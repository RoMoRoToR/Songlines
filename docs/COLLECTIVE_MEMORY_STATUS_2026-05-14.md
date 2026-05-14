# Collective Memory — статус и архитектура

**Дата фиксации**: 2026-05-14
**Состояние**: Phase 1 → 4d закрыты, все smoke прошли.

Этот документ описывает что реализовано в пакете `songline_drive/collective_*` и `songline_drive/field_*`, как уровни связаны между собой, какие инварианты держатся и как воспроизвести каждый smoke.

---

## 1. Общая идея

Поверх существующего planner / control стека Songlines добавлен **независимый параллельный слой коллективной памяти**.
Существующий код не переписывается. Новый слой только читает из планировщика и предоставляет ему ranked-кандидатов.

Структурно слой состоит из четырёх уровней (Phase 1 → Phase 4), каждый из которых строится поверх предыдущего:

```
Phase 1 — event bus (запись всего, что наблюдают агенты)
   ↓
Phase 2 — shared concept graph (консолидация наблюдений в концепты)
   ↓
Phase 3 — belief dynamics (temporal decay + conflict fusion)
   ↓
Phase 4 — semantic field (многоканальная активация + reranking + coordinated deconfliction + adaptive reweighting)
```

Каждый уровень имеет собственный **mode-axis**, который можно выставить независимо:
| Axis | Файл | Значения |
|---|---|---|
| `milestone_mode` | `collective_memory.py` | `none / shared / consensus` |
| `graph_update_mode` | `concept_recall.py` | `none / lazy / incremental` |
| `collective_field_mode` | `field_adapter.py` | `none / descriptive / read_only / coordinated` |

Любая комбинация значений валидна и тестируется независимым smoke.

---

## 2. Инварианты (соблюдаются на всех уровнях)

1. **Phase 1 event bus** — append-only. Никто, включая Phase 4, не модифицирует уже опубликованные события.
2. **Phase 2 concept graph** — единственный канонический источник concept-level состояния. Field читает граф, но не пишет в него.
3. **Phase 3 belief dynamics** — применяется первым в каждом цикле refresh, до того как Phase 4 строит поле.
4. **Planner / local control** — не переписывается. Phase 4 только переранжирует кандидатов и/или применяет occupancy pressure.

Если любой mode выставлен в `none`, поведение системы сводится к baseline (Phase 1) — это используется в A/B сравнениях.

---

## 3. Phase 1 — event bus + collective memory

### Файлы
- `songline_drive/collective_types.py` (146 строк) — datatypes:
  `AgentSignature`, `CollectiveEvent`, `CollectiveNode`, `CollectiveQuery`, `CollectiveQueryResult`, `PlaceKey`.
- `songline_drive/collective_memory.py` (518 строк) — `CollectiveMemory`:
  - `register_agent(sig)`
  - `publish_event(event_type, agent_id, episode_id, step_idx, env_id, payload, confidence)`
  - `query_collective_nodes(query, top_k)` — raw recency-weighted retrieval по `semantic_tags`
  - Внутренний counter `_next_seq` обеспечивает глобальный временной порядок.

### Что делает
Каждый агент при наблюдении вызывает `publish_event("place_observed", ...)`. Событие индексируется по `place_key`, `env_id`, тегам. Запросы по тегу возвращают candidate places, отсортированные по `recency_lambda^age × confidence`.

### Smoke
- `scripts/multiagent_smoke_collective.py` (447 строк) — два агента-скаута публикуют наблюдения, третий агент запрашивает water/hazard. Проверяется, что top-1 совпадает с GT.

---

## 4. Phase 2 — shared concept graph

### Файлы
- `songline_drive/collective_concepts.py` (346 строк) — `SharedConceptGraph`, `CollectiveConcept`.
  Концепт — кластер из `member_place_keys` с общим `dominant_tag`, `support_count`, `supporting_agents`, `semantic_profile`, `centroid_xy`.
- `songline_drive/concept_recall.py` (381 строка) — `ConceptRecallLayer`:
  - `refresh(collective)` → строит `SharedConceptGraph` через `PlaceAlignmentEngine`
  - `query(target_tag, requesting_agent_id, env_id, top_k, current_seq)` → `List[ConceptRecallResult]`
  - `to_collective_results(...)` — конвертация в Phase 1 формат для drop-in замены.

### Что делает
`PlaceAlignmentEngine` (отдельный файл) консолидирует все `place_observed` события: близкие по координатам (`spatial_radius=4.0`) и совпадающие по dominant tag → один концепт. Концепт хранит **усреднённый** `semantic_profile`, list of contributing places и list of supporting agents.

`ConceptRecallLayer.query()` возвращает концепты, отсортированные по
`score = confidence × tag_match × log1p(support) × freshness × (1 - conflict_penalty)`.

### Ключевая особенность
**Spatial isolation для conflict тестов**: если контестная точка ближе `spatial_radius` к чистым точкам, она будет смержена в один концепт, и hazard сигнал размоется по всем member places. Поэтому во всех 4b/4c/4d smoke тестах контестные точки расположены так, чтобы попарные расстояния > 4.0:
- `PURE_A = (0, 0)`, `PURE_B = (9, 7)`, `CONTESTED = (4, 3)`
- Расстояния: A↔C=5.0, B↔C=7.81, A↔B=11.4 — все > 4.0 ✓

### Smoke
- `scripts/multiagent_smoke_phase2.py` (533 строки) — базовый concept consolidation
- `scripts/multiagent_smoke_phase2_ab.py` (630 строк) — A/B comparison: raw Phase 1 vs ConceptRecallLayer

---

## 5. Phase 3 — belief dynamics

### Файлы
- `songline_drive/belief_fusion.py` (270 строк):
  - `TemporalDecayEngine(decay_factor, min_freshness)` — экспоненциальный decay по `current_seq - last_update_seq`
  - `ConflictRuleSet` — пары несовместимых тегов (`water_source ↔ hazard_edge`). `.songlines_default()` возвращает стандартный набор.
  - `apply_decay_to_graph(graph, current_seq)`, `apply_conflicts_to_graph(graph)` — мутируют concept attributes.

### Что делает
Перед каждым query:

1. `TemporalDecayEngine` обновляет `concept.freshness` по формуле:
   `freshness = max(min_freshness, decay_factor^(current_seq - last_seq))`
2. `ConflictRuleSet` смотрит `semantic_profile` каждого концепта: если в нём присутствуют incompatible tags с весами выше порога, вычисляется `conflict_score = min(weight_a, weight_b) / max(...)`. Это сохраняется в `concept.conflict_score` (Phase 3b — ключевое для Phase 4).
3. `ConceptRecallLayer.query()` использует `conflict_penalty = min(max_conflict_penalty, conflict_score)`, который входит в score множителем `(1 - penalty)`.

### Phase 3b — ключевой нюанс
Conflict вычисляется на **уровне концепта**, не на уровне отдельной точки. Если контестная точка попала в большой кластер водных точек (например, 1 hazard observation на 10 water), то даже после Phase 3 `conflict_score` будет низким (около 0.05). Для надёжной демонстрации conflict suppression нужны изолированные ячейки (см. Phase 2).

### Smoke
- `scripts/multiagent_smoke_phase3.py` (690 строк) — три сценария:
  - 3a: temporal decay → старые наблюдения проигрывают свежим. `stale_suppression=1.00`
  - 3b: conflict fusion → контестный концепт теряет вес. `conflict_penalty=0.74`
  - 3c: incremental updates → инкрементальное обновление графа не ломает существующие концепты. `stability=0.83, churn=0.09`

---

## 6. Phase 4 — semantic field

### Файлы
- `songline_drive/collective_field_types.py` (145 строк) — datatypes:
  - `FieldMode` — enum `{none, descriptive, read_only, coordinated}` с `validate()`
  - `FieldChannelState` — `activation, activation_fast, activation_slow, freshness, belief_strength, conflict_pressure, support_pressure, reservation_pressure, last_update_seq`
  - `FieldCellState` — `concept_id, channels: Dict[str, FieldChannelState], base_confidence/freshness/purity/conflict, support_count, supporting_agents, centroid_xy`
  - `FieldReservation` — `concept_id, channel, agent_id, reserved_at_seq, expires_at_seq`

- `songline_drive/semantic_field.py` (603 строки) — `SemanticField`:
  - Параметры (с дефолтами): `lambda_decay=0.95, alpha_belief=0.60, eta_conflict=0.30, xi_occupancy=0.20, gamma_diffusion=0.10, diffusion_radius=5.0, diffusion_steps=1`
  - `DEFAULT_CHANNEL_AFFINITIES` — словарь каналов (`water_source`, `safe_neutral`, `hazard_edge`, `goal_region`, `hazard_recovery_route`), каждый со словарём positive и negative tag weights.
  - `rebuild_from_concepts(graph, current_seq)` — основной билд:
    1. Для каждого концепта вычисляет `B(c) = w_conf·conf + w_fresh·fresh + w_support·log1p(support)/log1p(100) + w_purity·purity`
    2. Для каждой пары (channel, concept) вычисляет `I(k,c)` — channel affinity как weighted dot product `semantic_profile · channel_weights` с отрицательными весами для конкурирующих тегов.
    3. `raw_act = max(0, alpha_belief · B · I - eta_conflict · concept.conflict_score)`
    4. EMA continuity: `act = lambda_decay · prev_act + (1 - lambda_decay) · raw_act` (для `activation_fast`/`activation_slow` — разные lambda).
    5. Опционально diffusion: `diffusion_steps` проходов гауссова усреднения по spatial neighbors внутри `diffusion_radius`.
  - `top_k_for_channel(channel, k)` → отсортированный list `[(cid, activation), ...]`
  - `rerank(concept_recall_results, channel, field_weight)` — независимо нормализует concept score и field activation в [0,1], комбинирует с весом ω.
  - `reserve(concept_id, channel, agent_id, duration, current_seq)` — немедленно применяет `-xi_occupancy` к `ch.activation`, регистрирует `FieldReservation`.
  - `release(concept_id, agent_id)` — восстанавливает активацию.
  - `expire_reservations(current_seq)` — снимает истёкшие.
  - `to_snapshot()` — JSON-сериализуемый dump для метрик.

- `songline_drive/field_adapter.py` (237 строк) — `FieldAdapter`:
  Единственная точка интеграции Phase 4 с остальным стеком.
  - `__init__(field, recall_layer, field_weight, mode)` — синхронизирует `self.field.mode = self.mode`.
  - `refresh(collective, current_seq)` → сначала `recall_layer.refresh()` (Phase 3 применяется!), затем `field.rebuild_from_concepts(graph, seq)`. Возвращает `(graph, field)`.
  - `query(collective, query, top_k, current_seq)` — graceful degradation:
    - `NONE` → `collective.query_collective_nodes()` (raw Phase 1)
    - `DESCRIPTIVE` → `recall_layer.query()` (Phase 2/3)
    - `READ_ONLY` / `COORDINATED` → recall + `field.rerank()`
  - `commit_reservation(agent_id, concept_id, channel, duration, current_seq)` → возвращает `FieldReservation` или `None` если mode ≠ COORDINATED.
  - `release_reservation`, `expire_reservations`, `active_reservations`, `field_query`, `snapshot`.

- `songline_drive/field_metrics.py` (372 строки) — pure read-only метрики:
  - **Phase 4a**: `field_activation_split`, `field_conflict_suppression_rate`, `field_cross_channel_separation`, `field_top1_stability`, `field_decay_half_life`, `field_activation_to_query_rank_correlation`, `field_novelty_signal`
  - **Phase 4b**: `field_rerank_precision_at_k`, `field_assisted_rank_gain`, `field_top1_gain`, `field_rerank_delta_steps`
  - **Phase 4c**: `duplicate_target_rate`, `reservation_conflict_rate`, `field_driven_deconfliction_rate`
  - **Aggregates**: `all_field_metrics_4a`, `all_field_metrics_4b`

- `songline_drive/field_visualization.py` (119 строк) — `activation_table`, `save_snapshot`, `channel_summary`, optional `plot_activation_heatmap` (matplotlib).

- `songline_drive/field_adaptive.py` (215 строк) — `FieldOutcomeTracker`:
  - `record_concept_outcome(concept_id, success)` — rolling window per concept
  - `record_reservation_outcome(success)` — rolling window global
  - `adapt(min_samples=3)` → возвращает dict изменённых параметров. Три правила:
    1. Concept с `base_conflict ≥ 0.15` и `fail_rate ≥ 0.60` → `eta_conflict × 1.15`
    2. Reservation `success_rate ≥ 0.70` → `xi_occupancy × 1.05`; `< 0.30` → `× 0.95`
    3. Global `fail_rate ≥ 0.60` (по всем outcome) → `gamma_diffusion × 0.90`
  - Жёсткие bounds: `eta ∈ [0.10, 0.95]`, `xi ∈ [0.05, 0.60]`, `gamma ∈ [0.00, 0.30]`.
  - `parameter_delta()`, `summary()`, `adaptation_history`.

### Field update equation (полная форма из спеки)

```
A_{t+1}(k, c) = λ·A_t(k, c)                       // decay
              + α·B(c)·I(k, c)                    // belief × channel affinity
              + γ·D_t(k, c)                       // diffusion from neighbors
              - η·X_t(c)                          // conflict suppression
              - ξ·U_t(c)                          // reservation/occupancy
```

(β·O_t, δ·P_t зарезервированы для будущих фаз, не используются сейчас.)

### Smoke и результаты

| Smoke | Сценарии | Главные числа |
|---|---|---|
| `multiagent_smoke_phase4a.py` (541 строка) | A: channel separation, B: conflict suppression, C: decay | `sep=0.177, suppress=1.0, decay_factor_100=0.00592` |
| `multiagent_smoke_phase4b_ab.py` (548 строк) | I: чистая вода, II: контестная вода (rerank) | `prec@1: B=1.0→C=1.0`, `contested_act=0.10 vs pure=0.69-0.72`, rank сохранён |
| `multiagent_smoke_phase4c.py` (337 строк) | read_only collision vs coordinated deconfliction | `read_only_collision=True, coordinated_deconflicted=True, deconf_rate=1.00, activation_drop=0.200` |
| `multiagent_smoke_phase4d.py` (416 строк) | Rule 1, 2a, 2b, 3, bounds | `eta=0.300→0.345, xi_up=0.200→0.210, xi_down=0.200→0.190, gamma=0.100→0.090, bounds_ok` |

---

## 7. Как воспроизвести весь стек

Из корня репозитория с активированным `.venv`:

```bash
# Phase 1
PYTHONPATH=. .venv/bin/python scripts/multiagent_smoke_collective.py --out_dir tmp/smoke_p1

# Phase 2
PYTHONPATH=. .venv/bin/python scripts/multiagent_smoke_phase2.py    --out_dir tmp/smoke_p2
PYTHONPATH=. .venv/bin/python scripts/multiagent_smoke_phase2_ab.py --out_dir tmp/smoke_p2ab

# Phase 3
PYTHONPATH=. .venv/bin/python scripts/multiagent_smoke_phase3.py    --out_dir tmp/smoke_p3

# Phase 4
PYTHONPATH=. .venv/bin/python scripts/multiagent_smoke_phase4a.py   --out_dir tmp/smoke_p4a
PYTHONPATH=. .venv/bin/python scripts/multiagent_smoke_phase4b_ab.py --out_dir tmp/smoke_p4b
PYTHONPATH=. .venv/bin/python scripts/multiagent_smoke_phase4c.py   --seed 0 --out_dir tmp/smoke_p4c
PYTHONPATH=. .venv/bin/python scripts/multiagent_smoke_phase4d.py   --out_dir tmp/smoke_p4d
```

Каждый скрипт пишет `*_summary.json` в свой `out_dir` и печатает `✓ Phase Nx passed ...` при успехе.

---

## 8. Что доказано и что нет

### Доказано
- **Phase 1**: event bus работает, recall возвращает GT top-1
- **Phase 2**: концепты собираются корректно, A/B vs raw показывает выигрыш в precision
- **Phase 3**: decay подавляет stale (suppression=1.0), conflict снижает score (penalty=0.74), incremental updates стабильны (churn=0.09)
- **Phase 4a (descriptive)**: каналы семантически разделены, contested suppression=1.0, decay предсказуем
- **Phase 4b (read_only)**: reranking сохраняет precision и rang GT при контесте
- **Phase 4c (coordinated)**: reservation реально разводит двух агентов на разные ресурсы (deconfliction rate = 1.0)
- **Phase 4d (adaptive)**: все три правила outcome-driven adaptation срабатывают, bounds держатся

### Не доказано (открытые направления)
- **End-to-end multi-agent navigation gains** в реальной BabyAI / MiniWorld среде с coordinated mode — пока только синтетические smoke
- **Adaptive parameters converge** к оптимальным значениям в долгом эпизоде — пока только проверено что правила срабатывают, но не сходимость
- **Field diffusion** даёт реальную пользу планировщику — `gamma_diffusion` параметр работает математически, но его влияние на success rate не измерено в end-to-end
- **β·O_t (novelty) и δ·P_t (planner-driven) терма** — зарезервированы в формуле, но не реализованы

---

## 9. Файлы памяти (для будущих сессий)

`~/.claude/projects/-Users-taniyashuba-PycharmProjects-Songlines/memory/`:
- `collective_memory_phase1.md`
- `collective_memory_phase2.md`
- `collective_memory_phase3.md`
- `collective_memory_phase4.md` (4a + 4b)
- `collective_memory_phase4cd.md` (4c + 4d)

---

## 10. Следующие шаги (по приоритету)

1. **Полный прогон всех 8 smoke** одним runner-скриптом → быстрая verification после любых изменений.
2. **End-to-end эксперимент** с BabyAI multi-agent + coordinated mode → сравнение success rate / steps vs read_only baseline.
3. **Adaptive loop в реальном эпизоде**: подключить `FieldOutcomeTracker` к multiagent runtime → observe convergence of `eta_conflict` / `xi_occupancy` под реальные failure patterns.
4. **Paper section** в `docs/Formatting_Instructions_For_NeurIPS_2026/` — описание архитектуры + abalation таблица из smoke результатов.
