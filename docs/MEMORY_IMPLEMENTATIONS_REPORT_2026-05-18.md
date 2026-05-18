# Отчёт по трём реализациям коллективной памяти

**Дата:** 2026-05-18
**Статус:** все три реализации закрыты, все эксперименты проходят детерминистически.

В репозитории Songlines сосуществуют **четыре параллельные архитектуры** мульти-агентной памяти. Каждая решает свою задачу из таксономии рецензентов (Иван Томилов / Наталья Гусарова):

| Реализация | Папка | Вариант рецензентов | Роль |
|---|---|---|---|
| `independent_memory/` | `independent_memory/` | **(1) independent** | явный lower-bound baseline, no communication |
| `songline_drive/` | `songline_drive/` | (2) центр (max) | основной single-/multi-agent stack с shared bus |
| `distributed_memory/` | `distributed_memory/` | (2) центр (mid) | per-agent memory + central ConsensusLayer |
| `peer_memory/` | `peer_memory/` | (3) communication | peer-to-peer без центра |

Этот отчёт описывает **что**, **как** и **зачем** в каждой из них, с особым вниманием к тому, **как строится граф** в каждом подходе.

> **История обновлений:**
> - 2026-05-18 первая версия: 3 реализации, вариант (1) подавался как degenerate case через `AgentMemory` напрямую
> - 2026-05-18 (обновление): добавлен явный пакет `independent_memory/` чтобы все 4 варианта имели first-class представление

---

## 0. Общие низкоуровневые примитивы

Все три реализации **переиспользуют** одни и те же базовые компоненты из `songline_drive/`:

| Файл | Что делает |
|---|---|
| `place_alignment.py` | `PlaceAlignmentEngine` — кластеризует наблюдения в концепты по spatial + semantic близости |
| `belief_fusion.py` | `TemporalDecayEngine`, `ConflictRuleSet` — Phase 3 dynamics |
| `collective_concepts.py` | `SharedConceptGraph`, `SharedConceptNode` — datatypes для графа |
| `collective_types.py` | `AgentSignature`, `CollectiveEvent`, `CollectiveQuery` — общие datatypes |

Разница между тремя реализациями — **на каком уровне эти примитивы инстанциируются**:
- В `songline_drive/` — глобально (один граф на всех)
- В `distributed_memory/` — per-agent (по одному графу на каждого) + центральный layer
- В `peer_memory/` — per-agent + per-agent merged view, без центра

---

## 0.5. `independent_memory/` — fully isolated agents (variant 1)

### 0.5.1 Что было сделано

Минимальный явный пакет для варианта (1) из таксономии рецензентов. Каждый агент полностью изолирован, никакой механизм коммуникации не существует.

### 0.5.2 Архитектура — без коммуникации вообще

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│IndependentAgt│    │IndependentAgt│    │IndependentAgt│
│   scout-A    │    │   scout-B    │    │   scout-C    │
│ ┌──────────┐ │    │ ┌──────────┐ │    │ ┌──────────┐ │
│ │event log │ │    │ │event log │ │    │ │event log │ │
│ │  graph   │ │    │ │  graph   │ │    │ │  graph   │ │
│ └──────────┘ │    │ └──────────┘ │    │ └──────────┘ │
└──────────────┘    └──────────────┘    └──────────────┘

         (НЕТ bus, НЕТ ConsensusLayer, НЕТ snapshots)

       IndependentRuntime — просто список агентов
```

### 0.5.3 Как строится граф

Точно так же как в `distributed_memory.AgentMemory` (тот же `PlaceAlignmentEngine`, тот же Phase 3 dynamics), **но** ничего наружу не идёт:

```python
agent = IndependentAgent("scout-A")
agent.observe((3, 4), {"water_source": 0.95}, episode_id=1, step_idx=0)
agent.refresh_local()  # граф строится из СВОИХ наблюдений
agent.local_query("water_source")  # единственный способ читать

# Это всё запрещено:
agent.snapshot()    # RuntimeError
agent.broadcast()   # RuntimeError
agent.receive()     # RuntimeError
```

### 0.5.4 Зачем отдельный пакет

Можно было бы использовать `distributed_memory.AgentMemory` напрямую (что и делалось изначально), но это **неявный** контракт — легко случайно построить агрегацию и не заметить. Отдельный `independent_memory/` делает контракт **uncircumventable на уровне API**: запрещённые методы реально кидают исключения.

Для симметрии с `distributed_memory/` и `peer_memory/` — все три варианта рецензентов имеют first-class представление с одинаковой surface area (`spawn_agent`, `observe`, `tick`, `local_query`).

### 0.5.5 Файлы

| Файл | LOC | Что внутри |
|---|---|---|
| `independent_agent.py` | 140 | `IndependentAgent` — wrapper над `AgentMemory` с явно запрещёнными методами |
| `independent_runtime.py` | 90 | `IndependentRuntime` — spawn + step, без aggregation |

Total: ~230 LOC.

### 0.5.6 Эксперимент

`experiments/independent_memory/exp01_isolation.py`:
- 3 агента в 3 регионах → каждый знает 1 концепт (свой)
- Centroid'ы соответствуют размещению
- `IndependentRuntime` не имеет `collective_query` / `consensus` / `bus`
- Forbidden методы (`snapshot`, `broadcast`, `receive`) кидают `RuntimeError`

Результат:
```
✓ 3 agents, each knows exactly 1 region
  (N=(1.0, 1.0), E=(8.0, 1.0), S=(4.0, 6.0))
  no collective query exposed
```

### 0.5.7 Когда использовать

- Как явный lower-bound в ablation
- Когда хотим точно знать "что доступно только из своих наблюдений"
- В paper: ветка `independent` в 3-way comparison

---

## 1. `songline_drive/` — Phase 1-4 (shared collective memory)

### 1.1 Что было сделано

Полный стек коллективной памяти в четырёх фазах + 8 smoke тестов + adaptive loop + end-to-end multi-agent эксперимент. Все smoke проходят детерминистически (0.93 секунды на полный прогон).

### 1.2 Архитектура — один общий граф

```
agent-A.observe(...)                       agent-B.observe(...)
        │                                          │
        └───────────────┐         ┌────────────────┘
                        ▼         ▼
                   CollectiveMemory  (ОДИН event bus)
                        │
                        ▼
                 ConceptRecallLayer
                        │
                        ▼
                 SharedConceptGraph  (ОДИН граф концептов)
                        │
                        ▼
                  SemanticField  (ОДНО поле активаций)
                        │
                        ▼
                   FieldAdapter
                        │
                        ▼
            agent-A.query(...) ◄──► agent-B.query(...)
                (читают из одного слоя)
```

**Ключевое:** все агенты пишут в **один** `CollectiveMemory` и читают из **одного** `SharedConceptGraph`. Никакой приватной памяти у агента нет.

### 1.3 Как строится граф

```python
# Phase 1: события записываются в общую ленту
collective.publish_event("place_observed", agent_id="agent-A",
    payload={"place_key": (3, 4), "semantic_tags": {"water_source": 0.95}, ...})

# Phase 2: ConceptRecallLayer.refresh() триггерит PlaceAlignmentEngine.build(),
# которое сканирует ВСЕ события и кластеризует их в концепты:
graph = recall_layer.refresh(collective)
# graph.concepts — Dict[concept_id, SharedConceptNode]
# Концепт = кластер мест с member_place_keys, dominant_tag, semantic_profile,
# supporting_agents (множество всех агентов, кто внёс наблюдение в этот кластер)

# Phase 3: на graph применяются:
# - TemporalDecayEngine — concept.freshness *= decay_factor^(current_seq - last_seq)
# - ConflictRuleSet — concept.conflict_score = f(incompatible tag pairs в profile)

# Phase 4: SemanticField.rebuild_from_concepts(graph) строит активации
# для каждой пары (channel, concept) по формуле:
# A(k,c) = λ·A_prev + α·B(c)·I(k,c) + γ·D - η·conflict - ξ·reservation
field.rebuild_from_concepts(graph, current_seq=seq)
```

**Алгоритм кластеризации (`PlaceAlignmentEngine`):**
1. Группировать events по `env_id`
2. Для каждой группы — собрать `descriptors` (xy, tag_profile, confidence)
3. Greedy clustering: новое наблюдение сливается с существующим концептом если:
   - `spatial_distance < spatial_radius` (default 4.0)
   - `tag_match_score > semantic_threshold` (default 0.45 для match, бонус 0.45 если dominant_tag совпадает)
4. После clustering — финализация: `centroid_xy`, `semantic_profile` (нормированный), `supporting_agents` (union по contributors)

**Важно:** `supporting_agents` — это **просто метка** "какие агенты внесли вклад". Trust per agent не моделируется, все равны.

### 1.4 Mode axes (три независимых)

| Axis | Файл | Значения |
|---|---|---|
| `milestone_mode` | `collective_memory.py` | `none / shared / consensus` |
| `graph_update_mode` | `concept_recall.py` | `none / lazy / incremental` |
| `collective_field_mode` | `field_adapter.py` | `none / descriptive / read_only / coordinated` |

`FieldMode.NONE` → возвращаемся к Phase 1 raw retrieval. Это baseline для ablation.

### 1.5 Phase 4 — semantic field formula

```
A_{t+1}(k, c) = λ·A_t(k, c)            ← exponential decay
              + α·B(c)·I(k, c)         ← belief × channel affinity
              + γ·D_t(k, c)            ← diffusion от соседей
              - η·X_t(c)               ← conflict suppression (Phase 3)
              - ξ·U_t(c)               ← reservation/occupancy (Phase 4c)
```

Defaults: λ=0.95, α=0.60, η=0.30, ξ=0.20, γ=0.10. Все exposed для `FieldOutcomeTracker` adaptive (Phase 4d).

### 1.6 Файлы

| Файл | LOC | Что внутри |
|---|---|---|
| `collective_types.py` | 146 | `AgentSignature, CollectiveEvent, CollectiveQuery, PlaceKey` |
| `collective_memory.py` | 518 | `CollectiveMemory` — event bus + raw retrieval |
| `collective_metrics.py` | 879 | Phase 1+ метрики |
| `collective_concepts.py` | 346 | `SharedConceptGraph, SharedConceptNode` |
| `concept_recall.py` | 381 | `ConceptRecallLayer` — concept-level recall |
| `belief_fusion.py` | 270 | `TemporalDecayEngine, ConflictRuleSet` (Phase 3) |
| `collective_field_types.py` | 145 | `FieldMode, FieldChannelState, FieldCellState, FieldReservation` |
| `semantic_field.py` | 603 | `SemanticField` — multi-channel activation |
| `field_adapter.py` | 237 | `FieldAdapter` — единая точка интеграции |
| `field_metrics.py` | 372 | Phase 4 метрики |
| `field_adaptive.py` | 215 | `FieldOutcomeTracker` — adaptive reweighting |
| `field_visualization.py` | 119 | snapshots, heatmaps |

### 1.7 Smoke + эксперименты

8 smoke (`scripts/multiagent_smoke_*.py`), все проходят, runner `scripts/run_all_smokes.py` (0.93s):
- Phase 1: GT top-1 recovery
- Phase 2: concept consolidation A/B
- Phase 3a/b/c: decay / conflict / incremental
- Phase 4a/b/c/d: channel separation / reranking / deconfliction / adaptive

End-to-end (`experiments/multiagent_navigation/`):
- `exp_field_modes_comparison.py` — coordinated 100% / 21 шагов vs read_only 0% / timeout
- `exp_adaptive_loop.py` — eta climbs 0.30 → 0.525 за 5 adapt-циклов

### 1.8 Где сидит "обучение"

В строгом смысле — нигде (нет градиентов). В широком — `FieldOutcomeTracker` адаптирует параметры поля по rule-based правилам:
- High-conflict concept с fail_rate ≥ 0.60 → `eta_conflict × 1.15`
- Reservation success ≥ 0.70 → `xi_occupancy × 1.05`
- Global failure ≥ 0.60 → `gamma_diffusion × 0.90`

Это **централизованная** адаптация (один tracker, одно поле, общее на всех).

### 1.9 Когда использовать

- Симметричные доверенные агенты в одной среде
- Нужен coordinated reservation deconfliction
- Нужен adaptive field tuning
- В архитектуре нет требования провенанса или партикальной наблюдаемости

### 1.10 Когда **не** использовать

- Untrusted источники (нет trust weighting)
- Партиальная наблюдаемость с разными "владениями" у агентов
- Нужно явное обнаружение inter-agent разногласий
- Требование децентрализации (рецензенты этого не примут как main)

---

## 2. `distributed_memory/` — Variant C (per-agent + central consensus)

### 2.1 Что было сделано

Альтернативная архитектура где у каждого агента **свой** event store и **свой** граф. Один центральный `ConsensusLayer` периодически делает merge снапшотов. 5 экспериментов, все проходят.

### 2.2 Архитектура — независимые графы, центральный аггрегатор

```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ AgentMemory │  │ AgentMemory │  │ AgentMemory │
│  scout-A    │  │  scout-B    │  │  scout-C    │
│ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │
│ │event log│ │  │ │event log│ │  │ │event log│ │
│ │  graph  │ │  │ │  graph  │ │  │ │  graph  │ │
│ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       │ snapshot()     │ snapshot()     │ snapshot()
       ▼                ▼                ▼
┌──────────────────────────────────────────────────┐
│              ConsensusLayer.merge()               │  ← ЕДИНЫЙ центр
│   union-find clustering across agents             │
│   + trust-weighted aggregation                    │
│   + disagreement detection                        │
└────────────────────┬─────────────────────────────┘
                     │
                     ▼
              ConsensusReport
              (один на всех)
                     ▲
                     │
       collective_query(...) ← все агенты читают одну report
```

### 2.3 Как строится граф

**Локальный граф у каждого агента** — точно так же как в `songline_drive/`, но **только из своих** наблюдений:

```python
# Каждый AgentMemory имеет приватный CollectiveMemory и приватный recall layer
agent_a = AgentMemory("scout-A")  # внутри: свой CollectiveMemory, свой ConceptRecallLayer

agent_a.observe((3, 4), {"water_source": 0.95}, episode_id=1, step_idx=0)
# → пишется ТОЛЬКО в agent_a.memory (приватный event bus)

agent_a.refresh_local()
# → ConceptRecallLayer.refresh() строит граф из ТОЛЬКО agent_a's observations
# → agent_a.memory.events — только свои события, концепты — только свои кластеры
```

**Cross-agent объединение — в `ConsensusLayer.merge(views)`**:

```python
# 1. Pool — собрать local concepts от всех агентов
pool = [(agent_id, concept_id, summary) for view in views for ...]

# 2. Cross-agent clustering (union-find):
# Два концепта от РАЗНЫХ агентов сливаются в один консенсус-кластер если:
#   - центроиды на расстоянии < consensus_radius (default 4.0)
#   - И (dominant_tag совпадает) ИЛИ (cosine similarity > threshold)
# Концепты от ОДНОГО агента никогда не сливаются.

# 3. Aggregation в каждом кластере:
# - centroid: trust-weighted mean (trust × log(1 + support))
# - semantic_profile: trust × log(1 + support) weighted sum, normalized
# - dominant_tag: argmax of normalized profile
# - confidence: trust-weighted mean

# 4. Disagreement detection:
# Для каждого консенсус-концепта — найти пары агентов с несовместимыми
# dominant_tag (по ConflictRuleSet из Phase 3)

# 5. Scoring:
# consensus_confidence = mean_conf × agreement × min(1, sqrt(n_agents / n_total))
```

Результат — `ConsensusReport` со списком `DistributedConcept`. **Один report на всех агентов** — все запросы через `runtime.collective_query()` читают именно его.

### 2.4 Trust model

`TrustModel` — **глобальный** скаляр trust per agent в `[0.1, 1.0]`:
- EMA update по outcomes: `trust = trust + α·(target - trust)`, где `target = max` если correct, `min` если wrong
- Используется как вес в `ConsensusLayer.merge()`

Это **симметричный** trust — у всех агентов одинаковое представление о trust каждого. Никакой `trust[A][B] ≠ trust[B][A]`.

### 2.5 Provenance preserved

Каждый `DistributedConcept` хранит список `AgentContribution`:
- `agent_id` — кто
- `local_concept_id` — какой именно концепт у себя локально
- `trust` — с каким весом он внёс вклад
- `local_dominant_tag, local_support, local_confidence, local_freshness`

Это сохраняет полный трекинг "кто что утверждал" — невозможно в `songline_drive/` (там есть только `supporting_agents` как множество).

### 2.6 Файлы

| Файл | LOC | Что внутри |
|---|---|---|
| `consensus_types.py` | 176 | `AgentMemoryView, DistributedConcept, AgentContribution, AgentDisagreement, ConsensusReport` |
| `trust_model.py` | 95 | `TrustModel` — глобальный скаляр + EMA |
| `agent_memory.py` | 220 | `AgentMemory` — per-agent wrapper над Phase 1-4 примитивами |
| `disagreement.py` | 114 | `detect_pairwise_disagreements, agreement_score` |
| `consensus_layer.py` | 328 | `ConsensusLayer.merge()` |
| `distributed_runtime.py` | 204 | `DistributedRuntime` — orchestrator (центральный) |
| `__init__.py` | 47 | exports |

Total: ~1100 LOC.

### 2.7 Эксперименты

`experiments/distributed_memory/`:
- `exp01_basic_per_agent.py` — privacy: 2 агента → раздельные графы (centroids @ (0,0) и (9,7))
- `exp02_consensus_alignment.py` — happy path: 2 агента на одном месте → 1 aligned concept
- `exp03_disagreement.py` — water vs hazard → flagged, agreement=0.0
- `exp04_trust_weighted_fusion.py` — high-trust majority выигрывает, flip → инверсия
- `exp05_partial_observability.py` — каждый видит регион, consensus → все 3 региона

### 2.8 Где централизация

- **`ConsensusLayer` — единственный исполнитель merge.** Никакой агент не запускает merge сам.
- **`TrustModel` живёт в Runtime.** Один объект на всех.
- **`_last_report` хранится в Runtime.** `collective_query()` читает именно его.
- **Union-find clustering глобальный** — алгоритм видит все пары `(agent_i, concept_j)` одновременно.

### 2.9 Где **нет** централизации

- `ConsensusLayer.merge()` — **stateless pure function**, между вызовами state не накапливается
- Никакой агент не пишет в чужую память — изоляция настоящая
- Снапшоты сериализуемы (`AgentMemoryView.to_dict()`) — готово к сети

### 2.10 Критика рецензентов

Это **federated** (CTDE-shaped), не **peer-to-peer**. `ConsensusLayer` — это "общий склад для графов", через который текла бы информация в обучении. Рецензенты сказали: "так не хотим, можно как ablation".

Текущая позиция: **`distributed_memory/` остаётся как ablation baseline** (вариант 2 в их таксономии). Main — `peer_memory/`.

### 2.11 Когда использовать

- Untrusted источники, нужен trust weighting
- Партиальная наблюдаемость
- Нужен provenance (кто что сказал)
- Нужно явное обнаружение разногласий
- Готов к роли "centralized baseline" в paper, не main

---

## 3. `peer_memory/` — peer-to-peer (no center)

### 3.1 Что было сделано

Полностью децентрализованная архитектура. Каждый агент держит свой private graph + свой private merged view. Inter-agent контакт только через passive `BroadcastBus`. 3 эксперимента, включая 3-way ablation.

### 3.2 Архитектура — без центра, каждый делает merge сам

```
┌────────────────────────────────────────────────────┐
│                    PeerRuntime                     │
│         (только scheduler — без агрегации)         │
│                                                    │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐      │
│  │PeerAgent │    │PeerAgent │    │PeerAgent │      │
│  │ scout-A  │    │ scout-B  │    │ scout-C  │      │
│  │┌────────┐│    │┌────────┐│    │┌────────┐│      │
│  ││AgentMem││    ││AgentMem││    ││AgentMem││      │
│  ││trust[] ││    ││trust[] ││    ││trust[] ││      │
│  ││PeerView││    ││PeerView││    ││PeerView││      │
│  │└────────┘│    │└────────┘│    │└────────┘│      │
│  └────┬─────┘    └────┬─────┘    └────┬─────┘      │
│       │               │               │            │
│       └───────────────┼───────────────┘            │
│                       │                            │
│              ┌────────▼────────┐                   │
│              │  BroadcastBus   │  ← пассивный      │
│              │ (только delivery)│    транспорт      │
│              └─────────────────┘                   │
└────────────────────────────────────────────────────┘
```

### 3.3 Как строится граф

**Личный граф каждого агента** — как в `distributed_memory/`:

```python
# PeerAgent внутри использует distributed_memory.AgentMemory
peer_a = PeerAgent("scout-A", bus)
peer_a.observe((3, 4), {"water_source": 0.95}, episode_id=1, step_idx=0)
peer_a.refresh_local()
# → строится приватный graph через PlaceAlignmentEngine
```

**Распространение через broadcast — periodic, не proximity:**

```python
# Каждые K тиков:
peer_a.broadcast_now()
# → собирает peer_a.memory.snapshot() (AgentMemoryView)
# → BroadcastBus.broadcast(sender_id, msg)
# → bus кладёт msg в PeerInbox каждого ДРУГОГО агента
```

**Каждый агент САМ делает merge на принятых сообщениях:**

```python
peer_a.process_inbox_and_merge()
# 1. Drain inbox — забрать новые сообщения
# 2. Обновить _last_known[peer_id] — кэш последнего snapshot от каждого пира
# 3. local_merge(own_snapshot, all_last_known, own_trust)
#    — та же union-find clustering логика что в ConsensusLayer
#    — но называется агентом на самом себе СО СВОИМ trust table
# 4. Сохранить результат как peer_a._peer_view
```

**Ключевое отличие от `distributed_memory/`:** функция `local_merge` — пустая обёртка над теми же алгоритмами кластеризации, но **выполняется per-agent**, не один раз централизованно. Результат — **N разных** `PeerView` объектов, по одному на агента, у каждого свой trust table.

### 3.4 Asymmetric trust — главное концептуальное отличие

```python
peer_a.trust.set("scout-B", 0.90)  # A очень доверяет B
peer_b.trust.set("scout-A", 0.20)  # B почти не доверяет A
```

Это **приватная таблица в каждом агенте**. `trust[A][B] != trust[B][A]` — нормально. У каждого агента своя `AsymmetricTrust`, у других нет к ней доступа.

Эффект на merge:
- В `peer_a.peer_view` — данные от B весят с фактором 0.90 (B сильно влияет на A's beliefs)
- В `peer_b.peer_view` — данные от A весят с фактором 0.20 (A почти не влияет)
- Результат: **разные beliefs у двух агентов о ОДНОМ И ТОМ ЖЕ месте**

Это и есть проверенное в `exp02_asymmetric_trust.py`: одинаковые наблюдения, разные tags, разный trust → `A_view.hazard_share=0.474` vs `B_view.water_share=0.167`.

### 3.5 Почему `_last_known` cache важен

Если бы каждый агент держал только **только что принятые** сообщения, между двумя broadcast-тиками peer-информация **исчезала бы** из merged view. Поэтому каждый агент кэширует "последний известный snapshot от каждого пира":

```python
# В PeerAgent:
self._last_known: Dict[str, BroadcastMessage] = {}

# При merge:
new_messages = self.bus.inbox(self.agent_id).drain()
for msg in new_messages:
    self._last_known[msg.sender_id] = msg  # newer overwrites older
# Merge ВСЕГДА использует full _last_known, не только свежие messages
```

Это делает peer_view стабильным между broadcast'ами.

### 3.6 `BroadcastBus` — чистый транспорт

`BroadcastBus` намеренно тупой:
- `register(agent_id)` — создать PeerInbox
- `broadcast(sender_id, msg)` — положить копию в inbox каждого ДРУГОГО агента
- `inbox(agent_id)` — вернуть inbox для дренажа

Что bus **не делает**:
- Не агрегирует / не мержит / не интерпретирует
- Не хранит global state (только per-agent buffers)
- Не решает кто с кем общается (это задача протокола)

**Тест на отсутствие централизации:** удалить `peer_runtime.py` — система продолжит работать через прямые вызовы `agent.broadcast_now()` и `agent.process_inbox_and_merge()`. `PeerRuntime` — это **scheduling sugar**, не центральное состояние.

### 3.7 Файлы

| Файл | LOC | Что внутри |
|---|---|---|
| `peer_types.py` | 137 | `BroadcastMessage, PeerInbox, PeerView, PeerMergeReport` |
| `peer_trust.py` | 100 | `AsymmetricTrust` — per-owner table |
| `broadcast_bus.py` | 89 | `BroadcastBus` — пассивный транспорт |
| `peer_merge.py` | 255 | `local_merge()` — per-agent merge function |
| `peer_agent.py` | 222 | `PeerAgent` — lifecycle + кэш _last_known |
| `peer_runtime.py` | 181 | `PeerRuntime` — scheduling, без агрегации |
| `__init__.py` | 40 | exports |

Total: ~1000 LOC.

### 3.8 Эксперименты

`experiments/peer_memory/`:
- `exp01_basic_broadcast.py` — 2 агента через 3 тика → каждый видит другого в **своём** view
- `exp02_asymmetric_trust.py` — асимметричный trust → divergent beliefs (0.474 vs 0.167)
- `exp03_three_way_ablation.py` — **главный**: independent / centralized / peer на одном scenario

3-way ablation результат:
```
mode             avg_cov/3   msgs   per-agent knowledge
independent           1.00      0   N=1  E=1  S=1
centralized           3.00      6   N=3  E=3  S=3
peer                  3.00      6   N=3  E=3  S=3
```

Peer достигает того же coverage что centralized, но **без** ConsensusLayer.

### 3.9 Когда использовать

- Main contribution для рецензентов — вариант 3
- Сценарии где агенты могут "не общаться" с некоторыми
- Robustness к отказу одного агента (нет single point of failure)
- Когда trust асимметричен (разный для разных агентов)
- Eventually consistent распределённые системы

### 3.10 Чего пока нет

- Proximity-based gossip (выбран periodic как чистый baseline)
- Delta messages (передаём full snapshots для простоты v1)
- Symmetric pairwise trust update (по парам)
- End-to-end навигационный сценарий с peer как main

---

## 4. Сравнительная матрица

| Свойство | `independent_memory/` | `songline_drive/` | `distributed_memory/` | `peer_memory/` |
|---|---|---|---|---|
| **Event store** | per-agent | shared (один на всех) | per-agent | per-agent |
| **Concept graph** | per-agent | shared (один) | per-agent | per-agent |
| **Aggregation** | **нет** | implicit (через shared graph) | central `ConsensusLayer.merge()` | per-agent `local_merge()` |
| **Кто делает merge** | никто | автоматически при `refresh()` | один центр | каждый агент сам |
| **Merged view** | **нет** (только локальный граф) | один (sам граф) | один (`ConsensusReport`) | **N** (один на агента) |
| **Trust model** | не нужен (нет пиров) | не моделируется | global scalar | **asymmetric pairwise** |
| **Provenance** | n/a | `supporting_agents: Set[str]` | `List[AgentContribution]` | `List[AgentContribution]` |
| **Disagreement detection** | **невозможно** | нет | да | да |
| **Reservation / coordination** | нет | да (Phase 4c) | нет | нет |
| **Adaptive params** | нет | да (Phase 4d) | нет | нет |
| **Single point of failure** | нет | да (shared graph) | да (Runtime + Layer) | **нет** |
| **CTDE-style** | нет | да | да | **нет** |
| **Вариант рецензентов** | **(1) independent** | (2) центр (макс) | (2) центр (средне) | **(3) communication** |

---

## 5. Сценарии когда какую выбрать

```
                     Untrusted источники?
                            │
                  ┌─────────┴─────────┐
                 yes                 no
                  │                   │
       ┌──────────┴─────────┐         │
       Нужна полная         │         │
       децентрализация?     │         │
       │                    │         │
    ┌──┴──┐                 │         │
    yes   no                │    Нужны reservation/
    │     │                 │    adaptive params?
    │     │                 │         │
    │     │                 │    ┌────┴────┐
    │     │                 │    yes       no
    │     │                 │     │         │
    ▼     ▼                 ▼     ▼         ▼
 peer_  distributed_   distributed_  songline_  songline_
 memory memory         memory        drive      drive
                                     (full)     (lightweight)
```

---

## 6. Соответствие таксономии рецензентов

> Иван Томилов: "если будут эксперименты с (1) полностью независимыми агентами, (2) агентами с центром, (3) агентами с коммуникацией, то это будет полный набор"

| Их вариант | Наша реализация | Эксперимент |
|---|---|---|
| (1) Independent | **`independent_memory/`** | `experiments/independent_memory/exp01_isolation.py` + `experiments/peer_memory/exp03_three_way_ablation.py` (ветка independent) |
| (2) Center | `distributed_memory/` (mid) + `songline_drive/` (max) | `experiments/distributed_memory/exp*` + Phase 1-4 smokes + `experiments/peer_memory/exp03` (ветка centralized) |
| (3) Communication | **`peer_memory/`** | `experiments/peer_memory/exp01-03` (главное) |

Все три варианта в одном репо. 3-way ablation в одном скрипте.

---

## 7. Ключевой архитектурный takeaway

Различие между тремя реализациями **не в алгоритме кластеризации** (он один и тот же — `PlaceAlignmentEngine` + Phase 3 dynamics) и **не в datatypes** (`AgentContribution`, `DistributedConcept` переиспользуются).

Различие в **уровне инстанциации**:

```
independent_memory/  ───►  N графов, 0 merger'ов
                            ├── N приватных event bus
                            ├── N локальных графов
                            └── (ничего больше — коммуникация запрещена)

songline_drive/      ───►  1 граф на всех
                            ├── 1 event bus
                            ├── 1 SharedConceptGraph
                            └── 1 SemanticField

distributed_memory/  ───►  N графов + 1 центральный merger
                            ├── N приватных event bus
                            ├── N локальных графов
                            └── 1 ConsensusLayer.merge() → 1 ConsensusReport

peer_memory/         ───►  N графов + N приватных merger'ов
                            ├── N приватных event bus
                            ├── N локальных графов
                            ├── 1 пассивный BroadcastBus (транспорт)
                            └── N agent.local_merge() → N PeerView
```

Это **одни и те же примитивы**, **разная топология** инстанциации.

Поэтому в репо можно одновременно держать все три варианта без дублирования логики и сравнивать их head-to-head в одном эксперименте (`exp03_three_way_ablation.py`).

---

## 8. Файлы экспериментов и smoke

### Phase 1-4 smoke (`scripts/`)
- `multiagent_smoke_collective.py` — Phase 1
- `multiagent_smoke_phase2.py`, `..._ab.py` — Phase 2
- `multiagent_smoke_phase3.py` — Phase 3
- `multiagent_smoke_phase4a.py` — Phase 4a (descriptive)
- `multiagent_smoke_phase4b_ab.py` — Phase 4b (reranking)
- `multiagent_smoke_phase4c.py` — Phase 4c (coordinated)
- `multiagent_smoke_phase4d.py` — Phase 4d (adaptive)
- `run_all_smokes.py` — runner, 8/8 за 0.93s

### Multi-agent navigation (`experiments/multiagent_navigation/`)
- `exp_field_modes_comparison.py` — descriptive vs read_only vs coordinated end-to-end
- `exp_adaptive_loop.py` — adaptive convergence в реальном episode

### Distributed memory (`experiments/distributed_memory/`)
- exp01 - exp05 (privacy, alignment, disagreement, trust, partial observability)

### Peer memory (`experiments/peer_memory/`)
- exp01 (broadcast), exp02 (asymmetric trust), exp03 (3-way ablation)

### Paper section
- `docs/Formatting_Instructions_For_NeurIPS_2026/collective_memory_appendix.tex` — описание архитектуры Phase 1-4 + ablation таблица + end-to-end + adaptive convergence

---

## 9. Что осталось открыто

| Направление | Статус |
|---|---|
| Proximity-based gossip в peer_memory | можно добавить как режим |
| Delta messages (не full snapshots) | optimization, не блокер |
| Peer-to-peer field fusion (поле через broadcasts) | следующий шаг если peer станет main |
| End-to-end навигация под peer_memory | следующий шаг для paper |
| Neural обучение поверх peer_view | будущее направление |
| Trust update от outcomes в реальном эпизоде | hook есть (`peer.report_outcome_from_peer`), не подключён |

---

## 10. TL;DR для рецензентов

- **(1) Independent**: `AgentMemory` standalone, доступно через `distributed_memory/`
- **(2) Center**: два уровня — `songline_drive/` (максимальная централизация, all-shared) и `distributed_memory/` (per-agent + ConsensusLayer). Оба позиционируются как **ablation baselines**.
- **(3) Communication**: `peer_memory/` — main contribution. Periodic broadcast, asymmetric trust, per-agent merged views, без центрального аггрегатора.

Полный 3-way ablation в одном скрипте: `experiments/peer_memory/exp03_three_way_ablation.py`.
