# Roadmap: Sprint 1–15, приоритеты, go/no-go

**Родитель:** [`README.md`](README.md), [`04_EXPERIMENTS.md`](04_EXPERIMENTS.md). Каждый спринт — deliverable, независимо оцениваемый (дисциплина серии: остановка после любого спринта сохраняет ценность предыдущих).

---

## Последовательность спринтов

### Sprint 1 — Certificate layer
Реализовать `MemoryCertificate`, `AuthorityState`, `ValidationEvent`, `AuthorityDecision` (`01_FORMAL_MODEL.md` §2, §5–7). Никакого LLM, никакой интеграции с runtime v1 пока — чистые datatypes + state machine + unit-тесты переходов.

### Sprint 2 — Provenance DAG
Реализовать `origin_ids` vs `provenance_parents`, независимый подсчёт `n_eff` (`03_METRICS.md` §5). Это техническая предпосылка Theorem 1 — без неё E1 невозможен.

### Sprint 3 — Social Amplification benchmark (E1)
Первый критический эксперимент. **Go/no-go точка №1:** если PAF не стабилизируется около 1 у provenance-aware плеча — чинить архитектуру провенанса ДО перехода к Sprint 4, не идти дальше с дефектным origin-tracking.

### Sprint 4 — Staleness/revocation (E3)
World-version change + `AuthorityState` переход в `EXPIRED`/`REVOKED`. Переиспользует закон дальности W2 без изменений (`02_THEOREMS.md` Theorem 2).

### Sprint 5 — Independent corroboration (E2)
Доказать liveness. **Go/no-go точка №2:** если authority не растёт с независимыми подтверждениями — калибровка τ_E/τ_U слишком консервативна, откатить и перекалибровать перед Sprint 6.

### Sprint 6 — Receiver-specific role authority (E4)
Разные роли → разные admission-решения при равном E. Формально закрывает четвёртое go/no-go свойство (`README.md` §6.4).

### Sprint 7 — Causal interventions (инфраструктура для E5)
Собрать utility labels через randomized memory masking (`01_FORMAL_MODEL.md` §4): `(s, m, r, y, Z)` tuples на 5–10% eligible decisions.

### Sprint 8 — Utility estimator (E5)
Обучить `τ̂_θ(s, m, r)`, проверить calibration против `τ_true` на детерминированном grid (E5.1) и её необходимость на стохастическом субстрате (E5.2).

### Sprint 9 — Structural schemas (E7)
Relations + conditions + exceptions поверх LCS/graph-matching (G1). Переход от similarity-cluster к relational schema.

### Sprint 10 — LLM semantic layer
LLM extraction (observation → structured Claim) + query former + schema proposer. **Важно:** LLM здесь — proposal generator, не authority — trust/provenance/admission остаются explicit external state, не делегируются LLM (`10_CODE_LAYOUT.md` фиксирует эту границу на уровне кода).

### Sprint 11 — Full multi-agent LLM (E9)
Persistent 4–8-agent эксперимент на реалистичном субстрате. Headline experiment программы.

### Sprint 12 — Long horizon (E10)
`H ∈ {10,...,1000}` episodes без сброса памяти между ними.

### Sprint 13 — Full baseline suite
RAG/shared memory/vector/per-agent — весь набор бейзлайнов Table 1 (`06_FIGURES_AND_TABLES.md`).

### Sprint 14 — Theory cleanup
Формализовать (в статье) только то, что подтверждено runtime — не наоборот. Category theory остаётся appendix-уровнем, если не появится composable algebra конкретно для certificates/admission (см. `09_PAPER_OUTLINE.md` §5).

### Sprint 15 — Paper
Сборка статьи по outline из `09_PAPER_OUTLINE.md`. Отдельно от текущей задачи — эта папка (`FRONTIER_MEMORY_AUTHORITY_2026-08-07/`) содержит только planning MD, не саму статью.

---

## Приоритеты (если ресурсов мало — что делать первым)

### Критически важно (без этого программа не имеет смысла как отдельный claim)
1. `MemoryCertificate` (Sprint 1).
2. `AuthorityState` state machine (Sprint 1).
3. Provenance DAG / `origin_ids` vs `provenance_parents` (Sprint 2).
4. Social amplification experiment E1 (Sprint 3).
5. Independent corroboration E2 (Sprint 5).
6. Revocation E3 (Sprint 4).
7. Full LLM multi-agent experiment E9 (Sprint 11) — без него весь claim остаётся demonstrated только на grid'е, той же критики, что уже была применена к «no external validity» в замороженной серии (`docs/REVIEW_RISK_REGISTER.md`).

### Очень желательно
8. Causal utility (Sprint 7–8, E5).
9. Role-conditioned authority (Sprint 6, E4).
10. Long-horizon evaluation (Sprint 12, E10).
11. Полный набор RAG/vector-memory бейзлайнов (Sprint 13).

### Можно после
12. Open-ended relational schemas (Sprint 9, E7–E8).
13. Category theory formalization (Sprint 14, только если появится load-bearing algebra).
14. Реальные роботы / внешние субстраты за пределами уже закрытых grid + synthetic continuous + VMAS (см. `docs/CLAIM_EVIDENCE_MATRIX.md` — «Мы не заявляем внешнюю валидность за пределами...»).

---

## Go/no-go критерии на каждой развилке

| После спринта | Критерий продолжения | Если не выполнен |
|---|---|---|
| Sprint 3 (E1) | PAF провенанс-aware плеча ∈ [0.9, 1.1] | Чинить provenance DAG, не переходить к Sprint 4 |
| Sprint 5 (E2) | authority монотонно растёт с n_independent | Перекалибровать τ_E/τ_U перед Sprint 6 |
| Sprint 3+4+5 (все три Theorem вместе) | минимум 3 из 4 go/no-go свойств `README.md` §6 подтверждены | Framing «authority protocol» не работает — публиковать как честный negative result (дисциплина `docs/SERIES_VERDICTS.md`), НЕ идти в Sprint 6+ |
| Sprint 8 (E5.2) | causal estimator бьёт heuristic/LLM-rated на стохастическом субстрате | Документировать границу применимости (randomized intervention слишком дорога на LLM) как честную находку, но НЕ блокирует Sprint 9+ (utility estimator — desirable, не critical) |
| Sprint 11 (E9) | Full authority CSM одновременно: низкий FAR + success ≥ trust/staleness CSM + token budget ≤ 2× | Если провал — определить, какая из трёх осей (E/U/S) не перенеслась на LLM-субстрат, это становится следующей итерацией, не поводом откатывать весь фронтир |

---

## Что НЕ делать сейчас (явный список, чтобы не размывать фокус)

Прямая проекция принципа «don't add features beyond what the task requires» на эту исследовательскую программу — не тратить время на:

- ещё один MiniGrid layout сверх уже покрытых сценариев дефицита;
- ещё один trust-threshold sweep (закон дальности W2 уже закрыт 6/6 EXACT, не трогать);
- улучшение success на +1% без изменения FAR/PAF/Revocation Latency — эти три метрики являются точкой всей программы, не success;
- ещё одну ручную semantic tag поверх уже закрытого W10-словаря;
- ещё один вариант broadcast cadence (K* уже выведен из framework, `docs/MEMORY_IMPLEMENTATIONS_REPORT_2026-05-18.md` §7.7);
- расширение category-theory appendix до появления load-bearing algebra;
- риторику про «коллективное сознание» — прямо запрещено roadmap 12.06 §6 п.5, остаётся в силе и для authority-фронтира;
- реальных роботов до полной LLM-валидации (Sprint 11 должен закрыться раньше, чем начинается разговор про embodiment).

---

**Файл:** `docs/FRONTIER_MEMORY_AUTHORITY_2026-08-07/07_ROADMAP_SPRINTS.md`
