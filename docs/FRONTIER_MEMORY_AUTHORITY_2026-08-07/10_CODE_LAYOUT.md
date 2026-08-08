# Планируемая раскладка кода: `authority_memory/` поверх Songlines Runtime v1

**Родитель:** [`01_FORMAL_MODEL.md`](01_FORMAL_MODEL.md), [`07_ROADMAP_SPRINTS.md`](07_ROADMAP_SPRINTS.md). **Это только план раскладки, без кода** — ни один файл ниже не создаётся сейчас. Написание кода начинается со Sprint 1 (`07_ROADMAP_SPRINTS.md`), после явного запроса на реализацию.

---

## 1. Принцип: новый пакет НЕ переписывает замороженный runtime

Замороженный runtime v1 живёт в `songlines/` (`record.py`/`analogy.py`/`alignment.py`/`runtime.py`/`config.py`, этап 6 рефактора, `docs/SONGLINES_V1_FREEZE.md` §1) плюс тонкие shim-реэкспорты в `experiments/song_grammar/{runtime,ucsm}.py`. Ни один инвариант этого слоя не трогается — тот же принцип, что уже применялся при добавлении Song Grammar над CSM и CSM над Phase 1–4 (`songline_drive/collective_*.py` не переписывался при появлении `peer_memory/`, `distributed_memory/`, `songlines/`).

Новый пакет — **надстройка**, читает существующие datatypes (`songlines.record`, `songlines.analogy.Certificate/Schema/Song`) и оборачивает их, не заменяет.

```
authority_memory/            ← НОВЫЙ пакет, на уровне songline_drive/ и songlines/
    __init__.py
    certificate.py             # MemoryCertificate (01_FORMAL_MODEL.md §2), Claim (§3)
    provenance_graph.py         # origin_ids vs provenance_parents, n_eff (03_METRICS.md §5)
    authority_state.py            # AuthorityState enum + transition table (01_FORMAL_MODEL.md §5)
    evidence.py                     # E_i(m,t) — evidential admissibility, закон дальности (Theorem 2)
    causal_utility.py                 # randomized intervention labels, τ̂_θ estimator (§4)
    admission.py                        # LCB-критерий, три гейта (E ∧ U ∧ V) → authority transition
    revocation.py                        # EXPIRED/SUPERSEDED/CONTESTED → REVOKED, Revocation Latency
    schema.py                             # relational schema (роли/precondition/effect), E7
    schema_induction.py                     # candidate relation extraction → schema proposal (E7)
    llm_semantics.py                          # LLM extractor/query-former/schema-proposer (Sprint 10),
                                                #   строго proposal-only — см. §3 ниже
    metrics.py                                  # FAR, Authority Precision/Recall, PAF (03_METRICS.md)
```

```
experiments/authority_memory/     ← эксперименты E1–E10, параллельно существующим
                                    #   experiments/song_grammar/, experiments/peer_memory/ и т.д.
    exp_e1_social_amplification.py
    exp_e2_independent_corroboration.py
    exp_e3_stale_truth.py
    exp_e4_role_dependent.py
    exp_e5_causal_utility_eval.py
    exp_e6_admission_ablation.py
    exp_e7_relational_schema.py
    exp_e8_exceptions.py
    exp_e9_llm_society.py
    exp_e10_long_horizon.py
    corruption_kit.py            # шесть типов controlled corruption (05_BENCHMARK_CORRUPTIONS.md §2)
    world_history.py             # long-horizon world-с-историей каркас (05_BENCHMARK_CORRUPTIONS.md §5)
```

## 2. Как datatypes соотносятся с существующими

| Новый datatype | Оборачивает / читает | Не заменяет |
|---|---|---|
| `MemoryCertificate` | `songlines.record` (record type `m=(G,C,E,U,P,T,R,F,A)`) | Record type runtime v1 остаётся как есть; certificate — дополнительный слой поверх (composition, не inheritance по умолчанию — решается в Sprint 1 по факту сигнатур `songlines/record.py`) |
| `evidence.py: E_i(m,t)` | существующий trust-EMA + staleness-гейт (`songline_drive/belief_fusion.py TemporalDecayEngine`) | Не переизобретает закон затухания — тот же `age_max = ln(trust·conf/τ)/α`, применённый на уровне authority, не только merge-веса |
| `admission.py` | S3 admission control (карантин + валидация на визите, `docs/FRONTIER_UCSM_2026-07-27.md` §S3) | Не заменяет S3-механику — формализует её как явную FSM с логируемыми `AuthorityDecision`, где раньше был неявный побочный эффект admission-функции |
| `schema.py` | `songlines.analogy.Schema/Song`, `Certificate` (graph-matching G1) | Не заменяет LCS/NW-alignment — добавляет relational layer (роли/precondition/effect) НАД существующим structural matching, как альтернативное/дополнительное `structural_relation` |
| `causal_utility.py` | UE1 replay estimator (`experiments/song_grammar/exp_ue1_utility_estimator.py`) | UE1 остаётся как baseline в E5 (§E5.1 сравнение на детерминированном grid), не удаляется |

## 3. Жёсткая архитектурная граница: LLM — proposal, не authority

Прямое требование из обсуждения, критично зафиксировать в коде, не только в документации: `llm_semantics.py` может ТОЛЬКО:
- извлекать structured `Claim` из natural-language observation (extractor);
- предлагать candidate relational schema (proposer);
- формулировать retrieval query из task description (query former);
- предлагать high-level decision из retrieved certificates.

`llm_semantics.py` НЕ МОЖЕТ:
- напрямую устанавливать `evidence_score`, `utility_mean/std`, или `authority_state` — эти поля пишутся только через `evidence.py`/`causal_utility.py`/`admission.py`, независимо от того, что предложил LLM-слой;
- обходить `provenance_graph.py` — LLM не решает, что является independent origin, это чисто структурное свойство транспортного графа.

Практически: `MemoryCertificate`, приходящий из `llm_semantics.py`, всегда создаётся в состоянии `RECEIVED`/`QUARANTINED` (`01_FORMAL_MODEL.md` §5) — точно так же, как любой другой certificate, независимо от источника. Тест на соблюдение границы: unit-тест, который мокает LLM так, чтобы он пытался напрямую установить `authority_state = ADMITTED`, и проверяет, что `admission.py` это игнорирует/перезаписывает.

## 4. Порядок реализации файлов (сопоставление со спринтами `07_ROADMAP_SPRINTS.md`)

| Файл | Спринт |
|---|---|
| `certificate.py`, `authority_state.py` | Sprint 1 |
| `provenance_graph.py`, `metrics.py` (PAF, n_eff) | Sprint 2 |
| `experiments/authority_memory/exp_e1_social_amplification.py`, `corruption_kit.py` (частично: C — provenance laundering) | Sprint 3 |
| `revocation.py`, `exp_e3_stale_truth.py`, `corruption_kit.py` (A — staleness) | Sprint 4 |
| `exp_e2_independent_corroboration.py` | Sprint 5 |
| `exp_e4_role_dependent.py`, `corruption_kit.py` (D — role-dependent) | Sprint 6 |
| `causal_utility.py` (сбор labels) | Sprint 7 |
| `causal_utility.py` (estimator), `exp_e5_causal_utility_eval.py` | Sprint 8 |
| `schema.py`, `schema_induction.py`, `exp_e7_relational_schema.py` | Sprint 9 |
| `llm_semantics.py` | Sprint 10 |
| `exp_e9_llm_society.py`, `world_history.py` | Sprint 11 |
| `exp_e10_long_horizon.py` | Sprint 12 |
| остальные бейзлайны для Table 1 (`06_FIGURES_AND_TABLES.md`) | Sprint 13 |

## 5. Тестовая дисциплина (наследуется от runtime v1)

Замороженный runtime держит invariant suite `songlines/tests` (11/11, `docs/REVIEW_RISK_REGISTER.md`). Новый пакет заводит параллельный `authority_memory/tests/` с минимальным набором с первого спринта:
- unit-тест на state machine (`01_FORMAL_MODEL.md` §5): каждый разрешённый/запрещённый переход;
- unit-тест на non-amplification на синтетическом графе транспорта без нового origin (прямая проверка Theorem 1 на уровне кода, отдельно от полноценного E1-эксперимента);
- unit-тест на границу LLM (§3 выше).

---

**Файл:** `docs/FRONTIER_MEMORY_AUTHORITY_2026-08-07/10_CODE_LAYOUT.md`
