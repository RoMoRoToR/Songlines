# План будущей статьи (outline — статья НЕ собирается сейчас, только структура)

**Родитель:** [`README.md`](README.md), все предыдущие документы папки. Эта статья — **не** переработка текущей `papers/collective_semantic_memory/` / `papers/three_papers/paper3_collective_memory.tex`. Замороженная серия (`docs/SONGLINES_V1_FREEZE.md`) остаётся как есть, как отдельная опубликованная работа. Эта — следующая, отдельная статья, для которой замороженный runtime v1 — foundation/prior work.

---

## 1. Главный claim (формулировка)

**Русский вариант:**
> Долговременная коллективная память — это не общий архив. Это распределённый протокол перехода от наблюдений и чужих свидетельств к знаниям, которым разрешено влиять на действие.

**English (для draft):**
> Long-term multi-agent memory is not primarily a storage or retrieval problem; it is an evidence-to-authority problem. Foreign memories should enter a receiver as testimony, not as action-authoritative knowledge. Authority is granted only after receiver-specific validation of provenance, current applicability, and causal decision value, while structural assimilation determines how the admitted record modifies existing knowledge.

## 2. Кандидаты заголовка

1. **Sharing Memory Is Not Sharing Authority: Receiver-Side Admission for Long-Horizon Multi-Agent Agents** — самый описательный, безопасный.
2. **From Testimony to Action Authority: Evidence-Gated Memory for Multi-Agent Agents** — сильнее подчёркивает pipeline (testimony → authority).
3. **Memory as an Authority Protocol for Multi-Agent AI** — самый широкий; требует самой сильной экспериментальной базы (весь E1–E10), не только Stage 0.

**Рекомендация на этом этапе:** держать вариант 1 или 2 как working title до прохождения go/no-go точки после Sprint 5 (`07_ROADMAP_SPRINTS.md`) — вариант 3 разблокируется только если E9 (Full LLM Society) подтверждает claim на реалистичном субстрате, не только на grid.

## 3. Структура (12 разделов)

### §1 Introduction
Проблема: shared memory conflates information transport с action authority. Мотивирующая эмпирика — distributed-truth paper (72.5%→14.17%, с оговоркой verification, `08_RELATED_WORK.md` §3). Три-four go/no-go свойства (`README.md` §6) как preview контрибьюций.

### §2 Problem formulation
Multi-agent testimony + persistent memory + nonstationarity. Формальная постановка: почему information availability ≠ action authority — не риторика, а формальное различие (§1 `01_FORMAL_MODEL.md`).

### §3 Memory certificates
`MemoryCertificate`, три оси E×U×S (`01_FORMAL_MODEL.md` §1–3). Явное разграничение с UCSM record type `m` — что наследуется, что добавляется.

### §4 Provenance and non-amplification
Provenance DAG (origin_ids vs provenance_parents), Theorem 1 (`02_THEOREMS.md` §Theorem 1). Central theoretical contribution раздела.

### §5 Receiver-specific authority
Authority state machine (`01_FORMAL_MODEL.md` §5), admission-критерий через LCB (§4), role/state-conditioned authority (E4).

### §6 Causal utility
Randomized memory intervention, отличие от replay-калиброванного UE1, admission через LCB(Û) (`01_FORMAL_MODEL.md` §4, E5).

### §7 Revocation and liveness
Theorem 2 (bounded stale authority) + Theorem 3 (liveness) — обе стороны safety/liveness баланса, явно с предупреждением про вырожденное решение «никому не верить» (`02_THEOREMS.md`).

### §8 Benchmark
Шесть типов controlled corruption (`05_BENCHMARK_CORRUPTIONS.md`), long-horizon протокол, evaluator ground truth отделённый от observable interface (продолжение Q/R/M/C методологии, не переизобретение).

### §9 Experiments
Controlled (E1–E8, grid) + LLM (E9–E10). Структура секции повторяет порядок `04_EXPERIMENTS.md`, но с реальными числами после прогонов.

### §10 Analysis
Failure modes, honest FAILs (если есть — дисциплина серии требует их публикации с механизмом, не скрытия).

### §11 Related work
По структуре `08_RELATED_WORK.md` §5 (multi-agent memory architectures → admission/trust layers → distributed-truth motivation → foundation).

### §12 Limitations
Open-world grounding, calibration качество LCB на LLM-субстрате, возможность collusion между несколькими лгущими агентами (не покрыто E1–E10 — один poisoner уже проверен в замороженной серии P1, но множественный coordinated adversary — открытый вопрос), масштаб N агентов сверх протестированного.

## 4. Что сохраняется из текущей статьи как background/foundation (НЕ передоказывается)

Прямо перечислено, чтобы Sprint 14/15 не тратили время на повторное доказательство уже закрытого (`docs/SERIES_VERDICTS.md`, `docs/CLAIM_EVIDENCE_MATRIX.md`):

- cadence result (bottleneck shift M↔C, K*);
- non-monotone communication (collective memory не монотонно лучше);
- minimal CSM (третий режим M×C);
- private-frame alignment (W7–W10, SE(2));
- two-axis formation (U×S, до появления E) и пять операций (MERGE/EXCEPTION/NEW_SCHEMA/REPEAT/DROP);
- learned formation controller (U2 bandit);
- replay utility estimator (UE1, Spearman 0.989) — переиспользуется как baseline в E5, не как central contribution новой статьи;
- receiver-side admission (S3) — это Stage 0 authority-протокола, формализуется дальше в §5 новой статьи, не изобретается с нуля;
- long-horizon результаты серии (bench30, resource accounting);
- VMAS, perception noise, adversarial provenance (P1) — все три остаются valid prior evidence, на которые новая статья ссылается, не повторяет.

## 5. Что НЕ делать в новой статье (перенесено из обсуждения, применимо к структуре)

- **Не** вводить ещё одну семибуквенную framework-метрику вместо/вместе с Q/R/M/C — Q/R/M/C остаётся companion diagnostic слоем, если вообще упоминается (см. `README.md` §5, разграничение с TAE-треком).
- **Не** делать category theory headline'ом раздела — appendix-уровень, если не появится composable algebra конкретно для certificates/admission operators (проверка: существует ли естественная категория `Testimony → Authority` с функториальными свойствами non-amplification; это отдельный research question, не гарантированный этим планом).
- **Не** заявлять «мы решили LLM multi-agent memory в общем виде» — прямое наследование `We do not claim` дисциплины серии (`docs/CLAIM_EVIDENCE_MATRIX.md` §«Явные We do not claim»).
- **Не** заявлять «коллективное сознание» ни в заголовке, ни в abstract — прямой перенос запрета из roadmap 12.06 §6 п.5.
- **Не** тащить все 14+ механизмов замороженной серии в main text новой статьи — она должна читаться как одна статья с одним центральным тезисом (authority ≠ information availability), не как продолжение «у нас уже 20 хороших идей».

## 6. Figures/Tables budget для новой статьи (тело, не appendix)

По аналогии с `docs/AAAI_COMPRESSION_PLAN.md` — заранее фиксировать бюджет, чтобы не разрастаться:

- **Figure 1** (pipeline, `06_FIGURES_AND_TABLES.md`) — обязательно в §3 или §5.
- **Figure 2** (social amplification vs independent corroboration) — headline figure, обязательно в §4 или §9, самая важная фигура всей статьи.
- **Figure 3** (staleness/revocation) — §7 или §9.
- **Figure 5** (LLM long-horizon, три панели) — §9, headline experiment.
- Figure 4 (utility calibration) — может уйти в appendix, если §6 (causal utility) окажется desirable, не critical частью финальной статьи.
- **Table 1** (method vs baselines) и **Table 2** (ablation) — обе в §9, обе обязательны в теле по аналогии с тем, как TAE-трек требует «Table 1 с blinded fault benchmark должна стать первым большим результатом, а не появляться только на стр. 6» — тот же принцип: главная таблица не может быть в конце.

---

**Файл:** `docs/FRONTIER_MEMORY_AUTHORITY_2026-08-07/09_PAPER_OUTLINE.md`
