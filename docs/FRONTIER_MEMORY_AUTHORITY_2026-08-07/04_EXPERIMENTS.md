# Десять экспериментов: E1–E10

**Родитель:** [`01_FORMAL_MODEL.md`](01_FORMAL_MODEL.md), [`02_THEOREMS.md`](02_THEOREMS.md), [`03_METRICS.md`](03_METRICS.md). Каждый эксперимент — независимый deliverable (дисциплина серии, `docs/FRONTIER_ROUTE_WARP_2026-07-02.md` и др.): остановка после любого сохраняет ценность предыдущих. **Ничего из этого не прогнано** — это регистрируемые протоколы и гипотезы, ожидаемые числа отсутствуют намеренно (заполняются после прогонов, дисциплина `docs/SERIES_VERDICTS.md`: предсказания на диск до исполнения).

Приоритет и порядок реализации — в [`07_ROADMAP_SPRINTS.md`](07_ROADMAP_SPRINTS.md). E1–E3 — критический путь (go/no-go программы, `README.md` §6). E4–E6 — очень желательны. E7–E10 — после подтверждения go/no-go, включают LLM-субстрат.

---

## E1 — Social Amplification (критический путь)

**Гипотеза.** Обычные shared/vector memory-системы увеличивают confidence в ложном claim при повторной social retransmission; provenance-aware authority-протокол — нет.

**Сценарий.**
```
Agent A receives one false observation X.
A tells B.
B tells C.
C tells D.
D tells A.  (опционально: замыкание цикла — самый жёсткий тест laundering)
```

**Плечи сравнения:**
1. shared context (naive concatenation);
2. vector memory (embedding retrieval, без provenance);
3. naive graph merge (union по счётчику source);
4. source-count trust (доверие ∝ числу разных agent_id, БЕЗ различения origin/transport);
5. полный provenance-aware authority-протокол (`E = F(O(m))`, §1 `02_THEOREMS.md`).

**Метрики:** `authority(claim, hops)` по каждому плечу; PAF (`03_METRICS.md` §4); n_eff.

**Ожидаемая форма результата (не число — форма кривой):** плечи 1–4 показывают рост authority с числом hops (или как минимум не-убывание); плечо 5 — плоская кривая, PAF ≈ 1.

**Acceptance / registered prediction:** PAF плеча 5 остаётся в [0.9, 1.1] по всем k∈{1,...,5} hops; плечи 1–3 показывают статистически значимый рост (непересекающиеся CI между k=1 и k=5).

**Что делает эксперимент честным, а не self-fulfilling:** ground truth claim X зафиксирован как ложный ДО прогона (аналог blinded fault injection из TAE-трека); авторитет измеряется evaluator-side, не self-report агентов.

---

## E2 — Independent Corroboration (критический путь)

**Гипотеза.** В противоположность E1: независимое подтверждение (не ретрансляция) реально увеличивает authority — иначе система вырождена в «никому не доверять» (провал liveness, Theorem 3).

**Сценарий, продолжающий E1:**
```
A observes X.  A → B → C.        (authority не растёт — по E1)
D independently observes X.       (authority должна вырасти)
E independently observes X.        (должна вырасти ещё)
```

**Метрики:** `authority(claim, n_independent_confirmations)` — должна монотонно расти к 1 (Theorem 3); сравнение наклона этой кривой против плоской кривой E1 — это и есть **Figure 2** (`06_FIGURES_AND_TABLES.md`).

**Acceptance:** authority строго растёт при добавлении каждого независимого подтверждения (не просто «не убывает» — иначе неотличимо от плеча E1 при малом noise).

**Почему этот эксперимент обязателен именно вторым:** без него E1 доказывает только «система параноидальна», не «система различает источник confirmation». E1+E2 вместе — единственная пара, которая демонстрирует разницу между count-based и origin-based authority.

---

## E3 — Stale Truth → False Belief (критический путь)

**Гипотеза.** Мир меняется; устаревший memory item продолжает распространяться у бейзлайнов дольше, чем допускает authority-протокол.

**Сценарий:**
```
Episode 0:    bridge = open
Episode 20:   bridge = closed   (ground truth меняется, не сообщается агентам явно)
Episode 21+:  старый certificate "bridge=open" продолжает циркулировать
```

**Плечи:** raw history; vector RAG; recency-only RAG; shared graph (без гейта); trust-only; staleness-only; полный authority-протокол.

**Метрики:**
```
P(unsafe action)                          — доля действий, основанных на invalid claim
Revocation Latency L_R (03_METRICS.md §3)
stale authority area = Σ_t A(m,t)·1[m is false]   — интеграл «сколько суммарно недопустимого authority прожило»
```

**Acceptance:** `L_R(full protocol) < L_R(staleness-only) < L_R(raw/vector/shared без гейта)`, и `stale authority area` минимальна у full protocol без просадки task success (иначе это не revocation, а просто отказ доверять новому тоже).

**Связь с уже закрытым:** прямое расширение S2 (`docs/FRONTIER_UCSM_2026-07-27.md`: world-clock необходим, недостаточен — rob −7.4%, independent 12/12 всё равно выигрывает) на произвольный claim, не только на place-record.

---

## E4 — Role-dependent Knowledge (receiver-specific authority)

**Гипотеза.** Authority — receiver- и role-specific: один и тот же claim может быть `ADMITTED` для одной роли и `REVOKED`/never-admitted для другой, при равном evidential score E.

**Сценарий.** Роли: Scout / Carrier / Fragile / Fast (переиспользует predator-prey асимметрию воплощений из UCSM Фазы 1, `docs/FRONTIER_SONG_GRAMMAR_2026-07-25.md` §5.1). Claim: «маршрут A проходим». Причинная полезность:
```
U_scout > 0
U_fragile < 0     (тот же маршрут опасен хрупкому агенту)
```

**Метрики:** `A_i(m)` vs `A_j(m)` при `E_i(m) = E_j(m)` (равная evidential admissibility, разное решение об authority) — прямая демонстрация четвёртого свойства go/no-go (`README.md` §6.4).

**Acceptance:** истинностное значение claim (E) идентично между ролями; admission-решение (A) статистически различается; regret агента, которому claim был неверно admitted (fragile, следующий за scout-маршрутом), измеримо выше, чем у агента с корректным role-gate.

---

## E5 — Utility Estimator: causal vs replay-calibrated

**Гипотеза.** Интервенционный estimator (§4 `01_FORMAL_MODEL.md`, обученный на randomized memory masking) даёт admission-решения не хуже replay-калиброванного UE1 на среде, где replay доступен, и остаётся валидным там, где replay недоступен (стохастический/LLM субстрат), в отличие от UE1.

**Протокол.** Сначала: ground-truth intervention environment (детерминированный grid, как в UE1) — получить `τ_true` через exact replay для каждого memory item. Затем сравнить оценщики:
1. heuristic importance (частота/recency);
2. LLM-rated importance (subjective);
3. текущий UE1 replay-estimator (Spearman 0.989 vs oracle — уже измерено);
4. новый causal intervention estimator (обучен на randomized Z=0/1 масках, §4 `01_FORMAL_MODEL.md`).

**Метрики:** Spearman(τ̂, τ_true); calibration curve; sign accuracy `P(sign τ̂ = sign τ_true)`; regret при admission по τ̂; admission precision/recall (`03_METRICS.md` §2).

**Acceptance (двухчастная регистрация):**
- **E5.1** (детерминированный grid, replay доступен): causal estimator в пределах ε от UE1 по Spearman/sign accuracy — показывает, что интервенционный подход не теряет качество там, где replay работает.
- **E5.2** (стохастический/LLM субстрат, replay НЕ доступен): causal estimator даёт sign accuracy значимо выше heuristic/LLM-rated baselines — показывает необходимость интервенционного подхода там, где UE1-подход в принципе неприменим.

**Честная развязка, если E5.2 не проходит:** это будет означать, что randomized intervention на LLM-субстрате слишком дорога/шумна для практического обучения — сама по себе публикуемая находка о границах применимости причинного admission на нетривиальном субстрате.

---

## E6 — Authority Gating Ablation

**Гипотеза.** Каждый из трёх компонентов admission-критерия (`E`-гейт, `U`-гейт/LCB, structural exception) необходим — убирание любого ухудшает либо safety (FAR/PAF), либо liveness (Authority Recall), причём разные абляции ломают РАЗНЫЕ свойства (не одно и то же).

**Дизайн — полный факториал:**
```
No gate (baseline: authority = admitted at RECEIVED)
E only
U only
E + U
E + provenance (origin-aware E, без LCB на U)
E + U + provenance
E + U + provenance + structural exception
FULL (все компоненты §4 01_FORMAL_MODEL.md)
```

**Метрики по каждой ячейке:** FAR, Authority Precision, Authority Recall, PAF, Revocation Latency, task success/cost.

**Acceptance / что доказывает эксперимент:** это главная защита от рецензентской атаки «всё улучшение пришло просто от freshness filter» (E1/E3-эффект без нового понятия authority) или «просто от utility-гейта, который уже есть в S3». Регистрируемое предсказание: `E only` закрывает большую часть E1/E3-эффекта (staleness/amplification — по существу temporal/provenance проблема), но не закрывает E4 (role-dependent authority требует U-гейта); `U only` не даёт safety-гарантии Theorem 1 (PAF может расти без E-гейта, потому что utility не видит provenance); только `FULL` проходит весь набор из четырёх go/no-go свойств одновременно.

---

## E7 — Structural Schema Formation (relational, не similarity-based)

**Гипотеза.** Relational schema (роли + relation + precondition + effect, §3 `01_FORMAL_MODEL.md`) переносится между мирами с разной surface-формой, но одинаковой causal-структурой, лучше, чем content-similarity схемы (текущий LCS/graph-matching из G1).

**Мир.** Семейство задач с одним структурным паттерном (bottleneck → passage → resource) под разными поверхностными признаками:
```
World A:  red door   → hallway → water
World B:  blue gate   → tunnel  → charger
World C:  rock opening → canyon  → food
```

**Плечи:** exact memory (без обобщения); embedding similarity; текущий graph analogy (G1, LCS/NW-alignment); relational schema (роли/precondition/effect, новое); relational schema + exception (структурное исключение, аналог EXCEPTION-операции UCSM, но на relational уровне).

**Метрики:** success на novel-world holdout (тот же протокол, что и во всей UCSM-серии — «фитнес только на свежих картах», `docs/FRONTIER_SONG_GRAMMAR_2026-07-25.md` §2а); биты схемы; false-transfer rate (перенос сработал, но структурно неверно).

**Acceptance:** relational schema даёт статистически значимо больший perfect-transfer rate на holdout при равном или меньшем бюджете бит, чем G1-baseline; следствие — Songline перестаёт быть «сжатым описанием маршрута» и становится «параметризованной причинной структурой» (см. `docs/FRONTIER_SONG_GRAMMAR_2026-07-25.md` §5, пункт 6, «полный π_mem»).

---

## E8 — Exceptions под relational schema

**Гипотеза.** Relational schema, дополненная explicit EXCEPTION (не overwrite, не усреднение), даёт лучший context-conditioned accuracy, чем альтернативы, при появлении context, где обычно верное правило ложно.

**Сценарий.** Схема `narrow corridor → safe resource route` обычно верна, но ложна при `carrying_large_object = true`.

**Плечи:** overwrite (новое правило замещает старое); average (взвешенное смешение); separate memory (два независимых, несвязанных правила без явной exception-связи); exception (текущий механизм UCSM); split schema (создание новой под-схемы по контексту).

**Метрики:** catastrophic overwrite rate (доля случаев, где верное общее правило было безвозвратно испорчено конфликтующим частным случаем); context-conditioned accuracy (правильный выбор правила в зависимости от контекста).

**Acceptance:** exception и split-schema доминируют overwrite/average по context-conditioned accuracy при сопоставимом бюджете бит; между exception и split-schema — открытый вопрос, разрешаемый эмпирически (split может быть дороже по битам, но точнее).

---

## E9 — Full LLM Society (headline experiment)

**Гипотеза.** На реалистичном (не grid) LLM-субстрате все три Theorem одновременно держатся: shared-memory LLM-агенты катастрофически распространяют одну ложную belief; authority-gated агенты сдерживают распространение при сохранении полезного transfer.

**Дизайн.** 4–8 LLM-агентов (не 1–3 — нужна достаточная глубина графа ретрансляции для E1-эффекта), 50–200 эпизодов, каждый со своей приватной persistent memory, ролью, коммуникацией, собственным authority-state.

**Плечи (полный набор — переиспользует таксономию roadmap 12.06 и добавляет authority-слой):**
```
Raw context
Summary memory
Vector RAG
Shared vector memory
Per-agent vector memory
Naive shared graph (без gating)
Trust/staleness CSM (текущий runtime v1, без явного authority state machine)
Full authority CSM (этот фронтир)
```

**Метрики:** success; cumulative cost; FAR; stale-memory-induced actions; token cost; retrieval latency; memory size; PAF (social amplification на реальном LLM-обмене); Authority Precision/Recall; revocation recovery time после явного world-change эпизода.

**Acceptance:** Full authority CSM — единственное плечо, одновременно проходящее (а) низкий FAR под инъецированной misinformation, (б) успешный transfer полезного знания (task success не хуже Trust/staleness CSM), (в) контролируемый token/memory budget (не хуже 2× относительно raw, по аналогии с уже принятым стандартом B1: «raw не догоняет и при 4× бюджета»).

**Что делает этот эксперимент headline, а не ещё одним LLM-smoke:** это первая проверка authority-протокола НЕ на детерминированном гриде — валидирует, что весь §1–2 фронтира не является артефактом replay-доступности среды.

---

## E10 — Long Horizon

**Гипотеза.** Преимущество authority-протокола не деградирует (и в некоторых метриках растёт) с горизонтом эпизодов — важно, поскольку весь claim программы именно про **long-term** collective memory, а не про однократный обмен.

**Дизайн.** `H ∈ {10, 50, 100, 250, 500, 1000}` эпизодов, без сброса памяти между ними (в отличие от типичного benchmark reset — прямое требование, иначе это не long-term memory benchmark, ср. критику в roadmap 12.06 §2.3 п.2).

**Метрики как функция H:** task success; memory size; FAR; token cost; stale-decision rate.

**Acceptance:** memory size растёт сублинейно с H (продолжение X1: кросс-семейный кодбук, `docs/FRONTIER_UCSM_2026-07-27.md`); FAR не растёт с H (иначе накопление старых admitted certificate деградирует систему — прямая проверка того, что revocation действительно работает на длинном горизонте, не только в коротком controlled-сценарии E3).

---

## Сводная таблица: эксперимент → theorem/свойство → метрика → приоритет

| # | Название | Проверяет | Метрика | Приоритет |
|---|---|---|---|---|
| E1 | Social Amplification | Theorem 1 | PAF, n_eff | **critical** |
| E2 | Independent Corroboration | Theorem 3 | authority(n_independent) | **critical** |
| E3 | Stale Truth | Theorem 2 | Revocation Latency, FAR | **critical** |
| E4 | Role-dependent Knowledge | receiver-specificity | AP/AR по ролям | desirable |
| E5 | Utility Estimator | §4 формальной модели | Spearman/sign accuracy/calibration | desirable |
| E6 | Authority Gating Ablation | необходимость каждого гейта | FAR/AP/AR по ячейкам | desirable |
| E7 | Structural Schema Formation | ось S за пределами similarity | perfect-transfer rate на holdout | later |
| E8 | Exceptions | ось S, context-conditioned | catastrophic overwrite rate | later |
| E9 | Full LLM Society | всё сразу, LLM-субстрат | FAR + success + token cost | later (после E1–E6) |
| E10 | Long Horizon | устойчивость во времени | memory/FAR как функция H | later |

Порядок реализации и распределение по спринтам — [`07_ROADMAP_SPRINTS.md`](07_ROADMAP_SPRINTS.md).

---

**Файл:** `docs/FRONTIER_MEMORY_AUTHORITY_2026-08-07/04_EXPERIMENTS.md`
