# Фронтир: Memory as Authority Protocol — расшарить память ≠ расшарить право действовать

**Дата:** 2026-08-07
**Статус:** research design (Stage 0 = существующая `collective_semantic_memory` / UCSM серия закрыта и заморожена как foundation, см. `docs/SONGLINES_V1_FREEZE.md`, `docs/SERIES_VERDICTS.md`); этот фронтир — следующая программа НАД замороженным runtime, не его переделка.
**Родитель:** `docs/FRONTIER_UCSM_2026-07-27.md` (utility × analogy → операции памяти), `docs/FRONTIER_SONG_GRAMMAR_2026-07-25.md` (онтогенез памяти), Paper 2/3 серии (`papers/collective_semantic_memory/`, `papers/three_papers/paper3_collective_memory.tex`).
**Не трогает:** ни один инвариант замороженного runtime v1 (record type, две оси формирования, пять операций, provenance/admission/reservation/rollback, safety-слои). Новый слой строится НАД ним, как отдельный пакет.

---

## 1. Центральный тезис одним абзацем

Замороженная серия уже доказала: shared archive — не монотонное благо; receiver-side admission («measured, not told») переводит обмен из вредного в полезный (S1→S2→S3); utility отвечает за полезность, но не за истинность — эпистемическая ответственность лежит на provenance/validity-контракте, а не на utility-эстиматоре. Это уже почти сформулированный принцип, но он размазан по механизмам (trust-EMA, staleness-гейт, admission control, quarantine) вместо того, чтобы быть **одним архитектурным объектом**.

Следующий шаг — сделать этот принцип явным и проверяемым:

> **Sharing information must not imply sharing action authority.**
> Передача информации агенту не должна автоматически давать этой информации право влиять на его действия.

Формально: между «агент получил testimony» и «testimony может менять его persistent world model и планирование» стоит явный, аудируемый, receiver-specific шлюз — **authority admission**. Коммуникация остаётся почти свободной (транспорт дёшев); авторитет — дорогой и контролируемый ресурс.

## 2. Что уже есть в замороженном runtime и что достраивается

| Компонент | Уже есть (runtime v1) | Чего не хватает |
|---|---|---|
| Provenance | origin-bound, без ретрансляции (flip-links исключений) | **DAG независимых origins** — различать «3 независимых наблюдения» от «1 наблюдение, 3 ретрансляции» |
| Trust/staleness | EMA + age-гейт (закон дальности) | Не формализовано как единая величина **evidential admissibility E** |
| Utility | контрфактическая маргинальная полезность (UE1: Spearman 0.989 против oracle) | Только replay-калиброванный estimator; нет **интервенционного causal utility** на стохастическом субстрате |
| Admission | S3: карантин + валидация на визите | Нет явной **машины состояний авторитета** (QUARANTINED→PROVISIONAL→ADMITTED→REVOKED) с логируемыми переходами |
| Structural | MERGE/EXCEPTION/NEW_SCHEMA/REPEAT/DROP по LCS/graph-matching (G1) | Схемы — content-similarity клас­теры, не **relational schema с ролями/precondition/effect** |
| Safety | anchor consensus + commit-top1 + prefix-verify (fail-open 0.0014) | Не связано явно с эпистемическим слоем (E); это geometric safety, не evidential safety |

Три существующие оси **U** (utility) и структурный матчинг **S** уже независимы (B1 в `CLAIM_EVIDENCE_MATRIX.md`). Этот фронтир вводит третью, до сих пор implicit ось — **E** (evidential admissibility) — и формализует функцию перехода `(E, U, S) → (authority, formation operation)`.

## 3. Объект памяти нового поколения

Было (UCSM): `m = (G_m, C_m, E_m, U_m, Φ_m, P_m, F_m)` — граф/условия/эффекты/полезность/аналогии/провенанс/отказы.

Становится: **memory certificate** — тот же record, плюс явное поле авторитета и граф происхождения вместо скалярного provenance-тега. Полная схема — [`01_FORMAL_MODEL.md`](01_FORMAL_MODEL.md).

## 4. Карта документов

| Файл | Содержание |
|---|---|
| [`01_FORMAL_MODEL.md`](01_FORMAL_MODEL.md) | `MemoryCertificate`, оси E×U×S, машина состояний авторитета, вспомогательные datatypes (`Claim`, `ValidationEvent`, `AuthorityDecision`, `ActionRecord`) |
| [`02_THEOREMS.md`](02_THEOREMS.md) | Non-amplification (safety), bounded stale authority / revocation, liveness — с условиями и набросками доказательств |
| [`03_METRICS.md`](03_METRICS.md) | Новый vocabulary: False Authority Rate, Authority Precision/Recall, Revocation Latency, Provenance Amplification Factor |
| [`04_EXPERIMENTS.md`](04_EXPERIMENTS.md) | Десять экспериментов E1–E10: гипотеза, протокол, бейзлайны, метрики, ожидаемая таблица, go/no-go |
| [`05_BENCHMARK_CORRUPTIONS.md`](05_BENCHMARK_CORRUPTIONS.md) | Шесть типов controlled corruption (staleness, false testimony, provenance laundering, role-dependent validity, context exception, semantic aliasing) + long-horizon протокол |
| [`06_FIGURES_AND_TABLES.md`](06_FIGURES_AND_TABLES.md) | ASCII-макеты Figure 1–5, шаблоны Table 1–2 (заполняются после прогонов, не сейчас) |
| [`07_ROADMAP_SPRINTS.md`](07_ROADMAP_SPRINTS.md) | Sprint 1–15, приоритеты (critical/desirable/later), go/no-go критерий программы |
| [`08_RELATED_WORK.md`](08_RELATED_WORK.md) | Позиционирование против A-MEM, AgeMem, Memory-R1, G-Memory, A-MAC, MemGate, Collaborative Memory, PBRC, distributed-truth paper |
| [`09_PAPER_OUTLINE.md`](09_PAPER_OUTLINE.md) | Структура будущей статьи (12 разделов), кандидаты заголовка, формулировка главного claim |
| [`10_CODE_LAYOUT.md`](10_CODE_LAYOUT.md) | Планируемая раскладка нового пакета `authority_memory/` поверх существующего runtime (без кода — только план) |

## 5. Связь с двумя другими открытыми треками

- **TAE workshop reframing** (`songlines_qrmc_measurement_framework` / `qrmc_aaai27`) — отдельный, не связанный с этим фронтиром трек: там речь про валидность самого Q/R/M/C протокола как measurement instrument. Тот трек — про переформулировку уже существующей статьи под CFP конкретного workshop'а, без нового кода. Здесь фиксируем это явно, чтобы не путать два направления: Q/R/M/C остаётся companion-диагностикой ("read-only" measurement layer), не переиспользуется как central framework для authority-протокола (см. §29 из обсуждения — решение НЕ вводить ещё одну семибуквенную framework-метрику).
- **UCSM / Song Grammar** — этот фронтир не заменяет и не переоткрывает Phase 1 (детерминированный прототип), Phase 2 (bandit), Phase 3 (evolution грамматики). Authority-слой встраивается как надстройка над `ucsm.py`/`runtime.py`, точно так же как Song Grammar встраивался над CSM без переписывания нижних слоёв.

## 6. Критерий, по которому программа проверяется целиком (go/no-go)

Продолжаем эту программу дальше первых трёх спринтов только если ОДНОВРЕМЕННО подтверждаются четыре свойства (полностью в [`04_EXPERIMENTS.md`](04_EXPERIMENTS.md) §E1–E3, [`02_THEOREMS.md`](02_THEOREMS.md)):

1. **Social repetition does not create evidence** — `A→B→C→D` не увеличивает authority ложного claim (PAF ≈ 1).
2. **Independent evidence does** — независимое подтверждение увеличивает authority.
3. **Authority is revocable** — после смены мира устаревший authority снимается быстрее, чем у бейзлайнов (raw/vector/shared graph).
4. **Task success не рушится** — система не вырождается в «никому не верить» (liveness).

Если из четырёх выполняется меньше трёх — framing «authority protocol» неверен, и это тоже честный, публикуемый результат (в духе дисциплины серии: провалы регистраций публикуются с механизмом, см. `docs/SERIES_VERDICTS.md`).

---

**Файл:** `docs/FRONTIER_MEMORY_AUTHORITY_2026-08-07/README.md`
**Создан:** 07.08.2026, по итогам обсуждения «утилита ≠ истина, коммуникация ≠ авторитет» — следующая программа над замороженным Songlines Runtime v1.
