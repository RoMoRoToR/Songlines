# Теоремы: non-amplification, revocation, liveness

**Родитель:** [`01_FORMAL_MODEL.md`](01_FORMAL_MODEL.md). Три свойства ниже — не «теоремы» в смысле замороженной серии (`docs/SONGLINES_V1_FREEZE.md` §6 запрещает `prove` кроме стандартных категорных лемм cocompleteness Pos в Part IV) — это **проверяемые invariant'ы дизайна**, регистрируемые до экспериментальной проверки, в духе `We claim` / `We show` из `docs/CLAIM_EVIDENCE_MATRIX.md`. Модальность здесь — **We claim** (архитектурное свойство, доказываемое конструктивно из определений §1 `01_FORMAL_MODEL.md`) плюс **We show** (эмпирическая проверка в [`04_EXPERIMENTS.md`](04_EXPERIMENTS.md)).

---

## Theorem 1 — Provenance non-amplification (safety, ось E)

**Формулировка.** Пусть evidential authority зависит от множества независимых первичных evidence origins:

```
E(m) = F(O(m))
```

где `O(m)` — множество независимых origin_ids, поддерживающих claim m (§2 `01_FORMAL_MODEL.md`). Если коммуникация создаёт только новые записи `provenance_parents` (транспортные рёбра, т.е. ретрансляция без нового наблюдения), но не новые `origin_ids`:

```
O(m') = O(m)   ⟹   E(m') = E(m)
```

**Следствие.** Произвольная глубина цепочки ретрансляции `A → B → C → D` не увеличивает evidential authority исходного claim, если ни один из hops не добавил независимое наблюдение.

**Почему это не тривиально для типичной memory-системы.** Naive shared/vector memory считает «источники подтверждения» по числу сообщений/агентов, а не по числу независимых origin. Именно этот механизм и есть корень уязвимости, найденной в свежей литературе про distributed truth в LLM multi-agent системах: одна ложная ключевая testimony обрушивает collective truth recovery с 72.5% до 14.17%, потому что ошибка, once accepted, дальше распространяется уже честными агентами как будто independent confirmation (см. [`08_RELATED_WORK.md`](08_RELATED_WORK.md)).

**Как проверяется.** [`04_EXPERIMENTS.md`](04_EXPERIMENTS.md) §E1 (Social Amplification) — прямое измерение `authority(claim, hops)` для нескольких архитектур; успех = authority не растёт с числом ретрансляций при отсутствии нового origin.

**Метрика для количественной проверки:** Provenance Amplification Factor, `03_METRICS.md` §4 — `PAF ≤ 1` для корректной системы без нового evidence.

---

## Theorem 2 — Bounded stale authority / revocation (safety, ось время)

**Формулировка.** Пусть evidential admissibility затухает экспоненциально с возрастом (закон дальности W2 переносится без изменений, только формально пере-adресован на уровень E, а не только на уровень merge-веса):

```
E_i(m, t) = E_i(m, t0) · e^(−λ(t − t0))
```

и запись остаётся в состоянии допускающем authority только при `E_i(m, t) ≥ τ_E`. Тогда существует конечный горизонт:

```
t_max = t0 + (1/λ) · ln( E_i(m,t0) / τ_E )
```

после которого запись НЕ может оставаться `ADMITTED` без нового подтверждающего наблюдения (переход в `EXPIRED` — обязателен по конструкции, не по эвристике).

**Следствие (no indefinite stale authority).** Ни одно foreign observation не может сохранять authority бесконечно без refresh evidence — независимо от того, сколько раз оно было использовано успешно в прошлом.

**Связь с уже измеренным.** Это формализация закона дальности `age_max(trust) = ln(trust·conf/τ)/α`, подтверждённого 6/6 EXACT в W2 (`docs/SERIES_VERDICTS.md` Часть II), но теперь как свойство authority-слоя, а не только merge-веса.

**Как проверяется.** [`04_EXPERIMENTS.md`](04_EXPERIMENTS.md) §E3 (Stale Truth → False Belief) — измеряется `Revocation Latency` (`03_METRICS.md` §3) против бейзлайнов (raw history, vector RAG, shared graph без гейта).

---

## Theorem 3 — Liveness (дополнение к safety)

**Проблема, которую закрывает.** Theorem 1 и 2 — чисто safety-свойства («плохое знание не получает authority»). Вырожденное решение, тривиально удовлетворяющее обоим, — «никому не верить вообще» (τ_E → 1, τ_U → ∞). Нужно доказать, что это НЕ то, что происходит.

**Формулировка (informal, требует калибровки на эксперименте, не выводится чисто аналитически как Theorem 1–2).** При:
- истинном и полезном claim m (τ(m) > 0 в смысле причинной полезности §4 `01_FORMAL_MODEL.md`);
- достаточном числе независимых validation opportunities (наблюдений/визитов);
- bounded estimator error (σ_i(m) не растёт неограниченно с числом наблюдений — стандартное условие консистентности estimator'а);
- стабильном (или предсказуемо дрейфующем в пределах закона старения) состоянии мира;

вероятность admission стремится к единице с ростом числа независимых подтверждений:

```
P(A_i(m) = ADMITTED)  →  1   при  n_independent(m) → ∞
```

**Статус.** В отличие от Theorem 1 (чисто конструктивная, следует из определения `E = F(O(m))`) и Theorem 2 (следует из формы затухания, уже измеренной в W2), Theorem 3 — эмпирическое утверждение о работе конкретного admission-критерия (§4 `01_FORMAL_MODEL.md`) и оценщика Û. Она **не гарантирована архитектурой автоматически** — если τ_U или β откалиброваны слишком консервативно, liveness может не выполняться даже при честном claim. Это именно та часть, ради которой нужен experiment §E2 (Independent Corroboration), а не только математический вывод.

**Как проверяется.** [`04_EXPERIMENTS.md`](04_EXPERIMENTS.md) §E2 — кривая `authority(claim, n_independent_confirmations)` должна монотонно расти к 1, в отличие от кривой по числу ретрансляций (Theorem 1), которая должна быть плоской.

---

## Как safety и liveness совмещаются: два разных графика с одной осью X

Central figure программы (детали — [`06_FIGURES_AND_TABLES.md`](06_FIGURES_AND_TABLES.md) Figure 2):

```
authority
   │                                    independent confirmations (liveness, Th.3)
   │                                   ↗
   │                              ↗
   │                         ↗
   │────────────────────  (flat: PAF≈1, Th.1)
   │  retransmissions (без нового evidence, safety)
   └──────────────────────────────────────────────► n
```

Если обе кривые получаются как предсказано — это единственная убедительная демонстрация того, что authority-протокол не просто «более параноидальная» версия существующей памяти, а различает **источник** confirmation, не просто его частоту.

---

## Четыре результата, которые нужны минимально (go/no-go программы)

Повторяется из `README.md` §6 для полноты этого документа — это критерий, по которому вся программа принимается или честно закрывается как negative result:

1. **Social repetition does not create evidence.** `A→B→C→D ⇏ authority↑` (Theorem 1).
2. **Independent evidence does.** `e_A + e_B + e_C ⇒ authority↑` (Theorem 3).
3. **Authority is revocable within bounded latency.** `true@t1 ⇏ authoritative@t2` без refresh (Theorem 2), и latency меньше, чем у raw/vector/shared-graph бейзлайнов.
4. **Authority is receiver-specific.** `A_i(m) ≠ A_j(m)` для одного и того же factual m при разных ролях/состояниях i, j — это свойство не сформулировано как отдельная теорема (оно тривиально следует из role_conditions/state_conditions в §2 `01_FORMAL_MODEL.md`), но обязательно к эмпирической демонстрации в [`04_EXPERIMENTS.md`](04_EXPERIMENTS.md) §E4.

Если из четырёх свойств подтверждается меньше трёх — framing «authority protocol» как отдельного принципа не работает, и это тоже честный публикуемый результат в дисциплине серии (`docs/SERIES_VERDICTS.md`: «самые информативные результаты серии — именно честные провалы»).

---

**Файл:** `docs/FRONTIER_MEMORY_AUTHORITY_2026-08-07/02_THEOREMS.md`
