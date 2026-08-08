# Related work: позиционирование против свежей литературы

**Родитель:** [`README.md`](README.md), [`02_THEOREMS.md`](02_THEOREMS.md). **Важно перед подачей:** ссылки и цитаты ниже взяты из обсуждения (устный обзор, не проверенный этой сессией напрямую по первоисточникам) — перед включением в реальную статью нужен независимый verification pass по каждой ссылке (номера arXiv, точные claims, даты), тот же принцип, что уже применяется в проекте («Related work: позиционирование; сверить ссылки перед подачей», `docs/FRONTIER_UCSM_2026-07-27.md` §4).

---

## 1. Почему простого «мы делаем умную память» уже недостаточно

Поле уже занято достаточно плотно: A-MEM динамически организует и связывает memories; AgeMem и Memory-R1 обучают агентов самим выполнять memory operations (store/merge/drop как policy, не hand-crafted правила — прямая параллель с UCSM Phase 2 bandit-контроллером, но там LLM/RL учится на operations, а не на formation-matrix); G-Memory строит многоуровневую память специально для multi-agent систем. Все три — про **что запоминать и как организовывать**, не про **кому разрешено действовать по чужому знанию**. Это и есть свободная ниша.

## 2. Admission/gating сам по себе — уже занятая территория, нужно точнее

Нельзя продавать «нужно проверять чужое testimony» как абсолютно новую идею — это уже частично закрыто:

- **A-MAC** — adaptive memory admission control. Пересекается с S3 (`docs/FRONTIER_UCSM_2026-07-27.md`) почти буквально по названию механизма («admission control»). Отличие, которое нужно чётко сформулировать в статье: A-MAC (судя по названию и позиционированию) — adaptive threshold/policy для решения admit/reject, но не строит явный provenance DAG с различением origin/transport (Theorem 1) и не формализует authority как отдельную многосостояночную FSM, отделённую от evidential score.
- **MemGate** — рассматривает memory retrieval как trust boundary. Близко к идее «коммуникация ≠ authority», но, судя по формулировке, trust boundary — это retrieval-time gate (пропускать/не пропускать в контекст), а не persistent authority state с revocation. Ключевое отличие: у нас authority — не одноразовое retrieval-решение, а состояние с историей переходов (`ADMITTED → CONTESTED/SUPERSEDED → REVOKED`), аудируемое post-hoc.
- **Collaborative Memory** — занимается provenance и asymmetric access control. Ближайший конкурент по слову «provenance». Отличие: access control там, судя по всему, про то, КТО может читать/писать (permission model), не про то, СКОЛЬКО evidential weight несёт ретранслированное свидетельство (Theorem 1 — это не access control, это epistemic accounting).
- **PBRC (Preregistered Belief Revision Contracts)** — концептуально самая близкая работа: явно разделяет открытую коммуникацию и допустимое изменение belief. Это практически тот же тезис, что и центральный тезис этого фронтира («sharing information ≠ sharing authority»), сформулированный с других слов («belief revision contract»). **Формулировка «мы первые поняли, что testimony надо проверять» будет слишком широкой claim** — нужно чётко указать, что отличается: (а) provenance DAG с origin/transport различением и proof-обязанным non-amplification (Theorem 1) — не просто contract на пересмотр belief, а конкретный механизм против social amplification через циклы ретрансляции; (б) receiver-specific, role-conditioned authority (не единая belief revision policy на агента, а per-certificate, per-role state); (в) integration именно с long-horizon collective **task** memory (route/place transfer), не generic belief update.

## 3. Мотивирующая эмпирика: проблема не искусственная

Работа про **distributed truth в LLM multi-agent системах** (позиционируется как август 2026) показывает, что одна ложная ключевая testimony обрушивает collective truth recovery с 72.5% до 14.17%, причём ошибочная информация затем принимается и распространяется уже честными агентами. Это прямая, независимо полученная эмпирическая мотивация Theorem 1 (`02_THEOREMS.md` §1) — если verified, эту цифру стоит процитировать в Introduction будущей статьи как «проблема, которую решает non-amplification», а не как «related work, которую мы улучшаем» — это скорее подтверждение существования проблемы, чем конкурирующий метод.

**Обязательно перед использованием:** проверить точный arXiv id, дату публикации, точную формулировку чисел (72.5%/14.17%) — эти цифры пришли из устного пересказа в разговоре, не из независимого чтения первоисточника этой сессией.

## 4. Как выглядит таблица позиционирования (шаблон для будущей статьи)

| Работа | Provenance DAG (origin≠transport) | Receiver-specific authority | Persistent revocable state | Causal (не replay-only) utility | Multi-agent task memory |
|---|:-:|:-:|:-:|:-:|:-:|
| A-MEM | — | — | — | — | — |
| AgeMem / Memory-R1 | — | — | частично (learned ops) | — | частично |
| G-Memory | — | частично (multi-agent) | — | — | ✓ |
| A-MAC | частично (adaptive gate) | — | — | — | — |
| MemGate | — (trust boundary, не DAG) | — | — | — | — |
| Collaborative Memory | частично (access control) | частично | — | — | ✓ |
| PBRC | — | — | частично (revision contract) | — | — |
| **Этот фронтир** | ✓ (Theorem 1) | ✓ (E4, role_conditions) | ✓ (FSM §5 `01_FORMAL_MODEL.md`) | ✓ (§4, randomized intervention) | ✓ (наследует UCSM/CSM runtime) |

**Эта таблица — гипотеза расстановки, не проверенный факт** про конкурентов (колонки для чужих работ заполнены по смыслу названий/пересказа из обсуждения, не по чтению статей). Перед подачей — обязательный verification pass: прочитать каждую работу и перепроверить каждую клетку.

## 5. Порядок Related Work в будущей статье (не единственно верный, но рекомендуемый)

1. **Multi-agent collective memory architectures** (A-MEM, AgeMem, Memory-R1, G-Memory) — что делают, почему они не отвечают на вопрос authority.
2. **Admission/trust/access-control layers** (A-MAC, MemGate, Collaborative Memory, PBRC) — самые близкие конкуренты, детальное отличие по таблице §4.
3. **Distributed truth / misinformation propagation in LLM societies** (distributed-truth paper) — эмпирическая мотивация масштаба проблемы, не конкурирующий метод.
4. **Foundation this work builds on** — собственная серия (UCSM/CSM/Song Grammar), явно как prior work, не related work в смысле competitors.

---

**Файл:** `docs/FRONTIER_MEMORY_AUTHORITY_2026-08-07/08_RELATED_WORK.md`
