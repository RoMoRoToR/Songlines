# Формальная модель: MemoryCertificate, оси E×U×S, машина состояний авторитета

**Родитель:** [`README.md`](README.md). Расширяет record type UCSM (`m = (G_m, C_m, E_m, U_m, Φ_m, P_m, F_m)`, см. `docs/FRONTIER_UCSM_2026-07-27.md` §2) третьей осью и явной моделью авторитета.

---

## 1. Три независимые оси memory item

Формирование памяти уже различает **U** (полезность) и **S** (структурная аналогия) как независимые оси (B1, `CLAIM_EVIDENCE_MATRIX.md`). Этот фронтир вводит третью:

### E — Evidential admissibility
«Можно ли вообще доверять этому утверждению?» Зависит от provenance, возраста, world-version, надёжности источника, независимости evidence, наличия противоречий, собственных наблюдений receiver'а.

```
E_i(m, t) — скаляр в [0, 1], receiver-specific и time-dependent
```

### U — Decision utility (уже есть, переиспользуется без изменений)
«Стоит ли этой информацией пользоваться для задачи конкретного агента?» Контрфактическая маргинальная полезность из UCSM (`U_{i,ι}(m|M) = cost_i(M,ι) − cost_i(M∪{m},ι)`), либо её причинный аналог (см. §4 ниже).

### S — Structural assimilability (уже есть, переиспользуется без изменений)
«Куда информация попадает относительно существующей памяти?» Результат — одна из пяти операций UCSM: MERGE / EXCEPTION / NEW_SCHEMA / REPEAT / DROP.

**Функция перехода:**

```
(E, U, S) → (authority_state, formation_operation)
```

Ключевое разделение ответственности: **E отвечает за истинность/допустимость, U — за полезность, S — за структурную интеграцию.** Ни одна ось не может подменить другую — это и есть операционализация тезиса «utility ≠ truth» из уже закрытой S3 (`FRONTIER_UCSM` §6, «свидетельство ≠ доказательство»).

---

## 2. MemoryCertificate — новый центральный объект

Замена/расширение `m` из UCSM. Не переписывает record type runtime v1 — оборачивает его.

```
MemoryCertificate
    certificate_id          # уникальный id этого конкретного certificate

    claim                    # StructuredClaim (см. §3) — НЕ строка/blob

    origin_ids               # set[str] — первичные evidence-наблюдения,
                              #   на которые claim восходит (без учёта транспорта)
    provenance_parents        # set[str] — от кого получен ЭТОТ конкретный экземпляр
                              #   (транспортный граф, отдельно от origin)

    source_agent              # кто это certificate непосредственно передал
    receiver_agent             # None до получения; заполняется при приёме

    created_world_version      # версия мировых часов в момент наблюдения
    observed_at                 # тик наблюдения
    valid_until                  # None или тик, после которого валидность считается 0
                                  #   (age_max закон дальности, W2 — переносится без изменений)

    role_conditions              # dict — при каких ролях получателя claim применим
    state_conditions               # dict — при каких состояниях receiver'а применим

    evidence_score                  # E_i(m, t) на момент последней переоценки
    utility_mean                     # Û — точечная оценка причинной полезности
    utility_std                       # σ — неопределённость оценки (для LCB, см. §4)

    structural_relation                # MERGE | EXCEPTION | NEW_SCHEMA | REPEAT | DROP
    exception_ids                        # list[str] — известные контрпримеры

    authority_state                       # AuthorityState (см. §5)
```

Критическое разграничение: **`origin_ids` и `provenance_parents` — разные множества.** `provenance_parents` растёт при каждой ретрансляции (A→B→C→D даёт цепочку из 3 hops); `origin_ids` не растёт от ретрансляции — только от нового независимого наблюдения. Это разграничение — единственная техническая предпосылка Theorem 1 (non-amplification, [`02_THEOREMS.md`](02_THEOREMS.md) §1).

---

## 3. Claim — структурированное содержание, не строка

Одна из типичных ошибок memory-систем — держать content как неструктурированный текст, из-за чего provenance становится развитым, а содержание — нет. Минимальная структура (может быть triple или small graph):

```
Claim
    subject       # напр. "route_17"
    relation        # напр. "safe_for"
    object            # напр. "fragile_agent"
    conditions          # dict, напр. {"door_state": "open"}
```

Пример для UCSM song-грамматики: куплет-сигнатура (σ) уже структурирована (замкнутый пучок тегов); Claim здесь — обобщение той же идеи на произвольные relational факты, не только на топо-метрические куплеты.

---

## 4. Причинная полезность вместо replay-калибровки

Текущий UE1 (`docs/FRONTIER_UCSM_2026-07-27.md` §UE1) обучен на exact counterfactual replay — это valid ТОЛЬКО там, где replay доступен (детерминированный grid). Для authority-протокола на стохастическом/LLM субстрате нужна интервенционная версия.

**Определение (randomized memory intervention):**

```
Z_{i,m,t} = 1  если m доступно policy агента i в момент t
Z_{i,m,t} = 0  если m скрыто (masked)

τ_i(m, s) = E[Y | do(Z=1), s] − E[Y | do(Z=0), s]
```

**Сбор данных:** во время training для 5–10% eligible decisions — randomized 50/50 маскирование memory item, логировать `(s, m, r, y, Z)`. После обучить `τ̂_θ(s, m, r)` — уже без необходимости rollout на inference.

**Admission-критерий через lower confidence bound**, не через точечную оценку:

```
LCB_i(m) = Û_i(m) − β·σ_i(m)

A_i(m) = 1  ⟺  E_i(m) ≥ τ_E  ∧  LCB_i(m) ≥ τ_U  ∧  V_i(m) = 1
```

где V_i(m) — applicability/role-condition (0/1). Это заменяет текущий `admission по (Û > 0)` (S3-версия) на uncertainty-aware версию — необходимо, потому что на LLM-субстрате Û не имеет replay-точности grid-мира.

---

## 5. Машина состояний авторитета

Новый явный элемент архитектуры — не скалярный trust-вес, а состояние с логируемыми переходами.

```
RECEIVED
   │
   ▼
QUARANTINED   ← доступно reasoning layer, ЗАПРЕЩЕНО как durable action authority
   │
   ▼
PROVISIONAL   ← ограниченное использование (exploration/route-proposal), не пишет в persistent memory
   │
   ▼
ADMITTED      ← участвует в planning, может менять persistent semantic memory
   │
  ┌┴────────────┐
  ▼             ▼
CONTESTED    SUPERSEDED / EXPIRED
  │             │
  └──────┬──────┘
         ▼
      REVOKED
```

Семантика переходов:

| Переход | Условие | Кто логирует |
|---|---|---|
| RECEIVED → QUARANTINED | автоматически при приёме | receiver |
| QUARANTINED → PROVISIONAL | E_i(m) ≥ τ_E (базовая evidential admissibility, без utility) | receiver |
| PROVISIONAL → ADMITTED | LCB_i(m) ≥ τ_U ∧ V_i(m)=1 (полный admission-критерий §4) | receiver, на визите/использовании — «measured, not told», переносится из S3 без изменений |
| ADMITTED → CONTESTED | появилось несовместимое evidence (та же ConflictRuleSet, что в belief_fusion.py) | receiver |
| ADMITTED → SUPERSEDED | появилась более новая версия того же claim | receiver |
| ADMITTED/PROVISIONAL → EXPIRED | текущий тик > valid_until (закон дальности W2, не меняется) | receiver (пассивно, по таймеру) |
| CONTESTED/SUPERSEDED/EXPIRED → REVOKED | финальное снятие authority | receiver |

Каждый переход создаёт **`AuthorityDecision`** (см. §7) — это и есть аудируемый след, отвечающий на вопрос «почему агент поверил этому свидетельству».

---

## 6. ValidationEvent — след эмпирической проверки

```
ValidationEvent
    certificate_id
    receiver_id
    local_observation    # что receiver сам наблюдал в момент использования
    outcome               # успех/провал использования
    world_version
    support               # True/False — подтвердило ли собственное наблюдение claim
```

Это машинерия S3 (карантин → валидация на визите), формализованная как отдельный logged event, а не побочный эффект admission-функции.

---

## 7. AuthorityDecision — аудируемое решение

```
AuthorityDecision
    certificate_id
    previous_state
    new_state
    evidence_score      # E на момент решения
    utility_lcb            # LCB(Û) на момент решения
    reason                   # какое из трёх условий admission не прошло/прошло
    timestamp
```

## 8. ActionRecord — трассировка действий до evidence

Каждое memory-driven действие получает собственную запись:

```
ActionRecord
    action
    supporting_certificate_ids   # какие certificate обосновали это действие
    query
    selected_schema
```

Комбинация `ActionRecord.supporting_certificate_ids` + `AuthorityDecision`-цепочка + `provenance_parents`/`origin_ids` даёт **causal audit trail от действия до первичного наблюдения**:

```
bad action → certificate_84 → message from agent C → copied from B → copied from A → original observation e17
```

Это не natural-language rationalization (в отличие от LLM chain-of-thought) — это логированная причинная цепочка внутри runtime, проверяемая программно. Практическая ценность: при провале эпизода можно механически определить, какое исходное наблюдение (не какой агент) стало корнем ошибки — прямое продолжение дисциплины серии «ни одна клетка не цитируется без установленного механизма отказа» (`docs/PROJECT_STATUS_2026-07-18.md` §3).

---

## 9. Где заканчивается формальная работа этого документа

Этот документ фиксирует ТОЛЬКО datatypes и state machine. Свойства (safety/liveness) — в [`02_THEOREMS.md`](02_THEOREMS.md). Как эти свойства проверяются экспериментально — в [`04_EXPERIMENTS.md`](04_EXPERIMENTS.md). Как это ложится в код поверх существующего runtime — в [`10_CODE_LAYOUT.md`](10_CODE_LAYOUT.md).

---

**Файл:** `docs/FRONTIER_MEMORY_AUTHORITY_2026-08-07/01_FORMAL_MODEL.md`
