# Goals: Semantic Customer Personalization

**Дата:** 2026-06-09
**Базис:** переиспользует архитектуру `songline_drive/` (Phase 1–4) и идею semantic intention из `GOALS_semantic_intention_water.md`.
**Контекст:** кафе с 2+ камерами (entrance/counter и seating), POS-системой и опциональной программой лояльности.

---

## 1. Main Goal

Перейти от агрегированного подсчёта посетителей (Little's Law) к **семантически структурированной памяти о клиентах**, где каждый клиент представлен как `CustomerNode` в graph memory, и решения о персонализации (приветствие, рекомендация, оффер) принимаются через ту же цепочку intent → predicate → ranked target → action, что и навигация в Songlines.

Целевая схема:

```
ContextState (время, очередь, смена)
    ↓
IntentType (GREET_RETURNING / SUGGEST_USUAL / RECOMMEND_NEW / LOYALTY_REWARD)
    ↓
SemanticPredicate (face_match ∧ behavioral_consistency)
    ↓
Candidate customers in CustomerGraph
    ↓
Ranked target customer (через SemanticField channels)
    ↓
Recommendation action (текст / промо / приветствие)
```

---

## 2. Core Principle

Главная цель не в том, чтобы «узнавать клиентов по лицу как special-case».

Главная цель такая:

* научить систему **выбирать и обслуживать клиентов, определяемых семантическими паттернами поведения**, а не заранее заданными ID

Face recognition — это просто scene-level evidence (как `water_visible` в Songlines). Сама идентификация клиента — это результат clustering этого evidence + поведенческих признаков + temporal dynamics.

---

## 3. Why Customer Personalization (as next case study)

После водной задачи персонализация клиентов — следующий естественный case study общей архитектуры, потому что:

* клиент задаётся как **тип сущности с признаками**, а не как известный ID
* признаки клиента — это semantic evidence (face embedding + поведенческие tags)
* активация intent зависит от **context state** (время суток, загруженность, наличие новинок) — прямой аналог `AgentState` (thirst/energy)
* масштабируется к более общим задачам:
  * upsell-кандидаты
  * клиенты на грани оттока
  * сегменты для маркетинга
  * персональные лимиты для loyalty

Важно: water-task и customer-task используют **одну и ту же** generic intent abstraction. Никакого customer-specific shortcut в planner.

---

## 4. Two-Tier Sampling Strategy

CV-стек работает с **двумя интервалами snapshot одновременно**, потому что разные задачи имеют разные требования по частоте.

### 4.1 30-секундный snapshot — глобальный подсчёт и occupancy

* Покрывает все зоны (zal, counter, entrance).
* Используется для Little's Law (подсчёт посетителей за день).
* Достаточен, потому что усреднение интегральное и шум сокращается на масштабе сотен снапшотов.
* Дешёв по compute: 1440 кадров на 12-часовой день.

### 4.2 10-секундный snapshot — face recognition в entrance/counter zone

* Только для зоны кассы.
* Триггерится либо постоянно, либо при детекции человека в зоне (smart trigger).
* Нужен потому что face quality (фронтальность, фокус) на одном кадре низкая, и нужны множественные попытки в течение dwell клиента у кассы.
* 4320 кадров на 12-часовой день, но только для одной зоны → compute контролируем.

### 4.3 Почему именно такой split

Математически. Пусть клиент проводит у кассы D секунд, и каждый snapshot захватывает frontal face с вероятностью p = 0.4–0.6 (типичная цифра для CCTV at counter angle). Вероятность хотя бы одного качественного захвата за визит:

```
P_capture(D, Δt, p) = 1 - (1 - p)^(D / Δt)
```

Для D = 60s, p = 0.5:

| Δt | snapshots за визит | P_capture |
|---|---|---|
| 30s | 2 | 0.75 |
| 10s | 6 | 0.984 |
| 5s | 12 | 0.9998 |

Переход с 30s на 10s даёт скачок с 75% до 98.4% — это качественное изменение. Дальнейшее уменьшение даёт diminishing returns. **10 секунд — sweet spot для face recognition на CCTV.**

Для подсчёта (Σ Nt) разница 10s vs 30s менее принципиальна — variance падает в √3 ≈ 1.73 раза, но систематические ошибки остаются те же. Поэтому для подсчёта 30s достаточно.

---

## 5. Layer 1: Counting via Little's Law

### 5.1 Базовая формула

Для интервала snapshot Δt:

```
V_day = (Σ Nt × Δt) / T̄
```

где:
* `Σ Nt` — сумма всех значений «людей в зоне» по всем снапшотам за день;
* `Δt` — интервал между снапшотами (в тех же единицах, что T̄);
* `T̄` — среднее время пребывания посетителя в зоне.

### 5.2 Конкретные подстановки

**30-секундный snapshot, T̄ в минутах:**

```
V_day = (Σ Nt × 0.5) / T̄
```

**10-секундный snapshot, T̄ в минутах:**

```
V_day = (Σ Nt × 1/6) / T̄ = Σ Nt / (6 × T̄)
```

**10-секундный snapshot, T̄ в секундах:**

```
V_day = (Σ Nt × 10) / T̄_sec
```

Все три эквивалентны, главное — следить за единицами.

### 5.3 Калибровка детектора и time-binning (точная форма)

Реальная формула с поправкой на ошибки детектора и сегментацией по периодам дня:

```
V_day = Σᵢ [ (Δt × Σ_{t ∈ period_i} Nt) / (k_i × T̄_i) ]
```

где:
* `i` — индекс периода дня (утро / обед / вечер / поздний вечер);
* `k_i` — коэффициент калибровки детектора в периоде i (отношение detected_count / real_count, замеряется выборочно по 20–50 кадрам);
* `T̄_i` — среднее время пребывания в периоде i (утром takeaway короткий, в обед dine-in длинный).

Типичные значения для кафе:
* `k ∈ [0.85, 0.95]` для прилично настроенного YOLO/RetinaFace + detection.
* `T̄_counter`: 1.5–3 мин (выбрать + расплатиться).
* `T̄_zal`: 15–35 мин (зависит от формата).

### 5.4 Variance оценки (для honest reporting)

Стандартная ошибка V_day:

```
σ(V_day) ≈ V_day × √( σ²(T̄)/T̄² + 1/N )
```

где N — общее количество детекций за день (Σ Nt). При N ≈ 1000 и относительной σ(T̄)/T̄ = 0.2:

```
σ_rel ≈ √(0.04 + 0.001) ≈ 0.20
```

То есть **точность V_day ограничена в первую очередь точностью оценки T̄**, а не sample size. Переход 30s → 10s улучшает только второе слагаемое (1/N), которое и так маленькое.

**Вывод:** для повышения точности подсчёта главный рычаг — калибровка T̄ по чекам, а не уменьшение интервала.

---

## 6. Layer 2: Face Recognition Pipeline (10s zone)

Только для entrance/counter zone, на 10-секундных снапшотах.

### 6.1 Stages

```
Snapshot (10s) → Face detection → Quality filter → Embedding → Aggregation
```

1. **Face detection** (RetinaFace / MTCNN / YOLO-face). Возвращает bounding boxes лиц.
2. **Quality filter** — отсекает мусор перед extraction:
   * `box_height ≥ 80 px` (минимальное разрешение лица)
   * `frontal_score ≥ 0.6` (угол поворота головы)
   * `sharpness ≥ τ_blur` (Laplacian variance)
   * `illumination ∈ [τ_lo, τ_hi]` (не пересвет/недосвет)
3. **Embedding** (ArcFace / FaceNet) → вектор `e ∈ ℝ^512`, нормализованный (`||e|| = 1`).
4. **Aggregation per visit** — за один визит клиента у кассы накапливается до 6–12 эмбеддингов разного качества. Они **не усредняются сразу**, а сохраняются с quality score:
   ```
   visit_embeddings: List[(embedding, quality, timestamp)]
   ```

### 6.2 Match score между эмбеддингами

ArcFace/FaceNet выдают L2-нормализованные векторы, для которых:

```
cosine_similarity(e₁, e₂) = e₁ · e₂  ∈ [-1, 1]
cosine_distance(e₁, e₂) = 1 - e₁ · e₂  ∈ [0, 2]
```

«Тот же человек» если `cos_sim > τ_face`, где `τ_face ≈ 0.5–0.65` (зависит от модели).

### 6.3 Best-of-K query (вместо центроида)

При сопоставлении нового визита с известным клиентом, у которого хранится M эмбеддингов:

```
sim(visit, customer) = max_{j ∈ visit} max_{i ∈ customer} (e_j · e_i)
```

То есть **best-match across all pairs** — устойчивее центроида, потому что лица меняются с ракурсом, и центроид размывает идентичность. Хранить надо top-K (K = 5–10) лучших по quality эмбеддингов на клиента.

### 6.4 POS-linking timing

Заказ закрывается в POS в момент T_pos. Лицо клиента ищется в окне `[T_pos − 90s, T_pos − 5s]` (он стоял у кассы перед оплатой, не во время самой оплаты). Если в окне несколько разных лиц (false positives, очередь) → берётся то, чей quality × duration_at_counter максимальный.

```
face_for_order = argmax_{f in window} (Σ quality(f, t) × dwell_at_counter(f))
```

---

## 7. Layer 3: Customer Identity Resolution (adapted PlaceAlignmentEngine)

Аналог вашего `PlaceAlignmentEngine` в `songline_drive/place_alignment.py`. Заменяет spatial distance на face embedding distance, остальное переиспользует.

### 7.1 Clustering criterion

Новое наблюдение (visit + embeddings + order) сливается с существующим `CustomerNode` если оба условия:

```
face_sim(visit, customer) > τ_face        // визуальная похожесть
behavioral_match(visit, customer) > τ_beh  // поведенческая
```

Где:

```
behavioral_match = w_time × time_pattern_overlap(visit, customer)
                 + w_order × order_similarity(visit, customer)
                 + w_freq × visit_frequency_consistency(visit, customer)
```

Это дублирует логику `tag_match_score` из вашего движка, только в новом домене.

### 7.2 Disambiguation: близнецы / похожие лица

Если visit подходит к нескольким customers по face_sim, выбирается тот, у кого выше behavioral_match. Если оба высокие → создаётся **conflict marker**:

```
conflict_score(visit) = 1 - margin_to_next_best
margin = sim_top1 - sim_top2
```

Это аналог `concept.conflict_score` из вашего `belief_fusion.py`.

### 7.3 Создание нового customer

Если max sim < τ_face_new (более строгий порог, ~0.4):

```
customer = CustomerNode.new(
    initial_embedding=visit.best_embedding,
    first_seen=t_now,
    supporting_agents={cashier_id}
)
```

---

## 8. Customer Memory Schema (CustomerNode)

Расширение вашего `SharedConceptNode` для customer-domain.

```python
@dataclass
class CustomerNode:
    customer_id: str                              # uuid
    
    # Visual identity (face embeddings)
    face_embeddings: List[FaceEmbeddingRecord]    # top-K by quality
    embedding_centroid: ndarray                   # for fast prefilter
    
    # Behavioral profile (analog of semantic_profile)
    behavioral_tags: Dict[str, float]             # produced by LLM Role A
    
    # Raw event log (analog of Phase 1)
    visit_history: List[VisitEvent]               # timestamps, orders, dwell
    order_history: List[OrderEvent]               # tied to POS
    
    # Metadata
    first_seen_seq: int
    last_seen_seq: int
    visit_count: int
    lifetime_value: float
    
    # Phase 3 dynamics
    freshness: float                              # decayed
    conflict_score: float                         # identity uncertainty
    
    # Phase 4 fields (computed by SemanticField)
    channel_activations: Dict[str, FieldChannelState]
    
    # Provenance
    supporting_agents: Set[str]                   # cashiers / cameras
    consent_status: ConsentStatus                 # см. секцию 18
```

`FaceEmbeddingRecord`:

```python
@dataclass
class FaceEmbeddingRecord:
    embedding: ndarray            # 512-dim, L2-normalized
    quality: float                # 0..1
    captured_at_seq: int
    camera_id: str
```

`VisitEvent`:

```python
@dataclass
class VisitEvent:
    visit_id: str
    start_seq: int
    end_seq: int
    dwell_at_counter_sec: int
    captured_embeddings: List[FaceEmbeddingRecord]
    linked_order_id: Optional[str]
    cashier_id: Optional[str]
```

---

## 9. Belief Dynamics (Phase 3 adapted)

Прямо переиспользуется ваш `belief_fusion.py`, только меняется доменная интерпретация.

### 9.1 Temporal decay

Для customer freshness:

```
freshness(c) = max(min_freshness, decay_factor^Δseq)
```

Где `Δseq = current_seq - last_seen_seq(c)`. Sequence — это либо visit count, либо time-based ticks (например, 1 tick = 1 день).

Рекомендую time-based:

```
freshness(c) = max(min_freshness, decay^days_since_last_visit)
```

С `decay = 0.95` (per day) клиент с последним визитом 30 дней назад имеет freshness ≈ 0.21 — заметное падение, но ещё не ноль.

### 9.2 Conflict rules (identity-level)

Adapted `ConflictRuleSet` для customer domain. Несовместимые tag pairs:

* `frequent_morning` ↔ `frequent_evening` (один человек обычно один time slot)
* `oat_milk_only` ↔ `dairy_only`
* `dine_in_always` ↔ `takeaway_only`

Если в `behavioral_tags(c)` сильны оба, это сигнал что один профиль склеил двух разных клиентов.

```
conflict_score(c) = max over incompatible pairs (a, b):
    min(weight_a, weight_b) / max(weight_a, weight_b)
```

Высокий conflict_score → кандидат на split. Низкий → один цельный клиент.

### 9.3 Refresh порядок

В каждом цикле persistence:

1. `apply_decay_to_customers(graph, current_seq)`
2. `apply_conflicts_to_customers(graph)`
3. `field.rebuild_from_customers(graph, current_seq)`

Этот порядок дословно повторяет ваш Phase 3 → Phase 4 в `field_adapter.py`.

---

## 10. Semantic Field for Recommendations (Phase 4 adapted)

Ваш `SemanticField` переиспользуется, меняются только channels и semantics.

### 10.1 Recommendation channels

Вместо `water_source / hazard_edge / goal_region`:

| Channel | Цель | Positive tags | Negative tags |
|---|---|---|---|
| `greet_returning` | приветствие узнанного | `regular_visitor`, `recent_visit` | `new_customer`, `complaint_history` |
| `suggest_usual` | предложить «как обычно» | `consistent_order`, `morning_regular` | `experimental_taster`, `varied_orders` |
| `recommend_new` | новинка / сезон | `experimental_taster`, `varied_orders` | `strict_routine` |
| `upsell_dessert` | добавить десерт | `no_dessert_pattern`, `long_dwell` | `quick_takeaway`, `health_conscious` |
| `loyalty_reward` | бонус / промо | `near_loyalty_milestone`, `lapsed_recent` | `recent_promo_used` |
| `retention_push` | возврат отвалившегося | `lapsed_30d+`, `high_lifetime_value` | `new_customer`, `recent_visit` |

`DEFAULT_CHANNEL_AFFINITIES` для customer domain — словарь как в вашем `semantic_field.py`, только с этими tags.

### 10.2 Belief strength B(c)

Прямой перенос из Phase 4:

```
B(c) = w_conf × identification_confidence(c)
     + w_fresh × freshness(c)
     + w_support × log1p(visit_count(c)) / log1p(MAX_VISITS)
     + w_purity × (1 - conflict_score(c))
```

С теми же дефолтами весов из `semantic_field.py`. `identification_confidence` — это уверенность face match (см. секцию 7).

### 10.3 Channel affinity I(k, c)

```
I(k, c) = Σ_tag positive_weight(k, tag) × behavioral_tags(c)[tag]
        − Σ_tag negative_weight(k, tag) × behavioral_tags(c)[tag]
```

Это weighted dot product поведенческих tags клиента с tag-weights канала. Логика та же, что и для water/hazard channels.

### 10.4 Полная activation formula

```
A_{t+1}(k, c) = λ × A_t(k, c)                  // EMA continuity
              + α × B(c) × I(k, c)             // belief × affinity
              + γ × D_t(k, c)                  // diffusion (см. 10.5)
              − η × conflict_score(c)          // identity uncertainty penalty
              − ξ × reservation(c)             // multi-cashier lock
```

Дефолты из вашей спеки:
* `λ = 0.95` (decay)
* `α = 0.60` (belief weight)
* `γ = 0.10` (diffusion)
* `η = 0.30` (conflict suppression)
* `ξ = 0.20` (occupancy)

### 10.5 Diffusion term (γ) — переинтерпретация

В water-задаче diffusion усреднял spatial neighbors внутри `diffusion_radius`. Для customer-domain spatial proximity бессмыслена, но есть **embedding-space proximity**: похожие клиенты могут влиять друг на друга через collaborative filtering.

Два варианта:

**Вариант A (рекомендую для старта):** `γ = 0`. Diffusion отключён. Активация чисто per-customer.

**Вариант B (продвинутый):** заменить spatial KNN на embedding KNN:

```
D_t(k, c) = mean over top-N nearest customers c' by embedding:
            A_t(k, c') × similarity(c, c')
```

Это даёт «клиенты как X тоже любят Y». Но усложняет систему и требует валидации, что это даёт реальный gain. Сначала без него.

### 10.6 Reservations (multi-cashier)

Дословный перенос ваших `FieldReservation` на customer domain. Если cashier-A начинает обслуживать клиента c:

```
field.reserve(customer_id=c, channel=k, agent_id=cashier_A, duration=300s)
```

Это применяет `−ξ` к активациям клиента c на 300 секунд. Cashier-B на другой кассе **не получает** этого же клиента в своих top-recommendations, даже если он подходит. Это устраняет double-targeting.

`expire_reservations(current_seq)` сама снимает истёкшие.

### 10.7 Mode axis (выводится напрямую)

| Mode | Поведение |
|---|---|
| `none` | recommendations отключены, только подсчёт |
| `descriptive` | computed но не используются (для метрик) |
| `read_only` | top-k recommendation отдаётся cashier UI, без reservations |
| `coordinated` | full mode с reservations, для multi-cashier |

Те же 4 mode'а, что и в `FieldMode` enum.

---

## 11. Intent Selection

### 11.1 IntentTypes (customer-domain)

```python
class IntentType(Enum):
    # Existing (Songlines)
    FIND_GOAL_REGION = ...
    FIND_WATER_SOURCE = ...
    
    # New (Customer personalization)
    IDENTIFY_CUSTOMER = "identify_customer"
    GREET_RETURNING = "greet_returning"
    SUGGEST_USUAL = "suggest_usual"
    RECOMMEND_NEW = "recommend_new"
    UPSELL = "upsell"
    LOYALTY_REWARD = "loyalty_reward"
    RETENTION_PUSH = "retention_push"
    GUEST_FLOW = "guest_flow"           # для незарегистрированных
```

### 11.2 State-driven activation (rule-based MVP)

Прямой аналог вашего Sprint B v2 (`thirst_on/off_threshold` для воды).

```
if customer.consent_status != GRANTED:
    intent = GUEST_FLOW
elif customer.visit_count < 2:
    intent = GREET_RETURNING + IDENTIFY_CUSTOMER
elif days_since_last_visit > 30 and lifetime_value > τ_lv:
    intent = RETENTION_PUSH
elif near_loyalty_milestone(customer):
    intent = LOYALTY_REWARD
elif strong_usual_pattern(customer) and current_time matches pattern:
    intent = SUGGEST_USUAL
elif has_unsold_seasonal() and experimental_score(customer) > τ:
    intent = RECOMMEND_NEW
else:
    intent = SUGGEST_USUAL  # safe default
```

С hysteresis по `recommendation_cooldown` (не пушить одному клиенту чаще раза в X дней).

### 11.3 Trace requirements (по аналогии с Sprint B v2)

В traces писать:
* `previous_active_intent`
* `new_active_intent`
* `intent_switch_reason`
* `context_state_snapshot`

Для отладки и benchmark — как у вас уже сделано для water.

---

## 12. LLM Roles (где именно и почему)

LLM — **не часть единой сети**, а три отдельных функции над graph state. Каждая решает свою задачу с разной частотой.

### 12.1 Role A — Tag extraction from POS history (offline, periodic)

**Когда:** раз в неделю или после N новых заказов на клиента.
**Вход:** raw order history клиента (JSON списков заказов с timestamps).
**Выход:** `behavioral_tags: Dict[str, float]` для записи в `CustomerNode`.

```
Input prompt template:
  "Below is order history of a cafe customer. 
   Produce a JSON of behavioral tags with confidence weights in [0, 1]
   from this allowed tag vocabulary: {TAG_VOCABULARY}.
   Be conservative — only assign weights > 0.5 if pattern is clear from 3+ orders.
   Orders: {orders_as_jsonl}"

Output:
  {
    "morning_regular": 0.95,
    "oat_milk_drinker": 0.90,
    "no_dessert_pattern": 0.80,
    "workday_visitor": 0.85,
    "latte_default": 0.92
  }
```

Аналог вашего `scene_tokenizer.py` — превращает сырое observation в нормированные semantic tags.

### 12.2 Role B — Intent reasoning (online, optional)

**Когда:** при распознавании клиента у кассы.
**Вход:** customer profile + current context state.
**Выход:** chosen IntentType + reasoning.

Опционально — можно начать с rule-based из секции 11.2 и подключить LLM только если правила достигают потолка по точности.

### 12.3 Role C — Generation (online, presentation layer)

**Когда:** после выбора intent, перед показом cashier'у.
**Вход:** intent + customer profile + context.
**Выход:** текст приветствия / рекомендации.

Это **не часть** graph memory architecture. Это presentation layer поверх неё. Можно даже не LLM — для большинства intent'ов хватает template'ов.

### 12.4 Что НЕ делать

* Не превращать всё в end-to-end transformer. Graph memory — это state с явной семантикой; LLM поверх неё работает лучше, чем как замена ей.
* Не передавать в LLM сырые face embeddings или POS-данные — только агрегированные tags.
* Не использовать LLM в hot path подсчёта или identity resolution — это deterministic операции.

---

## 13. End-to-End Data Flow

```
                   ┌─────────────────────────────────────┐
                   │           CAMERAS                   │
                   │  zal (30s)  +  counter (10s)        │
                   └──────────────┬──────────────────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              │                                       │
              ▼                                       ▼
    ┌─────────────────┐                   ┌──────────────────────┐
    │ Counting branch │                   │ Face recognition     │
    │ (Little's Law)  │                   │ (entrance/counter)   │
    │                 │                   │                      │
    │ Σ Nt / T̄ per   │                   │ Detect → Quality →   │
    │ time bin        │                   │ Embedding (ArcFace)  │
    └────────┬────────┘                   └──────────┬───────────┘
             │                                       │
             ▼                                       ▼
    ┌─────────────────┐                   ┌──────────────────────┐
    │ Daily V_day     │                   │ visit_embeddings     │
    │ (per zone)      │                   │ + visit metadata     │
    └─────────────────┘                   └──────────┬───────────┘
                                                     │
                                                     ▼
                                          ┌──────────────────────┐
                                          │  POS link            │
                                          │  (T_pos ± window)    │
                                          └──────────┬───────────┘
                                                     │
                                                     ▼
                                          ┌──────────────────────┐
                                          │ CustomerAlignment    │
                                          │ Engine (Phase 2-like)│
                                          │ - face match         │
                                          │ - behavioral match   │
                                          │ - merge or create    │
                                          └──────────┬───────────┘
                                                     │
                                                     ▼
                                          ┌──────────────────────┐
                                          │  CustomerGraph       │
                                          │  (CustomerNodes)     │
                                          └──────────┬───────────┘
                                                     │
                                                     ▼
                            ┌────────────────────────┴────────────────────┐
                            │                                             │
                            ▼                                             ▼
                ┌────────────────────────┐                  ┌───────────────────────┐
                │ Phase 3 dynamics       │                  │  LLM Role A (offline) │
                │ - decay                │                  │  POS → behavioral_tags│
                │ - conflict rules       │                  └──────────┬────────────┘
                └────────────┬───────────┘                             │
                             │                                          │
                             ▼                                          ▼
                ┌────────────────────────────────────────────────────────────┐
                │                  SemanticField                             │
                │   A(k,c) = λA + αBI + γD - η·conflict - ξ·reservation     │
                │                                                            │
                │   channels: greet, suggest_usual, upsell, loyalty, ...    │
                └────────────────────────┬───────────────────────────────────┘
                                         │
                                         ▼
                            ┌────────────────────────┐
                            │ Intent selection       │
                            │ (rule-based, later LLM)│
                            └────────────┬───────────┘
                                         │
                                         ▼
                            ┌────────────────────────┐
                            │  Presentation layer    │
                            │  (LLM Role C / templ.) │
                            └────────────┬───────────┘
                                         │
                                         ▼
                            ┌────────────────────────┐
                            │  Cashier UI            │
                            │  ("Анна, латте с овс?")│
                            └────────────────────────┘
```

---

## 14. Target Architecture (mapped to your codebase)

### 14.1 Reuse as-is

* `songline_drive/collective_types.py` — `AgentSignature`, `CollectiveEvent`, `CollectiveQuery` работают без изменений (agent_id = cashier_id, event = visit_observation).
* `songline_drive/belief_fusion.py` — `TemporalDecayEngine`, `ConflictRuleSet` переиспользуются дословно, только с другим `default_rules()`.
* `songline_drive/collective_field_types.py` — `FieldMode`, `FieldChannelState`, `FieldReservation` идентичны.
* `songline_drive/semantic_field.py` — `SemanticField` класс переиспользуется. Меняется только `DEFAULT_CHANNEL_AFFINITIES` (под customer-channels).
* `songline_drive/field_adapter.py` — graceful degradation logic тождественна.
* `songline_drive/field_metrics.py` — большинство метрик применимы (precision@k, top1_stability, deconfliction_rate).

### 14.2 Adapt

* `songline_drive/place_alignment.py` → `customer_alignment.py` — заменить spatial distance на face embedding distance + behavioral match.
* `songline_drive/collective_concepts.py` → расширить `SharedConceptNode` либо завести параллельный `CustomerNode`.
* `songline_drive/concept_recall.py` → `customer_recall.py` — тот же шаблон, только query by face_query вместо tag_query.

### 14.3 New

* `customer_personalization/face_pipeline.py` — детектор + quality + embedding + aggregation.
* `customer_personalization/pos_link.py` — связка лица с заказом по timestamp window.
* `customer_personalization/llm_roles.py` — Role A (offline), Role B (optional online), Role C (generation).
* `customer_personalization/intent_policy.py` — rule-based intent selector + activation state.
* `customer_personalization/consent.py` — управление consent_status, отдельный flow для GRANTED / DENIED / NOT_ASKED.

### 14.4 Mode parallelism

Те же 3 независимых mode axes, что и для water:

| Axis | Аналог | Значения |
|---|---|---|
| `customer_memory_mode` | `milestone_mode` | `none / individual / shared` |
| `recommendation_graph_mode` | `graph_update_mode` | `none / lazy / incremental` |
| `recommendation_field_mode` | `collective_field_mode` | `none / descriptive / read_only / coordinated` |

При всех = `none` система сводится к чистому подсчёту без персонализации.

---

## 15. Sprint Plan

### Sprint A: Customer As Semantic Node

Цель: первый рабочий customer pipeline без LLM, без recommendations, только идентификация и подсчёт по идентифицированным.

* Ввести `IntentType.IDENTIFY_CUSTOMER`.
* Реализовать `face_pipeline.py` (detection + quality + ArcFace embedding).
* Реализовать `customer_alignment.py` (greedy clustering по face + behavioral).
* Расширить graph datatypes под `CustomerNode`.
* Smoke: 5 synthetic visit traces, 3 уникальных клиента, 2 повторных визита у двух из них → проверить что recall возвращает правильный customer_id.

**Done criterion:** trace показывает `intent=IDENTIFY_CUSTOMER, candidate_customer_id, identification_confidence` для каждого визита у кассы.

### Sprint A.1: LLM Tag Extraction

* Подключить LLM Role A.
* Определить `TAG_VOCABULARY` (50–100 поведенческих tags).
* Smoke: 10 synthetic customer histories → проверить воспроизводимость tags (temperature=0, two runs совпадают по top-5 tags).

**Done criterion:** customer node имеет `behavioral_tags` после batch-job.

### Sprint B: State-Driven Recommendation

* Подключить `recommendation_field_mode=descriptive`.
* Определить 6 channels из секции 10.1.
* Реализовать rule-based intent selector (секция 11.2).
* Smoke: один и тот же клиент в разных context state → разные top-channels.

**Done criterion:** trace показывает `top_channel, channel_activation, intent_selected, intent_switch_reason` для каждого визита.

### Sprint C: Multi-Cashier Coordination

* `recommendation_field_mode=coordinated`.
* Реализовать reservations для customer-level lock.
* Smoke: один клиент в очереди двух касс → reservation предотвращает double-recommendation.

**Done criterion:** `deconfliction_rate = 1.0` на синтетическом сценарии с 2 cashier'ами.

### Sprint D: Production calibration

* Калибровка `T̄_i` по чекам.
* Калибровка `k_i` (detector accuracy).
* Tuning `τ_face`, `τ_beh` по реальной выборке.
* End-to-end run на одном дне реальных данных.

**Done criterion:** ошибка V_day < 5% против ground truth по чекам.

---

## 16. Files To Extend / Create

Реальные пути по аналогии с Songlines layout.

### Extend (Songlines codebase)
* `songline_drive/types.py` — добавить `IntentType.IDENTIFY_CUSTOMER` и др.
* `songline_drive/intents.py` — customer-domain intent definitions.
* `songline_drive/belief_fusion.py` — добавить `customer_default_rules()` в `ConflictRuleSet`.
* `songline_drive/semantic_field.py` — добавить customer channels в `DEFAULT_CHANNEL_AFFINITIES`.

### New module: `customer_personalization/`

* `customer_personalization/types.py`
* `customer_personalization/face_pipeline.py`
* `customer_personalization/pos_link.py`
* `customer_personalization/customer_alignment.py`
* `customer_personalization/customer_recall.py`
* `customer_personalization/customer_field_adapter.py`
* `customer_personalization/intent_policy.py`
* `customer_personalization/consent.py`
* `customer_personalization/llm_roles.py`
* `customer_personalization/metrics.py`

### Scripts

* `scripts/customer_smoke_sprint_a.py`
* `scripts/customer_smoke_sprint_a1_llm_tags.py`
* `scripts/customer_smoke_sprint_b.py`
* `scripts/customer_smoke_sprint_c.py`

Каждый по образцу ваших `multiagent_smoke_*.py` — пишет `*_summary.json`, печатает `✓ Customer Sprint X passed`.

---

## 17. What Not To Do

* Не делать end-to-end neural net «CV+LLM в одной сети». Graph memory — central state, нейронки — функции над ней.
* Не делать customer-specific shortcut в planner — переиспользовать generic intent abstraction.
* Не использовать LLM в hot path identity resolution (deterministic) или counting (analytic).
* Не передавать face embeddings в LLM — это и юридически опасно, и не нужно.
* Не строить collaborative filtering (γ-diffusion) до того как single-customer recommendation работает.
* Не игнорировать consent status — нет согласия = `intent=GUEST_FLOW`, никаких embeddings в БД.
* Не уменьшать snapshot interval ниже 10s ради точности — diminishing returns, основной рычаг точности в калибровке T̄ и k.

---

## 18. Legal Considerations

Это **самая большая практическая проблема всего проекта**, не архитектурная. Архитектурно стек к ней готов.

### 18.1 Применимое регулирование (РФ)

* **152-ФЗ "О персональных данных", ст. 11** — биометрия (face embeddings включительно) требует **письменного согласия** субъекта.
* **572-ФЗ (2023+)** — биометрические шаблоны должны храниться/использоваться через **ЕБС** (Единая биометрическая система Ростелекома) для большинства коммерческих сценариев. Юридическая консультация обязательна.
* **Регистрация в Роскомнадзоре** как оператора биометрических ПД.
* **Хранение строго на территории РФ.**
* После поправок 2024 г.: штраф до 1.5 млн ₽ за инцидент.

### 18.2 Consent flow в архитектуре

Каждый `CustomerNode` имеет `consent_status: ConsentStatus`:

```python
class ConsentStatus(Enum):
    NOT_ASKED = "not_asked"       # клиент новый, не спрашивали
    DENIED = "denied"             # явный отказ
    GRANTED = "granted"           # подписанное согласие
    EXPIRED = "expired"           # согласие истекло
```

Логика:

* `NOT_ASKED` / `DENIED` / `EXPIRED` → `intent = GUEST_FLOW`. Face embeddings **не сохраняются** в долгосрочной памяти. Можно использовать transient embedding только в пределах одного визита для linkage с заказом (если это допустимо консультацией с юристом).
* `GRANTED` → полный pipeline активен.

### 18.3 Альтернатива — программа лояльности через QR

Самый чистый юридический путь: персонализация **только** для opt-in клиентов через QR-код / приложение. Face recognition остаётся опциональным «премиум» уровнем для тех, кто подписал отдельное согласие на биометрию. Большинство задач персонализации закрывается просто QR + POS-историей без камер.

**Рекомендация:** запустить весь intent/recommendation стек на data из QR loyalty (это та же архитектура без face части), и только после доказанного бизнес-эффекта добавлять face как опциональный layer.

---

## 19. Open Questions

* **Embedding-space diffusion (γ-term):** даёт ли реальный gain или это шум? Нужен ablation после Sprint B.
* **Best-of-K vs centroid storage:** какой K оптимален? Trade-off storage vs accuracy. Гипотеза: K = 5–10.
* **LLM Role A на каком интервале:** раз в неделю / на каждый новый заказ / по batch'ам? Влияет на стоимость API.
* **Cold start:** что делать с клиентами с 1–2 визитами? Возможно, отдельный `IntentType.ONBOARD_NEW` с другой логикой.
* **A/B framework:** как сравнивать модели с и без recommendations не нарушая customer experience? Time-sliced randomization?
* **Cross-store memory:** если у сети несколько кафе — переиспользовать `distributed_memory/` или `peer_memory/`? Гипотеза: `peer_memory/` лучше для сетей, потому что нет центрального single-point-of-failure.
* **Drift detection:** клиент изменил привычки (новая работа → другое время визитов) — как быстро система это поймает? Параметр `decay_factor` ключевой.

---

## 20. One-Sentence Goal

Не «узнавать каждого клиента по лицу», а:

* научить систему **выбирать и обслуживать клиентов, определяемых семантическими паттернами поведения** (а не заранее заданными ID), переиспользуя generic semantic-intention архитектуру Songlines.

Customer personalization — следующий полноценный case study после water, на той же абстракции.
