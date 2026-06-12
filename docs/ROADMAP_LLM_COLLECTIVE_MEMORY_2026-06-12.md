# Дорожная карта: от текущего символического стека к прототипу коллективной памяти LLM-агентов

**Дата:** 12 июня 2026
**Автор vision:** автор проекта
**Цель документа:** зафиксировать (а) что уже работает в коде, (б) куда мы идём, (в) с каким набором deliverables считаем что пришли. Документ не описывает академическую статью — он описывает инженерный продукт.

---

## 1. Vision одним абзацем

Построить прототип **распределённой долговременной коллективной памяти для мультиагентных LLM-систем**, в которой каждый агент:
1. имеет **свой собственный** символический concept-граф;
2. получает токены не из hand-crafted vocabulary, а из LLM, извлекающего семантические концепты из наблюдений (символическая токенизация на естественно-языковых данных);
3. сохраняет состояние памяти **между сессиями** (cross-episode persistence);
4. обменивается с другими агентами **через peer-to-peer broadcast с настраиваемой каденцией K**, без центрального агрегатора;
5. использует доверие к пирам и убывание свежести (trust × staleness) для merge incoming evidence.

Конечная цель prototype — продемонстрировать, что такой стек даёт **измеримое преимущество** над двумя доминирующими альтернативами: (а) shared context window (MetaGPT, AutoGen) и (б) vector embedding memory (MemGPT, Voyager, Generative Agents) — в задачах, где требуется кооперативное долговременное знание.

---

## 2. Где мы сейчас (карта существующего кода)

### 2.1 Что реально работает

| Подсистема | Расположение | Размер | Статус |
|---|---|---|---|
| Peer-to-peer broadcast runtime | `peer_memory/` | 1 024 LOC, 7 модулей | ✅ работает, протестировано на 35 640 прогонах |
| Per-agent symbolic graph | `peer_memory/peer_agent.py` | 222 LOC | ✅ каждый агент владеет собственным графом |
| Asymmetric trust model | `peer_memory/peer_trust.py` | 100 LOC | ✅ EMA-обновление от retrieval consistency |
| Place-belief aggregator с Dirichlet posteriors | `songline_drive/collective_memory.py` | 518 LOC | ✅ event-sourced, поддерживает per-place tag distributions |
| Минимальная Collective Semantic Memory (CSM) | `experiments/collective_semantic_memory/csm_memory.py` | 160 LOC | ✅ strict-dominate'ит fixed-K peer на 3 240 прогонах с non-overlapping 95% CI |
| Q/R/M/C diagnostic framework | `experiments/big_experiment/runner.py` (часть `run_one_config`) | измеряет conditional rates | ✅ работает на двух substrate'ах (custom grid + MiniGrid wrapper) |
| MiniGrid wrapper для portability | `experiments/minigrid_multiagent_wrapper/` | ~250 LOC | ✅ slope conditions хранят знаки $p<10^{-6}$ |
| Пакет для подачи в ИТМО как РИД (CADENZA) | `peer_memory/` + `docs/ITMO/` | заполнено 11.06.2026 | ✅ готов к подаче |

**Итого: ~1 700 LOC рабочего production-grade Python кода + воспроизводимый measurement pipeline.**

### 2.2 Что есть в Q/R/M/C diagnostic'е

Четырёхстадийная декомпозиция семантической навигационной памяти:
- $Q^\star$ — был ли сформирован непустой query;
- $R^\star$ — вернул ли retrieval семантически удовлетворяющий кандидат;
- $M^\star$ — закрепил ли planner за этим кандидатом конкретную цель;
- $C^\star$ — достиг ли агент цели.

На 35 640 прогонах верифицированы наклоны bottleneck-shift (slope-level claim): уменьшение K улучшает $P(M^\star|R^\star)$ и ухудшает $P(C^\star|M^\star)$, оба знака при $p<10^{-4}$.

Эта метрическая инфраструктура **сохраняется без изменений** для LLM-substrate — она читает события из логов, ей всё равно откуда пришли токены.

### 2.3 Чего реально нет (честный gap-аналитик)

1. **Нет LLM в loop'е.** Tokenization сейчас hand-crafted: символ `water_source` приходит из кода окружения, не из языковой модели.
2. **Нет cross-session persistence.** Память агента ресетится между benchmark-прогонами. Архитектура event-sourced, persistence добавить тривиально, но *сейчас её нет*.
3. **Нет NL-канала коммуникации между агентами.** Broadcast обменивает symbolic snapshots; естественноязыкового сообщения один-другому не передаёт.
4. **Нет realistic task.** 12×10 grid с одной целью water_source — не та сложность, на которой можно показать что long-term collective memory нужна. Нужен ALFWorld / TextWorld / Crafter уровень.
5. **Нет LLM agent class.** Решения принимает 3-уровневый rule-based planner. LLM-driven decision не реализован.

**Сводка:**

```
                              есть │ нет
─────────────────────────────────┼──────
 symbolic substrate                │
 per-agent graph                   │
 cadence-K broadcast               │
 trust × staleness                 │
 Q/R/M/C measurement               │
                                   │  LLM tag extractor
                                   │  LLM query former
                                   │  PeerLLMAgent decision layer
                                   │  cross-session persistence
                                   │  NL inter-agent comm channel
                                   │  realistic task (ALFWorld)
```

Это **~40% инфраструктуры, 0% LLM integration, 100% измерительный framework**.

### 2.4 Где наша уникальная ниша в существующей literature

```
                            single-agent  │  multi-agent
                          ────────────────┼────────────────
  vector memory          MemGPT, Voyager  │  Generative Agents
  shared context              —           │  MetaGPT, AutoGen
  symbolic per-agent          ?           │  ◄── ЗДЕСЬ
```

Правая нижняя клетка в literature **пустая**. Никто из работающих систем не имеет одновременно:
- per-agent independent symbolic graph,
- tunable broadcast cadence как structural axis,
- diagnostic framework для measuring где ломается коллектив.

Это defendable research niche. У нас уже есть три из трёх компонентов — на toy substrate. Прототип должен показать что они работают и на LLM substrate.

---

## 3. Куда мы идём (целевой prototype)

### 3.1 Определение «пришли»

Prototype считается выполненным, когда выполнены **все шесть** acceptance criteria:

1. **N ≥ 3 LLM-агентов** одновременно работают на кооперативной задаче (ALFWorld household или эквивалент).
2. **Каждый агент** имеет свой собственный символический concept graph (наследуется от `peer_memory.PeerAgent`).
3. **Tags извлекаются LLM-ом** из natural-language observations, не из hand-crafted кода.
4. **Память persistent**: между двумя сессиями того же агента состояние памяти восстанавливается из снапшота.
5. **Q/R/M/C events** на новой подложке эмитятся не-тривиально и стадии не вырождаются (как у CommNet — там $Q^\star=1$, $R/M$ collapse).
6. **Comparative baseline**: vector-memory baseline (LangChain `VectorStoreRetrieverMemory`) измерен на той же задаче. CSM-style стек либо доминирует по success rate с non-overlapping CI, либо мы получаем честный negative result.

### 3.2 Что НЕ входит в scope prototype'а

- Полная formalisation коллективного сознания. Это marketing-фраза для внутренней мотивации, не technical claim. В статью и тем более в evaluation не выносится.
- Production-quality multi-agent orchestration (sandboxing, security, billing, scale). Это инженерия после исследовательского prototype'а.
- Fine-tuning LLM. Используем off-the-shelf модели через API или ollama локально.
- Visual modalities. Только текст + (опционально) structured observations.

### 3.3 Целевой стек после реализации

```
┌──────────────────────────────────────────────────────────────┐
│  Multi-agent task environment (ALFWorld / TextWorld)         │
│  ─ кооперативная задача, требующая shared long-term memory   │
└──────────────────────────────────────────────────────────────┘
            │ NL observation        │ NL observation
            ▼                        ▼
┌───────────────────────┐  ┌───────────────────────┐  ...
│ PeerLLMAgent #1       │  │ PeerLLMAgent #2       │
│ ┌───────────────────┐ │  │ ┌───────────────────┐ │
│ │ LLM tag extractor │ │  │ │ LLM tag extractor │ │
│ │  obs → Dict[tag,c]│ │  │ │  obs → Dict[tag,c]│ │
│ └─────────┬─────────┘ │  │ └─────────┬─────────┘ │
│           │           │  │           │           │
│ ┌─────────▼─────────┐ │  │ ┌─────────▼─────────┐ │
│ │ Symbolic per-agent│ │  │ │ Symbolic per-agent│ │
│ │ graph + Dirichlet │ │  │ │ graph + Dirichlet │ │
│ │ + trust + staleness│ │  │ │ + trust + staleness│ │
│ │   (peer_memory)   │ │  │ │   (peer_memory)   │ │
│ └─────────┬─────────┘ │  │ └─────────┬─────────┘ │
│           │ retrieve   │  │           │ retrieve  │
│ ┌─────────▼─────────┐ │  │ ┌─────────▼─────────┐ │
│ │ LLM query former  │ │  │ │ LLM query former  │ │
│ │ task → req_tags   │ │  │ │ task → req_tags   │ │
│ └─────────┬─────────┘ │  │ └─────────┬─────────┘ │
│           │            │  │           │           │
│ ┌─────────▼─────────┐ │  │ ┌─────────▼─────────┐ │
│ │ LLM decision      │ │  │ │ LLM decision      │ │
│ │ candidates → action │  │ │ candidates → action │
│ └───────────────────┘ │  │ └───────────────────┘ │
└───────────┬───────────┘  └───────────┬───────────┘
            │                          │
            ▼                          ▼
┌──────────────────────────────────────────────────────────────┐
│  BroadcastBus (passive) — каждые K тиков snapshot exchange  │
│  ─ symbolic tags + NL annotation (краткое summary памяти)   │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  Q/R/M/C diagnostic logger (без изменений)                  │
│  ─ per-agent per-tick события Q*, R*, M*, C*                │
│  ─ агрегирование в conditional rates + success rate         │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  Cross-session persistence layer                            │
│  ─ JSON-dump events на конец сессии                         │
│  ─ восстановление при старте новой сессии того же агента    │
└──────────────────────────────────────────────────────────────┘
```

Жёлтый — новое (то, что строим). Синий — существующее (работает).

---

## 4. Дорожная карта (6 этапов, 4-6 недель)

Каждый этап — independent deliverable: можно остановиться после любого без потери ценности от предыдущих.

### Этап 1. LLM tag extractor (3-4 дня)
**Создаётся:** `experiments/llm_collective/llm_tag_extractor.py` (~150 LOC)
**Что делает:** observation text → `Dict[str, float]` (concept tag → confidence).
**Backend:** конфигурируемый (Llama-3-8B локально через ollama / Claude Haiku / GPT-4o-mini).
**Caching:** deterministic seed → cache hit для воспроизводимости.
**Бюджет:** $0 (ollama) или ~$5 (API на 5000 наблюдений).
**Acceptance:** smoke на 10 ALFWorld scene descriptions, тегов non-trivially >2 per scene.

### Этап 2. TextWorld / ALFWorld адаптер (2-3 дня)
**Создаётся:** `experiments/llm_collective/textworld_env.py` (~200 LOC)
**Что делает:** обёртка ALFWorld под существующий `build_env() → Built` интерфейс.
**Acceptance:** single-agent prototype проходит ALFWorld smoke (1 эпизод), Q/R/M/C события эмитятся.

### Этап 3. PeerLLMAgent (1 неделя)
**Создаётся:** `experiments/llm_collective/peer_llm_agent.py` (~400 LOC)
**Что делает:** subclass `peer_memory.PeerAgent`. LLM формулирует retrieval query из task. LLM принимает решение из retrieved candidates. Memory observe() остаётся unchanged — получает tags от extractor'а.
**Acceptance:** один LLM-агент с symbolic memory на ALFWorld проходит 5 эпизодов end-to-end.

### Этап 4. Cross-session persistence (3-4 дня)
**Создаётся:** `experiments/llm_collective/persistent_runtime.py` (~100 LOC)
**Что делает:** dump events на конец сессии, load на старте. Event-sourcing уже есть — serialization тривиальна.
**Acceptance:** agent на втором эпизоде использует знание из первого (можно verify через trace).

### Этап 5. Multi-agent LLM коллектив (1-1.5 недели) — **точка возврата к vision**
**Создаётся:** `experiments/llm_collective/run_collective.py` (~250 LOC)
**Что делает:** N=3 PeerLLMAgent на shared ALFWorld task с cadence-K broadcast. NL summary layer: каждый agent генерирует короткое описание памяти при broadcast. Receiving agents используют NL для richer trust update.
**Acceptance:** 3 LLM-агента кооперативно решают задачу, в логах видна ненулевая кросс-агентная передача знаний через broadcast.

### Этап 6. Q/R/M/C evaluation + baseline comparison (1 неделя)
**Создаётся:** `experiments/llm_collective/run_csm_llm_vs_vector.py` (~300 LOC)
**Что делает:** Тот же measurement protocol на новой подложке. Baseline: vector memory (LangChain `VectorStoreRetrieverMemory`). Метрики: success rate, Q/R/M/C profile, interpretability score (можно ли восстановить рассуждение agent'а).
**Acceptance:** один из трёх результатов:
- (A) Symbolic > vector с non-overlapping CI → готовая Q1-paper data.
- (B) Symbolic = vector, но interpretability strictly выше → niche paper про diagnosable multi-agent memory.
- (C) Symbolic < vector → честный negative result, понимаем границы подхода.

**Любой из трёх — publishable. Plan не предполагает что результат обязательно положительный.**

---

## 5. Бюджет (deliberately tight estimates)

### Time
- Solo focused работа: **6 недель** (один разработчик, 30+ часов/неделю)
- Solo part-time (10-15 ч/неделю): **3-4 месяца**
- C scaffold-помощником и фокус-сессиями: **4 недели**

### Compute / API
- Локальный Llama-3-8B через ollama: $0 (есть Mac с M2/M3 — комфортно работает)
- Если хочется качества Claude/GPT уровня: ~**$200-500** на всё 6 этапов (включая абляции)
- GPU не требуется — только inference, не training

### Risks
| Риск | Вероятность | Mitigation |
|---|---|---|
| LLM extractor производит шумные tags, Q/R/M/C events вырождаются | средняя | Этап 1 — независимый тест. Если показывает > 30% noise rate — переходить на structured output / few-shot examples / fine-tuned extractor. Если не помогает — это сам по себе publishable negative result. |
| ALFWorld multi-agent не поддерживается из коробки | низкая | Fallback: TextWorld (точно работает) или Crafter-multi |
| Vector baseline доминирует на простых задачах | средняя | Выбрать задачу где нужна **long-term** память (несколько сессий). Vector memory деградирует на long horizon из-за context bloat. |
| Symbol grounding evaluation субъективна | высокая | Это known issue области. Mitigation — фиксированный vocabulary + manual annotation 50 примеров для inter-rater agreement. |

---

## 6. Что мы НЕ делаем (явный список scope cuts)

Чтобы не размывать prototype:

1. **Не пишем академическую статью** до завершения Этапа 6. Параллельно подаём текущую Q/R/M/C работу на workshop как есть.
2. **Не делаем production deployment.** Цель — research prototype, не SaaS.
3. **Не fine-tune'им модели.** Off-the-shelf only.
4. **Не intgrate'ем с Voyager / MetaGPT / AutoGen напрямую.** Они — comparison points в литобзоре, не dependency'и.
5. **Не пытаемся заявлять «коллективное сознание»** в заголовке статьи. Внутренняя мотивация ≠ paper claim. В paper это будет «distributed long-term memory architecture for cooperative LLM agents with diagnosable failure modes».

---

## 7. Параллельные треки

Пока строится LLM-prototype, **независимо** ведутся:

- **Подача CADENZA как РИД в ИТМО** (документы подготовлены 11.06.2026, остался ручной fill personal data) → patent protection backbone'а.
- **Текущая Q/R/M/C work как workshop submission** в её нынешнем виде. 1 день работы на подготовку версии под workshop формат (4-9 страниц).
- **Возможный preprint на arXiv** текущей версии (29-31 страница) если хочется early citation cycle.

Эти три не блокируют roadmap. Их можно сделать в течение 1-2 недель параллельно Этапу 1.

---

## 8. Дерево решений: что делать на каждой точке возврата

```
[сейчас] ───► Этап 1 (LLM tag extractor)
                  │
                  ├──► extractor работает > 70% precision
                  │            │
                  │            ▼
                  │      Этап 2 (ALFWorld adapter)
                  │            │
                  │            ▼
                  │      Этап 3 (PeerLLMAgent single)
                  │            │
                  │       ┌────┴────┐
                  │       │         │
                  │  success    fail
                  │       │         │
                  │       ▼         ▼
                  │   Этап 4    pivot:
                  │   (persist)  понять почему LLM
                  │      │       не decide'ит — это
                  │      ▼       сам по себе finding
                  │   Этап 5
                  │   (multi)
                  │      │
                  │      ▼
                  │   Этап 6
                  │   (eval)
                  │
                  └──► extractor шумный (< 50% precision)
                              │
                              ▼
                         pivot: paper про
                         symbol grounding для
                         multi-agent. Использовать
                         какоhe-стек как negative
                         baseline.
```

В каждой точке возврата — **минимум 2-3 дня** на анализ результата прежде чем перейти к следующему этапу. Никаких «прошли быстро дальше» — каждый этап даёт нетривиальные insights которые меняют дизайн следующего.

---

## 9. Самопроверка после реализации (definition of done)

Готовы заявлять «у нас работает прототип распределённой коллективной долговременной памяти для LLM-агентов» когда:

- [ ] Сторонний разработчик может запустить `python -m experiments.llm_collective.run_collective` и получить трассу из N=3 LLM-агентов, кооперативно решающих ALFWorld task
- [ ] В трассе видно ненулевую кросс-агентную передачу знаний через broadcast
- [ ] Q/R/M/C события эмитятся и conditional rates не вырождены ($P(M|R)$, $P(C|M)$ — оба в диапазоне $[0.3, 0.95]$)
- [ ] Между двумя сессиями того же агента восстанавливается персистентная память (verified в trace)
- [ ] Vector baseline измерен на той же задаче, сравнение записано
- [ ] Один из трёх возможных вариантов result'а зафиксирован честно — без cherry-picking

---

## 10. Ближайший конкретный шаг

**Сегодня / завтра:** запустить Этап 1.

Я пишу `experiments/llm_collective/llm_tag_extractor.py` за один заход. Локальный Llama-3-8B через ollama (Mac M2/M3 справится). 10-наблюдательный smoke на ALFWorld scene descriptions. Печать extracted tags + cache hit/miss stats.

Что вы решаете в этой точке:
1. Какой backend по умолчанию — Llama-3-8B локально или Claude Haiku через API?
2. Vocabulary — фиксированный (open-set с post-hoc canonicalization) или свободный?
3. ALFWorld vs TextWorld для первого smoke — берём что проще?

Без ваших ответов на (1-3) Этап 1 не блокируется — defaults: ollama + open-set + ALFWorld. Можно переопределить позже без переписывания extractor'а.

---

**Дата создания:** 12.06.2026
**Последнее обновление:** 12.06.2026
**Файл:** `docs/ROADMAP_LLM_COLLECTIVE_MEMORY_2026-06-12.md`
