# Goals: Semantic Intention and Water Task

## Current Status

Текущий water-case уже можно считать рабочим demonstrator главной идеи проекта.

Что уже подтверждено:

* `FIND_WATER_SOURCE` добавлен как настоящий `IntentType`, а не как special-case в planner.
* water evidence проходит полный контур:
  * `scene -> semantic tags -> graph memory -> planner query -> target materialization -> waypoint/action`
* water task больше не завязан на заранее известный `goal_xy`;
* есть реальная behavioral validation:
  * retrieval по воде materialize-ится в trace как planner-level event;
  * water-task success считается по достижению water-marker, а не по штатному `Goal`.
* state-conditioned water activation тоже уже реализован и работает.

Короткий честный вывод:

* water-case уже рабочий как semantic-intention demonstrator;
* это ещё не большой benchmark claim, но это уже не просто plumbing.

## Main Goal

Перейти от постановки задачи через `goal_xy` к постановке через semantic intention.

Целевая схема должна быть такой:

`AgentState / task -> Intent -> Semantic predicate -> Candidate places in graph -> Ranked target -> Waypoint -> Local action`

То есть агент должен спрашивать не только:

* куда идти по координате

а в первую очередь:

* какое место в памяти и в текущей сцене лучше всего соответствует моему намерению

## Core Principle

Главная цель не в том, чтобы научить агента искать именно воду как special-case.

Главная цель такая:

* научить агента выбирать и достигать места, определяемые semantic patterns, а не заранее заданными координатами

`FIND_WATER_SOURCE` должен быть первым полноценным case study этой общей архитектуры.

## Why Water

Вода подходит как первый semantic task, потому что:

* она естественно задаётся как тип места, а не как координата
* её можно описать через semantic evidence
* её можно связать с `AgentState`, например через `thirst`
* она хорошо масштабируется к более общим задачам:
  * ресурс
  * пища
  * безопасное место
  * место отдыха
  * объект-паттерн

## Target Architecture

Нужна следующая цепочка:

1. `AgentState` определяет потребность или задачу.
2. Из неё выбирается `IntentType`.
3. `IntentType` порождает semantic predicate/query.
4. Scene layer собирает локальное semantic evidence.
5. Graph memory хранит evidence per node.
6. Planner ищет и ранжирует candidate nodes по соответствию намерению.
7. Выбранный target превращается в waypoint.
8. Local controller исполняет движение.

## Immediate Next Goal

Сделать первый полноценный semantic task:

* `FIND_WATER_SOURCE`

Этот intent должен быть задан не через координату, а через semantic predicate над признаками места.

## Sprint A: Water As Semantic Task

Цель спринта:

* сделать `FIND_WATER_SOURCE` как первый semantic task без multi-agent и без нового controller

Что нужно сделать:

1. Ввести новый intent:
   * `IntentType.FIND_WATER_SOURCE`
2. Добавить water-related semantic evidence в scene layer.
3. Хранить water evidence per node в graph memory.
4. Сделать predicate-based query для water.
5. Прогнать один smoke/demo case.

### Sprint A Status

Sprint A завершён.

Что уже реализовано:

* `IntentType.FIND_WATER_SOURCE`
* composite `SemanticTargetPredicate` для воды
* water evidence в `scene_encoder`
* water semantic tags в `scene_tokenizer`
* water evidence per node в `graph_memory`
* generic intent scoring и planner-level retrieval

Что это значит:

* вода уже представляется не только как scene-level признак,
* но и как graph-level semantic hypothesis about a place.

### Sprint A.1 Status

Добавлен минимальный water-task wrapper.

Что это дало:

* появился отдельный `task_mode = water_search_v1`;
* success теперь измеряется по достижению water-marker;
* `find_water_source` начал materialize-иться как реальный planner event;
* trace показывает `planner_query_intent = find_water_source`, candidate nodes и `target_node_id`.

Итог:

* переход от `goal_xy` к semantic target для первого case study уже состоялся.

### Files To Extend

* [types.py](/Users/taniyashuba/PycharmProjects/Songlines/songline_drive/types.py)
* [intents.py](/Users/taniyashuba/PycharmProjects/Songlines/songline_drive/intents.py)
* [scene_encoder.py](/Users/taniyashuba/PycharmProjects/Songlines/songline_drive/scene_encoder.py)
* [scene_tokenizer.py](/Users/taniyashuba/PycharmProjects/Songlines/songline_drive/scene_tokenizer.py)
* [graph_memory.py](/Users/taniyashuba/PycharmProjects/Songlines/songline_drive/graph_memory.py)
* [graph_rollout.py](/Users/taniyashuba/PycharmProjects/Songlines/songline_drive/graph_rollout.py)
* [agent_state.py](/Users/taniyashuba/PycharmProjects/Songlines/songline_drive/agent_state.py)

### New Intent

Добавить:

```python
IntentType.FIND_WATER_SOURCE
```

## Water Semantic Evidence

На первом этапе детектор воды может быть rule-based.

Нужны признаки уровня scene:

* `water_visible`
* `water_pattern_match`
* `water_accessible`
* `water_neighbor_context`
* `water_confidence_local`

В tokenizer это должно стать semantic tags:

* `water_source`
* `water_candidate`
* `water_nearby`

Важно:

* не делать здесь сразу сложную learned-модель
* не делать воду жёстким special-case в planner

## Graph Memory Goal

Graph memory уже хранит:

* `semantic_tag_counts`
* `semantic_tag_confidence`

Следующий шаг:

* хранить water-evidence per node
* различать как минимум:
  * вода явно видна
  * место похоже на источник воды
  * к воде можно безопасно подойти

Graph node должен стать semantic hypothesis about a place, а не просто visited phrase.

## Planner Goal

Planner не должен знать “что такое вода” как отдельную жёсткую сущность.

Он должен работать через общую абстракцию:

* `IntentType`
* `PlannerQuery`
* `SemanticPredicate`
* `candidate_nodes_for_intent(...)`

Для воды следующий шаг после простого threshold predicate:

* composite predicate / scoring predicate

Пример логики:

* высокий `water_source_confidence`
* `water_accessible == 1`
* `hazard_adjacent` не слишком высокий
* optional exploration terms, если вода ещё не подтверждена

## Sprint B: State-Driven Water Seeking

После Sprint A следующий слой:

* связать `FIND_WATER_SOURCE` с `AgentState`

Минимальная логика:

* если `thirst` высокая -> активировать `FIND_WATER_SOURCE`
* если `risk_budget` низкий -> искать только безопасную воду
* если `energy` низкая -> не уходить в дальний exploration
* если вода уже рядом -> переходить из broad search в approach mode

Это и есть настоящий переход от:

* идти к цели

к:

* иметь потребность
* выбирать semantic intention
* искать место по его свойствам

### Sprint B Status

Sprint B v2 тоже уже реализован.

Что подтверждено:

* `thirst` теперь реально участвует в выборе между `FIND_GOAL_REGION` и `FIND_WATER_SOURCE`;
* policy использует hysteresis через `thirst_on_threshold` / `thirst_off_threshold`;
* activation может удерживаться и по local water evidence;
* trace пишет:
  * `previous_active_intent`
  * `new_active_intent`
  * `intent_switch_reason`
  * `agent_state_thirst`

Практический результат:

* поздний `state_v1` терял retrieval;
* evidence-aware `state_v2` доведён до рабочего качества и вышел в паритет с fixed water intent на текущем demo scenario.

Честная формулировка:

* state-conditioned water seeking уже работает как механизм;
* теперь воду можно показывать как первый рабочий пример semantic intention + internal need.

## Sprint C: Multi-Agent Only After Water

К multi-agent стоит переходить только после semantic intention для одного агента.

Причина:

* иначе получится дублирование текущей waypoint-архитектуры, а не semantic coordination

Правильный следующий язык для multi-agent:

* агент A ищет воду
* агент B ищет безопасный маршрут
* semantic targets можно делить
* graph nodes можно claim-ить и шарить

## What Not To Do

Пока не нужно:

* делать multi-agent раньше water task
* делать воду special-case внутри planner
* смешивать это с очередным `LavaGap` tuning
* писать water-only hacks в local controller
* добавлять сложную learned detector stack до базового rule-based semantic layer

Отдельно:

* не возвращаться к `goal_xy` как к основной постановке water-task;
* не интерпретировать water-case как “ещё один LavaGap tuning”;
* не ломать generic intent abstraction ради water-specific shortcuts.

## Practical Sequence

Следующий рациональный порядок:

1. Ввести `FIND_WATER_SOURCE`.
2. Добавить water-related semantic tags в scene encoder/tokenizer.
3. Научить graph memory хранить water evidence per node.
4. Сделать composite predicate query для water.
5. Привязать activation к `thirst`.
6. Только потом идти в multi-agent.

Этот шаг уже частично выполнен:

* Sprint A завершён;
* Sprint A.1 завершён;
* Sprint B v2 завершён.

Следующий практический шаг для water-line теперь такой:

1. Сделать reproducible demo/benchmark block именно для water-case.
2. Проверить, даёт ли state-conditioned water retrieval gain поверх fixed water не только на коротком smoke, но и на более устойчивом сценарии.
3. Только после этого переходить к multi-agent semantic coordination.

## Technical Note

Water wrapper больше не ломает `gymnasium` observation validation.

Причина старого warning была такой:

* wrapper подменял `mission` строкой `"find the water source"`;
* это значение не входило в `MissionSpace`;
* из-за этого `observation_space.contains(obs)` возвращал `False`.

Теперь:

* `mission` среды не мутируется;
* описание water-task пишется в `info["task_mission"]`;
* warning про observation space закрыт.

## One-Sentence Goal

Не “научить агента искать воду по координате”, а:

* научить агента выбирать и достигать места, определяемые semantic patterns, а не заданными координатами

Вода — первый сильный case study для этой архитектуры.
