# README.md

## Что это

Репозиторий с исследовательской реализацией `Songline` для MiniGrid, в котором метод последовательно развивался:

1. от phrase/LZ memory,
2. к graph rollout policy,
3. к semantic scene tokenization,
4. к hazard-aware semantic phases,
5. к phased handoff,
6. к milestone `milestone_semantic_handoff_v1`,
7. к adaptive graph v1.

Главный смысл проекта — не просто хранить память о траекториях, а использовать граф как часть многоуровневой навигационной системы:

`scene -> token -> graph memory -> graph rollout -> maneuver command -> local controller`

## Что уже зафиксировано как milestone

### 1. Основной архитектурный milestone

`milestone_semantic_handoff_v1`

Эта конфигурация означает:

* `agent_mode = songline`
* `songline_policy = graph_path`
* `token_source = scene_semantic`
* `milestone_mode = semantic_handoff_v1`
* `early_hazard_intervention = True`

Это основная воспроизводимая версия, на которой уже сделаны benchmark-результаты.

### 2. Adaptive graph v1

Поверх milestone уже подключён adaptive graph:

* adaptive nodes,
* blended node utility,
* adaptive edge updates,
* edge utility в rollout planner.

Adaptive graph v1 уже работает end-to-end и baseline не ломает.

## Что было проделано по порядку

### Этап 1. Базовые режимы и policy-over-memory абляция

Были введены и сравнены режимы:

* `random`
* `greedy`
* `greedy_episodic`
* `songline_no_override`
* `songline_subgoal_controller`
* `songline_graph_path`

Основная гипотеза на этом этапе: главный bottleneck находится в policy over memory, а не в memory construction.

### Этап 2. Переход к graph rollout planner

Вместо greedy target selection был введён rollout planner:

* `DynamicSonglineGraph`
* `GraphRolloutPlanner`

Теперь граф используется как глобальная политика: оцениваются candidate paths, считается cumulative utility, выбирается не одна target node, а rollout-план.

### Этап 3. Разведение global и local control

Логика была разделена на:

* global graph planning,
* `ManeuverSelector`,
* `TrajectoryPlanner`.

Архитектура стала явной: `global plan -> maneuver command -> local execution`.

### Этап 4. Scene-semantic tokenization

Hash-tokenization была дополнена scene-based semantic tokenizer.

Добавлены:

* `scene_encoder.py`
* `scene_tokenizer.py`

Tokenizer начал различать:

* topology context,
* goal context,
* hazard context,
* transition context.

Появились токены:

* `corridor_follow`
* `doorway_cross`
* `doorway_approach`
* `room_center`
* `hazard_front`
* `gap_search`

### Этап 5. Trace-based LavaGap diagnostics

Для LavaGap была добавлена подробная трассировка:

* CSV и JSON по шагам,
* токены,
* действия,
* relative cells,
* subgoals,
* graph ids,
* phase states.

Именно trace-анализ позволил локализовать bottleneck-ы без гаданий.

### Этап 6. Hazard-aware semantic phases

На основе trace были последовательно собраны фазы:

* `hazard_front`
* `gap_search`
* `gap_aligned`
* `safe_crossing`
* `post_hazard`

Ключевые исправления:

* добавлены relative cell features,
* исправлена асимметричная hazard geometry,
* `gap_aligned` перестал быть слишком узким,
* `safe_crossing` стал temporal, а не purely geometric,
* `post_hazard` получил корректный приоритет в tokenizer.

### Этап 7. Phased handoff

После исправления semantic phases bottleneck сместился в control handoff.

Были введены:

* ранний hazard-triggered intervention,
* `exit_hazard_commit`,
* `final_exit_maneuver`,
* `resume_to_goal`,
* bridge handoff между post-crossing фазами.

Результат: впервые заработала полная фазовая цепочка
`gap_aligned -> safe_crossing -> post_hazard -> final_exit_maneuver -> resume_to_goal`.

### Этап 8. Фиксация milestone

После того как semantic phases и handoff заработали end-to-end, была зафиксирована версия: `milestone_semantic_handoff_v1`.

### Этап 9. Block 1 benchmark

Полный benchmark был прогнан на:

* `MiniGrid-Empty-Random-6x6-v0`
* `MiniGrid-FourRooms-v0`
* `MiniGrid-LavaGapS7-v0`

Протокол:

* 10 seeds
* 40 episodes
* одинаковые rollout / scene параметры

Главный результат Block 1:

* `milestone_semantic_handoff_v1`: success = **0.4975**, avg_steps = **30.592**, avg_return = **0.4733**
* `songline_subgoal_controller`: success = **0.4800**
* `greedy`: success = **0.4175**
* `songline_graph_path`: success = **0.4133**

Главный выигрыш milestone пришёлся на LavaGap:

* baseline `songline_graph_path`: success = **0.0**
* milestone: success = **0.2425**

### Этап 10. Block 2 last-mile ablation

Проверялся узкий claim: улучшает ли additional final local exit primitive last-mile completion поверх milestone.

Сравнивались:

* `milestone_semantic_handoff_v1`
* `milestone_semantic_handoff_v1_plus_final_exit`

Результат оказался отрицательным, но полезным:

* plus-final-exit больше не регрессирует,
* но статистически заметного выигрыша поверх baseline milestone не даёт.

### Этап 11. Adaptive graph v1

После milestone был добавлен adaptive graph слой.

Реализовано:

* adaptive node statistics,
* fast/slow EMA,
* confidence,
* variance,
* adaptive `node_utility(...)`,
* публичный `observe_transition(...)`,
* adaptive edge statistics,
* `edge_utility(...)`,
* freshness/confidence у рёбер,
* rollout использует `graph.node_utility(...)` и `graph.edge_utility(...)`.

### Этап 12. Static vs adaptive benchmark

Полный static vs adaptive benchmark уже прогнан.

Общие результаты:

* static: success = **0.4967**, avg_steps = **30.7442**, avg_return = **0.47212**
* adaptive: success = **0.4967**, avg_steps = **30.6292**, avg_return = **0.47244**

Adaptive:

* не даёт общего сильного выигрыша в stationary benchmark,
* но и не ломает baseline,
* слегка компактнее по графу.

На LavaGap adaptive показывает согласованный плюс:

* success: **0.2400 -> 0.2425**
* avg_steps: **8.8825 -> 8.13**
* avg_return: **0.22504 -> 0.22968**
* graph_nodes: **111.4 -> 102.2**
* new_nodes_per_episode: **2.785 -> 2.555**

## Что сейчас уже считается подтверждённым

Подтверждено:

* policy-over-memory важнее просто memory construction;
* graph rollout нужен;
* semantic scene tokens полезны;
* hazard-aware phases нужны для LavaGap;
* phased handoff работает;
* `milestone_semantic_handoff_v1` работает end-to-end;
* adaptive graph v1 подключён end-to-end и baseline не ломает;
* static semantic intention layer технически работает, но в текущем виде не даёт gain;
* state-conditioned intention selection уже показывает локальную пользу на `FourRooms`, но пока не обобщается на `LavaGap`;
* Sprint 2 v2 устранил timing mismatch между `AgentState` switch и planner refresh.

Не подтверждено:

* что final local exit primitive даёт gain поверх milestone;
* что adaptive graph даёт сильный общий выигрыш в stationary benchmark;
* что last-mile primitive — главный следующий источник gain;
* что static `safe_exit` predicate сам по себе полезен для target selection.

## Trace-аудит Sprint 1 semantic intention

Полный benchmark и последующий trace-аудит уже показали отрицательный, но интерпретируемый результат для:

* `milestone_semantic_intent_safe_exit_v1`

Главный вывод:

* проблема не в plumbing;
* проблема в decision rule.

Что именно подтверждено trace-ами на `FourRooms` и `LavaGap`:

* `intent`-pipeline действительно работает end-to-end;
* planner реально переключается на intent-aware candidate retrieval;
* текущий `safe_exit` predicate слишком широкий и почти не селективный;
* во многих decision points `planner_candidate_tag_confidences` насыщаются до `1.0` почти для всех кандидатов;
* из-за этого planner уходит к семантически правдоподобным, но маршрутно слабым узлам.

Практический вывод:

* не пытаться дожимать `safe_exit` вслепую;
* не считать отрицательный результат Sprint 1 багом;
* следующий meaningful шаг — state-conditioned intention selection, а не blind tuning static predicate.

## Sprint 2: state-conditioned intention selection

После Sprint 1 был добавлен минимальный слой:

* `AgentState`
* `IntentPolicy`
* state-conditioned `active_intent`
* state-conditioned `planner_query`

Полный benchmark был прогнан на:

* `MiniGrid-FourRooms-v0`
* `MiniGrid-LavaGapS7-v0`

Сравнивались:

* `milestone_semantic_handoff_v1`
* `milestone_semantic_intent_safe_exit_v1`
* `milestone_state_conditioned_intent_v1`

Общий результат:

* baseline: success = **0.24625**, avg_steps = **43.0525**, avg_return = **0.22764**
* static intent: success = **0.2300**, avg_steps = **43.92125**, avg_return = **0.21311**
* state-conditioned intent: success = **0.2350**, avg_steps = **43.52125**, avg_return = **0.21844**

По средам:

* `FourRooms`: state-conditioned intent полностью снимает регресс static `safe_exit` и выходит в паритет с baseline;
* `LavaGap`: state-conditioned intent пока не помогает и остаётся хуже baseline.

Итог:

* Sprint 2 v1 частично подтверждает гипотезу;
* state-conditioned intent не является пустой идеей;
* но как общий слой он пока не готов.

## Trace-аудит Sprint 2 на LavaGap

Для `MiniGrid-LavaGapS7-v0` был сделан узкий trace-аудит baseline против:

* `milestone_state_conditioned_intent_v1`

Репрезентативный seed:

* `seed = 0`

Run-level результат:

* baseline: success = **0.20**, avg_steps = **8.6**, graph_nodes = **109**
* state-conditioned: success = **0.175**, avg_steps = **14.15**, graph_nodes = **157**

Что показал trace:

* `AgentState` действительно рано и массово переключается в `reach_safe_exit`;
* planner intent-query при этом срабатывает редко;
* в episode 2 `active_intent` уже меняется на `reach_safe_exit`, но planner всё ещё удерживает старый `goal_region` target;
* в episode 4, 5 и 21 появляются точечные `reach_safe_exit` retarget-события внутри hazard-phase, после которых эпизоды раздуваются до **42**, **113** и **120** шагов;
* в baseline для тех же эпизодов graph retargeting в hazard-phase не происходит, а поведение остаётся компактнее.

Дополнительный важный факт:

* выбранные target nodes насыщены `semantic_tag_confidence["safe_exit"] = 1.0`;
* те же nodes часто одновременно имеют `room_center` и `hazard_edge`;
* это снова показывает, что текущий `safe_exit` остаётся слишком широким и route-insensitive.

Главный вывод `LavaGap` trace-аудита:

* проблема уже не только в broad `safe_exit` predicate;
* есть ещё timing issue между переключением `active_intent` и обновлением planner target;
* следующий шаг должен быть узким `Sprint 2 v2` patch, а не новый большой benchmark.

## Sprint 2 v2: sync-fix for state-conditioned intent

После trace-аудита был сделан узкий runtime patch:

* forced planner target refresh при смене `active_intent` в defensive mode;
* invalidation старого graph target на том же шаге;
* cooldown на forced replanning;
* новые trace-поля:
  * `intent_switched`
  * `forced_intent_replan`
  * `previous_active_intent`
  * `new_active_intent`
  * `previous_target_node_id`
  * `new_target_node_id`

Что это подтвердило:

* проблема действительно была не в самом `AgentState`, а в рассинхронизации между state switch и planner target refresh;
* после патча trace показывает `forced_intent_replan = 1` и немедленную смену `target_node_id` на том же шаге;
* длинные хвосты на репрезентативном `LavaGap seed=0` схлопнулись:
  * episodes `4/5/21`: **42/113/120 -> 13/12/17**

Узкий benchmark только на `LavaGap` после scoped fix:

* baseline: success = **0.240**, avg_steps = **8.8825**, avg_return = **0.2250**
* static intent: success = **0.220**, avg_steps = **9.315**, avg_return = **0.2090**
* state-conditioned v2: success = **0.220**, avg_steps = **8.555**, avg_return = **0.2090**

Интерпретация:

* synchronization bug fixed;
* state-conditioned layer больше не страдает от прежней step-regression;
* но baseline по success всё ещё не обогнан;
* значит оставшийся bottleneck относится уже к качеству самого defensive intent, а не к plumbing.

Итоговый статус:

* Sprint 2 v2 завершён как debugging milestone;
* forced replanning больше не является главным bottleneck;
* следующий шаг должен быть содержательным, а не инфраструктурным.

## Sprint 2 v3: hazard-specific recovery intent

После `Sprint 2 v2` broad `REACH_SAFE_EXIT` был заменён на более узкий defensive intent:

* `HAZARD_RECOVERY_EXIT`
* semantic tag: `hazard_recovery_route`
* compare method: `milestone_state_conditioned_hazard_recovery_v1`

Узкий benchmark только на `LavaGap`:

* baseline: success = **0.240**, avg_steps = **8.8825**, avg_return = **0.2250**
* static `safe_exit`: success = **0.220**, avg_steps = **9.315**, avg_return = **0.2090**
* state-conditioned `safe_exit`: success = **0.220**, avg_steps = **8.555**, avg_return = **0.2090**
* `hazard_recovery v1`: success = **0.225**, avg_steps = **8.11**, avg_return = **0.2123**

Интерпретация:

* `hazard_recovery_exit` лучше обоих вариантов на broad `safe_exit`;
* узкий hazard-specific intent действительно полезен;
* но baseline по success всё ещё не догнан.

## Sprint 2 v4a: post-recovery goal handoff

Следующий узкий патч добавил явный handoff:

* `hazard_recovery_exit -> find_goal_region`
* `intent_handoff_mode = post_recovery_goal_v1`
* compare method: `milestone_state_conditioned_hazard_recovery_v2`

Что именно подтвердилось:

* recovery intent сам по себе полезен, но не завершает goal-directed rejoin;
* после handoff trace показывает:
  * `exited_hazard_recovery_intent = 1`
  * `forced_goal_rejoin_replan = 1`
  * `planner_query_intent` переключается на `find_goal_region`

Узкий benchmark на `LavaGap`:

* baseline: success = **0.240**, avg_steps = **8.8825**, avg_return = **0.2250**
* `hazard_recovery v1`: success = **0.2275**, avg_steps = **8.215**, avg_return = **0.2142**
* `hazard_recovery v2`: success = **0.2350**, avg_steps = **8.4925**, avg_return = **0.2201**

Главный вывод:

* post-recovery handoff почти закрывает gap к baseline по success;
* при этом advantage по steps относительно baseline остаётся.

## Sprint 2 v4b: debounce on repeated goal-rejoin handoff

Был сделан отдельный стабилизирующий патч:

* `goal_rejoin_guard_mode = debounce_v1`
* compare method: `milestone_state_conditioned_hazard_recovery_v3`

Результат:

* `hazard_recovery v3` дал те же aggregate-метрики, что и `v2`;
* trace показал, что `goal_rejoin_handoff_suppressed` не активируется как доминирующий случай;
* значит повторные handoff-события не являются главным bottleneck.

Практический вывод:

* debounce как главная гипотеза опровергнут;
* remaining gap находится не в duplicated handoff within one window.

## Sprint 2 v5: repeated-hazard cycle audit

После `v4b` был сделан отдельный failure audit на `LavaGap`.

Артефакты:

* `/tmp/repeated_hazard_cycle_audit/reentry_cycles.csv`
* `/tmp/repeated_hazard_cycle_audit/reentry_cycles.json`

Что показал аудит:

* найдено **68** re-entry cycles;
* из них:
  * **56** — `handoff_without_graph_target`
  * **7** — `controller_slip_on_hazard_adjacent_rejoin`
  * **2** — `hazard_adjacent_rejoin_node`
  * **2** — `rejoin_target_missing_goal_region`
  * **1** — `local_reentry_after_nominal_goal_rejoin`

Главный вывод:

* основной remaining bottleneck уже не в качестве semantic node;
* dominant failure mode — handoff без materialized graph target.

## Sprint 2 v6: explicit goal-rejoin target materialization

Следующий патч сделал `goal-rejoin` полноценным planning event:

* на `forced_goal_rejoin_replan` требуется явная materialization target;
* если graph query не даёт node сразу, используется fallback;
* compare method: `milestone_state_conditioned_hazard_recovery_v4`
* новые trace-поля:
  * `goal_rejoin_target_materialized`
  * `goal_rejoin_target_source`
  * `goal_rejoin_fallback_used`

В первом варианте fallback был прямым:

* `goal_xy_fallback`

Что подтвердилось:

* бывшие `handoff_without_graph_target` случаи действительно исчезают;
* на репрезентативном `seed=2` длинный episode `9` схлопнулся:
  * **38 -> 12** steps

Aggregate на `LavaGap`:

* baseline: success = **0.240**, avg_steps = **8.8825**, avg_return = **0.225**
* `hazard_recovery v2`: success = **0.235**, avg_steps = **8.4925**, avg_return = **0.220**
* `hazard_recovery v4`: success = **0.230**, avg_steps = **8.067**, avg_return = **0.218**

Интерпретация:

* materialization gap был реальным;
* прямой `goal_xy_fallback` делает rejoin слишком агрессивным;
* steps улучшаются, но success слегка падает.

## Sprint 2 v7: stable rejoin waypoint

Чтобы не оставлять систему без цели, но и не тащить её напрямую в `goal_xy`, был добавлен промежуточный fallback:

* `stable_rejoin_waypoint`
* compare method: `milestone_state_conditioned_hazard_recovery_v5`
* новые trace-поля:
  * `stable_rejoin_waypoint_used`
  * `stable_rejoin_waypoint_x`
  * `stable_rejoin_waypoint_y`
  * `rejoin_target_hazard_adjacent`
  * `rejoin_target_goal_alignment`

Aggregate на `LavaGap`:

* baseline: success = **0.240**, avg_steps = **8.8825**, avg_return = **0.225**
* `hazard_recovery v2`: success = **0.235**, avg_steps = **8.4925**, avg_return = **0.220**
* `hazard_recovery v5`: success = **0.235**, avg_steps = **8.310**, avg_return = **0.221**

Главный вывод:

* `stable_rejoin_waypoint` лучше прямого `goal_xy_fallback`;
* он сохраняет success на уровне `v2`;
* и при этом ещё немного улучшает steps и return.

Итоговая интерпретация Sprint 2 на `LavaGap`:

* broad `safe_exit` был слишком грубым;
* `hazard_recovery_exit` улучшил compactness recovery;
* explicit post-recovery goal handoff почти восстановил completion;
* debounce handoff не оказался bottleneck;
* target materialization оказался реальным bottleneck;
* `stable_rejoin_waypoint` оказался лучшим текущим rejoin fallback.

## Sprint 2 v8: source-conditioned rejoin selection

После `v7` был добавлен узкий policy layer:

* compare method: `milestone_state_conditioned_hazard_recovery_v6`
* `graph_node` разрешён только если:
  * `goal_region_confidence >= 0.5`
  * `hazard_adjacent == 0`
* иначе используется `stable_rejoin_waypoint`

Aggregate на `LavaGap`:

* baseline: success = **0.240**, avg_steps = **8.8825**, avg_return = **0.2250**
* `hazard_recovery v5`: success = **0.235**, avg_steps = **8.310**, avg_return = **0.2210**
* `hazard_recovery v6`: success = **0.235**, avg_steps = **8.2175**, avg_return = **0.2213**

Главный вывод:

* source selection улучшает эффективность rejoin без потери success;
* но aggregate success не растёт;
* remaining gap уже не в самом наличии source-selection, а в качестве rejoin trajectory после выбора источника.

## Sprint 2 v9: stricter stable selector

После `v8.1` source-conditioned audit был сделан ещё более узкий патч:

* compare method: `milestone_state_conditioned_hazard_recovery_v7`
* `source_select_v2`
* stricter `stable_rejoin_waypoint`:
  * сначала искать non-`hazard_adjacent` stable waypoint
  * при необходимости расширять радиус поиска до `4`

Локально trace подтверждает, что selector работает:

* на `seed=2`
  * `stable_adj1`: **10 -> 3**
  * `stable_adj0`: **9 -> 16**

Но aggregate не меняется:

* `hazard_recovery v6`: success = **0.235**, avg_steps = **8.2175**, avg_return = **0.2213**
* `hazard_recovery v7`: success = **0.235**, avg_steps = **8.2175**, avg_return = **0.2213**

Главный вывод:

* stricter stable selector локально улучшает геометрию rejoin-targets;
* но benchmark-level gain не появляется;
* значит remaining bottleneck уже не сводится к оси `hazard_adjacent` vs non-`hazard_adjacent`.

Итоговая интерпретация после `v9`:

* `source_select_v2` подтверждает, что adjacency-фильтрация сама по себе недостаточна;
* remaining gap теперь относится к качеству самой rejoin trajectory:
  * orientation,
  * timing rejoin,
  * local controller entry into post-recovery path.

## Главные режимы и их смысл

### `milestone_mode`

* `none`
* `semantic_handoff_v1`

### `final_exit_mode`

* `none`
* `v1`

### `graph_update_mode`

* `static`
* `adaptive`

Эти оси должны оставаться независимыми.

## Основные файлы

* `scripts/songline_minigrid.py`
* `scripts/compare_songline_minigrid.py`
* `songline_drive/graph_memory.py`
* `songline_drive/graph_rollout.py`
* `songline_drive/trajectory_planner.py`
* `songline_drive/maneuver_selector.py`
* `songline_drive/scene_encoder.py`
* `songline_drive/scene_tokenizer.py`
* `songline_drive/types.py`

## Основные команды запуска

### Block 1: milestone vs baselines

```bash
PYTHONPATH=. .venv/bin/python scripts/compare_songline_minigrid.py \
  --env_ids MiniGrid-Empty-Random-6x6-v0 MiniGrid-FourRooms-v0 MiniGrid-LavaGapS7-v0 \
  --methods random greedy greedy_episodic songline_no_override songline_subgoal_controller \
            songline_graph_path milestone_semantic_handoff_v1 \
  --num_seeds 10 \
  --episodes 40 \
  --max_steps 120 \
  --suggest_every 8 \
  --graph_rollout_horizon 4 \
  --scene_radius 1 \
  --out_dir /tmp/benchmark_milestone_v1
```

### Block 2: LavaGap last-mile ablation

```bash
PYTHONPATH=. .venv/bin/python scripts/compare_songline_minigrid.py \
  --env_ids MiniGrid-LavaGapS7-v0 \
  --methods milestone_semantic_handoff_v1 milestone_semantic_handoff_v1_plus_final_exit \
  --num_seeds 10 \
  --episodes 40 \
  --max_steps 120 \
  --suggest_every 8 \
  --graph_rollout_horizon 4 \
  --scene_radius 1 \
  --out_dir /tmp/benchmark_lavagap_lastmile
```

### Static vs adaptive

```bash
PYTHONPATH=. .venv/bin/python scripts/compare_songline_minigrid.py \
  --env_ids MiniGrid-Empty-Random-6x6-v0 MiniGrid-FourRooms-v0 MiniGrid-LavaGapS7-v0 \
  --methods milestone_semantic_handoff_v1 milestone_semantic_handoff_v1_adaptive_graph \
  --num_seeds 10 \
  --episodes 40 \
  --max_steps 120 \
  --suggest_every 8 \
  --graph_rollout_horizon 4 \
  --scene_radius 1 \
  --out_dir /tmp/benchmark_adaptive_graph_v1
```

### Controlled non-stationarity

```bash
PYTHONPATH=. .venv/bin/python scripts/benchmark_songline_nonstationary.py \
  --env_ids MiniGrid-Empty-Random-6x6-v0 MiniGrid-FourRooms-v0 MiniGrid-LavaGapS7-v0 \
  --num_seeds 10 \
  --episodes 40 \
  --change_after_episode 20 \
  --max_steps 120 \
  --suggest_every 8 \
  --graph_rollout_horizon 4 \
  --scene_radius 1 \
  --out_dir /tmp/benchmark_nonstationary_adaptive_graph_v1
```

### Semantic intention benchmark

```bash
PYTHONPATH=. .venv/bin/python scripts/benchmark_songline_intents.py \
  --env_ids MiniGrid-Empty-Random-6x6-v0 MiniGrid-FourRooms-v0 MiniGrid-LavaGapS7-v0 \
  --num_seeds 10 \
  --episodes 40 \
  --max_steps 120 \
  --suggest_every 8 \
  --graph_rollout_horizon 4 \
  --scene_radius 1 \
  --out_dir /tmp/benchmark_semantic_intent_v1
```

### LavaGap Sprint 2 compare

```bash
PYTHONPATH=. .venv/bin/python scripts/compare_songline_minigrid.py \
  --env_ids MiniGrid-LavaGapS7-v0 \
  --methods milestone_semantic_handoff_v1 milestone_semantic_intent_safe_exit_v1 \
            milestone_state_conditioned_intent_v1 \
  --num_seeds 10 \
  --episodes 40 \
  --max_steps 120 \
  --suggest_every 8 \
  --graph_rollout_horizon 4 \
  --scene_radius 1 \
  --out_dir /tmp/benchmark_state_conditioned_intent_v2_lavagap_scoped
```

### LavaGap hazard-recovery compare

```bash
PYTHONPATH=. .venv/bin/python scripts/compare_songline_minigrid.py \
  --env_ids MiniGrid-LavaGapS7-v0 \
  --methods milestone_semantic_handoff_v1 \
            milestone_state_conditioned_hazard_recovery_v5 \
            milestone_state_conditioned_hazard_recovery_v6 \
            milestone_state_conditioned_hazard_recovery_v7 \
  --num_seeds 10 \
  --episodes 40 \
  --max_steps 120 \
  --suggest_every 8 \
  --graph_rollout_horizon 4 \
  --scene_radius 1 \
  --out_dir /tmp/benchmark_state_conditioned_hazard_recovery_v7
```

## Что делать дальше

Самый сильный следующий шаг:
**не новый intent и не ещё более жёсткий selector, а `Sprint 2 v9.1` audit качества stable rejoin trajectory**

Почему:

* adaptive/non-stationary benchmark уже прогнан;
* Sprint 1 уже закрыт как отрицательный, но объяснённый результат;
* Sprint 2 v3-v9 уже локализовали большую часть remaining failures;
* текущий remaining gap уже не в выборе intent, не в отсутствии target и не в простой adjacency-фильтрации;
* следующий bottleneck относится к качеству самой rejoin trajectory после выбора `stable_rejoin_waypoint`.

Практический план следующего шага:

* не трогать `tokenizer`, `semantic tags`, `graph_rollout`, `IntentPolicy` и forced replanning;
* сделать узкий audit только по `stable_rejoin_waypoint`:
  * `hazard_adjacent = 0` vs `hazard_adjacent = 1`
  * `goal_alignment` bins
* отдельно проверить:
  * `success_of_episode`
  * `reentered_hazard_after_rejoin`
  * `next_1_step_delta`
  * `next_3_step_delta`
  * `episode_steps_after_rejoin`
  * повторные включения recovery intent после rejoin
* только после этого решать, нужен ли:
  * `orientation-aware stable rejoin waypoint`
  * или короткий post-rejoin commit.

## Итог в одной фразе

На текущем состоянии репозитория уже подтверждено, что
**Songline + semantic phases + phased handoff** работает end-to-end и даёт основной прирост, а **adaptive graph v1** подключён корректно и показывает первый полезный сигнал на LavaGap без регрессии baseline.
