# README_songline_repo.md

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
* adaptive graph v1 подключён end-to-end и baseline не ломает.

Не подтверждено:

* что final local exit primitive даёт gain поверх milestone;
* что adaptive graph даёт сильный общий выигрыш в stationary benchmark;
* что last-mile primitive — главный следующий источник gain.

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

## Что делать дальше

Самый сильный следующий шаг:
**controlled non-stationarity benchmark**

Почему:

* в stationary benchmark adaptive уже показал “not worse + slight LavaGap signal”;
* основная ценность adaptive updates должна проявляться при меняющейся среде.

## Итог в одной фразе

На текущем состоянии репозитория уже подтверждено, что
**Songline + semantic phases + phased handoff** работает end-to-end и даёт основной прирост, а **adaptive graph v1** подключён корректно и показывает первый полезный сигнал на LavaGap без регрессии baseline.