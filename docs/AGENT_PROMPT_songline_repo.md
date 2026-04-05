
# AGENT PROMPT.md

## Назначение

Этот блок нужен агенту, который продолжает работу в репозитории.

Главная задача агента:

* не ломать уже зафиксированный milestone,
* продолжать разработку через узкие проверяемые гипотезы,
* не смешивать benchmark plumbing, semantic logic, local control и adaptive graph в один большой неинтерпретируемый патч.

## Что уже считается зафиксированным

### Основной milestone

`milestone_semantic_handoff_v1`

Он включает:

* `agent_mode = songline`
* `songline_policy = graph_path`
* `token_source = scene_semantic`
* `milestone_mode = semantic_handoff_v1`
* `early_hazard_intervention = True`

### Adaptive layer

`adaptive_graph_v1`

Он уже включает:

* adaptive nodes,
* adaptive edge updates,
* blended rollout utility,
* `graph_update_mode = adaptive`

## Что уже доказано

1. Бутылочное горлышко находится в `policy over memory`, а не просто в `memory construction`.
2. `graph rollout` полезнее, чем greedy target-node selection.
3. `scene_semantic` токены полезны и интерпретируемы.
4. Для LavaGap нужны explicit semantic phases.
5. `phased handoff` закрывает основной architectural bottleneck.
6. `milestone_semantic_handoff_v1` работает end-to-end.
7. `adaptive graph v1` подключён end-to-end и не ломает baseline.
8. В stationary benchmark adaptive graph не даёт общего сильного выигрыша, но показывает полезный сигнал на LavaGap.

## Что пока не доказано

1. Что `final_exit_mode=v1` даёт статистический gain поверх milestone.
2. Что adaptive graph существенно лучше static в stationary benchmark.
3. Что last-mile primitive сейчас является главным рычагом дальнейшего улучшения.

## Нельзя ломать

### 1. Milestone baseline

Если ты меняешь что-то новое, `milestone_semantic_handoff_v1` должен оставаться воспроизводимым.

### 2. Compare pipeline

Новые режимы обязаны корректно проходить через:

* `method_to_config(...)`
* run_results
* aggregate_by_env
* aggregate_overall
* summary_table

### 3. Независимость осей конфигурации

Нельзя смешивать:

* `milestone_mode`
* `final_exit_mode`
* `graph_update_mode`

Они должны оставаться независимыми.

## Основные оси конфигурации

### `milestone_mode`

* `none`
* `semantic_handoff_v1`

### `final_exit_mode`

* `none`
* `v1`

### `graph_update_mode`

* `static`
* `adaptive`

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

## Правила работы агента

1. Не делай большой патч без гипотезы.
2. После каждого meaningful change делай проверку: `py_compile`, smoke-run, trace или benchmark.
3. Если проблема неясна — сначала trace.
4. Не выдавай отрицательный результат за баг.

## Какие benchmark-и уже есть

### Block 1

* Empty
* FourRooms
* LavaGap

### Block 2

* milestone baseline
* milestone + final exit

### Static vs adaptive

* milestone static graph
* milestone adaptive graph

## Основные уже существующие метрики

### Общие

* `success_rate`
* `avg_steps_to_goal`
* `avg_return`
* `graph_nodes`
* `graph_edges`
* `new_nodes_per_episode`
* `intervention_rate`
* `subgoal_reach_rate`
* `goal_distance_delta_per_intervention`

### Фазовые

* `fraction_gap_aligned`
* `fraction_safe_crossing`
* `fraction_post_hazard`
* `fraction_final_exit_maneuver`
* `fraction_resume_to_goal`
* `mean_max_phase_depth`

### Last-mile

* `fraction_post_hazard_progress`
* `fraction_resume_to_goal_progress`
* `fraction_post_hazard_to_success`
* `fraction_resume_to_goal_to_success`
* `conditional_post_hazard_success`
* `conditional_resume_to_goal_success`

## Текущее научное состояние проекта

* semantic handoff milestone уже доказан;
* adaptive graph v1 уже стабилен и end-to-end подключён;
* общий stationary gain adaptive graph пока слабый;
* strongest positive adaptive signal уже виден на LavaGap;
* controlled non-stationarity benchmark уже прогнан;
* static semantic intention layer уже прогнан и дал отрицательный, но объяснённый результат;
* Sprint 2 v1 уже прогнан;
* state-conditioned intention selection исправляет failure mode Sprint 1 на `FourRooms`, но пока не обобщается на `LavaGap`;
* Sprint 2 v2 уже устранил timing mismatch между `AgentState` switch и planner refresh.

## Что уже известно про Sprint 1 intention layer

Проверенный negative result:

* `milestone_semantic_intent_safe_exit_v1` не даёт прироста поверх milestone baseline;
* plumbing intention layer работает корректно end-to-end;
* провал находится в decision rule, а не в benchmark plumbing;
* текущий `safe_exit` predicate слишком широкий и плохо селективный;
* на `FourRooms` это даёт лишний churn и рост графа;
* на `LavaGap` intent может включаться слишком рано и уводить от hazard-relevant structure.

Практический вывод для следующего агента:

* не пытайся ещё раз тюнить `safe_exit` вслепую;
* не интерпретируй Sprint 1 как баг;
* если хочешь менять intention layer до Sprint 2, делай только узкий trace-based predicate experiment;
* Sprint 1 уже закрыт;
* Sprint 2 v2 уже закрыл debugging-слой на `LavaGap`;
* следующий фокус — узкий `Sprint 2 v3` с новым defensive intent.

## Что уже известно про Sprint 2 state-conditioned layer

Проверенный промежуточный результат:

* `milestone_state_conditioned_intent_v1` лучше static `safe_exit`, но пока не лучше baseline в aggregate;
* на `FourRooms` state-conditioned слой полностью снимает регресс Sprint 1 и выходит в паритет с baseline;
* на `LavaGap` state-conditioned слой остаётся хуже baseline.

Что уже показал узкий trace-аудит `LavaGap seed=0`:

* `AgentState` рано и массово переключается в `reach_safe_exit`;
* planner intent-query при этом срабатывает редко;
* в episode 2 `active_intent` уже становится `reach_safe_exit`, но planner всё ещё держит старый `goal_region` target;
* в episode 4, 5 и 21 появляются `reach_safe_exit` retarget-события внутри hazard-phase, после которых эпизоды раздуваются до 42, 113 и 120 шагов;
* выбранные target nodes насыщены `safe_exit = 1.0` и часто одновременно помечены как `room_center` / `hazard_edge`.

Практический вывод:

* проблема `LavaGap` уже не только в широком predicate;
* есть ещё timing issue между state switch и planner target refresh;
* следующий патч должен быть узким и LavaGap-specific по механике, а не новым общим benchmark-циклом.

## Что уже известно про Sprint 2 v2 sync-fix

Проверенный результат:

* forced planner refresh на intent switch действительно чинит обнаруженную рассинхронизацию;
* trace теперь показывает `forced_intent_replan = 1` и немедленную смену `target_node_id` на том же шаге;
* на `LavaGap seed=0` длинные эпизоды `42/113/120` схлопнулись до `13/12/17`;
* в scoped benchmark по `LavaGap` state-conditioned v2 улучшил свои steps:
  * old state-conditioned v1: **9.82**
  * state-conditioned v2: **8.555**
* baseline всё ещё лучше по success:
  * baseline: success **0.240**
  * state-conditioned v2: success **0.220**

Практический вывод:

* synchronization bug fixed;
* remaining gap to baseline is now a policy problem, not a plumbing problem;
* forced replanning, tokenizer and compare pipeline больше не являются главным bottleneck.

## Что делать дальше

### Приоритет 1

Сделать **Sprint 2 v3: новый defensive intent для LavaGap**.

Минимальный смысл:

* не менять `AgentState` plumbing и forced replanning;
* заменить broad `REACH_SAFE_EXIT` на более узкий hazard-specific intent;
* тестировать сначала только на `MiniGrid-LavaGapS7-v0`.

### Приоритет 2

Только после локального `LavaGap` успеха можно снова прогонять более широкий compare.

### Не делать сейчас

* не переписывать tokenizer заново,
* не ломать milestone semantic chain,
* не смешивать adaptive benchmark с last-mile plumbing,
* не дожимать `safe_exit` blind tuning-ом без нового trace claim,
* не трогать больше forced replanning без нового trace claim,
* не запускать новый широкий benchmark до локального `LavaGap` v3 результата.

## Команды

### Block 1

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

### Block 2

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

## Финальная памятка агенту

Перед любым новым коммитом проверь:

1. Какую гипотезу я тестирую?
2. Какой минимальный слой я меняю?
3. Не ломаю ли я `milestone_semantic_handoff_v1`?
4. Нужен ли trace или benchmark?
5. Какой артефакт докажет, что патч реально сработал?

Если ответы неясны, патч делать нельзя.
