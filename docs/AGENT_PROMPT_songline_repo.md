
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
* Sprint 2 v2 уже устранил timing mismatch между `AgentState` switch и planner refresh;
* Sprint 2 v3 (`hazard_recovery_exit`) оказался лучше broad `safe_exit` на `LavaGap`;
* Sprint 2 v4a показал, что explicit post-recovery handoff обратно в `find_goal_region` действительно нужен;
* Sprint 2 v4b показал отрицательный, но полезный результат: debounce repeated handoff не является главным bottleneck;
* Sprint 2 v5 audit локализовал dominant failure mode как `handoff_without_graph_target`;
* Sprint 2 v6 подтвердил, что explicit target materialization нужно, но прямой `goal_xy_fallback` слишком агрессивен;
* Sprint 2 v7 (`stable_rejoin_waypoint`) сейчас является лучшим узким rejoin fallback на `LavaGap`.
* Sprint 2 v8 (`source_select_v1`) улучшил rejoin efficiency без потери success;
* Sprint 2 v9 (`source_select_v2`) локально вытесняет `hazard_adjacent` stable targets, но не даёт benchmark-level gain.

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
* Sprint 2 v2-v7 уже закрыли debugging- и recovery-layer на `LavaGap`;
* следующий фокус — не новый broad intent, а качество post-materialization rejoin source.

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

## Что уже известно про Sprint 2 v3-v7 на LavaGap

### `milestone_state_conditioned_hazard_recovery_v1`

Проверенный результат:

* `hazard_recovery_exit` лучше, чем оба варианта на broad `safe_exit`;
* это полезный recovery-specific intent;
* но он ещё не восстанавливает completion до baseline.

Aggregate:

* baseline: success **0.240**, steps **8.8825**, return **0.2250**
* `hazard_recovery v1`: success **0.225**, steps **8.11**, return **0.2123**

### `milestone_state_conditioned_hazard_recovery_v2`

Проверенный результат:

* explicit post-recovery handoff `hazard_recovery_exit -> find_goal_region` нужен и полезен;
* trace показывает:
  * `exited_hazard_recovery_intent = 1`
  * `forced_goal_rejoin_replan = 1`
  * `planner_query_intent = find_goal_region`
* этот слой почти восстанавливает gap к baseline по success, сохраняя часть выигрыша по steps.

Aggregate:

* baseline: success **0.240**, steps **8.8825**, return **0.2250**
* `hazard_recovery v2`: success **0.2350**, steps **8.4925**, return **0.2201**

### `milestone_state_conditioned_hazard_recovery_v3`

Проверенный negative result:

* debounce repeated goal-rejoin handoff не меняет aggregate-метрики;
* `goal_rejoin_handoff_suppressed` не появляется как доминирующий trace case;
* значит repeated handoff внутри одного окна не является главным bottleneck.

Практический вывод:

* не возвращайся к handoff-debounce как к основной гипотезе без нового trace claim.

### Sprint 2 v5 audit

Проверенный failure audit:

* найдено **68** repeated-hazard cycles;
* из них **56** — `handoff_without_graph_target`;
* остальные случаи значительно реже:
  * `controller_slip_on_hazard_adjacent_rejoin`
  * `hazard_adjacent_rejoin_node`
  * `rejoin_target_missing_goal_region`
  * `local_reentry_after_nominal_goal_rejoin`

Главный вывод:

* dominant failure mode после `v2` уже не node quality;
* основной gap был в том, что handoff часто не materialize-ил target.

### `milestone_state_conditioned_hazard_recovery_v4`

Проверенный результат:

* explicit target materialization действительно нужен;
* прямой `goal_xy_fallback` закрывает `handoff_without_graph_target`;
* но делает rejoin trajectory слишком агрессивной.

Aggregate:

* baseline: success **0.240**, steps **8.8825**, return **0.225**
* `hazard_recovery v4`: success **0.230**, steps **8.067**, return **0.218**

Практический вывод:

* materialization gap closed;
* remaining bottleneck moved from “no target” to “quality of fallback rejoin trajectory”.

### `milestone_state_conditioned_hazard_recovery_v5`

Проверенный текущий best narrow result for `LavaGap`:

* `stable_rejoin_waypoint` лучше прямого `goal_xy_fallback`;
* он сохраняет success на уровне `v2`;
* и одновременно улучшает steps и немного return.

Aggregate:

* baseline: success **0.240**, steps **8.8825**, return **0.225**
* `hazard_recovery v2`: success **0.235**, steps **8.4925**, return **0.220**
* `hazard_recovery v5`: success **0.235**, steps **8.310**, return **0.221**

Практический вывод:

* current best narrow `LavaGap` variant is `v5`;
* следующий слой должен сравнивать источники rejoin target, а не снова менять intent.

### `milestone_state_conditioned_hazard_recovery_v6`

Проверенный результат:

* source-conditioned rejoin selection полезен;
* `graph_node` берётся только если он non-`hazard_adjacent` и имеет сильный `goal_region`;
* иначе используется `stable_rejoin_waypoint`.

Aggregate:

* `hazard_recovery v5`: success **0.235**, steps **8.310**, return **0.2210**
* `hazard_recovery v6`: success **0.235**, steps **8.2175**, return **0.2213**

Практический вывод:

* source selection улучшает эффективность;
* но оставшийся gap до baseline не закрывается.

### `milestone_state_conditioned_hazard_recovery_v7`

Проверенный negative result:

* stricter stable selector действительно вытесняет `hazard_adjacent` stable targets локально;
* на `seed=2`:
  * `stable_adj1`: **10 -> 3**
  * `stable_adj0`: **9 -> 16**
* но aggregate не меняется:
  * `hazard_recovery v6`: success **0.235**, steps **8.2175**, return **0.2213**
  * `hazard_recovery v7`: success **0.235**, steps **8.2175**, return **0.2213**

Практический вывод:

* adjacency-фильтрация сама по себе уже не является главным bottleneck;
* remaining gap относится к качеству самой rejoin trajectory.

## Что делать дальше

### Приоритет 1

Сделать **Sprint 2 v9.1 audit: conditional outcomes for `stable_rejoin_waypoint` by `hazard_adjacent` and `goal_alignment`**.

Минимальный смысл:

* не менять `IntentPolicy`, `tokenizer`, `semantic tags`, `graph_rollout`, forced replanning и compare plumbing;
* audit-ить только forced rejoin events для `v7`;
* сравнить `stable_rejoin_waypoint` при:
  * `hazard_adjacent = 0`
  * `hazard_adjacent = 1`
  * разных `goal_alignment` bins
* проверить:
  * `success_of_episode`
  * `reentered_hazard_after_rejoin`
  * `next_1_step_delta`
  * `next_3_step_delta`
  * длину эпизода после rejoin
  * сколько раз после rejoin снова активируется recovery intent
* тестировать сначала только на `MiniGrid-LavaGapS7-v0`.

### Приоритет 2

Только после локального `LavaGap` результата можно снова прогонять более широкий compare.

### Не делать сейчас

* не переписывать tokenizer заново,
* не ломать milestone semantic chain,
* не смешивать adaptive benchmark с last-mile plumbing,
* не дожимать `safe_exit` blind tuning-ом без нового trace claim,
* не трогать больше forced replanning без нового trace claim,
* не возвращаться к debounce repeated handoff как к главной гипотезе,
* не возвращаться к raw materialization-vs-no-materialization как к главной гипотезе,
* не возвращаться к adjacency-only filtering как к главной гипотезе,
* не запускать новый широкий benchmark до локального `LavaGap` `v9.1` audit результата.

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

## Финальная памятка агенту

Перед любым новым коммитом проверь:

1. Какую гипотезу я тестирую?
2. Какой минимальный слой я меняю?
3. Не ломаю ли я `milestone_semantic_handoff_v1`?
4. Нужен ли trace или benchmark?
5. Какой артефакт докажет, что патч реально сработал?

Если ответы неясны, патч делать нельзя.
