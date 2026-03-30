
# AGENT_PROMPT_songline_repo.md

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
* следующий сильный шаг — controlled non-stationarity benchmark.

## Что делать дальше

### Приоритет 1

Сделать **controlled non-stationarity benchmark** для сравнения:

* `milestone_semantic_handoff_v1`
* `milestone_semantic_handoff_v1_adaptive_graph`

### Приоритет 2

Только если есть очень конкретный trace-based провал, можно делать узкий local patch.

### Не делать сейчас

* не переписывать tokenizer заново,
* не ломать milestone semantic chain,
* не смешивать adaptive benchmark с last-mile plumbing.

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

## Финальная памятка агенту

Перед любым новым коммитом проверь:

1. Какую гипотезу я тестирую?
2. Какой минимальный слой я меняю?
3. Не ломаю ли я `milestone_semantic_handoff_v1`?
4. Нужен ли trace или benchmark?
5. Какой артефакт докажет, что патч реально сработал?

Если ответы неясны, патч делать нельзя.
