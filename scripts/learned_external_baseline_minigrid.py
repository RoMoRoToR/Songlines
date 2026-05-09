import argparse
import json
import os
import sys
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.songline_minigrid import (
    _oracle_controller_action,
    build_env,
    build_local_planner,
    ensure_dir,
    get_goal_position,
    get_rest_position,
    get_water_position,
    manhattan,
)


ACTIONS = (0, 1, 2)
LEARNED_METHOD_NAME = "external_learned_q_table_grid_obs"
LEARNED_DQN_METHOD_NAME = "external_learned_dqn_grid_obs"
LEARNED_BC_METHOD_NAME = "external_learned_bc_grid_obs"


def grid_cell_code(cell):
    if cell is None:
        return 0
    if cell.type == "wall":
        return 1
    if cell.type == "goal":
        return 2
    if cell.type == "lava":
        return 3
    if cell.type == "door":
        return 4
    if cell.type == "key":
        return 5
    if cell.type == "ball":
        return 6
    if cell.type == "box":
        return 7
    return 8


def build_state_key(env):
    grid = env.unwrapped.grid
    flat_codes = []
    for y in range(grid.height):
        for x in range(grid.width):
            flat_codes.append(int(grid_cell_code(grid.get(x, y))))
    agent_x, agent_y = env.unwrapped.agent_pos
    direction = getattr(env.unwrapped, "agent_dir", 0)
    return tuple(flat_codes + [int(agent_x), int(agent_y), int(direction)])


def resolve_task_target(env, args):
    task_mode = str(getattr(args, "task_mode", "default"))
    if task_mode == "water_search_v1":
        return get_water_position(env)
    if task_mode == "rest_search_v1":
        return get_rest_position(env)
    return get_goal_position(env)


def build_state_vector(env, args=None):
    grid = env.unwrapped.grid
    flat_codes = []
    for y in range(grid.height):
        for x in range(grid.width):
            flat_codes.append(float(grid_cell_code(grid.get(x, y))) / 8.0)
    agent_x, agent_y = env.unwrapped.agent_pos
    direction = getattr(env.unwrapped, "agent_dir", 0)
    flat_codes.extend(
        [
            float(agent_x) / max(1.0, float(grid.width - 1)),
            float(agent_y) / max(1.0, float(grid.height - 1)),
            float(direction) / 3.0,
        ]
    )
    if args is not None:
        target_xy = resolve_task_target(env, args)
        if target_xy is None:
            flat_codes.extend([0.0, 0.0])
        else:
            flat_codes.extend(
                [
                    float(target_xy[0]) / max(1.0, float(grid.width - 1)),
                    float(target_xy[1]) / max(1.0, float(grid.height - 1)),
                ]
            )
    return np.asarray(flat_codes, dtype=np.float32)


def task_success_from_transition(reward, info):
    if reward > 0.0:
        return 1
    if int(info.get("water_task_success", 0)) == 1:
        return 1
    if int(info.get("rest_task_success", 0)) == 1:
        return 1
    return 0


def default_learned_args(args):
    defaults = {
        "agent_mode": "learned_external",
        "songline_policy": "q_table_grid_obs",
        "learned_algo": "q_table",
        "commit_to_corridor": False,
        "learned_train_steps": 12000,
        "learned_alpha": 0.4,
        "learned_gamma": 0.99,
        "learned_epsilon_start": 1.0,
        "learned_epsilon_end": 0.05,
        "learned_epsilon_decay_fraction": 0.7,
        "learned_eval_epsilon": 0.0,
        "learned_batch_size": 64,
        "learned_target_update": 500,
        "learned_replay_size": 20000,
        "learned_hidden_dim": 128,
        "learned_lr": 1e-3,
        "learned_warmup_steps": 1000,
        "learned_bc_epochs": 6,
    }
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    return args


def epsilon_for_step(args, step_idx):
    decay_steps = max(1, int(float(args.learned_train_steps) * float(args.learned_epsilon_decay_fraction)))
    frac = min(1.0, float(step_idx) / float(decay_steps))
    return float(args.learned_epsilon_start) + frac * float(args.learned_epsilon_end - args.learned_epsilon_start)


def sample_training_seed(args, rng):
    span = max(8, int(args.episodes))
    return int(args.seed) + int(rng.randint(0, span))


def choose_action(q_table, state_key, epsilon, rng):
    if rng.rand() < float(epsilon) or state_key not in q_table:
        return int(rng.choice(ACTIONS))
    return int(np.argmax(q_table[state_key]))


class DQNPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(num_actions)),
        )

    def forward(self, x):
        return self.net(x)


def train_q_table(args):
    args = default_learned_args(args)
    env = build_env(args)
    rng = np.random.RandomState(int(args.seed))
    q_table = {}
    _, _ = env.reset(seed=sample_training_seed(args, rng))
    state_key = build_state_key(env)

    train_episode_returns = []
    train_episode_successes = []
    current_episode_return = 0.0
    current_episode_success = 0

    def ensure_state(key):
        if key not in q_table:
            q_table[key] = np.zeros(len(ACTIONS), dtype=np.float32)
        return q_table[key]

    for step_idx in range(int(args.learned_train_steps)):
        epsilon = epsilon_for_step(args, step_idx)
        q_values = ensure_state(state_key)
        action = choose_action(q_table, state_key, epsilon=epsilon, rng=rng)
        _, reward, terminated, truncated, info = env.step(int(action))
        next_key = build_state_key(env)
        next_values = ensure_state(next_key)
        done = bool(terminated or truncated)
        td_target = float(reward) + float(args.learned_gamma) * (0.0 if done else float(np.max(next_values)))
        q_values[action] = q_values[action] + float(args.learned_alpha) * (td_target - q_values[action])

        current_episode_return += float(reward)
        current_episode_success = max(int(current_episode_success), int(task_success_from_transition(float(reward), dict(info or {}))))
        state_key = next_key

        if done:
            train_episode_returns.append(float(current_episode_return))
            train_episode_successes.append(int(current_episode_success))
            current_episode_return = 0.0
            current_episode_success = 0
            _, _ = env.reset(seed=sample_training_seed(args, rng))
            state_key = build_state_key(env)

    env.close()
    train_stats = {
        "learned_algo": "q_table",
        "train_steps": int(args.learned_train_steps),
        "train_episodes": int(len(train_episode_returns)),
        "train_success_rate": 0.0 if not train_episode_successes else float(np.mean(train_episode_successes)),
        "train_avg_return": 0.0 if not train_episode_returns else float(np.mean(train_episode_returns)),
        "num_states": int(len(q_table)),
        "num_actions": int(len(ACTIONS)),
    }
    return q_table, train_stats


def choose_dqn_action(model, state_vec, epsilon, rng):
    if rng.rand() < float(epsilon):
        return int(rng.choice(ACTIONS))
    with torch.no_grad():
        state_tensor = torch.as_tensor(state_vec, dtype=torch.float32).unsqueeze(0)
        q_values = model(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())


def train_dqn(args):
    args = default_learned_args(args)
    env = build_env(args)
    rng = np.random.RandomState(int(args.seed))
    torch.manual_seed(int(args.seed))

    _, _ = env.reset(seed=sample_training_seed(args, rng))
    initial_state = build_state_vector(env, args=args)
    model = DQNPolicy(len(initial_state), int(args.learned_hidden_dim), len(ACTIONS))
    target_model = DQNPolicy(len(initial_state), int(args.learned_hidden_dim), len(ACTIONS))
    target_model.load_state_dict(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.learned_lr))
    replay = deque(maxlen=int(args.learned_replay_size))

    state_vec = initial_state
    target_xy = resolve_task_target(env, args)
    prev_distance = None if target_xy is None else float(manhattan(np.asarray(env.unwrapped.agent_pos, dtype=np.int32), target_xy))
    train_episode_returns = []
    train_episode_successes = []
    current_episode_return = 0.0
    current_episode_success = 0

    for step_idx in range(int(args.learned_train_steps)):
        epsilon = epsilon_for_step(args, step_idx)
        action = choose_dqn_action(model, state_vec, epsilon=epsilon, rng=rng)
        _, reward, terminated, truncated, info = env.step(int(action))
        next_state_vec = build_state_vector(env, args=args)
        done = bool(terminated or truncated)
        shaped_reward = float(reward)
        if target_xy is not None:
            next_distance = float(manhattan(np.asarray(env.unwrapped.agent_pos, dtype=np.int32), target_xy))
            if prev_distance is not None:
                shaped_reward += 0.05 * (prev_distance - next_distance)
            shaped_reward -= 0.0025
            prev_distance = next_distance
        replay.append(
            (
                np.asarray(state_vec, dtype=np.float32),
                int(action),
                float(shaped_reward),
                np.asarray(next_state_vec, dtype=np.float32),
                float(done),
            )
        )

        if len(replay) >= int(args.learned_batch_size) and step_idx >= int(args.learned_warmup_steps):
            idxs = rng.choice(len(replay), size=int(args.learned_batch_size), replace=False)
            batch = [replay[int(i)] for i in idxs]
            states = torch.as_tensor(np.stack([item[0] for item in batch]), dtype=torch.float32)
            actions = torch.as_tensor([item[1] for item in batch], dtype=torch.int64)
            rewards = torch.as_tensor([item[2] for item in batch], dtype=torch.float32)
            next_states = torch.as_tensor(np.stack([item[3] for item in batch]), dtype=torch.float32)
            dones = torch.as_tensor([item[4] for item in batch], dtype=torch.float32)

            q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_q = torch.max(target_model(next_states), dim=1).values
                targets = rewards + float(args.learned_gamma) * (1.0 - dones) * next_q
            loss = F.smooth_l1_loss(q_values, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (step_idx + 1) % int(args.learned_target_update) == 0:
            target_model.load_state_dict(model.state_dict())

        current_episode_return += float(reward)
        current_episode_success = max(int(current_episode_success), int(task_success_from_transition(float(reward), dict(info or {}))))
        state_vec = next_state_vec

        if done:
            train_episode_returns.append(float(current_episode_return))
            train_episode_successes.append(int(current_episode_success))
            current_episode_return = 0.0
            current_episode_success = 0
            _, _ = env.reset(seed=sample_training_seed(args, rng))
            state_vec = build_state_vector(env, args=args)
            target_xy = resolve_task_target(env, args)
            prev_distance = None if target_xy is None else float(manhattan(np.asarray(env.unwrapped.agent_pos, dtype=np.int32), target_xy))

    env.close()
    target_model.load_state_dict(model.state_dict())
    train_stats = {
        "learned_algo": "dqn_mlp",
        "train_steps": int(args.learned_train_steps),
        "train_episodes": int(len(train_episode_returns)),
        "train_success_rate": 0.0 if not train_episode_successes else float(np.mean(train_episode_successes)),
        "train_avg_return": 0.0 if not train_episode_returns else float(np.mean(train_episode_returns)),
        "num_actions": int(len(ACTIONS)),
        "replay_final_size": int(len(replay)),
        "hidden_dim": int(args.learned_hidden_dim),
    }
    return model, train_stats


def train_behavior_cloning(args):
    args = default_learned_args(args)
    env = build_env(args)
    _, trajectory_planner = build_local_planner(args)
    rng = np.random.RandomState(int(args.seed))
    torch.manual_seed(int(args.seed))

    dataset_states = []
    dataset_actions = []
    samples_target = max(2000, int(args.learned_train_steps))
    collected = 0
    rollout_idx = 0
    while collected < samples_target:
        _, _ = env.reset(seed=sample_training_seed(args, rng) + rollout_idx)
        target_xy = resolve_task_target(env, args)
        if target_xy is None:
            rollout_idx += 1
            continue
        for _ in range(int(args.max_steps)):
            state_vec = build_state_vector(env, args=args)
            action = _oracle_controller_action(env, trajectory_planner, target_xy)
            dataset_states.append(state_vec)
            dataset_actions.append(int(action))
            collected += 1
            _, reward, terminated, truncated, info = env.step(int(action))
            if terminated or truncated or task_success_from_transition(float(reward), dict(info or {})):
                break
            if collected >= samples_target:
                break
        rollout_idx += 1

    states = torch.as_tensor(np.stack(dataset_states), dtype=torch.float32)
    actions = torch.as_tensor(dataset_actions, dtype=torch.int64)
    model = DQNPolicy(states.shape[1], int(args.learned_hidden_dim), len(ACTIONS))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.learned_lr))
    batch_size = int(min(len(dataset_states), max(32, int(args.learned_batch_size))))

    for _ in range(int(args.learned_bc_epochs)):
        perm = torch.randperm(states.shape[0])
        for start in range(0, states.shape[0], batch_size):
            idx = perm[start : start + batch_size]
            logits = model(states[idx])
            loss = F.cross_entropy(logits, actions[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    env.close()
    train_stats = {
        "learned_algo": "bc_oracle",
        "train_steps": int(args.learned_train_steps),
        "train_samples": int(len(dataset_states)),
        "num_actions": int(len(ACTIONS)),
        "hidden_dim": int(args.learned_hidden_dim),
        "bc_epochs": int(args.learned_bc_epochs),
    }
    return model, train_stats


def build_episode_metrics(args, ep_idx, env_id, task_mode, step_count, episode_return, success, info):
    info = dict(info or {})
    failure_taxonomy = "success_without_retrieval" if int(success) == 1 else "no_retrieval_attempt"
    algo = str(getattr(args, "learned_algo", "q_table"))
    if algo == "dqn_mlp":
        method_name = LEARNED_DQN_METHOD_NAME
        policy_name = "dqn_grid_obs"
    elif algo == "bc_oracle":
        method_name = LEARNED_BC_METHOD_NAME
        policy_name = "bc_grid_obs"
    else:
        method_name = LEARNED_METHOD_NAME
        policy_name = "q_table_grid_obs"
    return {
        "episode": int(ep_idx + 1),
        "env_id": str(env_id),
        "task_mode": str(task_mode),
        "agent_mode": "learned_external",
        "songline_policy": policy_name,
        "method": method_name,
        "seed": int(args.seed),
        "return": float(episode_return),
        "steps_to_goal": int(step_count),
        "steps": int(step_count),
        "success": int(success),
        "water_task_success": int(info.get("water_task_success", 0)),
        "rest_task_success": int(info.get("rest_task_success", 0)),
        "query_attempt_count": 0,
        "query_nonempty_rate": 0.0,
        "retrieval_precision_at_k": 0.0,
        "query_satisfaction_rate": 0.0,
        "semantic_target_materialization_rate": 0.0,
        "semantic_target_materialized_any": 0,
        "post_retrieval_completion": 0,
        "retrieval_failure_empty": 0,
        "retrieval_failure_unsatisfied": 0,
        "semantic_materialization_failure": 0,
        "control_failure_after_retrieval": 0,
        "failure_taxonomy": failure_taxonomy,
        "graph_nodes": 0,
        "graph_edges": 0,
        "intervention_rate": 0.0,
        "plan_hit_rate": 0.0,
        "phrase_length_mean": 0.0,
        "subgoal_reach_rate": 0.0,
        "goal_distance_delta_per_intervention": 0.0,
        "node_reuse_rate": 0.0,
        "new_nodes_per_episode": 0,
        "graph_path_length": 0.0,
    }


def summarise_eval(args, episode_metrics):
    successes = [int(item["success"]) for item in episode_metrics]
    episode_returns = [float(item["return"]) for item in episode_metrics]
    episode_lengths = [int(item["steps_to_goal"]) for item in episode_metrics]
    success_rate = 0.0 if not successes else float(np.mean(successes))
    avg_return = 0.0 if not episode_returns else float(np.mean(episode_returns))
    avg_steps = 0.0 if not episode_lengths else float(np.mean(episode_lengths))
    algo = str(getattr(args, "learned_algo", "q_table"))
    if algo == "dqn_mlp":
        method_name = LEARNED_DQN_METHOD_NAME
        policy_name = "dqn_grid_obs"
    elif algo == "bc_oracle":
        method_name = LEARNED_BC_METHOD_NAME
        policy_name = "bc_grid_obs"
    else:
        method_name = LEARNED_METHOD_NAME
        policy_name = "q_table_grid_obs"
    return {
        "env_id": str(args.env_id),
        "task_mode": str(getattr(args, "task_mode", "default")),
        "agent_mode": "learned_external",
        "songline_policy": policy_name,
        "method": method_name,
        "episodes": int(args.episodes),
        "max_steps": int(args.max_steps),
        "seed": int(args.seed),
        "success_rate": float(success_rate),
        "success_rate_pre_change": float(success_rate),
        "success_rate_post_change": 0.0,
        "success_rate_change_delta": 0.0,
        "avg_return": float(avg_return),
        "avg_steps_to_goal": float(avg_steps),
        "avg_steps": float(avg_steps),
        "intervention_rate": 0.0,
        "plan_hit_rate": 0.0,
        "graph_nodes": 0.0,
        "graph_edges": 0.0,
        "phrase_length_mean": 0.0,
        "subgoal_reach_rate": 0.0,
        "goal_distance_delta_per_intervention": 0.0,
        "node_reuse_rate": 0.0,
        "new_nodes_per_episode": 0.0,
        "graph_path_length": 0.0,
        "fraction_gap_aligned": 0.0,
        "fraction_safe_crossing": 0.0,
        "fraction_post_hazard": 0.0,
        "fraction_final_exit_maneuver": 0.0,
        "fraction_resume_to_goal": 0.0,
        "fraction_post_hazard_progress": 0.0,
        "fraction_resume_to_goal_progress": 0.0,
        "fraction_post_hazard_to_success": 0.0,
        "fraction_resume_to_goal_to_success": 0.0,
        "conditional_post_hazard_success": 0.0,
        "conditional_resume_to_goal_success": 0.0,
        "mean_max_phase_depth": 0.0,
        "query_attempt_count": 0.0,
        "query_nonempty_rate": 0.0,
        "retrieval_precision_at_k": 0.0,
        "query_satisfaction_rate": 0.0,
        "semantic_target_materialization_rate": 0.0,
        "post_retrieval_completion_rate": 0.0,
        "completion_given_materialized": 0.0,
        "local_resource_guidance_usage_rate": 0.0,
        "goal_rejoin_fallback_assist_usage_rate": 0.0,
        "retrieval_failure_empty_rate": 0.0,
        "retrieval_failure_unsatisfied_rate": 0.0,
        "semantic_materialization_failure_rate": 0.0,
        "control_failure_after_retrieval_rate": 0.0,
        "local_resource_guidance_enabled": int(not bool(getattr(args, "disable_local_resource_guidance", False))),
        "goal_rejoin_fallback_assists_enabled": int(not bool(getattr(args, "disable_goal_rejoin_fallback_assists", False))),
    }


def run_learned_baseline_experiment(args, export_outputs=True, verbose=True):
    args = default_learned_args(args)
    ensure_dir(args.out_dir)
    learned_algo = str(getattr(args, "learned_algo", "q_table"))
    if learned_algo == "dqn_mlp":
        method_name = LEARNED_DQN_METHOD_NAME
    elif learned_algo == "bc_oracle":
        method_name = LEARNED_BC_METHOD_NAME
    else:
        method_name = LEARNED_METHOD_NAME
    if learned_algo == "dqn_mlp":
        policy_model, train_stats = train_dqn(args)
    elif learned_algo == "bc_oracle":
        policy_model, train_stats = train_behavior_cloning(args)
    else:
        policy_model, train_stats = train_q_table(args)
    env = build_env(args)
    episode_metrics = []
    episode_returns = []
    episode_lengths = []
    success_flags = []

    for ep_idx in range(int(args.episodes)):
        _, info = env.reset(seed=int(args.seed) + ep_idx)
        state_key = build_state_key(env)
        episode_return = 0.0
        success = 0
        final_info = dict(info or {})
        step_count = 0
        for step_idx in range(int(args.max_steps)):
            step_count = step_idx + 1
            eval_rng = np.random.RandomState(int(args.seed) + ep_idx * 97 + step_idx)
            if learned_algo in {"dqn_mlp", "bc_oracle"}:
                state_vec = build_state_vector(env, args=args)
                action = choose_dqn_action(policy_model, state_vec, epsilon=float(args.learned_eval_epsilon), rng=eval_rng)
            else:
                action = choose_action(policy_model, state_key, epsilon=float(args.learned_eval_epsilon), rng=eval_rng)
            _, reward, terminated, truncated, info = env.step(int(action))
            state_key = build_state_key(env)
            episode_return += float(reward)
            final_info = dict(info or {})
            success = max(int(success), int(task_success_from_transition(float(reward), final_info)))
            if terminated or truncated:
                break

        item = build_episode_metrics(
            args=args,
            ep_idx=ep_idx,
            env_id=args.env_id,
            task_mode=getattr(args, "task_mode", "default"),
            step_count=step_count,
            episode_return=episode_return,
            success=success,
            info=final_info,
        )
        episode_metrics.append(item)
        episode_returns.append(float(episode_return))
        episode_lengths.append(int(step_count))
        success_flags.append(int(success))
        if verbose:
            print(
                f"Episode {ep_idx + 1:03d} | method={method_name} | "
                f"return={episode_return:.3f} | steps={step_count:03d} | success={success}"
            )

    env.close()
    summary = summarise_eval(args, episode_metrics)
    run_summary = {
        "env_id": str(args.env_id),
        "task_mode": str(getattr(args, "task_mode", "default")),
        "agent_mode": "learned_external",
        "songline_policy": (
            "dqn_grid_obs"
            if learned_algo == "dqn_mlp"
            else ("bc_grid_obs" if learned_algo == "bc_oracle" else "q_table_grid_obs")
        ),
        "method": method_name,
        "episodes": int(args.episodes),
        "max_steps": int(args.max_steps),
        "seed": int(args.seed),
        "episode_returns": episode_returns,
        "episode_lengths": episode_lengths,
        "successes": success_flags,
        "successes_pre_change": list(success_flags),
        "successes_post_change": [],
        "graph_nodes": [0 for _ in episode_metrics],
        "graph_edges": [0 for _ in episode_metrics],
        "intervention_rate": [0.0 for _ in episode_metrics],
        "plan_hit_rate": [0.0 for _ in episode_metrics],
        "phrase_length_mean": [0.0 for _ in episode_metrics],
        "subgoal_reached": [0.0 for _ in episode_metrics],
        "goal_distance_delta_per_intervention": [0.0 for _ in episode_metrics],
        "node_reuse_rate": [0.0 for _ in episode_metrics],
        "new_nodes_per_episode": [0 for _ in episode_metrics],
        "graph_path_length": [0.0 for _ in episode_metrics],
        "has_gap_aligned": [0 for _ in episode_metrics],
        "has_safe_crossing": [0 for _ in episode_metrics],
        "has_post_hazard": [0 for _ in episode_metrics],
        "has_final_exit_maneuver": [0 for _ in episode_metrics],
        "has_resume_to_goal": [0 for _ in episode_metrics],
        "has_post_hazard_progress": [0 for _ in episode_metrics],
        "has_resume_to_goal_progress": [0 for _ in episode_metrics],
        "post_hazard_to_success": [0 for _ in episode_metrics],
        "resume_to_goal_to_success": [0 for _ in episode_metrics],
        "max_phase_depth": [0 for _ in episode_metrics],
        "query_attempt_count": [0 for _ in episode_metrics],
        "query_nonempty_rate": [0.0 for _ in episode_metrics],
        "retrieval_precision_at_k": [0.0 for _ in episode_metrics],
        "query_satisfaction_rate": [0.0 for _ in episode_metrics],
        "semantic_target_materialization_rate": [0.0 for _ in episode_metrics],
        "semantic_target_materialized_any": [0 for _ in episode_metrics],
        "post_retrieval_completion": [0 for _ in episode_metrics],
        "local_resource_guidance_used": [0 for _ in episode_metrics],
        "goal_rejoin_fallback_assist_used": [0 for _ in episode_metrics],
        "retrieval_failure_empty": [0 for _ in episode_metrics],
        "retrieval_failure_unsatisfied": [0 for _ in episode_metrics],
        "semantic_materialization_failure": [0 for _ in episode_metrics],
        "control_failure_after_retrieval": [0 for _ in episode_metrics],
        "episode_metrics": episode_metrics,
        "training_stats": train_stats,
    }

    if export_outputs:
        with open(os.path.join(args.out_dir, "episodes.json"), "w") as file_obj:
            json.dump(episode_metrics, file_obj, indent=2)
        with open(os.path.join(args.out_dir, "summary.json"), "w") as file_obj:
            payload = dict(summary)
            payload["training_stats"] = train_stats
            json.dump(payload, file_obj, indent=2)
        with open(os.path.join(args.out_dir, "run_summary.json"), "w") as file_obj:
            json.dump(run_summary, file_obj, indent=2)

    return run_summary, summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="MiniGrid-Empty-Random-6x6-v0")
    parser.add_argument("--task_mode", type=str, default="default", choices=["default", "water_search_v1", "rest_search_v1"])
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--water_success_radius", type=int, default=1)
    parser.add_argument("--rest_success_radius", type=int, default=1)
    parser.add_argument("--disable_local_resource_guidance", action="store_true")
    parser.add_argument("--disable_goal_rejoin_fallback_assists", action="store_true")
    parser.add_argument("--learned_algo", type=str, default="q_table", choices=["q_table", "dqn_mlp", "bc_oracle"])
    parser.add_argument("--learned_train_steps", type=int, default=12000)
    parser.add_argument("--learned_alpha", type=float, default=0.4)
    parser.add_argument("--learned_gamma", type=float, default=0.99)
    parser.add_argument("--learned_epsilon_start", type=float, default=1.0)
    parser.add_argument("--learned_epsilon_end", type=float, default=0.05)
    parser.add_argument("--learned_epsilon_decay_fraction", type=float, default=0.7)
    parser.add_argument("--learned_eval_epsilon", type=float, default=0.0)
    parser.add_argument("--learned_batch_size", type=int, default=64)
    parser.add_argument("--learned_target_update", type=int, default=500)
    parser.add_argument("--learned_replay_size", type=int, default=20000)
    parser.add_argument("--learned_hidden_dim", type=int, default=128)
    parser.add_argument("--learned_lr", type=float, default=1e-3)
    parser.add_argument("--learned_warmup_steps", type=int, default=1000)
    parser.add_argument("--learned_bc_epochs", type=int, default=6)
    parser.add_argument("--out_dir", type=str, default="tmp/learned_external_baseline_minigrid")
    return parser.parse_args()


def main():
    args = parse_args()
    run_learned_baseline_experiment(args, export_outputs=True, verbose=True)


if __name__ == "__main__":
    main()
