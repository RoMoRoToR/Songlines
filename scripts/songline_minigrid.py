import os
import json
import math
import argparse
from collections import defaultdict

import numpy as np
import gymnasium as gym
import minigrid
import matplotlib.pyplot as plt

from utils.lz_memory import LZMapMemory, SymbolicTokenizer


DIR_TO_VEC = {
    0: np.array((1, 0), dtype=np.int32),   # right
    1: np.array((0, 1), dtype=np.int32),   # down
    2: np.array((-1, 0), dtype=np.int32),  # left
    3: np.array((0, -1), dtype=np.int32),  # up
}


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def local_symbolic_observation(env, radius=1):
    """
    Строит компактное символическое наблюдение:
    - направление агента
    - локальное окно (2r+1)x(2r+1) вокруг агента
    - тип клетки кодируется коротким числом
    """
    unwrapped = env.unwrapped
    ax, ay = unwrapped.agent_pos
    direction = unwrapped.agent_dir
    grid = unwrapped.grid

    patch = []
    for dy in range(-radius, radius + 1):
        row = []
        for dx in range(-radius, radius + 1):
            x = ax + dx
            y = ay + dy
            if x < 0 or y < 0 or x >= grid.width or y >= grid.height:
                row.append(99)
                continue
            cell = grid.get(x, y)
            if cell is None:
                row.append(0)
            else:
                cell_name = cell.type
                if cell_name == "wall":
                    row.append(1)
                elif cell_name == "goal":
                    row.append(2)
                elif cell_name == "lava":
                    row.append(3)
                elif cell_name == "door":
                    row.append(4)
                elif cell_name == "key":
                    row.append(5)
                elif cell_name == "ball":
                    row.append(6)
                elif cell_name == "box":
                    row.append(7)
                else:
                    row.append(8)
        patch.append(row)

    patch = np.array(patch, dtype=np.int32).reshape(-1)
    obs_vec = np.concatenate([np.array([direction], dtype=np.int32), patch])
    return obs_vec.astype(np.float32)


def manhattan(a, b):
    return int(abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1])))


def choose_action_toward(env, target_xy):
    """
    Простой локальный контроллер:
    поворачивает агента к target_xy, затем идёт вперёд.
    """
    unwrapped = env.unwrapped
    ax, ay = unwrapped.agent_pos
    agent_dir = unwrapped.agent_dir

    tx, ty = int(target_xy[0]), int(target_xy[1])
    dx = tx - ax
    dy = ty - ay

    if dx == 0 and dy == 0:
        return 2  # forward

    if abs(dx) >= abs(dy):
        desired_dir = 0 if dx > 0 else 2
    else:
        desired_dir = 1 if dy > 0 else 3

    if desired_dir == agent_dir:
        return 2  # forward

    right_turns = (desired_dir - agent_dir) % 4
    left_turns = (agent_dir - desired_dir) % 4

    if left_turns <= right_turns:
        return 0  # left
    return 1  # right


def random_safe_action(rng):
    return int(rng.choice([0, 1, 2]))


def get_goal_position(env):
    grid = env.unwrapped.grid
    for y in range(grid.height):
        for x in range(grid.width):
            cell = grid.get(x, y)
            if cell is not None and cell.type == "goal":
                return np.array([x, y], dtype=np.int32)
    return None


def export_run_summary(out_dir, run_summary):
    with open(os.path.join(out_dir, "run_summary.json"), "w") as f:
        json.dump(run_summary, f, indent=2)

    episodes = np.arange(1, len(run_summary["episode_returns"]) + 1)

    plt.figure(figsize=(6, 4))
    plt.plot(episodes, run_summary["episode_returns"])
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Episode Returns")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "episode_returns.png"))
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(episodes, run_summary["episode_lengths"])
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.title("Episode Lengths")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "episode_lengths.png"))
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(episodes, run_summary["graph_nodes"])
    plt.plot(episodes, run_summary["graph_edges"])
    plt.xlabel("Episode")
    plt.ylabel("Count")
    plt.title("Graph Growth by Episode")
    plt.legend(["nodes", "edges"])
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "graph_growth_by_episode.png"))
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="MiniGrid-Empty-Random-6x6-v0")
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--max_steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epsilon", type=float, default=0.15)
    parser.add_argument("--suggest_every", type=int, default=8)
    parser.add_argument("--min_goal_visits", type=int, default=2)
    parser.add_argument("--top_k_goals", type=int, default=5)
    parser.add_argument("--tokenizer_mode", type=str, default="hash_sign", choices=["argmax", "hash_sign"])
    parser.add_argument("--tokenizer_proj_dim", type=int, default=16)
    parser.add_argument("--out_dir", type=str, default="tmp/songline_minigrid")
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    env = gym.make(args.env_id, render_mode="rgb_array")
    rng = np.random.RandomState(args.seed)

    tokenizer = SymbolicTokenizer(
        mode=args.tokenizer_mode,
        proj_dim=args.tokenizer_proj_dim,
        seed=args.seed,
    )
    memory = LZMapMemory(min_goal_visits=args.min_goal_visits)

    run_summary = {
        "env_id": args.env_id,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "episode_returns": [],
        "episode_lengths": [],
        "successes": [],
        "graph_nodes": [],
        "graph_edges": [],
        "intervention_rate": [],
        "plan_hit_rate": [],
    }

    total_step_idx = 0
    subgoal_xy = None
    position_visit_counter = defaultdict(int)

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        goal_xy = get_goal_position(env)
        episode_return = 0.0
        success = 0
        previous_goal_distance = None

        for step in range(args.max_steps):
            unwrapped = env.unwrapped
            agent_xy = np.array(unwrapped.agent_pos, dtype=np.int32)
            position_visit_counter[(int(agent_xy[0]), int(agent_xy[1]))] += 1

            obs_vec = local_symbolic_observation(env, radius=1)
            token = tokenizer.encode(obs_vec)
            memory.update_token(token, total_step_idx)
            memory.observe(
                total_step_idx,
                pose_xy=agent_xy,
                goal_xy=goal_xy,
                reward=0.0,
            )

            if step % args.suggest_every == 0:
                proposal = memory.suggest_subgoal(top_k=args.top_k_goals)
                if proposal is not None:
                    gx = int(round(proposal["goal_xy"][0]))
                    gy = int(round(proposal["goal_xy"][1]))
                    gx = max(0, min(gx, env.unwrapped.grid.width - 1))
                    gy = max(0, min(gy, env.unwrapped.grid.height - 1))
                    subgoal_xy = np.array([gx, gy], dtype=np.int32)

            if subgoal_xy is not None and rng.rand() > args.epsilon:
                action = choose_action_toward(env, subgoal_xy)
            else:
                if goal_xy is not None and rng.rand() > 0.5:
                    action = choose_action_toward(env, goal_xy)
                else:
                    action = random_safe_action(rng)

            obs, reward, terminated, truncated, info = env.step(action)
            total_step_idx += 1
            episode_return += float(reward)

            agent_xy_new = np.array(env.unwrapped.agent_pos, dtype=np.int32)

            current_goal_distance = None
            if goal_xy is not None:
                current_goal_distance = manhattan(agent_xy_new, goal_xy)

            if previous_goal_distance is not None and current_goal_distance is not None:
                improved = current_goal_distance < previous_goal_distance
                memory.record_plan_outcome(improved)

            if current_goal_distance is not None:
                previous_goal_distance = current_goal_distance

            memory.observe(
                total_step_idx,
                pose_xy=agent_xy_new,
                goal_xy=goal_xy,
                reward=float(reward),
            )

            if reward > 0:
                success = 1
                subgoal_xy = goal_xy.copy()

            if terminated or truncated:
                break

        edge_count = sum(len(v) for v in memory.edges.values())

        run_summary["episode_returns"].append(float(episode_return))
        run_summary["episode_lengths"].append(step + 1)
        run_summary["successes"].append(int(success))
        run_summary["graph_nodes"].append(int(len(memory.nodes)))
        run_summary["graph_edges"].append(int(edge_count))
        run_summary["intervention_rate"].append(
            0.0 if memory.intervention_attempts == 0 else memory.interventions / memory.intervention_attempts
        )
        run_summary["plan_hit_rate"].append(
            0.0 if memory.plan_total == 0 else memory.plan_hits / memory.plan_total
        )

        print(
            f"Episode {ep + 1:03d} | "
            f"return={episode_return:.3f} | "
            f"steps={step + 1:03d} | "
            f"success={success} | "
            f"nodes={len(memory.nodes)} | "
            f"edges={edge_count}"
        )

    memory.export(args.out_dir, env_idx=0)
    export_run_summary(args.out_dir, run_summary)

    summary = {
        "success_rate": float(np.mean(run_summary["successes"])) if run_summary["successes"] else 0.0,
        "avg_return": float(np.mean(run_summary["episode_returns"])) if run_summary["episode_returns"] else 0.0,
        "avg_steps": float(np.mean(run_summary["episode_lengths"])) if run_summary["episode_lengths"] else 0.0,
        "final_nodes": int(len(memory.nodes)),
        "final_edges": int(sum(len(v) for v in memory.edges.values())),
        "intervention_rate": 0.0 if memory.intervention_attempts == 0 else memory.interventions / memory.intervention_attempts,
        "plan_hit_rate": 0.0 if memory.plan_total == 0 else memory.plan_hits / memory.plan_total,
    }

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nFinal summary:")
    for k, v in summary.items():
        print(f"{k}: {v}")

    env.close()


if __name__ == "__main__":
    main()