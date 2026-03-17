#!/usr/bin/env python3
import argparse
import json
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.lz_memory import LZMapMemory, SymbolicTokenizer


def main():
    parser = argparse.ArgumentParser("Standalone smoke test for LZ pipeline")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--emb-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--tokenizer_mode", type=str, default="argmax",
                        choices=["argmax", "hash_sign"])
    parser.add_argument("--tokenizer_proj_dim", type=int, default=32)
    parser.add_argument("--lz_min_goal_visits", type=int, default=3)
    parser.add_argument("--lz_topk_goals", type=int, default=5)
    parser.add_argument("--suggest_every", type=int, default=25)
    parser.add_argument("--out_dir", type=str, default="tmp/lz_smoke")
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    tokenizer = SymbolicTokenizer(
        mode=args.tokenizer_mode,
        proj_dim=args.tokenizer_proj_dim,
        seed=args.seed,
    )
    memory = LZMapMemory(min_goal_visits=args.lz_min_goal_visits)

    goal_center = np.array([120.0, 120.0], dtype=np.float32)
    interventions = []
    for step in range(args.steps):
        # Synthetic embedding with periodic structure to force phrase repeats.
        base = rng.randn(args.emb_dim).astype(np.float32)
        base[(step // 10) % 8] += 4.0
        token = tokenizer.encode(base)
        memory.update_token(token, step)

        pose = np.array([
            120.0 + 40.0 * np.sin(step / 15.0),
            120.0 + 40.0 * np.cos(step / 15.0),
        ], dtype=np.float32)
        dist = float(np.linalg.norm(pose - goal_center))
        reward = float(max(0.0, 1.0 - dist / 80.0))
        memory.observe(step, pose_xy=pose, goal_xy=goal_center, reward=reward)

        if (step + 1) % args.suggest_every == 0:
            proposal = memory.suggest_subgoal(top_k=args.lz_topk_goals)
            if proposal is not None:
                proposal = {
                    "goal_xy": [float(proposal["goal_xy"][0]),
                                float(proposal["goal_xy"][1])],
                    "node_id": int(proposal["node_id"]),
                    "path_len": int(proposal["path_len"]),
                    "mean_reward": float(proposal["mean_reward"]),
                }
            interventions.append({
                "step": step,
                "proposal": proposal,
            })
            if proposal is not None:
                memory.record_plan_outcome(improved=True)

    os.makedirs(args.out_dir, exist_ok=True)
    memory.export(args.out_dir, env_idx=0)
    with open(os.path.join(args.out_dir, "interventions.json"), "w") as f:
        json.dump(interventions, f, indent=2)

    print("Smoke test complete.")
    print("nodes={}, edges={}, interventions={}".format(
        len(memory.nodes),
        sum(len(v) for v in memory.edges.values()),
        memory.interventions,
    ))
    print("Artifacts:", os.path.abspath(args.out_dir))


if __name__ == "__main__":
    main()
