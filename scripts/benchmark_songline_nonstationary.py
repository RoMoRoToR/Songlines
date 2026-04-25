import argparse
from types import SimpleNamespace

from scripts.compare_songline_minigrid import DEFAULT_ENV_IDS, run_comparison


DEFAULT_METHODS = [
    "milestone_semantic_handoff_v1",
    "milestone_semantic_handoff_v1_adaptive_graph",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_ids", nargs="+", default=DEFAULT_ENV_IDS)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, choices=DEFAULT_METHODS)
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--num_seeds", type=int, default=10)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--max_steps", type=int, default=120)
    parser.add_argument("--epsilon", type=float, default=0.15)
    parser.add_argument("--suggest_every", type=int, default=8)
    parser.add_argument("--intervention_patience", type=int, default=4)
    parser.add_argument("--min_goal_visits", type=int, default=2)
    parser.add_argument("--top_k_goals", type=int, default=5)
    parser.add_argument("--graph_rollout_horizon", type=int, default=4)
    parser.add_argument("--scene_radius", type=int, default=1)
    parser.add_argument("--tokenizer_mode", type=str, default="hash_sign", choices=["argmax", "hash_sign"])
    parser.add_argument("--tokenizer_proj_dim", type=int, default=16)
    parser.add_argument("--token_source", type=str, default="symbolic_hash", choices=["symbolic_hash", "scene_semantic", "scene_patch_hash"])
    parser.add_argument("--env_change_mode", type=str, default="goal_shift_v1", choices=["goal_shift_v1"])
    parser.add_argument("--change_after_episode", type=int, default=-1)
    parser.add_argument("--out_dir", type=str, default="tmp/songline_nonstationary_benchmark")
    return parser.parse_args()


def main():
    args = parse_args()
    change_after_episode = args.change_after_episode
    if change_after_episode < 0:
        change_after_episode = max(1, args.episodes // 2)

    compare_args = SimpleNamespace(
        env_ids=args.env_ids,
        methods=args.methods,
        seed_start=args.seed_start,
        num_seeds=args.num_seeds,
        seeds=args.seeds,
        episodes=args.episodes,
        max_steps=args.max_steps,
        epsilon=args.epsilon,
        suggest_every=args.suggest_every,
        intervention_patience=args.intervention_patience,
        min_goal_visits=args.min_goal_visits,
        top_k_goals=args.top_k_goals,
        graph_rollout_horizon=args.graph_rollout_horizon,
        token_source=args.token_source,
        scene_radius=args.scene_radius,
        env_change_mode=args.env_change_mode,
        change_after_episode=change_after_episode,
        tokenizer_mode=args.tokenizer_mode,
        tokenizer_proj_dim=args.tokenizer_proj_dim,
        out_dir=args.out_dir,
    )
    run_comparison(compare_args)


if __name__ == "__main__":
    main()
