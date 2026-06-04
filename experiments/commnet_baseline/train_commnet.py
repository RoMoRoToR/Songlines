"""Part 2.2 — train CommNet policy with REINFORCE + baseline.

Scenario: scarcity case (N=3 agents, M=2 waters), asymmetric layout, hazard 0.05.
Curriculum: train on seed-varying episodes; report success on held-out seeds.

Reward shaping:
  +1.0 per agent on success (reaching water)
  -0.01 per step (encourages efficiency)
  -0.1 per hazard hit

Training budget: 2000 episodes; loss = -mean(log π(a) · (R - V)) + value-MSE.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
import torch.nn.functional as F

from experiments.big_experiment.env_factory import build_env
from experiments.commnet_baseline.commnet_agent import CommNetPolicy, encode_observation


def run_episode(env_seed: int, policy: CommNetPolicy,
                n_agents: int, n_waters: int, hazard: float,
                step_limit: int = 80, layout: str = "asymmetric",
                train: bool = True, device: str = "cpu"):
    built = build_env(
        n_agents=n_agents, n_waters=n_waters,
        layout=layout, hazard_density=hazard,
        seed=env_seed, step_limit=step_limit,
    )
    env = built.env
    agent_ids = built.agent_ids

    log_probs: List[torch.Tensor] = []
    values: List[torch.Tensor] = []
    rewards: List[float] = []
    entropies: List[torch.Tensor] = []

    for tick in range(step_limit):
        obs = np.stack([encode_observation(env, aid) for aid in agent_ids])
        obs_t = torch.tensor(obs, device=device)
        logits, value, _ = policy(obs_t)
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        log_p = dist.log_prob(actions)
        ent = dist.entropy()

        actions_dict = {aid: int(a) for aid, a in zip(agent_ids, actions.tolist())}
        result = env.step(actions_dict)

        step_reward = 0.0
        for aid in agent_ids:
            ag = env.agents[aid]
            # Per-step shaping
            step_reward -= 0.01
            if result.info[aid].cell_tag == "hazard_edge":
                step_reward -= 0.1
            if ag.success and tick > 0:
                # Reward only the tick of first success
                if hasattr(ag, "_rewarded"):
                    pass
                else:
                    step_reward += 1.0
                    ag._rewarded = True

        rewards.append(step_reward)
        log_probs.append(log_p.sum())
        values.append(value.mean())
        entropies.append(ent.mean())

        if result.all_succeeded:
            break

    n_succeeded = sum(1 for ag in env.agents.values() if ag.success)
    success_rate = n_succeeded / max(1, n_agents)
    total_steps = tick + 1
    return {
        "log_probs": log_probs,
        "values": values,
        "rewards": rewards,
        "entropies": entropies,
        "success_rate": success_rate,
        "n_succeeded": n_succeeded,
        "steps": total_steps,
    }


def reinforce_loss(ep: Dict, gamma: float = 0.99,
                   value_coef: float = 0.5, ent_coef: float = 0.01):
    rewards = ep["rewards"]
    log_probs = ep["log_probs"]
    values = ep["values"]
    entropies = ep["entropies"]

    # Discounted returns
    G = 0.0
    returns: List[float] = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)
    returns.reverse()
    returns_t = torch.tensor(returns, dtype=torch.float32)
    if returns_t.numel() > 1 and returns_t.std() > 1e-6:
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-6)

    log_probs_t = torch.stack(log_probs)
    values_t = torch.stack(values)
    advantages = returns_t - values_t.detach()

    policy_loss = -(log_probs_t * advantages).mean()
    value_loss = F.mse_loss(values_t, returns_t)
    entropy_loss = -torch.stack(entropies).mean()
    return policy_loss + value_coef * value_loss + ent_coef * entropy_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_episodes", type=int, default=2000)
    parser.add_argument("--n_agents", type=int, default=3)
    parser.add_argument("--n_waters", type=int, default=2)
    parser.add_argument("--hazard", type=float, default=0.05)
    parser.add_argument("--layout", default="asymmetric")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--out_dir", default="experiments/commnet_baseline")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    torch.manual_seed(0)
    np.random.seed(0)
    policy = CommNetPolicy()
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)

    log = {
        "episode": [], "reward_sum": [], "success_rate": [], "steps": [],
        "loss": [],
    }
    t0 = time.time()
    running_succ: List[float] = []

    for ep_idx in range(args.n_episodes):
        env_seed = ep_idx % 200  # train across 200 env seeds, cycling
        ep = run_episode(
            env_seed, policy, args.n_agents, args.n_waters, args.hazard,
            layout=args.layout, train=True,
        )
        loss = reinforce_loss(ep)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        opt.step()

        log["episode"].append(ep_idx)
        log["reward_sum"].append(float(sum(ep["rewards"])))
        log["success_rate"].append(ep["success_rate"])
        log["steps"].append(ep["steps"])
        log["loss"].append(float(loss.detach()))
        running_succ.append(ep["success_rate"])
        running_succ = running_succ[-100:]

        if ep_idx % 100 == 0:
            print(f"  ep {ep_idx:5d}  loss={float(loss):+.3f}  "
                  f"R_sum={sum(ep['rewards']):+.2f}  "
                  f"succ_rate(last100)={statistics.mean(running_succ):.3f}  "
                  f"elapsed={time.time()-t0:.0f}s")

    # Final eval on held-out seeds 250..299 (50 episodes)
    print("\nEvaluating on held-out seeds 250..299 (50 episodes)...")
    eval_succ = []
    eval_steps = []
    eval_n_succ = []
    for s in range(250, 300):
        ep = run_episode(
            s, policy, args.n_agents, args.n_waters, args.hazard,
            layout=args.layout, train=False,
        )
        eval_succ.append(ep["success_rate"])
        eval_steps.append(ep["steps"])
        eval_n_succ.append(ep["n_succeeded"])

    summary = {
        "n_episodes_trained": args.n_episodes,
        "n_agents": args.n_agents, "n_waters": args.n_waters,
        "layout": args.layout, "hazard_density": args.hazard,
        "train_seconds": time.time() - t0,
        "train_last100_success_rate_mean": statistics.mean(running_succ),
        "eval_success_rate_mean": statistics.mean(eval_succ),
        "eval_success_rate_std": statistics.stdev(eval_succ) if len(eval_succ) > 1 else 0.0,
        "eval_mean_steps": statistics.mean(eval_steps),
        "eval_n_succ_distribution": dict(
            (k, eval_n_succ.count(k)) for k in sorted(set(eval_n_succ))
        ),
    }
    print(f"\n=== Eval summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    with open(os.path.join(args.out_dir, "train_log.json"), "w") as f:
        json.dump(log, f)
    with open(os.path.join(args.out_dir, "eval_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    torch.save(policy.state_dict(), os.path.join(args.out_dir, "commnet_policy.pt"))
    print(f"\nSaved → {args.out_dir}/{{train_log.json,eval_summary.json,commnet_policy.pt}}")


if __name__ == "__main__":
    main()
