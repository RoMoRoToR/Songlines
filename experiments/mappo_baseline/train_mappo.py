"""
MAPPO baseline (Yu et al., 2022) on the scarcity scenario -- the stronger
cooperative-MARL baseline reviewers asked for beyond CommNet.

MAPPO = PPO with decentralized actors + a CENTRALIZED critic V(s) conditioned
on the concatenation of all agents' observations (CTDE). Actors reuse the
CommNet policy network (its actor head); the centralized critic is a separate
MLP [N*F] -> 128 -> 128 -> 1. Rollout/GAE/clipped-update logic mirrors
experiments/commnet_ppo_baseline/train_ppo_commnet.py.

After training, evaluate stage observability with the SAME Q/R/M/C logger:
  experiments/commnet_baseline/eval_with_qrmc.py --policy_path <out>/mappo_policy.pt

Run (cluster, CPU ok):
  PYTHONPATH=. python experiments/mappo_baseline/train_mappo.py \
      --total_updates 300 --out_dir tmp/cluster/mappo
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.big_experiment.env_factory import build_env
from experiments.commnet_baseline.commnet_agent import CommNetPolicy, encode_observation
from experiments.commnet_ppo_baseline.train_ppo_commnet import compute_gae


class CentralizedCritic(nn.Module):
    """V(s): MLP over the concatenation of all agents' encoded observations."""

    def __init__(self, n_agents: int, feat_dim: int = 131, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_agents * feat_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs_all: torch.Tensor) -> torch.Tensor:  # [N, F] -> scalar
        return self.net(obs_all.reshape(1, -1)).squeeze()


def rollout(env_seed, policy, critic, n_agents, n_targets, hazard,
            step_limit=80, layout="asymmetric"):
    built = build_env(n_agents=n_agents, n_waters=n_targets, layout=layout,
                      hazard_density=hazard, seed=env_seed, step_limit=step_limit)
    env, agent_ids = built.env, built.agent_ids
    obs_l, act_l, logp_l, val_l, rew_l = [], [], [], [], []
    for tick in range(step_limit):
        obs = np.stack([encode_observation(env, aid) for aid in agent_ids])
        obs_t = torch.tensor(obs)
        with torch.no_grad():
            logits, _, _ = policy(obs_t)
            v = critic(obs_t)                      # centralized value
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        result = env.step({aid: int(x) for aid, x in zip(agent_ids, actions.tolist())})
        step_r = 0.0
        for aid in agent_ids:
            ag = env.agents[aid]
            step_r -= 0.01
            if result.info[aid].cell_tag == "hazard_edge":
                step_r -= 0.1
            if ag.success and tick > 0 and not getattr(ag, "_rewarded", False):
                step_r += 1.0
                ag._rewarded = True
        obs_l.append(obs); act_l.append(actions.numpy())
        logp_l.append(dist.log_prob(actions).numpy())
        val_l.append(float(v)); rew_l.append(step_r)
        if result.all_succeeded:
            break
    return {"obs": obs_l, "actions": act_l, "old_logp": logp_l,
            "values": val_l, "rewards": rew_l,
            "n_succeeded": sum(1 for ag in env.agents.values() if ag.success),
            "n_agents": n_agents}


def mappo_update(policy, critic, opt, rollouts, *, clip_eps=0.2, epochs=4,
                 value_coef=0.5, ent_coef=0.01):
    fo, fa, fl, fadv, fret = [], [], [], [], []
    for ep in rollouts:
        adv = compute_gae(ep["rewards"], ep["values"])
        ret = [a + v for a, v in zip(adv, ep["values"])]
        fo.extend(ep["obs"]); fa.extend(ep["actions"])
        fl.extend(ep["old_logp"]); fadv.extend(adv); fret.extend(ret)
    obs_t = torch.tensor(np.array(fo)); act_t = torch.tensor(np.array(fa))
    old_t = torch.tensor(np.array(fl))
    adv_t = torch.tensor(fadv, dtype=torch.float32)
    ret_t = torch.tensor(fret, dtype=torch.float32)
    if adv_t.numel() > 1:
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-6)
    T = obs_t.shape[0]
    for _ in range(epochs):
        for t in range(T):
            logits, _, _ = policy(obs_t[t])
            dist = torch.distributions.Categorical(logits=logits)
            new_logp = dist.log_prob(act_t[t])
            ratio = (new_logp - old_t[t]).exp()
            s1 = ratio * adv_t[t]
            s2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_t[t]
            v = critic(obs_t[t])
            loss = (-torch.min(s1, s2).mean()
                    + value_coef * F.mse_loss(v, ret_t[t])
                    - ent_coef * dist.entropy().mean())
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(policy.parameters()) + list(critic.parameters()), 1.0)
            opt.step()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total_updates", type=int, default=300)
    ap.add_argument("--rollouts_per_update", type=int, default=64)
    ap.add_argument("--n_agents", type=int, default=3)
    ap.add_argument("--n_targets", type=int, default=2)
    ap.add_argument("--hazard", type=float, default=0.05)
    ap.add_argument("--layout", default="asymmetric")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out_dir", default="tmp/cluster/mappo")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    torch.manual_seed(0); np.random.seed(0)

    policy = CommNetPolicy()
    critic = CentralizedCritic(a.n_agents)
    opt = torch.optim.Adam(list(policy.parameters()) + list(critic.parameters()),
                           lr=a.lr)
    t0 = time.time(); seed_iter = 0; running: List[float] = []
    for upd in range(a.total_updates):
        rolls = []
        for _ in range(a.rollouts_per_update):
            rolls.append(rollout(seed_iter % 200, policy, critic,
                                 a.n_agents, a.n_targets, a.hazard,
                                 layout=a.layout))
            seed_iter += 1
        mappo_update(policy, critic, opt, rolls)
        succ = float(np.mean([r["n_succeeded"] / r["n_agents"] for r in rolls]))
        running = (running + [succ])[-50:]
        if upd % 5 == 0:
            print(f"  upd {upd:4d}  succ={succ:.3f}  last50={np.mean(running):.3f}"
                  f"  {time.time()-t0:.0f}s", flush=True)
    torch.save(policy.state_dict(), os.path.join(a.out_dir, "mappo_policy.pt"))
    torch.save(critic.state_dict(), os.path.join(a.out_dir, "mappo_critic.pt"))
    print(f"saved -> {a.out_dir}/mappo_policy.pt  "
          f"(eval next with eval_with_qrmc.py --policy_path ...)")


if __name__ == "__main__":
    main()
