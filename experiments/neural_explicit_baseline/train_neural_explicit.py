"""Train the neural-explicit-interface baseline (package J) with PPO.

Same scarcity scenario as experiments/mappo_baseline/train_mappo.py
(N=3 agents, M=2 waters, asymmetric layout, hazard 0.05, step_limit 80),
same env (experiments.big_experiment.env_factory.build_env), same obs
encoding -- but the policy is factored into explicit Q/R/M/C heads (see
experiments/neural_explicit_baseline/agent.py) coupled ONLY through a
shared append-only external memory (memory_store.py).  No comm channel.

Per-tick decision flow per (non-succeeded) agent:
  1. write current observation into the shared append-only store
  2. h = encoder(obs131)
  3. Q: sample query token (require-tag or NO_QUERY)         -> Q-event
  4. R: if query != NO_QUERY and store non-empty, learned retriever scores
     ALL entries, top-k candidates surface                   -> R-event
  5. M: lock head samples over [keep-lock] + k candidates; choosing a
     candidate commits env-visible lock
     (env.agents[aid].locked_target = (x, y))                -> M-event
  6. C: controller samples motor action given (h, lock features);
     arrival at the locked cell                              -> C-event

Rewards are the env-native per-agent rewards (-0.01 step, -0.5 hazard,
+1.0 on reaching water).  Optional --lock_shaping adds potential-based
shaping toward the CURRENT lock (phi = -manhattan(pos, lock)/(W+H)); it is
disclosed, potential-based (policy-invariant in the limit) and rewards
using the lock mechanism, not reaching water per se.  Default 0 (off).

Training: PPO (clipped), joint log-prob = sum of the active heads'
log-probs, GAE per agent-trajectory.  During PPO epochs the top-k
candidate SET is frozen from rollout time (standard approximation);
retriever/lock scores over those candidates are re-computed with
gradients (retriever score is mixed into the lock logit, so R gets
gradient through M's log-prob).

The curve JSON is dumped EVERY --dump_every updates (not only at the end).

Smoke (local):
  PYTHONPATH=. .venv/bin/python experiments/neural_explicit_baseline/train_neural_explicit.py \
      --total_updates 40 --rollouts_per_update 16 --out_dir tmp/neural_explicit_smoke
Full (cluster): see cluster/submit_neural_explicit.sh
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
import torch.nn.functional as F

from experiments.big_experiment.env_factory import build_env
from experiments.commnet_baseline.commnet_agent import encode_observation
from experiments.commnet_ppo_baseline.train_ppo_commnet import compute_gae
from experiments.neural_explicit_baseline.agent import (
    LOCK_FEAT_DIM, NeuralExplicitPolicy, lock_features,
)
from experiments.neural_explicit_baseline.memory_store import (
    ENTRY_FEAT_DIM, NO_QUERY, QUERY_VOCAB, AppendOnlyMemory,
)

GAMMA = 0.99


# ────────────────────────────────────────────────────────────── rollout

def rollout(env_seed: int, policy: NeuralExplicitPolicy, a,
            event_log: Optional[List[dict]] = None) -> dict:
    """Collect one episode.  Returns per-agent transition arrays + stats.

    If ``event_log`` is a list, structured Q/R/M/C events are appended
    (one dict per event, JSONL-ready).
    """
    built = build_env(n_agents=a.n_agents, n_waters=a.n_targets,
                      layout=a.layout, hazard_density=a.hazard,
                      seed=env_seed, step_limit=a.step_limit)
    env, agent_ids = built.env, built.agent_ids
    water_set = set(built.water_positions)
    memory = AppendOnlyMemory()
    K = a.k_retrieval

    locks: Dict[str, Optional[Tuple[int, int]]] = {aid: None for aid in agent_ids}
    lock_tags: Dict[str, str] = {}
    lock_arrival_logged: Dict[str, bool] = {aid: False for aid in agent_ids}

    traj = {aid: {"obs": [], "query_tok": [], "has_ret": [],
                  "cand_feats": [], "cand_mask": [], "cand_scores_frozen": [],
                  "lock_choice": [], "lock_feat": [], "motor": [],
                  "old_logp": [], "value": [], "reward": []}
            for aid in agent_ids}
    ev = {"q_events": 0, "r_events": 0, "m_events": 0, "c_events": 0,
          "r_cand_total": 0, "r_cand_satisfying": 0,
          "m_top1": 0, "m_lock_is_water": 0, "decision_ticks": 0}

    for tick in range(a.step_limit):
        active = [aid for aid in agent_ids if not env.agents[aid].success]
        if not active:
            break
        for aid in active:
            memory.write_agent_observation(env, aid, tick)

        actions: Dict[str, int] = {}
        pre_phi: Dict[str, float] = {}
        with torch.no_grad():
            for aid in active:
                ag = env.agents[aid]
                obs = encode_observation(env, aid)
                h = policy.encode(torch.from_numpy(obs).unsqueeze(0))  # [1,H]
                ev["decision_ticks"] += 1

                # ── Q: emit discrete query
                q_dist = torch.distributions.Categorical(
                    logits=policy.query_logits(h))
                q_tok = int(q_dist.sample())
                q_logp = float(q_dist.log_prob(torch.tensor(q_tok)))
                if q_tok != NO_QUERY:
                    ev["q_events"] += 1
                    if event_log is not None:
                        event_log.append({
                            "episode": env_seed, "tick": tick, "agent": aid,
                            "type": "Q", "query": QUERY_VOCAB[q_tok]})

                # ── R: learned retrieval over the full store
                has_ret = (q_tok != NO_QUERY) and len(memory) > 0
                cand_feats = np.zeros((K, ENTRY_FEAT_DIM), dtype=np.float32)
                cand_mask = np.zeros(K, dtype=bool)
                cand_scores = np.zeros(K, dtype=np.float32)
                cand_entries: List = []
                if has_ret:
                    feats_all = memory.features(
                        (ag.x, ag.y), tick, env.width, env.height, a.step_limit)
                    scores = policy.retrieval_scores(
                        h, torch.tensor([q_tok]),
                        torch.from_numpy(feats_all).unsqueeze(0))[0]  # [E]
                    k_eff = min(K, len(memory))
                    top = torch.topk(scores, k_eff)
                    for j, idx in enumerate(top.indices.tolist()):
                        cand_feats[j] = feats_all[idx]
                        cand_mask[j] = True
                        cand_scores[j] = float(scores[idx])
                        cand_entries.append(memory.entries[idx])
                    ev["r_events"] += 1
                    n_sat = sum(1 for e in cand_entries
                                if e.tag == QUERY_VOCAB[q_tok])
                    ev["r_cand_total"] += k_eff
                    ev["r_cand_satisfying"] += n_sat
                    if event_log is not None:
                        event_log.append({
                            "episode": env_seed, "tick": tick, "agent": aid,
                            "type": "R", "query": QUERY_VOCAB[q_tok],
                            "store_size": len(memory), "k": k_eff,
                            "candidates": [
                                {"xy": [e.x, e.y], "tag": e.tag,
                                 "score": round(float(s), 4),
                                 "satisfies_query": e.tag == QUERY_VOCAB[q_tok]}
                                for e, s in zip(cand_entries, cand_scores)],
                            "frac_satisfying": n_sat / k_eff})

                # ── M: target-lock commit
                lock_choice, lock_logp = 0, 0.0
                if has_ret:
                    l_logits = policy.lock_logits(
                        h, torch.from_numpy(cand_feats).unsqueeze(0),
                        torch.from_numpy(cand_scores).unsqueeze(0),
                        torch.from_numpy(cand_mask).unsqueeze(0))
                    l_dist = torch.distributions.Categorical(logits=l_logits[0])
                    lock_choice = int(l_dist.sample())
                    lock_logp = float(l_dist.log_prob(torch.tensor(lock_choice)))
                    if lock_choice > 0:  # commit
                        e = cand_entries[lock_choice - 1]
                        locks[aid] = (e.x, e.y)
                        lock_tags[aid] = e.tag
                        lock_arrival_logged[aid] = False
                        # env-visible commit:
                        env.agents[aid].locked_target = (e.x, e.y)
                        ev["m_events"] += 1
                        is_top1 = (lock_choice == 1)
                        ev["m_top1"] += int(is_top1)
                        ev["m_lock_is_water"] += int((e.x, e.y) in water_set)
                        if event_log is not None:
                            event_log.append({
                                "episode": env_seed, "tick": tick,
                                "agent": aid, "type": "M",
                                "lock_xy": [e.x, e.y], "lock_tag": e.tag,
                                "candidate_rank": lock_choice,
                                "is_top1_retrieval": is_top1})

                # ── C: motor action given lock
                lf = lock_features((ag.x, ag.y), locks[aid],
                                   env.width, env.height)
                m_dist = torch.distributions.Categorical(
                    logits=policy.motor_logits(h, lf.unsqueeze(0))[0])
                motor = int(m_dist.sample())
                m_logp = float(m_dist.log_prob(torch.tensor(motor)))
                v = float(policy.value(h, lf.unsqueeze(0)))
                actions[aid] = motor

                if locks[aid] is not None:
                    lx, ly = locks[aid]
                    pre_phi[aid] = -(abs(lx - ag.x) + abs(ly - ag.y)) \
                        / (env.width + env.height)

                t = traj[aid]
                t["obs"].append(obs)
                t["query_tok"].append(q_tok)
                t["has_ret"].append(has_ret)
                t["cand_feats"].append(cand_feats)
                t["cand_mask"].append(cand_mask)
                t["cand_scores_frozen"].append(cand_scores)
                t["lock_choice"].append(lock_choice)
                t["lock_feat"].append(lf.numpy())
                t["motor"].append(motor)
                t["old_logp"].append(q_logp + (lock_logp if has_ret else 0.0)
                                     + m_logp)
                t["value"].append(v)

        result = env.step(actions)

        for aid in active:
            ag = env.agents[aid]
            r = result.rewards[aid]
            if a.lock_shaping > 0 and locks[aid] is not None and aid in pre_phi:
                lx, ly = locks[aid]
                post_phi = -(abs(lx - ag.x) + abs(ly - ag.y)) \
                    / (env.width + env.height)
                r += a.lock_shaping * (GAMMA * post_phi - pre_phi[aid])
            traj[aid]["reward"].append(r)
            # C-event: arrival at the committed lock (or success)
            arrived = locks[aid] is not None and (ag.x, ag.y) == locks[aid]
            if (arrived or ag.success) and not lock_arrival_logged[aid]:
                lock_arrival_logged[aid] = True
                ev["c_events"] += 1
                if event_log is not None:
                    event_log.append({
                        "episode": env_seed, "tick": tick, "agent": aid,
                        "type": "C",
                        "lock_xy": list(locks[aid]) if locks[aid] else None,
                        "arrived_at_lock": bool(arrived),
                        "success": bool(ag.success)})

        if result.all_succeeded:
            break

    ev["n_succeeded"] = sum(1 for ag in env.agents.values() if ag.success)
    ev["n_agents"] = a.n_agents
    ev["store_size_final"] = len(memory)
    return {"traj": traj, "stats": ev}


# ─────────────────────────────────────────────────────────── PPO update

def flatten_rollouts(rollouts: List[dict]) -> Optional[Dict[str, torch.Tensor]]:
    batch = {k: [] for k in ("obs", "query_tok", "has_ret", "cand_feats",
                             "cand_mask", "cand_scores_frozen", "lock_choice",
                             "lock_feat", "motor", "old_logp", "adv", "ret")}
    for r in rollouts:
        for aid, t in r["traj"].items():
            T = len(t["reward"])
            if T == 0:
                continue
            adv = compute_gae(t["reward"], t["value"][:T])
            ret = [ad + v for ad, v in zip(adv, t["value"][:T])]
            for k in ("obs", "query_tok", "has_ret", "cand_feats", "cand_mask",
                      "cand_scores_frozen", "lock_choice", "lock_feat",
                      "motor", "old_logp"):
                batch[k].extend(t[k][:T])
            batch["adv"].extend(adv)
            batch["ret"].extend(ret)
    if not batch["obs"]:
        return None
    out = {
        "obs": torch.from_numpy(np.array(batch["obs"], dtype=np.float32)),
        "query_tok": torch.tensor(batch["query_tok"], dtype=torch.long),
        "has_ret": torch.tensor(batch["has_ret"], dtype=torch.bool),
        "cand_feats": torch.from_numpy(np.array(batch["cand_feats"],
                                                dtype=np.float32)),
        "cand_mask": torch.from_numpy(np.array(batch["cand_mask"])),
        "lock_choice": torch.tensor(batch["lock_choice"], dtype=torch.long),
        "lock_feat": torch.from_numpy(np.array(batch["lock_feat"],
                                               dtype=np.float32)),
        "motor": torch.tensor(batch["motor"], dtype=torch.long),
        "old_logp": torch.tensor(batch["old_logp"], dtype=torch.float32),
        "adv": torch.tensor(batch["adv"], dtype=torch.float32),
        "ret": torch.tensor(batch["ret"], dtype=torch.float32),
    }
    if out["adv"].numel() > 1:
        out["adv"] = (out["adv"] - out["adv"].mean()) / (out["adv"].std() + 1e-6)
    return out


def ppo_update(policy: NeuralExplicitPolicy, opt, batch, *,
               clip_eps=0.2, epochs=4, minibatch=512,
               value_coef=0.5, ent_coef=0.01) -> dict:
    B = batch["obs"].shape[0]
    losses = []
    for _ in range(epochs):
        perm = torch.randperm(B)
        for s in range(0, B, minibatch):
            idx = perm[s:s + minibatch]
            obs = batch["obs"][idx]
            h = policy.encode(obs)
            lf = batch["lock_feat"][idx]
            has_ret = batch["has_ret"][idx]

            q_dist = torch.distributions.Categorical(
                logits=policy.query_logits(h))
            q_logp = q_dist.log_prob(batch["query_tok"][idx])

            # Lock head: recompute retriever scores on the frozen candidate
            # set WITH gradient (this is how R gets trained through M).
            cand_feats = batch["cand_feats"][idx]
            cand_mask = batch["cand_mask"][idx].clone()
            # rows without retrieval: give one dummy valid slot to keep the
            # Categorical well-defined; their contribution is masked out.
            dummy = ~cand_mask.any(dim=1)
            cand_mask[dummy, 0] = True
            retr_scores = policy.retrieval_scores(
                h, batch["query_tok"][idx].clamp(max=len(QUERY_VOCAB) - 1),
                cand_feats)
            l_logits = policy.lock_logits(h, cand_feats, retr_scores, cand_mask)
            l_dist = torch.distributions.Categorical(logits=l_logits)
            l_logp = l_dist.log_prob(batch["lock_choice"][idx])
            l_logp = torch.where(has_ret, l_logp, torch.zeros_like(l_logp))
            l_ent = torch.where(has_ret, l_dist.entropy(),
                                torch.zeros_like(l_logp))

            m_dist = torch.distributions.Categorical(
                logits=policy.motor_logits(h, lf))
            m_logp = m_dist.log_prob(batch["motor"][idx])

            new_logp = q_logp + l_logp + m_logp
            ratio = (new_logp - batch["old_logp"][idx]).exp()
            adv = batch["adv"][idx]
            s1 = ratio * adv
            s2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
            v = policy.value(h, lf)
            entropy = (q_dist.entropy() + l_ent + m_dist.entropy()).mean()
            loss = (-torch.min(s1, s2).mean()
                    + value_coef * F.mse_loss(v, batch["ret"][idx])
                    - ent_coef * entropy)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))
    return {"loss_mean": float(np.mean(losses))}


# ──────────────────────────────────────────────────────────────── main

def aggregate_event_stats(rolls: List[dict]) -> dict:
    tot = {}
    for k in ("q_events", "r_events", "m_events", "c_events", "r_cand_total",
              "r_cand_satisfying", "m_top1", "m_lock_is_water",
              "decision_ticks"):
        tot[k] = sum(r["stats"][k] for r in rolls)
    d = max(1, tot["decision_ticks"])
    return {
        "q_rate": tot["q_events"] / d,
        "r_rate": tot["r_events"] / d,
        "m_commits_per_episode": tot["m_events"] / len(rolls),
        "c_events_per_episode": tot["c_events"] / len(rolls),
        "r_frac_cand_satisfying": (tot["r_cand_satisfying"]
                                   / max(1, tot["r_cand_total"])),
        "m_frac_lock_is_top1": tot["m_top1"] / max(1, tot["m_events"]),
        "m_frac_lock_is_water": (tot["m_lock_is_water"]
                                 / max(1, tot["m_events"])),
    }


def dump_curve(a, curve, path):
    with open(path, "w") as f:
        json.dump({"total_updates": a.total_updates,
                   "rollouts_per_update": a.rollouts_per_update,
                   "n_agents": a.n_agents, "n_targets": a.n_targets,
                   "layout": a.layout, "hazard": a.hazard,
                   "k_retrieval": a.k_retrieval,
                   "lock_shaping": a.lock_shaping, "seed": a.seed,
                   "curve": curve}, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total_updates", type=int, default=400)
    ap.add_argument("--rollouts_per_update", type=int, default=64)
    ap.add_argument("--n_agents", type=int, default=3)
    ap.add_argument("--n_targets", type=int, default=2)
    ap.add_argument("--hazard", type=float, default=0.05)
    ap.add_argument("--layout", default="asymmetric")
    ap.add_argument("--step_limit", type=int, default=80)
    ap.add_argument("--k_retrieval", type=int, default=4)
    ap.add_argument("--lock_shaping", type=float, default=0.0,
                    help="potential-based shaping toward current lock (0=off)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent_coef", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dump_every", type=int, default=10,
                    help="dump curve JSON every N updates (not only at end)")
    ap.add_argument("--dump_events_episodes", type=int, default=3,
                    help="episodes of full Q/R/M/C JSONL events dumped at end")
    ap.add_argument("--out_dir", default="tmp/neural_explicit")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    policy = NeuralExplicitPolicy()
    n_params = sum(p.numel() for p in policy.parameters())
    opt = torch.optim.Adam(policy.parameters(), lr=a.lr)
    print(f"NeuralExplicitPolicy: {n_params} params  "
          f"(Q/R/M/C explicit heads, K={a.k_retrieval})", flush=True)

    curve_path = os.path.join(a.out_dir, "neural_explicit_curve.json")
    t0 = time.time()
    seed_iter = 0
    running: List[float] = []
    curve = []
    for upd in range(a.total_updates):
        rolls = []
        for _ in range(a.rollouts_per_update):
            rolls.append(rollout(seed_iter % 200, policy, a))
            seed_iter += 1
        batch = flatten_rollouts(rolls)
        upd_info = ppo_update(policy, opt, batch,
                              ent_coef=a.ent_coef) if batch else {}
        succ = float(np.mean([r["stats"]["n_succeeded"]
                              / r["stats"]["n_agents"] for r in rolls]))
        running = (running + [succ])[-50:]
        ev = aggregate_event_stats(rolls)
        curve.append({"update": upd, "success": succ,
                      "success_smooth50": float(np.mean(running)),
                      **ev, **upd_info})
        if upd % 5 == 0 or upd == a.total_updates - 1:
            print(f"  upd {upd:4d}  succ={succ:.3f}  "
                  f"last50={np.mean(running):.3f}  q={ev['q_rate']:.2f}  "
                  f"sat={ev['r_frac_cand_satisfying']:.2f}  "
                  f"top1={ev['m_frac_lock_is_top1']:.2f}  "
                  f"lockw={ev['m_frac_lock_is_water']:.2f}  "
                  f"{time.time()-t0:.0f}s", flush=True)
        if (upd + 1) % a.dump_every == 0 or upd == a.total_updates - 1:
            dump_curve(a, curve, curve_path)  # periodic! not only at end
            torch.save(policy.state_dict(),
                       os.path.join(a.out_dir, "neural_explicit_policy.pt"))

    # Final: dump structured Q/R/M/C event logs on held-out seeds (250..)
    events: List[dict] = []
    with torch.no_grad():
        for s in range(250, 250 + a.dump_events_episodes):
            rollout(s, policy, a, event_log=events)
    with open(os.path.join(a.out_dir, "qrmc_events.jsonl"), "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    dump_curve(a, curve, curve_path)
    torch.save(policy.state_dict(),
               os.path.join(a.out_dir, "neural_explicit_policy.pt"))
    print(f"saved -> {curve_path} + neural_explicit_policy.pt + "
          f"qrmc_events.jsonl ({len(events)} events)  "
          f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
