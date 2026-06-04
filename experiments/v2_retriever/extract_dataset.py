"""Part 2.1 — extract (features, label) dataset from existing hazard-recovery runs.

Diagnostic finding from data inspection:
  349 query calls across 10 seeds; only 32 had ANY candidate (91% empty).
  Of the 32 with candidates, 31 were "satisfied" (selected was semantically valid).
  → The retrieval bottleneck is candidate GENERATION, not candidate RANKING.

We therefore extract a per-step candidate-generation dataset:
  Features:  agent_state numerical fields + semantic_tags dict (per-tag confidences)
             + one-hot active_intent
  Label:     was this step's symbolic node EVER selected as a candidate during the
             original benchmark run (i.e. did the symbolic memory ever flag it as a
             hazard-recovery candidate)?  This isolates "candidate-worthy state"
             from "actually retrieved" downstream filtering.

Output: dataset_v2.npz with X (n_steps, d), y (n_steps,), seed (n_steps,), env (n_steps,)
"""

from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

ROOT = "/Users/taniyashuba/PycharmProjects/Songlines/tmp/article_revision_10seeds_20260501/hazard_recovery"
METHOD = "milestone_state_conditioned_hazard_recovery_v7"

INTENT_KEYS = [
    "find_goal_region", "hazard_recovery_exit", "find_water", "find_rest",
    "explore", "default_goal",
]

TAG_KEYS = [
    "corridor", "goal_region", "hazard_edge", "hazard_recovery_route",
    "near_water", "open_safe_rest_zone", "post_hazard_goal_rejoin",
    "rest_candidate", "adjacent_hazard",
]

NUMERIC_AGENT_KEYS = ["energy", "thirst", "risk_budget"]


def _extract_features(step_record: Dict) -> np.ndarray:
    feats: List[float] = []
    agent = step_record.get("agent_state", {})
    for k in NUMERIC_AGENT_KEYS:
        feats.append(float(agent.get(k, 0.0)))
    tags = step_record.get("semantic_tags", {})
    for k in TAG_KEYS:
        feats.append(float(tags.get(k, 0.0)))
    intent = step_record.get("active_intent") or agent.get("active_intent") or ""
    for k in INTENT_KEYS:
        feats.append(1.0 if intent == k else 0.0)
    return np.asarray(feats, dtype=np.float32)


def _extract_seed_dir(seed_dir: str, seed_id: int) -> Tuple[List[np.ndarray], List[int]]:
    method_dir = os.path.join(seed_dir, METHOD)
    if not os.path.isdir(method_dir):
        return [], []
    # Episode records: per-step features
    ep_path = os.path.join(method_dir, "episode_records.json")
    if not os.path.exists(ep_path):
        return [], []
    with open(ep_path) as f:
        episodes = json.load(f)
    # Query debug: which symbolic_node_ids ever became candidates
    qd_path = os.path.join(method_dir, "query_debug.json")
    candidate_node_ids: set = set()
    if os.path.exists(qd_path):
        with open(qd_path) as f:
            qd = json.load(f)
        for entry in qd:
            for nid in entry.get("candidate_node_ids", []):
                candidate_node_ids.add(int(nid))

    X_list = []
    y_list = []
    for ep in episodes:
        for sr in ep.get("step_records", []):
            x = _extract_features(sr)
            nid = sr.get("symbolic_node_id") or sr.get("node_id")
            label = 1 if (nid is not None and int(nid) in candidate_node_ids) else 0
            X_list.append(x)
            y_list.append(label)
    return X_list, y_list


def main():
    assists = "assists_on"
    env_dir = os.path.join(ROOT, assists, "MiniGrid-LavaGapS7-v0")
    seed_dirs = sorted(glob.glob(os.path.join(env_dir, "seed_*")),
                       key=lambda p: int(p.split("_")[-1]))

    all_X = []
    all_y = []
    all_seed = []

    for sd in seed_dirs:
        seed_id = int(os.path.basename(sd).split("_")[-1])
        X, y = _extract_seed_dir(sd, seed_id)
        all_X.extend(X)
        all_y.extend(y)
        all_seed.extend([seed_id] * len(X))
        print(f"  seed {seed_id}: {len(X)} steps, {sum(y)} positive labels")

    X = np.stack(all_X)
    y = np.asarray(all_y, dtype=np.int64)
    seed = np.asarray(all_seed, dtype=np.int64)

    out = os.path.join(os.path.dirname(__file__), "dataset_v2.npz")
    np.savez(out, X=X, y=y, seed=seed,
             feature_names=np.array(
                 NUMERIC_AGENT_KEYS + TAG_KEYS
                 + [f"intent={k}" for k in INTENT_KEYS]
             ))

    print(f"\nDataset: X={X.shape}, y={y.shape}")
    print(f"Total positive: {y.sum()} / {len(y)} ({y.mean()*100:.1f}%)")
    print(f"Unique seeds: {sorted(set(seed.tolist()))}")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
