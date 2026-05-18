"""End-to-end multi-agent navigation experiment.

Two agents must each find a water_source.  There are two clean water
cells on the grid.  The expected difference between field modes:

    descriptive (baseline)
        Both agents query memory independently.  They may BOTH pick the
        same water cell (the closer / fresher / higher-confidence one).
        Only one can reach it; the other has to find the second water
        cell late.  Expected: longer mean episode length, lower
        coordination.

    read_only
        Field reranking is applied, but no reservations.  Behavior
        similar to descriptive in this small scenario; reranking matters
        most when concept ranking is ambiguous, which is not the case
        with two clean water cells.

    coordinated
        Agent-A reserves its target water cell.  Agent-B's query then
        sees the reservation penalty and picks the OTHER water cell.
        Expected: each agent goes to a different water cell from step 1,
        episode terminates faster.

We run N seeds × 3 modes and compare:

  - success_rate (fraction of seeds where both agents reach water)
  - mean episode length (steps to terminate)
  - duplicate_target_rate (do both agents go to the same place initially?)
  - mean_hazard_hits

Usage::

    PYTHONPATH=. .venv/bin/python experiments/multiagent_navigation/exp_field_modes_comparison.py \\
        --n_seeds 10 --step_limit 60 --out_dir tmp/multiagent_nav_modes
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from songline_drive.belief_fusion import ConflictRuleSet, TemporalDecayEngine
from songline_drive.collective_field_types import FieldMode
from songline_drive.collective_memory import CollectiveMemory
from songline_drive.collective_types import AgentSignature
from songline_drive.concept_recall import ConceptRecallLayer
from songline_drive.field_adapter import FieldAdapter
from songline_drive.place_alignment import PlaceAlignmentEngine
from songline_drive.semantic_field import SemanticField

from multiagent_env import (
    BaselineRandomPlanner,
    CoordinatedFieldPlanner,
    GreedyMemoryPlanner,
    HAZARD,
    MultiAgentGridWorld,
    WATER,
    publish_observation_to_memory,
)

ENV_ID = "multiagent-grid-10x8"
WATER_TAG = "water_source"

# Two water cells, well separated; agents start in opposite corners
WATER_A: Tuple[int, int] = (2, 2)
WATER_B: Tuple[int, int] = (7, 5)
HAZARDS: List[Tuple[int, int]] = [(4, 4), (5, 2)]
START_A: Tuple[int, int] = (0, 0)
START_B: Tuple[int, int] = (9, 7)


# ─────────────────────────────────────────────────────── builders


def build_grid(seed: int) -> MultiAgentGridWorld:
    env = MultiAgentGridWorld(
        width=10, height=8, step_limit=80,
        observation_radius=8, rng_seed=seed,
    )
    env.set_cell(*WATER_A, WATER)
    env.set_cell(*WATER_B, WATER)
    for hx, hy in HAZARDS:
        env.set_cell(hx, hy, HAZARD)
    env.spawn("agent-A", start_xy=START_A, target_tag=WATER_TAG, direction=0)
    env.spawn("agent-B", start_xy=START_B, target_tag=WATER_TAG, direction=2)
    return env


def make_adapter(mode: str) -> Tuple[CollectiveMemory, FieldAdapter]:
    collective = CollectiveMemory(recency_lambda=0.97)
    collective.register_agent(AgentSignature("agent-A", role="agent", trust=1.0))
    collective.register_agent(AgentSignature("agent-B", role="agent", trust=1.0))

    engine = PlaceAlignmentEngine(
        semantic_threshold=0.45, spatial_radius=2.0,
        tag_match_bonus=0.45, min_confidence=0.05,
    )
    recall = ConceptRecallLayer(
        engine, only_dominant_tag=True, min_concept_support=1,
        decay_engine=TemporalDecayEngine(),
        conflict_rules=ConflictRuleSet.songlines_default(),
    )
    field = SemanticField(
        channels=[WATER_TAG, "hazard_edge", "safe_neutral"],
        mode=mode, lambda_decay=0.95, alpha_belief=0.60,
        eta_conflict=0.30, xi_occupancy=0.30, gamma_diffusion=0.10,
        diffusion_steps=1,
    )
    return collective, FieldAdapter(field, recall, field_weight=0.40, mode=mode)


# ─────────────────────────────────────────────────────── seed scout


def seed_memory_with_scout(
    env: MultiAgentGridWorld,
    collective: CollectiveMemory,
    seed: int,
) -> None:
    """Pre-populate memory with a few observations so planners have signal.

    A short "scout" pass: each agent observes its own current cell + neighbors
    for 3 steps of NOOP.  No actions, just observations.
    """
    rng = np.random.default_rng(seed)
    for step_i in range(3):
        for agent_id in env.agents:
            obs = env._observation(agent_id)
            publish_observation_to_memory(
                collective, agent_id, ENV_ID,
                episode_id=0, step_idx=step_i, obs=obs,
            )


# ─────────────────────────────────────────────────────── run one episode


def run_episode(
    mode: str,
    seed: int,
    planner_factory,
    *,
    step_limit: int,
) -> Dict[str, Any]:
    env = build_grid(seed)
    collective, adapter = make_adapter(mode)
    seed_memory_with_scout(env, collective, seed)
    adapter.refresh(collective)

    # Track each agent's chosen target (concept_id of top-1 at step 0)
    # before they start moving.  This lets us compute duplicate_target_rate.
    initial_targets: Dict[str, Optional[str]] = {}

    for agent_id in env.agents:
        items = adapter.field.top_k_for_channel(WATER_TAG, k=3)
        cid = items[0][0] if items else None
        initial_targets[agent_id] = cid
        # In coordinated mode, commit the reservation immediately so the
        # next agent's query sees the penalty.
        if mode == FieldMode.COORDINATED and cid is not None:
            adapter.commit_reservation(
                agent_id=agent_id, concept_id=cid,
                channel=WATER_TAG, duration=step_limit,
                current_seq=collective._next_seq,
            )

    planners = {aid: planner_factory(seed=seed * 100 + i)
                for i, aid in enumerate(env.agents.keys())}

    episode_id = 1
    step_idx = 0
    while True:
        actions = {}
        for aid in env.agents:
            actions[aid] = planners[aid].choose(
                env, aid, collective, adapter,
                current_seq=collective._next_seq,
                episode_id=episode_id, step_idx=step_idx,
            )
        result = env.step(actions)
        step_idx += 1

        # Publish observations to memory
        for aid, obs in result.obs.items():
            publish_observation_to_memory(
                collective, aid, ENV_ID,
                episode_id=episode_id, step_idx=step_idx, obs=obs,
            )

        # Periodically refresh memory (every 5 steps) so new observations
        # propagate to the field
        if step_idx % 5 == 0:
            adapter.refresh(collective)

        if result.done or step_idx >= step_limit:
            break

    duplicate_target = (
        initial_targets["agent-A"] is not None
        and initial_targets["agent-A"] == initial_targets["agent-B"]
    )

    n_succeeded = sum(1 for ag in env.agents.values() if ag.success)
    hazard_hits = sum(ag.n_hazard_hits for ag in env.agents.values())

    return {
        "mode": mode,
        "seed": seed,
        "all_succeeded": result.all_succeeded,
        "n_succeeded": n_succeeded,
        "episode_steps": step_idx,
        "duplicate_initial_target": duplicate_target,
        "initial_targets": initial_targets,
        "hazard_hits": hazard_hits,
        "agent_a_final_xy": (env.agents["agent-A"].x, env.agents["agent-A"].y),
        "agent_b_final_xy": (env.agents["agent-B"].x, env.agents["agent-B"].y),
        "agent_a_success": env.agents["agent-A"].success,
        "agent_b_success": env.agents["agent-B"].success,
    }


# ─────────────────────────────────────────────────────── aggregate


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {}
    return {
        "n_seeds": len(rows),
        "success_rate_both": (
            sum(1 for r in rows if r["all_succeeded"]) / len(rows)
        ),
        "success_rate_any": (
            sum(r["n_succeeded"] for r in rows) / (2 * len(rows))
        ),
        "mean_episode_steps": statistics.mean(r["episode_steps"] for r in rows),
        "stdev_episode_steps": (
            statistics.stdev(r["episode_steps"] for r in rows)
            if len(rows) > 1 else 0.0
        ),
        "duplicate_initial_target_rate": (
            sum(1 for r in rows if r["duplicate_initial_target"]) / len(rows)
        ),
        "mean_hazard_hits": statistics.mean(r["hazard_hits"] for r in rows),
    }


# ─────────────────────────────────────────────────────── main


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_seeds", type=int, default=10)
    parser.add_argument("--step_limit", type=int, default=60)
    parser.add_argument("--out_dir", default="tmp/multiagent_nav_modes")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    modes = [
        ("descriptive", FieldMode.DESCRIPTIVE, GreedyMemoryPlanner),
        ("read_only",   FieldMode.READ_ONLY,   GreedyMemoryPlanner),
        ("coordinated", FieldMode.COORDINATED, CoordinatedFieldPlanner),
    ]

    all_runs: List[Dict[str, Any]] = []
    aggregates: Dict[str, Any] = {}

    for label, mode, planner_cls in modes:
        rows: List[Dict[str, Any]] = []
        for seed in range(args.n_seeds):
            row = run_episode(
                mode, seed,
                planner_factory=lambda *, seed=seed, pcls=planner_cls: pcls(rng_seed=seed),
                step_limit=args.step_limit,
            )
            rows.append(row)
            all_runs.append(row)
        aggregates[label] = aggregate(rows)

    out_path = os.path.join(args.out_dir, "multiagent_nav_modes_summary.json")
    with open(out_path, "w") as f:
        json.dump({"aggregates": aggregates, "runs": all_runs}, f, indent=2)

    print(f"Multi-agent navigation comparison ({args.n_seeds} seeds, step_limit={args.step_limit})")
    print("=" * 75)
    header = (
        f"{'mode':<14}  {'success_both':>13}  {'mean_steps':>11}  "
        f"{'dup_target':>11}  {'hazard_hits':>12}"
    )
    print(header)
    print("-" * 75)
    for label, agg in aggregates.items():
        print(
            f"{label:<14}  {agg['success_rate_both']:>13.2f}  "
            f"{agg['mean_episode_steps']:>11.1f}  "
            f"{agg['duplicate_initial_target_rate']:>11.2f}  "
            f"{agg['mean_hazard_hits']:>12.2f}"
        )
    print(f"\nSummary → {out_path}")

    # Soft assertions (we don't crash on failure — just report)
    coord = aggregates.get("coordinated", {})
    base = aggregates.get("read_only", {})
    if coord and base:
        delta_steps = base["mean_episode_steps"] - coord["mean_episode_steps"]
        delta_dup = base["duplicate_initial_target_rate"] - coord["duplicate_initial_target_rate"]
        print(
            f"\nCoordinated vs read_only:  Δmean_steps={delta_steps:+.1f}  "
            f"Δdup_target_rate={delta_dup:+.2f}"
        )


if __name__ == "__main__":
    main()
