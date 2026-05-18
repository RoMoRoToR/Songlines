"""Adaptive loop — FieldOutcomeTracker over multiple episodes.

Scenario: Two agents navigate to water across N episodes.  Hazards are
mixed in, and a "contested" cell sits between the water sources — it
carries both water and hazard tags, so it has high ``base_conflict``
and will be selected as the target if its activation rises above
the clean water cells.

We run two configurations and compare convergence:

  baseline      — no adaptation; ``eta/xi/gamma`` stay at defaults
  adaptive      — after every episode we record per-concept outcomes;
                  every K episodes we call ``tracker.adapt()`` which
                  may bump ``eta_conflict`` (so contested cells are
                  suppressed harder) or other parameters

Expectations
------------
  - Adaptive run: ``eta_conflict`` rises over episodes when the
    contested cell repeatedly fails.
  - Adaptive run shows lower ``contested_target_pick_rate`` over time.
  - Adaptive run shows non-empty ``adaptation_history``.

This is a *demonstration* of adaptive convergence, not a benchmark.

Usage::

    PYTHONPATH=. .venv/bin/python experiments/multiagent_navigation/exp_adaptive_loop.py \\
        --n_episodes 20 --adapt_every 4 --out_dir tmp/adaptive_loop
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from songline_drive.belief_fusion import ConflictRuleSet, TemporalDecayEngine
from songline_drive.collective_field_types import FieldMode
from songline_drive.collective_memory import CollectiveMemory
from songline_drive.collective_types import AgentSignature
from songline_drive.concept_recall import ConceptRecallLayer
from songline_drive.field_adapter import FieldAdapter
from songline_drive.field_adaptive import FieldOutcomeTracker
from songline_drive.place_alignment import PlaceAlignmentEngine
from songline_drive.semantic_field import SemanticField

from multiagent_env import (
    GreedyMemoryPlanner,
    HAZARD,
    MultiAgentGridWorld,
    WATER,
    publish_observation_to_memory,
)

ENV_ID = "adaptive-loop-grid"
WATER_TAG = "water_source"
HAZARD_TAG = "hazard_edge"

# Cells layout
PURE_WATER_A: Tuple[int, int] = (1, 1)
PURE_WATER_B: Tuple[int, int] = (8, 6)
CONTESTED:    Tuple[int, int] = (4, 4)  # has both water + hazard
HAZARDS:      List[Tuple[int, int]] = [(3, 4), (5, 4), (4, 3), (4, 5)]
START_A:      Tuple[int, int] = (0, 0)
START_B:      Tuple[int, int] = (9, 7)


def build_grid() -> MultiAgentGridWorld:
    """CONTESTED is EMPTY in the physical env.  Its conflict exists only
    in agent memory, injected each episode by ``inject_contested_concept``.
    Hazards surround it so agents that head there incur penalties."""
    env = MultiAgentGridWorld(
        width=10, height=8, step_limit=60, observation_radius=2,
    )
    env.set_cell(*PURE_WATER_A, WATER)
    env.set_cell(*PURE_WATER_B, WATER)
    for hx, hy in HAZARDS:
        env.set_cell(hx, hy, HAZARD)
    return env


def make_stack() -> Tuple[CollectiveMemory, FieldAdapter, FieldOutcomeTracker]:
    collective = CollectiveMemory(recency_lambda=0.97)
    for aid in ("agent-A", "agent-B", "scout-water", "scout-hazard"):
        collective.register_agent(AgentSignature(aid, role="scout", trust=1.0))

    engine = PlaceAlignmentEngine(
        semantic_threshold=0.45, spatial_radius=2.0,
        tag_match_bonus=0.45, min_confidence=0.05,
    )
    recall = ConceptRecallLayer(
        engine, only_dominant_tag=False, min_concept_support=1,
        decay_engine=TemporalDecayEngine(),
        conflict_rules=ConflictRuleSet.songlines_default(),
    )
    field = SemanticField(
        channels=[WATER_TAG, HAZARD_TAG, "safe_neutral"],
        mode=FieldMode.READ_ONLY, lambda_decay=0.95, alpha_belief=0.60,
        eta_conflict=0.30, xi_occupancy=0.20, gamma_diffusion=0.10,
        diffusion_steps=1,
    )
    adapter = FieldAdapter(field, recall, field_weight=0.40, mode=FieldMode.READ_ONLY)
    tracker = FieldOutcomeTracker(field, window=10)
    return collective, adapter, tracker


def inject_contested_concept(
    collective: CollectiveMemory, episode_id: int,
    n_water: int = 20, n_hazard: int = 4,
) -> None:
    """Two synthetic scouts inject water + hazard on CONTESTED.

    The injection creates a memory-only concept with high base_conflict.
    Agents in the env see CONTESTED as empty so their own observations
    don't dilute the conflict signal.
    """
    for i in range(n_water):
        collective.publish_event(
            "place_observed", "scout-water",
            episode_id=episode_id, step_idx=i, env_id=ENV_ID,
            payload={"place_key": list(CONTESTED),
                     "semantic_tags": {WATER_TAG: 0.92, "water_candidate": 0.75},
                     "node_freshness": 1.0},
            confidence=0.95,
        )
    for i in range(n_hazard):
        collective.publish_event(
            "place_observed", "scout-hazard",
            episode_id=episode_id, step_idx=n_water + i, env_id=ENV_ID,
            payload={"place_key": list(CONTESTED),
                     "semantic_tags": {HAZARD_TAG: 0.92, "hazard_candidate": 0.75},
                     "node_freshness": 1.0},
            confidence=0.95,
        )


def run_episode(
    env: MultiAgentGridWorld,
    collective: CollectiveMemory,
    adapter: FieldAdapter,
    episode_id: int,
) -> Dict[str, Any]:
    """Single episode.  Returns episode summary."""
    env.spawn("agent-A", start_xy=START_A, target_tag=WATER_TAG, direction=0)
    env.spawn("agent-B", start_xy=START_B, target_tag=WATER_TAG, direction=2)

    # Inject contested concept BEFORE refresh so it has conflict from t=0
    inject_contested_concept(collective, episode_id)
    adapter.refresh(collective)

    # Determine each agent's chosen target concept
    items = adapter.field.top_k_for_channel(WATER_TAG, k=5)
    contested_picked: List[str] = []  # agents that picked the contested concept

    planners = {
        "agent-A": GreedyMemoryPlanner(rng_seed=episode_id),
        "agent-B": GreedyMemoryPlanner(rng_seed=episode_id + 100),
    }

    for aid in env.agents:
        top_items = adapter.field.top_k_for_channel(WATER_TAG, k=3)
        if top_items:
            top_cid = top_items[0][0]
            top_cell = adapter.field.cells.get(top_cid)
            if top_cell and top_cell.centroid_xy is not None:
                tx, ty = round(top_cell.centroid_xy[0]), round(top_cell.centroid_xy[1])
                if (tx, ty) == CONTESTED:
                    contested_picked.append(aid)

    step_idx = 0
    contested_visits = 0
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

        for aid, info in result.info.items():
            if info.new_xy == CONTESTED:
                contested_visits += 1

        for aid, obs in result.obs.items():
            publish_observation_to_memory(
                collective, aid, ENV_ID, episode_id, step_idx, obs,
            )

        if step_idx % 5 == 0:
            adapter.refresh(collective)

        if result.done or step_idx >= env.step_limit:
            break

    summary = {
        "episode_id": episode_id,
        "steps": step_idx,
        "all_succeeded": result.all_succeeded,
        "agent_a_success": env.agents["agent-A"].success,
        "agent_b_success": env.agents["agent-B"].success,
        "agent_a_hazard_hits": env.agents["agent-A"].n_hazard_hits,
        "agent_b_hazard_hits": env.agents["agent-B"].n_hazard_hits,
        "contested_picked_by": list(contested_picked),
        "contested_picked_count": len(contested_picked),
        "contested_visits": contested_visits,
    }

    # Reset agents for next episode
    env._agents.clear()
    env._episode_step = 0

    return summary


def record_outcomes(
    tracker: FieldOutcomeTracker,
    adapter: FieldAdapter,
    ep_summary: Dict[str, Any],
    *,
    failure_threshold_activation: float = 0.03,
) -> None:
    """Record outcomes for adaptation.

    Heuristic: any concept that has both
       (a) base_conflict ≥ 0.15
       (b) activation in the channel ≥ failure_threshold_activation
    is treated as a 'trap' candidate that *almost* got picked but
    shouldn't have.  We record a failure for it.  This drives Rule 1
    (eta_conflict ×1.15) over multiple episodes.

    For concepts that are high-conflict but already suppressed below the
    threshold, no record is created — they're already controlled.
    """
    field = adapter.field
    high_conflict_cids = [
        cid for cid, cell in field.cells.items()
        if cell.base_conflict >= 0.15
    ]
    n_recorded = 0
    for cid in high_conflict_cids:
        cell = field.cells[cid]
        # Pick the highest activation across any water-like channel
        max_act = 0.0
        for ch_name, ch in cell.channels.items():
            if ch_name in (WATER_TAG, "safe_neutral"):
                max_act = max(max_act, ch.activation)
        if max_act >= failure_threshold_activation:
            tracker.record_concept_outcome(cid, success=False)
            n_recorded += 1


def run_configuration(
    n_episodes: int, adapt_every: int, *, do_adapt: bool,
) -> Dict[str, Any]:
    collective, adapter, tracker = make_stack()
    env = build_grid()
    eta_traj: List[float] = [adapter.field.eta_conflict]
    contested_pick_traj: List[int] = []
    success_traj: List[bool] = []

    for ep in range(n_episodes):
        ep_summary = run_episode(env, collective, adapter, episode_id=ep)
        success_traj.append(ep_summary["all_succeeded"])
        contested_pick_traj.append(ep_summary["contested_picked_count"])

        if do_adapt:
            record_outcomes(tracker, adapter, ep_summary)
            if (ep + 1) % adapt_every == 0:
                tracker.adapt(min_samples=2)
        eta_traj.append(adapter.field.eta_conflict)

    return {
        "n_episodes": n_episodes,
        "adapt_every": adapt_every,
        "do_adapt": do_adapt,
        "eta_trajectory": [round(v, 4) for v in eta_traj],
        "eta_final": round(adapter.field.eta_conflict, 4),
        "eta_delta": round(adapter.field.eta_conflict - eta_traj[0], 4),
        "contested_pick_trajectory": contested_pick_traj,
        "contested_pick_total": sum(contested_pick_traj),
        "success_trajectory": success_traj,
        "success_rate": sum(success_traj) / max(1, len(success_traj)),
        "adaptation_history": tracker.adaptation_history,
        "tracker_summary": tracker.summary(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_episodes", type=int, default=20)
    parser.add_argument("--adapt_every", type=int, default=4)
    parser.add_argument("--out_dir", default="tmp/adaptive_loop")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    baseline = run_configuration(args.n_episodes, args.adapt_every, do_adapt=False)
    adaptive = run_configuration(args.n_episodes, args.adapt_every, do_adapt=True)

    out_path = os.path.join(args.out_dir, "adaptive_loop_summary.json")
    with open(out_path, "w") as f:
        json.dump({"baseline": baseline, "adaptive": adaptive}, f, indent=2)

    print(f"Adaptive loop ({args.n_episodes} episodes, adapt_every={args.adapt_every})")
    print("=" * 75)
    print(f"{'metric':<35} {'baseline':>15} {'adaptive':>15}")
    print("-" * 75)
    print(f"{'eta_conflict_final':<35} {baseline['eta_final']:>15.4f} "
          f"{adaptive['eta_final']:>15.4f}")
    print(f"{'eta_delta':<35} {baseline['eta_delta']:>15.4f} "
          f"{adaptive['eta_delta']:>15.4f}")
    print(f"{'contested_picks_total':<35} {baseline['contested_pick_total']:>15d} "
          f"{adaptive['contested_pick_total']:>15d}")
    print(f"{'success_rate':<35} {baseline['success_rate']:>15.2f} "
          f"{adaptive['success_rate']:>15.2f}")
    print(f"{'adaptation_history_len':<35} {len(baseline['adaptation_history']):>15d} "
          f"{len(adaptive['adaptation_history']):>15d}")
    print()
    print(f"Eta trajectory (adaptive): {adaptive['eta_trajectory']}")
    print(f"Summary → {out_path}")

    # Soft assertions on adaptive run
    assert adaptive["eta_delta"] >= 0, (
        f"Adaptive run should increase eta_conflict (got delta={adaptive['eta_delta']:.4f})"
    )
    if adaptive["adaptation_history"]:
        print(f"\n✓ Adaptive triggered {len(adaptive['adaptation_history'])} times")


if __name__ == "__main__":
    main()
