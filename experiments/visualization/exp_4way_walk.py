"""4-way visual comparison — memory-driven planner.

Three agents walk a 12x10 grid that contains three water cells.  Each
agent's task is to FIND water.  No agent has any a-priori knowledge of
where water is.  Movement is driven entirely by a memory-driven planner:

    1. If memory currently exposes any water target → head to the
       closest known target (deterministic A*-lite: nearest by manhattan,
       turn-then-forward navigation).
    2. Otherwise: explore by preferring unvisited adjacent cells.
    3. Last resort: deterministic RNG pick.

The planner is variant-agnostic — same code in all four runs.  The only
thing that changes between variants is **what the memory query returns**:

    independent   →   agent.local_query(water)   — only own observations
    shared        →   shared_graph.water_centroids — every agent's obs
    centralized   →   ConsensusReport.top_k(water) — central merge
    peer          →   agent.peer_view.top_k(water) — own peer-merged view

Because behaviour depends on memory, **trajectories now genuinely differ
across the four variants** — that is the whole point.  An agent in the
independent variant will wander until it personally stumbles into water;
an agent in the shared variant will beeline toward water as soon as
ANY agent has seen it; the peer variant lies in between, depending on
the broadcast cadence.

Usage::

    PYTHONPATH=. .venv/bin/python experiments/visualization/exp_4way_walk.py \\
        --n_ticks 36 --out_dir tmp/visualization_4way
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from songline_drive.belief_fusion import ConflictRuleSet, TemporalDecayEngine
from songline_drive.collective_memory import CollectiveMemory
from songline_drive.collective_types import AgentSignature
from songline_drive.concept_recall import ConceptRecallLayer
from songline_drive.place_alignment import PlaceAlignmentEngine

from distributed_memory import DistributedRuntime
from independent_memory import IndependentRuntime
from peer_memory import PeerRuntime

from multiagent_env import (
    EMPTY, FORWARD, HAZARD, NOOP, TURN_LEFT, TURN_RIGHT,
    MultiAgentGridWorld, WALL, WATER, publish_observation_to_memory,
)
from multiagent_env.grid_world import DIR_DELTAS


# ─────────────────────────────────────────────────────── scenario


ENV_ID = "viz-grid"
WATER_TAG = "water_source"

GRID_W, GRID_H = 12, 10

# Water cells positioned ASYMMETRICALLY relative to agent starts so that
# *every* agent benefits from collective memory, not just one.
#
# Reasoning:
#   - agent-A starts NW (0,0).  Its "prefer unvisited" exploration heads
#     east-south.  Water in NW corner area is HARD for A to discover solo
#     because A walks AWAY from it.
#   - agent-B starts NE (11,0) facing west.  Natural path goes west-south.
#     Water in NE corner is HARD for B.
#   - agent-C starts SW (0,9) facing north.  Natural path goes north-east.
#     Water in SW area is HARD for C.
#
# Each agent's "own" water is roughly in the direction OPPOSITE to its
# natural exploration.  Solo, every agent struggles.  With collective
# memory, whichever agent stumbles into ANY water immediately informs
# the other two — all three then redistribute efficiently.
WATER_CELLS = [
    (3, 8),    # SW area    — A naturally goes east, B naturally goes west: HARD for both
    (8, 7),    # S-centre   — far from every start, but a possible stumble for B going south
    (10, 2),   # NE area    — A might reach late, B is closest but starts facing AWAY
]

# Hazards positioned to discourage straight-line beelines through the centre.
HAZARD_CELLS = [(5, 4), (5, 5), (6, 4), (6, 6)]

# Three agents in three corners.  Starting directions chosen so each
# agent's "FORWARD" leads to its natural exploration direction.
AGENT_SPEC = [
    {"id": "agent-A", "start": (0, 0), "color": "#e74c3c"},   # face east
    {"id": "agent-B", "start": (11, 0), "color": "#3498db"},  # face west
    {"id": "agent-C", "start": (0, 9), "color": "#27ae60"},   # face north
]


# ─────────────────────────────────────────────────────── env builder


def build_env(seed: int = 0) -> MultiAgentGridWorld:
    env = MultiAgentGridWorld(
        width=GRID_W, height=GRID_H, step_limit=200,
        observation_radius=2, rng_seed=seed,
    )
    for x, y in WATER_CELLS:
        env.set_cell(x, y, WATER)
    for x, y in HAZARD_CELLS:
        env.set_cell(x, y, HAZARD)
    for spec in AGENT_SPEC:
        # Agent-B at (11, 0) starts facing west (dir=2) so it doesn't immediately bump.
        # Agent-C at (0, 9) starts facing north (dir=3) for the same reason.
        if spec["id"] == "agent-B":
            direction = 2
        elif spec["id"] == "agent-C":
            direction = 3
        else:
            direction = 0
        env.spawn(spec["id"], start_xy=spec["start"],
                  target_tag=WATER_TAG, direction=direction)
    return env


def make_alignment_engine() -> PlaceAlignmentEngine:
    return PlaceAlignmentEngine(
        semantic_threshold=0.45, spatial_radius=2.0,
        tag_match_bonus=0.45, min_confidence=0.05,
    )


# ─────────────────────────────────────────────────────── planner


def _stable_rng(agent_id: str, tick: int, variant: str) -> np.random.Generator:
    """RNG keyed by (agent, tick, variant) for reproducibility."""
    key = f"{agent_id}|{tick}|{variant}".encode()
    seed = int(hashlib.md5(key).hexdigest()[:8], 16)
    return np.random.default_rng(seed)


def _direction_toward(from_xy: Tuple[int, int], to_xy: Tuple[int, int]) -> int:
    dx = to_xy[0] - from_xy[0]
    dy = to_xy[1] - from_xy[1]
    if abs(dx) >= abs(dy):
        return 0 if dx > 0 else 2  # east or west
    return 1 if dy > 0 else 3       # south or north


def _can_step(env: MultiAgentGridWorld, ag, target_xy: Tuple[int, int]) -> bool:
    nx, ny = target_xy
    if nx < 0 or nx >= env.width or ny < 0 or ny >= env.height:
        return False
    if env.cell(nx, ny) == WALL:
        return False
    for other in env.agents.values():
        if other.agent_id != ag.agent_id and (other.x, other.y) == (nx, ny):
            return False
    return True


def _turn_or_forward(ag_direction: int, target_dir: int) -> int:
    if ag_direction == target_dir:
        return FORWARD
    diff = (target_dir - ag_direction) % 4
    if diff == 1:
        return TURN_RIGHT
    if diff == 3:
        return TURN_LEFT
    return TURN_LEFT  # 180° — arbitrary


@dataclass
class PlannerState:
    """Per-agent state the planner maintains independently for each variant."""

    agent_id: str
    visited: Set[Tuple[int, int]] = field(default_factory=set)
    # When the agent is locked onto a target it remembers it across ticks
    # so micro-fluctuations in the memory query (e.g., a centroid moving by
    # 0.1 due to a new observation) don't make the agent flip targets.
    locked_target: Optional[Tuple[int, int]] = None


def plan_action(
    state: PlannerState,
    env: MultiAgentGridWorld,
    memory_targets: List[Tuple[float, float]],
    tick: int,
    variant: str,
) -> int:
    """Universal memory-driven planner.  Returns one action int.

    All four variants run this exact function — the only thing that
    differs is the ``memory_targets`` list, which encodes "what the
    memory layer is telling this agent about water right now".
    """
    ag = env.agents[state.agent_id]
    state.visited.add((ag.x, ag.y))

    if ag.success:
        return NOOP

    # ── Tier 1: navigate toward a known water target ──────────────
    if memory_targets:
        # Skip targets that another successful agent is sitting on.
        # When a peer has claimed a water cell, we cannot stand on it
        # anyway (env blocks the move), so prefer the next-closest one.
        occupied_by_other = set()
        for other_aid, other in env.agents.items():
            if other_aid != state.agent_id and other.success:
                occupied_by_other.add((other.x, other.y))

        def dist(xy):
            return abs(xy[0] - ag.x) + abs(xy[1] - ag.y)

        # If our locked target became occupied, drop the lock
        if state.locked_target in occupied_by_other:
            state.locked_target = None

        candidates = sorted(memory_targets, key=dist)
        chosen: Optional[Tuple[int, int]] = None
        for cand in candidates:
            tx, ty = int(round(cand[0])), int(round(cand[1]))
            if (tx, ty) in occupied_by_other:
                continue
            chosen = (tx, ty)
            break

        if chosen is not None:
            tx, ty = chosen
            # Stick with the locked target unless the new is meaningfully closer
            if state.locked_target is not None and state.locked_target not in occupied_by_other:
                lock_d = abs(state.locked_target[0] - ag.x) + abs(state.locked_target[1] - ag.y)
                new_d = abs(tx - ag.x) + abs(ty - ag.y)
                if lock_d - new_d < 2:
                    tx, ty = state.locked_target
            state.locked_target = (tx, ty)

            if (tx, ty) == (ag.x, ag.y):
                return NOOP
            target_dir = _direction_toward((ag.x, ag.y), (tx, ty))
            if ag.direction == target_dir:
                dx, dy = DIR_DELTAS[target_dir]
                if _can_step(env, ag, (ag.x + dx, ag.y + dy)):
                    return FORWARD
                # Blocked — fall through to exploration
            else:
                return _turn_or_forward(ag.direction, target_dir)

    # ── Tier 2: exploration — prefer unvisited adjacent cells ─────
    state.locked_target = None
    # First check forward
    fwd_dir = ag.direction
    fwd_dx, fwd_dy = DIR_DELTAS[fwd_dir]
    fwd_xy = (ag.x + fwd_dx, ag.y + fwd_dy)
    if _can_step(env, ag, fwd_xy) and fwd_xy not in state.visited:
        return FORWARD

    # Then try turning to a direction with an unvisited reachable cell
    for try_dir in [(ag.direction + 1) % 4, (ag.direction + 3) % 4,
                    (ag.direction + 2) % 4]:
        dx, dy = DIR_DELTAS[try_dir]
        check_xy = (ag.x + dx, ag.y + dy)
        if _can_step(env, ag, check_xy) and check_xy not in state.visited:
            return _turn_or_forward(ag.direction, try_dir)

    # ── Tier 3: all neighbours visited or blocked — just move forward
    # if possible, otherwise random turn
    if _can_step(env, ag, fwd_xy):
        return FORWARD
    rng = _stable_rng(state.agent_id, tick, variant)
    return int(rng.integers(0, 2))  # turn left or right


# ─────────────────────────────────────────────────────── per-variant capture


@dataclass
class TickSnapshot:
    """What we capture from one variant at one tick."""

    tick: int
    agent_positions: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    agent_directions: Dict[str, int] = field(default_factory=dict)
    agent_success: Dict[str, bool] = field(default_factory=dict)
    # Per-agent: list of (x, y) concept centroids the agent currently "knows"
    known_concepts_per_agent: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)
    all_concept_centroids: List[Tuple[float, float]] = field(default_factory=list)


# ─────────────────────────────────────────────────────── variant: INDEPENDENT


def run_independent(n_ticks: int) -> List[TickSnapshot]:
    env = build_env()
    rt = IndependentRuntime(env_id=ENV_ID)
    for spec in AGENT_SPEC:
        rt.spawn_agent(spec["id"])
    planners = {spec["id"]: PlannerState(spec["id"]) for spec in AGENT_SPEC}

    snaps: List[TickSnapshot] = []
    for tick in range(n_ticks):
        # 1. Each agent observes its current neighbourhood and publishes to
        #    its OWN independent memory.
        for aid in env.agents:
            obs = env._observation(aid)
            for cell in obs.get("cells", []):
                tag = cell["tag"]
                if tag in ("wall", "safe_neutral"):
                    continue
                rt.observe(aid, cell["xy"], {tag: 0.95},
                           episode_id=1, step_idx=tick, confidence=0.95)
        rt.tick()  # local refresh, no cross-agent comms

        # 2. Capture state BEFORE acting
        snap = TickSnapshot(tick=tick)
        all_centroids = set()
        per_agent_targets: Dict[str, List[Tuple[float, float]]] = {}
        for aid, ag in env.agents.items():
            snap.agent_positions[aid] = (ag.x, ag.y)
            snap.agent_directions[aid] = ag.direction
            snap.agent_success[aid] = ag.success
            results = rt.local_query(aid, WATER_TAG, top_k=5)
            known = [r.centroid_xy for r in results if r.centroid_xy is not None]
            snap.known_concepts_per_agent[aid] = known
            per_agent_targets[aid] = known
            all_centroids.update(known)
        snap.all_concept_centroids = sorted(all_centroids)
        snaps.append(snap)

        # 3. Each agent plans + acts
        actions: Dict[str, int] = {}
        for aid in env.agents:
            actions[aid] = plan_action(
                planners[aid], env, per_agent_targets[aid], tick, "independent",
            )
        env.step(actions)
    return snaps


# ─────────────────────────────────────────────────────── variant: SHARED


def run_shared(n_ticks: int) -> List[TickSnapshot]:
    env = build_env()
    collective = CollectiveMemory(recency_lambda=0.97)
    for spec in AGENT_SPEC:
        collective.register_agent(AgentSignature(spec["id"], role="agent", trust=1.0))
    engine = make_alignment_engine()
    recall = ConceptRecallLayer(
        engine, only_dominant_tag=False, min_concept_support=1,
        decay_engine=TemporalDecayEngine(),
        conflict_rules=ConflictRuleSet.songlines_default(),
    )
    planners = {spec["id"]: PlannerState(spec["id"]) for spec in AGENT_SPEC}

    snaps: List[TickSnapshot] = []
    for tick in range(n_ticks):
        for aid in env.agents:
            obs = env._observation(aid)
            publish_observation_to_memory(collective, aid, ENV_ID,
                                          episode_id=1, step_idx=tick, obs=obs)
        graph = recall.refresh(collective)

        water_centroids = [
            node.centroid_xy
            for node in graph.concepts.values()
            if node.dominant_tag == WATER_TAG and node.centroid_xy is not None
        ]

        snap = TickSnapshot(tick=tick)
        for aid, ag in env.agents.items():
            snap.agent_positions[aid] = (ag.x, ag.y)
            snap.agent_directions[aid] = ag.direction
            snap.agent_success[aid] = ag.success
            snap.known_concepts_per_agent[aid] = list(water_centroids)
        snap.all_concept_centroids = list(water_centroids)
        snaps.append(snap)

        actions: Dict[str, int] = {}
        for aid in env.agents:
            actions[aid] = plan_action(
                planners[aid], env, water_centroids, tick, "shared",
            )
        env.step(actions)
    return snaps


# ─────────────────────────────────────────────────────── variant: CENTRALIZED


def run_centralized(n_ticks: int) -> List[TickSnapshot]:
    env = build_env()
    rt = DistributedRuntime(env_id=ENV_ID, consensus_radius=2.5)
    for spec in AGENT_SPEC:
        rt.spawn_agent(spec["id"])
    planners = {spec["id"]: PlannerState(spec["id"]) for spec in AGENT_SPEC}

    snaps: List[TickSnapshot] = []
    for tick in range(n_ticks):
        for aid in env.agents:
            obs = env._observation(aid)
            for cell in obs.get("cells", []):
                tag = cell["tag"]
                if tag in ("wall", "safe_neutral"):
                    continue
                rt.observe(aid, cell["xy"], {tag: 0.95},
                           episode_id=1, step_idx=tick, confidence=0.95)
        rt.tick()

        report = rt.last_report
        water_centroids = [
            c.centroid_xy for c in report.distributed_concepts
            if c.consensus_dominant_tag == WATER_TAG
        ]

        snap = TickSnapshot(tick=tick)
        for aid, ag in env.agents.items():
            snap.agent_positions[aid] = (ag.x, ag.y)
            snap.agent_directions[aid] = ag.direction
            snap.agent_success[aid] = ag.success
            snap.known_concepts_per_agent[aid] = list(water_centroids)
        snap.all_concept_centroids = list(water_centroids)
        snaps.append(snap)

        actions: Dict[str, int] = {}
        for aid in env.agents:
            actions[aid] = plan_action(
                planners[aid], env, water_centroids, tick, "centralized",
            )
        env.step(actions)
    return snaps


# ─────────────────────────────────────────────────────── variant: PEER


def run_peer(n_ticks: int, broadcast_every_k: int = 4) -> List[TickSnapshot]:
    env = build_env()
    rt = PeerRuntime(env_id=ENV_ID, broadcast_every_k=broadcast_every_k,
                     consensus_radius=2.5)
    for spec in AGENT_SPEC:
        rt.spawn_agent(spec["id"])
    planners = {spec["id"]: PlannerState(spec["id"]) for spec in AGENT_SPEC}

    snaps: List[TickSnapshot] = []
    for tick in range(n_ticks):
        for aid in env.agents:
            obs = env._observation(aid)
            for cell in obs.get("cells", []):
                tag = cell["tag"]
                if tag in ("wall", "safe_neutral"):
                    continue
                rt.observe(aid, cell["xy"], {tag: 0.95},
                           episode_id=1, step_idx=tick, confidence=0.95)
        rt.tick()

        snap = TickSnapshot(tick=tick)
        all_centroids = set()
        per_agent_targets: Dict[str, List[Tuple[float, float]]] = {}
        for aid, ag in env.agents.items():
            snap.agent_positions[aid] = (ag.x, ag.y)
            snap.agent_directions[aid] = ag.direction
            snap.agent_success[aid] = ag.success
            results = rt.peer_query(aid, WATER_TAG, top_k=5)
            known = [c.centroid_xy for c in results]
            snap.known_concepts_per_agent[aid] = known
            per_agent_targets[aid] = known
            all_centroids.update(known)
        snap.all_concept_centroids = sorted(all_centroids)
        snaps.append(snap)

        actions: Dict[str, int] = {}
        for aid in env.agents:
            actions[aid] = plan_action(
                planners[aid], env, per_agent_targets[aid], tick, "peer",
            )
        env.step(actions)
    return snaps


# ─────────────────────────────────────────────────────── rendering


def render_panel(
    ax,
    snap: TickSnapshot,
    title: str,
    trails: Dict[str, List[Tuple[int, int]]],
) -> None:
    ax.set_xlim(-0.5, GRID_W - 0.5)
    ax.set_ylim(GRID_H - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(GRID_W))
    ax.set_yticks(range(GRID_H))
    ax.tick_params(labelsize=6)
    ax.grid(True, color="lightgray", linewidth=0.3)

    for x, y in WATER_CELLS:
        ax.add_patch(mpatches.Rectangle(
            (x - 0.5, y - 0.5), 1, 1,
            facecolor="#aed6f1", edgecolor="#3498db", linewidth=0.8,
        ))
        ax.text(x, y, "W", ha="center", va="center", fontsize=9,
                color="#21618c", fontweight="bold")
    for x, y in HAZARD_CELLS:
        ax.add_patch(mpatches.Rectangle(
            (x - 0.5, y - 0.5), 1, 1,
            facecolor="#f5b7b1", edgecolor="#c0392b", linewidth=0.8,
        ))
        ax.text(x, y, "X", ha="center", va="center", fontsize=9,
                color="#922b21", fontweight="bold")

    for cx, cy in snap.all_concept_centroids:
        ax.add_patch(mpatches.Circle(
            (cx, cy), 0.18,
            facecolor="none", edgecolor="purple", linewidth=1.2,
        ))

    for spec in AGENT_SPEC:
        aid = spec["id"]
        color = spec["color"]
        path = trails.get(aid, [])
        if len(path) > 1:
            xs = [p[0] for p in path]
            ys = [p[1] for p in path]
            ax.plot(xs, ys, color=color, linewidth=1.5, alpha=0.45)

    arrow_dirs = {0: (0.30, 0), 1: (0, 0.30), 2: (-0.30, 0), 3: (0, -0.30)}
    for spec in AGENT_SPEC:
        aid = spec["id"]
        color = spec["color"]
        if aid not in snap.agent_positions:
            continue
        x, y = snap.agent_positions[aid]
        direction = snap.agent_directions.get(aid, 0)
        dx, dy = arrow_dirs[direction]
        is_success = snap.agent_success.get(aid, False)
        # Successful agents drawn as a star, in-progress agents as circle
        if is_success:
            ax.scatter([x], [y], s=240, c=color, marker="*", edgecolors="black",
                       linewidths=0.7, zorder=10)
        else:
            ax.add_patch(mpatches.Circle((x, y), 0.30, facecolor=color,
                                         edgecolor="black", linewidth=0.6, zorder=10))
            ax.arrow(x, y, dx, dy, head_width=0.18, head_length=0.10,
                     fc="white", ec="black", linewidth=0.4, zorder=11)
        ax.text(x, y - 0.50, aid[-1], ha="center", va="center",
                fontsize=6, color=color, fontweight="bold")

    for spec in AGENT_SPEC:
        aid = spec["id"]
        color = spec["color"]
        if aid not in snap.agent_positions:
            continue
        ax_xy = snap.agent_positions[aid]
        known = snap.known_concepts_per_agent.get(aid, [])
        for cx, cy in known:
            ax.plot([ax_xy[0], cx], [ax_xy[1], cy],
                    color=color, linewidth=0.5, alpha=0.45,
                    linestyle="--", zorder=5)

    n_succ = sum(1 for v in snap.agent_success.values() if v)
    ax.set_title(f"{title}  [success {n_succ}/3]", fontsize=10)


def render_frame(
    tick: int,
    snaps: Dict[str, TickSnapshot],
    trails: Dict[str, Dict[str, List[Tuple[int, int]]]],
    out_path: str,
) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(13, 11))
    plt.subplots_adjust(top=0.93, bottom=0.07, left=0.05, right=0.97,
                        hspace=0.28, wspace=0.15)
    ax_map = {
        "independent": axs[0, 0],
        "shared":      axs[0, 1],
        "centralized": axs[1, 0],
        "peer":        axs[1, 1],
    }
    titles = {
        "independent": "(1) independent — каждый знает только своё",
        "shared":      "(2 max) shared bus — один общий граф",
        "centralized": "(2 mid) centralized — ConsensusLayer аггрегирует",
        "peer":        "(3) peer — broadcast каждые K тиков",
    }
    for variant, snap in snaps.items():
        render_panel(ax_map[variant], snap, titles[variant], trails.get(variant, {}))

    counts = []
    for variant in ["independent", "shared", "centralized", "peer"]:
        snap = snaps[variant]
        n_known_each = [len(v) for v in snap.known_concepts_per_agent.values()]
        n_succ = sum(1 for v in snap.agent_success.values() if v)
        counts.append(f"{variant}={n_known_each} succ={n_succ}")

    fig.suptitle(
        f"4-way memory-driven planner — tick {tick:02d}\n"
        f"  per agent (concepts known, success):  {' | '.join(counts)}",
        fontsize=11, y=0.985,
    )

    legend_handles = [
        mpatches.Patch(facecolor="#aed6f1", edgecolor="#3498db",
                       label="water (W)"),
        mpatches.Patch(facecolor="#f5b7b1", edgecolor="#c0392b",
                       label="hazard (X)"),
        mpatches.Patch(facecolor="none", edgecolor="purple",
                       label="known concept"),
    ]
    for spec in AGENT_SPEC:
        legend_handles.append(
            mpatches.Patch(facecolor=spec["color"], edgecolor="black",
                           label=f"{spec['id']}")
        )
    legend_handles.append(
        plt.Line2D([0], [0], marker="*", color="black",
                   markerfacecolor="gold", markersize=11, linewidth=0,
                   label="reached water (success)")
    )
    legend_handles.append(
        plt.Line2D([0], [0], color="gray", linestyle="--", linewidth=0.7,
                   label="agent → known concept")
    )
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=len(legend_handles), fontsize=8,
               bbox_to_anchor=(0.5, 0.005), frameon=False)

    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────── main


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_ticks", type=int, default=40)
    parser.add_argument("--out_dir", default="tmp/visualization_4way")
    parser.add_argument("--broadcast_every_k", type=int, default=4)
    parser.add_argument("--gif", action="store_true", default=True)
    parser.add_argument("--frame_duration_ms", type=int, default=350)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    frames_dir = os.path.join(args.out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    print(f"Running 4 variants × {args.n_ticks} ticks ...")
    snaps_per_variant: Dict[str, List[TickSnapshot]] = {}
    snaps_per_variant["independent"] = run_independent(args.n_ticks)
    snaps_per_variant["shared"] = run_shared(args.n_ticks)
    snaps_per_variant["centralized"] = run_centralized(args.n_ticks)
    snaps_per_variant["peer"] = run_peer(args.n_ticks, args.broadcast_every_k)

    print("Rendering frames ...")
    trails_per_variant: Dict[str, Dict[str, List[Tuple[int, int]]]] = {
        v: {spec["id"]: [] for spec in AGENT_SPEC}
        for v in snaps_per_variant
    }
    frame_paths: List[str] = []
    for tick in range(args.n_ticks):
        snaps_now: Dict[str, TickSnapshot] = {}
        for variant, snaps in snaps_per_variant.items():
            snap = snaps[tick]
            snaps_now[variant] = snap
            for aid, xy in snap.agent_positions.items():
                trail = trails_per_variant[variant].setdefault(aid, [])
                if not trail or trail[-1] != xy:
                    trail.append(xy)
        out_png = os.path.join(frames_dir, f"frame_{tick:03d}.png")
        render_frame(tick, snaps_now, trails_per_variant, out_png)
        frame_paths.append(out_png)
        if tick % 5 == 0:
            print(f"  rendered tick {tick}/{args.n_ticks}")

    # Per-variant statistics
    stats: Dict[str, Any] = {}
    for variant, snaps in snaps_per_variant.items():
        last = snaps[-1]
        # When did each agent first succeed?
        first_success: Dict[str, Optional[int]] = {}
        for aid in last.agent_success:
            first_success[aid] = None
            for t, snap in enumerate(snaps):
                if snap.agent_success.get(aid):
                    first_success[aid] = t
                    break
        stats[variant] = {
            "n_succeeded_final": sum(1 for v in last.agent_success.values() if v),
            "first_success_tick": first_success,
            "final_knowledge": {
                aid: len(c) for aid, c in last.known_concepts_per_agent.items()
            },
            "total_unique_positions": {
                aid: len(set(trails_per_variant[variant][aid]))
                for aid in last.agent_positions
            },
        }

    summary = {
        "n_ticks": args.n_ticks,
        "broadcast_every_k": args.broadcast_every_k,
        "scenario": {
            "grid_size": [GRID_W, GRID_H],
            "water_cells": WATER_CELLS,
            "hazard_cells": HAZARD_CELLS,
            "agents": AGENT_SPEC,
        },
        "stats": stats,
        "frames_dir": frames_dir,
        "n_frames": len(frame_paths),
    }
    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    gif_path = os.path.join(args.out_dir, "4way_walk.gif")
    if args.gif:
        print(f"Combining {len(frame_paths)} frames into GIF ...")
        try:
            import imageio.v2 as imageio_v2
        except ImportError:
            import imageio as imageio_v2
        images = [imageio_v2.imread(p) for p in frame_paths]
        duration_s = args.frame_duration_ms / 1000.0
        try:
            imageio_v2.mimsave(gif_path, images, duration=duration_s, loop=0)
        except Exception:
            imageio_v2.mimsave(gif_path, images, duration=args.frame_duration_ms)

    print("=" * 75)
    print(f"✓ Visualization done")
    print(f"  Frames:  {frames_dir}  ({len(frame_paths)} PNGs)")
    if args.gif:
        print(f"  GIF:     {gif_path}")
    print(f"  Summary: {summary_path}")
    print()

    print("Per-variant stats:")
    print(f"{'variant':<14} {'n_succ':>7} {'first_success_tick':>30} {'n_known':>14}")
    print("-" * 75)
    for variant in ["independent", "shared", "centralized", "peer"]:
        s = stats[variant]
        fs = s["first_success_tick"]
        fs_str = ", ".join(f"{aid[-1]}={t}" for aid, t in fs.items())
        nk = s["final_knowledge"]
        nk_str = "  ".join(f"{aid[-1]}={v}" for aid, v in nk.items())
        print(f"{variant:<14} {s['n_succeeded_final']}/3{'':<3} "
              f"{fs_str:>30}  {nk_str:>14}")


if __name__ == "__main__":
    main()
