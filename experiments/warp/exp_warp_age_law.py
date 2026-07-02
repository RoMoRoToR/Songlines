"""W2 — the warp-distance law: witness–traveler protocol (design §4, H-W3).

CSM includes foreign evidence with weight  trust · exp(−α·age) · conf
and inclusion threshold τ (merge_threshold).  The closed-form gate:

    age_max(trust) = (1/α) · ln(trust · conf / τ)     (conf = 0.95)

Predictions tested here:

  1. Gate curve (W2a): a traveler's query stops returning the witness's
     water at age_max ± 15%; fixed-K peer has NO gate (flat curve).
  2. Completion breakpoint (W2b): full navigation episodes.  The
     feasibility inequality  a₀ + travel_time(d) ≤ age_max  puts the
     breakpoint on the initial-age axis at ≈ age_max − travel_time(d).
  3. Trust-flip (design §4.3): trust ≤ τ/conf zeroes W* events entirely.
  4. Pruning cell (design §4.4): at K=2 the snapshot pruning horizon
     (10·K = 20 ticks) binds before the exponential gate at high trust.

Protocol: witness A observes water once and goes silent (its snapshot is
injected with a parameterised age and never refreshed); traveler B has
never seen the water — every lock it makes on it is a strict W*.
Everything is deterministic; no seeds needed.

Usage::

    PYTHONPATH=. .venv/bin/python experiments/warp/exp_warp_age_law.py
"""

from __future__ import annotations

import json
import math
import os
import sys
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from distributed_memory.agent_memory import AgentMemory
from experiments.collective_semantic_memory.csm_memory import (
    CSMMemory, WATER_TAG, _Snapshot,
)
from experiments.warp.warp_runner import run_warp_episode
from multiagent_env import MultiAgentGridWorld, WATER
from peer_memory.broadcast_bus import BroadcastBus
from peer_memory.peer_agent import PeerAgent

ALPHA = 0.05
TAU = 0.30
CONF = 0.95
TRUSTS = [1.0, 0.8, 0.6, 0.4, 0.25]
AGES = list(range(0, 37, 2))
AGE_STEP = 2
DISTANCES = [6, 12]
STEP_PAD = 15  # step_limit = travel distance + pad (too short to stumble)
OBS_RADIUS = 2

# Discrete-time refinement of the feasibility inequality (§4.2): the
# traveler sees the water itself once it is within OBS_RADIUS, and own
# evidence sustains the (already-frozen-phi) lock from there on.  So the
# foreign gate only has to survive the query at tick
#     t_gate = d − OBS_RADIUS − 1
# and the completion breakpoint on the initial-age axis is
#     bp(a0) :  a0 + t_gate ≤ age_max   →   bp = age_max − t_gate.


def predicted_age_max(trust: float) -> float:
    if trust * CONF <= TAU:
        return 0.0
    return math.log(trust * CONF / TAU) / ALPHA


# ────────────────────────────────────────── memory wrappers


class CSMWitnessMemory:
    """CSM with one silent witness snapshot of parameterised age.

    Trust updates are frozen: trust is the controlled independent
    variable of this protocol.
    """

    name = "csm"

    def __init__(self, *, initial_age: int, trust: float,
                 water_xy: Tuple[int, int], broadcast_every_k: int = 8):
        self.mem = CSMMemory(
            agent_ids=["traveler", "witness"],
            broadcast_every_k=broadcast_every_k,
        )
        self.mem._update_trust_from_top = lambda state, top_xy: None
        st = self.mem._states["traveler"]
        st.trust["witness"] = trust
        st.peer_snapshots.append(_Snapshot(
            sender="witness", tick=-initial_age,
            places={tuple(water_xy): {WATER_TAG: CONF}},
        ))

    def observe(self, agent_id, cells, tick):
        self.mem.observe(agent_id, cells, tick)

    def tick(self, tick_idx):
        self.mem.tick(tick_idx)

    def query(self, agent_id):
        return self.mem.query(agent_id)


class PeerWitnessMemory:
    """Fixed-K peer arm: witness snapshot cached in ``_last_known``.

    Driven through direct PeerAgent calls (no runtime — the design's
    'remove the runtime' pattern).  peer merge has no staleness term,
    so the gate is predicted NOT to close.
    """

    name = "peer"

    def __init__(self, *, initial_age: int, trust: float,
                 water_xy: Tuple[int, int]):
        from peer_memory.peer_types import BroadcastMessage

        bus = BroadcastBus()
        self.agent = PeerAgent("traveler", bus, env_id="w2",
                               consensus_radius=2.5)
        self.agent.trust._trust["witness"] = trust

        witness = AgentMemory("witness", role="scout", env_id="w2", trust=1.0)
        witness.observe(tuple(water_xy), {WATER_TAG: CONF},
                        episode_id=1, step_idx=0, confidence=CONF)
        witness.refresh_local()
        self.agent._last_known["witness"] = BroadcastMessage(
            sender_id="witness", sent_at_step=-initial_age,
            snapshot=witness.snapshot(),
        )
        self.rt = SimpleNamespace(
            agent=lambda aid: self.agent,
            all_agents=lambda: [self.agent],
            tick_count=0,
        )

    def observe(self, agent_id, cells, tick):
        for cell in cells:
            tag = cell["tag"]
            if tag in ("wall", "safe_neutral"):
                continue
            self.agent.observe(tuple(cell["xy"]), {tag: CONF},
                               episode_id=1, step_idx=tick, confidence=CONF)

    def tick(self, tick_idx):
        self.agent.refresh_local()
        self.agent.tick_step()
        self.agent.process_inbox_and_merge()
        self.rt.tick_count = self.agent.step_counter

    def query(self, agent_id):
        return [c.centroid_xy for c in
                self.agent.peer_query(WATER_TAG, top_k=5)
                if c.centroid_xy is not None]


# ────────────────────────────────────────── W2a: gate curve


def gate_open(arch: str, trust: float, age: int, k: int = 8) -> bool:
    """Does the traveler's query still expose the witness's water?"""
    water_xy = (9, 3)
    if arch == "csm":
        mem = CSMWitnessMemory(initial_age=age, trust=trust,
                               water_xy=water_xy, broadcast_every_k=k)
        mem.tick(0)
    else:
        mem = PeerWitnessMemory(initial_age=age, trust=trust,
                                water_xy=water_xy)
        mem.tick(0)
    targets = mem.query("traveler")
    return any(abs(t[0] - water_xy[0]) <= 0.6 and abs(t[1] - water_xy[1]) <= 0.6
               for t in targets)


def run_w2a() -> Dict[str, Any]:
    out: Dict[str, Any] = {"csm": {}, "peer": {}}
    for trust in TRUSTS:
        # empirical gate: largest age at which the candidate is still there
        open_ages = [a for a in range(0, 61) if gate_open("csm", trust, a)]
        emp = max(open_ages) if open_ages else -1
        pred = predicted_age_max(trust)
        rel_err = (abs(emp - pred) / pred) if pred > 0 else None
        # ages are sampled on an integer grid: the true crossing lies in
        # (emp, emp+1], so floor(pred) == emp is an exact hit.
        out["csm"][trust] = {
            "empirical_age_max": emp, "predicted_age_max": round(pred, 2),
            "rel_err": None if rel_err is None else round(rel_err, 3),
            "within_15pct": (
                (math.floor(pred) == emp or (rel_err is not None
                                             and rel_err <= 0.15))
                if pred > 0 else (emp == -1)),
        }
        peer_open = [gate_open("peer", trust, a) for a in (0, 20, 40, 60, 120)]
        out["peer"][trust] = {
            "open_at_ages_0_20_40_60_120": peer_open,
            "gate_never_closes": all(peer_open),
        }
    return out


# ────────────────────────────────────────── W2b: completion episodes


def run_traveler_episode(arch: str, trust: float, initial_age: int,
                         distance: int, k: int = 8) -> Dict[str, Any]:
    width = max(26, distance + 10)
    start = (1, 3)
    water_xy = (1 + distance, 3)
    env = MultiAgentGridWorld(width=width, height=7,
                              step_limit=distance + STEP_PAD,
                              observation_radius=2, rng_seed=0)
    env.set_cell(*water_xy, WATER)
    env.spawn("traveler", start_xy=start, target_tag=WATER_TAG, direction=0)

    if arch == "csm":
        memory = CSMWitnessMemory(initial_age=initial_age, trust=trust,
                                  water_xy=water_xy, broadcast_every_k=k)
    else:
        memory = PeerWitnessMemory(initial_age=initial_age, trust=trust,
                                   water_xy=water_xy)

    metrics, log = run_warp_episode(
        env, ["traveler"], [water_xy], memory,
        step_limit=distance + STEP_PAD, variant_tag=arch,
    )
    strict = [e for e in log.m_star_events() if e.w_star_strict]
    return {
        "arch": arch, "k": k, "trust": trust, "initial_age": initial_age,
        "distance": distance,
        "w_star_strict": len(strict) > 0,
        "completed": any(e.completed for e in strict),
        "t_succ": log.first_success_tick.get("traveler"),
    }


def run_w2b() -> List[Dict[str, Any]]:
    rows = []
    for trust in TRUSTS:
        for d in DISTANCES:
            for a0 in AGES:
                rows.append(run_traveler_episode("csm", trust, a0, d, k=8))
    # pruning cell (K=2, high trust): horizon 10·K = 20 binds before gate
    for a0 in AGES:
        rows.append(run_traveler_episode("csm", 1.0, a0, 6, k=2))
    # fixed-K peer arm: no gate expected
    for trust in (1.0, 0.6):
        for a0 in AGES:
            rows.append(run_traveler_episode("peer", trust, a0, 6))
    return rows


def breakpoint_table(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Empirical completion breakpoint per (arch, k, trust, d)."""
    table: Dict[str, Any] = {}
    keys = sorted({(r["arch"], r["k"], r["trust"], r["distance"])
                   for r in rows})
    for arch, k, trust, d in keys:
        cell = sorted([r for r in rows
                       if (r["arch"], r["k"], r["trust"], r["distance"])
                       == (arch, k, trust, d)],
                      key=lambda r: r["initial_age"])
        succ_ages = [r["initial_age"] for r in cell if r["completed"]]
        emp_bp = max(succ_ages) if succ_ages else -1
        # travel overhead measured on the freshest run of this cell
        base = next((r for r in cell if r["initial_age"] == 0), None)
        travel = (base["t_succ"] if base and base["t_succ"] is not None
                  else d)
        gate = predicted_age_max(trust)
        pred_bp: Optional[float] = None
        if arch == "csm":
            t_gate = max(0, d - OBS_RADIUS - 1)
            pred_bp = gate - t_gate
            if k <= 4:
                # pruning executes on broadcast ticks: the snapshot must
                # survive the last broadcast tick before t_gate.
                t_bcast = (t_gate // k) * k
                pred_bp = min(pred_bp, 10 * k - t_bcast)
            pred_bp = max(-1.0, pred_bp)
        entry = {
            "empirical_breakpoint_age": emp_bp,
            "predicted_breakpoint_age": (round(pred_bp, 1)
                                         if pred_bp is not None else None),
            "travel_time": travel,
            "n_strict_W": sum(1 for r in cell if r["w_star_strict"]),
            "monotone_step": all(
                r["completed"] >= r2["completed"]
                for r, r2 in zip(cell, cell[1:])),
        }
        if pred_bp is not None and gate > 0:
            # tolerance: the coarser of the age-sampling grid and 15% of
            # the gate (H-W3's stated band)
            tol = max(AGE_STEP, 0.15 * gate)
            entry["abs_err_ticks"] = abs(emp_bp - pred_bp)
            entry["within_15pct_of_gate"] = abs(emp_bp - pred_bp) <= tol
        table[f"{arch}-k{k}|trust={trust}|d={d}"] = entry
    return table


def main() -> None:
    out_dir = "tmp/warp/w2_age_law"
    os.makedirs(out_dir, exist_ok=True)

    print("W2a — gate curve (lock existence vs evidence age)")
    w2a = run_w2a()
    for trust, v in w2a["csm"].items():
        print(f"  csm trust={trust:<4} empirical={v['empirical_age_max']:>3} "
              f"predicted={v['predicted_age_max']:>6}  "
              f"within15%={v['within_15pct']}")
    for trust, v in w2a["peer"].items():
        print(f"  peer trust={trust:<4} gate_never_closes="
              f"{v['gate_never_closes']}")

    print("\nW2b — completion episodes (witness–traveler)")
    rows = run_w2b()
    table = breakpoint_table(rows)
    for key, v in table.items():
        print(f"  {key:<28} emp_bp={v['empirical_breakpoint_age']:>3} "
              f"pred_bp={v['predicted_breakpoint_age']} "
              f"travel={v['travel_time']} strictW={v['n_strict_W']} "
              f"monotone={v['monotone_step']} "
              f"ok15={v.get('within_15pct_of_gate', '-')}")

    # ── verdicts ──────────────────────────────────────────────────
    csm_gate_ok = all(v["within_15pct"] for v in w2a["csm"].values())
    peer_flat_ok = all(v["gate_never_closes"] for v in w2a["peer"].values())
    flip = w2a["csm"][0.25]
    trust_flip_ok = flip["empirical_age_max"] == -1
    csm_bp = {k: v for k, v in table.items()
              if k.startswith("csm") and v.get("predicted_breakpoint_age")
              is not None}
    bp_ok = all(v.get("within_15pct_of_gate", False) for v in csm_bp.values()
                if v["predicted_breakpoint_age"] >= 0)

    verdict = {
        "H_W3_gate_within_15pct": csm_gate_ok,
        "H_W3_peer_no_gate": peer_flat_ok,
        "trust_flip_zeroes_warp": trust_flip_ok,
        "completion_breakpoint_within_15pct": bp_ok,
    }
    with open(os.path.join(out_dir, "w2_results.json"), "w") as f:
        json.dump({"w2a": {str(k): v for k, v in w2a["csm"].items()},
                   "w2a_peer": {str(k): v for k, v in w2a["peer"].items()},
                   "w2b_rows": rows, "breakpoints": table,
                   "verdict": verdict}, f, indent=2)

    print("\n" + "=" * 60)
    for k, v in verdict.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print("=" * 60)
    print(f"Saved: {out_dir}/w2_results.json")


if __name__ == "__main__":
    main()
