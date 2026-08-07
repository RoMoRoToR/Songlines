"""W2 hold-out v2 — parameter-disjoint registered predictions (H-W3).

Reviewer ask: the original hold-out (exp_warp_age_law_holdout.py) had 6
cells and varied only (K, trust, d).  This v2 tests the closed-form
admissibility/rupture boundary on a grid where EVERY continuous
parameter of the gate is at a value never used in any prior run:

    trusts  tau_i in {0.35, 0.55, 0.65, 0.85, 0.95}   (old: 1.0/.9/.8/.7/.6/.5/.4/.25)
    alpha        in {0.15, 0.25}                       (old: 0.05 only)
    tau (incl.)  in {0.20, 0.40}                       (old: 0.30 only)
    conf c       in {0.90, 0.99}                       (old: 0.95 only)
    d            in {5, 7, 11, 13, 17}                 (old: 6/8/9/12/15)
    K            in {2, 4, 8}
    delay L      in {0, 2, 5}                          (old: no delay arm)

NOTE on alpha: the registered design listed alpha in {0.05, 0.15, 0.25}
with the instruction to drop any value already used.  The prior alpha
was 0.05, so 0.05 is EXCLUDED.  With alpha >= 0.15 the pruning horizon
can never bind (max gate = ln(0.95*0.99/0.2)/0.15 = 10.3 < 20 = 10*K_min),
so the required "pruning beats gate" cells use a dedicated UNSEEN small
alpha = 0.03 instead.

Fixed model (identical to the one registered before the first hold-out;
no constant was re-fit):

    gate condition at a query:  trust * exp(-alpha*age) * conf >= tau
    age_max   = ln(trust*conf/tau) / alpha        (never if trust*conf <= tau)
    t_gate    = d - OBS_RADIUS - 1                (last foreign-only query)
    bp_gate   = age_max - (t_gate - L)
    bp_prune  = 10*K - t_bcast + L,  t_bcast = largest broadcast tick
                in [L, t_gate]                    (no bound if none exists)
    bp        = floor(min(bp_gate, bp_prune));  never if bp < 0
    strictness bound: L > t_gate  =>  never      (own observation of the
                water preempts the foreign lock, so no strict W* exists)

Message-delay arm: the witness snapshot is delivered to the traveler at
tick L; a0 parameterises the evidence age AT DELIVERY (snapshot tick
stamp = L - a0).  Registered prediction: both bounds shift UP by
exactly L (less aging time remains between delivery and the decisive
query), and L > t_gate zeroes strict W* entirely.

Protocol discipline: all predictions are computed from the formula and
fsynced to disk (predictions*.json, with a UTC timestamp) BEFORE the
first episode runs — separate code steps.  Empirical breakpoints are
measured on an age grid of step 1; a hit must be exact.

Usage::

    PYTHONPATH=. .venv/bin/python experiments/warp/exp_boundary_holdout_v2.py --smoke
    PYTHONPATH=. .venv/bin/python experiments/warp/exp_boundary_holdout_v2.py          # full grid

Mechanics are imported from exp_warp_age_law (untouched): same env
geometry, same warp runner, same witness-memory pattern generalised to
parameterised (alpha, tau, conf, delay).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from experiments.collective_semantic_memory.csm_memory import (
    CSMMemory, WATER_TAG, _Snapshot,
)
from experiments.warp.exp_warp_age_law import OBS_RADIUS, STEP_PAD
from experiments.warp.warp_runner import run_warp_episode
from multiagent_env import MultiAgentGridWorld, WATER

OUT_DIR_DEFAULT = "tmp/warp/boundary_holdout_v2"
HARD_CAP_AGE = 80          # absolute ceiling of the age scan
PRUNE_HORIZON_MULT = 10    # CSMMemory prunes snapshots older than 10*K

# ── parameter-disjoint grid (see module docstring for provenance) ────
TRUSTS = [0.35, 0.55, 0.65, 0.85, 0.95]
ALPHAS = [0.15, 0.25]
ALPHA_PRUNE = 0.03         # unseen small alpha: only value where pruning can bind
TAUS = [0.20, 0.40]
CONFS = [0.90, 0.99]
DS = [5, 7, 11, 13, 17]
KS = [2, 4, 8]

# Explicit never-cells: 4 trust-flip (trust*conf <= tau — only tr=0.35
# with tau=0.4 satisfies this on the disjoint grid) + 4 deep-negative
# gate cells (age_max < t_gate), spread over alpha/tau/conf/d.
NEVER_CELLS = [
    # (trust, alpha, tau, conf, d, K)
    (0.35, 0.15, 0.40, 0.90, 7, 2),    # trust flip
    (0.35, 0.25, 0.40, 0.90, 17, 4),   # trust flip
    (0.35, 0.15, 0.40, 0.99, 13, 8),   # trust flip (0.3465 <= 0.4)
    (0.35, 0.25, 0.40, 0.99, 5, 2),    # trust flip
    (0.95, 0.25, 0.20, 0.99, 17, 8),   # gate 6.2 < t_gate 14
    (0.55, 0.15, 0.40, 0.99, 11, 4),   # gate 2.1 < t_gate 8
    (0.65, 0.25, 0.20, 0.90, 13, 2),   # gate 4.3 < t_gate 10
    (0.85, 0.15, 0.40, 0.90, 17, 8),   # gate 4.3 < t_gate 14
]

# Dedicated pruning-binding cells: 10*K - t_bcast must undercut the gate.
PRUNING_CELLS = [
    # (trust, tau, conf, d, K)
    (0.95, 0.20, 0.99, 5, 2),    # gate 51.6, prune 18  -> prune binds
    (0.95, 0.20, 0.99, 13, 2),   # gate 51.6, prune 10  -> prune binds
    (0.85, 0.20, 0.90, 7, 4),    # gate 44.7, prune 36  -> prune binds
    (0.95, 0.20, 0.90, 17, 4),   # gate 48.4, prune 28  -> prune binds
]

# Delay arm bases; each is run at L in {0, 2, 5}.
DELAY_BASES = [
    # (trust, alpha, tau, conf, d, K)
    (0.95, 0.15, 0.20, 0.99, 11, 8),  # gate cell: bp 2 -> 4 -> 7
    (0.85, 0.15, 0.20, 0.90, 7, 8),   # L=5 > t_gate=4: strictness kills W*
    (0.65, 0.15, 0.20, 0.99, 13, 8),  # never-cell RESCUED by L=5
]


@dataclass(frozen=True)
class Cell:
    trust: float
    alpha: float
    tau: float
    conf: float
    d: int
    k: int
    delay: int = 0

    @property
    def name(self) -> str:
        return (f"a{self.alpha}|tau{self.tau}|c{self.conf}"
                f"|tr{self.trust}|d{self.d}|K{self.k}|L{self.delay}")


# Smoke subset: 10 cells spanning every prediction regime — gate-bound,
# trust-flip never, deep-negative never, pruning-bound, delay shift,
# delay strictness-kill, delay rescue-of-never.
SMOKE_CELLS = [
    Cell(0.35, 0.15, 0.20, 0.90, 5, 2, 0),    # gate, bp 1
    Cell(0.55, 0.15, 0.20, 0.99, 7, 4, 0),    # gate, bp 2
    Cell(0.95, 0.15, 0.20, 0.99, 11, 8, 0),   # gate, bp 2 (delay base)
    Cell(0.95, 0.15, 0.20, 0.99, 11, 8, 2),   # delay +2 -> bp 4
    Cell(0.95, 0.15, 0.20, 0.99, 11, 8, 5),   # delay +5 -> bp 7
    Cell(0.65, 0.15, 0.20, 0.99, 13, 8, 5),   # never rescued by delay -> bp 2
    Cell(0.55, 0.25, 0.20, 0.90, 13, 2, 0),   # never (gate < t_gate)
    Cell(0.35, 0.25, 0.40, 0.90, 17, 4, 0),   # never (trust flip: tr*c <= tau)
    Cell(0.95, ALPHA_PRUNE, 0.20, 0.99, 5, 2, 0),   # pruning binds, bp 18
    Cell(0.85, ALPHA_PRUNE, 0.20, 0.90, 7, 4, 0),   # pruning binds, bp 36
]


def build_cells() -> List[Cell]:
    """Deterministic selection by predicted regime (registered a priori).

    K does not enter the gate bound, and at alpha >= 0.15 the pruning
    horizon (>= 20) can never bind, so the 600-point factorial collapses
    to 54 distinct positive (trust, alpha, tau, conf, d) combos.  We
    take ALL of them, assigning K by deterministic rotation (a handful
    of combos also recur in SMOKE_CELLS with a different K — kept as
    explicit K-invariance probes).  Plus: 8 explicit never-cells,
    4 pruning-binding cells (unseen alpha=0.03), 3 delay bases
    x L in {0, 2, 5}.
    """
    cells: List[Cell] = list(SMOKE_CELLS)  # pinned: smoke must be a subset
    i = 0
    for alpha in ALPHAS:
        for tau in TAUS:
            for conf in CONFS:
                for trust in TRUSTS:
                    for d in DS:
                        cell = Cell(trust, alpha, tau, conf, d,
                                    KS[i % len(KS)], 0)
                        i += 1
                        if predict(cell)["predicted_bp"] >= 0:
                            cells.append(cell)
    for trust, alpha, tau, conf, d, k in NEVER_CELLS:
        cells.append(Cell(trust, alpha, tau, conf, d, k, 0))
    for trust, tau, conf, d, k in PRUNING_CELLS:
        cells.append(Cell(trust, ALPHA_PRUNE, tau, conf, d, k, 0))
    for trust, alpha, tau, conf, d, k in DELAY_BASES:
        for lag in (0, 2, 5):
            cells.append(Cell(trust, alpha, tau, conf, d, k, lag))
    # dedupe, preserving order
    seen, unique = set(), []
    for c in cells:
        if c.name not in seen:
            seen.add(c.name)
            unique.append(c)
    return unique


# ── step 1: registered predictions (pure formula, no episodes) ───────


def predict(cell: Cell) -> Dict[str, Any]:
    t_gate = max(0, cell.d - OBS_RADIUS - 1)
    entry: Dict[str, Any] = {"t_gate": t_gate}
    if cell.trust * cell.conf <= cell.tau:
        entry.update({"age_max": 0.0, "binding": "trust_flip",
                      "predicted_bp": -1})
        return entry
    age_max = math.log(cell.trust * cell.conf / cell.tau) / cell.alpha
    entry["age_max"] = round(age_max, 3)
    if cell.delay > t_gate:
        # own observation of the water precedes (or ties) delivery:
        # every lock is self-contaminated, no strict W* can exist.
        entry.update({"binding": "strictness_delay", "predicted_bp": -1})
        return entry
    bp_gate = age_max - (t_gate - cell.delay)
    entry["bp_gate"] = round(bp_gate, 3)
    bp = bp_gate
    binding = "gate"
    t_bcast = (t_gate // cell.k) * cell.k  # largest broadcast tick <= t_gate
    if t_bcast >= cell.delay:  # snapshot is present at that pruning event
        bp_prune = PRUNE_HORIZON_MULT * cell.k - t_bcast + cell.delay
        entry["bp_pruning"] = bp_prune
        if bp_prune < bp:
            bp, binding = bp_prune, "pruning"
    entry["binding"] = binding
    entry["predicted_bp"] = math.floor(bp) if bp >= 0 else -1
    if abs(bp_gate - round(bp_gate)) < 0.02:
        entry["grid_edge_warning"] = True  # boundary within 0.02 of integer
    return entry


def register_predictions(cells: List[Cell], path: str) -> Dict[str, Any]:
    predictions = {c.name: {**asdict(c), **predict(c)} for c in cells}
    payload = {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": ("bp = floor(min(ln(trust*conf/tau)/alpha - (t_gate - L), "
                  "10K - t_bcast + L)); t_gate = d - obs_radius - 1; "
                  "never if trust*conf <= tau, bp < 0, or L > t_gate"),
        "obs_radius": OBS_RADIUS,
        "n_cells": len(cells),
        "predictions": predictions,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    return predictions


# ── step 2: witness–traveler episodes ────────────────────────────────


class CSMWitnessMemoryV2:
    """Parameterised witness memory: alpha/tau/conf free, optional lag.

    Same pattern as exp_warp_age_law.CSMWitnessMemory: one silent
    witness snapshot, trust frozen (it is the controlled variable).
    With delay L the snapshot sits in transit and is appended to the
    traveler's peer_snapshots at tick L; its stamp is L - initial_age,
    so initial_age is the evidence age AT DELIVERY.
    """

    name = "csm"

    def __init__(self, cell: Cell, initial_age: int,
                 water_xy: Tuple[int, int]):
        self.mem = CSMMemory(
            agent_ids=["traveler", "witness"],
            broadcast_every_k=cell.k,
            staleness_alpha=cell.alpha,
            merge_threshold=cell.tau,
        )
        self.mem._update_trust_from_top = lambda state, top_xy: None
        st = self.mem._states["traveler"]
        st.trust["witness"] = cell.trust
        self._pending: Optional[_Snapshot] = _Snapshot(
            sender="witness", tick=cell.delay - initial_age,
            places={tuple(water_xy): {WATER_TAG: cell.conf}},
        )
        self._deliver_at = cell.delay

    def observe(self, agent_id, cells, tick):
        self.mem.observe(agent_id, cells, tick)

    def tick(self, tick_idx):
        if self._pending is not None and tick_idx >= self._deliver_at:
            self.mem._states["traveler"].peer_snapshots.append(self._pending)
            self._pending = None
        self.mem.tick(tick_idx)  # pruning on broadcast ticks sees the snapshot

    def query(self, agent_id):
        return self.mem.query(agent_id)


def run_traveler_episode_v2(cell: Cell, initial_age: int) -> Dict[str, Any]:
    """Identical geometry to exp_warp_age_law.run_traveler_episode."""
    width = max(26, cell.d + 10)
    start = (1, 3)
    water_xy = (1 + cell.d, 3)
    env = MultiAgentGridWorld(width=width, height=7,
                              step_limit=cell.d + STEP_PAD,
                              observation_radius=OBS_RADIUS, rng_seed=0)
    env.set_cell(*water_xy, WATER)
    env.spawn("traveler", start_xy=start, target_tag=WATER_TAG, direction=0)

    memory = CSMWitnessMemoryV2(cell, initial_age, water_xy)
    metrics, log = run_warp_episode(
        env, ["traveler"], [water_xy], memory,
        step_limit=cell.d + STEP_PAD, variant_tag="csm",
    )
    strict = [e for e in log.m_star_events() if e.w_star_strict]
    return {
        "initial_age": initial_age,
        "w_star_strict": len(strict) > 0,
        "completed": any(e.completed for e in strict),
        "t_succ": log.first_success_tick.get("traveler"),
    }


def measure_cell(cell: Cell, pred_bp: int) -> Dict[str, Any]:
    """Empirical strict-W completion breakpoint, age grid step 1.

    Scans at least to pred_bp + 6 and always 6 past the last observed
    success (so a miss upward is detected, not censored), hard cap 80.
    """
    max_age = max(6, pred_bp + 6)
    succ_ages: List[int] = []
    a0 = 0
    while a0 <= min(max_age, HARD_CAP_AGE):
        r = run_traveler_episode_v2(cell, a0)
        if r["completed"]:
            succ_ages.append(a0)
            max_age = max(max_age, a0 + 6)
        a0 += 1
    emp_bp = max(succ_ages) if succ_ages else -1
    return {
        "empirical_bp": emp_bp,
        "monotone_step": succ_ages == list(range(emp_bp + 1)),
        "censored_at_cap": bool(succ_ages) and emp_bp >= HARD_CAP_AGE,
        "n_episodes": a0,
    }


def classify(pred: int, emp: int) -> str:
    if emp == pred:
        return "exact"
    if abs(emp - pred) == 1:
        return "off_by_one"
    return "miss"


# ── driver ───────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--smoke", action="store_true",
                    help="10-cell local subset (all prediction regimes)")
    ap.add_argument("--out", default=OUT_DIR_DEFAULT)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    suffix = "_smoke" if args.smoke else ""
    pred_path = os.path.join(args.out, f"predictions{suffix}.json")
    res_path = os.path.join(args.out, f"results{suffix}.json")

    all_cells = build_cells()
    if args.smoke:
        by_name = {c.name: c for c in all_cells}
        missing = [c.name for c in SMOKE_CELLS if c.name not in by_name]
        if missing:
            raise SystemExit(f"smoke cells not in the registered design: "
                             f"{missing}")
        cells = list(SMOKE_CELLS)
    else:
        cells = all_cells

    # ── STEP 1: write predictions to disk BEFORE any episode ─────────
    predictions = register_predictions(cells, pred_path)
    n_never = sum(1 for p in predictions.values() if p["predicted_bp"] == -1)
    n_prune = sum(1 for p in predictions.values()
                  if p.get("binding") == "pruning")
    print(f"Registered {len(cells)} predictions -> {pred_path}")
    print(f"  never-cells: {n_never}   pruning-bound cells: {n_prune}   "
          f"delay cells: {sum(1 for c in cells if c.delay > 0)}")

    # ── STEP 2: run the witness–traveler protocol per cell ───────────
    print("\nRunning held-out episodes (age step 1) …")
    started_at = datetime.now(timezone.utc).isoformat()
    results: Dict[str, Any] = {}
    counts = {"exact": 0, "off_by_one": 0, "miss": 0}
    for idx, cell in enumerate(cells):
        pred = predictions[cell.name]
        meas = measure_cell(cell, pred["predicted_bp"])
        verdict = classify(pred["predicted_bp"], meas["empirical_bp"])
        counts[verdict] += 1
        results[cell.name] = {**pred, **meas, "verdict": verdict}
        print(f"  [{idx + 1:>2}/{len(cells)}] {cell.name:<46} "
              f"pred={pred['predicted_bp']:>3} emp={meas['empirical_bp']:>3} "
              f"{verdict.upper():<10} bind={pred['binding']:<16} "
              f"monotone={meas['monotone_step']}")

    # delay arm: empirical bp shift vs L against the registered shift
    delay_shift = {}
    for trust, alpha, tau, conf, d, k in DELAY_BASES:
        base = Cell(trust, alpha, tau, conf, d, k, 0).name
        if base not in results:
            continue
        for lag in (2, 5):
            name = Cell(trust, alpha, tau, conf, d, k, lag).name
            if name not in results:
                continue
            b_emp = results[base]["empirical_bp"]
            l_emp = results[name]["empirical_bp"]
            delay_shift[name] = {
                "L": lag,
                "empirical_shift": (l_emp - b_emp
                                    if -1 not in (b_emp, l_emp) else None),
                "base_emp": b_emp, "delayed_emp": l_emp,
                "predicted_shift": (lag if results[name]["predicted_bp"] != -1
                                    and results[base]["predicted_bp"] != -1
                                    else None),
            }

    summary = {
        "mode": "smoke" if args.smoke else "full",
        "predictions_registered_at": None,
        "episodes_started_at": started_at,
        "n_cells": len(cells),
        "counts": counts,
        "n_never_predicted": n_never,
        "n_pruning_bound": n_prune,
        "all_exact": counts["exact"] == len(cells),
        "misses": [name for name, r in results.items()
                   if r["verdict"] == "miss"],
    }
    with open(pred_path) as f:
        summary["predictions_registered_at"] = json.load(f)[
            "registered_at_utc"]
    with open(res_path, "w") as f:
        json.dump({"summary": summary, "delay_shift": delay_shift,
                   "results": results}, f, indent=2)

    print("\n" + "=" * 68)
    print(f"  cells={len(cells)}  exact={counts['exact']}  "
          f"off_by_one={counts['off_by_one']}  miss={counts['miss']}")
    for name, ds in delay_shift.items():
        print(f"  delay {name}: shift emp={ds['empirical_shift']} "
              f"pred={ds['predicted_shift']} (L={ds['L']})")
    if summary["misses"]:
        print("  MISSES (raw, reported as-is — do not re-fit):")
        for m in summary["misses"]:
            print(f"    {m}: pred={results[m]['predicted_bp']} "
                  f"emp={results[m]['empirical_bp']}")
    print("=" * 68)
    print(f"Saved: {pred_path}\n       {res_path}")


if __name__ == "__main__":
    main()
