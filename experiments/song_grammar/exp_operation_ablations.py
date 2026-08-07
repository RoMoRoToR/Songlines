"""E — Operation ablations: necessity of the five formation operations.

Reviewer claim addressed: "the five operations (MERGE, EXCEPTION,
NEW_SCHEMA, REPEAT, DROP) are a hand-designed taxonomy; prove their
necessity, especially EXCEPTION."

Design.  The two-axis decision matrix (utility x analogy) lives in
``songlines.analogy.decide`` and the mechanics in
``SonglineMemory.consider``.  We keep the DECISION function intact and
ablate one OPERATION at a time by rewriting the decided op to a
neighbouring one, then run the exact same candidate stream through
every arm.  Utility stays counterfactual and closed-loop: each arm's
marginal utility is computed against ITS OWN memory, so an arm that
corrupts its memory also mis-measures utility downstream — that is the
point, not a confound.

Candidate stream per world (extends U1's five-cell stream with extra
duplicates and a second novel world so that REPEAT-bloat and the
exception-heap are visible):

  1. new         song to water in world A, empty memory
  2-4. repeat    the same song three more times (marginal utility 0)
  5. novel-B     song to water in structurally different world B
  6. novel-C     song to water in structurally different world C
  7. irrelevant  song to a GOAL cell while intent is water
  8. conflict    world A' = A with the water secretly moved >= D_THR:
                 the stored A-schema now misleads; the new song is
                 useful AND structurally analogous (the EXCEPTION cell)

Arms (op-rewrite policies; everything else identical):

  full               control: the unmodified matrix.
  no_exception_merge EXCEPTION -> MERGE (last-write-wins assimilation
                     of decision-conflicting evidence).
  no_exception_drop  EXCEPTION -> DROP (conflicting evidence discarded).
  no_new_schema      NEW_SCHEMA -> EXCEPTION attached to the nearest
                     schema.  Bootstrap exemption: with EMPTY memory
                     there is no parent to attach to, so the very first
                     store is allowed through — otherwise the arm can
                     never store anything and is a trivial strawman.
  no_repeat          REPEAT -> NEW_SCHEMA (low-utility simple events
                     create fresh records instead of updating support).
  merge_or_drop      binary policy: any high-utility op -> MERGE onto
                     the nearest schema (bootstrap: MERGE into empty
                     memory creates the record); any low-utility op ->
                     DROP.

Registered predictions (written 2026-08-07, BEFORE any run; if an
ablation does NOT hurt, that is a finding against the necessity claim
and is reported as such):

  E.1 no_exception_merge: at least one decision-conflicting MERGE
      (analogy D >= D_THR) per world; the A-schema is silently
      overwritten by the A'-route, so first-try success on the ORIGINAL
      world A collapses and cost_A rises far above the full matrix
      (U1 measured 12.25 vs 62.6 for the last-write-wins baseline).
  E.2 no_exception_drop: world A stays first-try correct, but on the
      corrupted world A' the arm pays phantom-detour + blind sweep:
      cost_V well above full's cost_V; exception_count == 0 (evidence
      of the conflict is lost — cert carries no failure record).
  E.3 no_new_schema: schema_count stays at 1 (bootstrap only); novel
      worlds B and C are mis-filed as "exceptions" under the unrelated
      parent A, polluting A's certificate with >= 2 fake failure
      entries; exception_count grows to >= 3 vs 1 for full.  CAVEAT
      registered up front: under this insertion-order consumer the COST
      on B/C may not degrade (the songs are still in memory and still
      transport); if so we report the damage as structural (wrong
      parent links, corrupted trust/failure bookkeeping), not cost.
  E.4 no_repeat: memory grows by ~3 duplicate records vs full with NO
      cost improvement on any world (duplicates have zero marginal
      utility by construction); support statistics are destroyed
      (every record keeps support == 1, uncertainty stays 1.0).
  E.5 merge_or_drop: combination of corruption and loss — structurally
      unrelated worlds are last-write-wins-merged into a single slot;
      memory_items == 1 at the end, >= 2 cross-structure merges,
      first-try success on A collapses.
  E.6 full (control): reproduces U1 on the extended stream — ops ==
      [NEW_SCHEMA, REPEAT, REPEAT, REPEAT, NEW_SCHEMA, NEW_SCHEMA,
      DROP, EXCEPTION] in every valid world; memory_items == 4
      (A, B, C, exception); first-try correct on A; on A' the
      exception caps the cost below the no-exception arms.

v1 outcome (smoke, seeds 0-21, recorded 2026-08-07 AFTER the first
run; kept verbatim per registered discipline): E.1-E.5 PASSED as
registered.  E.6 FAILED on exactly one clause — cost_V(full)=20.8 was
NOT below cost_V(no_exception_merge)=15.2.  The falsified sub-claim
was a registration defect, not a model defect: last-write-wins is
first-try correct on the corrupted world ALONE (it holds only the A'
route), which it buys by destroying the original world (cost_A 52.2
vs 13.0, success_A 0.00 vs 1.00).  A single-world cost comparison can
therefore never establish the necessity of EXCEPTION; the necessity
claim is inherently two-world.  v2 registration (before the fresh-seed
cluster run; smoke seeds are burned for E.7 and the full run uses
disjoint seed blocks 400+/800+/1200+):

  E.6 (v2, control only): exact op sequence, memory_items == 4,
      first-try success on A, and cost_V(full) < cost_V(
      no_exception_drop).  The merge-arm comparison moves to E.7.
  E.7 (exception two-world necessity): with regret(arm, w) =
      cost(arm, w) - min over arms of cost(w), every arm LACKING the
      exception mechanism (no_exception_merge, no_exception_drop,
      merge_or_drop) has max-regret over {A, V} strictly greater than
      full's in every world; the arms that keep EXCEPTION
      (no_new_schema, no_repeat) match full's costs exactly.  I.e.
      only exception-bearing memories sit at the two-world Pareto
      point: correct on the original world AND cheap on the corrupted
      one.

Metrics per arm and world: consumer cost + first-try success on the
ORIGINAL world A, the CORRUPTED world A' (V), and the novel worlds
B/C; memory items, schema count, exception count; corruption events
(MERGE applied where the incoming song's end-displacement conflicts,
ana.D >= D_THR — i.e. a merged schema now sings to the wrong target)
and cross-structure merges (MERGE where ana.share < SHARE_THR);
certificate failure entries.  Plus the formal decision table of the
full matrix, enumerated FROM THE CODE (``decide``) over the mutually
exclusive condition cells, written as markdown.

Usage::

    PYTHONPATH=. .venv/bin/python \
        experiments/song_grammar/exp_operation_ablations.py \
        --n-worlds 4 --out tmp/operation_ablations_smoke
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from experiments.song_grammar.exp_s0_song_smoke import (
    TRAVELER_START, bfs_path)
from experiments.song_grammar.exp_u1_ucsm_smoke import (
    D_THR, SHARE_THR, U_THR, blind_cost, build_goal_world, build_variant,
    consumer_cost, utility, valid_transport, witness_song)
from experiments.song_grammar.ucsm import (
    Certificate, Schema, SonglineMemory, analogy, decide, nearest)
from experiments.warp.exp_warp_landmark_ablation import build_world
from multiagent_env import WALL

GridXY = Tuple[int, int]
DEFAULT_OUT = "tmp/operation_ablations_smoke"
Policy = Callable[[str, Optional[int], Optional[Dict[str, float]]], str]


# ── ablated memory: same mechanics, op rewritten by the arm ────────

class AblatedMemory(SonglineMemory):
    """SonglineMemory whose decided op is rewritten by an arm policy.

    ``decide`` is untouched; only the OPERATION applied differs.  The
    apply-mechanics below mirror ``SonglineMemory.consider`` exactly
    (songlines/analogy.py) so the control arm is bit-identical."""

    def __init__(self, policy: Policy, u_thr: float, share_thr: float,
                 d_thr: float):
        super().__init__(u_thr, share_thr, d_thr)
        self.policy = policy
        self.ops_orig: List[str] = []
        self.corrupt_merges = 0        # MERGE with ana.D >= d_thr
        self.cross_merges = 0          # MERGE with ana.share < share_thr

    def consider(self, cand, utility: float, episode_id: str,
                 conditions: Dict[str, Any]) -> str:
        idx, ana = nearest(cand, self.schemas)
        op = decide(utility, ana, self.u_thr, self.share_thr, self.d_thr)
        self.ops_orig.append(op)
        final = self.policy(op, idx, ana)
        if final == "REPEAT" and idx is not None:
            self.schemas[idx].support += 1
            self.schemas[idx].cert.uncertainty = \
                1.0 / self.schemas[idx].support
            self.schemas[idx].cert.evidence.append(episode_id)
        elif final == "MERGE" and idx is not None:
            if ana is not None and ana["D"] >= self.d_thr:
                self.corrupt_merges += 1
            if ana is not None and ana["share"] < self.share_thr:
                self.cross_merges += 1
            s = self.schemas[idx]
            s.song = cand                      # last write wins
            s.support += 1
            s.cert.delta_v = max(s.cert.delta_v, utility)
            s.cert.evidence.append(episode_id)
        elif final == "NEW_SCHEMA":
            self.schemas.append(Schema(cand, Certificate(
                conditions, utility, 1.0, [episode_id])))
        elif final == "EXCEPTION" and idx is not None:
            self.schemas[idx].cert.failures.append(
                {"episode": episode_id, "distortion": ana["D"]})
            self.schemas.append(Schema(cand, Certificate(
                conditions, utility, 1.0, [episode_id]),
                kind="exception", parent=idx))
        self.log.append({"episode": episode_id, "op": final,
                         "op_matrix": op, "utility": round(utility, 2)})
        return final


# ── arm policies (op rewrites) ─────────────────────────────────────

def pol_full(op, idx, ana):
    return op


def pol_no_exception_merge(op, idx, ana):
    return "MERGE" if op == "EXCEPTION" else op


def pol_no_exception_drop(op, idx, ana):
    return "DROP" if op == "EXCEPTION" else op


def pol_no_new_schema(op, idx, ana):
    # bootstrap exemption: an exception needs a parent (see header)
    if op == "NEW_SCHEMA" and idx is not None:
        return "EXCEPTION"
    return op


def pol_no_repeat(op, idx, ana):
    return "NEW_SCHEMA" if op == "REPEAT" else op


def pol_merge_or_drop(op, idx, ana):
    if op in ("MERGE", "NEW_SCHEMA", "EXCEPTION"):
        # MERGE into empty memory is defined as creating the record;
        # otherwise the arm could never store anything (strawman)
        return "MERGE" if idx is not None else "NEW_SCHEMA"
    return "DROP"


ARMS: Dict[str, Policy] = {
    "full": pol_full,
    "no_exception_merge": pol_no_exception_merge,
    "no_exception_drop": pol_no_exception_drop,
    "no_new_schema": pol_no_new_schema,
    "no_repeat": pol_no_repeat,
    "merge_or_drop": pol_merge_or_drop,
}

FULL_OPS_EXPECTED = ["NEW_SCHEMA", "REPEAT", "REPEAT", "REPEAT",
                     "NEW_SCHEMA", "NEW_SCHEMA", "DROP", "EXCEPTION"]


# ── decision table, enumerated from decide() itself ────────────────

def decision_table() -> Tuple[List[Dict[str, str]], str]:
    """Enumerate decide() over the mutually exclusive condition cells
    (2 utility levels x 4 analogy classes = 8 cells, exhaustive by
    construction: share>=thr is binary, D>=thr only defined on top of
    it, ana=None is the empty-memory case)."""
    ana_cells = [
        ("none (empty memory / no schema)", "ana is None", None),
        ("complex (share < SHARE_THR)", "share < share_thr",
         {"L": 9, "share": SHARE_THR - 0.05, "D": 0}),
        ("simple, decision-consistent",
         "share >= share_thr and D < d_thr",
         {"L": 1, "share": 1.0, "D": 0}),
        ("simple, decision-CONFLICTING",
         "share >= share_thr and D >= d_thr",
         {"L": 1, "share": 1.0, "D": D_THR}),
    ]
    rows = []
    for u_name, u in (("high (U >= U_THR)", U_THR),
                      ("low (U < U_THR)", U_THR - 1.0)):
        for a_name, cond, ana in ana_cells:
            op = decide(u, ana, U_THR, SHARE_THR, D_THR)
            rows.append({"utility": u_name, "analogy": a_name,
                         "condition": cond, "operation": op})
    md = ["| utility | analogy | condition (from `decide`) | operation |",
          "|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['utility']} | {r['analogy']} "
                  f"| `{r['condition']}` | **{r['operation']}** |")
    md.append("")
    md.append("Note (from code): for low utility the conflict predicate "
              "is never evaluated — a low-utility, simple, conflicting "
              "candidate is a REPEAT, not an EXCEPTION; the exception "
              "mechanism is reserved for evidence that is worth acting "
              "on. `conflict` implies `simple` by construction, so "
              "'complex + conflicting' is not a reachable cell.")
    return rows, "\n".join(md)


# ── world construction (U1 machinery + a second novel world) ───────

def find_novel_worlds(seed: int, n: int):
    found = []
    for k in range(100, 260):
        env_b, water_b = build_world(seed + k)
        if (env_b.cell(*TRAVELER_START) == WALL
                or not bfs_path(env_b, TRAVELER_START, water_b)):
            continue
        song_b = witness_song(env_b, water_b)
        if not valid_transport(env_b, song_b, water_b):
            continue
        found.append((env_b, song_b))
        if len(found) == n:
            return found
    return found


def build_stream(seed: int):
    """Same construction-validity discipline as U1 (fail-closed), plus:
    the novel songs must be structurally complex w.r.t. A and each
    other (else the NOVEL cells would not exercise NEW_SCHEMA), and
    the conflict song must be simple+conflicting w.r.t. A (else the
    CONFLICT cell would not exercise EXCEPTION)."""
    env_a, water_a = build_world(seed)
    env_g, goal = build_goal_world(seed + 200)
    env_v, water_v, _old = build_variant(seed)
    if env_g is None or env_v is None:
        return None
    song_a = witness_song(env_a, water_a)
    song_g = witness_song(env_g, goal)
    song_v = witness_song(env_v, water_v)
    novel = find_novel_worlds(seed, 2)
    if len(novel) < 2:
        return None
    (env_b, song_b), (env_c, song_c) = novel
    if not (env_a.cell(*TRAVELER_START) != WALL
            and bfs_path(env_a, TRAVELER_START, water_a)
            and bfs_path(env_v, TRAVELER_START, water_v)
            and valid_transport(env_a, song_a, water_a)
            and valid_transport(env_v, song_v, water_v)
            and valid_transport(env_g, song_g, goal)
            and utility(env_g, [], song_g) < U_THR):
        return None
    ana_v = analogy(song_v, song_a)
    if not (ana_v["share"] >= SHARE_THR and ana_v["D"] >= D_THR):
        return None                       # conflict cell not exercised
    for x, y in ((song_b, song_a), (song_c, song_a), (song_c, song_b)):
        if analogy(x, y)["share"] >= SHARE_THR:
            return None                   # novel cell not exercised
    stream = [
        (f"s{seed}-new", env_a, song_a),
        (f"s{seed}-repeat1", env_a, song_a),
        (f"s{seed}-repeat2", env_a, song_a),
        (f"s{seed}-repeat3", env_a, song_a),
        (f"s{seed}-novelB", env_b, song_b),
        (f"s{seed}-novelC", env_c, song_c),
        (f"s{seed}-irrelevant", env_g, song_g),
        (f"s{seed}-conflict", env_v, song_v),
    ]
    eval_envs = {"A": env_a, "V": env_v, "B": env_b, "C": env_c}
    return stream, eval_envs


# ── run one arm on one world ───────────────────────────────────────

def run_arm(policy: Policy, stream) -> AblatedMemory:
    mem = AblatedMemory(policy, U_THR, SHARE_THR, D_THR)
    for ep_id, env, cand in stream:
        u = utility(env, mem.ordered(), cand)   # closed loop: own memory
        mem.consider(cand, u, ep_id, {"n_couplets": len(cand)})
    return mem


def eval_arm(mem: AblatedMemory, eval_envs) -> Dict[str, Any]:
    worlds = {}
    for name, env in eval_envs.items():
        blind = blind_cost(env, TRAVELER_START)
        cc = consumer_cost(env, mem.ordered())
        worlds[name] = {
            "cost": cc["cost"], "blind": blind,
            "phantom_first": cc["phantom_first"],
            "first_try_success": (not cc["phantom_first"])
            and cc["cost"] < blind}
    schemas = mem.ordered()
    return {
        "ops": [e["op"] for e in mem.log],
        "ops_matrix": mem.ops_orig,
        "worlds": worlds,
        "memory_items": len(schemas),
        "schema_count": sum(1 for s in schemas if s.kind == "schema"),
        "exception_count": sum(1 for s in schemas
                               if s.kind == "exception"),
        "corrupt_merges": mem.corrupt_merges,
        "cross_merges": mem.cross_merges,
        "cert_failure_entries": sum(len(s.cert.failures)
                                    for s in schemas if s.cert),
        "max_support": max((s.support for s in schemas), default=0),
    }


def run_world(seed: int) -> Optional[Dict[str, Any]]:
    built = build_stream(seed)
    if built is None:
        return None
    stream, eval_envs = built
    row: Dict[str, Any] = {"seed": seed, "arms": {}}
    for name, pol in ARMS.items():
        mem = run_arm(pol, stream)
        row["arms"][name] = eval_arm(mem, eval_envs)
    return row


# ── aggregation + registered verdicts ──────────────────────────────

def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def aggregate(rows) -> Dict[str, Any]:
    agg: Dict[str, Any] = {}
    for arm in ARMS:
        a = [r["arms"][arm] for r in rows]
        agg[arm] = {
            "mean_cost": {w: round(mean([x["worlds"][w]["cost"]
                                         for x in a]), 1)
                          for w in ("A", "V", "B", "C")},
            "success_rate": {w: mean([x["worlds"][w]["first_try_success"]
                                      for x in a])
                             for w in ("A", "V", "B", "C")},
            "mean_memory_items": mean([x["memory_items"] for x in a]),
            "mean_schema_count": mean([x["schema_count"] for x in a]),
            "mean_exception_count": mean([x["exception_count"]
                                          for x in a]),
            "mean_corrupt_merges": mean([x["corrupt_merges"] for x in a]),
            "mean_cross_merges": mean([x["cross_merges"] for x in a]),
            "mean_cert_failure_entries": mean(
                [x["cert_failure_entries"] for x in a]),
        }
    return agg


def verdicts(rows, agg) -> Dict[str, bool]:
    full, nem, ned = agg["full"], agg["no_exception_merge"], \
        agg["no_exception_drop"]
    nns, nrp, mod = agg["no_new_schema"], agg["no_repeat"], \
        agg["merge_or_drop"]
    e1 = (nem["success_rate"]["A"] < full["success_rate"]["A"]
          and nem["mean_cost"]["A"] > full["mean_cost"]["A"]
          and all(r["arms"]["no_exception_merge"]["corrupt_merges"] >= 1
                  for r in rows))
    e2 = (ned["mean_cost"]["V"] > full["mean_cost"]["V"]
          and ned["success_rate"]["A"] == full["success_rate"]["A"]
          and ned["mean_exception_count"] == 0)
    e3 = (nns["mean_exception_count"] >= 3
          and nns["mean_schema_count"] == 1
          and nns["mean_cert_failure_entries"]
          >= full["mean_cert_failure_entries"] + 2)
    e4 = (nrp["mean_memory_items"] >= full["mean_memory_items"] + 3
          and all(r["arms"]["no_repeat"]["worlds"][w]["cost"]
                  == r["arms"]["full"]["worlds"][w]["cost"]
                  for r in rows for w in ("A", "V", "B", "C"))
          and all(r["arms"]["no_repeat"]["max_support"] == 1
                  for r in rows))
    e5 = (all(r["arms"]["merge_or_drop"]["memory_items"] == 1
              for r in rows)
          and mod["mean_cross_merges"] >= 2
          and mod["success_rate"]["A"] < full["success_rate"]["A"])
    e6 = (all(r["arms"]["full"]["ops"] == FULL_OPS_EXPECTED
              for r in rows)
          and all(r["arms"]["full"]["memory_items"] == 4 for r in rows)
          and full["success_rate"]["A"] == 1.0
          and full["mean_cost"]["V"] < ned["mean_cost"]["V"])
    # E.7 (v2): per-world max-regret over {A, V}
    no_exc_arms = ("no_exception_merge", "no_exception_drop",
                   "merge_or_drop")
    exc_arms = ("no_new_schema", "no_repeat")
    e7 = True
    for r in rows:
        best = {w: min(r["arms"][a]["worlds"][w]["cost"] for a in ARMS)
                for w in ("A", "V")}
        regret = {a: max(r["arms"][a]["worlds"][w]["cost"] - best[w]
                         for w in ("A", "V")) for a in ARMS}
        if not all(regret[a] > regret["full"] for a in no_exc_arms):
            e7 = False
        if not all(r["arms"][a]["worlds"][w]["cost"]
                   == r["arms"]["full"]["worlds"][w]["cost"]
                   for a in exc_arms for w in ("A", "V")):
            e7 = False
    return {
        "E.1_no_exception_merge_corrupts_A": e1,
        "E.2_no_exception_drop_loses_evidence_on_V": e2,
        "E.3_no_new_schema_structural_damage": e3,
        "E.4_no_repeat_bloat_without_benefit": e4,
        "E.5_merge_or_drop_corruption_plus_loss": e5,
        "E.6_full_matrix_control_v2": e6,
        "E.7_exception_two_world_necessity_v2": e7,
    }


# ── main ───────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-worlds", type=int, default=4)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--seed-scan", type=int, default=80,
                    help="how many base seeds to scan for valid worlds")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # registered BEFORE evaluating (mirrors the module docstring)
    with open(os.path.join(args.out, "e_registered.json"), "w") as f:
        json.dump({
            "constants": {"U_THR": U_THR, "SHARE_THR": SHARE_THR,
                          "D_THR": D_THR},
            "arms": list(ARMS),
            "full_ops_expected": FULL_OPS_EXPECTED,
            "E.1": "no_exception_merge: >=1 conflicting MERGE per world; "
                   "success_A drops, cost_A >> full",
            "E.2": "no_exception_drop: A intact, cost_V >> full, "
                   "exceptions == 0 (evidence lost)",
            "E.3": "no_new_schema: schema_count == 1, exceptions >= 3, "
                   "parent cert polluted by >= 2 fake failures; cost on "
                   "B/C may NOT degrade (registered caveat: structural, "
                   "not cost, damage under this consumer)",
            "E.4": "no_repeat: memory_items >= full + 3, identical costs "
                   "everywhere, support stats destroyed",
            "E.5": "merge_or_drop: memory_items == 1, >= 2 "
                   "cross-structure merges, success_A collapses",
            "E.6_v2": "full control: exact op sequence, 4 items, "
                      "first-try A, cost_V(full) < cost_V("
                      "no_exception_drop)",
            "E.7_v2": "exception two-world necessity: max-regret over "
                      "{A, V} of every exception-lacking arm > full's, "
                      "per world; exception-keeping arms match full "
                      "exactly",
            "v1_outcome": "v1 (smoke seeds 0-21): E.1-E.5 PASS as "
                          "registered; E.6 v1 FAILED on the clause "
                          "cost_V(full) < cost_V(no_exception_merge) "
                          "(20.8 vs 15.2) — a registration defect: "
                          "last-write-wins wins the corrupted world "
                          "ALONE by destroying the original (cost_A "
                          "52.2 vs 13.0). Necessity of EXCEPTION is a "
                          "two-world claim; re-registered as E.6 v2 + "
                          "E.7 v2, to be confirmed on disjoint seed "
                          "blocks (400+/800+/1200+) in the cluster run.",
        }, f, indent=2)

    table_rows, table_md = decision_table()
    ops_seen = {r["operation"] for r in table_rows}
    assert ops_seen == {"MERGE", "EXCEPTION", "NEW_SCHEMA", "REPEAT",
                        "DROP"}, ops_seen
    with open(os.path.join(args.out, "decision_table.md"), "w") as f:
        f.write(table_md + "\n")

    rows, skipped = [], 0
    for seed in range(args.seed_start, args.seed_start + args.seed_scan):
        if len(rows) >= args.n_worlds:
            break
        row = run_world(seed)
        if row is None:
            skipped += 1
            continue
        rows.append(row)
    if not rows:
        print("No valid worlds found (fail-closed); nothing to report.")
        return
    with open(os.path.join(args.out, "e_rows.json"), "w") as f:
        json.dump(rows, f, indent=1)

    agg = aggregate(rows)
    verd = verdicts(rows, agg)
    with open(os.path.join(args.out, "e_results.json"), "w") as f:
        json.dump({"n_worlds": len(rows), "skipped": skipped,
                   "aggregate": agg, "verdicts": verd,
                   "decision_table": table_rows}, f, indent=2)

    hdr = (f"{'arm':<20} {'cost_A':>7} {'cost_V':>7} {'cost_B':>7} "
           f"{'cost_C':>7} {'ok_A':>5} {'items':>6} {'schem':>6} "
           f"{'exc':>4} {'corr':>5} {'xmrg':>5} {'fail':>5}")
    print(f"worlds={len(rows)} skipped={skipped}")
    print(hdr)
    print("-" * len(hdr))
    for arm in ARMS:
        a = agg[arm]
        print(f"{arm:<20} {a['mean_cost']['A']:>7} {a['mean_cost']['V']:>7} "
              f"{a['mean_cost']['B']:>7} {a['mean_cost']['C']:>7} "
              f"{a['success_rate']['A']:>5.2f} "
              f"{a['mean_memory_items']:>6.1f} "
              f"{a['mean_schema_count']:>6.1f} "
              f"{a['mean_exception_count']:>4.1f} "
              f"{a['mean_corrupt_merges']:>5.1f} "
              f"{a['mean_cross_merges']:>5.1f} "
              f"{a['mean_cert_failure_entries']:>5.1f}")
    print("=" * len(hdr))
    for k, v in verd.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print("=" * len(hdr))
    print(f"Saved: {args.out}/e_results.json")


if __name__ == "__main__":
    main()
