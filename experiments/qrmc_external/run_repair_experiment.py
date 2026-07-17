"""Repair experiment: does a Q/R/M/C diagnosis select the intervention
that actually restores task completion, and does it do so better than
simpler diagnostics of the same logs?

Design (specified in repair_spec.json before any run):
  * Stage-level repair menu, applied WITHOUT knowledge of the injected
    fault: Q -- restore the query channel (recall tool + standard
    instruction); R -- consolidate memory (true record names the item;
    reliable recall); M -- remove records that name the item at a
    non-true place; C -- relax execution (budget +4; reliable
    goto/take).
  * Arms: every repair x every fault variant (+ control, to price
    false positives) x 3 frameworks x 20 seeds, llama3.1. The
    no-repair baseline is the existing blinded-benchmark data
    (rows_meta_llama.json).
  * Each diagnostic method selects the repair equal to its predicted
    stage ("none" -> no repair). Methods and their per-cell
    predictions come from diagnostic_baselines.py, computed on the
    ORIGINAL (unrepaired) profiles.
  * Metrics per method over fault cells:
      RepairGain   = mean( S[chosen] - S[none] )
      RepairRegret = mean( S[best]   - S[chosen] )
    plus random-repair and oracle-best comparators, and the cost of
    repairs applied to healthy controls.

Pre-specified predictions:
  P-REP-1  the matched repair (arm == true stage) restores >= 50% of
           the control-fault completion gap on the five structural
           faults, on every framework;
  P-REP-2  RepairRegret(Q/R/M/C) < RepairRegret of progress,
           two-stage, and conformance baselines;
  P-REP-3  repairs applied to the clean control change completion by
           at most 0.15 in absolute value.
  Risk     the C-repair (+4 budget) may act as a weak universal
           repair via longer blind search; if so, regret differences
           shrink -- reported either way.

Usage::

    PYTHONPATH=. .venv/bin/python experiments/qrmc_external/run_repair_experiment.py \
        [--seeds 20] [--analyze-only]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np

from experiments.qrmc_external.memory_house import (
    ALL_VARIANTS, FAULT_STAGE, MemoryHouse,
)
from experiments.qrmc_external import run_external_validation as ext

OUT_DIR = "tmp/qrmc_external"
FRAMEWORKS = ["openai_sdk", "langgraph", "autogen"]
REPAIRS = ["Q", "R", "M", "C"]
MODEL = "llama3.1:latest"


def run_episodes(seeds: int) -> list:
    rows_path = os.path.join(OUT_DIR, "rows_repair_llama.json")
    rows = []
    if os.path.exists(rows_path):
        rows = json.load(open(rows_path))
        print(f"resuming: {len(rows)} episodes done")
    done = {(r["framework"], r["variant"], r["repair"], r["seed"])
            for r in rows}
    ext.MODEL = MODEL
    variants = [v for v in ALL_VARIANTS]
    for fw in FRAMEWORKS:
        for variant in variants:
            for rep in REPAIRS:
                for seed in range(seeds):
                    if (fw, variant, rep, seed) in done:
                        continue
                    house = MemoryHouse(variant, seed, repair=rep)
                    try:
                        ext.ADAPTERS[fw](house)
                    except Exception as e:
                        house.log.append({"tool": "error",
                                          "arg": str(e)[:200], "n": -1})
                    row = {"framework": fw, "variant": variant,
                           "repair": rep, "seed": seed, **house.qrmc()}
                    rows.append(row)
                    print(f"  {fw:<10} {variant:<14} rep={rep} s{seed:<2} "
                          f"C{row['C']} ({row['n_tool_calls']} calls)",
                          flush=True)
                    with open(rows_path, "w") as f:
                        json.dump(rows, f, indent=1)
    return rows


def analyze(rows: list, seeds: int) -> dict:
    # S[fw][variant][arm] = completion rate; arm "none" from the
    # original blinded benchmark
    base = json.load(open(os.path.join(OUT_DIR, "rows_meta_llama.json")))
    S = defaultdict(dict)
    for fw in FRAMEWORKS:
        for v in ALL_VARIANTS:
            b = [r["C"] for r in base
                 if r["framework"] == fw and r["variant"] == v]
            S[(fw, v)]["none"] = float(np.mean(b))
            for rep in REPAIRS:
                a = [r["C"] for r in rows if r["framework"] == fw
                     and r["variant"] == v and r["repair"] == rep]
                if a:
                    S[(fw, v)][rep] = float(np.mean(a))

    # per-method diagnoses on the ORIGINAL profiles
    from experiments.qrmc_external.diagnostic_baselines import (
        RULES, load_cells,
    )
    cells = [c for c in load_cells()
             if c["model"] == "llama" and c["bound"]
             and c["variant"] != "control_holdout"]
    report = {"per_method": {}, "arms": {f"{fw}|{v}": S[(fw, v)]
                                         for (fw, v) in S}}
    fault_cells = [c for c in cells if c["truth"] != "none"]

    for name, rule in RULES.items():
        gains, regrets, picks = [], [], []
        for c in fault_cells:
            fw, v = c["framework"], c["variant"]
            arms = S[(fw, v)]
            pred = rule(c["profile"], c["ctrl"])
            chosen = pred if pred in REPAIRS else "none"
            best = max(arms, key=lambda a: arms[a])
            gains.append(arms[chosen] - arms["none"])
            regrets.append(arms[best] - arms[chosen])
            picks.append({"fw": fw, "variant": v, "pred": pred,
                          "chosen": chosen, "best": best,
                          "gain": round(arms[chosen] - arms["none"], 3)})
        report["per_method"][name] = {
            "repair_gain": round(float(np.mean(gains)), 3),
            "repair_regret": round(float(np.mean(regrets)), 3),
            "picks": picks}

    # comparators
    rnd_gain, rnd_reg, orc_gain = [], [], []
    for c in fault_cells:
        arms = S[(c["framework"], c["variant"])]
        best = max(arms, key=lambda a: arms[a])
        rg = [arms[rep] - arms["none"] for rep in REPAIRS]
        rnd_gain.append(float(np.mean(rg)))
        rnd_reg.append(arms[best] - float(np.mean([arms[r] for r in REPAIRS])))
        orc_gain.append(arms[best] - arms["none"])
    report["random_repair"] = {"repair_gain": round(float(np.mean(rnd_gain)), 3),
                               "repair_regret": round(float(np.mean(rnd_reg)), 3)}
    report["oracle_best"] = {"repair_gain": round(float(np.mean(orc_gain)), 3),
                             "repair_regret": 0.0}

    # matched-repair recovery on structural faults (P-REP-1)
    structural = ["q_no_tool", "r_starved", "r_corrupted",
                  "c_budget", "c_take_broken"]
    rec = []
    for fw in FRAMEWORKS:
        ctrl_S = S[(fw, "control")]["none"]
        for v in structural:
            gap = ctrl_S - S[(fw, v)]["none"]
            if gap <= 0:
                continue
            got = S[(fw, v)][FAULT_STAGE[v]] - S[(fw, v)]["none"]
            rec.append(got / gap)
    report["P_REP_1_matched_recovery_min"] = round(float(np.min(rec)), 3)
    report["P_REP_1_pass_ge_050_all"] = bool(np.min(rec) >= 0.5)

    # control harm (P-REP-3)
    harms = []
    for fw in FRAMEWORKS:
        for rep in REPAIRS:
            if rep in S[(fw, "control")]:
                harms.append(abs(S[(fw, "control")][rep]
                                 - S[(fw, "control")]["none"]))
    report["P_REP_3_max_control_harm"] = round(float(np.max(harms)), 3)
    report["P_REP_3_pass_le_015"] = bool(np.max(harms) <= 0.15)

    qr = report["per_method"]["qrmc"]["repair_regret"]
    report["P_REP_2_pass"] = all(
        qr < report["per_method"][m]["repair_regret"]
        for m in ["progress", "two_stage", "conformance"])

    slim = json.loads(json.dumps(report))
    for m in slim["per_method"].values():
        m.pop("picks", None)
    with open(os.path.join(OUT_DIR, "repair_verdict.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'method':<22} {'gain':>7} {'regret':>7}")
    for name, m in report["per_method"].items():
        print(f"{name:<22} {m['repair_gain']:>7} {m['repair_regret']:>7}")
    print(f"{'random_repair':<22} {report['random_repair']['repair_gain']:>7} "
          f"{report['random_repair']['repair_regret']:>7}")
    print(f"{'oracle_best':<22} {report['oracle_best']['repair_gain']:>7} "
          f"{0.0:>7}")
    for k in ["P_REP_1_pass_ge_050_all", "P_REP_2_pass", "P_REP_3_pass_le_015"]:
        print(f"  [{'PASS' if report[k] else 'FAIL'}] {k}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--analyze-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    spec_path = os.path.join(OUT_DIR, "repair_spec.json")
    if not os.path.exists(spec_path):
        json.dump({"date": "2026-07-17", "model": MODEL,
                   "menu": {"Q": "restore query channel",
                            "R": "consolidate memory + reliable recall",
                            "M": "remove untrue item records",
                            "C": "budget +4 + reliable goto/take"},
                   "P_REP_1": "matched repair restores >=50% of the "
                              "control-fault gap on structural faults",
                   "P_REP_2": "regret(qrmc) < regret(progress, "
                              "two_stage, conformance)",
                   "P_REP_3": "repairs on control change completion "
                              "by <= 0.15",
                   "risk": "C-repair (+4 budget) may be a weak "
                           "universal repair via longer blind search"},
                  open(spec_path, "w"), indent=2)
        print(f"spec written: {spec_path}")

    if a.analyze_only:
        rows = json.load(open(os.path.join(OUT_DIR, "rows_repair_llama.json")))
    else:
        rows = run_episodes(a.seeds)
    analyze(rows, a.seeds)


if __name__ == "__main__":
    main()
