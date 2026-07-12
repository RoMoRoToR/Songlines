"""Blinded fault-localization meta-evaluation of the Q/R/M/C protocol.

Answers the reviewer's strongest methodological objection: the original
external validation engineered failures whose localization follows from
the event definitions ("tautology").  Here the protocol is tested as a
CLASSIFIER: ten fault types (2-3 per stage, structural AND behavioural)
are injected into MemoryHouse, and a decision rule --- fixed and
registered BEFORE any episode is run --- must recover the injected
stage from the aggregate Q/R/M/C profile alone.

Design
------
* 10 faults + control (see FAULT_STAGE in memory_house.py).  Ground
  truth = the stage-owning component the fault was injected into.
* Blinding: the decision rule is code written before the runs; at
  analysis time it receives anonymised numeric profiles (no variant
  names).  A labelled control cell (seeds 0..19) is given to the rule
  for calibration --- a practitioner always has a baseline; a HELD-OUT
  control cell (seeds 20..39) enters the blind set with ground truth
  "none", measuring the false-positive rate.
* Manipulation check: a fault must BIND (change behaviour or the
  trace) before localization of it is scored.  Registered binding
  criteria per family below; non-binding cells are reported, not
  scored (analogous to an instrument-relevance check).
* Metrics: stage-level top-1 accuracy per framework, full confusion
  matrix, FP rate on held-out control, cross-framework agreement.
  Fault-level identity within a stage is NOT claimed: e.g. r_starved
  vs r_corrupted are expected to be indistinguishable from the four
  rates --- the confusion structure within a stage is itself a
  finding about the protocol's resolution.

Registered decision rule (first-failing-link on marginal diagnostics,
thresholds fixed a priori; ctrl = labelled control cell of the same
(framework, model)):

    1. Q  <= 0.6                                   -> "Q"
    2. R  <= ctrl.R - 0.25                         -> "R"
    3. exhausted >= 0.75 and ctrl.C - C >= 0.25    -> "C"
    4. ctrl.M1 - M1 >= 0.20 or ctrl.M - M >= 0.20  -> "M"
    5. ctrl.C - C >= 0.25                          -> "C"
    6. otherwise                                   -> "none"

Registered predictions (before any run):
  P-META-1 (llama3.1): stage-level accuracy >= 7/11 blind cells per
     framework; the five structural faults (q_no_tool, r_starved,
     r_corrupted, c_budget, c_take_broken) localize 5/5 on every
     framework; held-out control -> "none" on every framework.
  P-META-2 (qwen3:4b): per-framework accuracy >= llama's; the
     M-family faults produce a visible M1 drop (>= 0.20) on >= 2/3
     frameworks (qwen's clean commitment discipline M1=0.90-0.95
     makes M-faults legible where llama's dirty baseline hides them).
  P-META-3: the predicted stage label agrees across the three
     frameworks in >= 8/11 blind cells (llama).
  Registered risks (expected failure modes, stated in advance):
     q_misleading may not bind (prompt-only fault; llama may ignore
     the note) --- if Q drops < 0.20 vs control the cell is excluded
     by the manipulation check; m_ambiguous / m_duplicate on llama
     may misclassify as C or none because llama's control M1 is
     already 0.25-0.50 --- this is the "weak committer hides M-faults"
     phenomenon the paper reports, and the qwen contrast (P-META-2)
     is the designed test of it.

Usage::

    PYTHONPATH=. .venv/bin/python experiments/qrmc_external/run_meta_evaluation.py \
        --model llama3.1:latest --tag meta_llama [--seeds 20] [--analyze-only]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from experiments.qrmc_external.memory_house import (
    ALL_VARIANTS, FAULT_STAGE, MemoryHouse,
)
from experiments.qrmc_external import run_external_validation as ext

OUT_DIR = "tmp/qrmc_external"
FRAMEWORKS = ["openai_sdk", "langgraph", "autogen"]
META_VARIANTS = [v for v in ALL_VARIANTS]  # control + 10 faults

# ── the registered decision rule (do not edit after registration) ───

THRESH = {"Q_ABS": 0.6, "D_R": 0.25, "D_M": 0.20, "D_C": 0.25,
          "EXH": 0.75}


def decide(cell: Dict[str, float], ctrl: Dict[str, float]) -> str:
    """Blind localization: numeric profiles only, no variant names."""
    t = THRESH
    if cell["Q"] <= t["Q_ABS"]:
        return "Q"
    if cell["R"] <= ctrl["R"] - t["D_R"]:
        return "R"
    if (cell["exhausted"] >= t["EXH"]
            and ctrl["C"] - cell["C"] >= t["D_C"]):
        return "C"
    if (ctrl["M1"] - cell["M1"] >= t["D_M"]
            or ctrl["M"] - cell["M"] >= t["D_M"]):
        return "M"
    if ctrl["C"] - cell["C"] >= t["D_C"]:
        return "C"
    return "none"


def binds(variant: str, cell: Dict[str, float],
          ctrl: Dict[str, float]) -> bool:
    """Manipulation check: did the injected fault change anything?
    Structural and content faults bind by construction; the prompt-only
    fault must move the Q rate; runtime-flaky faults must show up in
    the trace (they do, with p=0.7/0.5 over 20 seeds)."""
    if variant == "q_misleading":
        return ctrl["Q"] - cell["Q"] >= 0.20
    return True


# ── runner ───────────────────────────────────────────────────────────


def run_episodes(frameworks: List[str], seeds: int, tag: str) -> List[Dict]:
    rows_path = os.path.join(OUT_DIR, f"rows_{tag}.json")
    rows: List[Dict[str, Any]] = []
    if os.path.exists(rows_path):
        rows = json.load(open(rows_path))
        print(f"resuming: {len(rows)} episodes already done")
    done = {(r["framework"], r["variant"], r["seed"]) for r in rows}

    for fw in frameworks:
        for variant in META_VARIANTS:
            n_seeds = 2 * seeds if variant == "control" else seeds
            for seed in range(n_seeds):
                if (fw, variant, seed) in done:
                    continue
                house = MemoryHouse(variant, seed)
                try:
                    ext.ADAPTERS[fw](house)
                except Exception as e:
                    house.log.append({"tool": "error",
                                      "arg": str(e)[:200], "n": -1})
                row = {"framework": fw, "variant": variant,
                       "seed": seed, **house.qrmc()}
                rows.append(row)
                print(f"  {fw:<10} {variant:<14} s{seed:<2} "
                      f"Q{row['Q']}R{row['R']}M{row['M']}C{row['C']} "
                      f"({row['n_tool_calls']} calls)", flush=True)
                with open(rows_path, "w") as f:
                    json.dump(rows, f, indent=1)
    return rows


# ── blind analysis ───────────────────────────────────────────────────


def analyze(rows: List[Dict], frameworks: List[str], seeds: int,
            tag: str) -> Dict[str, Any]:
    keys = ["Q", "R", "M", "C", "M1", "exhausted"]

    def agg(rs: List[Dict]) -> Dict[str, float]:
        out = {k: sum(r[k] for r in rs) / len(rs) for k in keys}
        out["n"] = len(rs)
        return out

    report: Dict[str, Any] = {"per_framework": {}, "thresholds": THRESH}
    stages = ["Q", "R", "M", "C", "none"]
    pooled_conf = {s: {p: 0 for p in stages} for s in stages}
    fw_labels: Dict[str, Dict[str, str]] = {}

    for fw in frameworks:
        fwr = [r for r in rows if r["framework"] == fw]
        ctrl = agg([r for r in fwr if r["variant"] == "control"
                    and r["seed"] < seeds])
        # blind set: 10 faults + held-out control (anonymised: decide()
        # sees numbers only; names are used here solely for scoring)
        cells: Dict[str, Dict[str, float]] = {}
        for v in META_VARIANTS:
            if v == "control":
                cells["control_holdout"] = agg(
                    [r for r in fwr if r["variant"] == "control"
                     and r["seed"] >= seeds])
            else:
                cells[v] = agg([r for r in fwr if r["variant"] == v])

        results, correct, scored = {}, 0, 0
        fw_labels[fw] = {}
        for v, cell in cells.items():
            variant = "control" if v == "control_holdout" else v
            truth = FAULT_STAGE[variant]
            pred = decide(cell, ctrl)
            fw_labels[fw][v] = pred
            bound = binds(variant, cell, ctrl)
            if bound:
                scored += 1
                correct += int(pred == truth)
                pooled_conf[truth][pred] += 1
            results[v] = {"truth": truth, "pred": pred,
                          "bound": bound, "profile": cell}
        report["per_framework"][fw] = {
            "control_calibration": ctrl, "cells": results,
            "accuracy": f"{correct}/{scored}",
            "accuracy_frac": round(correct / max(scored, 1), 3)}

    # cross-framework agreement on predicted labels
    blind_names = [v for v in fw_labels[frameworks[0]]]
    agree = sum(1 for v in blind_names
                if len({fw_labels[fw][v] for fw in frameworks}) == 1)
    report["cross_framework_agreement"] = f"{agree}/{len(blind_names)}"
    report["confusion_matrix_pooled"] = pooled_conf

    # verdicts against the registered predictions
    structural = ["q_no_tool", "r_starved", "r_corrupted",
                  "c_budget", "c_take_broken"]
    struct_ok = all(
        report["per_framework"][fw]["cells"][v]["pred"]
        == FAULT_STAGE[v]
        for fw in frameworks for v in structural)
    holdout_ok = all(
        report["per_framework"][fw]["cells"]["control_holdout"]["pred"]
        == "none" for fw in frameworks)
    acc_ok = all(
        int(report["per_framework"][fw]["accuracy"].split("/")[0]) >= 7
        for fw in frameworks)
    report["verdict"] = {
        "P_META_1_accuracy_ge_7_of_11": acc_ok,
        "P_META_1_structural_5of5_all_fw": struct_ok,
        "P_META_1_holdout_control_none": holdout_ok,
        "P_META_3_cross_fw_agreement_ge_8_of_11": agree >= 8,
    }
    with open(os.path.join(OUT_DIR, f"meta_verdict_{tag}.json"),
              "w") as f:
        json.dump(report, f, indent=2)
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--frameworks", nargs="+", default=FRAMEWORKS)
    ap.add_argument("--model", default=ext.MODEL)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--analyze-only", action="store_true")
    a = ap.parse_args()
    ext.MODEL = a.model
    tag = a.tag or ("meta_" + a.model.split(":")[0].replace(".", "_"))
    os.makedirs(OUT_DIR, exist_ok=True)

    # registration is write-once: never overwrite a pre-run file
    reg_path = os.path.join(OUT_DIR, "registered_meta.json")
    if not os.path.exists(reg_path):
        with open(reg_path, "w") as f:
            json.dump({
                "registered_before_runs": True,
                "date": "2026-07-10",
                "taxonomy": FAULT_STAGE,
                "decision_rule": ("first-failing-link on marginal "
                                  "diagnostics; see decide() and "
                                  "THRESH in run_meta_evaluation.py"),
                "thresholds": THRESH,
                "blinding": ("rule fixed pre-run; receives anonymised "
                             "numeric profiles + a labelled control "
                             "cell (seeds 0..19); held-out control "
                             "(seeds 20..39) enters the blind set as "
                             "'none'"),
                "manipulation_check": ("q_misleading scored only if "
                                       "Q drops >= 0.20 vs control"),
                "P_META_1": ("llama3.1: accuracy >= 7/11 per "
                             "framework; structural 5/5 everywhere; "
                             "held-out control -> none everywhere"),
                "P_META_2": ("qwen3:4b: accuracy >= llama per "
                             "framework; M-family M1-drop >= 0.20 on "
                             ">= 2/3 frameworks"),
                "P_META_3": ("cross-framework label agreement >= 8/11 "
                             "(llama)"),
                "registered_risks": ("q_misleading may not bind; "
                                     "m_ambiguous/m_duplicate on llama "
                                     "may read as C or none (weak "
                                     "baseline commitment); fault "
                                     "identity within a stage not "
                                     "claimed (r_starved vs "
                                     "r_corrupted expected "
                                     "indistinguishable)"),
                "scope": {"frameworks": FRAMEWORKS, "seeds": 20,
                          "models": ["llama3.1:latest", "qwen3:4b"]},
            }, f, indent=2)
        print(f"registered: {reg_path}")

    rows_path = os.path.join(OUT_DIR, f"rows_{tag}.json")
    if a.analyze_only:
        rows = json.load(open(rows_path))
    else:
        rows = run_episodes(a.frameworks, a.seeds, tag)

    report = analyze(rows, a.frameworks, a.seeds, tag)
    print("=" * 64)
    for fw in a.frameworks:
        r = report["per_framework"][fw]
        print(f"{fw:<11} accuracy {r['accuracy']}")
        for v, c in r["cells"].items():
            mark = "OK " if c["pred"] == c["truth"] else "MISS"
            b = "" if c["bound"] else " (non-binding, excluded)"
            print(f"    [{mark}] {v:<16} truth={c['truth']:<4} "
                  f"pred={c['pred']}{b}")
    print(f"cross-framework agreement: "
          f"{report['cross_framework_agreement']}")
    for k, v in report["verdict"].items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"Saved: {OUT_DIR}/meta_verdict_{tag}.json")


if __name__ == "__main__":
    main()
