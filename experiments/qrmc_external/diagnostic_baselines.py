"""Comparison of Q/R/M/C with diagnostic baselines on the blinded
fault-localization benchmark (reviewer request: show that the
four-stage protocol outperforms simpler analyses of the same logs).

All methods receive the SAME inputs as the Q/R/M/C rule did: the
anonymised per-cell aggregate profile plus the labelled calibration
control of the same (model, framework). No method sees the injected
fault name. All rules below were fixed before scoring
(baselines_spec.json is written before any evaluation); thresholds
are matched to the Q/R/M/C rule's (0.25 drop; 0.6 absolute; 0.2 for
first-commitment) so no baseline is disadvantaged by a stricter
cutoff.

Baselines
---------
B1 success-only     : success rate and tool-call counts only; can
                      flag failure but names no stage.
B2 progress         : AgentBoard-style milestones = the four marginal
                      event rates read as a progress scale; diagnosis
                      = first milestone whose rate drops vs control.
B3 two-stage R/E    : RAGChecker-style split retrieval (Q,R) vs
                      execution (M,C); side with the larger drop;
                      exact-stage credit via the canonical mapping
                      retrieval->R, execution->C.
B4 conformance-lite : process-conformance first deviation on the
                      canonical sequence recall -> relevant record ->
                      first goto correct -> take, using the
                      order-sensitive M1; identical to Q/R/M/C's rule
                      minus the budget-exhaustion branch.
B5 learned LR       : multinomial logistic regression on cell
                      aggregates; leave-one-fault-type-out,
                      leave-one-framework-out, leave-one-model-out.

Metrics: exact stage accuracy, macro-F1 over {Q,R,M,C,none}, control
false-positive rate (held-out controls), McNemar exact test vs
Q/R/M/C, bootstrap CIs resampling fault types (not episodes).

Usage::

    PYTHONPATH=. .venv/bin/python experiments/qrmc_external/diagnostic_baselines.py
"""

from __future__ import annotations

import itertools
import json
import math
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np

from experiments.qrmc_external.memory_house import FAULT_STAGE
from experiments.qrmc_external.run_meta_evaluation import (
    THRESH, decide, binds,
)

OUT_DIR = "tmp/qrmc_external"
MODELS = {"llama": "rows_meta_llama.json", "qwen": "rows_meta_qwen.json"}
FRAMEWORKS = ["openai_sdk", "langgraph", "autogen"]
KEYS = ["Q", "R", "M", "C", "M1", "exhausted", "n_tool_calls"]
STAGES = ["Q", "R", "M", "C", "none"]
SEEDS = 20
RNG = np.random.default_rng(0)


# ── baseline decision rules (fixed before scoring) ──────────────────

def b1_success(cell, ctrl):
    return "failure" if ctrl["C"] - cell["C"] >= 0.25 else "none"


def b2_progress(cell, ctrl):
    for st in ["Q", "R", "M", "C"]:
        if ctrl[st] - cell[st] >= 0.25:
            return st
    return "none"


def b3_twostage(cell, ctrl):
    r_drop = max(ctrl["Q"] - cell["Q"], ctrl["R"] - cell["R"])
    e_drop = max(ctrl["M"] - cell["M"], ctrl["C"] - cell["C"])
    if max(r_drop, e_drop) < 0.25:
        return "none"
    return "R" if r_drop >= e_drop else "C"  # canonical mapping


def b3_side(cell, ctrl):
    r_drop = max(ctrl["Q"] - cell["Q"], ctrl["R"] - cell["R"])
    e_drop = max(ctrl["M"] - cell["M"], ctrl["C"] - cell["C"])
    if max(r_drop, e_drop) < 0.25:
        return "none"
    return "retrieval" if r_drop >= e_drop else "execution"


def b4_conformance(cell, ctrl):
    if cell["Q"] <= 0.6:
        return "Q"
    if ctrl["R"] - cell["R"] >= 0.25:
        return "R"
    if ctrl["M1"] - cell["M1"] >= 0.20:
        return "M"
    if ctrl["C"] - cell["C"] >= 0.25:
        return "C"
    return "none"


def qrmc_rule(cell, ctrl):
    return decide(cell, ctrl)


RULES = {"success_only": b1_success, "progress": b2_progress,
         "two_stage": b3_twostage, "conformance": b4_conformance,
         "qrmc": qrmc_rule}


# ── data loading ─────────────────────────────────────────────────────

def load_cells(out_dir=OUT_DIR):
    """-> list of dicts: model, framework, variant, truth, bound,
    profile (means), ctrl (calibration control means)."""
    cells = []
    for model, fname in MODELS.items():
        rows = json.load(open(os.path.join(out_dir, fname)))
        for fw in FRAMEWORKS:
            fwr = [r for r in rows if r["framework"] == fw]

            def agg(rs):
                return {k: float(np.mean([r[k] for r in rs]))
                        for k in KEYS}

            ctrl = agg([r for r in fwr if r["variant"] == "control"
                        and r["seed"] < SEEDS])
            for v in sorted({r["variant"] for r in fwr}):
                if v == "control":
                    prof = agg([r for r in fwr if r["variant"] == v
                                and r["seed"] >= SEEDS])
                    name = "control_holdout"
                else:
                    prof = agg([r for r in fwr if r["variant"] == v])
                    name = v
                truth = FAULT_STAGE["control" if v == "control" else v]
                cells.append({
                    "model": model, "framework": fw, "variant": name,
                    "truth": truth, "profile": prof, "ctrl": ctrl,
                    "bound": binds("control" if v == "control" else v,
                                   prof, ctrl)})
    return cells


# ── metrics ──────────────────────────────────────────────────────────

def macro_f1(pairs):
    f1s = []
    for st in STAGES:
        tp = sum(1 for t, p in pairs if t == st and p == st)
        fp = sum(1 for t, p in pairs if t != st and p == st)
        fn = sum(1 for t, p in pairs if t == st and p != st)
        if tp + fp + fn == 0:
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s))


def mcnemar(correct_a, correct_b):
    """exact binomial McNemar on discordant pairs."""
    b = sum(1 for x, y in zip(correct_a, correct_b) if x and not y)
    c = sum(1 for x, y in zip(correct_a, correct_b) if not x and y)
    n = b + c
    if n == 0:
        return 1.0, b, c
    p = sum(math.comb(n, k) for k in range(min(b, c) + 1)) * 2 / 2 ** n
    return min(1.0, p), b, c


def boot_ci_by_fault(records, n_iter=4000):
    """bootstrap accuracy resampling fault types (incl. holdout ctrl)."""
    by_type = defaultdict(list)
    for r in records:
        by_type[r["variant"]].append(r["correct"])
    types = list(by_type)
    accs = []
    for _ in range(n_iter):
        sample = RNG.choice(len(types), size=len(types), replace=True)
        vals = [v for i in sample for v in by_type[types[i]]]
        accs.append(np.mean(vals))
    return float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))


# ── learned baseline ─────────────────────────────────────────────────

def features(cell):
    p, c = cell["profile"], cell["ctrl"]
    f = [p[k] for k in KEYS] + [c[k] - p[k] for k in KEYS]
    f[KEYS.index("n_tool_calls")] /= 6.0
    f[len(KEYS) + KEYS.index("n_tool_calls")] /= 6.0
    return f


def learned_lr(cells, regime):
    from sklearn.linear_model import LogisticRegression
    scored = [c for c in cells if c["bound"]]
    X = np.array([features(c) for c in scored])
    y = np.array([c["truth"] for c in scored])
    preds = {}
    if regime == "fault":
        groups = [c["variant"] for c in scored]
    elif regime == "framework":
        groups = [c["framework"] for c in scored]
    else:
        groups = [c["model"] for c in scored]
    for g in sorted(set(groups)):
        tr = [i for i, gg in enumerate(groups) if gg != g]
        te = [i for i, gg in enumerate(groups) if gg == g]
        clf = LogisticRegression(max_iter=2000, multi_class="multinomial")
        clf.fit(X[tr], y[tr])
        for i, p in zip(te, clf.predict(X[te])):
            preds[i] = p
    return [(scored[i], preds[i]) for i in sorted(preds)]


# ── main ─────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    spec_path = os.path.join(OUT_DIR, "baselines_spec.json")
    if not os.path.exists(spec_path):
        json.dump({
            "date": "2026-07-17",
            "note": "baseline rules fixed before scoring; thresholds "
                    "matched to the Q/R/M/C rule (0.25/0.6/0.20); all "
                    "methods receive identical anonymised cell "
                    "aggregates + labelled calibration control; "
                    "non-binding cells excluded for all methods alike",
            "rules": {k: RULES[k].__doc__ or k for k in RULES},
            "thresholds": THRESH,
        }, open(spec_path, "w"), indent=2)
        print(f"spec written: {spec_path}")

    cells = load_cells()
    scored_cells = [c for c in cells if c["bound"]]
    print(f"{len(cells)} cells, {len(scored_cells)} scored "
          f"(non-binding excluded for all methods)\n")

    report = {"methods": {}}
    correct_vectors = {}
    for name, rule in RULES.items():
        recs = []
        for c in scored_cells:
            pred = rule(c["profile"], c["ctrl"])
            recs.append({**{k: c[k] for k in
                            ("model", "framework", "variant", "truth")},
                         "pred": pred, "correct": pred == c["truth"]})
        pairs = [(r["truth"], r["pred"]) for r in recs]
        acc = np.mean([r["correct"] for r in recs])
        ctrl_recs = [r for r in recs if r["variant"] == "control_holdout"]
        fpr = np.mean([r["pred"] != "none" for r in ctrl_recs])
        lo, hi = boot_ci_by_fault(recs)
        per_model = {m: float(np.mean([r["correct"] for r in recs
                                       if r["model"] == m]))
                     for m in MODELS}
        report["methods"][name] = {
            "accuracy": round(float(acc), 3), "ci": [round(lo, 3), round(hi, 3)],
            "macro_f1": round(macro_f1(pairs), 3),
            "control_fpr": round(float(fpr), 3),
            "per_model": {k: round(v, 3) for k, v in per_model.items()},
            "records": recs}
        correct_vectors[name] = [r["correct"] for r in recs]

    # B3 side-level accuracy (coarse credit)
    side_map = {"Q": "retrieval", "R": "retrieval",
                "M": "execution", "C": "execution", "none": "none"}
    side_recs = []
    for c in scored_cells:
        pred = b3_side(c["profile"], c["ctrl"])
        side_recs.append(pred == side_map[c["truth"]])
    report["two_stage_side_level_accuracy"] = round(float(np.mean(side_recs)), 3)

    # learned baseline, three transfer regimes
    for regime in ["fault", "framework", "model"]:
        recs = []
        for c, pred in learned_lr(cells, regime):
            recs.append({"variant": c["variant"],
                         "correct": pred == c["truth"],
                         "is_ctrl": c["variant"] == "control_holdout",
                         "pred": pred})
        acc = np.mean([r["correct"] for r in recs])
        fpr = np.mean([r["pred"] != "none" for r in recs if r["is_ctrl"]])
        report["methods"][f"learned_lr_leave_{regime}_out"] = {
            "accuracy": round(float(acc), 3),
            "control_fpr": round(float(fpr), 3)}
        correct_vectors[f"learned_lr_leave_{regime}_out"] = \
            [r["correct"] for r in recs]

    # McNemar vs qrmc (rule-based ones share the same cell order)
    report["mcnemar_vs_qrmc"] = {}
    for name in ["success_only", "progress", "two_stage", "conformance",
                 "learned_lr_leave_fault_out"]:
        a, b = correct_vectors["qrmc"], correct_vectors[name]
        if len(a) == len(b):
            p, x, y = mcnemar(a, b)
            report["mcnemar_vs_qrmc"][name] = {
                "p": round(p, 4), "qrmc_only_correct": x,
                "baseline_only_correct": y}

    # cost profile (qualitative)
    report["cost"] = {
        "success_only": {"training": False, "reruns": 0},
        "progress": {"training": False, "reruns": 0},
        "two_stage": {"training": False, "reruns": 0},
        "conformance": {"training": False, "reruns": 0},
        "learned_lr": {"training": True, "needs_fault_labels": True,
                       "reruns": 0},
        "qrmc": {"training": False, "reruns": 0},
    }

    for name, m in report["methods"].items():
        extra = (f" F1={m['macro_f1']}" if "macro_f1" in m else "")
        print(f"{name:<30} acc={m['accuracy']:<6}{extra} "
              f"FPR={m['control_fpr']}"
              + (f" CI={m['ci']}" if "ci" in m else ""))
    print("two_stage side-level acc:", report["two_stage_side_level_accuracy"])
    print("McNemar vs qrmc:", json.dumps(report["mcnemar_vs_qrmc"]))

    slim = {k: v for k, v in report.items()}
    for m in slim["methods"].values():
        m.pop("records", None)
    with open(os.path.join(OUT_DIR, "baselines_verdict.json"), "w") as f:
        json.dump(slim, f, indent=2)
    print(f"\nsaved {OUT_DIR}/baselines_verdict.json")


if __name__ == "__main__":
    main()
