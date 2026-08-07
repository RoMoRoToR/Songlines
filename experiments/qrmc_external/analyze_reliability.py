"""Sampling and threshold reliability checks for the Q/R/M/C meta-eval.

This analysis uses the existing 20-episode-per-cell blinded benchmark
records. It does not run any LLM episodes. The goal is to audit the
evaluation protocol itself:

* sample-size sensitivity: re-compute diagnoses after subsampling
  n in {5, 10, 15, 20} episodes from each calibration and blind cell;
* threshold sensitivity: re-compute full-data diagnoses after scaling
  the drop thresholds D_R, D_M, and D_C by 0.8..1.2.

The scored-cell set is fixed to the original 63 cells that passed the
full-data manipulation check. This keeps denominators comparable across
subsamples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np

from experiments.qrmc_external.diagnostic_baselines import (
    KEYS,
    load_cells,
    macro_f1,
)
from experiments.qrmc_external.memory_house import FAULT_STAGE
from experiments.qrmc_external.run_meta_evaluation import THRESH

OUT_DIR = "tmp/qrmc_external"
MODELS = {"llama": "rows_meta_llama.json", "qwen": "rows_meta_qwen.json"}
MODEL_IDS = {"llama": "llama3.1:latest", "qwen": "qwen3:4b"}
OLLAMA_MODEL_IDS = {"llama": "46e0c10c039e", "qwen": "359d7dd4bcda"}
FRAMEWORKS = ["openai_sdk", "langgraph", "autogen"]
DEFAULT_SAMPLE_SIZES = [5, 10, 15, 20]
DEFAULT_SCALES = [0.8, 0.9, 1.0, 1.1, 1.2]


def decide_scaled(cell: Dict[str, float], ctrl: Dict[str, float],
                  scale: float = 1.0) -> str:
    """Q/R/M/C decision rule with scaled drop thresholds.

    Absolute Q and exhaustion thresholds are left fixed; the sensitivity
    audit targets evaluator choices for stage-drop thresholds.
    """
    d_r = THRESH["D_R"] * scale
    d_m = THRESH["D_M"] * scale
    d_c = THRESH["D_C"] * scale
    if cell["Q"] <= THRESH["Q_ABS"]:
        return "Q"
    if cell["R"] <= ctrl["R"] - d_r:
        return "R"
    if cell["exhausted"] >= THRESH["EXH"] and ctrl["C"] - cell["C"] >= d_c:
        return "C"
    if ctrl["M1"] - cell["M1"] >= d_m or ctrl["M"] - cell["M"] >= d_m:
        return "M"
    if ctrl["C"] - cell["C"] >= d_c:
        return "C"
    return "none"


def mean_profile(rows: Iterable[Dict]) -> Dict[str, float]:
    rows = list(rows)
    return {k: float(np.mean([r[k] for r in rows])) for k in KEYS}


def load_rows(rows_dir: str = OUT_DIR) -> Dict[Tuple[str, str, str], List[Dict]]:
    grouped: Dict[Tuple[str, str, str], List[Dict]] = defaultdict(list)
    for model, fname in MODELS.items():
        rows = json.load(open(os.path.join(rows_dir, fname)))
        for row in rows:
            grouped[(model, row["framework"], row["variant"])].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda r: r["seed"])
    return grouped


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def input_manifest(rows_dir: str = OUT_DIR) -> Dict[str, Dict]:
    manifest = {}
    for model, fname in MODELS.items():
        path = os.path.join(rows_dir, fname)
        rows = json.load(open(path))
        counts: Dict[Tuple[str, str], int] = defaultdict(int)
        for row in rows:
            counts[(row["framework"], row["variant"])] += 1
        control_counts = [
            n for (_, variant), n in counts.items() if variant == "control"
        ]
        fault_counts = [
            n for (_, variant), n in counts.items() if variant != "control"
        ]
        manifest[model] = {
            "path": path,
            "model_id": MODEL_IDS[model],
            "ollama_model_id": OLLAMA_MODEL_IDS[model],
            "sha256": sha256(path),
            "rows": len(rows),
            "frameworks": sorted({row["framework"] for row in rows}),
            "variants": sorted({row["variant"] for row in rows}),
            "control_rows_per_framework": control_counts,
            "fault_rows_per_framework_variant_min": min(fault_counts),
            "fault_rows_per_framework_variant_max": max(fault_counts),
        }
    return manifest


def scored_cell_specs(rows_dir: str = OUT_DIR) -> List[Dict]:
    specs = []
    for cell in load_cells(rows_dir):
        if not cell["bound"]:
            continue
        blind_variant = "control" if cell["variant"] == "control_holdout" else cell["variant"]
        specs.append({
            "model": cell["model"],
            "framework": cell["framework"],
            "variant": cell["variant"],
            "blind_variant": blind_variant,
            "truth": cell["truth"],
        })
    return specs


def full_predictions(grouped: Dict[Tuple[str, str, str], List[Dict]],
                     specs: List[Dict], scale: float = 1.0) -> List[str]:
    preds = []
    for spec in specs:
        model, fw = spec["model"], spec["framework"]
        ctrl_rows = [r for r in grouped[(model, fw, "control")] if r["seed"] < 20]
        if spec["variant"] == "control_holdout":
            cell_rows = [r for r in grouped[(model, fw, "control")] if r["seed"] >= 20]
        else:
            cell_rows = grouped[(model, fw, spec["blind_variant"])]
        preds.append(decide_scaled(mean_profile(cell_rows), mean_profile(ctrl_rows), scale))
    return preds


def summarize_predictions(specs: List[Dict], preds: List[str],
                          baseline_preds: List[str] | None = None) -> Dict:
    pairs = [(s["truth"], p) for s, p in zip(specs, preds)]
    correct = [truth == pred for truth, pred in pairs]
    controls = [i for i, s in enumerate(specs) if s["truth"] == "none"]
    out = {
        "accuracy": float(np.mean(correct)),
        "macro_f1": float(macro_f1(pairs)),
        "control_fpr": float(np.mean([preds[i] != "none" for i in controls])),
    }
    if baseline_preds is not None:
        out["flip_rate_vs_full"] = float(np.mean([
            p != b for p, b in zip(preds, baseline_preds)
        ]))
    return out


def sample_rows(rows: List[Dict], n: int, rng: np.random.Generator,
                control_holdout: bool = False) -> List[Dict]:
    if control_holdout:
        pool = [r for r in rows if r["seed"] >= 20]
    else:
        pool = [r for r in rows if r["seed"] < 20]
    if n >= len(pool):
        return pool
    idx = rng.choice(len(pool), size=n, replace=False)
    return [pool[int(i)] for i in idx]


def subsample_predictions(grouped: Dict[Tuple[str, str, str], List[Dict]],
                          specs: List[Dict], n: int,
                          rng: np.random.Generator) -> List[str]:
    ctrl_profiles: Dict[Tuple[str, str], Dict[str, float]] = {}
    for model in MODELS:
        for fw in FRAMEWORKS:
            rows = grouped[(model, fw, "control")]
            ctrl_profiles[(model, fw)] = mean_profile(sample_rows(rows, n, rng))

    preds = []
    for spec in specs:
        model, fw = spec["model"], spec["framework"]
        if spec["variant"] == "control_holdout":
            rows = grouped[(model, fw, "control")]
            profile = mean_profile(sample_rows(rows, n, rng, control_holdout=True))
        else:
            rows = grouped[(model, fw, spec["blind_variant"])]
            profile = mean_profile(sample_rows(rows, n, rng))
        preds.append(decide_scaled(profile, ctrl_profiles[(model, fw)]))
    return preds


def percentile(xs: List[float]) -> List[float]:
    return [float(np.percentile(xs, 2.5)), float(np.percentile(xs, 97.5))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicates", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260807)
    ap.add_argument("--sample-sizes", type=int, nargs="+",
                    default=DEFAULT_SAMPLE_SIZES)
    ap.add_argument("--scales", type=float, nargs="+", default=DEFAULT_SCALES)
    ap.add_argument("--rows-dir", default=OUT_DIR)
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "reliability_verdict.json"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    grouped = load_rows(args.rows_dir)
    specs = scored_cell_specs(args.rows_dir)
    baseline = full_predictions(grouped, specs, scale=1.0)
    baseline_summary = summarize_predictions(specs, baseline)

    sampling = {}
    for n in args.sample_sizes:
        accs, f1s, fprs, flips = [], [], [], []
        for _ in range(args.replicates):
            preds = subsample_predictions(grouped, specs, n, rng)
            s = summarize_predictions(specs, preds, baseline)
            accs.append(s["accuracy"])
            f1s.append(s["macro_f1"])
            fprs.append(s["control_fpr"])
            flips.append(s["flip_rate_vs_full"])
        sampling[str(n)] = {
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_ci": percentile(accs),
            "macro_f1_mean": float(np.mean(f1s)),
            "macro_f1_ci": percentile(f1s),
            "control_fpr_mean": float(np.mean(fprs)),
            "flip_rate_vs_full_mean": float(np.mean(flips)),
            "flip_rate_vs_full_ci": percentile(flips),
        }

    threshold = {}
    for scale in args.scales:
        preds = full_predictions(grouped, specs, scale=scale)
        threshold[f"{scale:.2f}"] = summarize_predictions(specs, preds, baseline)

    report = {
        "derived_from_existing_episode_logs": True,
        "analysis_script": "experiments/qrmc_external/analyze_reliability.py",
        "command": (".venv/bin/python "
                    "experiments/qrmc_external/analyze_reliability.py "
                    f"--replicates {args.replicates} --seed {args.seed} "
                    f"--rows-dir {args.rows_dir} --out {args.out}"),
        "inputs": input_manifest(args.rows_dir),
        "source_rows": MODELS,
        "scored_cells": len(specs),
        "replicates": args.replicates,
        "seed": args.seed,
        "full_data": baseline_summary,
        "sampling": sampling,
        "threshold_scaling": threshold,
        "notes": {
            "sample_scheme": "without-replacement subsamples from the existing 20 episodes per cell; calibration controls are sampled once per model/framework/replicate",
            "scored_set": "fixed to the original cells passing full-data manipulation checks",
            "threshold_scaling": "scales only D_R, D_M, and D_C; Q_ABS and EXH remain fixed",
        },
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print(f"saved {args.out}")
    print("full-data:", json.dumps(baseline_summary, sort_keys=True))
    for n, row in sampling.items():
        print(f"n={n:>2} acc={row['accuracy_mean']:.3f} "
              f"f1={row['macro_f1_mean']:.3f} "
              f"flip={row['flip_rate_vs_full_mean']:.3f}")
    for scale, row in threshold.items():
        print(f"scale={scale} acc={row['accuracy']:.3f} "
              f"f1={row['macro_f1']:.3f} "
              f"flip={row['flip_rate_vs_full']:.3f}")


if __name__ == "__main__":
    main()
