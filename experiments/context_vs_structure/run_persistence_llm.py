"""
Experiment 2 (concept-aligned): long-lived context across sessions --
raw replay vs the Songlines graph, SAME model answering in both arms.

A persistent, slowly drifting world is explored session by session (partial
band sweeps). After each session the model is asked: "where is the water NOW?"

  raw_replay -- typical LLM usage: the FULL concatenated history of all session
                transcripts is re-injected into the prompt every session.
                Prompt grows linearly; freshness conflicts pile up.
  songlines  -- the same transcripts are consolidated into the graph-semantic
                store (latest state per place, staleness explicit); the model
                gets the compact graph dump. Prompt stays O(salient places).

Metrics per (arm, session): answer accuracy vs CURRENT water; prompt size.
Concept prediction: raw accuracy decays with session count while its prompt
explodes; songlines stays flat on both.

Run (local smoke): PYTHONPATH=. python experiments/context_vs_structure/run_persistence_llm.py \
    --backend ollama --models llama3.1:latest --sessions 6 --seeds 3
Cluster: --backend hf --models Qwen/Qwen2.5-7B-Instruct --sessions 8 --seeds 12
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from experiments.context_vs_structure.run_ctx_vs_struct import parse_target

W, H = 24, 18
SYSTEM = ("You answer questions about a robot's exploration history. "
          "Reply with EXACTLY one line: TARGET: x,y (integer grid coordinates).")


BAND_STARTS = [0, 5, 10, 13]   # rotating bands: full-grid coverage every 4 sessions


def session_lines(session: int, water, hazards, rng, r0: int = 0) -> List[str]:
    """One session = a 5-row band sweep; bands ROTATE so any drifted water is
    re-observed within <=4 sessions (stale periods in between are the point)."""
    y0 = BAND_STARTS[(session + r0) % len(BAND_STARTS)]
    lines = []
    t = 0
    for y in range(y0, y0 + 5):
        xs = range(W) if y % 2 == 0 else range(W - 1, -1, -1)
        for x in xs:
            tag = "empty"
            if (x, y) in hazards:
                tag = "hazard"
            if (x, y) == water:
                tag = "water"
            lines.append(f"s{session} t={t}: at ({x},{y}): {tag}")
            t += 1
    return lines


def consolidate(all_lines: List[str]) -> str:
    state: Dict[Tuple[int, int], Tuple[int, str]] = {}
    order = 0
    for ln in all_lines:
        m = re.match(r"s(\d+) t=(\d+): at \((\d+),(\d+)\): (.+)", ln)
        if not m:
            continue
        x, y, tag = int(m.group(3)), int(m.group(4)), m.group(5)
        state[(x, y)] = (order, f"s{m.group(1)}", tag)
        order += 1
    waters = [(xy, sid) for xy, (o, sid, tag) in state.items() if tag == "water"]
    hazards = [xy for xy, (o, sid, tag) in state.items() if tag == "hazard"]
    n_empty = sum(1 for v in state.values() if v[2] == "empty")
    out = ["SONGLINES MEMORY (graph, latest state per place):"]
    for (x, y), sid in sorted(waters, key=lambda w: -int(w[1][1:])):
        out.append(f"- water_source at ({x},{y})  [last confirmed session {sid}]")
    for (x, y) in sorted(hazards)[:8]:
        out.append(f"- hazard at ({x},{y})")
    out.append(f"- {n_empty} explored places currently empty (omitted)")
    return "\n".join(out)


def run_seed(backend, arm: str, S: int, p_move: float, seed: int) -> List[Dict]:
    rng = np.random.default_rng(seed)
    water = (int(rng.integers(1, W - 1)), int(rng.integers(1, H - 1)))
    hazards = set()
    while len(hazards) < 8:
        h = (int(rng.integers(0, W)), int(rng.integers(0, H)))
        if h != water:
            hazards.add(h)
    history: List[str] = []
    rows = []
    for s in range(S):
        if s > 0 and rng.random() < p_move:
            water = (int(rng.integers(1, W - 1)), int(rng.integers(1, H - 1)))
        history.extend(session_lines(s, water, hazards, rng, r0=seed % 4))
        body = ("FULL EXPLORATION HISTORY (all sessions, chronological):\n"
                + "\n".join(history)) if arm == "raw_replay" else consolidate(history)
        prompt = (body + "\n\nQUESTION: Where is the water NOW "
                  "(most recent state across all sessions)?"
                  "\nReply exactly one line: TARGET: x,y")
        out = backend.complete(prompt, system=SYSTEM, seed=seed * 100 + s,
                               max_tokens=24)
        tgt = parse_target(out)
        # correct iff matches current water AND the water has actually been
        # observed at its current site by now (else unanswerable -> skip row)
        seen_current = any(f"at ({water[0]},{water[1]}): water" in ln
                           for ln in history)
        if not seen_current:
            continue
        acc = int(tgt is not None and tgt == water)
        rows.append({"arm": arm, "session": s, "seed": seed, "acc": acc,
                     "prompt_chars": len(prompt), "hist_lines": len(history),
                     "raw_out": out[:40]})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["ollama", "hf"], default="hf")
    ap.add_argument("--models", nargs="+", default=["Qwen/Qwen2.5-7B-Instruct"])
    ap.add_argument("--sessions", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--p_move", type=float, default=0.35)
    ap.add_argument("--out_dir", default="tmp/cluster/persist_llm")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    for model in a.models:
        slug = model.replace("/", "_").replace(":", "_")
        cache = os.path.join(a.out_dir, slug + ".cache")
        if a.backend == "hf":
            from experiments.llm_collective.hf_backend import HFBackend
            backend = HFBackend(model=model, cache_dir=cache)
        else:
            from experiments.llm_collective.llm_backend import OllamaBackend
            backend = OllamaBackend(model=model, cache_dir=cache)
        rows: List[Dict] = []
        for arm in ("raw_replay", "songlines"):
            for seed in range(a.seeds):
                rows.extend(run_seed(backend, arm, a.sessions, a.p_move, seed))
            print(f"[{model}] arm={arm} done", flush=True)
        path = os.path.join(a.out_dir, f"runs_{slug}.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\n{'arm':>10} | " + " ".join(f"s={s}" for s in range(a.sessions))
              + "   <- accuracy per session")
        summ = {}
        for arm in ("raw_replay", "songlines"):
            accs, sizes = [], []
            for s in range(a.sessions):
                g = [r for r in rows if r["arm"] == arm and r["session"] == s]
                accs.append(sum(r["acc"] for r in g) / len(g) if g else float("nan"))
                sizes.append(int(np.mean([r["prompt_chars"] for r in g])) if g else 0)
            print(f"{arm:>10} | " + " ".join(f"{v:.2f}" for v in accs))
            print(f"{'chars':>10} | " + " ".join(f"{v//1000}k" for v in sizes))
            summ[arm] = {"acc_by_session": accs, "prompt_chars": sizes}
        summ["backend"] = backend.summary()
        with open(os.path.join(a.out_dir, f"summary_{slug}.json"), "w") as f:
            json.dump(summ, f, indent=2)
        if hasattr(backend, "close"):
            backend.close()


if __name__ == "__main__":
    main()
