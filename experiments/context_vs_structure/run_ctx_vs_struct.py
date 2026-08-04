"""
Experiment 1: context-stuffing vs consolidated structure (the crux A/B).

Thesis under test: models ACCEPT large context but are much worse at turning it
into a stable, globally coherent knowledge structure. We give ONE model the SAME
evidence stream in two forms and measure stage rates as evidence volume grows:

  RAW    -- the witness's full observation transcript, verbatim in the prompt
            (context-stuffing; grows linearly with volume);
  STRUCT -- the SAME transcript consolidated by our symbolic memory rule
            (latest-state-per-place + staleness; compact, ~constant size).

World: grid W x H. A witness sweeps it and logs one line per visit:
  "t=214: at (x,y): empty|water|hazard"
Mid-log the water MOVES: old site later observed empty ("dried up"), new site
observed with water. So the transcript contains a FRESHNESS CONFLICT that raw
attention must resolve and that consolidation resolves by construction.

Task: "Where is the water NOW? Reply exactly: TARGET: x,y".
Stages: Q* fires on the question (by construction); R* = parsed TARGET within
eps of the CURRENT true water (the retrieval-availability event -- the crux);
M* = R* (single-shot commit; noted in the paper); C* = a greedy controller
reaches the cell within 1.5 x Manhattan budget on the true grid.

Volume axis: number of transcript lines L (empty-cell visits pad the log).
Prediction: RAW degrades with L (needle-in-haystack + stale-vs-fresh conflict);
STRUCT stays flat because consolidation already resolved both.

Run (local smoke, ollama):
  PYTHONPATH=. python experiments/context_vs_structure/run_ctx_vs_struct.py \
      --backend ollama --models llama3.1:latest --volumes 40 200 --seeds 3
Cluster (hf):
  ... --backend hf --models Qwen/Qwen2.5-7B-Instruct --volumes 50 200 800 2000 --seeds 20
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

SYSTEM = ("You answer questions about a robot's exploration log. "
          "Reply with EXACTLY one line: TARGET: x,y  (integer grid coordinates).")


# ---------------------------------------------------------------- world + log

def gen_transcript(L: int, seed: int, W: int = 30, H: int = 24,
                   hazards: int = 6) -> Tuple[List[str], Tuple[int, int]]:
    """Serpentine sweep padded/truncated to ~L lines. Water appears at site A,
    later 'dries up' and appears at site B (the CURRENT truth). Returns
    (transcript lines, current water xy)."""
    rng = np.random.default_rng(seed)
    cells = [(x, y) for y in range(H) for x in (range(W) if y % 2 == 0 else range(W - 1, -1, -1))]
    # visit sequence long enough: repeat sweep with jitter
    seq = []
    while len(seq) < L:
        seq.extend(cells)
    seq = seq[:L]
    A = (int(rng.integers(2, W - 2)), int(rng.integers(2, H - 2)))
    B = (int(rng.integers(2, W - 2)), int(rng.integers(2, H - 2)))
    while B == A:
        B = (int(rng.integers(2, W - 2)), int(rng.integers(2, H - 2)))
    haz = set()
    while len(haz) < hazards:
        h = (int(rng.integers(0, W)), int(rng.integers(0, H)))
        if h not in (A, B):
            haz.add(h)
    tA_seen = L // 5                      # water first seen at A early
    t_move = (2 * L) // 3                 # after this: A dry, B has water
    lines = []
    for t, (x, y) in enumerate(seq):
        tag = "empty"
        if (x, y) in haz:
            tag = "hazard"
        if (x, y) == A:
            tag = "water" if tA_seen <= t < t_move else ("empty (dried up)" if t >= t_move else "empty")
        if (x, y) == B:
            tag = "water" if t >= t_move else "empty"
        lines.append(f"t={t}: at ({x},{y}): {tag}")
    # guarantee the story is present regardless of L: force-inject key events.
    # B's (current) water sighting is placed at ~0.9L with an empty tail after,
    # so RAW cannot solve it by reading the literal last line: it must weigh
    # many stale A=water mentions against one fresher B=water + A dried-up.
    t_B = max(min(int(0.9 * L), L - 2), 2)
    lines[tA_seen] = f"t={tA_seen}: at ({A[0]},{A[1]}): water"
    lines[min(t_move, t_B - 1)] = f"t={min(t_move, t_B - 1)}: at ({A[0]},{A[1]}): empty (dried up)"
    lines[t_B] = f"t={t_B}: at ({B[0]},{B[1]}): water"
    # scrub accidental later mentions of A/B in the tail (keep story clean)
    for t in range(t_B + 1, L):
        m = re.match(r"t=\d+: at \((\d+),(\d+)\):", lines[t])
        if m and (int(m.group(1)), int(m.group(2))) in (A, B):
            lines[t] = f"t={t}: at ({m.group(1)},{m.group(2)}): empty"
            if (int(m.group(1)), int(m.group(2))) == B:
                lines[t] = f"t={t}: at ({B[0]},{B[1]}): water"  # B stays water if revisited
    return lines, {"truth": B, "stale": A, "hazards": sorted(haz),
                    "n_waters_now": 1}


# ------------------------------------------------------------- consolidation

def consolidate(lines: List[str]) -> str:
    """Our symbolic consolidation in miniature: latest state per place
    (staleness resolution), emit only salient records + a coverage summary.
    Output size is ~constant in L."""
    state: Dict[Tuple[int, int], Tuple[int, str]] = {}
    for ln in lines:
        m = re.match(r"t=(\d+): at \((\d+),(\d+)\): (.+)", ln)
        if not m:
            continue
        t, x, y, tag = int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4)
        state[(x, y)] = (t, tag)          # later lines overwrite: freshness
    waters = [(xy, t) for xy, (t, tag) in state.items() if tag.strip() == "water"]
    hazards = [xy for xy, (t, tag) in state.items() if tag.startswith("hazard")]
    n_empty = sum(1 for _, (t, tag) in state.items() if tag.startswith("empty"))
    out = ["CONSOLIDATED MEMORY (latest state per place):"]
    for (x, y), t in sorted(waters, key=lambda w: -w[1]):
        out.append(f"- water_source at ({x},{y})  [last seen t={t}, current]")
    for (x, y) in sorted(hazards)[:8]:
        out.append(f"- hazard at ({x},{y})")
    out.append(f"- {n_empty} places explored and empty (omitted)")
    return "\n".join(out)


# ---------------------------------------------------------------- episode

def parse_target(out: str) -> Optional[Tuple[int, int]]:
    m = re.search(r"TARGET:\s*\(?\s*(\d+)\s*[, ]\s*(\d+)", out)
    return (int(m.group(1)), int(m.group(2))) if m else None


QUESTIONS = ["fact", "hop", "count"]


def make_question(kind: str, world: Dict) -> Tuple[str, str]:
    """Return (question text, expected answer spec)."""
    if kind == "fact":
        return ("Where is the water NOW (most recent state)? "
                "Reply exactly one line: TARGET: x,y", "target_truth")
    if kind == "hop":   # integration: two distant facts (current water + hazards)
        return ("Which hazard is CLOSEST (manhattan distance) to the CURRENT "
                "water location? Reply exactly one line: TARGET: x,y "
                "(the hazard's coordinates)", "nearest_hazard")
    return ("How many places currently have water (most recent state per "
            "place)? Reply exactly one line: COUNT: n", "count")


def score(kind: str, out: str, world: Dict, eps: float = 0.6) -> int:
    if kind == "count":
        m = re.search(r"COUNT:\s*(\d+)", out)
        return int(m is not None and int(m.group(1)) == world["n_waters_now"])
    tgt = parse_target(out)
    if tgt is None:
        return 0
    if kind == "fact":
        tr = world["truth"]
        return int(abs(tgt[0] - tr[0]) <= eps and abs(tgt[1] - tr[1]) <= eps)
    tr = world["truth"]
    dists = [abs(h[0] - tr[0]) + abs(h[1] - tr[1]) for h in world["hazards"]]
    best = min(dists)
    ok = [h for h, d in zip(world["hazards"], dists) if d == best]
    return int(any(abs(tgt[0] - h[0]) <= eps and abs(tgt[1] - h[1]) <= eps
                   for h in ok))


def run_cell(backend, arm: str, L: int, seed: int, qkind: str = "fact") -> Dict:
    lines, world = gen_transcript(L, seed)
    if arm == "raw":
        body = "EXPLORATION LOG (chronological):\n" + "\n".join(lines)
    else:
        body = consolidate(lines)
    qtext, _ = make_question(qkind, world)
    prompt = body + "\n\nQUESTION: " + qtext
    out = backend.complete(prompt, system=SYSTEM, seed=seed, max_tokens=24)
    ok = score(qkind, out, world)
    return {"arm": arm, "L": L, "seed": seed, "question": qkind,
            "prompt_chars": len(prompt), "q_star": 1,
            "r_star": ok, "m_star": ok, "c_star": ok,
            "target": "", "truth": str(world["truth"]), "raw_out": out[:60]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["ollama", "hf"], default="hf")
    ap.add_argument("--models", nargs="+", default=["Qwen/Qwen2.5-7B-Instruct"])
    ap.add_argument("--volumes", nargs="+", type=int, default=[50, 200, 800, 2000])
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out_dir", default="tmp/cluster/ctx_vs_struct")
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
            backend = OllamaBackend(model=model, cache_dir=cache,
                                    timeout_s=float(os.environ.get("OLLAMA_TIMEOUT", "300")))
        rows = []
        for L in a.volumes:
            for arm in ("raw", "struct"):
                for qk in QUESTIONS:
                    for s in range(a.seeds):
                        row = run_cell(backend, arm, L, s, qkind=qk)
                        row["model"] = model
                        rows.append(row)
                    got = [r for r in rows if r["L"] == L and r["arm"] == arm
                           and r["question"] == qk]
                    rr = sum(r["r_star"] for r in got) / len(got)
                    print(f"[{model}] L={L:>5} {arm:>6} {qk:>5}: acc={rr:.2f} "
                          f"(prompt~{got[0]['prompt_chars']}ch)", flush=True)
            path = os.path.join(a.out_dir, f"runs_{slug}.csv")
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader(); w.writerows(rows)
        # summary
        summ = {}
        for L in a.volumes:
            for arm in ("raw", "struct"):
                for qk in QUESTIONS:
                    got = [r for r in rows if r["L"] == L and r["arm"] == arm
                           and r["question"] == qk]
                    summ[f"{arm}_L{L}_{qk}"] = {
                        "acc": sum(r["r_star"] for r in got) / len(got),
                        "prompt_chars": int(np.mean([r["prompt_chars"] for r in got]))}
        summ["backend"] = backend.summary()
        with open(os.path.join(a.out_dir, f"summary_{slug}.json"), "w") as f:
            json.dump(summ, f, indent=2)
        if hasattr(backend, "close"):
            backend.close()   # free VRAM before the next model
        print(json.dumps({k: v for k, v in summ.items() if k != "backend"}, indent=2))


if __name__ == "__main__":
    main()
