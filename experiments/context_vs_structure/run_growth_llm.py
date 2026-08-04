"""
Experiment 3 LLM probe: large multi-agent systems -- raw logs vs Songlines map.

Three agents explore a shared world over G sessions, each logging observations
in its OWN PRIVATE coordinate frame (secret offsets). Same model, two arms:

  raw   -- typical usage: all three agents' raw logs concatenated (frames
          unaligned, and the prompt says so). Grows ~linearly with G x N.
  graph -- the Songlines machinery aligns frames (fingerprint consensus,
          align_frames) and merges evidence into ONE coherent map in agent-0's
          frame; the model gets the compact map. Grows with alignment quality.

Global questions (per checkpoint G):
  count -- "how many DISTINCT water sources exist?" (cross-agent dedup: the
           same water appears at different private coords in different logs).

Concept prediction (crossing curves): graph accuracy RISES with G (alignment
locks as fingerprints accumulate -- the Exp-3 exact% curve), raw accuracy FALLS
with G (log volume + frame confusion). Deterministic given seeds.

Smoke: PYTHONPATH=. python experiments/context_vs_structure/run_growth_llm.py \
   --backend ollama --models llama3.1:latest --checkpoints 1 2 --seeds 2
Cluster: --backend hf --models Qwen/Qwen2.5-7B-Instruct --checkpoints 1 2 4 8 --seeds 10
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

from experiments.warp.semantic_identity import align_frames, fingerprint

W, H, NAG, EPS = 26, 20, 3, 0.6
SYSTEM = ("You answer questions about exploration logs from multiple robots. "
          "Follow the exact answer format requested.")


def gen_world(seed):
    rng = np.random.default_rng(seed)
    waters = []
    while len(waters) < 3:
        w = (int(rng.integers(2, W - 2)), int(rng.integers(2, H - 2)))
        if w not in waters:
            waters.append(w)
    hazards = set()
    while len(hazards) < int(0.08 * W * H):
        h = (int(rng.integers(0, W)), int(rng.integers(0, H)))
        if h not in waters:
            hazards.add(h)
    offs = {i: (int(rng.integers(-9, 10)), int(rng.integers(-9, 10)))
            for i in range(NAG)}
    return waters, hazards, offs, rng


def tag_at(xy, waters, hazards):
    if xy in waters:
        return "water"
    if xy in hazards:
        return "hazard"
    return "empty"


def obs_cells(xy, waters, hazards):
    """radius-2 diamond of (true_xy, tag)."""
    out = []
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            if abs(dx) + abs(dy) > 2:
                continue
            p = (xy[0] + dx, xy[1] + dy)
            if 0 <= p[0] < W and 0 <= p[1] < H:
                out.append((p, tag_at(p, waters, hazards)))
    return out


def sweep(agent, g, waters, hazards, off, rng, store, log, seen_w):
    y0 = int(rng.integers(0, H - 5))
    t = 0
    for y in range(y0, y0 + 5):
        xs = range(W) if y % 2 == 0 else range(W - 1, -1, -1)
        for x in xs:
            cells = obs_cells((x, y), waters, hazards)
            pxy = (x + off[0], y + off[1])
            pcells = [{"xy": (p[0] + off[0], p[1] + off[1]), "tag":
                       ("water_source" if tg == "water" else
                        "hazard_edge" if tg == "hazard" else "open")}
                      for p, tg in cells]
            store[pxy] = fingerprint(pxy, pcells)
            here = tag_at((x, y), waters, hazards)
            log.append(f"agent{agent} s{g} t={t}: at ({pxy[0]},{pxy[1]}): {here}")
            if here == "water":
                seen_w.add((x, y))
            t += 1


def build_graph_map(stores, seen_w_priv, offs):
    """Songlines arm: align each agent's frame to agent-0, transport water
    sightings, dedup within EPS. Returns (map text, waters in a0 frame,
    aligned_count)."""
    a0 = []
    for w in seen_w_priv[0]:
        a0.append((w[0] + offs[0][0], w[1] + offs[0][1]))
    aligned = 0
    for j in range(1, NAG):
        res = align_frames(stores[0], stores[j])
        if res.offset is None:
            continue
        aligned += 1
        for w in seen_w_priv[j]:
            pj = (w[0] + offs[j][0], w[1] + offs[j][1])
            a0.append((pj[0] + res.offset[0], pj[1] + res.offset[1]))
    # dedup
    uniq: List[Tuple[int, int]] = []
    for w in a0:
        if not any(abs(w[0] - u[0]) <= EPS and abs(w[1] - u[1]) <= EPS for u in uniq):
            uniq.append(w)
    txt = ["SONGLINES GLOBAL MAP (all agents merged, agent0 coordinates):"]
    for (x, y) in sorted(uniq):
        txt.append(f"- water_source at ({x},{y})")
    txt.append(f"(frames aligned: {aligned}/{NAG-1} peers; duplicates merged)")
    return "\n".join(txt), uniq, aligned


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["ollama", "hf"], default="hf")
    ap.add_argument("--models", nargs="+", default=["Qwen/Qwen2.5-7B-Instruct"])
    ap.add_argument("--checkpoints", nargs="+", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out_dir", default="tmp/cluster/growth_llm")
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
        for seed in range(a.seeds):
            waters, hazards, offs, rng = gen_world(seed)
            stores = {i: {} for i in range(NAG)}
            logs = {i: [] for i in range(NAG)}
            seen_w = {i: set() for i in range(NAG)}   # true coords per agent
            g = 0
            for G in sorted(a.checkpoints):
                while g < G:
                    for i in range(NAG):
                        sweep(i, g, waters, hazards, offs[i],
                              np.random.default_rng(seed * 1000 + g * 10 + i),
                              stores[i], logs[i], seen_w[i])
                    g += 1
                n_true = len({w for i in range(NAG) for w in seen_w[i]})
                if n_true == 0:
                    continue
                gmap, uniq, aligned = build_graph_map(stores, seen_w, offs)
                raw_body = ("Logs from 3 robots. IMPORTANT: each robot uses its "
                            "OWN private coordinate frame (unknown offsets "
                            "between frames).\n"
                            + "\n".join(logs[0]) + "\n" + "\n".join(logs[1])
                            + "\n" + "\n".join(logs[2]))
                for arm, body in (("raw", raw_body), ("graph", gmap)):
                    # Q1: count distinct waters (observed so far)
                    q1 = (body + "\n\nQUESTION: How many DISTINCT water sources "
                          "have been observed in total (the same physical water "
                          "seen by two robots must be counted ONCE)? "
                          "Reply exactly: COUNT: n")
                    o1 = backend.complete(q1, system=SYSTEM,
                                          seed=seed * 100 + G, max_tokens=16)
                    m = re.search(r"COUNT:\s*(\d+)", o1)
                    acc1 = int(m is not None and int(m.group(1)) == n_true)
                    rows.append({"model": model, "seed": seed, "G": G,
                                 "arm": arm, "q": "count", "acc": acc1,
                                 "prompt_chars": len(q1), "out": o1[:40]})
                    print(f"[{model}] G={G} {arm:>5} count: acc={acc1} "
                          f"({len(q1)//1000}k ch)", flush=True)
        path = os.path.join(a.out_dir, f"runs_{slug}.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        summ = {}
        for G in sorted(a.checkpoints):
            for arm in ("raw", "graph"):
                g_ = [r for r in rows if r["G"] == G and r["arm"] == arm]
                if g_:
                    summ[f"{arm}_G{G}"] = {
                        "acc": sum(r["acc"] for r in g_) / len(g_),
                        "prompt_chars": int(np.mean([r["prompt_chars"] for r in g_]))}
        summ["backend"] = backend.summary()
        with open(os.path.join(a.out_dir, f"summary_{slug}.json"), "w") as f:
            json.dump(summ, f, indent=2)
        print(json.dumps({k: v for k, v in summ.items() if k != "backend"},
                         indent=2))
        if hasattr(backend, "close"):
            backend.close()


if __name__ == "__main__":
    main()
