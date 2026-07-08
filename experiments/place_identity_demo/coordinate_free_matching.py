"""
Coordinate-free place correspondence across agents (Paper B, coordinate-anchor removal).

Question the collective-memory paper defers: when agents do NOT share a global
(x,y) frame, merge cannot be built directly -- correspondence (which of my
places = which of yours) must be recovered first. We test this quantitatively.

Setup
-----
- T ground-truth places, each with a latent semantic profile.
- N agents. Each agent observes a random subset of the places and reports, per
  place, a (raw_key, tag_profile). Two frame regimes:
    * shared_frame=True  : all agents' raw_key = true position + small noise
                           (the paper's current assumption).
    * shared_frame=False : each agent applies its OWN private rigid transform
                           (rotation + translation) to positions -> global keys
                           are meaningless across agents. Semantic profiles are
                           frame-invariant; noisy/partial.

Two measurements
----------------
1. Cross-agent matching quality (pairwise precision/recall/F1 vs ground truth)
   for COORDINATE_ONLY / SEMANTIC_ONLY / HYBRID.
2. Alignment defect = the categorical adjunction round-trip error. For an agent
   pair (A,B): F maps each A-place to its best B-match, G the reverse; the
   defect is the fraction of A-places with a true B-correspondent for which
   G(F(x)) != x (the eta_X : X -> G(F(X)) loop is non-identity). This is the
   forwarded theorem's defect made numeric -- a computable indicator of
   correspondence quality, i.e. what the categorical model lets you *compute*.

Deterministic given --seed. Reuses songline_drive.place_identity unchanged.
"""
from __future__ import annotations
import argparse, math, os, sys
from itertools import combinations
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np

from songline_drive.place_identity import (
    IdentityMode, PlaceIdentityEngine, PlaceObservation,
)

TAG_VOCAB = ["water_source", "wet", "rest_area", "shelter", "open",
             "passable", "hazard_edge", "rocky", "vegetation", "shade"]


def _make_truth(T: int, rng) -> List[Dict[str, float]]:
    """T latent places, each a sparse profile over the vocab (distinct dominant tags)."""
    profiles = []
    for t in range(T):
        prof = {}
        dom = TAG_VOCAB[t % len(TAG_VOCAB)]
        prof[dom] = float(rng.uniform(0.80, 0.95))
        for tag in rng.choice(TAG_VOCAB, size=3, replace=False):
            prof.setdefault(tag, float(rng.uniform(0.25, 0.55)))
        profiles.append(prof)
    return profiles


def _rigid(pos, theta, tx, ty):
    x, y = pos
    return (math.cos(theta) * x - math.sin(theta) * y + tx,
            math.sin(theta) * x + math.cos(theta) * y + ty)


def _observe(profiles, T, N, shared_frame, prof_noise, prof_keep, pos_noise, rng):
    """Return observations tagged with (agent, true_place) ground truth."""
    true_pos = [(float(rng.uniform(0, 12)), float(rng.uniform(0, 10))) for _ in range(T)]
    obs, truth = [], []
    for a in range(N):
        # private frame if not shared
        if shared_frame:
            theta, tx, ty = 0.0, 0.0, 0.0
        else:
            theta = float(rng.uniform(-math.pi, math.pi)); tx = float(rng.uniform(-8, 8)); ty = float(rng.uniform(-8, 8))
        seen = rng.choice(T, size=max(2, int(round(T * rng.uniform(0.6, 1.0)))), replace=False)
        for t in seen:
            px, py = _rigid(true_pos[t], theta, tx, ty)
            px += float(rng.normal(0, pos_noise)); py += float(rng.normal(0, pos_noise))
            prof = dict(profiles[t])
            # noise + partial view
            prof = {k: min(1.0, max(0.0, v + float(rng.normal(0, prof_noise)))) for k, v in prof.items()}
            keep = max(1, int(round(len(prof) * prof_keep)))
            prof = dict(sorted(prof.items(), key=lambda kv: -kv[1])[:keep])
            obs.append(PlaceObservation(agent_id=f"ag{a}", raw_key=(px, py),
                                        position_sigma=pos_noise + 0.5,
                                        tag_profile=prof, env_id="cf"))
            truth.append((a, int(t)))
    return obs, truth


def _pairwise_prf(pred_labels, truth):
    """Pairwise precision/recall/F1 over CROSS-agent observation pairs."""
    tp = fp = fn = 0
    for i, j in combinations(range(len(truth)), 2):
        if truth[i][0] == truth[j][0]:
            continue  # only cross-agent pairs matter for correspondence
        same_true = truth[i][1] == truth[j][1]
        same_pred = pred_labels[i] == pred_labels[j]
        if same_pred and same_true: tp += 1
        elif same_pred and not same_true: fp += 1
        elif not same_pred and same_true: fn += 1
    prec = tp / (tp + fp) if tp + fp else 1.0
    rec = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def _best_match(engine, obs, src_idx, dst_indices):
    """F/G component: best same-place match of obs[src_idx] among dst_indices, or None."""
    best, best_s = None, -1.0
    for j in dst_indices:
        s = engine.score_pair(obs[src_idx], obs[j])
        if s.is_same and s.combined_score > best_s:
            best, best_s = j, s.combined_score
    return best


def _alignment_defect(engine, obs, truth):
    """Mean over ordered agent pairs of the eta_X : X -> G(F(X)) round-trip
    failure rate, restricted to A-places that truly have a B-correspondent
    (the categorical adjunction defect made numeric)."""
    by_agent: Dict[str, List[int]] = {}
    for idx, (a, _) in enumerate(truth):
        by_agent.setdefault(a, []).append(idx)
    true_place = {idx: truth[idx][1] for idx in range(len(truth))}
    agents = sorted(by_agent)
    defects = []
    for a in agents:
        for b in agents:
            if a == b:
                continue
            A, B = by_agent[a], by_agent[b]
            b_places = {true_place[j] for j in B}
            broken = total = 0
            for x in A:
                if true_place[x] not in b_places:
                    continue  # no true correspondent -> not counted
                total += 1
                fx = _best_match(engine, obs, x, B)          # F: A -> B
                if fx is None:
                    broken += 1; continue
                gfx = _best_match(engine, obs, fx, A)        # G: B -> A
                if gfx is None or true_place[gfx] != true_place[x]:
                    broken += 1  # eta_X is not identity: round-trip lands elsewhere
            if total:
                defects.append(broken / total)
    return float(np.mean(defects)) if defects else float("nan")


def _run_cell(mode, shared_frame, prof_noise, prof_keep, pos_noise, T, N, seed):
    rng = np.random.default_rng(seed)
    profiles = _make_truth(T, rng)
    obs, truth = _observe(profiles, T, N, shared_frame, prof_noise, prof_keep, pos_noise, rng)
    engine = PlaceIdentityEngine(mode=mode, identity_threshold=0.55,
                                 semantic_weight=0.6, spatial_weight=0.4,
                                 spatial_sigma_scale=1.5)
    labels = engine.resolve_identities(obs)          # dict idx -> canonical_id
    pred = [labels[i] for i in range(len(obs))]
    prec, rec, f1 = _pairwise_prf(pred, truth)
    defect = _alignment_defect(engine, obs, truth)
    return prec, rec, f1, defect


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=6)
    ap.add_argument("--N", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=20)
    args = ap.parse_args()
    modes = [IdentityMode.COORDINATE_ONLY, IdentityMode.SEMANTIC_ONLY, IdentityMode.HYBRID]

    def avg(mode, shared, pn, pk, posn):
        rows = [_run_cell(mode, shared, pn, pk, posn, args.T, args.N, s) for s in range(args.seeds)]
        arr = np.array(rows)  # cols: prec,rec,f1,defect
        return arr.mean(axis=0)

    print(f"Coordinate-free place correspondence  (T={args.T} places, N={args.N} agents, {args.seeds} seeds)\n")
    for shared in (True, False):
        tag = "SHARED frame (paper's assumption)" if shared else "NO shared frame (private per-agent frames)"
        print(f"=== {tag} ===")
        print(f"{'mode':>16} | {'precision':>9} {'recall':>7} {'F1':>6} | {'align-defect':>12}")
        for m in modes:
            p, r, f1, d = avg(m, shared, 0.08, 0.8, 0.15)
            print(f"{m.value:>16} | {p:>9.3f} {r:>7.3f} {f1:>6.3f} | {d:>12.3f}")
        print()

    print("=== Alignment defect vs profile noise (HYBRID, no shared frame) ===")
    print(f"{'prof_noise':>10} | {'F1':>6} | {'align-defect':>12}")
    for pn in [0.02, 0.08, 0.15, 0.25, 0.40]:
        _, _, f1, d = avg(IdentityMode.HYBRID, False, pn, 0.8, 0.4)
        print(f"{pn:>10.2f} | {f1:>6.3f} | {d:>12.3f}")
    print("\nReading: COORDINATE_ONLY collapses without a shared frame (F1 low, defect high);")
    print("semantic/HYBRID signature matching recovers correspondence; the alignment defect")
    print("(eta_X round-trip failure rate) rises with noise and tracks matching degradation --")
    print("a computable categorical indicator, not a re-description.")


if __name__ == "__main__":
    main()
