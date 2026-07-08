"""
Symbol (Sigma) alignment: aligning tag alphabets across agents through shared
referents -- the next frontier after coordinate-frame recovery (Section 6).

Setup. Two agents describe the SAME places but with DIFFERENT tag alphabets:
agent B's alphabet is a secret relabelling (and, in the hard variant, a
coarsening) of agent A's, with observation noise. Place correspondence (the
anchor set) is what Section 6 already recovers; here we isolate the symbol
step. Symbols are aligned by GROUNDING: on anchored (matched) places, the
co-occurrence of an A-tag and a B-tag is evidence they mean the same feature.

Categorical object (Barwise-Seligman information flow / Chu). The recovered
alignment is an infomorphism (f: places_A->places_B, fhat: Sigma_B->Sigma_A)
with the biconditional  I_B(q,y) <-> I_A(p, fhat(y))  on anchors. The
Sigma-alignment defect is the fraction of anchored (place, B-tag) pairs where
that biconditional fails -- the symbol analogue of the frame adjunction defect.

Arms:
  naive     -- assume the alphabets coincide by index ("same tags"): breaks
               under relabelling (fails open: confident wrong translation).
  recovered -- fhat learned from co-occurrence over anchors.
  oracle    -- the true secret map (upper bound).

Deterministic given --seed. Pure numpy.
Run: PYTHONPATH=. .venv/bin/python experiments/warp/symbol_alignment.py
"""
from __future__ import annotations
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np


def gen(T, KA, noise, coarsen, anchor_frac, seed):
    """Return incidence matrices I_A (T x KA), I_B (T x KB), the ground-truth
    B->A map (list of sets), and the anchor mask over places."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.25, 0.6, size=KA)            # per-feature base rate
    IA = (rng.random((T, KA)) < base).astype(int)
    # ensure the target feature (index 0) actually occurs
    IA[rng.choice(T, size=max(3, T // 4), replace=False), 0] = 1

    perm = rng.permutation(KA)                        # secret relabelling B-index -> A-feature
    if coarsen:
        # merge two A-features into one B-tag: B has KA-1 tags
        merge_a, merge_b = perm[0], perm[1]
        KB = KA - 1
        gt = []          # gt[b] = set of A-features that B-tag b covers
        IB = np.zeros((T, KB), dtype=int)
        bi = 0
        used = set()
        for a in perm:
            if a in used:
                continue
            if a in (merge_a, merge_b):
                cover = {merge_a, merge_b}; used |= cover
            else:
                cover = {a}; used.add(a)
            gt.append(cover)
            col = (IA[:, list(cover)].max(axis=1))
            IB[:, bi] = col; bi += 1
        KB = bi
        IB = IB[:, :KB]
    else:
        KB = KA
        gt = [{int(perm[b])} for b in range(KB)]      # bijection
        IB = np.zeros((T, KB), dtype=int)
        for b in range(KB):
            IB[:, b] = IA[:, perm[b]]
    # observation noise on B
    flip = rng.random(IB.shape) < noise
    IB = IB ^ flip.astype(int)
    anchors = rng.random(T) < anchor_frac
    return IA, IB, gt, anchors, perm


def recover_map(IA, IB, anchors):
    """fhat[b] = A-tag most associated with B-tag b over anchored places
    (cosine of incidence columns)."""
    A, B = IA[anchors], IB[anchors]
    KA, KB = A.shape[1], B.shape[1]
    fhat = np.zeros(KB, dtype=int)
    for b in range(KB):
        vb = B[:, b].astype(float)
        best, bs = 0, -1.0
        for a in range(KA):
            va = A[:, a].astype(float)
            na, nb = np.linalg.norm(va), np.linalg.norm(vb)
            c = (va @ vb) / (na * nb) if na > 0 and nb > 0 else 0.0
            if c > bs:
                bs, best = c, a
        fhat[b] = best
    return fhat


def infomorphism_defect(IA, IB, fhat, anchors):
    """Fraction of anchored (place, B-tag) pairs violating I_B(q,y) == I_A(p, fhat(y))."""
    A, B = IA[anchors], IB[anchors]
    viol = (B != A[:, fhat])
    return float(viol.mean())


def target_f1(IA, IB, b_pred, anchors):
    """F1 of identifying the target feature (A-index 0) from B's evidence via
    the arm's believed B-tag b_pred, over anchored places."""
    if b_pred is None or b_pred >= IB.shape[1]:
        return 0.0
    true = IA[anchors, 0]
    pred = IB[anchors, b_pred]
    tp = int(((pred == 1) & (true == 1)).sum()); fp = int(((pred == 1) & (true == 0)).sum())
    fn = int(((pred == 0) & (true == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    return 2 * p * r / (p + r) if p + r else 0.0


def run_cell(noise, coarsen, anchor_frac, T, KA, seed):
    IA, IB, gt, anchors, perm = gen(T, KA, noise, coarsen, anchor_frac, seed)
    fhat = recover_map(IA, IB, anchors)
    KB = IB.shape[1]
    # translation accuracy: fhat[b] in ground-truth cover of b
    acc = float(np.mean([fhat[b] in gt[b] for b in range(KB)]))
    dfS = infomorphism_defect(IA, IB, fhat, anchors)
    # target = A-feature 0. Which B-tag do the arms believe means it?
    b_oracle = next((b for b in range(KB) if 0 in gt[b]), None)             # true
    b_recov = next((b for b in range(KB) if fhat[b] == 0), None)            # learned
    b_naive = 0                                                             # "same tags": B-index 0
    f1o = target_f1(IA, IB, b_oracle, anchors)
    f1r = target_f1(IA, IB, b_recov, anchors)
    f1n = target_f1(IA, IB, b_naive, anchors)
    return acc, dfS, f1n, f1r, f1o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=120); ap.add_argument("--KA", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=20)
    a = ap.parse_args()

    def avg(noise, coarsen, af):
        r = np.array([run_cell(noise, coarsen, af, a.T, a.KA, s) for s in range(a.seeds)])
        return r.mean(axis=0)  # acc, def, f1_naive, f1_recov, f1_oracle

    print(f"Symbol (Sigma) alignment via grounding on shared places "
          f"(T={a.T} places, |Sigma_A|={a.KA}, {a.seeds} seeds)\n")
    print("=== Relabelling only (bijection), vs observation noise; anchor_frac=0.6 ===")
    print(f"{'noise':>6} | {'transl.acc':>10} {'def_Sigma':>10} | "
          f"{'F1 naive':>9} {'F1 recov':>9} {'F1 oracle':>9}")
    for nz in [0.0, 0.05, 0.10, 0.20, 0.30]:
        acc, df, fn, fr, fo = avg(nz, False, 0.6)
        print(f"{nz:>6.2f} | {acc:>10.2f} {df:>10.3f} | {fn:>9.2f} {fr:>9.2f} {fo:>9.2f}")

    print("\n=== Anchor-set size (relabelling, noise=0.10) ===")
    print(f"{'anchor%':>7} | {'transl.acc':>10} {'def_Sigma':>10} {'F1 recov':>9}")
    for af in [0.15, 0.30, 0.60, 1.0]:
        acc, df, fn, fr, fo = avg(0.10, False, af)
        print(f"{af:>7.0%} | {acc:>10.2f} {df:>10.3f} {fr:>9.2f}")

    print("\n=== Coarsening variant (two A-features -> one B-tag, noise=0.10, anchor=0.6) ===")
    acc, df, fn, fr, fo = avg(0.10, True, 0.6)
    print(f"transl.acc={acc:.2f}  def_Sigma={df:.3f}  F1: naive={fn:.2f} recov={fr:.2f} oracle={fo:.2f}")

    print("\nReading: 'naive' (assume identical alphabets) collapses on target ID because")
    print("B relabelled the tags; co-occurrence over anchored places RECOVERS the map")
    print("(transl.acc high, def_Sigma low) and recovers oracle-level target ID. def_Sigma")
    print("rises with noise and as anchors shrink -- the infomorphism witness, analogue of")
    print("the frame adjunction defect. Coarsening bounds accuracy (no faithful bijection).")


if __name__ == "__main__":
    main()
