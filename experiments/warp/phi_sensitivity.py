"""Threshold sensitivity for the warp predicate (reviewer request).

The paper classifies an M*-lock as a warp at foreign-evidence share
phi >= 0.8.  This script recomputes the two strata --- warp share and
completion within each stratum --- at thresholds phi in {0.5..0.95}
from the raw per-lock records of the W1 run (tmp/warp/w1_gain/
w1_rows.jsonl: 2,400 episodes, 23k locks, continuous phi per lock),
grouped by cadence K.  The claim under test: the stratification is a
property of provenance, not of the 0.8 cutoff --- at every threshold
the foreign stratum completes far below the own stratum, and the warp
share falls with K.

Usage::

    PYTHONPATH=. .venv/bin/python experiments/warp/phi_sensitivity.py
"""

from __future__ import annotations

import json
from collections import defaultdict

ROWS = "tmp/warp/w1_gain/w1_rows.jsonl"
OUT = "tmp/warp/phi_sensitivity.json"
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def main() -> None:
    # per-lock records: (K, phi, completed) --- peer episodes only,
    # full-visibility arm (mask_foreign False), matching the paper's
    # strata analysis
    locks = []
    with open(ROWS) as f:
        for line in f:
            ep = json.loads(line)
            if ep.get("architecture") != "peer" or ep.get("mask_foreign"):
                continue
            k = ep.get("broadcast_every_k")
            for ev in ep.get("events", []):
                if ev.get("phi") is None:
                    continue
                locks.append((k, float(ev["phi"]),
                              bool(ev.get("completed"))))
    ks = sorted({k for k, _, _ in locks})
    print(f"{len(locks)} locks over K={ks}")

    out = {}
    hdr = f"{'thr':>5} {'K':>3} {'share':>6} {'P(C|W)':>7} " \
          f"{'P(C|own)':>8} {'n_W':>6} {'n_own':>6}"
    print(hdr)
    qualitative_ok = True
    for thr in THRESHOLDS:
        out[str(thr)] = {}
        for k in ks:
            grp = [(p, c) for kk, p, c in locks if kk == k]
            warp = [c for p, c in grp if p >= thr]
            own = [c for p, c in grp if p < thr]
            share = len(warp) / len(grp) if grp else 0.0
            pcw = sum(warp) / len(warp) if warp else float("nan")
            pco = sum(own) / len(own) if own else float("nan")
            out[str(thr)][str(k)] = {
                "warp_share": round(share, 4),
                "p_C_given_W": round(pcw, 4) if warp else None,
                "p_C_given_own": round(pco, 4) if own else None,
                "n_W": len(warp), "n_own": len(own)}
            print(f"{thr:>5} {k:>3} {share:>6.3f} {pcw:>7.3f} "
                  f"{pco:>8.3f} {len(warp):>6} {len(own):>6}")
            if warp and own and not (pcw < pco):
                qualitative_ok = False
        # share must fall with K at this threshold
        shares = [out[str(thr)][str(k)]["warp_share"] for k in ks]
        if not all(shares[i] >= shares[i + 1] - 0.03
                   for i in range(len(shares) - 1)):
            print(f"  note: share not strictly falling at thr={thr}: "
                  f"{shares}")

    out["verdict"] = {
        "foreign_stratum_below_own_at_every_threshold_and_K":
            qualitative_ok}
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[{'PASS' if qualitative_ok else 'FAIL'}] stratification "
          f"holds at every threshold; saved {OUT}")


if __name__ == "__main__":
    main()
