#!/usr/bin/env python3
"""P1 analysis on existing per-lock logs (tmp/warp/w1_gain/w1_rows.jsonl):
1. Continuous P(C | phi in bins) curve (review: don't rest on binary theta).
2. Matched own-vs-foreign comparison within covariate strata
   (distance at lock, snapshot age, co-locks) — selection-bias check.
No new runs; purely descriptive re-analysis of logged events.
"""
import json, math
from collections import defaultdict

ROWS = "/Users/taniyashuba/PycharmProjects/Songlines/tmp/warp/w1_gain/w1_rows.jsonl"

events = []
with open(ROWS) as f:
    for line in f:
        ep = json.loads(line)
        arm = ep.get("arm") or ep.get("architecture") or "?"
        for e in ep.get("events", []):
            e["_arm"] = arm
            events.append(e)

print(f"episodes parsed; total lock events: {len(events)}")
keys = sorted(events[0].keys())
print("event keys:", keys)

def rate(evs):
    n = len(evs)
    c = sum(1 for e in evs if e.get("completed"))
    return c / n if n else float("nan"), n

# ---- 1. Continuous phi curve ----
print("\n=== P(C | phi bin), pooled and per arm ===")
bins = [(0.0, 1e-9), (1e-9, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8),
        (0.8, 0.999), (0.999, 1.01)]
labels = ["=0", "(0,.2)", "[.2,.4)", "[.4,.6)", "[.6,.8)", "[.8,1)", "=1"]
arms = sorted({e["_arm"] for e in events})
hdr = "bin      " + "".join(f"{a:>18}" for a in arms) + "     pooled"
print(hdr)
for (lo, hi), lab in zip(bins, labels):
    row = f"{lab:9}"
    sel_all = [e for e in events if lo <= e["phi"] < hi]
    for a in arms:
        sel = [e for e in sel_all if e["_arm"] == a]
        r, n = rate(sel)
        row += f"  {r:6.3f} (n={n:5d})" if n else "        --       "
    r, n = rate(sel_all)
    row += f"  {r:6.3f}/{n}"
    print(row)

# ---- 2. Matched strata: distance at lock x age x co-locks ----
print("\n=== matched own (phi=0) vs foreign (phi>=0.8) within covariate strata ===")
def dist_bucket(e):
    d = e.get("warp_radius_cells")
    if d is None: return None
    return "d<=2" if d <= 2 else ("d3-6" if d <= 6 else "d>6")
def age_bucket(e):
    a = e.get("source_snapshot_age")
    if a in (None, -1): return "age:na"
    return "age<=5" if a <= 5 else "age>5"
def co_bucket(e):
    c = e.get("co_locked") or 0
    return "co=0" if c == 0 else "co>=1"

strata = defaultdict(lambda: {"own": [], "for": []})
for e in events:
    db = dist_bucket(e)
    if db is None: continue
    key = (db, co_bucket(e))
    if e["phi"] < 1e-9:
        strata[key]["own"].append(e)
    elif e["phi"] >= 0.8:
        strata[key]["for"].append(e)

print(f"{'stratum':22}{'P(C|own)':>12}{'n_own':>7}{'P(C|foreign)':>14}{'n_for':>7}{'gap':>8}")
tot_w, gap_w = 0.0, 0.0
for key in sorted(strata):
    o_r, o_n = rate(strata[key]["own"])
    f_r, f_n = rate(strata[key]["for"])
    if o_n >= 20 and f_n >= 20:
        gap = o_r - f_r
        w = min(o_n, f_n)
        tot_w += w; gap_w += gap * w
        print(f"{str(key):22}{o_r:12.3f}{o_n:7d}{f_r:14.3f}{f_n:7d}{gap:8.3f}")
print(f"\nweighted (min-n) mean within-stratum gap P(C|own)-P(C|foreign): "
      f"{gap_w/tot_w:.3f}" if tot_w else "no overlapping strata")

# crude unadjusted gap for comparison
own_all = [e for e in events if e["phi"] < 1e-9]
for_all = [e for e in events if e["phi"] >= 0.8]
ro, no = rate(own_all); rf, nf = rate(for_all)
print(f"unadjusted: P(C|own)={ro:.3f} (n={no}), P(C|foreign)={rf:.3f} (n={nf}), gap={ro-rf:.3f}")

# covariate imbalance summary
def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals)/len(vals) if vals else float("nan")
print("\n=== covariate imbalance (own vs foreign) ===")
for cov in ("warp_radius_cells", "source_snapshot_age", "co_locked"):
    mo = mean([e.get(cov) if e.get(cov) != -1 else None for e in own_all])
    mf = mean([e.get(cov) if e.get(cov) != -1 else None for e in for_all])
    print(f"{cov:22} own={mo:7.2f}  foreign={mf:7.2f}")
