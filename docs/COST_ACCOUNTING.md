# Full cost accounting (Stage 9)

Closes claim-matrix debt A13: distinguish the *pure song codec* from
the *complete protocol wire cost*, and give the Pareto view so the
method's advantage is honest under maintenance overhead.

## 9.1 Pure codec vs full protocol (per transferred record)

Codec constants (`songlines/record.py`): tag key 6 bits, length
field 4, beat 10; certificate 64, provenance 48, time/version 32.

| Quantity | Bits | % of snapshot (5690) |
|---|---|---|
| **Pure song codec** (nodes + edges only) | 366 | **6.4%** |
| + certificate (utility profile, support, uncertainty) | +64 | |
| + provenance (origin, uid, immutable episode ref) | +48 | |
| + time/version | +32 | |
| **Full protocol payload** | **510** | **9.0%** |

Corrected paper wording (now in Part III §anatomy): *"the relational
song itself uses 6.4% of snapshot bits; the complete protocol —
adding certificate, provenance and world-version metadata — uses
9.0%."* Still an order of magnitude below the full snapshot, and the
metadata is exactly what buys admission, audit and staleness.

## 9.2 Runtime cost proxies

Measured per policy in the drivers: `match_ops` (alignment latency
proxy), `memory_bits` (incl. quarantine + immutable layer),
`wire_bits` (full codec), duplicate/contention events. Full
wall-clock/token profiling of the LLM arms (L1) is per-model and
reported there; the deterministic arms are CPU-seconds-scale.

Notable runtime finding (already in Part V §acceptance): on the
continuous substrate the **unconsolidated raw baseline is
computationally intractable** at the full horizon — O(E²) matching
over an ever-growing store — a cost the byte accounting alone misses.

## 9.3 Pareto view (team utility × memory × wire × safety)

The advantage must survive being plotted, not reduced to one scalar.
Filled from the unified equal-budget benchmark
(`exp_b_unified`, 6 policies × 12 test seeds × 300 episodes); numbers
land when the cluster run completes.

Unified benchmark (6 policies × **30 test seeds (100–129)** × 300
episodes × 6 agents; Stage-8 rerun, was 12 seeds; `phantom` =
phantom-first-visit under drift, NOT the safety-critical transport
fail-open of N1v2/C1.4):

| Policy | team cost ↓ | success ↑ | phantom ↓ | memory (KB) ↓ | wire (KB) ↓ |
|---|---:|---:|---:|---:|---:|
| independent | 136.9 | 0.03 | 0.76 | 16.7 | 0 |
| decision_centric (DeMem) | 164.5 | 0.05 | 0.89 | 18.2 | 24.9 |
| execution_path (Mage) | 180.6 | 0.05 | 0.90 | 34.7 | 44.2 |
| graph_memory (RIR) | 181.2 | 0.01 | 0.14 | 35.4 | 37.1 |
| learned_formation (Mem-α−) | 170.8 | 0.05 | 0.90 | 21.8 | 22.4 |
| **songline_full** | **135.1** | 0.03 | 0.73 | 44.2 | 48.9 |

**BU.1 PASS (30 seeds):** songline_full beats the best direct baseline
(decision_centric, 164.5) by **17.9%** on team cost, **30/30 paired
seeds**, two-sided sign test **p=1.9e-9**, world-block bootstrap 95% CI
on the paired team-cost gap **[28.1, 30.7]** (disjoint from 0), and
beats independent by 1.4% (20/30 seeds). Every communicating baseline is
*worse than independent* — the S1 lesson generalises: exchange
without an admission contract is net-harmful at horizon, and the
modern methods (decision-distortion merge, execution-path sharing,
RIR retrieval, learned formation) all lack it.

**Honest Pareto reading (not a free lunch):** songline wins team cost
and is safer (lower phantom) than 3 of 4 baselines; graph_memory buys
the lowest phantom (0.14) but at the highest team cost (181 — it
refuses/mis-retrieves conservatively). songline spends MORE memory
and wire (44/49 KB vs 17–35/22–44) — the metadata that buys
admission and audit. The claim is a superior Pareto point (cost +
safety), paid for in bits, not a dominance on every axis. B1 already
showed the bit deficit is not what holds raw back: capping raw to
4× the budget does not close the cost gap.

---
**Claim matrix update:** A13 now reads *We show* (pure 6.4% / full
9.0%), no longer *open*.
