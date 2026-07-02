# Minimal Collective Semantic Memory — acceptance status

A first instantiation of *collective semantic memory* — the engineering
target this paper sets up. Built as a peer-broadcast architecture (K=8)
with three explicit rules:

- **Merge**: trust-weighted, staleness-discounted majority vote on
  cell-tag confidence at each candidate place.
- **Trust**: per-peer trust score updated by EMA from retrieval-success
  consistency (peers whose snapshots match the local locked target
  gain weight).
- **Staleness**: each broadcast snapshot carries a tick stamp;
  weight decays exponentially with age (rate α = 0.05/tick).

## Result: acceptance PASS

3{,}240-run sweep across the full multi-agent scarcity scenario:
- $N \in \{3,5,8\}$ agents, $T \in \{2,3,5\}$ targets ($N>T$)
- 3 layouts (symmetric/asymmetric/random)
- 3 hazard densities (0.0/0.05/0.10)
- 20 seeds per cell
- 6 architectures: peer-broadcast at $K \in \{1,4,8,16,64\}$ plus CSM

| Architecture | $P(M^\star\,\vert\,R^\star)$ | $P(C^\star\,\vert\,M^\star)$ | **Success rate** (95% bootstrap CI) |
|---|---|---|---|
| peer-K1       | 0.991 | 0.586 | 0.577 [0.572, 0.586] |
| peer-K4       | 0.955 | 0.623 | 0.581 [0.572, 0.590] |
| peer-K8       | 0.730 | 0.871 | 0.599 [0.591, 0.607] |
| peer-K16      | 0.697 | 0.892 | 0.597 [0.589, 0.606] |
| peer-K64      | 0.684 | 0.900 | 0.601 [0.594, 0.609] |
| **CSM**       | **0.782** | **0.825** | **0.617 [0.610, 0.624]** |

**CSM strictly dominates ALL 5 fixed-cadence peer architectures on
success rate** (CSM's lower CI bound exceeds every peer-K's upper CI
bound). Improvements:

- vs K=1:  +0.040 success (0.617 vs 0.577)
- vs K=4:  +0.036 success
- vs K=8:  +0.019 success
- vs K=16: +0.020 success
- vs K=64: +0.017 success

## Interpretation

On the M×C trade-off plane, CSM sits at $(P(M^\star|R^\star),
P(C^\star|M^\star)) = (0.78, 0.83)$ — between the fast-broadcast
M-saturation regime (K≤4, M=0.95+, C<0.65) and the slow-broadcast
C-saturation regime (K≥16, M<0.70, C>0.89). No fixed cadence reaches
the (0.78, 0.83) point; the CSM achieves it by trading off
*adaptively* via trust × staleness on the same K=8 backbone.

The framework's measurement instrument (Q/R/M/C conditional rates)
detected the improvement: CSM didn't beat fixed-K on either M or C in
isolation; it occupied a different point in M×C space that nonetheless
yields higher end-to-end success.

## Reproduce

```bash
python -m experiments.collective_semantic_memory.smoke_test
python -m experiments.collective_semantic_memory.run_csm_vs_peer
```

Wall-clock: ~5 minutes on a single CPU core for the 3,240-run sweep.

## Artefacts

```
experiments/collective_semantic_memory/
├── csm_memory.py           # CSMMemory class (merge / trust / staleness)
├── smoke_test.py           # 5-seed sanity check
├── run_csm_vs_peer.py      # 3240-run CSM vs peer-K sweep
└── RESULTS.md              # this file
tmp/csm_vs_peer/runs.csv    # raw 3240 runs
tmp/csm_vs_peer/summary.json
```

## Paper impact

The §5 "A minimal collective semantic memory" section now contains:
- The three-rule design space (merge/trust/staleness)
- The CSM specification (10 lines)
- The 3,240-run dominance result
- The honest scope: this is a **minimal** prototype that demonstrates
  the framework's measurement protocol applied to a new architecture.
  Richer CSM variants (selective consolidation, intent-weighted trust,
  adaptive cadence $K_t$) remain future work.

The title now justifiably mentions "with a Minimal Collective Semantic
Memory" because such an artefact is delivered and measured.
