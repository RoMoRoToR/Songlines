# VMAS continuous-substrate portability test (reviewer point #1)

**Question.** Do the M↔C bottleneck-shift slopes of Empirical Claim 1
(`P(M*|R*)` falls, `P(C*|M*)` rises as cadence `K` grows) survive when the
**same** symbolic peer memory + planner run on a *continuous-state* substrate
(VMAS) instead of a grid? This is the reviewer's top request: at least one
result outside grid-world.

**Design.** We do **not** change the memory stack. A thin bridge
(`continuous_bridge.py`) discretises continuous `(x,y)` into coarse grid cells
and tags cells near a water landmark `water_source`, so `build_memory("peer", …)`
and the operational Q/R/M/C definitions apply unchanged. The controller acts in
continuous space toward the materialised waypoint. Only the substrate is
continuous; everything symbolic is reused.

## Status — preliminary positive

- `continuous_bridge.py` — **done**, unit-tested (`python continuous_bridge.py`).
- `run_vmas_portability.py` — **runnable**, executes end-to-end on VMAS (custom
  `WaterSearch` scenario: N agents + T water landmarks, holonomic continuous
  dynamics), reuses the symbolic peer memory, logs Q/R/M/C, sweeps `K`.
- **Both slope signs of Empirical Claim 1 reproduce** on the continuous
  substrate in the coupling regime. With `N=8, T=3, max_steps=25, 40 seeds`
  over `K ∈ {1,2,4,8,16}`:

  | K | P(M*\|R*) | P(C*\|M*) |
  |---|-----------|-----------|
  | 1  | 1.000 | 0.182 |
  | 2  | 0.583 | 0.305 |
  | 4  | 0.448 | 0.405 |
  | 8  | 0.305 | 0.608 |
  | 16 | 0.278 | 0.639 |

  `Spearman(P(M*|R*), K) = -1.00`, `Spearman(P(C*|M*), K) = +1.00` — exactly the
  predicted directions (M falls, C rises as broadcast slows). Beyond `K≈32` the
  system enters the independent regime (rare broadcast → no contention → M
  rebounds), the same regime boundary seen on the grid.

### Honest caveats (why this is preliminary, not headline)
- Single scenario config (`N=8, T=3`), short horizon, no cluster-robust CIs yet.
- Memory is fed via the discretisation **bridge**, so this is "continuous
  dynamics + discretised symbolic memory," not fully continuous memory.
- The M-slope needs the occupancy-sensitive materialisation rule (a candidate
  whose target is already claimed does not lock) — faithful to the grid planner,
  but the effect only appears once that rule is in place.
- To promote to a headline result: scale to ≥3 scarcity cells × layouts × ≥40
  seeds, add cluster-robust CIs (reuse `analyze_effect_sizes.py`), and vary the
  scenario to show the signs are not config-specific.

Reproduce: `PYTHONPATH=. python experiments/vmas_portability/run_vmas_portability.py --Ks 1 2 4 8 16 --n_agents 8 --n_waters 3 --seeds 40 --max_steps 25`

## Install & run

```bash
pip install vmas            # pulls torch (already present), gym, pyglet
PYTHONPATH=. python experiments/vmas_portability/run_vmas_portability.py --smoke
# larger:
PYTHONPATH=. python experiments/vmas_portability/run_vmas_portability.py \
    --Ks 1 4 16 64 --n_agents 8 --n_waters 2 --seeds 12 --max_steps 120
```

## Knobs that matter for reproducing the shift

- `SENSE_R` (cell sensing radius): large → retrieval saturates (`P(M*|R*)≈1`,
  no M-slope). Small → memory freshness matters and the M-slope can appear.
- scarcity `n_agents > n_waters`: needed for the C-side occupancy contention
  that produces the falling `P(C*|M*)` at fast cadence.
- `--seeds`: the grid result used 40; a few seeds is not enough to resolve a
  moderate slope (see the cluster-robust analysis in the main paper).

## Next steps to turn this into a paper result

1. Tune `SENSE_R`, world size, and `(N,T)` so both `P(M*|R*)` and `P(C*|M*)`
   are away from saturation at the baseline cadence.
2. Scale the sweep (≥20 seeds × scarcity cells) and compute cluster-robust CIs
   with the same block bootstrap as `experiments/big_experiment/analyze_effect_sizes.py`.
3. If the slope signs match (`∂P(M*|R*)/∂K<0`, `∂P(C*|M*)/∂K>0`), add a
   VMAS row to the portability paragraph; if not, report the divergence
   honestly (it would bound the claim to discrete substrates).
