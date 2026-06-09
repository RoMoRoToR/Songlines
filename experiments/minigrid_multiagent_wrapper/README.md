# MiniGrid multi-agent wrapper — acceptance checklist

Closes the "two-substrate seam" by re-running the multi-agent
water-search scarcity scenario on a MiniGrid-based substrate. Once
this passes, the Limitations rewrite drops "two environments, openly
reported" and replaces it with "the multi-agent claims replicate on a
standard MiniGrid-based substrate (§7.7)".

## Scope (deliberately minimal)

- One layout (FourRooms)
- Peer-broadcast architecture only
- N = 3, T = 2
- K ∈ {1, 4, 8, 16, 64} (5 cadences — enough to show slope + minimum)
- 20 seeds
- = 100 runs

Total wall-clock at 8 cores: ≤ 30 minutes (extrapolating from custom-env timing).

## Implementation plan

1. **env_wrapper.py** (the actual engineering work — 2-3 weeks):
   - Pick a MiniGrid layout (FourRooms-like with grid_size=17)
   - Place N agents on the grid; place T water-tagged cells
   - Per-tick: each agent emits a MiniGrid action; apply them in
     deterministic order; resolve collisions by no-op
   - Cell-tag dictionary: extend MiniGrid tags with `water_source` and
     `hazard_edge` to match the custom env vocabulary
   - Episode-end: when all agents have reached a water cell OR
     step_limit (120 ticks)

2. **PettingZoo wrapper** (optional, day 1): expose the env via
   `pettingzoo.ParallelEnv` so MARL libraries (PPO/IPPO/MAPPO) can be
   plugged in for future learned-baseline work.

3. **run_portability_sweep.py** (already scaffolded): drives the 100
   runs and runs the acceptance check (slope signs + interior minimum).

## Acceptance criteria (paste into §7.7 once they pass)

- (a) Q/R/M/C events emit non-trivially on the new substrate
      (P(Q*) > 0.9, P(R*|Q*) > 0.9; first two stages saturated as on
      the custom env)
- (b) Spearman slope signs of Proposition 3 hold:
      P(M*|R*) decreases in K (r < 0, p < 0.05)
      P(C*|M*) increases in K (r > 0, p < 0.05)
- (c) Mean t_succ has an interior minimum strictly below both
      endpoints with non-overlapping 95% bootstrap CIs

If only 2 of 3 pass: report verbatim what failed and treat it as a
partial portability result, not a portability claim.

## Paper edits if accepted

Add new subsection §7.7 "Portability to standard substrate" with:
- 1 paragraph stating scope + acceptance status
- Table: 5×3 matrix (K × [P(M|R), P(C|M), t_succ]) with 95% CIs
- Optional: 1 figure (mini version of fig_bottleneck_shift.pdf for
  the portability sweep)

Rewrite Limitations entry "Two environments, openly reported" to:
"The multi-agent claims of §7.5 replicate on a standard MiniGrid-based
substrate (§7.7) with identical operational Q/R/M/C definitions and
preserved slope signs, eliminating the cross-substrate gap."

## Fallback if the wrapper buckles by July

VMAS or Melting Pot with the same scarcity scenario, but the cost of
entry is higher (continuous state requires redefining ε in the M event
and re-validating the Q/R/M/C estimators). MiniGrid-wrapper is the
priority path.
