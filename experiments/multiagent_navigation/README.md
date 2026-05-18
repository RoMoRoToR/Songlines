# Multi-agent navigation experiments

End-to-end navigation experiments using `multiagent_env/` (custom multi-agent
grid world) + `songline_drive/` (Phase 1-4 collective memory).

## Why custom env (not BabyAI)

BabyAI / MiniGrid envs are framework-level single-agent — `agent_pos` is
a scalar, not a list. Building a true multi-agent variant required:
- Two agents tracked in one grid
- Joint step() with collision rules
- Local semantic observations from each agent's POV

This is implemented in `multiagent_env/grid_world.py` as
`MultiAgentGridWorld`.

## Experiments

### `exp_field_modes_comparison.py`

**Scenario**: Two agents both need water. Two clean water cells exist on
the grid (well separated, both valid). Agents start in opposite corners.

**Comparison**: Same setup × 3 field modes × N seeds.

| Mode | Mechanism |
|---|---|
| `descriptive` | Phase 2/3 only — concept recall without field reranking |
| `read_only` | Phase 4b — field reranks candidates |
| `coordinated` | Phase 4c — agents reserve their target; reservation penalises others' queries |

**Metrics**:
- `success_rate_both` — fraction of seeds where BOTH agents reach water
- `mean_episode_steps` — steps to terminate (lower = faster)
- `duplicate_initial_target_rate` — fraction of seeds where both agents pick the same target at t=0
- `mean_hazard_hits` — cells with `hazard_edge` tag traversed

**Result (10 seeds, step_limit=80)**:

| mode | success_both | mean_steps | dup_target | hazard_hits |
|---|---|---|---|---|
| descriptive | 0.00 | 80.0 | 1.00 | 2.00 |
| read_only   | 0.00 | 80.0 | 1.00 | 2.00 |
| coordinated | 1.00 | 21.0 | 0.00 | 4.00 |

**Interpretation**:
- `descriptive` and `read_only` both fail because both agents pick the same water cell. One reaches it; the other gets stuck (the occupied cell blocks movement).
- `coordinated` succeeds 100%: agent-A reserves water-A → agent-B's field query sees water-B as top-1 → both agents head to different targets from t=0.
- Coordinated agents traverse more total cells (higher hazard hits) but finish in 1/4 the time.

## Run

```bash
PYTHONPATH=. .venv/bin/python experiments/multiagent_navigation/exp_field_modes_comparison.py \
    --n_seeds 10 --step_limit 80 --out_dir tmp/multiagent_nav_modes
```
