# Independent memory experiments

Runnable demonstrations of `independent_memory/` (variant 1 — fully
isolated agents).

## Run all

```bash
PYTHONPATH=. .venv/bin/python experiments/independent_memory/exp01_isolation.py
```

## Experiments

### `exp01_isolation.py` — isolation contract holds

Three agents in three disjoint regions, each observes only its own
water cell.  Verifies:

- Each agent's `local_query` returns exactly one concept (own region only)
- Centroids match the per-agent placement
- The runtime exposes no `collective_query`, no `consensus`, no `bus`
- Calling forbidden methods (`snapshot`, `broadcast`, `receive`) on an
  `IndependentAgent` raises `RuntimeError`

**Key result**: `N=(1.0, 1.0), E=(8.0, 1.0), S=(4.0, 6.0)` — three private graphs, no leakage.

This experiment is the **lower-bound baseline** referenced by the
headline 3-way ablation in
`experiments/peer_memory/exp03_three_way_ablation.py`.
