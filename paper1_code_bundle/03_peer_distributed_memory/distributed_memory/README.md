# Distributed memory experiments

Runnable demonstrations of the `distributed_memory` package
(per-agent isolation + consensus fusion).

Each experiment:
- Is a standalone script
- Builds a controlled scenario (no randomness)
- Prints `✓ ExpNN passed ...` on success
- Writes a JSON summary to `tmp/distributed_expNN/`
- Hard-asserts the key claim

## Running

Single experiment:
```bash
PYTHONPATH=. .venv/bin/python experiments/distributed_memory/exp01_basic_per_agent.py
```

All experiments:
```bash
for f in experiments/distributed_memory/exp*.py; do
  PYTHONPATH=. .venv/bin/python "$f"
done
```

## What each experiment proves

### exp01_basic_per_agent.py — graph isolation
Two agents observe two different places. After local refresh:
- Each agent has exactly 1 concept
- Centroids are at *different* coordinates (different places)
- Each agent's event count is exactly what it observed (no leakage)

**Key result**: `scout-A: 1 concept @ (0.0, 0.0)  scout-B: 1 concept @ (9.0, 7.0)`

### exp02_consensus_alignment.py — happy-path consensus
Two agents observe the *same* place with compatible tags + each has private places.
- 1 distributed concept aligns both agents at the shared place
- 2 isolated distributed concepts for the private places
- Aligned concept has agreement ≈ 1.0

**Key result**: `aligned=1  isolated=2  shared_centroid=(3.0, 4.0)  agreement=1.000`

### exp03_disagreement.py — inter-agent conflict
Two agents observe the same place but tag it incompatibly (water vs hazard).
- Consensus still merges them spatially
- `disagreement_flags` is non-empty
- `inter_agent_agreement` drops near 0
- `consensus_confidence` drops to ≈ 0

**Key result**: `agreement=0.000  flags=scout-A:water_source vs scout-B:hazard_edge`

### exp04_trust_weighted_fusion.py — trust-driven majority
Three agents disagree. Two with high trust claim water, one with low trust claims hazard.
- Consensus picks water (majority trust mass wins)
- Flip trusts → consensus flips to hazard
- All three agents always appear in `contributions` (provenance preserved)

**Key result**: `scenario_a: tag=water_source (water=1.000, hazard=0.020)  |  scenario_b: tag=hazard_edge`

### exp05_partial_observability.py — knowledge transfer
Three agents each see one different region. Locally, each only knows its own.
After consensus tick, all three regions are visible via `collective_query`.
- Each agent's `local_query` returns 1 concept (local knowledge only)
- `n_aligned=0` (regions are spatially disjoint)
- `n_isolated=3` (all three regions surface in consensus)
- `collective_query` returns all 3 regions to any caller

**Key result**: `each agent locally sees 1 concept  collective exposes 3 regions`

## Cumulative claim

Together these 5 experiments demonstrate the full Variant C contract:

1. **Privacy** — each agent has an isolated graph (exp01)
2. **Alignment** — compatible peer observations fuse (exp02)
3. **Conflict surfacing** — incompatible peer observations are flagged, not silently averaged (exp03)
4. **Trust weighting** — consensus rotates with trust mass (exp04)
5. **Emergent collective knowledge** — locally-private becomes globally-accessible (exp05)
