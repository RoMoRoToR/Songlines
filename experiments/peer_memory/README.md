# Peer-to-peer memory experiments

Runnable demonstrations of `peer_memory/` (peer-to-peer collective
memory without a central aggregator).

## Run all

```bash
for f in experiments/peer_memory/exp*.py; do
  PYTHONPATH=. .venv/bin/python "$f"
done
```

## Experiments

### `exp01_basic_broadcast.py` — basic protocol works
Two agents in different regions broadcast periodically. After one
broadcast cycle, each agent's *private* `peer_view` contains the
other's observations.

**Key result**: `A sees 2 concepts (heard from B); B sees 2 concepts (heard from A)` — two SEPARATE peer_views.

### `exp02_asymmetric_trust.py` — divergent beliefs from divergent trust
Both agents observe the SAME place but tag it differently
(water vs hazard). Trust is asymmetric: A trusts B at 0.9, B trusts A
at 0.2. The two agents' peer_views diverge accordingly.

**Key result**: `A_view hazard_share=0.474 > B_view water_share=0.167` — same world, different beliefs because of trust asymmetry.

### `exp03_three_way_ablation.py` — the reviewers' triad
The headline comparison: independent vs centralized vs peer-to-peer
on the same scenario (three agents, three regions, knowledge propagation).

**Key result**:
```
mode             avg_cov/3   msgs   per-agent
independent           1.00      0   N=1  E=1  S=1
centralized           3.00      6   N=3  E=3  S=3
peer                  3.00      6   N=3  E=3  S=3
```

- Independent: each agent knows only its own region (1/3)
- Centralized: all agents query the SAME central report (3/3) — single point of failure
- Peer: each agent has its OWN merged view (3/3) — no central layer

Same coverage as centralized **without** a `ConsensusLayer` or shared report.

## Mapping to reviewers' taxonomy

| Reviewers' label | Our package | Experiment |
|---|---|---|
| (1) Fully independent agents | `distributed_memory.AgentMemory` used standalone | exp03 independent branch |
| (2) Agents with a center | `distributed_memory.ConsensusLayer` | exp03 centralized branch |
| (3) Agents with communication | `peer_memory/` (this package) | exp01, exp02, exp03 peer branch |
