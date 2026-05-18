# independent_memory

Variant (1) from the reviewers' taxonomy: **fully independent agents**
with no mechanism to exchange information.

Each `IndependentAgent` has its own private event store and concept
graph (same low-level machinery as the other agent flavours).  The
difference is what is **deliberately absent**:

- No `snapshot()` export — calling it raises `RuntimeError`
- No `broadcast()` / `receive()` methods — both raise
- The runtime has no `collective_query`, no `consensus`, no `bus`

This is the controlled **lower-bound baseline**: it shows what an
agent can achieve using only its own observations.  Compare against:

- `distributed_memory/` (variant 2 mid) — central `ConsensusLayer`
- `peer_memory/` (variant 3) — peer-to-peer broadcast
- `songline_drive/` (variant 2 max) — shared event bus

## Why a dedicated package

For symmetry with the other two implementations and so the "no
communication" baseline is **explicit and uncircumventable**.  Using
`AgentMemory` standalone (as we did initially) works, but is implicit
— it would be easy to accidentally introduce aggregation without
noticing.  This package makes the contract explicit at the API level.

## Quickstart

```python
from independent_memory import IndependentRuntime

rt = IndependentRuntime(env_id="my-env")
rt.spawn_agent("scout-A")
rt.spawn_agent("scout-B")

rt.observe("scout-A", (3, 4), {"water_source": 0.9},
           episode_id=1, step_idx=0)

for _ in range(5):
    rt.tick()   # each agent independently refreshes its private graph

# The ONLY query method is local_query — there is no collective query.
results = rt.local_query("scout-A", "water_source", top_k=3)
```

## Files

| File | Purpose |
|---|---|
| `independent_agent.py` | `IndependentAgent` — wraps a private graph, blocks communication methods |
| `independent_runtime.py` | `IndependentRuntime` — spawn + step; no aggregation |

## Experiments

See `experiments/independent_memory/`:

| Experiment | What it shows |
|---|---|
| `exp01_isolation.py` | Three agents in three regions → each knows only its own region; communication methods raise |

This package's main role is as the `independent` arm of the headline
3-way ablation in `experiments/peer_memory/exp03_three_way_ablation.py`.
