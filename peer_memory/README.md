# peer_memory

Peer-to-peer collective memory. Each agent maintains its own private
graph + its own private merged view. Inter-agent contact happens only
through a passive `BroadcastBus` (in-process message router).

**There is no central aggregator.** No `ConsensusLayer`, no global
report, no shared trust table. Compare to `distributed_memory/`
(Variant C) which has the same per-agent isolation but adds a central
`ConsensusLayer` to fuse snapshots — this package replaces that
centraliser with peer-to-peer gossip.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                       PeerRuntime                       │
│   (passive scheduler — no aggregation)                  │
│                                                         │
│  ┌────────────┐   ┌────────────┐   ┌────────────┐       │
│  │ PeerAgent  │   │ PeerAgent  │   │ PeerAgent  │       │
│  │  scout-A   │   │  scout-B   │   │  scout-C   │       │
│  │ ┌────────┐ │   │ ┌────────┐ │   │ ┌────────┐ │       │
│  │ │AgentMem│ │   │ │AgentMem│ │   │ │AgentMem│ │       │
│  │ │trust[]│  │   │ │trust[]│  │   │ │trust[]│  │       │
│  │ │PeerView│ │   │ │PeerView│ │   │ │PeerView│ │       │
│  │ └────────┘ │   │ └────────┘ │   │ └────────┘ │       │
│  └─────┬──────┘   └─────┬──────┘   └─────┬──────┘       │
│        │                │                │              │
│        └────────────────┼────────────────┘              │
│                         │                               │
│                  ┌──────▼──────┐                        │
│                  │BroadcastBus │ (transport only,       │
│                  │ no logic)   │  delivers messages)    │
│                  └─────────────┘                        │
└─────────────────────────────────────────────────────────┘
```

Each `PeerAgent` owns:
- An `AgentMemory` (private event store + concept graph)
- An `AsymmetricTrust` table (`trust[peer_id] -> [0,1]`, may differ across agents)
- A `_last_known` cache of the last snapshot received from each peer
- Its own `PeerView` produced by its OWN merge

Two agents starting from identical observations can produce different
`PeerView`s if their trust tables differ. There is no notion of "the
consensus" — every agent has its own consensus.

## Key files

| File | Purpose |
|---|---|
| `peer_types.py` | `BroadcastMessage`, `PeerInbox`, `PeerView`, `PeerMergeReport` |
| `peer_trust.py` | `AsymmetricTrust` — per-owner, per-peer trust with EMA updates |
| `broadcast_bus.py` | Passive in-process message router. No business logic. |
| `peer_merge.py` | `local_merge()` — pure function each agent calls on itself |
| `peer_agent.py` | `PeerAgent` — wraps everything + broadcast/receive/merge lifecycle |
| `peer_runtime.py` | `PeerRuntime` — scheduling sugar; no aggregation |

## Quickstart

```python
from peer_memory import PeerRuntime

rt = PeerRuntime(env_id="my-env", broadcast_every_k=3)
rt.spawn_agent("scout-A")
rt.spawn_agent("scout-B")

# Asymmetric trust (optional — defaults to 0.7 symmetrically)
rt.agent("scout-A").trust.set("scout-B", 0.9)
rt.agent("scout-B").trust.set("scout-A", 0.2)

# Each agent observes privately
rt.observe("scout-A", (3, 4), {"water_source": 0.9},
           episode_id=1, step_idx=0)

# Tick: rebuild local graphs, maybe broadcast, merge own + received
for _ in range(6):
    rt.tick()

# Query EACH AGENT'S PRIVATE VIEW — no global query exists
view_a = rt.agent("scout-A").peer_view
view_b = rt.agent("scout-B").peer_view
# view_a and view_b may differ
```

## Periodic broadcast protocol

Each tick:
1. Every agent calls `refresh_local()` (private graph rebuild)
2. Every agent increments its internal step counter
3. If `tick_count % broadcast_every_k == 0`, every agent calls `broadcast_now()`
4. Every agent calls `process_inbox_and_merge()`:
   - Drains new messages from its `PeerInbox`
   - Updates its `_last_known[peer_id]` cache (newer overwrites older)
   - Runs `local_merge(own_snapshot, all_last_known_messages, own_trust)`
   - Stores result as new `PeerView`

The `_last_known` cache is what makes the view stable between
broadcasts: an agent doesn't "forget" peer information on a non-broadcast tick.

## Comparison with other packages

| Property | `songline_drive/` | `distributed_memory/` | `peer_memory/` |
|---|---|---|---|
| Event store | shared | per-agent | per-agent |
| Concept graph | shared | per-agent | per-agent |
| Aggregation | implicit (one graph) | central `ConsensusLayer` | per-agent `local_merge` |
| Trust | not modelled | global scalar | per-agent asymmetric |
| Merged view | one (the graph) | one (`ConsensusReport`) | N (one per agent) |
| Single point of failure | yes | yes (Runtime + ConsensusLayer) | no |
| Communication pattern | shared substrate | central aggregator | peer-to-peer broadcast |
| Matches reviewers' taxonomy | (2) "agents with a center" | (2) "agents with a center" | (3) "agents with communication" |

## Verifying there is no centralisation

Delete `peer_runtime.py`. All other code still works:

```python
from peer_memory import BroadcastBus, PeerAgent

bus = BroadcastBus()
a = PeerAgent("A", bus); b = PeerAgent("B", bus)
a.observe((1,1), {"water_source": 0.9}, episode_id=1, step_idx=0)
a.refresh_local(); a.tick_step(); a.broadcast_now()
b.refresh_local(); b.tick_step(); b.process_inbox_and_merge()
print(b.peer_view.contributing_peer_ids)  # ['A']
```

`PeerRuntime` is pure scheduling sugar — it owns no state that
aggregates across agents.

## Experiments

See `experiments/peer_memory/`:

| Experiment | What it shows |
|---|---|
| `exp01_basic_broadcast.py` | Two agents in different regions → after broadcast each sees both |
| `exp02_asymmetric_trust.py` | Same observations + different trust → different peer_views |
| `exp03_three_way_ablation.py` | independent vs centralized vs peer (the reviewers' triad) |
