# distributed_memory

Per-agent isolated memory + cross-agent consensus fusion.

This is **Variant C** in the per-agent / shared trade-off space:
- **Variant A** (lightweight): per-agent views over a shared event bus
- **Variant B** (middle): per-agent memory + gossip protocol
- **Variant C — this package**: full per-agent isolation + explicit consensus layer

Each agent owns a private `AgentMemory` (event store, concept graph, optional semantic field).
Agents never read each other's memory directly. Cross-agent integration
happens only through `ConsensusLayer.merge()` operating on snapshots.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      DistributedRuntime                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │ AgentMemory │  │ AgentMemory │  │ AgentMemory │   ...        │
│  │  scout-A    │  │  scout-B    │  │  scout-C    │              │
│  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │              │
│  │ │event log│ │  │ │event log│ │  │ │event log│ │              │
│  │ │concept  │ │  │ │concept  │ │  │ │concept  │ │              │
│  │ │graph    │ │  │ │graph    │ │  │ │graph    │ │              │
│  │ │(opt)    │ │  │ │(opt)    │ │  │ │(opt)    │ │              │
│  │ │field    │ │  │ │field    │ │  │ │field    │ │              │
│  │ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │              │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘              │
│         │ snapshot()     │ snapshot()     │ snapshot()          │
│         ▼                ▼                ▼                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    ConsensusLayer                        │   │
│  │   spatial alignment → trust-weighted fusion              │   │
│  │   → disagreement detection → DistributedConcept[]        │   │
│  └──────────────────────────────────────────────────────────┘   │
│         │                                                       │
│         ▼                                                       │
│   ConsensusReport (queryable)                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Files

| File | Purpose |
|---|---|
| `consensus_types.py` | Datatypes: `AgentMemoryView`, `DistributedConcept`, `AgentContribution`, `AgentDisagreement`, `ConsensusReport` |
| `trust_model.py` | `TrustModel` — scalar trust per agent with EMA outcome updates |
| `agent_memory.py` | `AgentMemory` — single-agent wrapper around Phase 1-4 primitives |
| `disagreement.py` | `detect_pairwise_disagreements`, `agreement_score` |
| `consensus_layer.py` | `ConsensusLayer.merge()` — pool + cluster + aggregate + flag |
| `distributed_runtime.py` | `DistributedRuntime` — orchestrates tick / observe / query |

## Quickstart

```python
from distributed_memory import DistributedRuntime

rt = DistributedRuntime(env_id="my-env")
rt.spawn_agent("scout-A", trust=1.0)
rt.spawn_agent("scout-B", trust=0.9)

# Agents observe (private to each agent's memory)
rt.observe("scout-A", place_key=(3, 4),
           semantic_tags={"water_source": 0.95},
           episode_id=1, step_idx=0)
rt.observe("scout-B", place_key=(3, 4),
           semantic_tags={"water_source": 0.92},
           episode_id=1, step_idx=0)

# Refresh local graphs + run consensus
report = rt.tick()

# Local query — only this agent's observations
local = rt.local_query("scout-A", "water_source", top_k=3)

# Collective query — fused across all agents
collective = rt.collective_query("water_source", top_k=3)
```

## Consensus algorithm

`ConsensusLayer.merge(views)` does:

1. **Pool**: collect all `local_concepts` from all `AgentMemoryView` snapshots.
2. **Cluster across agents** (union-find):
   Two local concepts from *different* agents merge into the same consensus
   cluster iff:
   - their centroids are within `consensus_radius` (default 4.0)
   - AND either (same dominant tag) OR (cosine similarity of semantic_profile ≥ threshold)
   Concepts from the same agent never merge — that agent's own clustering is respected.
3. **Aggregate** each cluster:
   - **centroid**: trust-weighted mean of contributing centroids
   - **semantic_profile**: trust × log(1+support)-weighted mean, normalized
   - **dominant_tag**: argmax of consensus profile
   - **confidence**: trust-weighted mean of local confidences
4. **Detect disagreement**: scan pairs of contributions for incompatible
   dominant tags using the Phase 3 `ConflictRuleSet`. Compute severity
   from trust × profile mass on conflicting tags.
5. **Score**: `consensus_confidence = mean_conf × agreement × multi_agent_factor`
   - `agreement ∈ [0, 1]` decreases with severity of pairwise disagreements
   - `multi_agent_factor = min(1.0, sqrt(n_agents / n_total))` rewards
     concepts confirmed by many agents

## Key design choices

**No shared event bus.** Each `AgentMemory` has its own `CollectiveMemory`.
Compare to `songline_drive/collective_memory.py` where all agents publish to
the same bus.

**Consensus is stateless.** `ConsensusLayer.merge()` is a pure function of
input snapshots. No internal accumulator. Callers persist reports themselves.

**Trust is scalar.** No pairwise trust, no Bayesian machinery. One number
per agent in `[trust_min, trust_max]`, updated by `update_from_outcome()`
with an EMA step. Easy to inspect and tune.

**Provenance preserved.** Every `DistributedConcept` carries the list of
`AgentContribution` objects — you always know which agent claimed what.

**Phase 3 compatibility.** The same `ConflictRuleSet` that powers
intra-concept tag conflict in Phase 3 also drives inter-agent disagreement
detection here. Single source of truth for "incompatible tags".

**Optional Phase 4 field per agent.** Set `enable_field=True` in the
runtime constructor — each agent gets its own `SemanticField` for local
reranking. Field is *not* aggregated by the consensus layer (yet); it's
purely a per-agent acceleration. Cross-agent field fusion is a possible
extension.

## When to use this vs Phase 1-4

| Scenario | Use Phase 1-4 collective | Use distributed_memory |
|---|---|---|
| Single trusted environment, all agents see everything | ✓ | |
| Need provenance per observation | ✓ | ✓ |
| Need trust weighting | | ✓ |
| Need partial observability (each agent has limited view) | | ✓ |
| Need to detect inter-agent disagreement explicitly | | ✓ |
| Want belief revision when peer concepts arrive | | ✓ |
| Production multi-agent system with untrusted sources | | ✓ |

## Experiments

See `experiments/distributed_memory/` for runnable demonstrations:

| Experiment | What it shows |
|---|---|
| `exp01_basic_per_agent.py` | Two agents observe different places → private graphs, no leakage |
| `exp02_consensus_alignment.py` | Two agents see same place with same tags → 1 aligned consensus concept |
| `exp03_disagreement.py` | Two agents disagree on tag → flagged, agreement drops, confidence drops |
| `exp04_trust_weighted_fusion.py` | High-trust majority wins; flip trusts → consensus flips |
| `exp05_partial_observability.py` | Each agent sees one region; consensus exposes all regions to all agents |

Run all:

```bash
for f in experiments/distributed_memory/exp*.py; do
  PYTHONPATH=. .venv/bin/python "$f"
done
```
