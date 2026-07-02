"""IndependentAgent — fully isolated agent with no inter-agent communication.

This is variant (1) from the reviewers' taxonomy:
    "полностью независимые агенты"

Each agent has its private event store and concept graph (just like
``distributed_memory.AgentMemory``), but with **no mechanism whatsoever**
to exchange information with other agents.  There is:

  - No ConsensusLayer
  - No BroadcastBus
  - No snapshot export for cross-agent merge
  - No trust table (trust is meaningless when no peers exist)

This is the **lower bound** baseline: it shows what an agent can do
using only its own observations.  Compare against:

  - ``distributed_memory/`` (variant 2 mid)  — same per-agent isolation +
                                                central aggregator
  - ``peer_memory/``        (variant 3)      — same per-agent isolation +
                                                peer-to-peer broadcast
  - ``songline_drive/``     (variant 2 max)  — agents share an event bus

We deliberately wrap ``distributed_memory.AgentMemory`` and forbid the
``snapshot()`` method by raising — to make it impossible to accidentally
construct an aggregation pipeline.  If you need snapshot semantics,
you're not building an independent agent.

Lifecycle each tick:

  1. observe(...)
  2. refresh_local()
  3. local_query(...)
  4. plan / act

That's it.  No step accesses another agent.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from songline_drive.belief_fusion import ConflictRuleSet, TemporalDecayEngine
from songline_drive.concept_recall import ConceptRecallResult
from songline_drive.place_alignment import PlaceAlignmentEngine

from distributed_memory.agent_memory import AgentMemory


class _NoCommunicationError(RuntimeError):
    """Raised when an independent agent is asked to do something that
    would require communication with another agent."""


class IndependentAgent:
    """Fully isolated agent.  Cannot exchange anything with other agents.

    Parameters
    ----------
    agent_id : str
    env_id : str
    role : str
    alignment_engine, decay_engine, conflict_rules
        Same low-level primitives as the other agent flavours.

    Notes
    -----
    The only intentionally-removed capability versus
    ``distributed_memory.AgentMemory`` is the ability to **export** a
    snapshot.  Local observation, local graph refresh, and local
    querying all work identically.
    """

    def __init__(
        self,
        agent_id: str,
        *,
        env_id: str = "default",
        role: str = "scout",
        alignment_engine: Optional[PlaceAlignmentEngine] = None,
        decay_engine: Optional[TemporalDecayEngine] = None,
        conflict_rules: Optional[ConflictRuleSet] = None,
    ) -> None:
        self.agent_id = agent_id
        self.env_id = env_id
        # Reuse AgentMemory for the actual graph machinery but never
        # expose its snapshot() method.
        self._memory = AgentMemory(
            agent_id, role=role, env_id=env_id, trust=1.0,
            alignment_engine=alignment_engine,
            decay_engine=decay_engine,
            conflict_rules=conflict_rules,
        )
        self._step_counter = 0

    # ──────────────────────────────────────────────────── observe

    def observe(
        self,
        place_key: Tuple[Any, ...],
        semantic_tags: Dict[str, float],
        *,
        episode_id: int,
        step_idx: int,
        confidence: float = 1.0,
        node_freshness: float = 1.0,
    ) -> None:
        self._memory.observe(
            place_key, semantic_tags,
            episode_id=episode_id, step_idx=step_idx,
            confidence=confidence, node_freshness=node_freshness,
        )

    def refresh_local(self) -> None:
        self._memory.refresh_local()

    # ──────────────────────────────────────────────────── query

    def local_query(self, target_tag: str, top_k: int = 5) -> List[ConceptRecallResult]:
        """The ONLY query method.  Returns concepts from THIS agent's graph."""
        return self._memory.local_query(target_tag, top_k=top_k)

    # ──────────────────────────────────────────────────── explicitly forbidden

    def snapshot(self):
        raise _NoCommunicationError(
            "IndependentAgent does not support snapshot export.  "
            "An independent agent has no mechanism to share state.  "
            "If you need this, use distributed_memory.AgentMemory or peer_memory.PeerAgent."
        )

    def receive(self, *args, **kwargs):
        raise _NoCommunicationError(
            "IndependentAgent cannot receive messages from other agents."
        )

    def broadcast(self, *args, **kwargs):
        raise _NoCommunicationError(
            "IndependentAgent cannot broadcast.  If you need broadcast, "
            "use peer_memory.PeerAgent instead."
        )

    # ──────────────────────────────────────────────────── diagnostics

    def tick_step(self) -> None:
        self._step_counter += 1

    @property
    def step_counter(self) -> int:
        return self._step_counter

    def stats(self) -> Dict[str, Any]:
        # Use AgentMemory.snapshot() *internally* purely for stats —
        # caller cannot reach this snapshot through the public API.
        internal = self._memory.snapshot()
        return {
            "agent_id": self.agent_id,
            "step_counter": self._step_counter,
            "n_local_concepts": internal.n_local_concepts,
            "n_events": internal.n_events,
            "communication_disabled": True,
        }
