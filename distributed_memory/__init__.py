"""Distributed memory — per-agent isolated views with consensus fusion.

Each agent owns a private ``AgentMemory`` (event store + concept graph +
optional field).  Agents never share memory directly.  Cross-agent
integration happens through ``ConsensusLayer.merge()`` which fuses
per-agent snapshots into a ``ConsensusReport`` of ``DistributedConcept``
objects with trust-weighted aggregation and disagreement flagging.

Top-level entry point is ``DistributedRuntime``.

Differs from the standard collective memory pipeline
(``songline_drive/collective_*``):
    - Each agent has its own ``CollectiveMemory`` (event store)
    - Each agent has its own ``ConceptRecallLayer``
    - Cross-agent fusion is explicit, not implicit
    - Trust weighting + disagreement detection happen at fusion time
"""

from distributed_memory.agent_memory import AgentMemory
from distributed_memory.consensus_layer import ConsensusLayer
from distributed_memory.consensus_types import (
    AgentContribution,
    AgentDisagreement,
    AgentMemoryView,
    ConsensusReport,
    DistributedConcept,
)
from distributed_memory.disagreement import (
    agreement_score,
    detect_pairwise_disagreements,
)
from distributed_memory.distributed_runtime import DistributedRuntime
from distributed_memory.trust_model import TrustModel

__all__ = [
    "AgentContribution",
    "AgentDisagreement",
    "AgentMemory",
    "AgentMemoryView",
    "ConsensusLayer",
    "ConsensusReport",
    "DistributedConcept",
    "DistributedRuntime",
    "TrustModel",
    "agreement_score",
    "detect_pairwise_disagreements",
]
