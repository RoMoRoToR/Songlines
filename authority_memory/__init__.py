"""Authority Memory --- authority-control layer over Songlines
Runtime v1 (Sprint 1: certificate + authority state machine; Sprint 2:
provenance DAG; Sprint 4: staleness/revocation; Sprint 6: the
three-gate admission criterion; Sprint 7-8: randomized-intervention
label collection and the trained causal-utility estimator; Sprint 9:
relational schemas).

Central thesis (``docs/FRONTIER_MEMORY_AUTHORITY_2026-08-07/README.md``):
sharing information must not imply sharing action authority. This
package does not replace or modify ``songlines/`` (the frozen
runtime, ``docs/SONGLINES_V1_FREEZE.md``) --- it wraps it with a
receiver-specific, revocable authority layer.

Public API (Sprints 1-2-4-6-7-8-9 scope only --- the LLM semantic
layer lands in Sprint 10, see
``docs/FRONTIER_MEMORY_AUTHORITY_2026-08-07/07_ROADMAP_SPRINTS.md``):

    from authority_memory import Claim, MemoryCertificate, receive
    from authority_memory import (AuthorityState, ALLOWED_TRANSITIONS,
                                  ValidationEvent, AuthorityDecision,
                                  InvalidAuthorityTransition, transition,
                                  has_action_authority, has_limited_authority)
    from authority_memory import (ProvenanceGraph, EvidenceOrigin, RelayEdge,
                                  DuplicateOriginError)
    from authority_memory import n_eff, provenance_amplification_factor
    from authority_memory import (decay, expiry_horizon, is_stale,
                                  apply_staleness, apply_world_version_check,
                                  revoke_expired)
    from authority_memory import is_applicable, decide_admission, apply_admission
    from authority_memory import (InterventionLabel, randomized_mask,
                                  collect_labels, empirical_tau, standard_error)
    from authority_memory import CausalUtilityEstimator, fit_estimator, predict_tau
    from authority_memory import (RelationalSchema, WorldGraph,
                                  match_schema_to_graph, unique_match,
                                  schema_bits, induce_schema)
"""

from authority_memory.admission import (apply_admission, decide_admission,
                                        is_applicable)
from authority_memory.authority_state import (
    ALLOWED_TRANSITIONS, DURABLE_ACTION_STATES, LIMITED_USE_STATES,
    TERMINAL_STATES, AuthorityDecision, AuthorityState,
    InvalidAuthorityTransition, ValidationEvent, has_action_authority,
    has_limited_authority, transition)
from authority_memory.causal_utility import (CausalUtilityEstimator,
                                             InterventionLabel,
                                             collect_labels, empirical_tau,
                                             fit_estimator, predict_tau,
                                             randomized_mask,
                                             standard_error)
from authority_memory.certificate import (STRUCTURAL_RELATIONS, Claim,
                                          MemoryCertificate, receive)
from authority_memory.metrics import n_eff, provenance_amplification_factor
from authority_memory.provenance_graph import (DuplicateOriginError,
                                               EvidenceOrigin,
                                               ProvenanceGraph, RelayEdge)
from authority_memory.revocation import (EXPIRABLE_STATES,
                                         REVOCABLE_STATES,
                                         apply_staleness,
                                         apply_world_version_check, decay,
                                         expiry_horizon, is_stale,
                                         revoke_expired)
from authority_memory.schema import (RelationalSchema, WorldGraph,
                                     match_schema_to_graph, schema_bits,
                                     unique_match)
from authority_memory.schema_induction import induce_schema

__all__ = [
    "Claim", "MemoryCertificate", "receive", "STRUCTURAL_RELATIONS",
    "AuthorityState", "ALLOWED_TRANSITIONS", "DURABLE_ACTION_STATES",
    "LIMITED_USE_STATES", "TERMINAL_STATES",
    "ValidationEvent", "AuthorityDecision", "InvalidAuthorityTransition",
    "transition", "has_action_authority", "has_limited_authority",
    "ProvenanceGraph", "EvidenceOrigin", "RelayEdge",
    "DuplicateOriginError", "n_eff", "provenance_amplification_factor",
    "decay", "expiry_horizon", "is_stale", "apply_staleness",
    "apply_world_version_check", "revoke_expired", "EXPIRABLE_STATES",
    "REVOCABLE_STATES", "is_applicable", "decide_admission",
    "apply_admission", "InterventionLabel", "randomized_mask",
    "collect_labels", "empirical_tau", "standard_error",
    "CausalUtilityEstimator", "fit_estimator", "predict_tau",
    "RelationalSchema", "WorldGraph", "match_schema_to_graph",
    "unique_match", "schema_bits", "induce_schema",
]
