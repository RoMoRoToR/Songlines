"""Inter-agent disagreement detection.

A *disagreement* is when two or more agents observed roughly the same
location but produced incompatible local beliefs about it (most
commonly: incompatible dominant tags).

Unlike Phase 3 conflict — which fuses incompatible tags *within* a
single concept — disagreement is detected *across* agents whose local
concepts happen to align spatially.
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Tuple

from songline_drive.belief_fusion import ConflictRuleSet

from distributed_memory.consensus_types import (
    AgentContribution,
    AgentDisagreement,
)


def _profile_overlap(profile_a: Dict[str, float], profile_b: Dict[str, float]) -> float:
    """Min over shared tags as a proxy for the strength of disagreement."""
    keys = set(profile_a) | set(profile_b)
    if not keys:
        return 0.0
    total = 0.0
    for k in keys:
        total += min(profile_a.get(k, 0.0), profile_b.get(k, 0.0))
    return total / max(1, len(keys))


def detect_pairwise_disagreements(
    consensus_id: str,
    centroid_xy: Tuple[float, float],
    contributions: List[AgentContribution],
    contribution_profiles: Dict[str, Dict[str, float]],
    rules: ConflictRuleSet,
) -> List[AgentDisagreement]:
    """Find pairs of agents with incompatible dominant tags at the same place.

    Parameters
    ----------
    consensus_id : str
        Identifier of the consensus concept where the disagreement lives.
    centroid_xy : Tuple[float, float]
        Centroid of the consensus concept (for diagnostics).
    contributions : list of AgentContribution
        Per-agent local belief at this consensus concept.
    contribution_profiles : dict
        Mapping ``agent_id -> local semantic_profile``.  Used to compute
        severity beyond just the dominant tag.
    rules : ConflictRuleSet
        Phase 3 conflict rules (incompat pairs).
    """
    flags: List[AgentDisagreement] = []
    incompat_pairs = _incompat_pair_set(rules)
    if not incompat_pairs:
        return flags

    for c_a, c_b in itertools.combinations(contributions, 2):
        tag_a, tag_b = c_a.local_dominant_tag, c_b.local_dominant_tag
        pair = tuple(sorted([tag_a, tag_b]))
        if pair not in incompat_pairs:
            continue

        # Severity: blend trust-weighted profile mass on incompatible tags
        prof_a = contribution_profiles.get(c_a.agent_id, {})
        prof_b = contribution_profiles.get(c_b.agent_id, {})
        mass_a = prof_a.get(tag_a, 0.0) * c_a.trust
        mass_b = prof_b.get(tag_b, 0.0) * c_b.trust
        severity = min(1.0, mass_a + mass_b)

        flags.append(AgentDisagreement(
            consensus_id=consensus_id,
            centroid_xy=centroid_xy,
            agent_a=c_a.agent_id,
            agent_b=c_b.agent_id,
            tag_a=tag_a,
            tag_b=tag_b,
            severity=severity,
        ))
    return flags


def _incompat_pair_set(rules: ConflictRuleSet) -> set:
    """Extract incompatible tag pairs from a ConflictRuleSet as a normalized set."""
    pairs = set()
    for rule in getattr(rules, "rules", []):
        tag_a = getattr(rule, "positive_tag", None)
        tag_b = getattr(rule, "negative_tag", None)
        if tag_a and tag_b:
            pairs.add(tuple(sorted([tag_a, tag_b])))
    return pairs


def agreement_score(
    contributions: List[AgentContribution],
    disagreements: List[AgentDisagreement],
) -> float:
    """Compute an agreement score in [0, 1] for one consensus concept.

    1.0 — all agents agree on the dominant tag.
    0.0 — every pair of agents disagrees with maximum severity.
    """
    n = len(contributions)
    if n < 2:
        return 1.0
    max_pairs = n * (n - 1) // 2
    severity_sum = sum(d.severity for d in disagreements)
    return max(0.0, 1.0 - severity_sum / max_pairs)
