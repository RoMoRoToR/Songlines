"""Local merge — each agent runs this OVER ITS OWN snapshot + peer messages.

This is the peer-to-peer analogue of ``ConsensusLayer.merge`` from
``distributed_memory``.  Two important differences:

  1. It is called **by an agent on itself**, not by a central coordinator.
     The first argument is "my own snapshot"; the rest are messages this
     agent has received from peers.

  2. It uses the agent's **private** ``AsymmetricTrust`` table to weight
     contributions.  Trust toward peer P is what *this agent* believes
     about P; other agents may use different weights for the same P.

The result is a ``PeerView``: this agent's belief about the world
given everything it knows.  No two agents share a PeerView, and two
agents may produce different PeerViews from the same set of messages
if their trust tables differ.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from songline_drive.belief_fusion import ConflictRuleSet
from songline_drive.collective_concepts import euclidean_xy, normalize_profile

from distributed_memory.consensus_types import (
    AgentContribution,
    AgentMemoryView,
)
from distributed_memory.disagreement import (
    agreement_score,
    detect_pairwise_disagreements,
)

from peer_memory.peer_trust import AsymmetricTrust
from peer_memory.peer_types import (
    BroadcastMessage,
    PeerMergeReport,
    PeerView,
)


def local_merge(
    own_snapshot: AgentMemoryView,
    peer_messages: List[BroadcastMessage],
    trust: AsymmetricTrust,
    *,
    consensus_radius: float = 4.0,
    profile_similarity_threshold: float = 0.5,
    conflict_rules: Optional[ConflictRuleSet] = None,
    multi_agent_reward: bool = True,
    at_step: int = 0,
) -> Tuple[PeerView, PeerMergeReport]:
    """Merge own snapshot + peer snapshots into a private PeerView.

    Parameters
    ----------
    own_snapshot : AgentMemoryView
        This agent's own local-graph snapshot.
    peer_messages : list of BroadcastMessage
        Messages received from peers (each carrying that peer's snapshot).
    trust : AsymmetricTrust
        This agent's private trust table.
    consensus_radius, profile_similarity_threshold, conflict_rules,
    multi_agent_reward
        Same semantics as ``ConsensusLayer``.

    Returns
    -------
    PeerView, PeerMergeReport
        The view this agent will query, plus a diagnostic report.
    """
    conflict_rules = conflict_rules or ConflictRuleSet.songlines_default()

    # Step 1: pool concepts (own + all peers)
    pool: List[Tuple[str, str, Dict, float]] = []  # (agent_id, cid, summary, w)

    # Own concepts: full self-trust
    own_w = trust.trust_in(own_snapshot.agent_id)  # == trust_max by convention
    for cid, summary in own_snapshot.local_concepts.items():
        if summary.get("centroid_xy") is None:
            continue
        pool.append((own_snapshot.agent_id, cid, summary, own_w))

    # Peer concepts: weighted by this agent's trust in the peer
    contributing_peer_ids: List[str] = []
    for msg in peer_messages:
        peer_id = msg.sender_id
        peer_trust = trust.trust_in(peer_id)
        if peer_id not in contributing_peer_ids:
            contributing_peer_ids.append(peer_id)
        for cid, summary in msg.snapshot.local_concepts.items():
            if summary.get("centroid_xy") is None:
                continue
            pool.append((peer_id, cid, summary, peer_trust))

    # Step 2: union-find clustering across SOURCES
    n = len(pool)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(n):
        for j in range(i + 1, n):
            agent_i, _, s_i, _ = pool[i]
            agent_j, _, s_j, _ = pool[j]
            if agent_i == agent_j:
                continue  # never merge two concepts from same source
            if not _should_merge(s_i, s_j, consensus_radius,
                                 profile_similarity_threshold):
                continue
            union(i, j)

    clusters: Dict[int, List[int]] = {}
    for idx in range(n):
        clusters.setdefault(find(idx), []).append(idx)

    # Step 3: aggregate each cluster into a DistributedConcept (peer flavour)
    distributed_concepts = []
    all_disagreements = []
    n_total_sources = 1 + len(contributing_peer_ids)

    for cluster_idx, (_root, indices) in enumerate(clusters.items()):
        contributions: List[AgentContribution] = []
        profiles_by_agent: Dict[str, Dict[str, float]] = {}
        sum_x = sum_y = 0.0
        total_xy_w = 0.0
        sum_conf = 0.0
        total_conf_w = 0.0
        profile_acc: Dict[str, float] = {}

        for idx in indices:
            agent_id, local_cid, summary, w_trust = pool[idx]
            support = int(summary.get("support_count", 1) or 1)
            xy = summary["centroid_xy"]
            tag = summary.get("dominant_tag", "")
            confidence = float(summary.get("confidence", 0.0))
            freshness = float(summary.get("freshness", 0.0))
            profile = dict(summary.get("semantic_profile", {}))

            w_combined = w_trust * math.log1p(support)
            sum_x += xy[0] * w_combined
            sum_y += xy[1] * w_combined
            total_xy_w += w_combined
            sum_conf += confidence * w_trust
            total_conf_w += w_trust
            for k, v in profile.items():
                profile_acc[k] = profile_acc.get(k, 0.0) + v * w_combined

            contributions.append(AgentContribution(
                agent_id=agent_id, local_concept_id=local_cid,
                trust=w_trust, local_dominant_tag=tag,
                local_support=support, local_confidence=confidence,
                local_freshness=freshness,
            ))
            profiles_by_agent[agent_id] = profile

        centroid = (
            (sum_x / total_xy_w, sum_y / total_xy_w)
            if total_xy_w > 0
            else pool[indices[0]][2]["centroid_xy"]
        )
        consensus_profile = normalize_profile(profile_acc)
        consensus_dominant_tag = (
            max(consensus_profile.items(), key=lambda kv: kv[1])[0]
            if consensus_profile else contributions[0].local_dominant_tag
        )
        mean_conf = sum_conf / total_conf_w if total_conf_w > 0 else 0.0

        consensus_id = f"peer-{own_snapshot.agent_id}-{cluster_idx:04d}-{consensus_dominant_tag}"
        disagreements = detect_pairwise_disagreements(
            consensus_id=consensus_id, centroid_xy=centroid,
            contributions=contributions,
            contribution_profiles=profiles_by_agent,
            rules=conflict_rules,
        )
        agreement = agreement_score(contributions, disagreements)

        n_sources_here = len(set(c.agent_id for c in contributions))
        multi_factor = (
            min(1.0, math.sqrt(n_sources_here / max(1, n_total_sources)))
            if multi_agent_reward else 1.0
        )
        consensus_confidence = mean_conf * agreement * multi_factor

        from distributed_memory.consensus_layer import DistributedConceptExt
        dc = DistributedConceptExt(
            consensus_id=consensus_id, centroid_xy=centroid,
            consensus_dominant_tag=consensus_dominant_tag,
            consensus_profile=consensus_profile,
            consensus_confidence=consensus_confidence,
            inter_agent_agreement=agreement,
            contributions=contributions,
            disagreement_flags=[
                f"{d.agent_a}:{d.tag_a} vs {d.agent_b}:{d.tag_b} (sev={d.severity:.2f})"
                for d in disagreements
            ],
            disagreement_flags_objects=disagreements,
        )
        distributed_concepts.append(dc)
        all_disagreements.extend(disagreements)

    view = PeerView(
        owner_id=own_snapshot.agent_id,
        formed_at_step=at_step,
        distributed_concepts=distributed_concepts,
        disagreements=all_disagreements,
        n_peer_messages_used=len(peer_messages),
        contributing_peer_ids=contributing_peer_ids,
    )
    report = PeerMergeReport(
        owner_id=own_snapshot.agent_id, at_step=at_step,
        n_local_concepts=own_snapshot.n_local_concepts,
        n_peer_messages=len(peer_messages),
        peer_ids=contributing_peer_ids,
        n_clusters_formed=len(distributed_concepts),
        n_disagreements=len(all_disagreements),
    )
    return view, report


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)
    dot = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in keys)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


def _should_merge(
    a: Dict, b: Dict, radius: float, sim_threshold: float,
) -> bool:
    xy_a = a.get("centroid_xy")
    xy_b = b.get("centroid_xy")
    if xy_a is None or xy_b is None:
        return False
    if euclidean_xy(xy_a, xy_b) > radius:
        return False
    if a.get("dominant_tag") == b.get("dominant_tag"):
        return True
    sim = _cosine(a.get("semantic_profile", {}), b.get("semantic_profile", {}))
    return sim >= sim_threshold
