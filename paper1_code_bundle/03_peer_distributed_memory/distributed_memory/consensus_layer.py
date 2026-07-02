"""ConsensusLayer — fuses per-agent local graphs into a distributed view.

Algorithm
---------

Given a list of ``AgentMemoryView`` snapshots:

1. **Pool** all local concepts from all agents, tagged with their source.
2. **Cluster** local concepts across agents by spatial proximity
   (``consensus_radius``) and either tag-match or profile similarity.
   Each cluster becomes one ``DistributedConcept``.
3. **Aggregate** within each cluster:
   - centroid_xy: trust-weighted mean
   - semantic_profile: trust × support weighted mean, then normalized
   - dominant_tag: argmax of the consensus profile
   - confidence: trust-weighted mean of local confidences
4. **Detect inter-agent disagreements** within clusters using the Phase 3
   ``ConflictRuleSet``.
5. **Compute** ``consensus_confidence``:
   ``confidence_mean × agreement_score × multi_agent_factor``
   where ``multi_agent_factor = min(1.0, sqrt(n_agents / n_total_agents))``
   — rewards concepts seen by many agents.

The layer is **stateless** between calls — each ``merge()`` is a pure
function of the input snapshots.  Callers persist the report themselves.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from songline_drive.belief_fusion import ConflictRuleSet
from songline_drive.collective_concepts import normalize_profile, euclidean_xy

from distributed_memory.consensus_types import (
    AgentContribution,
    AgentMemoryView,
    ConsensusReport,
    DistributedConcept,
)
from distributed_memory.disagreement import (
    agreement_score,
    detect_pairwise_disagreements,
)
from distributed_memory.trust_model import TrustModel


class ConsensusLayer:
    """Cross-agent consensus aggregator.

    Parameters
    ----------
    trust_model : TrustModel, optional
        Source of per-agent trust scores.  If omitted, a default model is
        created and the ``trust`` field on each snapshot is treated as
        authoritative.
    consensus_radius : float
        Two local concepts from different agents are candidates for the
        same consensus cluster if their centroids are within this distance.
    profile_similarity_threshold : float
        Minimum cosine similarity to merge concepts whose dominant tags
        differ.  (Concepts with matching dominant tags are merged
        regardless of similarity provided they are within radius.)
    conflict_rules : ConflictRuleSet, optional
        Phase 3 incompatibility rules.  Used to flag disagreements.
    multi_agent_reward : bool
        If True, applies a ``sqrt(n_agents / n_total)`` factor in
        ``consensus_confidence`` so concepts confirmed by multiple agents
        score higher.
    """

    def __init__(
        self,
        trust_model: Optional[TrustModel] = None,
        *,
        consensus_radius: float = 4.0,
        profile_similarity_threshold: float = 0.5,
        conflict_rules: Optional[ConflictRuleSet] = None,
        multi_agent_reward: bool = True,
    ) -> None:
        self.trust_model = trust_model or TrustModel()
        self.consensus_radius = float(consensus_radius)
        self.profile_similarity_threshold = float(profile_similarity_threshold)
        self.conflict_rules = conflict_rules or ConflictRuleSet.songlines_default()
        self.multi_agent_reward = bool(multi_agent_reward)
        self._merge_count = 0

    # ──────────────────────────────────────────────────────── merge

    def merge(self, views: List[AgentMemoryView]) -> ConsensusReport:
        """Fuse a list of agent snapshots into a single ConsensusReport."""
        self._merge_count += 1

        # Step 1: Pool all local concepts with their source agent
        pool: List[Tuple[str, str, Dict[str, Any]]] = []  # (agent_id, cid, summary)
        n_agents = len(views)
        for view in views:
            for cid, summary in view.local_concepts.items():
                if summary.get("centroid_xy") is None:
                    continue
                pool.append((view.agent_id, cid, summary))

        # Step 2: Cluster across agents
        clusters = self._cluster_across_agents(pool)

        # Steps 3-5: Build DistributedConcepts
        distributed: List[DistributedConcept] = []
        all_disagreements = []
        n_aligned = 0
        n_isolated = 0
        agreements: List[float] = []

        for cluster_idx, cluster in enumerate(clusters):
            dc = self._aggregate_cluster(
                cluster_idx, cluster, n_total_agents=n_agents,
            )
            distributed.append(dc)
            all_disagreements.extend(dc.disagreement_flags_objects)
            agreements.append(dc.inter_agent_agreement)
            if dc.n_agents > 1:
                n_aligned += 1
            else:
                n_isolated += 1

        avg_agreement = (
            sum(agreements) / len(agreements) if agreements else 1.0
        )

        report = ConsensusReport(
            distributed_concepts=distributed,
            disagreements=all_disagreements,
            n_agents=n_agents,
            n_aligned=n_aligned,
            n_isolated=n_isolated,
            avg_agreement=avg_agreement,
        )
        return report

    # ──────────────────────────────────────────────────── clustering

    def _cluster_across_agents(
        self,
        pool: List[Tuple[str, str, Dict[str, Any]]],
    ) -> List[List[Tuple[str, str, Dict[str, Any]]]]:
        """Union-find clustering of (agent_id, local_cid, summary) triples.

        Two items merge iff they come from different agents (we never
        merge two concepts from the same agent — that's the agent's own
        clustering decision) AND they pass the spatial + semantic gate.
        """
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
                agent_i, _, summary_i = pool[i]
                agent_j, _, summary_j = pool[j]
                if agent_i == agent_j:
                    continue
                if not self._should_merge(summary_i, summary_j):
                    continue
                union(i, j)

        clusters_by_root: Dict[int, List[Tuple[str, str, Dict[str, Any]]]] = {}
        for idx, item in enumerate(pool):
            root = find(idx)
            clusters_by_root.setdefault(root, []).append(item)

        return list(clusters_by_root.values())

    def _should_merge(
        self,
        a: Dict[str, Any],
        b: Dict[str, Any],
    ) -> bool:
        xy_a = a.get("centroid_xy")
        xy_b = b.get("centroid_xy")
        if xy_a is None or xy_b is None:
            return False
        if euclidean_xy(xy_a, xy_b) > self.consensus_radius:
            return False

        # Same dominant tag → merge regardless of similarity
        if a.get("dominant_tag") == b.get("dominant_tag"):
            return True

        # Different dominant tags → require profile similarity
        sim = _cosine(a.get("semantic_profile", {}), b.get("semantic_profile", {}))
        return sim >= self.profile_similarity_threshold

    # ──────────────────────────────────────────────────── aggregation

    def _aggregate_cluster(
        self,
        cluster_idx: int,
        cluster: List[Tuple[str, str, Dict[str, Any]]],
        n_total_agents: int,
    ) -> "DistributedConceptExt":
        contributions: List[AgentContribution] = []
        profiles_by_agent: Dict[str, Dict[str, float]] = {}

        # Weighted accumulators
        sum_x = sum_y = 0.0
        total_xy_weight = 0.0
        sum_conf = 0.0
        total_conf_weight = 0.0
        profile_acc: Dict[str, float] = {}

        for agent_id, local_cid, summary in cluster:
            trust = self.trust_model.get(agent_id)
            support = int(summary.get("support_count", 1) or 1)
            xy = summary["centroid_xy"]
            tag = summary.get("dominant_tag", "")
            confidence = float(summary.get("confidence", 0.0))
            freshness = float(summary.get("freshness", 0.0))
            profile = dict(summary.get("semantic_profile", {}))

            # Weights
            w_xy = trust * math.log1p(support)
            w_profile = trust * math.log1p(support)

            sum_x += xy[0] * w_xy
            sum_y += xy[1] * w_xy
            total_xy_weight += w_xy

            sum_conf += confidence * trust
            total_conf_weight += trust

            for k, v in profile.items():
                profile_acc[k] = profile_acc.get(k, 0.0) + v * w_profile

            contributions.append(AgentContribution(
                agent_id=agent_id,
                local_concept_id=local_cid,
                trust=trust,
                local_dominant_tag=tag,
                local_support=support,
                local_confidence=confidence,
                local_freshness=freshness,
            ))
            profiles_by_agent[agent_id] = profile

        # Centroid
        if total_xy_weight > 0:
            centroid_xy = (sum_x / total_xy_weight, sum_y / total_xy_weight)
        else:
            centroid_xy = cluster[0][2]["centroid_xy"]

        # Consensus profile + dominant tag
        consensus_profile = normalize_profile(profile_acc)
        if consensus_profile:
            consensus_dominant_tag = max(
                consensus_profile.items(), key=lambda kv: kv[1]
            )[0]
        else:
            consensus_dominant_tag = contributions[0].local_dominant_tag

        # Confidence (trust-weighted mean)
        mean_conf = sum_conf / total_conf_weight if total_conf_weight > 0 else 0.0

        consensus_id = f"consensus-{cluster_idx:04d}-{consensus_dominant_tag}"

        # Disagreement detection
        disagreements = detect_pairwise_disagreements(
            consensus_id=consensus_id,
            centroid_xy=centroid_xy,
            contributions=contributions,
            contribution_profiles=profiles_by_agent,
            rules=self.conflict_rules,
        )
        agreement = agreement_score(contributions, disagreements)

        # Multi-agent reward factor
        if self.multi_agent_reward and n_total_agents > 0:
            multi_factor = min(1.0, math.sqrt(len(contributions) / n_total_agents))
        else:
            multi_factor = 1.0

        consensus_confidence = mean_conf * agreement * multi_factor

        dc = DistributedConceptExt(
            consensus_id=consensus_id,
            centroid_xy=centroid_xy,
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
        return dc


# ────────────────────────────────────────────────────── helpers


class DistributedConceptExt(DistributedConcept):
    """Internal subclass that also carries disagreement objects (not just strings)."""

    def __init__(self, *args, disagreement_flags_objects=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.disagreement_flags_objects = list(disagreement_flags_objects or [])


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)
    dot = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in keys)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0
