"""Place identity resolution across agents with partial/noisy observations.

Replaces the coordinate-grounded identity assumption

    (env_id, x, y)  ==  same place

with probabilistic identity scoring that works when agents have
different, partial, or noisy views of the same location.

Three regimes are defined explicitly so the boundary between the
trivial case and the real research problem is testable:

  COORDINATE_ONLY  — current system; (x,y) hard key; fails under any noise.
  SEMANTIC_ONLY    — tag profiles only; no spatial prior at all.
  HYBRID           — soft spatial Gaussian prior × semantic likelihood.

The HYBRID regime is the honest middle ground: it degrades gracefully
as positional uncertainty rises (falling back toward semantic-only) and
as tag profiles become sparser (falling back toward spatial-only).

The three hidden assumptions of the old system — shared coordinates,
stable dominant_tag, single env_id — are each made explicit as
parameters here, so experiments can stress each one independently.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from songline_drive.collective_concepts import cosine_similarity


# ──────────────────────────────────────────────────── types

class IdentityMode(str, Enum):
    """
    COORDINATE_ONLY  Exact (x,y) match required. Identity is trivial when
                     agents share a global frame; breaks immediately under noise.
    SEMANTIC_ONLY    Tag-profile cosine similarity only. No coordinate
                     assumption at all. Robust to any positional noise but
                     collapses when profiles have no tag overlap.
    HYBRID           Combines a soft Gaussian spatial prior with semantic
                     likelihood. The two weights are explicit parameters so
                     the contribution of each axis is measurable.
    """
    COORDINATE_ONLY = "coordinate_only"
    SEMANTIC_ONLY   = "semantic_only"
    HYBRID          = "hybrid"


@dataclass
class PlaceObservation:
    """One agent's observation of one place.

    Parameters
    ----------
    agent_id:
        Which agent made this observation.
    raw_key:
        The (x, y) this agent *believes* the place to be at.
        Under coordinate noise this may differ between agents for the
        same physical location.
    position_sigma:
        Positional uncertainty (std dev, grid units). 0 → exact coordinate;
        >0 → noisy. This is the quantity that exposes the hidden assumption.
    tag_profile:
        Tag-confidence dict from this agent's partial view.
        May be sparse: not all tags are visible to all agents.
    env_id:
        Environment. Observations from different env_ids are never merged
        unless cross_env=True is set on the engine.
    observation_seq:
        Global step at which this was made (used for staleness, not identity).
    """
    agent_id: str
    raw_key: Tuple[float, float]
    position_sigma: float
    tag_profile: Dict[str, float]
    env_id: Optional[str] = None
    observation_seq: int = 0


@dataclass
class IdentityScore:
    """Result of comparing two PlaceObservations for identity."""
    obs_a_key: Tuple[float, float]
    obs_b_key: Tuple[float, float]
    spatial_score: float    # 0..1; Gaussian overlap on noisy positions
    semantic_score: float   # 0..1; cosine similarity of tag profiles
    combined_score: float   # final; mode-dependent combination
    is_same: bool           # decision at engine threshold
    mode: IdentityMode


@dataclass
class IdentityCluster:
    """A group of observations resolved to the same canonical place."""
    canonical_id: int               # index of the seed observation
    member_indices: List[int]       # all observation indices in this cluster
    agent_ids: List[str]            # agents that contributed
    mean_position: Tuple[float, float]
    fused_profile: Dict[str, float] # weighted average of member profiles
    position_spread: float          # max pairwise distance across members


# ──────────────────────────────────────────────────── engine

class PlaceIdentityEngine:
    """
    Resolves place identity across agents without assuming shared coordinates.

    The key departure: place_key = (x, y) is a *noisy observation with
    uncertainty*, not a ground-truth key. Two observations from different
    agents are the same place when their combined spatial + semantic
    evidence exceeds identity_threshold.

    Parameters
    ----------
    mode:
        Which identity regime to use (see IdentityMode).
    identity_threshold:
        Minimum combined_score to declare two observations the same place.
        0.5 is a neutral prior; raise for stricter merging.
    spatial_sigma_scale:
        Multiplier on position_sigma when computing spatial overlap.
        Controls how much positional uncertainty is tolerated.
    semantic_weight, spatial_weight:
        Convex combination weights in HYBRID mode (must sum to 1).
        semantic_weight=1, spatial_weight=0 collapses to SEMANTIC_ONLY.
    cross_env:
        Allow identity resolution across env_id boundaries.
        Off by default; turn on for transfer / cross-layout experiments.
    min_profile_overlap:
        Minimum fraction of shared tag keys required before semantic
        score is trusted. Below this, sparse profiles are unreliable.
        Set to 0 to disable.
    """

    def __init__(
        self,
        mode: IdentityMode = IdentityMode.HYBRID,
        identity_threshold: float = 0.55,
        spatial_sigma_scale: float = 1.0,
        semantic_weight: float = 0.6,
        spatial_weight: float = 0.4,
        cross_env: bool = False,
        min_profile_overlap: float = 0.0,
    ) -> None:
        if abs(semantic_weight + spatial_weight - 1.0) > 1e-6:
            raise ValueError("semantic_weight + spatial_weight must equal 1.0")
        self.mode = mode
        self.identity_threshold = identity_threshold
        self.spatial_sigma_scale = spatial_sigma_scale
        self.semantic_weight = semantic_weight
        self.spatial_weight = spatial_weight
        self.cross_env = cross_env
        self.min_profile_overlap = min_profile_overlap

    # ---------------------------------------------------------------- public

    def score_pair(
        self,
        obs_a: PlaceObservation,
        obs_b: PlaceObservation,
    ) -> IdentityScore:
        """Compute pairwise identity score. No side effects."""

        # Hard gate: different envs → always distinct (unless cross_env)
        if (
            not self.cross_env
            and obs_a.env_id is not None
            and obs_b.env_id is not None
            and obs_a.env_id != obs_b.env_id
        ):
            return IdentityScore(
                obs_a_key=obs_a.raw_key, obs_b_key=obs_b.raw_key,
                spatial_score=0.0, semantic_score=0.0,
                combined_score=0.0, is_same=False, mode=self.mode,
            )

        spatial_score  = self._spatial_score(obs_a, obs_b)
        semantic_score = self._semantic_score(obs_a, obs_b)

        if self.mode == IdentityMode.COORDINATE_ONLY:
            # Strict: non-zero distance → different place in hard-key world
            combined = 1.0 if spatial_score >= 1.0 else 0.0

        elif self.mode == IdentityMode.SEMANTIC_ONLY:
            combined = semantic_score

        else:  # HYBRID
            combined = (
                self.spatial_weight  * spatial_score
                + self.semantic_weight * semantic_score
            )

        return IdentityScore(
            obs_a_key=obs_a.raw_key,
            obs_b_key=obs_b.raw_key,
            spatial_score=spatial_score,
            semantic_score=semantic_score,
            combined_score=combined,
            is_same=combined >= self.identity_threshold,
            mode=self.mode,
        )

    def resolve_identities(
        self,
        observations: List[PlaceObservation],
    ) -> Dict[int, int]:
        """
        Cluster observations into canonical places.

        Two observations from the **same agent** are never merged (an
        agent's own map is internally consistent; merging within-agent
        would conflate distinct places that happen to look similar).

        Returns
        -------
        mapping : Dict[int, int]
            obs_index → canonical_id, where canonical_id is the index
            of the earliest observation in the cluster (union-find root).
        """
        n = len(observations)
        parent = list(range(n))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i: int, j: int) -> None:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[rj] = ri  # earlier index becomes root

        for i in range(n):
            for j in range(i + 1, n):
                if observations[i].agent_id == observations[j].agent_id:
                    continue
                if self.score_pair(observations[i], observations[j]).is_same:
                    union(i, j)

        return {i: find(i) for i in range(n)}

    def build_clusters(
        self,
        observations: List[PlaceObservation],
    ) -> List[IdentityCluster]:
        """
        Resolve identities and return fully-described clusters.

        Each cluster carries a fused semantic profile and position
        estimate, ready to hand off to downstream planning or retrieval.
        """
        mapping = self.resolve_identities(observations)

        groups: Dict[int, List[int]] = {}
        for obs_idx, root in mapping.items():
            groups.setdefault(root, []).append(obs_idx)

        clusters: List[IdentityCluster] = []
        for root, indices in sorted(groups.items()):
            members = [observations[i] for i in indices]
            agent_ids = list({m.agent_id for m in members})

            # Weighted position: uniform average (extend with trust weights if needed)
            xs = [m.raw_key[0] for m in members]
            ys = [m.raw_key[1] for m in members]
            mean_pos = (sum(xs) / len(xs), sum(ys) / len(ys))

            # Position spread: max pairwise distance
            spread = 0.0
            for i, mi in enumerate(members):
                for mj in members[i + 1:]:
                    d = math.sqrt(
                        (mi.raw_key[0] - mj.raw_key[0]) ** 2
                        + (mi.raw_key[1] - mj.raw_key[1]) ** 2
                    )
                    spread = max(spread, d)

            # Fused profile: simple average across members
            fused: Dict[str, float] = {}
            for m in members:
                for tag, conf in m.tag_profile.items():
                    fused[tag] = fused.get(tag, 0.0) + conf
            fused = {k: v / len(members) for k, v in fused.items()}

            clusters.append(IdentityCluster(
                canonical_id=root,
                member_indices=indices,
                agent_ids=agent_ids,
                mean_position=mean_pos,
                fused_profile=fused,
                position_spread=spread,
            ))

        return clusters

    # ---------------------------------------------------------------- internals

    def _spatial_score(
        self, a: PlaceObservation, b: PlaceObservation
    ) -> float:
        """
        Spatial identity score in [0, 1].

        Exact match → 1.0.
        COORDINATE_ONLY with non-zero distance → 0.0.
        HYBRID / SEMANTIC_ONLY: Gaussian overlap
            score = exp(-d² / (2 * σ_eff²))
        where σ_eff = spatial_sigma_scale × max(σ_a, σ_b, 0.5).

        The floor at 0.5 grid unit prevents σ=0 from making the Gaussian
        a Dirac delta — even a "perfect" position observation has ≥0.5 unit
        discretization uncertainty in a grid world.
        """
        ax, ay = a.raw_key
        bx, by = b.raw_key
        d_sq = (ax - bx) ** 2 + (ay - by) ** 2

        if d_sq == 0.0:
            return 1.0
        if self.mode == IdentityMode.COORDINATE_ONLY:
            return 0.0

        sigma_eff = self.spatial_sigma_scale * max(
            a.position_sigma, b.position_sigma, 0.5
        )
        return math.exp(-d_sq / (2.0 * sigma_eff ** 2))

    def _semantic_score(
        self, a: PlaceObservation, b: PlaceObservation
    ) -> float:
        """
        Semantic identity score: cosine similarity of tag profiles.

        Partial profiles (one agent hasn't observed all tags) are handled
        naturally: missing tags contribute 0 to the dot product, so
        partial overlap lowers the score gracefully.

        If min_profile_overlap > 0 and the fraction of shared tag keys is
        below that threshold, the score is penalised proportionally.
        This prevents two nearly-empty profiles from spuriously matching.
        """
        if not a.tag_profile or not b.tag_profile:
            return 0.0

        sim = cosine_similarity(a.tag_profile, b.tag_profile)

        if self.min_profile_overlap > 0.0:
            keys_a = set(a.tag_profile)
            keys_b = set(b.tag_profile)
            all_keys = keys_a | keys_b
            if all_keys:
                overlap_frac = len(keys_a & keys_b) / len(all_keys)
                if overlap_frac < self.min_profile_overlap:
                    sim *= overlap_frac / self.min_profile_overlap

        return sim
