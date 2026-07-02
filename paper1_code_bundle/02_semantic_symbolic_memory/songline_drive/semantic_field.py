"""Phase 4 — multi-channel semantic activation field over the concept graph.

The field is a real-valued function A(channel, concept) ∈ [0, 1].

Update equation (Phase 4 spec §4):

    A_{t+1}(k,c) = λ·A_t(k,c)
                 + α·B(c)·I(k,c)
                 + β·O_t(k,c)
                 + γ·D_t(k,c)
                 + δ·P_t(k,c)
                 - η·X_t(c)
                 - ξ·U_t(c)

    B(c)    belief strength from Phase 2/3 concept attributes
    I(k,c)  channel affinity — dot product of semantic_profile with channel weights
    O_t     observation pressure (new events — Phase 4c)
    D_t     diffusion from spatially neighbouring concepts (γ-term)
    P_t     intent/role pressure from active agents (Phase 4c)
    X_t(c)  conflict penalty (from Phase 3b conflict_score)
    U_t(c)  occupancy/reservation penalty (Phase 4c)

Phase 4a uses only α·B·I and -η·X (static rebuild).
Phase 4b adds λ·A_t (decay between rebuilds) and γ·D (diffusion).
Phase 4c adds β·O, δ·P, ξ·U (full dynamic field).

Invariants (§0):
  1. Phase 1 event bus is never written to by the field.
  2. Phase 2 concept graph remains canonical; field lives on top.
  3. Phase 3 decay/conflict are applied first; field reads pre-computed results.
  4. Planner/local control not rewritten; field only reranks candidates.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from songline_drive.collective_field_types import (
    AgentFieldFootprint,
    FieldCellState,
    FieldChannelState,
    FieldMode,
    FieldQuery,
    FieldQueryResult,
    FieldReservation,
)


# ─────────────────────────────────────────── default channel affinity weights


DEFAULT_CHANNEL_AFFINITIES: Dict[str, Dict[str, float]] = {
    "water_source": {
        "water_source": 1.00,
        "water_candidate": 0.50,
        "water_visible": 0.40,
        "near_water": 0.30,
        "adjacent_hazard": -0.40,
        "hazard_edge": -0.60,
    },
    "safe_neutral": {
        "safe_neutral": 1.00,
        "corridor": 0.30,
        "room_center": 0.40,
        "open_safe_rest_zone": 0.60,
        "adjacent_hazard": -0.50,
        "hazard_edge": -0.90,
    },
    "hazard_edge": {
        "hazard_edge": 1.00,
        "adjacent_hazard": 0.60,
        "water_source": -0.20,
        "safe_neutral": -0.30,
    },
    "goal_region": {
        "goal_region": 1.00,
        "open_goal": 0.40,
        "hazard_edge": -0.70,
        "adjacent_hazard": -0.40,
    },
    "hazard_recovery_route": {
        "hazard_recovery_route": 1.00,
        "post_hazard_goal_rejoin": 0.60,
        "hazard_edge": -0.40,
    },
}


class SemanticField:
    """Multi-channel semantic activation field over the concept graph.

    Parameters
    ----------
    channels:
        List of semantic channels to track. Defaults to all keys of
        ``DEFAULT_CHANNEL_AFFINITIES``.
    channel_affinities:
        Override the default affinity weight tables.
    lambda_decay:
        λ — per-step multiplicative decay applied by ``decay()``.
    alpha_belief:
        α — weight of the belief·affinity input term.
    beta_obs, gamma_diffusion, delta_intent, eta_conflict, xi_occupancy:
        β, γ, δ, η, ξ — as in the update equation above.
    w_conf, w_fresh, w_support, w_purity:
        Weights for belief-strength B(c).
    diffusion_radius:
        Maximum centroid distance for a pair of concepts to be treated as
        diffusion neighbours (same unit as centroid_xy coordinates).
    diffusion_steps:
        Number of diffusion passes applied after each rebuild.
    ema_fast, ema_slow:
        EMA blend factors for activation_fast / activation_slow.
    mode:
        ``FieldMode`` constant controlling how the field participates in
        retrieval (descriptive / read_only / coordinated / none).
    """

    def __init__(
        self,
        channels: Optional[List[str]] = None,
        channel_affinities: Optional[Dict[str, Dict[str, float]]] = None,
        *,
        lambda_decay: float = 0.95,
        alpha_belief: float = 0.60,
        beta_obs: float = 0.20,
        gamma_diffusion: float = 0.10,
        delta_intent: float = 0.15,
        eta_conflict: float = 0.30,
        xi_occupancy: float = 0.20,
        w_conf: float = 0.40,
        w_fresh: float = 0.30,
        w_support: float = 0.20,
        w_purity: float = 0.10,
        diffusion_radius: float = 5.0,
        diffusion_steps: int = 1,
        ema_fast: float = 0.70,
        ema_slow: float = 0.10,
        mode: str = FieldMode.DESCRIPTIVE,
    ) -> None:
        self.channels: List[str] = list(
            channels or DEFAULT_CHANNEL_AFFINITIES.keys()
        )
        self.channel_affinities: Dict[str, Dict[str, float]] = (
            channel_affinities or DEFAULT_CHANNEL_AFFINITIES
        )
        self.lambda_decay = float(lambda_decay)
        self.alpha_belief = float(alpha_belief)
        self.beta_obs = float(beta_obs)
        self.gamma_diffusion = float(gamma_diffusion)
        self.delta_intent = float(delta_intent)
        self.eta_conflict = float(eta_conflict)
        self.xi_occupancy = float(xi_occupancy)
        self.w_conf = float(w_conf)
        self.w_fresh = float(w_fresh)
        self.w_support = float(w_support)
        self.w_purity = float(w_purity)
        self.diffusion_radius = float(diffusion_radius)
        self.diffusion_steps = int(diffusion_steps)
        self.ema_fast = float(ema_fast)
        self.ema_slow = float(ema_slow)
        self.mode = FieldMode.validate(mode)

        self._cells: Dict[str, FieldCellState] = {}
        self._build_seq: int = -1
        self._reservations: Dict[Tuple[str, str], FieldReservation] = {}
        self._footprints: List[AgentFieldFootprint] = []

    # ──────────────────────────────────────────── belief / affinity helpers

    def _belief_strength_from_concept(self, concept: Any) -> float:
        """Compute B(c) from a graph concept node.

        B(c) = w_conf·conf + w_fresh·fresh + w_support·log1p(support)/log1p(100) + w_purity·purity
        """
        conf = float(concept.confidence)
        fresh = float(concept.freshness)
        purity = max(0.0, 1.0 - float(concept.conflict_score))
        support_norm = math.log1p(int(concept.support_count)) / math.log1p(100)
        return (
            self.w_conf * conf
            + self.w_fresh * fresh
            + self.w_support * support_norm
            + self.w_purity * purity
        )

    def _channel_affinity(
        self, channel: str, semantic_profile: Dict[str, float]
    ) -> float:
        """Compute I(k, c) ∈ [0, ∞) — channel affinity for a concept profile.

        Positive weights contribute; negative weights suppress.
        Result is clamped to 0 from below (no negative affinity).
        """
        weights = self.channel_affinities.get(channel, {})
        pos = sum(
            float(semantic_profile.get(tag, 0.0)) * w
            for tag, w in weights.items()
            if w > 0
        )
        neg = sum(
            float(semantic_profile.get(tag, 0.0)) * abs(w)
            for tag, w in weights.items()
            if w < 0
        )
        return max(0.0, pos - neg)

    # ───────────────────────────────────────────────────── rebuild / diffuse

    def rebuild_from_concepts(
        self,
        graph: Any,           # SharedConceptGraph
        current_seq: int = 0,
    ) -> None:
        """Full rebuild of field state from concept graph.

        Reads pre-computed ``concept.conflict_score`` (Phase 3b) and
        ``concept.freshness`` (Phase 3a) — both must be applied to the graph
        before calling this method (via ``ConceptRecallLayer.refresh()``).

        Phase 4a activation:  A(k,c) = α·B(c)·I(k,c) - η·X(c)
        After rebuild, ``diffusion_steps`` diffusion passes are applied.
        """
        prev_cells = self._cells  # keep for EMA continuity
        self._cells = {}
        self._build_seq = current_seq

        for cid, concept in graph.concepts.items():
            belief = self._belief_strength_from_concept(concept)
            conflict = float(concept.conflict_score)
            purity = max(0.0, 1.0 - conflict)
            freshness = float(concept.freshness)

            cell = FieldCellState(
                concept_id=cid,
                base_confidence=float(concept.confidence),
                base_freshness=freshness,
                base_purity=purity,
                base_conflict=conflict,
                support_count=int(concept.support_count),
                supporting_agents=list(concept.supporting_agents),
                centroid_xy=concept.centroid_xy,
            )

            prev_cell = prev_cells.get(cid)
            for channel in self.channels:
                affinity = self._channel_affinity(channel, concept.semantic_profile)
                raw_act = max(
                    0.0,
                    self.alpha_belief * belief * affinity
                    - self.eta_conflict * conflict,
                )
                # EMA continuity: blend previous activation if available
                if prev_cell is not None and channel in prev_cell.channels:
                    prev_act = prev_cell.channels[channel].activation
                    prev_fast = prev_cell.channels[channel].activation_fast
                    prev_slow = prev_cell.channels[channel].activation_slow
                else:
                    prev_act = prev_fast = prev_slow = raw_act

                # Apply lambda decay to previous activation, then add new input
                decayed_prev = self.lambda_decay * prev_act
                new_act = max(0.0, decayed_prev + raw_act) / 2.0  # blend

                # For fresh builds (no previous), just use raw
                if prev_cell is None:
                    new_act = raw_act

                new_fast = self.ema_fast * new_act + (1.0 - self.ema_fast) * prev_fast
                new_slow = self.ema_slow * new_act + (1.0 - self.ema_slow) * prev_slow

                ch = FieldChannelState(
                    channel=channel,
                    activation=new_act,
                    activation_fast=new_fast,
                    activation_slow=new_slow,
                    freshness=freshness,
                    belief_strength=belief,
                    conflict_pressure=conflict,
                    support_pressure=affinity,
                    last_update_seq=current_seq,
                )
                cell.channels[channel] = ch

            self._cells[cid] = cell

        for _ in range(self.diffusion_steps):
            self._apply_diffusion()

    def _apply_diffusion(self) -> None:
        """One diffusion pass: each concept absorbs γ·weighted_avg(neighbours).

        Neighbours are concepts whose centroid_xy is within diffusion_radius.
        Weight is exp(-dist / diffusion_radius) (Gaussian-like).
        """
        concepts = list(self._cells.items())
        deltas: Dict[Tuple[str, str], float] = {}  # (cid, channel) → delta

        for cid, cell in concepts:
            if cell.centroid_xy is None:
                continue
            for channel in self.channels:
                ch = cell.channels.get(channel)
                if ch is None:
                    continue
                total_w = 0.0
                weighted = 0.0
                for cid2, cell2 in concepts:
                    if cid2 == cid or cell2.centroid_xy is None:
                        continue
                    ch2 = cell2.channels.get(channel)
                    if ch2 is None:
                        continue
                    dist = math.dist(cell.centroid_xy, cell2.centroid_xy)
                    if dist > self.diffusion_radius:
                        continue
                    w = math.exp(-dist / self.diffusion_radius)
                    weighted += w * ch2.activation
                    total_w += w
                if total_w > 0:
                    deltas[(cid, channel)] = (
                        self.gamma_diffusion * weighted / total_w
                    )

        for (cid, channel), delta in deltas.items():
            ch = self._cells[cid].channels.get(channel)
            if ch is not None:
                ch.activation = max(0.0, ch.activation + delta)

    # ──────────────────────────────────────────────────────────── decay

    def decay(self, current_seq: int, steps: Optional[int] = None) -> None:
        """Apply temporal decay λ^steps to all activations.

        If ``steps`` is None, computed from (current_seq - _build_seq).
        Useful to simulate time passing without a full rebuild.
        """
        n = steps if steps is not None else max(0, current_seq - self._build_seq)
        if n == 0:
            return
        factor = self.lambda_decay ** n
        for cell in self._cells.values():
            for ch in cell.channels.values():
                ch.activation = max(0.0, ch.activation * factor)
                ch.activation_fast = max(0.0, ch.activation_fast * factor)
                ch.activation_slow = max(0.0, ch.activation_slow * factor)

    # ─────────────────────────────────────────────────────────────── query

    def query(
        self,
        channel: str,
        requesting_agent_id: str,
        env_id: Optional[str] = None,
        top_k: int = 3,
        min_activation: float = 0.0,
        graph: Optional[Any] = None,
        current_seq: int = 0,
    ) -> List[FieldQueryResult]:
        """Query concepts sorted by field activation for a given channel.

        Parameters
        ----------
        channel:
            The semantic channel to query (must be in ``self.channels``).
        graph:
            If provided, member_places in results are filtered by ``env_id``.
        """
        results: List[FieldQueryResult] = []

        for cid, cell in self._cells.items():
            ch = cell.channels.get(channel)
            if ch is None or ch.activation < min_activation:
                continue

            member_places: List[Tuple[str, Any]] = []
            if graph is not None:
                concept = graph.concepts.get(cid)
                if concept is not None:
                    places = list(concept.member_place_keys)
                    if env_id is not None:
                        places = [(e, k) for e, k in places if e == env_id]
                    member_places = places

            results.append(
                FieldQueryResult(
                    concept_id=cid,
                    channel=channel,
                    activation=ch.activation,
                    concept_score=cell.base_confidence,
                    field_score=ch.activation,
                    combined_score=ch.activation,
                    centroid_xy=cell.centroid_xy,
                    member_places=member_places,
                    supporting_agents=list(cell.supporting_agents),
                    explanation={
                        "belief_strength": ch.belief_strength,
                        "channel_affinity": ch.support_pressure,
                        "conflict_pressure": ch.conflict_pressure,
                        "freshness": ch.freshness,
                        "base_confidence": cell.base_confidence,
                        "base_purity": cell.base_purity,
                    },
                )
            )

        results.sort(key=lambda r: r.activation, reverse=True)
        return results[:top_k]

    def explain_score(self, concept_id: str, channel: str) -> Dict[str, float]:
        """Return the activation breakdown for (concept, channel)."""
        cell = self._cells.get(concept_id)
        if cell is None:
            return {}
        ch = cell.channels.get(channel)
        if ch is None:
            return {}
        return {
            "activation": ch.activation,
            "belief_strength": ch.belief_strength,
            "channel_affinity": ch.support_pressure,
            "conflict_pressure": ch.conflict_pressure,
            "freshness": ch.freshness,
            "activation_fast": ch.activation_fast,
            "activation_slow": ch.activation_slow,
            "base_confidence": cell.base_confidence,
            "base_purity": cell.base_purity,
            "base_conflict": cell.base_conflict,
            "support_count": cell.support_count,
        }

    # ──────────────────────────────────────────────────── reranking (Phase 4b)

    def rerank(
        self,
        concept_recall_results: List[Any],
        channel: str,
        field_weight: float = 0.30,
    ) -> List[Any]:
        """Rerank concept recall results by combined concept_score + field activation.

        Input ``concept_recall_results`` must have ``.concept_id`` and ``.score``
        attributes (``ConceptRecallResult`` from ``concept_recall.py``).

        Both concept scores and field activations are independently normalised to
        [0, 1] before combining, so neither dominates purely due to scale.

        Returns the same objects in reranked order (no copies made).
        """
        if not concept_recall_results:
            return []

        activations = {
            cid: cell.channels[channel].activation
            for cid, cell in self._cells.items()
            if channel in cell.channels
        }
        if not activations:
            return concept_recall_results

        max_score = max(
            (float(r.score) for r in concept_recall_results), default=1.0
        ) or 1.0
        max_act = max(activations.values()) or 1.0

        scored: List[Tuple[float, Any]] = []
        for r in concept_recall_results:
            norm_c = float(r.score) / max_score
            norm_f = activations.get(r.concept_id, 0.0) / max_act
            combined = (1.0 - field_weight) * norm_c + field_weight * norm_f
            scored.append((combined, r))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored]

    # ──────────────────────────────────────────── reservation stubs (Phase 4c)

    def reserve(
        self,
        concept_id: str,
        channel: str,
        agent_id: str,
        duration: int,
        current_seq: int,
    ) -> Optional[FieldReservation]:
        """Soft-reserve (concept, channel) for ``agent_id``.

        Mode check is intentionally absent: the FieldAdapter controls whether
        reservations are allowed based on its own FieldMode.  This keeps the
        SemanticField free of mode-switching logic.

        Side-effects (immediate, not deferred to next rebuild):
        - ``ch.reservation_pressure`` tracks cumulative reservation load.
        - ``ch.activation`` is immediately reduced by ``xi_occupancy`` so that
          other agents querying the field right after reservation see the penalised
          value and are naturally redirected to alternatives.
        """
        res = FieldReservation(
            concept_id=concept_id,
            channel=channel,
            agent_id=agent_id,
            reserved_at_seq=current_seq,
            expires_at_seq=current_seq + duration,
        )
        self._reservations[(concept_id, agent_id)] = res
        cell = self._cells.get(concept_id)
        if cell is not None:
            ch = cell.channels.get(channel)
            if ch is not None:
                ch.reservation_pressure = min(
                    1.0, ch.reservation_pressure + self.xi_occupancy
                )
                # Immediate occupancy penalty: −ξ·U_t
                ch.activation = max(0.0, ch.activation - self.xi_occupancy)
        return res

    def release(self, concept_id: str, agent_id: str) -> None:
        """Release a reservation and restore the occupancy penalty on activation."""
        res = self._reservations.pop((concept_id, agent_id), None)
        if res is None:
            return
        cell = self._cells.get(concept_id)
        if cell is not None:
            ch = cell.channels.get(res.channel)
            if ch is not None:
                ch.reservation_pressure = max(
                    0.0, ch.reservation_pressure - self.xi_occupancy
                )
                # Restore the penalty (next rebuild will compute fresh activation)
                ch.activation = min(1.0, ch.activation + self.xi_occupancy)

    def expire_reservations(self, current_seq: int) -> int:
        """Expire and release all reservations past their TTL."""
        keys = [
            k for k, r in self._reservations.items()
            if r.expires_at_seq <= current_seq
        ]
        for cid, aid in keys:
            self.release(cid, aid)
        return len(keys)

    # ──────────────────────────────────────────────────────── introspection

    @property
    def cells(self) -> Dict[str, FieldCellState]:
        return self._cells

    @property
    def build_seq(self) -> int:
        return self._build_seq

    def activation_for(self, concept_id: str, channel: str) -> float:
        cell = self._cells.get(concept_id)
        if cell is None:
            return 0.0
        ch = cell.channels.get(channel)
        return ch.activation if ch is not None else 0.0

    def top_k_for_channel(
        self, channel: str, k: int = 3
    ) -> List[Tuple[str, float]]:
        """Return [(concept_id, activation)] sorted descending."""
        items = [
            (cid, cell.channels[channel].activation)
            for cid, cell in self._cells.items()
            if channel in cell.channels
        ]
        items.sort(key=lambda x: x[1], reverse=True)
        return items[:k]

    def to_snapshot(self) -> Dict:
        """Serialise field state to a JSON-compatible dict."""
        return {
            "build_seq": self._build_seq,
            "mode": self.mode,
            "channels": self.channels,
            "cells": {
                cid: {
                    "base_confidence": round(cell.base_confidence, 5),
                    "base_purity": round(cell.base_purity, 5),
                    "base_conflict": round(cell.base_conflict, 5),
                    "support_count": cell.support_count,
                    "centroid_xy": (
                        [round(v, 3) for v in cell.centroid_xy]
                        if cell.centroid_xy else None
                    ),
                    "channels": {
                        ch_name: {
                            "activation": round(ch.activation, 5),
                            "activation_fast": round(ch.activation_fast, 5),
                            "activation_slow": round(ch.activation_slow, 5),
                            "belief_strength": round(ch.belief_strength, 5),
                            "channel_affinity": round(ch.support_pressure, 5),
                            "conflict_pressure": round(ch.conflict_pressure, 5),
                            "freshness": round(ch.freshness, 5),
                        }
                        for ch_name, ch in cell.channels.items()
                    },
                }
                for cid, cell in self._cells.items()
            },
        }
