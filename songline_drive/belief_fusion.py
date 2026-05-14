"""Phase 3 — temporal decay (3a) + conflict fusion (3b).

Two engines, both read-only with respect to CollectiveMemory:

``TemporalDecayEngine`` (Phase 3a)
    Computes per-concept decayed confidence and freshness based on how
    many global ``wall_clock_seq`` steps have passed since the concept
    last received support.  Concepts below ``deactivation_threshold``
    are "stale"; concepts that receive new evidence and rise above
    ``recovery_threshold`` have "recovered".

``ConflictRuleSet`` + ``ConflictRule`` (Phase 3b)
    Domain-specific rules that penalise a concept whose semantic profile
    contains competing positive and negative signals (e.g. water_source
    alongside hazard_edge).  The penalty is score-multiplicative, not
    additive — a pure water concept is not penalised at all; a concept
    that accumulates both water and hazard evidence is suppressed in
    proportion to how strong both signals are.

Both engines plug into ``ConceptRecallLayer`` via constructor arguments;
they can also be applied directly in metrics or diagnostic scripts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


# ─────────────────────────────────────────────────────────── Phase 3a helpers


@dataclass
class TemporalDecayEngine:
    """Per-concept temporal decay and lifecycle management.

    Parameters
    ----------
    decay_lambda:
        Multiplicative decay per ``wall_clock_seq`` step.  Matches the
        ``recency_lambda`` in ``CollectiveMemory`` (default 0.97) so that
        a concept that stopped receiving support loses ~50% confidence
        after ~23 steps and drops below 0.10 after ~75 steps.
    deactivation_threshold:
        Decayed confidence below which the concept is considered *stale*
        and should not appear in query results.
    recovery_threshold:
        Decayed confidence above which a previously-stale concept is
        considered *recovered* after receiving fresh support.
    """

    decay_lambda: float = 0.97
    deactivation_threshold: float = 0.10
    recovery_threshold: float = 0.30

    # ---------------------------------------------------------------- core

    def decayed_confidence(self, concept: Any, current_seq: int) -> float:
        """Concept confidence after applying temporal decay."""
        if concept.last_seen_seq < 0:
            return 0.0
        age = max(0, int(current_seq) - int(concept.last_seen_seq))
        return float(concept.confidence) * (self.decay_lambda ** age)

    def decayed_freshness(self, concept: Any, current_seq: int) -> float:
        """Freshness after decay (clamped to [0, 1])."""
        if concept.last_seen_seq < 0:
            return 0.0
        age = max(0, int(current_seq) - int(concept.last_seen_seq))
        return min(1.0, float(concept.freshness) * (self.decay_lambda ** age))

    def is_active(self, concept: Any, current_seq: int) -> bool:
        return self.decayed_confidence(concept, current_seq) >= self.deactivation_threshold

    def steps_to_deactivation(self, concept: Any, current_seq: int) -> Optional[int]:
        """Steps from *now* until this concept's confidence drops below threshold.

        Returns ``None`` if already inactive or confidence is 0."""
        dc = self.decayed_confidence(concept, current_seq)
        if dc <= 0.0:
            return None
        if dc < self.deactivation_threshold:
            return 0
        if self.decay_lambda >= 1.0:
            return None  # never decays
        steps = math.log(self.deactivation_threshold / dc) / math.log(self.decay_lambda)
        return max(0, int(math.ceil(steps)))

    # ---------------------------------------------------------------- graph-level

    def apply_to_graph(
        self,
        graph: Any,
        current_seq: int,
    ) -> Dict[str, bool]:
        """Update ``concept.freshness`` on every concept and return
        ``{concept_id: is_active}`` mapping.

        Side-effect: writes decayed freshness back to each concept so that
        subsequent queries can read ``concept.freshness`` as the current
        decay-adjusted value.  ``concept.confidence`` is *not* mutated —
        raw confidence is preserved for recovery tracking.
        """
        status: Dict[str, bool] = {}
        for cid, concept in graph.concepts.items():
            concept.freshness = self.decayed_freshness(concept, current_seq)
            status[cid] = self.is_active(concept, current_seq)
        return status

    def filter_active(
        self,
        graph: Any,
        current_seq: int,
        results: List[Any],  # List[ConceptQueryResult] or similar
    ) -> List[Any]:
        """Filter a list of query results to only include active concepts."""
        return [
            r for r in results
            if self.is_active(graph.concepts[r.concept_id], current_seq)
            if r.concept_id in graph.concepts
        ]


# ─────────────────────────────────────────────────────────── Phase 3b helpers


@dataclass(frozen=True)
class ConflictRule:
    """A single directed conflict: ``negative_tag`` evidence penalises
    the score of concepts that also carry ``positive_tag`` evidence.

    ``conflict_weight`` ∈ [0, 1]: 0 = no penalty, 1 = full suppression
    when both signals are at maximum confidence.
    """

    positive_tag: str
    negative_tag: str
    conflict_weight: float = 0.5


class ConflictRuleSet:
    """Domain-specific set of conflict rules applied during concept scoring.

    The conflict penalty for a concept is:

    .. code-block::

        penalty = min(1.0, Σ_rules  pos_signal × neg_signal × rule.weight)

    The adjusted score is:

    .. code-block::

        adjusted = base_score × (1 - penalty)

    This is multiplicative: a concept with no conflicting signal is
    unaffected; one where both signals are at full strength and
    conflict_weight = 1.0 gets score = 0.

    Parameters
    ----------
    rules:
        List of ``ConflictRule`` instances.
    """

    def __init__(self, rules: List[ConflictRule]) -> None:
        self.rules = list(rules)

    # ---------------------------------------------------------------- factory

    @classmethod
    def songlines_default(cls) -> "ConflictRuleSet":
        """Default rules for the Songlines domain.

        Priority ordering (by conflict_weight):
        - goal_region ↔ hazard_edge (1.0)  — never navigate into a hazard goal
        - water_source ↔ hazard_edge (0.85) — water at a hazard cell is suspect
        - water_source ↔ adjacent_hazard (0.50) — partial suppression
        - safe_neutral ↔ hazard_edge (0.70) — safe cell next to hazard is suspect
        - safe_neutral ↔ adjacent_hazard (0.40)
        """
        return cls(rules=[
            ConflictRule("goal_region", "hazard_edge", 1.0),
            ConflictRule("goal_region", "adjacent_hazard", 0.60),
            ConflictRule("water_source", "hazard_edge", 0.85),
            ConflictRule("water_source", "adjacent_hazard", 0.50),
            ConflictRule("safe_neutral", "hazard_edge", 0.70),
            ConflictRule("safe_neutral", "adjacent_hazard", 0.40),
        ])

    # ---------------------------------------------------------------- core

    def conflict_penalty(self, concept: Any) -> float:
        """Conflict penalty ∈ [0, 1] for the given concept."""
        profile = concept.semantic_profile
        total = 0.0
        for rule in self.rules:
            pos = float(profile.get(rule.positive_tag, 0.0))
            neg = float(profile.get(rule.negative_tag, 0.0))
            if pos <= 0.0 or neg <= 0.0:
                continue
            total += pos * neg * float(rule.conflict_weight)
        return min(1.0, total)

    def adjusted_score(self, concept: Any, base_score: float) -> float:
        """Score after applying conflict penalty."""
        penalty = self.conflict_penalty(concept)
        return float(base_score) * (1.0 - penalty)

    def concept_purity(self, concept: Any) -> float:
        """Concept purity ∈ [0, 1].

        1.0 = no conflicting evidence, 0.0 = maximally conflicted.
        Low purity means the concept's member places carry strongly
        competing semantic signals."""
        return 1.0 - self.conflict_penalty(concept)

    def has_significant_conflict(
        self, concept: Any, threshold: float = 0.20
    ) -> bool:
        return self.conflict_penalty(concept) >= threshold

    def tag_conflict_pairs(
        self, concept: Any
    ) -> List[Tuple[str, str, float, float, float]]:
        """Diagnostic: return (pos_tag, neg_tag, pos_val, neg_val, contribution)
        for every rule that fires on this concept."""
        profile = concept.semantic_profile
        out = []
        for rule in self.rules:
            pos = float(profile.get(rule.positive_tag, 0.0))
            neg = float(profile.get(rule.negative_tag, 0.0))
            if pos > 0.0 and neg > 0.0:
                out.append((
                    rule.positive_tag, rule.negative_tag,
                    pos, neg,
                    pos * neg * rule.conflict_weight,
                ))
        return out

    # ---------------------------------------------------------------- graph-level

    def apply_to_graph(self, graph: Any) -> Dict[str, float]:
        """Compute and store ``conflict_score`` on every concept.

        Returns ``{concept_id: conflict_score}`` for diagnostic use.
        Does NOT modify ``confidence``; the raw signal is preserved.
        """
        scores: Dict[str, float] = {}
        for cid, concept in graph.concepts.items():
            concept.conflict_score = self.conflict_penalty(concept)
            scores[cid] = concept.conflict_score
        return scores

    def filter_clean(
        self,
        graph: Any,
        results: List[Any],
        max_penalty: float = 0.5,
    ) -> List[Any]:
        """Filter query results to concepts whose conflict penalty ≤ max_penalty."""
        out = []
        for r in results:
            concept = graph.concepts.get(r.concept_id)
            if concept is None:
                continue
            if self.conflict_penalty(concept) <= max_penalty:
                out.append(r)
        return out
