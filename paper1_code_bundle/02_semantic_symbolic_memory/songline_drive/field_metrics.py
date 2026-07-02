"""Phase 4 — semantic field metrics.

All functions are pure (read-only with respect to SemanticField).

Phase 4a — descriptive validation
    field_activation_split
    field_conflict_suppression_rate
    field_cross_channel_separation
    field_top1_stability
    field_decay_half_life
    field_activation_to_query_rank_correlation

Phase 4b — read_only reranking quality
    field_rerank_precision_at_k
    field_assisted_rank_gain
    field_top1_gain

Phase 4c — coordinated mode (stubs for future use)
    duplicate_target_rate
    reservation_conflict_rate
    field_driven_deconfliction_rate

Aggregates
    all_field_metrics_4a
    all_field_metrics_4b
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Dict, List, Optional, Set, Tuple

from songline_drive.semantic_field import SemanticField


# ─────────────────────────────────────────────────────── Phase 4a metrics


def field_activation_split(
    field: SemanticField,
    channel: str,
    concept_tag_map: Dict[str, str],
) -> Dict[str, float]:
    """Mean activation per dominant_tag for a given channel.

    Returns ``{dominant_tag: mean_activation}``.  Use this to verify
    water concepts have higher water_source activation than hazard concepts.
    """
    buckets: Dict[str, List[float]] = {}
    for cid, cell in field.cells.items():
        ch = cell.channels.get(channel)
        if ch is None:
            continue
        tag = concept_tag_map.get(cid, "unknown")
        buckets.setdefault(tag, []).append(ch.activation)
    return {
        tag: statistics.mean(vals)
        for tag, vals in buckets.items()
        if vals
    }


def field_conflict_suppression_rate(
    field: SemanticField,
    channel: str,
    conflict_threshold: float = 0.20,
    activation_threshold: Optional[float] = None,
) -> Optional[float]:
    """Fraction of high-conflict concepts suppressed below median activation.

    Returns ``None`` if no conflicted concepts exist (metric undefined).
    """
    all_acts = [
        cell.channels[channel].activation
        for cell in field.cells.values()
        if channel in cell.channels
    ]
    if not all_acts:
        return None

    act_cutoff = (
        activation_threshold
        if activation_threshold is not None
        else statistics.median(all_acts)
    )

    conflicted = [
        cid
        for cid, cell in field.cells.items()
        if cell.base_conflict >= conflict_threshold and channel in cell.channels
    ]
    if not conflicted:
        return None

    suppressed = sum(
        1 for cid in conflicted
        if field.cells[cid].channels[channel].activation < act_cutoff
    )
    return suppressed / len(conflicted)


def field_cross_channel_separation(
    field: SemanticField,
    channel_a: str,
    channel_b: str,
) -> float:
    """Mean absolute difference in activation between two channels.

    Higher = better semantic separation (water vs hazard should be high).
    """
    diffs: List[float] = []
    for cell in field.cells.values():
        a = cell.channels.get(channel_a)
        b = cell.channels.get(channel_b)
        if a is None or b is None:
            continue
        diffs.append(abs(a.activation - b.activation))
    return statistics.mean(diffs) if diffs else 0.0


def field_top1_stability(
    field: SemanticField,
    channel: str,
    n_queries: int = 3,
) -> bool:
    """True if repeated identical queries return the same top-1 concept.

    Since the field is deterministic, this verifies internal consistency.
    """
    top1s: Set[str] = set()
    for _ in range(n_queries):
        items = field.top_k_for_channel(channel, k=1)
        if items:
            top1s.add(items[0][0])
    return len(top1s) <= 1


def field_decay_half_life(lambda_decay: float) -> float:
    """Analytical half-life in steps for decay constant λ."""
    if lambda_decay >= 1.0 or lambda_decay <= 0.0:
        return float("inf")
    return math.log(0.5) / math.log(lambda_decay)


def field_activation_to_query_rank_correlation(
    field: SemanticField,
    channel: str,
    ranked_concept_ids: List[str],
) -> float:
    """Spearman rank correlation: field activation vs concept recall rank.

    Positive → higher activation correlates with higher concept score.
    Returns NaN if fewer than 2 concepts.
    """
    n = len(ranked_concept_ids)
    if n < 2:
        return float("nan")

    recall_ranks = {cid: i for i, cid in enumerate(ranked_concept_ids)}
    field_items = field.top_k_for_channel(channel, k=n + 10)
    field_rank_map = {cid: i for i, (cid, _) in enumerate(field_items)}

    pairs = [
        (recall_ranks[cid], field_rank_map.get(cid, n))
        for cid in ranked_concept_ids
    ]
    d_sq = sum((r1 - r2) ** 2 for r1, r2 in pairs)
    return 1.0 - (6.0 * d_sq) / (n * (n * n - 1))


def field_novelty_signal(
    field: SemanticField,
    channel: str,
    concept_id: str,
) -> Optional[float]:
    """Novelty signal: activation_fast - activation_slow for (concept, channel).

    Positive → recent surge in activation (new relevant evidence arrived).
    """
    cell = field.cells.get(concept_id)
    if cell is None:
        return None
    ch = cell.channels.get(channel)
    if ch is None:
        return None
    return ch.activation_fast - ch.activation_slow


# ─────────────────────────────────────────────────────── Phase 4b metrics


def field_rerank_precision_at_k(
    ranked_concept_ids: List[str],
    gt_concept_ids: Set[str],
    k: int = 3,
) -> float:
    """Fraction of top-k results that are GT concepts."""
    top_k = ranked_concept_ids[:k]
    if not top_k:
        return 0.0
    return sum(1 for cid in top_k if cid in gt_concept_ids) / len(top_k)


def field_assisted_rank_gain(
    baseline_ranks: Dict[str, int],
    field_ranks: Dict[str, int],
    gt_concept_ids: Set[str],
) -> float:
    """Mean rank improvement for GT concepts after field reranking.

    Positive = field moved GT concepts higher in the ranking.
    """
    gains: List[float] = []
    for cid in gt_concept_ids:
        base = baseline_ranks.get(cid)
        field = field_ranks.get(cid)
        if base is None or field is None:
            continue
        gains.append(float(base - field))
    return statistics.mean(gains) if gains else float("nan")


def field_top1_gain(
    baseline_top1: Optional[str],
    field_top1: Optional[str],
    gt_concept_ids: Set[str],
) -> int:
    """+1 if field top-1 is GT and baseline is not; -1 vice versa; 0 if tied."""
    base_ok = baseline_top1 in gt_concept_ids if baseline_top1 else False
    field_ok = field_top1 in gt_concept_ids if field_top1 else False
    if field_ok and not base_ok:
        return 1
    if base_ok and not field_ok:
        return -1
    return 0


def field_rerank_delta_steps(
    baseline_nav_steps: float,
    field_nav_steps: float,
) -> float:
    """Navigation step delta: positive means field saved steps."""
    return baseline_nav_steps - field_nav_steps


# ─────────────────────────────────────────── Phase 4c stubs (future)


def duplicate_target_rate(
    agent_targets: List[Tuple[str, str]],
) -> float:
    """Fraction of (agent, target_concept) pairs that share a target concept."""
    if len(agent_targets) < 2:
        return 0.0
    targets = [t for _, t in agent_targets]
    dupes = len(targets) - len(set(targets))
    return dupes / len(targets)


def reservation_conflict_rate(
    reservations: List[Any],
) -> float:
    """Fraction of reservations that overlap with another agent's reservation."""
    concept_agents: Dict[str, List[str]] = {}
    for r in reservations:
        concept_agents.setdefault(r.concept_id, []).append(r.agent_id)
    conflicted = sum(
        len(agents) - 1
        for agents in concept_agents.values()
        if len(agents) > 1
    )
    return conflicted / max(1, len(reservations))


def field_driven_deconfliction_rate(
    baseline_targets: List[Tuple[str, str]],   # [(agent_id, concept_id), ...]
    coordinated_targets: List[Tuple[str, str]],
) -> float:
    """Fraction of baseline collisions resolved by coordinated field mode.

    A "collision" is two or more agents targeting the same concept_id.
    Returns 1.0 if all collisions are resolved, 0.0 if none.
    Returns NaN if there were no baseline collisions (nothing to deconflict).
    """
    def collision_pairs(targets: List[Tuple[str, str]]) -> int:
        from collections import Counter
        counts = Counter(cid for _, cid in targets)
        return sum(c - 1 for c in counts.values() if c > 1)

    base_coll = collision_pairs(baseline_targets)
    if base_coll == 0:
        return float("nan")
    coord_coll = collision_pairs(coordinated_targets)
    return max(0.0, (base_coll - coord_coll) / base_coll)


# ─────────────────────────────────────────────────────── aggregates


def all_field_metrics_4a(
    field: SemanticField,
    channel: str,
    concept_tag_map: Dict[str, str],
    conflict_threshold: float = 0.20,
    reference_channel: Optional[str] = None,
    ranked_concept_ids: Optional[List[str]] = None,
) -> Dict:
    """Aggregate Phase 4a diagnostic metrics."""
    split = field_activation_split(field, channel, concept_tag_map)
    suppression = field_conflict_suppression_rate(
        field, channel, conflict_threshold
    )
    stable = field_top1_stability(field, channel)
    half_life = field_decay_half_life(field.lambda_decay)
    top3 = field.top_k_for_channel(channel, k=3)

    result: Dict = {
        "channel": channel,
        "activation_split_by_tag": split,
        "field_conflict_suppression_rate": suppression,
        "field_top1_stability": stable,
        "field_decay_half_life_steps": half_life,
        "top3_activations": [
            {"concept_id": cid, "activation": round(act, 5)}
            for cid, act in top3
        ],
    }

    if reference_channel is not None:
        result["field_cross_channel_separation"] = field_cross_channel_separation(
            field, channel, reference_channel
        )

    if ranked_concept_ids is not None and len(ranked_concept_ids) >= 2:
        result["field_activation_to_query_rank_correlation"] = (
            field_activation_to_query_rank_correlation(
                field, channel, ranked_concept_ids
            )
        )

    return result


def all_field_metrics_4b(
    field: SemanticField,
    channel: str,
    baseline_recall: List[str],
    field_recall: List[str],
    gt_concept_ids: Set[str],
    k: int = 3,
) -> Dict:
    """Aggregate Phase 4b reranking metrics."""
    baseline_ranks = {cid: i for i, cid in enumerate(baseline_recall)}
    field_ranks_map = {cid: i for i, cid in enumerate(field_recall)}

    return {
        "baseline_precision_at_k": field_rerank_precision_at_k(
            baseline_recall, gt_concept_ids, k
        ),
        "field_precision_at_k": field_rerank_precision_at_k(
            field_recall, gt_concept_ids, k
        ),
        "field_assisted_rank_gain": field_assisted_rank_gain(
            baseline_ranks, field_ranks_map, gt_concept_ids
        ),
        "field_top1_gain": field_top1_gain(
            baseline_recall[0] if baseline_recall else None,
            field_recall[0] if field_recall else None,
            gt_concept_ids,
        ),
    }
