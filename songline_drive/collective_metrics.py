"""Phase 1 collective-memory diagnostics.

Eight metrics mirroring the roadmap. Every metric is provenance-aware:
it can be replayed from the same append-only event log that powers the
substrate, so paper-style audits stay reproducible.

For Phase 1 a couple of metrics are intentionally proxies:
- ``field_activation_to_success_correlation`` returns the correlation
  between *fused tag confidence* and episode success. In Phase 4 this
  will be replaced by the actual ``Φ_t(p,c)`` semantic field; the
  signature stays the same so callers do not change.
- ``time_to_collective_convergence`` operates on the recorded events,
  not on a field; the meaning ("at which global step did the team
  reach consensus on this concept") is consistent across phases.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from songline_drive.collective_memory import CollectiveMemory, _normalize_place_key
from songline_drive.collective_types import BeliefRecord


# ----------------------------------------------------------------- fusion helper


def _fuse_at(
    records: Iterable[BeliefRecord],
    recency_lambda: float,
    cur_seq: int,
    exclude_agent_id: Optional[str] = None,
) -> Tuple[float, Set[str]]:
    total = 0.0
    norm = 0.0
    agents: Set[str] = set()
    for record in records:
        if exclude_agent_id is not None and record.agent_id == exclude_agent_id:
            continue
        recency = recency_lambda ** max(0, cur_seq - record.wall_clock_seq)
        total += record.confidence * recency
        norm += abs(recency)
        agents.add(record.agent_id)
    if norm <= 0.0:
        return 0.0, agents
    return total / norm, agents


# --------------------------------------------------------------------- metric 1


def collective_memory_reuse_rate(
    collective: CollectiveMemory,
    requesting_agent_id: Optional[str] = None,
) -> float:
    """Fraction of recorded queries that returned at least one place backed
    by some contributing agent (i.e. memory was actually used as memory,
    not as an empty surface)."""
    reads = collective.reads_log()
    if requesting_agent_id is not None:
        reads = [r for r in reads if r["requesting_agent_id"] == requesting_agent_id]
    if not reads:
        return 0.0
    reused = sum(1 for r in reads if r["contributing_agents"])
    return reused / len(reads)


# --------------------------------------------------------------------- metric 2


def other_agent_knowledge_use_rate(
    collective: CollectiveMemory,
    requesting_agent_id: Optional[str] = None,
) -> float:
    """Fraction of recorded queries whose top result was supported by at
    least one agent OTHER than the requester. This is the most direct
    smoke-level signal that the substrate is acting as a shared (not
    just personal) memory."""
    reads = collective.reads_log()
    if requesting_agent_id is not None:
        reads = [r for r in reads if r["requesting_agent_id"] == requesting_agent_id]
    if not reads:
        return 0.0
    used_other = sum(1 for r in reads if r.get("used_other_agent_knowledge"))
    return used_other / len(reads)


# --------------------------------------------------------------------- metric 3


def concept_transfer_precision(
    collective: CollectiveMemory,
    ground_truth_places_by_tag: Dict[str, Set[Tuple[Any, ...]]],
    concept_tag: str,
    min_fused: float = 0.3,
) -> Optional[float]:
    """Precision: among places where collective memory currently holds
    ``concept_tag`` above ``min_fused`` fused confidence, what fraction
    is actually in the ground-truth set for that tag?"""
    truth = {tuple(p) for p in ground_truth_places_by_tag.get(concept_tag, set())}
    if not truth:
        return None
    cur_seq = collective._next_seq  # noqa: SLF001 — diagnostic, intentional
    total = 0
    correct = 0
    for (env_id, place_key), aggregate in collective.place_beliefs.items():
        records = aggregate.tag_records.get(concept_tag, [])
        if not records:
            continue
        fused, _ = _fuse_at(records, collective.recency_lambda, cur_seq)
        if fused < min_fused:
            continue
        total += 1
        if tuple(place_key) in truth:
            correct += 1
    if total == 0:
        return 0.0
    return correct / total


# --------------------------------------------------------------------- metric 4


def belief_conflict_resolution_accuracy(
    collective: CollectiveMemory,
    ground_truth_tag_per_place: Dict[Tuple[Any, ...], str],
    min_alternate_support: int = 2,
) -> Optional[float]:
    """For every place where at least two tags accumulated positive
    support (i.e. there was a real conflict to resolve), check whether
    the currently-dominant fused tag matches ground truth.

    A place that received only one tag is excluded — there was nothing
    to resolve."""
    truth = {tuple(k): str(v) for k, v in ground_truth_tag_per_place.items()}
    if not truth:
        return None
    cur_seq = collective._next_seq  # noqa: SLF001
    total = 0
    correct = 0
    for (env_id, place_key), aggregate in collective.place_beliefs.items():
        if tuple(place_key) not in truth:
            continue
        tag_fused: Dict[str, float] = {}
        for tag, records in aggregate.tag_records.items():
            if not records:
                continue
            fused, _ = _fuse_at(records, collective.recency_lambda, cur_seq)
            if fused > 0.0:
                tag_fused[tag] = fused
        if len(tag_fused) < min_alternate_support:
            continue
        best_tag = max(tag_fused.items(), key=lambda kv: kv[1])[0]
        total += 1
        if best_tag == truth[tuple(place_key)]:
            correct += 1
    if total == 0:
        return None
    return correct / total


# --------------------------------------------------------------------- metric 5


def field_activation_to_success_correlation(
    collective: CollectiveMemory,
    episode_outcomes: Sequence[Dict[str, Any]],
    target_tag: Optional[str] = None,
) -> Optional[float]:
    """Pearson correlation between the mean fused tag confidence of the
    places an agent actually visited during the episode and episode
    success.

    Phase 1 proxy for ``Φ_t(p,c)``: replaces field activation with the
    average ``fused(target_tag)`` over visited places at the *moment
    the episode ended*. Returns ``None`` if there are fewer than two
    eligible episodes or no variance.
    """
    if not episode_outcomes:
        return None
    cur_seq = collective._next_seq  # noqa: SLF001
    xs: List[float] = []
    ys: List[float] = []
    for ep in episode_outcomes:
        visited = ep.get("visited_place_keys") or []
        if not visited:
            continue
        env_id = str(ep.get("env_id", ""))
        ep_tag = target_tag or ep.get("target_concept_tag")
        if not ep_tag:
            continue
        place_scores: List[float] = []
        for raw in visited:
            place_key = _normalize_place_key(raw)
            aggregate = collective.place_beliefs.get((env_id, place_key))
            if aggregate is None:
                continue
            fused, _ = _fuse_at(
                aggregate.tag_records.get(ep_tag, []),
                collective.recency_lambda,
                cur_seq,
            )
            place_scores.append(fused)
        if not place_scores:
            continue
        xs.append(sum(place_scores) / len(place_scores))
        ys.append(float(ep.get("success", 0.0)))
    if len(xs) < 2:
        return None
    return _pearson(xs, ys)


# --------------------------------------------------------------------- metric 6


def time_to_collective_convergence(
    collective: CollectiveMemory,
    concept_tag: str,
    convergence_min_agents: int = 2,
    convergence_min_score: Optional[float] = None,
) -> Optional[int]:
    """Earliest ``wall_clock_seq`` at which some place reached the fused
    confidence threshold on ``concept_tag`` with at least
    ``convergence_min_agents`` distinct supporters.

    Replays the event log in seq order — so the answer is independent
    of post-hoc decay state and reproducible from the JSONL dump.
    """
    threshold = (
        float(convergence_min_score)
        if convergence_min_score is not None
        else collective.convergence_min_score
    )
    streams: Dict[Tuple[str, Tuple[Any, ...]], List[Tuple[int, str, float]]] = {}
    events_sorted = sorted(collective.all_events(), key=lambda e: e.provenance.wall_clock_seq)
    earliest: Optional[int] = None
    for event in events_sorted:
        tags = event.payload.get("semantic_tags") or {}
        if concept_tag not in tags:
            continue
        place_key = _normalize_place_key(event.payload.get("place_key"))
        if not place_key:
            continue
        trust = collective._trust(event.provenance.agent_id)  # noqa: SLF001
        raw_conf = max(0.0, float(tags[concept_tag])) * event.confidence * trust
        if raw_conf <= 0.0:
            continue
        stream = streams.setdefault((event.provenance.env_id, place_key), [])
        stream.append((event.provenance.wall_clock_seq, event.provenance.agent_id, raw_conf))
        agents = {a for _, a, _ in stream}
        if len(agents) < convergence_min_agents:
            continue
        cur_seq = event.provenance.wall_clock_seq
        total = 0.0
        norm = 0.0
        for seq, _agent, conf in stream:
            recency = collective.recency_lambda ** max(0, cur_seq - seq)
            total += conf * recency
            norm += recency
        fused = total / norm if norm > 0 else 0.0
        if fused >= threshold:
            if earliest is None or event.provenance.wall_clock_seq < earliest:
                earliest = event.provenance.wall_clock_seq
            break
    return earliest


# --------------------------------------------------------------------- metric 7


def collective_semantic_path_completion(
    collective: CollectiveMemory,
    episode_outcomes: Sequence[Dict[str, Any]],
    min_fused: float = 0.3,
) -> Optional[float]:
    """Episode-level analogue of ``semantic_path_completion`` from the
    paper: for each episode with a declared ``target_concept_tag`` and
    a ``terminal_place_key``, did the agent end on a place that the
    collective memory currently endorses as that concept?"""
    if not episode_outcomes:
        return None
    cur_seq = collective._next_seq  # noqa: SLF001
    total = 0
    matched = 0
    for ep in episode_outcomes:
        tag = ep.get("target_concept_tag")
        terminal = ep.get("terminal_place_key")
        if not tag or terminal is None:
            continue
        total += 1
        env_id = str(ep.get("env_id", ""))
        aggregate = collective.place_beliefs.get((env_id, _normalize_place_key(terminal)))
        if aggregate is None:
            continue
        fused, _ = _fuse_at(
            aggregate.tag_records.get(tag, []),
            collective.recency_lambda,
            cur_seq,
        )
        if fused >= min_fused:
            matched += 1
    if total == 0:
        return None
    return matched / total


# --------------------------------------------------------------------- metric 8


def agent_contribution_entropy(collective: CollectiveMemory) -> float:
    """Shannon entropy (bits) of the distribution of events over agents.

    High entropy = workload spread across the team; low entropy = one
    agent dominates the substrate. Returns ``0.0`` when fewer than two
    agents have contributed."""
    counts: Dict[str, int] = {}
    for event in collective.all_events():
        counts[event.provenance.agent_id] = counts.get(event.provenance.agent_id, 0) + 1
    total = sum(counts.values())
    if total <= 0 or len(counts) <= 1:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


# --------------------------------------------------------------------- metric 9


def observation_to_use_latency(
    collective: CollectiveMemory,
    *,
    requesting_agent_id: Optional[str] = None,
    only_other_agent_sources: bool = True,
) -> Optional[Dict[str, float]]:
    """Per-query latency between the *earliest* contributing observation
    and the moment the requester actually read that place.

    For each logged query that returned at least one supporting record,
    we look up the records the substrate would have fused and take
    ``read_seq - min(contributing_seq)``. By default we keep only
    records published by some agent OTHER than the requester — that is
    exactly the "delay from one agent's observation to another agent's
    use" Phase 1 asks for.

    Returns a small distribution summary in global wall-clock-seq units
    (one unit = one published event), or ``None`` if there is nothing
    to measure."""
    reads = collective.reads_log()
    if requesting_agent_id is not None:
        reads = [r for r in reads if r["requesting_agent_id"] == requesting_agent_id]
    if not reads:
        return None

    deltas: List[int] = []
    for entry in reads:
        env_id = entry.get("env_id")
        place_key = _normalize_place_key(entry.get("place_key", ()))
        target_tag = str(entry.get("target_tag", ""))
        read_seq = int(entry.get("wall_clock_seq", 0))
        requester = entry.get("requesting_agent_id")
        aggregate = collective.place_beliefs.get((env_id, place_key))
        if aggregate is None:
            continue
        candidate_seqs: List[int] = []
        for record in aggregate.tag_records.get(target_tag, []):
            if record.wall_clock_seq > read_seq:
                continue
            if only_other_agent_sources and record.agent_id == requester:
                continue
            candidate_seqs.append(record.wall_clock_seq)
        if not candidate_seqs:
            continue
        deltas.append(read_seq - min(candidate_seqs))

    if not deltas:
        return None
    deltas_sorted = sorted(deltas)
    n = len(deltas_sorted)
    median = (
        deltas_sorted[n // 2]
        if n % 2 == 1
        else (deltas_sorted[n // 2 - 1] + deltas_sorted[n // 2]) / 2
    )
    return {
        "n": float(n),
        "min": float(deltas_sorted[0]),
        "median": float(median),
        "mean": float(sum(deltas_sorted) / n),
        "max": float(deltas_sorted[-1]),
        "unit": "wall_clock_seq",
    }


# ----------------------------------------------------------------- concept metrics (Phase 2)
# These operate on a SharedConceptGraph built by PlaceAlignmentEngine,
# not directly on CollectiveMemory, so they accept the graph as first arg.


def concept_coverage_rate(
    graph: Any,  # SharedConceptGraph — typed loosely to avoid circular import
    ground_truth_places_by_tag: Dict[str, Set[Tuple[Any, ...]]],
    target_tag: str,
) -> Optional[float]:
    """Fraction of ground-truth places for ``target_tag`` that appear as
    members of any concept with that dominant_tag.

    0.0 = concept graph missed all GT places for this tag.
    1.0 = every GT place is represented in the concept vocabulary."""
    truth = {tuple(p) for p in ground_truth_places_by_tag.get(target_tag, set())}
    if not truth:
        return None
    covered: Set[Tuple[Any, ...]] = set()
    for concept in graph.concepts.values():
        if concept.dominant_tag != target_tag:
            continue
        for _env_id, place_key in concept.member_place_keys:
            covered.add(tuple(place_key))
    return len(covered & truth) / len(truth)


def cross_agent_concept_support_rate(
    graph: Any,
    min_agents: int = 2,
) -> float:
    """Fraction of concepts supported by at least ``min_agents`` distinct agents.

    The headline Phase 2 signal: if all concepts are still single-agent-local,
    the clustering did not achieve cross-agent consolidation."""
    if not graph.concepts:
        return 0.0
    multi = sum(
        1 for c in graph.concepts.values()
        if len(c.supporting_agents) >= min_agents
    )
    return multi / len(graph.concepts)


def concept_query_precision(
    graph: Any,
    ground_truth_places_by_tag: Dict[str, Set[Tuple[Any, ...]]],
    target_tag: str,
    top_k: int = 3,
    only_dominant_tag: bool = True,
) -> Optional[float]:
    """Precision of concept-level query: among the top-k concepts returned
    by ``graph.query_concepts`` for ``target_tag``, what fraction of their
    member places fall in the ground-truth set for that tag?

    ``only_dominant_tag=True`` (default) filters to concepts whose dominant
    tag matches ``target_tag``, eliminating noise-contaminated results and
    yielding a precision figure that reflects the clustering quality rather
    than the false-positive noise floor."""
    truth = {tuple(p) for p in ground_truth_places_by_tag.get(target_tag, set())}
    if not truth:
        return None
    results = graph.query_concepts(
        target_tag=target_tag, top_k=top_k, only_dominant_tag=only_dominant_tag
    )
    if not results:
        return 0.0
    total = 0
    correct = 0
    for res in results:
        concept = graph.concepts.get(res.concept_id)
        if concept is None:
            continue
        for _env_id, place_key in concept.member_place_keys:
            total += 1
            if tuple(place_key) in truth:
                correct += 1
    if total == 0:
        return 0.0
    return correct / total


def concept_member_diversity_entropy(graph: Any) -> float:
    """Mean Shannon entropy (bits) of the semantic profile across all concepts.

    Low entropy = concepts are semantically pure (good, tight clustering).
    High entropy = concepts mix many tag types (ambiguous or noisy clustering).
    A value near 0 after Phase 2 clustering on a clean world means concepts
    have stable, single-dominant semantics."""
    from songline_drive.collective_concepts import profile_entropy  # local to avoid circular
    if not graph.concepts:
        return 0.0
    entropies = [
        profile_entropy(c.semantic_profile)
        for c in graph.concepts.values()
        if c.semantic_profile
    ]
    if not entropies:
        return 0.0
    return sum(entropies) / len(entropies)


def all_concept_metrics(
    graph: Any,
    ground_truth_places_by_tag: Optional[Dict[str, Set[Tuple[Any, ...]]]] = None,
    target_tag: Optional[str] = None,
    min_agents_for_cross: int = 2,
) -> Dict[str, Any]:
    """Aggregated Phase 2 concept-level report."""
    report: Dict[str, Any] = {
        "graph_stats": graph.stats(),
        "cross_agent_concept_support_rate": cross_agent_concept_support_rate(
            graph, min_agents=min_agents_for_cross
        ),
        "concept_member_diversity_entropy": concept_member_diversity_entropy(graph),
    }
    if ground_truth_places_by_tag and target_tag:
        report["concept_coverage_rate"] = concept_coverage_rate(
            graph, ground_truth_places_by_tag, target_tag
        )
        report["concept_query_precision"] = concept_query_precision(
            graph, ground_truth_places_by_tag, target_tag
        )
    return report


# ---------------------------------------------------------------- Phase 3 metrics
# All Phase 3 metrics accept typed objects from belief_fusion.py as first args
# to avoid circular imports; duck-typing is used throughout.


# ――― Phase 3a: temporal decay ―――――――――――――――――――――――――――――――――――――――――――――――


def stale_concept_suppression_rate(
    graph: Any,
    current_seq: int,
    decay_engine: Any,
    min_stale_age: int = 50,
) -> float:
    """Fraction of truly stale concepts that have decayed below the deactivation
    threshold.

    "Truly stale" = ``current_seq - concept.last_seen_seq ≥ min_stale_age``.
    "Suppressed" = ``decay_engine.decayed_confidence(concept, current_seq)
    < decay_engine.deactivation_threshold``.

    High rate (→1) means the temporal decay correctly silences old concepts.
    Low rate means concepts persist even when they should have been forgotten."""
    truly_stale = [
        c for c in graph.concepts.values()
        if (current_seq - c.last_seen_seq) >= min_stale_age
    ]
    if not truly_stale:
        return 0.0
    suppressed = sum(
        1 for c in truly_stale
        if not decay_engine.is_active(c, current_seq)
    )
    return suppressed / len(truly_stale)


def refreshed_concept_recovery_rate(
    graph_before: Any,
    graph_after: Any,
    decay_engine: Any,
    current_seq_before: int,
    current_seq_after: int,
    concept_ids: Optional[List[str]] = None,
) -> Optional[float]:
    """Fraction of concepts that were stale at ``current_seq_before`` and
    recovered (active again) at ``current_seq_after``.

    ``concept_ids``: restrict to specific concept IDs. If None, all concepts
    present in both graphs are evaluated."""
    ids = set(concept_ids) if concept_ids else (
        set(graph_before.concepts) & set(graph_after.concepts)
    )
    if not ids:
        return None
    were_stale = [
        cid for cid in ids
        if cid in graph_before.concepts
        and not decay_engine.is_active(graph_before.concepts[cid], current_seq_before)
    ]
    if not were_stale:
        return None
    recovered = sum(
        1 for cid in were_stale
        if cid in graph_after.concepts
        and decay_engine.is_active(graph_after.concepts[cid], current_seq_after)
    )
    return recovered / len(were_stale)


def latency_to_deactivation(
    graph: Any,
    current_seq: int,
    decay_engine: Any,
) -> Dict[str, Any]:
    """For each currently active concept, how many more seq-steps until it
    deactivates (drops below threshold), assuming no new support arrives.

    Returns a summary dict with ``min``, ``median``, ``mean``, ``max``
    over active concepts, plus the per-concept mapping ``by_concept``."""
    steps_by_id: Dict[str, Optional[int]] = {}
    for cid, concept in graph.concepts.items():
        if decay_engine.is_active(concept, current_seq):
            steps_by_id[cid] = decay_engine.steps_to_deactivation(concept, current_seq)
    finite = [s for s in steps_by_id.values() if s is not None]
    if not finite:
        return {"n_active": len(steps_by_id), "min": None, "median": None, "mean": None, "max": None, "by_concept": steps_by_id}
    finite_sorted = sorted(finite)
    n = len(finite_sorted)
    med = (
        finite_sorted[n // 2]
        if n % 2 == 1
        else (finite_sorted[n // 2 - 1] + finite_sorted[n // 2]) / 2
    )
    return {
        "n_active": len(steps_by_id),
        "min": finite_sorted[0],
        "median": med,
        "mean": sum(finite_sorted) / n,
        "max": finite_sorted[-1],
        "by_concept": steps_by_id,
    }


# ――― Phase 3b: conflict fusion ――――――――――――――――――――――――――――――――――――――――――――――


def conflict_resolution_accuracy(
    graph: Any,
    conflict_rules: Any,
    ground_truth_dominant_tag: Dict[str, str],
    conflict_threshold: float = 0.20,
) -> Optional[float]:
    """For concepts with significant conflict (penalty ≥ conflict_threshold),
    check if the dominant tag (post-penalty suppression) matches ground truth.

    ``ground_truth_dominant_tag``: ``{concept_id: correct_dominant_tag}``.

    Returns None if no conflicted concepts match ground truth keys."""
    conflicted = [
        c for c in graph.concepts.values()
        if conflict_rules.has_significant_conflict(c, threshold=conflict_threshold)
        and c.concept_id in ground_truth_dominant_tag
    ]
    if not conflicted:
        return None
    correct = sum(
        1 for c in conflicted
        if c.dominant_tag == ground_truth_dominant_tag[c.concept_id]
    )
    return correct / len(conflicted)


def concept_purity_under_contradiction(
    graph: Any,
    conflict_rules: Any,
    conflict_threshold: float = 0.20,
) -> Optional[float]:
    """Mean ``conflict_rules.concept_purity()`` for concepts that have
    significant conflict (penalty ≥ conflict_threshold).

    Low mean purity (→0) means conflicted concepts are heavily contaminated.
    High mean purity (→1) means conflicts are mild and the dominant signal
    still wins cleanly."""
    conflicted = [
        c for c in graph.concepts.values()
        if conflict_rules.has_significant_conflict(c, threshold=conflict_threshold)
    ]
    if not conflicted:
        return None
    purities = [conflict_rules.concept_purity(c) for c in conflicted]
    return sum(purities) / len(purities)


def false_persistence_rate(
    graph: Any,
    current_seq: int,
    decay_engine: Any,
    conflict_rules: Any,
    target_tag: str,
    conflict_threshold: float = 0.20,
) -> float:
    """Fraction of ``target_tag`` concepts that are simultaneously:

    1. Stale (decayed_confidence < deactivation_threshold), AND
    2. Still returned by ``graph.query_concepts`` (i.e. not yet suppressed).

    High rate = concepts that should be forgotten are still influencing
    retrieval — a combined failure of temporal decay AND conflict fusion."""
    all_results = graph.query_concepts(
        target_tag=target_tag,
        only_dominant_tag=True,
        top_k=len(graph.concepts) + 1,
    )
    if not all_results:
        return 0.0
    falsely_persistent = sum(
        1 for r in all_results
        if r.concept_id in graph.concepts
        and not decay_engine.is_active(graph.concepts[r.concept_id], current_seq)
    )
    return falsely_persistent / len(all_results)


# ――― Phase 3c: incremental updates ―――――――――――――――――――――――――――――――――――――――――


def update_stability(
    snap_before: Dict[Tuple[Any, ...], str],
    snap_after: Dict[Tuple[Any, ...], str],
) -> float:
    """Jaccard similarity of (place_key, concept_id) assignment pairs before
    and after an incremental update.

    ``snap_before / after``: ``{place_key: concept_id}`` dicts (from
    ``IncrementalAlignmentEngine.snapshot_membership()``).

    1.0 = identical assignment, 0.0 = completely reshuffled.  High stability
    means the incremental update is conservative: existing assignments are
    preserved and only genuinely changed places are moved."""
    if not snap_before and not snap_after:
        return 1.0
    before_set = set(snap_before.items())
    after_set = set(snap_after.items())
    intersection = len(before_set & after_set)
    union = len(before_set | after_set)
    return intersection / union if union > 0 else 1.0


def concept_churn_rate(
    snap_before: Dict[Tuple[Any, ...], str],
    snap_after: Dict[Tuple[Any, ...], str],
) -> float:
    """Fraction of places that changed concept membership between two snapshots.

    Low churn (→0) = graph is stable across incremental updates.
    High churn (→1) = most places moved to a different concept."""
    shared_keys = set(snap_before) & set(snap_after)
    if not shared_keys:
        return 0.0
    changed = sum(
        1 for k in shared_keys
        if snap_before[k] != snap_after[k]
    )
    return changed / len(shared_keys)


def reuse_after_incremental_update(
    top1_before: Optional[Tuple[Any, str]],
    top1_after: Optional[Tuple[Any, str]],
) -> bool:
    """True if the top-1 concept query result (concept_id) is the same before
    and after an incremental update.

    ``top1_before / after``: ``(place_key, concept_id)`` of the top result,
    or None if no result was returned.

    Episode-level proxy for "an agent that committed to a concept before the
    update can still use the same concept after the update"."""
    if top1_before is None and top1_after is None:
        return True
    if top1_before is None or top1_after is None:
        return False
    return top1_before[1] == top1_after[1]  # compare concept_id


def all_phase3_metrics(
    graph: Any,
    current_seq: int,
    decay_engine: Any,
    conflict_rules: Any,
    *,
    snap_before: Optional[Dict] = None,
    snap_after: Optional[Dict] = None,
    target_tag: Optional[str] = None,
    ground_truth_dominant_tag: Optional[Dict[str, str]] = None,
    min_stale_age: int = 50,
    conflict_threshold: float = 0.20,
) -> Dict[str, Any]:
    """Aggregated Phase 3a + 3b + 3c report."""
    report: Dict[str, Any] = {}

    # 3a
    report["stale_concept_suppression_rate"] = stale_concept_suppression_rate(
        graph, current_seq, decay_engine, min_stale_age=min_stale_age
    )
    report["latency_to_deactivation"] = latency_to_deactivation(
        graph, current_seq, decay_engine
    )

    # 3b
    if target_tag:
        report["false_persistence_rate"] = false_persistence_rate(
            graph, current_seq, decay_engine, conflict_rules,
            target_tag, conflict_threshold=conflict_threshold,
        )
    report["concept_purity_under_contradiction"] = concept_purity_under_contradiction(
        graph, conflict_rules, conflict_threshold=conflict_threshold
    )
    if ground_truth_dominant_tag:
        report["conflict_resolution_accuracy"] = conflict_resolution_accuracy(
            graph, conflict_rules, ground_truth_dominant_tag, conflict_threshold=conflict_threshold
        )

    # 3c
    if snap_before is not None and snap_after is not None:
        report["update_stability"] = update_stability(snap_before, snap_after)
        report["concept_churn_rate"] = concept_churn_rate(snap_before, snap_after)

    return report


# ----------------------------------------------------------------- aggregation


def all_metrics(
    collective: CollectiveMemory,
    *,
    episode_outcomes: Optional[Sequence[Dict[str, Any]]] = None,
    ground_truth_places_by_tag: Optional[Dict[str, Set[Tuple[Any, ...]]]] = None,
    ground_truth_tag_per_place: Optional[Dict[Tuple[Any, ...], str]] = None,
    target_concept_tag: Optional[str] = None,
    requesting_agent_id: Optional[str] = None,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "stats": collective.stats(),
        "collective_memory_reuse_rate": collective_memory_reuse_rate(
            collective, requesting_agent_id=requesting_agent_id
        ),
        "other_agent_knowledge_use_rate": other_agent_knowledge_use_rate(
            collective, requesting_agent_id=requesting_agent_id
        ),
        "observation_to_use_latency": observation_to_use_latency(
            collective, requesting_agent_id=requesting_agent_id
        ),
        "agent_contribution_entropy": agent_contribution_entropy(collective),
    }
    if ground_truth_places_by_tag and target_concept_tag:
        report["concept_transfer_precision"] = concept_transfer_precision(
            collective, ground_truth_places_by_tag, target_concept_tag
        )
    if ground_truth_tag_per_place:
        report["belief_conflict_resolution_accuracy"] = belief_conflict_resolution_accuracy(
            collective, ground_truth_tag_per_place
        )
    if episode_outcomes is not None:
        report["field_activation_to_success_correlation"] = (
            field_activation_to_success_correlation(
                collective, episode_outcomes, target_tag=target_concept_tag
            )
        )
        report["collective_semantic_path_completion"] = (
            collective_semantic_path_completion(collective, episode_outcomes)
        )
    if target_concept_tag is not None:
        report["time_to_collective_convergence"] = time_to_collective_convergence(
            collective, target_concept_tag
        )
    return report


# ------------------------------------------------------------------- statistics


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    n = len(xs)
    if n < 2 or n != len(ys):
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sx2 = sum((x - mx) ** 2 for x in xs)
    sy2 = sum((y - my) ** 2 for y in ys)
    if sx2 <= 0.0 or sy2 <= 0.0:
        return 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / math.sqrt(sx2 * sy2)
