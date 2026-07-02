"""LLM-driven agent with symbolic per-agent memory.

This is the minimal Phase A agent. It uses:
  • LLMTagExtractor    — observation text → symbolic tags (Dirichlet-like)
  • LLMQueryFormer     — task → required/preferred/penalty tags
  • LLMDecider         — observation + candidates → chosen action

The memory is a thin Dict-based store that supports the same
observe()/query() shape as peer_memory.PeerAgent. We keep this
intentionally simple in Phase A; Phase B will substitute
peer_memory.PeerRuntime for the multi-agent broadcast loop.
"""

from __future__ import annotations

import dataclasses as dc
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from experiments.llm_collective.llm_backend import OllamaBackend
from experiments.llm_collective.llm_tag_extractor import LLMTagExtractor
from experiments.llm_collective.llm_query_former import LLMQueryFormer, Query
from experiments.llm_collective.llm_decider import LLMDecider


GridXY = Tuple[int, int]


@dc.dataclass
class _PlaceRecord:
    xy: GridXY
    tag_conf: Dict[str, float] = dc.field(default_factory=dict)
    visits: int = 0
    last_tick: int = -1


class SimpleSymbolicMemory:
    """Per-agent symbolic memory.

    observe(): tag updates via max-confidence merge (Phase A; Phase B
               will reuse peer_memory.PeerAgent's Dirichlet posteriors).
    query():   scores places against (req, pref, pen) via weighted
               log-score with required-must-match policy. Returns
               candidates in score order with their tag profiles.
    """

    def __init__(
        self,
        required_match_threshold: float = 0.3,
        pref_weight: float = 0.5,
        pen_weight: float = 1.0,
    ) -> None:
        self.places: Dict[GridXY, _PlaceRecord] = {}
        self.required_match_threshold = float(required_match_threshold)
        self.pref_weight = float(pref_weight)
        self.pen_weight = float(pen_weight)

    def observe(self, xy: GridXY, tags: Dict[str, float], tick: int) -> None:
        rec = self.places.get(xy)
        if rec is None:
            rec = _PlaceRecord(xy=xy)
            self.places[xy] = rec
        for t, c in tags.items():
            rec.tag_conf[t] = max(rec.tag_conf.get(t, 0.0), float(c))
        rec.visits += 1
        rec.last_tick = tick

    def query(self, q: Query) -> List[Dict]:
        out = []
        for xy, rec in self.places.items():
            # Required-tag gate
            if q.required:
                ok = all(
                    rec.tag_conf.get(t, 0.0) >= self.required_match_threshold
                    for t in q.required
                )
                if not ok:
                    continue
            score = 0.0
            for t in q.preferred:
                score += self.pref_weight * rec.tag_conf.get(t, 0.0)
            for t in q.penalty:
                score -= self.pen_weight * rec.tag_conf.get(t, 0.0)
            # Always include required-bonus
            for t in q.required:
                score += rec.tag_conf.get(t, 0.0)
            out.append({
                "id": f"p{xy[0]}_{xy[1]}",
                "xy": xy,
                "score": score,
                "tags": dict(rec.tag_conf),
                "visits": rec.visits,
            })
        out.sort(key=lambda c: -c["score"])
        return out


class PeerLLMAgent:
    """LLM agent with symbolic memory. Drop-in for one-agent run loop."""

    def __init__(
        self,
        aid: str = "a0",
        backend: Optional[OllamaBackend] = None,
        memory: Optional[SimpleSymbolicMemory] = None,
    ) -> None:
        self.aid = aid
        self.backend = backend or OllamaBackend()
        self.memory = memory or SimpleSymbolicMemory()
        self.extractor = LLMTagExtractor(backend=self.backend)
        self.query_former = LLMQueryFormer(backend=self.backend)
        self.decider = LLMDecider(backend=self.backend)
        self._cached_query: Optional[Query] = None
        self._locked_target: Optional[GridXY] = None

    def observe(self, xy: GridXY, obs_text: str, tick: int,
                seed: int = 0) -> Dict[str, float]:
        tags = self.extractor.extract(obs_text, seed=seed)
        self.memory.observe(xy, tags, tick)
        return tags

    def form_query(self, task_text: str, seed: int = 0) -> Query:
        if self._cached_query is None:
            self._cached_query = self.query_former.form(task_text, seed=seed)
        return self._cached_query

    def reset_query(self) -> None:
        self._cached_query = None
        self._locked_target = None

    def decide(
        self,
        observation_text: str,
        task_text: str,
        allowed_actions: List[str],
        seed: int = 0,
    ) -> Tuple[str, Query, List[Dict], Optional[Dict]]:
        """Returns (chosen_action, query, candidates, locked_candidate_or_None).

        This is the single Q/R/M/C event-emitting step:
          Q*  = query formed (always true for this LLM agent)
          R*  = candidates list non-empty AND chosen candidate satisfies task
          M*  = decider locked onto a concrete candidate id (target)
          C*  = (emitted by the env, when agent succeeds)
        """
        q = self.form_query(task_text, seed=seed)
        candidates = self.memory.query(q)
        d = self.decider.decide(
            observation=observation_text,
            task=task_text,
            candidates=candidates,
            allowed_actions=allowed_actions,
            seed=seed,
        )
        locked = None
        if d.target_id:
            for c in candidates:
                if c["id"] == d.target_id:
                    locked = c
                    self._locked_target = c["xy"]
                    break
        return d.action, q, candidates, locked
