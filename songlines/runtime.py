"""Songline Memory Runtime v1 --- the per-agent orchestrator.

One method, one record type, one cycle:

    observation -> candidate memory -> utility/analogy decision ->
    immutable store -> certificate -> peer quarantine -> local
    admission -> frame-free landmark matching -> route retrieval ->
    reservation -> execution / rupture fallback.

The agent composes the substrate-agnostic pieces --- ``record``
(state + costs), ``analogy`` (the two-axis formation matrix), and
``alignment`` (frame-free consumption).  Grid/continuous substrates
supply only observations and a utility function; the runtime never
touches world coordinates.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from songlines.record import (
    BEAT_BITS, KEY_BITS, LEN_BITS, ROLE_NAMES, SHARE_THR, D_THR,
    TIME_BITS, U_THR, Config, Record, record_bits)
from songlines.analogy import Schema, nearest
from songlines.alignment import song_target

GridXY = Tuple[float, float]


class SonglineAgent:
    """One agent's full memory runtime."""

    simfn = None   # pluggable constellation similarity (continuous)

    def __init__(self, aid: int, role_name: str, cfg: Config):
        self.aid = aid
        self.role_name = role_name
        self.cfg = cfg
        self.records: List[Record] = []
        self.episodic: List[Dict[str, Any]] = []   # immutable layer
        self.quarantine: Dict[int, List[Record]] = {}
        self.known_version: Dict[int, int] = {}
        self.visited: Set[int] = set()
        self.received: Set[Tuple[int, int]] = set()
        self.now = 0
        self.match_ops = 0                          # latency proxy

    # ── world clock ────────────────────────────────────────────────
    def note_version(self, fam: int, ver: int) -> None:
        if ver > self.known_version.get(fam, -1):
            self.known_version[fam] = ver

    def admissible(self, rec: Record) -> bool:
        if not self.cfg.world_clock:
            return True
        return rec.version >= self.known_version.get(rec.family, -1)

    # ── formation ──────────────────────────────────────────────────
    def form(self, song, intent: str, fam: int, ver: int, t: int,
             role_u: Dict[str, float]) -> str:
        self.now = t
        if self.cfg.immutable:
            self.episodic.append({"song": song, "fam": fam, "t": t,
                                  "ver": ver, "intent": intent})
        u = role_u[self.role_name]
        rel = [r for r in self.records if r.intent == intent]
        idx_rel, ana = nearest(song, [Schema(r.song, cert=None)
                                      for r in rel])
        idx = (self.records.index(rel[idx_rel])
               if idx_rel is not None and rel else None)
        simple = ana is not None and ana["share"] >= SHARE_THR
        conflict = simple and ana["D"] >= D_THR
        rec = Record(song, intent, fam, dict(role_u), self.aid,
                     (self.aid, t), t, ver)
        if not self.cfg.utility_gate:
            self.records.append(rec)
            return "STORE_ALL"
        if u < U_THR:
            if simple and idx is not None:
                self.records[idx].support += 1
                return "REPEAT"
            return "DROP"
        if conflict and self.cfg.exceptions and idx is not None:
            rec.kind, rec.parent = "exception", idx
            self.records.append(rec)
            return "EXCEPTION"
        if simple and idx is not None:
            old = self.records[idx]
            old.song, old.t, old.version = song, t, ver
            old.support += 1
            old.role_u = {r: max(old.role_u.get(r, 0.0),
                                 role_u.get(r, 0.0)) for r in ROLE_NAMES}
            return "MERGE"
        self.records.append(rec)
        return "NEW_SCHEMA"

    # ── exchange ───────────────────────────────────────────────────
    def outbox(self, since: int) -> List[Record]:
        return [r for r in self.records
                if (not self.cfg.provenance or r.origin == self.aid)
                and r.t > since]

    def receive(self, rec: Record, sender: Optional[int] = None
                ) -> None:
        if rec.uid in self.received:
            return
        # origin-bound provenance: a record whose claimed origin is
        # not the channel sender is laundered testimony --- rejected
        if self.cfg.provenance and sender is not None \
                and rec.origin != sender:
            return
        self.received.add(rec.uid)
        self.note_version(rec.family, rec.version)
        adm = self.cfg.admission
        if adm == "none" or (adm == "visit"
                             and rec.family in self.visited):
            self._admit(rec, rec.role_u)
            return
        self.quarantine.setdefault(rec.family, []).append(rec)

    def _admit(self, rec: Record, role_u: Dict[str, float]) -> None:
        if self.cfg.world_clock and rec.version < \
                self.known_version.get(rec.family, -1):
            return
        r2 = Record(rec.song, rec.intent, rec.family, dict(role_u),
                    rec.origin, rec.uid, rec.t, rec.version)
        rel = [r for r in self.records if r.intent == rec.intent]
        idx_rel, ana = nearest(rec.song, [Schema(r.song, cert=None)
                                          for r in rel])
        idx = (self.records.index(rel[idx_rel])
               if idx_rel is not None and rel else None)
        simple = ana is not None and ana["share"] >= SHARE_THR
        conflict = simple and ana["D"] >= D_THR
        if conflict and self.cfg.exceptions and idx is not None:
            r2.kind, r2.parent = "exception", idx
            self.records.append(r2)
        elif simple and idx is not None and not self.cfg.exceptions:
            self.records[idx] = r2                 # overwrite ablation
        else:
            self.records.append(r2)

    def on_visit(self, env, fam: int, ver: int, t: int,
                 utility_fn: Callable) -> None:
        self.visited.add(fam)
        self.note_version(fam, ver)
        for q in self.quarantine.pop(fam, []):
            if self.cfg.world_clock and q.version < \
                    self.known_version.get(fam, -1):
                continue
            if self.cfg.admission == "util":
                u = utility_fn(env, self, q.song, q.intent)
                if u < U_THR:
                    continue
                role_u = dict(q.role_u)
                role_u[self.role_name] = u          # measured, not told
            else:
                role_u = q.role_u
            self._admit(q, role_u)

    # ── consumption ────────────────────────────────────────────────
    def targets(self, band_fps, intent: str, observe_fn=None,
                start=None) -> List[GridXY]:
        flipped = {r.parent for r in self.records
                   if r.kind == "exception"} if self.cfg.provenance \
            else set()
        def score(i, r):
            s = r.support * math.exp(-0.02 * (self.now - r.t))
            if i in flipped:
                s *= 0.25
            return s
        usable = [(i, r) for i, r in enumerate(self.records)
                  if r.intent == intent and self.admissible(r)
                  and r.role_u.get(self.role_name, 0.0) >= 0.0]
        usable.sort(key=lambda p: -score(*p))
        out, seen = [], set()
        scored = []       # (n_anchors, target, song) for commit_top1
        for _, r in usable:
            self.match_ops += len(r.song)
            res = song_target(r.song, band_fps, self.cfg.sim_threshold,
                              self.cfg.anchor_consensus,
                              self.cfg.closure_tol, self.simfn,
                              self.cfg.unimodal_tol, return_support=True)
            if res is None:
                continue
            tt, n_anchor = res
            if self.cfg.commit_top1:
                scored.append((n_anchor, tt, r.song))
            elif tt not in seen:
                seen.add(tt)
                out.append(tt)
        if self.cfg.commit_top1:
            if not scored:
                return []
            scored.sort(key=lambda p: -p[0])
            for best_n, best_t, best_song in scored:
                if self.cfg.commit_dominance > 0:
                    rival = max((n for n, t, _ in scored
                                 if abs(t[0] - best_t[0])
                                 + abs(t[1] - best_t[1]) > 1.0),
                                default=0)
                    if best_n < self.cfg.commit_dominance * max(1, rival):
                        return []
                if self.cfg.prefix_verify and observe_fn is not None:
                    if not self._prefix_ok(best_song, observe_fn,
                                           start):
                        continue       # phantom: try next / refuse
                return [best_t]
            return []
        return out

    def _prefix_ok(self, song, observe_fn, start) -> bool:
        """Walk the first prefix_verify couplets from `start`; the
        observed constellation must match each couplet (a wrong-family
        song diverges from the traveler's actual terrain)."""
        from experiments.warp.semantic_identity import cosine as _cos
        fsim = self.simfn or _cos
        pos = start
        checked = 0
        for c in song:
            if checked >= self.cfg.prefix_verify:
                break
            if c.get("beat") is None:
                continue
            pos = (pos[0] + c["beat"][0], pos[1] + c["beat"][1])
            if not c.get("sig"):
                continue
            obs = observe_fn(pos)
            if fsim(c["sig"], obs) < self.cfg.prefix_tol:
                return False
            checked += 1
        return True

    # ── accounting ─────────────────────────────────────────────────
    def memory_bits(self) -> int:
        total = sum(record_bits(r, self.cfg) for r in self.records)
        for q in self.quarantine.values():
            total += sum(record_bits(r, self.cfg) for r in q)
        if self.cfg.immutable:
            for e in self.episodic:
                total += sum(LEN_BITS + KEY_BITS * len(c["sig"])
                             + BEAT_BITS for c in e["song"]) + TIME_BITS
        return total
