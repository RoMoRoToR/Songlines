"""Semantic place identity wired into the full peer/CSM memory stack.

``SemanticFramePeerMemory`` is a drop-in multi-agent memory where NO
shared coordinate frame exists: every agent records fingerprints and
evidence in its own private frame (the true grid shifted by a secret
per-agent offset), broadcasts them on cadence K, and each receiver
recovers, PER SENDER, the frame translation from mutually-unique
matches of co-visited landmarks (semantic_identity.align_frames).
Alignment develops online: early in an episode agents have visited too
little to align, and foreign evidence is unusable until enough shared
landmarks accumulate — the memory fails closed until then.

Modes:
  'semantic'    — landmark-consensus frame recovery (the contribution);
  'coordinate'  — foreign private coordinates taken at face value (the
                  shared-frame assumption; poison under misalignment);

CSM composition: with ``csm_gate=True`` foreign evidence additionally
passes the trust x staleness inclusion rule of the main series
(trust * exp(-alpha*age) * conf >= tau), so the distance law of W2 is
predicted to hold UNCHANGED on top of semantic identity — the gate and
the frame recovery are orthogonal layers.

Privacy contract as everywhere in the series: no agent reads another's
memory; only broadcast payloads move, and they carry no global frame.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from experiments.warp.semantic_identity import (
    align_frames, align_frames_se2, fingerprint, rotate_point,
)

GridXY = Tuple[int, int]
WATER_TAG = "water_source"


class SemanticFramePeerMemory:
    def __init__(
        self,
        agent_ids: List[str],
        frame_offsets: Dict[str, GridXY],
        *,
        frame_rotations: Optional[Dict[str, int]] = None,
        aligner: str = "translation",   # 'translation' (W7/W8) | 'se2' (W9)
        mode: str = "semantic",
        broadcast_every_k: int = 4,
        csm_gate: bool = False,
        trust: float = 1.0,
        alpha: float = 0.05,
        tau: float = 0.30,
        conf: float = 0.95,
    ) -> None:
        assert mode in ("semantic", "coordinate")
        assert aligner in ("translation", "se2")
        self.mode = mode
        self.aligner = aligner
        self.k = broadcast_every_k
        self.offsets = {aid: tuple(off) for aid, off in frame_offsets.items()}
        self.rotations = {aid: (frame_rotations or {}).get(aid, 0) % 4
                          for aid in agent_ids}
        self.csm_gate = csm_gate
        self.trust, self.alpha, self.tau, self.conf = trust, alpha, tau, conf
        self.state: Dict[str, Dict[str, Any]] = {
            aid: {"fingerprints": {}, "evidence": {},
                  "foreign": {}, "align_cache": {}}
            for aid in agent_ids}

    # ── frame bookkeeping: private = R_r(true) + offset ───────────

    def _to_private(self, aid: str, true_xy: GridXY) -> GridXY:
        r, off = self.rotations[aid], self.offsets[aid]
        rx, ry = rotate_point((int(true_xy[0]), int(true_xy[1])), r)
        return (rx + off[0], ry + off[1])

    def _to_true(self, aid: str, private_xy: GridXY) -> GridXY:
        r, off = self.rotations[aid], self.offsets[aid]
        return rotate_point((private_xy[0] - off[0],
                             private_xy[1] - off[1]), (4 - r) % 4)

    # ── recording (true coords in, private frame stored) ──────────

    def observe(self, aid: str, true_agent_xy: GridXY,
                true_cells: List[Dict[str, Any]], tick: int) -> None:
        pxy = self._to_private(aid, true_agent_xy)
        cells_p = [{"xy": self._to_private(aid, c["xy"]), "tag": c["tag"]}
                   for c in true_cells]
        st = self.state[aid]
        st["fingerprints"][pxy] = fingerprint(pxy, cells_p)
        for c in cells_p:
            if c["tag"] == WATER_TAG:
                st["evidence"][tuple(c["xy"])] = {"tick": tick}

    def tick(self, tick_idx: int) -> None:
        if self.k <= 0 or tick_idx % self.k != 0:
            return
        for sender, st in self.state.items():
            payload = {"fingerprints": dict(st["fingerprints"]),
                       "evidence": {xy: dict(m)
                                    for xy, m in st["evidence"].items()}}
            for receiver, rst in self.state.items():
                if receiver != sender:
                    rst["foreign"][sender] = payload
                    rst["align_cache"].pop(sender, None)  # re-align lazily

    # ── frame recovery (cached per broadcast wave) ────────────────

    def _frame(self, receiver: str,
               sender: str) -> Optional[Tuple[int, GridXY]]:
        """(rotation, delta) mapping sender-private -> receiver-private,
        or None while landmarks are insufficient/ambiguous."""
        rst = self.state[receiver]
        if sender in rst["align_cache"]:
            return rst["align_cache"][sender]
        payload = rst["foreign"].get(sender)
        frame = None
        if payload is not None:
            if self.aligner == "se2":
                res = align_frames_se2(rst["fingerprints"],
                                       payload["fingerprints"])
                if res.rotation is not None:
                    frame = (res.rotation, res.delta)
            else:
                off = align_frames(rst["fingerprints"],
                                   payload["fingerprints"]).offset
                if off is not None:
                    frame = (0, off)
        rst["align_cache"][sender] = frame
        return frame

    def alignment_status(self, aid: str
                         ) -> Dict[str, Optional[Tuple[int, GridXY]]]:
        return {s: self._frame(aid, s)
                for s in self.state[aid]["foreign"]}

    # ── query (true coords out, for the runner/planner) ───────────

    def _weight(self, src_is_self: bool, age: int) -> float:
        trust = 1.0 if src_is_self else self.trust
        if not self.csm_gate:
            return trust * self.conf
        return trust * math.exp(-self.alpha * max(0, age)) * self.conf

    def _gathered(self, aid: str, tick: int
                  ) -> Dict[GridXY, Dict[str, float]]:
        """true-frame target -> {source: mass}, gate applied."""
        st = self.state[aid]
        out: Dict[GridXY, Dict[str, float]] = {}

        def add(receiver_private: GridXY, source: str,
                ev_tick: int) -> None:
            w = self._weight(source == aid, tick - ev_tick)
            if w < self.tau if self.csm_gate else w <= 0:
                return
            txy = self._to_true(aid, receiver_private)
            out.setdefault(txy, {})
            out[txy][source] = out[txy].get(source, 0.0) + w

        for pxy, meta in st["evidence"].items():
            add(pxy, aid, meta["tick"])
        for sender, payload in st["foreign"].items():
            if self.mode == "coordinate":
                # foreign numbers read as if they were in MY frame
                for pxy, meta in payload["evidence"].items():
                    add(tuple(pxy), sender, meta["tick"])
                continue
            frame = self._frame(aid, sender)
            if frame is None:
                continue  # fail closed until landmarks align
            r, delta = frame
            for pxy, meta in payload["evidence"].items():
                rp = rotate_point(tuple(pxy), r)
                add((rp[0] + delta[0], rp[1] + delta[1]),
                    sender, meta["tick"])
        return out

    def query(self, aid: str, tick: int) -> List[GridXY]:
        return list(self._gathered(aid, tick).keys())

    # ── warp provenance ───────────────────────────────────────────

    def phi(self, aid: str, true_target: GridXY, tick: int) -> float:
        masses = self._gathered(aid, tick).get(tuple(true_target))
        if not masses:
            return 0.0
        own = masses.get(aid, 0.0)
        foreign = sum(v for s, v in masses.items() if s != aid)
        total = own + foreign
        return foreign / total if total > 0 else 0.0

    def self_observed(self, aid: str, true_target: GridXY) -> bool:
        return self._to_private(aid, true_target) in \
            self.state[aid]["evidence"]
