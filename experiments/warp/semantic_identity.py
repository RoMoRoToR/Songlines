"""Semantic place identity — matching places by meaning, not coordinates.

Everywhere else in this series 'the same place' ultimately means 'close
(x, y) in a frame the environment hands to every agent'; semantics only
disambiguates within a spatial radius.  This module inverts that:

  1. A *place fingerprint* is the local semantic constellation around a
     visited cell — which tags (hazard, wall, water, grid border) sit at
     which RELATIVE offsets.  Fingerprints are translation-invariant and
     carry no global coordinates.
  2. Two agents match places by fingerprint similarity with an
     unambiguity margin: a foreign fingerprint is accepted only if it
     has exactly ONE good own-side candidate.  Repetitive worlds
     (perceptual aliasing) therefore yield NO matches — the system
     fails closed instead of confidently wrong.
  3. From >= MIN_CONSENSUS matched pairs that agree on a single
     translation, the receiver recovers the frame offset and transports
     ALL of the sender's evidence — including places the receiver has
     never seen (the warp targets) — into its own frame.

This is the songline mechanism made literal: shared landmarks anchor
the unknown part of the song.  Scope: translation-only (grid worlds
have an absolute orientation via the action set); rotation-invariant
fingerprints are future work.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

GridXY = Tuple[int, int]

# tags that carry identity information (safe_neutral floor does not)
SALIENT_TAGS = {"wall", "hazard_edge", "water_source", "goal", "void"}
# The grid is discrete and noise-free, so place identity is EXACT
# constellation equality; sub-unit thresholds admit spurious pairs.
SIM_THRESHOLD = 0.999
MIN_SIG_KEYS = 2         # near-empty fingerprints are alias factories
MIN_CONSENSUS = 3        # matched pairs required to trust an offset
CONSENSUS_SHARE = 0.8    # share of pairs that must agree on the offset


def fingerprint(agent_xy: GridXY, cells: List[Dict[str, Any]],
                radius: int = 2) -> Dict[str, float]:
    """Local semantic constellation, keyed by tag@relative-offset.

    Cells missing from the observation window (outside the grid) are
    encoded as 'void' — borders are landmarks too."""
    ax, ay = agent_xy
    seen: Dict[GridXY, str] = {}
    for c in cells:
        x, y = int(c["xy"][0]), int(c["xy"][1])
        seen[(x - ax, y - ay)] = c["tag"]
    sig: Dict[str, float] = {}
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if abs(dx) + abs(dy) > radius:
                continue
            tag = seen.get((dx, dy), "void")
            if tag in SALIENT_TAGS:
                sig[f"{tag}@{dx},{dy}"] = 1.0
    return sig


def cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)
    dot = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in keys)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


@dataclass
class AlignmentResult:
    offset: Optional[GridXY]          # sender->receiver translation
    n_matches: int
    n_ambiguous: int
    consensus_share: float
    matched_pairs: List[Tuple[GridXY, GridXY]] = field(default_factory=list)


def align_frames(own_fps: Dict[GridXY, Dict[str, float]],
                 foreign_fps: Dict[GridXY, Dict[str, float]]
                 ) -> AlignmentResult:
    """Estimate the translation between two agents' private frames from
    semantically matched fingerprints.

    A pair is accepted only if it is MUTUALLY unique: the foreign
    fingerprint has exactly one own-side candidate above threshold AND
    that own fingerprint has exactly one foreign-side candidate (the
    same one).  One-directional uniqueness is not enough — the sender
    typically knows far more places than the receiver, and a distant
    foreign cell with a similar constellation would otherwise 'uniquely'
    hit a wrong local cell.  Fingerprints with fewer than MIN_SIG_KEYS
    salient keys are skipped (featureless cells alias everywhere).  The
    offset is the modal delta over accepted pairs; it is trusted only
    with MIN_CONSENSUS pairs and CONSENSUS_SHARE agreement."""
    # A landmark must carry at least one CONTENT tag: void-only
    # fingerprints describe the shape of the world (borders), which is
    # rotationally near-symmetric for rectangles and feeds wrong-frame
    # hypotheses; content constellations (hazards, water, walls) are
    # the actual places.
    def informative(sig):
        return (len(sig) >= MIN_SIG_KEYS
                and any(not k.startswith("void") for k in sig))

    own_rich = {k: v for k, v in own_fps.items() if informative(v)}
    foreign_rich = {k: v for k, v in foreign_fps.items()
                    if informative(v)}
    pairs: List[Tuple[GridXY, GridXY]] = []
    n_ambiguous = 0
    for fxy, fsig in foreign_rich.items():
        candidates = [oxy for oxy, osig in own_rich.items()
                      if cosine(fsig, osig) >= SIM_THRESHOLD]
        if len(candidates) > 1:
            n_ambiguous += 1
            continue
        if len(candidates) != 1:
            continue
        oxy = candidates[0]
        back = [f2 for f2, fsig2 in foreign_rich.items()
                if cosine(own_rich[oxy], fsig2) >= SIM_THRESHOLD]
        if back == [fxy]:
            pairs.append((fxy, oxy))
        else:
            n_ambiguous += 1
    if not pairs:
        return AlignmentResult(None, 0, n_ambiguous, 0.0)

    deltas = Counter((oxy[0] - fxy[0], oxy[1] - fxy[1])
                     for fxy, oxy in pairs)
    (delta, count), = [deltas.most_common(1)[0]] if deltas else [((0, 0), 0)]
    share = count / len(pairs)
    if count >= MIN_CONSENSUS and share >= CONSENSUS_SHARE:
        return AlignmentResult(delta, len(pairs), n_ambiguous, share,
                               matched_pairs=pairs)
    return AlignmentResult(None, len(pairs), n_ambiguous, share,
                           matched_pairs=pairs)


# ── SE(2): rotation + translation ─────────────────────────────────
#
# Grid rotations r ∈ {0,1,2,3} (multiples of 90° CCW).  A foreign frame
# rotated by r relative to the receiver has BOTH its coordinates and
# its fingerprint offsets rotated, so one rotation hypothesis rotates
# the whole foreign payload and reuses the validated translation
# aligner.  The winner must DOMINATE: rectangular-world borders are
# 180°-symmetric (corner matches give a consistent delta for the wrong
# 180° hypothesis), so a unique-winner rule alone is not enough — the
# best rotation must carry at least DOMINANCE× the runner-up's
# consensus, or the aligner fails closed.

DOMINANCE = 2.0


def rotate_point(xy: GridXY, r: int) -> GridXY:
    x, y = xy
    r %= 4
    if r == 0:
        return (x, y)
    if r == 1:
        return (-y, x)
    if r == 2:
        return (-x, -y)
    return (y, -x)


def rotate_sig(sig: Dict[str, float], r: int) -> Dict[str, float]:
    if r % 4 == 0:
        return sig
    out: Dict[str, float] = {}
    for key, v in sig.items():
        tag, off = key.rsplit("@", 1)
        dx, dy = (int(s) for s in off.split(","))
        rx, ry = rotate_point((dx, dy), r)
        out[f"{tag}@{rx},{ry}"] = v
    return out


@dataclass
class SE2Result:
    rotation: Optional[int]
    delta: Optional[GridXY]
    per_rotation: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    n_winners: int = 0


def align_frames_se2(own_fps: Dict[GridXY, Dict[str, float]],
                     foreign_fps: Dict[GridXY, Dict[str, float]]
                     ) -> SE2Result:
    """Recover (rotation, translation) between two private frames.

    Each rotation hypothesis rotates the foreign payload and runs the
    translation aligner; a hypothesis 'wins' if it reaches delta
    consensus.  Accept only a DOMINANT winner (see above); several
    comparably-supported winners — e.g. a 180°-symmetric world — mean
    the frames are unrecoverable from these landmarks: fail closed."""
    per_rot: Dict[int, Dict[str, Any]] = {}
    winners: List[Tuple[int, GridXY, int]] = []  # (r, delta, support)
    for r in range(4):
        rot_fps = {rotate_point(xy, r): rotate_sig(sig, r)
                   for xy, sig in foreign_fps.items()}
        res = align_frames(own_fps, rot_fps)
        support = len([1 for fxy, oxy in res.matched_pairs
                       if (oxy[0] - fxy[0], oxy[1] - fxy[1]) == res.offset]) \
            if res.offset is not None else 0
        per_rot[r] = {"offset": res.offset, "n_matches": res.n_matches,
                      "support": support}
        if res.offset is not None:
            winners.append((r, res.offset, support))

    if len(winners) == 1:
        r, delta, _ = winners[0]
        return SE2Result(r, delta, per_rot, 1)
    if len(winners) > 1:
        winners.sort(key=lambda w: -w[2])
        if winners[0][2] >= DOMINANCE * winners[1][2]:
            r, delta, _ = winners[0]
            return SE2Result(r, delta, per_rot, len(winners))
    return SE2Result(None, None, per_rot, len(winners))


class SemanticIdentityMemory:
    """Peer place memory over PRIVATE coordinate frames.

    Each agent records fingerprints of visited cells and tagged
    evidence (e.g. water) in its OWN frame.  Broadcasts carry the
    sender's private coordinates — meaningless to the receiver until
    it aligns frames semantically.  ``mode``:

      'semantic'   — receiver aligns via ``align_frames`` and transports
                     foreign evidence through the recovered offset;
                     without consensus it refuses (fails closed);
      'coordinate' — receiver takes foreign coordinates at face value
                     (the shared-frame assumption of the main series;
                     under frame misalignment it fails open).
    """

    def __init__(self, agent_ids: List[str], mode: str = "semantic",
                 broadcast_every_k: int = 4):
        assert mode in ("semantic", "coordinate")
        self.mode = mode
        self.k = broadcast_every_k
        self.state: Dict[str, Dict[str, Any]] = {
            aid: {"fingerprints": {}, "evidence": {},   # own, private frame
                  "foreign": {}}                        # sender -> payload
            for aid in agent_ids}

    # ── recording (own frame) ─────────────────────────────────────

    def observe(self, aid: str, private_xy: GridXY,
                cells: List[Dict[str, Any]], tick: int) -> None:
        st = self.state[aid]
        st["fingerprints"][tuple(private_xy)] = fingerprint(
            private_xy, cells)
        ax, ay = private_xy
        for c in cells:
            if c["tag"] == "water_source":
                # cells arrive in the agent's own frame already
                st["evidence"][ (int(c["xy"][0]), int(c["xy"][1])) ] = {
                    "tag": c["tag"], "tick": tick}

    def tick(self, tick_idx: int) -> None:
        if self.k <= 0 or tick_idx % self.k != 0:
            return
        for sender, st in self.state.items():
            payload = {"fingerprints": dict(st["fingerprints"]),
                       "evidence": dict(st["evidence"])}
            for receiver, rst in self.state.items():
                if receiver != sender:
                    rst["foreign"][sender] = payload

    # ── query (receiver frame) ────────────────────────────────────

    def query(self, aid: str) -> Tuple[List[GridXY], Dict[str, Any]]:
        """Water targets in the receiver's own frame + diagnostics."""
        st = self.state[aid]
        targets = {xy: {aid} for xy in st["evidence"]}
        diag: Dict[str, Any] = {"alignments": {}}
        for sender, payload in st["foreign"].items():
            if self.mode == "coordinate":
                for xy in payload["evidence"]:
                    targets.setdefault(tuple(xy), set()).add(sender)
                continue
            res = align_frames(st["fingerprints"], payload["fingerprints"])
            diag["alignments"][sender] = {
                "offset": res.offset, "n_matches": res.n_matches,
                "n_ambiguous": res.n_ambiguous,
                "consensus_share": round(res.consensus_share, 2),
            }
            if res.offset is None:
                continue  # fail closed: no transport without consensus
            dx, dy = res.offset
            for (fx, fy) in payload["evidence"]:
                targets.setdefault((fx + dx, fy + dy), set()).add(sender)
        diag["per_target_sources"] = {str(k): sorted(v)
                                      for k, v in targets.items()}
        return list(targets.keys()), diag

    def phi(self, aid: str, target_xy: GridXY) -> float:
        """Foreign evidence share for a queried target (uniform source
        masses — evidence here is binary sightings)."""
        _, diag = self.query(aid)
        srcs = diag["per_target_sources"].get(str(tuple(target_xy)))
        if not srcs:
            return 0.0
        foreign = sum(1 for s in srcs if s != aid)
        return foreign / len(srcs)
