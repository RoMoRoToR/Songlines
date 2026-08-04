"""Songlines core — the memory record, the runtime config, and the
bit codec.

Substrate-agnostic: this module (and the whole ``songlines`` package)
owns the METHOD; grid/continuous worlds and experiment drivers live in
``experiments/``.  The reviewer's record type is
``m = (G, C, E, U, P, T, R, F, A)``; the fields map onto ``Record``
below, with frame correspondences (R) deliberately NOT stored ---
identity is re-derived at consumption (the frame-free contract).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

GridXY = Tuple[int, int]

# ── role names (substrate supplies the cost profiles) ──────────────
ROLE_NAMES: Tuple[str, ...] = ("fragile", "robust")

# ── full-codec wire/storage costs (bits) --- reviewer item 11 ──────
KEY_BITS = 6            # tag id (2) + offset-in-window (4)
LEN_BITS = 4            # signature length field
BEAT_BITS = 10          # dx, dy: 5 signed bits each
COORD_BITS = 8          # 4 + 4 for a 14x12 grid (snapshot cell)
CERT_BITS = 64          # utility profile, support, uncertainty
PROV_BITS = 48          # origin id + uid + immutable episode ref
TIME_BITS = 32          # observation time + world version
RESV_BITS = 24          # one reservation message

# ── frozen formation thresholds (see docs/SONGLINES_V1_FREEZE.md) ──
U_THR, SHARE_THR, D_THR = 5.0, 0.4, 3


@dataclass
class Config:
    """Every mechanism is a flag: arms and ablations of the benchmark
    are configurations of one runtime, never separate codebases."""
    landmarks: bool = True        # signatures in couplets (identity)
    beats: bool = True            # edges in couplets (transport)
    utility_gate: bool = True     # two-axis decision (else store all)
    exceptions: bool = True       # EXCEPTION op (else overwrite-merge)
    provenance: bool = True       # origin-bound: no relay, flip links
    world_clock: bool = True      # referent-version admissibility
    admission: str = "util"       # none | visit | util
    reservations: bool = True
    immutable: bool = True        # keep episodic originals
    sim_threshold: float = 0.999  # matching margin (lower under noise)
    anchor_consensus: int = 1     # safety: >=k anchors + loop closure
    closure_tol: float = 0.0      # beat-chain tolerance (continuous)
    unimodal_tol: float = 0.0     # continuous anchoring cluster radius
    commit_top1: bool = False     # safety: walk only the top target
    commit_dominance: float = 0.0 # safety: top >= this x best rival
    prefix_verify: int = 0        # safety: verify first N couplets
    prefix_tol: float = 0.5


@dataclass
class Record:
    """m = (G, C, E, U, P, T, R, F, A).  G=song, C=(intent,family),
    E implicit (reach intent tag), U=role_u, P=(origin,uid),
    T=(t,version), R re-derived at consumption, F=(kind,parent),
    A=admission/authority via world_clock + quarantine state."""
    song: List[Dict[str, Any]]            # G
    intent: str                           # C: applicability tag
    family: int                           # C: referent id
    role_u: Dict[str, float]              # U (per-role)
    origin: int                           # P
    uid: Tuple[int, int]                  # P
    t: int                                # T: observation time
    version: int                          # T: world version
    kind: str = "schema"                  # F: schema | exception
    parent: Optional[int] = None          # F
    support: int = 1


# ── cost accounting (one place --- reviewer item 9) ────────────────

def record_bits(rec: Record, cfg: Config) -> int:
    """Full-protocol record cost: song codec + certificate + (if on)
    provenance + timestamp/version."""
    song_bits = 0
    for c in rec.song:
        if cfg.landmarks:
            song_bits += LEN_BITS + KEY_BITS * len(c["sig"])
        if cfg.beats:
            song_bits += BEAT_BITS
    return song_bits + CERT_BITS + (PROV_BITS if cfg.provenance
                                    else 0) + TIME_BITS


def bits_of_song(song) -> int:
    """Pure song codec (nodes + edges only) --- NOT full protocol."""
    return sum(LEN_BITS + KEY_BITS * len(c["sig"]) + BEAT_BITS
               for c in song)


def bits_of_snapshot(fps) -> int:
    return COORD_BITS + sum(COORD_BITS + LEN_BITS + KEY_BITS * len(s)
                            for s in fps.values())
