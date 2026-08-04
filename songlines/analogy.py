"""UCSM — Utility-Certified Songline Memory (Stage 1: deterministic).

Core idea: usefulness and simplicity-of-analogy are two INDEPENDENT
axes, and the memory operation is a function of both:

    utility   analogy    operation
    high      simple     MERGE   (assimilate into the existing schema)
    high      simple+    EXCEPTION (conflicts with the schema's decision:
              conflict    store as a counterexample, do NOT overwrite)
    high      complex    NEW_SCHEMA
    low       simple     REPEAT  (update statistics, no new record)
    low       complex    DROP

Utility is COUNTERFACTUAL and MARGINAL: U(m | M) = cost(M) - cost(M+m)
via deterministic replay (the semantic-warp masking device generalised
from "foreign evidence on/off" to "this memory on/off").  A duplicate
of an already-stored schema therefore has U = 0 by construction --- no
similarity heuristic needed to detect repeats.

Analogy cost between two songs (couplet sequences):
    L      -- edit mass: couplets outside the signature-LCS;
    share  -- LCS length / min(len)  (structural overlap);
    D      -- decision distortion: manhattan distance between the
              end-to-end beat displacements (where the two songs
              ultimately SEND the walker).

A schema is stored with a utility CERTIFICATE: conditions, observed
counterfactual gain, uncertainty, evidence episode ids, known
failures.  Receivers recompute their own utility through their own
matching; the sender's number is testimony, not truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

Song = List[Dict[str, Any]]   # couplets: {"sig": {...}, "beat": (dx,dy), ...}


# ── certificates ───────────────────────────────────────────────────

@dataclass
class Certificate:
    conditions: Dict[str, Any]
    delta_v: float                 # observed counterfactual gain
    uncertainty: float             # 1 / support
    evidence: List[str]            # episode ids behind the schema
    failures: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Schema:
    song: Song
    cert: Certificate
    support: int = 1
    kind: str = "schema"           # "schema" | "exception"
    parent: Optional[int] = None   # index of the schema this excepts


# ── analogy ────────────────────────────────────────────────────────

def _sig_key(sig: Dict[str, float]) -> frozenset:
    return frozenset(sig.keys())


def _lcs(a: List[frozenset], b: List[frozenset]) -> int:
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            dp[i][j] = (dp[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1]
                        else max(dp[i - 1][j], dp[i][j - 1]))
    return dp[-1][-1]


def _displacement(song: Song) -> Tuple[int, int]:
    beats = [c.get("beat") for c in song]
    dx = sum(b[0] for b in beats if b is not None)
    dy = sum(b[1] for b in beats if b is not None)
    return dx, dy


def analogy(cand: Song, schema: Song) -> Dict[str, float]:
    """Structural analogy between two songs."""
    ka = [_sig_key(c["sig"]) for c in cand]
    kb = [_sig_key(c["sig"]) for c in schema]
    lcs = _lcs(ka, kb)
    da, db = _displacement(cand), _displacement(schema)
    return {
        "L": (len(ka) - lcs) + (len(kb) - lcs),
        "share": lcs / max(1, min(len(ka), len(kb))),
        "D": abs(da[0] - db[0]) + abs(da[1] - db[1]),
    }


def nearest(cand: Song, schemas: List[Schema]
            ) -> Tuple[Optional[int], Optional[Dict[str, float]]]:
    best_i, best = None, None
    for i, s in enumerate(schemas):
        a = analogy(cand, s.song)
        if best is None or a["share"] > best["share"]:
            best_i, best = i, a
    return best_i, best


# ── the two-axis decision ──────────────────────────────────────────

def decide(utility: float, ana: Optional[Dict[str, float]],
           u_thr: float, share_thr: float, d_thr: float) -> str:
    simple = ana is not None and ana["share"] >= share_thr
    conflict = simple and ana["D"] >= d_thr
    if utility >= u_thr:
        if conflict:
            return "EXCEPTION"
        return "MERGE" if simple else "NEW_SCHEMA"
    return "REPEAT" if simple else "DROP"


# ── memory ─────────────────────────────────────────────────────────

class SonglineMemory:
    """Ordered store of schemas; exceptions sit AFTER their parent, so
    a consumer tries the better-supported general schema first and
    falls through to the counterexample (fail-closed detour, never a
    silent overwrite)."""

    def __init__(self, u_thr: float, share_thr: float, d_thr: float):
        self.schemas: List[Schema] = []
        self.u_thr, self.share_thr, self.d_thr = u_thr, share_thr, d_thr
        self.log: List[Dict[str, Any]] = []

    def consider(self, cand: Song, utility: float, episode_id: str,
                 conditions: Dict[str, Any]) -> str:
        idx, ana = nearest(cand, self.schemas)
        op = decide(utility, ana, self.u_thr, self.share_thr, self.d_thr)
        if op == "REPEAT" and idx is not None:
            self.schemas[idx].support += 1
            self.schemas[idx].cert.uncertainty = \
                1.0 / self.schemas[idx].support
            self.schemas[idx].cert.evidence.append(episode_id)
        elif op == "MERGE" and idx is not None:
            s = self.schemas[idx]
            s.song = cand                      # refresh with newer route
            s.support += 1
            s.cert.delta_v = max(s.cert.delta_v, utility)
            s.cert.evidence.append(episode_id)
        elif op == "NEW_SCHEMA":
            self.schemas.append(Schema(cand, Certificate(
                conditions, utility, 1.0, [episode_id])))
        elif op == "EXCEPTION" and idx is not None:
            self.schemas[idx].cert.failures.append(
                {"episode": episode_id, "distortion": ana["D"]})
            self.schemas.append(Schema(cand, Certificate(
                conditions, utility, 1.0, [episode_id]),
                kind="exception", parent=idx))
        self.log.append({"episode": episode_id, "op": op,
                         "utility": round(utility, 2),
                         "analogy": ana})
        return op

    def ordered(self) -> List[Schema]:
        return list(self.schemas)   # insertion order: parents first
