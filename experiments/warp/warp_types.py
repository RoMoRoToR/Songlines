"""Datatypes for the semantic-warp measurement layer.

An ``MStarEvent`` is created at the moment the planner commits to a
target (lock).  Foreignness phi is computed ONCE at that moment and
frozen (design §3.1): if the agent later confirms the place with its
own eyes, phi is NOT diluted — otherwise every successful warp would
wash itself out on approach and P(C*|W*) would be unmeasurable.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

THETA_STRICT = 0.999   # phi >= this  →  strict warp (never self-observed)
THETA_SOFT = 0.8       # working soft-warp threshold (design §12, default 0.8)


@dataclass
class MStarEvent:
    """One planner lock, annotated with provenance at lock time."""

    agent_id: str
    tick: int
    target_xy: Tuple[int, int]
    is_real_water: bool

    # ── provenance (frozen at lock moment) ───────────────────────
    phi: float                      # foreign mass share in [0, 1]
    own_mass: float
    foreign_mass: float
    per_source_mass: Dict[str, float] = field(default_factory=dict)
    self_ever_observed: bool = False   # ledger cross-check for strict W*

    # ── warp annotations ─────────────────────────────────────────
    w_star_strict: bool = False     # phi >= THETA_STRICT and never self-seen
    w_star_soft: bool = False       # phi >= THETA_SOFT
    warp_radius_cells: int = 0      # manhattan(agent position, target) at lock
    warp_radius_visited: int = 0    # min manhattan(visited set, target) at lock
    source_snapshot_age: Optional[float] = None  # freshest foreign source age
    co_recipients: int = 0          # other receivers of the same source snapshot
    co_locked: int = 0              # other agents locked on ~same target at lock

    # ── outcome (filled at episode end) ──────────────────────────
    completed: bool = False         # agent reached THIS target while locked
    completion_tick: Optional[int] = None
    dropped_tick: Optional[int] = None   # lock abandoned/retargeted at this tick

    # ── W3 warp-drive fields ─────────────────────────────────────
    retracted: bool = False         # anti-M* rollback fired for this lock
    retraction_tick: Optional[int] = None
    rollback_latency: Optional[int] = None  # ticks from lock to retraction

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["target_xy"] = list(self.target_xy)
        return d


@dataclass
class WarpEpisodeLog:
    """All M* events of one episode plus per-agent outcomes."""

    events: List[MStarEvent] = field(default_factory=list)
    first_success_tick: Dict[str, Optional[int]] = field(default_factory=dict)
    success_without_lock: Dict[str, bool] = field(default_factory=dict)

    def m_star_events(self) -> List[MStarEvent]:
        return [e for e in self.events if e.is_real_water]

    def warp_metrics(self) -> Dict[str, Any]:
        """Aggregate warp metrics over this episode's real-water locks."""
        m = self.m_star_events()
        n_m = len(m)
        strict = [e for e in m if e.w_star_strict]
        soft = [e for e in m if e.w_star_soft]
        own = [e for e in m if not e.w_star_soft]

        def _rate(evts):
            return (sum(1 for e in evts if e.completed) / len(evts)
                    if evts else float("nan"))

        return {
            "n_m_star": n_m,
            "n_w_star_strict": len(strict),
            "n_w_star_soft": len(soft),
            "warp_share_strict": len(strict) / n_m if n_m else float("nan"),
            "warp_share_soft": len(soft) / n_m if n_m else float("nan"),
            "p_C_given_W_strict": _rate(strict),
            "p_C_given_W_soft": _rate(soft),
            "p_C_given_M_own": _rate(own),
            "mean_phi": (sum(e.phi for e in m) / n_m) if n_m else float("nan"),
            "mean_warp_radius_strict": (
                sum(e.warp_radius_cells for e in strict) / len(strict)
                if strict else float("nan")),
        }
