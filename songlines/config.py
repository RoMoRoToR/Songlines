"""Named arm and ablation configurations --- one registry.

Every benchmark arm and every ablation is a ``Config`` here, so no
experiment carries a hand-forked copy of the runtime logic (reviewer
item 6).  The integration driver (``exp_i1_integration``) and the
continuous driver (``exp_c1_continuous``) read from these.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, Tuple

from songlines.record import Config

FULL = Config()


def _c(**kw) -> Config:
    return replace(FULL, **kw)


# name -> (Config, communicating?)
ARMS: Dict[str, Tuple[Config, bool]] = {
    # ── main arms (grid integration benchmark, I1) ──────────────────
    "independent": (FULL, False),
    "raw_history": (_c(utility_gate=False, exceptions=False,
                       world_clock=False, admission="none",
                       reservations=False), True),
    "vector_sim": (_c(utility_gate=False, exceptions=False,
                      world_clock=False, admission="none",
                      reservations=False, beats=False), True),
    "graph_no_prov": (_c(provenance=False, admission="none",
                         reservations=False), True),
    "song_plain": (_c(world_clock=False, admission="none",
                      provenance=False, reservations=False), True),
    "song_trust": (_c(world_clock=False, admission="none",
                      reservations=False), True),
    "song_wclock": (_c(admission="none", reservations=False), True),
    "songline_full": (FULL, True),
    # ── one-flag ablations of the full method ───────────────────────
    "no_landmarks": (_c(landmarks=False), True),
    "no_beats": (_c(beats=False), True),
    "no_provenance": (_c(provenance=False), True),
    "no_worldclock": (_c(world_clock=False), True),
    "no_admission": (_c(admission="none"), True),
    "no_exceptions": (_c(exceptions=False), True),
    "no_reservations": (_c(reservations=False), True),
    "no_immutable": (_c(immutable=False), True),
    "no_utility_gate": (_c(utility_gate=False), True),
}

# ── continuous substrate (C1): same runtime, continuous matching ────
_C = dict(sim_threshold=0.6, closure_tol=0.7, anchor_consensus=2,
          unimodal_tol=1.6)
CONTINUOUS_ARMS: Dict[str, Tuple[Config, bool]] = {
    "independent": (_c(**_C), False),
    "raw_history": (_c(**_C, utility_gate=False, exceptions=False,
                       world_clock=False, admission="none",
                       reservations=False), True),
    "songline_full": (_c(**_C), True),
    "songline_safe": (_c(sim_threshold=0.6, closure_tol=1.2,
                         anchor_consensus=3, unimodal_tol=1.6,
                         commit_top1=True, prefix_verify=5,
                         prefix_tol=0.55), True),
}


def get(name: str, continuous: bool = False) -> Tuple[Config, bool]:
    return (CONTINUOUS_ARMS if continuous else ARMS)[name]
