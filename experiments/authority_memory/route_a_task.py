"""Authority Memory experiments — the route_A decision task, shared
by Sprint 7's collection smoke test and Sprint 8's E5 evaluation.

Deliberately noisy (a single decision must not reveal the causal
effect; only averaging over many randomized decisions does, which is
the entire point of the "no replay needed" claim): for a candidate
memory item asserting "route_A is traversable"
(``corruption_kit.role_dependent_claim``), the realised outcome of a
decision is

    y = BASE_REWARD - COST_PER_DISTANCE * distance_to_goal
        + Z * ROLE_UTILITY[role] + Gaussian noise (sigma=NOISE_SIGMA)

``distance_to_goal`` is a nuisance covariate: it moves the outcome
but not the treatment effect, and Z is randomized independent of it,
so it adds variance without introducing bias.

``paired_replay_tau`` is the OTHER way to query this same task: run
the SAME underlying noise draw twice, once per each value of Z, and
subtract --- the noise cancels exactly, giving the true tau for that
specific (state, role) with zero estimation error.  This is the
"UE1-equivalent" oracle for E5.1's deterministic-grid half
(``experiments/song_grammar/exp_ue1_utility_estimator.py`` is
calibrated on exactly this kind of exact-replay availability);
``make_decision_fn``/``collect_labels`` (unpaired, one Z draw per
decision) is what a substrate WITHOUT replay is stuck with, which is
E5.2's half.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple

from experiments.authority_memory.corruption_kit import ROLE_UTILITY

BASE_REWARD = 1.0
COST_PER_DISTANCE = 0.05
NOISE_SIGMA = 1.0
CERTIFICATE_ID = "cert-route-A"


def make_decision_fn(noise_seed: int):
    """Unpaired decision function: one independent noise draw per
    call.  This is the ONLY thing a non-replayable substrate can give
    an estimator --- no re-running the same decision with the other
    value of Z."""
    rng = random.Random(noise_seed)

    def decision_fn(state: Dict[str, Any], role: str, z: int) -> float:
        return (BASE_REWARD - COST_PER_DISTANCE * state["distance_to_goal"]
               + z * ROLE_UTILITY[role] + rng.gauss(0.0, NOISE_SIGMA))
    return decision_fn


def make_states(n_total: int, seed: int) -> List[Tuple[Dict[str, Any], str]]:
    rng = random.Random(seed)
    roles = list(ROLE_UTILITY)
    return [({"distance_to_goal": rng.uniform(1.0, 10.0)}, rng.choice(roles))
           for _ in range(n_total)]


def paired_replay_tau(state: Dict[str, Any], role: str,
                      noise_seed: int) -> float:
    """Exact counterfactual replay: the SAME noise draw applied under
    Z=1 and Z=0, subtracted --- gives ROLE_UTILITY[role] exactly,
    with zero estimation error, regardless of ``state``.  The
    deterministic-grid oracle E5.1 compares the trained estimator
    against.
    """
    y1 = (BASE_REWARD - COST_PER_DISTANCE * state["distance_to_goal"]
         + 1 * ROLE_UTILITY[role]
         + random.Random(noise_seed).gauss(0.0, NOISE_SIGMA))
    y0 = (BASE_REWARD - COST_PER_DISTANCE * state["distance_to_goal"]
         + 0 * ROLE_UTILITY[role]
         + random.Random(noise_seed).gauss(0.0, NOISE_SIGMA))
    return y1 - y0
