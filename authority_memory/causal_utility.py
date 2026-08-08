"""Authority Memory core — randomized memory intervention (Sprint 7:
label collection; Sprint 8: the trained estimator tau_hat_theta).

Operationalises the causal utility definition from
``docs/FRONTIER_MEMORY_AUTHORITY_2026-08-07/01_FORMAL_MODEL.md`` §4:

    Z_{i,m,t} = 1 if m is available to policy i at t, else 0
    tau_i(m, s) = E[Y | do(Z=1), s] - E[Y | do(Z=0), s]

This is the deliberate alternative to UE1's exact-replay estimator
(``experiments/song_grammar/exp_ue1_utility_estimator.py``, Spearman
0.989 against an oracle rollout) --- valid only where replay is
available (a deterministic grid).  Randomized intervention needs no
replay: for a chosen fraction of "eligible" decisions, mask m with
probability 0.5 and log the realised outcome; averaging within each Z
group over many decisions recovers tau even when a single decision's
outcome is noisy/stochastic --- the substrate this is FOR, not the
deterministic grid UE1 already handles.

``fit_estimator``/``predict_tau`` (Sprint 8) turn Sprint 7's raw
labels into a trained tau_hat_theta(s, m, r): an OLS fit of
    y ~ (role dummies) + (Z * role dummies) + (numeric state features)
with no shared intercept/main-Z term --- every role gets its own
intercept and its own treatment coefficient directly, so
``tau_hat_theta(role)`` reads off one fitted coefficient, not a
difference of two.  This is a genuine regression, not
``empirical_tau``'s raw group-mean difference: adding numeric state
features as covariates (when the task has any that affect the
outcome but not the treatment effect) reduces residual variance and
therefore standard error at the same sample size --- the standard
covariate-adjustment argument for randomized designs, not an
arbitrary embellishment.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class InterventionLabel:
    """One (s, m, r, y, Z) tuple --- the label unit Sprint 8's
    estimator will be trained on.  ``state``/``role`` are kept as
    separate fields (matching ``01_FORMAL_MODEL.md``'s own signature
    tau_i(m|s,r)) even though, in a given task, role may also appear
    inside the state dict --- the estimator is free to use either or
    both."""
    state: Dict[str, Any]
    certificate_id: str
    role: str
    outcome: float
    z: int          # 1 = memory available, 0 = masked


def randomized_mask(rng: random.Random, p_available: float = 0.5) -> int:
    """One Bernoulli draw for whether the memory item is available
    (Z=1) or masked (Z=0) for this eligible decision --- isolated as
    its own function so a caller can substitute a different masking
    probability or a paired schedule without touching
    ``collect_labels``."""
    return 1 if rng.random() < p_available else 0


def collect_labels(decision_fn: Callable[[Dict[str, Any], str, int], float],
                   states: Sequence[Tuple[Dict[str, Any], str]],
                   certificate_id: str, *, p_available: float = 0.5,
                   seed: int = 0) -> List[InterventionLabel]:
    """Run one randomized-masking trial per (state, role) pair in
    ``states`` and return the resulting labels.  ``decision_fn(state,
    role, z) -> outcome`` is the task/environment's own decision +
    outcome function --- this module does not know or care what task
    produced the number, only that Z was randomized BEFORE the
    outcome was realised (the causal, not merely correlational,
    property the whole approach depends on).
    """
    rng = random.Random(seed)
    labels = []
    for state, role in states:
        z = randomized_mask(rng, p_available)
        outcome = decision_fn(state, role, z)
        labels.append(InterventionLabel(
            state=state, certificate_id=certificate_id, role=role,
            outcome=outcome, z=z))
    return labels


def empirical_tau(labels: Sequence[InterventionLabel]) -> Tuple[float, int, int]:
    """tau_hat = mean(y | Z=1) - mean(y | Z=0) over ``labels`` --- the
    simplest unbiased estimator of the randomized intervention effect
    (no model, no covariates; a real regression estimator that USES
    state/role as features is Sprint 8).  Returns
    ``(tau_hat, n_treated, n_control)``; raises if either arm is
    empty (undefined without both).
    """
    treated = [l.outcome for l in labels if l.z == 1]
    control = [l.outcome for l in labels if l.z == 0]
    if not treated or not control:
        raise ValueError("empirical_tau needs at least one label in "
                        "each of Z=1 and Z=0")
    tau_hat = sum(treated) / len(treated) - sum(control) / len(control)
    return tau_hat, len(treated), len(control)


def standard_error(labels: Sequence[InterventionLabel]) -> float:
    """Standard error of ``empirical_tau`` under simple random
    assignment: sqrt(Var(Y|Z=1)/n1 + Var(Y|Z=0)/n0) --- the textbook
    two-sample SE, needed to judge whether an empirical tau is
    distinguishable from zero (or from a candidate true value) given
    how many labels were collected."""
    treated = [l.outcome for l in labels if l.z == 1]
    control = [l.outcome for l in labels if l.z == 0]
    if len(treated) < 2 or len(control) < 2:
        raise ValueError("standard_error needs at least 2 labels in "
                        "each of Z=1 and Z=0")
    return math.sqrt(statistics.variance(treated) / len(treated)
                     + statistics.variance(control) / len(control))


@dataclass(frozen=True)
class CausalUtilityEstimator:
    """A trained tau_hat_theta(s, m, r) for one certificate: OLS
    coefficients from ``fit_estimator``.  ``tau`` holds the fitted
    per-role treatment effect directly (no reference-category algebra
    needed at prediction time); ``alpha`` and ``beta_state`` are kept
    for inspection but are not part of ``predict_tau``'s contract ---
    this task's true effect does not depend on state, and the
    parameterisation enforces that (no Z*state interaction term is
    fit), matching the ground truth rather than merely failing to
    contradict it."""
    certificate_id: str
    roles: Tuple[str, ...]
    state_keys: Tuple[str, ...]
    alpha: Dict[str, float]
    tau: Dict[str, float]
    beta_state: Dict[str, float]


def fit_estimator(labels: Sequence[InterventionLabel],
                  state_keys: Sequence[str] = ()) -> CausalUtilityEstimator:
    """Fit tau_hat_theta by OLS on ``labels``: design matrix columns
    are [role dummies | Z * role dummies | numeric state features],
    no shared intercept or main-Z column (avoids the redundant
    reference-category parameterisation entirely --- every role's
    intercept and treatment coefficient is directly one column each).
    """
    if not labels:
        raise ValueError("fit_estimator needs at least one label")
    certificate_id = labels[0].certificate_id
    roles = tuple(sorted({l.role for l in labels}))
    role_index = {r: i for i, r in enumerate(roles)}
    n_role = len(roles)
    n_state = len(state_keys)
    n = len(labels)
    x = np.zeros((n, 2 * n_role + n_state))
    y = np.zeros(n)
    for row, label in enumerate(labels):
        ridx = role_index[label.role]
        x[row, ridx] = 1.0
        x[row, n_role + ridx] = float(label.z)
        for j, key in enumerate(state_keys):
            x[row, 2 * n_role + j] = label.state[key]
        y[row] = label.outcome
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    alpha = {r: float(beta[role_index[r]]) for r in roles}
    tau = {r: float(beta[n_role + role_index[r]]) for r in roles}
    beta_state = {k: float(beta[2 * n_role + j])
                 for j, k in enumerate(state_keys)}
    return CausalUtilityEstimator(certificate_id=certificate_id,
                                  roles=roles, state_keys=tuple(state_keys),
                                  alpha=alpha, tau=tau,
                                  beta_state=beta_state)


def predict_tau(estimator: CausalUtilityEstimator, role: str) -> float:
    """tau_hat_theta(s, m, r) for a query role --- the whole point of
    training rather than just averaging (Sprint 7's ``empirical_tau``)
    is that this generalises to a query the training data did not
    literally contain; for this task tau does not depend on state, so
    the fitted model correctly does not use state to predict it
    either (state affects the predicted OUTCOME level through
    ``alpha``/``beta_state``, never ``tau``).
    """
    if role not in estimator.tau:
        raise KeyError(f"role {role!r} was not present in training "
                       f"labels (known roles: {estimator.roles})")
    return estimator.tau[role]
