"""Sprint 7-8 invariant tests --- randomized-masking label collection
must recover a KNOWN synthetic causal effect (Sprint 7), and the
trained OLS estimator must match/beat the raw group-mean estimator at
the same sample size while generalising per-role coefficients
correctly (Sprint 8).  No pytest dependency: run

    PYTHONPATH=. python -m authority_memory.tests.test_causal_utility
"""

from __future__ import annotations

import random

from authority_memory.causal_utility import (collect_labels, empirical_tau,
                                             fit_estimator, predict_tau,
                                             randomized_mask,
                                             standard_error)

_checks = []


def check(name, cond):
    _checks.append((name, bool(cond)))


TRUE_TAU = 0.4
NOISE_SIGMA = 1.0


def _decision_fn(rng):
    def fn(state, role, z):
        return z * TRUE_TAU + rng.gauss(0.0, NOISE_SIGMA)
    return fn


def _states(n, seed):
    rng = random.Random(seed)
    return [({"x": rng.uniform(0, 1)}, "role-A") for _ in range(n)]


# ── randomized_mask: roughly balanced, seed-reproducible -------------
def test_randomized_mask_balance_and_determinism():
    rng1 = random.Random(0)
    draws = [randomized_mask(rng1) for _ in range(2000)]
    rate = sum(draws) / len(draws)
    check("randomized_mask_roughly_balanced_at_p_0.5",
          0.45 <= rate <= 0.55)
    check("randomized_mask_draws_are_0_or_1", set(draws) <= {0, 1})

    rng_a, rng_b = random.Random(42), random.Random(42)
    seq_a = [randomized_mask(rng_a) for _ in range(50)]
    seq_b = [randomized_mask(rng_b) for _ in range(50)]
    check("randomized_mask_reproducible_given_same_seed", seq_a == seq_b)


# ── empirical_tau recovers a known synthetic effect -------------------
def test_empirical_tau_recovers_known_effect():
    labels = collect_labels(_decision_fn(random.Random(1)),
                            _states(4000, seed=2), "cert-x", seed=3)
    tau_hat, n1, n0 = empirical_tau(labels)
    se = standard_error(labels)
    check("empirical_tau_within_4se_of_truth",
          abs(tau_hat - TRUE_TAU) < 4 * se)
    check("both_arms_nonempty_and_roughly_balanced",
          n1 > 1500 and n0 > 1500 and n1 + n0 == 4000)


def test_empirical_tau_requires_both_arms():
    all_treated = [collect_labels(_decision_fn(random.Random(1)),
                                  _states(5, seed=9), "cert-x",
                                  p_available=1.0, seed=9)][0]
    raised = False
    try:
        empirical_tau(all_treated)
    except ValueError:
        raised = True
    check("empirical_tau_raises_without_both_arms", raised)


# ── standard error shrinks as more labels are collected --------------
def test_standard_error_shrinks_with_sample_size():
    ses = []
    for n in (100, 500, 2000, 8000):
        labels = collect_labels(_decision_fn(random.Random(1)),
                                _states(n, seed=5), "cert-x", seed=6)
        ses.append(standard_error(labels))
    check("standard_error_monotonically_shrinks_with_n",
          all(ses[i] > ses[i + 1] for i in range(len(ses) - 1)))


# ── fit_estimator/predict_tau: per-role OLS coefficients -------------
ROLE_TAU = {"scout": 0.5, "carrier": 0.2, "fragile": -0.5, "fast": 0.3}


def _role_decision_fn(rng):
    def fn(state, role, z):
        return (state["baseline"] + z * ROLE_TAU[role]
               + rng.gauss(0.0, NOISE_SIGMA))
    return fn


def _role_states(n, seed):
    rng = random.Random(seed)
    roles = list(ROLE_TAU)
    return [({"baseline": rng.uniform(0, 2)}, rng.choice(roles))
           for _ in range(n)]


def test_fit_estimator_recovers_per_role_tau():
    labels = collect_labels(_role_decision_fn(random.Random(1)),
                            _role_states(8000, seed=2), "cert-route",
                            seed=3)
    estimator = fit_estimator(labels, state_keys=("baseline",))
    check("fit_estimator_known_roles_match",
          set(estimator.roles) == set(ROLE_TAU))
    check("fit_estimator_recovers_sign_for_every_role",
          all((predict_tau(estimator, r) > 0) == (ROLE_TAU[r] > 0)
             for r in ROLE_TAU))
    check("fit_estimator_close_to_true_tau_for_every_role",
          all(abs(predict_tau(estimator, r) - ROLE_TAU[r]) < 0.1
             for r in ROLE_TAU))


def test_fit_estimator_unknown_role_raises():
    labels = collect_labels(_role_decision_fn(random.Random(1)),
                            _role_states(500, seed=4), "cert-route",
                            seed=5)
    estimator = fit_estimator(labels, state_keys=("baseline",))
    raised = False
    try:
        predict_tau(estimator, "not-a-real-role")
    except KeyError:
        raised = True
    check("predict_tau_raises_on_unknown_role", raised)


def test_fit_estimator_matches_group_mean_without_covariate():
    # With no state covariate and a state-independent effect, the OLS
    # fit's per-role tau must equal empirical_tau on the SAME
    # per-role subset --- the model is a strict generalisation of the
    # group-mean estimator, not a different number by coincidence.
    labels = collect_labels(_role_decision_fn(random.Random(1)),
                            _role_states(3000, seed=6), "cert-route",
                            seed=7)
    estimator = fit_estimator(labels, state_keys=())
    ok = True
    for role in ROLE_TAU:
        role_labels = [l for l in labels if l.role == role]
        tau_hat, _, _ = empirical_tau(role_labels)
        ok = ok and abs(predict_tau(estimator, role) - tau_hat) < 1e-9
    check("fit_estimator_without_covariates_equals_group_mean", ok)


def test_fit_estimator_requires_labels():
    raised = False
    try:
        fit_estimator([])
    except ValueError:
        raised = True
    check("fit_estimator_raises_on_empty_labels", raised)


def main():
    for fn in (test_randomized_mask_balance_and_determinism,
              test_empirical_tau_recovers_known_effect,
              test_empirical_tau_requires_both_arms,
              test_standard_error_shrinks_with_sample_size,
              test_fit_estimator_recovers_per_role_tau,
              test_fit_estimator_unknown_role_raises,
              test_fit_estimator_matches_group_mean_without_covariate,
              test_fit_estimator_requires_labels):
        fn()
    ok = sum(1 for _, c in _checks if c)
    for name, c in _checks:
        print(f"  [{'PASS' if c else 'FAIL'}] {name}")
    print(f"{ok}/{len(_checks)} causal-utility checks passed")
    return 0 if ok == len(_checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
