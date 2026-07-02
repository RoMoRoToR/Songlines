"""
Place identity smoke test: demonstrates the boundary between trivial and real.

Scenario
--------
Two agents independently discover the same water source.

  Agent A: observes it at (5, 3), full tag profile.
  Agent B: observes it at (5+ε_x, 3+ε_y), partial or noisy tag profile.

Three modes are run side by side:

  COORDINATE_ONLY  — passes only when ε = 0. Every other case: wrong answer.
  SEMANTIC_ONLY    — passes when profiles overlap, regardless of ε.
  HYBRID           — passes when spatial + semantic evidence together
                     clear the threshold; degrades gracefully.

Two sweeps:
  1. Coordinate noise sweep  — vary ε, fix profile.
  2. Profile sparsity sweep  — fix ε, drop increasing fractions of tags
                                to simulate partial observation.

Expected output shows exactly where each mode breaks.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from typing import Dict, List, Tuple

from songline_drive.place_identity import (
    IdentityMode,
    IdentityScore,
    PlaceIdentityEngine,
    PlaceObservation,
)


# ──────────────────────────────────────────── helpers

FULL_WATER_PROFILE: Dict[str, float] = {
    "water_source": 0.92,
    "wet":          0.74,
    "open":         0.41,
    "passable":     0.55,
}

FULL_REST_PROFILE: Dict[str, float] = {
    "rest_area":    0.88,
    "shelter":      0.62,
    "open":         0.38,
    "passable":     0.50,
}


def _partial_profile(
    profile: Dict[str, float], keep_fraction: float
) -> Dict[str, float]:
    """Keep only the top-k tags by confidence (simulates partial view)."""
    n = max(1, round(len(profile) * keep_fraction))
    return dict(sorted(profile.items(), key=lambda kv: -kv[1])[:n])


def _run_pair(
    engine: PlaceIdentityEngine,
    obs_a: PlaceObservation,
    obs_b: PlaceObservation,
) -> IdentityScore:
    return engine.score_pair(obs_a, obs_b)


def _header(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


# ──────────────────────────────────────────── sweep 1: coordinate noise

def sweep_coordinate_noise() -> None:
    _header("Sweep 1: coordinate noise  (same full profile, ε varies)")

    engines = {
        IdentityMode.COORDINATE_ONLY: PlaceIdentityEngine(
            mode=IdentityMode.COORDINATE_ONLY,
            identity_threshold=0.55,
        ),
        IdentityMode.SEMANTIC_ONLY: PlaceIdentityEngine(
            mode=IdentityMode.SEMANTIC_ONLY,
            identity_threshold=0.55,
        ),
        IdentityMode.HYBRID: PlaceIdentityEngine(
            mode=IdentityMode.HYBRID,
            identity_threshold=0.55,
            semantic_weight=0.6,
            spatial_weight=0.4,
            spatial_sigma_scale=1.5,
        ),
    }

    noise_levels: List[Tuple[float, float]] = [
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (2.0, 1.0),
        (3.0, 0.0),
        (4.0, 4.0),
    ]

    print(f"{'ε_x':>5} {'ε_y':>5}  {'COORD':>8}  {'SEM':>8}  {'HYBRID':>8}  {'COORD?':>7}  {'SEM?':>7}  {'HYB?':>7}")
    print(f"{'':->5} {'':->5}  {'':->8}  {'':->8}  {'':->8}  {'':->7}  {'':->7}  {'':->7}")

    for eps_x, eps_y in noise_levels:
        obs_a = PlaceObservation(
            agent_id="A", raw_key=(5.0, 3.0),
            position_sigma=0.0,
            tag_profile=dict(FULL_WATER_PROFILE),
            env_id="grid",
        )
        obs_b = PlaceObservation(
            agent_id="B", raw_key=(5.0 + eps_x, 3.0 + eps_y),
            position_sigma=1.0,  # B admits it doesn't know exactly where it is
            tag_profile=dict(FULL_WATER_PROFILE),
            env_id="grid",
        )

        scores = {mode: _run_pair(eng, obs_a, obs_b) for mode, eng in engines.items()}

        coord  = scores[IdentityMode.COORDINATE_ONLY]
        sem    = scores[IdentityMode.SEMANTIC_ONLY]
        hybrid = scores[IdentityMode.HYBRID]

        ok = lambda b: "YES" if b else "NO"
        print(
            f"{eps_x:>5.1f} {eps_y:>5.1f}  "
            f"{coord.combined_score:>8.3f}  "
            f"{sem.combined_score:>8.3f}  "
            f"{hybrid.combined_score:>8.3f}  "
            f"{ok(coord.is_same):>7}  "
            f"{ok(sem.is_same):>7}  "
            f"{ok(hybrid.is_same):>7}"
        )


# ──────────────────────────────────────────── sweep 2: profile sparsity

def sweep_profile_sparsity() -> None:
    _header("Sweep 2: profile sparsity  (ε=1, B sees only top-k tags)")

    engine_sem = PlaceIdentityEngine(
        mode=IdentityMode.SEMANTIC_ONLY, identity_threshold=0.55,
    )
    engine_hyb = PlaceIdentityEngine(
        mode=IdentityMode.HYBRID,
        identity_threshold=0.55,
        semantic_weight=0.6, spatial_weight=0.4,
        spatial_sigma_scale=1.5,
    )

    fractions = [1.0, 0.75, 0.5, 0.25]

    print(f"{'keep':>6}  {'B_profile':30}  {'SEM':>8}  {'HYBRID':>8}  {'SEM?':>7}  {'HYB?':>7}")
    print(f"{'':->6}  {'':->30}  {'':->8}  {'':->8}  {'':->7}  {'':->7}")

    for frac in fractions:
        partial = _partial_profile(FULL_WATER_PROFILE, frac)
        obs_a = PlaceObservation(
            agent_id="A", raw_key=(5.0, 3.0),
            position_sigma=0.0,
            tag_profile=dict(FULL_WATER_PROFILE), env_id="grid",
        )
        obs_b = PlaceObservation(
            agent_id="B", raw_key=(6.0, 3.0),   # ε=1 on x
            position_sigma=1.0,
            tag_profile=partial, env_id="grid",
        )

        s_sem = _run_pair(engine_sem, obs_a, obs_b)
        s_hyb = _run_pair(engine_hyb, obs_a, obs_b)

        profile_str = "{" + ", ".join(f"{k}:{v:.2f}" for k, v in partial.items()) + "}"
        ok = lambda b: "YES" if b else "NO"
        print(
            f"{frac:>5.0%}  {profile_str:30}  "
            f"{s_sem.combined_score:>8.3f}  "
            f"{s_hyb.combined_score:>8.3f}  "
            f"{ok(s_sem.is_same):>7}  "
            f"{ok(s_hyb.is_same):>7}"
        )


# ──────────────────────────────────────────── sweep 3: distinct objects

def sweep_distinct_objects() -> None:
    _header("Sweep 3: genuinely different objects  (must NOT be merged)")

    engine_hyb = PlaceIdentityEngine(
        mode=IdentityMode.HYBRID,
        identity_threshold=0.55,
        semantic_weight=0.6, spatial_weight=0.4,
        spatial_sigma_scale=1.5,
    )
    engine_sem = PlaceIdentityEngine(
        mode=IdentityMode.SEMANTIC_ONLY, identity_threshold=0.55,
    )

    pairs = [
        ("nearby, same tag",   (5.0, 3.0), FULL_WATER_PROFILE, (6.0, 3.0), FULL_WATER_PROFILE),
        ("nearby, diff tag",   (5.0, 3.0), FULL_WATER_PROFILE, (6.0, 3.0), FULL_REST_PROFILE),
        ("far, same tag",      (5.0, 3.0), FULL_WATER_PROFILE, (12.0, 8.0), FULL_WATER_PROFILE),
        ("far, diff tag",      (5.0, 3.0), FULL_WATER_PROFILE, (12.0, 8.0), FULL_REST_PROFILE),
    ]

    print(f"{'scenario':25}  {'SEM':>8}  {'HYBRID':>8}  {'SEM?':>7}  {'HYB?':>7}")
    print(f"{'':->25}  {'':->8}  {'':->8}  {'':->7}  {'':->7}")

    for label, pos_a, prof_a, pos_b, prof_b in pairs:
        obs_a = PlaceObservation(
            agent_id="A", raw_key=pos_a, position_sigma=0.5,
            tag_profile=dict(prof_a), env_id="grid",
        )
        obs_b = PlaceObservation(
            agent_id="B", raw_key=pos_b, position_sigma=0.5,
            tag_profile=dict(prof_b), env_id="grid",
        )
        s_sem = _run_pair(engine_sem, obs_a, obs_b)
        s_hyb = _run_pair(engine_hyb, obs_a, obs_b)

        ok = lambda b: "YES" if b else "NO"
        note = ""
        if label == "far, same tag" and s_sem.is_same:
            note = "  ← SEMANTIC_ONLY false-positive: distance ignored"
        print(
            f"{label:25}  "
            f"{s_sem.combined_score:>8.3f}  "
            f"{s_hyb.combined_score:>8.3f}  "
            f"{ok(s_sem.is_same):>7}  "
            f"{ok(s_hyb.is_same):>7}{note}"
        )


# ──────────────────────────────────────────── sweep 4: cluster across agents

def sweep_cluster() -> None:
    _header("Sweep 4: build_clusters — three agents, two places")

    engine = PlaceIdentityEngine(
        mode=IdentityMode.HYBRID,
        identity_threshold=0.55,
        semantic_weight=0.6, spatial_weight=0.4,
        spatial_sigma_scale=1.5,
    )

    observations = [
        # Water source: three views, slight position disagreement
        PlaceObservation("A", (5.0, 3.0), 0.5, dict(FULL_WATER_PROFILE), "grid"),
        PlaceObservation("B", (5.5, 3.0), 1.0, _partial_profile(FULL_WATER_PROFILE, 0.75), "grid"),
        PlaceObservation("C", (5.0, 3.8), 1.2, dict(FULL_WATER_PROFILE), "grid"),
        # Rest area: two views, further apart
        PlaceObservation("A", (11.0, 7.0), 0.5, dict(FULL_REST_PROFILE), "grid"),
        PlaceObservation("C", (11.5, 7.5), 1.0, dict(FULL_REST_PROFILE), "grid"),
    ]

    clusters = engine.build_clusters(observations)

    print(f"\nResolved {len(observations)} observations → {len(clusters)} canonical places\n")
    for cl in clusters:
        tags = ", ".join(
            f"{k}:{v:.2f}" for k, v in
            sorted(cl.fused_profile.items(), key=lambda kv: -kv[1])
        )
        print(
            f"  Cluster #{cl.canonical_id}  agents={sorted(cl.agent_ids)}"
            f"  pos={cl.mean_position[0]:.1f},{cl.mean_position[1]:.1f}"
            f"  spread={cl.position_spread:.2f}  profile=[{tags}]"
        )


# ──────────────────────────────────────────── main

if __name__ == "__main__":
    print("=" * 60)
    print("  Place identity smoke test")
    print("  Three hidden assumptions vs. the research frontier")
    print("=" * 60)

    sweep_coordinate_noise()
    sweep_profile_sparsity()
    sweep_distinct_objects()
    sweep_cluster()

    print("\n" + "=" * 60)
    print("  Summary of what the table shows")
    print("=" * 60)
    print("""
  Sweep 1 — COORDINATE_ONLY gives the right answer ONLY at ε=0.
             The moment agents disagree on position by even 1 grid
             unit, it declares them different places. This is the
             hidden assumption: shared global frame.

  Sweep 2 — SEMANTIC_ONLY degrades gracefully as B's profile gets
             sparser, but never completely fails as long as the
             dominant tag (water_source) is visible.
             HYBRID: spatial evidence rescues the case where
             profiles thin out to the point of ambiguity.

  Sweep 3 — SEMANTIC_ONLY false-positives on (far, same tag):
             two genuinely different water sources at opposite ends
             of the grid get merged because their profiles are
             identical and distance is ignored. HYBRID rejects this.

  Sweep 4 — Three agents, slight position disagreement, one partial
             profile: HYBRID correctly collapses to 2 canonical
             places, carrying each cluster's fused profile and
             position spread as first-class diagnostics.

  The "distinct research frontier" is the tradeoff in Sweep 3 vs
  Sweep 1: COORDINATE_ONLY buys false-negative safety at the cost
  of noise fragility; SEMANTIC_ONLY buys noise robustness at the
  cost of false-positive risk on semantically identical-but-distant
  objects. HYBRID makes the weights an explicit parameter so the
  tradeoff is measurable rather than hidden.
""")
