"""Phase 4d — outcome-driven adaptive field parameter reweighting.

``FieldOutcomeTracker`` tracks navigation outcomes per concept and adjusts
``SemanticField`` hyperparameters based on empirical success/failure patterns.

Adaptive rules (Phase 4 spec §8):

1. High-conflict concept fails repeatedly
   → increase ``field.eta_conflict`` (stronger conflict suppression)

2. Reservation consistently correlates with success
   → maintain or increase ``field.xi_occupancy`` (stronger deconfliction)
   If reservation has poor success rate → reduce ``xi_occupancy``

3. Overall high failure rate across many outcomes
   → reduce ``field.gamma_diffusion`` (diffusion may be spreading misleading signal)

All rules use a rolling window and EMA-like multiplicative adjustments.
Parameters are clamped to hard bounds so the field cannot be driven to
degenerate extremes.

Usage::

    tracker = FieldOutcomeTracker(field, window=10)
    # ... agents navigate ...
    tracker.record_concept_outcome(contested_cid, success=False)
    tracker.record_concept_outcome(contested_cid, success=False)
    tracker.record_concept_outcome(contested_cid, success=False)
    changes = tracker.adapt(min_samples=3)
    # rebuild field with new parameters
    adapter.refresh(collective)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from songline_drive.semantic_field import SemanticField


class FieldOutcomeTracker:
    """Tracks navigation outcomes per concept and adapts field parameters.

    Parameters
    ----------
    field:
        The ``SemanticField`` whose parameters will be adjusted.
    window:
        Rolling window size for outcome history per concept.  Older outcomes
        are discarded when the buffer exceeds this size.

    Adaptive coefficient bounds
    ---------------------------
    eta_conflict  : [0.10, 0.95]
    xi_occupancy  : [0.05, 0.60]
    gamma_diffusion: [0.00, 0.30]
    """

    # Hard bounds for adaptive parameters
    ETA_CONFLICT_MIN: float = 0.10
    ETA_CONFLICT_MAX: float = 0.95
    XI_OCCUPANCY_MIN: float = 0.05
    XI_OCCUPANCY_MAX: float = 0.60
    GAMMA_DIFFUSION_MIN: float = 0.00
    GAMMA_DIFFUSION_MAX: float = 0.30

    def __init__(self, field: SemanticField, window: int = 10) -> None:
        self.field = field
        self.window = int(window)
        self._concept_outcomes: Dict[str, List[bool]] = {}
        self._reservation_success: List[bool] = []
        self._adaptation_history: List[Dict] = []

        # Store baseline values for reporting
        self._eta_conflict_init = field.eta_conflict
        self._xi_occupancy_init = field.xi_occupancy
        self._gamma_diffusion_init = field.gamma_diffusion

    # ─────────────────────────────────────────── record outcomes

    def record_concept_outcome(self, concept_id: str, success: bool) -> None:
        """Record whether navigating to ``concept_id`` succeeded."""
        buf = self._concept_outcomes.setdefault(concept_id, [])
        buf.append(success)
        if len(buf) > self.window:
            buf.pop(0)

    def record_reservation_outcome(self, success: bool) -> None:
        """Record whether an episode where a reservation was used succeeded."""
        self._reservation_success.append(success)
        if len(self._reservation_success) > self.window:
            self._reservation_success.pop(0)

    # ─────────────────────────────────────────── per-concept statistics

    def failure_rate_for(self, concept_id: str) -> Optional[float]:
        """Failure rate ∈ [0, 1] for ``concept_id``, or None if no data."""
        outcomes = self._concept_outcomes.get(concept_id, [])
        if not outcomes:
            return None
        return outcomes.count(False) / len(outcomes)

    def success_rate_for(self, concept_id: str) -> Optional[float]:
        fr = self.failure_rate_for(concept_id)
        return None if fr is None else 1.0 - fr

    # ─────────────────────────────────────────── adapt

    def adapt(self, min_samples: int = 3) -> Dict[str, float]:
        """Apply outcome-driven reweighting rules once.

        Returns a dict of ``{parameter_name: new_value}`` for any parameters
        that were actually changed.  Empty dict means no rule fired.

        Rebuild the field (``field.rebuild_from_concepts()``) after calling
        ``adapt()`` to propagate the new coefficients to activations.
        """
        changes: Dict[str, float] = {}

        # ── Rule 1: high-conflict concept with high failure rate
        #    → boost conflict suppression (eta_conflict ×1.15)
        for cid, outcomes in self._concept_outcomes.items():
            if len(outcomes) < min_samples:
                continue
            cell = self.field.cells.get(cid)
            if cell is None or cell.base_conflict < 0.15:
                continue
            fail_rate = outcomes.count(False) / len(outcomes)
            if fail_rate >= 0.60:
                new_eta = min(
                    self.ETA_CONFLICT_MAX,
                    self.field.eta_conflict * 1.15,
                )
                if new_eta != self.field.eta_conflict:
                    self.field.eta_conflict = new_eta
                    changes["eta_conflict"] = round(new_eta, 4)
                break  # one adjustment per adapt() call to avoid oscillation

        # ── Rule 2: reservation outcome rate
        #    success ≥ 0.70 → increase xi_occupancy ×1.05
        #    success < 0.30 → decrease xi_occupancy ×0.95
        res_outcomes = self._reservation_success
        if len(res_outcomes) >= min_samples:
            res_rate = sum(res_outcomes) / len(res_outcomes)
            if res_rate >= 0.70:
                new_xi = min(
                    self.XI_OCCUPANCY_MAX,
                    self.field.xi_occupancy * 1.05,
                )
            elif res_rate < 0.30:
                new_xi = max(
                    self.XI_OCCUPANCY_MIN,
                    self.field.xi_occupancy * 0.95,
                )
            else:
                new_xi = self.field.xi_occupancy
            if new_xi != self.field.xi_occupancy:
                self.field.xi_occupancy = new_xi
                changes["xi_occupancy"] = round(new_xi, 4)

        # ── Rule 3: global high failure rate → reduce diffusion
        all_outcomes = [s for o in self._concept_outcomes.values() for s in o]
        if len(all_outcomes) >= min_samples * 2:
            global_fail = all_outcomes.count(False) / len(all_outcomes)
            if global_fail >= 0.60:
                new_gamma = max(
                    self.GAMMA_DIFFUSION_MIN,
                    self.field.gamma_diffusion * 0.90,
                )
                if new_gamma != self.field.gamma_diffusion:
                    self.field.gamma_diffusion = new_gamma
                    changes["gamma_diffusion"] = round(new_gamma, 4)

        if changes:
            self._adaptation_history.append({
                "changes": changes,
                "n_concept_outcomes": len(all_outcomes),
                "n_reservation_outcomes": len(res_outcomes),
            })

        return changes

    # ─────────────────────────────────────────── introspection

    @property
    def adaptation_history(self) -> List[Dict]:
        return list(self._adaptation_history)

    def parameter_delta(self) -> Dict[str, float]:
        """Current parameter values relative to initial values."""
        return {
            "eta_conflict": round(self.field.eta_conflict - self._eta_conflict_init, 4),
            "xi_occupancy": round(self.field.xi_occupancy - self._xi_occupancy_init, 4),
            "gamma_diffusion": round(
                self.field.gamma_diffusion - self._gamma_diffusion_init, 4
            ),
        }

    def summary(self) -> Dict:
        """Snapshot of tracker state for logging / diagnostics."""
        return {
            "n_tracked_concepts": len(self._concept_outcomes),
            "n_reservation_outcomes": len(self._reservation_success),
            "n_adaptations": len(self._adaptation_history),
            "current_eta_conflict": round(self.field.eta_conflict, 4),
            "current_xi_occupancy": round(self.field.xi_occupancy, 4),
            "current_gamma_diffusion": round(self.field.gamma_diffusion, 4),
            "parameter_delta": self.parameter_delta(),
            "per_concept_failure_rate": {
                cid: round(self.failure_rate_for(cid), 4)
                for cid in self._concept_outcomes
                if self.failure_rate_for(cid) is not None
            },
        }
