"""Authority Memory core — staleness and revocation (Sprint 4).

Operationalises Theorem 2 (bounded stale authority,
``docs/FRONTIER_MEMORY_AUTHORITY_2026-08-07/02_THEOREMS.md``
§Theorem 2): evidential admissibility decays exponentially with age
--- the same functional FORM as the frozen runtime's distance law
(``age_max(trust) = ln(trust*conf/tau)/alpha``, W2, 6/6 EXACT
hold-out, ``docs/SERIES_VERDICTS.md``), ported to the authority
layer's own evidence_score/rate/threshold rather than reusing the
grid-specific alpha=0.05/tau=0.30 constants --- those were calibrated
for a merge-weight scale this layer does not share; "do not retune"
(``docs/SONGLINES_V1_FREEZE.md`` §6) applies to THAT calibration, not
to this new layer's own.

A second, independent trigger lives here too:
``apply_world_version_check`` --- the S2 lesson from the frozen UCSM
series (``docs/FRONTIER_UCSM_2026-07-27.md`` §S2): age-based decay
alone is necessary but not sufficient, because its clock is
unrelated to WHEN the world actually changed.  A receiver who
detects an explicit world-version mismatch on a visit must revoke
immediately, regardless of what the decay clock says.
"""

from __future__ import annotations

import math
from typing import Optional

from authority_memory.authority_state import AuthorityState, transition
from authority_memory.certificate import MemoryCertificate

# Only PROVISIONAL and ADMITTED carry authority that staleness can
# withdraw (ALLOWED_TRANSITIONS restricts EXPIRED to exactly these
# two sources); QUARANTINED never had durable authority to begin
# with, and CONTESTED/SUPERSEDED/EXPIRED/REVOKED already left the
# durable-authority path some other way.
EXPIRABLE_STATES = (AuthorityState.PROVISIONAL, AuthorityState.ADMITTED)

# EXPIRED, CONTESTED and SUPERSEDED are the three sources
# ALLOWED_TRANSITIONS permits into REVOKED.
REVOCABLE_STATES = (AuthorityState.EXPIRED, AuthorityState.CONTESTED,
                    AuthorityState.SUPERSEDED)


def decay(evidence_at_t0: float, age: float, rate: float) -> float:
    """E(t) = E(t0) * exp(-rate * age) --- the same functional form as
    the frozen runtime's distance law (W2), applied to this layer's
    own evidence_score instead of a merge weight."""
    return evidence_at_t0 * math.exp(-rate * age)


def expiry_horizon(evidence_at_t0: float, rate: float,
                   tau_e: float) -> Optional[float]:
    """t_max - t0 = (1/rate) * ln(E(t0)/tau_e) --- the age at which
    ``decay()`` first drops below ``tau_e``.  ``0.0`` if
    ``evidence_at_t0`` is already at or below ``tau_e`` (expired at
    t0); ``None`` if ``rate <= 0`` (no decay --- never expires by age
    alone, a caller error to rely on for a real certificate but a
    legitimate limiting case to be able to express)."""
    if rate <= 0:
        return None
    if evidence_at_t0 <= tau_e:
        return 0.0
    return math.log(evidence_at_t0 / tau_e) / rate


def is_stale(evidence_at_t0: float, age: float, rate: float,
            tau_e: float) -> bool:
    """True once ``decay()`` has dropped strictly below ``tau_e`` ---
    the one predicate both ``apply_staleness`` and any caller use to
    decide whether the age gate has fired yet."""
    return decay(evidence_at_t0, age, rate) < tau_e


def apply_staleness(certificate: MemoryCertificate, *, age: float,
                    rate: float, tau_e: float, utility_lcb: float,
                    reason: str, timestamp: int) -> bool:
    """Check ``certificate.evidence_score`` (treated as its value at
    age 0) against its age-decayed value and, if that has crossed
    below ``tau_e``, force EXPIRED via the real FSM.  Returns True iff
    a transition was applied; a certificate outside
    ``EXPIRABLE_STATES`` is left untouched --- this function does one
    job, the age gate, not a general state audit.
    """
    if certificate.authority_state not in EXPIRABLE_STATES:
        return False
    decayed = decay(certificate.evidence_score, age, rate)
    if decayed >= tau_e:
        return False
    transition(certificate, AuthorityState.EXPIRED, evidence_score=decayed,
              utility_lcb=utility_lcb, reason=reason, timestamp=timestamp)
    return True


def apply_world_version_check(certificate: MemoryCertificate, *,
                              current_world_version: int,
                              utility_lcb: float, reason: str,
                              timestamp: int) -> bool:
    """Force EXPIRED immediately on a detected world-version
    mismatch, independent of the age-decay clock: a certificate
    stamped at an old ``created_world_version`` must not survive a
    KNOWN version change merely because its decay clock has not
    caught up yet.  Returns True iff a transition was applied.
    """
    if certificate.authority_state not in EXPIRABLE_STATES:
        return False
    if certificate.created_world_version == current_world_version:
        return False
    transition(certificate, AuthorityState.EXPIRED,
              evidence_score=certificate.evidence_score,
              utility_lcb=utility_lcb, reason=reason, timestamp=timestamp)
    return True


def revoke_expired(certificate: MemoryCertificate, *, utility_lcb: float,
                   reason: str, timestamp: int) -> bool:
    """EXPIRED/CONTESTED/SUPERSEDED -> REVOKED, the terminal step. Kept
    separate from ``apply_staleness``/``apply_world_version_check``
    because not every certificate that leaves durable authority is
    revoked in the same tick (a receiver may hold it pending
    explanation before formally withdrawing authority,
    ``01_FORMAL_MODEL.md`` §5); callers decide the timing, this
    function only performs the one valid transition.
    """
    if certificate.authority_state not in REVOCABLE_STATES:
        return False
    transition(certificate, AuthorityState.REVOKED,
              evidence_score=certificate.evidence_score,
              utility_lcb=utility_lcb, reason=reason, timestamp=timestamp)
    return True
