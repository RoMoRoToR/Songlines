"""Backward-compatibility shim.

The runtime moved into the ``songlines`` package (Stage 6 refactor).
This module re-exports the public API so the validated experiment
drivers keep importing ``experiments.song_grammar.runtime`` unchanged.
New code should import from ``songlines`` directly.
"""

from __future__ import annotations

from songlines.record import (  # noqa: F401
    BEAT_BITS, CERT_BITS, COORD_BITS, KEY_BITS, LEN_BITS, PROV_BITS,
    RESV_BITS, ROLE_NAMES, SHARE_THR, TIME_BITS, D_THR, U_THR, Config,
    Record, bits_of_snapshot, bits_of_song, record_bits)
from songlines.alignment import song_target  # noqa: F401
from songlines.runtime import SonglineAgent  # noqa: F401

GridXY = tuple
