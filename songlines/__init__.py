"""Songlines --- utility-certified analogical memory for multi-agent
navigation (Runtime v1).

Public API (substrate-agnostic method core):

    from songlines import Config, Record, SonglineAgent, song_target
    from songlines import Schema, Certificate, analogy, nearest, decide
    from songlines import record_bits, bits_of_song, bits_of_snapshot
    from songlines.config import ARMS, get

Substrates (grid, continuous) and experiment drivers live under
``experiments/song_grammar/``.  See ``songlines/README.md`` for the
layer map and ``docs/SONGLINES_V1_FREEZE.md`` for the freeze record.
"""

from songlines.record import (
    BEAT_BITS, CERT_BITS, COORD_BITS, KEY_BITS, LEN_BITS, PROV_BITS,
    RESV_BITS, ROLE_NAMES, SHARE_THR, TIME_BITS, D_THR, U_THR, Config,
    Record, bits_of_snapshot, bits_of_song, record_bits)
from songlines.analogy import (
    Certificate, Schema, SonglineMemory, analogy, decide, nearest)
from songlines.alignment import song_target
from songlines.runtime import SonglineAgent

__all__ = [
    "Config", "Record", "SonglineAgent", "song_target",
    "Schema", "Certificate", "SonglineMemory", "analogy", "nearest",
    "decide", "record_bits", "bits_of_song", "bits_of_snapshot",
    "ROLE_NAMES", "U_THR", "SHARE_THR", "D_THR",
    "KEY_BITS", "LEN_BITS", "BEAT_BITS", "COORD_BITS",
    "CERT_BITS", "PROV_BITS", "TIME_BITS", "RESV_BITS",
]
