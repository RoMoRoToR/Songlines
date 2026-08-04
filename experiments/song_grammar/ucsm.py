"""Backward-compatibility shim.

The two-axis formation core (Schema, Certificate, analogy, nearest,
decide, SonglineMemory) moved into ``songlines.analogy`` (Stage 6
refactor).  This module re-exports it so validated drivers keep
importing ``experiments.song_grammar.ucsm`` unchanged.  New code
should import from ``songlines`` directly.
"""

from __future__ import annotations

from songlines.analogy import (  # noqa: F401
    Certificate, Schema, Song, SonglineMemory, analogy, decide,
    nearest)
