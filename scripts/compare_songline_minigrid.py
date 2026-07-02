"""Backward-compatible import shim for older paper scripts.

The MiniGrid comparison runner was renamed to ``compare_semnav_minigrid``.
Several paper-facing experiment scripts still import
``scripts.compare_songline_minigrid``. Keep this small shim so those
scripts remain reproducible without duplicating runner logic.
"""

from scripts.compare_semnav_minigrid import *  # noqa: F401,F403
