"""Append-only external memory for the neural-explicit baseline.

Deliberately NOT the symbolic songline pipeline: the store is a flat
append-only list of raw observations (position + semantic tag + tick +
observer).  There is no symbolic filtering, ranking, or consolidation --
ALL selection over this store is done by the learned retriever head.

Write policy (fixed, not learned -- it is part of the environment
interface, mirroring "agents record what they see"):
  * every tagged cell (water_source / hazard_edge / goal_region) inside an
    agent's observation radius is appended the first time ANY agent sees it;
  * the agent's own current cell is appended as a ``safe_neutral``
    breadcrumb the first time it is visited -- these act as distractor
    entries so retrieval is a real selection problem.

Entries are deduplicated on (x, y, tag): append-only semantics, nothing is
ever overwritten or removed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

import numpy as np

# Query vocabulary == tag vocabulary (+ explicit NO_QUERY token).
TAGS = ["water_source", "hazard_edge", "goal_region", "safe_neutral"]
TAG_TO_IDX = {t: i for i, t in enumerate(TAGS)}
NO_QUERY = len(TAGS)  # query-head token meaning "emit no query this tick"
QUERY_VOCAB = TAGS + ["NO_QUERY"]

# Entry feature layout (dim 9):
#   tag one-hot (4) | xy normalised (2) | xy relative to reader (2) | age (1)
ENTRY_FEAT_DIM = 9


@dataclass
class MemoryEntry:
    x: int
    y: int
    tag: str
    tick: int
    observer: str


class AppendOnlyMemory:
    """Shared episode-scoped append-only observation store."""

    def __init__(self) -> None:
        self.entries: List[MemoryEntry] = []
        self._seen: Set[Tuple[int, int, str]] = set()

    def __len__(self) -> int:
        return len(self.entries)

    def write(self, x: int, y: int, tag: str, tick: int, observer: str) -> bool:
        key = (x, y, tag)
        if key in self._seen:
            return False
        self._seen.add(key)
        self.entries.append(MemoryEntry(x, y, tag, tick, observer))
        return True

    def write_agent_observation(self, env, agent_id: str, tick: int) -> int:
        """Record what ``agent_id`` currently sees.  Returns #new entries."""
        ag = env.agents[agent_id]
        n_new = 0
        r = env.observation_radius
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if abs(dx) + abs(dy) > r:
                    continue
                cx, cy = ag.x + dx, ag.y + dy
                if not (0 <= cx < env.width and 0 <= cy < env.height):
                    continue
                tag = env.cell_tag(cx, cy)
                if tag in ("water_source", "hazard_edge", "goal_region"):
                    n_new += int(self.write(cx, cy, tag, tick, agent_id))
        # Breadcrumb distractor: own current cell if plain.
        if env.cell_tag(ag.x, ag.y) == "safe_neutral":
            n_new += int(self.write(ag.x, ag.y, "safe_neutral", tick, agent_id))
        return n_new

    def features(self, reader_xy: Tuple[int, int], tick: int,
                 width: int, height: int, step_limit: int) -> np.ndarray:
        """Feature matrix [E, ENTRY_FEAT_DIM] relative to the reading agent."""
        E = len(self.entries)
        out = np.zeros((E, ENTRY_FEAT_DIM), dtype=np.float32)
        ax, ay = reader_xy
        for i, e in enumerate(self.entries):
            out[i, TAG_TO_IDX[e.tag]] = 1.0
            out[i, 4] = e.x / max(1, width - 1)
            out[i, 5] = e.y / max(1, height - 1)
            out[i, 6] = (e.x - ax) / max(1, width - 1)
            out[i, 7] = (e.y - ay) / max(1, height - 1)
            out[i, 8] = (tick - e.tick) / max(1, step_limit)
        return out
