"""Minimal text-navigation environment for the LLM bridge experiment.

Design rationale: ALFWorld and full TextWorld have heavy setup
(graphics-less text engine + benchmark JSON download + cookbook
generators) that would burn a day of integration for an experiment
whose research question is narrower: do Q/R/M/C events emit
non-trivially when the *agent* is LLM-driven on a text substrate?

This env provides a 3-room household map with an apple in the kitchen
and a goal cell on the living-room table. The episode terminates when
the agent brings the apple to the table. We expose:

  • NL observations on each step  (-> LLM tag extractor)
  • a task string                  (-> LLM query former)
  • a fixed action vocabulary      (-> LLM decider)
  • ground-truth landmark coords   (-> Q/R/M/C R/M/C events)

The env is intentionally tiny so a CPU-bound 8B model finishes an
episode in < 1 minute.
"""

from __future__ import annotations

import dataclasses as dc
from typing import Dict, List, Optional, Tuple


GridXY = Tuple[int, int]


@dc.dataclass
class StepInfo:
    aid: str
    new_xy: GridXY
    cell_tag: str
    obs_text: str


@dc.dataclass
class StepResult:
    info: Dict[str, StepInfo]
    all_succeeded: bool


@dc.dataclass
class _Agent:
    aid: str
    x: int
    y: int
    carrying: Optional[str] = None
    success: bool = False


class TextNavEnv:
    """3-room linear map: kitchen (col 0-2), hall (col 3-5), living (col 6-8).

    Cells:
      (1,1) kitchen apple location
      (7,1) living-room table (goal)
      All others: floor in their room

    Actions:
      go_west / go_east / go_north / go_south
      take_apple / put_apple
      look
    """

    ACTIONS = [
        "go_west", "go_east", "go_north", "go_south",
        "take_apple", "put_apple", "look",
    ]

    APPLE_XY: GridXY = (1, 1)
    TABLE_XY: GridXY = (7, 1)

    WIDTH = 9
    HEIGHT = 3

    def __init__(self, step_limit: int = 30, seed: int = 0) -> None:
        self.step_limit = int(step_limit)
        self.seed = int(seed)
        self.tick = 0
        # Single agent for Phase A. Phase B will spawn N.
        self.agents: Dict[str, _Agent] = {"a0": _Agent(aid="a0", x=4, y=1)}
        # Place state — apple at kitchen until picked
        self.apple_present_at_kitchen = True
        self.apple_on_table = False

    # --- env helpers ----------------------------------------------------

    def room_of(self, x: int, y: int) -> str:
        if x <= 2:
            return "kitchen"
        if x <= 5:
            return "hall"
        return "living_room"

    def cell_tag(self, x: int, y: int) -> str:
        if (x, y) == self.APPLE_XY and self.apple_present_at_kitchen:
            return "apple"
        if (x, y) == self.TABLE_XY:
            return "table"
        return self.room_of(x, y)

    def _direction_to(self, ag: _Agent, tgt: GridXY) -> str:
        dx = tgt[0] - ag.x; dy = tgt[1] - ag.y
        bits = []
        if dx > 0: bits.append("east")
        elif dx < 0: bits.append("west")
        if dy > 0: bits.append("south")
        elif dy < 0: bits.append("north")
        return "/".join(bits) if bits else "here"

    def visible_neighborhood(self, aid: str) -> List[Tuple[Tuple[int,int], str]]:
        """Return [(xy, cell_tag), ...] for current cell + 4 neighbors."""
        ag = self.agents[aid]
        out = []
        for dx, dy in [(0,0), (-1,0), (1,0), (0,-1), (0,1)]:
            nx, ny = ag.x + dx, ag.y + dy
            if 0 <= nx < self.WIDTH and 0 <= ny < self.HEIGHT:
                out.append(((nx, ny), self.cell_tag(nx, ny)))
        return out

    def observe_text(self, aid: str) -> str:
        ag = self.agents[aid]
        room = self.room_of(ag.x, ag.y)
        parts = [f"You are in the {room} at column {ag.x} row {ag.y}."]
        # Visible items in current cell or adjacent cells
        if (ag.x, ag.y) == self.APPLE_XY and self.apple_present_at_kitchen:
            parts.append("There is an apple here on the counter — you can take it.")
        if abs(ag.x - self.TABLE_XY[0]) + abs(ag.y - self.TABLE_XY[1]) <= 1:
            if ag.carrying == "apple":
                parts.append("The table is right next to you. To finish the task choose action put_apple.")
            else:
                parts.append("There is a table here — you can put items on it.")
        # Always emit a directional pointer to the current semantic target
        if not ag.success:
            if ag.carrying == "apple":
                d = self._direction_to(ag, self.TABLE_XY)
                parts.append(f"You need to bring the apple to the living-room table ({d}).")
            elif self.apple_present_at_kitchen:
                d = self._direction_to(ag, self.APPLE_XY)
                parts.append(f"The apple is in the kitchen ({d} from here).")
        if ag.carrying == "apple":
            parts.append("You are holding an apple.")
        if self.apple_on_table:
            parts.append("The apple is on the table. Task complete.")
        return " ".join(parts)

    @property
    def task_text(self) -> str:
        return "Bring the apple from the kitchen to the living-room table."

    @property
    def allowed_actions(self) -> List[str]:
        return list(self.ACTIONS)

    def ground_truth_target_xy(self, ag: _Agent) -> GridXY:
        """The real semantic target the planner ought to be heading toward."""
        if ag.carrying == "apple":
            return self.TABLE_XY
        return self.APPLE_XY

    # --- runtime -------------------------------------------------------

    def step(self, actions: Dict[str, str]) -> StepResult:
        info: Dict[str, StepInfo] = {}
        for aid, ag in self.agents.items():
            a = actions.get(aid, "look")
            nx, ny = ag.x, ag.y
            if a == "go_west":
                nx = max(0, ag.x - 1)
            elif a == "go_east":
                nx = min(self.WIDTH - 1, ag.x + 1)
            elif a == "go_north":
                ny = max(0, ag.y - 1)
            elif a == "go_south":
                ny = min(self.HEIGHT - 1, ag.y + 1)
            elif a == "take_apple":
                # Allow take from current cell or any cell within reach 1
                if self.apple_present_at_kitchen:
                    ax, ay = self.APPLE_XY
                    if abs(ag.x - ax) + abs(ag.y - ay) <= 1:
                        ag.carrying = "apple"
                        self.apple_present_at_kitchen = False
            elif a == "put_apple":
                if ag.carrying == "apple":
                    tx, ty = self.TABLE_XY
                    if abs(ag.x - tx) + abs(ag.y - ty) <= 1:
                        ag.carrying = None
                        self.apple_on_table = True
                        ag.success = True
            ag.x, ag.y = nx, ny
            info[aid] = StepInfo(
                aid=aid, new_xy=(ag.x, ag.y),
                cell_tag=self.cell_tag(ag.x, ag.y),
                obs_text=self.observe_text(aid),
            )
        self.tick += 1
        return StepResult(
            info=info,
            all_succeeded=all(a.success for a in self.agents.values()),
        )
