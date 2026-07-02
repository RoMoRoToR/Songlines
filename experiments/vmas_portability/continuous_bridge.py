"""
Continuous->grid bridge for the VMAS portability test [reviewer #1].

The symbolic peer memory and planner operate on discrete place records with
semantic tags (see multiagent_env.grid_world._observation: a list of
{"xy", "value", "tag"} cells). To run the SAME memory stack on a continuous
VMAS substrate, we discretise continuous (x, y) into coarse grid cells and tag
cells near a water landmark as ``water_source``. This is the only adapter
needed; nothing in the memory/planner changes, which is the point of a
portability test.

Pure functions, no VMAS dependency -> unit-testable on its own.
"""
from typing import Dict, List, Sequence, Tuple

# grid CELL_TAGS values reused from multiagent_env.grid_world
EMPTY, WALL, WATER = 0, 1, 2
TAG_EMPTY, TAG_WATER = "open", "water_source"


def to_cell(xy: Tuple[float, float], cell_size: float, origin: float = 0.0) -> Tuple[int, int]:
    """Snap a continuous coordinate to an integer grid cell."""
    return (int((xy[0] - origin) // cell_size), int((xy[1] - origin) // cell_size))


def build_cells(
    agent_xy: Tuple[float, float],
    water_xys: Sequence[Tuple[float, float]],
    cell_size: float,
    radius: int = 2,
    water_tag_dist: float = 1.0,
    origin: float = 0.0,
) -> List[Dict]:
    """Discretised local observation around ``agent_xy`` in the grid ``cells``
    schema. A cell within ``water_tag_dist`` (continuous units) of any water
    landmark is tagged ``water_source``; otherwise ``open``. ``radius`` is the
    Manhattan cell radius of the local window, matching the grid env."""
    acx, acy = to_cell(agent_xy, cell_size, origin)
    cells: List[Dict] = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if abs(dx) + abs(dy) > radius:
                continue
            cx, cy = acx + dx, acy + dy
            # continuous centre of this cell
            ccx = origin + (cx + 0.5) * cell_size
            ccy = origin + (cy + 0.5) * cell_size
            is_water = any(
                abs(ccx - wx) <= water_tag_dist and abs(ccy - wy) <= water_tag_dist
                for wx, wy in water_xys
            )
            cells.append({
                "xy": (cx, cy),
                "value": WATER if is_water else EMPTY,
                "tag": TAG_WATER if is_water else TAG_EMPTY,
            })
    return cells


def cell_center_xy(cell: Tuple[int, int], cell_size: float, origin: float = 0.0) -> Tuple[float, float]:
    """Continuous centre of a grid cell -- used to turn a materialised (grid)
    target back into a continuous waypoint for the VMAS controller."""
    return (origin + (cell[0] + 0.5) * cell_size, origin + (cell[1] + 0.5) * cell_size)


if __name__ == "__main__":
    # self-test (no VMAS needed)
    cells = build_cells((0.05, 0.0), water_xys=[(0.3, 0.0)], cell_size=0.1, radius=2)
    watered = [c for c in cells if c["tag"] == TAG_WATER]
    print(f"cells={len(cells)}  water-tagged={len(watered)}  sample={cells[0]}")
    assert any(c["tag"] == TAG_WATER for c in cells), "expected a water cell near landmark"
    assert to_cell((0.35, -0.12), 0.1) == (3, -2)
    print("continuous_bridge self-test OK")
