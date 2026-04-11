from typing import Tuple

import numpy as np

from songline_drive.types import EgoState, LaneContext, RouteContext, SceneState, SignalContext


class MiniGridSceneEncoder:
    def __init__(self, radius: int = 1):
        self.radius = radius

    def encode(self, env) -> SceneState:
        unwrapped = env.unwrapped
        agent_pos = np.asarray(unwrapped.agent_pos, dtype=np.int32)
        agent_dir = int(unwrapped.agent_dir)
        goal_xy = self._get_goal_position(env)
        patch = self._local_patch(env, radius=self.radius)
        cardinal_neighbors = self._cardinal_neighbors(env)
        front_cell = self._front_cell(env)
        left_cell, right_cell, back_cell = self._relative_cells(env)

        hazard_near = float(np.any(patch == 3))
        hazard_front = float(front_cell == 3)
        hazard_left = float(left_cell == 3)
        hazard_right = float(right_cell == 3)
        wall_near = float(np.any(patch == 1) or np.any(patch == 99))
        goal_visible = bool(np.any(patch == 2))
        goal_front = float(front_cell == 2)
        front_safe = float(front_cell in (0, 2))
        lateral_blocked = float(left_cell in (1, 3, 99) and right_cell in (1, 3, 99))
        lateral_hazard = float(left_cell == 3 or right_cell == 3)
        asymmetric_gap_channel = float(
            front_safe > 0.0 and (left_cell in (1, 3, 99) or right_cell in (1, 3, 99))
        )
        hazard_asymmetric_gap_channel = float(
            front_safe > 0.0 and (hazard_left > 0.0 or hazard_right > 0.0)
        )
        free_cells = float(np.sum(patch == 0))
        total_cells = float(max(1, patch.size))
        free_space_score = free_cells / total_cells
        wall_count = int(np.sum((patch == 1) | (patch == 99)))
        free_neighbor_count = int(np.sum(cardinal_neighbors == 0))
        lava_neighbor_count = int(np.sum(cardinal_neighbors == 3))
        wall_neighbor_count = int(np.sum((cardinal_neighbors == 1) | (cardinal_neighbors == 99)))
        opposite_open = self._opposite_open_count(cardinal_neighbors)
        corridor_like = float(free_neighbor_count <= 2 and wall_count >= 4)
        doorway_like = float(free_neighbor_count >= 2 and opposite_open >= 1 and wall_count >= 2)
        open_space_like = float(free_neighbor_count >= 3 and wall_count <= 3)
        narrow_safe_channel = float(
            front_safe > 0.0 and (lateral_blocked > 0.0 or hazard_asymmetric_gap_channel > 0.0)
        )

        remaining_distance = 0.0
        goal_alignment = 0.0
        goal_dx = 0.0
        goal_dy = 0.0
        if goal_xy is not None:
            remaining_distance = float(np.abs(agent_pos - goal_xy).sum())
            goal_alignment = -remaining_distance
            goal_dx = float(goal_xy[0] - agent_pos[0])
            goal_dy = float(goal_xy[1] - agent_pos[1])
        goal_heading_alignment = self._goal_heading_alignment(agent_dir, goal_dx, goal_dy)
        is_safe_zone = float(
            front_safe == 1.0
            and hazard_front == 0.0
            and hazard_near == 0.0
        )
        is_goal_region = float(
            goal_visible
            or goal_front == 1.0
            or (goal_xy is not None and remaining_distance <= 2.0)
        )
        is_hazard_edge = float(hazard_near == 1.0 and front_safe == 1.0)
        is_room_center = float(open_space_like == 1.0 and free_neighbor_count >= 3)
        is_corridor = float(corridor_like == 1.0)
        is_hazard_recovery_route = float(
            front_safe == 1.0
            and hazard_near == 1.0
            and goal_heading_alignment >= 0.3
            and (
                narrow_safe_channel == 1.0
                or hazard_asymmetric_gap_channel == 1.0
                or lateral_hazard == 1.0
            )
        )

        return SceneState(
            ego_state=EgoState(
                x=float(agent_pos[0]),
                y=float(agent_pos[1]),
                yaw=float(agent_dir),
                speed=0.0,
                yaw_rate=0.0,
                acceleration=0.0,
            ),
            route_context=RouteContext(
                route_id=None,
                target_lane_id=None,
                remaining_distance=remaining_distance,
                goal_alignment=goal_alignment,
                goal_xy=None if goal_xy is None else (int(goal_xy[0]), int(goal_xy[1])),
                goal_visible=goal_visible,
            ),
            lane_context=LaneContext(
                current_lane_id=None,
                left_lane_id=None,
                right_lane_id=None,
                lane_curvature=0.0,
                drivable_width=float(self.radius * 2 + 1),
            ),
            agents=[],
            signals=SignalContext(
                traffic_light_state=None,
                stop_line_distance=None,
                crosswalk_distance=None,
            ),
            free_space_score=free_space_score,
            risk_features={
                "hazard_near": hazard_near,
                "hazard_front": hazard_front,
                "hazard_left": hazard_left,
                "hazard_right": hazard_right,
                "wall_near": wall_near,
                "goal_visible": float(goal_visible),
                "goal_front": goal_front,
                "front_cell": float(front_cell),
                "left_cell": float(left_cell),
                "right_cell": float(right_cell),
                "back_cell": float(back_cell),
                "front_safe": front_safe,
                "lateral_blocked": lateral_blocked,
                "lateral_hazard": lateral_hazard,
                "asymmetric_gap_channel": asymmetric_gap_channel,
                "hazard_asymmetric_gap_channel": hazard_asymmetric_gap_channel,
                "narrow_safe_channel": narrow_safe_channel,
                "wall_count": float(wall_count),
                "free_neighbor_count": float(free_neighbor_count),
                "lava_neighbor_count": float(lava_neighbor_count),
                "wall_neighbor_count": float(wall_neighbor_count),
                "corridor_like": corridor_like,
                "doorway_like": doorway_like,
                "open_space_like": open_space_like,
                "place_is_safe_zone": is_safe_zone,
                "place_is_goal_region": is_goal_region,
                "place_is_hazard_edge": is_hazard_edge,
                "place_is_room_center": is_room_center,
                "place_is_corridor": is_corridor,
                "place_is_hazard_recovery_route": is_hazard_recovery_route,
                "goal_heading_alignment": float(goal_heading_alignment),
                "goal_dx": goal_dx,
                "goal_dy": goal_dy,
            },
            local_patch=tuple(int(v) for v in patch.reshape(-1)),
        )

    def _local_patch(self, env, radius: int) -> np.ndarray:
        unwrapped = env.unwrapped
        ax, ay = unwrapped.agent_pos
        grid = unwrapped.grid

        patch = []
        for dy in range(-radius, radius + 1):
            row = []
            for dx in range(-radius, radius + 1):
                x = ax + dx
                y = ay + dy
                if x < 0 or y < 0 or x >= grid.width or y >= grid.height:
                    row.append(99)
                    continue
                cell = grid.get(x, y)
                if cell is None:
                    row.append(0)
                else:
                    row.append(self._encode_cell(cell.type))
            patch.append(row)
        return np.asarray(patch, dtype=np.int32)

    def _encode_cell(self, cell_type: str) -> int:
        if cell_type == "wall":
            return 1
        if cell_type == "goal":
            return 2
        if cell_type == "lava":
            return 3
        if cell_type == "door":
            return 4
        if cell_type == "key":
            return 5
        if cell_type == "ball":
            return 6
        if cell_type == "box":
            return 7
        return 8

    def _cardinal_neighbors(self, env) -> np.ndarray:
        unwrapped = env.unwrapped
        ax, ay = unwrapped.agent_pos
        offsets = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        values = []
        for dx, dy in offsets:
            values.append(self._cell_code(env, ax + dx, ay + dy))
        return np.asarray(values, dtype=np.int32)

    def _front_cell(self, env) -> int:
        unwrapped = env.unwrapped
        ax, ay = unwrapped.agent_pos
        dx, dy = self._dir_to_delta(int(unwrapped.agent_dir))
        return int(self._cell_code(env, ax + dx, ay + dy))

    def _relative_cells(self, env):
        unwrapped = env.unwrapped
        ax, ay = unwrapped.agent_pos
        agent_dir = int(unwrapped.agent_dir)
        front_dx, front_dy = self._dir_to_delta(agent_dir)
        left_dx, left_dy = self._dir_to_delta((agent_dir - 1) % 4)
        right_dx, right_dy = self._dir_to_delta((agent_dir + 1) % 4)
        back_dx, back_dy = self._dir_to_delta((agent_dir + 2) % 4)
        left_cell = self._cell_code(env, ax + left_dx, ay + left_dy)
        right_cell = self._cell_code(env, ax + right_dx, ay + right_dy)
        back_cell = self._cell_code(env, ax + back_dx, ay + back_dy)
        _ = (front_dx, front_dy)
        return int(left_cell), int(right_cell), int(back_cell)

    def _cell_code(self, env, x: int, y: int) -> int:
        grid = env.unwrapped.grid
        if x < 0 or y < 0 or x >= grid.width or y >= grid.height:
            return 99
        cell = grid.get(x, y)
        if cell is None:
            return 0
        return self._encode_cell(cell.type)

    def _opposite_open_count(self, neighbors: np.ndarray) -> int:
        east, south, west, north = [int(v) for v in neighbors]
        count = 0
        if east == 0 and west == 0:
            count += 1
        if north == 0 and south == 0:
            count += 1
        return count

    def _dir_to_delta(self, agent_dir: int):
        dir_to_delta = {
            0: (1, 0),
            1: (0, 1),
            2: (-1, 0),
            3: (0, -1),
        }
        return dir_to_delta[int(agent_dir)]

    def _goal_heading_alignment(self, agent_dir: int, goal_dx: float, goal_dy: float) -> float:
        if goal_dx == 0.0 and goal_dy == 0.0:
            return 1.0
        direction_vectors = {
            0: np.array([1.0, 0.0], dtype=np.float64),
            1: np.array([0.0, 1.0], dtype=np.float64),
            2: np.array([-1.0, 0.0], dtype=np.float64),
            3: np.array([0.0, -1.0], dtype=np.float64),
        }
        goal_vec = np.array([goal_dx, goal_dy], dtype=np.float64)
        goal_norm = np.linalg.norm(goal_vec)
        if goal_norm <= 1e-6:
            return 1.0
        heading = direction_vectors[int(agent_dir)]
        return float(np.dot(heading, goal_vec / goal_norm))

    def _get_goal_position(self, env) -> Tuple[int, int]:
        grid = env.unwrapped.grid
        for y in range(grid.height):
            for x in range(grid.width):
                cell = grid.get(x, y)
                if cell is not None and cell.type == "goal":
                    return np.array([x, y], dtype=np.int32)
        return None
