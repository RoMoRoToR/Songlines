import importlib
import math
import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from songline_drive.types import EgoState, LaneContext, RouteContext, SceneState, SignalContext


MINIWORLD_ENV_ALIASES = {
    "Hallway": "MiniWorld-Hallway-v0",
    "TMaze": "MiniWorld-TMaze-v0",
    "WallGap": "MiniWorld-WallGap-v0",
    "FourRooms": "MiniWorld-FourRooms-v0",
}


def is_miniworld_available() -> bool:
    return importlib.util.find_spec("miniworld") is not None


def ensure_miniworld_available():
    if not is_miniworld_available():
        raise RuntimeError(
            "MiniWorld is not installed in this environment. Install the `miniworld` package "
            "before running MiniWorld benchmarks or demos."
        )


def canonical_miniworld_env_id(env_id: str) -> str:
    return MINIWORLD_ENV_ALIASES.get(str(env_id), str(env_id))


def build_miniworld_env(env_id: str, render_mode: str = "rgb_array"):
    ensure_miniworld_available()
    os.environ.setdefault("PYGLET_HEADLESS", "true")
    try:
        import pyglet

        pyglet.options["headless"] = True
    except Exception:
        pass
    import gymnasium as gym
    try:
        import miniworld  # noqa: F401  # Registers env ids.
    except Exception as exc:
        raise RuntimeError(
            "MiniWorld is installed but its rendering backend is not usable in this environment. "
            "A working display or EGL-compatible headless OpenGL backend is required."
        ) from exc

    return gym.make(canonical_miniworld_env_id(env_id), render_mode=render_mode)


def check_miniworld_runtime() -> Tuple[bool, str]:
    if not is_miniworld_available():
        return False, "miniworld package not installed"
    os.environ.setdefault("PYGLET_HEADLESS", "true")
    try:
        import pyglet

        pyglet.options["headless"] = True
    except Exception:
        pass
    try:
        import miniworld  # noqa: F401

        return True, ""
    except Exception as exc:
        return False, str(exc)


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(float(angle)), math.cos(float(angle)))


def _heading_vec(heading_rad: float) -> np.ndarray:
    return np.asarray([math.cos(float(heading_rad)), math.sin(float(heading_rad))], dtype=np.float32)


def _to_xz(pos) -> Optional[np.ndarray]:
    if pos is None:
        return None
    arr = np.asarray(pos, dtype=np.float32).reshape(-1)
    if arr.size >= 3:
        return np.asarray([float(arr[0]), float(arr[2])], dtype=np.float32)
    if arr.size >= 2:
        return np.asarray([float(arr[0]), float(arr[1])], dtype=np.float32)
    return None


def get_agent_position_xz(env) -> np.ndarray:
    unwrapped = env.unwrapped
    agent = getattr(unwrapped, "agent", None)
    if agent is not None:
        pos = _to_xz(getattr(agent, "pos", None))
        if pos is not None:
            return pos
    if hasattr(unwrapped, "agent_pos"):
        pos = _to_xz(getattr(unwrapped, "agent_pos"))
        if pos is not None:
            return pos
    raise RuntimeError("Could not extract MiniWorld agent position.")


def get_agent_heading_rad(env) -> float:
    unwrapped = env.unwrapped
    agent = getattr(unwrapped, "agent", None)
    if agent is not None:
        for attr_name in ("dir", "heading", "yaw", "rot"):
            if hasattr(agent, attr_name):
                raw = getattr(agent, attr_name)
                if np.isscalar(raw):
                    value = float(raw)
                    if abs(value) > 2.0 * math.pi:
                        value = math.radians(value)
                    return _wrap_angle(value)
    for attr_name in ("agent_dir", "agent_heading", "agent_yaw"):
        if hasattr(unwrapped, attr_name):
            raw = getattr(unwrapped, attr_name)
            if np.isscalar(raw):
                value = float(raw)
                if abs(value) > 2.0 * math.pi:
                    value = math.radians(value)
                return _wrap_angle(value)
    return 0.0


def extract_rgb_observation(obs, env=None) -> np.ndarray:
    if isinstance(obs, np.ndarray) and obs.ndim == 3 and obs.shape[-1] in (3, 4):
        rgb = obs[..., :3]
        return np.asarray(rgb, dtype=np.uint8)
    if isinstance(obs, dict):
        for key in ("image", "rgb", "obs"):
            value = obs.get(key)
            if isinstance(value, np.ndarray) and value.ndim == 3 and value.shape[-1] in (3, 4):
                rgb = value[..., :3]
                return np.asarray(rgb, dtype=np.uint8)
    if env is not None:
        frame = env.render()
        if frame is not None:
            return np.asarray(frame[..., :3], dtype=np.uint8)
    return np.zeros((60, 80, 3), dtype=np.uint8)


def miniworld_symbolic_observation(obs, env, grid_h: int = 6, grid_w: int = 8) -> np.ndarray:
    rgb = extract_rgb_observation(obs, env=env).astype(np.float32)
    h, w = rgb.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((1 + grid_h * grid_w,), dtype=np.float32)
    gray = (0.299 * rgb[..., 0]) + (0.587 * rgb[..., 1]) + (0.114 * rgb[..., 2])
    ys = np.linspace(0, h - 1, num=grid_h, dtype=int)
    xs = np.linspace(0, w - 1, num=grid_w, dtype=int)
    patch = gray[np.ix_(ys, xs)]
    quantized = np.clip(np.rint(patch / 32.0), 0, 7).astype(np.int32).reshape(-1)
    heading = get_agent_heading_rad(env)
    heading_bin = int(np.floor(((heading + math.pi) / (2.0 * math.pi)) * 8.0)) % 8
    return np.concatenate([np.asarray([heading_bin], dtype=np.int32), quantized]).astype(np.float32)


def _iter_entities(env) -> Iterable[object]:
    unwrapped = env.unwrapped
    for attr_name in ("entities", "objs", "objects"):
        entities = getattr(unwrapped, attr_name, None)
        if entities:
            return list(entities)
    return []


def _entity_name(entity) -> str:
    return str(entity.__class__.__name__).lower()


def _entity_color(entity) -> str:
    return str(getattr(entity, "color", getattr(entity, "colour", ""))).lower()


def get_goal_position_xz(env) -> Optional[np.ndarray]:
    unwrapped = env.unwrapped
    for attr_name in ("goal_pos", "goal_pos_xz", "box_pos"):
        if hasattr(unwrapped, attr_name):
            pos = _to_xz(getattr(unwrapped, attr_name))
            if pos is not None:
                return pos

    best = None
    best_score = None
    for entity in _iter_entities(env):
        pos = _to_xz(getattr(entity, "pos", None))
        if pos is None:
            continue
        name = _entity_name(entity)
        color = _entity_color(entity)
        score = 0
        if "goal" in name:
            score += 10
        if "box" in name:
            score += 8
        if "ball" in name:
            score += 6
        if color == "red":
            score += 5
        if best_score is None or score > best_score:
            best = pos
            best_score = score
    return None if best is None else np.asarray(best, dtype=np.float32)


def _room_bounds(room) -> Optional[Tuple[float, float, float, float]]:
    options = [
        ("min_x", "max_x", "min_z", "max_z"),
        ("xmin", "xmax", "zmin", "zmax"),
        ("x1", "x2", "z1", "z2"),
    ]
    for names in options:
        if all(hasattr(room, name) for name in names):
            vals = [float(getattr(room, name)) for name in names]
            return vals[0], vals[1], vals[2], vals[3]
    return None


def _iter_rooms(env) -> List[object]:
    unwrapped = env.unwrapped
    rooms = getattr(unwrapped, "rooms", None)
    if rooms:
        return list(rooms)
    return []


def _find_room_for_xz(env, pos_xz: np.ndarray):
    x = float(pos_xz[0])
    z = float(pos_xz[1])
    for room in _iter_rooms(env):
        bounds = _room_bounds(room)
        if bounds is None:
            continue
        min_x, max_x, min_z, max_z = bounds
        if min_x <= x <= max_x and min_z <= z <= max_z:
            return room, bounds
    return None, None


def _goal_visual_proxy(rgb: np.ndarray) -> float:
    if rgb.size == 0:
        return 0.0
    red = rgb[..., 0].astype(np.float32)
    green = rgb[..., 1].astype(np.float32)
    blue = rgb[..., 2].astype(np.float32)
    red_mask = (red > 120.0) & (red > green + 25.0) & (red > blue + 25.0)
    return float(np.mean(red_mask))


class MiniWorldSceneEncoder:
    def __init__(self, view_bins: Tuple[int, int] = (6, 8)):
        self.view_bins = tuple(int(v) for v in view_bins)

    def encode(self, env, obs=None) -> SceneState:
        agent_xz = get_agent_position_xz(env)
        heading = get_agent_heading_rad(env)
        goal_xz = get_goal_position_xz(env)
        rgb = extract_rgb_observation(obs, env=env)

        current_room, bounds = _find_room_for_xz(env, agent_xz)
        room_width = 4.0
        room_depth = 4.0
        if bounds is not None:
            room_width = max(1e-3, float(bounds[1] - bounds[0]))
            room_depth = max(1e-3, float(bounds[3] - bounds[2]))
        room_area = float(room_width * room_depth)
        aspect_ratio = float(max(room_width, room_depth) / max(1e-3, min(room_width, room_depth)))
        corridor_like = float(min(room_width, room_depth) <= 2.6 and aspect_ratio >= 2.0)
        open_space_like = float(min(room_width, room_depth) >= 3.2 and room_area >= 14.0)
        room_center = np.asarray([0.0, 0.0], dtype=np.float32)
        doorway_like = 0.0
        if bounds is not None:
            room_center = np.asarray([(bounds[0] + bounds[1]) / 2.0, (bounds[2] + bounds[3]) / 2.0], dtype=np.float32)
            margin = min(
                abs(float(agent_xz[0]) - float(bounds[0])),
                abs(float(agent_xz[0]) - float(bounds[1])),
                abs(float(agent_xz[1]) - float(bounds[2])),
                abs(float(agent_xz[1]) - float(bounds[3])),
            )
            doorway_like = float(margin <= 0.85 and not corridor_like)

        remaining_distance = 0.0
        goal_alignment = 0.0
        goal_visible = False
        goal_front = 0.0
        goal_dx = 0.0
        goal_dz = 0.0
        heading_alignment = 0.0
        if goal_xz is not None:
            delta = np.asarray(goal_xz, dtype=np.float32) - np.asarray(agent_xz, dtype=np.float32)
            goal_dx = float(delta[0])
            goal_dz = float(delta[1])
            remaining_distance = float(np.linalg.norm(delta))
            heading_vec = _heading_vec(heading)
            norm = float(max(1e-6, np.linalg.norm(delta)))
            heading_alignment = float(np.dot(heading_vec, delta / norm))
            goal_alignment = float(max(-1.0, min(1.0, heading_alignment)) / max(1.0, remaining_distance))
            goal_visible = bool(heading_alignment >= 0.25 or _goal_visual_proxy(rgb) >= 0.01)
            goal_front = float(heading_alignment >= 0.70)

        free_space_score = float(min(1.0, room_area / 25.0))
        free_neighbor_count = 2 if corridor_like > 0.0 else (4 if open_space_like > 0.0 else 3)
        wall_count = 6 if corridor_like > 0.0 else (2 if open_space_like > 0.0 else 4)
        is_room_center = float(open_space_like > 0.0 and np.linalg.norm(agent_xz - room_center) <= 1.4)
        place_is_goal_region = float(goal_visible or (goal_xz is not None and remaining_distance <= 1.75))

        local_patch = miniworld_symbolic_observation(rgb, env=env, grid_h=self.view_bins[0], grid_w=self.view_bins[1])

        return SceneState(
            ego_state=EgoState(
                x=float(agent_xz[0]),
                y=float(agent_xz[1]),
                yaw=float(heading),
                speed=0.0,
                yaw_rate=0.0,
                acceleration=0.0,
            ),
            route_context=RouteContext(
                route_id=None,
                target_lane_id=None,
                remaining_distance=float(remaining_distance),
                goal_alignment=float(goal_alignment),
                goal_xy=None if goal_xz is None else (int(round(float(goal_xz[0]))), int(round(float(goal_xz[1])))),
                goal_visible=bool(goal_visible),
            ),
            lane_context=LaneContext(
                current_lane_id=None,
                left_lane_id=None,
                right_lane_id=None,
                lane_curvature=0.0,
                drivable_width=float(min(room_width, room_depth)),
            ),
            agents=[],
            signals=SignalContext(
                traffic_light_state=None,
                stop_line_distance=None,
                crosswalk_distance=None,
            ),
            free_space_score=float(free_space_score),
            risk_features={
                "hazard_near": 0.0,
                "hazard_front": 0.0,
                "hazard_left": 0.0,
                "hazard_right": 0.0,
                "wall_near": float(corridor_like > 0.0),
                "goal_visible": float(goal_visible),
                "goal_front": float(goal_front),
                "front_cell": 0.0,
                "left_cell": 0.0,
                "right_cell": 0.0,
                "back_cell": 0.0,
                "front_safe": 1.0,
                "lateral_blocked": float(corridor_like > 0.0),
                "lateral_hazard": 0.0,
                "asymmetric_gap_channel": 0.0,
                "hazard_asymmetric_gap_channel": 0.0,
                "narrow_safe_channel": 0.0,
                "wall_count": float(wall_count),
                "free_neighbor_count": float(free_neighbor_count),
                "lava_neighbor_count": 0.0,
                "wall_neighbor_count": float(max(0, wall_count - free_neighbor_count)),
                "corridor_like": float(corridor_like),
                "doorway_like": float(doorway_like),
                "open_space_like": float(open_space_like),
                "place_is_safe_zone": float(open_space_like > 0.0 or doorway_like > 0.0),
                "place_is_goal_region": float(place_is_goal_region),
                "place_is_hazard_edge": 0.0,
                "place_is_room_center": float(is_room_center),
                "place_is_corridor": float(corridor_like),
                "place_is_hazard_recovery_route": 0.0,
                "water_visible": 0.0,
                "water_pattern_match": 0.0,
                "water_accessible": 0.0,
                "water_neighbor_context": 0.0,
                "water_confidence_local": 0.0,
                "rest_visible": 0.0,
                "rest_pattern_match": 0.0,
                "rest_accessible": 0.0,
                "rest_neighbor_context": 0.0,
                "rest_confidence_local": 0.0,
                "goal_heading_alignment": float(heading_alignment),
                "goal_dx": float(goal_dx),
                "goal_dy": float(goal_dz),
            },
            local_patch=tuple(int(v) for v in local_patch.reshape(-1)),
        )


class MiniWorldTrajectoryPlanner:
    def __init__(self, turn_threshold_rad: float = 0.30, forward_distance_threshold: float = 0.55):
        self.turn_threshold_rad = float(turn_threshold_rad)
        self.forward_distance_threshold = float(forward_distance_threshold)

    def reset(self):
        return None

    def _action_value(self, env, action_name: str, default: int) -> int:
        actions = getattr(env.unwrapped, "actions", getattr(env, "actions", None))
        if actions is not None and hasattr(actions, action_name):
            raw = getattr(actions, action_name)
            try:
                return int(raw)
            except Exception:
                return int(getattr(raw, "value", default))
        return int(default)

    def next_action(self, env, maneuver_command):
        waypoint_xy = maneuver_command.get("waypoint_xy")
        if waypoint_xy is None:
            return self._action_value(env, "move_forward", 2)
        target_xz = np.asarray(waypoint_xy, dtype=np.float32)
        agent_xz = get_agent_position_xz(env)
        heading = get_agent_heading_rad(env)
        delta = target_xz - agent_xz
        distance = float(np.linalg.norm(delta))
        if distance <= self.forward_distance_threshold:
            return self._action_value(env, "move_forward", 2)
        desired_heading = math.atan2(float(delta[1]), float(delta[0]))
        err = _wrap_angle(desired_heading - heading)
        if abs(err) <= self.turn_threshold_rad:
            return self._action_value(env, "move_forward", 2)
        if err > 0.0:
            return self._action_value(env, "turn_left", 0)
        return self._action_value(env, "turn_right", 1)
