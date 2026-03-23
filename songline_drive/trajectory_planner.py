class TrajectoryPlanner:
    def __init__(self, commit_to_corridor=False):
        self.commit_to_corridor = bool(commit_to_corridor)
        self.corridor_commit_dir = None
        self.corridor_commit_steps = 0

    def reset(self):
        self.corridor_commit_dir = None
        self.corridor_commit_steps = 0

    def next_action(self, env, maneuver_command):
        if maneuver_command.get("command_type") == "resume_to_goal":
            if maneuver_command.get("final_exit_mode") == "v1":
                return self._follow_resume_to_goal_v1(env, maneuver_command)
            return self._follow_resume_to_goal(env, maneuver_command)
        if maneuver_command.get("command_type") == "final_exit_maneuver":
            return self._follow_final_exit(env, maneuver_command)
        if maneuver_command.get("command_type") in {
            "committed_corridor_cross",
            "exit_hazard_commit",
        }:
            return self._follow_external_commit(env, maneuver_command)
        waypoint_xy = maneuver_command.get("waypoint_xy")
        if waypoint_xy is None:
            return 2
        if maneuver_command.get("command_type") == "align_and_cross_gap":
            return self._choose_gap_align_action(env, waypoint_xy)
        self._clear_commit()
        return self._choose_action_toward(env, waypoint_xy)

    def _choose_gap_align_action(self, env, target_xy):
        unwrapped = env.unwrapped
        ax, ay = unwrapped.agent_pos
        agent_dir = int(unwrapped.agent_dir)

        if self.commit_to_corridor and self._commit_active():
            return self._follow_committed_corridor(env, agent_dir)

        if self._is_safe_front(env, agent_dir):
            alignment = self._heading_alignment(ax, ay, agent_dir, target_xy)
            if alignment >= 0.35:
                if self.commit_to_corridor and self._is_corridor_commit_candidate(env, agent_dir):
                    self._start_commit(agent_dir)
                return 2

        candidate_actions = [0, 1]
        best_action = 0
        best_score = None
        for action in candidate_actions:
            candidate_dir = self._apply_turn(agent_dir, action)
            front_safe = 1.0 if self._is_safe_front(env, candidate_dir) else 0.0
            lateral_hazard = 1.0 if self._lateral_hazard(env, candidate_dir) else 0.0
            lateral_blocked = 1.0 if self._lateral_blocked(env, candidate_dir) else 0.0
            alignment = self._heading_alignment(ax, ay, candidate_dir, target_xy)
            score = (2.0 * front_safe) + (1.25 * lateral_hazard) + (0.5 * lateral_blocked) + alignment
            if best_score is None or score > best_score:
                best_score = score
                best_action = action
        return best_action

    def _choose_action_toward(self, env, target_xy):
        unwrapped = env.unwrapped
        ax, ay = unwrapped.agent_pos
        agent_dir = unwrapped.agent_dir

        tx, ty = int(target_xy[0]), int(target_xy[1])
        dx = tx - ax
        dy = ty - ay

        if dx == 0 and dy == 0:
            return 2

        if abs(dx) >= abs(dy):
            desired_dir = 0 if dx > 0 else 2
        else:
            desired_dir = 1 if dy > 0 else 3

        if desired_dir == agent_dir:
            return 2

        right_turns = (desired_dir - agent_dir) % 4
        left_turns = (agent_dir - desired_dir) % 4

        if left_turns <= right_turns:
            return 0
        return 1

    def _apply_turn(self, agent_dir, action):
        if action == 0:
            return (agent_dir - 1) % 4
        if action == 1:
            return (agent_dir + 1) % 4
        return agent_dir

    def _dir_to_delta(self, agent_dir):
        return {
            0: (1, 0),
            1: (0, 1),
            2: (-1, 0),
            3: (0, -1),
        }[int(agent_dir)]

    def _cell_code(self, env, x, y):
        grid = env.unwrapped.grid
        if x < 0 or y < 0 or x >= grid.width or y >= grid.height:
            return 99
        cell = grid.get(x, y)
        if cell is None:
            return 0
        if cell.type == "wall":
            return 1
        if cell.type == "goal":
            return 2
        if cell.type == "lava":
            return 3
        if cell.type == "door":
            return 4
        return 8

    def _is_safe_front(self, env, agent_dir):
        ax, ay = env.unwrapped.agent_pos
        dx, dy = self._dir_to_delta(agent_dir)
        return self._cell_code(env, ax + dx, ay + dy) in (0, 2)

    def _lateral_hazard(self, env, agent_dir):
        ax, ay = env.unwrapped.agent_pos
        left_dir = (agent_dir - 1) % 4
        right_dir = (agent_dir + 1) % 4
        ldx, ldy = self._dir_to_delta(left_dir)
        rdx, rdy = self._dir_to_delta(right_dir)
        left_cell = self._cell_code(env, ax + ldx, ay + ldy)
        right_cell = self._cell_code(env, ax + rdx, ay + rdy)
        return left_cell == 3 or right_cell == 3

    def _lateral_blocked(self, env, agent_dir):
        ax, ay = env.unwrapped.agent_pos
        left_dir = (agent_dir - 1) % 4
        right_dir = (agent_dir + 1) % 4
        ldx, ldy = self._dir_to_delta(left_dir)
        rdx, rdy = self._dir_to_delta(right_dir)
        left_cell = self._cell_code(env, ax + ldx, ay + ldy)
        right_cell = self._cell_code(env, ax + rdx, ay + rdy)
        return left_cell in (1, 3, 99) or right_cell in (1, 3, 99)

    def _heading_alignment(self, ax, ay, agent_dir, target_xy):
        tx, ty = int(target_xy[0]), int(target_xy[1])
        dx = tx - ax
        dy = ty - ay
        if dx == 0 and dy == 0:
            return 1.0
        dir_dx, dir_dy = self._dir_to_delta(agent_dir)
        norm = (dx * dx + dy * dy) ** 0.5
        if norm <= 1e-6:
            return 1.0
        return ((dir_dx * dx) + (dir_dy * dy)) / norm

    def _is_corridor_commit_candidate(self, env, agent_dir):
        return self._is_safe_front(env, agent_dir) and self._lateral_hazard(env, agent_dir)

    def _start_commit(self, agent_dir, steps=3):
        self.corridor_commit_dir = int(agent_dir)
        self.corridor_commit_steps = int(steps)

    def _commit_active(self):
        return self.corridor_commit_dir is not None and self.corridor_commit_steps > 0

    def _clear_commit(self):
        self.corridor_commit_dir = None
        self.corridor_commit_steps = 0

    def _follow_committed_corridor(self, env, agent_dir):
        if not self._commit_active():
            return 2
        commit_dir = int(self.corridor_commit_dir)
        if agent_dir != commit_dir:
            right_turns = (commit_dir - agent_dir) % 4
            left_turns = (agent_dir - commit_dir) % 4
            return 0 if left_turns <= right_turns else 1
        if self._is_safe_front(env, agent_dir):
            self.corridor_commit_steps -= 1
            if self.corridor_commit_steps <= 0:
                self._clear_commit()
            return 2
        self._clear_commit()
        return 0

    def _follow_external_commit(self, env, maneuver_command):
        commit_dir = maneuver_command.get("hazard_commit_dir")
        if commit_dir is None:
            return 2
        command_type = maneuver_command.get("command_type")
        waypoint_xy = maneuver_command.get("waypoint_xy")
        agent_dir = int(env.unwrapped.agent_dir)
        commit_dir = int(commit_dir)
        if agent_dir != commit_dir:
            right_turns = (commit_dir - agent_dir) % 4
            left_turns = (agent_dir - commit_dir) % 4
            return 0 if left_turns <= right_turns else 1
        if self._is_safe_front(env, agent_dir):
            return 2
        left_dir = (commit_dir - 1) % 4
        right_dir = (commit_dir + 1) % 4
        if command_type == "exit_hazard_commit" and waypoint_xy is not None:
            return self._choose_exit_turn(env, waypoint_xy, left_dir, right_dir)
        left_safe = self._is_safe_front(env, left_dir)
        right_safe = self._is_safe_front(env, right_dir)
        left_hazard = self._lateral_hazard(env, left_dir)
        right_hazard = self._lateral_hazard(env, right_dir)
        if left_safe and not right_safe:
            return 0
        if right_safe and not left_safe:
            return 1
        if left_safe and right_safe:
            return 0 if left_hazard >= right_hazard else 1
        return 0

    def _choose_exit_turn(self, env, target_xy, left_dir, right_dir):
        ax, ay = env.unwrapped.agent_pos
        candidates = [(0, left_dir), (1, right_dir)]
        best_action = 0
        best_score = None
        for action, candidate_dir in candidates:
            front_safe = 1.0 if self._is_safe_front(env, candidate_dir) else 0.0
            alignment = self._heading_alignment(ax, ay, candidate_dir, target_xy)
            lateral_hazard = 1.0 if self._lateral_hazard(env, candidate_dir) else 0.0
            lateral_blocked = 1.0 if self._lateral_blocked(env, candidate_dir) else 0.0
            score = (2.5 * front_safe) + (1.5 * alignment) - (0.75 * lateral_hazard) - (0.25 * lateral_blocked)
            if best_score is None or score > best_score:
                best_score = score
                best_action = action
        return best_action

    def _follow_final_exit(self, env, maneuver_command):
        waypoint_xy = maneuver_command.get("waypoint_xy")
        if waypoint_xy is None:
            return 2
        unwrapped = env.unwrapped
        ax, ay = unwrapped.agent_pos
        agent_dir = int(unwrapped.agent_dir)

        if self._is_safe_front(env, agent_dir):
            alignment = self._heading_alignment(ax, ay, agent_dir, waypoint_xy)
            if alignment >= 0.35:
                return 2

        best_action = 0
        best_score = None
        for action, candidate_dir in [(0, (agent_dir - 1) % 4), (1, (agent_dir + 1) % 4)]:
            front_safe = 1.0 if self._is_safe_front(env, candidate_dir) else 0.0
            alignment = self._heading_alignment(ax, ay, candidate_dir, waypoint_xy)
            lateral_hazard = 1.0 if self._lateral_hazard(env, candidate_dir) else 0.0
            lateral_blocked = 1.0 if self._lateral_blocked(env, candidate_dir) else 0.0
            score = (2.5 * front_safe) + (2.25 * alignment) - (0.75 * lateral_hazard) - (0.25 * lateral_blocked)
            if best_score is None or score > best_score:
                best_score = score
                best_action = action
        return best_action

    def _follow_resume_to_goal(self, env, maneuver_command):
        waypoint_xy = maneuver_command.get("waypoint_xy")
        if waypoint_xy is None:
            return 2
        agent_dir = int(env.unwrapped.agent_dir)
        if self._is_safe_front(env, agent_dir):
            return 2
        return self._choose_action_toward(env, waypoint_xy)

    def _follow_resume_to_goal_v1(self, env, maneuver_command):
        waypoint_xy = maneuver_command.get("waypoint_xy")
        if waypoint_xy is None:
            return 2
        unwrapped = env.unwrapped
        ax, ay = unwrapped.agent_pos
        agent_dir = int(unwrapped.agent_dir)

        if self._is_safe_front(env, agent_dir):
            alignment = self._heading_alignment(ax, ay, agent_dir, waypoint_xy)
            if alignment >= 0.15:
                return 2

        if self._post_exit_forward_ok(env, waypoint_xy, agent_dir):
            return 2

        best_action = 0
        best_score = None
        for action, candidate_dir in [(0, (agent_dir - 1) % 4), (1, (agent_dir + 1) % 4)]:
            front_safe = 1.0 if self._is_safe_front(env, candidate_dir) else 0.0
            alignment = self._heading_alignment(ax, ay, candidate_dir, waypoint_xy)
            lateral_hazard = 1.0 if self._lateral_hazard(env, candidate_dir) else 0.0
            lateral_blocked = 1.0 if self._lateral_blocked(env, candidate_dir) else 0.0
            score = (3.0 * front_safe) + (1.5 * alignment) - (1.0 * lateral_hazard) - (0.25 * lateral_blocked)
            if best_score is None or score > best_score:
                best_score = score
                best_action = action
        return best_action

    def _post_exit_forward_ok(self, env, target_xy, agent_dir):
        ax, ay = env.unwrapped.agent_pos
        alignment = self._heading_alignment(ax, ay, agent_dir, target_xy)
        if alignment < 0.9:
            return False

        dx, dy = self._dir_to_delta(agent_dir)
        front_code = self._cell_code(env, ax + dx, ay + dy)
        if front_code != 3:
            return False

        left_dir = (agent_dir - 1) % 4
        right_dir = (agent_dir + 1) % 4
        ldx, ldy = self._dir_to_delta(left_dir)
        rdx, rdy = self._dir_to_delta(right_dir)
        left_code = self._cell_code(env, ax + ldx, ay + ldy)
        right_code = self._cell_code(env, ax + rdx, ay + rdy)

        left_open_right_blocked = left_code in (0, 2) and right_code in (1, 3, 99)
        right_open_left_blocked = right_code in (0, 2) and left_code in (1, 3, 99)
        return left_open_right_blocked or right_open_left_blocked

    def debug_state(self):
        return {
            "corridor_commit_active": int(self._commit_active()),
            "corridor_commit_dir": self.corridor_commit_dir,
            "corridor_commit_steps_left": int(self.corridor_commit_steps),
        }

    def should_preserve_commit(self, maneuver_command, hazard_phase_active):
        if not self.commit_to_corridor:
            return False
        if maneuver_command is None:
            return False
        if maneuver_command.get("command_type") != "align_and_cross_gap":
            return False
        return bool(hazard_phase_active and self._commit_active())
