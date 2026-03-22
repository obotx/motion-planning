import sys, os, time, math, threading
import numpy as np
import mujoco

from ompl import base as ob
from ompl import geometric as og

FLOOR_X      = (0.0, 8.0)
FLOOR_Y      = (-8.0, 0.0)
GRID_RES     = 0.1
ROBOT_RADIUS = 0.35
OBSTACLE_RECTS = []

ROBOT_START  = (3.0, -6.0)

WAYPOINT_REACH_DIST = 0.50
GOAL_REACH_DIST     = 0.55
WAYPOINT_TIMEOUT    = 180.0


class OmplPlanner:

    def __init__(self, model, data, base_body_name="base_footprint",
                 x_range=FLOOR_X, y_range=FLOOR_Y, solve_time=2.0):
        self.model          = model
        self.data           = data
        self.base_id        = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body_name)
        self.x_range        = x_range
        self.y_range        = y_range
        self.solve_time     = solve_time

        self.base_jnt_qposadr = self._find_base_qposadr()
        self._space, self._si = self._build_space()

        print(f"[OMPL] Planner ready. Base qposadr={self.base_jnt_qposadr}, "
              f"body_id={self.base_id}")

    def _find_base_qposadr(self):
        for j in range(self.model.njnt):
            body_id = self.model.jnt_bodyid[j]
            jtype   = self.model.jnt_type[j]
            if body_id == self.base_id and jtype == 0:
                return self.model.jnt_qposadr[j]
        for j in range(self.model.njnt):
            if self.model.jnt_type[j] == 0:
                return self.model.jnt_qposadr[j]
        raise RuntimeError("Could not find free joint for base_footprint")

    def _set_base_pose(self, x, y, yaw):
        adr = self.base_jnt_qposadr
        self.data.qpos[adr + 0] = x
        self.data.qpos[adr + 1] = y
        self.data.qpos[adr + 3] = math.cos(yaw / 2.0)
        self.data.qpos[adr + 4] = 0.0
        self.data.qpos[adr + 5] = 0.0
        self.data.qpos[adr + 6] = math.sin(yaw / 2.0)

    def _is_state_valid(self, state):
        x   = state[0]
        y   = state[1]
        yaw = state[2]

        self._set_base_pose(x, y, yaw)
        mujoco.mj_forward(self.model, self.data)

        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1 = c.geom1
            g2 = c.geom2
            b1 = self.model.geom_bodyid[g1]
            b2 = self.model.geom_bodyid[g2]
            if self._is_base_body(b1) or self._is_base_body(b2):
                return False
        return True

    def _is_base_body(self, body_id):
        bid = body_id
        for _ in range(5):
            if bid == self.base_id:
                return True
            bid = self.model.body_parentid[bid]
            if bid == 0:
                break
        return False

    def _build_space(self):
        space = ob.SE2StateSpace()

        bounds = ob.RealVectorBounds(2)
        bounds.setLow(0,  self.x_range[0])
        bounds.setHigh(0, self.x_range[1])
        bounds.setLow(1,  self.y_range[0])
        bounds.setHigh(1, self.y_range[1])
        space.setBounds(bounds)

        si = ob.SpaceInformation(space)
        si.setStateValidityChecker(ob.StateValidityCheckerFn(self._is_state_valid))
        si.setStateValidityCheckingResolution(0.02)
        si.setup()

        return space, si

    def plan(self, start_xy, goal_xy, start_yaw=0.0, goal_yaw=None):
        if goal_yaw is None:
            dx = goal_xy[0] - start_xy[0]
            dy = goal_xy[1] - start_xy[1]
            goal_yaw = math.atan2(dy, dx)

        start = ob.State(self._space)
        start().setX(start_xy[0])
        start().setY(start_xy[1])
        start().setYaw(start_yaw)

        goal = ob.State(self._space)
        goal().setX(goal_xy[0])
        goal().setY(goal_xy[1])
        goal().setYaw(goal_yaw)

        if not self._si.isValid(start()):
            print(f"[OMPL] Start {start_xy} is in collision — nudging")
            start = self._find_valid_state_near(start_xy, start_yaw)
            if start is None:
                print("[OMPL] Cannot find valid start state")
                return None

        if not self._si.isValid(goal()):
            print(f"[OMPL] Goal {goal_xy} is in collision — nudging")
            goal = self._find_valid_state_near(goal_xy, goal_yaw)
            if goal is None:
                print("[OMPL] Cannot find valid goal state")
                return None

        pdef = ob.ProblemDefinition(self._si)
        pdef.setStartAndGoalStates(start, goal)

        planner = og.RRTstar(self._si)
        planner.setProblemDefinition(pdef)
        planner.setRange(0.5)
        planner.setup()

        print(f"[OMPL] Planning {start_xy} → {goal_xy} (budget={self.solve_time}s)")
        solved = planner.solve(self.solve_time)

        if not solved:
            print("[OMPL] No solution found")
            return None

        path = pdef.getSolutionPath()
        path.interpolate(50)

        waypoints = []
        for i in range(path.getStateCount()):
            s = path.getState(i)
            waypoints.append((s.getX(), s.getY()))

        waypoints[-1] = goal_xy

        print(f"[OMPL] Path found: {len(waypoints)} waypoints")
        return waypoints

    def _find_valid_state_near(self, xy, yaw, radius=0.5, tries=20):
        state = ob.State(self._space)
        for i in range(tries):
            angle  = 2 * math.pi * i / tries
            nx     = xy[0] + radius * math.cos(angle)
            ny     = xy[1] + radius * math.sin(angle)
            state().setX(nx)
            state().setY(ny)
            state().setYaw(yaw)
            if self._si.isValid(state()):
                print(f"[OMPL] Nudged to ({nx:.2f}, {ny:.2f})")
                return state
        return None


def smooth_path(waypoints, passes=3):
    pts = list(waypoints)
    for _ in range(passes):
        new_pts = [pts[0]]
        for i in range(1, len(pts) - 1):
            x = 0.5 * pts[i][0] + 0.25 * (pts[i-1][0] + pts[i+1][0])
            y = 0.5 * pts[i][1] + 0.25 * (pts[i-1][1] + pts[i+1][1])
            new_pts.append((x, y))
        new_pts.append(pts[-1])
        pts = new_pts
    return pts


def decimate_path(waypoints, min_dist=0.3):
    if not waypoints:
        return waypoints
    result = [waypoints[0]]
    for wp in waypoints[1:-1]:
        if math.dist(result[-1], wp) >= min_dist:
            result.append(wp)
    result.append(waypoints[-1])
    return result


class OccupancyGrid:
    def __init__(self, *args, **kwargs):
        pass
    def print_stats(self):
        print("[OccupancyGrid] Stub — OMPL + MuJoCo collision detection is active.")


class AStarPlanner:
    def __init__(self, *args, **kwargs):
        pass


class InProcessNavigator:

    def __init__(self, sim):
        self.sim      = sim
        self._running = False
        self._thread  = None
        self.on_complete = None

        self._plan_data = mujoco.MjData(sim.model)
        self._plan_data.qpos[:] = sim.data.qpos[:]
        self._plan_data.qvel[:] = 0.0

        self.planner = OmplPlanner(
            model          = sim.model,
            data           = self._plan_data,
            base_body_name = "base_footprint",
            x_range        = FLOOR_X,
            y_range        = FLOOR_Y,
            solve_time     = 2.0
        )

        print("[OMPL Navigator] Initialized.")

    def navigate_to(self, goal_xy, on_complete=None):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self.on_complete = on_complete
        self._running    = True
        self._thread     = threading.Thread(
            target=self._run, args=(goal_xy,), daemon=True)
        self._thread.start()

    def cancel(self):
        self._running = False

    def _run(self, goal_xy):
        pose      = self.sim.localization()
        start_xy  = (pose[0], pose[1])
        start_yaw = pose[2]

        print(f"[OMPL NAV] Planning {start_xy} → {goal_xy}")

        self._plan_data.qpos[:] = self.sim.data.qpos[:]
        self._plan_data.qvel[:] = 0.0

        raw = self.planner.plan(start_xy, goal_xy, start_yaw=start_yaw)

        if raw is None:
            print("[OMPL NAV] No path found")
            if self.on_complete:
                self.on_complete(False)
            return

        path = decimate_path(raw, min_dist=0.30)
        print(f"[OMPL NAV] {len(path)} waypoints after decimation")

        success = self._follow(path, goal_xy)
        print(f"[OMPL NAV] Done — success={success}")
        if self.on_complete:
            self.on_complete(success)

    def _follow(self, path, final_goal):
        for i, (wx, wy) in enumerate(path):
            if not self._running:
                return False
            nx, ny = path[i+1] if i+1 < len(path) else final_goal
            yaw = math.atan2(ny - wy, nx - wx)
            with self.sim._target_lock:
                self.sim.target_base = np.array([wx, wy, yaw])
            tol = GOAL_REACH_DIST if i == len(path) - 1 else WAYPOINT_REACH_DIST
            t0  = time.time()
            last_print = 0.0
            while self._running:
                p    = self.sim.localization()
                dist = math.dist((p[0], p[1]), (wx, wy))
                now  = time.time()
                if now - last_print > 3.0:
                    print(f"[OMPL NAV]  wp{i+1} dist={dist:.3f}m "
                          f"robot=({p[0]:.2f},{p[1]:.2f}) target=({wx:.2f},{wy:.2f})")
                    last_print = now
                if dist < tol:
                    break
                if now - t0 > WAYPOINT_TIMEOUT:
                    print(f"[OMPL NAV] Waypoint {i+1} timed out (dist={dist:.3f}m)")
                    return False
                time.sleep(0.05)
        return self._running


if __name__ == "__main__":
    print("ompl_navigator.py — OMPL + MuJoCo path planner")
    print("Import InProcessNavigator and use it in play_m1.py")