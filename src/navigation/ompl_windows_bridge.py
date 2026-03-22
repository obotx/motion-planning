"""
ompl_windows_bridge.py  —  Windows-side OMPL navigator
========================================================
1. Calls OMPL RRT* planner in WSL (plan.py) via subprocess
2. Validates returned path using MuJoCo collision detection (mj_forward + ncon)
3. Executes validated path in simulation
"""

import subprocess, json, threading, time, math
import numpy as np
import mujoco

WAYPOINT_REACH_DIST = 0.60
GOAL_REACH_DIST     = 0.65
WAYPOINT_TIMEOUT    = 180.0
MIN_WAYPOINT_DIST   = 0.40

WSL_PYTHON  = "/home/user1/ompl_clean/bin/python3"
WSL_PLAN_PY = "/home/user1/ompl_bridge/plan.py"


# ---------------------------------------------------------------------------
# MuJoCo collision validator
# ---------------------------------------------------------------------------
class MujocoValidator:
    """
    Validates waypoints using MuJoCo native collision detection.
    Uses dedicated MjData — never touches main sim data.
    Only checks contacts involving the mobile base body.
    """

    def __init__(self, model, sim_data, base_body_name="base_footprint"):
        self.model   = model
        self.data    = mujoco.MjData(model)
        self.data.qpos[:] = sim_data.qpos[:]
        self.data.qvel[:] = 0.0
        self.base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body_name)
        self.base_qposadr = self._find_free_joint_adr()

        # Collect all geom IDs belonging to the base and its direct children
        self.base_geom_ids = self._collect_base_geoms()
        print(f"[MuJoCo Validator] Ready. base_id={self.base_id}, "
              f"base_geoms={len(self.base_geom_ids)}, qposadr={self.base_qposadr}")

    def _find_free_joint_adr(self):
        for j in range(self.model.njnt):
            if self.model.jnt_bodyid[j] == self.base_id and self.model.jnt_type[j] == 0:
                return self.model.jnt_qposadr[j]
        for j in range(self.model.njnt):
            if self.model.jnt_type[j] == 0:
                return self.model.jnt_qposadr[j]
        raise RuntimeError("No free joint found for base body")

    def _collect_base_geoms(self):
        """Collect geom IDs that belong to the base body only (not arms)."""
        geoms = set()
        for g in range(self.model.ngeom):
            if self.model.geom_bodyid[g] == self.base_id:
                geoms.add(g)
        return geoms

    def _set_base_xy(self, x, y):
        adr = self.base_qposadr
        self.data.qpos[adr + 0] = x
        self.data.qpos[adr + 1] = y
        self.data.qpos[adr + 3] = 1.0
        self.data.qpos[adr + 4] = 0.0
        self.data.qpos[adr + 5] = 0.0
        self.data.qpos[adr + 6] = 0.0

    def is_valid(self, x, y):
        """
        mj_forward() + check contacts involving base geoms only.
        Ignores arm/gripper contacts.
        """
        self._set_base_xy(x, y)
        mujoco.mj_forward(self.model, self.data)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 in self.base_geom_ids or c.geom2 in self.base_geom_ids:
                return False
        return True

    def sync(self, sim_data):
        self.data.qpos[:] = sim_data.qpos[:]
        self.data.qvel[:] = 0.0

    def filter_path(self, path):
        """Remove waypoints where base is in collision per MuJoCo."""
        valid = [path[0]]
        removed = 0
        for pt in path[1:]:
            if self.is_valid(pt[0], pt[1]):
                valid.append(pt)
            else:
                removed += 1
        if removed > 0:
            print(f"[MuJoCo Validator] Removed {removed} waypoints in collision")
        # Always keep final goal
        if valid[-1] != path[-1]:
            valid.append(path[-1])
        return valid


# ---------------------------------------------------------------------------
# Path post-processing
# ---------------------------------------------------------------------------
def decimate_path(path, min_dist=MIN_WAYPOINT_DIST):
    if not path:
        return path
    result = [path[0]]
    for wp in path[1:-1]:
        if math.dist(result[-1], wp) >= min_dist:
            result.append(wp)
    result.append(path[-1])
    return result


def smooth_path(waypoints, passes=3):
    pts = list(waypoints)
    for _ in range(passes):
        new_pts = [pts[0]]
        for i in range(1, len(pts) - 1):
            x = 0.5*pts[i][0] + 0.25*(pts[i-1][0] + pts[i+1][0])
            y = 0.5*pts[i][1] + 0.25*(pts[i-1][1] + pts[i+1][1])
            new_pts.append((x, y))
        new_pts.append(pts[-1])
        pts = new_pts
    return pts


# ---------------------------------------------------------------------------
# InProcessNavigator
# ---------------------------------------------------------------------------
class InProcessNavigator:
    def __init__(self, sim):
        self.sim          = sim
        self._cancel_flag = False
        self._nav_thread  = None
        self.validator    = MujocoValidator(sim.model, sim.data)
        # Pre-warm WSL
        try:
            subprocess.run(["wsl", "echo", "ok"], capture_output=True, timeout=10)
        except Exception:
            pass
        print("[OMPL Bridge] Navigator initialized.")

    def navigate_to(self, goal_xy, on_complete=None):
        self._cancel_flag = True
        if self._nav_thread and self._nav_thread.is_alive():
            self._nav_thread.join(timeout=2.0)
        self._cancel_flag = False
        self._nav_thread  = threading.Thread(
            target=self._run, args=(goal_xy, on_complete), daemon=True)
        self._nav_thread.start()

    def cancel(self):
        self._cancel_flag = True

    def _run(self, goal_xy, on_complete):
        try:
            x, y, yaw = self.sim.localization()
            payload = json.dumps({
                "start":      [float(x), float(y)],
                "goal":       [float(goal_xy[0]), float(goal_xy[1])],
                "solve_time": 3.0
            })

            print(f"[OMPL] Calling WSL: ({x:.2f},{y:.2f}) → ({float(goal_xy[0]):.2f},{float(goal_xy[1]):.2f})")

            result = subprocess.run(
                ["wsl", WSL_PYTHON, WSL_PLAN_PY],
                input=payload, capture_output=True, text=True, timeout=60)

            path_json = None
            for line in result.stdout.splitlines():
                line = line.strip()
                if line.startswith('[') or line.startswith('{'):
                    path_json = line
                    break

            if path_json is None:
                print(f"[OMPL] No JSON in output. stderr: {result.stderr[:200]}")
                if on_complete: on_complete(False)
                return

            data = json.loads(path_json)
            if isinstance(data, dict) and "error" in data:
                print(f"[OMPL] Planner error: {data['error']}")
                if on_complete: on_complete(False)
                return

            raw_path = [(float(pt[0]), float(pt[1])) for pt in data]
            print(f"[OMPL] {len(raw_path)} waypoints from OMPL")

            # MuJoCo collision validation
            self.validator.sync(self.sim.data)
            validated = self.validator.filter_path(raw_path)
            print(f"[MuJoCo] {len(validated)} waypoints after collision validation")

            # Post-process
            path = decimate_path(smooth_path(validated))
            print(f"[Nav] {len(path)} final waypoints")

            if len(path) < 2:
                print("[Nav] Path too short after validation — using raw OMPL path")
                path = decimate_path(smooth_path(raw_path))

            success = self._follow(path, goal_xy)
            print(f"[Nav] Done — success={success}")
            if on_complete: on_complete(success)

        except subprocess.TimeoutExpired:
            print("[OMPL] WSL timed out")
            if on_complete: on_complete(False)
        except Exception as e:
            print(f"[OMPL] Error: {e}")
            if on_complete: on_complete(False)

    def _follow(self, path, final_goal):
        for i, (wx, wy) in enumerate(path):
            if self._cancel_flag:
                return False
            nx, ny = path[i+1] if i+1 < len(path) else final_goal
            yaw = math.atan2(ny - wy, nx - wx)
            with self.sim._target_lock:
                self.sim.target_base = np.array([wx, wy, yaw])
            tol = GOAL_REACH_DIST if i == len(path)-1 else WAYPOINT_REACH_DIST
            t0  = time.time()
            last_print = 0.0
            while not self._cancel_flag:
                cx, cy, _ = self.sim.localization()
                dist = math.hypot(cx - wx, cy - wy)
                now = time.time()
                if now - last_print > 5.0:
                    print(f"[Nav] wp{i+1}/{len(path)} dist={dist:.2f}m "
                          f"pos=({cx:.2f},{cy:.2f}) target=({wx:.2f},{wy:.2f})")
                    last_print = now
                if dist < tol:
                    break
                if now - t0 > WAYPOINT_TIMEOUT:
                    print(f"[Nav] Waypoint {i+1} timed out at dist={dist:.2f}m")
                    return False
                time.sleep(0.05)
        return not self._cancel_flag