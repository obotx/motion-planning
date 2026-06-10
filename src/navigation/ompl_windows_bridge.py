
import os, sys
import subprocess, json, threading, time, math
import numpy as np
import mujoco

WAYPOINT_REACH_DIST = 0.25
GOAL_REACH_DIST     = 0.35
WAYPOINT_TIMEOUT    = 180.0
WAYPOINT_STALL_TIMEOUT = float(os.environ.get("AH_STALL_TIMEOUT", "5.0"))
WAYPOINT_STALL_MIN_IMPROVEMENT = 0.02
WAYPOINT_STALL_BACKUP_DIST = 0.25
WAYPOINT_STALL_BACKUP_TIMEOUT = 4.0
MIN_WAYPOINT_DIST   = 0.25
STRICT_MAX_COLLISION_STITCH_GAP = 0.75
DOCK_YAW_RATE = 0.6
FINAL_YAW_TOL       = float(os.environ.get("AH_FINAL_YAW_TOL", "0.06"))

BRIDGE_MODE = os.environ.get("OMPL_BRIDGE_MODE", "native").lower()
WSL_PYTHON  = os.environ.get("OMPL_WSL_PYTHON", "/home/user1/ompl_clean/bin/python3")
WSL_PLAN_PY = os.environ.get("OMPL_WSL_PLAN_PY", "/home/user1/ompl_bridge/plan.py")
LOCAL_PYTHON = os.environ.get("OMPL_PYTHON", sys.executable)
LOCAL_PLAN_PY = os.environ.get(
    "OMPL_PLAN_PY",
    os.path.join(os.path.dirname(__file__), "plan.py"),
)


class MujocoValidator:

    def __init__(self, model, sim_data, base_body_name="base_footprint"):
        self.model   = model
        self.data    = mujoco.MjData(model)
        self.data.qpos[:] = sim_data.qpos[:]
        self.data.qvel[:] = 0.0
        self.base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body_name)
        self.base_qposadr = self._find_free_joint_adr()

        self.base_geom_ids = self._collect_base_geoms()
        self.last_removed = 0
        self.last_max_gap = 0.0
        self.last_goal_dropped = False
        self.last_segment_invalid = False
        self.skip_segment_validation = False

        self.pickup_obj_ids = self._collect_pickup_objects()
        self.exempt_obj_ids = set()
        self.ignore_floor_objects = True
        self.chassis_clearance_radius = 0.20
        self.obj_clearance_margin     = 0.02
        if self.pickup_obj_ids:
            _ign = " (IGNORED for nav — walls/shelf only)" if self.ignore_floor_objects else ""
            print(f"[MuJoCo Validator] Tracking {len(self.pickup_obj_ids)} "
                  f"floor-object obstacles for nav clearance.{_ign}")

        print(f"[MuJoCo Validator] Ready. base_id={self.base_id}, "
              f"base_geoms={len(self.base_geom_ids)}, qposadr={self.base_qposadr}")

    def _collect_pickup_objects(self):
        ids = []
        for bid in range(self.model.nbody):
            name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            if name.startswith("pickup_obj_"):
                ids.append(int(bid))
        return ids

    def set_exempt_objects(self, body_ids):
        if body_ids is None:
            self.exempt_obj_ids = set()
        else:
            self.exempt_obj_ids = set(int(b) for b in body_ids)

    def set_exempt_all_except(self, keep_obstacle_bid):
        keep = int(keep_obstacle_bid)
        self.exempt_obj_ids = {
            bid for bid in self.pickup_obj_ids if bid != keep
        }

    def _object_radius_xy(self, body_id):
        max_r = 0.0
        for g in range(self.model.ngeom):
            if int(self.model.geom_bodyid[g]) == int(body_id):
                r = float(self.model.geom_size[g, 0])
                if r > max_r:
                    max_r = r
        return max_r

    def _clear_of_floor_objects(self, x, y):
        if self.ignore_floor_objects:
            return True
        if not self.pickup_obj_ids:
            return True
        for bid in self.pickup_obj_ids:
            if bid in self.exempt_obj_ids:
                continue
            ox = float(self.data.xpos[bid, 0])
            oy = float(self.data.xpos[bid, 1])
            min_dist = (self.chassis_clearance_radius
                        + self._object_radius_xy(bid)
                        + self.obj_clearance_margin)
            dx = x - ox
            dy = y - oy
            if (dx * dx + dy * dy) < (min_dist * min_dist):
                return False
        return True

    def _find_free_joint_adr(self):
        for j in range(self.model.njnt):
            if self.model.jnt_bodyid[j] == self.base_id and self.model.jnt_type[j] == 0:
                return self.model.jnt_qposadr[j]
        for j in range(self.model.njnt):
            if self.model.jnt_type[j] == 0:
                return self.model.jnt_qposadr[j]
        raise RuntimeError("No free joint found for base body")

    def _collect_base_geoms(self):
        arm_roots = {"Arm_1", "Arm_2"}

        def belongs_to_mobile_base(body_id):
            cur = int(body_id)
            while cur >= 0:
                name = mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, cur) or ""
                if name in arm_roots:
                    return False
                if cur == self.base_id:
                    return True
                parent = int(self.model.body_parentid[cur])
                if parent == cur:
                    break
                cur = parent
            return False

        geoms = set()
        for g in range(self.model.ngeom):
            if belongs_to_mobile_base(self.model.geom_bodyid[g]):
                geoms.add(g)
        return geoms

    def _is_robot_descendant(self, body_id):
        cur = int(body_id)
        while cur >= 0:
            if cur == self.base_id:
                return True
            parent = int(self.model.body_parentid[cur])
            if parent == cur:
                break
            cur = parent
        return False

    def _is_allowed_base_contact(self, other_geom):
        geom_name = mujoco.mj_id2name(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, int(other_geom)) or ""
        if geom_name == "floor":
            return True
        other_body = int(self.model.geom_bodyid[int(other_geom)])
        if self.ignore_floor_objects and other_body in self.pickup_obj_ids:
            return True
        return self._is_robot_descendant(other_body)

    def _set_base_pose(self, x, y, yaw=0.0):
        adr = self.base_qposadr
        self.data.qpos[adr + 0] = x
        self.data.qpos[adr + 1] = y
        self.data.qpos[adr + 3] = math.cos(yaw / 2.0)
        self.data.qpos[adr + 4] = 0.0
        self.data.qpos[adr + 5] = 0.0
        self.data.qpos[adr + 6] = math.sin(yaw / 2.0)

    def is_valid(self, x, y, yaw=0.0):
        self._set_base_pose(x, y, yaw)
        mujoco.mj_forward(self.model, self.data)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1 = int(c.geom1)
            g2 = int(c.geom2)
            if g1 in self.base_geom_ids or g2 in self.base_geom_ids:
                other = g2 if g1 in self.base_geom_ids else g1
                if self._is_allowed_base_contact(other):
                    continue
                return False
        if not self._clear_of_floor_objects(x, y):
            return False
        return True

    def sync(self, sim_data):
        self.data.qpos[:] = sim_data.qpos[:]
        self.data.qvel[:] = 0.0

    def filter_path(self, path, final_yaw=None, constant_yaw=False):
        _cyaw = float(final_yaw) if (constant_yaw and final_yaw is not None) else None
        if not path:
            self.last_removed = 0
            self.last_max_gap = 0.0
            self.last_goal_dropped = False
            return path
        valid = [path[0]]
        removed = 0
        self.last_goal_dropped = False
        for i in range(1, len(path)):
            wx, wy = path[i]
            if _cyaw is not None:
                yaw = _cyaw
            elif i < len(path) - 1:
                nx, ny = path[i + 1]
                yaw = math.atan2(ny - wy, nx - wx)
            elif final_yaw is not None:
                yaw = float(final_yaw)
            else:
                px, py = path[i - 1]
                yaw = math.atan2(wy - py, wx - px)
            if self.is_valid(wx, wy, yaw=yaw):
                valid.append((wx, wy))
            else:
                removed += 1
        if removed > 0:
            print(f"[MuJoCo Validator] Removed {removed} waypoints in collision")
        self.last_removed = removed
        if valid[-1] != path[-1]:
            tail_yaw = _cyaw if _cyaw is not None else (
                float(final_yaw) if final_yaw is not None else (
                    math.atan2(path[-1][1] - path[-2][1],
                               path[-1][0] - path[-2][0]) if len(path) >= 2 else 0.0))
            if self.is_valid(path[-1][0], path[-1][1], yaw=tail_yaw):
                valid.append(path[-1])
            else:
                print("[MuJoCo Validator] Dropped final goal because it is in collision")
                self.last_goal_dropped = True
        if len(valid) >= 2:
            self.last_max_gap = max(math.dist(a, b)
                                    for a, b in zip(valid[:-1], valid[1:]))
        else:
            self.last_max_gap = 0.0
        self.last_segment_invalid = False
        if self.skip_segment_validation:
            return valid
        SEG_SAMPLE_SPACING = 0.20
        for i in range(len(valid) - 1):
            ax, ay = valid[i]
            bx, by = valid[i + 1]
            seg_len = math.hypot(bx - ax, by - ay)
            if seg_len < SEG_SAMPLE_SPACING:
                continue
            seg_yaw = _cyaw if _cyaw is not None else math.atan2(by - ay, bx - ax)
            n_samples = max(1, int(seg_len / SEG_SAMPLE_SPACING))
            for k in range(1, n_samples):
                t = k / n_samples
                sx = ax + t * (bx - ax)
                sy = ay + t * (by - ay)
                if not self.is_valid(sx, sy, yaw=seg_yaw):
                    self.last_segment_invalid = True
                    break
            if self.last_segment_invalid:
                break
        return valid


def decimate_path(path, min_dist=MIN_WAYPOINT_DIST):
    if not path:
        return path
    result = [path[0]]
    for wp in path[1:-1]:
        if math.dist(result[-1], wp) >= min_dist:
            result.append(wp)
    result.append(path[-1])
    return result


def densify_path(path, max_step=0.20):
    if not path:
        return path
    dense = [path[0]]
    for p0, p1 in zip(path[:-1], path[1:]):
        dist = math.dist(p0, p1)
        steps = max(1, int(math.ceil(dist / max_step)))
        for k in range(1, steps + 1):
            t = k / steps
            dense.append((
                p0[0] + (p1[0] - p0[0]) * t,
                p0[1] + (p1[1] - p0[1]) * t,
            ))
    return dense


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


class InProcessNavigator:
    def __init__(self, sim):
        self.sim          = sim
        self.validator    = MujocoValidator(sim.model, sim.data)
        self._nav_gen     = 0
        self._gen_lock    = threading.Lock()
        if BRIDGE_MODE == "wsl":
            try:
                subprocess.run(["wsl", "echo", "ok"], capture_output=True, timeout=10)
            except Exception:
                pass
        print(f"[OMPL Bridge] Navigator initialized. mode={BRIDGE_MODE}")

    def _planner_command(self):
        if BRIDGE_MODE == "native":
            return [LOCAL_PYTHON, LOCAL_PLAN_PY]
        return ["wsl", WSL_PYTHON, WSL_PLAN_PY]

    def navigate_to(self, goal_xy, on_complete=None, goal_tolerance=None,
                    final_yaw=None, allow_goal_nudge=True, constant_yaw=False):
        with self._gen_lock:
            self._nav_gen += 1
            my_gen = self._nav_gen
        threading.Thread(
            target=self._run,
            args=(my_gen, goal_xy, on_complete, goal_tolerance, final_yaw,
                  allow_goal_nudge, constant_yaw),
            daemon=True).start()

    def cancel(self):
        with self._gen_lock:
            self._nav_gen += 1

    def _is_current(self, my_gen):
        with self._gen_lock:
            return my_gen == self._nav_gen

    def _run(self, my_gen, goal_xy, on_complete, goal_tolerance, final_yaw,
             allow_goal_nudge, constant_yaw=False):
        cb_fired = [False]
        def fire(ok):
            if cb_fired[0]:
                return
            cb_fired[0] = True
            if on_complete:
                on_complete(ok)

        try:
            x, y, yaw = self.sim.localization()
            payload = json.dumps({
                "start":      [float(x), float(y)],
                "goal":       [float(goal_xy[0]), float(goal_xy[1])],
                "solve_time": 1.5
            })

            print(
                f"[OMPL] Calling planner mode={BRIDGE_MODE}: "
                f"({x:.2f},{y:.2f}) -> "
                f"({float(goal_xy[0]):.2f},{float(goal_xy[1]):.2f})"
            )

            result = subprocess.run(
                self._planner_command(),
                input=payload, capture_output=True, text=True, timeout=60)

            if not self._is_current(my_gen):
                print(f"[Nav gen={my_gen}] superseded — exiting")
                return

            path_json = None
            for line in result.stdout.splitlines():
                line = line.strip()
                if line.startswith('[') or line.startswith('{'):
                    path_json = line
                    break

            if path_json is None:
                print(f"[OMPL] No JSON in output. stderr: {result.stderr[:200]}")
                fire(False)
                return

            data = json.loads(path_json)
            if isinstance(data, dict) and "error" in data:
                print(f"[OMPL] Planner error: {data['error']}")
                fire(False)
                return

            raw_path = [(float(pt[0]), float(pt[1])) for pt in data]
            print(f"[OMPL] {len(raw_path)} waypoints from OMPL")
            if raw_path:
                start_nudge = math.hypot(raw_path[0][0] - float(x),
                                         raw_path[0][1] - float(y))
                if start_nudge > 0.10:
                    print(f"[Nav] REFUSING path: planner nudged start from "
                          f"({x:.2f},{y:.2f}) to "
                          f"({raw_path[0][0]:.2f},{raw_path[0][1]:.2f}) "
                          f"(Δ={start_nudge:.2f}m).")
                    fire(False)
                    return
                goal_nudge = math.hypot(raw_path[-1][0] - float(goal_xy[0]),
                                        raw_path[-1][1] - float(goal_xy[1]))
                if (not allow_goal_nudge) and goal_nudge > 0.10:
                    print(f"[Nav] REFUSING path: planner nudged goal from "
                          f"({float(goal_xy[0]):.2f},{float(goal_xy[1]):.2f}) "
                          f"to ({raw_path[-1][0]:.2f},{raw_path[-1][1]:.2f}) "
                          f"(Δ={goal_nudge:.2f}m).")
                    fire(False)
                    return

            self.validator.sync(self.sim.data)
            validated = self.validator.filter_path(raw_path, final_yaw=final_yaw,
                                                    constant_yaw=constant_yaw)
            if not allow_goal_nudge and self.validator.last_goal_dropped:
                print("[Nav] REFUSING path: collision validation dropped the "
                      "exact pick/drop goal.")
                fire(False)
                return
            if (not allow_goal_nudge
                    and self.validator.last_segment_invalid):
                print("[Nav] REFUSING path: a sampled segment between "
                      "kept waypoints fails collision check.")
                fire(False)
                return
            print(f"[MuJoCo] {len(validated)} waypoints after collision validation "
                  f"(max_gap={self.validator.last_max_gap:.2f}m, segments_clear=True)")

            _sm_passes = (int(os.environ.get("AH_NAV_SMOOTH_PASSES", "14"))
                          if constant_yaw else 3)
            smooth_candidate = densify_path(smooth_path(validated, passes=_sm_passes))
            smooth_candidate = decimate_path(smooth_candidate)
            path = self.validator.filter_path(smooth_candidate,
                                              final_yaw=final_yaw,
                                              constant_yaw=constant_yaw)
            if (not allow_goal_nudge and
                    (self.validator.last_goal_dropped or
                     self.validator.last_segment_invalid)):
                reason = (
                    "dropped exact goal" if self.validator.last_goal_dropped
                    else "a sampled segment fails collision check"
                )
                print(f"[Nav] Smoothed strict path invalid ({reason}); "
                      "falling back to raw collision-validated path")
                fallback = decimate_path(densify_path(validated))
                path = self.validator.filter_path(fallback, final_yaw=final_yaw,
                                                  constant_yaw=constant_yaw)
                if not allow_goal_nudge and self.validator.last_goal_dropped:
                    print("[Nav] REFUSING path: fallback collision validation "
                          "dropped the exact pick/drop goal.")
                    fire(False)
                    return
                if (not allow_goal_nudge
                        and self.validator.last_segment_invalid):
                    print("[Nav] REFUSING path: fallback validation found "
                          "a segment between kept waypoints in collision.")
                    fire(False)
                    return
            print(f"[Nav] {len(path)} final waypoints "
                  f"(max_gap={self.validator.last_max_gap:.2f}m)")

            if len(path) < 2:
                print("[Nav] No safe path after MuJoCo validation")
                fire(False)
                return

            success = self._follow(my_gen, path, goal_xy, goal_tolerance, final_yaw,
                                   constant_yaw=constant_yaw)
            print(f"[Nav gen={my_gen}] Done — success={success}")
            if not self._is_current(my_gen):
                return
            fire(success)

        except subprocess.TimeoutExpired:
            print(f"[OMPL] Planner timed out mode={BRIDGE_MODE}")
            if self._is_current(my_gen):
                fire(False)
        except Exception as e:
            print(f"[OMPL] Error: {e}")
            if self._is_current(my_gen):
                fire(False)

    def _follow(self, my_gen, path, final_goal, goal_tolerance=None, final_yaw=None,
                constant_yaw=False):
        effective_goal = path[-1] if path else final_goal
        try:
            self.sim._nav_goal_xy = (float(effective_goal[0]),
                                     float(effective_goal[1]))
        except Exception:
            pass
        for i, (wx, wy) in enumerate(path):
            if not self._is_current(my_gen):
                return False
            nx, ny = path[i+1] if i+1 < len(path) else final_goal
            is_last = (i == len(path) - 1)
            if final_yaw is not None and (constant_yaw or is_last):
                wp_yaw = float(final_yaw)
            else:
                wp_yaw = math.atan2(ny - wy, nx - wx)
            _dock = (is_last and final_yaw is not None
                     and getattr(self.sim, "_base_near_goal_vel", None)
                     is not None)
            _dock_phase = "pos" if _dock else None
            _dock_target_yaw = (float(self.sim.localization()[2])
                                if _dock else wp_yaw)
            _dock_last_t = time.time()
            _wp_yaw_eff = _dock_target_yaw if _dock else wp_yaw
            with self.sim._target_lock:
                self.sim.target_base = np.array([wx, wy, _wp_yaw_eff])
            if is_last and goal_tolerance is not None:
                tol = float(goal_tolerance)
            else:
                tol = GOAL_REACH_DIST if is_last else WAYPOINT_REACH_DIST
            t0  = time.time()
            last_print = 0.0
            best_progress = float("inf")
            last_progress = t0
            while self._is_current(my_gen):
                now = time.time()
                cx, cy, cyaw = self.sim.localization()
                dist = math.hypot(cx - wx, cy - wy)
                pos_ok = dist < tol
                yaw_ok = True
                if is_last and final_yaw is not None:
                    yaw_err = abs(((cyaw - float(final_yaw) + math.pi)
                                   % (2 * math.pi)) - math.pi)
                    _eff_yaw_tol = (FINAL_YAW_TOL if _dock
                                    else float(os.environ.get(
                                        "AH_PICKUP_YAW_TOL", "0.175")))
                    yaw_ok = yaw_err < _eff_yaw_tol
                if _dock:
                    if _dock_phase == "pos":
                        if pos_ok:
                            _dock_phase = "yaw"
                            _dock_last_t = now
                            last_progress = now
                            print(f"[Nav] dock: position reached "
                                  f"(dist={dist:.2f}m) → gentle stationary "
                                  f"in-place yaw to final_yaw "
                                  f"({DOCK_YAW_RATE:.1f}rad/s)")
                        else:
                            yaw_ok = False
                            if os.environ.get("AH_DOCK_YAW_PROG", "1") == "1":
                                _e_yp = ((float(final_yaw) - _dock_target_yaw
                                          + math.pi) % (2 * math.pi)) - math.pi
                                _prog = max(0.0, 1.0 - dist / 1.0)
                                _dt_yp = max(0.0, now - _dock_last_t)
                                _dock_last_t = now
                                _step = 0.3 * _dt_yp * _prog
                                _dock_target_yaw += max(-_step, min(_step, _e_yp))
                    if _dock_phase == "yaw":
                        _dt = max(0.0, now - _dock_last_t)
                        _dock_last_t = now
                        _e = ((float(final_yaw) - _dock_target_yaw + math.pi)
                              % (2 * math.pi)) - math.pi
                        _mx = DOCK_YAW_RATE * _dt
                        _dock_target_yaw += max(-_mx, min(_mx, _e))
                    with self.sim._target_lock:
                        self.sim.target_base = np.array(
                            [wx, wy, _dock_target_yaw])
                    if os.environ.get("AH_CARRY_LOW_IMPRATIO", "1") == "1":
                        _drv_imp = float(os.environ.get("AH_IMPRATIO", "1.0"))
                        _yaw_imp = float(os.environ.get("AH_YAW_IMPRATIO", "80.0"))
                        self.sim.model.opt.impratio = (
                            _yaw_imp if _dock_phase == "yaw" else _drv_imp)
                progress = dist
                if is_last and final_yaw is not None:
                    progress += 0.20 * min(float(yaw_err), math.pi)
                if progress < best_progress - WAYPOINT_STALL_MIN_IMPROVEMENT:
                    best_progress = progress
                    last_progress = now
                if now - last_print > 5.0:
                    extra = ""
                    if is_last and final_yaw is not None:
                        extra = f"  yaw_err={math.degrees(yaw_err):.1f}deg"
                    print(f"[Nav gen={my_gen}] wp{i+1}/{len(path)} dist={dist:.2f}m "
                          f"pos=({cx:.2f},{cy:.2f}) target=({wx:.2f},{wy:.2f}){extra}")
                    last_print = now
                passed = False
                if (not is_last
                        and getattr(self.sim, "_base_near_goal_vel", None)
                        is not None):
                    _dnext = math.hypot(cx - nx, cy - ny)
                    if _dnext < dist:
                        passed = True
                    else:
                        _dx, _dy = nx - wx, ny - wy
                        _dn = math.hypot(_dx, _dy)
                        if _dn > 1e-6:
                            _ux, _uy = _dx / _dn, _dy / _dn
                            _proj = (cx - wx) * _ux + (cy - wy) * _uy
                            _lat = abs((cx - wx) * (-_uy) + (cy - wy) * _ux)
                            passed = (_proj > 0.0 and _lat < 1.0)
                if (pos_ok and yaw_ok) or passed:
                    break
                if now - last_progress > WAYPOINT_STALL_TIMEOUT:
                    extra = ""
                    if is_last and final_yaw is not None:
                        extra = f", yaw_err={math.degrees(yaw_err):.1f}deg"
                    print(f"[Nav] Waypoint {i+1} stalled at dist={dist:.2f}m"
                          f"{extra}; no progress for "
                          f"{WAYPOINT_STALL_TIMEOUT:.0f}s")
                    if os.environ.get("AH_NAV_PUSH_THROUGH", "0") == "1":
                        try:
                            _pbx, _pby, _pyaw = self.sim.localization()
                            if math.isfinite(_pbx) and math.isfinite(_pby):
                                _psave = {}
                                for _a, _v in (
                                        ("_base_max_vel_override",
                                         float(os.environ.get(
                                             "AH_NAV_PUSH_VEL", "13.0"))),
                                        ("_base_omega_max_override", 0.2),
                                        ("_base_cmd_slew_override", 0.08)):
                                    _psave[_a] = getattr(self.sim, _a, None)
                                    setattr(self.sim, _a, _v)
                                with self.sim._target_lock:
                                    self.sim.target_base = np.array(
                                        [wx, wy, _pyaw])
                                _pt0 = time.time()
                                _pdur = float(os.environ.get(
                                    "AH_NAV_PUSH_SECS", "3.0"))
                                while time.time() - _pt0 < _pdur:
                                    if not self._is_current(my_gen):
                                        break
                                    time.sleep(0.05)
                                for _a, _old in _psave.items():
                                    if _old is None:
                                        try:
                                            delattr(self.sim, _a)
                                        except Exception:
                                            pass
                                    else:
                                        setattr(self.sim, _a, _old)
                                _pfx, _pfy, _ = self.sim.localization()
                                _pmoved = (math.hypot(_pfx - _pbx, _pfy - _pby)
                                           if math.isfinite(_pfx) else 0.0)
                                print(f"[Nav] PUSH-THROUGH: forward shove "
                                      f"(wall-clear path = loose obj) moved "
                                      f"{_pmoved*100:.0f}cm toward wp{i+1}")
                                if _pmoved > 0.05:
                                    last_progress = time.time()
                                    best_progress = math.hypot(
                                        _pfx - wx, _pfy - wy)
                                    continue
                        except Exception as _pe:
                            print(f"[Nav] push-through skipped: {_pe}")
                    _rescue_saved = {}
                    try:
                        _bx, _by, _byaw = self.sim.localization()
                        if not (math.isfinite(_bx) and math.isfinite(_by)):
                            print("[Nav] Stall rescue skipped: non-finite pose")
                            return False
                        for _a in ("integral_x", "integral_y", "integral_yaw"):
                            if hasattr(self.sim, _a):
                                setattr(self.sim, _a, 0.0)
                        for _a in ("deriv_x", "deriv_y", "deriv_yaw",
                                   "prev_delta_x", "prev_delta_y",
                                   "prev_delta_yaw"):
                            if hasattr(self.sim, _a):
                                setattr(self.sim, _a, 0.0)
                        self.sim._prev_target_vel = None
                        for _a, _v in (("_base_max_vel_override", 1.5),
                                       ("_base_omega_max_override", 0.4),
                                       ("_base_cmd_slew_override", 0.05)):
                            _rescue_saved[_a] = getattr(self.sim, _a, None)
                            setattr(self.sim, _a, _v)
                        _vx = _bx - wx
                        _vy = _by - wy
                        _vn = math.hypot(_vx, _vy)
                        if _vn < 1e-3:
                            _ux = -math.cos(_byaw)
                            _uy = -math.sin(_byaw)
                        else:
                            _ux = _vx / _vn
                            _uy = _vy / _vn
                        _backup_x = _bx + _ux * WAYPOINT_STALL_BACKUP_DIST
                        _backup_y = _by + _uy * WAYPOINT_STALL_BACKUP_DIST
                        print(f"[Nav] Stall rescue: GENTLE back "
                              f"{WAYPOINT_STALL_BACKUP_DIST*100:.0f}cm "
                              f"(windup reset, vel-capped) "
                              f"({_bx:.2f},{_by:.2f}) → "
                              f"({_backup_x:.2f},{_backup_y:.2f})")
                        with self.sim._target_lock:
                            self.sim.target_base = np.array(
                                [_backup_x, _backup_y, _byaw])
                        _bk_t0 = time.time()
                        _bk_tol = 0.10
                        while time.time() - _bk_t0 < WAYPOINT_STALL_BACKUP_TIMEOUT:
                            if not self._is_current(my_gen):
                                break
                            _cx, _cy, _ = self.sim.localization()
                            if not (math.isfinite(_cx) and math.isfinite(_cy)):
                                print("[Nav] Stall rescue abort: pose went NaN")
                                break
                            if math.hypot(_cx - _backup_x,
                                          _cy - _backup_y) <= _bk_tol:
                                break
                            time.sleep(0.05)
                        _fx, _fy, _ = self.sim.localization()
                        _moved = (math.hypot(_fx - _bx, _fy - _by) * 100
                                  if math.isfinite(_fx) and math.isfinite(_fy)
                                  else float("nan"))
                        print(f"[Nav] Stall rescue done: chassis at "
                              f"({_fx:.2f},{_fy:.2f}) — moved {_moved:.0f}cm "
                              f"in {time.time() - _bk_t0:.1f}s")
                    except Exception as _bk_e:
                        print(f"[Nav] Stall rescue skipped: {_bk_e}")
                    finally:
                        for _a, _old in _rescue_saved.items():
                            if _old is None:
                                if hasattr(self.sim, _a):
                                    try:
                                        delattr(self.sim, _a)
                                    except Exception:
                                        pass
                            else:
                                setattr(self.sim, _a, _old)
                    return False
                if now - t0 > WAYPOINT_TIMEOUT:
                    print(f"[Nav] Waypoint {i+1} timed out at dist={dist:.2f}m")
                    return False
                time.sleep(0.05)
        if not self._is_current(my_gen):
            return False
        cx, cy, cyaw = self.sim.localization()
        dist = math.hypot(cx - effective_goal[0], cy - effective_goal[1])
        verify_tol = float(goal_tolerance) if goal_tolerance is not None else GOAL_REACH_DIST
        if dist > verify_tol + 0.05:
            print(f"[Nav] verify FAIL: final dist={dist:.3f}m to path-end "
                  f"({effective_goal[0]:.2f},{effective_goal[1]:.2f}) "
                  f"> tol={verify_tol:.3f}m+0.05")
            return False
        if final_yaw is not None:
            yaw_err = abs(((cyaw - float(final_yaw) + math.pi) % (2 * math.pi)) - math.pi)
            _vtol = (FINAL_YAW_TOL if constant_yaw
                     else float(os.environ.get("AH_PICKUP_YAW_TOL", "0.175")))
            if yaw_err > _vtol:
                print(f"[Nav] verify FAIL: yaw_err={math.degrees(yaw_err):.1f}deg "
                      f"> {math.degrees(_vtol):.1f}deg")
                return False
            print(f"[Nav] verify OK: final dist={dist:.3f}m  "
                  f"yaw_err={math.degrees(yaw_err):.1f}deg")
        else:
            print(f"[Nav] verify OK: final dist={dist:.3f}m")
        nudge_dist = math.hypot(effective_goal[0] - final_goal[0],
                                effective_goal[1] - final_goal[1])
        if nudge_dist > 0.10:
            print(f"[Nav] note: caller's goal ({final_goal[0]:.2f},{final_goal[1]:.2f}) "
                  f"was nudged by plan.py to ({effective_goal[0]:.2f},{effective_goal[1]:.2f}) "
                  f"— Δ={nudge_dist:.2f}m")
        return True
