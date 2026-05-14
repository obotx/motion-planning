"""
Windows-side OMPL navigator.

1. Calls the OMPL RRT* planner (plan.py) via subprocess (native or WSL).
2. Validates the returned path using MuJoCo collision detection.
3. Executes the validated path in simulation.
"""

import os, sys
import subprocess, json, threading, time, math
import numpy as np
import mujoco

WAYPOINT_REACH_DIST = 0.25
GOAL_REACH_DIST     = 0.35
WAYPOINT_TIMEOUT    = 180.0
# Stall detection — when the chassis hasn't progressed within this
# window the PID is saturated against an obstacle and additional
# waiting will not help.
WAYPOINT_STALL_TIMEOUT = 5.0
WAYPOINT_STALL_MIN_IMPROVEMENT = 0.02
# Stall recovery — back the chassis along the reverse-approach by this
# distance so the next plan attempt sees a collision-free start pose.
WAYPOINT_STALL_BACKUP_DIST = 0.50
WAYPOINT_STALL_BACKUP_TIMEOUT = 4.0
MIN_WAYPOINT_DIST   = 0.25
STRICT_MAX_COLLISION_STITCH_GAP = 0.75
FINAL_YAW_TOL       = 0.20   # rad (~11.5 deg) — final-waypoint yaw tolerance.

BRIDGE_MODE = os.environ.get("OMPL_BRIDGE_MODE", "native").lower()
WSL_PYTHON  = os.environ.get("OMPL_WSL_PYTHON", "/home/user1/ompl_clean/bin/python3")
WSL_PLAN_PY = os.environ.get("OMPL_WSL_PLAN_PY", "/home/user1/ompl_bridge/plan.py")
LOCAL_PYTHON = os.environ.get("OMPL_PYTHON", sys.executable)
LOCAL_PLAN_PY = os.environ.get(
    "OMPL_PLAN_PY",
    os.path.join(os.path.dirname(__file__), "plan.py"),
)


# ---------------------------------------------------------------------------
# MuJoCo collision validator
# ---------------------------------------------------------------------------
class MujocoValidator:
    """
    Validates waypoints using MuJoCo native collision detection.

    Uses a dedicated MjData (never touches the main sim) and only checks
    contacts involving the mobile-base body (arm/gripper contacts ignored).
    """

    def __init__(self, model, sim_data, base_body_name="base_footprint"):
        self.model   = model
        self.data    = mujoco.MjData(model)
        self.data.qpos[:] = sim_data.qpos[:]
        self.data.qvel[:] = 0.0
        self.base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body_name)
        self.base_qposadr = self._find_free_joint_adr()

        # base_footprint itself has no geoms; chassis/wheels live under child
        # bodies. Arms are also descendants and must be excluded.
        self.base_geom_ids = self._collect_base_geoms()
        self.last_removed = 0
        self.last_max_gap = 0.0
        self.last_goal_dropped = False
        self.last_segment_invalid = False
        self.skip_segment_validation = False  # set by relaxed fallback

        # Per-floor-object clearance check: every pickup_obj_* body in the
        # model is treated as a circular obstacle for the chassis during
        # nav.  The selected pick target / held object are temporarily
        # exempted via set_exempt_objects() so the chassis is allowed to
        # approach to standoff or carry distance.
        self.pickup_obj_ids = self._collect_pickup_objects()
        self.exempt_obj_ids = set()
        # Chassis-vs-floor-object clearance: rejection threshold is
        # chassis_radius + obj_radius + margin.  Tight enough that
        # dense object spawns still leave corridors for OMPL paths.
        self.chassis_clearance_radius = 0.20
        self.obj_clearance_margin     = 0.02
        if self.pickup_obj_ids:
            print(f"[MuJoCo Validator] Tracking {len(self.pickup_obj_ids)} "
                  f"floor-object obstacles for nav clearance.")

        print(f"[MuJoCo Validator] Ready. base_id={self.base_id}, "
              f"base_geoms={len(self.base_geom_ids)}, qposadr={self.base_qposadr}")

    def _collect_pickup_objects(self):
        """Find all pickup_obj_* bodies in the model and return their body IDs."""
        ids = []
        for bid in range(self.model.nbody):
            name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            if name.startswith("pickup_obj_"):
                ids.append(int(bid))
        return ids

    def set_exempt_objects(self, body_ids):
        """
        Exempt the given pickup-object body IDs from the nav-clearance
        check (e.g. the selected pick target during approach, or the
        held object during transport).  Pass an empty iterable to clear.
        """
        if body_ids is None:
            self.exempt_obj_ids = set()
        else:
            self.exempt_obj_ids = set(int(b) for b in body_ids)

    def set_exempt_all_except(self, keep_obstacle_bid):
        """Relaxed-clearance fallback: exempt every floor object except
        the one passed (the selected pick target stays a hard obstacle
        so the chassis cannot drive through it).
        """
        keep = int(keep_obstacle_bid)
        self.exempt_obj_ids = {
            bid for bid in self.pickup_obj_ids if bid != keep
        }

    def _object_radius_xy(self, body_id):
        """Best-effort outer radius of a pickup object's geoms (cylinder)."""
        max_r = 0.0
        for g in range(self.model.ngeom):
            if int(self.model.geom_bodyid[g]) == int(body_id):
                r = float(self.model.geom_size[g, 0])
                if r > max_r:
                    max_r = r
        return max_r

    def _clear_of_floor_objects(self, x, y):
        """True if (x, y) is far enough from every non-exempt pickup object."""
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
        """Collect mobile-base geom IDs while excluding arm/gripper descendants."""
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
        return self._is_robot_descendant(other_body)

    def _set_base_pose(self, x, y, yaw=0.0):
        """Place base at (x, y, yaw) — yaw is rotation about world Z."""
        adr = self.base_qposadr
        self.data.qpos[adr + 0] = x
        self.data.qpos[adr + 1] = y
        # Quaternion for rotation about Z by yaw radians
        self.data.qpos[adr + 3] = math.cos(yaw / 2.0)
        self.data.qpos[adr + 4] = 0.0
        self.data.qpos[adr + 5] = 0.0
        self.data.qpos[adr + 6] = math.sin(yaw / 2.0)

    def is_valid(self, x, y, yaw=0.0):
        """
        Run mj_forward() and check contacts involving base geoms only.
        Also enforce a chassis-radius clearance against every non-exempt
        pickup object so curved nav paths cannot graze floor objects.

        yaw: robot heading in radians. The chassis is asymmetric (arms extend
        forward), so a configuration valid at yaw=0 may collide at the actual
        heading; pass the heading the robot will have at this waypoint.
        """
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
        # Independent floor-object clearance: contact-based check above can
        # miss objects that are close in XY but not yet AABB-overlapping
        # at this yaw, so we additionally reject any pose within the
        # chassis-clearance radius of any non-exempt pickup object.
        if not self._clear_of_floor_objects(x, y):
            return False
        return True

    def sync(self, sim_data):
        self.data.qpos[:] = sim_data.qpos[:]
        self.data.qvel[:] = 0.0

    def filter_path(self, path, final_yaw=None):
        """
        Remove waypoints where the base is in collision per MuJoCo.

        Yaw at each waypoint is the heading toward the next waypoint (what
        the navigator commands during motion); for the final waypoint, use
        final_yaw if provided, else the heading-from-previous direction.
        """
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
            if i < len(path) - 1:
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
        # Re-append the original goal only if it differs from valid[-1] AND
        # is itself reachable at the final yaw.
        if valid[-1] != path[-1]:
            tail_yaw = float(final_yaw) if final_yaw is not None else (
                math.atan2(path[-1][1] - path[-2][1],
                           path[-1][0] - path[-2][0]) if len(path) >= 2 else 0.0)
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
        # Segment validation: sample at 20cm spacing between consecutive
        # kept waypoints.  Skipped in relaxed-clearance fallback.
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
            seg_yaw = math.atan2(by - ay, bx - ax)
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


# ---------------------------------------------------------------------------
# InProcessNavigator
# ---------------------------------------------------------------------------
class InProcessNavigator:
    def __init__(self, sim):
        self.sim          = sim
        self.validator    = MujocoValidator(sim.model, sim.data)
        # Generation token: every navigate_to() increments _nav_gen. Each
        # background thread captures its own token and self-exits when it
        # sees a newer token, making cancellation deterministic even when an
        # old thread is blocked in subprocess.run().
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
                    final_yaw=None, allow_goal_nudge=True):
        with self._gen_lock:
            self._nav_gen += 1
            my_gen = self._nav_gen
        threading.Thread(
            target=self._run,
            args=(my_gen, goal_xy, on_complete, goal_tolerance, final_yaw,
                  allow_goal_nudge),
            daemon=True).start()

    def cancel(self):
        # Bumping the generation token invalidates the running thread.
        with self._gen_lock:
            self._nav_gen += 1

    def _is_current(self, my_gen):
        with self._gen_lock:
            return my_gen == self._nav_gen

    def _run(self, my_gen, goal_xy, on_complete, goal_tolerance, final_yaw,
             allow_goal_nudge):
        # Single-shot callback dispatcher (avoid double-fire in error paths).
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
                "solve_time": 3.0
            })

            print(
                f"[OMPL] Calling planner mode={BRIDGE_MODE}: "
                f"({x:.2f},{y:.2f}) -> "
                f"({float(goal_xy[0]):.2f},{float(goal_xy[1]):.2f})"
            )

            result = subprocess.run(
                self._planner_command(),
                input=payload, capture_output=True, text=True, timeout=60)

            # If a newer navigate_to was issued while we were blocked in OMPL,
            # exit silently without firing the callback or driving the base.
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
                # Refuse a path whose start was nudged: the robot is likely
                # near a keepout and the nudge target may not be OMPL-safe.
                start_nudge = math.hypot(raw_path[0][0] - float(x),
                                         raw_path[0][1] - float(y))
                if start_nudge > 0.10:
                    print(f"[Nav] REFUSING path: planner nudged start from "
                          f"({x:.2f},{y:.2f}) to "
                          f"({raw_path[0][0]:.2f},{raw_path[0][1]:.2f}) "
                          f"(Δ={start_nudge:.2f}m).")
                    fire(False)
                    return
                # Pick candidates must be reached exactly because arm IK was
                # pre-screened for the caller's pose.
                goal_nudge = math.hypot(raw_path[-1][0] - float(goal_xy[0]),
                                        raw_path[-1][1] - float(goal_xy[1]))
                if (not allow_goal_nudge) and goal_nudge > 0.10:
                    print(f"[Nav] REFUSING path: planner nudged goal from "
                          f"({float(goal_xy[0]):.2f},{float(goal_xy[1]):.2f}) "
                          f"to ({raw_path[-1][0]:.2f},{raw_path[-1][1]:.2f}) "
                          f"(Δ={goal_nudge:.2f}m).")
                    fire(False)
                    return

            # MuJoCo collision validation, with yaw inferred per waypoint.
            self.validator.sync(self.sim.data)
            validated = self.validator.filter_path(raw_path, final_yaw=final_yaw)
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

            # Prefer the smoothed path, but if smoothing creates a stitch gap
            # that strict mode is trying to avoid, fall back to the densified
            # raw-valid waypoints.
            smooth_candidate = densify_path(smooth_path(validated))
            smooth_candidate = decimate_path(smooth_candidate)
            path = self.validator.filter_path(smooth_candidate,
                                              final_yaw=final_yaw)
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
                path = self.validator.filter_path(fallback, final_yaw=final_yaw)
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

            success = self._follow(my_gen, path, goal_xy, goal_tolerance, final_yaw)
            print(f"[Nav gen={my_gen}] Done — success={success}")
            if not self._is_current(my_gen):
                # Superseded mid-_follow; do not invoke caller's callback.
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

    def _follow(self, my_gen, path, final_goal, goal_tolerance=None, final_yaw=None):
        # The path's last waypoint is what the robot is actually driven to.
        # When plan.py nudges an invalid goal, path[-1] != final_goal — verify
        # against path[-1] so we don't reject a nav that reached the path end.
        effective_goal = path[-1] if path else final_goal
        for i, (wx, wy) in enumerate(path):
            if not self._is_current(my_gen):
                return False
            nx, ny = path[i+1] if i+1 < len(path) else final_goal
            is_last = (i == len(path) - 1)
            if is_last and final_yaw is not None:
                wp_yaw = float(final_yaw)
            else:
                wp_yaw = math.atan2(ny - wy, nx - wx)
            with self.sim._target_lock:
                self.sim.target_base = np.array([wx, wy, wp_yaw])
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
                # On the last waypoint, additionally require yaw to converge
                # within tolerance so we don't exit position-converged while
                # the base PD is still rotating to final_yaw.
                yaw_ok = True
                if is_last and final_yaw is not None:
                    yaw_err = abs(((cyaw - float(final_yaw) + math.pi)
                                   % (2 * math.pi)) - math.pi)
                    yaw_ok = yaw_err < FINAL_YAW_TOL
                progress = dist
                if is_last and final_yaw is not None:
                    # Count yaw convergence as progress so a valid in-place
                    # rotation does not look like a position stall.
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
                if pos_ok and yaw_ok:
                    break
                if now - last_progress > WAYPOINT_STALL_TIMEOUT:
                    extra = ""
                    if is_last and final_yaw is not None:
                        extra = f", yaw_err={math.degrees(yaw_err):.1f}deg"
                    print(f"[Nav] Waypoint {i+1} stalled at dist={dist:.2f}m"
                          f"{extra}; no progress for "
                          f"{WAYPOINT_STALL_TIMEOUT:.0f}s")
                    # Stall rescue — drive the chassis back along the
                    # reverse-approach direction so it exits any
                    # constraint region it is wedged against.  Without
                    # this the next plan attempt sees the chassis in a
                    # clearance-violating pose and the planner refuses.
                    try:
                        _bx, _by, _byaw = self.sim.localization()
                        # reverse-approach unit = from waypoint toward
                        # current pos (= direction chassis was coming
                        # from).  If degenerate (very small distance),
                        # use chassis yaw to back straight rearward.
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
                        print(f"[Nav] Stall rescue: backing chassis "
                              f"{WAYPOINT_STALL_BACKUP_DIST*100:.0f}cm "
                              f"along reverse-approach "
                              f"({_bx:.2f},{_by:.2f}) → "
                              f"({_backup_x:.2f},{_backup_y:.2f}) "
                              f"to exit collision region before failing")
                        with self.sim._target_lock:
                            self.sim.target_base = np.array(
                                [_backup_x, _backup_y, _byaw])
                        _bk_t0 = time.time()
                        _bk_tol = 0.10  # 10 cm — coarse, just need to escape
                        while time.time() - _bk_t0 < WAYPOINT_STALL_BACKUP_TIMEOUT:
                            if not self._is_current(my_gen):
                                break
                            _cx, _cy, _ = self.sim.localization()
                            if math.hypot(_cx - _backup_x,
                                          _cy - _backup_y) <= _bk_tol:
                                break
                            time.sleep(0.05)
                        _fx, _fy, _ = self.sim.localization()
                        print(f"[Nav] Stall rescue done: chassis at "
                              f"({_fx:.2f},{_fy:.2f}) — moved "
                              f"{math.hypot(_fx - _bx, _fy - _by)*100:.0f}cm "
                              f"in {time.time() - _bk_t0:.1f}s")
                    except Exception as _bk_e:
                        print(f"[Nav] Stall rescue skipped: {_bk_e}")
                    return False
                if now - t0 > WAYPOINT_TIMEOUT:
                    print(f"[Nav] Waypoint {i+1} timed out at dist={dist:.2f}m")
                    return False
                time.sleep(0.05)
        if not self._is_current(my_gen):
            return False
        # Verify-on-success: confirm final pose against the path's effective
        # goal (which may differ from final_goal when plan.py nudged an
        # invalid goal).
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
            if yaw_err > FINAL_YAW_TOL:
                print(f"[Nav] verify FAIL: yaw_err={math.degrees(yaw_err):.1f}deg "
                      f"> {math.degrees(FINAL_YAW_TOL):.1f}deg")
                return False
            print(f"[Nav] verify OK: final dist={dist:.3f}m  "
                  f"yaw_err={math.degrees(yaw_err):.1f}deg")
        else:
            print(f"[Nav] verify OK: final dist={dist:.3f}m")
        # If the path was nudged (path[-1] != caller's final_goal), warn.
        nudge_dist = math.hypot(effective_goal[0] - final_goal[0],
                                effective_goal[1] - final_goal[1])
        if nudge_dist > 0.10:
            print(f"[Nav] note: caller's goal ({final_goal[0]:.2f},{final_goal[1]:.2f}) "
                  f"was nudged by plan.py to ({effective_goal[0]:.2f},{effective_goal[1]:.2f}) "
                  f"— Δ={nudge_dist:.2f}m")
        return True
