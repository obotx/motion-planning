"""
MORPH-I OMPL arm planner (4-DOF: [h1, h2, a1, theta]).

Builds an OMPL state space and validity checker for the MORPH arm using
runtime joint/actuator maps (no hardcoded qpos/ctrl indices) and a separate
MjData for planning so the live sim state is never modified.

API notes:
  - si.allocState() is required (ob.State constructor removed in OMPL 2.0).
  - isValid() must return Python bool, not numpy.bool_.
  - XML paths must be absolute for nested includes.

Usage:
    bridge = MORPHBridge("src/env/market_world_m1.xml", arm=1)
    bridge.park_all_pickup_objects()
    q_goal = bridge.solve_ik(obj_world_pos + [0, 0, 0.20])
    path   = bridge.plan(HOME_Q, q_goal, timeout=8.0)
"""

import os
import numpy as np
import mujoco
from ompl import base as ob
from ompl import geometric as og


# Physical constants
D2            = 0.1     # column lateral separation (m)
ALPHA_MIN_DEG = 10.0    # minimum arm tilt from horizontal (degrees)
INTERP_RES    = 0.02    # one waypoint per 2 cm of state-space path length

# OMPL state space bounds (SI units). a1 upper limit is set by the
# ArmLeftJoint actuator ctrlrange (executable, not commanded, range).
JOINT_RANGES_ARM = [
    (0.05,  1.35),    # h1: ColumnLeftBearingJoint
    (0.05,  1.35),    # h2: ColumnRightBearingJoint
    (0.0,   0.60),    # a1: ArmLeftJoint
    (-3.14, 3.14),    # theta: BaseJoint
]

# Reference arm configurations (SI units: m, rad)
HOME_Q = [0.5, 0.9, 0.1, 0.0]    # canonical home/rest pose for ARM1 OMPL
# Static pose for the inactive ARM2 (acts as an obstacle while ARM1 plans).
# Intentionally singular for ARM1 OMPL (h1==h2 → alpha=0°); never used as an
# ARM1 planning state.
PARK_Q = [1.2, 1.2, 0.1, 0.0]

# Scale applied to the optional kinematic-calibration LUT correction.
# Measured deflection (free pendulum) is much larger than runtime
# deflection (actuator-resisted), so we apply a fraction.
CALIB_SCALE = 0.20


class MORPHValidityChecker(ob.StateValidityChecker):
    """
    Two-layer validity checker for MORPH-I single arm (4-DOF).

    Layer 1 — geometry (fast, no MuJoCo):
        |arctan2(h2-h1, d2)| >= alpha_min_deg

    Layer 2 — MuJoCo collision (~0.25 ms):
        Set qpos → mj_forward → inspect contacts vs allowed set.
    """

    def __init__(self, si, model, planning_data, qpos_map,
                 d2=D2, alpha_min_deg=ALPHA_MIN_DEG, allowed_pairs=None):
        super().__init__(si)
        self._model    = model
        self._data     = planning_data
        self._qpos     = qpos_map
        self._d2       = d2
        self._alpha_sq = alpha_min_deg * alpha_min_deg
        self._allowed  = allowed_pairs if allowed_pairs is not None else set()

    def isValid(self, state):
        h1, h2, a1, theta = state[0], state[1], state[2], state[3]

        # Layer 1: geometry. Cast to Python bool — nanobind rejects numpy.bool_.
        alpha_deg = np.degrees(np.arctan2(h2 - h1, self._d2))
        if not bool(alpha_deg * alpha_deg >= self._alpha_sq):
            return False

        # Layer 2: MuJoCo collision
        self._data.qpos[self._qpos["ColumnLeft"]]  = h1
        self._data.qpos[self._qpos["ColumnRight"]] = h2
        self._data.qpos[self._qpos["ArmLeft"]]     = a1
        self._data.qpos[self._qpos["Base"]]        = theta
        mujoco.mj_forward(self._model, self._data)

        for i in range(self._data.ncon):
            c = self._data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if (g1, g2) not in self._allowed and (g2, g1) not in self._allowed:
                return False

        return True


class MORPHBridge:
    """
    High-level interface: load model, set up OMPL, plan arm paths.

    Args:
        xml_path:       path to the MuJoCo scene XML
        arm:            1 or 2
        alpha_min_deg:  minimum arm tilt for geometry constraint
        timeout:        default OMPL planning timeout (seconds)
    """

    def __init__(self, xml_path, arm=1, alpha_min_deg=ALPHA_MIN_DEG,
                 timeout=5.0, use_calibration=False):
        self._timeout = timeout
        self._arm     = arm

        xml_path        = os.path.abspath(xml_path)
        self._model     = mujoco.MjModel.from_xml_path(xml_path)
        self._plan_data = mujoco.MjData(self._model)

        # Runtime maps — safe against XML topology changes
        self._qpos_map = self._build_qpos_map(arm)
        self._ctrl_map = self._build_ctrl_map(arm)

        self._rest_pairs = self._build_rest_pairs()
        allowed          = set(self._rest_pairs)

        self._space   = self._build_space()
        self._si      = ob.SpaceInformation(self._space)
        self._checker = MORPHValidityChecker(
            self._si, self._model, self._plan_data,
            self._qpos_map, alpha_min_deg=alpha_min_deg,
            allowed_pairs=allowed,
        )
        self._si.setStateValidityChecker(self._checker)
        self._si.setup()

        # Calibration LUT for kinematic-vs-physics deflection.  Off by
        # default — the planner runs in its original uncorrected mode
        # unless the caller passes use_calibration=True (or play_m1 is
        # launched with --use-calib).  Generated offline by
        # tools/calibrate_arm_kinematics.py.
        if use_calibration:
            self._calib = self._load_calibration()
        else:
            self._calib = None
            print("[MORPHBridge] Calibration LUT disabled (default).  Pass "
                  "use_calibration=True or run play_m1 with --use-calib to "
                  "enable IK pre-correction.")

    # ── Calibration LUT ──────────────────────────────────────────────────────

    def _load_calibration(self):
        """Load `data/arm_calibration.npz` if present.  Returns dict of
        grids + error tensor, or None if file absent / malformed."""
        import os
        # Project root is two dirs up from this file (src/navigation/).
        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.abspath(os.path.join(here, "..", ".."))
        npz_path = os.path.join(root, "data", "arm_calibration.npz")
        if not os.path.exists(npz_path):
            print(f"[MORPHBridge] No calibration LUT at {npz_path} — "
                  "IK runs uncorrected.  Run tools/calibrate_arm_kinematics.py "
                  "to generate one for tighter physics-vs-IK agreement.")
            return None
        try:
            d = np.load(npz_path)
            calib = {
                'h1_grid': np.asarray(d['h1_grid'], dtype=float),
                'h2_grid': np.asarray(d['h2_grid'], dtype=float),
                'a1_grid': np.asarray(d['a1_grid'], dtype=float),
                'error':   np.asarray(d['error'],   dtype=float),
            }
            print(f"[MORPHBridge] Loaded calibration LUT {npz_path}  "
                  f"grid={len(calib['h1_grid'])}×{len(calib['h2_grid'])}×"
                  f"{len(calib['a1_grid'])}  "
                  f"max_err_xy={np.linalg.norm(calib['error'][..., :2], axis=-1).max()*100:.1f}cm")
            return calib
        except Exception as e:
            print(f"[MORPHBridge] Failed to load LUT {npz_path}: {e}")
            return None

    def _calib_error(self, h1, h2, a1, theta):
        """Compute the world-frame deflection correction for the given
        arm pose.  Two-step transform:

        1. Trilinear-interpolate the LUT at (h1, h2, a1).  This gives the
           deflection in the chassis-local frame at theta=0 (calibration
           reference orientation).
        2. Rotate by theta around Z (arm Base joint rotates the arm
           inside the chassis frame).
        3. Rotate by the actual chassis world rotation (from plan_data)
           since the IK target is in world frame and plan_data is set
           up with the runtime chassis pose by `reset_plan_data_for_ik`.

        Returns a 3-vector in world frame to subtract from the IK target.
        """
        if self._calib is None:
            return np.zeros(3, dtype=float)
        hg = self._calib['h1_grid']
        kg = self._calib['h2_grid']
        ag = self._calib['a1_grid']
        err = self._calib['error']

        def _interp_axis(grid, v):
            v = float(np.clip(v, grid[0], grid[-1]))
            i = int(np.searchsorted(grid, v) - 1)
            i = max(0, min(len(grid) - 2, i))
            t = (v - grid[i]) / max(1e-9, grid[i + 1] - grid[i])
            return i, float(t)

        i, ti = _interp_axis(hg, h1)
        j, tj = _interp_axis(kg, h2)
        k, tk = _interp_axis(ag, a1)
        # 8-corner trilinear blend → chassis-local deflection at theta=0
        out = np.zeros(3, dtype=float)
        for di, wi in ((0, 1 - ti), (1, ti)):
            for dj, wj in ((0, 1 - tj), (1, tj)):
                for dk, wk in ((0, 1 - tk), (1, tk)):
                    out += wi * wj * wk * err[i + di, j + dj, k + dk]

        # Apply the runtime scale factor — see CALIB_SCALE comment.  The
        # LUT measures unloaded passive deflection; runtime actuators
        # resist most of it, so a scale of ~0.2 typically matches reality.
        out = out * CALIB_SCALE

        # Rotate by theta about Z to account for arm rotation inside
        # the chassis frame (LUT was recorded at theta=0).
        c = np.cos(theta)
        s = np.sin(theta)
        in_chassis = np.array([
            c * out[0] - s * out[1],
            s * out[0] + c * out[1],
            out[2],
        ], dtype=float)

        # Rotate by chassis world rotation matrix.  plan_data was set
        # up with the runtime chassis pose before IK; xmat reflects it.
        try:
            chassis_bid = mujoco.mj_name2id(
                self._model, mujoco.mjtObj.mjOBJ_BODY, "robot")
            if chassis_bid >= 0:
                R = self._plan_data.xmat[chassis_bid].reshape(3, 3)
                return R @ in_chassis
        except Exception:
            pass
        return in_chassis

    # ── Map builders ─────────────────────────────────────────────────────────

    def _build_qpos_map(self, arm):
        m = self._model

        def qi(jname):
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                raise RuntimeError(f"Joint '{jname}' not found in model")
            return int(m.jnt_qposadr[jid])

        return {
            "ColumnLeft":  qi(f"ColumnLeftBearingJoint_{arm}"),
            "ColumnRight": qi(f"ColumnRightBearingJoint_{arm}"),
            "ArmLeft":     qi(f"ArmLeftJoint_{arm}"),
            "Base":        qi(f"BaseJoint_{arm}"),
        }

    def _build_ctrl_map(self, arm):
        m = self._model

        def ci(aname):
            aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, aname)
            if aid < 0:
                raise RuntimeError(f"Actuator '{aname}' not found in model")
            return int(aid)

        return {
            "ColumnLeft":  ci(f"ColumnLeftBearingJointMotor_{arm}"),
            "ColumnRight": ci(f"ColumnRightBearingJointMotor_{arm}"),
            "ArmLeft":     ci(f"ArmLeftJointMotor_{arm}"),
            "Base":        ci(f"BaseJointMotor_{arm}"),
        }

    # ── Rest-state contact baseline ───────────────────────────────────────────

    def _build_rest_pairs(self):
        """
        Build the set of structural contact pairs always present at rest.
        These are whitelisted by the validity checker.

        body_group() walks the parent chain looking for arm/base anchor bodies.
        """
        arm1_map = self._build_qpos_map(1)
        arm2_map = self._build_qpos_map(2)

        ref_states = [
            HOME_Q,
            [0.05, 0.5, 0.1, 0.0],   # low reference
        ]

        def body_group(body_id):
            cur = int(body_id)
            for _ in range(30):
                name = mujoco.mj_id2name(
                    self._model, mujoco.mjtObj.mjOBJ_BODY, cur) or ""
                if name == "Arm_1":                return "arm1"
                if name == "Arm_2":                return "arm2"
                if name in ("robot", "base"):      return "base"
                if cur == 0:                       break
                cur = int(self._model.body_parentid[cur])
            return "scene"

        pairs = set()
        for ref_q in ref_states:
            rd = mujoco.MjData(self._model)
            mujoco.mj_resetData(self._model, rd)

            rd.qpos[arm1_map["ColumnLeft"]]  = ref_q[0]
            rd.qpos[arm1_map["ColumnRight"]] = ref_q[1]
            rd.qpos[arm1_map["ArmLeft"]]     = ref_q[2]
            rd.qpos[arm1_map["Base"]]        = ref_q[3]

            rd.qpos[arm2_map["ColumnLeft"]]  = PARK_Q[0]
            rd.qpos[arm2_map["ColumnRight"]] = PARK_Q[1]
            rd.qpos[arm2_map["ArmLeft"]]     = PARK_Q[2]
            rd.qpos[arm2_map["Base"]]        = PARK_Q[3]

            mujoco.mj_forward(self._model, rd)

            for i in range(rd.ncon):
                c = rd.contact[i]
                g1, g2 = int(c.geom1), int(c.geom2)
                grp1 = body_group(int(self._model.geom_bodyid[g1]))
                grp2 = body_group(int(self._model.geom_bodyid[g2]))

                allow = (
                    (grp1 == grp2 and grp1 in ("arm1", "arm2", "base")) or
                    ({grp1, grp2} == {"base", "scene"}) or
                    (grp1 == "scene" and grp2 == "scene")
                )
                if allow:
                    pairs.add((g1, g2))
                    pairs.add((g2, g1))

        return pairs

    # ── OMPL state space ──────────────────────────────────────────────────────

    def _build_space(self):
        space  = ob.RealVectorStateSpace(4)
        bounds = ob.RealVectorBounds(4)
        for i, (lo, hi) in enumerate(JOINT_RANGES_ARM):
            bounds.setLow(i, lo)
            bounds.setHigh(i, hi)
        space.setBounds(bounds)
        return space

    # ── Planning ──────────────────────────────────────────────────────────────

    def plan(self, start_q, goal_q, timeout=None):
        """
        Plan a collision-free path from start_q to goal_q (4D).

        Args:
            start_q: [h1, h2, a1, theta]
            goal_q:  [h1, h2, a1, theta]
            timeout: seconds (instance default if None)

        Returns:
            List of [h1, h2, a1, theta] waypoints, or None if planning failed.
            Waypoint count ≈ path_length / INTERP_RES (min 10).
        """
        t = timeout if timeout is not None else self._timeout

        pdef  = ob.ProblemDefinition(self._si)
        start = self._si.allocState()
        goal  = self._si.allocState()
        for i in range(4):
            start[i] = float(start_q[i])
            goal[i]  = float(goal_q[i])
        pdef.setStartAndGoalStates(start, goal)

        planner = og.RRTConnect(self._si)
        planner.setProblemDefinition(pdef)
        planner.setup()

        if not planner.solve(t):
            return None

        path = pdef.getSolutionPath()
        og.PathSimplifier(self._si).simplifyMax(path)

        n_wps = max(10, int(path.length() / INTERP_RES))
        path.interpolate(n_wps)

        return [[path.getState(i)[j] for j in range(4)]
                for i in range(path.getStateCount())]

    # ── IK ────────────────────────────────────────────────────────────────────

    def solve_ik(self, target_pos, n_seeds=8, threshold=0.02):
        """
        Solve IK for Gripper_Link1_1 to reach target_pos (WORLD frame).

        Cost combines reach error with an asymmetric tilt penalty that prefers
        a level pose for mid-height targets and tilts up/down for high/low
        targets. Seeds mix a biased "low + extended" pattern with HOME_Q and
        random seeds for robustness across shelf and floor targets.
        """
        from scipy.optimize import minimize
        import math as _math

        gripper_bid = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, "Gripper_Link1_1")
        if gripper_bid < 0:
            raise RuntimeError("Body 'Gripper_Link1_1' not found in model")

        target_pos = np.asarray(target_pos, dtype=float)

        # Choose preferred tilt direction based on target z. Note that h2 only
        # affects body positions during mj_step (passive RotationLeftJoint), so
        # encoding tilt direction via this preference produces the right
        # physical tilt at runtime even though plan_data uses mj_forward only.
        WRIST_Z_MID = 1.0
        prefer_downward = bool(target_pos[2] < WRIST_Z_MID)

        def cost_fn(x):
            self._plan_data.qpos[self._qpos_map["ColumnLeft"]]  = x[0]
            self._plan_data.qpos[self._qpos_map["ColumnRight"]] = x[1]
            self._plan_data.qpos[self._qpos_map["ArmLeft"]]     = x[2]
            self._plan_data.qpos[self._qpos_map["Base"]]        = x[3]
            mujoco.mj_forward(self._model, self._plan_data)
            reach_err = float(np.linalg.norm(
                self._plan_data.xpos[gripper_bid] - target_pos))
            # Piecewise tilt penalty: linear-cheap for mild tilt so reach can
            # dominate, quadratic-expensive past TILT_KNEE so the optimizer
            # cannot pick extreme spreads the runtime arm would not track.
            # Wrong-direction tilt stays linearly heavy at all magnitudes.
            diff = float(x[1]) - float(x[0])    # h2 - h1, positive = downward tilt
            TILT_KNEE = 0.30
            TILT_QUAD_K = 1.0
            if prefer_downward:
                preferred = max(0.0, diff)        # downward
                opposed   = max(0.0, -diff)       # upward
            else:
                preferred = max(0.0, -diff)
                opposed   = max(0.0, diff)
            preferred_penalty = (
                0.005 * preferred
                + TILT_QUAD_K * max(0.0, preferred - TILT_KNEE) ** 2
            )
            opposed_penalty = 0.10 * opposed
            return reach_err + preferred_penalty + opposed_penalty

        # Theta guess: rotate arm toward target XY from the robot base in world.
        robot_bid = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, "robot")
        if robot_bid >= 0:
            rpos = self._plan_data.xpos[robot_bid]
            theta_guess = float(_math.atan2(target_pos[1] - rpos[1],
                                            target_pos[0] - rpos[0]))
        else:
            theta_guess = 0.0
        # Clamp into joint range
        theta_guess = max(-3.14, min(3.14, theta_guess))

        # Biased seeds: a "low + extended" pattern that works well for floor
        # objects, with several h1 values for higher-shelf targets.
        biased_seeds = [
            np.array([0.05, 0.10, 0.60, theta_guess]),
            np.array([0.10, 0.15, 0.60, theta_guess]),
            np.array([0.15, 0.20, 0.55, theta_guess]),
            np.array([0.20, 0.25, 0.55, theta_guess]),
        ]
        seeds = biased_seeds + [np.array(HOME_Q, dtype=float)]
        rng = np.random.default_rng()
        for _ in range(max(0, n_seeds - len(seeds))):
            seeds.append(np.array([rng.uniform(lo, hi) for lo, hi in JOINT_RANGES_ARM]))

        best_q, best_err = None, float('inf')
        for seed in seeds:
            res = minimize(cost_fn, seed, method='SLSQP',
                           bounds=JOINT_RANGES_ARM, tol=1e-4)
            # Threshold on pure reach error (cost_fn also includes tilt penalty).
            self._plan_data.qpos[self._qpos_map["ColumnLeft"]]  = res.x[0]
            self._plan_data.qpos[self._qpos_map["ColumnRight"]] = res.x[1]
            self._plan_data.qpos[self._qpos_map["ArmLeft"]]     = res.x[2]
            self._plan_data.qpos[self._qpos_map["Base"]]        = res.x[3]
            mujoco.mj_forward(self._model, self._plan_data)
            reach_err = float(np.linalg.norm(
                self._plan_data.xpos[gripper_bid] - target_pos))
            # Rank by cost_fn (includes tilt penalty) for tie-breaking.
            if res.fun < best_err and self.is_valid(res.x) and reach_err <= threshold + 1e-6:
                best_err = res.fun
                best_q   = res.x.tolist()
            if best_err < 0.005:
                break

        if best_q is None:
            raise RuntimeError(
                f"IK failed: target={target_pos}, best_err=infm "
                f"(threshold={threshold}m). Consider increasing n_seeds or threshold.")

        return best_q

    def solve_ik_with_z_lift(self, target_pos, n_seeds=12,
                             max_lift=0.40, step=0.05, threshold=0.04):
        """
        Strict-IK variant for grasp/pick targets: never returns a solution
        for a target below the requested z. If the requested target_z is
        unreachable, lifts z by `step` and retries up to `max_lift`.

        Returns (q, actual_target) so the caller knows where the wrist will
        actually be. Unlike solve_ik_robust, never silently downgrades z.
        """
        target_pos = np.asarray(target_pos, dtype=float)
        last_err = None
        n_steps = max(1, int(round(max_lift / step))) + 1
        for i in range(n_steps):
            lift = i * step
            probe = target_pos.copy()
            probe[2] += lift
            try:
                q = self.solve_ik(probe, n_seeds=n_seeds, threshold=threshold)
                return q, probe
            except RuntimeError as e:
                last_err = e
        raise RuntimeError(
            f"solve_ik_with_z_lift: pre-grasp unreachable up to "
            f"+{max_lift:.2f}m of z. Last: {last_err}")

    def solve_ik_with_z_lift_link3(self, target_pos, n_seeds=12,
                                   max_iters=3, tol=0.005):
        """Z-lift IK aimed at Gripper_Link3_1 (palm) instead of Link1.

        The base solve_ik aims Link1 at the target.  The palm (Link3),
        where the fingers attach, sits a few centimetres past Link1
        along the wrist axis — so when the palm is what we care about
        (i.e. for grasp targeting), aiming Link1 at the requested
        target lands the palm short of the object.

        This method iteratively compensates: solve IK, measure the
        runtime Link1→Link3 vector at the resulting pose, subtract it
        from the target, re-solve.  Two or three passes are usually
        enough because the offset rotates with theta, and theta barely
        changes between iterations.

        Returns (q, link3_actual_target).  link3_actual_target is the
        world XYZ where the palm will actually land (Link1_actual +
        Link1→Link3 measured at q), accounting for any z-lift the
        wrapper applied to keep the target reachable.
        """
        link1_bid = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, "Gripper_Link1_1")
        link3_bid = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, "Gripper_Link3_1")
        if link1_bid < 0 or link3_bid < 0:
            # Fall back to the regular Link1-aimed IK if either body is
            # missing from the model (defensive — both are required by
            # the rest of the pipeline).
            return self.solve_ik_with_z_lift(target_pos, n_seeds=n_seeds)

        # GRIPPER_STANDOFF_XY already accounts for the rigid Link1→Link3
        # offset, so we only subtract `calib_corr` (passive-deflection
        # correction from the LUT, scaled by CALIB_SCALE).
        original_target = np.asarray(target_pos, dtype=float).copy()
        adjusted_target = original_target.copy()
        last_q = None
        last_link1_actual = None
        last_calib_corr = np.zeros(3, dtype=float)

        for it in range(max_iters):
            q, link1_actual = self.solve_ik_with_z_lift(
                adjusted_target, n_seeds=n_seeds)
            self._plan_data.qpos[self._qpos_map["ColumnLeft"]]  = q[0]
            self._plan_data.qpos[self._qpos_map["ColumnRight"]] = q[1]
            self._plan_data.qpos[self._qpos_map["ArmLeft"]]     = q[2]
            self._plan_data.qpos[self._qpos_map["Base"]]        = q[3]
            mujoco.mj_forward(self._model, self._plan_data)
            calib_corr = self._calib_error(q[0], q[1], q[2], q[3])
            new_adjusted = original_target - calib_corr
            residual = float(np.linalg.norm(new_adjusted - adjusted_target))
            adjusted_target = new_adjusted
            last_q = q
            last_link1_actual = np.asarray(link1_actual, dtype=float)
            last_calib_corr = calib_corr
            if residual < tol:
                break

        # Reported actual-target: where Link3 (palm) will physically
        # land = Link1_actual + rigid Link1→Link3 offset + calib_corr.
        link3_minus_link1 = (self._plan_data.xpos[link3_bid].copy()
                             - self._plan_data.xpos[link1_bid].copy())
        link3_actual_target = (last_link1_actual
                               + link3_minus_link1
                               + last_calib_corr)
        return last_q, link3_actual_target

    def solve_ik_robust(self, target_pos, n_seeds=12, z_perturbs=None):
        """
        Retry IK with z-offset perturbations when the exact target is unreachable.

        Tries target_pos, then target_pos+dz for each dz in z_perturbs.
        Useful for shelf heights that fall just outside the kinematic envelope.
        """
        if z_perturbs is None:
            z_perturbs = [0.0, 0.05, -0.05, 0.10, -0.10, 0.15]
        target_pos = np.asarray(target_pos, dtype=float)
        last_err = None
        for dz in z_perturbs:
            try:
                probe = target_pos.copy()
                probe[2] += dz
                return self.solve_ik(probe, n_seeds=n_seeds, threshold=0.04)
            except RuntimeError as e:
                last_err = e
        raise RuntimeError(
            f"solve_ik_robust: all {len(z_perturbs)} probes failed. Last: {last_err}")

    # ── Validity helpers ──────────────────────────────────────────────────────

    def is_valid(self, q):
        """Check if a single 4D config is collision-free."""
        s = self._si.allocState()
        for i in range(4):
            s[i] = float(q[i])
        return self._si.isValid(s)

    def update_allowed_pairs(self, extra_pairs):
        """Add extra allowed pairs (e.g. gripper touching carried object in plan_data)."""
        self._checker._allowed = set(self._rest_pairs) | set(extra_pairs)

    # ── Scene management in plan_data ────────────────────────────────────────

    def park_body(self, body_name, pos=(0.0, 0.0, 100.0)):
        """
        Move a free-joint body to pos in plan_data (not sim_data).

        Use before pre-approach planning: park the target object so the arm
        can reach near it without the validity checker flagging a collision.
        Use DIFFERENT XY offsets per object so parked objects don't collide.
        """
        bid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid < 0:
            raise ValueError(f"Body '{body_name}' not found in model")
        ja = int(self._model.body_jntadr[bid])
        if ja < 0:
            raise ValueError(f"Body '{body_name}' has no joint")
        qa = int(self._model.jnt_qposadr[ja])
        self._plan_data.qpos[qa]     = float(pos[0])
        self._plan_data.qpos[qa + 1] = float(pos[1])
        self._plan_data.qpos[qa + 2] = float(pos[2])
        self._plan_data.qpos[qa + 3] = 1.0
        self._plan_data.qpos[qa + 4] = 0.0
        self._plan_data.qpos[qa + 5] = 0.0
        self._plan_data.qpos[qa + 6] = 0.0
        mujoco.mj_forward(self._model, self._plan_data)

    def park_all_pickup_objects(self, n=10):
        """
        Park all pickup_obj_0..N-1 above the scene in plan_data.

        Call before any planning so floor objects don't trigger false collisions.
        Use staggered XY so parked objects don't collide each other.
        """
        for i in range(n):
            try:
                self.park_body(f"pickup_obj_{i}",
                               pos=(float(i) * 3.0, 50.0, 100.0))
            except ValueError:
                pass

    def sync_base_pose_from_sim(self, sim_data):
        """
        Copy the robot base freejoint pose from sim_data into plan_data.

        Call this before any arm planning so the validity checker's world-frame
        contacts reflect the robot's current position in the scene.
        """
        robot_bid = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, "robot")
        if robot_bid < 0:
            return
        ja = int(self._model.body_jntadr[robot_bid])
        if ja < 0:
            return
        qa = int(self._model.jnt_qposadr[ja])
        # freejoint qpos: [x, y, z, qw, qx, qy, qz]
        self._plan_data.qpos[qa:qa + 7] = sim_data.qpos[qa:qa + 7]
        mujoco.mj_forward(self._model, self._plan_data)

    def set_base_pose_xy_yaw(self, x, y, yaw, z=None):
        """
        Set a hypothetical robot base freejoint pose in plan_data.

        Used for virtual base-candidate screening before moving the real robot.
        Keeps current/freejoint z unless z is provided.
        """
        robot_bid = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, "robot")
        if robot_bid < 0:
            return
        ja = int(self._model.body_jntadr[robot_bid])
        if ja < 0:
            return
        qa = int(self._model.jnt_qposadr[ja])
        self._plan_data.qpos[qa + 0] = float(x)
        self._plan_data.qpos[qa + 1] = float(y)
        if z is not None:
            self._plan_data.qpos[qa + 2] = float(z)
        half = 0.5 * float(yaw)
        self._plan_data.qpos[qa + 3] = float(np.cos(half))
        self._plan_data.qpos[qa + 4] = 0.0
        self._plan_data.qpos[qa + 5] = 0.0
        self._plan_data.qpos[qa + 6] = float(np.sin(half))
        mujoco.mj_forward(self._model, self._plan_data)

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def model(self):
        return self._model

    @property
    def planning_data(self):
        return self._plan_data

    @property
    def qpos_map(self):
        return self._qpos_map

    @property
    def ctrl_map(self):
        return self._ctrl_map


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import time

    print("\n" + "=" * 65)
    print("arm_planner.py self-test  —  MORPHBridge dof=4")
    print("=" * 65)

    candidates = [
        "src/env/market_world_m1.xml",
        "src/env/market_world.xml",
    ]
    xml_path = None
    for c in candidates:
        if os.path.exists(c):
            xml_path = c
            break
    if xml_path is None:
        print("ERROR: scene XML not found. Run from motion-planning/ directory.")
        sys.exit(1)

    print(f"\nLoading: {xml_path}")
    bridge = MORPHBridge(xml_path, arm=1)
    m = bridge.model
    print(f"Model: nq={m.nq}  nu={m.nu}  nbody={m.nbody}  ngeom={m.ngeom}")
    print(f"qpos_map:  {bridge.qpos_map}")
    print(f"ctrl_map:  {bridge.ctrl_map}")
    canonical = sorted({(min(a, b), max(a, b)) for a, b in bridge._rest_pairs})
    print(f"Rest pairs ({len(canonical)} unique): {canonical}")

    bridge.park_all_pickup_objects()
    print("  [scene] All pickup objects parked in plan_data")

    # ── Validity checks ───────────────────────────────────────────────────────
    print("\n--- Validity checks ---")
    checks = [
        ("HOME_Q (expect valid)",              HOME_Q,               True),
        ("Singular h1==h2 (expect invalid)",   [0.5, 0.5, 0.2, 0.0], False),
        ("Low arm, valid height diff",          [0.3, 0.8, 0.1, 0.0], True),
        # PARK_Q is the ARM2 static pose; ARM1-singular by design.
        ("PARK_Q (expect ARM1-invalid; ARM2 static only)", PARK_Q, False),
    ]
    all_ok = True
    for desc, q, expect in checks:
        result = bridge.is_valid(q)
        ok = (result == expect)
        if not ok:
            all_ok = False
        print(f"  {'✓' if ok else '✗ FAIL'}  {desc}")

    # ── Planning ──────────────────────────────────────────────────────────────
    print("\n--- Planning tests ---")
    plan_cases = [
        ("Vertical transit",         [0.2, 0.5, 0.2, 0.0], [0.8, 1.1, 0.2, 0.0]),
        ("Height + reach + yaw",     [0.3, 0.7, 0.1, 0.0], [0.6, 1.0, 0.4, 1.0]),
        # High tilted retract target (non-singular ARM1 parked pose).
        ("HOME_Q → high tilted",     HOME_Q,                [0.9, 1.3, 0.05, 0.0]),
    ]
    for desc, sq, gq in plan_cases:
        t0   = time.time()
        path = bridge.plan(sq, gq, timeout=5.0)
        dt   = time.time() - t0
        if path is None:
            print(f"  ✗ FAIL  {desc}  (no path in {dt:.2f}s)")
            all_ok = False
        else:
            print(f"  ✓  {desc}  →  {len(path)} waypoints in {dt:.3f}s")

    # ── Performance ───────────────────────────────────────────────────────────
    print("\n--- Performance: isValid() ---")
    N  = 200
    qs = np.column_stack([
        np.random.uniform(0.1, 1.3, N),
        np.random.uniform(0.1, 1.3, N),
        np.random.uniform(0.0, 0.6, N),
        np.zeros(N),
    ])
    t0 = time.time()
    for q in qs:
        bridge.is_valid(q)
    dt_ms   = (time.time() - t0) / N * 1000
    perf_ok = dt_ms < 2.0
    print(f"  {dt_ms:.3f} ms/call  ({1000 / dt_ms:.0f} checks/sec)")
    print(f"  {'✓ fast enough' if perf_ok else '✗ too slow'} (threshold: 2 ms)")

    print()
    if all_ok and perf_ok:
        print("✓ arm_planner self-test PASSED")
        sys.exit(0)
    else:
        print("✗ arm_planner self-test FAILED")
        sys.exit(1)
