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
# Minimum arm tilt from horizontal — the validity check rejects poses
# where the parallel manipulator approaches singular (boom exactly
# horizontal at h1 == h2). Lowered through 10° → 3° → 1.5° as the
# side-grip mode emerged: the free-move screenshot showed a stable
# pose at 1.15° tilt (h2-h1 = -0.002 m), proving the parallel chain
# holds firmly with the column springs (stiffness=58000, damping=2000)
# even very close to singular. 1.5° corresponds to a 0.0026 m h2-h1
# differential — still away from the true singularity at 0° but
# letting the validity check accept the near-level poses side-grip
# needs. Top-down poses naturally return |h2-h1| ≥ 0.03 m (alpha ≈
# 17°), so they remain comfortably above this floor.
ALPHA_MIN_DEG = 1.5
INTERP_RES    = 0.02    # one waypoint per 2 cm of state-space path length

# IK debug printing toggle. When False the per-z-step / per-seed
# dumps and the "won by seed#X" lines are suppressed. Failure dumps
# (no valid solution found) still print regardless. Reduces log
# noise + I/O blocking on the main thread during heavy IK loops.
IK_DEBUG_VERBOSE = False

# Planning-only safety clearance between the arm/gripper and the
# robot chassis. The runtime sim model is unchanged.
# Mechanism: inflate `geom_margin` on chassis geoms in the planning
# model. MuJoCo then reports any geom-pair within margin in the
# contact list with a signed `dist` field. isValid() checks `dist`
# instead of just contact presence — for non-rest geom pairs we
# require `dist >= MIN_NONREST_CLEARANCE`. Rest-pair whitelist still
# applies (wheels-floor, intra-chassis, intra-arm contacts stay
# allowed).
# Why this matters:
# - Strict-binary validity (no margin) accepts a path that grazes
# the chassis at 0.1 mm clearance. Runtime physics + chassis
# yaw/translation during the pick then push the arm into actual
# contact, and the arm fights the chassis instead of reaching obj.
# - With a 1.2 cm enforced gap, the planner naturally routes around
# the chassis envelope, leaving room for runtime perturbations.
CHASSIS_PLAN_MARGIN     = 0.025  # margin (m) added to chassis geoms so
                                 # near-contacts within this band show
                                 # up in ncon (only enables visibility)
MIN_NONREST_CLEARANCE   = 0.003  # required signed distance for non-rest
                                 # pairs. c.dist >= this → valid;
                                 # below → reject (path too close).
                                 # Tried 1 mm — arm physically clipped
                                 # the LifePo4 battery body. 3 mm
                                 # keeps a real safety buffer while
                                 # still letting the IK fold reasonably
                                 # low.

# OMPL state space bounds (SI units). a1 upper limit is set by the
# ArmLeftJoint actuator ctrlrange (executable, not commanded, range).
# 8-DOF state: 4 arm slides + 4 wrist orientation joints. The wrist
# joints (HandBearingJoint, gripper_z/x/y_rotation) were always actuated
# in the model but ignored by the original 4-DOF planner; integrating
# them lets OMPL plan the full arm+wrist trajectory and lets the wrist
# tilt to a real top-down pose for proper 3-finger pickup.
JOINT_RANGES_ARM = [
    (0.05,  1.35),    # 0: h1   ColumnLeftBearingJoint
    (0.05,  1.35),    # 1: h2   ColumnRightBearingJoint
    (0.0,   0.60),    # 2: a1   ArmLeftJoint
    (-3.14, 3.14),    # 3: th   BaseJoint
    (-1.57, 1.57),    # 4: hb   HandBearingJoint (wrist pitch; XML ±1.5708)
    (-3.14, 3.14),    # 5: wz   gripper_z_rotation (palm roll; full rev required
                      # for side-grip orientations)
    (-0.80, 0.80),    # 6: wx   gripper_x_rotation (actuator ctrlrange ±0.8)
    (-0.80, 0.80),    # 7: wy   gripper_y_rotation (actuator ctrlrange ±0.8)
]
ARM_DOF = len(JOINT_RANGES_ARM)
WRIST_NEUTRAL = (0.0, 0.0, 0.0, 0.0)   # hb, wz, wx, wy at neutral pose

# Reference arm configurations (SI units: m, rad). Extended to 8-DOF;
# wrist defaults to neutral so 4-vector legacy callers behave identically.
HOME_Q = [0.5, 0.9, 0.1, 0.0,  0.0, 0.0, 0.0, 0.0]
# Static pose for the inactive ARM2 (acts as an obstacle while ARM1 plans).
# Intentionally singular for ARM1 OMPL (h1==h2 → alpha=0°); never used as an
# ARM1 planning state.
PARK_Q = [1.2, 1.2, 0.1, 0.0,  0.0, 0.0, 0.0, 0.0]


def _pad_q(q):
    """
    Accept a 4-vector (arm-only) or 8-vector (arm + wrist) and return an
    8-vector with wrist defaulting to WRIST_NEUTRAL.  Lets legacy callers
    continue passing 4-vectors during the 4→8-DOF transition.
    """
    if q is None:
        return None
    q = list(q)
    if len(q) == ARM_DOF:
        return [float(v) for v in q]
    if len(q) == 4:
        return [float(v) for v in q] + list(WRIST_NEUTRAL)
    raise ValueError(
        f"Arm q must have length 4 or {ARM_DOF}, got {len(q)}")

# Scale applied to the optional kinematic-calibration LUT correction.
# The LUT is recorded with both passive joints free, zero active-joint
# stiffness, and actuator ctrl matched to qpos — so it stores
# actuator-resisted deflection directly and the runtime scale is 1.0.
CALIB_SCALE = 1.0


class MORPHValidityChecker(ob.StateValidityChecker):
    """
    Two-layer validity checker for MORPH-I single arm (8-DOF: 4 arm
    slides + 4 wrist orientation joints).

    Layer 1 — geometry (fast, no MuJoCo):
        |arctan2(h2-h1, d2)| >= alpha_min_deg
        (depends only on h1, h2; wrist joints don't change arm tilt.)

    Layer 2 — MuJoCo collision (~0.25 ms):
        Set all 8 qpos entries → mj_forward → inspect contacts vs
        allowed set.  The wrist qpos writes let the collision check
        see the actual gripper orientation, so OMPL never plans a
        wrist tilt that drives the palm into nearby geometry.

        Clearance-aware: rest pairs (whitelisted at HOME) pass
        unconditionally; non-rest pairs require c.dist >=
        MIN_NONREST_CLEARANCE.  Combined with chassis-margin
        inflation, this gives the planner a real safety buffer
        instead of the previous binary contact check.
    """

    def __init__(self, si, model, planning_data, qpos_map,
                 d2=D2, alpha_min_deg=ALPHA_MIN_DEG, allowed_pairs=None,
                 min_nonrest_clearance=MIN_NONREST_CLEARANCE):
        super().__init__(si)
        self._model    = model
        self._data     = planning_data
        self._qpos     = qpos_map
        self._d2       = d2
        self._alpha_sq = alpha_min_deg * alpha_min_deg
        self._allowed  = allowed_pairs if allowed_pairs is not None else set()
        self._min_clearance = float(min_nonrest_clearance)

    def isValid(self, state):
        h1, h2, a1, theta = state[0], state[1], state[2], state[3]
        hb, wz, wx, wy    = state[4], state[5], state[6], state[7]

        # Layer 1: geometry. Cast to Python bool — nanobind rejects numpy.bool_.
        alpha_deg = np.degrees(np.arctan2(h2 - h1, self._d2))
        if not bool(alpha_deg * alpha_deg >= self._alpha_sq):
            return False

        # Layer 2: MuJoCo collision
        self._data.qpos[self._qpos["ColumnLeft"]]  = h1
        self._data.qpos[self._qpos["ColumnRight"]] = h2
        self._data.qpos[self._qpos["ArmLeft"]]     = a1
        self._data.qpos[self._qpos["Base"]]        = theta
        self._data.qpos[self._qpos["HandBearing"]] = hb
        self._data.qpos[self._qpos["WristZ"]]      = wz
        self._data.qpos[self._qpos["WristX"]]      = wx
        self._data.qpos[self._qpos["WristY"]]      = wy
        mujoco.mj_forward(self._model, self._data)

        for i in range(self._data.ncon):
            c = self._data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if (g1, g2) in self._allowed or (g2, g1) in self._allowed:
                # Whitelisted rest pair — always accepted regardless of dist
                continue
            # Non-rest pair: enforce signed-distance clearance. c.dist
            # is negative for penetration, 0 for touching, positive
            # within margin. Anything closer than min_clearance
            # (typ. 1.2 cm) is treated as too close → reject path.
            if c.dist < self._min_clearance:
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
                 timeout=5.0, use_calibration=False,
                 calib_wrist_mode="sidegrip"):
        self._timeout = timeout
        self._arm     = arm

        xml_path        = os.path.abspath(xml_path)
        self._model     = mujoco.MjModel.from_xml_path(xml_path)
        self._plan_data = mujoco.MjData(self._model)

        # Runtime maps — safe against XML topology changes
        self._qpos_map = self._build_qpos_map(arm)
        self._ctrl_map = self._build_ctrl_map(arm)

        # Planning-only clearance: inflate chassis geom_margin in the
        # planning model so the validity checker sees arm-vs-chassis
        # near-contacts (up to CHASSIS_PLAN_MARGIN) in ncon. isValid()
        # then enforces MIN_NONREST_CLEARANCE for any non-rest pair.
        # Runtime sim model is untouched — this is a SEPARATE MjModel.
        self._chassis_geom_ids = self._collect_chassis_geom_ids()
        for gid in self._chassis_geom_ids:
            self._model.geom_margin[gid] = CHASSIS_PLAN_MARGIN
        print(f"[ArmPlanner] planning-only chassis clearance: "
              f"margin={CHASSIS_PLAN_MARGIN*100:.1f}cm on "
              f"{len(self._chassis_geom_ids)} chassis geoms; "
              f"min non-rest clearance="
              f"{MIN_NONREST_CLEARANCE*100:.1f}cm")

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

        # Calibration LUT for kinematic-vs-physics deflection. Off by
        # default — the planner runs in its original uncorrected mode
        # unless the caller passes use_calibration=True (or play_m1 is
        # launched with --use-calib). Mode-specific LUTs are stored
        # at data/arm_calibration_<mode>.npz; `calib_wrist_mode`
        # selects which one to load (defaults to sidegrip — the
        # current STRICT-mode configuration). Generated offline by
        # tools/calibrate_arm_kinematics.py.
        self._calib_wrist_mode = calib_wrist_mode
        if use_calibration:
            self._calib = self._load_calibration(calib_wrist_mode)
        else:
            self._calib = None
            print("[MORPHBridge] Calibration LUT disabled (default).  Pass "
                  "use_calibration=True or run play_m1 with --use-calib to "
                  "enable IK pre-correction.")

    # ── Calibration LUT ──────────────────────────────────────────────────────

    def _load_calibration(self, wrist_mode):
        """Load mode-specific LUT from `data/arm_calibration_<mode>.npz`.
        Falls back to the legacy `arm_calibration.npz` if the mode-
        specific file is absent (for backward compatibility with the
        4-DOF era LUT).  Returns dict of grids + error tensor, or None
        if no file is loadable."""
        import os
        # Project root is two dirs up from this file (src/navigation/).
        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.abspath(os.path.join(here, "..", ".."))
        # Try mode-specific first, then legacy.
        candidates = [
            os.path.join(root, "data", f"arm_calibration_{wrist_mode}.npz"),
            os.path.join(root, "data", "arm_calibration.npz"),
        ]
        npz_path = None
        for path in candidates:
            if os.path.exists(path):
                npz_path = path
                break
        if npz_path is None:
            print(f"[MORPHBridge] No calibration LUT found.  Looked for:")
            for path in candidates:
                print(f"    {path}")
            print("    IK runs uncorrected.  Run "
                  f"tools/calibrate_arm_kinematics.py --wrist-mode {wrist_mode} "
                  "to generate one.")
            return None
        try:
            d = np.load(npz_path)
            calib = {
                'h1_grid': np.asarray(d['h1_grid'], dtype=float),
                'h2_grid': np.asarray(d['h2_grid'], dtype=float),
                'a1_grid': np.asarray(d['a1_grid'], dtype=float),
                'error':   np.asarray(d['error'],   dtype=float),
            }
            # Optional metadata fields (present in newer LUTs).
            lut_mode = None
            try:
                lut_mode = str(d['wrist_mode']) if 'wrist_mode' in d.files else None
            except Exception:
                lut_mode = None
            mode_str = f" mode={lut_mode}" if lut_mode else ""
            mismatch = (lut_mode is not None and lut_mode != wrist_mode)
            mismatch_warn = (" WARNING: LUT was built for "
                             f"'{lut_mode}' but requested '{wrist_mode}' —"
                             " calibration may be inaccurate" if mismatch
                             else "")
            print(f"[MORPHBridge] Loaded calibration LUT {npz_path}{mode_str}  "
                  f"grid={len(calib['h1_grid'])}×{len(calib['h2_grid'])}×"
                  f"{len(calib['a1_grid'])}  "
                  f"max_err_xy="
                  f"{np.linalg.norm(calib['error'][..., :2], axis=-1).max()*100:.1f}cm  "
                  f"max_err_z="
                  f"{np.abs(calib['error'][..., 2]).max()*100:.1f}cm"
                  f"{mismatch_warn}")
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

        # Apply the runtime scale factor — see CALIB_SCALE comment. The
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

        # Rotate by chassis world rotation matrix. plan_data was set
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

    def _collect_chassis_geom_ids(self):
        """Return geom ids belonging to the chassis/base subtree
        (descendants of the 'robot' or 'base' body), EXCLUDING any
        geoms in the arm subtrees ('Arm_1', 'Arm_2').

        These are the geoms whose `geom_margin` we inflate in the
        planning model so the validity checker can see arm-vs-chassis
        near-contacts.  Walks each geom's body up the parent chain;
        if an arm ancestor is found first it's an arm geom, if a
        chassis ancestor is found first it's a chassis geom.
        """
        m = self._model
        ids = []
        for gid in range(m.ngeom):
            cur = int(m.geom_bodyid[gid])
            is_chassis = False
            for _ in range(30):
                name = mujoco.mj_id2name(
                    m, mujoco.mjtObj.mjOBJ_BODY, cur) or ""
                if name in ("Arm_1", "Arm_2"):
                    is_chassis = False
                    break
                if name in ("robot", "base"):
                    is_chassis = True
                    break
                if cur == 0:
                    break
                cur = int(m.body_parentid[cur])
            if is_chassis:
                ids.append(gid)
        return ids

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
            "HandBearing": qi(f"HandBearingJoint_{arm}"),
            "WristZ":      qi(f"gripper_z_rotation_{arm}"),
            "WristX":      qi(f"gripper_x_rotation_{arm}"),
            "WristY":      qi(f"gripper_y_rotation_{arm}"),
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
            _pad_q(HOME_Q),
            _pad_q([0.05, 0.5, 0.1, 0.0]),   # low reference
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
            rd.qpos[arm1_map["HandBearing"]] = ref_q[4]
            rd.qpos[arm1_map["WristZ"]]      = ref_q[5]
            rd.qpos[arm1_map["WristX"]]      = ref_q[6]
            rd.qpos[arm1_map["WristY"]]      = ref_q[7]

            park_q = _pad_q(PARK_Q)
            rd.qpos[arm2_map["ColumnLeft"]]  = park_q[0]
            rd.qpos[arm2_map["ColumnRight"]] = park_q[1]
            rd.qpos[arm2_map["ArmLeft"]]     = park_q[2]
            rd.qpos[arm2_map["Base"]]        = park_q[3]
            rd.qpos[arm2_map["HandBearing"]] = park_q[4]
            rd.qpos[arm2_map["WristZ"]]      = park_q[5]
            rd.qpos[arm2_map["WristX"]]      = park_q[6]
            rd.qpos[arm2_map["WristY"]]      = park_q[7]

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
        space  = ob.RealVectorStateSpace(ARM_DOF)
        bounds = ob.RealVectorBounds(ARM_DOF)
        for i, (lo, hi) in enumerate(JOINT_RANGES_ARM):
            bounds.setLow(i, lo)
            bounds.setHigh(i, hi)
        space.setBounds(bounds)
        return space

    # ── Planning ──────────────────────────────────────────────────────────────

    def plan(self, start_q, goal_q, timeout=None):
        """
        Plan a collision-free path from start_q to goal_q (8-DOF: 4 arm
        slides + 4 wrist orientation joints).

        Accepts 4-vectors (legacy callers) and pads to 8 with neutral
        wrist for the transition period.

        Args:
            start_q: [h1, h2, a1, th, hb, wz, wx, wy]  (or 4-vec)
            goal_q:  [h1, h2, a1, th, hb, wz, wx, wy]  (or 4-vec)
            timeout: seconds (instance default if None)

        Returns:
            List of 8-element waypoints, or None if planning failed.
            Waypoint count ≈ path_length / INTERP_RES (min 10).
        """
        t = timeout if timeout is not None else self._timeout
        sq = _pad_q(start_q)
        gq = _pad_q(goal_q)

        pdef  = ob.ProblemDefinition(self._si)
        start = self._si.allocState()
        goal  = self._si.allocState()
        for i in range(ARM_DOF):
            start[i] = float(sq[i])
            goal[i]  = float(gq[i])
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

        return [[path.getState(i)[j] for j in range(ARM_DOF)]
                for i in range(path.getStateCount())]

    # ── IK ────────────────────────────────────────────────────────────────────

    def solve_ik(self, target_pos, n_seeds=8, threshold=0.02,
                 wrist_goal=None, wrist_weight=5.0,
                 target_body="Gripper_Link1_1",
                 seed_q=None):
        """
        Solve 8-DOF IK to reach target_pos (WORLD frame) with `target_body`.

        Cost combines reach error, tilt penalty, and a wrist preference
        term.  When `target_body` is `Gripper_Link3_1` (the palm) and
        `wrist_weight` is low on HandBearing, SLSQP uses HandBearing
        as a 5th positional DOF: rotating it swings the palm by 15-20
        cm around the Hand_Bearing pivot.  The other wrist DOFs
        (gripper_z/x/y_rotation) are typically held tight at the goal
        because they set the gripper-frame ORIENTATION (palm leveling,
        thumb-vs-fingers spin) and shouldn't drift.

        Args:
            target_pos:    (x, y, z) world target for `target_body`.
            n_seeds:       SLSQP seed count.
            threshold:     accept solutions with reach error <= threshold.
            wrist_goal:    (hb, wz, wx, wy) desired wrist pose; None → neutral.
            wrist_weight:  λ on the wrist preference term.  Accepts either
                           a scalar (applies to all 4 wrist dims) or a
                           4-tuple (hb_w, wz_w, wx_w, wy_w) for per-DOF
                           weights.  Use per-DOF for side-grip: low HB
                           weight (0.1) lets HandBearing be a reach DOF,
                           high wz/wx/wy weights (3.0) lock the gripper
                           orientation in place.
            target_body:   "Gripper_Link1_1" (wrist) for legacy/top-down,
                           "Gripper_Link3_1" (palm) for grasp targeting.

        Returns 8-element list [h1, h2, a1, th, hb, wz, wx, wy].
        """
        from scipy.optimize import minimize
        import math as _math

        gripper_bid = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, target_body)
        if gripper_bid < 0:
            raise RuntimeError(f"Body '{target_body}' not found in model")

        # Normalise wrist_weight to a 4-tuple.
        if np.isscalar(wrist_weight):
            ww = np.array([wrist_weight] * 4, dtype=float)
        else:
            ww = np.asarray(wrist_weight, dtype=float)
            if ww.shape != (4,):
                raise ValueError(
                    "wrist_weight must be a scalar or 4-tuple, got "
                    f"shape {ww.shape}")

        target_pos = np.asarray(target_pos, dtype=float)
        wg = (np.asarray(wrist_goal, dtype=float)
              if wrist_goal is not None
              else np.asarray(WRIST_NEUTRAL, dtype=float))

        # Side-grip detection: when HandBearing goal is near 0, the
        # gripper is held palm-horizontal and the right arm pose is a
        # LEVEL one (h1 ≈ h2) at mid-column height so the wrist hangs
        # below the trolley to obj-level. When hb is significantly
        # negative the gripper tilts down — the arm wants a
        # DOWNWARD-TILTED pose (h2 > h1) and lower columns so the
        # wrist sits above the obj for the descent.
        side_grip_mode = bool(abs(float(wg[0])) < 0.20)

        # Tilt-preference direction. For diagonal/top-down (legacy
        # behaviour) we keep the "low target → tilt down" heuristic.
        # For side grip we want LEVEL (no h2-h1 deviation), so the
        # tilt cost punishes any deviation equally.
        WRIST_Z_MID = 1.0
        prefer_downward = bool(target_pos[2] < WRIST_Z_MID)

        def _write_qpos(x):
            self._plan_data.qpos[self._qpos_map["ColumnLeft"]]  = x[0]
            self._plan_data.qpos[self._qpos_map["ColumnRight"]] = x[1]
            self._plan_data.qpos[self._qpos_map["ArmLeft"]]     = x[2]
            self._plan_data.qpos[self._qpos_map["Base"]]        = x[3]
            self._plan_data.qpos[self._qpos_map["HandBearing"]] = x[4]
            self._plan_data.qpos[self._qpos_map["WristZ"]]      = x[5]
            self._plan_data.qpos[self._qpos_map["WristX"]]      = x[6]
            self._plan_data.qpos[self._qpos_map["WristY"]]      = x[7]

        def cost_fn(x):
            _write_qpos(x)
            mujoco.mj_forward(self._model, self._plan_data)
            reach_err = float(np.linalg.norm(
                self._plan_data.xpos[gripper_bid] - target_pos))
            diff = float(x[1]) - float(x[0])    # h2 - h1, positive = downward tilt
            TILT_KNEE = 0.30
            TILT_QUAD_K = 1.0
            if side_grip_mode:
                # Side-grip cost: target h-diff = SIDE_TILT_TARGET with a
                # quadratic penalty outside SIDE_TILT_WINDOW. Extra
                # quadratic pulls toward known-good (h1, a1) keep SLSQP in
                # the reachable basin instead of collapsing to a small-tilt
                # local minimum that leaves the fingers short of the object.
                SIDE_TILT_TARGET = 0.20
                SIDE_TILT_WINDOW = 0.05
                SIDE_H1_TARGET   = 0.21
                SIDE_A1_TARGET   = 0.55
                tilt_dev = abs(diff - SIDE_TILT_TARGET)
                preferred = 0.0
                opposed = max(0.0, tilt_dev - SIDE_TILT_WINDOW)
                h1_pull = 0.5 * (float(x[0]) - SIDE_H1_TARGET) ** 2
                a1_pull = 2.0 * (float(x[2]) - SIDE_A1_TARGET) ** 2
            elif prefer_downward:
                # Diagonal / top-down with low target Z — slight downward
                # tilt is preferred (the original 4-DOF heuristic).
                preferred = max(0.0, diff)
                opposed   = max(0.0, -diff)
            else:
                preferred = max(0.0, -diff)
                opposed   = max(0.0, diff)
            preferred_penalty = (
                0.005 * preferred
                + TILT_QUAD_K * max(0.0, preferred - TILT_KNEE) ** 2
            )
            opposed_penalty = 0.10 * opposed
            # Wrist preference: per-DOF quadratic pull toward wrist_goal.
            # Side-grip callers pass per-DOF ww = (low, high, high, high)
            # so HB is free as a reach DOF while wx/wz/wy stay locked at
            # the orientation goal.
            wrist_dev = (np.asarray(x[4:8]) - wg) ** 2
            wrist_penalty = float(np.sum(ww * wrist_dev))
            # Side-grip-specific manual-pose pulls (h1, a1 toward the
            # known-working floor pickup values). Zero in other modes.
            manual_pull = 0.0
            if side_grip_mode:
                manual_pull = h1_pull + a1_pull
            return (reach_err + preferred_penalty + opposed_penalty
                    + wrist_penalty + manual_pull)

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

        # Seeds: choose the family based on the wrist orientation.
        # Side-grip mode (palm horizontal, hb ≈ 0): the wrist must reach
        # the obj's mid-height with the arm LEVEL (h1 ≈ h2). The hand
        # hangs from a mid-column trolley by the static chain offset
        # (~0.5 m), so to reach floor obj at z ≈ 0.1 m we need columns
        # at h ≈ 0.55–0.65 m. Seeds span that band.
        # Diagonal / top-down mode (hb < -0.2): the wrist sits ABOVE the
        # obj with the arm tilted down (h2 > h1). Use the original
        # "low + extended" seed family that worked for the 4-DOF planner.
        wg_list = wg.tolist()
        if side_grip_mode:
            # Side seeds — focus on the NEAR-LEVEL mid-column family.
            # The new architecture (IK plans from virtual chassis at
            # 0.52 m, chassis pushes forward after descent) means the
            # arm only needs the natural small tilt that the parallel
            # chain produces. The previous "tilted-arm + level-palm"
            # family (h1=0.05, h2=0.50) is no longer needed and was
            # the source of the chassis-clipping h2-h1=0.36 solutions.
            # Seeds vary h1/h2 around mid-column with a small valid
            # tilt (h2-h1=0.03 → alpha ≈ 17°, comfortably above
            # ALPHA_MIN_DEG=1.5°), and a few HB pre-tilts so SLSQP
            # can use HB as a fine reach adjustment if needed.
            def _ws(hb_seed):
                ws = list(wg)
                ws[0] = float(hb_seed)
                return ws
            biased_seeds = [
                # High-tilt seed near the known-good floor side-grip pose
                # (h1≈0.21, h2≈0.50, a1≈0.44). Tried first so SLSQP
                # starts in the reachable basin; if validity rejects, the
                # forward-tilt seeds below act as fallbacks.
                np.array([0.21, 0.50, 0.44, theta_guess] + _ws(wg[0])),
                # Forward-tilt seeds (h2 > h1). Side-grip on a floor obj
                # requires forward tilt; reverse tilt is invalid here.
                np.array([0.55, 0.58, 0.30, theta_guess] + _ws(wg[0])),
                np.array([0.60, 0.63, 0.30, theta_guess] + _ws(wg[0])),
                np.array([0.50, 0.53, 0.40, theta_guess] + _ws(wg[0])),
                np.array([0.55, 0.58, 0.40, theta_guess] + _ws(wg[0])),
                np.array([0.60, 0.63, 0.40, theta_guess] + _ws(wg[0])),
                # Near-level with slight HB adjustment for reach
                np.array([0.55, 0.58, 0.35, theta_guess] + _ws(-0.30)),
                np.array([0.60, 0.63, 0.35, theta_guess] + _ws(-0.30)),
            ]
        else:
            biased_seeds = [
                np.array([0.05, 0.10, 0.60, theta_guess] + wg_list),
                np.array([0.10, 0.15, 0.60, theta_guess] + wg_list),
                np.array([0.15, 0.20, 0.55, theta_guess] + wg_list),
                np.array([0.20, 0.25, 0.55, theta_guess] + wg_list),
            ]
        home_seed = np.array(_pad_q(HOME_Q), dtype=float)
        home_seed[4:8] = wg
        seeds = biased_seeds + [home_seed]
        # Warm-start: when the caller passes a prior valid solution as
        # `seed_q`, prepend it so SLSQP starts in a known-feasible basin.
        if seed_q is not None:
            try:
                sq = _pad_q(seed_q)
                warm = np.array(sq, dtype=float)
                # Overwrite the wrist block with the caller's wrist goal
                # so the seed matches the IK's wz/wx/wy bias.
                warm[4:8] = wg
                seeds = [warm] + seeds
            except Exception:
                pass
        rng = np.random.default_rng()
        for _ in range(max(0, n_seeds - len(seeds))):
            arm_rand = [rng.uniform(lo, hi)
                        for lo, hi in JOINT_RANGES_ARM[:4]]
            seeds.append(np.array(arm_rand + wg_list))

        best_q, best_err = None, float('inf')
        best_seed_idx = -1
        seed_diags = []   # (seed_idx, reach_err, valid, cost_fn) per seed
        for si, seed in enumerate(seeds):
            # SLSQP normally converges in <20 iterations; cap at 40 to
            # bound worst-case time on hard cost landscapes. Suboptimal
            # early stops are acceptable since we rank multiple seeds.
            res = minimize(cost_fn, seed, method='SLSQP',
                           bounds=JOINT_RANGES_ARM, tol=1e-4,
                           options={'maxiter': 40})
            # Threshold on pure reach error (cost_fn also includes tilt penalty).
            _write_qpos(res.x)
            mujoco.mj_forward(self._model, self._plan_data)
            reach_err = float(np.linalg.norm(
                self._plan_data.xpos[gripper_bid] - target_pos))
            valid = self.is_valid(res.x)
            seed_diags.append((si, reach_err, valid, float(res.fun),
                               float(res.x[0]), float(res.x[1])))
            # Rank by cost_fn (includes tilt penalty) for tie-breaking.
            if res.fun < best_err and valid and reach_err <= threshold + 1e-6:
                best_err = res.fun
                best_q   = res.x.tolist()
                best_seed_idx = si
            if best_err < 0.005:
                break

        if best_q is None:
            # Per-seed failure dump is high-volume — `solve_ik_with_z_lift`
            # invokes solve_ik() repeatedly while sweeping z, and most
            # raises are expected intermediate failures. Gate the dump
            # behind IK_DEBUG_VERBOSE.
            if IK_DEBUG_VERBOSE:
                print(f"[IK-DBG] IK failed.  Seed results:")
                for (si, re, va, cf, h1, h2) in seed_diags:
                    print(f"   seed#{si}: reach_err={re:.3f}m  valid={va}  "
                          f"cost={cf:.4f}  h1={h1:.3f} h2={h2:.3f}")
            raise RuntimeError(
                f"IK failed: target={target_pos}, best_err=infm "
                f"(threshold={threshold}m). Consider increasing n_seeds or threshold.")

        # Phase 2 diagnostic: which seed won? Helps diagnose whether
        # the manual high-tilt seed (#0) was actually picked or whether
        # SLSQP converged to a different basin from a small-tilt seed.
        # Only print for side-grip mode (where the manual seed is in
        # play) to keep diagonal/top-down logs clean.
        # Gated by IK_DEBUG_VERBOSE; flip True to re-enable for tuning.
        if (IK_DEBUG_VERBOSE
                and side_grip_mode and len(seed_diags) > 0):
            win = seed_diags[best_seed_idx]
            print(f"[IK-DBG] side-grip IK won by seed#{win[0]}: "
                  f"h1={win[4]:.3f} h2={win[5]:.3f} h-diff={win[5]-win[4]:+.3f}  "
                  f"reach_err={win[1]:.3f}m  cost={win[3]:.4f}")
            # If seed#0 (the manual high-tilt seed) was tried, also
            # report its result so we can see WHY it lost.
            if len(seed_diags) > 0:
                s0 = seed_diags[0]
                print(f"[IK-DBG]   seed#0 (manual high-tilt seed): "
                      f"h1={s0[4]:.3f} h2={s0[5]:.3f} h-diff={s0[5]-s0[4]:+.3f}  "
                      f"reach_err={s0[1]:.3f}m  valid={s0[2]}  cost={s0[3]:.4f}"
                      + ("  ← WINNING" if best_seed_idx == 0 else ""))

        return best_q

    def solve_ik_with_z_lift(self, target_pos, n_seeds=12,
                             max_lift=0.40, step=0.02, threshold=0.04,
                             wrist_goal=None, wrist_weight=5.0,
                             target_body="Gripper_Link1_1",
                             seed_q=None):
        """
        Strict-IK variant for grasp/pick targets: never returns a solution
        for a target below the requested z. If the requested target_z is
        unreachable, lifts z by `step` and retries up to `max_lift`.

        Returns (q, actual_target) so the caller knows where the wrist will
        actually be. Unlike solve_ik_robust, never silently downgrades z.

        Pass-through args (see solve_ik):
            wrist_goal, wrist_weight, target_body — used unchanged inside
            the lift-and-retry loop.  For side-grip, the typical call is
            `target_body="Gripper_Link3_1"` and `wrist_weight=0.5` so
            SLSQP can use HandBearing as a 5th reach DOF and put the palm
            (not the wrist) on the obj.

        Note on step size: lowered 0.05 → 0.02 (2 cm) so the IK lands on
        the lowest valid z instead of jumping past it in 5 cm chunks.
        At 5 cm step, asking for z=0.13 m on a floor obj would jump to
        0.18, 0.23, ... 0.48 — landing way above the user-desired mid-
        height.  At 2 cm step the same search lands at z=0.15-0.17 m.
        """
        target_pos = np.asarray(target_pos, dtype=float)
        last_err = None
        n_steps = max(1, int(round(max_lift / step))) + 1
        for i in range(n_steps):
            lift = i * step
            probe = target_pos.copy()
            probe[2] += lift
            try:
                q = self.solve_ik(probe, n_seeds=n_seeds,
                                  threshold=threshold,
                                  wrist_goal=wrist_goal,
                                  wrist_weight=wrist_weight,
                                  target_body=target_body,
                                  seed_q=seed_q)
                return q, probe
            except RuntimeError as e:
                last_err = e
        raise RuntimeError(
            f"solve_ik_with_z_lift: pre-grasp unreachable up to "
            f"+{max_lift:.2f}m of z. Last: {last_err}")

    def solve_ik_with_z_lift_link3(self, target_pos, n_seeds=12,
                                   max_iters=3, tol=0.005,
                                   wrist_goal=None):
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
            return self.solve_ik_with_z_lift(target_pos, n_seeds=n_seeds,
                                             wrist_goal=wrist_goal)

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
                adjusted_target, n_seeds=n_seeds, wrist_goal=wrist_goal)
            self._plan_data.qpos[self._qpos_map["ColumnLeft"]]  = q[0]
            self._plan_data.qpos[self._qpos_map["ColumnRight"]] = q[1]
            self._plan_data.qpos[self._qpos_map["ArmLeft"]]     = q[2]
            self._plan_data.qpos[self._qpos_map["Base"]]        = q[3]
            self._plan_data.qpos[self._qpos_map["HandBearing"]] = q[4]
            self._plan_data.qpos[self._qpos_map["WristZ"]]      = q[5]
            self._plan_data.qpos[self._qpos_map["WristX"]]      = q[6]
            self._plan_data.qpos[self._qpos_map["WristY"]]      = q[7]
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

    def solve_ik_robust(self, target_pos, n_seeds=12, z_perturbs=None,
                        wrist_goal=None):
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
                return self.solve_ik(probe, n_seeds=n_seeds, threshold=0.04,
                                     wrist_goal=wrist_goal)
            except RuntimeError as e:
                last_err = e
        raise RuntimeError(
            f"solve_ik_robust: all {len(z_perturbs)} probes failed. Last: {last_err}")

    # ── Validity helpers ──────────────────────────────────────────────────────

    def is_valid(self, q):
        """Check if an arm config is collision-free.

        Accepts a 4-vector (legacy callers — wrist defaulted to neutral)
        or an 8-vector (4 arm + 4 wrist DOFs).
        """
        qp = _pad_q(q)
        s = self._si.allocState()
        for i in range(ARM_DOF):
            s[i] = float(qp[i])
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
    print(f"arm_planner.py self-test  —  MORPHBridge dof={ARM_DOF}")
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
    # 4-vec entries exercise the legacy code path via _pad_q; 8-vec entries
    # confirm wrist DOFs are accepted directly.
    print("\n--- Validity checks ---")
    checks = [
        ("HOME_Q (8-vec, expect valid)",        HOME_Q,                True),
        ("Singular h1==h2 (4-vec, invalid)",    [0.5, 0.5, 0.2, 0.0], False),
        ("Low arm, valid height diff (4-vec)",  [0.3, 0.8, 0.1, 0.0], True),
        # PARK_Q is the ARM2 static pose; ARM1-singular by design.
        ("PARK_Q (expect ARM1-invalid)",        PARK_Q,               False),
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
    # 8-DOF samples: random arm + neutral wrist (zeros) so the perf number is
    # comparable to the pre-extension 4-DOF baseline.
    print("\n--- Performance: isValid() ---")
    N  = 200
    qs = np.column_stack([
        np.random.uniform(0.1, 1.3, N),
        np.random.uniform(0.1, 1.3, N),
        np.random.uniform(0.0, 0.6, N),
        np.zeros(N),
        np.zeros(N),
        np.zeros(N),
        np.zeros(N),
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
