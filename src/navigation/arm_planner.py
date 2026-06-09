
import os
import numpy as np
import mujoco
from ompl import base as ob
from ompl import geometric as og


D2            = 0.1
ALPHA_MIN_DEG = 1.5
INTERP_RES    = 0.02

IK_DEBUG_VERBOSE = False

CHASSIS_PLAN_MARGIN     = 0.025
MIN_NONREST_CLEARANCE   = 0.003

JOINT_RANGES_ARM = [
    (0.00,  1.40),
    (0.00,  1.40),
    (0.00,  0.625),
    (-3.14, 3.14),
    (-1.57, 1.57),
    (-3.14, 3.14),
    (-0.80, 0.80),
    (-0.80, 0.80),
]
ARM_DOF = len(JOINT_RANGES_ARM)
WRIST_NEUTRAL = (0.0, 0.0, 0.0, 0.0)

HOME_Q = [0.5, 0.9, 0.1, 0.0,  0.0, 0.0, 0.0, 0.0]
PARK_Q = [1.2, 1.2, 0.1, 0.0,  0.0, 0.0, 0.0, 0.0]


def _pad_q(q):
    if q is None:
        return None
    q = list(q)
    if len(q) == ARM_DOF:
        return [float(v) for v in q]
    if len(q) == 4:
        return [float(v) for v in q] + list(WRIST_NEUTRAL)
    raise ValueError(
        f"Arm q must have length 4 or {ARM_DOF}, got {len(q)}")

CALIB_SCALE = 1.0


class MORPHValidityChecker(ob.StateValidityChecker):

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

        alpha_deg = np.degrees(np.arctan2(h2 - h1, self._d2))
        if not bool(alpha_deg * alpha_deg >= self._alpha_sq):
            return False

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
                continue
            if c.dist < self._min_clearance:
                return False

        return True


CARRY_ANCHOR_FINGER_BODIES = ("finger_a_link_3_1",
                              "finger_b_link_3_1",
                              "finger_c_link_3_1")


def compute_carry_anchor_xyz(data, finger_body_ids):
    pts = [data.xpos[bid] for bid in finger_body_ids]
    return np.mean(pts, axis=0)


class MORPHBridge:

    def __init__(self, xml_path, arm=1, alpha_min_deg=ALPHA_MIN_DEG,
                 timeout=5.0, use_calibration=False,
                 calib_wrist_mode="sidegrip"):
        self._timeout = timeout
        self._arm     = arm

        xml_path        = os.path.abspath(xml_path)
        self._model     = mujoco.MjModel.from_xml_path(xml_path)
        self._plan_data = mujoco.MjData(self._model)

        self._qpos_map = self._build_qpos_map(arm)
        self._ctrl_map = self._build_ctrl_map(arm)

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
        try:
            import navigation.grasp_executor as _ge_clr
            if getattr(_ge_clr, 'ENABLE_NO_CHASSIS_PUSH', False):
                self._checker._min_clearance = 0.012
                print(f"[ArmPlanner] --no-chassis-push: raised IK/OMPL "
                      f"chassis clearance "
                      f"{MIN_NONREST_CLEARANCE*100:.1f}cm → 1.2cm "
                      f"(arm routes around base; avoids Arm_Left↔base jerk)")
        except Exception:
            pass
        self._si.setup()

        self._calib_wrist_mode = calib_wrist_mode
        if use_calibration:
            self._calib = self._load_calibration(calib_wrist_mode)
        else:
            self._calib = None
            print("[MORPHBridge] Calibration LUT disabled (default).  Pass "
                  "use_calibration=True or run play_m1 with --use-calib to "
                  "enable IK pre-correction.")

        self._reach_lut = None
        self._reach_tree = None
        self._reach_lut_tried = False


    def _load_calibration(self, wrist_mode):
        import os
        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.abspath(os.path.join(here, "..", ".."))
        _prefer_5d = False
        try:
            import navigation.grasp_executor as _ge_mod
            _prefer_5d = bool(getattr(_ge_mod, 'ENABLE_NO_CHASSIS_PUSH', False))
        except Exception:
            _prefer_5d = False
        candidates = []
        if _prefer_5d:
            candidates.append(
                os.path.join(root, "data",
                             f"arm_calibration_{wrist_mode}_dynamic.npz"))
            candidates.append(
                os.path.join(root, "data", f"arm_calibration_{wrist_mode}_5d.npz"))
        candidates += [
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
            ndim = int(d['ndim']) if 'ndim' in d.files else 3
            err_arr = np.asarray(d['error'], dtype=float)
            try:
                _lut_mode_raw = (str(d['wrist_mode']) if 'wrist_mode' in d.files
                                 else "")
            except Exception:
                _lut_mode_raw = ""
            _is_dynamic = "dynamic" in _lut_mode_raw.lower()
            clip_thresh = 0.60 if _is_dynamic else 0.25
            err_mag = np.linalg.norm(err_arr, axis=-1)
            bad_mask = err_mag > clip_thresh
            n_bad = int(bad_mask.sum())
            if n_bad > 0:
                err_arr[bad_mask] = 0.0
            calib = {
                'ndim':    ndim,
                'is_dynamic': _is_dynamic,
                'h1_grid': np.asarray(d['h1_grid'], dtype=float),
                'h2_grid': np.asarray(d['h2_grid'], dtype=float),
                'a1_grid': np.asarray(d['a1_grid'], dtype=float),
                'error':   err_arr,
                'n_corner_clipped': n_bad,
            }
            if ndim == 5:
                calib['hb_grid'] = np.asarray(d['hb_grid'], dtype=float)
                calib['wz_grid'] = np.asarray(d['wz_grid'], dtype=float)
            lut_mode = None
            try:
                lut_mode = str(d['wrist_mode']) if 'wrist_mode' in d.files else None
            except Exception:
                lut_mode = None
            mode_str = f" mode={lut_mode}" if lut_mode else ""
            _lut_base_mode = (lut_mode.replace("-dynamic", "")
                              if lut_mode else None)
            mismatch = (_lut_base_mode is not None
                        and _lut_base_mode != wrist_mode)
            mismatch_warn = (" WARNING: LUT was built for "
                             f"'{lut_mode}' but requested '{wrist_mode}' —"
                             " calibration may be inaccurate" if mismatch
                             else "")
            if ndim == 5:
                grid_desc = (f"{len(calib['h1_grid'])}×{len(calib['h2_grid'])}"
                             f"×{len(calib['a1_grid'])}"
                             f"×{len(calib['hb_grid'])}×{len(calib['wz_grid'])}"
                             f"  (5D)")
            else:
                grid_desc = (f"{len(calib['h1_grid'])}×{len(calib['h2_grid'])}"
                             f"×{len(calib['a1_grid'])}  (3D)")
            clipped_str = (f"  [clipped {calib['n_corner_clipped']} corner cells]"
                           if calib.get('n_corner_clipped', 0) > 0 else "")
            print(f"[MORPHBridge] Loaded calibration LUT {npz_path}{mode_str}  "
                  f"grid={grid_desc}  "
                  f"max_err_xy="
                  f"{np.linalg.norm(calib['error'][..., :2], axis=-1).max()*100:.1f}cm  "
                  f"max_err_z="
                  f"{np.abs(calib['error'][..., 2]).max()*100:.1f}cm"
                  f"{clipped_str}{mismatch_warn}")
            return calib
        except Exception as e:
            print(f"[MORPHBridge] Failed to load LUT {npz_path}: {e}")
            return None

    def is_dynamic_calib_loaded(self):
        if self._calib is None:
            return False
        return bool(self._calib.get('is_dynamic', False))

    def _calib_error(self, h1, h2, a1, theta, hb=None, wz=None):
        if self._calib is None:
            return np.zeros(3, dtype=float)
        err = self._calib['error']
        ndim = self._calib.get('ndim', 3)

        def _interp_axis(grid, v):
            v = float(np.clip(v, grid[0], grid[-1]))
            i = int(np.searchsorted(grid, v) - 1)
            i = max(0, min(len(grid) - 2, i))
            t = (v - grid[i]) / max(1e-9, grid[i + 1] - grid[i])
            return i, float(t)

        hg = self._calib['h1_grid']
        kg = self._calib['h2_grid']
        ag = self._calib['a1_grid']
        i, ti = _interp_axis(hg, h1)
        j, tj = _interp_axis(kg, h2)
        k, tk = _interp_axis(ag, a1)

        if ndim == 5:
            bg = self._calib['hb_grid']
            wg = self._calib['wz_grid']
            hb_v = float(bg[len(bg) // 2]) if hb is None else float(hb)
            wz_v = float(wg[len(wg) // 2]) if wz is None else float(wz)
            m, tm = _interp_axis(bg, hb_v)
            n, tn = _interp_axis(wg, wz_v)
            out = np.zeros(3, dtype=float)
            for di, wi in ((0, 1 - ti), (1, ti)):
                for dj, wj in ((0, 1 - tj), (1, tj)):
                    for dk, wk in ((0, 1 - tk), (1, tk)):
                        for dm, wm in ((0, 1 - tm), (1, tm)):
                            for dn, wn in ((0, 1 - tn), (1, tn)):
                                out += (wi * wj * wk * wm * wn
                                        * err[i + di, j + dj, k + dk,
                                              m + dm, n + dn])
        else:
            out = np.zeros(3, dtype=float)
            for di, wi in ((0, 1 - ti), (1, ti)):
                for dj, wj in ((0, 1 - tj), (1, tj)):
                    for dk, wk in ((0, 1 - tk), (1, tk)):
                        out += wi * wj * wk * err[i + di, j + dj, k + dk]

        out = out * CALIB_SCALE

        c = np.cos(theta)
        s = np.sin(theta)
        in_chassis = np.array([
            c * out[0] - s * out[1],
            s * out[0] + c * out[1],
            out[2],
        ], dtype=float)

        try:
            chassis_bid = mujoco.mj_name2id(
                self._model, mujoco.mjtObj.mjOBJ_BODY, "robot")
            if chassis_bid >= 0:
                R = self._plan_data.xmat[chassis_bid].reshape(3, 3)
                return R @ in_chassis
        except Exception:
            pass
        return in_chassis


    def _load_reachability_lut(self, wrist_mode):
        import os
        if self._reach_tree is not None:
            return (self._reach_lut['sample_q'],
                    self._reach_lut['sample_pos'],
                    self._reach_tree)
        if self._reach_lut_tried:
            return None, None, None
        self._reach_lut_tried = True
        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.abspath(os.path.join(here, "..", ".."))
        npz_path = os.path.join(
            root, "data", f"arm_reachability_{wrist_mode}.npz")
        if not os.path.exists(npz_path):
            print(f"[MORPHBridge] No reachability LUT found at {npz_path}. "
                  f"Run tools/build_reachability_lut.py --wrist-mode "
                  f"{wrist_mode} to build one. IK will use cold seeds.")
            return None, None, None
        try:
            from scipy.spatial import cKDTree
            d = np.load(npz_path)
            sample_q = np.asarray(d['sample_q'], dtype=float)
            sample_pos = np.asarray(d['sample_pos'], dtype=float)
            self_coll = np.asarray(d['self_collision'], dtype=bool)
            free_mask = ~self_coll
            free_pos = sample_pos[free_mask]
            free_q = sample_q[free_mask]
            tree = cKDTree(free_pos)
            self._reach_lut = {
                'sample_q':   free_q,
                'sample_pos': free_pos,
                'wrist_mode': wrist_mode,
            }
            self._reach_tree = tree
            print(f"[MORPHBridge] Loaded reachability LUT {npz_path}  "
                  f"({len(free_pos):,} free / {len(sample_pos):,} total)")
            return free_q, free_pos, tree
        except Exception as e:
            print(f"[MORPHBridge] Failed to load reachability LUT: {e}")
            return None, None, None

    def _query_reachability(self, target_pos_world, k=1):
        wrist_mode = self._calib_wrist_mode
        sample_q, sample_pos, tree = self._load_reachability_lut(wrist_mode)
        if tree is None:
            return None
        try:
            chassis_bid = mujoco.mj_name2id(
                self._model, mujoco.mjtObj.mjOBJ_BODY, "robot")
            if chassis_bid >= 0:
                R = self._plan_data.xmat[chassis_bid].reshape(3, 3)
                origin = self._plan_data.xpos[chassis_bid]
                target_local = R.T @ (np.asarray(target_pos_world,
                                                  dtype=float) - origin)
            else:
                target_local = np.asarray(target_pos_world, dtype=float)
        except Exception:
            target_local = np.asarray(target_pos_world, dtype=float)
        try:
            d, idx = tree.query(target_local, k=k)
            if k == 1:
                return sample_q[int(idx)].copy()
            else:
                return [sample_q[int(i)].copy() for i in idx]
        except Exception as e:
            print(f"[MORPHBridge] Reachability query failed: {e}")
            return None


    def _collect_chassis_geom_ids(self):
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


    def _build_rest_pairs(self):
        arm1_map = self._build_qpos_map(1)
        arm2_map = self._build_qpos_map(2)

        ref_states = [
            _pad_q(HOME_Q),
            _pad_q([0.05, 0.5, 0.1, 0.0]),
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


    def _build_space(self):
        space  = ob.RealVectorStateSpace(ARM_DOF)
        bounds = ob.RealVectorBounds(ARM_DOF)
        for i, (lo, hi) in enumerate(JOINT_RANGES_ARM):
            bounds.setLow(i, lo)
            bounds.setHigh(i, hi)
        space.setBounds(bounds)
        return space


    def plan(self, start_q, goal_q, timeout=None):
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


    def solve_ik(self, target_pos, n_seeds=8, threshold=0.02,
                 wrist_goal=None, wrist_weight=5.0,
                 target_body="Gripper_Link1_1",
                 seed_q=None,
                 tilt_weight_scale=1.0,
                 manual_pull_scale=1.0,
                 validity_penalty_scale=0.0,
                 approach_yaw=None,
                 axis_align_weight=0.0,
                 th_target=None,
                 th_weight=0.0,
                 column_center_weight=0.0):
        from scipy.optimize import minimize
        import math as _math

        gripper_bid = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, target_body)
        if gripper_bid < 0:
            raise RuntimeError(f"Body '{target_body}' not found in model")

        _axis_align_active = (approach_yaw is not None
                              and float(axis_align_weight) > 0.0)
        _thumb_axis_bid = -1
        _b_axis_bid = -1
        _c_axis_bid = -1
        if _axis_align_active:
            _thumb_axis_bid = mujoco.mj_name2id(
                self._model, mujoco.mjtObj.mjOBJ_BODY, "finger_a_link_3_1")
            _b_axis_bid = mujoco.mj_name2id(
                self._model, mujoco.mjtObj.mjOBJ_BODY, "finger_b_link_3_1")
            _c_axis_bid = mujoco.mj_name2id(
                self._model, mujoco.mjtObj.mjOBJ_BODY, "finger_c_link_3_1")
            if (_thumb_axis_bid < 0 or _b_axis_bid < 0
                    or _c_axis_bid < 0):
                _axis_align_active = False

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

        side_grip_mode = bool(abs(float(wg[0])) < 0.20)

        WRIST_Z_MID = 1.0
        prefer_downward = bool(target_pos[2] < WRIST_Z_MID)

        _bnd_lo = np.array([float(b[0]) for b in JOINT_RANGES_ARM])
        _bnd_hi = np.array([float(b[1]) for b in JOINT_RANGES_ARM])

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
            if not np.all(np.isfinite(x)):
                return 1e6
            x = np.clip(np.asarray(x, dtype=float), _bnd_lo, _bnd_hi)
            _write_qpos(x)
            mujoco.mj_forward(self._model, self._plan_data)
            reach_err = float(np.linalg.norm(
                self._plan_data.xpos[gripper_bid] - target_pos))
            diff = float(x[1]) - float(x[0])
            TILT_KNEE = 0.30
            TILT_QUAD_K = 1.0
            if side_grip_mode:
                _no_push_for_ik = False
                try:
                    import navigation.grasp_executor as _ge_for_ik
                    _no_push_for_ik = bool(getattr(
                        _ge_for_ik, 'ENABLE_NO_CHASSIS_PUSH', False))
                except Exception:
                    pass
                if _no_push_for_ik:
                    SIDE_TILT_TARGET = 0.08
                    SIDE_TILT_WINDOW = 0.04
                else:
                    SIDE_TILT_TARGET = 0.20
                    SIDE_TILT_WINDOW = 0.05
                SIDE_H1_TARGET   = 0.21
                SIDE_A1_TARGET   = 0.55
                tilt_dev = abs(diff - SIDE_TILT_TARGET)
                preferred = 0.0
                opposed = max(0.0, tilt_dev - SIDE_TILT_WINDOW)
                h1_pull = 0.5 * (float(x[0]) - SIDE_H1_TARGET) ** 2
                a1_pull = 2.0 * (float(x[2]) - SIDE_A1_TARGET) ** 2
                if _no_push_for_ik and abs(diff) > 0.12:
                    h1_pull += 80.0 * (abs(diff) - 0.12) ** 2
            elif prefer_downward:
                preferred = max(0.0, diff)
                opposed   = max(0.0, -diff)
            else:
                preferred = max(0.0, -diff)
                opposed   = max(0.0, diff)
            preferred_penalty = tilt_weight_scale * (
                0.005 * preferred
                + TILT_QUAD_K * max(0.0, preferred - TILT_KNEE) ** 2
            )
            opposed_penalty = tilt_weight_scale * 0.10 * opposed
            wrist_dev = (np.asarray(x[4:8]) - wg) ** 2
            wrist_penalty = float(np.sum(ww * wrist_dev))
            manual_pull = 0.0
            if side_grip_mode:
                manual_pull = manual_pull_scale * (h1_pull + a1_pull)
            validity_pen = 0.0
            if validity_penalty_scale > 0.0:
                alpha_deg = abs(_math.degrees(_math.atan2(
                    diff, self._checker._d2)))
                alpha_dev = max(0.0, ALPHA_MIN_DEG - alpha_deg)
                validity_pen += 0.1 * alpha_dev
                allowed = self._checker._allowed
                min_clr = self._checker._min_clearance
                pd = self._plan_data
                for i in range(pd.ncon):
                    c = pd.contact[i]
                    g1, g2 = int(c.geom1), int(c.geom2)
                    if (g1, g2) in allowed or (g2, g1) in allowed:
                        continue
                    if c.dist < min_clr:
                        validity_pen += 0.05 + (min_clr - c.dist)
                validity_pen *= validity_penalty_scale
            axis_align_pen = 0.0
            if _axis_align_active:
                thumb_xy = self._plan_data.xpos[_thumb_axis_bid][:2]
                b_xy = self._plan_data.xpos[_b_axis_bid][:2]
                c_xy = self._plan_data.xpos[_c_axis_bid][:2]
                bc_centroid = 0.5 * (b_xy + c_xy)
                axis_dir = bc_centroid - thumb_xy
                if float(axis_dir[0] ** 2 + axis_dir[1] ** 2) > 1e-9:
                    axis_yaw = _math.atan2(
                        float(axis_dir[1]), float(axis_dir[0]))
                    target_a = float(approach_yaw) + _math.pi / 2
                    target_b = float(approach_yaw) - _math.pi / 2

                    def _wrap(a):
                        while a > _math.pi:
                            a -= 2 * _math.pi
                        while a < -_math.pi:
                            a += 2 * _math.pi
                        return a

                    err_a = _wrap(axis_yaw - target_a)
                    err_b = _wrap(axis_yaw - target_b)
                    err = err_a if abs(err_a) < abs(err_b) else err_b
                    axis_align_pen = float(axis_align_weight) * err * err
            th_pen = 0.0
            if th_target is not None and th_weight > 0.0:
                _thd = float(x[3]) - float(th_target)
                while _thd > _math.pi:
                    _thd -= 2 * _math.pi
                while _thd < -_math.pi:
                    _thd += 2 * _math.pi
                th_pen = float(th_weight) * _thd * _thd
            col_pen = 0.0
            if column_center_weight > 0.0:
                _h1 = float(x[0]); _h2 = float(x[1])
                _lo, _hi = 0.18, 1.25
                col_pen = column_center_weight * (
                    max(0.0, _lo - _h1) ** 2 + max(0.0, _lo - _h2) ** 2
                    + max(0.0, _h1 - _hi) ** 2 + max(0.0, _h2 - _hi) ** 2)
            return (reach_err + preferred_penalty + opposed_penalty
                    + wrist_penalty + manual_pull + validity_pen
                    + axis_align_pen + th_pen + col_pen)

        robot_bid = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, "robot")
        if robot_bid >= 0:
            rpos = self._plan_data.xpos[robot_bid]
            theta_guess = float(_math.atan2(target_pos[1] - rpos[1],
                                            target_pos[0] - rpos[0]))
        else:
            theta_guess = 0.0
        theta_guess = max(-3.14, min(3.14, theta_guess))

        wg_list = wg.tolist()
        if side_grip_mode:
            def _ws(hb_seed):
                ws = list(wg)
                ws[0] = float(hb_seed)
                return ws
            biased_seeds = [
                np.array([0.21, 0.50, 0.44, theta_guess] + _ws(wg[0])),
                np.array([0.55, 0.58, 0.30, theta_guess] + _ws(wg[0])),
                np.array([0.60, 0.63, 0.30, theta_guess] + _ws(wg[0])),
                np.array([0.50, 0.53, 0.40, theta_guess] + _ws(wg[0])),
                np.array([0.55, 0.58, 0.40, theta_guess] + _ws(wg[0])),
                np.array([0.60, 0.63, 0.40, theta_guess] + _ws(wg[0])),
                np.array([0.55, 0.58, 0.35, theta_guess] + _ws(-0.30)),
                np.array([0.60, 0.63, 0.35, theta_guess] + _ws(-0.30)),
            ]
            if robot_bid >= 0:
                _ctt = float(np.linalg.norm(target_pos[:2] - rpos[:2]))
                if _ctt > 0.60:
                    biased_seeds.append(
                        np.array([0.05, 0.20, 0.62, theta_guess] + _ws(wg[0])))
                    _no_push_active = False
                    try:
                        import navigation.grasp_executor as _ge_mod
                        _no_push_active = bool(getattr(
                            _ge_mod, 'ENABLE_NO_CHASSIS_PUSH', False))
                    except Exception:
                        pass
                    if _no_push_active:
                        biased_seeds.append(
                            np.array([0.00, 0.05, 0.55, theta_guess] + _ws(0.30)))
                        biased_seeds.append(
                            np.array([0.05, 0.15, 0.60, theta_guess] + _ws(0.30)))
                        biased_seeds.append(
                            np.array([0.05, 0.25, 0.62, theta_guess] + _ws(0.15)))
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
        lut_seeds = []
        try:
            reach_seeds_k = self._query_reachability(target_pos, k=3)
            if reach_seeds_k is not None:
                if isinstance(reach_seeds_k, list):
                    candidates = reach_seeds_k
                else:
                    candidates = [reach_seeds_k]
                for rs in candidates:
                    warm = np.array(rs, dtype=float)
                    warm[4:8] = wg
                    lut_seeds.append(warm)
        except Exception:
            pass
        if seed_q is not None:
            try:
                sq = _pad_q(seed_q)
                warm = np.array(sq, dtype=float)
                warm[4:8] = wg
                seeds = [warm] + lut_seeds + seeds
            except Exception:
                seeds = lut_seeds + seeds
        else:
            seeds = lut_seeds + seeds
        rng = np.random.default_rng()
        for _ in range(max(0, n_seeds - len(seeds))):
            arm_rand = [rng.uniform(lo, hi)
                        for lo, hi in JOINT_RANGES_ARM[:4]]
            seeds.append(np.array(arm_rand + wg_list))

        best_q, best_err = None, float('inf')
        best_seed_idx = -1
        seed_diags = []
        for si, seed in enumerate(seeds):
            res = minimize(cost_fn, seed, method='SLSQP',
                           bounds=JOINT_RANGES_ARM, tol=1e-4,
                           options={'maxiter': 40})
            if not np.all(np.isfinite(res.x)):
                continue
            _rx = np.clip(np.asarray(res.x, dtype=float), _bnd_lo, _bnd_hi)
            _write_qpos(_rx)
            mujoco.mj_forward(self._model, self._plan_data)
            reach_err = float(np.linalg.norm(
                self._plan_data.xpos[gripper_bid] - target_pos))
            valid = self.is_valid(_rx)
            seed_diags.append((si, reach_err, valid, float(res.fun),
                               float(_rx[0]), float(_rx[1])))
            if res.fun < best_err and valid and reach_err <= threshold + 1e-6:
                best_err = res.fun
                best_q   = _rx.tolist()
                best_seed_idx = si
            if best_err < 0.005:
                break

        if best_q is None:
            if IK_DEBUG_VERBOSE:
                print(f"[IK-DBG] IK failed.  Seed results:")
                for (si, re, va, cf, h1, h2) in seed_diags:
                    print(f"   seed#{si}: reach_err={re:.3f}m  valid={va}  "
                          f"cost={cf:.4f}  h1={h1:.3f} h2={h2:.3f}")
            raise RuntimeError(
                f"IK failed: target={target_pos}, best_err=infm "
                f"(threshold={threshold}m). Consider increasing n_seeds or threshold.")

        if (IK_DEBUG_VERBOSE
                and side_grip_mode and len(seed_diags) > 0):
            win = seed_diags[best_seed_idx]
            print(f"[IK-DBG] side-grip IK won by seed#{win[0]}: "
                  f"h1={win[4]:.3f} h2={win[5]:.3f} h-diff={win[5]-win[4]:+.3f}  "
                  f"reach_err={win[1]:.3f}m  cost={win[3]:.4f}")
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
                             seed_q=None,
                             tilt_weight_scale=1.0,
                             manual_pull_scale=1.0,
                             validity_penalty_scale=0.0,
                             approach_yaw=None,
                             axis_align_weight=0.0,
                             th_target=None,
                             th_weight=0.0):
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
                                  seed_q=seed_q,
                                  tilt_weight_scale=tilt_weight_scale,
                                  manual_pull_scale=manual_pull_scale,
                                  validity_penalty_scale=validity_penalty_scale,
                                  approach_yaw=approach_yaw,
                                  axis_align_weight=axis_align_weight,
                                  th_target=th_target,
                                  th_weight=th_weight)
                return q, probe
            except RuntimeError as e:
                last_err = e
        raise RuntimeError(
            f"solve_ik_with_z_lift: pre-grasp unreachable up to "
            f"+{max_lift:.2f}m of z. Last: {last_err}")

    def solve_ik_no_z_lift(self, target_pos, n_seeds=4, threshold=0.04,
                           wrist_goal=None, wrist_weight=5.0,
                           target_body="Gripper_Link1_1",
                           seed_q=None,
                           tilt_weight_scale=1.0,
                           manual_pull_scale=1.0,
                           validity_penalty_scale=0.0):
        target_pos = np.asarray(target_pos, dtype=float)
        q = self.solve_ik(target_pos, n_seeds=n_seeds,
                          threshold=threshold,
                          wrist_goal=wrist_goal,
                          wrist_weight=wrist_weight,
                          target_body=target_body,
                          seed_q=seed_q,
                          tilt_weight_scale=tilt_weight_scale,
                          manual_pull_scale=manual_pull_scale,
                          validity_penalty_scale=validity_penalty_scale)
        return q, target_pos

    def differential_ik_step(self, current_q, target_body, residual_xyz,
                             joint_indices=(0, 1, 2, 3),
                             max_step_per_joint=0.05,
                             max_xy_step=0.06):
        bid = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, target_body)
        if bid < 0:
            raise RuntimeError(
                f"differential_ik_step: body '{target_body}' not found")

        qp = _pad_q(current_q)
        self._plan_data.qpos[self._qpos_map["ColumnLeft"]]  = qp[0]
        self._plan_data.qpos[self._qpos_map["ColumnRight"]] = qp[1]
        self._plan_data.qpos[self._qpos_map["ArmLeft"]]     = qp[2]
        self._plan_data.qpos[self._qpos_map["Base"]]        = qp[3]
        self._plan_data.qpos[self._qpos_map["HandBearing"]] = qp[4]
        self._plan_data.qpos[self._qpos_map["WristZ"]]      = qp[5]
        self._plan_data.qpos[self._qpos_map["WristX"]]      = qp[6]
        self._plan_data.qpos[self._qpos_map["WristY"]]      = qp[7]
        mujoco.mj_forward(self._model, self._plan_data)

        jacp = np.zeros((3, self._model.nv), dtype=np.float64)
        mujoco.mj_jacBody(self._model, self._plan_data, jacp, None, bid)

        qpos_map_keys = ["ColumnLeft", "ColumnRight", "ArmLeft", "Base",
                         "HandBearing", "WristZ", "WristX", "WristY"]
        dof_adrs = []
        for ji in joint_indices:
            joint_name = qpos_map_keys[ji]
            qadr = self._qpos_map[joint_name]
            for jnt_id in range(self._model.njnt):
                if int(self._model.jnt_qposadr[jnt_id]) == qadr:
                    dof_adrs.append(int(self._model.jnt_dofadr[jnt_id]))
                    break
            else:
                dof_adrs.append(qadr)
        J_sub = jacp[:2, :][:, dof_adrs]

        res_xy = np.asarray(residual_xyz[:2], dtype=float)
        res_mag = float(np.linalg.norm(res_xy))
        if res_mag > max_xy_step:
            res_xy = res_xy * (max_xy_step / res_mag)

        delta_q_sub = np.linalg.pinv(J_sub) @ res_xy

        for i in range(len(delta_q_sub)):
            if abs(delta_q_sub[i]) > max_step_per_joint:
                delta_q_sub[i] = (max_step_per_joint
                                  * (1.0 if delta_q_sub[i] > 0 else -1.0))

        delta_q = np.zeros(8, dtype=float)
        for i, ji in enumerate(joint_indices):
            delta_q[ji] = float(delta_q_sub[i])

        new_q = [qp[i] + delta_q[i] for i in range(8)]
        for _try in range(3):
            try:
                if self.is_valid(new_q):
                    return delta_q
            except Exception:
                pass
            delta_q = delta_q * 0.5
            new_q = [qp[i] + delta_q[i] for i in range(8)]
        raise RuntimeError(
            f"differential_ik_step: no valid q after 3 shrinks "
            f"(residual {res_mag*100:.1f}cm at {target_body})")

    def solve_ik_with_z_lift_link3(self, target_pos, n_seeds=12,
                                   max_iters=3, tol=0.005,
                                   wrist_goal=None):
        link1_bid = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, "Gripper_Link1_1")
        link3_bid = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, "Gripper_Link3_1")
        if link1_bid < 0 or link3_bid < 0:
            return self.solve_ik_with_z_lift(target_pos, n_seeds=n_seeds,
                                             wrist_goal=wrist_goal)

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
            calib_corr = self._calib_error(q[0], q[1], q[2], q[3],
                                            hb=q[4], wz=q[5])
            new_adjusted = original_target - calib_corr
            residual = float(np.linalg.norm(new_adjusted - adjusted_target))
            adjusted_target = new_adjusted
            last_q = q
            last_link1_actual = np.asarray(link1_actual, dtype=float)
            last_calib_corr = calib_corr
            if residual < tol:
                break

        link3_minus_link1 = (self._plan_data.xpos[link3_bid].copy()
                             - self._plan_data.xpos[link1_bid].copy())
        link3_actual_target = (last_link1_actual
                               + link3_minus_link1
                               + last_calib_corr)
        return last_q, link3_actual_target

    def _carry_anchor_body_ids(self):
        ids = getattr(self, "_carry_anchor_bids_cache", None)
        if ids is None:
            ids = []
            for nm in CARRY_ANCHOR_FINGER_BODIES:
                bid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, nm)
                if bid >= 0:
                    ids.append(bid)
            self._carry_anchor_bids_cache = ids
        return ids

    def solve_ik_with_z_lift_carry_anchor(self, anchor_target, n_seeds=8,
                                          max_iters=3, tol=0.005,
                                          wrist_goal=None, wrist_weight=5.0,
                                          seed_q=None, validity_penalty=50.0,
                                          column_center_weight=0.0):
        finger_ids = self._carry_anchor_body_ids()
        if len(finger_ids) != 3:
            return self.solve_ik_with_z_lift_link3(
                anchor_target, n_seeds=n_seeds, wrist_goal=wrist_goal)

        qmap = self._qpos_map
        anchor_target = np.asarray(anchor_target, dtype=float)
        adj = anchor_target.copy()
        best_q = None
        best_err = float("inf")
        best_anchor = None

        for _ in range(max_iters):
            try:
                q = self.solve_ik(
                    tuple(adj), n_seeds=n_seeds, threshold=0.05,
                    wrist_goal=wrist_goal, wrist_weight=wrist_weight,
                    seed_q=seed_q, target_body="Gripper_Link3_1",
                    validity_penalty_scale=validity_penalty,
                    column_center_weight=column_center_weight)
            except Exception:                       # noqa: BLE001
                break
            if q is None:
                break
            q = np.asarray(q, dtype=float)
            if not np.all(np.isfinite(q)):
                break
            for i in range(min(ARM_DOF, len(q))):
                lo, hi = JOINT_RANGES_ARM[i]
                q[i] = float(np.clip(q[i], lo, hi))
            for key, idx in (("ColumnLeft", 0), ("ColumnRight", 1),
                             ("ArmLeft", 2), ("Base", 3), ("HandBearing", 4),
                             ("WristZ", 5), ("WristX", 6), ("WristY", 7)):
                self._plan_data.qpos[qmap[key]] = q[idx]
            mujoco.mj_forward(self._model, self._plan_data)
            anchor_actual = compute_carry_anchor_xyz(self._plan_data, finger_ids)
            err_vec = anchor_target - anchor_actual
            err = float(np.linalg.norm(err_vec))
            if err < best_err:
                best_err = err
                best_q = list(q)
                best_anchor = anchor_actual.copy()
            seed_q = list(q)
            if err < tol:
                break
            adj = adj + 0.8 * err_vec

        return best_q, (best_anchor if best_anchor is not None else adj)

    def solve_ik_robust(self, target_pos, n_seeds=12, z_perturbs=None,
                        wrist_goal=None):
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


    def is_valid(self, q):
        qp = _pad_q(q)
        s = self._si.allocState()
        for i in range(ARM_DOF):
            s[i] = float(qp[i])
        return self._si.isValid(s)

    def update_allowed_pairs(self, extra_pairs):
        self._checker._allowed = set(self._rest_pairs) | set(extra_pairs)


    def park_body(self, body_name, pos=(0.0, 0.0, 100.0)):
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
        for i in range(n):
            try:
                self.park_body(f"pickup_obj_{i}",
                               pos=(float(i) * 3.0, 50.0, 100.0))
            except ValueError:
                pass

    def sync_base_pose_from_sim(self, sim_data):
        robot_bid = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, "robot")
        if robot_bid < 0:
            return
        ja = int(self._model.body_jntadr[robot_bid])
        if ja < 0:
            return
        qa = int(self._model.jnt_qposadr[ja])
        self._plan_data.qpos[qa:qa + 7] = sim_data.qpos[qa:qa + 7]
        mujoco.mj_forward(self._model, self._plan_data)

    def set_base_pose_xy_yaw(self, x, y, yaw, z=None):
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

    print("\n--- Validity checks ---")
    checks = [
        ("HOME_Q (8-vec, expect valid)",        HOME_Q,                True),
        ("Singular h1==h2 (4-vec, invalid)",    [0.5, 0.5, 0.2, 0.0], False),
        ("Low arm, valid height diff (4-vec)",  [0.3, 0.8, 0.1, 0.0], True),
        ("PARK_Q (expect ARM1-invalid)",        PARK_Q,               False),
    ]
    all_ok = True
    for desc, q, expect in checks:
        result = bridge.is_valid(q)
        ok = (result == expect)
        if not ok:
            all_ok = False
        print(f"  {'✓' if ok else '✗ FAIL'}  {desc}")

    print("\n--- Planning tests ---")
    plan_cases = [
        ("Vertical transit",         [0.2, 0.5, 0.2, 0.0], [0.8, 1.1, 0.2, 0.0]),
        ("Height + reach + yaw",     [0.3, 0.7, 0.1, 0.0], [0.6, 1.0, 0.4, 1.0]),
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
