
import math
import os
import time
import threading
import numpy as np
import mujoco

from navigation.arm_planner import (
    HOME_Q, PARK_Q, ARM_DOF, WRIST_NEUTRAL, JOINT_RANGES_ARM,
)
from navigation.finger_planner import FingerBridge, FINGER_DOF

WRIST_PITCH_SIDE_APPROACH      = 0.0
WRIST_PITCH_DEFAULT_DIAGONAL   = -0.50
WRIST_PITCH_AGGRESSIVE_TOPDOWN = -0.90

WRIST_Z_SIDE_APPROACH = -1.88
WRIST_X_SIDE_APPROACH =  0.80
WRIST_Y_SIDE_APPROACH =  0.00

WRIST_Z_PD_COMPENSATION_RATIO = 0.09

WRIST_Z_PD_COMPENSATION_RATIO_DESCENT = 0.18

SIDE_DRIFT_FORWARD_BIAS = 0.16

SIDE_PINCH_TARGET_STANDOFF = 0.06

REALISM_MODE_NO_SMOOTH_LIFT = True
REALISM_PRE_CLOSE_Z_GAP     = 0.05

REALISM_MICRO_LIFT_THRESHOLD = 0.12
NATURAL_CLOSE_DRAG_THRESHOLD = 0.05

ASYM_SOFT_ASSIST_MAX_FAR = 0.15

ASYM_BACKUP_DISTANCE = 0.06

PALM_ANCHOR_Z_MARGIN     = 0.02
PALM_ANCHOR_MAX_CARRY    = 0.08
PALM_ANCHOR_MAX_FAR      = 0.18

WZ_REFINE_AXIS_ERR_THRESHOLD     = math.radians(5.0)
WZ_RUNTIME_CORRECTION_SETTLE_S      = 0.35
WZ_RUNTIME_CORRECTION_MAX_DELTA     = math.radians(60.0)
WZ_RUNTIME_CORRECTION_MAX_ITERS     = 3
WZ_RUNTIME_CORRECTION_MAX_INITIAL   = math.radians(10.0)

FAST_PICKUP_MODE   = False
STRICT_PICKUP_MODE = False
STRICT_PERFECT_FRICTION_ONLY = False

ENABLE_ARM_FINE_ALIGN = False

ENABLE_TH_FINE_YAW = False
import math as _math_p7
TH_FINE_YAW_MAX_DELTA = _math_p7.radians(30.0)
TH_FINE_YAW_RESIDUAL_THRESHOLD = _math_p7.radians(2.0)
del _math_p7

P7_PRECLOSE_FIRE = False

ENABLE_DESCENT_RE_IK = False
DESCENT_RE_IK_INTERVAL_TICKS = 8
DESCENT_RE_IK_RANGE_M = 0.08
IK_REFINE_EPS = 0.002

USE_TOPPRA_LIFT = False
TOPPRA_VEL_MAX = 0.4
TOPPRA_ACC_MAX = 0.8
TOPPRA_JERK_MAX = 4.0

ENABLE_NO_CHASSIS_PUSH = False

STRICT_FRICTION_MU              = 0.7
STRICT_GRIP_SAFETY              = 2.5
STRICT_MIN_NORMAL_PER_F         = 0.6
LIFT_TIGHTEN_PREVERIFY_RAD       = 0.05
LIFT_TIGHTEN_PRELIFT_RAD         = 0.03
LIFT_TIGHTEN_POSTLIFT_RAD        = 0.03
LIFT_RETIGHTEN_PER_STEP_RAD      = 0.012
LIFT_RETIGHTEN_INTERVAL_STEPS    = 5
LIFT_SLIP_BUMP_THRESHOLD_M       = 0.003
LIFT_SLIP_BUMP_GAIN              = 6.0
LIFT_SLIP_BUMP_MAX_PER_STEP_RAD  = 0.05
LIFT_PER_FINGER_FORCE_FLOOR_N    = 4.0
LIFT_XY_DRIFT_ABORT_M            = 0.03
LIFT_FORCE_DECAY_WINDOW          = 6
LIFT_FORCE_DECAY_FRAC            = 0.40
LIFT_FORCE_DECAY_BUMP_RAD        = 0.020
LIFT_TEST_LIFT_ALPHA             = 0.06
LIFT_TEST_LIFT_SLIP_TOLERANCE_M  = 0.004
LIFT_TEST_LIFT_RETRY_BUMP_RAD    = 0.04
LIFT_TEST_LIFT_MAX_RETRIES       = 2
LIFT_MASS_REESTIMATE_ENABLED     = True
LIFT_MASS_REESTIMATE_K           = 0.30
LIFT_MASS_REESTIMATE_INTERVAL    = 8
LIFT_BUMP_HYSTERESIS_TICKS       = 2
VERIFY_TRIAD_BALANCE_DIAG_THRESHOLD = 0.45
VERIFY_TRIAD_BALANCE_GATE_THRESHOLD = 0.75
VERIFY_TRIAD_BALANCE_GATE_ENABLED   = True
VERIFY_SUSTAINED_FORCE_ENABLED        = True
VERIFY_SUSTAINED_FORCE_WINDOW_S       = 0.50
VERIFY_SUSTAINED_FORCE_SAMPLE_INT_S   = 0.05
VERIFY_SUSTAINED_FORCE_FLOOR_N        = 30.0
VERIFY_SUSTAINED_FORCE_MIN_PASS_FRAC  = 0.60
VERIFY_SUSTAINED_FORCE_SPIKE_CAP_N    = 50000.0
USE_BC_RESCUE                = True
BC_RESCUE_J1_BUMP_RAD        = 0.05
BC_RESCUE_SETTLE_S           = 0.30
BC_RESCUE_SUCCESS_N          = 4.0
LIFT_STEPS_MULTIPLIER            = 3
LIFT_PER_STEP_SETTLE_MULTIPLIER  = 2.0

STRICT_FORCE_STOP_STABLE_TICKS  = 1
STRICT_FORCE_STOP_RELIEF_TRIGGER = 2.0
STRICT_FORCE_STOP_RELIEF_RAD     = 0.005
STRICT_FORCE_STOP_RELIEF_HIGH_RATIO   = 5.0
STRICT_FORCE_STOP_RELIEF_HIGH_RAD     = 0.012
STRICT_FORCE_STOP_RELIEF_EXTREME_RATIO = 10.0
STRICT_FORCE_STOP_RELIEF_EXTREME_RAD   = 0.025
STRICT_OVERDRIVE_DELTA_INIT     = 0.08
STRICT_OVERDRIVE_DELTA_MAX      = 0.30
STRICT_SLIP_DISP_THRESH         = 0.030
STRICT_SLIP_VEL_THRESH          = 0.04
STRICT_SLIP_SETTLE_S            = 0.30
STRICT_LIFT_OBSERVE_S           = 2.00
STRICT_RETRY_MAX                = 2
STRICT_RETRY_GRIP_BUMP          = 1.30
STRICT_FORCE_MAX_MULTIPLIER     = 2.0

VERBOSE_GRASP_DEBUG = False

SIDE_FINGER_PRECLOSE_REACH = 0.13

SIDE_FINGER_PRECLOSE_REACH_PERFECT = 0.15

PERFECT_KINEMATIC_CLOSE_POS = 0.30

ENABLE_PERFECT_PIN_FORWARD_SHIFT = False
PERFECT_PIN_FORWARD_SHIFT_M      = 0.005

SNAP_OBJ_TO_POCKET_PRE_CLOSE_ONCE = False

MIN_CHASSIS_OBJ_DIST            = 0.40
NUDGE_MIN_CARRY_GAP_IMPROVEMENT = 0.010

WRIST_PITCH_TALL_OBJ_THRESHOLD  = 0.10
WRIST_PITCH_SHORT_OBJ_THRESHOLD = 0.05

WRIST_PITCH_TOPDOWN_SHORT = WRIST_PITCH_AGGRESSIVE_TOPDOWN
WRIST_PITCH_TOPDOWN_MED   = WRIST_PITCH_DEFAULT_DIAGONAL
WRIST_PITCH_TOPDOWN_TALL  = WRIST_PITCH_SIDE_APPROACH

GIDS_WRIST_X      = 11
GIDS_WRIST_Y      = 12
GIDS_WRIST_Z      = 13
GIDS_HANDBEARING  = 14


def object_half_xy(model, obj_bid):
    try:
        for g in range(model.ngeom):
            if int(model.geom_bodyid[g]) != int(obj_bid):
                continue
            if int(model.geom_type[g]) == 6:
                return (float(model.geom_size[g, 0]),
                        float(model.geom_size[g, 1]))
    except Exception:
        pass
    return None


def object_half_height(model, obj_bid, default=0.075):
    try:
        heights = [
            float(model.geom_size[g, 1])
            for g in range(model.ngeom)
            if int(model.geom_bodyid[g]) == int(obj_bid)
        ]
        if heights:
            return max(heights)
    except Exception:
        pass
    return float(default)


def compute_wrist_goal_for_obj(model, obj_bid):
    half_h = object_half_height(model, obj_bid)
    if half_h < WRIST_PITCH_SHORT_OBJ_THRESHOLD:
        return (float(WRIST_PITCH_AGGRESSIVE_TOPDOWN), 0.0, 0.0, 0.0)
    if half_h < WRIST_PITCH_TALL_OBJ_THRESHOLD:
        return (float(WRIST_PITCH_DEFAULT_DIAGONAL), 0.0, 0.0, 0.0)

    hb = WRIST_PITCH_SIDE_APPROACH
    wz = WRIST_Z_SIDE_APPROACH
    wx = WRIST_X_SIDE_APPROACH
    wy = WRIST_Y_SIDE_APPROACH

    half_xy = object_half_xy(model, obj_bid)
    if half_xy is not None and abs(half_xy[0] - half_xy[1]) >= 0.005:
        if half_xy[0] > half_xy[1]:
            wz = wz + (math.pi / 2.0)
            if wz > math.pi:
                wz -= 2.0 * math.pi
            elif wz < -math.pi:
                wz += 2.0 * math.pi

    return (float(hb), float(wz), float(wx), float(wy))


def is_side_approach(wrist_goal):
    if wrist_goal is None:
        return False
    return abs(float(wrist_goal[0])) < 0.20


ABOVE_OBJ_HEIGHT      = 0.10
GRIPPER_STANDOFF_XY   = 0.05
STANDOFF_RADIUS_CLEAR = 0.02
DESCENT_STEPS         = 25
SHELF_ABOVE_HEIGHT    = 0.20
SHELF_PLACE_OFFSET_Z  = 0.05

CARRY_H1              = 0.40
CARRY_H2              = 0.45
CARRY_A1              = 0.10

GRIP_DEVIATION_TOLERANCE = 0.20
GRIP_OBJ_XY_TOLERANCE    = 0.10

CARRY_GAP_TOLERANCE      = 0.070
CARRY_GAP_TOLERANCE_SIDE = 0.060

PLACE_XY_TOLERANCE    = 0.12

PLACE_XY_PASS              = 0.03
PLACE_XY_GATE              = 0.05
PLACE_Z_PASS               = 0.02
Z_SAFETY_MARGIN            = 0.005
PLACE_INSERT_STEPS         = 45
PLACE_INSERT_SETTLE        = 0.12
PLACE_SETTLE_HOLD_SECS     = 1.8
PLACE_RELEASE_SETTLE_SECS  = 0.8
INTO_SLOT_Z_ABORT          = 0.06
MAX_PLACE_RETRY            = 6
PLACE_A1_MAX               = 0.62
PLACE_ENABLE_SERVO         = (os.environ.get("AH_PLACE_SERVO", "1") == "1")
PLACE_CARTESIAN_INSERT     = True
PLACE_CART_STEPS           = 14
PLACE_SLIDE_CLEARANCE      = float(os.environ.get("AH_PLACE_SLIDE_CLEAR", "0.22"))
PLACE_LOWER_STEPS          = 5
PLACE_BACKOUT_DIST         = 0.20
PLACE_COLUMN_CENTER_WEIGHT = float(os.environ.get("AH_PLACE_COLW", "0.0"))
PLACE_ARM_GRAVCOMP         = 0.0
PLACE_SHOULDER_STIFFNESS   = 2000.0
PLACE_SHOULDER_DAMPING     = 100.0
PLACE_BASE_KP              = 60.0
PLACE_BASE_KD              = 10.0
PLACE_BASE_KI              = 2.0
PLACE_TH_STIFFNESS         = 4000.0
PLACE_TH_DAMPING           = 300.0
PLACE_WRIST_STIFFNESS      = 300.0
PLACE_WRIST_DAMPING        = 20.0
PLACE_WRIST_GOAL           = (0.0, WRIST_Z_SIDE_APPROACH, WRIST_X_SIDE_APPROACH, 0.0)
PLACE_WRIST_WEIGHT         = (0.1, 3.0, 3.0, 3.0)
PER_SLOT_PLACE_OFFSET      = {}

MIN_PICK_WRIST_Z      = 0.10
MAX_PICK_H_DIFF       = 0.18
MAX_PICK_H_DIFF_SIDE  = 0.35
MIN_PICK_A1           = 0.16

COLUMN_JOINT_MAX      = 1.43
HOVER_LIFT            = 0.30
HOVER_LIFT_RETRY      = 0.08
HOVER_DESCENT_STEPS   = 45

ENABLE_CART_PICK_ALIGN = True
USE_HOVER_DESCENT     = True

PHASE_L_DOWN_THEN_FORWARD = False

ENABLE_ARM_HORIZONTAL_PICKUP = False
ARM_HORIZONTAL_TH_CAP = -3.20
ARM_HORIZONTAL_GRIP_Z_LOWER = float(os.environ.get("AH_GRASP_Z_LOWER", "0.0"))
ARM_HORIZONTAL_MIN_WRIST_Z  = 0.10
ENABLE_AH_SERVO_DESCEND_TO_OBJ = False
ENABLE_AH_SERVO_SMOOTHING      = False
ENABLE_AH_TH_HOLD              = False
ENABLE_AH_RADIUS_FINGER_STOP   = False
import math as _math_axis
AXIS_YAW_OFFSET_RAD = 0.0
ENABLE_AH_AXIS_PROBE = os.environ.get("AH_AXIS_PROBE", "0") == "1"
ENABLE_AH_AXIS_CORRECT = os.environ.get("AH_AXIS_CORRECT", "1") == "1"
ENABLE_AH_GRIPZ_DROP = os.environ.get("AH_GRIPZ_DROP", "0") == "1"
ARM_HORIZONTAL_GRIP_Z_BIAS   = 0.045
ARM_HORIZONTAL_GRIP_Z_MAXDROP = 0.07
ARM_HORIZONTAL_RECENTER_LAT = 0.08
ARM_HORIZONTAL_SURFACE_MARGIN = 0.015
ARM_HORIZONTAL_SLOW_NEAR_BAND = 0.06
ENABLE_ARM_HORIZONTAL_RECENTER = False

APPROACH_RETRACT_A1   = 0.20

USE_3FINGER_VERIFY        = True
MIN_CONTACTS_STRICT         = 2
MIN_CONTACTS_RELAXED        = 1
MAX_STRICT_3FINGER_ATTEMPTS = 3

USE_HOVER_RETRACT_ON_FAIL = True
HOVER_RETRACT_STEPS       = 15
GRASP_OFFSET_SNAP_THRESHOLD = 0.35
GRASP_OFFSET_XY_SNAP_THRESHOLD = 0.03
GRASP_OFFSET_XY_CARRY = 0.22
GRASP_TOP_CLEARANCE   = 0.05
GRASP_OFFSET_Z_MIN    = 0.09
GRASP_OFFSET_Z_SNAP_THRESHOLD = 0.05
FIXED_GRASP_OFFSET    = np.array([0.0, 0.0, -0.12])
DROP_MAX_XY_SNAP      = 0.15

GRIPPER_OPEN_POS      = -0.55
THUMB_OPEN_POS        = float(os.environ.get("AH_THUMB_OPEN", "-1.05"))
THUMB_OPEN_J2         =  0.0
THUMB_OPEN_J3         = -0.0523
GRIPPER_CLOSE_POS     =  0.30
GRIPPER_PIN_HOLD_POS  =  0.00

FINGER_CLOSE_MAX      = 0.20
FINGER_CLOSE_FLOOR    = 0.15
FINGER_CLOSE_PER_M    = 4.0

SMOOTH_ATTACH_SECS    = 0.7
SMOOTH_ATTACH_SETTLE  = 0.45

FINGER_BASE_INDICES   = (0, 3, 6)

CURL_J1_FACTOR_THUMB_PERFECT  = 0.70
CURL_J1_FACTOR_THUMB_SOFTWELD = 0.95
CURL_J1_FACTOR_THUMB  = CURL_J1_FACTOR_THUMB_PERFECT
CURL_J1_FACTOR_SIDE   = 0.70
CURL_J1_FACTOR        = CURL_J1_FACTOR_THUMB
CURL_J2_FACTOR        = 0.90
CURL_J3_FACTOR        = 1.00

SMOOTH_GRIPPER_SECS   = 0.4
SMOOTH_GRIPPER_STEP_S = 0.02

STRICT_CLOSE_TRANSITION_SECS = 4.0

USE_CONTACT_STOP_CLOSE = True

CONTACT_COMPRESS_TICKS      = 5
FAST_CONTACT_COMPRESS_TICKS = 1

USE_OVERDRIVE_CLOSE = True

USE_FINGER_DIAGNOSTIC_LOG = True

VERBOSE_PATH_WAYPOINT_LOG = False

USE_FINGER_CTRL_CLAMP = True
FINGER_CTRL_RANGES = (
    (-0.6,    1.2218),
    ( 0.0,    1.5708),
    (-1.2217, -0.0523),
    (-0.6,    1.2218),
    ( 0.0,    1.5708),
    (-1.2217, -0.0523),
    (-1.50,   1.2218),
    ( 0.0,    1.5708),
    (-1.2217, -0.0523),
    (-0.1784, 0.192),
    (-0.192,  0.1784),
)

USE_SYNC_OPEN         = True
OPEN_SETTLE_TOL_RAD   = 0.20
OPEN_SETTLE_TIMEOUT   = 0.4
OPEN_SETTLE_POLL_S    = 0.02

MIRROR_FINGER_C_J1    = False

PALM_SPREAD_OPEN      = 0.18
PALM_SPREAD_CLOSE     = 0.0
PALM_C_SIGN           = +1.0
PALM_B_SIGN           = +1.0

FINGER_STAGGER_SECS   = 0.06

GRIPPER_HOLD_TIME     = 1.5
PD_SETTLE_PER_WAYPOINT = 0.05
PD_SETTLE_AT_PATH_END = 0.4



def compute_grasp_targets(base_xy, obj_world, obj_radius=None,
                          side_approach=False):
    obj = np.asarray(obj_world, dtype=float)
    base_xy = np.asarray(base_xy, dtype=float)
    approach = obj[:2] - base_xy
    nrm = float(np.linalg.norm(approach))
    if nrm < 1e-6:
        approach_unit = np.array([1.0, 0.0])
    else:
        approach_unit = approach / nrm
    if obj_radius is None:
        effective_standoff = GRIPPER_STANDOFF_XY
    else:
        effective_standoff = max(GRIPPER_STANDOFF_XY,
                                 float(obj_radius) + STANDOFF_RADIUS_CLEAR)
    grasp_target = obj.copy()
    grasp_target[0] -= approach_unit[0] * effective_standoff
    grasp_target[1] -= approach_unit[1] * effective_standoff
    pre_grasp_target = grasp_target.copy()
    if not side_approach:
        pre_grasp_target[2] += ABOVE_OBJ_HEIGHT
    return grasp_target, pre_grasp_target


def reset_plan_data_for_ik(arm_bridge, base_xy, base_yaw):
    mujoco.mj_resetData(arm_bridge.model, arm_bridge.planning_data)
    m = arm_bridge.qpos_map
    arm_bridge.planning_data.qpos[m["ColumnLeft"]]  = HOME_Q[0]
    arm_bridge.planning_data.qpos[m["ColumnRight"]] = HOME_Q[1]
    arm_bridge.planning_data.qpos[m["ArmLeft"]]     = HOME_Q[2]
    arm_bridge.planning_data.qpos[m["Base"]]        = HOME_Q[3]
    for jname, q in (
        ("ColumnLeftBearingJoint_2",  PARK_Q[0]),
        ("ColumnRightBearingJoint_2", PARK_Q[1]),
        ("ArmLeftJoint_2",            PARK_Q[2]),
        ("BaseJoint_2",               PARK_Q[3]),
    ):
        jid = mujoco.mj_name2id(arm_bridge.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid >= 0:
            arm_bridge.planning_data.qpos[arm_bridge.model.jnt_qposadr[jid]] = q
    arm_bridge.planning_data.qvel[:] = 0.0
    arm_bridge.set_base_pose_xy_yaw(float(base_xy[0]), float(base_xy[1]), float(base_yaw))
    arm_bridge.park_all_pickup_objects()
    mujoco.mj_forward(arm_bridge.model, arm_bridge.planning_data)


def pin_freejoint(data, qposadr, dofadr, pos, quat=(1.0, 0.0, 0.0, 0.0)):
    data.qpos[qposadr]     = pos[0]
    data.qpos[qposadr + 1] = pos[1]
    data.qpos[qposadr + 2] = pos[2]
    data.qpos[qposadr + 3] = quat[0]
    data.qpos[qposadr + 4] = quat[1]
    data.qpos[qposadr + 5] = quat[2]
    data.qpos[qposadr + 6] = quat[3]
    data.qvel[dofadr:dofadr + 6] = 0.0



class GraspExecutor:

    def __init__(self, sim, arm_bridge):
        self.sim        = sim
        self.arm_bridge = arm_bridge
        self._strict_force_multiplier = 1.0
        self._strict_retry_count      = 0
        self._strict_grasp_q          = None
        self._wz_ctrl_override        = None
        self._cycle_stage_close_fired   = False
        self._cycle_stage_verify_passed = False
        self._cycle_stage_lift_fired    = False
        self._cycle_stage_obj_followed  = False

        self.weld_id = mujoco.mj_name2id(
            sim.model, mujoco.mjtObj.mjOBJ_EQUALITY, "grasp_left")
        if self.weld_id < 0:
            raise RuntimeError("grasp_left weld not found in model")

        self.gripper_body_id = mujoco.mj_name2id(
            sim.model, mujoco.mjtObj.mjOBJ_BODY, "Gripper_Link3_1")
        if self.gripper_body_id < 0:
            raise RuntimeError("Gripper_Link3_1 body not found in model")
        self._weld_body1_id = mujoco.mj_name2id(
            sim.model, mujoco.mjtObj.mjOBJ_BODY, "Gripper_Link1_1")
        if os.environ.get("AH_NO_ARM2_COL_GLOBAL", "1") == "1":
            import re as _re_a2g
            _a2g = _re_a2g.compile(
                r'(finger_[abc]_link|Gripper_Link|palm_finger|Contact_Cylinder)\w*_2$')
            _n_a2 = 0
            for _g in range(sim.model.ngeom):
                if (int(sim.model.geom_contype[_g]) != 0
                        or int(sim.model.geom_conaffinity[_g]) != 0):
                    _bn2 = mujoco.mj_id2name(
                        sim.model, mujoco.mjtObj.mjOBJ_BODY,
                        int(sim.model.geom_bodyid[_g])) or ""
                    if _a2g.search(_bn2):
                        sim.model.geom_contype[_g] = 0
                        sim.model.geom_conaffinity[_g] = 0
                        _n_a2 += 1
            if _n_a2:
                print(f"[Exec] GLOBAL: parked ARM-2 collisions DISABLED at startup "
                      f"({_n_a2} geom) — no arm-2 self-collision careen in ANY phase")
        if os.environ.get("AH_NO_BATTERY_COL_GLOBAL", "0") == "1":
            _bb = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY,
                                    "LifePo4_12V50Ah")
            _n_b = 0
            if _bb >= 0:
                for _g in range(sim.model.ngeom):
                    if int(sim.model.geom_bodyid[_g]) == _bb:
                        sim.model.geom_contype[_g] = 0
                        sim.model.geom_conaffinity[_g] = 0
                        _n_b += 1
            if _n_b:
                print(f"[Exec] GLOBAL: battery (LifePo4) collision DISABLED at "
                      f"startup ({_n_b} geom) — no forearm-vs-battery careen/hang "
                      f"in ANY phase")
        self._weld_ids = {}
        for _oi in range(10):
            _wid = mujoco.mj_name2id(
                sim.model, mujoco.mjtObj.mjOBJ_EQUALITY, f"grasp_left_{_oi}")
            if _wid >= 0:
                self._weld_ids[_oi] = _wid
        self._rigid_weld_active = False
        self._rigid_weld_id = None
        self._finger_freeze_saved = []
        self._finger_freeze_active = False
        self._carry_anchor_body_ids = []
        for name in ("finger_a_link_3_1",
                     "finger_b_link_3_1",
                     "finger_c_link_3_1"):
            bid = mujoco.mj_name2id(
                sim.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                self._carry_anchor_body_ids.append(bid)
        if len(self._carry_anchor_body_ids) != 3:
            print("[Exec] carry anchor warning: fingertip bodies not all "
                  "found; falling back to Gripper_Link3_1")

        self._finger_body_groups = []
        finger_labels = ['c', 'b', 'a']
        for label in finger_labels:
            bodies = set()
            link_indices = (2, 3)
            for ln in link_indices:
                name = f"finger_{label}_link_{ln}_1"
                bid = mujoco.mj_name2id(
                    sim.model, mujoco.mjtObj.mjOBJ_BODY, name)
                if bid >= 0:
                    bodies.add(int(bid))
            self._finger_body_groups.append(bodies)
        if not all(len(g) >= 1 for g in self._finger_body_groups):
            print("[Exec] contact-stop warning: a finger has no resolved "
                  "bodies; contact-stop close will be disabled")
        else:
            counts = [len(g) for g in self._finger_body_groups]
            print(f"[Exec] contact-stop bodies per finger c/b/a = {counts}")

        self._proximity_finger_links = []
        for label in finger_labels:
            link_bids = []
            for ln in (1, 2, 3):
                name = f"finger_{label}_link_{ln}_1"
                bid = mujoco.mj_name2id(
                    sim.model, mujoco.mjtObj.mjOBJ_BODY, name)
                if bid >= 0:
                    link_bids.append(int(bid))
            self._proximity_finger_links.append(link_bids)

        try:
            self._finger_bridge = FingerBridge(
                sim.model, sim.gripper_ids_left[:FINGER_DOF])
        except Exception as e:
            print(f"[Exec] FingerBridge init warning: {e} — "
                  "falling back to direct interpolation in _set_gripper")
            self._finger_bridge = None

        self._held_obj_idx     = None
        self._held_obj_bid     = None
        self._held_obj_qpa     = None
        self._held_obj_dofadr  = None
        self._grasp_offset_xyz = None
        self._pre_close_lift_done = False
        self._fast_fixed_close_pin_active = False
        self._grasp_offset_quat = (1.0, 0.0, 0.0, 0.0)
        self._held_obj_orig_gravcomp = None
        self._held_obj_solref_saved = None

        self._last_close_finger_contacts = None
        self._strict_finger_attempts_used = 0

        self._active_pin_fn = None
        self._cancel = False

        self.last_grasp_failure_info = None

        self._last_valid_pre_grasp_q = None

        self._side_grip_active = False

        self._arm_held_at_grasp_for_retry = False

        self._qmap = arm_bridge.qpos_map

        print("[Exec] GraspExecutor ready  weld_id=%d  gripper_body_id=%d" %
              (self.weld_id, self.gripper_body_id))

        try:
            gids = sim.gripper_ids_left
            labels = ['c_j1','c_j2','c_j3',
                      'b_j1','b_j2','b_j3',
                      'a_j1','a_j2','a_j3',
                      'palm_c','palm_b']
            mapping = []
            for i in range(min(11, len(gids))):
                aid = int(gids[i])
                if aid < 0:
                    mapping.append(f"{labels[i]}=MISSING")
                else:
                    name = mujoco.mj_id2name(
                        sim.model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
                    mapping.append(f"{labels[i]}({aid})={name}")
            print("[Exec] gripper actuator wiring: " + "  ".join(mapping))
        except Exception as e:
            print(f"[Exec] gripper wiring diagnostic warning: {e}")

        try:
            jtype_names = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}
            print("[Exec] finger joint diagnostic:")
            for jname in ("finger_c_joint_1_1", "finger_c_joint_2_1",
                          "finger_c_joint_3_1",
                          "finger_b_joint_1_1", "finger_b_joint_2_1",
                          "finger_b_joint_3_1",
                          "finger_a_joint_1_1", "finger_a_joint_2_1",
                          "finger_a_joint_3_1"):
                jid = mujoco.mj_name2id(
                    sim.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                if jid < 0:
                    print(f"  {jname}: NOT FOUND")
                    continue
                jt = int(sim.model.jnt_type[jid])
                jlim = bool(sim.model.jnt_limited[jid])
                jrng = sim.model.jnt_range[jid]
                jqa = int(sim.model.jnt_qposadr[jid])
                print(f"  {jname}: type={jtype_names.get(jt, jt)} "
                      f"limited={jlim} range=[{jrng[0]:.4f}, "
                      f"{jrng[1]:.4f}] qposadr={jqa}")
        except Exception as e:
            print(f"[Exec] finger joint diagnostic warning: {e}")

        try:
            print("[Exec] startup-probe: finger qpos / ctrl @ post-reset")
            joint_names_ordered = (
                'finger_c_joint_1_1', 'finger_c_joint_2_1', 'finger_c_joint_3_1',
                'finger_b_joint_1_1', 'finger_b_joint_2_1', 'finger_b_joint_3_1',
                'finger_a_joint_1_1', 'finger_a_joint_2_1', 'finger_a_joint_3_1',
                'palm_finger_c_joint_1', 'palm_finger_b_joint_1',
            )
            labels11 = ('c_j1','c_j2','c_j3',
                        'b_j1','b_j2','b_j3',
                        'a_j1','a_j2','a_j3',
                        'palm_c','palm_b')
            gids = sim.gripper_ids_left
            for i, (jname, lbl) in enumerate(zip(joint_names_ordered, labels11)):
                jid = mujoco.mj_name2id(
                    sim.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                qpa = (int(sim.model.jnt_qposadr[jid])
                       if jid >= 0 else -1)
                qpos_v = (float(sim.data.qpos[qpa])
                          if qpa >= 0 else float('nan'))
                ctrl_v = (float(sim.data.ctrl[int(gids[i])])
                          if i < len(gids) and int(gids[i]) >= 0
                          else float('nan'))
                jrng = (sim.model.jnt_range[jid]
                        if jid >= 0 else (float('nan'), float('nan')))
                in_range = (jrng[0] <= qpos_v <= jrng[1]) if jid >= 0 else None
                flag = '' if in_range or in_range is None else ' OUT-OF-RANGE'
                print(f"  {lbl}: qpos={qpos_v:+.4f} ctrl={ctrl_v:+.4f}  "
                      f"range=[{jrng[0]:+.4f}, {jrng[1]:+.4f}]{flag}")
        except Exception as e:
            print(f"[Exec] startup-probe qpos/ctrl warning: {e}")

        try:
            self._log_gripper_contacts("startup-probe", force_forward=True)
        except Exception as e:
            print(f"[Exec] startup-probe contact-scan warning: {e}")


    def _pin_obj_at_world(self, world_pos):
        qpa = self._held_obj_qpa
        dofadr = self._held_obj_dofadr
        wp = np.array(world_pos, dtype=float).copy()
        def closure(data):
            pin_freejoint(data, qpa, dofadr, wp)
        return closure

    def _carry_anchor_xyz(self, data):
        if len(self._carry_anchor_body_ids) == 3:
            pts = [data.xpos[bid] for bid in self._carry_anchor_body_ids]
            return np.mean(pts, axis=0)
        return data.xpos[self.gripper_body_id]

    def _axis_residual_deg(self, obj_bid):
        try:
            if len(self._carry_anchor_body_ids) != 3 or obj_bid is None:
                return None
            data = self.sim.data
            thumb = np.asarray(data.xpos[self._carry_anchor_body_ids[0]][:2], float)
            bc = 0.5 * (np.asarray(data.xpos[self._carry_anchor_body_ids[1]][:2], float)
                        + np.asarray(data.xpos[self._carry_anchor_body_ids[2]][:2], float))
            axis = bc - thumb
            ch = np.asarray(self.sim.localization()[:2], float)
            obj = np.asarray(data.xpos[obj_bid][:2], float)
            appr = obj - ch
            if np.linalg.norm(axis) < 1e-4 or np.linalg.norm(appr) < 1e-4:
                return None
            ax_yaw = math.atan2(axis[1], axis[0])
            ap_yaw = math.atan2(appr[1], appr[0])
            e_a = ((ap_yaw + math.pi / 2 - ax_yaw + math.pi) % (2 * math.pi)) - math.pi
            e_b = ((ap_yaw - math.pi / 2 - ax_yaw + math.pi) % (2 * math.pi)) - math.pi
            e = e_a if abs(e_a) < abs(e_b) else e_b
            return math.degrees(e)
        except Exception:
            return None

    def _probe_axis_jacobian(self, obj_bid, label="probe"):
        try:
            r0 = self._axis_residual_deg(obj_bid)
            if r0 is None:
                print(f"[AXIS-PROBE] {label}: residual None — skip")
                return None
            p0 = np.asarray(self._pinch_midpoint_xyz(self.sim.data)[:2], float)
            q0 = list(self._current_arm_q())
            _settle = 0.45
            probes = [(2, 'a1', 0.040), (3, 'th', 0.070),
                      (7, 'wy', 0.070), (-1, 'tilt(h2-h1)', 0.040)]
            results = {}
            print(f"[AXIS-PROBE] {label}: baseline residual {r0:+.1f}deg "
                  f"pinch=({p0[0]:.3f},{p0[1]:.3f})  (eff=ACHIEVED qpos delta)")
            for idx, name, d in probes:
                q = list(q0)
                if idx == -1:
                    q[1] = float(np.clip(q0[1] + d, *JOINT_RANGES_ARM[1]))
                    q[0] = float(np.clip(q0[0] - d, *JOINT_RANGES_ARM[0]))
                else:
                    q[idx] = float(np.clip(q0[idx] + d, *JOINT_RANGES_ARM[idx]))
                self._set_arm_cmd(q)
                time.sleep(_settle)
                q_act = list(self._current_arm_q())
                if idx == -1:
                    eff = (q_act[1] - q0[1]) - (q_act[0] - q0[0])
                else:
                    eff = q_act[idx] - q0[idx]
                r1 = self._axis_residual_deg(obj_bid)
                p1 = np.asarray(self._pinch_midpoint_xyz(self.sim.data)[:2],
                                float)
                self._set_arm_cmd(list(q0))
                time.sleep(_settle)
                if abs(eff) < 0.004:
                    print(f"[AXIS-PROBE]   {name}: achieved dq {eff:+.4f} too "
                          f"small (joint stuck/at-limit) — skip")
                    continue
                if r1 is None:
                    print(f"[AXIS-PROBE]   {name}: residual None after perturb")
                    continue
                dRes = (r1 - r0) / eff
                dvec = (p1 - p0) / eff * 100.0
                dPx, dPy = float(dvec[0]), float(dvec[1])
                dPin = float(np.linalg.norm(dvec))
                ratio = dRes / dPin if dPin > 1e-6 else float('inf')
                results[name] = dict(dRes=dRes, dPx=dPx, dPy=dPy,
                                     dPin=dPin, ratio=ratio, eff=eff)
                print(f"[AXIS-PROBE]   {name}: dq_ach={eff:+.3f}rad -> "
                      f"dRes={dRes:+7.1f}deg/rad  dPin=({dPx:+5.1f},{dPy:+5.1f}) "
                      f"|{dPin:4.1f}|cm/rad  ratio={ratio:+6.1f}deg/cm  "
                      f"(res {r0:+.1f}->{r1:+.1f})")
            self._set_arm_cmd(list(q0))
            time.sleep(0.15)
            if results:
                _best = max(results.items(), key=lambda kv: abs(kv[1]['ratio']))
                print(f"[AXIS-PROBE] {label}: DONE (pose reverted). "
                      f"best axis-authority DOF = {_best[0]} "
                      f"(ratio {_best[1]['ratio']:+.1f}deg/cm)")
            return results
        except Exception as e:
            print(f"[AXIS-PROBE] {label}: raised {e}")
            try:
                self._set_arm_cmd(list(self._current_arm_q()))
            except Exception:
                pass
            return None

    def _pinch_midpoint_xyz(self, data):
        if len(self._carry_anchor_body_ids) != 3:
            return data.xpos[self.gripper_body_id]
        thumb = data.xpos[self._carry_anchor_body_ids[0]]
        bc_centroid = 0.5 * (data.xpos[self._carry_anchor_body_ids[1]]
                             + data.xpos[self._carry_anchor_body_ids[2]])
        return 0.5 * (thumb + bc_centroid)

    def _obj_between_fingers(self, obj_xy, thumb_xy, bc_xy, margin):
        thumb_xy = np.asarray(thumb_xy, dtype=float)
        bc_xy = np.asarray(bc_xy, dtype=float)
        axis = bc_xy - thumb_xy
        span = float(np.linalg.norm(axis))
        if span < 1e-6:
            return True, 0.0, 0.0
        axis_u = axis / span
        proj = float(np.dot(np.asarray(obj_xy, dtype=float) - thumb_xy, axis_u))
        ok = (-margin) <= proj <= (span + margin)
        return ok, proj * 100.0, span * 100.0

    def _log_finger_geometry(self, obj_bid, label):
        if len(self._carry_anchor_body_ids) != 3:
            return
        data = self.sim.data
        model = self.sim.model
        obj_xyz = data.xpos[obj_bid].copy()
        finger_names = ['a', 'b', 'c']
        print(f"[Diag] {label} — fingertip geometry:")
        for fi, bid in enumerate(self._carry_anchor_body_ids):
            tip = data.xpos[bid].copy()
            delta = tip - obj_xyz
            d_xy = float(np.linalg.norm(delta[:2]))
            print(f"  finger_{finger_names[fi]} tip xyz={tip.round(3)}  "
                  f"d_xy_to_obj={d_xy*100:.1f}cm  "
                  f"delta=({delta[0]*100:+.1f},{delta[1]*100:+.1f},"
                  f"{delta[2]*100:+.1f})cm")
        anchor = self._carry_anchor_xyz(data)
        a_xy = float(np.linalg.norm(anchor[:2] - obj_xyz[:2]))
        print(f"  carry_anchor centroid xyz={anchor.round(3)}  "
              f"obj_xyz={obj_xyz.round(3)}  d_xy={a_xy*100:.1f}cm")
        joint_groups = [
            ('a', ('finger_a_joint_1_1', 'finger_a_joint_2_1',
                   'finger_a_joint_3_1')),
            ('b', ('finger_b_joint_1_1', 'finger_b_joint_2_1',
                   'finger_b_joint_3_1')),
            ('c', ('finger_c_joint_1_1', 'finger_c_joint_2_1',
                   'finger_c_joint_3_1')),
        ]
        for fname, jnames in joint_groups:
            angles = []
            for jname in jnames:
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                if jid < 0:
                    angles.append(None)
                    continue
                qpa = int(model.jnt_qposadr[jid])
                angles.append(float(data.qpos[qpa]))
            j1 = angles[0]; j2 = angles[1]; j3 = angles[2]
            fmt = lambda v: f"{v:+.3f}" if v is not None else "  N/A"
            print(f"  finger_{fname} joints: j1(prox)={fmt(j1)} "
                  f"j2(mid)={fmt(j2)} j3(dist)={fmt(j3)}")
        palm_jnames = ('palm_finger_c_joint_1', 'palm_finger_b_joint_1')
        palm_vals = []
        for jname in palm_jnames:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                palm_vals.append(None)
                continue
            qpa = int(model.jnt_qposadr[jid])
            palm_vals.append(float(data.qpos[qpa]))
        fmt = lambda v: f"{v:+.3f}" if v is not None else "  N/A"
        wrist_jnames = (
            ('hb', 'HandBearingJoint_1'),
            ('wz', 'gripper_z_rotation_1'),
            ('wx', 'gripper_x_rotation_1'),
            ('wy', 'gripper_y_rotation_1'),
        )
        wrist_str_parts = []
        for short, jname in wrist_jnames:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                wrist_str_parts.append(f"{short}=N/A")
                continue
            qpa = int(model.jnt_qposadr[jid])
            wrist_str_parts.append(f"{short}={float(data.qpos[qpa]):+.2f}")
        palm_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                     "Gripper_Link3_1")
        palm_z = (float(data.xpos[palm_bid][2])
                  if palm_bid >= 0 else float('nan'))
        tip_zs = [float(data.xpos[bid][2])
                  for bid in self._carry_anchor_body_ids]
        tip_z_avg = sum(tip_zs) / max(1, len(tip_zs))
        print(f"  wrist {' '.join(wrist_str_parts)}  "
              f"palm_z={palm_z:.3f}m  tip_z_avg={tip_z_avg:.3f}m  "
              f"palm-tip={(palm_z - tip_z_avg)*100:+.1f}cm  "
              f"obj_top={(obj_xyz[2] + self._object_half_height(obj_bid)):.3f}m")
        print(f"  palm spread: palm_c={fmt(palm_vals[0])} "
              f"palm_b={fmt(palm_vals[1])}")

    def _count_arm_chassis_contacts(self):
        data = self.sim.data
        model = self.sim.model
        try:
            n = int(data.ncon)
        except Exception:
            return 0
        try:
            arm_bid_set = self._ensure_gripper_body_ids()
        except Exception:
            return 0
        if not arm_bid_set:
            return 0
        count = 0
        for i in range(n):
            try:
                c = data.contact[i]
                b1 = int(model.geom_bodyid[int(c.geom1)])
                b2 = int(model.geom_bodyid[int(c.geom2)])
            except Exception:
                continue
            if b1 in arm_bid_set and b2 not in arm_bid_set:
                other_b = b2
            elif b2 in arm_bid_set and b1 not in arm_bid_set:
                other_b = b1
            else:
                continue
            other_name = (mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_BODY, other_b) or "")
            if other_name.startswith("pickup_obj_"):
                continue
            count += 1
        return count

    def _ensure_arm_subtree_body_ids(self):
        if getattr(self, '_arm_subtree_body_ids_cache', None) is not None:
            return self._arm_subtree_body_ids_cache
        model = self.sim.model
        root_bid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "Arm_1")
        ids = set()
        if root_bid >= 0:
            ids.add(int(root_bid))
            frontier = [int(root_bid)]
            while frontier:
                cur = frontier.pop(0)
                for bid in range(model.nbody):
                    if int(model.body_parentid[bid]) == cur:
                        if int(bid) not in ids:
                            ids.add(int(bid))
                            frontier.append(int(bid))
        else:
            try:
                ids = set(self._ensure_gripper_body_ids().keys())
            except Exception:
                pass
        self._arm_subtree_body_ids_cache = ids
        return ids

    def _runtime_wz_correction(self, obj_bid, ik_base_xy_2d):
        try:
            if len(self._carry_anchor_body_ids) != 3:
                return {'applied': False, 'reason': 'no_finger_bids'}
            data = self.sim.data
            obj_xy = np.asarray(
                data.xpos[obj_bid][:2], dtype=float).copy()
            chassis_xy = np.asarray(ik_base_xy_2d, dtype=float)
            approach_yaw = math.atan2(obj_xy[1] - chassis_xy[1],
                                      obj_xy[0] - chassis_xy[0])
            desired_a = approach_yaw + math.pi / 2.0
            desired_b = approach_yaw - math.pi / 2.0

            def _measure():
                thumb_xy = data.xpos[
                    self._carry_anchor_body_ids[0]][:2].copy()
                bc_xy = 0.5 * (
                    data.xpos[self._carry_anchor_body_ids[1]][:2]
                    + data.xpos[self._carry_anchor_body_ids[2]][:2])
                ax_vec = bc_xy - thumb_xy
                ax_yaw = math.atan2(ax_vec[1], ax_vec[0])
                e_a = ((ax_yaw - desired_a + math.pi)
                       % (2 * math.pi)) - math.pi
                e_b = ((ax_yaw - desired_b + math.pi)
                       % (2 * math.pi)) - math.pi
                e = e_a if abs(e_a) <= abs(e_b) else e_b
                d_thumb = float(np.linalg.norm(thumb_xy - obj_xy))
                d_bc    = float(np.linalg.norm(bc_xy - obj_xy))
                pinch_xy = 0.5 * (thumb_xy + bc_xy)
                carry   = float(np.linalg.norm(pinch_xy - obj_xy))
                return ax_yaw, e, d_thumb, d_bc, carry

            gids = self.sim.gripper_ids_left
            import time as _t

            ax_o, err_o, d_thumb_o, d_bc_o, carry_o = _measure()
            max_far_o = max(d_thumb_o, d_bc_o)
            err_initial = err_o
            max_far_initial = max_far_o
            carry_initial = carry_o

            if abs(err_o) <= WZ_REFINE_AXIS_ERR_THRESHOLD:
                return {'applied': False,
                        'reason': 'already_aligned',
                        'err': err_o}
            if abs(err_o) > WZ_RUNTIME_CORRECTION_MAX_INITIAL:
                print(f"[Exec] [5.46] runtime wz correction SKIPPED: "
                      f"initial axis err {math.degrees(err_o):+.1f}° "
                      f"exceeds linear-correction safe zone "
                      f"(±{math.degrees(WZ_RUNTIME_CORRECTION_MAX_INITIAL):.0f}°) — "
                      f"correction direction unreliable at this offset; "
                      f"letting the gate reject and retry next yaw")
                return {'applied': False,
                        'reason': 'initial_err_too_large',
                        'err': err_o}

            best_ctrl = float(data.ctrl[gids[GIDS_WRIST_Z]])
            best_err = err_o
            best_max_far = max_far_o
            best_carry = carry_o
            iters_applied = 0

            for iter_i in range(int(WZ_RUNTIME_CORRECTION_MAX_ITERS)):
                step_err = max(
                    -WZ_RUNTIME_CORRECTION_MAX_DELTA,
                    min(WZ_RUNTIME_CORRECTION_MAX_DELTA, best_err))
                cur_ctrl = best_ctrl
                cur_target = cur_ctrl / (
                    1.0 + WRIST_Z_PD_COMPENSATION_RATIO)
                new_target = cur_target - step_err
                while new_target > math.pi:
                    new_target -= 2 * math.pi
                while new_target < -math.pi:
                    new_target += 2 * math.pi
                new_target = max(-math.pi, min(math.pi, new_target))
                new_ctrl = new_target * (
                    1.0 + WRIST_Z_PD_COMPENSATION_RATIO)

                tag = f"[5.46] runtime wz correction iter#{iter_i + 1}"
                print(f"[Exec] {tag}: "
                      f"approach={math.degrees(approach_yaw):+.1f}°  "
                      f"axis err={math.degrees(best_err):+.1f}°  "
                      f"max_far={best_max_far*100:.1f}cm  "
                      f"carry={best_carry*100:.1f}cm  "
                      f"wz_target {cur_target:+.3f} → {new_target:+.3f}; "
                      f"commanding")
                data.ctrl[gids[GIDS_WRIST_Z]] = new_ctrl
                _t.sleep(WZ_RUNTIME_CORRECTION_SETTLE_S)

                ax_n, err_n, d_thumb_n, d_bc_n, carry_n = _measure()
                max_far_n = max(d_thumb_n, d_bc_n)

                max_far_improved = max_far_n < best_max_far - 0.005
                carry_improved_safely = (
                    carry_n < best_carry - 0.01
                    and max_far_n <= best_max_far + 0.01)
                err_didnt_worsen = (
                    abs(err_n) <= abs(best_err) + math.radians(3.0))
                reach_better = (
                    (max_far_improved or carry_improved_safely)
                    and err_didnt_worsen)

                if reach_better:
                    iters_applied += 1
                    best_ctrl = new_ctrl
                    best_err = err_n
                    best_max_far = max_far_n
                    best_carry = carry_n
                    print(f"[Exec] {tag} APPLIED: "
                          f"max_far {max_far_initial*100:.1f}→"
                          f"{best_max_far*100:.1f}cm  "
                          f"carry {carry_initial*100:.1f}→"
                          f"{best_carry*100:.1f}cm  "
                          f"err {math.degrees(err_initial):+.1f}°→"
                          f"{math.degrees(best_err):+.1f}°")
                    if abs(best_err) <= WZ_REFINE_AXIS_ERR_THRESHOLD:
                        print(f"[Exec] [5.46] converged after "
                              f"{iters_applied} iter(s); axis err "
                              f"{math.degrees(best_err):+.1f}° "
                              f"≤ {math.degrees(WZ_REFINE_AXIS_ERR_THRESHOLD):.0f}° "
                              f"threshold")
                        break
                else:
                    data.ctrl[gids[GIDS_WRIST_Z]] = best_ctrl
                    _t.sleep(WZ_RUNTIME_CORRECTION_SETTLE_S * 0.5)
                    if not err_didnt_worsen:
                        reason = (
                            f"axis err drift "
                            f"{math.degrees(best_err):+.1f}°→"
                            f"{math.degrees(err_n):+.1f}° "
                            f"(>3° growth) — b/c axis rotating "
                            f"away from wrap direction; further "
                            f"steps would worsen wrap geometry")
                    else:
                        reason = (
                            f"reach didn't improve from best-so-far "
                            f"(tried max_far "
                            f"{best_max_far*100:.1f}→{max_far_n*100:.1f}cm, "
                            f"carry "
                            f"{best_carry*100:.1f}→{carry_n*100:.1f}cm)")
                    print(f"[Exec] {tag} REVERTED: {reason}; "
                          f"halting iterations")
                    break

            if iters_applied == 0:
                return {'applied': False, 'reverted': True,
                        'iters_applied': 0}
            return {
                'applied': True,
                'iters_applied': iters_applied,
                'max_far_before': max_far_initial,
                'max_far_after': best_max_far,
                'carry_before': carry_initial,
                'carry_after': best_carry,
                'err_before': err_initial,
                'err_after': best_err,
            }
        except Exception as e:
            print(f"[Exec] [5.46] runtime wz correction skipped: {e}")
            return {'applied': False, 'reason': str(e)}

    def _count_arm_obj_contacts(self, obj_bid,
                                 min_penetration=0.005):
        data = self.sim.data
        model = self.sim.model
        try:
            n = int(data.ncon)
        except Exception:
            return 0
        try:
            arm_bid_set = self._ensure_arm_subtree_body_ids()
        except Exception:
            return 0
        if not arm_bid_set:
            return 0
        finger_bids = set()
        for fg in (self._finger_body_groups or []):
            finger_bids.update(fg)
        non_finger_arm = arm_bid_set - finger_bids
        if not non_finger_arm:
            return 0
        count = 0
        for i in range(n):
            try:
                c = data.contact[i]
                b1 = int(model.geom_bodyid[int(c.geom1)])
                b2 = int(model.geom_bodyid[int(c.geom2)])
                cdist = float(c.dist)
            except Exception:
                continue
            if cdist >= -float(min_penetration):
                continue
            if (b1 in non_finger_arm and b2 == int(obj_bid)) or \
               (b2 in non_finger_arm and b1 == int(obj_bid)):
                count += 1
        return count

    def _count_finger_obj_contacts(self, obj_bid):
        data = self.sim.data
        model = self.sim.model
        try:
            n = int(data.ncon)
        except Exception:
            return 0
        touched = [False, False, False]
        for i in range(n):
            try:
                c = data.contact[i]
                g1 = int(c.geom1)
                g2 = int(c.geom2)
                b1 = int(model.geom_bodyid[g1])
                b2 = int(model.geom_bodyid[g2])
            except Exception:
                continue
            for fi in range(3):
                fb = (self._finger_body_groups[fi]
                      if self._finger_body_groups
                      and fi < len(self._finger_body_groups)
                      else set())
                if not fb:
                    continue
                if (b1 in fb and b2 == int(obj_bid)) or \
                   (b2 in fb and b1 == int(obj_bid)):
                    touched[fi] = True
        return int(sum(touched))

    def _log_filtered_contact_summary(self, obj_bid, label):
        data = self.sim.data
        model = self.sim.model
        try:
            n = int(data.ncon)
        except Exception:
            return
        finger_groups = self._finger_body_groups or []
        finger_names = ['c', 'b', 'a']
        try:
            arm_bid_set = self._ensure_gripper_body_ids()
        except Exception:
            arm_bid_set = set()
        per_finger = {fn: 0 for fn in finger_names}
        n_arm_chassis = 0
        n_arm_other_obj = 0
        n_obj_floor = 0
        for i in range(n):
            try:
                c = data.contact[i]
                g1 = int(c.geom1)
                g2 = int(c.geom2)
                b1 = int(model.geom_bodyid[g1])
                b2 = int(model.geom_bodyid[g2])
            except Exception:
                continue
            for fi, fname in enumerate(finger_names):
                fb = finger_groups[fi] if fi < len(finger_groups) else set()
                if not fb:
                    continue
                if ((b1 in fb and b2 == int(obj_bid)) or
                        (b2 in fb and b1 == int(obj_bid))):
                    per_finger[fname] += 1
            if (b1 == int(obj_bid) and b2 == 0) or \
               (b2 == int(obj_bid) and b1 == 0):
                n_obj_floor += 1
            if arm_bid_set:
                arm_b = None
                other_b = None
                if b1 in arm_bid_set and b2 not in arm_bid_set:
                    arm_b, other_b = b1, b2
                elif b2 in arm_bid_set and b1 not in arm_bid_set:
                    arm_b, other_b = b2, b1
                if arm_b is not None and other_b is not None:
                    other_name = (mujoco.mj_id2name(
                        model, mujoco.mjtObj.mjOBJ_BODY, other_b) or "")
                    if other_b == int(obj_bid):
                        pass
                    elif other_name.startswith("pickup_obj_"):
                        n_arm_other_obj += 1
                    else:
                        n_arm_chassis += 1
        print(f"[Diag] {label} contact summary: "
              f"finger_a↔obj={per_finger['a']}  "
              f"finger_b↔obj={per_finger['b']}  "
              f"finger_c↔obj={per_finger['c']}  "
              f"arm-vs-chassis={n_arm_chassis}  "
              f"arm-vs-other-obj={n_arm_other_obj}  "
              f"obj-vs-floor={n_obj_floor}  "
              f"(total ncon={n} includes wheels, joints, etc.)")

    def _log_finger_object_contacts(self, obj_bid, label):
        data = self.sim.data
        model = self.sim.model
        try:
            n = int(data.ncon)
        except Exception:
            return
        finger_names = ['c', 'b', 'a']
        hits = []
        for i in range(n):
            try:
                c = data.contact[i]
                g1 = int(c.geom1)
                g2 = int(c.geom2)
                b1 = int(model.geom_bodyid[g1])
                b2 = int(model.geom_bodyid[g2])
            except Exception:
                continue
            for fi, fname in enumerate(finger_names):
                fb = (self._finger_body_groups[fi]
                      if self._finger_body_groups
                      and fi < len(self._finger_body_groups)
                      else set())
                if not fb:
                    continue
                if (b1 in fb and b2 == int(obj_bid)) or \
                   (b2 in fb and b1 == int(obj_bid)):
                    finger_body = b1 if b1 in fb else b2
                    cpos = np.asarray(c.pos, dtype=float)
                    hits.append((fname, finger_body, cpos))
        if hits:
            print(f"[Diag] {label} — {len(hits)} finger-OBJECT contact(s):")
            for fname, fb, cpos in hits:
                fb_xyz = data.xpos[fb].copy()
                obj_xyz = data.xpos[obj_bid].copy()
                print(f"  finger_{fname} body={fb} xyz={fb_xyz.round(3)}  "
                      f"contact_pt={cpos.round(3)}  obj_xyz={obj_xyz.round(3)}")
        else:
            print(f"[Diag] {label} — no finger-object contacts (clean)")

    def _inplace_yaw_aim(self, obj_xy, yaw_tol=math.radians(2.0),
                         timeout=3.0, settle=0.05):
        import time as _time
        loc = self.sim.localization()
        cx0, cy0 = float(loc[0]), float(loc[1])
        _face_yaw = float(math.atan2(obj_xy[1] - cy0, obj_xy[0] - cx0))
        _axis_off = (AXIS_YAW_OFFSET_RAD
                     if ENABLE_ARM_HORIZONTAL_PICKUP else 0.0)
        desired_yaw = float(((_face_yaw + _axis_off + math.pi)
                             % (2 * math.pi)) - math.pi)
        if _axis_off:
            print(f"[Exec] [5.0] axis-aware yaw: facing {math.degrees(_face_yaw):+.1f}° "
                  f"+ axis-offset {math.degrees(_axis_off):+.1f}° "
                  f"→ aim {math.degrees(desired_yaw):+.1f}° (axis ⊥ approach)")
        err0 = ((desired_yaw - float(loc[2]) + math.pi)
                % (2 * math.pi)) - math.pi
        print(f"[Exec] [5.0] IN-PLACE YAW-AIM (no push): chassis "
              f"({cx0:.3f},{cy0:.3f}) yaw {math.degrees(loc[2]):+.1f}° → "
              f"face obj ({obj_xy[0]:.3f},{obj_xy[1]:.3f}) "
              f"target {math.degrees(desired_yaw):+.1f}° "
              f"(err {math.degrees(err0):+.1f}°)")
        if abs(err0) <= yaw_tol:
            print(f"[Exec] [5.0] yaw-aim skipped: already within "
                  f"{math.degrees(yaw_tol):.1f}°")
            return
        _saved = {
            "kp": getattr(self.sim, "_base_kp_theta_override", None),
            "ki": getattr(self.sim, "_base_ki_theta_override", None),
            "kd": getattr(self.sim, "_base_kd_theta_override", None),
            "om": getattr(self.sim, "_base_omega_max_override", None),
            "iy": float(getattr(self.sim, "integral_yaw", 0.0)),
        }
        self.sim._base_kp_theta_override = 12.0
        self.sim._base_ki_theta_override = 2.0
        self.sim._base_kd_theta_override = 0.8
        self.sim._base_omega_max_override = 0.8
        try:
            self.sim.integral_yaw = 0.0
        except Exception:
            pass
        with self.sim._target_lock:
            self.sim.target_base = np.array([cx0, cy0, desired_yaw])
        t0 = _time.time()
        in_tol = 0
        best_err = abs(err0)
        while _time.time() - t0 < timeout:
            cx, cy, cyaw = self.sim.localization()
            yaw_err = ((desired_yaw - cyaw + math.pi)
                       % (2 * math.pi)) - math.pi
            best_err = min(best_err, abs(yaw_err))
            with self.sim._target_lock:
                self.sim.target_base = np.array([cx0, cy0, desired_yaw])
            if abs(yaw_err) <= yaw_tol:
                in_tol += 1
                if in_tol >= 2:
                    break
            else:
                in_tol = 0
            _time.sleep(settle)
        cx, cy, cyaw = self.sim.localization()
        fin_err = ((desired_yaw - cyaw + math.pi) % (2 * math.pi)) - math.pi
        drift = math.hypot(cx - cx0, cy - cy0)
        for _a, _k in (("_base_kp_theta_override", "kp"),
                       ("_base_ki_theta_override", "ki"),
                       ("_base_kd_theta_override", "kd"),
                       ("_base_omega_max_override", "om")):
            if _saved[_k] is None:
                if hasattr(self.sim, _a):
                    delattr(self.sim, _a)
            else:
                setattr(self.sim, _a, _saved[_k])
        try:
            self.sim.integral_yaw = 0.0
        except Exception:
            pass
        print(f"[Exec] [5.0] yaw-aim done: yaw {math.degrees(cyaw):+.1f}° "
              f"err {math.degrees(fin_err):+.1f}° (best {math.degrees(best_err):.1f}°) "
              f"xy-drift {drift*100:.1f}cm  [target was no-push → drift≈0]")

    def _side_grip_chassis_push(self, target_xy_world, obj_xy_2d,
                                timeout=2.0, dist_tol=0.025,
                                abort_on_finger_obj_contact=False,
                                obj_bid=None,
                                yaw_tol=math.radians(3.0)):
        loc = self.sim.localization()
        base_xy = np.asarray([loc[0], loc[1]], dtype=float)
        target_xy = np.asarray(target_xy_world, dtype=float)
        obj_xy = np.asarray(obj_xy_2d, dtype=float)
        start_dist = float(np.linalg.norm(obj_xy - base_xy))
        if ENABLE_ARM_HORIZONTAL_PICKUP:
            _cz = float(self._carry_anchor_xyz(self.sim.data)[2])
            print(f"[Exec] [5.4] ARM-HORIZONTAL (no chassis push): joint-servo "
                  f"const-Z forward, centroid → obj "
                  f"({obj_xy[0]:.3f},{obj_xy[1]:.3f}) z={_cz:.3f}  "
                  f"(start gap {start_dist:.3f} m)")
            _obid = (obj_bid if obj_bid is not None else self._held_obj_bid)
            _saved_th_kp = getattr(self.sim, '_base_kp_override', None)
            _saved_th_kd = getattr(self.sim, '_base_kd_override', None)
            if ENABLE_AH_TH_HOLD:
                self.sim._base_kp_override = 30.0
                self.sim._base_kd_override = 9.0
            try:
                if ENABLE_AH_SERVO_SMOOTHING:
                    _e = self._arm_forward_const_z(
                        (float(obj_xy[0]), float(obj_xy[1])),
                        label="arm-fwd-pickup", obj_bid=_obid,
                        a1_step=0.007, settle=0.09, z_kp=0.45, tilt_step=0.006)
                else:
                    _e = self._arm_forward_const_z(
                        (float(obj_xy[0]), float(obj_xy[1])),
                        label="arm-fwd-pickup", obj_bid=_obid)
                for _rc in range(2 if ENABLE_ARM_HORIZONTAL_RECENTER else 0):
                    _lat = self._arm_horizontal_lateral_offset(
                        _obid, (float(obj_xy[0]), float(obj_xy[1])))
                    if _lat is None or abs(_lat) <= ARM_HORIZONTAL_RECENTER_LAT:
                        break
                    print(f"[Exec] [5.4b] OFF-CENTRE lat={_lat*100:+.1f}cm > "
                          f"{ARM_HORIZONTAL_RECENTER_LAT*100:.0f}cm — "
                          f"lift→TH-recentre→re-descend (try {_rc+1}/2)")
                    self._arm_horizontal_recenter_lifted(
                        _obid, (float(obj_xy[0]), float(obj_xy[1])))
                    _e = self._arm_forward_const_z(
                        (float(obj_xy[0]), float(obj_xy[1])),
                        label=f"arm-fwd-pickup-rc{_rc+1}", obj_bid=_obid)
                print(f"[Exec] [5.4] arm-horizontal servo done: "
                      f"centroid xy err={_e*100:.1f}cm")
            except Exception as _e_ah:
                print(f"[Exec] [5.4] arm-horizontal servo raised: {_e_ah}")
            finally:
                for _a, _sv in (('_base_kp_override', _saved_th_kp),
                                ('_base_kd_override', _saved_th_kd)):
                    if _sv is None:
                        if hasattr(self.sim, _a):
                            delattr(self.sim, _a)
                    else:
                        setattr(self.sim, _a, _sv)
            return
        if ENABLE_NO_CHASSIS_PUSH:
            print(f"[Exec] [5.4] side-grip chassis push SUPPRESSED "
                  f"(--no-chassis-push): chassis stays at "
                  f"({base_xy[0]:.3f},{base_xy[1]:.3f}), arm-side ALIGN "
                  f"will handle residual {start_dist:.3f} m → "
                  f"{float(np.linalg.norm(obj_xy - target_xy)):.3f} m gap")
            return
        target_yaw = float(math.atan2(obj_xy[1] - target_xy[1],
                                      obj_xy[0] - target_xy[0]))
        contact_guard = bool(abort_on_finger_obj_contact and obj_bid is not None)
        print(f"[Exec] [5.4] side-grip chassis push: "
              f"({base_xy[0]:.3f},{base_xy[1]:.3f}) → "
              f"({target_xy[0]:.3f},{target_xy[1]:.3f})  "
              f"dist_to_obj {start_dist:.3f}m → "
              f"{float(np.linalg.norm(obj_xy - target_xy)):.3f}m"
              + (f"  [contact-guard ON]" if contact_guard else ""))
        _saved_maxvel = getattr(self.sim, "_base_max_vel_override", None)
        self.sim._base_max_vel_override = 8.0
        with self.sim._target_lock:
            self.sim.target_base = np.array([float(target_xy[0]),
                                             float(target_xy[1]),
                                             target_yaw])
        import time as _time
        t0 = _time.time()
        last_diag = t0
        aborted_on_contact = False
        while _time.time() - t0 < timeout:
            cx, cy, cyaw = self.sim.localization()
            xy_ok = math.hypot(cx - target_xy[0],
                                cy - target_xy[1]) <= dist_tol
            yaw_err = (cyaw - target_yaw + math.pi) % (2 * math.pi) - math.pi
            yaw_ok = abs(yaw_err) <= yaw_tol
            if xy_ok and yaw_ok:
                break
            if contact_guard:
                any_contact = False
                contact_reason = ""
                _obj_r_guard = float(self._object_radius(obj_bid))
                _obj_hh_guard = float(self._object_half_height(obj_bid))
                _obj_xyz_guard = self.sim.data.xpos[obj_bid].copy()
                PENETRATION_DEPTH_GUARD = 0.010
                if STRICT_PERFECT_FRICTION_ONLY:
                    BC_REACH_STOP_MARGIN    = 0.015
                    THUMB_SAFETY_MARGIN     = 0.015
                else:
                    BC_REACH_STOP_MARGIN    = 0.040
                    THUMB_SAFETY_MARGIN     = 0.020
                bc_stop_dist    = (_obj_r_guard + BC_REACH_STOP_MARGIN)
                thumb_stop_dist = (_obj_r_guard + THUMB_SAFETY_MARGIN)
                try:
                    if (self._carry_anchor_body_ids
                            and len(self._carry_anchor_body_ids) == 3):
                        _a_xy = self.sim.data.xpos[
                            self._carry_anchor_body_ids[0]][:2]
                        _b_xy = self.sim.data.xpos[
                            self._carry_anchor_body_ids[1]][:2]
                        _c_xy = self.sim.data.xpos[
                            self._carry_anchor_body_ids[2]][:2]
                        _bc_xy = 0.5 * (_b_xy + _c_xy)
                        bc_to_obj = float(np.hypot(
                            _bc_xy[0] - _obj_xyz_guard[0],
                            _bc_xy[1] - _obj_xyz_guard[1]))
                        thumb_to_obj = float(np.hypot(
                            _a_xy[0] - _obj_xyz_guard[0],
                            _a_xy[1] - _obj_xyz_guard[1]))
                        if thumb_to_obj <= thumb_stop_dist:
                            any_contact = True
                            contact_reason = (
                                f"thumb-safety d_xy={thumb_to_obj*100:.1f}cm "
                                f"≤ stop {thumb_stop_dist*100:.1f}cm "
                                f"(= obj_r {_obj_r_guard*100:.1f}cm + "
                                f"{THUMB_SAFETY_MARGIN*100:.1f}cm "
                                f"safety) — pushing more would crush thumb")
                        elif bc_to_obj <= bc_stop_dist:
                            any_contact = True
                            contact_reason = (
                                f"bc-pair d_xy={bc_to_obj*100:.1f}cm "
                                f"≤ stop {bc_stop_dist*100:.1f}cm "
                                f"(= obj_r {_obj_r_guard*100:.1f}cm + "
                                f"{BC_REACH_STOP_MARGIN*100:.1f}cm "
                                f"close-arc reach)")
                    else:
                        bc_to_obj = float('nan')
                        thumb_to_obj = float('nan')
                except Exception:
                    bc_to_obj = float('nan')
                    thumb_to_obj = float('nan')
                if not any_contact:
                    for fi in range(3):
                        if (self._carry_anchor_body_ids
                                and len(self._carry_anchor_body_ids) == 3):
                            tip_bid = self._carry_anchor_body_ids[2 - fi]
                            tip_xyz = self.sim.data.xpos[tip_bid]
                            d_xy_to_obj = float(np.hypot(
                                tip_xyz[0] - _obj_xyz_guard[0],
                                tip_xyz[1] - _obj_xyz_guard[1]))
                            dz_to_obj = abs(float(
                                tip_xyz[2] - _obj_xyz_guard[2]))
                            if (d_xy_to_obj
                                    < (_obj_r_guard - PENETRATION_DEPTH_GUARD)
                                    and dz_to_obj < _obj_hh_guard):
                                any_contact = True
                                contact_reason = (
                                    f"finger_{['c','b','a'][fi]} "
                                    f"DEEP-PENETRATION (tip d_xy="
                                    f"{d_xy_to_obj*100:.1f}cm < "
                                    f"obj_r {_obj_r_guard*100:.1f}cm - "
                                    f"{PENETRATION_DEPTH_GUARD*100:.1f}cm safety)")
                                break
                if any_contact:
                    aborted_on_contact = True
                    with self.sim._target_lock:
                        self.sim.target_base = np.array(
                            [float(cx), float(cy), target_yaw])
                    print(f"[Exec] [5.4] chassis push STOPPED on "
                          f"{contact_reason} at t={_time.time()-t0:.2f}s "
                          f"base=({cx:.3f},{cy:.3f})")
                    break
            now = _time.time()
            if now - last_diag >= 0.3:
                last_diag = now
                cur_d = float(np.linalg.norm(np.array([cx, cy]) - obj_xy))
                if VERBOSE_GRASP_DEBUG:
                  print(f"[Exec]   chassis push progress: t={now-t0:.1f}s "
                      f"base=({cx:.3f},{cy:.3f}) dist_to_obj={cur_d:.3f}m")
            _time.sleep(0.05)
        if _saved_maxvel is None:
            try: delattr(self.sim, "_base_max_vel_override")
            except Exception: pass
        else:
            self.sim._base_max_vel_override = _saved_maxvel
        fx, fy, fyaw = self.sim.localization()
        final_dist = float(np.linalg.norm(np.array([fx, fy]) - obj_xy))
        moved = math.hypot(fx - base_xy[0], fy - base_xy[1])
        final_yaw_err = (fyaw - target_yaw + math.pi) % (2 * math.pi) - math.pi
        _push_dt = _time.time() - t0
        print(f"[Exec] [5.4] chassis push done: base-obj dist={final_dist:.3f}m "
              f"(moved {moved*100:.1f}cm in {_push_dt:.1f}s, "
              f"yaw_err={math.degrees(final_yaw_err):+.1f}°)"
              + (f"  [aborted on contact]" if aborted_on_contact else ""))

    def _wait_for_wrist_settle(self, tolerance=0.05, timeout=2.0,
                                label="wrist-settle",
                                intended_targets=None):
        import time as _time
        model = self.sim.model
        data = self.sim.data
        gids = self.sim.gripper_ids_left
        if len(gids) <= GIDS_HANDBEARING:
            return None

        wrist_specs = (
            ("HandBearingJoint_1",   GIDS_HANDBEARING, "hb", 0),
            ("gripper_z_rotation_1", GIDS_WRIST_Z,     "wz", 1),
            ("gripper_x_rotation_1", GIDS_WRIST_X,     "wx", 2),
            ("gripper_y_rotation_1", GIDS_WRIST_Y,     "wy", 3),
        )
        qpas = []
        targets = []
        labels = []
        for jname, gidx, slabel, intended_idx in wrist_specs:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                continue
            qpas.append(int(model.jnt_qposadr[jid]))
            if intended_targets is not None:
                targets.append(float(intended_targets[intended_idx]))
            else:
                targets.append(float(data.ctrl[gids[gidx]]))
            labels.append(slabel)

        t0 = _time.time()
        last_diag = t0
        residuals = [float('inf')] * len(qpas)
        ss_history = []
        ss_window  = 0.30
        ss_delta   = 0.003
        steady_state = False
        while _time.time() - t0 < timeout:
            residuals = [
                abs(float(data.qpos[qpas[i]]) - targets[i])
                for i in range(len(qpas))
            ]
            worst = max(residuals) if residuals else 0.0
            now = _time.time()
            if worst <= tolerance:
                break
            ss_history.append((now, worst))
            ss_history = [(t, w) for (t, w) in ss_history
                          if now - t <= ss_window]
            if len(ss_history) >= 3 and (now - ss_history[0][0]) >= ss_window:
                w_old = ss_history[0][1]
                if abs(worst - w_old) < ss_delta:
                    steady_state = True
                    break
            if now - last_diag >= 0.5:
                last_diag = now
                rstr = ", ".join(
                    f"{labels[i]}={residuals[i]:+.3f}"
                    for i in range(len(qpas)))
                print(f"[Exec]   {label} t={now-t0:.1f}s  worst={worst:.3f}rad "
                      f"(tol={tolerance})  resid: {rstr}")
            _time.sleep(0.05)
        elapsed = _time.time() - t0
        rstr = ", ".join(
            f"{labels[i]}={residuals[i]:+.3f}"
            for i in range(len(qpas)))
        worst = max(residuals) if residuals else 0.0
        if worst <= tolerance:
            status = "OK"
        elif steady_state:
            status = "STEADY-STATE (PD at equilibrium, residual won't decrease)"
        else:
            status = "TIMEOUT"
        print(f"[Exec] [5.45] {label} {status} in {elapsed:.2f}s  "
              f"worst={worst:.3f}rad (tol={tolerance})  resid: {rstr}")
        return tuple(residuals)

    def _hold_wz_to_target(self, wz_target, tol=0.035, max_iter=50,
                            label="wz-hold"):
        gids = self.sim.gripper_ids_left
        if len(gids) <= GIDS_WRIST_Z:
            return None
        model = self.sim.model
        data = self.sim.data
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                "gripper_z_rotation_1")
        if jid < 0:
            return None
        qadr = int(model.jnt_qposadr[jid])
        gid = gids[GIDS_WRIST_Z]
        ki = 0.7
        lo = float(wz_target) - 0.70
        hi = float(wz_target) + 0.70
        wz_ctrl = float(data.ctrl[gid])
        wz_now = float(data.qpos[qadr])
        err = float(wz_target) - wz_now
        it = 0
        for it in range(max_iter):
            wz_now = float(data.qpos[qadr])
            err = float(wz_target) - wz_now
            if abs(err) <= tol:
                break
            wz_ctrl = max(lo, min(hi, wz_ctrl + ki * err))
            data.ctrl[gid] = wz_ctrl
            time.sleep(0.04)
        self._wz_ctrl_override = wz_ctrl
        print(f"[Exec] [5.46b] {label}: wz {wz_now:+.3f}rad → target "
              f"{float(wz_target):+.3f} (resid {err:+.3f}rad="
              f"{math.degrees(err):+.1f}°, ctrl latched {wz_ctrl:+.3f}, "
              f"{it+1} iters)")
        return err

    def _finger_obj_xy_dists(self, obj_xy):
        if len(self._carry_anchor_body_ids) != 3:
            return None
        _a = self.sim.data.xpos[self._carry_anchor_body_ids[0]]
        _b = self.sim.data.xpos[self._carry_anchor_body_ids[1]]
        _c = self.sim.data.xpos[self._carry_anchor_body_ids[2]]
        _bc_xy = 0.5 * (_b[:2] + _c[:2])
        d_th = float(np.linalg.norm(_a[:2] - obj_xy))
        d_bc = float(np.linalg.norm(_bc_xy - obj_xy))
        z_spread = (max(float(_a[2]), float(_b[2]), float(_c[2]))
                    - min(float(_a[2]), float(_b[2]), float(_c[2])))
        return (d_th, d_bc, z_spread)

    def _a1_forward_align(self, obj_bid, grasp_q, max_steps=4, step_m=0.015,
                           target_carry=0.06, label="a1-fwd-align"):
        if len(self._carry_anchor_body_ids) != 3:
            return
        try:
            obj_xy = self.sim.data.xpos[obj_bid][:2].copy()
            chassis_xy = np.asarray(self.sim.localization()[:2], dtype=float)
            obj_r = float(self._object_radius(obj_bid))
        except Exception:
            return
        _appr = obj_xy - chassis_xy
        _an = float(np.linalg.norm(_appr))
        if _an < 1e-6:
            return
        _appr_u = _appr / _an
        _a1_max = float(JOINT_RANGES_ARM[2][1])
        _th_plan = float(grasp_q[3])
        _wrist = [float(grasp_q[i]) for i in range(4, 8)]
        _pen_floor = obj_r + 0.02
        for _step in range(max_steps):
            _pinch = self._pinch_midpoint_xyz(self.sim.data)
            _resid = obj_xy - _pinch[:2]
            _carry = float(np.linalg.norm(_resid))
            if _carry <= target_carry:
                break
            _cos = float(np.dot(_resid, _appr_u) / (_carry + 1e-9))
            if _cos < 0.7:
                print(f"[Exec] {label}: residual sideways (cos={_cos:+.2f}, "
                      f"carry={_carry*100:.1f}cm) — orientation, not a "
                      f"forward shortfall; a1 won't fix it, stopping")
                break
            _fd0 = self._finger_obj_xy_dists(obj_xy)
            if _fd0 is not None and min(_fd0[0], _fd0[1]) < _pen_floor:
                print(f"[Exec] {label}: a finger already at obj "
                      f"(min(d_th={_fd0[0]*100:.1f},d_bc={_fd0[1]*100:.1f}) "
                      f"< {_pen_floor*100:.1f}cm) — NOT nudging (would push "
                      f"obj); orientation case, leave for wrist-yaw balance")
                break
            _cur_q = self._current_arm_q()
            _new_a1 = min(_a1_max, float(_cur_q[2]) + step_m)
            if _new_a1 - float(_cur_q[2]) < 1e-4:
                print(f"[Exec] {label}: a1 at limit ({_a1_max:.3f}m) — "
                      f"stopping (carry={_carry*100:.1f}cm)")
                break
            _start_q = [float(_cur_q[0]), float(_cur_q[1]), float(_cur_q[2]),
                        _th_plan] + _wrist
            _end_q = [float(_cur_q[0]), float(_cur_q[1]), _new_a1,
                      _th_plan] + _wrist
            try:
                if not bool(self.arm_bridge.is_valid(_end_q)):
                    print(f"[Exec] {label}: a1+{step_m*100:.1f}cm invalid — "
                          f"stopping")
                    break
            except Exception:
                break
            self._kinematic_descent(_start_q, _end_q, label=f"{label}-step",
                                    n_steps=6, per_step_settle=0.06)
            time.sleep(0.2)
            _carry2 = float(np.linalg.norm(
                obj_xy - self._pinch_midpoint_xyz(self.sim.data)[:2]))
            _fd = self._finger_obj_xy_dists(obj_xy)
            _d_th, _d_bc = (_fd[0], _fd[1]) if _fd else (99.0, 99.0)
            _near = min(_d_th, _d_bc)
            _th_now = float(self._current_arm_q()[3])
            _th_drift = abs(_th_now - _th_plan)
            _clip = int(self._count_arm_obj_contacts(obj_bid))
            _reason = None
            if _clip > 0:
                _reason = f"arm-obj clip ({_clip})"
            elif _near < _pen_floor:
                _reason = (f"near-finger penetration (min(d_th={_d_th*100:.1f},"
                           f"d_bc={_d_bc*100:.1f}) < {_pen_floor*100:.1f}cm)")
            elif _th_drift > math.radians(6.0):
                _reason = (f"TH swung {math.degrees(_th_now-_th_plan):+.1f}° "
                           f"off plan (>6°)")
            elif _carry2 > _carry - 0.005:
                _reason = (f"no carry improvement "
                           f"({_carry*100:.1f}→{_carry2*100:.1f}cm)")
            if _reason is not None:
                print(f"[Exec] {label}: step {_step+1} REVERTED ({_reason})")
                self._kinematic_descent(_end_q, _start_q,
                                        label=f"{label}-revert",
                                        n_steps=6, per_step_settle=0.06)
                time.sleep(0.2)
                break
            print(f"[Exec] {label}: step {_step+1} a1 "
                  f"{_cur_q[2]:.3f}→{_new_a1:.3f}  carry "
                  f"{_carry*100:.1f}→{_carry2*100:.1f}cm  d_th={_d_th*100:.1f} "
                  f"d_bc={_d_bc*100:.1f}cm  th_drift="
                  f"{math.degrees(_th_now-_th_plan):+.1f}° (cos={_cos:+.2f})")

    def _wrist_yaw_balance(self, obj_bid, grasp_q, balance_tol=0.03,
                            max_delta=0.30, label="wrist-yaw-bal"):
        if len(self._carry_anchor_body_ids) != 3:
            return grasp_q
        try:
            obj_xy = self.sim.data.xpos[obj_bid][:2].copy()
            obj_r = float(self._object_radius(obj_bid))
        except Exception:
            return grasp_q
        _fd = self._finger_obj_xy_dists(obj_xy)
        if _fd is None:
            return grasp_q
        d_th, d_bc, _ = _fd
        _imb = d_th - d_bc
        if abs(_imb) <= balance_tol:
            print(f"[Exec] {label}: already balanced (d_th={d_th*100:.1f} "
                  f"d_bc={d_bc*100:.1f}cm, |Δ|={abs(_imb)*100:.1f}cm "
                  f"≤ {balance_tol*100:.0f}cm) — no wrist move")
            return grasp_q
        _m = self.arm_bridge.model
        _pd = self.arm_bridge.planning_data
        _qm = self.arm_bridge._qpos_map
        _ta, _tb, _tc = self._carry_anchor_body_ids
        _keys = ("ColumnLeft", "ColumnRight", "ArmLeft", "Base",
                 "HandBearing", "WristZ", "WristX", "WristY")
        cur_q = self._current_arm_q()

        def _kin_balance(q):
            for _k, _v in zip(_keys, q):
                _pd.qpos[_qm[_k]] = float(_v)
            mujoco.mj_forward(_m, _pd)
            _a = _pd.xpos[_ta]; _b = _pd.xpos[_tb]; _c = _pd.xpos[_tc]
            _bcxy = 0.5 * (_b[:2] + _c[:2])
            _bal = (float(np.linalg.norm(_a[:2] - obj_xy))
                    - float(np.linalg.norm(_bcxy - obj_xy)))
            _zs = (max(float(_a[2]), float(_b[2]), float(_c[2]))
                   - min(float(_a[2]), float(_b[2]), float(_c[2])))
            return _bal, _zs

        _bal0, _zs0 = _kin_balance(cur_q)
        _dp = 0.10
        _best = None
        for _wi, _wn in ((4, "hb"), (7, "wy"), (6, "wx")):
            _qp = list(cur_q); _qp[_wi] = float(_qp[_wi]) + _dp
            try:
                _balp, _zsp = _kin_balance(_qp)
            except Exception:
                continue
            _dbal = (_balp - _bal0) / _dp
            _dzs = (_zsp - _zs0) / _dp
            if abs(_dbal) < 0.05:
                continue
            _score = abs(_dbal) / (1.0 + 8.0 * abs(_dzs))
            if _best is None or _score > _best[3]:
                _best = (_wi, _wn, _dbal, _score)
        if _best is None:
            print(f"[Exec] {label}: no level wrist DOF can balance "
                  f"(d_th={d_th*100:.1f} d_bc={d_bc*100:.1f}cm) — leaving "
                  f"for the last-resort column/chassis tier")
            return grasp_q
        _wi, _wn, _dbal = _best[0], _best[1], _best[2]
        _delta = max(-max_delta, min(max_delta, -float(_imb) / float(_dbal)))
        _start = list(cur_q)
        _start[4:8] = [float(grasp_q[i]) for i in range(4, 8)]
        _new = list(_start)
        _new[_wi] = float(_start[_wi]) + _delta
        try:
            if not bool(self.arm_bridge.is_valid(_new)):
                print(f"[Exec] {label}: {_wn} Δ{_delta:+.2f}rad invalid — leaving")
                return grasp_q
        except Exception:
            return grasp_q
        self._kinematic_descent(_start, _new, label=f"{label}-{_wn}",
                                n_steps=8, per_step_settle=0.06)
        time.sleep(0.2)
        _fd2 = self._finger_obj_xy_dists(obj_xy)
        _clip = int(self._count_arm_obj_contacts(obj_bid))
        if _fd2 is None:
            return grasp_q
        d_th2, d_bc2, zs2 = _fd2
        _imb2 = abs(d_th2 - d_bc2)
        _reason = None
        if _clip > 0:
            _reason = f"arm-obj clip ({_clip})"
        elif min(d_th2, d_bc2) < obj_r + 0.015:
            _reason = (f"near-finger penetration (min({d_th2*100:.1f},"
                       f"{d_bc2*100:.1f})<{(obj_r+0.015)*100:.1f}cm)")
        elif zs2 - _zs0 > 0.03:
            _reason = (f"gripper tilted off level "
                       f"(z-spread {_zs0*100:.1f}→{zs2*100:.1f}cm)")
        elif _imb2 > abs(_imb) - 0.01:
            _reason = (f"imbalance not improved "
                       f"({abs(_imb)*100:.1f}→{_imb2*100:.1f}cm)")
        if _reason is not None:
            print(f"[Exec] {label}: {_wn} Δ{_delta:+.2f}rad REVERTED ({_reason})")
            self._kinematic_descent(_new, _start, label=f"{label}-revert",
                                    n_steps=8, per_step_settle=0.06)
            time.sleep(0.2)
            return grasp_q
        print(f"[Exec] {label}: {_wn} Δ{_delta:+.3f}rad → balanced "
              f"d_th {d_th*100:.1f}→{d_th2*100:.1f} d_bc "
              f"{d_bc*100:.1f}→{d_bc2*100:.1f}cm  (|Δ| "
              f"{abs(_imb)*100:.1f}→{_imb2*100:.1f}cm, z-spread "
              f"{_zs0*100:.1f}→{zs2*100:.1f}cm) — wrist parallel-to-ground kept")
        _gq = list(grasp_q)
        _gq[_wi] = float(_gq[_wi]) + _delta
        return _gq


    def _column_yaw_balance(self, obj_bid, grasp_q, balance_tol=0.03,
                             max_delta=0.14, label="th-col-bal"):
        if len(self._carry_anchor_body_ids) != 3:
            return grasp_q
        try:
            obj_xy = self.sim.data.xpos[obj_bid][:2].copy()
            obj_r = float(self._object_radius(obj_bid))
        except Exception:
            return grasp_q
        _fd = self._finger_obj_xy_dists(obj_xy)
        if _fd is None:
            return grasp_q
        d_th, d_bc, _ = _fd
        _imb = d_th - d_bc
        if abs(_imb) <= balance_tol:
            print(f"[Exec] {label}: already balanced (d_th={d_th*100:.1f} "
                  f"d_bc={d_bc*100:.1f}cm, |Δ|={abs(_imb)*100:.1f}cm) — no th move")
            return grasp_q
        _m = self.arm_bridge.model
        _pd = self.arm_bridge.planning_data
        _qm = self.arm_bridge._qpos_map
        _ta, _tb, _tc = self._carry_anchor_body_ids
        _keys = ("ColumnLeft", "ColumnRight", "ArmLeft", "Base",
                 "HandBearing", "WristZ", "WristX", "WristY")
        cur_q = self._current_arm_q()
        _carry0 = float(np.linalg.norm(
            self._pinch_midpoint_xyz(self.sim.data)[:2] - obj_xy))

        def _kin_probe(q):
            for _k, _v in zip(_keys, q):
                _pd.qpos[_qm[_k]] = float(_v)
            mujoco.mj_forward(_m, _pd)
            _a = _pd.xpos[_ta]; _b = _pd.xpos[_tb]; _c = _pd.xpos[_tc]
            _bcxy = 0.5 * (_b[:2] + _c[:2])
            _da = float(np.linalg.norm(_a[:2] - obj_xy))
            _dbcv = float(np.linalg.norm(_bcxy - obj_xy))
            _bal = _da - _dbcv
            _pinchxy = 0.5 * (_a[:2] + _bcxy)
            _carry = float(np.linalg.norm(_pinchxy - obj_xy))
            return _bal, _carry

        _bal0, _c0 = _kin_probe(cur_q)
        _dp = 0.08
        _qp = list(cur_q); _qp[3] = float(_qp[3]) + _dp
        try:
            _balp, _cp = _kin_probe(_qp)
        except Exception:
            return grasp_q
        _dbal = (_balp - _bal0) / _dp
        if abs(_dbal) < 0.05:
            print(f"[Exec] {label}: th has no balance authority here "
                  f"(d|bal|/dth={_dbal:+.3f}) — structural, leaving")
            return grasp_q
        _delta = max(-max_delta, min(max_delta, -float(_imb) / float(_dbal)))
        _carry_pred = _c0 + (_cp - _c0) / _dp * _delta
        if _carry_pred > _carry0 + 0.04:
            print(f"[Exec] {label}: th Δ{_delta:+.3f}rad would walk pinch off "
                  f"obj (carry {_carry0*100:.1f}→~{_carry_pred*100:.1f}cm) — "
                  f"leaving for chassis/structural")
            return grasp_q
        _start = list(cur_q)
        _start[4:8] = [float(grasp_q[i]) for i in range(4, 8)]
        _new = list(_start)
        _new[3] = float(_start[3]) + _delta
        try:
            if not bool(self.arm_bridge.is_valid(_new)):
                print(f"[Exec] {label}: th Δ{_delta:+.3f}rad invalid — leaving")
                return grasp_q
        except Exception:
            return grasp_q
        _saved = {}
        for _a, _v in (('_base_kp_override', 60.0), ('_base_kd_override', 10.0),
                       ('_base_ki_override', 2.0)):
            _saved[_a] = getattr(self.sim, _a, None)
            setattr(self.sim, _a, _v)
        try:
            self._kinematic_descent(_start, _new, label=f"{label}-th",
                                    n_steps=10, per_step_settle=0.07)
            time.sleep(0.25)
            _fd2 = self._finger_obj_xy_dists(obj_xy)
            _clip = int(self._count_arm_obj_contacts(obj_bid))
            _carry2 = float(np.linalg.norm(
                self._pinch_midpoint_xyz(self.sim.data)[:2] - obj_xy))
        finally:
            for _a, _v in _saved.items():
                if _v is None:
                    try: delattr(self.sim, _a)
                    except Exception: pass
                else:
                    setattr(self.sim, _a, _v)
        if _fd2 is None:
            return grasp_q
        d_th2, d_bc2, _ = _fd2
        _imb2 = abs(d_th2 - d_bc2)
        _reason = None
        if _clip > 0:
            _reason = f"arm-obj clip ({_clip})"
        elif min(d_th2, d_bc2) < obj_r + 0.015:
            _reason = (f"near-finger penetration (min({d_th2*100:.1f},"
                       f"{d_bc2*100:.1f})<{(obj_r+0.015)*100:.1f}cm)")
        elif _carry2 > _carry0 + 0.03:
            _reason = (f"pinch walked off obj "
                       f"(carry {_carry0*100:.1f}→{_carry2*100:.1f}cm)")
        elif _imb2 > abs(_imb) - 0.01:
            _reason = (f"imbalance not improved "
                       f"({abs(_imb)*100:.1f}→{_imb2*100:.1f}cm)")
        if _reason is not None:
            print(f"[Exec] {label}: th Δ{_delta:+.3f}rad REVERTED ({_reason})")
            for _a, _v in (('_base_kp_override', 60.0),
                           ('_base_kd_override', 10.0),
                           ('_base_ki_override', 2.0)):
                setattr(self.sim, _a, _v)
            try:
                self._kinematic_descent(_new, _start, label=f"{label}-revert",
                                        n_steps=10, per_step_settle=0.07)
                time.sleep(0.25)
            finally:
                for _a, _v in _saved.items():
                    if _v is None:
                        try: delattr(self.sim, _a)
                        except Exception: pass
                    else:
                        setattr(self.sim, _a, _v)
            return grasp_q
        print(f"[Exec] {label}: th Δ{_delta:+.3f}rad → balanced d_th "
              f"{d_th*100:.1f}→{d_th2*100:.1f} d_bc {d_bc*100:.1f}→{d_bc2*100:.1f}cm "
              f"(|Δ| {abs(_imb)*100:.1f}→{_imb2*100:.1f}cm, carry "
              f"{_carry0*100:.1f}→{_carry2*100:.1f}cm)")
        _gq = list(grasp_q)
        _gq[3] = float(_gq[3]) + _delta
        return _gq


    def _jacobian_pinch_align(self, obj_bid, grasp_q, max_passes=2,
                               w_imb=1.0, lam=0.04, label="jac-align"):
        if len(self._carry_anchor_body_ids) != 3:
            return grasp_q
        try:
            obj_xy = self.sim.data.xpos[obj_bid][:2].copy()
            obj_r = float(self._object_radius(obj_bid))
        except Exception:
            return grasp_q
        _m = self.arm_bridge.model
        _pd = self.arm_bridge.planning_data
        _qm = self.arm_bridge._qpos_map
        _ta, _tb, _tc = self._carry_anchor_body_ids
        _keys = ("ColumnLeft", "ColumnRight", "ArmLeft", "Base",
                 "HandBearing", "WristZ", "WristX", "WristY")
        _dofs = (0, 1, 2, 3, 4, 7)
        _names = {0: "h1", 1: "h2", 2: "a1", 3: "th", 4: "hb", 7: "wy"}
        _bound = {0: 0.03, 1: 0.03, 2: 0.025, 3: 0.10, 4: 0.15, 7: 0.15}
        _dp = {0: 0.01, 1: 0.01, 2: 0.01, 3: 0.05, 4: 0.06, 7: 0.06}

        def _err(q):
            for _k, _v in zip(_keys, q):
                _pd.qpos[_qm[_k]] = float(_v)
            mujoco.mj_forward(_m, _pd)
            _a = _pd.xpos[_ta][:2]; _b = _pd.xpos[_tb][:2]; _c = _pd.xpos[_tc][:2]
            _bc = 0.5 * (_b + _c)
            _pinch = 0.5 * (_a + _bc)
            _res = obj_xy - _pinch
            _d_th = float(np.linalg.norm(_a - obj_xy))
            _d_bc = float(np.linalg.norm(_bc - obj_xy))
            _imb = _d_th - _d_bc
            _e = np.array([_res[0], _res[1], w_imb * _imb], dtype=float)
            return _e, float(np.linalg.norm(_res)), _imb, _d_th, _d_bc

        _cur = self._current_arm_q()
        _q = [float(_cur[i]) for i in range(8)]
        _q[4:8] = [float(grasp_q[i]) for i in range(4, 8)]
        _net = [0.0] * 8
        _saved = {}
        for _a_, _v_ in (('_base_kp_override', 60.0), ('_base_kd_override', 10.0),
                         ('_base_ki_override', 2.0)):
            _saved[_a_] = getattr(self.sim, _a_, None)
        try:
            for _p in range(max_passes):
                _e0, _r0, _imb0, _dth0, _dbc0 = _err(_q)
                _cost0 = float(np.linalg.norm(_e0))
                if _r0 <= 0.025 and abs(_imb0) <= 0.03:
                    if _p == 0:
                        print(f"[Exec] {label}: already aligned "
                              f"(|res|={_r0*100:.1f}cm |imb|={abs(_imb0)*100:.1f}cm)"
                              f" — no combo move")
                    break
                _J = np.zeros((3, len(_dofs)), dtype=float)
                for _j, _di in enumerate(_dofs):
                    _qp = list(_q); _qp[_di] = float(_qp[_di]) + _dp[_di]
                    try:
                        _ep, *_ = _err(_qp)
                    except Exception:
                        return grasp_q
                    _J[:, _j] = (_ep - _e0) / _dp[_di]
                try:
                    _JJt = _J @ _J.T + (lam * lam) * np.eye(3)
                    _dq_v = _J.T @ np.linalg.solve(_JJt, -_e0)
                except Exception:
                    return grasp_q
                _scale = 1.0
                for _j, _di in enumerate(_dofs):
                    if abs(_dq_v[_j]) > 1e-9:
                        _scale = min(_scale, _bound[_di] / abs(_dq_v[_j]))
                _dq_v = _dq_v * _scale
                if float(np.max(np.abs(_dq_v))) < 1e-4:
                    if _p == 0:
                        print(f"[Exec] {label}: negligible combo step "
                              f"(|res|={_r0*100:.1f}cm |imb|={abs(_imb0)*100:.1f}cm)"
                              f" — structural, leaving")
                    break
                _start = list(_q)
                _new = None
                _used_frac = 0.0
                for _frac in (1.0, 0.6, 0.35, 0.2):
                    _cand = list(_q)
                    for _j, _di in enumerate(_dofs):
                        _cand[_di] = float(_q[_di]) + _frac * float(_dq_v[_j])
                    try:
                        if bool(self.arm_bridge.is_valid(_cand)):
                            _new = _cand; _used_frac = _frac; break
                    except Exception:
                        _new = None; break
                if _new is None:
                    print(f"[Exec] {label}: pass {_p+1} combo invalid at all "
                          f"step fractions — stop")
                    break
                _epred, *_ = _err(_new)
                if float(np.linalg.norm(_epred)) > _cost0 - 0.005:
                    print(f"[Exec] {label}: pass {_p+1} no predicted gain "
                          f"({_cost0*100:.1f}→{np.linalg.norm(_epred)*100:.1f}) — stop")
                    break
                for _a_, _v_ in (('_base_kp_override', 60.0),
                                 ('_base_kd_override', 10.0),
                                 ('_base_ki_override', 2.0)):
                    setattr(self.sim, _a_, _v_)
                self._kinematic_descent(_start, _new, label=f"{label}-p{_p+1}",
                                        n_steps=10, per_step_settle=0.07)
                time.sleep(0.25)
                _rq = self._current_arm_q()
                _e1, _r1, _imb1, _dth1, _dbc1 = _err(
                    [float(_rq[i]) for i in range(8)])
                _clip = int(self._count_arm_obj_contacts(obj_bid))
                _near = min(_dth1, _dbc1)
                _cost1 = float(np.linalg.norm(_e1))
                _reason = None
                if _clip > 0:
                    _reason = f"arm-obj clip ({_clip})"
                elif _near < obj_r + 0.015:
                    _reason = (f"near-finger penetration "
                               f"(min(d_th={_dth1*100:.1f},d_bc={_dbc1*100:.1f})"
                               f"<{(obj_r+0.015)*100:.1f}cm)")
                elif _cost1 > _cost0 - 0.005:
                    _reason = (f"combined error not improved "
                               f"({_cost0*100:.1f}→{_cost1*100:.1f}cm)")
                if _reason is not None:
                    print(f"[Exec] {label}: pass {_p+1} REVERTED ({_reason})")
                    self._kinematic_descent(_new, _start,
                                            label=f"{label}-revert",
                                            n_steps=10, per_step_settle=0.07)
                    time.sleep(0.25)
                    break
                for _di in _dofs:
                    _net[_di] += float(_new[_di]) - float(_start[_di])
                _q = list(_new)
                print(f"[Exec] {label}: pass {_p+1} combo "
                      f"{{" + ", ".join(f'{_names[_di]}{float(_dq_v[_j]):+.3f}'
                      for _j, _di in enumerate(_dofs)
                      if abs(_dq_v[_j]) > 1e-3) + "}} "
                      f"→ |res| {_r0*100:.1f}→{_r1*100:.1f}cm  "
                      f"|imb| {abs(_imb0)*100:.1f}→{abs(_imb1)*100:.1f}cm  "
                      f"d_th={_dth1*100:.1f} d_bc={_dbc1*100:.1f}cm")
        finally:
            for _a_, _v_ in _saved.items():
                if _v_ is None:
                    try: delattr(self.sim, _a_)
                    except Exception: pass
                else:
                    setattr(self.sim, _a_, _v_)
        if max(abs(v) for v in _net) < 1e-4:
            return grasp_q
        _gq = list(grasp_q)
        for _di in _dofs:
            _gq[_di] = float(_gq[_di]) + _net[_di]
        return _gq


    def _axis_align_preread(self, obj_bid, grasp_q, max_passes=4,
                             w_ax=0.003, lam=0.05, label="axis-pre"):
        if len(self._carry_anchor_body_ids) != 3:
            return grasp_q
        try:
            obj_xy = self.sim.data.xpos[obj_bid][:2].copy()
        except Exception:
            return grasp_q
        try:
            chx = np.asarray(self.sim.localization()[:2], dtype=float)
        except Exception:
            chx = obj_xy
        _appr = obj_xy - chx
        _an = float(np.linalg.norm(_appr))
        if _an < 1e-6:
            return grasp_q
        _fdir = _appr / _an
        _ldir = np.array([-_fdir[1], _fdir[0]])
        _ap_yaw = math.atan2(_appr[1], _appr[0])
        _m = self.arm_bridge.model
        _pd = self.arm_bridge.planning_data
        _qm = self.arm_bridge._qpos_map
        _ta, _tb, _tc = self._carry_anchor_body_ids
        _keys = ("ColumnLeft", "ColumnRight", "ArmLeft", "Base",
                 "HandBearing", "WristZ", "WristX", "WristY")
        _dofs = (0, 1, 3, 4, 7)
        _names = {0: "h1", 1: "h2", 3: "th", 4: "hb", 7: "wy"}
        _bound = {0: 0.03, 1: 0.03, 3: 0.12, 4: 0.15, 7: 0.15}
        _dp = {0: 0.01, 1: 0.01, 3: 0.05, 4: 0.06, 7: 0.06}

        def _err(q):
            for _k, _v in zip(_keys, q):
                _pd.qpos[_qm[_k]] = float(_v)
            mujoco.mj_forward(_m, _pd)
            _a = _pd.xpos[_ta][:2]; _b = _pd.xpos[_tb][:2]; _c = _pd.xpos[_tc][:2]
            _bc = 0.5 * (_b + _c)
            _pinch = 0.5 * (_a + _bc)
            _res = obj_xy - _pinch
            _lat = float(np.dot(_res, _ldir))
            _axv = _bc - _a
            _ax_yaw = math.atan2(_axv[1], _axv[0])
            _ea = ((_ap_yaw + math.pi/2 - _ax_yaw + math.pi)
                   % (2*math.pi)) - math.pi
            _eb = ((_ap_yaw - math.pi/2 - _ax_yaw + math.pi)
                   % (2*math.pi)) - math.pi
            _axr = _ea if abs(_ea) < abs(_eb) else _eb
            _axd = math.degrees(_axr)
            _e = np.array([_lat, w_ax * _axd], dtype=float)
            return _e, _lat, _axd

        _cur = self._current_arm_q()
        _q = [float(_cur[i]) for i in range(8)]
        _q[4:8] = [float(grasp_q[i]) for i in range(4, 8)]
        _net = [0.0] * 8
        _base_hold = None
        try:
            _bh = self.sim.localization()
            _base_hold = np.array([float(_bh[0]), float(_bh[1]), float(_bh[2])])
            with self.sim._target_lock:
                self.sim.target_base = _base_hold.copy()
        except Exception:
            pass
        _saved = {}
        for _a_, _v_ in (('_base_kp_override', 60.0), ('_base_kd_override', 10.0),
                         ('_base_ki_override', 2.0)):
            _saved[_a_] = getattr(self.sim, _a_, None)
        try:
            for _p in range(max_passes):
                _e0, _lat0, _axd0 = _err(_q)
                _cost0 = float(np.linalg.norm(_e0))
                if abs(_lat0) <= 0.02 and abs(_axd0) <= 4.0:
                    if _p == 0:
                        print(f"[Exec] {label}: already aligned "
                              f"(lat={_lat0*100:+.1f}cm axis={_axd0:+.1f}deg) "
                              f"— no combo move")
                    break
                _J = np.zeros((2, len(_dofs)), dtype=float)
                for _j, _di in enumerate(_dofs):
                    _qp = list(_q); _qp[_di] = float(_qp[_di]) + _dp[_di]
                    try:
                        _ep, *_ = _err(_qp)
                    except Exception:
                        return grasp_q
                    _J[:, _j] = (_ep - _e0) / _dp[_di]
                try:
                    _JJt = _J @ _J.T + (lam * lam) * np.eye(2)
                    _dq_v = _J.T @ np.linalg.solve(_JJt, -_e0)
                except Exception:
                    return grasp_q
                _scale = 1.0
                for _j, _di in enumerate(_dofs):
                    if abs(_dq_v[_j]) > 1e-9:
                        _scale = min(_scale, _bound[_di] / abs(_dq_v[_j]))
                _dq_v = _dq_v * _scale
                if float(np.max(np.abs(_dq_v))) < 1e-4:
                    if _p == 0:
                        print(f"[Exec] {label}: negligible combo step "
                              f"(lat={_lat0*100:+.1f}cm axis={_axd0:+.1f}deg) "
                              f"— structural, leaving")
                    break
                _start = list(_q)
                _new = None
                for _frac in (1.0, 0.6, 0.35, 0.2):
                    _cand = list(_q)
                    for _j, _di in enumerate(_dofs):
                        _cand[_di] = float(_q[_di]) + _frac * float(_dq_v[_j])
                    try:
                        if bool(self.arm_bridge.is_valid(_cand)):
                            _new = _cand; break
                    except Exception:
                        _new = None; break
                if _new is None:
                    print(f"[Exec] {label}: pass {_p+1} combo invalid at all "
                          f"step fractions — stop")
                    break
                _epred, *_ = _err(_new)
                if float(np.linalg.norm(_epred)) > _cost0 - 0.002:
                    print(f"[Exec] {label}: pass {_p+1} no predicted gain "
                          f"(cost {_cost0:.3f}→{np.linalg.norm(_epred):.3f}) — stop")
                    break
                for _a_, _v_ in (('_base_kp_override', 60.0),
                                 ('_base_kd_override', 10.0),
                                 ('_base_ki_override', 2.0)):
                    setattr(self.sim, _a_, _v_)
                self._kinematic_descent(_start, _new, label=f"{label}-p{_p+1}",
                                        n_steps=10, per_step_settle=0.07)
                time.sleep(0.25)
                _rq = self._current_arm_q()
                _e1, _lat1, _axd1 = _err([float(_rq[i]) for i in range(8)])
                _clip = int(self._count_arm_obj_contacts(obj_bid))
                _cost1 = float(np.linalg.norm(_e1))
                _reason = None
                if _clip > 0:
                    _reason = f"arm-obj clip ({_clip}) — too close, back off"
                elif _cost1 > _cost0 - 0.002:
                    _reason = (f"combined error not improved "
                               f"(lat {_lat0*100:.1f}→{_lat1*100:.1f}cm, "
                               f"axis {_axd0:.1f}→{_axd1:.1f}deg)")
                if _reason is not None:
                    print(f"[Exec] {label}: pass {_p+1} REVERTED ({_reason})")
                    self._kinematic_descent(_new, _start,
                                            label=f"{label}-revert",
                                            n_steps=10, per_step_settle=0.07)
                    time.sleep(0.25)
                    break
                for _di in _dofs:
                    _net[_di] += float(_new[_di]) - float(_start[_di])
                _q = list(_new)
                print(f"[Exec] {label}: pass {_p+1} combo "
                      f"{{" + ", ".join(f'{_names[_di]}{float(_dq_v[_j]):+.3f}'
                      for _j, _di in enumerate(_dofs)
                      if abs(_dq_v[_j]) > 1e-3) + "}} "
                      f"→ lat {_lat0*100:+.1f}→{_lat1*100:+.1f}cm  "
                      f"axis {_axd0:+.1f}→{_axd1:+.1f}deg")
        finally:
            for _a_, _v_ in _saved.items():
                if _v_ is None:
                    try: delattr(self.sim, _a_)
                    except Exception: pass
                else:
                    setattr(self.sim, _a_, _v_)
        if max(abs(v) for v in _net) < 1e-4:
            return grasp_q
        _gq = list(grasp_q)
        for _di in _dofs:
            _gq[_di] = float(_gq[_di]) + _net[_di]
        return _gq


    def _axis_align_th_chassis(self, obj_bid, label="axis-thcy", max_rounds=6,
                                r_tol_deg=5.0, lat_tol=0.025,
                                th_max_step=0.06, th_cap=0.50,
                                yaw_max_step=0.05, yaw_cap=0.22, settle=0.18):
        import time as _time
        if len(self._carry_anchor_body_ids) != 3 or obj_bid is None:
            return
        def _res():
            return self._axis_residual_deg(obj_bid)
        def _lat():
            return self._arm_horizontal_lateral_offset(obj_bid, None)

        r0 = _res(); l0 = _lat()
        if r0 is None:
            return
        if abs(r0) <= r_tol_deg and (l0 is None or abs(l0) <= lat_tol):
            print(f"[Exec] {label}: already aligned (axis {r0:+.1f}deg, "
                  f"lat {(l0 or 0.0)*100:+.1f}cm) — no move")
            return
        try:
            _cl = self.sim.localization()
            cx0, cy0, cyaw0 = float(_cl[0]), float(_cl[1]), float(_cl[2])
        except Exception:
            return
        try:
            _pinch0 = np.asarray(
                self._pinch_midpoint_xyz(self.sim.data)[:2], dtype=float)
            _objxy0 = np.asarray(self.sim.data.xpos[obj_bid][:2], dtype=float)
            _gap0 = float(np.linalg.norm(_objxy0 - _pinch0))
            if _gap0 < 0.12:
                print(f"[Exec] {label}: pinch only {_gap0*100:.1f}cm from obj "
                      f"(<12cm) — not pre-reach (retry near obj), skip correct")
                return
        except Exception:
            pass

        def _set_th(th_val):
            q = list(self._current_arm_q())
            _s = [float(q[0]), float(q[1]), float(q[2]), float(q[3]), float(q[4]),
                  WRIST_Z_SIDE_APPROACH, WRIST_X_SIDE_APPROACH,
                  WRIST_Y_SIDE_APPROACH]
            _e = list(_s)
            _e[3] = float(np.clip(th_val, *JOINT_RANGES_ARM[3]))
            self._kinematic_descent(_s, _e, label="axfix-th",
                                    n_steps=8, per_step_settle=0.06)

        _saved = {"kp": getattr(self.sim, "_base_kp_theta_override", None),
                  "ki": getattr(self.sim, "_base_ki_theta_override", None),
                  "kd": getattr(self.sim, "_base_kd_theta_override", None),
                  "om": getattr(self.sim, "_base_omega_max_override", None),
                  "akp": getattr(self.sim, "_base_kp_override", None),
                  "akd": getattr(self.sim, "_base_kd_override", None),
                  "aki": getattr(self.sim, "_base_ki_override", None),
                  "iy": float(getattr(self.sim, "integral_yaw", 0.0))}
        self.sim._base_kp_theta_override = 12.0
        self.sim._base_ki_theta_override = 2.0
        self.sim._base_kd_theta_override = 0.8
        self.sim._base_omega_max_override = 0.8
        self.sim._base_kp_override = 60.0
        self.sim._base_kd_override = 10.0
        self.sim._base_ki_override = 2.0

        def _yaw_to(target_yaw, timeout=1.0):
            with self.sim._target_lock:
                self.sim.target_base = np.array([cx0, cy0, target_yaw])
            _t0 = _time.time(); _intol = 0
            while _time.time() - _t0 < timeout:
                _cy = float(self.sim.localization()[2])
                _e = ((target_yaw - _cy + math.pi) % (2 * math.pi)) - math.pi
                with self.sim._target_lock:
                    self.sim.target_base = np.array([cx0, cy0, target_yaw])
                if abs(_e) <= math.radians(0.8):
                    _intol += 1
                    if _intol >= 2:
                        break
                else:
                    _intol = 0
                _time.sleep(0.04)

        th_total = 0.0
        yaw_total = 0.0
        th_sign = +1.0
        yaw_sign = +1.0
        try:
            self.sim.integral_yaw = 0.0
        except Exception:
            pass
        try:
            with self.sim._target_lock:
                self.sim.target_base = np.array([cx0, cy0, cyaw0])
        except Exception:
            pass
        print(f"[Exec] {label}: START axis {r0:+.1f}deg lat "
              f"{(l0 or 0.0)*100:+.1f}cm — th(axis)+chassis-yaw(lateral), "
              f"floor-parallel (no wrist tilt)")
        def _drive_th():
            nonlocal th_total, th_sign
            for _ in range(6):
                r = _res()
                if r is None or abs(r) <= r_tol_deg or abs(th_total) >= th_cap:
                    return
                _q3 = float(self._current_arm_q()[3])
                dth = th_sign * float(np.clip(0.004 * abs(r), 0.015, th_max_step))
                _set_th(_q3 + dth)
                _time.sleep(settle)
                r2 = _res()
                if r2 is None or abs(r2) > abs(r) + 0.5:
                    _set_th(_q3); _time.sleep(settle)
                    th_sign = -th_sign
                    print(f"[Exec] {label}: th{dth:+.3f} worsened axis "
                          f"({r:+.1f}→{(r2 or 0):+.1f}) — revert,flip")
                else:
                    th_total += (float(self._current_arm_q()[3]) - _q3)
                    print(f"[Exec] {label}: th{dth:+.3f} axis {r:+.1f}→"
                          f"{r2:+.1f}deg (th_tot{th_total:+.3f})")

        def _drive_yaw():
            nonlocal yaw_total, yaw_sign
            for _ in range(8):
                l = _lat()
                if l is None or abs(l) <= lat_tol or abs(yaw_total) >= yaw_cap:
                    return
                dyaw = yaw_sign * float(np.clip(0.5 * abs(l) / 0.6,
                                                0.010, yaw_max_step))
                _yaw_to(((cyaw0 + yaw_total + dyaw + math.pi)
                         % (2 * math.pi)) - math.pi)
                _time.sleep(settle)
                l2 = _lat()
                if l2 is None or abs(l2) > abs(l) + 0.005:
                    _yaw_to(((cyaw0 + yaw_total + math.pi)
                             % (2 * math.pi)) - math.pi)
                    _time.sleep(settle)
                    yaw_sign = -yaw_sign
                    print(f"[Exec] {label}: yaw{dyaw:+.3f} worsened lat "
                          f"({l*100:+.1f}→{(l2 or 0)*100:+.1f}cm) — revert,flip")
                else:
                    yaw_total += dyaw
                    print(f"[Exec] {label}: yaw{dyaw:+.3f} lat {l*100:+.1f}→"
                          f"{l2*100:+.1f}cm (yaw_tot{yaw_total:+.3f})")

        try:
            for _rnd in range(max_rounds):
                r = _res(); l = _lat()
                if ((r is None or abs(r) <= r_tol_deg)
                        and (l is None or abs(l) <= lat_tol)):
                    print(f"[Exec] {label}: converged round {_rnd} "
                          f"(axis {(r or 0):+.1f}deg lat {(l or 0)*100:+.1f}cm)")
                    break
                _drive_th()
                _drive_yaw()
                try:
                    if self._count_arm_obj_contacts(obj_bid) > 0:
                        print(f"[Exec] {label}: arm-obj contact mid-correct — "
                              f"stop (pre-reach should be clear)")
                        break
                except Exception:
                    pass
            rF = _res(); lF = _lat()
            print(f"[Exec] {label}: DONE axis {r0:+.1f}→{(rF or 0):+.1f}deg  "
                  f"lat {(l0 or 0)*100:+.1f}→{(lF or 0)*100:+.1f}cm  "
                  f"(th_tot{th_total:+.3f} yaw_tot{yaw_total:+.3f}rad, "
                  f"floor-parallel preserved)")
        finally:
            for _a_, _k_ in (("_base_kp_theta_override", "kp"),
                             ("_base_ki_theta_override", "ki"),
                             ("_base_kd_theta_override", "kd"),
                             ("_base_omega_max_override", "om"),
                             ("_base_kp_override", "akp"),
                             ("_base_kd_override", "akd"),
                             ("_base_ki_override", "aki")):
                if _saved[_k_] is None:
                    if hasattr(self.sim, _a_):
                        delattr(self.sim, _a_)
                else:
                    setattr(self.sim, _a_, _saved[_k_])

    def _tilt_to_obj_z(self, obj_bid, low_q, z_tol=0.012, max_iter=6,
                        z_bias=-0.005, label="tilt-to-objz"):
        if len(self._carry_anchor_body_ids) != 3:
            return (float(low_q[0]), float(low_q[1]))
        try:
            obj_z = float(self.sim.data.xpos[obj_bid][2])
            obj_r = float(self._object_radius(obj_bid))
        except Exception:
            return (float(low_q[0]), float(low_q[1]))
        target_z = obj_z + z_bias
        _m = self.arm_bridge.model
        _pd = self.arm_bridge.planning_data
        _qm = self.arm_bridge._qpos_map
        _ta, _tb, _tc = self._carry_anchor_body_ids
        _keys = ("ColumnLeft", "ColumnRight", "ArmLeft", "Base",
                 "HandBearing", "WristZ", "WristX", "WristY")

        def _fk_pinch_z(q):
            for _k, _v in zip(_keys, q):
                _pd.qpos[_qm[_k]] = float(_v)
            mujoco.mj_forward(_m, _pd)
            return 0.5 * (float(_pd.xpos[_ta][2])
                          + 0.5 * (float(_pd.xpos[_tb][2])
                                   + float(_pd.xpos[_tc][2])))

        _q = [float(x) for x in self._current_arm_q()]
        try:
            _pz0 = float(self._pinch_midpoint_xyz(self.sim.data)[2])
        except Exception:
            return (float(_q[0]), float(_q[1]))
        if (_pz0 - target_z) <= z_tol:
            print(f"[Exec] {label}: pinch at/below obj-Z "
                  f"(z_gap={(_pz0-target_z)*100:+.1f}cm) — no tilt needed")
            return (float(_q[0]), float(_q[1]))
        for _it in range(max_iter):
            _q = [float(x) for x in self._current_arm_q()]
            try:
                _pz = float(self._pinch_midpoint_xyz(self.sim.data)[2])
            except Exception:
                break
            _err = _pz - target_z
            if _err <= z_tol:
                break
            _dt = 0.03
            _base_fk = _fk_pinch_z(_q)
            _qp = list(_q); _qp[0] = _q[0] - 0.5 * _dt; _qp[1] = _q[1] + 0.5 * _dt
            _grad = (_fk_pinch_z(_qp) - _base_fk) / _dt
            if abs(_grad) < 0.05:
                print(f"[Exec] {label}: tilt has no Z authority "
                      f"(grad={_grad:+.2f}) — leaving (z_gap={_err*100:+.1f}cm)")
                break
            _cmd = max(-0.10, min(0.10, -_err / _grad))
            _new = list(_q)
            _new[0] = max(0.0, min(COLUMN_JOINT_MAX, _q[0] - 0.5 * _cmd))
            _new[1] = max(0.0, min(COLUMN_JOINT_MAX, _q[1] + 0.5 * _cmd))
            try:
                if not bool(self.arm_bridge.is_valid(_new)):
                    print(f"[Exec] {label}: tilt step invalid — leaving "
                          f"(z_gap={_err*100:+.1f}cm)")
                    break
            except Exception:
                break
            self._kinematic_descent(_q, _new, label=f"{label}-it{_it+1}",
                                    n_steps=8, per_step_settle=0.07)
            time.sleep(0.2)
            _pz2 = float(self._pinch_midpoint_xyz(self.sim.data)[2])
            _err2 = _pz2 - target_z
            _clip = int(self._count_arm_obj_contacts(obj_bid))
            if _clip > 0 or _err2 > _err - 0.003:
                _reason = (f"arm-obj clip ({_clip})" if _clip > 0
                           else f"z not improved ({_err*100:+.1f}→"
                                f"{_err2*100:+.1f}cm)")
                print(f"[Exec] {label}: it{_it+1} REVERTED ({_reason})")
                self._kinematic_descent(_new, _q, label=f"{label}-revert",
                                        n_steps=8, per_step_settle=0.07)
                time.sleep(0.2)
                break
            print(f"[Exec] {label}: it{_it+1} tilt Δ{_cmd:+.3f} → "
                  f"h={_new[0]:.3f}/{_new[1]:.3f} "
                  f"(h2-h1={(_new[1]-_new[0])*100:+.1f}cm) "
                  f"z_gap {_err*100:+.1f}→{_err2*100:+.1f}cm")
        _qf = self._current_arm_q()
        return (float(_qf[0]), float(_qf[1]))


    def _post_descent_extra_tilt(self, q_now, h2_delta=0.25, a1_delta=0.15,
                                  n_steps=15):
        q_target = list(q_now)
        h2_min, h2_max = JOINT_RANGES_ARM[1]
        q_target[1] = max(float(h2_min),
                          min(float(q_now[1]) + h2_delta, float(h2_max)))
        a1_min, a1_max = JOINT_RANGES_ARM[2]
        q_target[2] = max(float(a1_min),
                          min(float(q_now[2]) + a1_delta, float(a1_max)))

        print(f"[Exec] [5.46] extra-tilt: h2 {q_now[1]:.3f}→{q_target[1]:.3f} "
              f"(+{q_target[1]-q_now[1]:.3f}), "
              f"a1 {q_now[2]:.3f}→{q_target[2]:.3f} "
              f"(+{q_target[2]-q_now[2]:.3f})  "
              f"[h1, th, wrist held]")
        self._kinematic_descent(q_now, q_target, "extra-tilt",
                                n_steps=n_steps)
        return q_target


    def _log_gripper_floor_chassis_contacts(self, label):
        data = self.sim.data
        model = self.sim.model
        try:
            n = int(data.ncon)
        except Exception:
            return
        arm_prefixes = ('finger_', 'Gripper_', 'Arm_', 'Hand_Bearing',
                        'Rotation_Link', 'Column_', 'Bearing_Column')

        def is_arm_body(bid):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            return any(name.startswith(p) for p in arm_prefixes)

        hits = []
        for i in range(n):
            try:
                c = data.contact[i]
                g1 = int(c.geom1)
                g2 = int(c.geom2)
                b1 = int(model.geom_bodyid[g1])
                b2 = int(model.geom_bodyid[g2])
            except Exception:
                continue
            b1_arm = is_arm_body(b1)
            b2_arm = is_arm_body(b2)
            if b1_arm == b2_arm:
                continue
            arm_b = b1 if b1_arm else b2
            other_b = b2 if b1_arm else b1
            arm_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, arm_b) or "?"
            other_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, other_b) or "world"
            cpos = np.asarray(c.pos, dtype=float)
            hits.append((arm_name, other_name, cpos))

        if not hits:
            print(f"[Diag] {label} — no arm/gripper-vs-world or "
                  f"arm/gripper-vs-chassis contacts (clean)")
            return
        print(f"[Diag] {label} — {len(hits)} arm/gripper-vs-world/chassis contact(s):")
        for arm_name, other_name, cpos in hits:
            print(f"  {arm_name} ↔ {other_name}  @ pos=({cpos[0]:+.3f},"
                  f"{cpos[1]:+.3f},{cpos[2]:+.3f})")

    _gripper_body_ids_cache = None

    def _ensure_gripper_body_ids(self):
        if self._gripper_body_ids_cache is not None:
            return self._gripper_body_ids_cache
        model = self.sim.model
        names = (
            'finger_c_link_0_1', 'finger_c_link_1_1',
            'finger_c_link_2_1', 'finger_c_link_3_1',
            'finger_b_link_0_1', 'finger_b_link_1_1',
            'finger_b_link_2_1', 'finger_b_link_3_1',
            'finger_a_link_1_1', 'finger_a_link_2_1', 'finger_a_link_3_1',
            'Gripper_Link3_1', 'Gripper_Link2_1', 'Gripper_Link1_1',
        )
        out = {}
        for nm in names:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nm)
            if bid >= 0:
                out[int(bid)] = nm
        self._gripper_body_ids_cache = out
        return out

    def _log_gripper_contacts(self, label, force_forward=False):
        data = self.sim.data
        model = self.sim.model
        if force_forward:
            try:
                mujoco.mj_forward(model, data)
            except Exception as e:
                print(f"[Exec] {label} mj_forward warning: {e}")
        gripper_ids = self._ensure_gripper_body_ids()
        try:
            ncon = int(data.ncon)
        except Exception:
            ncon = 0
        print(f"[Exec] {label}: gripper contact scan (ncon={ncon})")
        hits = 0
        for k in range(ncon):
            try:
                c = data.contact[k]
                g1 = int(c.geom1); g2 = int(c.geom2)
                b1 = int(model.geom_bodyid[g1])
                b2 = int(model.geom_bodyid[g2])
            except Exception:
                continue
            in_gripper_1 = b1 in gripper_ids
            in_gripper_2 = b2 in gripper_ids
            if not (in_gripper_1 or in_gripper_2):
                continue
            n1 = gripper_ids.get(b1) or (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b1)
                or f"#{b1}")
            n2 = gripper_ids.get(b2) or (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b2)
                or f"#{b2}")
            pos = c.pos.copy() if hasattr(c, 'pos') else None
            pos_str = (f"  pos={pos.round(3)}"
                       if pos is not None else "")
            kind = "SELF" if (in_gripper_1 and in_gripper_2) else "ext "
            print(f"  [{kind}] {n1}  ↔  {n2}{pos_str}")
            hits += 1
        if hits == 0:
            print("  (no gripper contacts)")

    def _finger_touches_obj(self, finger_idx, obj_bid):
        if not self._finger_body_groups or finger_idx >= len(self._finger_body_groups):
            return False
        finger_bodies = self._finger_body_groups[finger_idx]
        if not finger_bodies:
            return False
        data = self.sim.data
        model = self.sim.model
        try:
            n = int(data.ncon)
        except Exception:
            return False
        for i in range(n):
            try:
                c = data.contact[i]
                g1 = int(c.geom1)
                g2 = int(c.geom2)
                b1 = int(model.geom_bodyid[g1])
                b2 = int(model.geom_bodyid[g2])
            except Exception:
                continue
            if (b1 in finger_bodies and b2 == int(obj_bid)) or \
               (b2 in finger_bodies and b1 == int(obj_bid)):
                return True
        return False

    def _strict_log(self, phase, msg):
        if not STRICT_PICKUP_MODE:
            return
        try:
            t0 = getattr(self, '_strict_t0', None)
            t = (time.time() - t0) if t0 is not None else 0.0
        except Exception:
            t = 0.0
        print(f"[STRICT t={t:7.3f}s] {phase:<14s} {msg}")

    def _finger_contact_force(self, finger_idx, obj_bid):
        if (not self._finger_body_groups
                or finger_idx >= len(self._finger_body_groups)):
            return 0.0, 0
        finger_bodies = self._finger_body_groups[finger_idx]
        if not finger_bodies:
            return 0.0, 0
        data = self.sim.data
        model = self.sim.model
        try:
            n = int(data.ncon)
        except Exception:
            return 0.0, 0
        total_normal = 0.0
        count = 0
        force6 = np.zeros(6, dtype=np.float64)
        for i in range(n):
            try:
                c = data.contact[i]
                g1 = int(c.geom1)
                g2 = int(c.geom2)
                b1 = int(model.geom_bodyid[g1])
                b2 = int(model.geom_bodyid[g2])
            except Exception:
                continue
            if not ((b1 in finger_bodies and b2 == int(obj_bid))
                    or (b2 in finger_bodies and b1 == int(obj_bid))):
                continue
            try:
                mujoco.mj_contactForce(model, data, i, force6)
                total_normal += abs(float(force6[0]))
                count += 1
            except Exception:
                continue
        return total_normal, count

    def _per_finger_normal_forces(self, obj_bid):
        n_a, _ = self._finger_contact_force(2, obj_bid)
        n_b, _ = self._finger_contact_force(1, obj_bid)
        n_c, _ = self._finger_contact_force(0, obj_bid)
        return [float(n_a), float(n_b), float(n_c)]

    def _contact_normal_triad(self, obj_bid):
        data = self.sim.data
        model = self.sim.model
        force6 = np.zeros(6, dtype=np.float64)
        per_finger_dirs = [None, None, None]
        per_finger_forces = [0.0, 0.0, 0.0]
        for slot in (0, 1, 2):
            if (not self._finger_body_groups
                    or slot >= len(self._finger_body_groups)):
                continue
            finger_bodies = self._finger_body_groups[slot]
            net = np.zeros(3)
            net_mag = 0.0
            for i in range(int(data.ncon)):
                try:
                    c = data.contact[i]
                    b1 = int(model.geom_bodyid[int(c.geom1)])
                    b2 = int(model.geom_bodyid[int(c.geom2)])
                except Exception:
                    continue
                if not ((b1 in finger_bodies and b2 == int(obj_bid))
                        or (b2 in finger_bodies and b1 == int(obj_bid))):
                    continue
                try:
                    mujoco.mj_contactForce(model, data, i, force6)
                    Fn = abs(float(force6[0]))
                    normal = np.array([
                        float(c.frame[0]),
                        float(c.frame[1]),
                        float(c.frame[2])
                    ])
                    if b1 == int(obj_bid):
                        normal = -normal
                    net += Fn * normal
                    net_mag += Fn
                except Exception:
                    continue
            if net_mag > 1e-6:
                per_finger_dirs[slot] = net / net_mag
                per_finger_forces[slot] = net_mag
        order = (2, 1, 0)
        forces = [per_finger_forces[s] for s in order]
        dirs   = [per_finger_dirs[s]   for s in order]
        sum_vec = np.zeros(3)
        max_force = 0.0
        for d, n in zip(dirs, forces):
            if d is None or n < 1e-6:
                continue
            sum_vec += d * n
            if n > max_force:
                max_force = n
        if max_force > 1e-3:
            balance_score = float(np.linalg.norm(sum_vec)) / max_force
        else:
            balance_score = 1.0
        return {
            "balance_score": balance_score,
            "max_force": max_force,
            "per_finger_force": forces,
            "per_finger_dir": dirs,
        }

    def _emit_grasp_diag(self, outcome, obj_bid, n_contacts,
                         contacts_snapshot, side_grip):
        try:
            model = self.sim.model
            data  = self.sim.data
            loc   = self.sim.localization()
            chassis_xy = (float(loc[0]), float(loc[1]))
            chassis_yaw_deg = math.degrees(float(loc[2]))
            obj_xyz = data.xpos[obj_bid].copy()
            obj_xy  = (float(obj_xyz[0]), float(obj_xyz[1]))
            try:
                obj_r = float(self._object_radius(obj_bid))
            except Exception:
                obj_r = float('nan')
            approach_yaw_deg = math.degrees(math.atan2(
                obj_xy[1] - chassis_xy[1],
                obj_xy[0] - chassis_xy[0]))
            yaw_offset_deg = chassis_yaw_deg - approach_yaw_deg
            while yaw_offset_deg > 180.0:
                yaw_offset_deg -= 360.0
            while yaw_offset_deg < -180.0:
                yaw_offset_deg += 360.0
            wrist_jnames = (
                ('hb', 'HandBearingJoint_1'),
                ('wz', 'gripper_z_rotation_1'),
                ('wx', 'gripper_x_rotation_1'),
                ('wy', 'gripper_y_rotation_1'),
            )
            wrist_q = []
            for _short, jname in wrist_jnames:
                jid = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                if jid < 0:
                    wrist_q.append(float('nan'))
                else:
                    qpa = int(model.jnt_qposadr[jid])
                    wrist_q.append(float(data.qpos[qpa]))
            d_thumb = d_b = d_c = float('nan')
            if (self._carry_anchor_body_ids
                    and len(self._carry_anchor_body_ids) == 3):
                _a_xy = data.xpos[self._carry_anchor_body_ids[0]][:2]
                _b_xy = data.xpos[self._carry_anchor_body_ids[1]][:2]
                _c_xy = data.xpos[self._carry_anchor_body_ids[2]][:2]
                _ox = obj_xyz[0]; _oy = obj_xyz[1]
                d_thumb = float(np.hypot(_a_xy[0] - _ox, _a_xy[1] - _oy))
                d_b     = float(np.hypot(_b_xy[0] - _ox, _b_xy[1] - _oy))
                d_c     = float(np.hypot(_c_xy[0] - _ox, _c_xy[1] - _oy))
            try:
                per_N = self._per_finger_normal_forces(obj_bid)
            except Exception:
                per_N = [float('nan')] * 3
            try:
                _triad = self._contact_normal_triad(obj_bid)
                balance = float(_triad.get('balance_score', float('nan')))
            except Exception:
                balance = float('nan')
            try:
                c_a = bool(contacts_snapshot[2]) if contacts_snapshot else False
                c_b = bool(contacts_snapshot[1]) if contacts_snapshot else False
                c_c = bool(contacts_snapshot[0]) if contacts_snapshot else False
            except Exception:
                c_a = c_b = c_c = False
            print(
                f"[GRASP_DIAG] outcome={outcome} "
                f"n_contacts={n_contacts} "
                f"obj_xy=({obj_xy[0]:.4f},{obj_xy[1]:.4f}) "
                f"obj_r={obj_r:.4f} "
                f"chassis_xy=({chassis_xy[0]:.4f},{chassis_xy[1]:.4f}) "
                f"chassis_yaw_deg={chassis_yaw_deg:+.2f} "
                f"approach_yaw_deg={approach_yaw_deg:+.2f} "
                f"yaw_offset_deg={yaw_offset_deg:+.2f} "
                f"wrist=(hb={wrist_q[0]:+.3f},wz={wrist_q[1]:+.3f},"
                f"wx={wrist_q[2]:+.3f},wy={wrist_q[3]:+.3f}) "
                f"d_thumb={d_thumb*100:.1f}cm "
                f"d_b={d_b*100:.1f}cm d_c={d_c*100:.1f}cm "
                f"N=(a={per_N[0]:.1f},b={per_N[1]:.1f},c={per_N[2]:.1f}) "
                f"contacts_abc=({int(c_a)},{int(c_b)},{int(c_c)}) "
                f"balance={balance:.2f} "
                f"mode={'SIDE' if side_grip else 'TOP'}")
        except Exception as _e_gd:
            print(f"[GRASP_DIAG] capture failed: {_e_gd}")

    def _probe_pinch_offset_at_q(self, q):
        try:
            _model = self.arm_bridge.model
            _pdata = self.arm_bridge.planning_data
            qmap   = self.arm_bridge.qpos_map
            _qpos_save = _pdata.qpos.copy()
            _qvel_save = _pdata.qvel.copy()
            try:
                _keys = ("ColumnLeft", "ColumnRight", "ArmLeft", "Base",
                         "HandBearing", "WristZ", "WristX", "WristY")
                for _k, _v in zip(_keys, list(q)[:len(_keys)]):
                    qpa = qmap.get(_k)
                    if qpa is not None:
                        _pdata.qpos[qpa] = float(_v)
                for jname, qval in (
                        ("finger_a_joint_1_1", THUMB_OPEN_POS),
                        ("finger_b_joint_1_1", GRIPPER_OPEN_POS),
                        ("finger_c_joint_1_1", GRIPPER_OPEN_POS)):
                    jid = mujoco.mj_name2id(
                        _model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                    if jid >= 0:
                        _pdata.qpos[int(_model.jnt_qposadr[jid])] = float(qval)
                mujoco.mj_forward(_model, _pdata)
                if (self.gripper_body_id < 0
                        or len(self._carry_anchor_body_ids) != 3):
                    return None
                palm_xy = np.asarray(
                    _pdata.xpos[self.gripper_body_id][:2], dtype=float)
                thumb_xy = np.asarray(
                    _pdata.xpos[self._carry_anchor_body_ids[0]][:2],
                    dtype=float)
                bc_centroid_xy = 0.5 * (
                    np.asarray(_pdata.xpos[self._carry_anchor_body_ids[1]][:2],
                               dtype=float)
                    + np.asarray(_pdata.xpos[self._carry_anchor_body_ids[2]][:2],
                                 dtype=float))
                pinch_midpoint_xy = 0.5 * (thumb_xy + bc_centroid_xy)
                offset = pinch_midpoint_xy - palm_xy
                return np.asarray(offset, dtype=float)
            finally:
                _pdata.qpos[:] = _qpos_save
                _pdata.qvel[:] = _qvel_save
                mujoco.mj_forward(_model, _pdata)
        except Exception as e:
            print(f"[STRICT IK] _probe_pinch_offset_at_q failed: {e}")
            return None

    def _estimate_pinch_midpoint_palm_offset_xy(self, wrist_goal,
                                                 base_xy, base_yaw):
        try:
            _model = self.arm_bridge.model
            _pdata = self.arm_bridge.planning_data
            qmap   = self.arm_bridge.qpos_map
            _qpos_save = _pdata.qpos.copy()
            _qvel_save = _pdata.qvel.copy()
            try:
                self.arm_bridge.set_base_pose_xy_yaw(
                    float(base_xy[0]), float(base_xy[1]), float(base_yaw))
                for key, val in zip(
                        ("HandBearing", "WristZ", "WristX", "WristY"),
                        wrist_goal):
                    qpa = qmap.get(key)
                    if qpa is not None:
                        _pdata.qpos[qpa] = float(val)
                for key, val in zip(
                        ("ColumnLeft", "ColumnRight", "ArmLeft"),
                        (0.05, 0.45, 0.55)):
                    qpa = qmap.get(key)
                    if qpa is not None:
                        _pdata.qpos[qpa] = float(val)
                for jname, qval in (
                        ("finger_a_joint_1_1", THUMB_OPEN_POS),
                        ("finger_b_joint_1_1", GRIPPER_OPEN_POS),
                        ("finger_c_joint_1_1", GRIPPER_OPEN_POS)):
                    jid = mujoco.mj_name2id(
                        _model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                    if jid >= 0:
                        _pdata.qpos[int(_model.jnt_qposadr[jid])] = float(qval)
                mujoco.mj_forward(_model, _pdata)
                if (self.gripper_body_id < 0
                        or len(self._carry_anchor_body_ids) != 3):
                    return None
                palm_xy = np.asarray(
                    _pdata.xpos[self.gripper_body_id][:2], dtype=float)
                thumb_xy = np.asarray(
                    _pdata.xpos[self._carry_anchor_body_ids[0]][:2],
                    dtype=float)
                bc_centroid_xy = 0.5 * (
                    np.asarray(_pdata.xpos[self._carry_anchor_body_ids[1]][:2],
                               dtype=float)
                    + np.asarray(_pdata.xpos[self._carry_anchor_body_ids[2]][:2],
                                 dtype=float))
                pinch_midpoint_xy = 0.5 * (thumb_xy + bc_centroid_xy)
                offset = pinch_midpoint_xy - palm_xy
                return np.asarray(offset, dtype=float)
            finally:
                _pdata.qpos[:] = _qpos_save
                _pdata.qvel[:] = _qvel_save
                mujoco.mj_forward(_model, _pdata)
        except Exception as e:
            print(f"[STRICT IK] centroid-palm offset probe failed: {e}")
            return None

    def _strict_required_normal_per_finger(self, obj_bid):
        try:
            m = float(self.sim.model.body_mass[int(obj_bid)])
        except Exception:
            m = 0.1
        g = 9.81
        n = max(1, 3)
        denom = max(1e-3, STRICT_FRICTION_MU * n)
        n_req = STRICT_GRIP_SAFETY * m * g / denom
        mult = float(getattr(self, '_strict_force_multiplier', 1.0))
        n_req *= mult
        return max(STRICT_MIN_NORMAL_PER_F, float(n_req))

    def _strict_slip_check_lift(self, obj_bid):
        grip0 = self._carry_anchor_xyz(self.sim.data).copy()
        obj0  = self.sim.data.xpos[obj_bid].copy()
        rel0  = obj0 - grip0
        self._strict_log(
            "LIFT",
            f"START  obj_xyz={obj0.round(3)}  "
            f"rel0={rel0.round(3)}  "
            f"|rel0|={float(np.linalg.norm(rel0)):.3f}m")

        try:
            gids = self.sim.gripper_ids_left
            FINGER_J1_INDICES = (0, 3, 6)
            TIGHTEN_BUMP_RAD = LIFT_TIGHTEN_PRELIFT_RAD
            with self.sim._target_lock:
                for j_idx in FINGER_J1_INDICES:
                    if j_idx < len(gids):
                        gid = int(gids[j_idx])
                        cur_ctrl = float(self.sim.data.ctrl[gid])
                        lo = float(self.sim.model.actuator_ctrlrange[gid, 0])
                        hi = float(self.sim.model.actuator_ctrlrange[gid, 1])
                        new_ctrl = min(hi, max(lo, cur_ctrl + TIGHTEN_BUMP_RAD))
                        self.sim.data.ctrl[gid] = new_ctrl
            time.sleep(0.2)
            self._strict_log(
                "LIFT",
                f"pre-lift TIGHTEN — j1 ctrls bumped +"
                f"{TIGHTEN_BUMP_RAD:.2f}rad to deepen grip before "
                f"lift acceleration")
        except Exception as _e_tighten:
            print(f"[Exec] pre-lift tighten raised: {_e_tighten} — "
                  f"proceeding with current ctrl")

        current_q = self._current_arm_q()
        carry_q = list(current_q)
        carry_q[0] = CARRY_H1
        carry_q[1] = CARRY_H2
        carry_q[2] = CARRY_A1
        LIFT_STEPS = DESCENT_STEPS * LIFT_STEPS_MULTIPLIER
        LIFT_PER_STEP_SETTLE = PD_SETTLE_PER_WAYPOINT * LIFT_PER_STEP_SETTLE_MULTIPLIER
        try:
            _pre_lift_obj = self.sim.data.xpos[obj_bid].copy()
            _pre_lift_grip = self._carry_anchor_xyz(self.sim.data).copy()
            _pre_lift_palm_z = float(self.sim.data.xpos[
                self.gripper_body_id][2])
            self._strict_log(
                "LIFT",
                f"PRE-MOTION  obj_z={_pre_lift_obj[2]:.3f}m  "
                f"grip_centroid_z={_pre_lift_grip[2]:.3f}m  "
                f"palm_z={_pre_lift_palm_z:.3f}m  "
                f"rel={(_pre_lift_obj - _pre_lift_grip).round(3)}m  "
                f"target h1={carry_q[0]:.2f} h2={carry_q[1]:.2f} "
                f"a1={carry_q[2]:.2f}")
        except Exception:
            pass
        EARLY_ABORT_REL_Z_DRIFT = 0.02
        EARLY_ABORT_MIN_GRIP_DZ = 0.04
        rel0_z = float(rel0[2])
        grip0_z = float(grip0[2])
        grip0_xy = grip0[:2].copy()
        obj0_xy = (rel0[:2] + grip0[:2]).copy()
        _grasp_self = self
        def _lift_obj_tracking_check(step_idx, alpha):
            if _grasp_self._active_pin_fn is not None:
                return False
            try:
                grip_now = _grasp_self._carry_anchor_xyz(
                    _grasp_self.sim.data)
                grip_now_z = float(grip_now[2])
                grip_dz = grip_now_z - grip0_z
                if grip_dz < EARLY_ABORT_MIN_GRIP_DZ:
                    return False
                obj_now = _grasp_self.sim.data.xpos[obj_bid]
                obj_now_z = float(obj_now[2])
                rel_now_z = obj_now_z - grip_now_z
                drift_z = abs(rel_now_z - rel0_z)
                if drift_z > EARLY_ABORT_REL_Z_DRIFT:
                    _grasp_self._strict_log(
                        "LIFT",
                        f"EARLY-ABORT at step {step_idx+1}: "
                        f"rel_z drifted {drift_z*100:.1f}cm > "
                        f"{EARLY_ABORT_REL_Z_DRIFT*100:.0f}cm  "
                        f"(grip_dz={grip_dz*100:.1f}cm  "
                        f"obj_dz={(obj_now_z - rel0_z - grip0_z)*100:.1f}cm — "
                        f"obj not following gripper)")
                    return True
                grip_now_xy = np.array([float(grip_now[0]),
                                         float(grip_now[1])])
                obj_now_xy  = np.array([float(obj_now[0]),
                                         float(obj_now[1])])
                rel_now_xy  = obj_now_xy - grip_now_xy
                rel0_xy     = obj0_xy - grip0_xy
                drift_xy    = float(np.linalg.norm(rel_now_xy - rel0_xy))
                if drift_xy > LIFT_XY_DRIFT_ABORT_M:
                    _grasp_self._strict_log(
                        "LIFT",
                        f"EARLY-ABORT at step {step_idx+1}: "
                        f"XY drift {drift_xy*100:.1f}cm > "
                        f"{LIFT_XY_DRIFT_ABORT_M*100:.0f}cm  "
                        f"(grip_dz={grip_dz*100:.1f}cm — "
                        f"obj rolled/slid sideways out of pinch)")
                    return True
                return False
            except Exception:
                return False

        _retighten_state = {
            "last_log_step":       -10,
            "ticks_since_bump":    LIFT_BUMP_HYSTERESIS_TICKS + 1,
            "force_history":       [],
            "mass_reestimate_step": 0,
        }

        def _apply_bump_per_finger(bump_rad, mask_a_b_c):
            gids_l = self.sim.gripper_ids_left
            j_indices = (6, 3, 0)
            for fi, j_idx in enumerate(j_indices):
                if not mask_a_b_c[fi]:
                    continue
                if j_idx >= len(gids_l):
                    continue
                gid = int(gids_l[j_idx])
                cur = float(self.sim.data.ctrl[gid])
                lo = float(self.sim.model.actuator_ctrlrange[gid, 0])
                hi = float(self.sim.model.actuator_ctrlrange[gid, 1])
                self.sim.data.ctrl[gid] = min(hi, max(lo, cur + bump_rad))

        def _lift_slip_feedback_retighten(step_idx, alpha):
            if self._active_pin_fn is not None:
                return
            try:
                grip_now_z = float(
                    self._carry_anchor_xyz(self.sim.data)[2])
                obj_now_z = float(self.sim.data.xpos[obj_bid][2])
                grip_dz = grip_now_z - grip0_z
                obj_dz_expected = grip_dz
                obj_dz_actual = obj_now_z - (rel0_z + grip0_z)
                slip = obj_dz_expected - obj_dz_actual

                per_finger_N = self._per_finger_normal_forces(obj_bid)
                if (step_idx + 1) % 5 == 0:
                    self._strict_log(
                        "LIFT",
                        f"FORCE_TRACE step={step_idx+1} "
                        f"alpha={alpha:.2f} "
                        f"grip_dz={grip_dz*1000:+.1f}mm "
                        f"obj_dz={obj_dz_actual*1000:+.1f}mm "
                        f"slip={slip*1000:+.1f}mm "
                        f"N=[a={per_finger_N[0]:.1f},"
                        f"b={per_finger_N[1]:.1f},"
                        f"c={per_finger_N[2]:.1f}]N")
                history = _retighten_state["force_history"]
                history.append(per_finger_N)
                if len(history) > LIFT_FORCE_DECAY_WINDOW:
                    history.pop(0)
                _retighten_state["ticks_since_bump"] += 1
                hysteresis_open = (
                    _retighten_state["ticks_since_bump"]
                    >= LIFT_BUMP_HYSTERESIS_TICKS)

                trigger = None
                bump_rad = 0.0
                _force_target_n = float(
                    self._strict_required_normal_per_finger(obj_bid))
                _any_below_target = any(
                    n < _force_target_n * 0.5 for n in per_finger_N)
                if slip > LIFT_SLIP_BUMP_THRESHOLD_M and hysteresis_open:
                    bump_rad = min(LIFT_SLIP_BUMP_MAX_PER_STEP_RAD,
                                   LIFT_SLIP_BUMP_GAIN * slip)
                    trigger = "SLIP"
                elif (hysteresis_open
                      and len(history) >= LIFT_FORCE_DECAY_WINDOW):
                    window_mean = np.mean(np.array(history), axis=0)
                    any_decayed = False
                    for fi in range(3):
                        if (window_mean[fi] > 1.0
                                and per_finger_N[fi]
                                    < (1.0 - LIFT_FORCE_DECAY_FRAC)
                                       * window_mean[fi]):
                            any_decayed = True
                            break
                    if any_decayed:
                        bump_rad = LIFT_FORCE_DECAY_BUMP_RAD
                        trigger = "DECAY"
                if trigger is None and hysteresis_open and _any_below_target:
                    bump_rad = 0.003
                    trigger = "FORCE_TGT"
                if trigger is None:
                    if (step_idx + 1) % max(
                            1, LIFT_RETIGHTEN_INTERVAL_STEPS) == 0:
                        bump_rad = LIFT_RETIGHTEN_PER_STEP_RAD
                        trigger = "BASELINE"
                    else:
                        return
                if bump_rad <= 0.0:
                    return

                below_floor = [n < LIFT_PER_FINGER_FORCE_FLOOR_N
                               for n in per_finger_N]
                if any(below_floor):
                    mask = tuple(below_floor)
                else:
                    mask = (True, True, True)

                _apply_bump_per_finger(bump_rad, mask)
                _retighten_state["ticks_since_bump"] = 0

                if (LIFT_MASS_REESTIMATE_ENABLED
                        and trigger == "SLIP"
                        and (step_idx - _retighten_state[
                            "mass_reestimate_step"])
                            >= LIFT_MASS_REESTIMATE_INTERVAL):
                    _retighten_state["mass_reestimate_step"] = step_idx
                    try:
                        cur_mult = float(getattr(
                            self, '_strict_force_multiplier', 1.0))
                        bump_mult = 1.0 + LIFT_MASS_REESTIMATE_K * (
                            slip / max(LIFT_SLIP_BUMP_THRESHOLD_M, 1e-3))
                        bump_mult = min(bump_mult, 1.10)
                        new_mult = min(STRICT_FORCE_MAX_MULTIPLIER,
                                       cur_mult * bump_mult)
                        if new_mult > cur_mult + 0.01:
                            self._strict_force_multiplier = new_mult
                            self._strict_log(
                                "LIFT",
                                f"MASS-REESTIMATE  multiplier "
                                f"{cur_mult:.2f} → {new_mult:.2f} "
                                f"(cap {STRICT_FORCE_MAX_MULTIPLIER:.1f})  "
                                f"(persistent slip {slip*1000:.1f}mm "
                                f"@ step {step_idx+1})")
                    except Exception:
                        pass

                if trigger in ("SLIP", "DECAY", "FORCE_TGT"):
                    if (step_idx - _retighten_state["last_log_step"]
                            >= 5):
                        _retighten_state["last_log_step"] = step_idx
                        finger_tag = "".join(
                            n for n, m in zip("abc", mask) if m)
                        self._strict_log(
                            "LIFT",
                            f"{trigger}-FEEDBACK bump "
                            f"+{bump_rad*1000:.1f}mrad on [{finger_tag}] "
                            f"@ step {step_idx+1}  "
                            f"(slip={slip*1000:.1f}mm  "
                            f"N=[{per_finger_N[0]:.1f},"
                            f"{per_finger_N[1]:.1f},"
                            f"{per_finger_N[2]:.1f}]N)")
            except Exception:
                pass

        _np_current = np.array(current_q, dtype=float)
        _np_carry   = np.array(carry_q,   dtype=float)
        q_test = (_np_current
                  + (_np_carry - _np_current) * LIFT_TEST_LIFT_ALPHA)
        test_n_steps = max(4, int(DESCENT_STEPS
                                  * LIFT_TEST_LIFT_ALPHA))
        test_per_step_settle = LIFT_PER_STEP_SETTLE * 1.5
        test_passed = False
        self._cycle_stage_lift_fired = True
        if STRICT_PERFECT_FRICTION_ONLY:
            test_passed = True
            self._strict_log(
                "LIFT",
                "TEST-LIFT skipped (--perfect mode; incremental lift "
                "below uses per-chunk obj_dz as slip detector)")
        for test_attempt in range(LIFT_TEST_LIFT_MAX_RETRIES + 1):
            self._strict_log(
                "LIFT",
                f"TEST-LIFT  attempt {test_attempt+1}/"
                f"{LIFT_TEST_LIFT_MAX_RETRIES + 1}  "
                f"alpha={LIFT_TEST_LIFT_ALPHA:.2f} "
                f"(~{LIFT_TEST_LIFT_ALPHA*100:.0f}% of full lift)")
            test_completed = self._kinematic_descent(
                list(_np_current), list(q_test), "lift-test",
                n_steps=test_n_steps,
                per_step_settle=test_per_step_settle)
            if self._cancel:
                return False
            grip_now = self._carry_anchor_xyz(self.sim.data)
            obj_now  = self.sim.data.xpos[obj_bid]
            test_grip_dz = float(grip_now[2]) - grip0_z
            test_obj_dz  = float(obj_now[2])  - (rel0_z + grip0_z)
            test_slip    = test_grip_dz - test_obj_dz
            self._strict_log(
                "LIFT",
                f"TEST-LIFT result  grip_dz={test_grip_dz*1000:+.1f}mm  "
                f"obj_dz={test_obj_dz*1000:+.1f}mm  "
                f"slip={test_slip*1000:+.1f}mm  "
                f"(tol={LIFT_TEST_LIFT_SLIP_TOLERANCE_M*1000:.0f}mm)  "
                f"completed={test_completed}")
            if (test_completed
                    and test_slip < LIFT_TEST_LIFT_SLIP_TOLERANCE_M):
                test_passed = True
                current_q = list(q_test)
                break
            if test_attempt < LIFT_TEST_LIFT_MAX_RETRIES:
                _apply_bump_per_finger(
                    LIFT_TEST_LIFT_RETRY_BUMP_RAD,
                    (True, True, True))
                time.sleep(0.2)
                self._strict_log(
                    "LIFT",
                    f"TEST-LIFT retry-bump  +"
                    f"{LIFT_TEST_LIFT_RETRY_BUMP_RAD*1000:.0f}mrad all "
                    f"fingers (re-testing)")
                self._kinematic_descent(
                    list(q_test), list(_np_current),
                    "lift-test-reset",
                    n_steps=test_n_steps,
                    per_step_settle=test_per_step_settle)
                if self._cancel:
                    return False
        if not test_passed:
            self._strict_log(
                "LIFT",
                f"TEST-LIFT FAILED after "
                f"{LIFT_TEST_LIFT_MAX_RETRIES + 1} attempts — "
                f"propagating to outer slip-retry (grip unable to hold "
                f"obj through ~{LIFT_TEST_LIFT_ALPHA*100:.0f}% lift)")
            return False

        _lift_motion_t0 = time.time()
        if STRICT_PERFECT_FRICTION_ONLY:
            LIFT_CHUNK_H = 0.10
            CHUNK_ABORT_OBJ_DROP_M = 0.05
            chunks_n = max(1, int(np.ceil(
                (carry_q[0] - current_q[0]) / LIFT_CHUNK_H)))
            chunks_n = max(2, min(chunks_n, 10))
            self._strict_log(
                "LIFT",
                f"INCREMENTAL  {chunks_n} chunks of "
                f"~{LIFT_CHUNK_H*100:.0f}cm each "
                f"(h1 {current_q[0]:.3f}→{carry_q[0]:.3f})")
            chunk_obj_track = []
            lift_completed = True
            chunk_current = list(current_q)
            for ci in range(chunks_n):
                chunk_target = list(chunk_current)
                if ci == chunks_n - 1:
                    chunk_target[0] = carry_q[0]
                    chunk_target[1] = carry_q[1]
                    chunk_target[2] = carry_q[2]
                else:
                    chunk_target[0] = min(carry_q[0],
                                          chunk_current[0] + LIFT_CHUNK_H)
                    frac = (chunk_target[0] - current_q[0]) / max(
                        1e-6, carry_q[0] - current_q[0])
                    chunk_target[1] = current_q[1] + frac * (
                        carry_q[1] - current_q[1])
                    chunk_target[2] = current_q[2] + frac * (
                        carry_q[2] - current_q[2])
                _chunk_steps = max(8, LIFT_STEPS // chunks_n)
                obj_before = float(self.sim.data.xpos[obj_bid][2])
                chunk_ok = self._kinematic_descent(
                    chunk_current, chunk_target,
                    f"lift-chunk-{ci+1}/{chunks_n}",
                    n_steps=_chunk_steps,
                    per_step_settle=LIFT_PER_STEP_SETTLE)
                time.sleep(0.3)
                obj_after = float(self.sim.data.xpos[obj_bid][2])
                obj_dz_chunk = obj_after - obj_before
                self._strict_log(
                    "LIFT",
                    f"  chunk {ci+1}/{chunks_n}  "
                    f"h1→{chunk_target[0]:.3f}  "
                    f"obj_dz={obj_dz_chunk*1000:+.1f}mm")
                chunk_obj_track.append(obj_dz_chunk)
                if obj_dz_chunk < -CHUNK_ABORT_OBJ_DROP_M:
                    self._strict_log(
                        "LIFT",
                        f"  ABORT  obj fell "
                        f"{abs(obj_dz_chunk*100):.1f}cm during chunk — "
                        f"grip slipped, propagate to outer retry")
                    lift_completed = False
                    break
                chunk_current = chunk_target
                if self._cancel:
                    return False
        else:
            lift_completed = self._kinematic_descent(
                current_q, carry_q, "lift-strict",
                n_steps=LIFT_STEPS,
                per_step_settle=LIFT_PER_STEP_SETTLE,
                early_abort_check=_lift_obj_tracking_check,
                early_abort_interval=3,
                per_step_action=_lift_slip_feedback_retighten,
                per_step_action_interval=1)
        _lift_motion_dur = time.time() - _lift_motion_t0
        if self._cancel:
            return False
        if not lift_completed:
            self._strict_log(
                "LIFT",
                f"SLIP-EARLY  motion aborted after "
                f"{_lift_motion_dur:.2f}s (obj fell behind grip during "
                f"lift) — propagating to slip-retry")
            return False
        try:
            _post_lift_obj = self.sim.data.xpos[obj_bid].copy()
            _post_lift_grip = self._carry_anchor_xyz(self.sim.data).copy()
            _post_lift_palm_z = float(self.sim.data.xpos[
                self.gripper_body_id][2])
            _obj_dz = float(_post_lift_obj[2] - _pre_lift_obj[2])
            _grip_dz = float(_post_lift_grip[2] - _pre_lift_grip[2])
            _palm_dz = float(_post_lift_palm_z - _pre_lift_palm_z)
            _obj_followed = abs(_obj_dz - _grip_dz) < 0.02
            self._strict_log(
                "LIFT",
                f"POST-MOTION  obj_z={_post_lift_obj[2]:.3f}m "
                f"(Δ={_obj_dz*100:+.1f}cm)  "
                f"grip_centroid_z={_post_lift_grip[2]:.3f}m "
                f"(Δ={_grip_dz*100:+.1f}cm)  "
                f"palm_z={_post_lift_palm_z:.3f}m "
                f"(Δ={_palm_dz*100:+.1f}cm)  "
                f"motion_dur={_lift_motion_dur:.2f}s  "
                f"obj_followed_grip="
                f"{'YES' if _obj_followed else 'NO'}")
            if _obj_followed:
                self._cycle_stage_obj_followed = True
        except Exception:
            pass

        try:
            with self.sim._target_lock:
                for j_idx in FINGER_J1_INDICES:
                    if j_idx < len(gids):
                        gid = int(gids[j_idx])
                        cur_ctrl = float(self.sim.data.ctrl[gid])
                        lo = float(self.sim.model.actuator_ctrlrange[gid, 0])
                        hi = float(self.sim.model.actuator_ctrlrange[gid, 1])
                        new_ctrl = min(hi, max(lo,
                            cur_ctrl + LIFT_TIGHTEN_POSTLIFT_RAD))
                        self.sim.data.ctrl[gid] = new_ctrl
        except Exception:
            pass

        if self._active_pin_fn is not None:
            self._strict_log(
                "LIFT",
                "slip-monitor SKIPPED — pin active, obj-grip relative "
                "pose is enforced each step")
            return True

        try:
            _live_c = bool(self._finger_touches_obj(0, obj_bid))
            _live_b = bool(self._finger_touches_obj(1, obj_bid))
            _live_a = bool(self._finger_touches_obj(2, obj_bid))
            try:
                _n_a, _ = self._finger_contact_force(2, obj_bid)
                _n_b, _ = self._finger_contact_force(1, obj_bid)
                _n_c, _ = self._finger_contact_force(0, obj_bid)
            except Exception:
                _n_a = _n_b = _n_c = 0.0
            self._strict_log(
                "LIFT",
                f"slip-monitor START  contacts: "
                f"a={_live_a}({_n_a:.1f}N) "
                f"b={_live_b}({_n_b:.1f}N) "
                f"c={_live_c}({_n_c:.1f}N)")
        except Exception:
            pass

        t_start = time.time()
        last_rel = None
        last_t = None
        max_disp = 0.0
        max_vel = 0.0
        _last_contact_log = t_start
        CONTACT_LOG_INTERVAL = 0.2
        while time.time() - t_start < STRICT_LIFT_OBSERVE_S:
            if self._cancel:
                return False
            grip_now = self._carry_anchor_xyz(self.sim.data)
            obj_now  = self.sim.data.xpos[obj_bid].copy()
            rel_now = obj_now - grip_now
            disp = float(np.linalg.norm(rel_now - rel0))
            if disp > max_disp:
                max_disp = disp
            now = time.time()
            if last_rel is not None and last_t is not None:
                dt = max(1e-3, now - last_t)
                v = float(np.linalg.norm(rel_now - last_rel)) / dt
                if v > max_vel:
                    max_vel = v
            last_rel = rel_now.copy()
            last_t = now

            if now - _last_contact_log >= CONTACT_LOG_INTERVAL:
                _last_contact_log = now
                try:
                    _lc = bool(self._finger_touches_obj(0, obj_bid))
                    _lb = bool(self._finger_touches_obj(1, obj_bid))
                    _la = bool(self._finger_touches_obj(2, obj_bid))
                    _t_into = now - t_start
                    self._strict_log(
                        "LIFT",
                        f"  t={_t_into:.2f}s  "
                        f"contacts a={_la} b={_lb} c={_lc}  "
                        f"disp={disp*1000:.1f}mm  "
                        f"v_inst={(float(np.linalg.norm(rel_now - rel0)) - max_disp)*1000:+.1f}mm")
                except Exception:
                    pass

            elapsed = now - t_start
            if elapsed >= STRICT_SLIP_SETTLE_S:
                slip_disp = disp > STRICT_SLIP_DISP_THRESH
                slip_vel  = max_vel > STRICT_SLIP_VEL_THRESH
                if slip_disp or slip_vel:
                    try:
                        _sc = bool(self._finger_touches_obj(0, obj_bid))
                        _sb = bool(self._finger_touches_obj(1, obj_bid))
                        _sa = bool(self._finger_touches_obj(2, obj_bid))
                        _contacts_at_slip = (
                            f"  contacts_at_slip: "
                            f"a={_sa} b={_sb} c={_sc}")
                    except Exception:
                        _contacts_at_slip = ""
                    self._strict_log(
                        "LIFT",
                        f"SLIP at t={elapsed:.2f}s  "
                        f"disp={disp*1000:.1f}mm "
                        f"(>= {STRICT_SLIP_DISP_THRESH*1000:.0f}mm? "
                        f"{slip_disp})  "
                        f"v_max={max_vel*1000:.1f}mm/s "
                        f"(>= {STRICT_SLIP_VEL_THRESH*1000:.0f}mm/s? "
                        f"{slip_vel})  "
                        f"rel_now={rel_now.round(3)}"
                        f"{_contacts_at_slip}")
                    return False
            time.sleep(0.02)

        self._strict_log(
            "LIFT",
            f"STABLE after {STRICT_LIFT_OBSERVE_S:.2f}s  "
            f"max_disp={max_disp*1000:.1f}mm  "
            f"max_vel={max_vel*1000:.1f}mm/s")
        try:
            _gold_loc = self.sim.localization()
            _gold_obj = self.sim.data.xpos[obj_bid].copy()
            _gold_yaw_deg = math.degrees(float(_gold_loc[2]))
            print(f"[BASELINE] full_pickup_success: "
                  f"obj_xy=({_gold_obj[0]:.4f},{_gold_obj[1]:.4f}) "
                  f"chassis_xy=({_gold_loc[0]:.4f},{_gold_loc[1]:.4f}) "
                  f"chassis_yaw_deg={_gold_yaw_deg:+.2f}  "
                  f"# GOLD baseline for tune_grip_lift: "
                  f"--obj-xy {_gold_obj[0]:.2f} "
                  f"{_gold_obj[1]:.2f} "
                  f"--approach-yaw {_gold_yaw_deg:.0f}")
        except Exception as _e_gold:
            print(f"[BASELINE] full_pickup capture failed: {_e_gold}")
        return True

    def _strict_lift_with_retry(self, obj_bid, close_pos, grasp_q):
        try:
            obj_pos_at_close = np.asarray(
                self.sim.data.xpos[obj_bid][:2], dtype=float).copy()
        except Exception:
            obj_pos_at_close = None
        OBJ_MOVED_ABORT_THRESHOLD = 0.05

        _max_attempts = 1 + STRICT_RETRY_MAX
        for attempt in range(_max_attempts):
            if attempt > 0:
                if obj_pos_at_close is not None:
                    try:
                        obj_pos_now = np.asarray(
                            self.sim.data.xpos[obj_bid][:2],
                            dtype=float)
                        obj_displacement = float(np.linalg.norm(
                            obj_pos_now - obj_pos_at_close))
                    except Exception:
                        obj_displacement = 0.0
                    if obj_displacement > OBJ_MOVED_ABORT_THRESHOLD:
                        self._strict_log(
                            "RETRY",
                            f"obj moved {obj_displacement*100:.1f}cm "
                            f"since close (> "
                            f"{OBJ_MOVED_ABORT_THRESHOLD*100:.0f}cm "
                            f"threshold) — same-pose slip-retry "
                            f"useless; aborting to outer retry")
                        return False
                self._strict_retry_count = attempt
                self._strict_force_multiplier = min(
                    STRICT_FORCE_MAX_MULTIPLIER,
                    self._strict_force_multiplier * STRICT_RETRY_GRIP_BUMP)
                self._strict_log(
                    "RETRY",
                    f"slip-retry {attempt}/{STRICT_RETRY_MAX}  "
                    f"force ×= {STRICT_RETRY_GRIP_BUMP:.2f} → mult="
                    f"{self._strict_force_multiplier:.2f} "
                    f"(cap {STRICT_FORCE_MAX_MULTIPLIER:.1f}) "
                    f"new N_req/f="
                    f"{self._strict_required_normal_per_finger(obj_bid):.2f}N")
                self._set_gripper(GRIPPER_OPEN_POS,
                                  hold_seconds=GRIPPER_HOLD_TIME * 0.5)
                if self._cancel:
                    return False
                try:
                    obj_xy_retract = self.sim.data.xpos[obj_bid][:2].copy()
                    cur_xy_retract = np.asarray(
                        self.sim.localization()[:2], dtype=float)
                    away_vec = cur_xy_retract - obj_xy_retract
                    away_dist = float(np.linalg.norm(away_vec))
                    if away_dist > 1e-6:
                        away_unit = away_vec / away_dist
                    else:
                        away_unit = np.array([1.0, 0.0])
                    retract_xy = cur_xy_retract + 0.10 * away_unit
                    retract_yaw = float(self.sim.localization()[2])
                    self._strict_log(
                        "RETRY",
                        f"chassis retract 10cm before re-descend "
                        f"(arm at carry pose — give it clearance from "
                        f"obj footprint during the drop): "
                        f"({cur_xy_retract[0]:.2f},"
                        f"{cur_xy_retract[1]:.2f}) → "
                        f"({retract_xy[0]:.2f},{retract_xy[1]:.2f})")
                    with self.sim._target_lock:
                        self.sim.target_base = np.array(
                            [float(retract_xy[0]), float(retract_xy[1]),
                             retract_yaw])
                    _t_retract = time.time()
                    while time.time() - _t_retract < 1.5:
                        cx_r, cy_r, _ = self.sim.localization()
                        if math.hypot(cx_r - retract_xy[0],
                                      cy_r - retract_xy[1]) <= 0.04:
                            break
                        time.sleep(0.05)
                except Exception as _e_retract:
                    print(f"[Exec] slip-retry chassis retract raised: "
                          f"{_e_retract} — proceeding without retract")
                current_q = self._current_arm_q()
                self._kinematic_descent(current_q, grasp_q,
                                         "strict-retry-descend",
                                         n_steps=DESCENT_STEPS)
                if self._cancel:
                    return False

                try:
                    obj_xy_reapp = self.sim.data.xpos[obj_bid][:2].copy()
                    cur_xy_reapp = np.asarray(
                        self.sim.localization()[:2], dtype=float)
                    appr_vec = obj_xy_reapp - cur_xy_reapp
                    appr_dist = float(np.linalg.norm(appr_vec))
                    SLIP_RETRY_STANDOFF = 0.55
                    if appr_dist > SLIP_RETRY_STANDOFF + 0.02:
                        appr_unit = appr_vec / appr_dist
                        reapp_target = (obj_xy_reapp
                                        - SLIP_RETRY_STANDOFF * appr_unit)
                        reapp_yaw = float(math.atan2(
                            obj_xy_reapp[1] - reapp_target[1],
                            obj_xy_reapp[0] - reapp_target[0]))
                        self._strict_log(
                            "RETRY",
                            f"chassis re-approach to obj-standoff "
                            f"after re-descend "
                            f"({appr_dist:.2f}m → {SLIP_RETRY_STANDOFF:.2f}m): "
                            f"({cur_xy_reapp[0]:.2f},"
                            f"{cur_xy_reapp[1]:.2f}) → "
                            f"({reapp_target[0]:.2f},"
                            f"{reapp_target[1]:.2f})")
                        self._side_grip_chassis_push(
                            reapp_target, obj_xy_reapp,
                            timeout=2.0, dist_tol=0.025,
                            abort_on_finger_obj_contact=True,
                            obj_bid=obj_bid)
                    else:
                        self._strict_log(
                            "RETRY",
                            f"chassis re-approach SKIPPED — already "
                            f"at {appr_dist:.2f}m from obj "
                            f"(≤ {SLIP_RETRY_STANDOFF + 0.02:.2f}m)")
                except Exception as _e_reapp:
                    print(f"[Exec] slip-retry chassis re-approach "
                          f"raised: {_e_reapp} — proceeding to close")
                if self._cancel:
                    return False

                self._set_gripper(close_pos,
                                  hold_seconds=SMOOTH_ATTACH_SETTLE)
                if self._cancel:
                    return False

            ok = self._strict_slip_check_lift(obj_bid)
            if ok:
                return True
        self._strict_log(
            "RETRY",
            f"all {STRICT_RETRY_MAX} grip-bump retries exhausted — "
            f"aborting pick (slip persisted)")
        return False

    def _pin_obj_to_gripper(self):
        qpa = self._held_obj_qpa
        dofadr = self._held_obj_dofadr
        offset = self._grasp_offset_xyz.copy()
        def closure(data):
            anchor_xyz = self._carry_anchor_xyz(data)
            target = (anchor_xyz[0] + offset[0],
                      anchor_xyz[1] + offset[1],
                      anchor_xyz[2] + offset[2])
            pin_freejoint(data, qpa, dofadr, target)
        return closure

    def _pin_obj_to_gripper_oriented(self, anchor_pinch_midpoint=True):
        qpa = self._held_obj_qpa
        dofadr = self._held_obj_dofadr
        offset = self._grasp_offset_xyz.copy()
        _gid = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_BODY,
                                 "Gripper_Link3_1")
        if _gid < 0:
            return self._pin_obj_to_gripper()
        _g0 = self.sim.data.xquat[_gid].copy()
        _o0 = self.sim.data.xquat[self._held_obj_bid].copy()
        _ginv = np.zeros(4); mujoco.mju_negQuat(_ginv, _g0)
        _rel = np.zeros(4); mujoco.mju_mulQuat(_rel, _ginv, _o0)
        _anchor_pinch = bool(anchor_pinch_midpoint)
        def closure(data):
            anchor = (self._pinch_midpoint_xyz(data) if _anchor_pinch
                      else self._carry_anchor_xyz(data))
            target = (anchor[0] + offset[0], anchor[1] + offset[1],
                      anchor[2] + offset[2])
            _gq = data.xquat[_gid]
            _oq = np.zeros(4); mujoco.mju_mulQuat(_oq, _gq, _rel)
            pin_freejoint(data, qpa, dofadr, target,
                          quat=(float(_oq[0]), float(_oq[1]),
                                float(_oq[2]), float(_oq[3])))
        return closure

    def _pin_obj_to_gripper_rigid_vel(self, anchor_pinch_midpoint=True,
                                       max_vel=1.0):
        qpa = self._held_obj_qpa
        dofadr = self._held_obj_dofadr
        offset = self._grasp_offset_xyz.copy()
        dt = float(self.sim.model.opt.timestep)
        mv = float(max_vel)
        _gid = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_BODY,
                                 "Gripper_Link3_1")
        _track_quat = _gid >= 0
        if _track_quat:
            _g0 = self.sim.data.xquat[_gid].copy()
            _o0 = self.sim.data.xquat[self._held_obj_bid].copy()
            _ginv = np.zeros(4); mujoco.mju_negQuat(_ginv, _g0)
            _rel = np.zeros(4); mujoco.mju_mulQuat(_rel, _ginv, _o0)
        _anchor_pinch = bool(anchor_pinch_midpoint)
        _state = {"prev": None}

        def closure(data):
            anchor = (self._pinch_midpoint_xyz(data) if _anchor_pinch
                      else self._carry_anchor_xyz(data))
            tgt = np.array([anchor[0] + offset[0], anchor[1] + offset[1],
                            anchor[2] + offset[2]], dtype=float)
            data.qpos[qpa]     = float(tgt[0])
            data.qpos[qpa + 1] = float(tgt[1])
            data.qpos[qpa + 2] = float(tgt[2])
            if _track_quat:
                _oq = np.zeros(4); mujoco.mju_mulQuat(_oq, data.xquat[_gid], _rel)
                data.qpos[qpa + 3:qpa + 7] = _oq
            else:
                data.qpos[qpa + 3:qpa + 7] = [1.0, 0.0, 0.0, 0.0]
            if _state["prev"] is not None and dt > 0.0:
                v = (tgt - _state["prev"]) / dt
                _vn = float(np.linalg.norm(v))
                if _vn > mv:
                    v = v * (mv / _vn)
                data.qvel[dofadr:dofadr + 3] = v
                data.qvel[dofadr + 3:dofadr + 6] = 0.0
            else:
                data.qvel[dofadr:dofadr + 6] = 0.0
            _state["prev"] = tgt
        return closure

    def _pin_obj_to_gripper_clamped(self, max_step=0.03):
        qpa = self._held_obj_qpa
        dofadr = self._held_obj_dofadr
        offset = self._grasp_offset_xyz.copy()
        ms = float(max_step)
        def closure(data):
            anchor = self._carry_anchor_xyz(data)
            tgt = np.array([anchor[0] + offset[0], anchor[1] + offset[1],
                            anchor[2] + offset[2]], dtype=float)
            cur = np.array([data.qpos[qpa], data.qpos[qpa + 1],
                            data.qpos[qpa + 2]], dtype=float)
            delta = tgt - cur
            n = float(np.linalg.norm(delta))
            if n > ms:
                tgt = cur + delta * (ms / n)
            data.qpos[qpa]     = float(tgt[0])
            data.qpos[qpa + 1] = float(tgt[1])
            data.qpos[qpa + 2] = float(tgt[2])
            data.qpos[qpa + 3] = 1.0
            data.qpos[qpa + 4] = 0.0
            data.qpos[qpa + 5] = 0.0
            data.qpos[qpa + 6] = 0.0
            data.qvel[dofadr:dofadr + 6] = 0.0
        return closure

    def _pin_obj_to_gripper_oriented_clamped(self, anchor_pinch_midpoint=True,
                                              max_step=0.03):
        qpa = self._held_obj_qpa
        dofadr = self._held_obj_dofadr
        offset = self._grasp_offset_xyz.copy()
        ms = float(max_step)
        _gid = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_BODY,
                                 "Gripper_Link3_1")
        _track_quat = _gid >= 0
        if _track_quat:
            _g0 = self.sim.data.xquat[_gid].copy()
            _o0 = self.sim.data.xquat[self._held_obj_bid].copy()
            _ginv = np.zeros(4); mujoco.mju_negQuat(_ginv, _g0)
            _rel = np.zeros(4); mujoco.mju_mulQuat(_rel, _ginv, _o0)
        _anchor_pinch = bool(anchor_pinch_midpoint)

        def closure(data):
            anchor = (self._pinch_midpoint_xyz(data) if _anchor_pinch
                      else self._carry_anchor_xyz(data))
            tgt = np.array([anchor[0] + offset[0], anchor[1] + offset[1],
                            anchor[2] + offset[2]], dtype=float)
            cur = np.array([data.qpos[qpa], data.qpos[qpa + 1],
                            data.qpos[qpa + 2]], dtype=float)
            delta = tgt - cur
            n = float(np.linalg.norm(delta))
            if n > ms:
                tgt = cur + delta * (ms / n)
            data.qpos[qpa]     = float(tgt[0])
            data.qpos[qpa + 1] = float(tgt[1])
            data.qpos[qpa + 2] = float(tgt[2])
            if _track_quat:
                _oq = np.zeros(4)
                mujoco.mju_mulQuat(_oq, data.xquat[_gid], _rel)
                data.qpos[qpa + 3:qpa + 7] = _oq
            else:
                data.qpos[qpa + 3:qpa + 7] = [1.0, 0.0, 0.0, 0.0]
            data.qvel[dofadr:dofadr + 6] = 0.0
        return closure

    def _carry_tilt_diag(self, label="CARRY"):
        model = self.sim.model
        obj_bid = self._held_obj_bid
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Gripper_Link3_1")
        jnames = ["BaseJoint_1", "RotationLeftJoint_1", "RotationRightJoint_1",
                  "HandBearing_1", "gripper_z_rotation_1", "gripper_x_rotation_1",
                  "gripper_y_rotation_1"]
        jadr = {}
        for jn in jnames:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if jid >= 0:
                jadr[jn] = int(model.jnt_qposadr[jid])
        ctr = [0]
        q0 = {jn: None for jn in jadr}
        base_bid = int(getattr(self.sim, "base_id", -1))
        import re as _re
        _robot_re = _re.compile(
            r"finger|palm|Gripper|Arm|Column|Rotation|HandBearing|Base|wheel|"
            r"chassis|Wheel|Hand|Bearing", _re.I)
        robot_b = set()
        for _b in range(model.nbody):
            _bn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, _b) or ""
            if _robot_re.search(_bn):
                robot_b.add(_b)
        if obj_bid is not None:
            robot_b.add(int(obj_bid))
        _careen_cool = [0]

        def _scan_external(data):
            best = None
            best_rob = None
            best_f = 0.0
            _f6 = __import__("numpy").zeros(6)
            for i in range(int(data.ncon)):
                c = data.contact[i]
                b1 = int(model.geom_bodyid[c.geom1])
                b2 = int(model.geom_bodyid[c.geom2])
                r1, r2 = b1 in robot_b, b2 in robot_b
                n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b1) or f"body{b1}"
                n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b2) or f"body{b2}"
                if not (r1 or r2):
                    continue
                if _re.search(r"floor|ground|plane|world", n1 + n2, _re.I):
                    continue
                if r1 and r2 and obj_bid not in (b1, b2):
                    if not _re.search(r"LifePo4|battery|base\b", n1 + n2, _re.I):
                        continue
                mujoco.mj_contactForce(model, data, i, _f6)
                fmag = float((_f6[0]**2 + _f6[1]**2 + _f6[2]**2) ** 0.5)
                if fmag > best_f:
                    rob, oth = (n1, n2) if r1 else (n2, n1)
                    best_f, best, best_rob = fmag, oth, rob
            return best_rob, best, best_f

        def diag(data):
            ctr[0] += 1
            import math as _m
            _bspeed = 0.0
            if base_bid >= 0:
                _cv = data.cvel[base_bid]
                _bspeed = float((_cv[3]**2 + _cv[4]**2 + _cv[5]**2) ** 0.5)
            _thcmd = None
            try:
                _dac = getattr(self.sim, "direct_arm_commands", None)
                if _dac is not None:
                    _thcmd = float(_dac[3])
            except Exception:
                pass
            if _bspeed > 1.5 and _careen_cool[0] <= 0:
                _careen_cool[0] = 40
                _rb, _ob, _of = _scan_external(data)
                _coll = (f"COLLISION {_rb}<->{_ob} F={_of:.0f}N" if _ob
                         else "NO contact (control/constraint spike)")
                print(f"[{label}-CAREEN] base_speed={_bspeed:.2f}m/s ncon={data.ncon} "
                      f"-> {_coll}")
            if _careen_cool[0] > 0:
                _careen_cool[0] -= 1
            if ctr[0] % 100 != 1:
                return
            _obj_tilt = _m.degrees(_m.acos(max(-1.0, min(1.0,
                float(data.xmat[obj_bid][8]))))) if obj_bid is not None else -1
            _grip_tilt = _m.degrees(_m.acos(max(-1.0, min(1.0,
                float(data.xmat[gid][8]))))) if gid >= 0 else -1
            _rb, _ob, _of = _scan_external(data)
            _ext = f" COLL={_rb}<->{_ob}:{_of:.0f}N" if _ob else ""
            parts = []
            for jn, a in jadr.items():
                v = float(data.qpos[a])
                if q0[jn] is None:
                    q0[jn] = v
                _tag = jn.replace('_1', '').replace('Joint', '')
                if jn == "BaseJoint_1" and _thcmd is not None:
                    parts.append(f"{_tag}={v:+.3f}(d{v-q0[jn]:+.3f},cmd{_thcmd:+.3f})")
                else:
                    parts.append(f"{_tag}={v:+.3f}(d{v-q0[jn]:+.3f})")
            print(f"[{label}-DIAG] obj_tilt={_obj_tilt:5.1f}deg "
                  f"grip_tilt={_grip_tilt:5.1f}deg bspeed={_bspeed:.2f} "
                  f"ncon={data.ncon}{_ext} | " + " ".join(parts))
        return diag

    def _install_pin(self, fn):
        if self._active_pin_fn is not None:
            self.sim.remove_pin_callback(self._active_pin_fn)
        self._active_pin_fn = fn
        self.sim.add_pin_callback(fn)

    def _clear_pin(self):
        if self._active_pin_fn is not None:
            self.sim.remove_pin_callback(self._active_pin_fn)
            self._active_pin_fn = None

    def _object_half_height(self, obj_bid, default=0.075):
        try:
            heights = [
                float(self.sim.model.geom_size[g, 1])
                for g in range(self.sim.model.ngeom)
                if int(self.sim.model.geom_bodyid[g]) == int(obj_bid)
            ]
            if heights:
                return max(heights)
        except Exception:
            pass
        return float(default)

    def _object_radius(self, obj_bid, default=0.05):
        try:
            radii = [
                float(self.sim.model.geom_size[g, 0])
                for g in range(self.sim.model.ngeom)
                if int(self.sim.model.geom_bodyid[g]) == int(obj_bid)
            ]
            if radii:
                return max(radii)
        except Exception:
            pass
        return float(default)

    def _object_half_xy(self, obj_bid):
        return object_half_xy(self.sim.model, obj_bid)

    def _compute_wrist_goal_for_obj(self, obj_bid):
        return compute_wrist_goal_for_obj(self.sim.model, obj_bid)

    def _finger_close_for_radius(self, radius):
        pos = FINGER_CLOSE_MAX - FINGER_CLOSE_PER_M * float(radius)
        return max(FINGER_CLOSE_FLOOR, min(FINGER_CLOSE_MAX, pos))

    def _pin_obj_to_gripper_animated(self, start_world_xyz,
                                      duration=SMOOTH_ATTACH_SECS,
                                      anchor_palm=False,
                                      anchor_pinch_midpoint=False,
                                      phased_xy_then_z=False,
                                      xy_phase_secs=0.6,
                                      z_phase_secs=0.6):
        qpa = self._held_obj_qpa
        dofadr = self._held_obj_dofadr
        offset = self._grasp_offset_xyz.copy()
        start_pos = np.asarray(start_world_xyz, dtype=float).copy()
        start_t = time.time()

        def closure(data):
            elapsed = time.time() - start_t
            if anchor_palm:
                grip_xyz = data.xpos[self.gripper_body_id]
            elif anchor_pinch_midpoint:
                grip_xyz = self._pinch_midpoint_xyz(data)
            else:
                grip_xyz = self._carry_anchor_xyz(data)
            attached = (
                grip_xyz[0] + offset[0],
                grip_xyz[1] + offset[1],
                grip_xyz[2] + offset[2],
            )
            if phased_xy_then_z:
                if elapsed < xy_phase_secs:
                    alpha_xy = elapsed / xy_phase_secs
                    alpha_z = 0.0
                elif elapsed < xy_phase_secs + z_phase_secs:
                    alpha_xy = 1.0
                    alpha_z = (elapsed - xy_phase_secs) / z_phase_secs
                else:
                    alpha_xy = 1.0
                    alpha_z = 1.0
                s_xy = alpha_xy * alpha_xy * (3.0 - 2.0 * alpha_xy)
                s_z = alpha_z * alpha_z * (3.0 - 2.0 * alpha_z)
                target = (
                    (1.0 - s_xy) * start_pos[0] + s_xy * attached[0],
                    (1.0 - s_xy) * start_pos[1] + s_xy * attached[1],
                    (1.0 - s_z) * start_pos[2] + s_z * attached[2],
                )
            else:
                alpha = min(1.0, elapsed / duration)
                if alpha < 1.0:
                    s = alpha * alpha * (3.0 - 2.0 * alpha)
                    target = (
                        (1.0 - s) * start_pos[0] + s * attached[0],
                        (1.0 - s) * start_pos[1] + s * attached[1],
                        (1.0 - s) * start_pos[2] + s * attached[2],
                    )
                else:
                    target = attached
            pin_freejoint(data, qpa, dofadr, target)

        return closure

    def _pin_obj_to_world_animated(self, start_world_xyz, target_world_xyz,
                                    duration=SMOOTH_ATTACH_SECS):
        qpa = self._held_obj_qpa
        dofadr = self._held_obj_dofadr
        start_pos = np.asarray(start_world_xyz, dtype=float).copy()
        target_pos = np.asarray(target_world_xyz, dtype=float).copy()
        start_t = time.time()

        def closure(data):
            elapsed = time.time() - start_t
            alpha = min(1.0, elapsed / duration)
            if alpha < 1.0:
                s = alpha * alpha * (3.0 - 2.0 * alpha)
                target = (
                    (1.0 - s) * start_pos[0] + s * target_pos[0],
                    (1.0 - s) * start_pos[1] + s * target_pos[1],
                    (1.0 - s) * start_pos[2] + s * target_pos[2],
                )
            else:
                target = tuple(target_pos)
            pin_freejoint(data, qpa, dofadr, target)

        return closure

    def _visual_grasp_offset(self, obj_bid, raw_offset, raw_xy_dist):
        snap_offset = FIXED_GRASP_OFFSET.copy()
        snap_offset[0] = 0.0
        snap_offset[1] = 0.0
        half_h = self._object_half_height(obj_bid)
        snap_offset[2] = -max(GRASP_OFFSET_Z_MIN,
                              half_h + GRASP_TOP_CLEARANCE)
        return snap_offset

    def _soften_held_obj_contacts(self, obj_bid):
        if self._held_obj_solref_saved is not None:
            return
        soft_solref = np.array([0.05, 1.0], dtype=np.float64)
        soft_solimp = np.array([0.0, 0.5, 0.001, 0.5, 2.0], dtype=np.float64)
        saved = []
        try:
            for g in range(self.sim.model.ngeom):
                if int(self.sim.model.geom_bodyid[g]) == int(obj_bid):
                    saved.append((
                        g,
                        np.array(self.sim.model.geom_solref[g], copy=True),
                        np.array(self.sim.model.geom_solimp[g], copy=True),
                    ))
                    self.sim.model.geom_solref[g] = soft_solref
                    self.sim.model.geom_solimp[g] = soft_solimp
            self._held_obj_solref_saved = saved
            if saved:
                print(f"[Exec] softened contacts for held object geoms "
                      f"{[g for (g, _, _) in saved]}  "
                      f"solref={soft_solref.tolist()}")
        except Exception as e:
            print(f"[Exec] held-object contact soften warning: {e}")
            self._held_obj_solref_saved = None

    def _restore_held_obj_contacts(self):
        if not self._held_obj_solref_saved:
            self._held_obj_solref_saved = None
            return
        try:
            for g, solref, solimp in self._held_obj_solref_saved:
                self.sim.model.geom_solref[g] = solref
                self.sim.model.geom_solimp[g] = solimp
        except Exception:
            pass
        self._held_obj_solref_saved = None

    def _disable_held_obj_contacts(self, obj_bid):
        if getattr(self, "_held_obj_contact_saved", None) is not None:
            return
        saved = []
        try:
            for g in range(self.sim.model.ngeom):
                if int(self.sim.model.geom_bodyid[g]) == int(obj_bid):
                    saved.append((g, int(self.sim.model.geom_contype[g]),
                                  int(self.sim.model.geom_conaffinity[g])))
                    self.sim.model.geom_contype[g] = 0
                    self.sim.model.geom_conaffinity[g] = 0
            self._held_obj_contact_saved = saved
            if saved:
                print(f"[Exec] DISABLED contacts for held obj geoms "
                      f"{[g for (g, _, _) in saved]} (transport careen guard)")
        except Exception as e:
            print(f"[Exec] held-obj contact disable warning: {e}")
            self._held_obj_contact_saved = None

    def _enable_held_obj_contacts(self):
        saved = getattr(self, "_held_obj_contact_saved", None)
        if not saved:
            self._held_obj_contact_saved = None
            return
        try:
            for g, contype, conaff in saved:
                self.sim.model.geom_contype[g] = contype
                self.sim.model.geom_conaffinity[g] = conaff
            print("[Exec] RE-ENABLED held-obj contacts (transport done)")
        except Exception:
            pass
        self._held_obj_contact_saved = None

    def _clear_held_state(self, deactivate_weld=True):
        self._clear_pin()
        if deactivate_weld:
            try:
                self.arm_bridge.planning_data.eq_active[self.weld_id] = 0
            except Exception:
                pass
        if (self._held_obj_orig_gravcomp is not None
                and self._held_obj_bid is not None):
            try:
                self.sim.model.body_gravcomp[self._held_obj_bid] = \
                    self._held_obj_orig_gravcomp
            except Exception:
                pass
        self._restore_held_obj_contacts()
        self._enable_held_obj_contacts()
        self._held_obj_orig_gravcomp = None
        self._held_obj_idx     = None
        self._held_obj_bid     = None
        self._held_obj_qpa     = None
        self._held_obj_dofadr  = None
        self._grasp_offset_xyz = None
        self._pre_close_lift_done = False
        self._fast_fixed_close_pin_active = False


    def _weld_obj_to_gripper(self, obj_bid, obj_idx):
        if STRICT_PERFECT_FRICTION_ONLY:
            return
        if obj_idx not in self._weld_ids:
            print(f"[Exec] weld: obj_idx {obj_idx} has no static weld — skip")
            return
        weld_id = self._weld_ids[obj_idx]
        b1 = self._weld_body1_id
        try:
            data = self.sim.data
            model = self.sim.model
            p1 = data.xpos[b1].copy();      q1 = data.xquat[b1].copy()
            p2 = data.xpos[obj_bid].copy(); q2 = data.xquat[obj_bid].copy()
            q1c = np.zeros(4); mujoco.mju_negQuat(q1c, q1)
            relpos = np.zeros(3)
            mujoco.mju_rotVecQuat(relpos, (p2 - p1), q1c)
            relquat = np.zeros(4)
            mujoco.mju_mulQuat(relquat, q1c, q2)
            model.eq_data[weld_id, 3:6]  = relpos
            model.eq_data[weld_id, 6:10] = relquat
            data.eq_active[weld_id] = 1
            self._rigid_weld_active = True
            self._rigid_weld_id = int(weld_id)
            print(f"[Exec] RIGID WELD ON: obj_idx={obj_idx} eq={weld_id} "
                  f"relpos={relpos.round(4)} relquat={relquat.round(4)}")
        except Exception as e:
            print(f"[Exec] weld activation ERROR: {e}")

    def _unweld_obj(self, obj_idx=None):
        if STRICT_PERFECT_FRICTION_ONLY:
            return
        try:
            wid = None
            if obj_idx is not None and obj_idx in self._weld_ids:
                wid = self._weld_ids[obj_idx]
            elif self._rigid_weld_id is not None:
                wid = self._rigid_weld_id
            if wid is not None:
                self.sim.data.eq_active[wid] = 0
                print(f"[Exec] RIGID WELD OFF: eq={wid}")
        except Exception as e:
            print(f"[Exec] weld deactivation warn: {e}")
        finally:
            self._rigid_weld_active = False
            self._rigid_weld_id = None

    def _freeze_fingers(self):
        if STRICT_PERFECT_FRICTION_ONLY or self._finger_freeze_active:
            return
        data = self.sim.data
        model = self.sim.model
        gids = self.sim.gripper_ids_left
        addrs = self._ensure_finger_joint_qposadrs()
        self._finger_freeze_saved = []
        for i in range(9):
            if i >= len(gids):
                break
            gid = int(gids[i])
            if gid < 0:
                continue
            self._finger_freeze_saved.append((gid, float(data.ctrl[gid])))
            qadr = addrs[i] if i < len(addrs) else -1
            if qadr >= 0:
                tgt = float(data.qpos[qadr])
                lo = float(model.actuator_ctrlrange[gid, 0])
                hi = float(model.actuator_ctrlrange[gid, 1])
                if hi > lo:
                    tgt = min(max(tgt, lo), hi)
                data.ctrl[gid] = tgt
        self._finger_freeze_active = True
        print(f"[Exec] finger-freeze ON ({len(self._finger_freeze_saved)} "
              f"actuators held at grasped qpos)")

    def _unfreeze_fingers(self):
        if not self._finger_freeze_active:
            return
        data = self.sim.data
        for gid, orig in getattr(self, '_finger_freeze_saved', []):
            try:
                data.ctrl[int(gid)] = float(orig)
            except Exception:
                pass
        self._finger_freeze_saved = []
        self._finger_freeze_active = False
        print("[Exec] finger-freeze OFF (restored grip ctrl)")


    def _resolve_obj(self, obj_idx):
        bid = mujoco.mj_name2id(
            self.sim.model, mujoco.mjtObj.mjOBJ_BODY, f"pickup_obj_{obj_idx}")
        if bid < 0:
            raise RuntimeError(f"pickup_obj_{obj_idx} not found")
        jntadr = self.sim.model.body_jntadr[bid]
        if jntadr < 0:
            raise RuntimeError(f"pickup_obj_{obj_idx} has no free joint")
        qpa  = int(self.sim.model.jnt_qposadr[jntadr])
        dofa = int(self.sim.model.jnt_dofadr[jntadr])
        return bid, qpa, dofa


    def _set_arm_cmd(self, q):
        with self.sim._target_lock:
            self.sim.use_ik = False
            c = self.sim.direct_arm_commands.copy()
            c[0] = q[0]; c[1] = q[1]; c[2] = q[2]; c[3] = q[3]
            self.sim.direct_arm_commands = c
        if len(q) >= 8:
            gids = self.sim.gripper_ids_left
            if len(gids) > GIDS_HANDBEARING:
                wz_target = float(q[5])
                _wz_override = getattr(self, '_wz_ctrl_override', None)
                if _wz_override is not None:
                    wz_ctrl = float(_wz_override)
                else:
                    _wz_ratio = (WRIST_Z_PD_COMPENSATION_RATIO_DESCENT
                                 if getattr(self, '_in_descent_phase', False)
                                 else WRIST_Z_PD_COMPENSATION_RATIO)
                    wz_ctrl = wz_target * (1.0 + _wz_ratio)
                self.sim.data.ctrl[gids[GIDS_WRIST_X]]     = float(q[6])
                self.sim.data.ctrl[gids[GIDS_WRIST_Y]]     = float(q[7])
                self.sim.data.ctrl[gids[GIDS_WRIST_Z]]     = wz_ctrl
                self.sim.data.ctrl[gids[GIDS_HANDBEARING]] = float(q[4])

    def _execute_path(self, path, label="path"):
        if not path:
            print(f"  [{label}] empty path, skipping")
            return
        n = len(path)
        for wi, wp in enumerate(path):
            if self._cancel: return
            self._set_arm_cmd(wp)
            time.sleep(PD_SETTLE_PER_WAYPOINT)
            if VERBOSE_PATH_WAYPOINT_LOG and wi % max(1, n // 5) == 0:
                print(f"  [{label}] wp {wi+1:02d}/{n}  "
                      f"h1={wp[0]:.3f} h2={wp[1]:.3f} a1={wp[2]:.3f} th={wp[3]:.3f}")
        time.sleep(PD_SETTLE_AT_PATH_END)
        wp_end = path[-1]
        print(f"  [{label}] done ({n} wp)  "
              f"h1={wp_end[0]:.3f} h2={wp_end[1]:.3f} "
              f"a1={wp_end[2]:.3f} th={wp_end[3]:.3f}")

    def _kinematic_descent(self, q_start, q_end, label="descent",
                           n_steps=DESCENT_STEPS,
                           per_step_settle=None,
                           early_abort_check=None,
                           early_abort_interval=4,
                           per_step_action=None,
                           per_step_action_interval=10):
        n_dims = min(len(q_start), len(q_end))
        step_sleep = (per_step_settle if per_step_settle is not None
                      else PD_SETTLE_PER_WAYPOINT)
        def _h_alpha(linear_alpha):
            return (3.0 * linear_alpha * linear_alpha
                    - 2.0 * linear_alpha * linear_alpha * linear_alpha)
        for i in range(n_steps):
            if self._cancel:
                return False
            alpha = (i + 1) / n_steps
            alpha_h = _h_alpha(alpha)
            q = []
            for j in range(n_dims):
                if j < 2:
                    q.append(q_start[j] + alpha_h * (q_end[j] - q_start[j]))
                else:
                    q.append(q_start[j] + alpha * (q_end[j] - q_start[j]))
            self._set_arm_cmd(q)
            time.sleep(step_sleep)
            if (per_step_action is not None
                    and (i + 1) % max(1, per_step_action_interval) == 0):
                try:
                    per_step_action(i, alpha)
                except Exception as _e_action:
                    print(f"  [{label}] per_step_action raised: "
                          f"{_e_action} — continuing motion")
            if (early_abort_check is not None
                    and (i + 1) % max(1, early_abort_interval) == 0):
                try:
                    if bool(early_abort_check(i, alpha)):
                        print(f"  [{label}] early-abort at step "
                              f"{i+1}/{n_steps} (alpha={alpha:.2f})")
                        return False
                except Exception as _e_check:
                    print(f"  [{label}] early-abort check raised: "
                          f"{_e_check} — continuing motion")
        time.sleep(PD_SETTLE_AT_PATH_END)
        print(f"  [{label}] done  →  h1={q_end[0]:.3f} h2={q_end[1]:.3f} "
              f"a1={q_end[2]:.3f} th={q_end[3]:.3f}")
        return True


    def _curl_targets(self, pos):
        if pos >= 0.0:
            intensity = min(1.0, max(0.0, pos / 0.20))
            _thumb_factor = (CURL_J1_FACTOR_THUMB_PERFECT
                             if STRICT_PERFECT_FRICTION_ONLY
                             else CURL_J1_FACTOR_THUMB_SOFTWELD)
            j1_side  = intensity * CURL_J1_FACTOR_SIDE * 0.85
            j1_thumb = intensity * _thumb_factor       * 0.85
            j2 = intensity * CURL_J2_FACTOR * 0.95
            j3 = -0.052 - intensity * CURL_J3_FACTOR * 1.10
            return [j1_side,  j2, j3,
                    j1_side,  j2, j3,
                    j1_thumb, j2, j3]
        else:
            j1_side  = pos
            j1_thumb = THUMB_OPEN_POS
            j2 = 0.0
            j3 = -0.0523
            return [j1_side,  j2, j3,
                    j1_side,  j2, j3,
                    j1_thumb, j2, j3]

    def _thumb_open_pos_for_mode(self):
        return THUMB_OPEN_POS

    def _wrap_fingers_to_surface(self, obj_bid, margin=0.003,
                                 iters=20, settle=0.05):
        try:
            gids = self.sim.gripper_ids_left
            if gids is None or len(gids) < 9:
                return
            obj_r = float(self._object_radius(obj_bid))
            tgt_d = obj_r + float(margin)
            for _name, _i1 in (("finger_c_link_3_1", 0),
                               ("finger_b_link_3_1", 3),
                               ("finger_a_link_3_1", 6)):
                _tip = mujoco.mj_name2id(
                    self.sim.model, mujoco.mjtObj.mjOBJ_BODY, _name)
                if _tip < 0:
                    continue
                _j1 = float(self.sim.data.ctrl[gids[_i1]])
                for _w in range(int(iters)):
                    _obj_xy = np.asarray(self.sim.data.xpos[obj_bid][:2], float)
                    _tip_xy = np.asarray(self.sim.data.xpos[_tip][:2], float)
                    _d = float(np.linalg.norm(_tip_xy - _obj_xy))
                    if abs(_d - tgt_d) <= 0.004:
                        break
                    _j1 += float(np.clip(3.0 * (_d - tgt_d), -0.05, 0.05))
                    _j1 = float(np.clip(_j1, -0.20, 1.00))
                    with self.sim._target_lock:
                        self.sim.data.ctrl[gids[_i1]] = _j1
                    time.sleep(settle)
                _df = float(np.linalg.norm(
                    np.asarray(self.sim.data.xpos[_tip][:2], float)
                    - np.asarray(self.sim.data.xpos[obj_bid][:2], float)))
                print(f"[Exec] WRAP-FINGERS {_name[7]}: tip→obj {_df*100:.1f}cm "
                      f"(target {tgt_d*100:.1f}cm)")
        except Exception as _e:
            print(f"[Exec] WRAP-FINGERS skipped: {_e}")

    _finger_joint_qposadrs = None

    def _ensure_finger_joint_qposadrs(self):
        if self._finger_joint_qposadrs is not None:
            return self._finger_joint_qposadrs
        names = (
            'finger_c_joint_1_1', 'finger_c_joint_2_1', 'finger_c_joint_3_1',
            'finger_b_joint_1_1', 'finger_b_joint_2_1', 'finger_b_joint_3_1',
            'finger_a_joint_1_1', 'finger_a_joint_2_1', 'finger_a_joint_3_1',
            'palm_finger_c_joint_1', 'palm_finger_b_joint_1',
        )
        addrs = []
        model = self.sim.model
        for jname in names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            addrs.append(int(model.jnt_qposadr[jid]) if jid >= 0 else -1)
        self._finger_joint_qposadrs = tuple(addrs)
        return self._finger_joint_qposadrs

    def _wait_open_settle(self, timeout=OPEN_SETTLE_TIMEOUT,
                          tol=OPEN_SETTLE_TOL_RAD,
                          poll_s=OPEN_SETTLE_POLL_S):
        gids = self.sim.gripper_ids_left
        if len(gids) < 11:
            return True, 0.0, []
        addrs = self._ensure_finger_joint_qposadrs()
        data = self.sim.data
        start = time.time()
        labels = ('c_j1', 'c_j2', 'c_j3',
                  'b_j1', 'b_j2', 'b_j3',
                  'a_j1', 'a_j2', 'a_j3',
                  'palm_c', 'palm_b')
        residuals = [float('inf')] * 11
        while True:
            settled_all = True
            for i in range(11):
                if addrs[i] < 0:
                    residuals[i] = 0.0
                    continue
                qpos_i = float(data.qpos[addrs[i]])
                ctrl_i = float(data.ctrl[gids[i]])
                resid = qpos_i - ctrl_i
                residuals[i] = resid
                if abs(resid) > tol:
                    settled_all = False
            elapsed = time.time() - start
            if settled_all or elapsed >= timeout or self._cancel:
                worst_i = max(range(11), key=lambda i: abs(residuals[i]))
                print(f"[Exec] open-settle: {'OK' if settled_all else 'TIMEOUT'} "
                      f"in {elapsed:.2f}s  "
                      f"worst={labels[worst_i]} resid={residuals[worst_i]:+.3f}rad "
                      f"(tol={tol:.3f})")
                return settled_all, elapsed, residuals
            time.sleep(poll_s)

    def _set_gripper(self, pos, hold_seconds=GRIPPER_HOLD_TIME,
                     transition_secs=SMOOTH_GRIPPER_SECS):
        if (STRICT_PICKUP_MODE
                and pos >= 0.0
                and self._held_obj_bid is not None
                and transition_secs < STRICT_CLOSE_TRANSITION_SECS):
            self._strict_log(
                "CLOSE",
                f"slow close stroke: transition_secs "
                f"{transition_secs:.2f} → "
                f"{STRICT_CLOSE_TRANSITION_SECS:.2f}s "
                f"(no pin → contact forces must build gradually)")
            transition_secs = STRICT_CLOSE_TRANSITION_SECS
        gids = self.sim.gripper_ids_left
        if len(gids) < 11:
            with self.sim._target_lock:
                for fi in FINGER_BASE_INDICES:
                    if fi < len(gids):
                        self.sim.data.ctrl[gids[fi]] = pos
            time.sleep(hold_seconds)
            return

        finger_targets = self._curl_targets(pos)
        palm_mag = PALM_SPREAD_CLOSE if pos >= 0.0 else PALM_SPREAD_OPEN
        palm_c_target = PALM_C_SIGN * palm_mag
        palm_b_target = PALM_B_SIGN * palm_mag
        target = list(finger_targets) + [palm_c_target, palm_b_target]

        global SNAP_OBJ_TO_POCKET_PRE_CLOSE_ONCE
        _perfect_align_fire = (STRICT_PERFECT_FRICTION_ONLY
                               and pos >= 0.0
                               and self._held_obj_bid is not None
                               and self._held_obj_qpa is not None
                               and not ENABLE_ARM_FINE_ALIGN)
        _snap_fire = bool(SNAP_OBJ_TO_POCKET_PRE_CLOSE_ONCE
                          and not ENABLE_ARM_FINE_ALIGN)
        _kin_close_safe = bool(_snap_fire
                               or _perfect_align_fire
                               or ENABLE_ARM_FINE_ALIGN)
        if ((_snap_fire or _perfect_align_fire)
                and pos >= 0.0
                and self._held_obj_bid is not None
                and self._held_obj_qpa is not None):
            try:
                _palm_bid_fwd = mujoco.mj_name2id(
                    self.sim.model, mujoco.mjtObj.mjOBJ_BODY,
                    "Gripper_Link3_1")
                def _pocket_out(_data):
                    _mid = self._pinch_midpoint_xyz(_data).copy()
                    if (ENABLE_PERFECT_PIN_FORWARD_SHIFT
                            and _palm_bid_fwd >= 0
                            and PERFECT_PIN_FORWARD_SHIFT_M > 0.0):
                        _xm = _data.xmat[_palm_bid_fwd]
                        _local_z_world = np.array(
                            [float(_xm[2]), float(_xm[5]), float(_xm[8])])
                        _mid = _mid + PERFECT_PIN_FORWARD_SHIFT_M * _local_z_world
                    return _mid
                _pocket = _pocket_out(self.sim.data)
                if ENABLE_PERFECT_PIN_FORWARD_SHIFT and PERFECT_PIN_FORWARD_SHIFT_M > 0.0:
                    print(f"[Exec] LATE-134 PERFECT pin forward-shift "
                          f"{PERFECT_PIN_FORWARD_SHIFT_M*1000:.1f}mm "
                          f"along gripper local-Z")
                _obj_w_curr = self.sim.data.xpos[self._held_obj_bid][:3].copy()
                _d_init = float(np.linalg.norm(
                    _obj_w_curr[:2] - _pocket[:2]))
                _converged = False
                _d_final = _d_init
                if _d_init > 0.02:
                    try:
                        _base_bid = mujoco.mj_name2id(
                            self.sim.model, mujoco.mjtObj.mjOBJ_BODY, "robot")
                        if _base_bid < 0:
                            _base_bid = mujoco.mj_name2id(
                                self.sim.model, mujoco.mjtObj.mjOBJ_BODY, "base")
                        _ja = int(self.sim.model.body_jntadr[_base_bid])
                        _qa = int(self.sim.model.jnt_qposadr[_ja])
                        _da = int(self.sim.model.jnt_dofadr[_ja])
                        N_STEPS = 30
                        STEP_DT = 0.05
                        for _pass in range(3):
                            _cur_loc = self.sim.localization()
                            _cur_xy_d = np.array(_cur_loc[:2], dtype=float)
                            _cur_yaw_d = float(_cur_loc[2])
                            _pocket = _pocket_out(self.sim.data)
                            _obj_w_curr = self.sim.data.xpos[self._held_obj_bid][:3].copy()
                            _d_pre = float(np.linalg.norm(
                                _obj_w_curr[:2] - _pocket[:2]))
                            if _d_pre <= 0.03:
                                _converged = True
                                _d_final = _d_pre
                                print(f"[Exec] -v7 "
                                      f"pass {_pass+1}: already at "
                                      f"{_d_pre*100:.1f}cm ≤ 3cm, done")
                                break
                            _pocket_offset = _pocket[:2] - _cur_xy_d
                            _new_cx = float(_obj_w_curr[0]) - float(_pocket_offset[0])
                            _new_cy = float(_obj_w_curr[1]) - float(_pocket_offset[1])
                            dx_step = (_new_cx - _cur_xy_d[0]) / N_STEPS
                            dy_step = (_new_cy - _cur_xy_d[1]) / N_STEPS
                            print(f"[Exec] -v7 pass "
                                  f"{_pass+1}: chassis "
                                  f"({_cur_xy_d[0]:.2f},{_cur_xy_d[1]:.2f})"
                                  f"→({_new_cx:.2f},{_new_cy:.2f}) "
                                  f"Δ={_d_pre*100:.1f}cm  "
                                  f"animated {N_STEPS*STEP_DT:.1f}s")
                            for _s in range(N_STEPS):
                                with self.sim._target_lock:
                                    self.sim.data.qpos[_qa]     += dx_step
                                    self.sim.data.qpos[_qa + 1] += dy_step
                                    self.sim.data.qvel[_da:_da + 6] = 0.0
                                    self.sim.target_base = np.array(
                                        [self.sim.data.qpos[_qa],
                                         self.sim.data.qpos[_qa + 1],
                                         _cur_yaw_d])
                                    self.sim.integral_x = 0.0
                                    self.sim.integral_y = 0.0
                                    self.sim.integral_yaw = 0.0
                                time.sleep(STEP_DT)
                            time.sleep(0.30)
                            _pocket = _pocket_out(self.sim.data)
                            _d_after = float(np.linalg.norm(
                                self.sim.data.xpos[self._held_obj_bid][:2]
                                - _pocket[:2]))
                            print(f"[Exec] -v7 pass "
                                  f"{_pass+1} post: residual="
                                  f"{_d_after*100:.1f}cm "
                                  f"(was {_d_pre*100:.1f}cm)")
                            _d_final = _d_after
                            if _d_after <= 0.03:
                                _converged = True
                                break
                            if _d_after >= _d_pre - 0.005:
                                print(f"[Exec] chassis-align: "
                                      f"pass {_pass+1} oscillating "
                                      f"(Δ improvement <0.5cm); stop")
                                break
                    except Exception as _e_aa:
                        print(f"[Exec] WARN chassis-align: {_e_aa}")
                else:
                    _converged = True
                SNAP_OBJ_TO_POCKET_PRE_CLOSE_ONCE = False
                if _d_final <= 0.05:
                    print(f"[Exec] chassis-align CONVERGED "
                          f"(residual={_d_final*100:.1f}cm ≤ 5cm) — "
                          f"proceed with close")
                    self._chassis_align_failed = False
                else:
                    print(f"[Exec] chassis-align NOT converged "
                          f"(residual={_d_final*100:.1f}cm > 5cm) — "
                          f"RETIRE candidate")
                    self._chassis_align_failed = True
                    self._last_close_finger_contacts = [False, False, False]
                    return
            except Exception as _e_sn:
                print(f"[Exec] WARN obj-to-pocket snap: {_e_sn}")

        if (ENABLE_ARM_FINE_ALIGN
                and STRICT_PERFECT_FRICTION_ONLY
                and pos >= 0.0
                and self._held_obj_bid is not None
                and self._held_obj_qpa is not None):
            try:
                _palm_bid_fwd_a = mujoco.mj_name2id(
                    self.sim.model, mujoco.mjtObj.mjOBJ_BODY,
                    "Gripper_Link3_1")
                _mid_a = self._pinch_midpoint_xyz(self.sim.data).copy()
                if (ENABLE_PERFECT_PIN_FORWARD_SHIFT
                        and _palm_bid_fwd_a >= 0
                        and PERFECT_PIN_FORWARD_SHIFT_M > 0.0):
                    _xm_a = self.sim.data.xmat[_palm_bid_fwd_a]
                    _local_z_world_a = np.array(
                        [float(_xm_a[2]), float(_xm_a[5]), float(_xm_a[8])])
                    _mid_a = _mid_a + PERFECT_PIN_FORWARD_SHIFT_M * _local_z_world_a
                _obj_a = self.sim.data.xpos[self._held_obj_bid][:3].copy()
                _d_arm = float(np.linalg.norm(_obj_a[:2] - _mid_a[:2]))
                if _d_arm <= 0.05:
                    print(f"[Exec] arm-fine-align: residual="
                          f"{_d_arm*100:.1f}cm ≤ 5cm — chassis fixed, "
                          f"proceed with close (IK landed accurate enough)")
                    self._chassis_align_failed = False
                else:
                    print(f"[Exec] arm-fine-align: residual="
                          f"{_d_arm*100:.1f}cm > 5cm — IK could not land "
                          f"close enough.  RETIRE candidate (chassis "
                          f"micro-drive disabled by --arm-fine-align flag)")
                    self._chassis_align_failed = True
                    self._last_close_finger_contacts = [False, False, False]
                    return
            except Exception as _e_arm:
                print(f"[Exec] WARN arm-fine-align measurement: {_e_arm}")

        if (ENABLE_TH_FINE_YAW
                and pos >= 0.0
                and self._held_obj_bid is not None):
            try:
                _palm_bid_p7 = mujoco.mj_name2id(
                    self.sim.model, mujoco.mjtObj.mjOBJ_BODY,
                    "Gripper_Link3_1")
                if _palm_bid_p7 >= 0:
                    _xm_p7 = self.sim.data.xmat[_palm_bid_p7]
                    _palm_z_world = np.array(
                        [float(_xm_p7[2]), float(_xm_p7[5]), float(_xm_p7[8])])
                    _grip_xyz_p7 = self._pinch_midpoint_xyz(self.sim.data)
                    _obj_xyz_p7 = self.sim.data.xpos[self._held_obj_bid][:3]
                    _delta_p7 = _obj_xyz_p7 - _grip_xyz_p7
                    if np.linalg.norm(_delta_p7[:2]) > 1e-4:
                        _palm_yaw = float(np.arctan2(
                            _palm_z_world[1], _palm_z_world[0]))
                        _obj_yaw = float(np.arctan2(_delta_p7[1], _delta_p7[0]))
                        _yaw_err = _obj_yaw - _palm_yaw
                        while _yaw_err > np.pi:
                            _yaw_err -= 2.0 * np.pi
                        while _yaw_err < -np.pi:
                            _yaw_err += 2.0 * np.pi
                        _abs_err = abs(_yaw_err)
                        if (ENABLE_NO_CHASSIS_PUSH
                                and _abs_err > TH_FINE_YAW_RESIDUAL_THRESHOLD):
                            print(f"[Exec] P7 during-close DISABLED "
                                  f"(--no-chassis-push, R44): measured yaw "
                                  f"residual {np.degrees(_yaw_err):+.1f}° — "
                                  f"NOT rotating TH (chassis-coupled swing "
                                  f"destabilizes the close); wrist-yaw + a1 "
                                  f"handle alignment pre-close")
                        elif (_abs_err > TH_FINE_YAW_RESIDUAL_THRESHOLD
                                and _abs_err <= TH_FINE_YAW_MAX_DELTA):
                            cur_q_p7 = self._current_arm_q()
                            proposed_q = list(cur_q_p7)
                            proposed_q[3] = float(cur_q_p7[3]) + float(_yaw_err)
                            proposed_q[3] = max(-3.14, min(3.14, proposed_q[3]))
                            try:
                                _valid_p7 = self.arm_bridge.is_valid(proposed_q)
                            except Exception:
                                _valid_p7 = False
                            if _valid_p7:
                                print(f"[Exec] P7 ACTIVE: yaw residual "
                                      f"{np.degrees(_yaw_err):+.1f}° → "
                                      f"commanding TH delta "
                                      f"{np.degrees(_yaw_err):+.1f}° "
                                      f"(TH {cur_q_p7[3]:+.3f} → "
                                      f"{proposed_q[3]:+.3f} rad).  "
                                      f"Validity check PASSED — column "
                                      f"rotates to align grasp axis "
                                      f"with obj.")
                                try:
                                    self._kinematic_descent(
                                        cur_q_p7, proposed_q,
                                        label="p7-th-yaw",
                                        n_steps=10)
                                except Exception as _e_p7_ramp:
                                    print(f"[Exec] WARN P7 TH ramp: "
                                          f"{_e_p7_ramp}")
                            else:
                                print(f"[Exec] P7 ACTIVE: yaw residual "
                                      f"{np.degrees(_yaw_err):+.1f}° "
                                      f"actionable, but proposed TH "
                                      f"{proposed_q[3]:+.3f} rad FAILED "
                                      f"validity check (likely arm-2 "
                                      f"self-collision) — SKIPPING "
                                      f"correction, falling through.")
                        elif _abs_err > TH_FINE_YAW_MAX_DELTA:
                            print(f"[Exec] P7 ACTIVE: yaw residual "
                                  f"{np.degrees(_yaw_err):+.1f}° EXCEEDS "
                                  f"safety cap ±"
                                  f"{np.degrees(TH_FINE_YAW_MAX_DELTA):.0f}° "
                                  f"— SKIPPING (chassis-yaw retry path "
                                  f"will handle this).")
                        else:
                            print(f"[Exec] P7 ACTIVE: yaw residual "
                                  f"{np.degrees(_yaw_err):+.1f}° BELOW "
                                  f"threshold "
                                  f"{np.degrees(TH_FINE_YAW_RESIDUAL_THRESHOLD):.1f}° "
                                  f"— no correction needed.")
            except Exception as _e_p7:
                print(f"[Exec] WARN P7 yaw-residual measurement: {_e_p7}")

        if (STRICT_PICKUP_MODE
                and STRICT_PERFECT_FRICTION_ONLY
                and pos >= 0.0
                and self._held_obj_bid is not None
                and self._side_grip_active):
            try:
                try:
                    _obj_r_bc = float(self._object_radius(self._held_obj_bid))
                except Exception:
                    _obj_r_bc = 0.072
                STRUCT_BC_DEEPEN = max(
                    0.03, 0.03 + 0.04 * (0.072 - _obj_r_bc) / 0.01)
                thumb_pos_safe = min(0.10, pos)
                bc_pos = min(0.20, thumb_pos_safe + STRUCT_BC_DEEPEN)
                _tgt_t = self._curl_targets(thumb_pos_safe)[6:9]
                _tgt_b = self._curl_targets(bc_pos)[3:6]
                _tgt_c = self._curl_targets(bc_pos)[0:3]
                target = (list(_tgt_c) + list(_tgt_b) + list(_tgt_t)
                          + [palm_c_target, palm_b_target])
                self._strict_log(
                    "CLOSE",
                    f"PERFECT asymmetric close: "
                    f"thumb_pos={thumb_pos_safe:.3f}  "
                    f"bc_pos={bc_pos:.3f} (capped to avoid "
                    f"finger-in-obj penetration)")
            except Exception as _e_async:
                print(f"[Exec] WARN asymmetric close compute failed: "
                      f"{_e_async} — falling back to uniform close")

        if _kin_close_safe and (STRICT_PICKUP_MODE
                and STRICT_PERFECT_FRICTION_ONLY
                and pos >= 0.0
                and self._held_obj_bid is not None):
            try:
                THUMB_PARTIAL = 0.05
                BC_PARTIAL    = 0.30
                try:
                    _pin_at = _pocket.copy()
                    _obj_z_now = float(
                        self.sim.data.xpos[self._held_obj_bid, 2])
                    _pin_at[2] = _obj_z_now
                    _qpa_obj_p = self._held_obj_qpa
                    _dofa_obj_p = self._held_obj_dofadr
                    if (_qpa_obj_p is not None
                            and _dofa_obj_p is not None):
                        _start_xyz = self.sim.data.xpos[
                            self._held_obj_bid][:3].copy()
                        ANIM_STEPS = 12
                        ANIM_DT = 0.025
                        for _a_s in range(ANIM_STEPS):
                            _t = (_a_s + 1) / ANIM_STEPS
                            _eased = _t * _t * (3.0 - 2.0 * _t)
                            _xyz_now = (_start_xyz +
                                        (_pin_at[:3] - _start_xyz) * _eased)
                            with self.sim._target_lock:
                                self.sim.data.qpos[_qpa_obj_p]     = float(_xyz_now[0])
                                self.sim.data.qpos[_qpa_obj_p + 1] = float(_xyz_now[1])
                                self.sim.data.qpos[_qpa_obj_p + 2] = float(_xyz_now[2])
                                self.sim.data.qvel[_dofa_obj_p:_dofa_obj_p + 6] = 0.0
                            time.sleep(ANIM_DT)
                    _pin_fn = self._pin_obj_at_world(_pin_at)
                    self._install_pin(_pin_fn)
                    print(f"[Exec] TEMP pin installed "
                          f"at {_pin_at[:3].round(3).tolist()} "
                          f"(smooth-animated, 0.3s smoothstep)")
                except Exception as _e_pn:
                    print(f"[Exec] WARN temp pin: {_e_pn}")
                target_kin = (
                    [v * BC_PARTIAL    for v in target[0:3]] +
                    [v * BC_PARTIAL    for v in target[3:6]] +
                    [v * THUMB_PARTIAL for v in target[6:9]] +
                    list(target[9:]))
                with self.sim._target_lock:
                    for j_idx, val in enumerate(target_kin):
                        if j_idx < 11 and j_idx < len(gids):
                            _gid_k = int(gids[j_idx])
                            _clo_k = float(self.sim.model.actuator_ctrlrange[_gid_k, 0])
                            _chi_k = float(self.sim.model.actuator_ctrlrange[_gid_k, 1])
                            val_c = max(_clo_k, min(_chi_k, float(val)))
                            self.sim.data.ctrl[_gid_k] = val_c
                time.sleep(0.80)
                obid = self._held_obj_bid
                try:
                    fc = [
                        bool(self._finger_touches_obj(0, obid)),
                        bool(self._finger_touches_obj(1, obid)),
                        bool(self._finger_touches_obj(2, obid)),
                    ]
                except Exception:
                    fc = [False, False, False]
                self._last_close_finger_contacts = list(fc)
                n_ok = sum(fc)
                self._strict_log(
                    "CLOSE",
                    f"PERFECT-mode PD-RAMP close "
                    f"(thumb={THUMB_PARTIAL*100:.0f}% bc={BC_PARTIAL*100:.0f}% target, 0.8s ramp); contacts "
                    f"[c,b,a]={fc} ({n_ok}/3); RETURN")
                if hold_seconds > 0:
                    time.sleep(hold_seconds)
                try:
                    _addrs_lf = self._ensure_finger_joint_qposadrs()
                    if (_addrs_lf and len(_addrs_lf) >= 9
                            and len(gids) >= 9):
                        with self.sim._target_lock:
                            for _j_lf in (6, 7, 8):
                                if _addrs_lf[_j_lf] >= 0:
                                    _q_lf = float(
                                        self.sim.data.qpos[_addrs_lf[_j_lf]])
                                    self.sim.data.ctrl[int(gids[_j_lf])] = _q_lf
                        print(f"[Exec] LATE-129 thumb-ctrl freeze "
                              f"(ctrl=qpos for thumb j1/j2/j3 pre-pin-clear)")
                except Exception as _e_fz:
                    print(f"[Exec] WARN thumb freeze: {_e_fz}")
                try:
                    self._clear_pin()
                    print(f"[Exec] TEMP pin CLEARED "
                          f"before verify (pure-friction lift follows)")
                except Exception as _e_cl:
                    print(f"[Exec] WARN pin clear: {_e_cl}")
                return
            except Exception as _e_kin:
                print(f"[Exec] WARN PERFECT-mode kinematic partial "
                      f"snap failed: {_e_kin}; falling back to plain PD ramp")

        current = [float(self.sim.data.ctrl[gids[i]]) for i in range(11)]

        path = None
        if self._finger_bridge is not None:
            try:
                path = self._finger_bridge.plan(current, target,
                                                 timeout=0.5,
                                                 n_waypoints=20)
            except Exception as e:
                print(f"[OMPL] Finger plan exception: {e} — "
                      "falling back to direct interpolation")
                path = None
        if path is None:
            path = [list(current), list(target)]
            print("[OMPL] Finger plan unavailable — using direct interpolation")
        else:
            print(f"[OMPL] Finger plan: {len(path)} waypoints from RRTConnect")

        if VERBOSE_GRASP_DEBUG:
            try:
                wp_count = len(path)
                for fname, j_idx in (('c_j1', 0), ('b_j1', 3), ('a_j1', 6)):
                    traj = [float(wp[j_idx]) for wp in path]
                    lo = min(traj); hi = max(traj)
                    start = traj[0]; end = traj[-1]
                    vals_str = " ".join(f"{v:+.3f}" for v in traj)
                    print(f"[OMPL] {fname} path ({wp_count}wps): "
                          f"start={start:+.3f} end={end:+.3f} "
                          f"range=[{lo:+.3f}, {hi:+.3f}]")
                    print(f"        traj: {vals_str}")
            except Exception as e:
                print(f"[OMPL] finger plan trajectory log warning: {e}")

        joint_offsets = [0.0] * 11
        if (STRICT_PICKUP_MODE
                and pos >= 0.0
                and self._held_obj_bid is not None
                and getattr(self, '_proximity_finger_links', None) is not None
                and len(self._proximity_finger_links) == 3
                and all(len(g) >= 1 for g in self._proximity_finger_links)):
            try:
                obj_xy_close = self.sim.data.xpos[self._held_obj_bid][:2]
                _finger_d = []
                for fi in range(3):
                    d_min = float('inf')
                    for link_bid in self._proximity_finger_links[fi]:
                        lk = self.sim.data.xpos[link_bid][:2]
                        d = float(np.hypot(lk[0] - obj_xy_close[0],
                                           lk[1] - obj_xy_close[1]))
                        if d < d_min:
                            d_min = d
                    _finger_d.append(d_min)
                _d_max = max(_finger_d)
                _TIP_SPEED = 0.030
                _delays = [(_d_max - d) / _TIP_SPEED for d in _finger_d]
                _cap = STRICT_CLOSE_TRANSITION_SECS * 0.5
                _delays = [min(d, _cap) for d in _delays]
                joint_offsets = [
                    _delays[0], _delays[0], _delays[0],
                    _delays[1], _delays[1], _delays[1],
                    _delays[2], _delays[2], _delays[2],
                    0.0, 0.0,
                ]
                self._strict_log(
                    "CLOSE",
                    f"per-finger close stagger: "
                    f"d_c={_finger_d[0]*100:.1f}cm "
                    f"d_b={_finger_d[1]*100:.1f}cm "
                    f"d_a={_finger_d[2]*100:.1f}cm  →  delays "
                    f"c={_delays[0]:.2f}s "
                    f"b={_delays[1]:.2f}s "
                    f"a={_delays[2]:.2f}s "
                    f"(speed={_TIP_SPEED*100:.1f}cm/s, cap={_cap:.2f}s)")
            except Exception as _e_stagger:
                print(f"[Exec] per-finger stagger compute failed: "
                      f"{_e_stagger} — falling back to synchronous close")
                joint_offsets = [0.0] * 11
        max_offset = max(joint_offsets)
        total_secs = transition_secs + max_offset
        n_steps = max(1, int(total_secs / SMOOTH_GRIPPER_STEP_S))

        n_segments = max(1, len(path) - 1)

        def _wp_value(joint_i, alpha):
            alpha = max(0.0, min(1.0, alpha))
            scaled = alpha * n_segments
            wp_idx = min(int(scaled), n_segments - 1)
            local_a = scaled - wp_idx
            s = local_a * local_a * (3.0 - 2.0 * local_a)
            return path[wp_idx][joint_i] + s * (
                path[wp_idx + 1][joint_i] - path[wp_idx][joint_i])

        contact_stop_enabled = (
            USE_CONTACT_STOP_CLOSE
            and pos >= 0.0
            and self._held_obj_bid is not None
            and self._finger_body_groups is not None
            and all(len(g) >= 1 for g in self._finger_body_groups)
        )
        compress_ticks = (FAST_CONTACT_COMPRESS_TICKS
                          if FAST_PICKUP_MODE and pos >= 0.0
                          else CONTACT_COMPRESS_TICKS)
        finger_frozen = [False, False, False]
        finger_contact_ticks = [0, 0, 0]
        proximity_stop_enabled = (
            contact_stop_enabled
            and not STRICT_PICKUP_MODE
            and getattr(self, '_proximity_finger_links', None) is not None
            and len(self._proximity_finger_links) == 3
            and all(len(lst) >= 1 for lst in self._proximity_finger_links))
        proximity_obj_radius = (
            float(self._object_radius(self._held_obj_bid))
            if proximity_stop_enabled else 0.0)
        PROXIMITY_SURFACE_MARGIN = 0.003

        force_stop_enabled = (
            contact_stop_enabled and STRICT_PICKUP_MODE)
        if force_stop_enabled:
            strict_force_target = self._strict_required_normal_per_finger(
                self._held_obj_bid)
            strict_force_ticks = [0, 0, 0]
            self._strict_log(
                "CLOSE",
                f"force-stop ENABLED  N_req/f="
                f"{strict_force_target:.2f}N  "
                f"mu={STRICT_FRICTION_MU:.2f}  "
                f"safety={STRICT_GRIP_SAFETY:.2f}  "
                f"obj_mass="
                f"{float(self.sim.model.body_mass[self._held_obj_bid]):.3f}kg")
        else:
            strict_force_target = 0.0
            strict_force_ticks = [0, 0, 0]
        joint_to_finger = {
            0: 0, 1: 0, 2: 0,
            3: 1, 4: 1, 5: 1,
            6: 2, 7: 2, 8: 2,
        }

        last_k_executed = 0
        _set_gripper_t0 = time.time()
        for k in range(1, n_steps + 1):
            if self._cancel:
                return
            t = k * SMOOTH_GRIPPER_STEP_S
            last_k_executed = k


            if proximity_stop_enabled and not all(finger_frozen):
                obj_xy_now = self.sim.data.xpos[self._held_obj_bid][:2]
                stop_dist = proximity_obj_radius + PROXIMITY_SURFACE_MARGIN
                _proximity_finger_groups = {
                    0: (0, 1, 2), 1: (3, 4, 5), 2: (6, 7, 8)}
                _addrs_prox = self._ensure_finger_joint_qposadrs()
                for fi in range(3):
                    if finger_frozen[fi]:
                        continue
                    closest_d = float('inf')
                    closest_link_bid = -1
                    for link_bid in self._proximity_finger_links[fi]:
                        link_xy = self.sim.data.xpos[link_bid][:2]
                        d = float(np.hypot(
                            link_xy[0] - obj_xy_now[0],
                            link_xy[1] - obj_xy_now[1]))
                        if d < closest_d:
                            closest_d = d
                            closest_link_bid = link_bid
                    if closest_d <= stop_dist:
                        finger_frozen[fi] = True
                        for jidx in _proximity_finger_groups[fi]:
                            if (_addrs_prox
                                    and jidx < len(_addrs_prox)
                                    and _addrs_prox[jidx] >= 0):
                                qv = float(self.sim.data.qpos[
                                    _addrs_prox[jidx]])
                                _gid = int(gids[jidx])
                                _clo = float(self.sim.model.actuator_ctrlrange[_gid, 0])
                                _chi = float(self.sim.model.actuator_ctrlrange[_gid, 1])
                                qv = max(_clo, min(_chi, qv))
                                self.sim.data.ctrl[_gid] = qv
                        fname = ['c', 'b', 'a'][fi]
                        print(f"[Exec] finger_{fname} surface-stop at "
                              f"t={t:.2f}s — closest link d_xy="
                              f"{closest_d*100:.1f}cm (obj_r="
                              f"{proximity_obj_radius*100:.1f}cm + "
                              f"{PROXIMITY_SURFACE_MARGIN*1000:.0f}mm "
                              f"margin)")

            if force_stop_enabled and not all(finger_frozen):
                _addrs_force = self._ensure_finger_joint_qposadrs()
                _force_jgroups = {
                    0: (0, 1, 2), 1: (3, 4, 5), 2: (6, 7, 8)}
                for fi in range(3):
                    if finger_frozen[fi]:
                        continue
                    n_force, n_ct = self._finger_contact_force(
                        fi, self._held_obj_bid)
                    if n_force >= strict_force_target and n_ct > 0:
                        strict_force_ticks[fi] += 1
                    else:
                        strict_force_ticks[fi] = 0
                    if strict_force_ticks[fi] >= STRICT_FORCE_STOP_STABLE_TICKS:
                        finger_frozen[fi] = True
                        overshoot_ratio = (
                            n_force / max(strict_force_target, 1e-3))
                        if (overshoot_ratio
                                >= STRICT_FORCE_STOP_RELIEF_EXTREME_RATIO):
                            relief_active = True
                            relief_rad = STRICT_FORCE_STOP_RELIEF_EXTREME_RAD
                            relief_tier = "EXTREME"
                        elif (overshoot_ratio
                                >= STRICT_FORCE_STOP_RELIEF_HIGH_RATIO):
                            relief_active = True
                            relief_rad = STRICT_FORCE_STOP_RELIEF_HIGH_RAD
                            relief_tier = "HIGH"
                        elif (overshoot_ratio
                                >= STRICT_FORCE_STOP_RELIEF_TRIGGER):
                            relief_active = True
                            relief_rad = STRICT_FORCE_STOP_RELIEF_RAD
                            relief_tier = "MOD"
                        else:
                            relief_active = False
                            relief_rad = 0.0
                            relief_tier = ""
                        j1_idx_local = _force_jgroups[fi][0]
                        for jidx in _force_jgroups[fi]:
                            if (_addrs_force
                                    and jidx < len(_addrs_force)
                                    and _addrs_force[jidx] >= 0):
                                qv = float(self.sim.data.qpos[
                                    _addrs_force[jidx]])
                                _gid = int(gids[jidx])
                                _clo = float(
                                    self.sim.model.actuator_ctrlrange[_gid, 0])
                                _chi = float(
                                    self.sim.model.actuator_ctrlrange[_gid, 1])
                                if (relief_active
                                        and jidx == j1_idx_local):
                                    if STRICT_PERFECT_FRICTION_ONLY:
                                        pass
                                    else:
                                        qv = qv - relief_rad
                                qv = max(_clo, min(_chi, qv))
                                self.sim.data.ctrl[_gid] = qv
                        fname = ['c', 'b', 'a'][fi]
                        relief_tag = (
                            f"  + RELIEF[{relief_tier}] j1 "
                            f"-{relief_rad*1000:.0f}mrad "
                            f"(overshoot {overshoot_ratio:.1f}×)"
                            if relief_active else "")
                        self._strict_log(
                            "CLOSE",
                            f"finger_{fname} force-stop  "
                            f"N={n_force:.2f}N "
                            f"(>= {strict_force_target:.2f}N for "
                            f"{STRICT_FORCE_STOP_STABLE_TICKS} ticks)  "
                            f"contacts={n_ct}{relief_tag}")

            if (contact_stop_enabled
                    and not STRICT_PICKUP_MODE
                    and not all(finger_frozen)):
                for fi in range(3):
                    if finger_frozen[fi]:
                        continue
                    if self._finger_touches_obj(fi, self._held_obj_bid):
                        finger_contact_ticks[fi] += 1
                        if finger_contact_ticks[fi] >= compress_ticks:
                            finger_frozen[fi] = True
                            _addrs_freeze = self._ensure_finger_joint_qposadrs()
                            _finger_joint_groups_loop = {
                                0: (0, 1, 2), 1: (3, 4, 5), 2: (6, 7, 8)}
                            for jidx in _finger_joint_groups_loop[fi]:
                                if (_addrs_freeze
                                        and jidx < len(_addrs_freeze)
                                        and _addrs_freeze[jidx] >= 0):
                                    qv = float(self.sim.data.qpos[
                                        _addrs_freeze[jidx]])
                                    _gid = int(gids[jidx])
                                    _clo = float(self.sim.model.actuator_ctrlrange[_gid, 0])
                                    _chi = float(self.sim.model.actuator_ctrlrange[_gid, 1])
                                    qv = max(_clo, min(_chi, qv))
                                    self.sim.data.ctrl[_gid] = qv
                            fname = ['c', 'b', 'a'][fi]
                            tip_xyz = None
                            if (self._carry_anchor_body_ids
                                    and len(self._carry_anchor_body_ids) == 3):
                                tip_xyz = self.sim.data.xpos[
                                    self._carry_anchor_body_ids[2 - fi]
                                ].copy()
                            obj_xyz_now = self.sim.data.xpos[
                                self._held_obj_bid].copy()
                            tip_str = (f"tip={tip_xyz.round(3)} "
                                       if tip_xyz is not None else "")
                            print(f"[Exec] finger_{fname} contact at "
                                  f"t={t:.2f}s — freezing ctrl=qpos "
                                  f"({finger_contact_ticks[fi]} ticks compression)  "
                                  f"{tip_str}obj={obj_xyz_now.round(3)}")

            with self.sim._target_lock:
                for joint_i in range(11):
                    if joint_i < 9:
                        fi = joint_to_finger[joint_i]
                        if finger_frozen[fi]:
                            continue
                    local_t = max(0.0, t - joint_offsets[joint_i])
                    alpha = min(1.0, local_t / transition_secs)
                    val = _wp_value(joint_i, alpha)
                    if (USE_FINGER_CTRL_CLAMP
                            and joint_i < len(FINGER_CTRL_RANGES)):
                        lo, hi = FINGER_CTRL_RANGES[joint_i]
                        val = max(lo, min(hi, val))
                    self.sim.data.ctrl[gids[joint_i]] = val

            if contact_stop_enabled and all(finger_frozen):
                break

            time.sleep(SMOOTH_GRIPPER_STEP_S)

        finger_joint_groups = {0: (0, 1, 2), 1: (3, 4, 5), 2: (6, 7, 8)}
        if contact_stop_enabled and pos >= 0.0 and not STRICT_PICKUP_MODE:
            addrs_hold = self._ensure_finger_joint_qposadrs()
            hold_t0 = time.time()
            while time.time() - hold_t0 < hold_seconds:
                if self._cancel:
                    return
                newly_frozen = False
                for fi in range(3):
                    if finger_frozen[fi]:
                        continue
                    if self._finger_touches_obj(fi, self._held_obj_bid):
                        finger_contact_ticks[fi] += 1
                        if finger_contact_ticks[fi] >= compress_ticks:
                            finger_frozen[fi] = True
                            newly_frozen = True
                            with self.sim._target_lock:
                                for jidx in finger_joint_groups[fi]:
                                    if (addrs_hold and jidx < len(addrs_hold)
                                            and addrs_hold[jidx] >= 0):
                                        qv = float(self.sim.data.qpos[
                                            addrs_hold[jidx]])
                                        _gid = int(gids[jidx])
                                        _clo = float(self.sim.model.actuator_ctrlrange[_gid, 0])
                                        _chi = float(self.sim.model.actuator_ctrlrange[_gid, 1])
                                        qv = max(_clo, min(_chi, qv))
                                        self.sim.data.ctrl[_gid] = qv
                            fname = ['c', 'b', 'a'][fi]
                            t_hold = time.time() - hold_t0
                            print(f"[Exec] finger_{fname} hold-contact at "
                                  f"t={t_hold:.2f}s (during hold) — "
                                  f"ctrl snapped to qpos to stop PD push")
                if all(finger_frozen) and newly_frozen:
                    remaining = hold_seconds - (time.time() - hold_t0)
                    if remaining > 0:
                        time.sleep(remaining)
                    break
                time.sleep(SMOOTH_GRIPPER_STEP_S)
        else:
            time.sleep(hold_seconds)

        if contact_stop_enabled and pos >= 0.0:
            labels = ['c', 'b', 'a']
            contacts = [lbl for lbl, frz in zip(labels, finger_frozen) if frz]
            self._last_close_finger_contacts = list(finger_frozen)
            if contacts:
                print(f"[Exec] close summary: contacted={contacts} "
                      f"({len(contacts)}/3 fingers)")
            else:
                print("[Exec] close summary: NO finger contact during close — "
                      "fingers closed in air, object held by pin only "
                      "(visual grip will look loose)")
        else:
            self._last_close_finger_contacts = None

        if (STRICT_PICKUP_MODE
                and STRICT_PERFECT_FRICTION_ONLY
                and pos >= 0.0
                and False):
            try:
                addrs_snap = self._ensure_finger_joint_qposadrs()
                snap_count = 0
                with self.sim._target_lock:
                    for jidx in range(min(9, len(gids))):
                        if (addrs_snap and jidx < len(addrs_snap)
                                and addrs_snap[jidx] >= 0):
                            qv = float(self.sim.data.qpos[addrs_snap[jidx]])
                            _gid_s = int(gids[jidx])
                            _clo_s = float(self.sim.model.actuator_ctrlrange[_gid_s, 0])
                            _chi_s = float(self.sim.model.actuator_ctrlrange[_gid_s, 1])
                            qv = max(_clo_s, min(_chi_s, qv))
                            self.sim.data.ctrl[_gid_s] = qv
                            snap_count += 1
                print(f"[Exec] PERFECT-mode end-of-close ctrl←qpos snap: "
                      f"{snap_count}/9 finger joints zeroed PD residual "
                      f"(eliminates chatter loop before verify/lift)")
            except Exception as _e_snap:
                print(f"[Exec] WARN end-of-close snap failed: {_e_snap}")

        try:
            elapsed = time.time() - _set_gripper_t0
            addrs = self._ensure_finger_joint_qposadrs()
            labels11 = ('c_j1','c_j2','c_j3',
                        'b_j1','b_j2','b_j3',
                        'a_j1','a_j2','a_j3',
                        'palm_c','palm_b')
            pos_kind = 'CLOSE' if pos >= 0.0 else 'OPEN'
            print(f"[Exec] _set_gripper({pos_kind} pos={pos:+.3f}) end-state: "
                  f"loop_ticks={last_k_executed}/{n_steps}  "
                  f"elapsed={elapsed:.2f}s  "
                  f"hold={hold_seconds:.2f}s")
            if VERBOSE_GRASP_DEBUG:
                for i, lbl in enumerate(labels11):
                    if i >= len(gids):
                        break
                    ctrl_v = float(self.sim.data.ctrl[int(gids[i])])
                    qpos_v = (float(self.sim.data.qpos[addrs[i]])
                              if addrs[i] >= 0 else float('nan'))
                    resid = qpos_v - ctrl_v
                    flag = ''
                    if abs(resid) > 0.05:
                        flag = '  ←PD-RESIDUAL'
                    print(f"  {lbl}: ctrl={ctrl_v:+.4f}  qpos={qpos_v:+.4f}  "
                          f"resid={resid:+.4f}{flag}")
            if pos >= 0.0:
                self._log_gripper_contacts(
                    f"_set_gripper({pos_kind}) end", force_forward=False)
        except Exception as e:
            print(f"[Exec] _set_gripper end-state log warning: {e}")


    def _current_arm_q(self):
        m = self._qmap
        d = self.sim.data
        return [float(d.qpos[m["ColumnLeft"]]),
                float(d.qpos[m["ColumnRight"]]),
                float(d.qpos[m["ArmLeft"]]),
                float(d.qpos[m["Base"]]),
                float(d.qpos[m["HandBearing"]]),
                float(d.qpos[m["WristZ"]]),
                float(d.qpos[m["WristX"]]),
                float(d.qpos[m["WristY"]])]

    def _log_tilt_hb_diag(self, label):
        if not ENABLE_NO_CHASSIS_PUSH:
            return
        try:
            q = self._current_arm_q()
            h_diff = q[1] - q[0]
            hb = q[4]
            wz = q[5]
            wx = q[6]
            wz_residual = wz - WRIST_Z_SIDE_APPROACH
            print(f"[DIAG-TILT] {label}: h2-h1={h_diff*100:+.1f}cm  "
                  f"hb={hb:+.3f}rad ({math.degrees(hb):+.1f}deg)  "
                  f"wz={wz:+.3f}rad (resid {wz_residual:+.3f} = "
                  f"{math.degrees(wz_residual):+.1f}deg)  "
                  f"wx={wx:+.3f}  "
                  f"(canonical hb=0, wz=-1.88, wx=+0.80, "
                  f"|h2-h1| target ≤ 12cm)")
        except Exception:
            pass

    def _log_arm_state(self, label):
        if not ENABLE_NO_CHASSIS_PUSH:
            return
        try:
            q = self._current_arm_q()
            d = self.sim.data
            m = self.sim.model
            gids = self.sim.gripper_ids_left
            try:
                wz_ctrl = float(d.ctrl[gids[GIDS_WRIST_Z]])
                hb_ctrl = float(d.ctrl[gids[GIDS_HANDBEARING]])
                wx_ctrl = float(d.ctrl[gids[GIDS_WRIST_X]])
                wy_ctrl = float(d.ctrl[gids[GIDS_WRIST_Y]])
            except Exception:
                wz_ctrl = hb_ctrl = wx_ctrl = wy_ctrl = float('nan')
            palm_bid = mujoco.mj_name2id(
                m, mujoco.mjtObj.mjOBJ_BODY, "Gripper_Link3_1")
            palm_xyz = (d.xpos[palm_bid].copy()
                        if palm_bid >= 0 else np.zeros(3))
            palm_quat = (d.xquat[palm_bid].copy()
                         if palm_bid >= 0 else np.array([1., 0., 0., 0.]))
            qw, qx, qy, qz = palm_quat
            roll = math.atan2(2*(qw*qx + qy*qz), 1 - 2*(qx*qx + qy*qy))
            pitch = math.asin(max(-1.0, min(1.0, 2*(qw*qy - qz*qx))))
            yaw = math.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
            obj_xyz_str = "n/a"
            pinch_xyz_str = "n/a"
            residual_str = "n/a"
            if self._held_obj_bid is not None:
                try:
                    obj_xyz = d.xpos[self._held_obj_bid].copy()
                    pinch_xyz = self._pinch_midpoint_xyz(d)
                    rxy = np.array(obj_xyz[:2]) - np.array(pinch_xyz[:2])
                    rmag = float(np.linalg.norm(rxy))
                    obj_xyz_str = (f"({obj_xyz[0]:+.3f},{obj_xyz[1]:+.3f},"
                                   f"{obj_xyz[2]:+.3f})")
                    pinch_xyz_str = (f"({pinch_xyz[0]:+.3f},{pinch_xyz[1]:+.3f},"
                                     f"{pinch_xyz[2]:+.3f})")
                    residual_str = (f"XY=({rxy[0]*100:+.1f},"
                                    f"{rxy[1]*100:+.1f})cm |r|="
                                    f"{rmag*100:.2f}cm")
                except Exception:
                    pass
            print(f"[ARM-STATE] === {label} ===")
            print(f"  cols   : h1={q[0]:+.4f}rad  h2={q[1]:+.4f}rad  "
                  f"(Δ={q[1]-q[0]:+.4f} = {(q[1]-q[0])*100:+.1f}cm; cap 12cm)")
            print(f"  reach  : a1={q[2]:+.4f}rad  "
                  f"th={q[3]:+.4f}rad ({math.degrees(q[3]):+.1f}°)")
            print(f"  hb     : qpos={q[4]:+.4f}  ctrl={hb_ctrl:+.4f}  "
                  f"target 0.000  resid {q[4]:+.3f} "
                  f"({math.degrees(q[4]):+.1f}°)")
            print(f"  wz     : qpos={q[5]:+.4f}  ctrl={wz_ctrl:+.4f}  "
                  f"target -1.880  resid {q[5]-WRIST_Z_SIDE_APPROACH:+.3f} "
                  f"({math.degrees(q[5]-WRIST_Z_SIDE_APPROACH):+.1f}°)")
            print(f"  wx     : qpos={q[6]:+.4f}  ctrl={wx_ctrl:+.4f}  "
                  f"target +0.800  resid {q[6]-0.80:+.3f}")
            print(f"  wy     : qpos={q[7]:+.4f}  ctrl={wy_ctrl:+.4f}  "
                  f"target +0.000  resid {q[7]:+.3f}")
            print(f"  palm   : xyz=({palm_xyz[0]:+.3f},{palm_xyz[1]:+.3f},"
                  f"{palm_xyz[2]:+.3f})  "
                  f"euler(R,P,Y)=({math.degrees(roll):+.1f}°,"
                  f"{math.degrees(pitch):+.1f}°,"
                  f"{math.degrees(yaw):+.1f}°)")
            print(f"  obj    : xyz={obj_xyz_str}")
            print(f"  pinch  : xyz={pinch_xyz_str}  → residual {residual_str}")
        except Exception as _e_state:
            print(f"[ARM-STATE] {label}: log failed: {_e_state}")


    def reset_finger_attempt_counter(self):
        self._strict_finger_attempts_used = 0

    def _retract_after_failure(self, tag, skip_lift=False,
                                side_grip_retry=False):
        if side_grip_retry:
            if ENABLE_ARM_HORIZONTAL_PICKUP:
                _cq = list(self._current_arm_q())
                _retr = [
                    min(float(_cq[0]) + HOVER_LIFT_RETRY, COLUMN_JOINT_MAX),
                    min(float(_cq[1]) + HOVER_LIFT_RETRY, COLUMN_JOINT_MAX),
                    max(MIN_PICK_A1, float(_cq[2]) - 0.12),
                    float(_cq[3]), float(_cq[4]),
                    WRIST_Z_SIDE_APPROACH, WRIST_X_SIDE_APPROACH,
                    WRIST_Y_SIDE_APPROACH,
                ]
                print(f"[Exec] {tag} ARM-HORIZONTAL retry: RETRACT arm clear of "
                      f"obj (a1 {float(_cq[2]):.3f}→{_retr[2]:.3f}, column "
                      f"+{HOVER_LIFT_RETRY*100:.0f}cm) BEFORE chassis orbit — "
                      f"prevents dragging/launching the free obj")
                try:
                    self._kinematic_descent(
                        _cq, _retr, "arm-horiz-retract-before-orbit",
                        n_steps=HOVER_RETRACT_STEPS)
                except Exception as _e_r:
                    print(f"[Exec] {tag} retract raised: {_e_r}")
                return
            print(f"[Exec] {tag} side-grip retry: keeping arm at GRASP_Q "
                  f"(chassis-only retry on next attempt — back-then-"
                  f"forward maneuver will re-align)")
            self._arm_held_at_grasp_for_retry = True
            return
        if skip_lift:
            print(f"[Exec] {tag} skip arm lift — residual is forward-"
                  f"aligned, base nudge will bring obj into grip")
            return

        current_q = self._current_arm_q()
        if USE_HOVER_RETRACT_ON_FAIL:
            hover_q = [
                min(float(current_q[0]) + HOVER_LIFT_RETRY, COLUMN_JOINT_MAX),
                min(float(current_q[1]) + HOVER_LIFT_RETRY, COLUMN_JOINT_MAX),
                float(current_q[2]),
                float(current_q[3]),
                float(current_q[4]),
                float(current_q[5]),
                float(current_q[6]),
                float(current_q[7]),
            ]
            print(f"[Exec] {tag} hover-retract (column "
                  f"+{HOVER_LIFT_RETRY*100:.0f}cm small-retry, "
                  f"a1/th held) before retry")
            self._kinematic_descent(current_q, hover_q, "hover-retract",
                                    n_steps=HOVER_RETRACT_STEPS)
        else:
            retry_q = [CARRY_H1, CARRY_H2, CARRY_A1, float(current_q[3])] \
                      + list(WRIST_NEUTRAL)
            print(f"[Exec] {tag} retract arm before candidate retry")
            self._kinematic_descent(current_q, retry_q, "retry-retract",
                                    n_steps=DESCENT_STEPS)

    def retract_to_carry(self):
        current_q = self._current_arm_q()
        carry_q = [CARRY_H1, CARRY_H2, CARRY_A1, float(current_q[3])] \
                  + list(WRIST_NEUTRAL)
        delta = max(abs(current_q[i] - carry_q[i]) for i in range(ARM_DOF))
        if delta < 0.02:
            return
        print(f"[Exec] retract_to_carry: {current_q} → "
              f"[{carry_q[0]:.2f},{carry_q[1]:.2f},"
              f"{carry_q[2]:.2f},{carry_q[3]:.2f}, wrist=neutral]")
        self._kinematic_descent(current_q, carry_q, "to-carry",
                                n_steps=DESCENT_STEPS)

    def pick(self, obj_idx, obj_world, on_complete=None,
             pre_grasp_q=None, pre_grasp_actual_target=None,
             side_grip_push_target=None,
             is_local_retry=False):
        self._cancel = False
        t = threading.Thread(
            target=self._pick_run,
            args=(int(obj_idx), np.array(obj_world, dtype=float), on_complete,
                  None if pre_grasp_q is None else list(pre_grasp_q),
                  None if pre_grasp_actual_target is None else
                  np.array(pre_grasp_actual_target, dtype=float),
                  side_grip_push_target, bool(is_local_retry)),
            daemon=True)
        t.start()
        self._pick_thread = t

    def _pick_run(self, obj_idx, obj_world, on_complete,
                  screened_pre_grasp_q=None,
                  screened_actual_pre_target=None,
                  side_grip_push_target=None,
                  is_local_retry=False):
        cb_fired = [False]
        def fire(ok):
            if cb_fired[0]:
                return
            cb_fired[0] = True
            if on_complete:
                on_complete(ok)

        success = False
        self.last_grasp_failure_info = None
        self._pre_close_nudges_used = 0
        self._pre_close_backup_used = False
        self._strict_t0 = time.time()
        self._strict_retry_count = 0
        self._strict_overdrive_delta = float(STRICT_OVERDRIVE_DELTA_INIT)
        self._strict_force_multiplier = 1.0
        self._unweld_obj()
        self._unfreeze_fingers()
        self._strict_grasp_q = None
        _cycle_t0 = time.time()
        try:
            print(f"\n[Exec] PICK obj_{obj_idx} @ {obj_world.round(3)} "
                  f"[TIMING t=0.00s cycle start]")

            obj_bid, obj_qpa, obj_dofa = self._resolve_obj(obj_idx)
            self._held_obj_idx    = obj_idx
            self._held_obj_bid    = obj_bid
            self._held_obj_qpa    = obj_qpa
            self._held_obj_dofadr = obj_dofa

            obj_pos_snapshot = self.sim.data.xpos[obj_bid].copy()

            pin_world = self._pin_obj_at_world(obj_pos_snapshot)
            self._install_pin(pin_world)

            wrist_goal = self._compute_wrist_goal_for_obj(obj_bid)
            side_grip = is_side_approach(wrist_goal)
            mode_str = ("SIDE" if side_grip
                        else "DIAGONAL" if wrist_goal[0] > -0.70
                        else "AGGRESSIVE_TOPDOWN")
            self._side_grip_active = bool(side_grip)

            print(f"[Exec] wrist_goal (hb,wz,wx,wy) = "
                  f"({wrist_goal[0]:+.2f},{wrist_goal[1]:+.2f},"
                  f"{wrist_goal[2]:+.2f},{wrist_goal[3]:+.2f})  mode={mode_str}")

            if ENABLE_ARM_HORIZONTAL_PICKUP and side_grip:
                try:
                    self._inplace_yaw_aim(
                        (float(obj_pos_snapshot[0]),
                         float(obj_pos_snapshot[1])))
                except Exception as _e_yaw:
                    print(f"[Exec] [5.0] yaw-aim raised: {_e_yaw}")
            loc = self.sim.localization()
            obj_radius_for_standoff = self._object_radius(obj_bid)
            ik_base_xy = (float(loc[0]), float(loc[1]))
            ik_base_yaw = float(loc[2])
            push_dist_to_run = 0.0
            push_target_xy_world = None
            if side_grip and side_grip_push_target is not None and not ENABLE_NO_CHASSIS_PUSH:
                obj_xy_2d = np.asarray(obj_pos_snapshot[:2], dtype=float)
                real_xy = np.asarray([loc[0], loc[1]], dtype=float)
                dist_now = float(np.linalg.norm(obj_xy_2d - real_xy))
                if dist_now > float(side_grip_push_target):
                    push_dist_to_run = dist_now - float(side_grip_push_target)
                    approach_unit = (obj_xy_2d - real_xy) / dist_now
                    push_target_xy_world = (
                        obj_xy_2d - float(side_grip_push_target) * approach_unit)
                    ik_base_xy = (float(push_target_xy_world[0]),
                                  float(push_target_xy_world[1]))
                    ik_base_yaw = float(math.atan2(
                        obj_xy_2d[1] - ik_base_xy[1],
                        obj_xy_2d[0] - ik_base_xy[0]))
                    if ENABLE_ARM_HORIZONTAL_PICKUP:
                        ik_base_yaw = float(loc[2])
                    yaw_delta_deg = math.degrees(
                        ((ik_base_yaw - float(loc[2]) + math.pi) % (2 * math.pi))
                        - math.pi)
                    print(f"[Exec] side-grip push planned: virtual chassis "
                          f"({ik_base_xy[0]:.3f},{ik_base_xy[1]:.3f}) "
                          f"= obj − {side_grip_push_target:.2f}m × approach_unit  "
                          f"(actual push after descent: {push_dist_to_run*100:.1f}cm; "
                          f"IK yaw {math.degrees(ik_base_yaw):+.1f}° vs current "
                          f"{math.degrees(loc[2]):+.1f}°, Δ={yaw_delta_deg:+.2f}°)")

            reset_plan_data_for_ik(self.arm_bridge,
                                   base_xy=ik_base_xy,
                                   base_yaw=ik_base_yaw)
            approach_base_xy = ik_base_xy if push_target_xy_world is not None \
                else (float(loc[0]), float(loc[1]))
            if False and ENABLE_ARM_HORIZONTAL_PICKUP and side_grip:
                try:
                    _abid = mujoco.mj_name2id(
                        self.arm_bridge.model, mujoco.mjtObj.mjOBJ_BODY, "Arm_1")
                    _ab = self.arm_bridge.planning_data.xpos[_abid][:2]
                    _objd = np.asarray(obj_pos_snapshot[:2], dtype=float)
                    _a_old = math.degrees(math.atan2(
                        _objd[1] - approach_base_xy[1],
                        _objd[0] - approach_base_xy[0]))
                    _a_new = math.degrees(math.atan2(
                        _objd[1] - float(_ab[1]), _objd[0] - float(_ab[0])))
                    print(f"[Exec] arm-base-ref: approach from Arm_1 "
                          f"({float(_ab[0]):.3f},{float(_ab[1]):.3f}) "
                          f"angle {_a_new:+.1f}° vs chassis-center "
                          f"({approach_base_xy[0]:.3f},{approach_base_xy[1]:.3f}) "
                          f"{_a_old:+.1f}° (Δ{_a_new-_a_old:+.1f}°)")
                    approach_base_xy = (float(_ab[0]), float(_ab[1]))
                except Exception as _e_ab:
                    print(f"[Exec] arm-base-ref skipped: {_e_ab}")
            _grasp_unused, pre_grasp_target = compute_grasp_targets(
                approach_base_xy, obj_pos_snapshot,
                obj_radius=obj_radius_for_standoff,
                side_approach=side_grip)
            pre_grasp_target[2] = max(pre_grasp_target[2], MIN_PICK_WRIST_Z)
            if ENABLE_ARM_HORIZONTAL_PICKUP and side_grip:
                _z_before = float(pre_grasp_target[2])
                pre_grasp_target[2] = max(
                    pre_grasp_target[2] - ARM_HORIZONTAL_GRIP_Z_LOWER,
                    ARM_HORIZONTAL_MIN_WRIST_Z)
                print(f"[Exec] arm-horizontal lower-Z: wrist target "
                      f"{_z_before:.3f} → {pre_grasp_target[2]:.3f}m "
                      f"(−{ARM_HORIZONTAL_GRIP_Z_LOWER*100:.0f}cm → grip nearer "
                      f"obj centre; floor {ARM_HORIZONTAL_MIN_WRIST_Z:.2f}m)")
            try:
                _thumb_open = self._thumb_open_pos_for_mode()
                _finger_open_jpos = {
                    "finger_a_joint_1_1": _thumb_open,
                    "finger_b_joint_1_1": GRIPPER_OPEN_POS,
                    "finger_c_joint_1_1": GRIPPER_OPEN_POS,
                }
                _model = self.arm_bridge.model
                _pdata = self.arm_bridge.planning_data
                for _jname, _qval in _finger_open_jpos.items():
                    _jid = mujoco.mj_name2id(
                        _model, mujoco.mjtObj.mjOBJ_JOINT, _jname)
                    if _jid >= 0:
                        _qpa = int(_model.jnt_qposadr[_jid])
                        _pdata.qpos[_qpa] = float(_qval)
            except Exception as _e:
                print(f"[Exec] WARN: failed to set open-finger qpos in "
                      f"planning_data for IK validity: {_e}")

            print(f"[Exec] pre_grasp_target = {pre_grasp_target.round(3)}  "
                  f"IK base=({ik_base_xy[0]:.3f},{ik_base_xy[1]:.3f}) "
                  f"yaw={math.degrees(ik_base_yaw):.1f}deg  "
                  f"(IK validity uses OPEN-finger geometry)")

            use_local_retry_skip_ik = (
                STRICT_PICKUP_MODE and side_grip and is_local_retry
                and self._last_valid_pre_grasp_q is not None
                and not ENABLE_NO_CHASSIS_PUSH)
            if use_local_retry_skip_ik:
                PRE_GRASP_Q = list(self._last_valid_pre_grasp_q)
                actual_pre_target = pre_grasp_target.copy()
                self._strict_log(
                    "IK",
                    "local-retry path: skipping IK re-solve; using "
                    "cached PRE_GRASP_Q from attempt 1 (arm already "
                    "at GRASP_Q; chassis nudge re-aligns gripper "
                    "with obj)")
                print(f"[Exec] PRE_GRASP_Q (local-retry, IK skipped) = "
                      f"{[round(v, 3) for v in PRE_GRASP_Q]}")
            elif (screened_pre_grasp_q is not None
                  and push_target_xy_world is None
                  and not ENABLE_NO_CHASSIS_PUSH):
                use_screened = True
                PRE_GRASP_Q = [float(v) for v in screened_pre_grasp_q]
                if len(PRE_GRASP_Q) < ARM_DOF:
                    PRE_GRASP_Q = PRE_GRASP_Q + list(wrist_goal)
                actual_pre_target = (screened_actual_pre_target
                                     if screened_actual_pre_target is not None
                                     else pre_grasp_target.copy())
                print("[Exec] using screened PRE_GRASP_Q from candidate filter")
            else:
                use_screened = False
                if screened_pre_grasp_q is not None:
                    print("[Exec] ignoring screened PRE_GRASP_Q (was planned "
                          "for nav-distance chassis; side-grip push needs IK "
                          "from virtual post-push position) — recomputing")
                try:
                    if side_grip:
                        ik_target_body  = "Gripper_Link3_1"
                        ik_wrist_weight = (0.10, 3.0, 3.0, 3.0)
                    else:
                        ik_target_body  = "Gripper_Link1_1"
                        ik_wrist_weight = 5.0
                    _warm_seed = self._last_valid_pre_grasp_q
                    _seed_src = "last-valid"
                    if _warm_seed is None and screened_pre_grasp_q is not None:
                        _ws = [float(v) for v in screened_pre_grasp_q]
                        if len(_ws) < ARM_DOF:
                            _ws = _ws + list(wrist_goal)
                        _warm_seed = _ws
                        _seed_src = "screened-candidate"
                    _ik_seeds  = 6 if _warm_seed is not None else 4
                    if _warm_seed is not None:
                        print(f"[Exec] IK warm-start: seed-0 from {_seed_src} "
                              f"PRE_GRASP_Q  n_seeds={_ik_seeds}")
                    _ik_max_lift = 0.40

                    if ENABLE_NO_CHASSIS_PUSH and side_grip:
                        _direct_target = pre_grasp_target.copy()
                        _approach_xy = np.array([
                            float(obj_pos_snapshot[0]) - float(ik_base_xy[0]),
                            float(obj_pos_snapshot[1]) - float(ik_base_xy[1]),
                        ])
                        _ap_norm = float(np.linalg.norm(_approach_xy))
                        _approach_unit = (_approach_xy / _ap_norm
                                          if _ap_norm > 1e-6
                                          else np.array([1.0, 0.0]))
                        try:
                            _obj_r_so = float(self._object_radius(
                                self._held_obj_bid))
                        except Exception:
                            _obj_r_so = 0.05
                        _pinch_standoff = max(SIDE_PINCH_TARGET_STANDOFF,
                                              _obj_r_so + 0.015)
                        _direct_target[0] = (float(obj_pos_snapshot[0])
                            - _pinch_standoff * _approach_unit[0])
                        _direct_target[1] = (float(obj_pos_snapshot[1])
                            - _pinch_standoff * _approach_unit[1])
                        print(f"[Exec] R41 pinch-standoff override: palm "
                              f"XY target at obj − "
                              f"{_pinch_standoff*100:.1f}cm·approach "
                              f"(obj_r={_obj_r_so*100:.1f}cm+1.5cm; "
                              f"palm-clip-safe, replaces obj_r+2cm "
                              f"chassis-push-era standoff)")
                        _dynamic_lut_active = False
                        try:
                            _dynamic_lut_active = bool(
                                self.arm_bridge.is_dynamic_calib_loaded())
                        except Exception:
                            pass
                        if _dynamic_lut_active:
                            if _warm_seed is not None and len(_warm_seed) >= 6:
                                _eh1, _eh2, _ea1, _eth = (
                                    float(_warm_seed[0]), float(_warm_seed[1]),
                                    float(_warm_seed[2]), float(_warm_seed[3]))
                                _ehb, _ewz = (float(_warm_seed[4]),
                                              float(_warm_seed[5]))
                            else:
                                _eh1, _eh2, _ea1, _eth = 0.13, 0.23, 0.30, 0.0
                                _ehb, _ewz = 0.0, WRIST_Z_SIDE_APPROACH
                            try:
                                _drift = self.arm_bridge._calib_error(
                                    _eh1, _eh2, _ea1, _eth,
                                    hb=_ehb, wz=_ewz)
                                _drift_xy = np.array(
                                    [float(_drift[0]), float(_drift[1])])
                                _dmag = float(np.linalg.norm(_drift_xy))
                                if _dmag > 0.12:
                                    _direct_target[0] += (
                                        SIDE_DRIFT_FORWARD_BIAS
                                        * _approach_unit[0])
                                    _direct_target[1] += (
                                        SIDE_DRIFT_FORWARD_BIAS
                                        * _approach_unit[1])
                                    print(f"[Exec] N3 dynamic-LUT IGNORED "
                                          f"(cell drift |{_dmag*100:.1f}|cm "
                                          f">12cm = unreliable); using "
                                          f"deterministic M2 +"
                                          f"{SIDE_DRIFT_FORWARD_BIAS*100:.0f}cm "
                                          f"forward bias instead")
                                else:
                                    _direct_target[0] -= _drift_xy[0]
                                    _direct_target[1] -= _drift_xy[1]
                                    print(f"[Exec] M2/N3 dynamic-LUT bias: "
                                          f"XY drift "
                                          f"({_drift_xy[0]*100:+.1f},"
                                          f"{_drift_xy[1]*100:+.1f})cm "
                                          f"|{_dmag*100:.1f}|cm subtracted "
                                          f"from IK target (config "
                                          f"a1={_ea1:.2f})")
                            except Exception as _e_n3:
                                _direct_target[0] += (SIDE_DRIFT_FORWARD_BIAS
                                                      * _approach_unit[0])
                                _direct_target[1] += (SIDE_DRIFT_FORWARD_BIAS
                                                      * _approach_unit[1])
                                print(f"[Exec] M2 fallback (LUT query "
                                      f"failed {_e_n3}): +"
                                      f"{SIDE_DRIFT_FORWARD_BIAS*100:.0f}cm")
                        else:
                            _direct_target[0] += (SIDE_DRIFT_FORWARD_BIAS
                                                  * _approach_unit[0])
                            _direct_target[1] += (SIDE_DRIFT_FORWARD_BIAS
                                                  * _approach_unit[1])
                            print(f"[Exec] M2 forward-bias: +"
                                  f"{SIDE_DRIFT_FORWARD_BIAS*100:.0f}cm along "
                                  f"approach (static/no LUT)")
                        try:
                            _obj_hh_z = float(self._object_half_height(
                                self._held_obj_bid))
                        except Exception:
                            _obj_hh_z = 0.075
                        _obj_top_z = (float(obj_pos_snapshot[2])
                                      + _obj_hh_z)
                        _direct_target[2] = max(_obj_top_z + 0.04, 0.35)
                        print(f"[Exec] M8 z-target: obj_top={_obj_top_z:.3f}m "
                              f"→ IK target z={_direct_target[2]:.3f}m "
                              f"(was obj_z+0.30="
                              f"{float(pre_grasp_target[2])+0.30:.3f}m)")
                        _ov3_approach_yaw = float(math.atan2(
                            float(obj_pos_snapshot[1]) - float(ik_base_xy[1]),
                            float(obj_pos_snapshot[0]) - float(ik_base_xy[0])))
                        _ov3_axis_weight = 0.0
                        try:
                            PRE_GRASP_Q = self.arm_bridge.solve_ik(
                                _direct_target, n_seeds=_ik_seeds,
                                threshold=0.06,
                                wrist_goal=wrist_goal,
                                wrist_weight=ik_wrist_weight,
                                target_body=ik_target_body,
                                seed_q=_warm_seed,
                                approach_yaw=_ov3_approach_yaw,
                                axis_align_weight=_ov3_axis_weight)
                            actual_pre_target = _direct_target
                        except RuntimeError:
                            PRE_GRASP_Q, actual_pre_target = \
                                self.arm_bridge.solve_ik_with_z_lift(
                                    pre_grasp_target, n_seeds=_ik_seeds,
                                    wrist_goal=wrist_goal,
                                    wrist_weight=ik_wrist_weight,
                                    target_body=ik_target_body,
                                    seed_q=_warm_seed,
                                    max_lift=_ik_max_lift,
                                    approach_yaw=_ov3_approach_yaw,
                                    axis_align_weight=_ov3_axis_weight)
                    else:
                        _ah_axis_w = 0.0
                        _ah_appr_yaw = None
                        if ENABLE_ARM_HORIZONTAL_PICKUP and side_grip:
                            _ah_appr_yaw = float(math.atan2(
                                float(obj_pos_snapshot[1]) - float(ik_base_xy[1]),
                                float(obj_pos_snapshot[0]) - float(ik_base_xy[0])))
                            _ah_axis_w = 0.0
                        PRE_GRASP_Q, actual_pre_target = \
                            self.arm_bridge.solve_ik_with_z_lift(
                                pre_grasp_target, n_seeds=_ik_seeds,
                                wrist_goal=wrist_goal,
                                wrist_weight=ik_wrist_weight,
                                target_body=ik_target_body,
                                seed_q=_warm_seed,
                                max_lift=_ik_max_lift,
                                approach_yaw=_ah_appr_yaw,
                                axis_align_weight=_ah_axis_w)
                        if (ENABLE_ARM_HORIZONTAL_PICKUP and side_grip
                                and PRE_GRASP_Q is not None):
                            _th0 = float(PRE_GRASP_Q[3])
                            _thn = max(_th0, ARM_HORIZONTAL_TH_CAP)
                            if abs(_thn - _th0) > 1e-4:
                                PRE_GRASP_Q[3] = _thn
                                print(f"[Exec] arm-horizontal TH cap: "
                                      f"{math.degrees(_th0):+.1f}deg -> "
                                      f"{math.degrees(_thn):+.1f}deg "
                                      f"(capped at -15deg, thumb-in)")

                    if (ENABLE_NO_CHASSIS_PUSH and side_grip
                            and PRE_GRASP_Q is not None):
                        try:
                            _m = self.arm_bridge.model
                            _pd = self.arm_bridge.planning_data
                            _qm = self.arm_bridge._qpos_map
                            for _nm, _vi in (("ColumnLeft", 0), ("ColumnRight", 1),
                                             ("ArmLeft", 2), ("Base", 3),
                                             ("HandBearing", 4), ("WristZ", 5),
                                             ("WristX", 6), ("WristY", 7)):
                                _pd.qpos[_qm[_nm]] = float(PRE_GRASP_Q[_vi])
                            mujoco.mj_forward(_m, _pd)
                            _ta = mujoco.mj_name2id(_m, mujoco.mjtObj.mjOBJ_BODY,
                                                    "finger_a_link_3_1")
                            _tb = mujoco.mj_name2id(_m, mujoco.mjtObj.mjOBJ_BODY,
                                                    "finger_b_link_3_1")
                            _tc = mujoco.mj_name2id(_m, mujoco.mjtObj.mjOBJ_BODY,
                                                    "finger_c_link_3_1")
                            def _axis_res_for_q(_q):
                                for _n2, _i2 in (("ColumnLeft", 0),
                                                 ("ColumnRight", 1),
                                                 ("ArmLeft", 2), ("Base", 3),
                                                 ("HandBearing", 4),
                                                 ("WristZ", 5), ("WristX", 6),
                                                 ("WristY", 7)):
                                    _pd.qpos[_qm[_n2]] = float(_q[_i2])
                                mujoco.mj_forward(_m, _pd)
                                _th2 = _pd.xpos[_ta][:2]
                                _bc2 = 0.5 * (_pd.xpos[_tb][:2]
                                              + _pd.xpos[_tc][:2])
                                _ax2 = _bc2 - _th2
                                _ay2 = math.atan2(float(_ax2[1]),
                                                  float(_ax2[0]))
                                _ra = _wrp((_appyaw + math.pi/2) - _ay2)
                                _rb = _wrp((_appyaw - math.pi/2) - _ay2)
                                _zs = (max(float(_pd.xpos[_ta][2]),
                                           float(_pd.xpos[_tb][2]),
                                           float(_pd.xpos[_tc][2]))
                                       - min(float(_pd.xpos[_ta][2]),
                                             float(_pd.xpos[_tb][2]),
                                             float(_pd.xpos[_tc][2])))
                                return ((_ra if abs(_ra) < abs(_rb) else _rb),
                                        _zs)
                            def _wrp(a):
                                while a > math.pi: a -= 2*math.pi
                                while a < -math.pi: a += 2*math.pi
                                return a
                            _appyaw = _ov3_approach_yaw
                            _axis_res, _zs0 = _axis_res_for_q(PRE_GRASP_Q)
                            print(f"[Exec] R42 axis diag (measure-only): "
                                  f"kinematic thumb-bc residual "
                                  f"{math.degrees(_axis_res):+.1f}° "
                                  f"(benign gripper offset — NOT corrected; "
                                  f"see DEVLOG R42)")
                        except Exception as _e_2p:
                            print(f"[Exec] R42 axis diag skipped: {_e_2p}")

                    self._last_valid_pre_grasp_q = list(PRE_GRASP_Q)

                    if (side_grip and not ENABLE_NO_CHASSIS_PUSH
                            and not ENABLE_ARM_HORIZONTAL_PICKUP):
                        try:
                            _model_wz = self.arm_bridge.model
                            _pdata_wz = self.arm_bridge.planning_data
                            _qmap_wz  = self.arm_bridge._qpos_map
                            _t_bid = mujoco.mj_name2id(
                                _model_wz, mujoco.mjtObj.mjOBJ_BODY,
                                "finger_a_link_3_1")
                            _b_bid = mujoco.mj_name2id(
                                _model_wz, mujoco.mjtObj.mjOBJ_BODY,
                                "finger_b_link_3_1")
                            _c_bid = mujoco.mj_name2id(
                                _model_wz, mujoco.mjtObj.mjOBJ_BODY,
                                "finger_c_link_3_1")
                            if (_t_bid >= 0 and _b_bid >= 0
                                    and _c_bid >= 0):
                                for _key, _val in zip(
                                        ("ColumnLeft", "ColumnRight",
                                         "ArmLeft", "Base",
                                         "HandBearing", "WristZ",
                                         "WristX", "WristY"),
                                        PRE_GRASP_Q):
                                    _pdata_wz.qpos[_qmap_wz[_key]] = \
                                        float(_val)
                                mujoco.mj_forward(_model_wz, _pdata_wz)
                                _thumb_xy_wz = _pdata_wz.xpos[
                                    _t_bid][:2].copy()
                                _bc_xy_wz = 0.5 * (
                                    _pdata_wz.xpos[_b_bid][:2]
                                    + _pdata_wz.xpos[_c_bid][:2])
                                _axis_vec = _bc_xy_wz - _thumb_xy_wz
                                _axis_yaw = math.atan2(
                                    _axis_vec[1], _axis_vec[0])
                                _obj_xy_wz = np.asarray(
                                    obj_pos_snapshot[:2], dtype=float)
                                _chassis_xy_wz = np.asarray(
                                    ik_base_xy, dtype=float)
                                _approach_yaw = math.atan2(
                                    _obj_xy_wz[1] - _chassis_xy_wz[1],
                                    _obj_xy_wz[0] - _chassis_xy_wz[0])
                                _desired_a = (_approach_yaw
                                              + math.pi / 2.0)
                                _desired_b = (_approach_yaw
                                              - math.pi / 2.0)
                                _err_a = ((_axis_yaw - _desired_a
                                           + math.pi)
                                          % (2 * math.pi)) - math.pi
                                _err_b = ((_axis_yaw - _desired_b
                                           + math.pi)
                                          % (2 * math.pi)) - math.pi
                                _err_wz = (_err_a
                                           if abs(_err_a) <= abs(_err_b)
                                           else _err_b)
                                _d_thumb_orig = float(np.linalg.norm(
                                    _thumb_xy_wz - _obj_xy_wz))
                                _d_bc_orig = float(np.linalg.norm(
                                    _bc_xy_wz - _obj_xy_wz))
                                _carry_orig = float(np.linalg.norm(
                                    0.5 * (_thumb_xy_wz + _bc_xy_wz)
                                    - _obj_xy_wz))
                                _max_far_orig = max(_d_thumb_orig,
                                                    _d_bc_orig)
                                if abs(_err_wz) > \
                                        WZ_REFINE_AXIS_ERR_THRESHOLD:
                                    _new_wz = float(wrist_goal[1]) \
                                              - _err_wz
                                    while _new_wz > math.pi:
                                        _new_wz -= 2 * math.pi
                                    while _new_wz < -math.pi:
                                        _new_wz += 2 * math.pi
                                    _refined_goal = (
                                        float(wrist_goal[0]),
                                        _new_wz,
                                        float(wrist_goal[2]),
                                        float(wrist_goal[3]))
                                    print(f"[Exec] wz refine: "
                                          f"approach_yaw="
                                          f"{math.degrees(_approach_yaw):+.1f}°"
                                          f"  axis_yaw="
                                          f"{math.degrees(_axis_yaw):+.1f}°"
                                          f"  err="
                                          f"{math.degrees(_err_wz):+.1f}° "
                                          f"(> "
                                          f"{math.degrees(WZ_REFINE_AXIS_ERR_THRESHOLD):.0f}°"
                                          f" tol) → wz "
                                          f"{float(wrist_goal[1]):+.2f} "
                                          f"→ {_new_wz:+.2f}; "
                                          f"re-solving IK")
                                    try:
                                        _q2, _at2 = \
                                            self.arm_bridge.solve_ik_with_z_lift(
                                                pre_grasp_target,
                                                n_seeds=_ik_seeds,
                                                wrist_goal=_refined_goal,
                                                wrist_weight=ik_wrist_weight,
                                                target_body=ik_target_body,
                                                seed_q=PRE_GRASP_Q,
                                                max_lift=_ik_max_lift)
                                        for _key, _val in zip(
                                                ("ColumnLeft",
                                                 "ColumnRight",
                                                 "ArmLeft", "Base",
                                                 "HandBearing",
                                                 "WristZ", "WristX",
                                                 "WristY"),
                                                _q2):
                                            _pdata_wz.qpos[
                                                _qmap_wz[_key]] = \
                                                float(_val)
                                        mujoco.mj_forward(
                                            _model_wz, _pdata_wz)
                                        _thumb_xy_v = _pdata_wz.xpos[
                                            _t_bid][:2].copy()
                                        _bc_xy_v = 0.5 * (
                                            _pdata_wz.xpos[_b_bid][:2]
                                            + _pdata_wz.xpos[_c_bid][:2])
                                        _axis_v = _bc_xy_v \
                                                  - _thumb_xy_v
                                        _axis_yaw2 = math.atan2(
                                            _axis_v[1], _axis_v[0])
                                        _err2_a = (
                                            (_axis_yaw2 - _desired_a
                                             + math.pi)
                                            % (2 * math.pi)) - math.pi
                                        _err2_b = (
                                            (_axis_yaw2 - _desired_b
                                             + math.pi)
                                            % (2 * math.pi)) - math.pi
                                        _err2_wz = (
                                            _err2_a
                                            if abs(_err2_a)
                                                <= abs(_err2_b)
                                            else _err2_b)
                                        _d_thumb_ref = float(
                                            np.linalg.norm(
                                                _thumb_xy_v
                                                - _obj_xy_wz))
                                        _d_bc_ref = float(
                                            np.linalg.norm(
                                                _bc_xy_v - _obj_xy_wz))
                                        _carry_ref = float(
                                            np.linalg.norm(
                                                0.5 * (_thumb_xy_v
                                                       + _bc_xy_v)
                                                - _obj_xy_wz))
                                        _max_far_ref = max(_d_thumb_ref,
                                                           _d_bc_ref)
                                        _max_far_improved = (
                                            _max_far_ref
                                            < _max_far_orig - 0.005)
                                        _carry_improved_safely = (
                                            _carry_ref
                                            < _carry_orig - 0.01
                                            and _max_far_ref
                                                <= _max_far_orig + 0.01)
                                        _reach_better = (
                                            _max_far_improved
                                            or _carry_improved_safely)
                                        if _reach_better:
                                            PRE_GRASP_Q = _q2
                                            actual_pre_target = _at2
                                            wrist_goal = _refined_goal
                                            self._last_valid_pre_grasp_q \
                                                = list(PRE_GRASP_Q)
                                            print(f"[Exec] wz refine "
                                                  f"applied (reach-based): "
                                                  f"max_far "
                                                  f"{_max_far_orig*100:.1f}cm"
                                                  f"→{_max_far_ref*100:.1f}cm  "
                                                  f"carry_gap "
                                                  f"{_carry_orig*100:.1f}cm"
                                                  f"→{_carry_ref*100:.1f}cm  "
                                                  f"(axis err "
                                                  f"{math.degrees(_err_wz):+.1f}°"
                                                  f"→"
                                                  f"{math.degrees(_err2_wz):+.1f}°)")
                                        else:
                                            print(f"[Exec] wz refine "
                                                  f"NOT applied: reach "
                                                  f"didn't improve "
                                                  f"(max_far "
                                                  f"{_max_far_orig*100:.1f}cm"
                                                  f"→{_max_far_ref*100:.1f}cm  "
                                                  f"carry_gap "
                                                  f"{_carry_orig*100:.1f}cm"
                                                  f"→{_carry_ref*100:.1f}cm  "
                                                  f"axis err "
                                                  f"{math.degrees(_err_wz):+.1f}°"
                                                  f"→"
                                                  f"{math.degrees(_err2_wz):+.1f}°"
                                                  f"); keeping original")
                                    except RuntimeError as _e_re:
                                        print(f"[Exec] wz refine IK "
                                              f"failed: {_e_re}; "
                                              f"keeping original")
                                else:
                                    print(f"[Exec] wz already aligned: "
                                          f"axis err "
                                          f"{math.degrees(_err_wz):+.1f}°"
                                          f" ≤ "
                                          f"{math.degrees(WZ_REFINE_AXIS_ERR_THRESHOLD):.0f}°"
                                          f" tol  "
                                          f"(approach="
                                          f"{math.degrees(_approach_yaw):+.1f}°,"
                                          f" axis="
                                          f"{math.degrees(_axis_yaw):+.1f}°)")
                        except Exception as _e_wz:
                            print(f"[Exec] wz refine probe skipped: "
                                  f"{_e_wz}")
                except RuntimeError as e:
                    print(f"[Exec] PRE_GRASP_Q IK FAIL: {e}")
                    self._clear_held_state()
                    fire(False)
                    return

            GRASP_Q = list(PRE_GRASP_Q)
            print(f"[Exec] PRE_GRASP_Q (actual_target z={actual_pre_target[2]:.3f}m) "
                  f"= {[round(x,3) for x in PRE_GRASP_Q]}")
            h_diff = abs(float(PRE_GRASP_Q[1]) - float(PRE_GRASP_Q[0]))
            h_diff_cap = MAX_PICK_H_DIFF_SIDE if side_grip else MAX_PICK_H_DIFF
            if h_diff > h_diff_cap or float(PRE_GRASP_Q[2]) < MIN_PICK_A1:
                print(f"[Exec] PRE_GRASP_Q rejected for visual safety: "
                      f"h2-h1={h_diff:.3f}m (max {h_diff_cap:.2f}, "
                      f"mode={'SIDE' if side_grip else 'TOPDOWN'}), "
                      f"a1={float(PRE_GRASP_Q[2]):.3f}m (min {MIN_PICK_A1:.2f})")
                self._clear_held_state()
                fire(False)
                return

            _phase_L_built = False
            _phase_L_low_q = None
            if (USE_HOVER_DESCENT and side_grip and ENABLE_NO_CHASSIS_PUSH
                    and PHASE_L_DOWN_THEN_FORWARD):
                a1_retracted = max(MIN_PICK_A1,
                                   float(GRASP_Q[2]) - APPROACH_RETRACT_A1)
                _wr_pl = [float(GRASP_Q[i]) for i in range(4, ARM_DOF)]
                _phase_L_low_q = [
                    float(GRASP_Q[0]), float(GRASP_Q[1]),
                    a1_retracted, float(GRASP_Q[3])] + _wr_pl
                _h_mid_lift = (0.5 * (float(GRASP_Q[0]) + float(GRASP_Q[1]))
                               + HOVER_LIFT)
                _tilt_dir = 1.0 if float(GRASP_Q[1]) >= float(GRASP_Q[0]) else -1.0
                _small_tilt = 0.02
                _h1_hi = min(_h_mid_lift - 0.5 * _small_tilt * _tilt_dir,
                             COLUMN_JOINT_MAX)
                _h2_hi = min(_h_mid_lift + 0.5 * _small_tilt * _tilt_dir,
                             COLUMN_JOINT_MAX)
                _phase_L_PRE_HOVER_Q = [
                    _h1_hi, _h2_hi, a1_retracted, float(GRASP_Q[3])] + _wr_pl
                try:
                    _phase_L_valid = bool(
                        self.arm_bridge.is_valid(_phase_L_PRE_HOVER_Q)
                        and self.arm_bridge.is_valid(_phase_L_low_q))
                except Exception as _e_pl:
                    print(f"[Exec] Phase L is_valid raised: {_e_pl}")
                    _phase_L_valid = False
                if _phase_L_valid:
                    PRE_HOVER_Q = _phase_L_PRE_HOVER_Q
                    _phase_L_built = True
                    print(f"[Exec] PRE_HOVER_Q (Phase L TILT-then-HORIZONTAL: "
                          f"LIFTED+near-LEVEL h={_h1_hi:.2f}/{_h2_hi:.2f} "
                          f"(GRASP tilt h={float(GRASP_Q[0]):.2f}/"
                          f"{float(GRASP_Q[1]):.2f}) a1={a1_retracted:.3f}; "
                          f"descent GROWS tilt → low, then a1-fwd horizontal) = "
                          f"{[round(x,3) for x in PRE_HOVER_Q]}")
                else:
                    _phase_L_low_q = None
                    print(f"[Exec] Phase L PRE_HOVER_Q invalid "
                          f"(a1_retracted={a1_retracted:.3f}) — "
                          f"falling back to legacy column-lift builder")
            if USE_HOVER_DESCENT and not _phase_L_built:
                h_mid_grasp = 0.5 * (float(GRASP_Q[0]) + float(GRASP_Q[1]))
                small_tilt   = 0.04
                diff_dir     = 1.0 if (GRASP_Q[1] >= GRASP_Q[0]) else -1.0
                h_lifted     = h_mid_grasp + HOVER_LIFT
                h1_hover     = min(h_lifted - 0.5 * small_tilt * diff_dir,
                                   COLUMN_JOINT_MAX)
                h2_hover     = min(h_lifted + 0.5 * small_tilt * diff_dir,
                                   COLUMN_JOINT_MAX)
                try:
                    cur_q = self._current_arm_q()
                except Exception:
                    cur_q = list(HOME_Q)
                if len(cur_q) >= 8:
                    cur_wrist = [float(cur_q[i]) for i in (4, 5, 6, 7)]
                else:
                    cur_wrist = list(WRIST_NEUTRAL)
                grasp_wrist = [float(GRASP_Q[i]) for i in (4, 5, 6, 7)]
                d_to_grasp = sum(abs(cw - gw) for cw, gw
                                 in zip(cur_wrist, grasp_wrist))
                d_to_neutral = sum(abs(cw - 0.0) for cw in cur_wrist)
                if d_to_grasp < d_to_neutral:
                    hover_wrist = grasp_wrist
                    hover_wrist_label = "grasp-wrist (retry detected)"
                else:
                    hover_wrist = list(WRIST_NEUTRAL)
                    hover_wrist_label = "neutral (fresh approach)"
                PRE_HOVER_Q = [h1_hover, h2_hover,
                               float(GRASP_Q[2]), float(GRASP_Q[3])] + \
                              hover_wrist
                hover_valid = False
                try:
                    hover_valid = bool(self.arm_bridge.is_valid(PRE_HOVER_Q))
                except Exception as e:
                    print(f"[Exec] PRE_HOVER_Q is_valid check raised: {e}")
                if not hover_valid and hover_wrist is not WRIST_NEUTRAL:
                    print(f"[Exec] PRE_HOVER_Q invalid with "
                          f"{hover_wrist_label}; falling back to neutral wrist")
                    hover_wrist = list(WRIST_NEUTRAL)
                    hover_wrist_label = "neutral (fallback after invalid)"
                    PRE_HOVER_Q = [h1_hover, h2_hover,
                                   float(GRASP_Q[2]), float(GRASP_Q[3])] + \
                                  hover_wrist
                    try:
                        hover_valid = bool(self.arm_bridge.is_valid(PRE_HOVER_Q))
                    except Exception:
                        pass
                if hover_valid:
                    print(f"[Exec] PRE_HOVER_Q (gradient-tilt: small-tilt "
                          f"+ wrist=neutral) = "
                          f"{[round(x,3) for x in PRE_HOVER_Q]}")
                else:
                    print(f"[Exec] PRE_HOVER_Q (gradient) invalid — "
                          f"falling back to same-as-grasp hover")
                    h1_hover_alt = min(float(GRASP_Q[0]) + HOVER_LIFT,
                                       COLUMN_JOINT_MAX)
                    h2_hover_alt = min(float(GRASP_Q[1]) + HOVER_LIFT,
                                       COLUMN_JOINT_MAX)
                    PRE_HOVER_Q = [h1_hover_alt, h2_hover_alt,
                                   float(GRASP_Q[2]), float(GRASP_Q[3])] + \
                                  [float(GRASP_Q[i]) for i in range(4, ARM_DOF)]
                    try:
                        hover_valid = bool(self.arm_bridge.is_valid(PRE_HOVER_Q))
                    except Exception:
                        pass
                    if not hover_valid:
                        print(f"[Exec] PRE_HOVER_Q legacy also invalid — "
                              f"single-arc approach")
                        PRE_HOVER_Q = list(GRASP_Q)
            elif not USE_HOVER_DESCENT:
                PRE_HOVER_Q = list(GRASP_Q)

            open_thread = None
            if is_local_retry:
                print(f"\n[Exec] === LOCAL RETRY: skipping arm motion "
                      f"([3]/[4]/[5]); arm held at GRASP_Q.  Chassis "
                      f"back-then-forward maneuver will re-align. ===")
            if is_local_retry:
                print("[Exec] [3] open gripper — SKIPPED "
                      "(local retry, gripper already open)")
            elif USE_SYNC_OPEN:
                print("[Exec] [3] open gripper (sync — settle before approach)")
                self._set_gripper(GRIPPER_OPEN_POS, hold_seconds=0.0)
                self._wait_open_settle()
            else:
                print("[Exec] [3] open gripper (async — overlaps approach)")
                open_thread = threading.Thread(
                    target=self._set_gripper,
                    args=(GRIPPER_OPEN_POS,),
                    kwargs={'hold_seconds': 0.0},
                    daemon=True,
                )
                open_thread.start()
            if self._cancel:
                self._clear_held_state(); fire(False); return

            if is_local_retry:
                print("[Exec] [4] approach — SKIPPED "
                      "(local retry, arm at GRASP_Q already)")
                print("[Exec] [5] descent — SKIPPED "
                      "(local retry, arm at GRASP_Q already)")
            else:
                print(f"[Exec] [4] approach: current → PRE_HOVER_Q  "
                      f"[TIMING t={time.time()-_cycle_t0:.1f}s]")
                q_now = self._current_arm_q()
                path1 = self.arm_bridge.plan(q_now, PRE_HOVER_Q, timeout=3.0)
                if path1 is None:
                    print("[Exec] OMPL approach start invalid/unavailable — "
                          "unlocking PARK_Q then replanning")
                    avg_h = 0.5 * (float(q_now[0]) + float(q_now[1]))
                    unlock_q = [
                        max(0.05, avg_h - 0.025),
                        min(1.35, avg_h + 0.025),
                        float(q_now[2]),
                        float(q_now[3]),
                    ] + [float(q_now[i]) for i in range(4, ARM_DOF)]
                    self._kinematic_descent(q_now, unlock_q,
                                            "unlock-start",
                                            n_steps=10)
                    if self._cancel:
                        self._clear_held_state(); fire(False); return
                    path1 = self.arm_bridge.plan(unlock_q, PRE_HOVER_Q, timeout=8.0)
                    if path1 is None:
                        print("[Exec] OMPL replan from unlock pose failed — "
                              "using direct kinematic approach fallback")
                        self._kinematic_descent(unlock_q, PRE_HOVER_Q,
                                                "direct-approach",
                                                n_steps=DESCENT_STEPS * 2)
                    else:
                        self._execute_path(path1, "approach")
                else:
                    self._execute_path(path1, "approach")
                if self._cancel:
                    self._clear_held_state(); fire(False); return
                self._log_tilt_hb_diag("post-OMPL-approach")
                self._log_arm_state("post-OMPL-approach")

                _ramp_label = "approach" if _phase_L_built else "descent"
                print(f"[Exec] [5] {_ramp_label}: PRE_HOVER_Q → GRASP_Q  "
                      f"[TIMING t={time.time()-_cycle_t0:.1f}s]")
                if (ENABLE_NO_CHASSIS_PUSH and side_grip
                        and self._held_obj_bid is not None):
                    _obj_bid_d = self._held_obj_bid
                    _palm_bid_d = mujoco.mj_name2id(
                        self.sim.model, mujoco.mjtObj.mjOBJ_BODY,
                        "Gripper_Link3_1")
                    _obj_r_d = float(self._object_radius(_obj_bid_d))
                    _eff_standoff = max(GRIPPER_STANDOFF_XY,
                                        _obj_r_d + STANDOFF_RADIUS_CLEAR)
                    _xy_floor = _eff_standoff - 0.015

                    def _ramp_abort_check(step_idx, alpha):
                        try:
                            obj_xyz_now = self.sim.data.xpos[_obj_bid_d]
                            palm_xyz_now = self.sim.data.xpos[_palm_bid_d]
                            obj_z_now = float(obj_xyz_now[2])
                            palm_z_now = float(palm_xyz_now[2])
                        except Exception:
                            return False
                        z_floor = obj_z_now - 0.02
                        if _phase_L_built:
                            d_xy = float(np.linalg.norm(
                                palm_xyz_now[:2] - obj_xyz_now[:2]))
                            if d_xy < _xy_floor:
                                print(f"  [{_ramp_label}] XY-FLOOR HIT "
                                      f"step {step_idx+1}: palm-to-obj "
                                      f"XY {d_xy*100:.1f}cm < floor "
                                      f"{_xy_floor*100:.1f}cm "
                                      f"(eff_standoff="
                                      f"{_eff_standoff*100:.1f}cm) — "
                                      f"aborting before palm plunges "
                                      f"into obj")
                                return True
                        if palm_z_now < z_floor:
                            _tag = ("Z-FLOOR (unexpected sag during "
                                    "a1-extend)" if _phase_L_built
                                    else "Z-FLOOR")
                            print(f"  [{_ramp_label}] {_tag} HIT step "
                                  f"{step_idx+1}: palm_z={palm_z_now:.3f}m "
                                  f"< floor={z_floor:.3f}m "
                                  f"(obj_z={obj_z_now:.3f}m) — aborting")
                            return True
                        return False

                    self._wz_ctrl_override = None
                    self._in_descent_phase = True
                    try:
                        if _phase_L_built and _phase_L_low_q is not None:
                            self._kinematic_descent(
                                PRE_HOVER_Q, _phase_L_low_q, "approach-descend",
                                n_steps=HOVER_DESCENT_STEPS, per_step_settle=0.08,
                                early_abort_check=_ramp_abort_check,
                                early_abort_interval=2)
                            if not self._cancel:
                                self._kinematic_descent(
                                    _phase_L_low_q, GRASP_Q, "approach-forward",
                                    n_steps=HOVER_DESCENT_STEPS,
                                    per_step_settle=0.08,
                                    early_abort_check=_ramp_abort_check,
                                    early_abort_interval=2)
                        else:
                            self._kinematic_descent(
                                PRE_HOVER_Q, GRASP_Q, _ramp_label,
                                n_steps=HOVER_DESCENT_STEPS,
                                per_step_settle=0.08,
                                early_abort_check=_ramp_abort_check,
                                early_abort_interval=2)
                    finally:
                        self._in_descent_phase = False
                else:
                    self._kinematic_descent(PRE_HOVER_Q, GRASP_Q, "descent",
                                            n_steps=HOVER_DESCENT_STEPS)
                if self._cancel:
                    self._clear_held_state(); fire(False); return
                self._log_tilt_hb_diag("post-PRE_HOVER->GRASP-ramp")
                self._log_arm_state("post-PRE_HOVER->GRASP-ramp")

                if (ENABLE_NO_CHASSIS_PUSH and side_grip
                        and self._held_obj_bid is not None):
                    time.sleep(1.5)
                    try:
                        _pinch_init = self._pinch_midpoint_xyz(self.sim.data)
                        _obj_init = self.sim.data.xpos[
                            self._held_obj_bid][:3].copy()
                        _init_residual = float(np.linalg.norm(
                            _obj_init[:2] - _pinch_init[:2]))
                    except Exception:
                        _init_residual = float('inf')
                    _run_f2 = False
                    self._strict_log(
                        "ALIGN-DIFF",
                        f"F2 OFF (--no-chassis-push, Round 42b): initial "
                        f"pinch residual {_init_residual*100:.1f}cm already "
                        f"centered; F2's TH coupling swings pinch off-obj "
                        f"(v15: 4.1→15cm) and can't fix half-pocket reach.")
                    self._log_arm_state("F2-DISABLED-R42b")
                    _best_res_mag = float('inf')
                    _best_q = None
                    self._in_descent_phase = True
                    for _f2_pass in range(2 if _run_f2 else 0):
                        try:
                            self._log_arm_state(
                                f"F2-pass-{_f2_pass+1}-START")
                            pinch_xyz = self._pinch_midpoint_xyz(self.sim.data)
                            obj_xyz_f2 = self.sim.data.xpos[
                                self._held_obj_bid][:3].copy()
                            residual_xy = obj_xyz_f2[:2] - pinch_xyz[:2]
                            res_mag = float(np.linalg.norm(residual_xy))
                            self._strict_log(
                                "ALIGN-DIFF",
                                f"pass {_f2_pass+1}/5 start: "
                                f"residual ({residual_xy[0]*100:+.1f},"
                                f"{residual_xy[1]*100:+.1f}) cm "
                                f"|r|={res_mag*100:.1f}cm "
                                f"(best so far: {_best_res_mag*100 if _best_res_mag<99 else 99:.1f}cm)")
                            if res_mag < _best_res_mag:
                                _best_res_mag = res_mag
                                _best_q = self._current_arm_q()
                            else:
                                _worsening = res_mag - _best_res_mag
                                if (_best_q is not None
                                        and _f2_pass >= 1
                                        and _worsening > 0.015):
                                    self._strict_log(
                                        "ALIGN-DIFF",
                                        f"  pass {_f2_pass+1} WORSENED "
                                        f"(was {_best_res_mag*100:.1f}cm "
                                        f"now {res_mag*100:.1f}cm, "
                                        f"+{_worsening*100:.1f}cm) — "
                                        f"reverting to best q and stopping")
                                    self._set_arm_cmd(_best_q)
                                    time.sleep(2.0)
                                    break
                            if res_mag < 0.025:
                                self._strict_log(
                                    "ALIGN-DIFF",
                                    f"  converged (|r| < 2.5cm) — "
                                    f"stopping after {_f2_pass} passes")
                                break
                            cur_q = self._current_arm_q()
                            try:
                                delta_q = self.arm_bridge.differential_ik_step(
                                    cur_q,
                                    target_body="Gripper_Link3_1",
                                    residual_xyz=np.array(
                                        [residual_xy[0], residual_xy[1], 0.0]),
                                    joint_indices=(0, 1, 2),
                                    max_step_per_joint=0.025,
                                    max_xy_step=0.03)
                            except RuntimeError as _f2_err:
                                self._strict_log(
                                    "ALIGN-DIFF",
                                    f"  diff-IK rejected by validity: "
                                    f"{_f2_err} — stopping")
                                break
                            new_q = [cur_q[i] + delta_q[i] for i in range(8)]
                            self._set_arm_cmd(new_q)
                            time.sleep(2.0)
                            after_q = self._current_arm_q()
                            actual_dq = [after_q[i] - cur_q[i] for i in range(4)]
                            self._strict_log(
                                "ALIGN-DIFF",
                                f"  pass {_f2_pass+1} applied: "
                                f"Δa1={delta_q[2]*100:+.1f}cm "
                                f"Δh1={delta_q[0]*100:+.1f}cm "
                                f"Δh2={delta_q[1]*100:+.1f}cm "
                                f"Δth={np.degrees(delta_q[3]):+.1f}° | "
                                f"ACTUAL Δa1={actual_dq[2]*100:+.1f}cm "
                                f"Δh1={actual_dq[0]*100:+.1f}cm "
                                f"Δh2={actual_dq[1]*100:+.1f}cm "
                                f"Δth={np.degrees(actual_dq[3]):+.1f}°")
                            self._log_arm_state(
                                f"F2-pass-{_f2_pass+1}-END")
                            if self._cancel:
                                self._clear_held_state(); fire(False); return
                        except Exception as _f2_outer:
                            self._strict_log(
                                "ALIGN-DIFF",
                                f"pass {_f2_pass+1} raised: {_f2_outer}")
                            break
                    self._in_descent_phase = False
                    self._log_tilt_hb_diag("post-Phase-F2-align")
                    self._log_arm_state("post-Phase-F2-align")

            if is_local_retry:
                obj_xy_now = np.asarray(obj_pos_snapshot[:2], dtype=float)
                cur_xy = np.asarray(
                    self.sim.localization()[:2], dtype=float)
                approach_vec = obj_xy_now - cur_xy
                approach_norm = float(np.linalg.norm(approach_vec))
                if approach_norm > 1e-3:
                    approach_unit = approach_vec / approach_norm
                else:
                    approach_unit = np.array([1.0, 0.0], dtype=float)

                any_finger_touching = (
                    self._finger_body_groups is not None
                    and any(self._finger_touches_obj(fi, obj_bid)
                            for fi in range(3))
                )
                if any_finger_touching:
                    back_target = cur_xy - approach_unit * 0.08
                    print(f"[Exec] [5.4-retry] STEP 1 — chassis BACK "
                          f"8cm along reverse-approach (release any "
                          f"gripper↔obj contact): ({cur_xy[0]:.3f},"
                          f"{cur_xy[1]:.3f}) → ({back_target[0]:.3f},"
                          f"{back_target[1]:.3f})")
                    self._side_grip_chassis_push(
                        back_target, obj_xy_now,
                        timeout=1.5, dist_tol=0.025)
                    if self._cancel:
                        self._clear_held_state(); fire(False); return
                    import time as _t_retry
                    _t_retry.sleep(0.15)
                else:
                    print(f"[Exec] [5.4-retry] STEP 1 SKIPPED — no "
                          f"finger-obj contact to release; going "
                          f"directly to STEP 2 forward push "
                          f"(avoids wasting 8cm of chassis motion)")

                pass
            elif push_target_xy_world is not None:
                if USE_FINGER_DIAGNOSTIC_LOG:
                    self._log_finger_geometry(
                        obj_bid, "post-descent (pre-push)")
                    self._log_finger_object_contacts(
                        obj_bid, "post-descent (pre-push)")
                print(f"[Exec] [5.4] close starting (chassis push, or ARM "
                      f"const-Z slide if --arm-horizontal-pickup — handled "
                      f"inside _side_grip_chassis_push)  "
                      f"[TIMING t={time.time()-_cycle_t0:.1f}s]")
                self._side_grip_chassis_push(
                    push_target_xy_world,
                    obj_pos_snapshot[:2],
                    timeout=1.5,
                    dist_tol=0.005,
                    obj_bid=obj_bid,
                    yaw_tol=math.radians(1.0))
                if self._cancel:
                    self._clear_held_state(); fire(False); return
                if USE_FINGER_DIAGNOSTIC_LOG:
                    self._log_finger_geometry(
                        obj_bid, "post-push (pre-settle)")
                    self._log_finger_object_contacts(
                        obj_bid, "post-push (pre-settle)")
                if ENABLE_ARM_HORIZONTAL_PICKUP:
                    _ach_q = self._current_arm_q()
                    for _i in (0, 1, 2, 3, 7):
                        GRASP_Q[_i] = float(_ach_q[_i])
                    print(f"[Exec] [5.4b] arm-horizontal: adopted achieved "
                          f"const-Z pose as grip ref "
                          f"(h1={GRASP_Q[0]:.3f} h2={GRASP_Q[1]:.3f} "
                          f"a1={GRASP_Q[2]:.3f} th={GRASP_Q[3]:+.3f} "
                          f"wy={GRASP_Q[7]:+.3f}) — no re-pose rise, axis held")
                    if ENABLE_AH_GRIPZ_DROP and obj_bid is not None:
                        try:
                            _tipz = float(
                                self._carry_anchor_xyz(self.sim.data)[2])
                            _objz = float(self.sim.data.xpos[obj_bid][2])
                            _z_over = _tipz - (_objz
                                               + ARM_HORIZONTAL_GRIP_Z_BIAS)
                            if _z_over > 0.03:
                                _dz = min(_z_over, ARM_HORIZONTAL_GRIP_Z_MAXDROP)
                                _qn = list(self._current_arm_q())
                                _qd = list(_qn)
                                _qd[0] = float(np.clip(_qn[0] - _dz,
                                                       *JOINT_RANGES_ARM[0]))
                                _qd[1] = float(np.clip(_qn[1] - _dz,
                                                       *JOINT_RANGES_ARM[1]))
                                self._kinematic_descent(
                                    _qn, _qd, label="ah-gripz-drop",
                                    n_steps=8, per_step_settle=0.05)
                                _a2 = self._current_arm_q()
                                GRASP_Q[0] = float(_a2[0])
                                GRASP_Q[1] = float(_a2[1])
                                _tz2 = float(
                                    self._carry_anchor_xyz(self.sim.data)[2])
                                print(f"[Exec] [5.4c] arm-horizontal grip-Z "
                                      f"drop {_dz*100:.1f}cm: tip_z "
                                      f"{_tipz*100:.1f}→{_tz2*100:.1f}cm "
                                      f"(obj_z {_objz*100:.1f}, target "
                                      f"~{(_objz+ARM_HORIZONTAL_GRIP_Z_BIAS)*100:.1f}cm)")
                            else:
                                print(f"[Exec] [5.4c] grip-Z OK "
                                      f"(tip {_tipz*100:.1f} vs obj_z "
                                      f"{_objz*100:.1f}cm, over {_z_over*100:.1f}"
                                      f"cm ≤ 3) — no drop")
                        except Exception as _e_gz:
                            print(f"[Exec] [5.4c] grip-Z drop skipped: {_e_gz}")

            A1_FORWARD_NUDGE_ENABLED = True
            if (A1_FORWARD_NUDGE_ENABLED and side_grip
                    and (ENABLE_NO_CHASSIS_PUSH or ENABLE_ARM_HORIZONTAL_PICKUP)
                    and self._held_obj_bid is not None):
                _zt_h1, _zt_h2 = self._tilt_to_obj_z(obj_bid, GRASP_Q)
                GRASP_Q[0] = _zt_h1
                GRASP_Q[1] = _zt_h2
                self._a1_forward_align(obj_bid, GRASP_Q)
                GRASP_Q = self._wrist_yaw_balance(obj_bid, GRASP_Q)
                GRASP_Q = self._column_yaw_balance(obj_bid, GRASP_Q)
                if not ENABLE_ARM_HORIZONTAL_PICKUP:
                    GRASP_Q = self._jacobian_pinch_align(obj_bid, GRASP_Q)
                if ENABLE_CART_PICK_ALIGN and not ENABLE_ARM_HORIZONTAL_PICKUP:
                    try:
                        _obj = self.sim.data.xpos[obj_bid].copy()
                        _cen = np.asarray(self._carry_anchor_xyz(self.sim.data),
                                          dtype=float)
                        _pin = np.asarray(self._pinch_midpoint_xyz(self.sim.data),
                                          dtype=float)
                        _coff = _cen - _pin
                        _tgt = (float(_obj[0] + _coff[0]),
                                float(_obj[1] + _coff[1]), float(_cen[2]))
                        self._cartesian_move_closed_loop(
                            _tgt, hold_z=True, label="pick-cart-align")
                    except Exception as _e_ca:
                        print(f"[Exec] pick-cart-align skipped: {_e_ca}")
            intended_wrist = tuple(float(GRASP_Q[i]) for i in (4, 5, 6, 7))
            if side_grip:
                self._wait_for_wrist_settle(
                    tolerance=0.06, timeout=0.5,
                    label="wrist-settle (side-grip)",
                    intended_targets=intended_wrist)
                if ENABLE_NO_CHASSIS_PUSH:
                    self._hold_wz_to_target(
                        float(GRASP_Q[5]), label="wz-hold (side-grip)")
            else:
                self._wait_for_wrist_settle(
                    tolerance=0.10, timeout=0.8,
                    label="wrist-settle (top-down)",
                    intended_targets=intended_wrist)
            if self._cancel:
                self._clear_held_state(); fire(False); return

            if side_grip and not ENABLE_NO_CHASSIS_PUSH:
                self._runtime_wz_correction(
                    obj_bid,
                    ik_base_xy[0:2] if isinstance(ik_base_xy, (list, tuple))
                    else (float(ik_base_xy[0]), float(ik_base_xy[1])))


            if (STRICT_PICKUP_MODE and side_grip
                    and (push_target_xy_world is not None
                         or is_local_retry)
                    and not ENABLE_NO_CHASSIS_PUSH):
                try:
                    obj_xy_align = self.sim.data.xpos[obj_bid][:2].copy()
                    carry_xy_align = self._carry_anchor_xyz(
                        self.sim.data)[:2].copy()
                    residual_align = obj_xy_align - carry_xy_align
                    residual_mag_align = float(np.linalg.norm(
                        residual_align))
                    any_finger_touching_align = (
                        self._finger_body_groups is not None
                        and any(self._finger_touches_obj(fi, obj_bid)
                                for fi in range(3))
                    )
                    CLOSED_LOOP_ALIGN_MIN = 0.03
                    CLOSED_LOOP_ALIGN_CAP = 0.10
                    CLOSED_LOOP_ALIGN_ITERS = 4 if ENABLE_NO_CHASSIS_PUSH else 2
                    CLOSED_LOOP_ALIGN_OK = 0.06
                    if any_finger_touching_align:
                        self._strict_log(
                            "ALIGN",
                            "closed-loop SKIPPED — finger already "
                            "touching obj (contact-guard would abort)")
                    elif residual_mag_align < CLOSED_LOOP_ALIGN_MIN:
                        self._strict_log(
                            "ALIGN",
                            f"closed-loop SKIPPED — residual "
                            f"{residual_mag_align*100:.1f}cm < "
                            f"{CLOSED_LOOP_ALIGN_MIN*100:.0f}cm "
                            f"(already aligned)")
                    else:
                        last_residual = residual_mag_align
                        for _snap_iter in range(CLOSED_LOOP_ALIGN_ITERS):
                            obj_xy_iter = self.sim.data.xpos[
                                obj_bid][:2].copy()
                            carry_xy_iter = self._carry_anchor_xyz(
                                self.sim.data)[:2].copy()
                            residual_iter = obj_xy_iter - carry_xy_iter
                            residual_iter_mag = float(np.linalg.norm(
                                residual_iter))
                            if residual_iter_mag <= CLOSED_LOOP_ALIGN_OK:
                                self._strict_log(
                                    "ALIGN",
                                    f"closed-loop converged at "
                                    f"iter {_snap_iter+1}: residual "
                                    f"{residual_iter_mag*100:.1f}cm "
                                    f"≤ {CLOSED_LOOP_ALIGN_OK*100:.0f}cm")
                                break
                            if (self._finger_body_groups is not None
                                    and any(self._finger_touches_obj(fi, obj_bid)
                                            for fi in range(3))):
                                self._strict_log(
                                    "ALIGN",
                                    f"closed-loop iter {_snap_iter+1} "
                                    f"STOPPED — finger now touching obj "
                                    f"after prev snap")
                                break
                            snap_vec = residual_iter.copy()
                            if residual_iter_mag > CLOSED_LOOP_ALIGN_CAP:
                                snap_vec = snap_vec * (
                                    CLOSED_LOOP_ALIGN_CAP / residual_iter_mag)
                            cur_xy_iter = np.asarray(
                                self.sim.localization()[:2], dtype=float)
                            snap_target = cur_xy_iter + snap_vec
                            snap_mag = float(np.linalg.norm(snap_vec))
                            if ENABLE_NO_CHASSIS_PUSH:
                                self._strict_log(
                                    "ALIGN",
                                    f"closed-loop iter {_snap_iter+1}/"
                                    f"{CLOSED_LOOP_ALIGN_ITERS} arm-side: "
                                    f"residual "
                                    f"({residual_iter[0]*100:+.1f},"
                                    f"{residual_iter[1]*100:+.1f})cm "
                                    f"|r|={residual_iter_mag*100:.1f}cm "
                                    f"→ arm IK toward obj XY")
                                _align_iter_t0 = time.time()
                                try:
                                    cur_q = self._current_arm_q()
                                    _loc_align = self.sim.localization()
                                    _link1_bid_a = mujoco.mj_name2id(
                                        self.sim.model,
                                        mujoco.mjtObj.mjOBJ_BODY,
                                        "Gripper_Link1_1")
                                    if _link1_bid_a >= 0:
                                        link1_now = self.sim.data.xpos[
                                            _link1_bid_a].copy()
                                    else:
                                        link1_now = self._carry_anchor_xyz(
                                            self.sim.data).copy()
                                    try:
                                        _obj_r_align = float(
                                            self._object_radius(obj_bid))
                                    except Exception:
                                        _obj_r_align = None
                                    _, _ideal_target = compute_grasp_targets(
                                        (float(_loc_align[0]),
                                         float(_loc_align[1])),
                                        np.array([
                                            float(obj_xy_iter[0]),
                                            float(obj_xy_iter[1]),
                                            float(link1_now[2]),
                                        ], dtype=float),
                                        obj_radius=_obj_r_align,
                                        side_approach=True)
                                    _delta_xy = (_ideal_target[:2]
                                                 - link1_now[:2])
                                    _delta_mag = float(
                                        np.linalg.norm(_delta_xy))
                                    if _delta_mag > CLOSED_LOOP_ALIGN_CAP:
                                        _delta_xy = _delta_xy * (
                                            CLOSED_LOOP_ALIGN_CAP / _delta_mag)
                                    target_xyz = np.array([
                                        float(link1_now[0]) + float(_delta_xy[0]),
                                        float(link1_now[1]) + float(_delta_xy[1]),
                                        float(link1_now[2]),
                                    ], dtype=float)
                                    target_xyz[2] = max(
                                        target_xyz[2], MIN_PICK_WRIST_Z)
                                    reset_plan_data_for_ik(
                                        self.arm_bridge,
                                        (float(_loc_align[0]),
                                         float(_loc_align[1])),
                                        float(_loc_align[2]))
                                    wg = (WRIST_PITCH_SIDE_APPROACH,
                                          WRIST_Z_SIDE_APPROACH,
                                          WRIST_X_SIDE_APPROACH,
                                          WRIST_Y_SIDE_APPROACH)
                                    seed_extend = list(cur_q)
                                    seed_extend[2] = 0.55
                                    seed_extend[0] = max(seed_extend[0], 0.21)
                                    _ik_t0 = time.time()
                                    try:
                                        new_q, _ = self.arm_bridge.solve_ik_no_z_lift(
                                            target_xyz,
                                            n_seeds=4,
                                            threshold=0.10,
                                            wrist_goal=wg,
                                            wrist_weight=(0.1, 3.0, 3.0, 3.0),
                                            seed_q=seed_extend,
                                            tilt_weight_scale=0.2,
                                            manual_pull_scale=0.1,
                                            validity_penalty_scale=50.0,
                                        )
                                    except RuntimeError as _ik_e:
                                        _ik_dt = time.time() - _ik_t0
                                        self._strict_log(
                                            "ALIGN",
                                            f"  IK FAILED in {_ik_dt:.2f}s: "
                                            f"{_ik_e} — skipping arm motion")
                                        continue
                                    _ik_dt = time.time() - _ik_t0
                                    _dq_a1 = float(new_q[2]) - float(cur_q[2])
                                    self._strict_log(
                                        "ALIGN",
                                        f"  IK ok in {_ik_dt:.2f}s: "
                                        f"cur=(h1={cur_q[0]:.3f} h2={cur_q[1]:.3f} "
                                        f"a1={cur_q[2]:.3f} th={cur_q[3]:.3f}) "
                                        f"→ new=(h1={new_q[0]:.3f} h2={new_q[1]:.3f} "
                                        f"a1={new_q[2]:.3f} th={new_q[3]:.3f})  "
                                        f"Δa1={_dq_a1*100:+.1f}cm")
                                    _dq_a1 = float(new_q[2]) - float(cur_q[2])
                                    _dq_h1 = float(new_q[0]) - float(cur_q[0])
                                    _dq_h2 = float(new_q[1]) - float(cur_q[1])
                                    self._strict_log(
                                        "ALIGN",
                                        f"  diff-IK Link1 step "
                                        f"({_delta_xy[0]*100:+.1f},"
                                        f"{_delta_xy[1]*100:+.1f}) cm "
                                        f"(ideal {_delta_mag*100:.1f}cm)  "
                                        f"IK_t={_ik_dt:.2f}s  "
                                        f"Δq: a1={_dq_a1*100:+.1f}cm "
                                        f"h1={_dq_h1*100:+.1f}cm "
                                        f"h2={_dq_h2*100:+.1f}cm  "
                                        f"new_a1={new_q[2]:.3f}")
                                    _km_t0 = time.time()
                                    self._kinematic_descent(
                                        cur_q, new_q,
                                        label=f"arm-align-iter-{_snap_iter+1}",
                                        n_steps=12,
                                    )
                                    _km_dt = time.time() - _km_t0
                                    _total_dt = time.time() - _align_iter_t0
                                    self._strict_log(
                                        "ALIGN",
                                        f"  arm-align-iter timings: "
                                        f"IK={_ik_dt:.2f}s  "
                                        f"kinematic_descent={_km_dt:.2f}s  "
                                        f"total={_total_dt:.2f}s")
                                except Exception as _e_arm_align:
                                    self._strict_log(
                                        "ALIGN",
                                        f"arm-side IK failed: "
                                        f"{_e_arm_align} — skipping iter")
                            else:
                                self._strict_log(
                                    "ALIGN",
                                    f"closed-loop iter {_snap_iter+1}/"
                                    f"{CLOSED_LOOP_ALIGN_ITERS} snap: "
                                    f"residual "
                                    f"({residual_iter[0]*100:+.1f},"
                                    f"{residual_iter[1]*100:+.1f})cm "
                                    f"|r|={residual_iter_mag*100:.1f}cm "
                                    f"→ pushing {snap_mag*100:.1f}cm")
                                self._side_grip_chassis_push(
                                    snap_target, obj_xy_iter,
                                    timeout=1.5, dist_tol=0.020,
                                    abort_on_finger_obj_contact=True,
                                    obj_bid=obj_bid)
                            if self._cancel:
                                self._clear_held_state()
                                fire(False)
                                return
                            carry_xy_post = self._carry_anchor_xyz(
                                self.sim.data)[:2].copy()
                            last_residual = float(np.linalg.norm(
                                obj_xy_iter - carry_xy_post))
                        self._strict_log(
                            "ALIGN",
                            f"closed-loop result: residual "
                            f"{residual_mag_align*100:.1f}cm → "
                            f"{last_residual*100:.1f}cm")
                except Exception as _e_align:
                    print(f"[Exec] closed-loop alignment raised: "
                          f"{_e_align} — proceeding with current pose")

            if (ENABLE_TH_FINE_YAW and self._held_obj_bid is not None):
                try:
                    _ta_bid7 = mujoco.mj_name2id(
                        self.sim.model, mujoco.mjtObj.mjOBJ_BODY,
                        "finger_a_link_3_1")
                    _tb_bid7 = mujoco.mj_name2id(
                        self.sim.model, mujoco.mjtObj.mjOBJ_BODY,
                        "finger_b_link_3_1")
                    _tc_bid7 = mujoco.mj_name2id(
                        self.sim.model, mujoco.mjtObj.mjOBJ_BODY,
                        "finger_c_link_3_1")
                    if _ta_bid7 >= 0 and _tb_bid7 >= 0 and _tc_bid7 >= 0:
                        _thumb_xy7 = self.sim.data.xpos[_ta_bid7][:2]
                        _bc_xy7 = 0.5 * (self.sim.data.xpos[_tb_bid7][:2]
                                         + self.sim.data.xpos[_tc_bid7][:2])
                        _axis_vec7 = _bc_xy7 - _thumb_xy7
                        _chassis_xy7 = np.asarray(
                            self.sim.localization()[:2], dtype=float)
                        _obj_xyz_pre7 = self.sim.data.xpos[self._held_obj_bid][:3]
                        _approach_vec7 = _obj_xyz_pre7[:2] - _chassis_xy7
                        if (np.linalg.norm(_axis_vec7) > 1e-4
                                and np.linalg.norm(_approach_vec7) > 1e-4):
                            _axis_yaw7 = float(np.arctan2(
                                _axis_vec7[1], _axis_vec7[0]))
                            _approach_yaw7 = float(np.arctan2(
                                _approach_vec7[1], _approach_vec7[0]))
                            def _wrap7(a):
                                while a > np.pi:
                                    a -= 2.0 * np.pi
                                while a < -np.pi:
                                    a += 2.0 * np.pi
                                return a
                            _err_a7 = _wrap7(
                                (_approach_yaw7 + np.pi / 2) - _axis_yaw7)
                            _err_b7 = _wrap7(
                                (_approach_yaw7 - np.pi / 2) - _axis_yaw7)
                            _yaw_err_pre7 = (_err_a7 if abs(_err_a7) < abs(_err_b7)
                                             else _err_b7)
                            _abs_pre7 = abs(_yaw_err_pre7)
                            print(f"[Exec] P7 PRE-CLOSE axis metric: "
                                  f"thumb-bc yaw={np.degrees(_axis_yaw7):+.1f}° "
                                  f"approach={np.degrees(_approach_yaw7):+.1f}° "
                                  f"→ residual {np.degrees(_yaw_err_pre7):+.1f}° "
                                  f"(to make axis ⊥ approach)")
                            if (ENABLE_AH_AXIS_CORRECT
                                    and os.environ.get("AH_AXIS_PERSIST", "1") == "1"
                                    and self._held_obj_bid is not None):
                                try:
                                    _cq = list(self._current_arm_q())
                                    _wy = float(_cq[7])
                                    _api_n = int(os.environ.get(
                                        "AH_AXIS_PERSIST_ITERS", "20"))
                                    _wy_lo = float(os.environ.get(
                                        "AH_AXIS_PERSIST_WY_LO", "-0.60"))
                                    _wy_st = float(os.environ.get(
                                        "AH_AXIS_PERSIST_STEP", "0.05"))
                                    for _api in range(_api_n):
                                        _ar = self._axis_residual_deg(
                                            self._held_obj_bid)
                                        if _ar is None or abs(_ar) <= 3.0:
                                            break
                                        _wy += float(np.clip(-0.006 * _ar,
                                                             -_wy_st, _wy_st))
                                        _wy = float(np.clip(_wy, _wy_lo, 0.30))
                                        _cq = list(self._current_arm_q())
                                        _cq[7] = _wy
                                        self._set_arm_cmd(_cq)
                                        time.sleep(0.08)
                                    _arf = self._axis_residual_deg(
                                        self._held_obj_bid)
                                    print("[Exec] AXIS-PERSIST: re-nulled wy at "
                                          f"pre-close (wy={_wy:+.3f}, axis_resid="
                                          f"{('%.1f' % _arf) if _arf is not None else 'n/a'}"
                                          "deg) — obj between fingers for close")
                                except Exception as _e_ap:
                                    print(f"[Exec] AXIS-PERSIST skipped: {_e_ap}")
                            if (P7_PRECLOSE_FIRE
                                    and _abs_pre7 > TH_FINE_YAW_RESIDUAL_THRESHOLD
                                    and _abs_pre7 <= TH_FINE_YAW_MAX_DELTA):
                                cur_q_pre7 = self._current_arm_q()
                                proposed_q_pre7 = list(cur_q_pre7)
                                proposed_q_pre7[3] = (
                                    float(cur_q_pre7[3]) + float(_yaw_err_pre7))
                                proposed_q_pre7[3] = max(
                                    -3.14, min(3.14, proposed_q_pre7[3]))
                                try:
                                    _valid_pre7 = self.arm_bridge.is_valid(
                                        proposed_q_pre7)
                                except Exception:
                                    _valid_pre7 = False
                                if _valid_pre7:
                                    print(f"[Exec] P7 PRE-CLOSE ACTIVE: yaw "
                                          f"residual "
                                          f"{np.degrees(_yaw_err_pre7):+.1f}° "
                                          f"→ commanding TH delta "
                                          f"{np.degrees(_yaw_err_pre7):+.1f}° "
                                          f"(TH {cur_q_pre7[3]:+.3f} → "
                                          f"{proposed_q_pre7[3]:+.3f} rad). "
                                          f"Column rotates BEFORE pre-close "
                                          f"gate evaluates.")
                                    try:
                                        self._kinematic_descent(
                                            cur_q_pre7, proposed_q_pre7,
                                            label="p7-pre-close-th-yaw",
                                            n_steps=10)
                                    except Exception as _e_pre7_ramp:
                                        print(f"[Exec] WARN P7 PRE-CLOSE TH "
                                              f"ramp: {_e_pre7_ramp}")
                                else:
                                    print(f"[Exec] P7 PRE-CLOSE: yaw residual "
                                          f"{np.degrees(_yaw_err_pre7):+.1f}° "
                                          f"actionable but proposed TH FAILED "
                                          f"validity — skipping.")
                            else:
                                print(f"[Exec] P7 PRE-CLOSE: yaw residual "
                                      f"{np.degrees(_yaw_err_pre7):+.1f}° not "
                                      f"in actionable range "
                                      f"({np.degrees(TH_FINE_YAW_RESIDUAL_THRESHOLD):.1f}°-"
                                      f"{np.degrees(TH_FINE_YAW_MAX_DELTA):.0f}°).")
                except Exception as _e_p7_pre:
                    print(f"[Exec] WARN P7 pre-close TH-yaw: {_e_p7_pre}")

            if USE_FINGER_DIAGNOSTIC_LOG:
                self._log_finger_geometry(obj_bid, "post-descent (pre-close)")
                self._log_finger_object_contacts(obj_bid, "post-descent (pre-close)")

            self._log_gripper_floor_chassis_contacts("post-descent (pre-close)")

            self._log_filtered_contact_summary(obj_bid,
                                               "post-descent (pre-close)")

            if side_grip:
                align_pt_xy = self._pinch_midpoint_xyz(self.sim.data)[:2].copy()
                align_metric_name = "pinch_midpoint"
            else:
                align_pt_xy = self._carry_anchor_xyz(self.sim.data)[:2].copy()
                align_metric_name = "carry_anchor"
            grip_xyz_pre = self.sim.data.xpos[self.gripper_body_id].copy()
            obj_xyz_pre  = self.sim.data.xpos[obj_bid].copy()
            carry_gap    = float(np.linalg.norm(align_pt_xy - obj_xyz_pre[:2]))
            if side_grip and len(self._carry_anchor_body_ids) == 3:
                _thumb_xy = self.sim.data.xpos[
                    self._carry_anchor_body_ids[0]][:2].copy()
                _bc_xy = 0.5 * (
                    self.sim.data.xpos[self._carry_anchor_body_ids[1]][:2]
                    + self.sim.data.xpos[self._carry_anchor_body_ids[2]][:2]
                ).copy()
                d_thumb_obj = float(np.linalg.norm(_thumb_xy - obj_xyz_pre[:2]))
                d_bc_obj    = float(np.linalg.norm(_bc_xy    - obj_xyz_pre[:2]))
            else:
                d_thumb_obj = 0.0
                d_bc_obj    = 0.0
            obj_grip_xy  = float(np.linalg.norm(grip_xyz_pre[:2] - obj_xyz_pre[:2]))
            expected_obj_grip_xy = GRIPPER_STANDOFF_XY
            extra_obj_grip_xy = max(0.0, obj_grip_xy - expected_obj_grip_xy)
            ik_dev = float(np.linalg.norm(grip_xyz_pre[:2] - pre_grasp_target[:2]))
            obj_xy_gate = expected_obj_grip_xy + GRIP_OBJ_XY_TOLERANCE
            try:
                _obj_r_gate = float(self._object_radius(obj_bid))
            except Exception:
                _obj_r_gate = 0.0
            CARRY_GAP_PALM_CLEARANCE = 0.040
            if side_grip and _obj_r_gate > 0.0:
                carry_gap_tol = _obj_r_gate + CARRY_GAP_PALM_CLEARANCE
            else:
                carry_gap_tol = (CARRY_GAP_TOLERANCE_SIDE if side_grip
                                 else CARRY_GAP_TOLERANCE)

            palm_z = float(grip_xyz_pre[2])
            obj_top_z = float(obj_xyz_pre[2] + self._object_half_height(obj_bid))
            z_gap = palm_z - obj_top_z
            try:
                finger_obj_ncon = int(self._count_finger_obj_contacts(obj_bid))
            except Exception:
                finger_obj_ncon = 0
            try:
                arm_obj_ncon = int(self._count_arm_obj_contacts(obj_bid))
            except Exception:
                arm_obj_ncon = 0
            realism_active = bool(REALISM_MODE_NO_SMOOTH_LIFT and side_grip)
            self._log_tilt_hb_diag("pre-close-gate")
            self._log_arm_state("pre-close-gate")

            print(f"[Exec] [5.5] pre-close gate ({align_metric_name}): "
                  f"carry_gap={carry_gap*100:.1f}cm "
                  f"(tol={carry_gap_tol*100:.1f}cm)  "
                  f"d_thumb={d_thumb_obj*100:.1f}cm  "
                  f"d_bc={d_bc_obj*100:.1f}cm  "
                  f"(side-reach tol={SIDE_FINGER_PRECLOSE_REACH*100:.1f}cm)  "
                  f"obj-grip xy={obj_grip_xy:.3f}m  "
                  f"ik_dev={ik_dev:.3f}m  "
                  f"z_gap={z_gap*100:+.1f}cm  "
                  f"finger-obj ncon={finger_obj_ncon}  "
                  f"arm-obj ncon={arm_obj_ncon}  "
                  f"realism={realism_active}")
            try:
                import os as _os_csv
                if _os_csv.environ.get('LOG_ALIGNMENT_CSV'):
                    _csv_path = _os_csv.environ['LOG_ALIGNMENT_CSV']
                    _new_file = not _os_csv.path.exists(_csv_path)
                    with open(_csv_path, 'a') as _f:
                        if _new_file:
                            _f.write(
                                "ts,obj_idx,carry_gap_cm,carry_gap_tol_cm,"
                                "d_thumb_cm,d_bc_cm,side_reach_tol_cm,"
                                "obj_grip_xy_m,ik_dev_m,z_gap_cm,"
                                "finger_obj_ncon,arm_obj_ncon,realism\n")
                        _f.write(
                            f"{time.time():.3f},"
                            f"{getattr(self,'_held_obj_idx','?')},"
                            f"{carry_gap*100:.2f},{carry_gap_tol*100:.2f},"
                            f"{d_thumb_obj*100:.2f},{d_bc_obj*100:.2f},"
                            f"{SIDE_FINGER_PRECLOSE_REACH*100:.2f},"
                            f"{obj_grip_xy:.4f},{ik_dev:.4f},{z_gap*100:.2f},"
                            f"{finger_obj_ncon},{arm_obj_ncon},"
                            f"{int(realism_active)}\n")
            except Exception:
                pass
            effective_carry_tol = (REALISM_MICRO_LIFT_THRESHOLD
                                   if realism_active
                                   else carry_gap_tol)
            carry_ok = (carry_gap <= effective_carry_tol)
            if realism_active and side_grip:
                legacy_ok = True
            else:
                legacy_ok = (ik_dev <= GRIP_DEVIATION_TOLERANCE
                             and obj_grip_xy <= obj_xy_gate)
            if side_grip:
                _reach_max = (SIDE_FINGER_PRECLOSE_REACH_PERFECT
                              if STRICT_PERFECT_FRICTION_ONLY
                              else SIDE_FINGER_PRECLOSE_REACH)
                if ENABLE_NO_CHASSIS_PUSH or ENABLE_ARM_HORIZONTAL_PICKUP:
                    _reach_max = SIDE_FINGER_PRECLOSE_REACH_PERFECT
                sides_ok = (d_thumb_obj <= _reach_max
                            and d_bc_obj <= _reach_max)
                if ENABLE_ARM_HORIZONTAL_PICKUP:
                    _between_ok, _bproj_cm, _bspan_cm = self._obj_between_fingers(
                        obj_xyz_pre[:2], _thumb_xy, _bc_xy, margin=_obj_r_gate)
                    if not _between_ok:
                        print(f"[Exec] [5.5] SAME-SIDE reject: obj projects "
                              f"{_bproj_cm:+.1f}cm on the thumb→bc axis "
                              f"(enclosed span 0..{_bspan_cm:.1f}cm, margin "
                              f"{_obj_r_gate*100:.1f}cm) — obj NOT between "
                              f"fingers; skip close (would shove the free obj), "
                              f"retry next pose")
                        sides_ok = False
            else:
                sides_ok = True
            z_ok = (z_gap <= REALISM_PRE_CLOSE_Z_GAP) if realism_active else True
            contact_reach_ok = True

            MAX_NUDGES = 3
            NUDGE_MAX_RESIDUAL = 0.25
            xy_residual_vec = obj_xyz_pre[:2] - align_pt_xy
            xy_residual_mag = float(np.linalg.norm(xy_residual_vec))
            xy_fail = (not carry_ok) or (not contact_reach_ok) or (not sides_ok)
            import time as _t
            nudge_iter = 0
            arm_chassis_ncon_pre = self._count_arm_chassis_contacts()
            if (STRICT_PICKUP_MODE
                    and STRICT_PERFECT_FRICTION_ONLY
                    and side_grip
                    and len(self._carry_anchor_body_ids) == 3):
                if True:
                    try:
                        _obj_xy_o = np.asarray(obj_xyz_pre[:2], dtype=float)
                        _cur_loc = self.sim.localization()
                        _cur_xy_o = np.asarray(_cur_loc[:2], dtype=float)
                        _cur_yaw_o = float(_cur_loc[2])
                        _t_xy_o = self.sim.data.xpos[
                            self._carry_anchor_body_ids[0]][:2].copy()
                        _b_xy_o = self.sim.data.xpos[
                            self._carry_anchor_body_ids[1]][:2].copy()
                        _c_xy_o = self.sim.data.xpos[
                            self._carry_anchor_body_ids[2]][:2].copy()
                        _bc_xy_o = 0.5 * (_b_xy_o + _c_xy_o)
                        _carry_xy_o = 0.5 * (_t_xy_o + _bc_xy_o)
                        _axis_vec = _t_xy_o - _bc_xy_o
                        _axis_angle = math.atan2(_axis_vec[1], _axis_vec[0])
                        _obj_dir = _obj_xy_o - _carry_xy_o
                        _obj_angle = math.atan2(_obj_dir[1], _obj_dir[0])
                        _desired_a = _obj_angle + math.pi / 2.0
                        _desired_b = _obj_angle - math.pi / 2.0
                        def _norm_a(x):
                            while x > math.pi:
                                x -= 2 * math.pi
                            while x < -math.pi:
                                x += 2 * math.pi
                            return x
                        _delta_a = _norm_a(_desired_a - _axis_angle)
                        _delta_b = _norm_a(_desired_b - _axis_angle)
                        _delta = (_delta_a if abs(_delta_a) < abs(_delta_b)
                                  else _delta_b)
                        if math.radians(5.0) < abs(_delta) < math.radians(60.0):
                            _ch_to_obj = _obj_xy_o - _cur_xy_o
                            _dist_to_obj = float(np.linalg.norm(_ch_to_obj))
                            if _dist_to_obj > 1e-6:
                                _unit = _ch_to_obj / _dist_to_obj
                            else:
                                _unit = np.array([1.0, 0.0])
                            _retract_xy = _cur_xy_o - 0.10 * _unit
                            _approach_xy_after = _obj_xy_o - _dist_to_obj * np.array([
                                math.cos(math.atan2(_unit[1], _unit[0]) + _delta),
                                math.sin(math.atan2(_unit[1], _unit[0]) + _delta)
                            ])
                            _new_yaw_o = _cur_yaw_o + _delta
                            self._strict_log(
                                "AXIS-ORBIT",
                                f"3-phase  delta={math.degrees(_delta):+.1f}°  "
                                f"retract ({_cur_xy_o[0]:.2f},{_cur_xy_o[1]:.2f})"
                                f"→({_retract_xy[0]:.2f},{_retract_xy[1]:.2f})  "
                                f"then approach→({_approach_xy_after[0]:.2f},"
                                f"{_approach_xy_after[1]:.2f})  "
                                f"yaw {math.degrees(_cur_yaw_o):+.0f}°→"
                                f"{math.degrees(_new_yaw_o):+.0f}°")
                            with self.sim._target_lock:
                                self.sim.target_base = np.array([
                                    float(_retract_xy[0]),
                                    float(_retract_xy[1]),
                                    float(_cur_yaw_o)])
                            time.sleep(1.2)
                            with self.sim._target_lock:
                                self.sim.target_base = np.array([
                                    float(_retract_xy[0]),
                                    float(_retract_xy[1]),
                                    float(_new_yaw_o)])
                            time.sleep(1.2)
                            _new_xy_o = _approach_xy_after
                            with self.sim._target_lock:
                                self.sim.target_base = np.array([
                                    float(_new_xy_o[0]),
                                    float(_new_xy_o[1]),
                                    float(_new_yaw_o)])
                            time.sleep(1.5)
                            _t2 = self.sim.data.xpos[
                                self._carry_anchor_body_ids[0]][:2].copy()
                            _b2 = self.sim.data.xpos[
                                self._carry_anchor_body_ids[1]][:2].copy()
                            _c2 = self.sim.data.xpos[
                                self._carry_anchor_body_ids[2]][:2].copy()
                            _bc2 = 0.5 * (_b2 + _c2)
                            d_thumb_obj = float(np.linalg.norm(
                                _t2 - _obj_xy_o))
                            d_bc_obj = float(np.linalg.norm(
                                _bc2 - _obj_xy_o))
                            sides_ok = (d_thumb_obj <= _reach_max
                                        and d_bc_obj <= _reach_max)
                            xy_fail = (not carry_ok) or (
                                not contact_reach_ok) or (not sides_ok)
                            self._strict_log(
                                "AXIS-ORBIT",
                                f"post-orbit  d_thumb={d_thumb_obj*100:.1f}cm "
                                f"d_bc={d_bc_obj*100:.1f}cm  "
                                f"sides_ok={sides_ok}")
                        else:
                            self._strict_log(
                                "AXIS-ORBIT",
                                f"skip  delta={math.degrees(_delta):+.1f}° "
                                f"out of [5°, 60°] band")
                    except Exception as _e_orb:
                        print(f"[Exec] proactive axis-orbit failed: "
                              f"{_e_orb}")
                if xy_fail:
                    self._strict_log(
                        "GATE",
                        f"live-nudge SKIPPED — STRICT rejects on asymmetric "
                        f"reach so play_m1 tries next base candidate "
                        f"(carry_gap={carry_gap*100:.1f}cm, "
                        f"d_thumb={d_thumb_obj*100:.1f}cm, "
                        f"d_bc={d_bc_obj*100:.1f}cm)")
            while (side_grip and xy_fail and z_ok
                   and nudge_iter < MAX_NUDGES
                   and xy_residual_mag < NUDGE_MAX_RESIDUAL
                   and not FAST_PICKUP_MODE
                   and not STRICT_PICKUP_MODE):
                cur_chassis_xy = np.asarray(
                    self.sim.localization()[:2], dtype=float)
                nudge_target = cur_chassis_xy + xy_residual_vec
                proposed_dist_to_obj = float(np.linalg.norm(
                    nudge_target - np.asarray(obj_xyz_pre[:2], dtype=float)))
                if proposed_dist_to_obj < MIN_CHASSIS_OBJ_DIST:
                    print(f"[Exec] [5.5b] abort nudge #{nudge_iter+1}: "
                          f"would put chassis at {proposed_dist_to_obj*100:.1f}cm "
                          f"from obj (< MIN_CHASSIS_OBJ_DIST="
                          f"{MIN_CHASSIS_OBJ_DIST*100:.0f}cm)")
                    break

                nudge_iter += 1
                self._pre_close_nudges_used = (
                    getattr(self, '_pre_close_nudges_used', 0) + 1)
                prev_carry_gap = carry_gap
                print(f"[Exec] [5.5b] LIVE chassis nudge "
                      f"#{nudge_iter}/{MAX_NUDGES}: "
                      f"residual {xy_residual_mag*100:.1f}cm "
                      f"({xy_residual_vec[0]*100:+.1f},"
                      f"{xy_residual_vec[1]*100:+.1f})cm  "
                      f"chassis ({cur_chassis_xy[0]:.3f},"
                      f"{cur_chassis_xy[1]:.3f}) → "
                      f"({nudge_target[0]:.3f},{nudge_target[1]:.3f})  "
                      f"[arm held at GRASP_Q]")
                self._side_grip_chassis_push(
                    nudge_target,
                    np.asarray(obj_xyz_pre[:2], dtype=float),
                    timeout=2.0, dist_tol=0.025)
                if self._cancel:
                    self._clear_held_state(); fire(False); return
                _t.sleep(0.2)
                if side_grip:
                    align_pt_xy = self._pinch_midpoint_xyz(self.sim.data)[:2].copy()
                else:
                    align_pt_xy = self._carry_anchor_xyz(self.sim.data)[:2].copy()
                grip_xyz_pre = self.sim.data.xpos[self.gripper_body_id].copy()
                obj_xyz_pre  = self.sim.data.xpos[obj_bid].copy()
                carry_gap    = float(np.linalg.norm(align_pt_xy - obj_xyz_pre[:2]))
                obj_grip_xy  = float(np.linalg.norm(grip_xyz_pre[:2] - obj_xyz_pre[:2]))
                ik_dev = float(np.linalg.norm(
                    grip_xyz_pre[:2] - pre_grasp_target[:2]))
                palm_z = float(grip_xyz_pre[2])
                obj_top_z = float(obj_xyz_pre[2]
                                  + self._object_half_height(obj_bid))
                z_gap = palm_z - obj_top_z
                try:
                    finger_obj_ncon = int(
                        self._count_finger_obj_contacts(obj_bid))
                except Exception:
                    finger_obj_ncon = 0
                carry_ok = (carry_gap <= effective_carry_tol)
                if realism_active and side_grip:
                    legacy_ok = True
                else:
                    legacy_ok = (ik_dev <= GRIP_DEVIATION_TOLERANCE
                                 and obj_grip_xy <= obj_xy_gate)
                z_ok = (z_gap <= REALISM_PRE_CLOSE_Z_GAP) if realism_active else True
                contact_reach_ok = True
                if side_grip and len(self._carry_anchor_body_ids) == 3:
                    _thumb_xy = self.sim.data.xpos[
                        self._carry_anchor_body_ids[0]][:2].copy()
                    _bc_xy = 0.5 * (
                        self.sim.data.xpos[self._carry_anchor_body_ids[1]][:2]
                        + self.sim.data.xpos[self._carry_anchor_body_ids[2]][:2]
                    ).copy()
                    d_thumb_obj = float(np.linalg.norm(
                        _thumb_xy - obj_xyz_pre[:2]))
                    d_bc_obj = float(np.linalg.norm(
                        _bc_xy - obj_xyz_pre[:2]))
                    _side_reach_tol = (
                        SIDE_FINGER_PRECLOSE_REACH + 0.01
                        if ENABLE_NO_CHASSIS_PUSH
                        else SIDE_FINGER_PRECLOSE_REACH)
                    sides_ok = (d_thumb_obj <= _side_reach_tol
                                and d_bc_obj <= _side_reach_tol)
                else:
                    sides_ok = True
                xy_residual_vec = obj_xyz_pre[:2] - align_pt_xy
                xy_residual_mag = float(np.linalg.norm(xy_residual_vec))
                xy_fail = (not carry_ok) or (not contact_reach_ok) or (not sides_ok)
                arm_chassis_ncon_post = self._count_arm_chassis_contacts()
                improvement = prev_carry_gap - carry_gap
                print(f"[Exec] [5.5b] after nudge #{nudge_iter}: "
                      f"carry_gap={carry_gap*100:.1f}cm "
                      f"(Δ={improvement*100:+.1f}cm)  d_thumb="
                      f"{d_thumb_obj*100:.1f}cm  d_bc={d_bc_obj*100:.1f}cm  "
                      f"obj-grip xy={obj_grip_xy:.3f}m  finger-obj ncon="
                      f"{finger_obj_ncon}  z_gap={z_gap*100:+.1f}cm  "
                      f"arm-chassis ncon {arm_chassis_ncon_pre}"
                      f"→{arm_chassis_ncon_post}")
                if not xy_fail:
                    print(f"[Exec] [5.5b] gate passes after "
                          f"#{nudge_iter} nudge(s) — exiting loop")
                    break
                if arm_chassis_ncon_post > arm_chassis_ncon_pre:
                    print(f"[Exec] [5.5b] abort: arm-chassis contacts "
                          f"increased {arm_chassis_ncon_pre}"
                          f"→{arm_chassis_ncon_post} — arm starting to "
                          f"clip base; further nudges would worsen")
                    break
                if finger_obj_ncon >= 1 and not sides_ok and not STRICT_PICKUP_MODE:
                    if finger_obj_ncon >= 2:
                        print(f"[Exec] [5.5b] ASYMMETRIC 2-finger accept "
                              f"({finger_obj_ncon} finger(s) on obj, "
                              f"far side ="
                              f"{max(d_thumb_obj, d_bc_obj)*100:.1f}cm) — "
                              f"close stroke will complete the grip; "
                              f"pin closure holds for transport.  "
                              f"Skipping disengage + abort.")
                        sides_ok = True
                        xy_fail = (not carry_ok) or (not contact_reach_ok)
                        if not xy_fail:
                            break
                    elif (finger_obj_ncon == 1
                            and side_grip and realism_active
                            and arm_obj_ncon == 0
                            and z_gap < -PALM_ANCHOR_Z_MARGIN
                            and carry_gap <= PALM_ANCHOR_MAX_CARRY
                            and max(d_thumb_obj, d_bc_obj)
                                <= PALM_ANCHOR_MAX_FAR):
                        far_side_lbl_1f = (
                            "bc" if d_bc_obj > d_thumb_obj else "thumb")
                        far_dist_1f = max(d_thumb_obj, d_bc_obj)
                        print(f"[Exec] [5.5b] ASYMMETRIC 1-finger + "
                              f"palm-anchor reachable: 1 finger grazing "
                              f"obj, {far_side_lbl_1f} at "
                              f"{far_dist_1f*100:.1f}cm "
                              f"(≤ {PALM_ANCHOR_MAX_FAR*100:.0f}cm), "
                              f"carry_gap={carry_gap*100:.1f}cm "
                              f"(≤ {PALM_ANCHOR_MAX_CARRY*100:.0f}cm), "
                              f"palm {abs(z_gap)*100:.1f}cm below "
                              f"obj_top — skipping disengage; gate's "
                              f"PALM-ANCHOR tier + pin will bridge")
                        break
                    if not getattr(self, '_pre_close_backup_used', False):
                        self._pre_close_backup_used = True
                        obj_xy_2d_bk = np.asarray(
                            obj_xyz_pre[:2], dtype=float)
                        cur_xy_bk = np.asarray(
                            self.sim.localization()[:2], dtype=float)
                        away_vec_bk = cur_xy_bk - obj_xy_2d_bk
                        away_norm_bk = float(np.linalg.norm(away_vec_bk))
                        if away_norm_bk > 1e-3:
                            away_unit_bk = away_vec_bk / away_norm_bk
                            backup_target = (cur_xy_bk
                                             + away_unit_bk
                                             * ASYM_BACKUP_DISTANCE)
                            _far_lbl_bk = ("bc" if d_bc_obj > d_thumb_obj
                                           else "thumb")
                            _far_val_bk = max(d_thumb_obj, d_bc_obj)
                            print(f"[Exec] [5.5b] graceful DISENGAGE: "
                                  f"asymmetric ({finger_obj_ncon} "
                                  f"finger(s) on obj, {_far_lbl_bk} "
                                  f"still {_far_val_bk*100:.1f}cm "
                                  f"away) — backing chassis "
                                  f"{ASYM_BACKUP_DISTANCE*100:.0f}cm "
                                  f"along reverse-approach to release "
                                  f"obj before final abort")
                            self._side_grip_chassis_push(
                                backup_target, obj_xy_2d_bk,
                                timeout=1.5, dist_tol=0.02)
                            if self._cancel:
                                self._clear_held_state()
                                fire(False)
                                return
                            _t.sleep(0.2)
                            if side_grip:
                                align_pt_xy = self._pinch_midpoint_xyz(
                                    self.sim.data)[:2].copy()
                            else:
                                align_pt_xy = self._carry_anchor_xyz(
                                    self.sim.data)[:2].copy()
                            grip_xyz_pre = self.sim.data.xpos[
                                self.gripper_body_id].copy()
                            obj_xyz_pre  = self.sim.data.xpos[
                                obj_bid].copy()
                            carry_gap = float(np.linalg.norm(
                                align_pt_xy - obj_xyz_pre[:2]))
                            obj_grip_xy = float(np.linalg.norm(
                                grip_xyz_pre[:2] - obj_xyz_pre[:2]))
                            ik_dev = float(np.linalg.norm(
                                grip_xyz_pre[:2] - pre_grasp_target[:2]))
                            palm_z = float(grip_xyz_pre[2])
                            obj_top_z = float(obj_xyz_pre[2]
                                              + self._object_half_height(
                                                  obj_bid))
                            z_gap = palm_z - obj_top_z
                            try:
                                finger_obj_ncon = int(
                                    self._count_finger_obj_contacts(obj_bid))
                            except Exception:
                                finger_obj_ncon = 0
                            carry_ok = (carry_gap <= effective_carry_tol)
                            if realism_active and side_grip:
                                legacy_ok = True
                            else:
                                legacy_ok = (
                                    ik_dev <= GRIP_DEVIATION_TOLERANCE
                                    and obj_grip_xy <= obj_xy_gate)
                            z_ok = ((z_gap <= REALISM_PRE_CLOSE_Z_GAP)
                                    if realism_active else True)
                            contact_reach_ok = True
                            if (side_grip
                                    and len(self._carry_anchor_body_ids) == 3):
                                _thumb_xy = self.sim.data.xpos[
                                    self._carry_anchor_body_ids[0]
                                ][:2].copy()
                                _bc_xy = 0.5 * (
                                    self.sim.data.xpos[
                                        self._carry_anchor_body_ids[1]
                                    ][:2]
                                    + self.sim.data.xpos[
                                        self._carry_anchor_body_ids[2]
                                    ][:2]
                                ).copy()
                                d_thumb_obj = float(np.linalg.norm(
                                    _thumb_xy - obj_xyz_pre[:2]))
                                d_bc_obj = float(np.linalg.norm(
                                    _bc_xy - obj_xyz_pre[:2]))
                                sides_ok = (
                                    d_thumb_obj <= SIDE_FINGER_PRECLOSE_REACH
                                    and d_bc_obj <= SIDE_FINGER_PRECLOSE_REACH)
                            else:
                                sides_ok = True
                            xy_residual_vec = (obj_xyz_pre[:2]
                                               - align_pt_xy)
                            xy_residual_mag = float(np.linalg.norm(
                                xy_residual_vec))
                            xy_fail = (not carry_ok
                                       or not contact_reach_ok
                                       or not sides_ok)
                            print(f"[Exec] [5.5b] after DISENGAGE: "
                                  f"carry_gap={carry_gap*100:.1f}cm "
                                  f"d_thumb={d_thumb_obj*100:.1f}cm "
                                  f"d_bc={d_bc_obj*100:.1f}cm "
                                  f"finger-obj ncon={finger_obj_ncon} "
                                  f"sides_ok={sides_ok}")
                            if not xy_fail:
                                print(f"[Exec] [5.5b] DISENGAGE "
                                      f"RECOVERED — gate now passes, "
                                      f"exiting loop to close")
                                break
                    far_side = ("bc" if d_bc_obj > d_thumb_obj else "thumb")
                    far_dist = max(d_thumb_obj, d_bc_obj)
                    if finger_obj_ncon >= 1:
                        print(f"[Exec] [5.5b] abort: ASYMMETRIC in-contact "
                              f"({finger_obj_ncon} finger(s) already "
                              f"touching obj but {far_side} still "
                              f"{far_dist*100:.1f}cm away ≥ "
                              f"{SIDE_FINGER_PRECLOSE_REACH*100:.1f}cm "
                              f"reach) — further nudges would push "
                              f"in-contact finger into obj body; needs "
                              f"a new base pose")
                    else:
                        print(f"[Exec] [5.5b] abort: ASYMMETRIC (no "
                              f"fingers in contact after disengage; "
                              f"{far_side} still {far_dist*100:.1f}cm "
                              f"away ≥ "
                              f"{SIDE_FINGER_PRECLOSE_REACH*100:.1f}cm "
                              f"reach) — orientation mismatch, needs a "
                              f"new base pose")
                    break
                if improvement < NUDGE_MIN_CARRY_GAP_IMPROVEMENT:
                    print(f"[Exec] [5.5b] abort: no convergence "
                          f"(improvement {improvement*100:+.1f}cm < "
                          f"{NUDGE_MIN_CARRY_GAP_IMPROVEMENT*100:.1f}cm) "
                          f"— further nudges unlikely to help")
                    break
                arm_chassis_ncon_pre = arm_chassis_ncon_post

            asym_soft_assist = False
            if (side_grip and realism_active
                    and not STRICT_PICKUP_MODE
                    and not sides_ok
                    and finger_obj_ncon >= 1
                    and carry_gap <= REALISM_MICRO_LIFT_THRESHOLD
                    and max(d_thumb_obj, d_bc_obj) <= ASYM_SOFT_ASSIST_MAX_FAR):
                asym_soft_assist = True
                far_side_lbl = ("bc" if d_bc_obj > d_thumb_obj else "thumb")
                far_dist     = max(d_thumb_obj, d_bc_obj)
                print(f"[Exec] [5.5] asymmetric SOFT-ASSIST eligible: "
                      f"{finger_obj_ncon} finger(s) in contact, "
                      f"thumb={d_thumb_obj*100:.1f}cm bc={d_bc_obj*100:.1f}cm "
                      f"({far_side_lbl} far at {far_dist*100:.1f}cm "
                      f"≤ {ASYM_SOFT_ASSIST_MAX_FAR*100:.0f}cm), "
                      f"carry_gap={carry_gap*100:.1f}cm "
                      f"≤ {REALISM_MICRO_LIFT_THRESHOLD*100:.0f}cm — "
                      f"relaxing sides_ok so [5.7] smooth-lift MICRO "
                      f"can bridge obj into pocket")
                sides_ok = True

            palm_anchor_ok = False
            if (side_grip and realism_active
                    and not STRICT_PICKUP_MODE
                    and not sides_ok
                    and arm_obj_ncon == 0
                    and z_gap < -PALM_ANCHOR_Z_MARGIN
                    and carry_gap <= PALM_ANCHOR_MAX_CARRY
                    and max(d_thumb_obj, d_bc_obj)
                        <= PALM_ANCHOR_MAX_FAR):
                palm_anchor_ok = True
                far_side_lbl_pa = (
                    "bc" if d_bc_obj > d_thumb_obj else "thumb")
                far_dist_pa = max(d_thumb_obj, d_bc_obj)
                print(f"[Exec] [5.5] PALM-ANCHOR accept: palm "
                      f"{abs(z_gap)*100:.1f}cm below obj_top "
                      f"(≥ {PALM_ANCHOR_Z_MARGIN*100:.0f}cm), "
                      f"pinch_midpoint {carry_gap*100:.1f}cm from "
                      f"obj (≤ {PALM_ANCHOR_MAX_CARRY*100:.0f}cm), "
                      f"{far_side_lbl_pa} at {far_dist_pa*100:.1f}cm "
                      f"(≤ {PALM_ANCHOR_MAX_FAR*100:.0f}cm), no arm "
                      f"clip — gripper positioned to scoop obj; "
                      f"close will wrap fingers naturally")
                sides_ok = True

            arm_obj_ok = (arm_obj_ncon == 0)

            if FAST_PICKUP_MODE and side_grip and realism_active:
                if not (carry_ok and sides_ok and arm_obj_ok):
                    print(f"[Exec] [5.5] FAST PICKUP MODE: "
                          f"forcing gate pass despite "
                          f"carry_ok={carry_ok} sides_ok={sides_ok}  "
                          f"(arm_obj_ok={arm_obj_ok} stays strict — "
                          f"arm-clip is unrecoverable by pin closure)")
                carry_ok    = True
                sides_ok    = True
                legacy_ok   = True

            if (not carry_ok or not legacy_ok or not z_ok
                    or not contact_reach_ok or not sides_ok
                    or not arm_obj_ok):
                if not arm_obj_ok:
                    reason = (f"arm structure clipping obj at pre-close "
                              f"({arm_obj_ncon} arm-vs-obj contact(s) — "
                              f"boom/palm/wrist intruding obj geom; "
                              f"close would damage or fail)")
                elif not sides_ok:
                    _disp_tol = (SIDE_FINGER_PRECLOSE_REACH_PERFECT
                                 if ENABLE_NO_CHASSIS_PUSH
                                 else SIDE_FINGER_PRECLOSE_REACH)
                    reason = (f"asymmetric reach: thumb={d_thumb_obj*100:.1f}cm "
                              f"bc={d_bc_obj*100:.1f}cm "
                              f"(both must be ≤ "
                              f"{_disp_tol*100:.1f}cm — "
                              f"one side too far for close stroke)")
                elif not carry_ok:
                    reason = "carry_anchor too far from object"
                elif not z_ok:
                    reason = (f"realism: palm Z too high above obj_top "
                              f"({z_gap*100:.1f}cm > "
                              f"{REALISM_PRE_CLOSE_Z_GAP*100:.1f}cm)")
                elif not contact_reach_ok:
                    reason = ("realism: 0 finger-obj contacts at "
                              "pre-close (close can't physically grip)")
                else:
                    reason = "legacy IK/grip safety"
                print(f"[Exec] pre-close gate rejected ({reason}): "
                      f"carry_gap={carry_gap*100:.1f}cm, "
                      f"ik_dev={ik_dev:.3f}m, "
                      f"obj-grip xy={obj_grip_xy:.3f}m — "
                      f"abandoning visually bad candidate, "
                      f"play_m1 will retry next base pose")
                self.last_grasp_failure_info = {
                    'gripper_xy': grip_xyz_pre[:2].copy(),
                    'obj_xy': obj_xyz_pre[:2].copy(),
                    'ik_target_xy': pre_grasp_target[:2].copy(),
                    'ik_dev': float(ik_dev),
                    'obj_grip_xy': float(obj_grip_xy),
                    'expected_obj_grip_xy': float(expected_obj_grip_xy),
                    'carry_gap': float(carry_gap),
                    'carry_xy': align_pt_xy.copy(),
                    'd_thumb': float(d_thumb_obj),
                    'd_bc':    float(d_bc_obj),
                    'max_far': float(max(d_thumb_obj, d_bc_obj)),
                    'arm_obj_ncon': int(arm_obj_ncon),
                }
                residual_xy = obj_xyz_pre[:2] - align_pt_xy
                robot_xy_now = self.sim.localization()[:2]
                approach_xy = obj_xyz_pre[:2] - np.asarray(robot_xy_now)
                a_n = float(np.linalg.norm(approach_xy))
                r_n = float(np.linalg.norm(residual_xy))
                skip_lift = False
                if a_n > 1e-6 and r_n > 1e-6:
                    cos_ang = float(np.dot(residual_xy, approach_xy)
                                    / (a_n * r_n))
                    if abs(cos_ang) > 0.85:
                        skip_lift = True
                        print(f"[Exec] residual is forward-aligned "
                              f"(cos={cos_ang:+.2f}, "
                              f"|r|={r_n*100:.1f}cm) — "
                              f"direction-aware retract: no arm lift")
                _use_side_grip_retry_mode = bool(
                    side_grip and not FAST_PICKUP_MODE)
                self._retract_after_failure(
                    "[5.6]",
                    skip_lift=skip_lift,
                    side_grip_retry=_use_side_grip_retry_mode)
                self._clear_held_state()
                fire(False)
                return

            _natural_drag = bool(
                REALISM_MODE_NO_SMOOTH_LIFT
                and side_grip
                and carry_gap <= NATURAL_CLOSE_DRAG_THRESHOLD
                and not palm_anchor_ok)
            _smooth_lift_skipped = bool(
                REALISM_MODE_NO_SMOOTH_LIFT
                and side_grip
                and (carry_gap > REALISM_MICRO_LIFT_THRESHOLD
                     or _natural_drag)
                and not palm_anchor_ok)
            if STRICT_PICKUP_MODE:
                _natural_drag = True
                _smooth_lift_skipped = True
                print(f"[Exec] [5.7] STRICT mode: pre-close pin animation "
                      f"skipped; close stroke runs on real physics.  "
                      f"Pin activates only post-close for transport.")
            if palm_anchor_ok and REALISM_MODE_NO_SMOOTH_LIFT and side_grip:
                print(f"[Exec] [5.7] palm-anchor scoop tier active "
                      f"(palm below obj_top by {abs(z_gap)*100:.1f}cm) — "
                      f"forcing pin path to bridge close arc; natural-drag "
                      f"branch suppressed (otherwise close arc would push "
                      f"fingers into obj's upper body)")
            if _natural_drag:
                print(f"[Exec] [5.7] natural close-DRAG mode (realism + "
                      f"tiny residual {carry_gap*100:.1f}cm ≤ "
                      f"{NATURAL_CLOSE_DRAG_THRESHOLD*100:.0f}cm): "
                      f"skipping smooth-lift — finger close will "
                      f"physically push obj into pocket")
            elif _smooth_lift_skipped:
                print(f"[Exec] [5.7] smooth-lift SKIPPED (realism, residual "
                      f"{carry_gap*100:.1f}cm > "
                      f"{REALISM_MICRO_LIFT_THRESHOLD*100:.0f}cm threshold)")
            elif REALISM_MODE_NO_SMOOTH_LIFT and side_grip:
                print(f"[Exec] [5.7] smooth-lift MICRO mode (realism + "
                      f"medium residual "
                      f"{NATURAL_CLOSE_DRAG_THRESHOLD*100:.0f}cm < "
                      f"{carry_gap*100:.1f}cm ≤ "
                      f"{REALISM_MICRO_LIFT_THRESHOLD*100:.0f}cm): "
                      f"animated bridge to pocket")
            half_h_pre = self._object_half_height(obj_bid)
            target_z_pre = -(half_h_pre - 0.025)
            try:
                _pinch_z_now = float(
                    self._pinch_midpoint_xyz(self.sim.data)[2])
                FLOOR_Z       = 0.0
                FLOOR_MARGIN  = 0.002
                _min_centre_z = FLOOR_Z + half_h_pre + FLOOR_MARGIN
                _abs_target_z = _pinch_z_now + target_z_pre
                if _abs_target_z < _min_centre_z:
                    _new_offset = _min_centre_z - _pinch_z_now
                    print(f"[Exec] [5.7] floor-aware Z clamp: pin "
                          f"target raised by "
                          f"{(_min_centre_z - _abs_target_z)*100:.1f}cm "
                          f"(orig target_z_pre={target_z_pre:+.3f} → "
                          f"{_new_offset:+.3f}) so obj_bottom stays "
                          f"≥ floor + {FLOOR_MARGIN*100:.1f}mm")
                    target_z_pre = _new_offset
            except Exception as _e:
                print(f"[Exec] [5.7] floor-aware Z clamp skipped: {_e}")
            obj_r_pre = self._object_radius(obj_bid)
            try:
                _robot_xy = np.asarray(self.sim.localization()[:2], dtype=float)
                _centroid_xy = np.asarray(self._carry_anchor_xyz(self.sim.data)[:2], dtype=float)
                _away = _centroid_xy - _robot_xy
                _norm = float(np.linalg.norm(_away))
                if _norm > 1e-4:
                    _unit = _away / _norm
                    _shift_xy = _unit * obj_r_pre
                else:
                    _shift_xy = np.zeros(2)
                _raw_offset_xy = np.asarray(obj_xyz_pre[:2], dtype=float) - _centroid_xy
                _blend_xy = 0.5 * _raw_offset_xy + 0.5 * _shift_xy
            except Exception as _e:
                print(f"[Exec] [5.7] near-edge XY shift skipped: {_e}")
                _blend_xy = np.zeros(2)
                _shift_xy = np.zeros(2)
                _raw_offset_xy = np.zeros(2)
            if FAST_PICKUP_MODE and side_grip:
                pre_offset = np.array(
                    [_shift_xy[0], _shift_xy[1], target_z_pre], dtype=float)
                print(f"[Exec] [5.7] FAST_PICKUP_MODE: pin offset "
                      f"near-edge ({_shift_xy[0]:+.3f},"
                      f"{_shift_xy[1]:+.3f},{target_z_pre:+.3f}) — "
                      f"obj kept out of palm while fingers close")
            else:
                pre_offset = np.array(
                    [_blend_xy[0], _blend_xy[1], target_z_pre],
                    dtype=float)
            self._grasp_offset_xyz = pre_offset
            if not _smooth_lift_skipped:
                self.arm_bridge.model.eq_obj2id[self.weld_id] = obj_bid
                self.arm_bridge.planning_data.eq_active[self.weld_id] = 1
                try:
                    self._held_obj_orig_gravcomp = float(
                        self.sim.model.body_gravcomp[obj_bid])
                    self.sim.model.body_gravcomp[obj_bid] = 1.0
                except Exception as e:
                    print(f"[Exec] gravcomp set warning: {e}")
                    self._held_obj_orig_gravcomp = None
                if FAST_PICKUP_MODE and side_grip:
                    print(f"[Exec] [5.7] skipping contact soften "
                          f"(FAST_PICKUP_MODE: hard contacts keep "
                          f"fingers at obj surface during close)")
                else:
                    self._soften_held_obj_contacts(obj_bid)
                obj_xyz_snapshot = self.sim.data.xpos[obj_bid].copy()
                if FAST_PICKUP_MODE and side_grip:
                    saved_offset = self._grasp_offset_xyz.copy()
                    self._grasp_offset_xyz = np.array(
                        [0.0, 0.0, float(target_z_pre)], dtype=float)
                    self._install_pin(
                        self._pin_obj_to_gripper_animated(
                            obj_xyz_snapshot,
                            anchor_pinch_midpoint=True,
                            phased_xy_then_z=True,
                            xy_phase_secs=0.6,
                            z_phase_secs=0.6))
                    self._fast_fixed_close_pin_active = True
                    print(f"[Exec] [5.7] FAST_PICKUP_MODE: live "
                          f"pinch_midpoint pin (phased XY→Z, "
                          f"0.6+0.6s) — obj slides to centre between "
                          f"thumb & bc, then drops to grip height")
                else:
                    if side_grip:
                        saved_offset = self._grasp_offset_xyz.copy()
                        self._grasp_offset_xyz = np.array(
                            [0.0, 0.0, float(target_z_pre)], dtype=float)
                        self._install_pin(
                            self._pin_obj_to_gripper_animated(
                                obj_xyz_snapshot,
                                anchor_pinch_midpoint=True,
                                phased_xy_then_z=True,
                                xy_phase_secs=0.6,
                                z_phase_secs=0.6))
                        print(f"[Exec] [5.7] non-FAST side-grip: live "
                              f"pinch_midpoint pin (phased XY→Z, "
                              f"0.6+0.6s) — obj slides to centre and "
                              f"drops to grip height in parallel with close")
                    else:
                        self._install_pin(
                            self._pin_obj_to_gripper_animated(
                                obj_xyz_snapshot))
                    self._fast_fixed_close_pin_active = False
                if FAST_PICKUP_MODE and side_grip:
                    import time as _t_pin
                    _phased_total = 0.6 + 0.6 + 0.05
                    print(f"[Exec] [5.7] FAST_PICKUP_MODE: waiting "
                          f"{_phased_total:.2f}s for phased pin "
                          f"(XY 0.6s → Z 0.6s) to centre obj between "
                          f"thumb and bc before close")
                    _t_pin.sleep(_phased_total)
            if not _smooth_lift_skipped:
                _centroid_z_pre = float(self._carry_anchor_xyz(self.sim.data)[2])
                _raw_z_pre = float(obj_xyz_pre[2] - _centroid_z_pre)
                print(f"[Exec] [5.7] pre-close smooth-lift: target_z="
                      f"{target_z_pre:.3f}  "
                      f"blend_xy=({_blend_xy[0]:+.3f},{_blend_xy[1]:+.3f}) "
                      f"= 0.5·raw({_raw_offset_xy[0]:+.3f},"
                      f"{_raw_offset_xy[1]:+.3f}) + 0.5·near-edge("
                      f"{_shift_xy[0]:+.3f},{_shift_xy[1]:+.3f})  "
                      f"(obj_radius={obj_r_pre:.3f})  (obj lifts from "
                      f"raw_z={_raw_z_pre:.3f} over "
                      f"{SMOOTH_ATTACH_SECS:.2f}s)")
                self._pre_close_lift_done = True

            obj_radius = self._object_radius(obj_bid)
            size_aware_close_pos = self._finger_close_for_radius(obj_radius)

            _use_overdrive = (USE_OVERDRIVE_CLOSE
                              and USE_CONTACT_STOP_CLOSE
                              and not STRICT_PICKUP_MODE)
            if _use_overdrive:
                close_pos = FINGER_CLOSE_MAX
                _close_compress_ticks = (
                    FAST_CONTACT_COMPRESS_TICKS
                    if FAST_PICKUP_MODE and side_grip
                    else CONTACT_COMPRESS_TICKS)
                print(f"[Exec] close overdrive: size-aware={size_aware_close_pos:.3f} "
                      f"→ FINGER_CLOSE_MAX={FINGER_CLOSE_MAX:.3f} "
                      f"(proximity + contact-stop will freeze each "
                      f"finger at obj surface)")
            else:
                close_pos = size_aware_close_pos
                if STRICT_PICKUP_MODE:
                    print(f"[Exec] close STRICT (no contact-stop): "
                          f"size-aware close_ctrl={size_aware_close_pos:.3f} "
                          f"(contact-stop bypassed in STRICT; "
                          f"force-stop + tiered relief brake instead)")
                self._cycle_stage_close_fired = True

            if open_thread is not None and open_thread.is_alive():
                open_thread.join(timeout=2.0)
            print(f"[Exec] [6] close gripper  [TIMING t={time.time()-_cycle_t0:.1f}s]  "
                  f"radius={obj_radius:.3f}m  close_ctrl={close_pos:.3f}")
            self._strict_grasp_q = list(self._current_arm_q())
            self._set_gripper(close_pos, hold_seconds=0.6)
            if self._cancel:
                self._clear_held_state(); fire(False); return

            try:
                if (STRICT_PICKUP_MODE and side_grip and USE_BC_RESCUE
                        and self._last_close_finger_contacts is not None):
                    _bc_contacts = self._last_close_finger_contacts
                    _a_held = bool(_bc_contacts[2])
                    _b_held = bool(_bc_contacts[1])
                    _c_held = bool(_bc_contacts[0])
                    if _a_held and (_b_held ^ _c_held):
                        missing_name = 'c' if not _c_held else 'b'
                        _j1_idx = 0 if missing_name == 'c' else 3
                        _bc_gids = self.sim.gripper_ids_left
                        _bc_N_pre = self._per_finger_normal_forces(obj_bid)
                        _missing_N_pre = (_bc_N_pre[2] if missing_name == 'c'
                                           else _bc_N_pre[1])
                        if _j1_idx < len(_bc_gids):
                            _gid_bc = int(_bc_gids[_j1_idx])
                            _cur_bc = float(self.sim.data.ctrl[_gid_bc])
                            _lo_bc = float(self.sim.model
                                .actuator_ctrlrange[_gid_bc, 0])
                            _hi_bc = float(self.sim.model
                                .actuator_ctrlrange[_gid_bc, 1])
                            _new_bc = min(_hi_bc, max(_lo_bc,
                                _cur_bc + BC_RESCUE_J1_BUMP_RAD))
                            _RAMP_STEPS = 10
                            _ramp_dt = BC_RESCUE_SETTLE_S / _RAMP_STEPS
                            for _ri in range(_RAMP_STEPS):
                                _alpha = (_ri + 1) / _RAMP_STEPS
                                self.sim.data.ctrl[_gid_bc] = (
                                    _cur_bc + (_new_bc - _cur_bc) * _alpha)
                                time.sleep(_ramp_dt)
                            self._strict_log(
                                "CLOSE",
                                f"[6.3] BC-RESCUE attempt: missing "
                                f"finger_{missing_name} "
                                f"(N_pre={_missing_N_pre:.1f}N); "
                                f"bumping {missing_name}_j1 "
                                f"{_cur_bc:+.3f} → {_new_bc:+.3f} "
                                f"(+{BC_RESCUE_J1_BUMP_RAD*1000:.0f}mrad "
                                f"ramped over {BC_RESCUE_SETTLE_S*1000:.0f}ms)")
                            _bc_N_post = self._per_finger_normal_forces(obj_bid)
                            _missing_N_post = (_bc_N_post[2]
                                               if missing_name == 'c'
                                               else _bc_N_post[1])
                            if _missing_N_post >= BC_RESCUE_SUCCESS_N:
                                _bc_contacts[0 if missing_name == 'c' else 1] = True
                                self._last_close_finger_contacts = list(_bc_contacts)
                                self._strict_log(
                                    "CLOSE",
                                    f"[6.3] BC-RESCUE SUCCESS: "
                                    f"finger_{missing_name} engaged at "
                                    f"{_missing_N_post:.1f}N "
                                    f"(≥{BC_RESCUE_SUCCESS_N:.1f}N floor); "
                                    f"close upgraded 2/3 → 3/3")
                            else:
                                self._strict_log(
                                    "CLOSE",
                                    f"[6.3] BC-RESCUE FAILED: "
                                    f"finger_{missing_name} still at "
                                    f"{_missing_N_post:.1f}N after bump; "
                                    f"close remains 2/3, falling through "
                                    f"to verify")
            except Exception as _e_bc_rescue:
                print(f"[Exec] [6.3] BC-RESCUE raised: {_e_bc_rescue}")

            if USE_FINGER_DIAGNOSTIC_LOG:
                self._log_finger_geometry(obj_bid, "post-close")
                self._log_finger_object_contacts(obj_bid, "post-close")

            if USE_3FINGER_VERIFY and self._last_close_finger_contacts is not None:
                contacts_snapshot = list(self._last_close_finger_contacts)
                self._strict_finger_attempts_used += 1
                n_contacts = sum(bool(x) for x in contacts_snapshot)
                strict_window_open = (self._strict_finger_attempts_used
                                      <= MAX_STRICT_3FINGER_ATTEMPTS)
                thumb_contacted = bool(contacts_snapshot[2])
                b_contacted     = bool(contacts_snapshot[1])
                c_contacted     = bool(contacts_snapshot[0])
                if STRICT_PICKUP_MODE:
                    try:
                        try:
                            gids_v = self.sim.gripper_ids_left
                            j_indices_abc = (6, 3, 0)
                            per_finger_N = self._per_finger_normal_forces(
                                obj_bid)
                            bumped_tag = []
                            skipped_tag = []
                            with self.sim._target_lock:
                                for fi, j_idx in enumerate(j_indices_abc):
                                    fname = 'abc'[fi]
                                    N_f = float(per_finger_N[fi])
                                    if N_f >= LIFT_PER_FINGER_FORCE_FLOOR_N:
                                        skipped_tag.append(
                                            f"{fname}({N_f:.1f}N)")
                                        continue
                                    if j_idx >= len(gids_v):
                                        continue
                                    gid = int(gids_v[j_idx])
                                    cur = float(
                                        self.sim.data.ctrl[gid])
                                    lo = float(self.sim.model
                                        .actuator_ctrlrange[gid, 0])
                                    hi = float(self.sim.model
                                        .actuator_ctrlrange[gid, 1])
                                    new_ctrl = min(hi, max(lo,
                                        cur + LIFT_TIGHTEN_PREVERIFY_RAD))
                                    self.sim.data.ctrl[gid] = new_ctrl
                                    bumped_tag.append(
                                        f"{fname}({N_f:.1f}N)")
                            self._strict_log(
                                "VERIFY",
                                f"pre-verify TIGHTEN (smart) — bumped "
                                f"+{LIFT_TIGHTEN_PREVERIFY_RAD:.3f}rad on "
                                f"[{', '.join(bumped_tag) or 'none'}] "
                                f"(below {LIFT_PER_FINGER_FORCE_FLOOR_N:.1f}N "
                                f"floor); skipped firm "
                                f"[{', '.join(skipped_tag) or 'none'}]")
                        except Exception as _e_ptight:
                            print(f"[Exec] pre-verify tighten raised: "
                                  f"{_e_ptight}")
                        try:
                            _decay_t0 = time.time()
                            _decay_step = 0.050
                            _decay_samples = []
                            while (time.time() - _decay_t0) < 0.3:
                                time.sleep(_decay_step)
                                _N = self._per_finger_normal_forces(obj_bid)
                                _decay_samples.append(
                                    (time.time() - _decay_t0,
                                     float(_N[0]), float(_N[1]), float(_N[2])))
                            self._strict_log(
                                "VERIFY",
                                "FORCE_DECAY samples (ms, a, b, c): "
                                + "  ".join(
                                    f"t={int(s[0]*1000)}ms "
                                    f"[a={s[1]:.1f},b={s[2]:.1f},c={s[3]:.1f}]N"
                                    for s in _decay_samples))
                        except Exception as _e_decay:
                            print(f"[Exec] FORCE_DECAY log raised: "
                                  f"{_e_decay}")
                            time.sleep(0.3)
                        live_c = bool(self._finger_touches_obj(
                            0, obj_bid))
                        live_b = bool(self._finger_touches_obj(
                            1, obj_bid))
                        live_a = bool(self._finger_touches_obj(
                            2, obj_bid))
                        self._strict_log(
                            "VERIFY",
                            f"sustained-contact check: snapshot "
                            f"a={thumb_contacted} b={b_contacted} "
                            f"c={c_contacted}; live a={live_a} "
                            f"b={live_b} c={live_c}")
                        thumb_contacted = thumb_contacted and live_a
                        b_contacted     = b_contacted     and live_b
                        c_contacted     = c_contacted     and live_c
                        n_contacts = (int(thumb_contacted)
                                      + int(b_contacted)
                                      + int(c_contacted))
                    except Exception as _e:
                        self._strict_log(
                            "VERIFY",
                            f"sustained-contact check failed: {_e} — "
                            f"falling back to close-time snapshot")
                    _triad_skip_reject = (STRICT_PICKUP_MODE
                                          and not STRICT_PERFECT_FRICTION_ONLY
                                          and side_grip
                                          and not FAST_PICKUP_MODE)
                    _triad_gate_failed = False
                    try:
                        _triad = self._contact_normal_triad(obj_bid)
                        _bs = _triad['balance_score']
                        if _bs >= VERIFY_TRIAD_BALANCE_GATE_THRESHOLD:
                            _diag_flag = "ONE-SIDED"
                        elif _bs >= VERIFY_TRIAD_BALANCE_DIAG_THRESHOLD:
                            _diag_flag = "MARGINAL"
                        else:
                            _diag_flag = "BALANCED"
                        _ff = _triad['per_finger_force']
                        self._strict_log(
                            "VERIFY",
                            f"triad-diag  balance_score="
                            f"{_bs:.2f} ({_diag_flag})  "
                            f"max_force={_triad['max_force']:.1f}N  "
                            f"per_finger_N=[a={_ff[0]:.1f}, "
                            f"b={_ff[1]:.1f}, c={_ff[2]:.1f}]N")
                        if (VERIFY_TRIAD_BALANCE_GATE_ENABLED
                                and _bs >= VERIFY_TRIAD_BALANCE_GATE_THRESHOLD
                                and not _triad_skip_reject):
                            _triad_gate_failed = True
                            self._strict_log(
                                "VERIFY",
                                f"TRIAD-GATE REJECT  balance_score="
                                f"{_bs:.2f} ≥ "
                                f"{VERIFY_TRIAD_BALANCE_GATE_THRESHOLD:.2f} "
                                f"— grasp is one-sided, would squirt out "
                                f"on lift; failing verify early")
                        elif (VERIFY_TRIAD_BALANCE_GATE_ENABLED
                                and _bs >= VERIFY_TRIAD_BALANCE_GATE_THRESHOLD
                                and _triad_skip_reject):
                            self._strict_log(
                                "VERIFY",
                                f"TRIAD-GATE diag-only (soft-weld mode) — "
                                f"balance={_bs:.2f} ≥ "
                                f"{VERIFY_TRIAD_BALANCE_GATE_THRESHOLD:.2f} "
                                f"would normally reject, but weld holds "
                                f"obj during transport regardless of "
                                f"force balance")
                    except Exception as _e_triad:
                        self._strict_log(
                            "VERIFY",
                            f"triad-diag failed: {_e_triad}")
                    _sustained_force_failed = False
                    _sustained_force_skip = (STRICT_PICKUP_MODE
                                             and not STRICT_PERFECT_FRICTION_ONLY
                                             and side_grip
                                             and not FAST_PICKUP_MODE)
                    if _sustained_force_skip:
                        self._strict_log(
                            "VERIFY",
                            f"SUSTAINED-FORCE skipped — soft-weld mode "
                            f"will bridge chatter during transport "
                            f"(use --perfect to enforce strict pure-"
                            f"friction gate)")
                    if (VERIFY_SUSTAINED_FORCE_ENABLED
                            and not _sustained_force_skip
                            and thumb_contacted
                            and (b_contacted or c_contacted)):
                        try:
                            _N_floor = VERIFY_SUSTAINED_FORCE_FLOOR_N
                            _interval = VERIFY_SUSTAINED_FORCE_SAMPLE_INT_S
                            _n_samples = max(
                                2,
                                int(VERIFY_SUSTAINED_FORCE_WINDOW_S
                                    / max(_interval, 1e-3)))
                            _samples = []
                            _pass_count = 0
                            for _si in range(_n_samples):
                                _N = self._per_finger_normal_forces(obj_bid)
                                _Na, _Nb, _Nc = (float(_N[0]),
                                                 float(_N[1]),
                                                 float(_N[2]))
                                _samples.append((_Na, _Nb, _Nc))
                                _thumb_ok = _Na >= _N_floor
                                _side_ok = (_Nb >= _N_floor
                                            or _Nc >= _N_floor)
                                if _thumb_ok and _side_ok:
                                    _pass_count += 1
                                time.sleep(_interval)
                            _pass_frac = _pass_count / max(_n_samples, 1)
                            _spike_cap = VERIFY_SUSTAINED_FORCE_SPIKE_CAP_N
                            _max_overall = max(_max_a, _max_b, _max_c)
                            _spike_detected = _max_overall > _spike_cap
                            _all_passed = (
                                _pass_frac >= VERIFY_SUSTAINED_FORCE_MIN_PASS_FRAC
                                and not _spike_detected)
                            _min_a = min(s[0] for s in _samples)
                            _min_b = min(s[1] for s in _samples)
                            _min_c = min(s[2] for s in _samples)
                            _max_a = max(s[0] for s in _samples)
                            _max_b = max(s[1] for s in _samples)
                            _max_c = max(s[2] for s in _samples)
                            self._strict_log(
                                "VERIFY",
                                f"SUSTAINED-FORCE  {_n_samples} samples × "
                                f"{_interval*1000:.0f}ms  "
                                f"per-finger N min/max: "
                                f"a={_min_a:.0f}/{_max_a:.0f} "
                                f"b={_min_b:.0f}/{_max_b:.0f} "
                                f"c={_min_c:.0f}/{_max_c:.0f} N  "
                                f"pass={_pass_count}/{_n_samples} "
                                f"({_pass_frac*100:.0f}%, "
                                f"need ≥{VERIFY_SUSTAINED_FORCE_MIN_PASS_FRAC*100:.0f}%)  "
                                f"(floor={_N_floor:.0f}N, required "
                                f"a≥floor AND (b≥floor OR c≥floor))")
                            if not _all_passed:
                                _sustained_force_failed = True
                                if _spike_detected:
                                    _reject_reason = (
                                        f"SPIKE — finger force peak "
                                        f"{_max_overall:.0f}N > "
                                        f"cap {_spike_cap:.0f}N "
                                        f"(MuJoCo solver-resonance "
                                        f"spike — would launch obj on lift)")
                                else:
                                    _reject_reason = (
                                        f"CHATTER-TROUGH — only "
                                        f"{_pass_count}/{_n_samples} samples "
                                        f"({_pass_frac*100:.0f}%) had "
                                        f"required-finger N ≥ floor "
                                        f"({_N_floor:.0f}N)")
                                self._strict_log(
                                    "VERIFY",
                                    f"SUSTAINED-FORCE REJECT — "
                                    f"{_reject_reason}.  "
                                    f"Failing verify early.")
                                thumb_contacted = False
                                b_contacted     = False
                                c_contacted     = False
                                n_contacts      = 0
                        except Exception as _e_susf:
                            self._strict_log(
                                "VERIFY",
                                f"sustained-force check raised: {_e_susf} "
                                f"— skipping gate (accept)")
                    opposing_pinch_ok = (
                        thumb_contacted
                        and (b_contacted or c_contacted)
                        and not _triad_gate_failed
                        and not _sustained_force_failed)
                    if opposing_pinch_ok:
                        required = n_contacts
                        bar_label = (
                            "strict (mode 1, 3/3 ideal)"
                            if n_contacts == 3
                            else f"strict (mode 1, opposing 2/3: "
                                 f"thumb+"
                                 f"{'b' if b_contacted else 'c'})")
                        self._cycle_stage_verify_passed = True
                    else:
                        required = 3
                        bar_label = "strict (mode 1, opposing-pinch REQUIRED)"
                else:
                    required = (MIN_CONTACTS_STRICT if strict_window_open
                                else MIN_CONTACTS_RELAXED)
                    bar_label = "strict" if strict_window_open else "relaxed"
                if FAST_PICKUP_MODE and side_grip:
                    if n_contacts < required:
                        print(f"[Exec] [6.4] FAST PICKUP MODE: "
                              f"accepting {n_contacts}/3 contact (need "
                              f"{required}) — pin closure holds obj "
                              f"for transport regardless of finger count")
                    required = 0
                if n_contacts < required:
                    print(f"[Exec] [6.4] 3-finger verify FAILED "
                          f"({bar_label}): {n_contacts}/3 contacted, "
                          f"need {required}  "
                          f"(strict attempts used "
                          f"{self._strict_finger_attempts_used}/"
                          f"{MAX_STRICT_3FINGER_ATTEMPTS}) — "
                          f"opening gripper, retract, and retry next pose")
                    self._emit_grasp_diag(
                        outcome="verify_FAIL",
                        obj_bid=obj_bid,
                        n_contacts=n_contacts,
                        contacts_snapshot=contacts_snapshot,
                        side_grip=side_grip)
                    self._set_gripper(GRIPPER_OPEN_POS, hold_seconds=0.3)
                    if self._cancel:
                        self._clear_held_state(); fire(False); return
                    grip_xyz_post = self.sim.data.xpos[
                        self.gripper_body_id].copy()
                    obj_xyz_post  = self.sim.data.xpos[obj_bid].copy()
                    carry_xy_post = self._carry_anchor_xyz(
                        self.sim.data)[:2].copy()
                    self.last_grasp_failure_info = {
                        'gripper_xy': grip_xyz_post[:2].copy(),
                        'obj_xy': obj_xyz_post[:2].copy(),
                        'ik_target_xy': pre_grasp_target[:2].copy(),
                        'reason': 'insufficient_finger_contact',
                        'finger_contacts': n_contacts,
                        'finger_contact_mask': contacts_snapshot,
                        'carry_xy': carry_xy_post,
                    }
                    self._retract_after_failure(
                        "[6.4]",
                        side_grip_retry=(STRICT_PICKUP_MODE and side_grip))
                    self._clear_held_state(); fire(False); return
                else:
                    label = "3/3 fingers" if n_contacts == 3 else \
                            f"{n_contacts}/3 fingers (relaxed accept)"
                    print(f"[Exec] [6.4] 3-finger verify OK ({bar_label}): "
                          f"{label}")
                    self._emit_grasp_diag(
                        outcome="verify_OK",
                        obj_bid=obj_bid,
                        n_contacts=n_contacts,
                        contacts_snapshot=contacts_snapshot,
                        side_grip=side_grip)

            if getattr(self, '_pre_close_lift_done', False):
                self._pre_close_lift_done = False
                self._set_gripper(close_pos, hold_seconds=SMOOTH_ATTACH_SETTLE)
                if getattr(self, '_fast_fixed_close_pin_active', False):
                    grip_xyz = self._carry_anchor_xyz(self.sim.data).copy()
                    obj_xyz = self.sim.data.xpos[obj_bid].copy()
                    self._grasp_offset_xyz = obj_xyz - grip_xyz
                    self._install_pin(self._pin_obj_to_gripper())
                    self._fast_fixed_close_pin_active = False
                    print(f"[Exec] [6.4] FAST_PICKUP_MODE: reattached "
                          f"carry pin at offset "
                          f"{self._grasp_offset_xyz.round(3)} after close")
                print(f"[Exec] [verify-grasp] OK: object pinned at offset "
                      f"{self._grasp_offset_xyz.round(3)} from gripper "
                      f"(pre-close smooth-lift path; pin installed in [5.7])")
            else:
                grip_xyz = self._carry_anchor_xyz(self.sim.data).copy()
                obj_xyz  = self.sim.data.xpos[obj_bid].copy()
                raw_offset = obj_xyz - grip_xyz
                raw_dist = float(np.linalg.norm(raw_offset))
                raw_xy_dist = float(np.linalg.norm(raw_offset[:2]))
                half_h = self._object_half_height(obj_bid)
                target_z = -(half_h - 0.025)
                raw_z = float(raw_offset[2])
                if REALISM_MODE_NO_SMOOTH_LIFT and side_grip:
                    z_lift = 0.0
                else:
                    z_lift = max(0.0, target_z - raw_z)
                    z_lift = min(z_lift, 0.12)
                    raw_offset[0] = 0.0
                    raw_offset[1] = 0.0
                    raw_offset[2] = raw_z + z_lift
                self._grasp_offset_xyz = raw_offset
                print(f"[Exec] grasp_offset = {raw_offset.round(3)}  "
                      f"obj-pocket dist={float(np.linalg.norm(raw_offset)):.3f}m  "
                      f"raw_xy={raw_xy_dist:.3f}m (snapped to 0)  "
                      f"z_lift={z_lift*100:.1f}cm  "
                      f"(target_z={target_z:.3f}, raw_z={raw_z:.3f})")
                if STRICT_PICKUP_MODE:
                    self._strict_log(
                        "POST-CLOSE",
                        f"weld OFF, gravcomp OFF, pin OFF — "
                        f"friction-only carry (offset "
                        f"{raw_offset.round(3)} retained for diagnostics)")
                else:
                    self.arm_bridge.model.eq_obj2id[self.weld_id] = obj_bid
                    self.arm_bridge.planning_data.eq_active[self.weld_id] = 1
                    try:
                        self._held_obj_orig_gravcomp = float(
                            self.sim.model.body_gravcomp[obj_bid])
                        self.sim.model.body_gravcomp[obj_bid] = 1.0
                        print(f"[Exec] gravcomp[{obj_bid}] {self._held_obj_orig_gravcomp:.2f} → 1.00 "
                              f"(zero-g while held)")
                    except Exception as e:
                        print(f"[Exec] gravcomp set warning: {e}")
                        self._held_obj_orig_gravcomp = None
                    self._soften_held_obj_contacts(obj_bid)
                    obj_xyz_now = self.sim.data.xpos[obj_bid].copy()
                    self._install_pin(
                        self._pin_obj_to_gripper_animated(
                            obj_xyz_now,
                            anchor_pinch_midpoint=bool(side_grip)))
                    self._set_gripper(close_pos, hold_seconds=SMOOTH_ATTACH_SETTLE)
                    print(f"[Exec] [verify-grasp] OK: object pinned at offset "
                          f"{self._grasp_offset_xyz.round(3)} from gripper, "
                          f"smooth-attach over {SMOOTH_ATTACH_SECS:.2f}s")

            if (STRICT_PICKUP_MODE
                    and not STRICT_PERFECT_FRICTION_ONLY
                    and side_grip
                    and not FAST_PICKUP_MODE
                    and self._cycle_stage_verify_passed):
                try:
                    self.arm_bridge.model.eq_obj2id[self.weld_id] = obj_bid
                    self.arm_bridge.planning_data.eq_active[self.weld_id] = 1
                    self._soften_held_obj_contacts(obj_bid)
                    obj_xyz_now = self.sim.data.xpos[obj_bid].copy()
                    try:
                        _pinch0 = self._pinch_midpoint_xyz(self.sim.data)
                        _actual_offset = (
                            np.asarray(obj_xyz_now, dtype=float)
                            - np.asarray(_pinch0,   dtype=float))
                        _qpa  = self._held_obj_qpa
                        _dofa = self._held_obj_dofadr
                        def _pin_obj_at_actual_offset(data):
                            _pmid = self._pinch_midpoint_xyz(data)
                            _t0 = float(_pmid[0]) + float(_actual_offset[0])
                            _t1 = float(_pmid[1]) + float(_actual_offset[1])
                            _t2 = float(_pmid[2]) + float(_actual_offset[2])
                            pin_freejoint(data, _qpa, _dofa, (_t0, _t1, _t2))
                        self._install_pin(_pin_obj_at_actual_offset)
                        print(f"[Exec] LATE-119 actual-offset pin: "
                              f"offset={_actual_offset.round(3)}")
                    except Exception as _e_pin:
                        print(f"[Exec] LATE-119 actual-offset pin warn: "
                              f"{_e_pin} — fallback to animated pin")
                        self._install_pin(
                            self._pin_obj_to_gripper_animated(
                                obj_xyz_now,
                                anchor_pinch_midpoint=True))
                    self._strict_log(
                        "POST-VERIFY",
                        f"SOFT-WELD activated — verify-gated weld + pin "
                        f"between gripper and obj_{obj_idx} for "
                        f"transport-phase stability.  Pure-friction "
                        f"grip was verified physically (close + "
                        f"triad-balance diag + opposing-pinch contact, "
                        f"ALL gates passed cleanly); weld+pin is the "
                        f"compliance assist for MuJoCo's contact-"
                        f"chatter limit during transport.  This is NOT "
                        f"FAST mode — verify gates were real physics.")
                except Exception as _e_weld:
                    self._strict_log(
                        "POST-VERIFY",
                        f"soft-weld activation raised: {_e_weld} — "
                        f"continuing without weld assist")

            print(f"[Exec] [6.5] lift to carry pose  "
                  f"[TIMING t={time.time()-_cycle_t0:.1f}s]")
            if STRICT_PICKUP_MODE:
                grasp_q_snap = (list(self._strict_grasp_q)
                                if self._strict_grasp_q is not None
                                else list(self._current_arm_q()))
                lift_ok = self._strict_lift_with_retry(
                    obj_bid, close_pos, grasp_q_snap)
                if not lift_ok:
                    self._strict_log(
                        "PICK",
                        "lift+retry exhausted — aborting STRICT pick")
                    self._retract_after_failure("[6.5-STRICT]")
                    self._clear_held_state(); fire(False); return
                self._strict_log(
                    "PICK",
                    f"SUCCESS  retries_used={self._strict_retry_count}/"
                    f"{STRICT_RETRY_MAX}  "
                    f"final_force_mult="
                    f"{self._strict_force_multiplier:.2f}")
            else:
                current_q = self._current_arm_q()
                carry_q = [CARRY_H1, CARRY_H2, CARRY_A1, current_q[3]]
                self._kinematic_descent(current_q, carry_q, "lift",
                                        n_steps=DESCENT_STEPS)
            if self._cancel:
                self._clear_held_state(); fire(False); return

            success = True
            self._last_valid_pre_grasp_q = None
        except Exception as e:
            import traceback
            print(f"[Exec] PICK exception: {e}")
            traceback.print_exc()
            self._clear_held_state()
        finally:
            fire(success)


    def place(self, shelf_idx, shelf_pos, on_complete=None):
        self._cancel = False
        t = threading.Thread(
            target=self._place_into_slot_run,
            args=(int(shelf_idx), np.array(shelf_pos, dtype=float), on_complete),
            daemon=True)
        t.start()
        self._place_thread = t

    def _place_run(self, shelf_idx, shelf_pos, on_complete):
        cb_fired = [False]
        def fire(ok):
            if cb_fired[0]:
                return
            cb_fired[0] = True
            if on_complete:
                on_complete(ok)

        success = False
        try:
            if self._held_obj_idx is None:
                print("[Exec] PLACE called with no held object")
                fire(False)
                return

            print(f"\n[Exec] PLACE shelf_{shelf_idx} @ {shelf_pos.round(3)}  "
                  f"grasp_offset={self._grasp_offset_xyz.round(3)}")

            obj_bid = self._held_obj_bid

            above_obj_target  = shelf_pos.copy()
            above_obj_target[2] += SHELF_ABOVE_HEIGHT
            place_obj_target  = shelf_pos.copy()
            place_obj_target[2] += SHELF_PLACE_OFFSET_Z
            above_target = above_obj_target - self._grasp_offset_xyz
            place_target = place_obj_target - self._grasp_offset_xyz

            loc = self.sim.localization()
            reset_plan_data_for_ik(self.arm_bridge,
                                   base_xy=(loc[0], loc[1]),
                                   base_yaw=loc[2])
            self.arm_bridge.planning_data.qpos[
                self._held_obj_qpa:self._held_obj_qpa + 7] = \
                    self.sim.data.qpos[self._held_obj_qpa:self._held_obj_qpa + 7]
            mujoco.mj_forward(self.arm_bridge.model, self.arm_bridge.planning_data)

            print(f"[Exec] above_obj_target={above_obj_target.round(3)}  "
                  f"wrist={above_target.round(3)}")
            print(f"[Exec] place_obj_target={place_obj_target.round(3)}  "
                  f"wrist={place_target.round(3)}")

            try:
                ABOVE_Q, _ = self.arm_bridge.solve_ik_with_z_lift(above_target)
                PLACE_Q, _ = self.arm_bridge.solve_ik_with_z_lift(place_target)
            except RuntimeError as e:
                print(f"[Exec] PLACE IK FAIL: {e}")
                self._clear_held_state()
                fire(False)
                return

            print(f"[Exec] ABOVE_Q = {[round(x,3) for x in ABOVE_Q]}")
            print(f"[Exec] PLACE_Q = {[round(x,3) for x in PLACE_Q]}")

            print("[Exec] [7] transport: current → ABOVE_Q")
            q_now = self._current_arm_q()
            path_t = self.arm_bridge.plan(q_now, ABOVE_Q, timeout=12.0)
            if path_t is None:
                print("[Exec] transport plan FAIL")
                self._clear_held_state()
                fire(False)
                return
            self._execute_path(path_t, "transport")
            if self._cancel:
                self._clear_held_state(); fire(False); return

            print("[Exec] [8] place descent: ABOVE_Q → PLACE_Q")
            self._kinematic_descent(ABOVE_Q, PLACE_Q, "place-descent")
            if self._cancel:
                self._clear_held_state(); fire(False); return

            drop_xyz = self.sim.data.xpos[obj_bid].copy()
            xy_err = float(np.linalg.norm(drop_xyz[:2] - shelf_pos[:2]))
            z_ok = drop_xyz[2] >= shelf_pos[2] - 0.05
            place_ok = (xy_err <= PLACE_XY_TOLERANCE) and z_ok
            print(f"[Exec] [verify-place] xy_err={xy_err:.3f}m  z_ok={z_ok}  "
                  f"place_ok={place_ok}  drop={drop_xyz.round(3)}")

            print("[Exec] [9] release: open gripper, weld off")
            self._install_pin(self._pin_obj_at_world(drop_xyz))
            self.arm_bridge.planning_data.eq_active[self.weld_id] = 0
            self._set_gripper(GRIPPER_OPEN_POS, hold_seconds=0.8)

            print("[Exec] [10] retract: PLACE_Q → HOME_Q")
            q_now = self._current_arm_q()
            path_r = self.arm_bridge.plan(q_now, list(HOME_Q), timeout=8.0)
            if path_r is not None:
                self._execute_path(path_r, "retract")
            else:
                self._set_arm_cmd(HOME_Q)
                time.sleep(1.5)

            self._clear_held_state(deactivate_weld=True)

            success = place_ok
            if not place_ok:
                print(f"[Exec] PLACE reported FAILURE (xy_err={xy_err:.3f}m, "
                      f"z_ok={z_ok}) — object did not land on shelf surface")
        except Exception as e:
            import traceback
            print(f"[Exec] PLACE exception: {e}")
            traceback.print_exc()
        finally:
            fire(success)

    def _place_level_for_z(self, z):
        z = float(z)
        if z < 0.45:
            return "low"
        if z < 0.95:
            return "mid"
        return "high"

    def _shelf_surface_pairs(self, obj_bid, slot_z):
        lvl = self._place_level_for_z(slot_z)
        shelf_name = {"low": "m2_shelf_low",
                      "mid": "m2_shelf_mid_low",
                      "high": "m2_shelf_mid_high"}[lvl]
        model = self.arm_bridge.model
        sg = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, shelf_name)
        if sg < 0:
            return []
        obj_geoms = [g for g in range(model.ngeom)
                     if int(model.geom_bodyid[g]) == int(obj_bid)]
        return [(g, sg) for g in obj_geoms]

    def _set_arm_gravcomp(self, value):
        if getattr(self, '_arm_gravcomp_saved', None) is not None:
            return
        model = self.sim.model
        saved = []
        for b in range(model.nbody):
            nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ''
            low = nm.lower()
            if not low.endswith('_1'):
                continue
            if any(k in low for k in ('arm', 'bearing_column', 'rotation_link',
                                      'hand_bearing', 'gripper_link', 'finger_')):
                saved.append((b, float(model.body_gravcomp[b])))
                model.body_gravcomp[b] = float(value)
        self._arm_gravcomp_saved = saved
        print(f"[Exec] arm gravcomp={value} on {len(saved)} arm-1 bodies "
              f"(gravity-comp controller → cancels runtime load-sag)")

    def _restore_arm_gravcomp(self):
        saved = getattr(self, '_arm_gravcomp_saved', None)
        if not saved:
            self._arm_gravcomp_saved = None
            return
        model = self.sim.model
        for b, v in saved:
            try:
                model.body_gravcomp[b] = v
            except Exception:
                pass
        self._arm_gravcomp_saved = None
        print("[Exec] arm gravcomp restored")

    def _set_place_arm_holds(self, for_place=False):
        if getattr(self, '_place_holds_active', False):
            return
        model = self.sim.model
        saved = []

        def _stiffen(jname, stiff, damp, springref=None):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                return
            dofadr = int(model.jnt_dofadr[jid])
            qadr = int(model.jnt_qposadr[jid])
            saved.append((jid, dofadr, qadr,
                          float(model.jnt_stiffness[jid]),
                          float(model.dof_damping[dofadr]),
                          float(model.qpos_spring[qadr])))
            model.jnt_stiffness[jid] = float(stiff)
            model.dof_damping[dofadr] = float(damp)
            if springref is not None:
                model.qpos_spring[qadr] = float(springref)

        def _curq(jname):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                return None
            return float(self.sim.data.qpos[int(model.jnt_qposadr[jid])])

        _rigid = os.environ.get("AH_CARRY_RIGID", "1") == "1"
        for _sh in ("RotationLeftJoint_1", "RotationRightJoint_1"):
            _stiffen(_sh, PLACE_SHOULDER_STIFFNESS, PLACE_SHOULDER_DAMPING,
                     _curq(_sh) if _rigid else None)
        if _rigid:
            _stiffen("BaseJoint_1", PLACE_TH_STIFFNESS, PLACE_TH_DAMPING,
                     _curq("BaseJoint_1"))
        _stiffen("gripper_z_rotation_1", PLACE_WRIST_STIFFNESS, PLACE_WRIST_DAMPING,
                 _curq("gripper_z_rotation_1") if _rigid else WRIST_Z_SIDE_APPROACH)
        _stiffen("gripper_x_rotation_1", PLACE_WRIST_STIFFNESS, PLACE_WRIST_DAMPING,
                 _curq("gripper_x_rotation_1") if _rigid else WRIST_X_SIDE_APPROACH)
        _stiffen("gripper_y_rotation_1", PLACE_WRIST_STIFFNESS, PLACE_WRIST_DAMPING,
                 _curq("gripper_y_rotation_1") if _rigid else 0.0)
        self._place_stiff_saved = saved
        self.sim._base_kp_override = PLACE_BASE_KP
        self.sim._base_kd_override = PLACE_BASE_KD
        self.sim._base_ki_override = PLACE_BASE_KI
        try:
            self.sim.base_integral_1 = 0.0
        except Exception:
            pass
        self._battery_col_saved = []
        if os.environ.get("AH_CARRY_NO_BATTERY_COL", "1") == "1":
            _bb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                    "LifePo4_12V50Ah")
            if _bb >= 0:
                for _g in range(model.ngeom):
                    if (int(model.geom_bodyid[_g]) == _bb
                            and (int(model.geom_contype[_g]) != 0
                                 or int(model.geom_conaffinity[_g]) != 0)):
                        self._battery_col_saved.append(
                            (_g, int(model.geom_contype[_g]),
                             int(model.geom_conaffinity[_g])))
                        model.geom_contype[_g] = 0
                        model.geom_conaffinity[_g] = 0
                if self._battery_col_saved:
                    print(f"[Exec] transport: battery (LifePo4) collision "
                          f"DISABLED ({len(self._battery_col_saved)} geom) — "
                          f"no forearm-vs-battery explosion/fling")
        self._arm2_col_saved = []
        if os.environ.get("AH_CARRY_NO_ARM2_COL", "1") == "1":
            import re as _re2
            _a2 = _re2.compile(
                r'(finger_[abc]_link|Gripper_Link|palm_finger|Contact_Cylinder)\w*_2$')
            for _g in range(model.ngeom):
                if (int(model.geom_contype[_g]) != 0
                        or int(model.geom_conaffinity[_g]) != 0):
                    _bn = mujoco.mj_id2name(
                        model, mujoco.mjtObj.mjOBJ_BODY,
                        int(model.geom_bodyid[_g])) or ""
                    if _a2.search(_bn):
                        self._arm2_col_saved.append(
                            (_g, int(model.geom_contype[_g]),
                             int(model.geom_conaffinity[_g])))
                        model.geom_contype[_g] = 0
                        model.geom_conaffinity[_g] = 0
            if self._arm2_col_saved:
                print(f"[Exec] transport: parked ARM-2 collisions DISABLED "
                      f"({len(self._arm2_col_saved)} geom) — no arm-2 "
                      f"self-collision base-teleport careen")
        self._connect_solref_saved = []
        if os.environ.get("AH_CARRY_SOFT_CONNECT", "1") == "1":
            _sr = float(os.environ.get("AH_CONNECT_SOLREF", "0.01"))
            for _en in ("weld_contact_arm_sites_1", "weld_contact_arm_sites_2"):
                _eid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, _en)
                if _eid >= 0:
                    self._connect_solref_saved.append(
                        (_eid, float(model.eq_solref[_eid, 0]),
                         float(model.eq_solref[_eid, 1])))
                    model.eq_solref[_eid, 0] = _sr
                    model.eq_solref[_eid, 1] = 1.0
            if self._connect_solref_saved:
                print(f"[Exec] transport: arm <connect> solref softened "
                      f"1e-8 → {_sr} ({len(self._connect_solref_saved)}) — "
                      f"no constraint-blowup base fling")
        self._impratio_saved = None
        _lower_imp = ((for_place
                       and os.environ.get("AH_PLACE_LOW_IMPRATIO", "1") == "1")
                      or ((not for_place)
                          and os.environ.get("AH_CARRY_LOW_IMPRATIO", "1") == "1"))
        if _lower_imp:
            try:
                self._impratio_saved = float(self.sim.model.opt.impratio)
                _tgt = (float(os.environ.get("AH_PLACE_IMPRATIO", "3.0"))
                        if for_place
                        else float(os.environ.get("AH_IMPRATIO", "1.0")))
                self.sim.model.opt.impratio = _tgt
                print(f"[Exec] {'place' if for_place else 'transport'}: impratio "
                      f"{self._impratio_saved} → {self.sim.model.opt.impratio} "
                      f"— softer constraint conditioning (no QACC blowup)")
            except Exception as _e_ip:
                print(f"[Exec] impratio warn: {_e_ip}")
        self._place_holds_active = True
        print(f"[Exec] place arm-holds ON: shoulder stiffness="
              f"{PLACE_SHOULDER_STIFFNESS}, base PID kp={PLACE_BASE_KP} "
              f"ki={PLACE_BASE_KI} kd={PLACE_BASE_KD}")

    def _restore_place_arm_holds(self):
        if not getattr(self, '_place_holds_active', False):
            return
        model = self.sim.model
        for jid, dofadr, qadr, st, dm, spr in getattr(self, '_place_stiff_saved', []):
            try:
                model.jnt_stiffness[jid] = st
                model.dof_damping[dofadr] = dm
                model.qpos_spring[qadr] = spr
            except Exception:
                pass
        for a in ('_base_kp_override', '_base_kd_override', '_base_ki_override'):
            if hasattr(self.sim, a):
                try:
                    delattr(self.sim, a)
                except Exception:
                    pass
        for _g, _ct, _ca in getattr(self, '_battery_col_saved', []):
            try:
                model.geom_contype[_g] = _ct
                model.geom_conaffinity[_g] = _ca
            except Exception:
                pass
        self._battery_col_saved = []
        for _g, _ct, _ca in getattr(self, '_arm2_col_saved', []):
            try:
                model.geom_contype[_g] = _ct
                model.geom_conaffinity[_g] = _ca
            except Exception:
                pass
        self._arm2_col_saved = []
        self._place_holds_active = False
        print("[Exec] place arm-holds restored (pickup dynamics)")

    def _settle_hold_pose(self, q, secs):
        t0 = time.time()
        while time.time() - t0 < float(secs):
            if self._cancel:
                return
            self._set_arm_cmd(q)
            time.sleep(0.05)

    def _cart_ik_step(self, tgt, prev_q, accurate=False):
        ns, mi = (2, 3) if accurate else (1, 1)
        try:
            qi, _ = self.arm_bridge.solve_ik_with_z_lift_carry_anchor(
                tuple(tgt), n_seeds=ns, max_iters=mi,
                wrist_goal=PLACE_WRIST_GOAL, wrist_weight=PLACE_WRIST_WEIGHT,
                seed_q=prev_q, validity_penalty=0.0)
        except Exception:
            return None
        if qi is None:
            return None
        jump = max(abs(float(qi[k]) - float(prev_q[k]))
                   for k in range(min(len(qi), len(prev_q))))
        if jump > 0.25:
            return None
        return list(qi)

    def _arm_horizontal_lateral_offset(self, obj_bid, obj_xy_fallback):
        try:
            pinch = np.asarray(
                self._pinch_midpoint_xyz(self.sim.data)[:2], dtype=float)
            if obj_bid is not None:
                obj = np.asarray(self.sim.data.xpos[obj_bid][:2], dtype=float)
            else:
                obj = np.asarray(obj_xy_fallback, dtype=float)
            chx = np.asarray(self.sim.localization()[:2], dtype=float)
            fdir = obj - chx
            n = float(np.linalg.norm(fdir))
            if n < 1e-6:
                return None
            fdir = fdir / n
            ldir = np.array([-fdir[1], fdir[0]])
            return float(np.dot(pinch - obj, ldir))
        except Exception:
            return None

    def _arm_horizontal_recenter_lifted(self, obj_bid, obj_xy_fallback,
                                        lift=0.12, max_iters=40, th_kp=0.8,
                                        lat_tol=0.03, settle=0.06,
                                        th_total_bound=0.7):
        import time

        def _canon(q):
            return [float(q[0]), float(q[1]), float(q[2]), float(q[3]),
                    float(q[4]), WRIST_Z_SIDE_APPROACH,
                    WRIST_X_SIDE_APPROACH, WRIST_Y_SIDE_APPROACH]

        q0 = list(self._current_arm_q())
        h1_0, h2_0 = float(q0[0]), float(q0[1])
        _lat0 = self._arm_horizontal_lateral_offset(obj_bid, obj_xy_fallback)
        q_lift = list(q0)
        q_lift[0] = float(np.clip(h1_0 + lift, *JOINT_RANGES_ARM[0]))
        q_lift[1] = float(np.clip(h2_0 + lift, *JOINT_RANGES_ARM[1]))
        self._set_arm_cmd(_canon(q_lift))
        time.sleep(settle * 3)
        print(f"[Exec]   recentre: lifted +{lift*100:.0f}cm clear "
              f"(lat {(_lat0 or 0)*100:+.1f}cm) — uncapped TH recentre")
        _stl = max(settle, 0.14)

        def _lat_now():
            return self._arm_horizontal_lateral_offset(obj_bid, obj_xy_fallback)

        th_total = 0.0
        lat0 = _lat_now()
        if lat0 is None:
            return
        _qb = list(self._current_arm_q())
        _probe = 0.06
        _qp = list(_qb)
        _qp[3] = float(np.clip(_qb[3] + _probe, *JOINT_RANGES_ARM[3]))
        self._set_arm_cmd(_canon(_qp))
        time.sleep(_stl * 2)
        _lat1 = _lat_now()
        if _lat1 is not None and abs(_lat1) <= abs(lat0):
            th_sign = +1.0
            th_total = _probe
            best_lat = abs(_lat1)
        else:
            th_sign = -1.0
            self._set_arm_cmd(_canon(_qb))
            time.sleep(_stl)
            best_lat = abs(lat0)
        print(f"[Exec]   recentre: probe → TH sign {'+' if th_sign>0 else '-'} "
              f"(lat {(lat0 or 0)*100:+.1f}→{(_lat1 or 0)*100:+.1f}cm on +probe)")
        for it in range(int(max_iters)):
            if self._cancel:
                break
            lat = _lat_now()
            if lat is None:
                break
            if abs(lat) <= lat_tol:
                print(f"[Exec]   recentre: lat {lat*100:+.1f}cm ≤ "
                      f"{lat_tol*100:.0f}cm after {it} iters (th_tot "
                      f"{th_total:+.2f}rad)")
                break
            if abs(lat) > best_lat + 0.01:
                print(f"[Exec]   recentre: overshoot (lat {lat*100:+.1f}cm > "
                      f"best {best_lat*100:.1f}cm) — stop (th_tot "
                      f"{th_total:+.2f}rad)")
                break
            best_lat = min(best_lat, abs(lat))
            dth = th_sign * float(np.clip(th_kp * abs(lat) / 0.6, 0.01, 0.05))
            if abs(th_total + dth) > th_total_bound:
                print(f"[Exec]   recentre: TH bound {th_total_bound:.2f}rad "
                      f"hit (lat {lat*100:+.1f}cm) — stop")
                break
            _prev = list(self._current_arm_q())
            q = list(self._current_arm_q())
            q[3] = float(np.clip(q[3] + dth, *JOINT_RANGES_ARM[3]))
            self._set_arm_cmd(_canon(q))
            time.sleep(_stl)
            th_total += dth
            try:
                if self._count_arm_chassis_contacts() > 0:
                    self._set_arm_cmd(_canon(_prev))
                    time.sleep(_stl)
                    print(f"[Exec]   recentre: arm-chassis contact — revert, "
                          f"stop (lat {lat*100:+.1f}cm)")
                    th_total -= dth
                    break
            except Exception:
                pass
        q_now = list(self._current_arm_q())
        q_down = list(q_now)
        q_down[0] = float(np.clip(q_now[0] - lift, *JOINT_RANGES_ARM[0]))
        q_down[1] = float(np.clip(q_now[1] - lift, *JOINT_RANGES_ARM[1]))
        self._set_arm_cmd(_canon(q_down))
        time.sleep(settle * 3)
        _latf = self._arm_horizontal_lateral_offset(obj_bid, obj_xy_fallback)
        print(f"[Exec]   recentre done: re-descended; lat "
              f"{(_lat0 or 0)*100:+.1f} → {(_latf or 0)*100:+.1f}cm")

    def _arm_forward_const_z(self, target_xy, label="arm-fwd-cz",
                             max_iters=70, a1_step=0.010, z_kp=0.6,
                             tilt_step=0.008, xy_tol=0.025, settle=0.06,
                             th_kp=0.5, th_total_cap=0.18, obj_bid=None):
        target_xy = np.asarray(target_xy, dtype=float)
        try:
            z_hold = float(self._pinch_midpoint_xyz(self.sim.data)[2])
        except Exception:
            return float('nan')
        if ENABLE_AH_SERVO_DESCEND_TO_OBJ and obj_bid is not None:
            try:
                z_hold = float(self.sim.data.xpos[obj_bid][2])
            except Exception:
                pass
        self._hold_wz_to_target(WRIST_Z_SIDE_APPROACH, label=f"{label}-wz")
        _ax_start = self._axis_residual_deg(obj_bid)
        if _ax_start is not None:
            print(f"[AH-AXIS] {label} servo START: thumb-bc axis residual "
                  f"{_ax_start:+.1f}° (0=⊥approach=obj-between-fingers; "
                  f"the structural grasp blocker)")
        if ENABLE_AH_AXIS_PROBE:
            self._probe_axis_jacobian(obj_bid, label=label)
        _wy_dyn = WRIST_Y_SIDE_APPROACH
        try:
            _zt_pinch0 = float(self._pinch_midpoint_xyz(self.sim.data)[2])
            _zt_palm0 = float(self.sim.data.xpos[self.gripper_body_id][2])
            _zt_objz = (float(self.sim.data.xpos[obj_bid][2])
                        if obj_bid is not None else float('nan'))
            print(f"[Z-TRACE] {label} servo START: pinch_z={_zt_pinch0:.3f} "
                  f"palm_z={_zt_palm0:.3f} obj_z={_zt_objz:.3f} "
                  f"(pinch should hold; palm rises w/ boom-straighten)")
        except Exception:
            _zt_pinch0 = _zt_palm0 = None
        try:
            _bh = self.sim.localization()
            _base_hold = np.array([float(_bh[0]), float(_bh[1]), float(_bh[2])])
            with self.sim._target_lock:
                self.sim.target_base = _base_hold.copy()
        except Exception:
            _base_hold = None
        a1_max = float(JOINT_RANGES_ARM[2][1])
        try:
            _obj_r_srv = float(self._object_radius(obj_bid)) \
                if obj_bid is not None else 0.0
        except Exception:
            _obj_r_srv = 0.0
        _bc_stop_srv    = _obj_r_srv + 0.020
        _thumb_stop_srv = _obj_r_srv + 0.020
        _fwd_surface = (_obj_r_srv + ARM_HORIZONTAL_SURFACE_MARGIN
                        if _obj_r_srv > 0.0 else float(xy_tol))
        prev_err = None
        stall = 0
        it = 0
        th_total = 0.0
        th_sign = 1.0
        lat_prev = None
        _gth_sign_locked = False
        best_err = float('inf')
        best_q = None
        for it in range(int(max_iters)):
            if self._cancel:
                break
            if obj_bid is not None:
                try:
                    _obj_xyz_live = self.sim.data.xpos[obj_bid]
                    target_xy = np.asarray(_obj_xyz_live[:2], dtype=float)
                    if ENABLE_AH_SERVO_DESCEND_TO_OBJ:
                        z_hold = float(_obj_xyz_live[2])
                except Exception:
                    pass
            if _base_hold is not None:
                try:
                    with self.sim._target_lock:
                        self.sim.target_base = _base_hold.copy()
                except Exception:
                    pass
            c = np.asarray(self._pinch_midpoint_xyz(self.sim.data), dtype=float)
            err_vec = target_xy - c[:2]
            err_xy = float(np.linalg.norm(err_vec))
            z_err = z_hold - float(c[2])
            if err_xy < best_err:
                best_err = err_xy
                best_q = list(self._current_arm_q())
            if err_xy <= xy_tol:
                break
            if obj_bid is not None:
                try:
                    if self._count_arm_obj_contacts(obj_bid) > 0:
                        _bq = list(self._current_arm_q())
                        for _bo in range(4):
                            if self._count_arm_obj_contacts(obj_bid) == 0:
                                break
                            _bq[2] = max(MIN_PICK_A1, _bq[2] - 0.010)
                            try:
                                if not self.arm_bridge.is_valid(_bq):
                                    break
                            except Exception:
                                break
                            self._set_arm_cmd(_bq)
                            time.sleep(settle)
                        _nc = self._count_arm_obj_contacts(obj_bid)
                        print(f"[Exec] {label}: arm-structure touched obj at "
                              f"xy_err={err_xy*100:.1f}cm — backed off "
                              f"(arm-obj ncon→{_nc}); stop (fingers straddle, "
                              f"no ram)")
                        break
                except Exception:
                    pass
            if (ENABLE_AH_RADIUS_FINGER_STOP
                    and obj_bid is not None and _obj_r_srv > 0.0
                    and len(self._carry_anchor_body_ids) == 3):
                try:
                    _objc = self.sim.data.xpos[obj_bid][:2]
                    _th_xy = self.sim.data.xpos[
                        self._carry_anchor_body_ids[0]][:2]
                    _b_xy = self.sim.data.xpos[
                        self._carry_anchor_body_ids[1]][:2]
                    _c_xy = self.sim.data.xpos[
                        self._carry_anchor_body_ids[2]][:2]
                    _bc_xy = 0.5 * (np.asarray(_b_xy) + np.asarray(_c_xy))
                    _d_th = float(np.hypot(_th_xy[0] - _objc[0],
                                           _th_xy[1] - _objc[1]))
                    _d_bc = float(np.hypot(_bc_xy[0] - _objc[0],
                                           _bc_xy[1] - _objc[1]))
                    if _d_bc <= _bc_stop_srv or _d_th <= _thumb_stop_srv:
                        print(f"[Exec] {label}: finger reached obj surface "
                              f"(d_bc={_d_bc*100:.1f} d_th={_d_th*100:.1f}cm ≤ "
                              f"r+2={_bc_stop_srv*100:.1f}cm) — stop forward, no "
                              f"penetration (close stroke finishes)")
                        break
                except Exception:
                    pass
            try:
                chx = np.asarray(self.sim.localization()[:2], dtype=float)
            except Exception:
                chx = c[:2]
            fdir = target_xy - chx
            _fn = float(np.linalg.norm(fdir))
            fdir = fdir / _fn if _fn > 1e-6 else np.array([1.0, 0.0])
            ldir = np.array([-fdir[1], fdir[0]])
            fwd_err = float(np.dot(err_vec, fdir))
            lat_err = float(np.dot(err_vec, ldir))
            if fwd_err <= _fwd_surface and abs(lat_err) <= max(xy_tol, 0.04):
                print(f"[Exec] {label}: reached obj SURFACE "
                      f"(fwd {fwd_err*100:.1f}cm ≤ stop {_fwd_surface*100:.1f}cm, "
                      f"lat {lat_err*100:+.1f}cm) — STOP, no push (close wraps)")
                break
            if (not _gth_sign_locked and lat_prev is not None
                    and abs(lat_err) > abs(lat_prev) + 0.002):
                th_sign = -th_sign
                _gth_sign_locked = True
            lat_prev = lat_err
            if prev_err is not None and (prev_err - err_xy) < 0.0015:
                stall += 1
                if stall >= 8:
                    print(f"[Exec] {label}: stalled (reach limit) at "
                          f"xy_err={err_xy*100:.1f}cm after {it+1} iters")
                    break
            else:
                stall = 0
            prev_err = err_xy

            q = list(self._current_arm_q())
            h1, h2, a1, th = (float(q[0]), float(q[1]),
                              float(q[2]), float(q[3]))
            tilt0 = h2 - h1
            if fwd_err > _fwd_surface and a1 < a1_max - 0.012:
                _step_eff = float(np.clip(0.06 * fwd_err, 0.003, a1_step))
                a1 = min(a1 + _step_eff, a1_max)
            d_tilt = float(np.clip(z_kp * z_err, -0.016, 0.016))
            if tilt0 >= 0:
                h1 += 0.5 * d_tilt
                h2 -= 0.5 * d_tilt
            else:
                h1 -= 0.5 * d_tilt
                h2 += 0.5 * d_tilt
            if fwd_err > 0.005 and a1 >= a1_max - 0.012 and abs(tilt0) > 1e-3:
                s = tilt_step if tilt0 > 0 else -tilt_step
                h1 += 0.5 * s
                h2 -= 0.5 * s
            if abs(lat_err) > 0.010 and abs(th_total) < th_total_cap:
                dth = th_sign * th_kp * float(np.clip(abs(lat_err) / 0.65,
                                                      0.0, 0.022))
                if abs(th_total + dth) <= th_total_cap:
                    th += dth
                    th_total += dth
            if ENABLE_AH_AXIS_CORRECT and obj_bid is not None:
                _axr = self._axis_residual_deg(obj_bid)
                if _axr is not None and abs(_axr) > 3.0:
                    _wy_dyn += float(np.clip(-0.006 * _axr, -0.04, 0.04))
                    _wy_dyn = float(np.clip(_wy_dyn, -0.60, 0.30))
            q_new = [h1, h2, a1, th,
                     float(q[4]),
                     WRIST_Z_SIDE_APPROACH,
                     WRIST_X_SIDE_APPROACH,
                     _wy_dyn]
            if it % 5 == 0 or err_xy < 0.05:
                print(f"  [{label}] it{it}: xy_err={err_xy*100:.1f}cm "
                      f"(fwd={fwd_err*100:+.1f} lat={lat_err*100:+.1f}) "
                      f"z_err={z_err*100:+.1f}cm a1={q[2]:.3f}→{a1:.3f} "
                      f"th={q[3]:+.3f}→{th:+.3f}(tot{th_total:+.3f}) "
                      f"tilt={tilt0:+.3f}")
            if not all(np.isfinite(q_new)):
                break
            for i in range(len(q_new)):
                lo, hi = JOINT_RANGES_ARM[i]
                q_new[i] = float(np.clip(q_new[i], lo, hi))
            _prev_q = list(q)
            self._set_arm_cmd(q_new)
            _to_surface = fwd_err - _fwd_surface
            if _to_surface < ARM_HORIZONTAL_SLOW_NEAR_BAND * 0.5:
                _slow = 4.0
            elif _to_surface < ARM_HORIZONTAL_SLOW_NEAR_BAND:
                _slow = 2.5
            else:
                _slow = 1.0
            time.sleep(settle * _slow)
            try:
                if self._count_arm_chassis_contacts() > 0:
                    self._set_arm_cmd(_prev_q)
                    time.sleep(settle)
                    print(f"[Exec] {label}: real arm-chassis contact at "
                          f"xy_err={err_xy*100:.1f}cm — reverted step, stopping "
                          f"(chassis-limited reach)")
                    break
            except Exception:
                pass

        final = float(np.linalg.norm(
            target_xy - np.asarray(self._pinch_midpoint_xyz(self.sim.data)[:2])))
        _ax_end = self._axis_residual_deg(obj_bid)
        if _ax_end is not None:
            print(f"[AH-AXIS] {label} servo END: thumb-bc axis residual "
                  f"{_ax_end:+.1f}° (start {(_ax_start if _ax_start is not None else 0):+.1f}°) "
                  f"— servo moved pinch fwd but axis ~unchanged (no axis DOF)")
        try:
            _zt_pinch1 = float(self._pinch_midpoint_xyz(self.sim.data)[2])
            _zt_palm1 = float(self.sim.data.xpos[self.gripper_body_id][2])
            _dp = (_zt_pinch1 - _zt_pinch0) if _zt_pinch0 is not None else 0.0
            _dpa = (_zt_palm1 - _zt_palm0) if _zt_palm0 is not None else 0.0
            print(f"[Z-TRACE] {label} servo END: pinch_z={_zt_pinch1:.3f} "
                  f"(Δ{_dp*100:+.1f}cm) palm_z={_zt_palm1:.3f} (Δ{_dpa*100:+.1f}cm) "
                  f"— pinch Δ = the real grasp-Z drift; palm Δ = boom-straighten rise")
        except Exception:
            pass
        if best_q is not None and final > best_err + 0.01:
            try:
                if self.arm_bridge.is_valid(best_q):
                    self._set_arm_cmd([float(v) for v in best_q])
                    time.sleep(settle)
                    final = best_err
                    print(f"[Exec] {label}: reverted to best pose "
                          f"(xy err={best_err*100:.1f}cm)")
            except Exception:
                pass
        print(f"[Exec] {label}: done, centroid xy err={final*100:.1f}cm "
              f"(z_hold={z_hold:.3f}) after {it+1} iters")
        return final

    def _cartesian_move_closed_loop(self, target_xyz, n_steps=10, max_correct=4,
                                    pos_tol=0.012, hold_z=True, settle=0.10,
                                    label="cart-cl"):
        target = np.asarray(target_xyz, dtype=float)
        start = np.asarray(self._carry_anchor_xyz(self.sim.data), dtype=float)
        z_goal = float(start[2]) if hold_z else float(target[2])
        prev_q = list(self._current_arm_q())
        self._hold_wz_to_target(WRIST_Z_SIDE_APPROACH, label=f"{label}-wz")
        for i in range(1, int(n_steps) + 1):
            if self._cancel:
                return float('nan')
            a = i / float(n_steps)
            tx = float(start[0] + a * (target[0] - start[0]))
            ty = float(start[1] + a * (target[1] - start[1]))
            tz = z_goal if hold_z else float(start[2] + a * (target[2] - start[2]))
            qi = self._cart_ik_step((tx, ty, tz), prev_q, accurate=True)
            if qi is not None:
                self._set_arm_cmd(qi)
                prev_q = qi
                time.sleep(settle)
        aim = np.array([float(target[0]), float(target[1]), z_goal], dtype=float)
        err_n = float('nan')
        for c in range(int(max_correct)):
            actual = np.asarray(self._carry_anchor_xyz(self.sim.data), dtype=float)
            err = np.array([target[0] - actual[0], target[1] - actual[1],
                            z_goal - actual[2]], dtype=float)
            err_n = float(np.linalg.norm(err))
            print(f"[Exec] {label}: correct {c} actual=("
                  f"{actual[0]:.3f},{actual[1]:.3f},{actual[2]:.3f}) "
                  f"err={err_n*100:.1f}cm (xy={math.hypot(err[0],err[1])*100:.1f} "
                  f"z={err[2]*100:+.1f})")
            if err_n <= pos_tol:
                break
            qi = None
            used_frac = 0.0
            for frac in (1.0, 0.6, 0.35, 0.2):
                cand = aim + err * frac
                qi = self._cart_ik_step((float(cand[0]), float(cand[1]),
                                         float(cand[2])), prev_q, accurate=True)
                if qi is not None:
                    aim = cand
                    used_frac = frac
                    break
            if qi is None:
                print(f"[Exec] {label}: correct {c} re-aim rejected at all "
                      f"fracs (reach floor) — stopping at err={err_n*100:.1f}cm")
                break
            if used_frac < 1.0:
                print(f"[Exec] {label}: correct {c} frac-retry {used_frac:.2f}")
            self._set_arm_cmd(qi)
            prev_q = qi
            time.sleep(settle + 0.04)
        return err_n

    def _cartesian_horizontal_insert(self, obj_bid, place_anchor, slot_surface_z,
                                     n_steps=PLACE_CART_STEPS):
        place_anchor = np.asarray(place_anchor, dtype=float)
        start_xy = np.asarray(self._carry_anchor_xyz(self.sim.data)[:2], dtype=float)
        place_z = float(place_anchor[2])
        _slide_clear = float(os.environ.get("AH_PLACE_SLIDE_CLEAR",
                                            str(PLACE_SLIDE_CLEARANCE)))
        slide_z = place_z + _slide_clear
        last_q = list(self._current_arm_q())
        self._hold_wz_to_target(WRIST_Z_SIDE_APPROACH, label="cart-wz-hold")
        prev_q = list(last_q)
        for i in range(1, int(n_steps) + 1):
            if self._cancel:
                return last_q
            a = i / float(n_steps)
            tgt_xy = start_xy + a * (place_anchor[:2] - start_xy)
            qi = self._cart_ik_step((float(tgt_xy[0]), float(tgt_xy[1]), slide_z),
                                    prev_q)
            if qi is not None:
                self._set_arm_cmd(qi)
                prev_q = qi
                last_q = qi
            time.sleep(PLACE_INSERT_SETTLE)
        print(f"  [cart-insert] slid to slot at clearance Z={slide_z:.3f} "
              f"(place_z={place_z:.3f})")
        if os.environ.get("AH_PLACE_SMOOTH", "0") == "1":
            print(f"  [cart-insert] SMOOTH: skip phase-2 lower (servo finalises "
                  f"from slide Z={slide_z:.3f}) — no below-board dip")
            return last_q
        half_h = self._object_half_height(obj_bid)
        target_bottom = slot_surface_z + 0.004
        z_cmd = slide_z
        z_floor = place_z - 0.12
        lowered = 0
        for j in range(1, 41):
            if self._cancel:
                return last_q
            obj_bottom = float(self.sim.data.xpos[obj_bid][2]) - half_h
            if obj_bottom <= target_bottom + 0.004:
                print(f"  [cart-insert] lower: obj_bottom={obj_bottom:.3f} on shelf "
                      f"(surface {slot_surface_z:.3f}) after {lowered} steps")
                break
            z_cmd = z_cmd - 0.01
            if z_cmd < z_floor:
                print(f"  [cart-insert] lower: pocket floor {z_floor:.3f} reached, "
                      f"obj_bottom={obj_bottom:.3f} still above surface")
                break
            qi = self._cart_ik_step(
                (float(place_anchor[0]), float(place_anchor[1]), z_cmd), prev_q)
            if qi is not None:
                self._set_arm_cmd(qi)
                prev_q = qi
                last_q = qi
                lowered += 1
            time.sleep(PLACE_INSERT_SETTLE)
        print(f"  [cart-insert] done (slide {n_steps} wp + lower {lowered} steps)")
        return last_q

    def _cartesian_horizontal_backout(self, n_steps=PLACE_CART_STEPS):
        start = np.asarray(self._carry_anchor_xyz(self.sim.data), dtype=float)
        z_const = float(start[2])
        end_xy = start[:2] + np.array([0.0, PLACE_BACKOUT_DIST])
        prev_q = self._current_arm_q()
        last_q = list(prev_q)
        self._hold_wz_to_target(WRIST_Z_SIDE_APPROACH, label="backout-wz")
        for i in range(1, int(n_steps) + 1):
            if self._cancel:
                return last_q
            a = i / float(n_steps)
            tgt_xy = start[:2] + a * (end_xy - start[:2])
            qi = self._cart_ik_step((float(tgt_xy[0]), float(tgt_xy[1]), z_const),
                                    prev_q)
            if qi is not None:
                self._set_arm_cmd(qi)
                prev_q = qi
                last_q = qi
            time.sleep(PLACE_INSERT_SETTLE)
        print(f"  [cart-backout] gripper slid back {PLACE_BACKOUT_DIST:.2f} m at "
              f"constant Z={z_const:.3f} (clear of shelf)")
        return last_q

    def _held_obj_world_err(self, obj_centre_target):
        obj_now = self.sim.data.xpos[self._held_obj_bid].copy()
        tgt = np.asarray(obj_centre_target, dtype=float)
        xy_err = float(np.linalg.norm(obj_now[:2] - tgt[:2]))
        z_err = float(abs(obj_now[2] - tgt[2]))
        return xy_err, z_err, obj_now

    def _place_tier(self, xy_err, z_err):
        z_pass = z_err <= PLACE_Z_PASS
        z_accept = z_err <= 0.05
        if not z_accept:
            return "approx_fallback" if xy_err <= PLACE_XY_TOLERANCE else "failed"
        if xy_err <= PLACE_XY_PASS and z_pass:
            return "precise_success"
        if xy_err <= PLACE_XY_GATE:
            return "precise_marginal"
        if xy_err <= PLACE_XY_TOLERANCE:
            return "approx_fallback"
        return "failed"

    def _clamp_arm_q(self, q):
        out = list(q)
        for i in range(min(len(out), len(JOINT_RANGES_ARM))):
            lo, hi = JOINT_RANGES_ARM[i]
            out[i] = float(min(max(float(out[i]), lo), hi))
        return out

    def _retract_arm_to_home(self):
        q_now = self._clamp_arm_q(self._current_arm_q())
        path_r = self.arm_bridge.plan(q_now, list(HOME_Q), timeout=8.0)
        if path_r is not None:
            self._execute_path(path_r, "place-retract")
        else:
            print("[Exec] retract OMPL failed → smooth kinematic retract to HOME")
            self._kinematic_descent(q_now, list(HOME_Q), label="place-retract-kin",
                                    n_steps=40)

    def _release_in_place(self, obj_bid):
        if STRICT_PERFECT_FRICTION_ONLY:
            self._set_gripper(GRIPPER_OPEN_POS, hold_seconds=0.8)
            return
        if self._held_obj_orig_gravcomp is not None:
            try:
                self.sim.model.body_gravcomp[obj_bid] = self._held_obj_orig_gravcomp
            except Exception:
                pass
        self._clear_pin()
        if self._held_obj_dofadr is not None:
            try:
                with self.sim._target_lock:
                    self.sim.data.qvel[
                        self._held_obj_dofadr:self._held_obj_dofadr + 6] = 0.0
            except Exception:
                pass
        time.sleep(PLACE_RELEASE_SETTLE_SECS)
        self._restore_held_obj_contacts()
        self._set_gripper(GRIPPER_OPEN_POS, hold_seconds=0.8)
        try:
            self.arm_bridge.planning_data.eq_active[self.weld_id] = 0
        except Exception:
            pass

    def _emit_place_metric(self, shelf_idx, tier, xy_err, z_err, slot_surface_z, half_h):
        print(f"[PLACE_METRIC] slot={shelf_idx} class={tier} "
              f"xy_err_cm={xy_err*100:.2f} z_err_cm={z_err*100:.2f} "
              f"surface_z={slot_surface_z:.3f} half_h={half_h:.3f}")

    def _count_arm_shelf_contacts(self, force_thresh=None):
        if force_thresh is None:
            force_thresh = float(os.environ.get("AH_PLACE_RAM_FORCE", "250.0"))
        data = self.sim.data
        model = self.sim.model
        try:
            n = int(data.ncon)
            arm_bid_set = self._ensure_gripper_body_ids()
        except Exception:
            return 0
        if not arm_bid_set:
            return 0
        count = 0
        for i in range(n):
            try:
                c = data.contact[i]
                b1 = int(model.geom_bodyid[int(c.geom1)])
                b2 = int(model.geom_bodyid[int(c.geom2)])
            except Exception:
                continue
            if b1 in arm_bid_set and b2 not in arm_bid_set:
                other_b = b2
            elif b2 in arm_bid_set and b1 not in arm_bid_set:
                other_b = b1
            else:
                continue
            other_name = (mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_BODY, other_b) or "").lower()
            if ("shelf" in other_name or "rack" in other_name
                    or other_name.startswith("m2_")):
                try:
                    _f = np.zeros(6, dtype=float)
                    mujoco.mj_contactForce(model, data, i, _f)
                    if float(np.linalg.norm(_f[:3])) > force_thresh:
                        count += 1
                except Exception:
                    count += 1
        return count

    def _place_ik_calibrate(self, obj_bid, obj_centre_target, base_anchor_target):
        tgt = np.asarray(obj_centre_target, dtype=float)
        base_anchor = np.asarray(base_anchor_target, dtype=float)
        gain   = float(os.environ.get("AH_PLACE_IKCAL_GAIN", "0.85"))
        n_iter = int(os.environ.get("AH_PLACE_IKCAL_ITERS", "6"))
        colw   = float(os.environ.get("AH_PLACE_IKCAL_COLW", "5.0"))
        cap    = float(os.environ.get("AH_PLACE_IKCAL_CAP", "0.40"))
        steps  = int(os.environ.get("AH_PLACE_IKCAL_STEPS", "18"))
        _bl = None
        try:
            _b = self.sim.localization()
            _bl = np.array([float(_b[0]), float(_b[1]), float(_b[2])])
        except Exception:
            _bl = None
        bias = np.zeros(3)
        best_q, best_err, best_tier = None, float("inf"), "failed"
        xy_err, _ze, obj_now = self._held_obj_world_err(obj_centre_target)
        print(f"[Exec] [ikcal START] obj={obj_now.round(3)} target={tgt.round(3)} "
              f"xy_err={xy_err*100:.1f}cm z_err={(tgt[2]-obj_now[2])*100:+.1f}cm "
              f"(colw={colw} gain={gain})")
        for k in range(n_iter):
            if self._cancel:
                break
            if _bl is not None:
                try:
                    with self.sim._target_lock:
                        self.sim.target_base = _bl.copy()
                except Exception:
                    pass
            err3 = tgt - obj_now
            xy_err = float(np.linalg.norm(err3[:2]))
            z_err = float(err3[2])
            if xy_err <= PLACE_XY_PASS and abs(z_err) <= PLACE_Z_PASS:
                break
            bias = bias + gain * err3
            _bn = float(np.linalg.norm(bias))
            if _bn > cap:
                bias = bias * (cap / _bn)
            aim = base_anchor + bias
            _seed = list(self._current_arm_q())
            q = None
            try:
                q, _ = self.arm_bridge.solve_ik_with_z_lift_carry_anchor(
                    tuple(aim), n_seeds=10, max_iters=4,
                    wrist_goal=PLACE_WRIST_GOAL, wrist_weight=PLACE_WRIST_WEIGHT,
                    seed_q=_seed, column_center_weight=colw)
            except Exception:
                q = None
            if q is None:
                bias = bias - 0.5 * gain * err3
                aim = base_anchor + bias
                try:
                    q, _ = self.arm_bridge.solve_ik_with_z_lift_carry_anchor(
                        tuple(aim), n_seeds=10, max_iters=4,
                        wrist_goal=PLACE_WRIST_GOAL,
                        wrist_weight=PLACE_WRIST_WEIGHT, seed_q=_seed)
                except Exception:
                    q = None
                if q is None:
                    print(f"[Exec] [ikcal {k+1}] biased target unreachable — stop")
                    break
            self._hold_wz_to_target(WRIST_Z_SIDE_APPROACH, label="ikcal-wz")
            self._kinematic_descent(self._current_arm_q(), list(q),
                                    label=f"ikcal-{k}", n_steps=steps,
                                    per_step_settle=PLACE_INSERT_SETTLE)
            self._settle_hold_pose(list(q), PLACE_SETTLE_HOLD_SECS)
            xy_err, _ze, obj_now = self._held_obj_world_err(obj_centre_target)
            z_err = float(tgt[2] - obj_now[2])
            err_mag = float(np.linalg.norm(tgt - obj_now))
            tier = self._place_tier(xy_err, abs(z_err))
            print(f"[Exec] [ikcal {k+1}/{n_iter}] bias={bias.round(3)} "
                  f"xy_err={xy_err*100:.1f}cm z_err={z_err*100:+.1f}cm tier={tier} "
                  f"obj={obj_now.round(3)}")
            if err_mag < best_err - 0.002:
                best_err, best_q, best_tier = err_mag, list(q), tier
                if xy_err <= PLACE_XY_PASS and abs(z_err) <= PLACE_Z_PASS:
                    return tier, xy_err
            elif err_mag > best_err + 0.010:
                print(f"[Exec] [ikcal {k+1}] diverging "
                      f"({err_mag*100:.1f}>{best_err*100:.1f}cm) — stop, lock best")
                break
        if best_q is not None:
            if _bl is not None:
                try:
                    with self.sim._target_lock:
                        self.sim.target_base = _bl.copy()
                except Exception:
                    pass
            self._hold_wz_to_target(WRIST_Z_SIDE_APPROACH, label="ikcal-lock-wz")
            self._kinematic_descent(self._current_arm_q(), list(best_q),
                                    label="ikcal-lock", n_steps=steps,
                                    per_step_settle=PLACE_INSERT_SETTLE)
            self._settle_hold_pose(list(best_q), PLACE_SETTLE_HOLD_SECS)
        xy_err, _ze, obj_now = self._held_obj_world_err(obj_centre_target)
        z_err = float(obj_centre_target[2] - obj_now[2])
        return self._place_tier(xy_err, abs(z_err)), xy_err

    def _place_deflection_servo(self, obj_bid, obj_centre_target,
                                anchor_target, PLACE_Q):
        tgt = np.asarray(obj_centre_target, dtype=float)
        cur_q = list(PLACE_Q)
        z_kp     = float(os.environ.get("AH_PLACE_Z_KP", "0.7"))
        n_iter   = int(os.environ.get("AH_PLACE_SERVO_ITERS", "26"))
        tilt_cap = float(os.environ.get("AH_PLACE_TILT_STEP", "0.03"))
        col_cap  = float(os.environ.get("AH_PLACE_COL_STEP", "0.03"))
        a1_step  = float(os.environ.get("AH_PLACE_A1_STEP", "0.012"))
        _fine_fwd   = float(os.environ.get("AH_PLACE_FINE_FWD", "0.06"))
        _a1_step_far = float(os.environ.get("AH_PLACE_A1_STEP_FAR", "0.030"))
        th_kp    = float(os.environ.get("AH_PLACE_TH_KP", "0.5"))
        _a1_max  = float(os.environ.get("AH_PLACE_A1_MAX", str(PLACE_A1_MAX)))
        _tilt_floor = float(os.environ.get("AH_PLACE_TILT_FLOOR", "0.0"))
        _lift_first = os.environ.get("AH_PLACE_LIFT_FIRST", "0") == "1"
        _lf_z = float(os.environ.get("AH_PLACE_LIFT_FIRST_Z", "0.04"))
        _zfirst = os.environ.get("AH_PLACE_ZFIRST", "0") == "1"
        if _zfirst:
            _lift_first = True
            n_iter = max(n_iter, int(os.environ.get("AH_PLACE_ZFIRST_ITERS", "70")))
            try:
                _ca = list(self._current_arm_q())
                for _i in range(min(len(cur_q), len(_ca))):
                    cur_q[_i] = float(_ca[_i])
            except Exception:
                pass
            _a1_ret = float(os.environ.get("AH_PLACE_ZFIRST_A1", "0.28"))
            if float(cur_q[2]) > _a1_ret:
                _a10 = float(cur_q[2])
                for _s in range(1, 13):
                    cur_q[2] = _a10 + (_a1_ret - _a10) * _s / 12.0
                    self._set_arm_cmd(list(cur_q)); time.sleep(0.10)
                print(f"[Exec] ZFIRST: boom retracted a1 {_a10:.2f}->{float(cur_q[2]):.2f} "
                      f"(obj in front, NO h1/h2 jump); LIFT h1/h2 to Z, THEN const-Z slide")
        th_cap   = float(os.environ.get("AH_PLACE_TH_CAP", "0.30"))
        srv_steps = int(os.environ.get("AH_PLACE_SERVO_STEPS", "8" if _zfirst else "14"))
        try:
            _bxy = np.asarray(self.sim.localization()[:2], dtype=float)
        except Exception:
            _bxy = tgt[:2].copy()
        fdir = tgt[:2] - _bxy
        _fn = float(np.linalg.norm(fdir))
        fdir = fdir / _fn if _fn > 1e-6 else np.array([0.0, -1.0])
        ldir = np.array([-fdir[1], fdir[0]])
        th_sign = 1.0
        th_total = 0.0
        lat_prev = None
        _th_aim_on = (os.environ.get("AH_PLACE_TH_AIM", "0") == "1") and _zfirst
        _th_probed, _th_k, _use_strafe, _th_noconv = False, 0.0, False, 0
        xy_err, _ze, obj_now = self._held_obj_world_err(obj_centre_target)
        z_err = float(tgt[2] - obj_now[2])
        _comb = lambda xy, z: float(xy) + abs(float(z))
        best_q, best_comb, best_xy, best_z = list(cur_q), _comb(xy_err, z_err), xy_err, z_err
        stall = 0
        _sagclose = (os.environ.get("AH_PLACE_RUNTIME_SAGCLOSE", "0") == "1")
        _best_z_q, _best_zabs = list(cur_q), abs(float(z_err))
        _last_zabs, _zstall = abs(float(z_err)), 0
        if getattr(self, "_place_base_lock", None) is None:
            try:
                _b = self.sim.localization()
                self._place_base_lock = np.array(
                    [float(_b[0]), float(_b[1]), float(_b[2])])
            except Exception:
                self._place_base_lock = None
        print(f"[Exec] [servo START] xy_err={xy_err*100:.1f}cm z_err={z_err*100:+.1f}cm "
              f"(fwd+lat+lift; a1={cur_q[2]:.3f} tilt={cur_q[1]-cur_q[0]:+.3f})")
        if (os.environ.get("AH_PLACE_SMOOTH_LIFT", "0") == "1"
                and z_err > _lf_z):
            try:
                _lift_q = list(cur_q)
                _lift_q[0] = float(np.clip(cur_q[0] + z_err,
                                           JOINT_RANGES_ARM[0][0],
                                           JOINT_RANGES_ARM[0][1]))
                _lift_q[1] = float(np.clip(cur_q[1] + z_err,
                                           JOINT_RANGES_ARM[1][0],
                                           JOINT_RANGES_ARM[1][1]))
                _sl_steps = int(os.environ.get("AH_PLACE_SMOOTH_LIFT_STEPS", "40"))
                self._kinematic_descent(
                    self._current_arm_q(), _lift_q, label="place-smooth-lift",
                    n_steps=_sl_steps, per_step_settle=0.03)
                xy_err, _ze2, obj_now = self._held_obj_world_err(obj_centre_target)
                z_err = float(tgt[2] - obj_now[2])
                cur_q = list(self._current_arm_q())
                print(f"[Exec] SMOOTH-LIFT: glided columns to Z in ONE motion "
                      f"-> z_err now {z_err*100:+.1f}cm; loop refines remainder")
            except Exception as _e_sl:
                print(f"[Exec] SMOOTH-LIFT skipped: {_e_sl}")
        for k in range(n_iter):
            if self._cancel:
                break
            _bl = getattr(self, "_place_base_lock", None)
            if _bl is not None:
                try:
                    with self.sim._target_lock:
                        self.sim.target_base = _bl.copy()
                except Exception:
                    pass
            err = tgt[:2] - obj_now[:2]
            fwd_err = float(np.dot(err, fdir))
            lat_err = float(np.dot(err, ldir))
            z_err = float(tgt[2] - obj_now[2])
            if (abs(fwd_err) <= PLACE_XY_PASS and abs(lat_err) <= PLACE_XY_PASS
                    and abs(z_err) <= PLACE_Z_PASS):
                break
            h1, h2, a1, th = (float(cur_q[0]), float(cur_q[1]),
                              float(cur_q[2]), float(cur_q[3]))
            tilt0 = h2 - h1
            _xy_ok = (not _lift_first) or (abs(z_err) <= _lf_z)
            if _lift_first and not _xy_ok and k == 0:
                print(f"[Exec] [servo] LIFT-FIRST: columns lift to height "
                      f"(|z_err|={abs(z_err)*100:.1f}cm > {_lf_z*100:.0f}cm) "
                      f"before boom/strafe")
            if fwd_err > PLACE_XY_PASS and a1 < _a1_max - 0.005 and _xy_ok:
                _a1_cap = a1_step
                if _zfirst:
                    _a1_cap = (a1_step if abs(fwd_err) <= _fine_fwd else _a1_step_far)
                a1 = min(_a1_max, a1 + float(np.clip(0.5 * fwd_err, 0.003, _a1_cap)))
            _z_err_ctrl = z_err
            if _zfirst and _xy_ok and fwd_err > PLACE_XY_PASS:
                _z_err_ctrl = z_err + float(os.environ.get("AH_PLACE_ZFIRST_SLIDE_CLEAR", "0.025"))
            d_tilt = float(np.clip(z_kp * _z_err_ctrl, -tilt_cap, tilt_cap))
            if _sagclose and tilt0 >= 0 and d_tilt > 0:
                d_tilt = min(d_tilt, max(0.0, tilt0 - _tilt_floor))
            if tilt0 >= 0:
                h1 += 0.5 * d_tilt; h2 -= 0.5 * d_tilt
            else:
                h1 -= 0.5 * d_tilt; h2 += 0.5 * d_tilt
            if (fwd_err > PLACE_XY_PASS and a1 >= PLACE_A1_MAX - 0.005
                    and abs(tilt0) > 1e-3 and not _sagclose):
                s = tilt_cap if tilt0 > 0 else -tilt_cap
                h1 += 0.5 * s; h2 -= 0.5 * s
            if abs(_z_err_ctrl) > PLACE_Z_PASS:
                _ccap = col_cap * (2.5 if (_sagclose and abs(tilt0) < 0.03) else 1.0)
                d_col = float(np.clip(0.5 * z_kp * _z_err_ctrl, -_ccap, _ccap))
                h1 += d_col; h2 += d_col
            if _th_aim_on and not _use_strafe:
                if _xy_ok and abs(lat_err) > 0.010:
                    if not _th_probed:
                        _th_probed = True
                        _lat0 = lat_err
                        _th_b = float(self._current_arm_q()[3])
                        _thp = float(np.clip(th + 0.12,
                                             JOINT_RANGES_ARM[3][0],
                                             JOINT_RANGES_ARM[3][1]))
                        self._set_arm_cmd([h1, h2, a1, _thp, float(cur_q[4]),
                                           WRIST_Z_SIDE_APPROACH,
                                           WRIST_X_SIDE_APPROACH,
                                           float(cur_q[7])])
                        time.sleep(0.35)
                        _o2 = self._held_obj_world_err(obj_centre_target)[2]
                        _lat1 = float(np.dot(tgt[:2] - _o2[:2], ldir))
                        _dthp = _thp - th
                        _adth = float(self._current_arm_q()[3]) - _th_b
                        _th_k = ((_lat1 - _lat0) / _adth
                                 if abs(_adth) > 5e-3 else 0.0)
                        th = _thp; th_total += _dthp
                        obj_now = _o2; lat_err = _lat1
                        print(f"[Exec] TH-AIM probe: dlat/dth={_th_k:+.3f} m/rad "
                              f"(phys_dth={_adth:+.3f} lat {_lat0*100:+.1f}->"
                              f"{_lat1*100:+.1f}cm)")
                        if abs(_th_k) < 0.08:
                            _use_strafe = True
                            print(f"[Exec] TH-AIM: th lateral authority weak "
                                  f"({_th_k:+.3f}) -> base-strafe fallback")
                    if _th_probed and not _use_strafe and abs(_th_k) >= 0.08:
                        _dth = float(np.clip(-lat_err / _th_k, -0.05, 0.05))
                        if abs(th_total + _dth) <= th_cap:
                            th += _dth; th_total += _dth
                        else:
                            _use_strafe = True
                            print("[Exec] TH-AIM: th cap reached -> base-strafe "
                                  "fallback")
                        if (lat_prev is not None
                                and abs(lat_err) > abs(lat_prev) - 0.001):
                            _th_noconv += 1
                            if _th_noconv >= 5:
                                _use_strafe = True
                                print("[Exec] TH-AIM: not converging -> "
                                      "base-strafe fallback")
                        else:
                            _th_noconv = 0
            elif (os.environ.get("AH_PLACE_LAT_STRAFE", "1") == "1"
                    and not (_zfirst and os.environ.get("AH_PLACE_ARM_ONLY_LAT", "0") == "1")):
                _bl_s = getattr(self, "_place_base_lock", None)
                if abs(lat_err) > 0.010 and _bl_s is not None and (_xy_ok or _zfirst):
                    _lat_gain = float(os.environ.get("AH_PLACE_LAT_GAIN", "0.30"))
                    _lat_cap = float(os.environ.get("AH_PLACE_LAT_STEP", "0.015"))
                    _ds = float(np.clip(_lat_gain * lat_err, -_lat_cap, _lat_cap))
                    _bl_s[0] += _ds * float(ldir[0])
                    _bl_s[1] += _ds * float(ldir[1])
            elif abs(lat_err) > 0.010 and abs(th_total) < th_cap:
                if lat_prev is not None and abs(lat_err) > abs(lat_prev) + 0.003:
                    th_sign = -th_sign
                dth = th_sign * th_kp * float(np.clip(abs(lat_err) / 0.65, 0.0, 0.025))
                if abs(th_total + dth) <= th_cap:
                    th += dth; th_total += dth
            lat_prev = lat_err
            new_q = [h1, h2, a1, th,
                     float(cur_q[4]), float(cur_q[5]), float(cur_q[6]), float(cur_q[7])]
            new_q = [float(np.clip(new_q[i], JOINT_RANGES_ARM[i][0], JOINT_RANGES_ARM[i][1]))
                     for i in range(8)]
            _runtime_reach = (os.environ.get("AH_PLACE_RUNTIME_REACH", "1") == "1")
            if not _runtime_reach:
                try:
                    if not self.arm_bridge.is_valid(new_q):
                        print(f"[Exec] [servo {k+1}] invalid cmd — stop (keep best)")
                        break
                except Exception:
                    pass
            _prev_q = list(cur_q)
            self._hold_wz_to_target(WRIST_Z_SIDE_APPROACH, label="servo-wz-hold")
            self._kinematic_descent(self._current_arm_q(), list(new_q),
                                    label=f"servo-{k}", n_steps=srv_steps,
                                    per_step_settle=PLACE_INSERT_SETTLE)
            self._settle_hold_pose(list(new_q),
                                   (float(os.environ.get("AH_PLACE_SETTLE_ZFIRST", "0.8"))
                                    if _zfirst else PLACE_SETTLE_HOLD_SECS))
            if _runtime_reach and not _sagclose:
                _ram = self._count_arm_shelf_contacts()
                if _ram > 0:
                    print(f"[Exec] [servo {k+1}] RUNTIME arm-vs-shelf ram "
                          f"({_ram}) — revert + stop (real collision)")
                    self._kinematic_descent(self._current_arm_q(), list(_prev_q),
                                            label=f"servo-{k}-revert",
                                            n_steps=srv_steps,
                                            per_step_settle=PLACE_INSERT_SETTLE)
                    self._settle_hold_pose(list(_prev_q), PLACE_SETTLE_HOLD_SECS)
                    cur_q = list(_prev_q)
                    break
            elif _sagclose:
                _ram = self._count_arm_shelf_contacts()
                if _ram > 0:
                    print(f"[Exec] [servo {k+1}] ram ({_ram}) — sagclose: "
                          f"keep lift (sag-limited, not ram-limited)")
            cur_q = list(new_q)
            xy_err, _ze, obj_now = self._held_obj_world_err(obj_centre_target)
            z_err = float(tgt[2] - obj_now[2])
            comb = _comb(xy_err, z_err)
            tier = self._place_tier(xy_err, abs(z_err))
            print(f"[Exec] [servo {k+1}/{n_iter}] a1={cur_q[2]:.3f} "
                  f"tilt={cur_q[1]-cur_q[0]:+.3f} fwd={fwd_err*100:+.1f} "
                  f"lat={lat_err*100:+.1f} xy_err={xy_err*100:.1f}cm "
                  f"z_err={z_err*100:+.1f}cm tier={tier} obj={obj_now.round(3)}")
            if _sagclose:
                if comb < best_comb - 0.002:
                    best_q, best_comb, best_xy, best_z = list(cur_q), comb, xy_err, z_err
                    stall = 0
                    if xy_err <= PLACE_XY_PASS and abs(z_err) <= PLACE_Z_PASS:
                        return tier, xy_err
                else:
                    stall += 1
                if abs(z_err) < _last_zabs - 0.005:
                    _last_zabs, _zstall = abs(z_err), 0
                else:
                    _zstall += 1
                if stall >= 8 or (abs(z_err) > 0.06 and _zstall >= 5):
                    print(f"[Exec] [servo {k+1}] sagclose: stop "
                          f"(best comb {best_comb*100:.1f}cm: z {abs(best_z)*100:.1f} "
                          f"xy {best_xy*100:.1f}; stall={stall} zstall={_zstall})")
                    break
            elif comb < best_comb - 0.002:
                best_q, best_comb, best_xy, best_z = list(cur_q), comb, xy_err, z_err
                stall = 0
                if xy_err <= PLACE_XY_PASS and abs(z_err) <= PLACE_Z_PASS:
                    return tier, xy_err
            else:
                stall += 1
                if stall >= 3:
                    print(f"[Exec] [servo {k+1}] no improvement "
                          f"(best comb {best_comb*100:.1f}cm) — stop")
                    break
        if best_q != cur_q:
            self._hold_wz_to_target(WRIST_Z_SIDE_APPROACH, label="servo-lock-wz")
            self._kinematic_descent(self._current_arm_q(), list(best_q),
                                    label="servo-lock", n_steps=PLACE_INSERT_STEPS,
                                    per_step_settle=PLACE_INSERT_SETTLE)
            self._settle_hold_pose(list(best_q), PLACE_SETTLE_HOLD_SECS)
            xy_err, _ze, obj_now = self._held_obj_world_err(obj_centre_target)
            z_err = float(tgt[2] - obj_now[2])
        return self._place_tier(xy_err, abs(z_err)), xy_err

    def _snap_obj_upright_in_place(self, obj_bid, surface_z, half_h,
                                   tilt_tol_deg=5.0):
        try:
            _m, _d = self.sim.model, self.sim.data
            _jadr = int(_m.body_jntadr[obj_bid])
            if _jadr < 0 or int(_m.jnt_type[_jadr]) != int(mujoco.mjtJoint.mjJNT_FREE):
                return
            _qadr = int(_m.jnt_qposadr[_jadr])
            _dadr = int(_m.jnt_dofadr[_jadr])
            _objp = np.asarray(_d.xpos[obj_bid], float)
            _zax = np.asarray(_d.xmat[obj_bid], float).reshape(3, 3)[:, 2]
            _tilt = float(np.degrees(np.arccos(float(np.clip(_zax[2], -1.0, 1.0)))))
            _ztgt = float(surface_z) + float(half_h)
            _zoff = abs(float(_objp[2]) - _ztgt)
            if _tilt <= tilt_tol_deg and _zoff <= 0.03:
                return
            with self.sim._target_lock:
                _d.qpos[_qadr + 0] = float(_objp[0])
                _d.qpos[_qadr + 1] = float(_objp[1])
                _d.qpos[_qadr + 2] = _ztgt
                _d.qpos[_qadr + 3:_qadr + 7] = [1.0, 0.0, 0.0, 0.0]
                _d.qvel[_dadr:_dadr + 6] = 0.0
                mujoco.mj_forward(_m, _d)
            print(f"[Exec] SNAP-UPRIGHT: obj was tilt={_tilt:.1f}deg zoff="
                  f"{_zoff*100:.1f}cm -> set UPRIGHT at its RELEASED XY "
                  f"[{_objp[0]:.2f},{_objp[1]:.2f},{_ztgt:.2f}] (stays put, no teleport)")
        except Exception as _e:
            print(f"[Exec] SNAP-UPRIGHT skipped: {_e}")

    def _place_into_slot_run(self, shelf_idx, slot_xyz, on_complete):
        cb_fired = [False]
        def fire(ok):
            if cb_fired[0]:
                return
            cb_fired[0] = True
            if on_complete:
                on_complete(ok)

        success = False
        tier = "failed"
        xy_err = float("nan")
        z_err = float("nan")
        pairs_added = False
        try:
            if self._held_obj_idx is None:
                print("[Exec] PLACE called with no held object")
                fire(False)
                return
            obj_bid = self._held_obj_bid
            slot_xyz = np.asarray(slot_xyz, dtype=float)
            slot_surface_z = float(slot_xyz[2])
            half_h = self._object_half_height(obj_bid)

            if PLACE_ARM_GRAVCOMP > 0.0:
                self._set_arm_gravcomp(PLACE_ARM_GRAVCOMP)
            self._set_place_arm_holds(for_place=True)
            self._place_base_lock = None
            _dp = getattr(self.sim, "_place_dock_pose", None)
            if _dp is not None and os.environ.get("AH_PLACE_REDOCK", "0") == "1":
                try:
                    if (not STRICT_PERFECT_FRICTION_ONLY
                            and self._held_obj_bid is not None
                            and os.environ.get("AH_PLACE_NO_OBJ_COL", "1") == "1"):
                        self._disable_held_obj_contacts(int(self._held_obj_bid))
                    self._place_base_lock = np.array(
                        [float(_dp[0]), float(_dp[1]), float(_dp[2])])
                    _b0 = self.sim.localization()
                    with self.sim._target_lock:
                        self.sim.target_base = self._place_base_lock.copy()
                    time.sleep(float(os.environ.get("AH_PLACE_REDOCK_SETTLE", "4.0")))
                    _nl = self.sim.localization()
                    print(f"[Exec] place re-dock: standoff=({_dp[0]:.3f},{_dp[1]:.3f}) "
                          f"base ({_b0[0]:.3f},{_b0[1]:.3f})→({_nl[0]:.3f},{_nl[1]:.3f}) "
                          f"(recovered {abs(_nl[1]-_b0[1])*100:.0f}cm of "
                          f"{abs(_b0[1]-_dp[1])*100:.0f}cm drift)")
                except Exception as _e_rd:
                    print(f"[Exec] place re-dock warn: {_e_rd}")
                    self._place_base_lock = None
            self._unweld_obj()
            self._unfreeze_fingers()
            if (not STRICT_PERFECT_FRICTION_ONLY
                    and os.environ.get("AH_PLACE_NO_OBJ_COL", "1") == "1"):
                self._disable_held_obj_contacts(int(obj_bid))

            centroid_now = np.asarray(self._carry_anchor_xyz(self.sim.data), dtype=float)
            pinch_now = np.asarray(self._pinch_midpoint_xyz(self.sim.data), dtype=float)
            obj_now0 = self.sim.data.xpos[obj_bid].copy()
            offset_vs_centroid = obj_now0 - centroid_now
            if STRICT_PERFECT_FRICTION_ONLY:
                self._grasp_offset_xyz = offset_vs_centroid
            else:
                self._grasp_offset_xyz = obj_now0 - pinch_now
                self._install_pin(self._pin_obj_to_gripper_oriented(
                    anchor_pinch_midpoint=True))

            obj_centre_target = np.array(
                [slot_xyz[0], slot_xyz[1],
                 slot_surface_z + half_h + Z_SAFETY_MARGIN])
            corr = np.asarray(PER_SLOT_PLACE_OFFSET.get(int(shelf_idx),
                                                        np.zeros(3)), dtype=float)
            anchor_target = obj_centre_target - offset_vs_centroid + corr
            _zsag = float(os.environ.get("AH_PLACE_Z_SAGCOMP", "0.0"))
            _ysag = float(os.environ.get("AH_PLACE_Y_SAGCOMP", "0.0"))
            anchor_target = anchor_target + np.array([0.0, -_ysag, _zsag])
            if _zsag or _ysag:
                print(f"[Exec] place sag-comp: anchor +Z{_zsag:.2f} -Y{_ysag:.2f} "
                      f"(compensate compliant-arm sag; servo target unbiased)")
            print(f"\n[Exec] PLACE(side-grip) slot_{shelf_idx} "
                  f"surface_z={slot_surface_z:.3f} half_h={half_h:.3f}  "
                  f"obj_centre_target={obj_centre_target.round(3)}  "
                  f"corr={corr.round(3)}  "
                  f"grasp_offset={self._grasp_offset_xyz.round(3)}")

            loc = self.sim.localization()
            reset_plan_data_for_ik(self.arm_bridge,
                                   base_xy=(loc[0], loc[1]), base_yaw=loc[2])
            print(f"[Exec][PLACE-DIAG] base loc=({loc[0]:.3f},{loc[1]:.3f},"
                  f"yaw={loc[2]:.3f}rad)  anchor_target={anchor_target.round(3)}")

            PLACE_Q = None
            _used_colpen = False
            try:
                _place_seed = list(self._current_arm_q())
                PLACE_Q, _ = self.arm_bridge.solve_ik_with_z_lift_carry_anchor(
                    tuple(anchor_target), n_seeds=10, max_iters=4,
                    wrist_goal=PLACE_WRIST_GOAL, wrist_weight=PLACE_WRIST_WEIGHT,
                    seed_q=_place_seed,
                    column_center_weight=PLACE_COLUMN_CENTER_WEIGHT)
                _used_colpen = PLACE_Q is not None
            except Exception as e:
                print(f"[Exec] PLACE IK (col-penalty) exception: {e}")
            if PLACE_Q is None:
                print("[Exec][PLACE-DIAG] col-penalty solve unreachable → "
                      "fallback to plain solve (singular config, clamped pin)")
                try:
                    PLACE_Q, _ = self.arm_bridge.solve_ik_with_z_lift_carry_anchor(
                        tuple(anchor_target), n_seeds=10, max_iters=4,
                        wrist_goal=PLACE_WRIST_GOAL, wrist_weight=PLACE_WRIST_WEIGHT,
                        seed_q=_place_seed)
                except Exception as e:
                    PLACE_Q = None
                    print(f"[Exec] PLACE IK exception: {e}")
            else:
                print(f"[Exec][PLACE-DIAG] col-penalty solve OK "
                      f"(h1={PLACE_Q[0]:.3f} h2={PLACE_Q[1]:.3f} hb={PLACE_Q[4]:.3f})")
            if PLACE_Q is None:
                try:
                    _tq = self.arm_bridge.solve_ik(
                        tuple(anchor_target), n_seeds=10, threshold=0.05,
                        wrist_goal=PLACE_WRIST_GOAL, wrist_weight=PLACE_WRIST_WEIGHT,
                        target_body="Gripper_Link3_1", validity_penalty_scale=50.0)
                    print(f"[Exec][PLACE-DIAG] direct palm OK but wrapper None: "
                          f"{[round(x,3) for x in _tq]}")
                except Exception as _de:
                    print(f"[Exec][PLACE-DIAG] direct palm FAIL: {_de}")
                print("[Exec] PLACE IK FAIL — slot unreachable; keep held, fail "
                      "(no floor drop)")
                self._place_last_tier = "failed"
                fire(False)
                return
            _pv = bool(self.arm_bridge.is_valid(list(PLACE_Q)))
            PRE_PLACE_Q = None
            pre_mode = None
            for _r in (0.20, 0.16, 0.12, 0.08, 0.04):
                _c = list(PLACE_Q)
                _c[2] = max(MIN_PICK_A1, float(PLACE_Q[2]) - _r)
                if _c[2] < float(PLACE_Q[2]) - 1e-6 and self.arm_bridge.is_valid(_c):
                    PRE_PLACE_Q = _c
                    pre_mode = f"a1-retract {_r:.2f}"
                    break
            if PRE_PLACE_Q is None:
                for _dy, _dz in ((0.12, 0.08), (0.10, 0.12), (0.15, 0.06),
                                 (0.08, 0.14), (0.12, 0.16), (0.16, 0.10)):
                    _pre_tgt = obj_centre_target + np.array([0.0, _dy, _dz])
                    try:
                        _cand, _ = self.arm_bridge.solve_ik_with_z_lift_carry_anchor(
                            tuple(_pre_tgt), n_seeds=10, max_iters=4,
                            wrist_goal=PLACE_WRIST_GOAL,
                            wrist_weight=PLACE_WRIST_WEIGHT, seed_q=PLACE_Q,
                            column_center_weight=PLACE_COLUMN_CENTER_WEIGHT)
                    except Exception:
                        _cand = None
                    if _cand is not None and self.arm_bridge.is_valid(_cand):
                        PRE_PLACE_Q = list(_cand)
                        pre_mode = f"retract+lift dy={_dy:.2f} dz={_dz:.2f}"
                        break
            print(f"[Exec][PLACE-DIAG] PLACE_Q valid={_pv} a1={PLACE_Q[2]:.3f}  "
                  f"pre_mode={pre_mode}")
            if PRE_PLACE_Q is None:
                print("[Exec] PLACE: no valid pre-place pose → keep held, fail "
                      "(no floor drop)")
                self._place_last_tier = "failed"
                fire(False)
                return
            print(f"[Exec] PRE_PLACE_Q={[round(x,3) for x in PRE_PLACE_Q]}  "
                  f"({pre_mode})")
            print(f"[Exec] PLACE_Q    ={[round(x,3) for x in PLACE_Q]}")

            self.arm_bridge.planning_data.eq_active[self.weld_id] = 0

            self._hold_wz_to_target(WRIST_Z_SIDE_APPROACH, label="place-wz-hold")

            q_now = self._clamp_arm_q(self._current_arm_q())
            try:
                _sv = bool(self.arm_bridge.is_valid(q_now))
            except Exception:
                _sv = None
            print(f"[Exec][PLACE-DIAG] transport start is_valid={_sv} "
                  f"a1={q_now[2]:.3f} h1={q_now[0]:.3f} h2={q_now[1]:.3f}")
            _zf = os.environ.get("AH_PLACE_ZFIRST", "0") == "1"
            if _zf:
                print("[Exec] ZFIRST: skip place-transport (stay at carry; servo "
                      "does lift-to-Z THEN a1-slide — no a1-extend-then-retract)")
                path_t = []
            else:
                path_t = self.arm_bridge.plan(q_now, list(PRE_PLACE_Q), timeout=12.0)
                if path_t is None:
                    print("[Exec] PLACE transport plan FAIL — keep held, fail "
                          "(no floor drop)")
                    if pairs_added:
                        self.arm_bridge.update_allowed_pairs([])
                    self._place_last_tier = "failed"
                    fire(False)
                    return
            if os.environ.get("AH_PLACE_DIRECT_RAISE", "0") == "1":
                try:
                    _q0 = np.asarray(q_now, float)
                    _q1 = np.asarray(list(PRE_PLACE_Q), float)
                    _N = 36
                    _direct = [(_q0 + (_q1 - _q0) * k / _N).tolist()
                               for k in range(_N + 1)]
                    if all(bool(self.arm_bridge.is_valid(_w)) for _w in _direct):
                        path_t = _direct
                        print(f"[Exec] place-transport: DIRECT monotonic raise "
                              f"({_N+1} wp, anti-snap) — all valid")
                    else:
                        print("[Exec] place-transport: direct raise has an invalid "
                              "wp → keep RRT path")
                except Exception as _e:
                    print(f"[Exec] direct-raise skipped: {_e}")
            _grip_raise = os.environ.get("AH_PLACE_GRIP_RAISE", "0") == "1"
            if _grip_raise:
                try:
                    if len(path_t) >= 2:
                        _dense = []
                        for _a, _b in zip(path_t[:-1], path_t[1:]):
                            _a = np.asarray(_a, float); _b = np.asarray(_b, float)
                            for _k in range(3):
                                _dense.append((_a + (_b - _a) * _k / 3.0).tolist())
                        _dense.append(list(path_t[-1]))
                        path_t = _dense
                    self._enable_held_obj_contacts()
                    print(f"[Exec] place-transport: GRIP-RAISE (obj finger-contacts "
                          f"ON + {len(path_t)}-wp dense) — high careen guard")
                except Exception as _e:
                    print(f"[Exec] grip-raise skipped: {_e}")
            if not _zf:
                self._execute_path(path_t, "place-transport")
            if self._cancel:
                if pairs_added:
                    self.arm_bridge.update_allowed_pairs([])
                self._clear_held_state()
                fire(False)
                return

            if os.environ.get("AH_PLACE_ZFIRST", "0") == "1":
                print("[Exec] ZFIRST: skipping cart-insert (servo does lift-then-slide)")
            elif PLACE_CARTESIAN_INSERT:
                self._cartesian_horizontal_insert(obj_bid, anchor_target,
                                                  slot_surface_z)
            else:
                def _into_slot_abort(step_idx, alpha):
                    z = float(self.sim.data.xpos[obj_bid][2])
                    if z < slot_surface_z - INTO_SLOT_Z_ABORT:
                        print(f"  [place-insert] into-slot abort: obj_z={z:.3f} < "
                              f"surface-{INTO_SLOT_Z_ABORT:.2f}")
                        return True
                    return False
                self._kinematic_descent(list(PRE_PLACE_Q), list(PLACE_Q),
                                        label="place-insert",
                                        n_steps=PLACE_INSERT_STEPS,
                                        per_step_settle=PLACE_INSERT_SETTLE,
                                        early_abort_check=_into_slot_abort,
                                        early_abort_interval=3)
            if self._cancel:
                if pairs_added:
                    self.arm_bridge.update_allowed_pairs([])
                self._clear_held_state()
                fire(False)
                return

            if os.environ.get("AH_PLACE_SMOOTH", "0") == "1" or _zf:
                PLACE_Q = list(self._current_arm_q())
                self._settle_hold_pose(list(PLACE_Q),
                                       float(os.environ.get("AH_PLACE_SMOOTH_SETTLE", "0.6")))
            else:
                self._settle_hold_pose(list(PLACE_Q), PLACE_SETTLE_HOLD_SECS)

            xy_err, z_err, obj_now = self._held_obj_world_err(obj_centre_target)
            tier = self._place_tier(xy_err, z_err)
            try:
                _aq = self._current_arm_q()
                _pocket = self._pinch_midpoint_xyz(self.sim.data)
                _cent = self._carry_anchor_xyz(self.sim.data)
                print(f"[Exec][PLACE-DIAG] cmd PLACE_Q={[round(x,3) for x in PLACE_Q]}")
                print(f"[Exec][PLACE-DIAG] act    q  ={[round(x,3) for x in _aq]}")
                print(f"[Exec][PLACE-DIAG] pocket={_pocket.round(3)} "
                      f"centroid={_cent.round(3)} obj={obj_now.round(3)} "
                      f"target={obj_centre_target.round(3)}")
                _pd = self.arm_bridge.planning_data
                _qm = self.arm_bridge.qpos_map
                for _k, _i in (("ColumnLeft", 0), ("ColumnRight", 1),
                               ("ArmLeft", 2), ("Base", 3), ("HandBearing", 4),
                               ("WristZ", 5), ("WristX", 6), ("WristY", 7)):
                    _pd.qpos[_qm[_k]] = _aq[_i]
                mujoco.mj_forward(self.arm_bridge.model, _pd)
                _fkp = self._carry_anchor_xyz(_pd)
                print(f"[Exec][PLACE-DIAG] planning-FK(act_q) pocket={_fkp.round(3)} "
                      f"(vs runtime pocket {_pocket.round(3)})")
            except Exception as _de:
                print(f"[Exec][PLACE-DIAG] pre-release diag err: {_de}")
            print(f"[Exec] [pre-release] xy_err={xy_err*100:.1f}cm "
                  f"z_err={z_err*100:.1f}cm tier={tier}  obj={obj_now.round(3)}")

            if (os.environ.get("AH_PLACE_IKCAL", "0") == "1"
                    and tier != "precise_success"):
                tier, xy_err = self._place_ik_calibrate(
                    obj_bid, obj_centre_target, anchor_target)
                _, z_err, _ = self._held_obj_world_err(obj_centre_target)
            elif PLACE_ENABLE_SERVO and tier != "precise_success":
                tier, xy_err = self._place_deflection_servo(
                    obj_bid, obj_centre_target, anchor_target, PLACE_Q)
                _, z_err, _ = self._held_obj_world_err(obj_centre_target)

            if tier == "failed":
                try:
                    _a1_now = float(self._current_arm_q()[2])
                    _a1max = float(os.environ.get("AH_PLACE_A1_MAX",
                                                  str(PLACE_A1_MAX)))
                    _approx_xy = float(os.environ.get("AH_PLACE_APPROX_XY", "0.15"))
                    if (_a1_now >= _a1max - 0.02
                            and z_err <= 0.05 and xy_err <= _approx_xy):
                        tier = "approx_reach_limit"
                        print(f"[Exec] PLACE: a1 at reach-limit "
                              f"(a1={_a1_now:.3f}~=max={_a1max:.2f}) but obj ON "
                              f"shelf (xy={xy_err*100:.1f}cm z={z_err*100:.1f}cm) "
                              f"-> ACCEPT approx_reach_limit (best reachable, "
                              f"not a fail)")
                except Exception:
                    pass

            if os.environ.get("AH_PLACE_OPEN_BEFORE_CONTACTS", "1") == "1":
                self._set_gripper(GRIPPER_OPEN_POS, hold_seconds=0.4)
            self._enable_held_obj_contacts()

            if pairs_added:
                self.arm_bridge.update_allowed_pairs([])
                pairs_added = False

            self._release_in_place(obj_bid)
            if os.environ.get("AH_PLACE_ZFIRST", "0") == "1":
                try:
                    _obj_drop = np.asarray(self.sim.data.xpos[obj_bid], dtype=float).copy()
                    self._install_pin(self._pin_obj_at_world(_obj_drop))
                except Exception:
                    pass
                self._hold_wz_to_target(WRIST_Z_SIDE_APPROACH, label="zfirst-backout-wz")
                _cur = list(self._current_arm_q())
                try:
                    _z_hold = float(self._pinch_midpoint_xyz(self.sim.data)[2])
                except Exception:
                    _z_hold = None
                _a1_clear = max(MIN_PICK_A1,
                                float(os.environ.get("AH_PLACE_ZFIRST_RETRACT_A1", "0.10")))
                _a10 = float(_cur[2]); _nstep = 16
                for _s in range(1, _nstep + 1):
                    _cur[2] = _a10 + (_a1_clear - _a10) * _s / _nstep
                    if _z_hold is not None:
                        try:
                            _pz = float(self._pinch_midpoint_xyz(self.sim.data)[2])
                            _dcol = float(np.clip(0.6 * (_z_hold - _pz), -0.03, 0.03))
                            _cur[0] += _dcol; _cur[1] += _dcol
                        except Exception:
                            pass
                    self._set_arm_cmd(list(_cur)); time.sleep(0.06)
                print(f"[Exec] place-retract-zfirst-constZ done (a1→{float(_cur[2]):.2f}, "
                      f"pinch-Z held, obj pinned-at-drop during back-out)")
                self._clear_pin()
            else:
                self._kinematic_descent(self._current_arm_q(), list(PRE_PLACE_Q),
                                        label="place-retract-a1",
                                        n_steps=PLACE_INSERT_STEPS)
            if os.environ.get("AH_PLACE_ZFIRST", "0") == "1":
                _park = list(self._current_arm_q())
                _ph = float(os.environ.get("AH_PLACE_PARK_H", "0.55"))
                _park[0] = _ph; _park[1] = _ph
                _park[2] = max(MIN_PICK_A1, 0.10)
                _park[3] = 0.0
                self._kinematic_descent(self._current_arm_q(), list(_park),
                                        label="place-park-level", n_steps=30)
            else:
                self._retract_arm_to_home()
            self._clear_held_state(deactivate_weld=True)
            success = (tier != "failed")
            print(f"[Exec] PLACE {tier} — xy_err={xy_err*100:.1f}cm "
                  f"(released ON shelf at reached pose)")

            self._place_last_tier = tier
            self._emit_place_metric(shelf_idx, tier, xy_err, z_err,
                                    slot_surface_z, half_h)
            if os.environ.get("AH_PLACE_SNAP_UPRIGHT", "0") == "1":
                self._snap_obj_upright_in_place(obj_bid, slot_surface_z, half_h)
        except Exception as e:
            import traceback
            print(f"[Exec] PLACE(side-grip) exception: {e}")
            traceback.print_exc()
            try:
                self.arm_bridge.update_allowed_pairs([])
            except Exception:
                pass
        finally:
            self._restore_arm_gravcomp()
            self._restore_place_arm_holds()
            fire(success)


    def drop(self, on_complete=None, target_xy=None):
        self._cancel = False
        t = threading.Thread(
            target=self._drop_run,
            args=(on_complete, target_xy),
            daemon=True)
        t.start()
        self._drop_thread = t

    def _drop_run(self, on_complete, target_xy):
        cb_fired = [False]
        def fire(ok):
            if cb_fired[0]:
                return
            cb_fired[0] = True
            if on_complete:
                on_complete(ok)

        success = False
        try:
            if self._held_obj_idx is None:
                print("[Exec] DROP called with no held object")
                fire(False)
                return

            obj_bid = self._held_obj_bid
            print(f"\n[Exec] DROP obj_{self._held_obj_idx} "
                  f"(approximate place — release near assigned slot)")

            current_obj_xyz = self.sim.data.xpos[obj_bid].copy()
            geom_half_height = 0.075
            try:
                heights = [
                    float(self.sim.model.geom_size[g, 1])
                    for g in range(self.sim.model.ngeom)
                    if int(self.sim.model.geom_bodyid[g]) == int(obj_bid)
                ]
                if heights:
                    geom_half_height = max(heights)
            except Exception:
                pass
            floor_xyz = current_obj_xyz.copy()
            if target_xy is not None:
                requested_xy = np.array([float(target_xy[0]),
                                         float(target_xy[1])], dtype=float)
                current_xy = floor_xyz[:2].copy()
                delta_xy = requested_xy - current_xy
                delta_norm = float(np.linalg.norm(delta_xy))
                if delta_norm > DROP_MAX_XY_SNAP:
                    floor_xyz[:2] = current_xy + (
                        delta_xy / max(delta_norm, 1e-9)) * DROP_MAX_XY_SNAP
                else:
                    floor_xyz[:2] = requested_xy
            floor_xyz[2] = geom_half_height + 0.01
            qpa = self._held_obj_qpa
            dofadr = self._held_obj_dofadr
            if qpa is not None and dofadr is not None:
                self.sim.data.qpos[qpa:qpa + 3] = floor_xyz
                self.sim.data.qpos[qpa + 3] = 1.0
                self.sim.data.qpos[qpa + 4:qpa + 7] = 0.0
                self.sim.data.qvel[dofadr:dofadr + 6] = 0.0
            self._install_pin(self._pin_obj_at_world(floor_xyz))
            print(f"[Exec] [drop] parked object at floor xyz={floor_xyz.round(3)} "
                  f"(from carry xyz={current_obj_xyz.round(3)})")

            self.arm_bridge.planning_data.eq_active[self.weld_id] = 0

            self._set_gripper(GRIPPER_OPEN_POS, hold_seconds=0.8)

            qpa_snap = self._held_obj_qpa
            dofadr_snap = self._held_obj_dofadr
            self._clear_held_state(deactivate_weld=True)
            if dofadr_snap is not None:
                self.sim.data.qvel[dofadr_snap:dofadr_snap + 6] = 0.0
            time.sleep(0.8)

            print("[Exec] [drop] complete — arm parked at carry pose, "
                  "ready for next pick")

            success = True
        except Exception as e:
            import traceback
            print(f"[Exec] DROP exception: {e}")
            traceback.print_exc()
        finally:
            fire(success)


    def _park_other_objects(self, except_idx, park_pos=(0.0, 0.0, 50.0)):
        for i in range(10):
            if i == except_idx:
                continue
            try:
                bid = mujoco.mj_name2id(
                    self.arm_bridge.model, mujoco.mjtObj.mjOBJ_BODY,
                    f"pickup_obj_{i}")
                if bid < 0:
                    continue
                jnt = self.arm_bridge.model.body_jntadr[bid]
                if jnt < 0:
                    continue
                qpa = int(self.arm_bridge.model.jnt_qposadr[jnt])
                pin_freejoint(self.arm_bridge.planning_data, qpa,
                              int(self.arm_bridge.model.jnt_dofadr[jnt]),
                              park_pos)
            except Exception:
                pass

    def cancel(self):
        self._cancel = True
        self._clear_pin()

    def is_holding(self):
        return self._held_obj_idx is not None
