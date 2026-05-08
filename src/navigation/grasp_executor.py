"""
grasp_executor.py — OMPL pick-and-place executor.

Hold mechanism:
    pin_freejoint (object qpos set every sim step) plus the grasp_left
    weld constraint so OMPL transport sees the carried object as part
    of robot geometry.

Execution mode:
    PD control via direct_arm_commands[0:4] (ARM1 only).  ARM2 is parked
    at PARK_Q.  Pin callbacks are registered with sim.add_pin_callback
    so they run inside step_simulation right before mj_step.

Pipeline:
    [3]  Open gripper
    [4]  OMPL approach   HOME -> PRE_GRASP_Q          pin: object at world pos
    [5]  Linear descent  PRE_GRASP_Q -> GRASP_Q       pin: object at world pos
    [6]  Close gripper + activate weld + record grasp_offset
    [7]  OMPL transport  GRASP_Q -> ABOVE_SHELF_Q     pin: object on gripper
    [8]  Linear descent  ABOVE_SHELF_Q -> PLACE_Q     pin: object on gripper
    [9]  Open gripper + deactivate weld + pin object at drop
    [10] OMPL retract    PLACE_Q -> HOME_Q            pin: object held at drop
"""

import math
import time
import threading
import numpy as np
import mujoco

from navigation.arm_planner import HOME_Q, PARK_Q
from navigation.finger_planner import FingerBridge, FINGER_DOF


# ── Tunables ─────────────────────────────────────────────────────────────
ABOVE_OBJ_HEIGHT      = 0.10    # PRE_GRASP IK = obj_pos + [0, 0, ABOVE_OBJ_HEIGHT]
GRIPPER_STANDOFF_XY   = 0.05    # XY standoff from object centre to IK target
STANDOFF_RADIUS_CLEAR = 0.02    # effective standoff = max(GRIPPER_STANDOFF_XY, radius + this)
DESCENT_STEPS         = 25      # waypoints in PRE_GRASP_Q -> GRASP_Q linear interp
SHELF_ABOVE_HEIGHT    = 0.20    # ABOVE_SHELF: object target this far above shelf
SHELF_PLACE_OFFSET_Z  = 0.05    # PLACE: object target this far above shelf surface

# Carry pose joint values used after grasp to lift the object before base
# transport.
CARRY_H1              = 0.60
CARRY_H2              = 0.65
CARRY_A1              = 0.10

# Pre-close acceptance gate.  IK runs against plan_data which does not
# include passive RotationLeftJoint deflection, so the runtime wrist may
# fall short of the IK target.  We measure deviation from the IK target
# (not raw distance to the object) because the IK target is intentionally
# placed GRIPPER_STANDOFF_XY behind the object by design.
GRIP_DEVIATION_TOLERANCE = 0.08
# Margin above the expected standoff for the obj-grip XY check.
GRIP_OBJ_XY_TOLERANCE = 0.05

# Place verification: drop_xyz must be within PLACE_XY_TOLERANCE of the
# shelf target (XY) and within 5cm vertically.
PLACE_XY_TOLERANCE    = 0.12

# Geometry guards.
MIN_PICK_WRIST_Z      = 0.44    # minimum wrist Z target for the pick descent
MAX_PICK_H_DIFF       = 0.18    # max |h2 - h1| for an accepted IK solution
MIN_PICK_A1           = 0.16    # visual guard: avoid folding gripper into body
# Snap to a fixed offset when the recorded grasp_offset is too large
# (wrist couldn't reach the object) so transport/place stays clean.
GRASP_OFFSET_SNAP_THRESHOLD = 0.35   # if |raw_offset| > this -> snap to FIXED
GRASP_OFFSET_XY_SNAP_THRESHOLD = 0.03  # snap when horizontal component alone is too large
GRASP_OFFSET_XY_CARRY = 0.22
GRASP_TOP_CLEARANCE   = 0.05    # object top sits this far below the gripper ref
GRASP_OFFSET_Z_MIN    = 0.09    # don't pull tiny objects into wrist geometry
GRASP_OFFSET_Z_SNAP_THRESHOLD = 0.05
FIXED_GRASP_OFFSET    = np.array([0.0, 0.0, -0.12])  # fallback/default only
DROP_MAX_XY_SNAP      = 0.15    # max XY teleport at release so drops don't look like magic

GRIPPER_OPEN_POS      = -0.55   # open-hand pre-grasp ctrl
GRIPPER_CLOSE_POS     =  0.30   # full close (upper bound for the size-aware close)
GRIPPER_PIN_HOLD_POS  =  0.00   # legacy relaxed hold; superseded by size-aware hold

# Size-aware finger close mapping.  Smaller objects need fingers closed
# more; larger objects need fingers stopping earlier so they appear to
# wrap around the cylinder surface.
#   close_ctrl = FINGER_CLOSE_MAX - FINGER_CLOSE_PER_M * radius
# clamped to [FINGER_CLOSE_FLOOR, FINGER_CLOSE_MAX].
FINGER_CLOSE_MAX      = 0.20    # max close ctrl (smallest object)
FINGER_CLOSE_FLOOR    = 0.12    # min close ctrl across all object sizes
FINGER_CLOSE_PER_M    = 4.0     # ctrl backs off this fast as radius grows

# Smooth-attach interpolation: when the object pin switches from
# "pinned at world pose" to "pinned at gripper offset", interpolate over
# this duration instead of teleporting.
SMOOTH_ATTACH_SECS    = 0.4
# Hold time after the pin is installed so the smooth-attach animation
# completes before the arm starts the lift.
SMOOTH_ATTACH_SETTLE  = 0.45

# Finger actuator layout: gripper_ids_left has 9 finger joint actuators
# arranged as 3 fingers x 3 joints (proximal, middle, distal).
#   0,1,2 = finger_c (j1, j2, j3)
#   3,4,5 = finger_b (j1, j2, j3)
#   6,7,8 = finger_a (j1, j2, j3)
# Indices 9, 10 are palm spread joints (palm_finger_c, palm_finger_b).
FINGER_BASE_INDICES   = (0, 3, 6)   # base index of each finger (j1)

# Curl profile: distal joints curl more than proximal so the fingertip
# wraps around the object surface.
CURL_J1_FACTOR        = 0.70    # proximal (joint_1)
CURL_J2_FACTOR        = 0.90    # middle (joint_2)
CURL_J3_FACTOR        = 1.00    # distal (joint_3) — full close

# Smooth gripper close/open animation: interpolate ctrl values over this
# duration instead of snapping.
SMOOTH_GRIPPER_SECS   = 0.4
SMOOTH_GRIPPER_STEP_S = 0.02    # per-step sleep — ~50 Hz interpolation

# Feature flag: contact-stop close.  When True, _set_gripper monitors
# per-finger contacts with the held object during the close interpolation;
# each finger freezes its joint ctrls the moment its geoms touch the
# cylinder.  When False, falls back to position-only close.
USE_CONTACT_STOP_CLOSE = True

# Palm spread ctrl values.  Idx 9 = palm_finger_c_joint_1, idx 10 =
# palm_finger_b_joint_1.  Driving them wider during approach and back to
# narrow at grasp produces an "open hand reaching, then closing on object"
# choreography.
PALM_SPREAD_OPEN      = 0.18    # ctrl when gripper is open / approaching
PALM_SPREAD_CLOSE     = 0.0     # ctrl when gripper is grasping (default narrow)

# Per-finger stagger between successive fingers starting their close motion.
FINGER_STAGGER_SECS   = 0.06

GRIPPER_HOLD_TIME     = 1.5     # s — let fingers settle after open/close
PD_SETTLE_PER_WAYPOINT = 0.05   # s — pause between waypoints
PD_SETTLE_AT_PATH_END = 0.6     # s — pause after final waypoint for PD convergence


# ── Helpers ──────────────────────────────────────────────────────────────

def compute_grasp_targets(base_xy, obj_world, obj_radius=None):
    """
    Compute (grasp_target, pre_grasp_target) using the reference's
    approach-vector standoff pattern.  Both virtual screening and the
    runtime GraspExecutor must agree on this — otherwise screening passes
    but execution fails.

    Args:
        base_xy:    (x, y) of robot base (or candidate base for screening)
        obj_world:  (3,) object world position
        obj_radius: optional object cylinder radius.  When given, the
                    standoff adapts so the wrist sits at least
                    STANDOFF_RADIUS_CLEAR outside the object surface
                    (otherwise the wrist gets buried in big cylinders
                    and fingers go through the body).

    Returns:
        grasp_target:     IK target for GRASP_Q (palm at obj height,
                          pulled back effective_standoff along approach)
        pre_grasp_target: IK target for PRE_GRASP_Q (grasp_target + Z offset)
    """
    obj = np.asarray(obj_world, dtype=float)
    base_xy = np.asarray(base_xy, dtype=float)
    approach = obj[:2] - base_xy
    nrm = float(np.linalg.norm(approach))
    if nrm < 1e-6:
        approach_unit = np.array([1.0, 0.0])
    else:
        approach_unit = approach / nrm
    # obj_radius is retained in the signature for API symmetry; the fixed
    # standoff aligns the fingertips with the object surface for the
    # current cylinder radius range.
    standoff = GRIPPER_STANDOFF_XY
    grasp_target = obj.copy()
    grasp_target[0] -= approach_unit[0] * standoff
    grasp_target[1] -= approach_unit[1] * standoff
    pre_grasp_target = grasp_target.copy()
    pre_grasp_target[2] += ABOVE_OBJ_HEIGHT
    return grasp_target, pre_grasp_target


def reset_plan_data_for_ik(arm_bridge, base_xy, base_yaw):
    """
    Reset arm_bridge.planning_data to a deterministic clean state for IK/OMPL.

    Always uses set_base_pose_xy_yaw — clean XY + yaw only, no roll/pitch
    contamination from physics.  Caller must provide base_xy and base_yaw
    explicitly.  For runtime execution, extract them from sim.localization().
    For virtual screening, use the candidate base position + face-object yaw.

    ARM1 set to HOME_Q, ARM2 to PARK_Q, all pickup objects parked, qvel
    zeroed, and mj_forward called.  Identical plan_data state for both
    screening and execution paths.
    """
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
    """Pin a body's free joint at a fixed world (pos, quat) with zero velocity."""
    data.qpos[qposadr]     = pos[0]
    data.qpos[qposadr + 1] = pos[1]
    data.qpos[qposadr + 2] = pos[2]
    data.qpos[qposadr + 3] = quat[0]
    data.qpos[qposadr + 4] = quat[1]
    data.qpos[qposadr + 5] = quat[2]
    data.qpos[qposadr + 6] = quat[3]
    data.qvel[dofadr:dofadr + 6] = 0.0


# ── Executor ─────────────────────────────────────────────────────────────

class GraspExecutor:
    """
    Pick-and-place executor.  Constructed once in play_m1.py main(), then
    .pick(obj_idx, obj_world, ...) and .place(shelf_idx, shelf_pos, ...) are
    called from a background thread per task.
    """

    def __init__(self, sim, arm_bridge):
        self.sim        = sim
        self.arm_bridge = arm_bridge

        # Resolve weld + body indices once
        self.weld_id = mujoco.mj_name2id(
            sim.model, mujoco.mjtObj.mjOBJ_EQUALITY, "grasp_left")
        if self.weld_id < 0:
            raise RuntimeError("grasp_left weld not found in model")

        # Gripper_Link3_1 is the palm body where the three fingers attach
        # via their palm_finger_*_joint_1 joints.  Used as the pin
        # reference so the raw grasp_offset stays small.
        self.gripper_body_id = mujoco.mj_name2id(
            sim.model, mujoco.mjtObj.mjOBJ_BODY, "Gripper_Link3_1")
        if self.gripper_body_id < 0:
            raise RuntimeError("Gripper_Link3_1 body not found in model")
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

        # Per-finger body groups for contact-stop close.  Order matches
        # gripper_ids_left:
        #   index 0 = finger_c (gripper joints 0-2)
        #   index 1 = finger_b (gripper joints 3-5)
        #   index 2 = finger_a (gripper joints 6-8)
        # finger_a has no link_0_1 in this model (only b and c have a
        # palm-spread joint).  We probe link_0_1 through link_3_1 and
        # accept whatever exists; the tip body (link_3_1) is the most
        # important and is present on all three fingers.
        self._finger_body_groups = []
        finger_labels = ['c', 'b', 'a']
        for label in finger_labels:
            bodies = set()
            for ln in (0, 1, 2, 3):
                name = f"finger_{label}_link_{ln}_1"
                bid = mujoco.mj_name2id(
                    sim.model, mujoco.mjtObj.mjOBJ_BODY, name)
                if bid >= 0:
                    bodies.add(int(bid))
            self._finger_body_groups.append(bodies)
        # Contact-stop is usable as long as each finger has at least
        # one body resolved (typically the tip, link_3_1).
        if not all(len(g) >= 1 for g in self._finger_body_groups):
            print("[Exec] contact-stop warning: a finger has no resolved "
                  "bodies; contact-stop close will be disabled")
        else:
            counts = [len(g) for g in self._finger_body_groups]
            print(f"[Exec] contact-stop bodies per finger c/b/a = {counts}")

        # OMPL motion planner for the gripper's 11 finger DOFs.  The grasp
        # closure / open trajectory is generated via this planner; see
        # navigation/finger_planner.py for the state space and planner
        # setup.  _set_gripper walks the returned waypoints with
        # smoothstep timing.
        try:
            self._finger_bridge = FingerBridge(
                sim.model, sim.gripper_ids_left[:FINGER_DOF])
        except Exception as e:
            print(f"[Exec] FingerBridge init warning: {e} — "
                  "falling back to direct interpolation in _set_gripper")
            self._finger_bridge = None

        # Hold state
        self._held_obj_idx     = None
        self._held_obj_bid     = None
        self._held_obj_qpa     = None
        self._held_obj_dofadr  = None
        self._grasp_offset_xyz = None     # obj_xyz - gripper_xyz at grasp moment
        self._grasp_offset_quat = (1.0, 0.0, 0.0, 0.0)
        # Gravity compensation: while held, set body_gravcomp[obj_bid] = 1.0
        # so MuJoCo cancels gravity for this body.  Without this, gravity
        # drifts the object between pin snaps and causes visible jitter.
        self._held_obj_orig_gravcomp = None
        # While the object is pinned to the gripper, soften its contact
        # solver references (solref/solimp) so fingers stop at the
        # surface without fighting the pin solver.  Saved as a list of
        # (geom_id, solref_copy, solimp_copy) for restoration on release.
        self._held_obj_solref_saved = None

        # Active pin callback (function reference, registered with sim)
        self._active_pin_fn = None
        self._cancel = False

        # ARM1 qpos map (already built by arm_bridge — alias for clarity)
        self._qmap = arm_bridge.qpos_map

        print("[Exec] GraspExecutor ready  weld_id=%d  gripper_body_id=%d" %
              (self.weld_id, self.gripper_body_id))

        # One-time diagnostic: verify all 9 finger actuators + 2 palm
        # actuators are resolved and report which slot drives which joint.
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

    # ── Pin closures ─────────────────────────────────────────────────────

    def _pin_obj_at_world(self, world_pos):
        """Return a closure that pins the held object at world_pos every step."""
        qpa = self._held_obj_qpa
        dofadr = self._held_obj_dofadr
        wp = np.array(world_pos, dtype=float).copy()
        def closure(data):
            pin_freejoint(data, qpa, dofadr, wp)
        return closure

    def _carry_anchor_xyz(self, data):
        """World XYZ of the closed-finger pocket used for visual carry."""
        if len(self._carry_anchor_body_ids) == 3:
            pts = [data.xpos[bid] for bid in self._carry_anchor_body_ids]
            return np.mean(pts, axis=0)
        return data.xpos[self.gripper_body_id]

    def _finger_touches_obj(self, finger_idx, obj_bid):
        """Return True iff any sim contact pairs a body on finger
        `finger_idx`'s chain with the held object body `obj_bid`.
        Used by contact-stop close to freeze a finger's ctrls at the
        moment its tip meets the cylinder."""
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

    def _pin_obj_to_gripper(self):
        """Return a closure that pins the held object at fixed offset from the finger pocket."""
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
        """Best-effort vertical half-height for a pickup object body."""
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
        """Best-effort cylinder radius for a pickup object body.

        For cylinder geoms, geom_size[0] is the radius.  Returns the
        maximum across all geoms on the body so the finger close pose
        accounts for the outermost surface.
        """
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

    def _finger_close_for_radius(self, radius):
        """Map object radius to a finger ctrl value such that the closed
        fingers contact the object's outer surface.  Used by the [6]
        close-gripper command and by the post-pin carry hold pose so
        the fingers stay closed around the object during transport."""
        pos = FINGER_CLOSE_MAX - FINGER_CLOSE_PER_M * float(radius)
        return max(FINGER_CLOSE_FLOOR, min(FINGER_CLOSE_MAX, pos))

    def _pin_obj_to_gripper_animated(self, start_world_xyz,
                                      duration=SMOOTH_ATTACH_SECS):
        """Pin closure that smoothly interpolates the object from its
        current world position to the gripper-relative carry offset over
        `duration` seconds using smoothstep easing (3t^2 - 2t^3).  After
        interpolation completes the closure behaves identically to the
        static `_pin_obj_to_gripper`."""
        qpa = self._held_obj_qpa
        dofadr = self._held_obj_dofadr
        offset = self._grasp_offset_xyz.copy()
        start_pos = np.asarray(start_world_xyz, dtype=float).copy()
        start_t = time.time()

        def closure(data):
            elapsed = time.time() - start_t
            alpha = min(1.0, elapsed / duration)
            grip_xyz = self._carry_anchor_xyz(data)
            attached = (
                grip_xyz[0] + offset[0],
                grip_xyz[1] + offset[1],
                grip_xyz[2] + offset[2],
            )
            if alpha < 1.0:
                # Smoothstep: 3t^2 - 2t^3
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

    def _visual_grasp_offset(self, obj_bid, raw_offset, raw_xy_dist):
        """Return the canonical carry offset (object centered under the gripper)."""
        snap_offset = FIXED_GRASP_OFFSET.copy()
        snap_offset[0] = 0.0
        snap_offset[1] = 0.0
        half_h = self._object_half_height(obj_bid)
        snap_offset[2] = -max(GRASP_OFFSET_Z_MIN,
                              half_h + GRASP_TOP_CLEARANCE)
        return snap_offset

    def _soften_held_obj_contacts(self, obj_bid):
        """Soften contact solver references for the held object's geoms
        so fingers stop at the surface without the contact forces
        fighting the pin closure solver.  Overrides `geom_solref`
        (longer time constant) and `geom_solimp` (lower impedance);
        original per-geom values are saved for restoration on release.
        """
        if self._held_obj_solref_saved is not None:
            return
        # Soft contact parameters: lengthened time constant and low
        # impedance so contact response is gentle but still prevents
        # visual interpenetration with finger geoms.
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

    def _clear_held_state(self, deactivate_weld=True):
        """Reset all hold state.  Idempotent.  Called on any pick/place
        failure path so subsequent attempts start clean."""
        self._clear_pin()
        if deactivate_weld:
            try:
                # Weld is only ever active in plan_data; touching
                # sim_data.eq_active mid-simulation triggers
                # `mj_makeConstraint: nefc mis-allocation`.
                self.arm_bridge.planning_data.eq_active[self.weld_id] = 0
            except Exception:
                pass
        # Restore the original body_gravcomp value (was set to 1.0 while
        # pinned to the gripper).  Only restores if a value was saved
        # and we still know which body to restore it on.
        if (self._held_obj_orig_gravcomp is not None
                and self._held_obj_bid is not None):
            try:
                self.sim.model.body_gravcomp[self._held_obj_bid] = \
                    self._held_obj_orig_gravcomp
            except Exception:
                pass
        self._restore_held_obj_contacts()
        self._held_obj_orig_gravcomp = None
        self._held_obj_idx     = None
        self._held_obj_bid     = None
        self._held_obj_qpa     = None
        self._held_obj_dofadr  = None
        self._grasp_offset_xyz = None

    # ── Object resolution ────────────────────────────────────────────────

    def _resolve_obj(self, obj_idx):
        """Return (body_id, qposadr, dofadr) for pickup_obj_<obj_idx>."""
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

    # ── Arm execution (PD, via direct_arm_commands) ──────────────────────

    def _set_arm_cmd(self, q):
        with self.sim._target_lock:
            self.sim.use_ik = False
            c = self.sim.direct_arm_commands.copy()
            c[0] = q[0]; c[1] = q[1]; c[2] = q[2]; c[3] = q[3]
            self.sim.direct_arm_commands = c

    def _execute_path(self, path, label="path"):
        if not path:
            print(f"  [{label}] empty path, skipping")
            return
        n = len(path)
        for wi, wp in enumerate(path):
            if self._cancel: return
            self._set_arm_cmd(wp)
            time.sleep(PD_SETTLE_PER_WAYPOINT)
            if wi % max(1, n // 5) == 0:
                print(f"  [{label}] wp {wi+1:02d}/{n}  "
                      f"h1={wp[0]:.3f} h2={wp[1]:.3f} a1={wp[2]:.3f} th={wp[3]:.3f}")
        time.sleep(PD_SETTLE_AT_PATH_END)
        print(f"  [{label}] done — final cmd written")

    def _kinematic_descent(self, q_start, q_end, label="descent",
                           n_steps=DESCENT_STEPS):
        """Linear joint-space interpolation, executed via PD waypoints."""
        for i in range(n_steps):
            if self._cancel: return
            alpha = (i + 1) / n_steps
            q = [q_start[j] + alpha * (q_end[j] - q_start[j]) for j in range(4)]
            self._set_arm_cmd(q)
            time.sleep(PD_SETTLE_PER_WAYPOINT)
        time.sleep(PD_SETTLE_AT_PATH_END)
        print(f"  [{label}] done  →  h1={q_end[0]:.3f} h2={q_end[1]:.3f} "
              f"a1={q_end[2]:.3f} th={q_end[3]:.3f}")

    # ── Gripper ──────────────────────────────────────────────────────────

    @staticmethod
    def _curl_targets(pos):
        """Map a single close/open intensity to per-joint ctrl targets
        for all 9 finger joints (3 fingers x {proximal, middle, distal}).

        The three finger joints have different ctrl ranges in this
        model (DELTO M3-style gripper):
          j1 (proximal): [-1.0,   1.22]   positive ctrl curls inward
          j2 (middle):   [ 0.0,   1.57]   positive ctrl curls inward
          j3 (distal):   [-1.22, -0.052]  negative ctrl curls inward
                                          (rest at -0.052, full curl at -1.22)

        For closing (pos >= 0) we scale a [0, 1] intensity across each
        joint's curl direction; the distal joint curls more than the
        middle which curls more than the proximal so the fingertips
        wrap around the object.  For opening (pos < 0) all joints
        return to their rest/open positions.

        Returns a list of 9 target ctrl values in gripper_ids_left order.
        """
        if pos >= 0.0:
            # `pos` is a close-ctrl value in roughly [0, 0.20].  Map to a
            # [0, 1] intensity for joint-range scaling.
            intensity = min(1.0, max(0.0, pos / 0.20))
            j1 = intensity * CURL_J1_FACTOR * 0.85
            j2 = intensity * CURL_J2_FACTOR * 0.95
            # j3 is negative-curl: rest = -0.052, fully curled = -1.22.
            j3 = -0.052 - intensity * CURL_J3_FACTOR * 1.10
        else:
            # Opening — return joints to their rest/open positions.
            j1 = pos                # negative ctrl drives j1 outward
            j2 = 0.0                # j2 rest at lower bound of its range
            j3 = -0.052             # j3 rest at upper bound of its range
        # Tile across the 3 fingers
        return [j1, j2, j3, j1, j2, j3, j1, j2, j3]

    def _set_gripper(self, pos, hold_seconds=GRIPPER_HOLD_TIME,
                     transition_secs=SMOOTH_GRIPPER_SECS):
        """Choreograph a full gripper motion.

        1. Synchronous per-finger timing: all three fingers start
           closing at t=0 and finish at t=transition_secs.  The OMPL
           plan and smoothstep interpolation give the motion a curve.

        2. Curl profile: within each finger the distal joint (j3) curls
           more than the middle (j2) and proximal (j1) so the fingertips
           wrap around the object surface.

        3. Palm spread synchronised with gripper state: PALM_SPREAD_OPEN
           when opening (pos < 0), PALM_SPREAD_CLOSE when closing.
        """
        gids = self.sim.gripper_ids_left
        # Need at least 11 indices: 9 finger joints + 2 palm joints.
        if len(gids) < 11:
            # Defensive fallback — drive only the proximal joints.
            with self.sim._target_lock:
                for fi in FINGER_BASE_INDICES:
                    if fi < len(gids):
                        self.sim.data.ctrl[gids[fi]] = pos
            time.sleep(hold_seconds)
            return

        finger_targets = self._curl_targets(pos)         # 9 values
        palm_target = PALM_SPREAD_CLOSE if pos >= 0.0 else PALM_SPREAD_OPEN
        # Full 11-DOF target (9 finger joints + 2 palm spread joints)
        target = list(finger_targets) + [palm_target, palm_target]

        # Snapshot the starting ctrl values for all 11 DOFs.
        current = [float(self.sim.data.ctrl[gids[i]]) for i in range(11)]

        # ── OMPL motion plan for the finger trajectory ─────────────────
        # Generates the close/open trajectory via OMPL RRTConnect in an
        # 11-DOF state space (3 fingers x 3 joints + 2 palm spread).
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
            # Fallback: a degenerate 2-state path (current, target).
            # Executed with the same smoothstep timing as a planned path.
            path = [list(current), list(target)]
            print("[OMPL] Finger plan unavailable — using direct interpolation")
        else:
            print(f"[OMPL] Finger plan: {len(path)} waypoints from RRTConnect")

        # ── Execute the planned path with synchronous per-finger timing ───
        # All joint_offsets = 0 so the three fingers move together;
        # the OMPL plan + smoothstep within each joint provides the
        # motion curve.
        joint_offsets = [0.0] * 11
        max_offset = max(joint_offsets)
        total_secs = transition_secs + max_offset
        n_steps = max(1, int(total_secs / SMOOTH_GRIPPER_STEP_S))

        n_segments = max(1, len(path) - 1)

        def _wp_value(joint_i, alpha):
            """Sample the OMPL path at fractional position alpha ∈ [0,1]."""
            alpha = max(0.0, min(1.0, alpha))
            scaled = alpha * n_segments
            wp_idx = min(int(scaled), n_segments - 1)
            local_a = scaled - wp_idx
            s = local_a * local_a * (3.0 - 2.0 * local_a)   # smoothstep
            return path[wp_idx][joint_i] + s * (
                path[wp_idx + 1][joint_i] - path[wp_idx][joint_i])

        # Contact-stop close: per-finger contact monitoring during the
        # close path.  When a finger's bodies touch the held object,
        # freeze that finger's three joint ctrls at their current values
        # for the rest of this _set_gripper call.  Other fingers continue
        # advancing toward their targets.
        contact_stop_enabled = (
            USE_CONTACT_STOP_CLOSE
            and pos >= 0.0                              # closing only
            and self._held_obj_bid is not None          # have an object
            and self._finger_body_groups is not None
            and all(len(g) >= 1 for g in self._finger_body_groups)
        )
        finger_frozen = [False, False, False]   # [c, b, a]
        # joint_i (0-8) → finger group index (0=c, 1=b, 2=a):
        joint_to_finger = {
            0: 0, 1: 0, 2: 0,    # finger_c
            3: 1, 4: 1, 5: 1,    # finger_b
            6: 2, 7: 2, 8: 2,    # finger_a
        }

        for k in range(1, n_steps + 1):
            if self._cancel:
                return
            t = k * SMOOTH_GRIPPER_STEP_S

            # Per-step contact check — flip finger_frozen[fi] = True the
            # first step this finger touches the held object.
            if contact_stop_enabled and not all(finger_frozen):
                for fi in range(3):
                    if finger_frozen[fi]:
                        continue
                    if self._finger_touches_obj(fi, self._held_obj_bid):
                        finger_frozen[fi] = True
                        fname = ['c', 'b', 'a'][fi]
                        print(f"[Exec] finger_{fname} contact at "
                              f"t={t:.2f}s — freezing ctrl")

            with self.sim._target_lock:
                for joint_i in range(11):
                    if joint_i < 9:
                        fi = joint_to_finger[joint_i]
                        if finger_frozen[fi]:
                            # Don't advance this finger's ctrl any
                            # further — its tip is on the surface.
                            continue
                    local_t = max(0.0, t - joint_offsets[joint_i])
                    alpha = min(1.0, local_t / transition_secs)
                    self.sim.data.ctrl[gids[joint_i]] = _wp_value(
                        joint_i, alpha)

            # Early exit if all three fingers are in contact — no
            # benefit to keeping the loop running.
            if contact_stop_enabled and all(finger_frozen):
                # Fall through to the hold below.
                break

            time.sleep(SMOOTH_GRIPPER_STEP_S)
        time.sleep(hold_seconds)

    # ── Reading current arm config ──────────────────────────────────────

    def _current_arm_q(self):
        m = self._qmap
        d = self.sim.data
        return [float(d.qpos[m["ColumnLeft"]]),
                float(d.qpos[m["ColumnRight"]]),
                float(d.qpos[m["ArmLeft"]]),
                float(d.qpos[m["Base"]])]

    # ── Public: Pick ─────────────────────────────────────────────────────

    def pick(self, obj_idx, obj_world, on_complete=None,
             pre_grasp_q=None, pre_grasp_actual_target=None):
        """Pick ARM1 to the object at obj_world.  Runs in a background
        thread.  Caller: play_m1.py after base nav has positioned the
        robot at a virtually-screened pick standoff."""
        self._cancel = False
        t = threading.Thread(
            target=self._pick_run,
            args=(int(obj_idx), np.array(obj_world, dtype=float), on_complete,
                  None if pre_grasp_q is None else list(pre_grasp_q),
                  None if pre_grasp_actual_target is None else
                  np.array(pre_grasp_actual_target, dtype=float)),
            daemon=True)
        t.start()
        self._pick_thread = t

    def _pick_run(self, obj_idx, obj_world, on_complete,
                  screened_pre_grasp_q=None,
                  screened_actual_pre_target=None):
        # Single-shot callback dispatcher: guarantees on_complete is
        # invoked at most once per pick attempt.
        cb_fired = [False]
        def fire(ok):
            if cb_fired[0]:
                return
            cb_fired[0] = True
            if on_complete:
                on_complete(ok)

        success = False
        try:
            print(f"\n[Exec] PICK obj_{obj_idx} @ {obj_world.round(3)}")

            obj_bid, obj_qpa, obj_dofa = self._resolve_obj(obj_idx)
            self._held_obj_idx    = obj_idx
            self._held_obj_bid    = obj_bid
            self._held_obj_qpa    = obj_qpa
            self._held_obj_dofadr = obj_dofa

            # Snapshot the object's current world pose — pin keeps it here
            # for the entire approach + descent so it cannot drift.
            obj_pos_snapshot = self.sim.data.xpos[obj_bid].copy()

            # ── Pin object at its world position for the whole approach
            pin_world = self._pin_obj_at_world(obj_pos_snapshot)
            self._install_pin(pin_world)

            # ── Compute IK targets ──
            # XY offset along approach + obj_z + ABOVE_OBJ_HEIGHT, then
            # clamp wrist z to the calibrated minimum reachable height
            # so we don't ask IK for a target the arm cannot reach.
            robot_xy = self.sim.localization()[:2]
            obj_radius_for_standoff = self._object_radius(obj_bid)
            _grasp_unused, pre_grasp_target = compute_grasp_targets(
                robot_xy, obj_pos_snapshot,
                obj_radius=obj_radius_for_standoff)
            pre_grasp_target[2] = max(pre_grasp_target[2], MIN_PICK_WRIST_Z)

            loc = self.sim.localization()
            reset_plan_data_for_ik(self.arm_bridge,
                                   base_xy=(loc[0], loc[1]),
                                   base_yaw=loc[2])
            print(f"[Exec] pre_grasp_target = {pre_grasp_target.round(3)}  "
                  f"robot=({loc[0]:.3f},{loc[1]:.3f}) yaw={math.degrees(loc[2]):.1f}deg")

            # Strict z-lift IK returns (q, actual_target).  If requested
            # z is unreachable, the helper lifts z until reachable and
            # reports the actual reached target.
            if screened_pre_grasp_q is not None:
                PRE_GRASP_Q = [float(v) for v in screened_pre_grasp_q]
                actual_pre_target = (screened_actual_pre_target
                                     if screened_actual_pre_target is not None
                                     else pre_grasp_target.copy())
                print("[Exec] using screened PRE_GRASP_Q from candidate filter")
            else:
                try:
                    PRE_GRASP_Q, actual_pre_target = \
                        self.arm_bridge.solve_ik_with_z_lift(pre_grasp_target)
                except RuntimeError as e:
                    print(f"[Exec] PRE_GRASP_Q IK FAIL: {e}")
                    self._clear_held_state()
                    fire(False)
                    return

            # GRASP_Q == PRE_GRASP_Q: the arm cannot physically reach
            # obj_z for floor objects, so fingers close at the achievable
            # wrist height and the pin closure carries the object
            # thereafter at the recorded grasp offset.
            GRASP_Q = list(PRE_GRASP_Q)
            print(f"[Exec] PRE_GRASP_Q (actual_target z={actual_pre_target[2]:.3f}m) "
                  f"= {[round(x,3) for x in PRE_GRASP_Q]}")
            h_diff = abs(float(PRE_GRASP_Q[1]) - float(PRE_GRASP_Q[0]))
            if h_diff > MAX_PICK_H_DIFF or float(PRE_GRASP_Q[2]) < MIN_PICK_A1:
                print(f"[Exec] PRE_GRASP_Q rejected for visual safety: "
                      f"h2-h1={h_diff:.3f}m (max {MAX_PICK_H_DIFF:.2f}), "
                      f"a1={float(PRE_GRASP_Q[2]):.3f}m (min {MIN_PICK_A1:.2f})")
                self._clear_held_state()
                fire(False)
                return

            # ── [3] Open gripper in parallel with the approach ──
            # The open animates in a background thread while the OMPL
            # approach plan executes; open completes well before the
            # approach finishes.
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

            # ── [4] OMPL approach current → PRE_GRASP_Q ──
            print("[Exec] [4] approach: current → PRE_GRASP_Q")
            q_now = self._current_arm_q()
            path1 = self.arm_bridge.plan(q_now, PRE_GRASP_Q, timeout=8.0)
            if path1 is None:
                # OMPL rejected the start state (most commonly the
                # h1==h2 singular PARK_Q on the very first pick, where
                # arm_planner.is_valid requires alpha >= ALPHA_MIN_DEG).
                # Nudge h1/h2 apart into a nearby valid pose and replan.
                print("[Exec] OMPL approach start invalid/unavailable — "
                      "unlocking PARK_Q then replanning")
                avg_h = 0.5 * (float(q_now[0]) + float(q_now[1]))
                unlock_q = [
                    max(0.05, avg_h - 0.025),
                    min(1.35, avg_h + 0.025),
                    float(q_now[2]),
                    float(q_now[3]),
                ]
                self._kinematic_descent(q_now, unlock_q,
                                        "unlock-start",
                                        n_steps=10)
                if self._cancel:
                    self._clear_held_state(); fire(False); return
                path1 = self.arm_bridge.plan(unlock_q, PRE_GRASP_Q, timeout=8.0)
                if path1 is None:
                    # Last-resort fallback: direct kinematic approach so
                    # the log is explicit that this was not OMPL-planned.
                    print("[Exec] OMPL replan from unlock pose failed — "
                          "using direct kinematic approach fallback")
                    self._kinematic_descent(unlock_q, PRE_GRASP_Q,
                                            "direct-approach",
                                            n_steps=DESCENT_STEPS * 2)
                else:
                    self._execute_path(path1, "approach")
            else:
                self._execute_path(path1, "approach")
            if self._cancel:
                self._clear_held_state(); fire(False); return

            # ── [5] Linear descent PRE_GRASP_Q → GRASP_Q ──
            print("[Exec] [5] descent: PRE_GRASP_Q → GRASP_Q")
            self._kinematic_descent(PRE_GRASP_Q, GRASP_Q, "descent")
            if self._cancel:
                self._clear_held_state(); fire(False); return

            # ── [5.5] Pre-close acceptance gate ──
            # Read directly from sim.data — do NOT call mj_forward from
            # this background thread (concurrent ops on one MjData
            # cause SIGSEGV).  Deviation is measured against the IK
            # target (which sits GRIPPER_STANDOFF_XY behind the object),
            # not the object itself.
            grip_xyz_pre = self.sim.data.xpos[self.gripper_body_id].copy()
            obj_xyz_pre  = self.sim.data.xpos[obj_bid].copy()
            obj_grip_xy  = float(np.linalg.norm(grip_xyz_pre[:2] - obj_xyz_pre[:2]))
            expected_obj_grip_xy = GRIPPER_STANDOFF_XY
            extra_obj_grip_xy = max(0.0, obj_grip_xy - expected_obj_grip_xy)
            ik_dev = float(np.linalg.norm(grip_xyz_pre[:2] - pre_grasp_target[:2]))
            # Object-XY tolerance is relative to the expected standoff.
            obj_xy_gate = expected_obj_grip_xy + GRIP_OBJ_XY_TOLERANCE
            print(f"[Exec] [5.5] pre-close gate: obj-grip xy={obj_grip_xy:.3f}m  "
                  f"(expected≈{expected_obj_grip_xy:.3f}m, "
                  f"extra≈{extra_obj_grip_xy:.3f}m)  "
                  f"deviation_from_ik_target={ik_dev:.3f}m  "
                  f"(ik_tol={GRIP_DEVIATION_TOLERANCE:.2f}m, "
                  f"obj_xy_gate={obj_xy_gate:.2f}m)")
            if (ik_dev > GRIP_DEVIATION_TOLERANCE
                    or obj_grip_xy > obj_xy_gate):
                print(f"[Exec] pre-close gate rejected: "
                      f"ik_dev={ik_dev:.3f}m, obj-grip xy={obj_grip_xy:.3f}m — "
                      f"abandoning visually bad candidate, "
                      f"play_m1 will retry next base pose")
                current_q = self._current_arm_q()
                retry_q = [CARRY_H1, CARRY_H2, CARRY_A1, current_q[3]]
                print("[Exec] [5.6] retract arm before candidate retry")
                self._kinematic_descent(current_q, retry_q, "retry-retract",
                                        n_steps=DESCENT_STEPS)
                self._clear_held_state()
                fire(False)
                return

            # ── [6] Close gripper + activate weld + record grasp_offset ──
            # Size-aware close: smaller objects close more, larger
            # objects stop earlier so the fingers wrap around the
            # cylinder surface.  Contacts are kept at default (hard)
            # stiffness during the close so MuJoCo stops the fingertips
            # at the object surface; softening is applied afterwards
            # so the carry phase does not fight the pin solver.
            obj_radius = self._object_radius(obj_bid)
            close_pos  = self._finger_close_for_radius(obj_radius)
            # Make sure the async open from step [3] has completed
            # before we start the close (they share the finger
            # actuators).  The join is cheap insurance.
            if open_thread is not None and open_thread.is_alive():
                open_thread.join(timeout=2.0)
            print(f"[Exec] [6] close gripper  "
                  f"radius={obj_radius:.3f}m  close_ctrl={close_pos:.3f}")
            self._set_gripper(close_pos, hold_seconds=1.2)
            if self._cancel:
                self._clear_held_state(); fire(False); return

            # NOTE: do NOT call mj_forward(sim.data) from this background
            # thread — the main render thread runs mj_step concurrently
            # on the same MjData, and MuJoCo is not thread-safe for
            # concurrent ops on one MjData (SIGSEGV).  sim.data.xpos is
            # already up-to-date from the main thread's last mj_step.
            grip_xyz = self._carry_anchor_xyz(self.sim.data).copy()
            obj_xyz  = self.sim.data.xpos[obj_bid].copy()
            raw_offset = obj_xyz - grip_xyz
            raw_dist = float(np.linalg.norm(raw_offset))
            raw_xy_dist = float(np.linalg.norm(raw_offset[:2]))
            # Keep raw XY exactly (that's where the fingertips closed
            # on the cylinder, captured against the fingertip-pocket
            # centroid).  Adjust only Z to settle the object upward
            # into the finger pocket if it closed below the pocket.
            half_h = self._object_half_height(obj_bid)
            target_z = -(half_h + 0.025)  # 2.5cm overlap into pocket
            raw_z = float(raw_offset[2])
            z_lift = max(0.0, target_z - raw_z)
            z_lift = min(z_lift, 0.12)    # safety cap on the upward settle
            raw_offset[2] = raw_z + z_lift
            self._grasp_offset_xyz = raw_offset
            print(f"[Exec] grasp_offset = {raw_offset.round(3)}  "
                  f"obj-pocket dist={float(np.linalg.norm(raw_offset)):.3f}m  "
                  f"xy={raw_xy_dist:.3f}m  z_lift={z_lift*100:.1f}cm  "
                  f"(target_z={target_z:.3f}, raw_z={raw_z:.3f})")

            # Activate weld in plan_data only so OMPL transport sees
            # the carried object as part of robot geometry.
            self.arm_bridge.model.eq_obj2id[self.weld_id] = obj_bid
            self.arm_bridge.planning_data.eq_active[self.weld_id] = 1

            # Gravity compensation on the held object: MuJoCo cancels
            # gravity for this body so the pin closure does not fight a
            # falling object every frame.  Saved value is restored on
            # release in _clear_held_state.
            try:
                self._held_obj_orig_gravcomp = float(
                    self.sim.model.body_gravcomp[obj_bid])
                self.sim.model.body_gravcomp[obj_bid] = 1.0
                print(f"[Exec] gravcomp[{obj_bid}] {self._held_obj_orig_gravcomp:.2f} → 1.00 "
                      f"(zero-g while held)")
            except Exception as e:
                print(f"[Exec] gravcomp set warning: {e}")
                self._held_obj_orig_gravcomp = None

            # Soften held-object contacts before installing the
            # smooth-attach pin so the upward Z settle into the
            # fingertip pocket during interpolation produces only
            # gentle force against the closed fingers.
            self._soften_held_obj_contacts(obj_bid)

            # Switch pin closure: object follows gripper at grasp_offset.
            # Use the animated variant so the object smoothly slides
            # into the carry pose over SMOOTH_ATTACH_SECS.
            obj_xyz_now = self.sim.data.xpos[obj_bid].copy()
            self._install_pin(
                self._pin_obj_to_gripper_animated(obj_xyz_now))

            # Hold the fingers closed around the object during carry
            # using the same size-aware close pose as step [6].
            # hold_seconds covers the smooth-attach animation so it
            # completes before the lift starts.
            self._set_gripper(close_pos, hold_seconds=SMOOTH_ATTACH_SETTLE)

            print(f"[Exec] [verify-grasp] OK: object pinned at offset "
                  f"{self._grasp_offset_xyz.round(3)} from gripper, "
                  f"smooth-attach over {SMOOTH_ATTACH_SECS:.2f}s")

            # ── [6.5] LIFT — raise arm to carry pose for safe transport ──
            # Without this the gripper would stay at pick height during
            # base nav and drag the held object across the floor.
            print("[Exec] [6.5] lift to carry pose")
            current_q = self._current_arm_q()
            carry_q = [CARRY_H1, CARRY_H2, CARRY_A1, current_q[3]]
            self._kinematic_descent(current_q, carry_q, "lift",
                                    n_steps=DESCENT_STEPS)
            if self._cancel:
                self._clear_held_state(); fire(False); return

            success = True
        except Exception as e:
            import traceback
            print(f"[Exec] PICK exception: {e}")
            traceback.print_exc()
            self._clear_held_state()
        finally:
            fire(success)

    # ── Public: Place ────────────────────────────────────────────────────

    def place(self, shelf_idx, shelf_pos, on_complete=None):
        self._cancel = False
        t = threading.Thread(
            target=self._place_run,
            args=(int(shelf_idx), np.array(shelf_pos, dtype=float), on_complete),
            daemon=True)
        t.start()
        self._place_thread = t

    def _place_run(self, shelf_idx, shelf_pos, on_complete):
        # Single-shot callback dispatcher.
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

            # IK targets the object's desired position; the wrist is
            # offset by -grasp_offset from that.
            above_obj_target  = shelf_pos.copy()
            above_obj_target[2] += SHELF_ABOVE_HEIGHT
            place_obj_target  = shelf_pos.copy()
            place_obj_target[2] += SHELF_PLACE_OFFSET_Z
            # wrist_target = obj_target - grasp_offset
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
                # Clear held state so the next attempt starts clean
                # (otherwise the object remains pinned, the weld stays
                # active in plan_data, and gravcomp stays at 1.0).
                self._clear_held_state()
                fire(False)
                return

            print(f"[Exec] ABOVE_Q = {[round(x,3) for x in ABOVE_Q]}")
            print(f"[Exec] PLACE_Q = {[round(x,3) for x in PLACE_Q]}")

            # ── [7] OMPL transport current → ABOVE_Q ──
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

            # ── [8] Linear descent ABOVE_Q → PLACE_Q ──
            print("[Exec] [8] place descent: ABOVE_Q → PLACE_Q")
            self._kinematic_descent(ABOVE_Q, PLACE_Q, "place-descent")
            if self._cancel:
                self._clear_held_state(); fire(False); return

            # Capture the drop point — read directly, do NOT call
            # mj_forward from this background thread (concurrent ops
            # on the shared MjData cause SIGSEGV).
            drop_xyz = self.sim.data.xpos[obj_bid].copy()
            xy_err = float(np.linalg.norm(drop_xyz[:2] - shelf_pos[:2]))
            z_ok = drop_xyz[2] >= shelf_pos[2] - 0.05
            place_ok = (xy_err <= PLACE_XY_TOLERANCE) and z_ok
            print(f"[Exec] [verify-place] xy_err={xy_err:.3f}m  z_ok={z_ok}  "
                  f"place_ok={place_ok}  drop={drop_xyz.round(3)}")

            # ── [9] Open gripper + deactivate weld + pin obj at drop ──
            print("[Exec] [9] release: open gripper, weld off")
            # Pin the object at the drop point so opening fingers doesn't
            # send it flying laterally.
            self._install_pin(self._pin_obj_at_world(drop_xyz))
            # Weld is only in plan_data (sim_data was never activated).
            self.arm_bridge.planning_data.eq_active[self.weld_id] = 0
            self._set_gripper(GRIPPER_OPEN_POS, hold_seconds=0.8)

            # ── [10] OMPL retract PLACE_Q → HOME_Q ──
            print("[Exec] [10] retract: PLACE_Q → HOME_Q")
            q_now = self._current_arm_q()
            path_r = self.arm_bridge.plan(q_now, list(HOME_Q), timeout=8.0)
            if path_r is not None:
                self._execute_path(path_r, "retract")
            else:
                # Fallback: direct command
                self._set_arm_cmd(HOME_Q)
                time.sleep(1.5)

            # Clean up: pin removed, weld off, held state cleared.
            self._clear_held_state(deactivate_weld=True)

            # Report False if the verify-place check above found
            # xy_err out of tolerance or the object below the shelf
            # surface.  Release/retract still completed cleanly, but
            # the orchestrator must not claim "place complete" on a
            # drop that missed.
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

    # ── Public: Approximate drop ──────────────────────────────────────────

    def drop(self, on_complete=None, target_xy=None):
        """Approximate place: park the held object on the floor near
        the requested slot target and open the gripper.  Used in place
        of the full place() pipeline when only an approximate delivery
        is needed (avoids dropping from carry height into shelf/robot
        geometry, which can destabilise MuJoCo)."""
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

            # Place the object gently on the floor at its current XY
            # (or at target_xy when supplied) and open the gripper.
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
            # Explicitly zero the object's freejoint qpos+qvel before
            # installing the floor pin so the solver never sees an
            # implicit velocity inconsistent with the new pinned pose
            # during the swap from carry pin to floor pin.
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

            # Deactivate weld in plan_data (only place it was ever active).
            self.arm_bridge.planning_data.eq_active[self.weld_id] = 0

            # Open gripper.
            self._set_gripper(GRIPPER_OPEN_POS, hold_seconds=0.8)

            # Clear held state (pin removed, gravcomp restored, indices
            # reset).  Snapshot dofadr beforehand so we can zero qvel
            # once more after pin removal as a defensive guard.
            qpa_snap = self._held_obj_qpa
            dofadr_snap = self._held_obj_dofadr
            self._clear_held_state(deactivate_weld=True)
            if dofadr_snap is not None:
                self.sim.data.qvel[dofadr_snap:dofadr_snap + 6] = 0.0
            time.sleep(0.8)

            # Leave the arm at carry pose: non-singular and OMPL-valid,
            # ready for the next pick's OMPL approach plan.
            print("[Exec] [drop] complete — arm parked at carry pose, "
                  "ready for next pick")

            success = True
        except Exception as e:
            import traceback
            print(f"[Exec] DROP exception: {e}")
            traceback.print_exc()
        finally:
            fire(success)

    # ── Helpers ──────────────────────────────────────────────────────────

    def _park_other_objects(self, except_idx, park_pos=(0.0, 0.0, 50.0)):
        """Park all pickup_obj_N except `except_idx` in plan_data only."""
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
