"""
grasp_controller.py  —  Milestone 2: Grasping  (v4)
====================================================

WHY SINGLE-ARM GRASP
---------------------
v1-v3 moved BOTH arms simultaneously toward the object.
Both arms are offset ±0.15 m in Y from the robot centre.
Their fingertips end up on opposite sides of the object, so the
teleport-weld midpoint floats in empty air between the two arms.

Fix: select ONE arm (whichever has the smaller rotation angle to the
object), move only that arm, keep the other folded at home.
The object then snaps to the ACTIVE arm's fingertip — visually correct.

ARM SELECTION
-------------
_select_arm(obj_world) computes |theta| for each arm and picks the one
that needs less rotation (more natural reach direction).
The inactive arm is held at HOME position (h=0.30, a1=0.05, theta=0).

WRIST FLIP FIX (carried over from v3)
--------------------------------------
HandBearing actuator (gripper_ids[14]) is driven to HAND_DOWN during
approach and HAND_CARRY during lift to prevent gravity-induced wrist spin.

OBJECT PLACEMENT (carried over + improved from v3)
---------------------------------------------------
Uses palm body (finger attachment point) of the ACTIVE arm only.
GRASP_Z_OFFSET moves the object slightly into the finger curl.

CONSOLE OUTPUT
--------------
[Grasp] v4 ready — ee1=X ee2=X  palm1=X palm2=X  arm_geoms=N
[Grasp] Selected arm: LEFT  θ=42.3°
[Grasp] Step 1..6 as before
"""

import threading
import time
import math
import numpy as np
import mujoco

# ── Constants ─────────────────────────────────────────────────────────────────
COLUMN_LOW      = 0.00   # m — minimum column height (arms at lowest)
COLUMN_HIGH     = 0.55   # m — lift height
COLUMN_HOME     = 0.30   # m — inactive arm resting height
ARM_EXTEND      = 0.50   # m — horizontal extension toward object
ARM_RETRACT     = 0.25   # m — pull in when lifting
ARM_HOME        = 0.05   # m — inactive arm extension (folded)

GRIPPER_OPEN    = -1.0
GRIPPER_CLOSE   = 0.80

# HandBearing joint angles (gripper_ids index 14)
HAND_NEUTRAL    = 0.00   # home / folded
HAND_DOWN       = 1.30   # fingers pointing toward floor during approach
HAND_CARRY      = 0.70   # fingers cradling object during lift

# Object placement
GRASP_Z_OFFSET  = -0.05  # m below palm/fingertip body — puts object in finger curl
FLOOR_CLAMP     = 0.03   # m — never teleport object below this z

# Timing
SETTLE_RAISE    = 0.5    # s
SETTLE_ORIENT   = 2.0    # s
SETTLE_LOWER    = 1.5    # s
SETTLE_GRIPPER  = 0.8    # s
SETTLE_LIFT     = 2.0    # s

ATTACH_RADIUS   = 0.55   # m — proximity success threshold
MAX_RETRIES     = 3

# Arm mount offsets from robot centre (XML: Arm_1 pos="0.15 0.15", Arm_2 pos="0.15 -0.15")
ARM_OFFSETS = {
    'left':  (0.15,  0.15),
    'right': (0.15, -0.15),
}


# ── RL Policy ─────────────────────────────────────────────────────────────────
class GraspPolicy:
    """Contextual bandit — learns theta offset to improve grasp centering."""
    def __init__(self, lr=0.06, explore_std=0.06):
        self.lr          = lr
        self.explore_std = explore_std
        self.weight      = 0.0
        self._noise      = 0.0
        self.n_updates   = 0

    def get_offset(self):
        self._noise = float(np.random.normal(0.0, self.explore_std))
        return self.weight + self._noise

    def update(self, reward):
        self.weight    += self.lr * reward * self._noise
        self.weight     = float(np.clip(self.weight, -0.4, 0.4))
        self.n_updates += 1
        print(f"[RL] Update #{self.n_updates}  reward={reward:+.1f}  "
              f"theta_offset={self.weight:.3f} rad")


# ── Grasp Controller ──────────────────────────────────────────────────────────
class GraspController:

    def __init__(self, sim):
        self.sim    = sim
        self.policy = GraspPolicy()

        self._cancel      = False
        self._thread      = None
        self._active_arm  = None   # 'left' or 'right' — set at grasp time
        self._carry_wrist = HAND_CARRY

        # Teleport-weld state
        self._held_obj_idx  = None
        self._held_qpos_adr = None

        # ── Body IDs ──────────────────────────────────────────────────────
        # Wrist (Gripper_Link1) — used for proximity check
        self._ee   = {
            'left':  self._bid("Gripper_Link1_1"),
            'right': self._bid("Gripper_Link1_2"),
        }

        # Palm bodies — used for object placement (closer to fingers than wrist)
        # Try Gripper_Link3 first (deepest wrist body before fingers), then Link1
        self._palm = {
            'left':  (self._bid("Gripper_Link3_1") or
                      self._bid("Gripper_Link2_1") or
                      self._ee['left']),
            'right': (self._bid("Gripper_Link3_2") or
                      self._bid("Gripper_Link2_2") or
                      self._ee['right']),
        }

        # Fix fallback to sim.end_effector_id if not found
        for k in ('left', 'right'):
            if self._ee[k] is None:
                self._ee[k] = sim.end_effector_id
            if self._palm[k] is None:
                self._palm[k] = sim.end_effector_id

        # Gripper geom IDs for contact check
        self._gripper_geoms = self._collect_gripper_geoms()

        print(f"[Grasp] v4 ready — "
              f"ee=({self._ee['left']},{self._ee['right']})  "
              f"palm=({self._palm['left']},{self._palm['right']})  "
              f"gripper_geoms={len(self._gripper_geoms)}")

    # ── ID helpers ────────────────────────────────────────────────────────────
    def _bid(self, name):
        v = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_BODY, name)
        return v if v >= 0 else None

    def _collect_gripper_geoms(self):
        kw = {"gripper", "finger", "palm", "hand", "wrist",
              "link_0", "link_1", "link_2", "link_3"}
        geoms = set()
        for g in range(self.sim.model.ngeom):
            name = (mujoco.mj_id2name(self.sim.model, mujoco.mjtObj.mjOBJ_BODY,
                                      self.sim.model.geom_bodyid[g]) or "").lower()
            if any(k in name for k in kw):
                geoms.add(g)
        return geoms

    # ── Arm selection ─────────────────────────────────────────────────────────
    def _compute_theta(self, obj_world, arm):
        """Arm-local BaseJoint angle to point toward obj_world."""
        rx, ry, ryaw = self.sim.localization()
        cy, sy = math.cos(ryaw), math.sin(ryaw)
        ox, oy = ARM_OFFSETS[arm]
        ax = rx + cy * ox - sy * oy
        ay = ry + sy * ox + cy * oy
        world_angle = math.atan2(obj_world[1] - ay, obj_world[0] - ax)
        local_angle = world_angle - ryaw
        return (local_angle + math.pi) % (2 * math.pi) - math.pi

    def _select_arm(self, obj_world):
        """Pick whichever arm needs less rotation to face the object."""
        tl = abs(self._compute_theta(obj_world, 'left'))
        tr = abs(self._compute_theta(obj_world, 'right'))
        chosen = 'left' if tl <= tr else 'right'
        print(f"[Grasp] Selected arm: {chosen.upper()}  "
              f"θ_left={math.degrees(tl):.1f}°  θ_right={math.degrees(tr):.1f}°")
        return chosen

    # ── Joint control ─────────────────────────────────────────────────────────
    def _ids(self, arm):
        """Return gripper actuator id list for the given arm."""
        return (self.sim.gripper_ids_left  if arm == 'left'
                else self.sim.gripper_ids_right)

    def _gripper(self, arm, finger_val, wrist_angle=None):
        """Set finger actuators and optionally HandBearing for one arm."""
        ids = self._ids(arm)
        for idx in [0, 3, 6]:
            if idx < len(ids):
                self.sim.data.ctrl[ids[idx]] = finger_val
        if wrist_angle is not None and len(ids) > 14:
            self.sim.data.ctrl[ids[14]] = wrist_angle

    def _wrist(self, arm, angle):
        ids = self._ids(arm)
        if len(ids) > 14:
            self.sim.data.ctrl[ids[14]] = angle

    def _set_arm_joints(self, arm, h1, h2, a1, theta):
        """Move ONE arm; hold the other at home position."""
        inactive = 'right' if arm == 'left' else 'left'
        # Active arm
        if arm == 'left':
            cmds = [h1, h2, a1, theta,          # arm1 (left)
                    COLUMN_HOME, COLUMN_HOME, ARM_HOME, 0.0]   # arm2 (right) home
        else:
            cmds = [COLUMN_HOME, COLUMN_HOME, ARM_HOME, 0.0,   # arm1 (left) home
                    h1, h2, a1, theta]           # arm2 (right)
        with self.sim._target_lock:
            self.sim.use_ik = False
            self.sim.direct_arm_commands = np.array(cmds, dtype=float)

    # ── Success checks ────────────────────────────────────────────────────────
    def _proximity_ok(self, arm, obj_world):
        ee_pos = self.sim.data.xpos[self._ee[arm]]
        return np.linalg.norm(ee_pos - np.array(obj_world)) < ATTACH_RADIUS

    def _contact_ok(self, obj_idx):
        bid = mujoco.mj_name2id(
            self.sim.model, mujoco.mjtObj.mjOBJ_BODY, f"pickup_obj_{obj_idx}")
        obj_geoms = {g for g in range(self.sim.model.ngeom)
                     if self.sim.model.geom_bodyid[g] == bid}
        for i in range(self.sim.data.ncon):
            c = self.sim.data.contact[i]
            if ((c.geom1 in self._gripper_geoms and c.geom2 in obj_geoms) or
                (c.geom2 in self._gripper_geoms and c.geom1 in obj_geoms)):
                return True
        return False

    # ── Teleport-weld ─────────────────────────────────────────────────────────
    def _qpos_adr(self, obj_idx):
        bid = mujoco.mj_name2id(
            self.sim.model, mujoco.mjtObj.mjOBJ_BODY, f"pickup_obj_{obj_idx}")
        if bid < 0:
            return None
        jnt = self.sim.model.body_jntadr[bid]
        return self.sim.model.jnt_qposadr[jnt] if jnt >= 0 else None

    def _attach(self, obj_idx):
        self._held_obj_idx  = obj_idx
        self._held_qpos_adr = self._qpos_adr(obj_idx)
        print(f"[Grasp] obj_{obj_idx} attached  "
              f"arm={self._active_arm}  qpos_adr={self._held_qpos_adr}")

    def _detach(self):
        self._held_obj_idx  = None
        self._held_qpos_adr = None

    def update_held_object(self):
        """
        Call EVERY simulation step from play_m1.py main loop.

        1. Keeps driving the ACTIVE arm's wrist to prevent gravity flip.
        2. Snaps the held object to the ACTIVE arm's palm position only
           (not the midpoint between both arms — that caused the floating bug).
        """
        if self._held_qpos_adr is None or self._active_arm is None:
            return

        # 1. Hold wrist angle — prevents flip every step
        self._wrist(self._active_arm, self._carry_wrist)

        # 2. Snap object to active arm's palm body (not wrist midpoint!)
        palm_pos = self.sim.data.xpos[self._palm[self._active_arm]].copy()
        palm_pos[2] += GRASP_Z_OFFSET
        palm_pos[2]  = max(palm_pos[2], FLOOR_CLAMP)

        adr = self._held_qpos_adr
        self.sim.data.qpos[adr + 0] = palm_pos[0]
        self.sim.data.qpos[adr + 1] = palm_pos[1]
        self.sim.data.qpos[adr + 2] = palm_pos[2]
        self.sim.data.qpos[adr + 3] = 1.0   # upright orientation
        self.sim.data.qpos[adr + 4] = 0.0
        self.sim.data.qpos[adr + 5] = 0.0
        self.sim.data.qpos[adr + 6] = 0.0

        # Kill velocity
        bid = mujoco.mj_name2id(
            self.sim.model, mujoco.mjtObj.mjOBJ_BODY,
            f"pickup_obj_{self._held_obj_idx}")
        jnt = self.sim.model.body_jntadr[bid]
        if jnt >= 0:
            dof = self.sim.model.jnt_dofadr[jnt]
            self.sim.data.qvel[dof:dof + 6] = 0.0

    # ── Public API ────────────────────────────────────────────────────────────
    def grasp(self, obj_idx, obj_world_pos, on_complete=None):
        self._cancel = True
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._cancel = False
        self._thread = threading.Thread(
            target=self._run,
            args=(int(obj_idx), np.array(obj_world_pos, dtype=float), on_complete),
            daemon=True)
        self._thread.start()

    def cancel(self):
        self._cancel = True
        if self._active_arm:
            self._wrist(self._active_arm, HAND_NEUTRAL)
        self._detach()

    def is_holding(self):
        return self._held_obj_idx is not None

    def get_held_idx(self):
        return self._held_obj_idx

    # ── State machine ─────────────────────────────────────────────────────────
    def _run(self, obj_idx, obj_world, on_complete):
        print(f"[Grasp] Starting grasp of obj_{obj_idx} at {obj_world}")

        # Select active arm once per grasp (consistent across retries)
        arm = self._select_arm(obj_world)
        self._active_arm = arm
        inactive = 'right' if arm == 'left' else 'left'

        for attempt in range(MAX_RETRIES):
            if self._cancel:
                break
            print(f"[Grasp] Attempt {attempt + 1}/{MAX_RETRIES}  arm={arm.upper()}")

            theta_offset = self.policy.get_offset()
            theta = self._compute_theta(obj_world, arm) + theta_offset

            # Step 1: Open active gripper, fold inactive arm, raise ────────
            print("[Grasp] Step 1 — open, raise")
            self._gripper(arm, GRIPPER_OPEN, wrist_angle=HAND_NEUTRAL)
            self._gripper(inactive, GRIPPER_OPEN, wrist_angle=HAND_NEUTRAL)
            self._set_arm_joints(arm, h1=0.20, h2=0.20, a1=0.10, theta=theta)
            time.sleep(SETTLE_RAISE)
            if self._cancel: break

            # Step 2: Rotate wrist DOWN + orient arm ───────────────────────
            print(f"[Grasp] Step 2 — orient  θ={math.degrees(theta):.1f}°")
            self._gripper(arm, GRIPPER_OPEN, wrist_angle=HAND_DOWN)
            self._set_arm_joints(arm, h1=0.10, h2=0.10,
                                 a1=ARM_EXTEND * 0.5, theta=theta)
            time.sleep(SETTLE_ORIENT)
            if self._cancel: break

            # Step 3: Lower columns + full extension ───────────────────────
            print("[Grasp] Step 3 — lower + extend")
            self._set_arm_joints(arm, h1=COLUMN_LOW, h2=COLUMN_LOW,
                                 a1=ARM_EXTEND, theta=theta)
            sub = 3
            for _ in range(sub):
                self._wrist(arm, HAND_DOWN)     # re-drive every sub-step
                time.sleep(SETTLE_LOWER / sub)
                if self._cancel: break
            if self._cancel: break

            # Step 4: Close gripper ────────────────────────────────────────
            print("[Grasp] Step 4 — close + lock wrist")
            self._gripper(arm, GRIPPER_CLOSE, wrist_angle=HAND_DOWN)
            time.sleep(SETTLE_GRIPPER)
            if self._cancel: break

            # Step 5: Check success ────────────────────────────────────────
            contact   = self._contact_ok(obj_idx)
            proximity = self._proximity_ok(arm, obj_world)
            success   = contact or proximity

            print(f"[Grasp] Step 5 — contact={contact}  proximity={proximity}  "
                  f"ncon={self.sim.data.ncon}  → {'SUCCESS' if success else 'FAILED'}")
            self.policy.update(1.0 if success else -1.0)

            if success:
                # Step 6a: Attach + retract slightly ──────────────────────
                self._attach(obj_idx)
                self._carry_wrist = HAND_CARRY
                print(f"[Grasp] Step 6a — retract a1={ARM_RETRACT}")
                self._set_arm_joints(arm, h1=COLUMN_LOW, h2=COLUMN_LOW,
                                     a1=ARM_RETRACT, theta=theta)
                time.sleep(0.5)

                # Step 6b: Lift ───────────────────────────────────────────
                print(f"[Grasp] Step 6b — lift h={COLUMN_HIGH:.2f}m")
                self._gripper(arm, GRIPPER_CLOSE, wrist_angle=HAND_CARRY)
                self._set_arm_joints(arm, h1=COLUMN_HIGH, h2=COLUMN_HIGH,
                                     a1=ARM_RETRACT, theta=theta)
                time.sleep(SETTLE_LIFT)
                print("[Grasp] ✓ Object lifted")
                if on_complete:
                    on_complete(True)
                return

            # Failed — reset for retry
            print("[Grasp] Failed — resetting")
            self._gripper(arm, GRIPPER_OPEN, wrist_angle=HAND_NEUTRAL)
            self._set_arm_joints(arm, h1=0.25, h2=0.25, a1=0.10, theta=0.0)
            time.sleep(SETTLE_ORIENT)

        # All retries exhausted
        print("[Grasp] All attempts failed")
        self._gripper(arm, GRIPPER_OPEN, wrist_angle=HAND_NEUTRAL)
        self._set_arm_joints(arm, h1=0.25, h2=0.25, a1=0.10, theta=0.0)
        self._active_arm = None
        if on_complete:
            on_complete(False)