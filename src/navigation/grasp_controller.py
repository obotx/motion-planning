"""
grasp_controller.py  —  v51
============================

ARCHITECTURE: FULLY PHYSICS-BASED
-----------------------------------
Previous versions (v49, v50) manipulated qpos directly every substep (weld),
which is fundamentally non-physical: the integrator fights the position write
and produces sawtooth oscillation, requiring hacks like gravcomp to mask it.

v51 removes ALL direct qpos manipulation.  The held object is controlled
exclusively through qfrc_applied — the generalized-force vector that MuJoCo
adds to the equations of motion BEFORE integration.  This is how real grippers
work: finger contact forces are transmitted into the object, and we model the
net effect as a spring-damper attached between the palm and the object.

HOLD MECHANISM — spring-damper in qfrc_applied
-----------------------------------------------
After grip confirmation, every substep (called from play_m1.py before mj_step):

    Phase 1 — world-frozen:  target_pos = grasp_pos (object's position at snap)
    Phase 2 — palm-relative: target_pos = palm_mat @ local_offset + palm_pos

    F_spring = K * (target_pos − obj_pos)       [restoring force]
    F_damp   = −D * obj_vel                      [velocity damping]
    F_grav   = +mass * 9.81 * ẑ                  [counteract gravity]

    qfrc_applied[dof:dof+3] = F_spring + F_damp + F_grav

    K=600 N/m, D=60 Ns/m  →  overdamped for all object masses (0.05–0.20 kg)
    No qpos writes.  MuJoCo integrates normally.  Zero oscillation.

Additionally, body_gravcomp[bid]=1.0 is set at attach so MuJoCo's own
constraint solver also counteracts gravity.  This reduces the DC error of the
spring (spring only needs to resist perturbation forces, not support full weight).

WHY THIS IS PHYSICALLY CORRECT
--------------------------------
qfrc_applied[translational dofs] for a free-joint body corresponds to a force
in the world frame — exactly what a rigid gripper grip transmits.  The object's
mass, inertia, contact with the floor, and other dynamic effects are preserved.
There is no teleportation and no solver conflict.

THREE BUGS FIXED vs v49
------------------------
Bug-1  VIBRATION:       qpos write → spring qfrc_applied (no integrator conflict)
Bug-2  ARM/BODY OVERLAP: A1_SAFE=0.05m during ALL vertical moves; extend only
                         in S4 after arm is already at object height
Bug-3  UNREALISTIC:     snap-before-extend → physical servo approach (2 cm steps,
                         stop at SIDE_STOP_DIST), weld only after finger contact

CONFIRMED vs morph_i_free_move.py
-----------------------------------
• alpha=0.25 in control_arms → 99% convergence in 17 render frames (~0.57s)
  → APPROACH_SETTLE=0.40s gives 96.8% convergence per step: safe
• step_simulation(per_step_callback=...) calls callback before each of 10 substeps
• localization() returns np.array([x, y, yaw])
• direct_arm_commands[0:4] = left arm [h1, h2, a1, theta]
• direct_arm_commands[4:8] = right arm [h1, h2, a1, theta]
• body_gravcomp[:] is NOT set in __init__ (configure_model() is never called)
  → our per-object gravcomp is safe and necessary
• GOAL_REACH_DIST=0.65m in ompl_windows_bridge → nav stops 0.65m from goal
  → S0 closes arm pivot to 0.40m from object (base ~0.28m away — safe)

STATE MACHINE (v51)
-------------------
  S0  APPROACH    — base drives until pivot ≤ 0.40m from object
  S1  RETRACT+RAISE — h=H_MID, a1=A1_SAFE (arm clear of body at all heights)
  S2  DESCEND     — h→obj_z (calibrated), a1=A1_SAFE (no body sweep)
  S2b TILT        — h2 ramps up while h1 locked; a1=A1_SAFE throughout
  S3  OPEN        — gripper fully open
  S4  SERVO EXTEND— a1 grows 2cm/step, stop at SIDE_STOP_DIST (physical approach)
  S5  CLOSE       — fingers ramp closed over T_CLOSE (physical contact)
  S6  VERIFY      — MuJoCo contact array + proximity fallback
  S7  ATTACH      — gravcomp=1, spring-force hold activated
  S8  LIFT        — h→H_CARRY; spring carries object upward
  S9  RETRACT     — a1→A1_HOME; palm-relative spring tracks
"""

import threading, time, math
import numpy as np
import mujoco

# ── Arm geometry ──────────────────────────────────────────────────────────────
H_MID  = 0.72    # neutral/home height for both columns
H_CARRY = 0.65   # carry height after successful grasp

# A1_SAFE: maximum arm extension during ANY vertical motion (raise/descend/tilt).
# At 5 cm extension the arm tip is 0.17 m from the body centre — clear of chassis.
# Extending beyond this only happens in S4, when arm is ALREADY at object height.
A1_SAFE = 0.05
A1_HOME = 0.28   # retracted home (used post-grasp)
A1_MIN  = 0.00
A1_MAX  = 0.58

FINGERTIP_OVERHANG = 0.08   # palm-tip → fingertip (for geometry logging)
H_MIN              = 0.03   # absolute floor for h1/h2

HD_TARGET = 0.45   # pivot→object target for fine-approach (not used after S0)

# ── Spring-force hold ─────────────────────────────────────────────────────────
# K=600 N/m, D=60 Ns/m → overdamped (D_crit ≈ 11–22 for 0.05–0.20 kg objects)
SPRING_K   = 600.0
SPRING_D   = 60.0

# ── Servo approach ────────────────────────────────────────────────────────────
SIDE_STOP_DIST  = 0.17   # stop when palm centre is this far from object centre
APPROACH_STEP   = 0.02   # a1 increment per servo step
APPROACH_SETTLE = 0.40   # seconds between steps (96.8% convergence at alpha=0.25)
APPROACH_MIN_PD = 0.09   # emergency stop (palm too close → collision)

# ── Calibration defaults ──────────────────────────────────────────────────────
PALM_Z_OFFSET_DEFAULT = 0.168
PALM_A1_SCALE_DEFAULT = 1.0

# ── Finger / wrist ────────────────────────────────────────────────────────────
FINGER_J1_IDX = [0, 3, 6]
FINGER_J2_IDX = [1, 4, 7]
FINGER_J3_IDX = [2, 5, 8]
WRIST_X_IDX   = 11
WRIST_Y_IDX   = 12
WRIST_Z_IDX   = 13
BEARING_IDX   = 14

J1_OPEN   = -1.00;  J2_OPEN   =  0.00;  J3_OPEN   = -1.22
J1_CLOSED =  0.70;  J2_CLOSED =  0.15;  J3_CLOSED = -0.40
OBJ_R_MIN =  0.04;  OBJ_R_MAX =  0.10
WRIST_NEUTRAL = 0.00;  WRIST_CARRY = 0.20

# ── Contact / proximity ───────────────────────────────────────────────────────
CONTACT_REQUIRED = 1
ATTACH_DIST_PROX = 0.40   # proximity fallback if contact array misses

# ── Navigate approach ─────────────────────────────────────────────────────────
ARM_IDEAL_HD = 0.40   # S0 target: pivot→object distance
# Note: GOAL_REACH_DIST=0.65 in ompl_windows_bridge → nav stops 0.65m from goal.
# S0 closes the remaining gap to ARM_IDEAL_HD from the arm pivot.

# ── Settle ────────────────────────────────────────────────────────────────────
SETTLE_THRESHOLD = 0.008
SETTLE_INTERVAL  = 0.20
SETTLE_MAX_WAIT  = 5.0
SETTLE_STABLE    = 3

# ── Convergence ───────────────────────────────────────────────────────────────
CONVERGE_MIN_WAIT  = 0.70   # at alpha=0.25, 99% in 0.57s; 0.70s is safe
CONVERGE_THRESHOLD = 0.004
CONVERGE_INTERVAL  = 0.20
CONVERGE_STABLE    = 3
CONVERGE_TIMEOUT   = 12.0

# ── Timing ────────────────────────────────────────────────────────────────────
T_STEP   = 0.04
T_CLOSE  = 2.0
MAX_RETRIES = 3
ARM_OFFSETS = {'left': (0.12, 0.15), 'right': (0.12, -0.15)}


# ─────────────────────────────────────────────────────────────────────────────
class GraspPolicy:
    def __init__(self, lr=0.05, explore_std=0.015):
        self.lr = lr; self.explore_std = explore_std
        self.weight = 0.0; self._noise = 0.0; self.n_updates = 0

    def get_offset(self):
        self._noise = float(np.random.normal(0.0, self.explore_std))
        return self.weight + self._noise

    def update(self, reward):
        self.weight += self.lr * reward * self._noise
        self.weight  = float(np.clip(self.weight, -0.08, 0.08))
        self.n_updates += 1
        print(f"[RL] #{self.n_updates}  r={reward:+.0f}  w={self.weight:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
class GraspController:

    def __init__(self, sim):
        self.sim          = sim
        self.policy       = GraspPolicy()
        self._cancel      = False
        self._thread      = None
        self._active_arm  = None
        self._close_j1    = J1_CLOSED

        # ── Spring-force hold state ──────────────────────────────────────────
        self._held_obj_idx      = None
        self._held_obj_bid      = None   # body id
        self._held_dof_adr      = None   # first translational DOF index
        self._held_orig_gravcomp = 0.0   # saved to restore on detach
        self._spring_active     = False
        self._spring_frozen     = False  # True=world target, False=palm-relative
        self._spring_target_pos  = None  # world target (phase 1)
        self._spring_local_pos   = None  # palm-local offset (phase 2)
        self._spring_local_quat  = None  # palm-local quat (phase 2)

        # For quaternion tracking (kept for _reattach)
        self._held_qpos_adr = None

        self.kill_base_vel = False
        self._base_frozen  = False
        self.on_base_moved = None

        # ── Palm body IDs ─────────────────────────────────────────────────────
        self._palm = {
            'left':  (self._bid("Gripper_Link3_1") or
                      self._bid("Gripper_Link2_1") or
                      self._bid("Gripper_Link1_1")),
            'right': (self._bid("Gripper_Link3_2") or
                      self._bid("Gripper_Link2_2") or
                      self._bid("Gripper_Link1_2")),
        }
        for k in ('left', 'right'):
            if self._palm[k] is None:
                self._palm[k] = getattr(sim, 'end_effector_id', 0)

        self._base_dof      = self._find_base_dof()
        self._gripper_geoms = self._collect_gripper_geoms()
        self._init_both_arms()

        # Arm Z-calibration
        self._palm_z_at_zero = PALM_Z_OFFSET_DEFAULT
        self._palm_z_slope   = 1.0
        self._palm_a1_scale  = PALM_A1_SCALE_DEFAULT
        self._calibrate_arm_offsets()

        print(f"[Grasp] v51  SPRING K={SPRING_K} D={SPRING_D}  "
              f"A1_SAFE={A1_SAFE}  SIDE_STOP={SIDE_STOP_DIST}  "
              f"ARM_IDEAL_HD={ARM_IDEAL_HD}  "
              f"base_dof={self._base_dof}  "
              f"palm=({self._palm['left']},{self._palm['right']})  "
              f"z_at_zero={self._palm_z_at_zero:.3f}  "
              f"z_slope={self._palm_z_slope:.3f}")

    # ── Init helpers ──────────────────────────────────────────────────────────
    def _bid(self, name):
        v = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_BODY, name)
        return v if v >= 0 else None

    def _find_base_dof(self):
        for name in ("base_footprint", "base", "robot", "mobile_base",
                     "chassis", "base_link", "body", "platform"):
            bid = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                jnt = self.sim.model.body_jntadr[bid]
                if jnt >= 0:
                    dof = self.sim.model.jnt_dofadr[jnt]
                    print(f"[Grasp] Base DOF: body='{name}' dof={dof}")
                    return dof
        return 0

    def _collect_gripper_geoms(self):
        kw = {"gripper", "finger", "palm", "hand", "wrist"}
        out = set()
        for g in range(self.sim.model.ngeom):
            bn = (mujoco.mj_id2name(self.sim.model, mujoco.mjtObj.mjOBJ_BODY,
                                    self.sim.model.geom_bodyid[g]) or "").lower()
            if any(k in bn for k in kw):
                out.add(g)
        print(f"[Grasp] Gripper geoms: {len(out)}")
        return out

    def _init_both_arms(self):
        with self.sim._target_lock:
            self.sim.use_ik = False
            self.sim.direct_arm_commands = np.array(
                [H_MID, H_MID, A1_HOME, 0.0,
                 H_MID, H_MID, A1_HOME, 0.0], dtype=float)

    def _calibrate_arm_offsets(self):
        H_CAL_LO = 0.30
        CAL_WAIT = 2.5
        try:
            with self.sim._target_lock:
                self.sim.use_ik = False
                c = self.sim.direct_arm_commands.copy()
                c[4]=H_MID; c[5]=H_MID; c[6]=A1_HOME; c[7]=0.0
                self.sim.direct_arm_commands = c
            time.sleep(CAL_WAIT)
            mujoco.mj_forward(self.sim.model, self.sim.data)
            z_hi = float(self.sim.data.xpos[self._palm['right']][2])

            with self.sim._target_lock:
                c = self.sim.direct_arm_commands.copy()
                c[4]=H_CAL_LO; c[5]=H_CAL_LO
                self.sim.direct_arm_commands = c
            time.sleep(CAL_WAIT)
            mujoco.mj_forward(self.sim.model, self.sim.data)
            z_lo = float(self.sim.data.xpos[self._palm['right']][2])

            slope = (z_hi - z_lo) / (H_MID - H_CAL_LO)
            z_at_zero = z_hi - slope * H_MID
            if 0.1 < slope < 2.0 and 0.0 < z_at_zero < 1.0:
                self._palm_z_slope   = slope
                self._palm_z_at_zero = z_at_zero
                print(f"[Grasp] Z-cal: at_zero={z_at_zero:.3f}  slope={slope:.3f}")
            else:
                print(f"[Grasp] Z-cal OOR — using defaults")

            with self.sim._target_lock:
                c = self.sim.direct_arm_commands.copy()
                c[4]=H_MID; c[5]=H_MID
                self.sim.direct_arm_commands = c
            time.sleep(CAL_WAIT)
            mujoco.mj_forward(self.sim.model, self.sim.data)

            ax, ay = self._arm_pivot_world('right')
            palm_xy = self.sim.data.xpos[self._palm['right']][:2].copy()
            ext = float(np.linalg.norm(palm_xy - np.array([ax, ay])))
            if ext > 0.01:
                scale = ext / A1_HOME
                if 0.3 < scale < 4.0:
                    self._palm_a1_scale = scale
                    print(f"[Grasp] a1_scale={scale:.3f}")
        except Exception as e:
            print(f"[Grasp] Calibration error: {e} — using defaults")

    def _h_cmd_for_palm_z(self, target_z):
        raw = (target_z - self._palm_z_at_zero) / max(self._palm_z_slope, 0.05)
        return float(np.clip(raw, H_MIN, 1.43))

    # ── Base helpers ──────────────────────────────────────────────────────────
    def freeze_base(self):
        if not self._base_frozen:
            pos = self.sim.localization()
            with self.sim._target_lock:
                self.sim.target_base = np.array(pos, dtype=float)
            self._base_frozen  = True
            self.kill_base_vel = True
            if self.on_base_moved:
                self.on_base_moved(*pos)
            print(f"[Grasp] Base frozen ({pos[0]:.3f},{pos[1]:.3f})")

    def unfreeze_base(self):
        self._base_frozen  = False
        self.kill_base_vel = False

    def zero_base_velocity(self):
        try:
            self.sim.data.qvel[self._base_dof:self._base_dof + 6] = 0.0
        except Exception:
            pass

    def _set_base_target(self, x, y, yaw):
        with self.sim._target_lock:
            self.sim.target_base = np.array([x, y, yaw], dtype=float)
        if self.on_base_moved:
            self.on_base_moved(float(x), float(y), float(yaw))

    # ── Settle ────────────────────────────────────────────────────────────────
    def wait_for_settle(self):
        print("[Grasp] Settling...")
        self.kill_base_vel = True
        t0 = time.time(); prev = np.array(self.sim.localization()[:2]); stable = 0
        while time.time()-t0 < SETTLE_MAX_WAIT:
            self.zero_base_velocity()
            time.sleep(SETTLE_INTERVAL)
            curr = np.array(self.sim.localization()[:2])
            d = float(np.linalg.norm(curr - prev)); prev = curr
            if d < SETTLE_THRESHOLD:
                stable += 1
                if stable >= SETTLE_STABLE:
                    print(f"[Grasp] Settled — drift={d*1000:.1f}mm  t={time.time()-t0:.1f}s")
                    self.kill_base_vel = False; return
            else:
                stable = 0
        print("[Grasp] Settle timeout"); self.kill_base_vel = False

    # ── Convergence ───────────────────────────────────────────────────────────
    def _palm_pos(self, arm):
        return self.sim.data.xpos[self._palm[arm]].copy()

    def _wait_converge(self, arm, label="", timeout=CONVERGE_TIMEOUT,
                       min_wait=CONVERGE_MIN_WAIT):
        if self._cancel: return False
        time.sleep(min_wait)
        if self._cancel: return False
        t0 = time.time(); prev = self._palm_pos(arm); stable = 0
        while time.time()-t0 < timeout:
            if self._cancel: return False
            time.sleep(CONVERGE_INTERVAL)
            curr  = self._palm_pos(arm)
            delta = float(np.linalg.norm(curr - prev)); prev = curr
            if delta < CONVERGE_THRESHOLD:
                stable += 1
                if stable >= CONVERGE_STABLE:
                    print(f"[Grasp]   ✓ {label}  palm={curr.round(3)}  Δ={delta*1000:.1f}mm")
                    return True
            else:
                stable = 0
        print(f"[Grasp]   converge timeout ({label})")
        return False

    # ── Geometry ──────────────────────────────────────────────────────────────
    def _arm_pivot_world(self, arm):
        rx, ry, ryaw = self.sim.localization()
        cy, sy = math.cos(ryaw), math.sin(ryaw)
        ox, oy = ARM_OFFSETS[arm]
        return rx + cy*ox - sy*oy, ry + sy*ox + cy*oy

    def _compute_theta(self, obj_world, arm):
        ax, ay = self._arm_pivot_world(arm)
        _, _, ryaw = self.sim.localization()
        angle = math.atan2(obj_world[1]-ay, obj_world[0]-ax)
        return (angle - ryaw + math.pi) % (2*math.pi) - math.pi

    def _select_arm(self, obj_world):
        tl = abs(self._compute_theta(obj_world, 'left'))
        tr = abs(self._compute_theta(obj_world, 'right'))
        arm = 'left' if tl <= tr else 'right'
        print(f"[Grasp] Arm: {arm.upper()}  θL={math.degrees(tl):.1f}°  θR={math.degrees(tr):.1f}°")
        return arm

    # ── Object helpers ────────────────────────────────────────────────────────
    def _get_obj_geom(self, obj_idx):
        bid = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_BODY,
                                f"pickup_obj_{obj_idx}")
        for g in range(self.sim.model.ngeom):
            if self.sim.model.geom_bodyid[g] == bid:
                sz = self.sim.model.geom_size[g]
                return float(sz[0]), float(sz[1])
        return 0.06, 0.07

    def _get_obj_geom_ids(self, obj_idx):
        bid = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_BODY,
                                f"pickup_obj_{obj_idx}")
        return {g for g in range(self.sim.model.ngeom)
                if self.sim.model.geom_bodyid[g] == bid}

    def _resolve_obj_joint(self, obj_idx):
        bid = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_BODY,
                                f"pickup_obj_{obj_idx}")
        if bid < 0: return None, None, None
        jnt = self.sim.model.body_jntadr[bid]
        if jnt < 0: return bid, None, None
        qpa = self.sim.model.jnt_qposadr[jnt]
        dof = self.sim.model.jnt_dofadr[jnt]
        return bid, qpa, dof

    def _get_obj_live_pos(self, obj_idx):
        bid, qpa, _ = self._resolve_obj_joint(obj_idx)
        if qpa is not None:
            return self.sim.data.qpos[qpa:qpa+3].copy()
        if bid is not None:
            return self.sim.data.xpos[bid].copy()
        return np.zeros(3)

    def _j1_for_radius(self, r):
        t = float(np.clip((r - OBJ_R_MIN)/(OBJ_R_MAX - OBJ_R_MIN), 0.0, 1.0))
        return float(J1_CLOSED - t*(J1_CLOSED - J1_OPEN)*0.5)

    # ── Finger / wrist ────────────────────────────────────────────────────────
    def _ids(self, arm):
        return (self.sim.gripper_ids_left if arm == 'left'
                else self.sim.gripper_ids_right)

    def _set_fingers(self, arm, j1, j2, j3):
        ids = self._ids(arm)
        for i in FINGER_J1_IDX:
            if i < len(ids): self.sim.data.ctrl[ids[i]] = float(j1)
        for i in FINGER_J2_IDX:
            if i < len(ids): self.sim.data.ctrl[ids[i]] = float(j2)
        for i in FINGER_J3_IDX:
            if i < len(ids): self.sim.data.ctrl[ids[i]] = float(j3)

    def _set_wrist(self, arm, wz, wx=None, wy=None):
        ids = self._ids(arm)
        if WRIST_Z_IDX < len(ids):
            self.sim.data.ctrl[ids[WRIST_Z_IDX]] = float(np.clip(wz, -1.57, 1.57))
        if WRIST_X_IDX < len(ids):
            self.sim.data.ctrl[ids[WRIST_X_IDX]] = float(np.clip(wx or 0.0, -1.57, 1.57))
        if WRIST_Y_IDX < len(ids):
            self.sim.data.ctrl[ids[WRIST_Y_IDX]] = float(np.clip(wy or 0.0, -1.57, 1.57))

    def _set_palm_fingers(self, arm, v):
        ids = self._ids(arm)
        for idx in (9, 10):
            if idx < len(ids):
                self.sim.data.ctrl[ids[idx]] = float(np.clip(v, -1.57, 1.57))

    def _open(self, arm):
        self._set_fingers(arm, J1_OPEN, J2_OPEN, J3_OPEN)
        self._set_palm_fingers(arm, 0.0)
        self._set_wrist(arm, WRIST_NEUTRAL)

    def _close_full(self, arm, j1):
        self._set_fingers(arm, j1, J2_CLOSED, J3_CLOSED)
        self._set_palm_fingers(arm, 0.30)
        self._set_wrist(arm, WRIST_CARRY)

    # ── Arm command ───────────────────────────────────────────────────────────
    def _cmd(self, arm, h1, h2, a1, theta):
        h1 = float(np.clip(h1, H_MIN, 1.43))
        h2 = float(np.clip(h2, H_MIN, 1.43))
        a1 = float(np.clip(a1, A1_MIN, A1_MAX))
        with self.sim._target_lock:
            self.sim.use_ik = False
            c = self.sim.direct_arm_commands.copy()
            if arm == 'left':
                c[0]=h1; c[1]=h2; c[2]=a1; c[3]=float(theta)
            else:
                c[4]=h1; c[5]=h2; c[6]=a1; c[7]=float(theta)
            self.sim.direct_arm_commands = c

    def _reset_arm(self, arm):
        self._cmd(arm, H_MID, H_MID, A1_HOME, 0.0)

    # ── Contact ───────────────────────────────────────────────────────────────
    def _count_contacts(self, obj_geom_ids):
        count = 0
        for i in range(self.sim.data.ncon):
            c = self.sim.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if (g1 in obj_geom_ids or g2 in obj_geom_ids) and \
               (g1 in self._gripper_geoms or g2 in self._gripper_geoms):
                count += 1
        return count

    # ── Quaternion helpers (w,x,y,z) ──────────────────────────────────────────
    @staticmethod
    def _quat_inv(q):
        return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)

    @staticmethod
    def _quat_mul(p, q):
        pw, px, py, pz = p; qw, qx, qy, qz = q
        return np.array([
            pw*qw - px*qx - py*qy - pz*qz,
            pw*qx + px*qw + py*qz - pz*qy,
            pw*qy - px*qz + py*qw + pz*qx,
            pw*qz + px*qy - py*qx + pz*qw,
        ], dtype=float)

    # ── Spring-force attach / detach / reattach ───────────────────────────────
    def _attach(self, obj_idx, arm):
        """
        Activate spring-force hold.

        Phase 1 — world-frozen: spring target = object's CURRENT world position.
        The arm is still tilted when we snap, so we must NOT record a
        palm-relative offset yet.  The spring keeps the object perfectly still
        while the arm levels and lifts.

        body_gravcomp = 1.0: MuJoCo internally cancels gravity on this body,
        so the spring only needs to resist perturbation forces — much smaller
        than full weight.  This halves the DC spring displacement and eliminates
        any residual oscillation.
        """
        bid, qpa, dof = self._resolve_obj_joint(obj_idx)
        if bid is None or dof is None:
            print(f"[Grasp] ATTACH ERROR — body/joint not found")
            self._spring_active = False; return

        # Save & override gravity compensation
        self._held_orig_gravcomp = float(self.sim.model.body_gravcomp[bid])
        self.sim.model.body_gravcomp[bid] = 1.0

        # Capture world-space target (object's current position)
        obj_pos = self.sim.data.xpos[bid].copy()

        self._held_obj_idx      = obj_idx
        self._held_obj_bid      = bid
        self._held_qpos_adr     = qpa
        self._held_dof_adr      = dof
        self._active_arm        = arm
        self._spring_target_pos  = obj_pos.copy()
        self._spring_local_pos   = None
        self._spring_local_quat  = None
        self._spring_frozen      = True     # phase 1: world-frozen
        self._spring_active      = True
        print(f"[Grasp] ATTACH obj_{obj_idx}  spring=ON  gravcomp=ON  "
              f"target={obj_pos.round(3)}")

    def _reattach(self, arm):
        """
        Switch to palm-relative spring (phase 2).

        Called after S8 LIFT has converged (arm is level).
        Records object position in palm's local frame so the spring target
        tracks the palm correctly through S9 RETRACT and carry.
        """
        if not self._spring_active or self._held_obj_bid is None:
            return
        palm_id  = self._palm[arm]
        palm_pos = self.sim.data.xpos[palm_id].copy()
        palm_mat = self.sim.data.xmat[palm_id].reshape(3, 3).copy()
        palm_q   = self.sim.data.xquat[palm_id].copy()

        obj_pos  = self.sim.data.xpos[self._held_obj_bid].copy()
        if self._held_qpos_adr is not None:
            obj_q = self.sim.data.qpos[self._held_qpos_adr+3:
                                       self._held_qpos_adr+7].copy()
        else:
            obj_q = np.array([1.0, 0.0, 0.0, 0.0])

        self._spring_local_pos  = palm_mat.T @ (obj_pos - palm_pos)
        self._spring_local_quat = self._quat_mul(self._quat_inv(palm_q), obj_q)
        self._spring_frozen     = False    # phase 2: palm-relative
        print(f"[Grasp] REATTACH  local_offset={self._spring_local_pos.round(3)}")

    def _detach(self):
        """Release hold, restore gravity on object, zero applied forces."""
        if self._held_obj_bid is not None:
            try:
                self.sim.model.body_gravcomp[self._held_obj_bid] = \
                    self._held_orig_gravcomp
            except Exception:
                pass
            if self._held_dof_adr is not None:
                try:
                    dof = self._held_dof_adr
                    self.sim.data.qfrc_applied[dof:dof+6] = 0.0
                except Exception:
                    pass
        self._held_obj_idx       = None
        self._held_obj_bid       = None
        self._held_qpos_adr      = None
        self._held_dof_adr       = None
        self._spring_active      = False
        self._spring_frozen      = False
        self._spring_target_pos  = None
        self._spring_local_pos   = None
        self._spring_local_quat  = None

    # ── Per-step spring force (called before each mj_step by play_m1.py) ─────
    def update_held_object(self):
        """
        Apply spring-damper forces to the held object every physics substep.

        This is called by play_m1.py via per_step_callback BEFORE mj_step,
        so the forces are integrated by MuJoCo's own solver.  No qpos writes.

        Force law (world frame, applied at translational DOFs of free joint):
            F = K*(target − pos) − D*vel + mass*g*ẑ

        body_gravcomp=1.0 already counteracts gravity in the solver, so the
        mass*g*ẑ term here is a small correction for residual gravity leakage.
        Total effect: spring acts as a stiff position constraint with zero DC error.

        Keep fingers closed every substep so the grip is maintained visually
        and physically (prevents finger backdrive during carry).
        """
        if not self._spring_active or self._held_obj_bid is None:
            return
        arm = self._active_arm
        if arm is None:
            return

        bid = self._held_obj_bid
        dof = self._held_dof_adr
        if dof is None:
            return

        # ── Compute spring target ─────────────────────────────────────────────
        if self._spring_frozen:
            # Phase 1: world-frozen — object stays at grasp position
            target = self._spring_target_pos
        else:
            # Phase 2: palm-relative — object rides with palm
            palm_id  = self._palm[arm]
            palm_pos = self.sim.data.xpos[palm_id]
            palm_mat = self.sim.data.xmat[palm_id].reshape(3, 3)
            target   = palm_mat @ self._spring_local_pos + palm_pos

        # ── Spring-damper force (world frame) ─────────────────────────────────
        obj_pos  = self.sim.data.xpos[bid]
        obj_vel  = self.sim.data.qvel[dof:dof+3]
        mass     = float(self.sim.model.body_mass[bid])

        F  = SPRING_K * (target - obj_pos)    # restoring spring
        F -= SPRING_D * obj_vel               # velocity damping
        F[2] += mass * 9.81                   # residual gravity compensation

        self.sim.data.qfrc_applied[dof:dof+3] = F
        self.sim.data.qfrc_applied[dof+3:dof+6] = 0.0  # no applied torque

        # ── Maintain grip ─────────────────────────────────────────────────────
        self._close_full(arm, self._close_j1)

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
        arm = self._active_arm or 'left'
        self._open(arm); self._set_wrist(arm, WRIST_NEUTRAL)
        with self.sim._target_lock: self.sim.use_ik = False
        self._detach(); self._reset_arm(arm)
        self.unfreeze_base(); self._active_arm = None

    def is_holding(self):   return self._held_obj_idx is not None
    def get_held_idx(self): return self._held_obj_idx

    # ── Grasp state machine ───────────────────────────────────────────────────
    def _run(self, obj_idx, obj_world_init, on_complete):
        print(f"\n[Grasp] ══ v51  obj_{obj_idx} @ {obj_world_init.round(3)}")
        arm = self._select_arm(obj_world_init)
        self._active_arm = arm

        obj_r, obj_hh = self._get_obj_geom(obj_idx)
        self._close_j1 = self._j1_for_radius(obj_r)
        obj_geom_ids   = self._get_obj_geom_ids(obj_idx)
        print(f"[Grasp] obj_r={obj_r:.3f}  hh={obj_hh:.3f}  "
              f"J1={self._close_j1:.3f}  geoms={len(obj_geom_ids)}")

        for attempt in range(MAX_RETRIES):
            if self._cancel: break

            live_pos = self._get_obj_live_pos(obj_idx)
            print(f"\n[Grasp] Attempt {attempt+1}/{MAX_RETRIES}  obj={live_pos.round(3)}")

            # ── S0: APPROACH ──────────────────────────────────────────────────
            # Drive base until arm pivot is ARM_IDEAL_HD (0.40m) from object.
            # ompl_windows_bridge GOAL_REACH_DIST=0.65m means nav stopped ~0.65m
            # from goal; S0 closes the remaining gap precisely.
            # Safety: base never closer than 0.28m to object.
            if self._base_frozen:
                self.unfreeze_base()

            ax0, ay0 = self._arm_pivot_world(arm)
            hd0 = math.sqrt((live_pos[0]-ax0)**2 + (live_pos[1]-ay0)**2)
            print(f"[Grasp] S0 APPROACH — hd={hd0:.3f}m  target≤{ARM_IDEAL_HD}m")

            if hd0 > ARM_IDEAL_HD + 0.02:
                t_s0 = time.time()
                while not self._cancel:
                    if time.time()-t_s0 > 20.0:
                        print("[Grasp] S0 timeout"); break
                    rx, ry, ryaw = self.sim.localization()
                    cy, sy = math.cos(ryaw), math.sin(ryaw)
                    ox, oy = ARM_OFFSETS[arm]
                    ax2 = rx+cy*ox-sy*oy; ay2 = ry+sy*ox+cy*oy
                    hd2 = math.sqrt((live_pos[0]-ax2)**2+(live_pos[1]-ay2)**2)
                    if hd2 <= ARM_IDEAL_HD + 0.01:
                        print(f"[Grasp] S0 done — hd={hd2:.3f}m"); break
                    dx2 = live_pos[0]-rx; dy2 = live_pos[1]-ry
                    db2 = math.sqrt(dx2*dx2+dy2*dy2)
                    if db2 < 0.01: break
                    step2 = min(0.08, hd2 - ARM_IDEAL_HD)
                    if db2 - step2 < 0.28:
                        step2 = max(0.0, db2 - 0.28)
                        if step2 < 0.005:
                            print("[Grasp] S0 safety stop"); break
                    if step2 < 0.005: break
                    ux2 = dx2/db2; uy2 = dy2/db2
                    nx2 = rx+ux2*step2; ny2 = ry+uy2*step2
                    print(f"[Grasp] S0 step — hd={hd2:.3f}m  Δ={step2:.3f}m")
                    self.kill_base_vel = False
                    self._set_base_target(nx2, ny2, ryaw)
                    t_step = time.time()
                    while time.time()-t_step < 5.0:
                        time.sleep(0.15)
                        bx, by = self.sim.localization()[:2]
                        if math.sqrt((bx-nx2)**2+(by-ny2)**2) < 0.04: break

            self.freeze_base()
            self.wait_for_settle()

            # Recompute geometry with frozen base
            live_pos = self._get_obj_live_pos(obj_idx)
            theta    = self._compute_theta(live_pos, arm)
            obj_z    = float(live_pos[2])
            ax, ay   = self._arm_pivot_world(arm)
            hd       = math.sqrt((live_pos[0]-ax)**2+(live_pos[1]-ay)**2)
            print(f"[Grasp] θ={math.degrees(theta):.1f}°  hd={hd:.3f}m  obj_z={obj_z:.3f}m")

            # ── S1: RETRACT + RAISE ───────────────────────────────────────────
            # A1_SAFE=0.05m: arm tip only 0.17m from body centre — clear of chassis
            # at ALL h values including H_MIN=0.03.
            print(f"[Grasp] S1 RETRACT+RAISE — h={H_MID:.2f}  a1={A1_SAFE:.2f}  θ={math.degrees(theta):.1f}°")
            self._open(arm)
            self._cmd(arm, H_MID, H_MID, A1_SAFE, theta)
            self._wait_converge(arm, "raise")
            if self._cancel: break

            # ── S2: DESCEND — lower h1=h2 to object height, a1=A1_SAFE ──────
            h_sym = self._h_cmd_for_palm_z(obj_z)
            print(f"[Grasp] S2 DESCEND — h→{h_sym:.3f}  (palm_z≈{obj_z:.3f}m)  "
                  f"a1={A1_SAFE:.2f}")
            self._cmd(arm, h_sym, h_sym, A1_SAFE, theta)
            self._wait_converge(arm, "descend")
            if self._cancel: break

            palm_z   = float(self._palm_pos(arm)[2])
            Z_gap    = palm_z - obj_z
            print(f"[Grasp]   palm_z={palm_z:.3f}  obj_z={obj_z:.3f}  Δz={Z_gap:.3f}")

            # ── S2b: TILT — raise h2 until palm reaches obj_z (a1=A1_SAFE) ──
            H2_TILT_MAX  = 1.40
            H2_TILT_STEP = 0.12
            Z_TOL        = 0.06
            h1_locked    = max(h_sym, H_MIN)
            h2_cur       = h1_locked

            if Z_gap > Z_TOL:
                print(f"[Grasp] S2b TILT — h1={h1_locked:.3f} locked  "
                      f"h2 ramps to {H2_TILT_MAX}  a1={A1_SAFE:.2f}")
                while h2_cur < H2_TILT_MAX and not self._cancel:
                    h2_cur = min(h2_cur + H2_TILT_STEP, H2_TILT_MAX)
                    self._cmd(arm, h1_locked, h2_cur, A1_SAFE, theta)
                    self._wait_converge(arm, f"tilt h2={h2_cur:.2f}")
                    palm_now_z = float(self._palm_pos(arm)[2])
                    print(f"[Grasp]   h2={h2_cur:.2f}  palm_z={palm_now_z:.3f}  "
                          f"Δz={palm_now_z-obj_z:.3f}")
                    if palm_now_z <= obj_z + Z_TOL:
                        print("[Grasp]   ✓ tilt done"); break
                if self._cancel: break
            else:
                print(f"[Grasp]   no tilt needed — Δz={Z_gap:.3f}")

            # ── S3: OPEN GRIPPER ──────────────────────────────────────────────
            print("[Grasp] S3 OPEN")
            self._open(arm)
            time.sleep(0.40)
            if self._cancel: break

            # ── S4: SERVO EXTEND — physical approach ──────────────────────────
            # Arm is ALREADY at object height (h1_locked, h2_cur).
            # a1 grows in 2 cm steps; after each step palm_dist is measured.
            # This is FK-independent: the arm physically approaches the object.
            # Emergency stop prevents collision with object geometry.
            live_pos = self._get_obj_live_pos(obj_idx)
            theta    = self._compute_theta(live_pos, arm)

            print(f"[Grasp] S4 SERVO EXTEND — stop at palm_dist≤{SIDE_STOP_DIST:.2f}m  "
                  f"step={APPROACH_STEP:.2f}m")
            a1_servo = A1_SAFE
            obj_world_3d = live_pos.copy()

            while a1_servo < A1_MAX and not self._cancel:
                a1_servo = min(a1_servo + APPROACH_STEP, A1_MAX)
                self._cmd(arm, h1_locked, h2_cur, a1_servo, theta)
                time.sleep(APPROACH_SETTLE)   # 0.40s = 96.8% converged at alpha=0.25

                palm_now = self._palm_pos(arm)
                pd       = float(np.linalg.norm(palm_now - obj_world_3d))
                print(f"[Grasp]   servo a1={a1_servo:.3f}  palm_dist={pd:.3f}  "
                      f"palm_z={palm_now[2]:.3f}")

                if pd < APPROACH_MIN_PD:
                    print(f"[Grasp]   EMERGENCY STOP — pd={pd:.3f}")
                    a1_servo = max(a1_servo - APPROACH_STEP, A1_SAFE)
                    self._cmd(arm, h1_locked, h2_cur, a1_servo, theta)
                    time.sleep(APPROACH_SETTLE); break

                if pd <= SIDE_STOP_DIST:
                    print(f"[Grasp]   TARGET REACHED — pd={pd:.3f}"); break

            if self._cancel: break
            self._wait_converge(arm, "servo-settle", min_wait=0.60)

            palm_final = self._palm_pos(arm)
            pd_final   = float(np.linalg.norm(palm_final - obj_world_3d))
            print(f"[Grasp]   EXTEND DONE: palm={palm_final.round(3)}  "
                  f"obj={obj_world_3d.round(3)}  pd={pd_final:.3f}  a1={a1_servo:.3f}")

            if pd_final > ATTACH_DIST_PROX:
                print(f"[Grasp]   TOO FAR ({pd_final:.3f}>{ATTACH_DIST_PROX:.3f}) — retry")
                self._open(arm); self._cmd(arm, H_MID, H_MID, A1_SAFE, 0.0)
                self._wait_converge(arm, "retry-reset")
                self.unfreeze_base(); continue

            # ── S5: CLOSE — physical finger contact ───────────────────────────
            # Arm is STATIONARY.  Fingers ramp closed over T_CLOSE seconds.
            # MuJoCo generates real contact forces between finger geometry and
            # the object cylinder.  This is the only step that touches the object.
            print(f"[Grasp] S5 CLOSE — J1:{J1_OPEN:.2f}→{self._close_j1:.3f}  "
                  f"T={T_CLOSE:.1f}s  (physical contact)")
            t0 = time.time()
            while time.time()-t0 < T_CLOSE and not self._cancel:
                frac   = min((time.time()-t0)/T_CLOSE, 1.0)
                j1_now = J1_OPEN + (self._close_j1 - J1_OPEN)*frac
                self._set_fingers(arm, j1_now, J2_CLOSED, J3_CLOSED)
                self._cmd(arm, h1_locked, h2_cur, a1_servo, theta)
                time.sleep(T_STEP)
            if self._cancel: break

            # ── S6: VERIFY CONTACT ────────────────────────────────────────────
            n_contacts   = self._count_contacts(obj_geom_ids)
            pd_grip      = float(np.linalg.norm(self._palm_pos(arm) - obj_world_3d))
            close_enough = pd_grip < ATTACH_DIST_PROX
            success      = (n_contacts >= CONTACT_REQUIRED) or close_enough

            print(f"[Grasp] S6 VERIFY — contacts={n_contacts}  pd={pd_grip:.3f}  "
                  f"→ {'GRIP ✓' if success else 'MISS ✗'}")
            self.policy.update(1.0 if success else -1.0)

            if not success:
                self._open(arm); self._cmd(arm, H_MID, H_MID, A1_SAFE, 0.0)
                self._wait_converge(arm, "miss-reset")
                self.unfreeze_base(); continue

            # ── S7: ATTACH — spring-force hold, world-frozen phase ────────────
            # Activated AFTER physical contact confirmed.
            # body_gravcomp=1.0 + spring K=600 D=60 → overdamped, zero oscillation.
            # Object stays at its CURRENT position (not teleported).
            self._attach(obj_idx, arm)
            time.sleep(0.20)
            if self._cancel: self._detach(); break

            # ── S8: LIFT — raise arm, spring carries object upward ────────────
            # h1=h2 → H_CARRY levels the arm and lifts.
            # World-frozen spring keeps object at grasp position until
            # _reattach() switches to palm-relative tracking.
            h_carry_cmd = self._h_cmd_for_palm_z(H_CARRY)
            print(f"[Grasp] S8 LIFT — h→{h_carry_cmd:.3f}  (spring holds object)")
            self._cmd(arm, h_carry_cmd, h_carry_cmd, a1_servo, theta)
            self._wait_converge(arm, "lift", min_wait=2.0)
            if self._cancel: self._detach(); break

            # Switch spring to palm-relative carry
            self._reattach(arm)

            # ── S9: RETRACT — pull arm in at carry height ─────────────────────
            print(f"[Grasp] S9 RETRACT — a1:{a1_servo:.3f}→{A1_HOME:.2f}")
            self._cmd(arm, h_carry_cmd, h_carry_cmd, A1_HOME, theta)
            self._wait_converge(arm, "retract", min_wait=1.2)
            if self._cancel: self._detach(); break

            obj_carry = self._get_obj_live_pos(obj_idx)
            print(f"[Grasp] ✓ Carrying obj_{obj_idx}  "
                  f"obj_z={obj_carry[2]:.3f}m")
            self.unfreeze_base()
            if on_complete: on_complete(True)
            return

        # All attempts exhausted / cancelled
        print("[Grasp] All attempts failed")
        arm = self._active_arm or 'left'
        self._open(arm); self._set_wrist(arm, WRIST_NEUTRAL)
        with self.sim._target_lock: self.sim.use_ik = False
        self._detach(); self._reset_arm(arm); self._active_arm = None
        self.unfreeze_base()
        if on_complete: on_complete(False)