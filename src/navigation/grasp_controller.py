
import threading, time, math
import numpy as np
import mujoco

H_MID  = 0.72
H_CARRY = 0.65

A1_SAFE = 0.05
A1_HOME = 0.28
A1_MIN  = 0.00
A1_MAX  = 0.58

FINGERTIP_OVERHANG = 0.08
H_MIN              = 0.03

HD_TARGET = 0.45

SPRING_K   = 600.0
SPRING_D   = 60.0

SIDE_STOP_DIST  = 0.17
APPROACH_STEP   = 0.02
APPROACH_SETTLE = 0.40
APPROACH_MIN_PD = 0.09

PALM_Z_OFFSET_DEFAULT = 0.168
PALM_A1_SCALE_DEFAULT = 1.0

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

CONTACT_REQUIRED = 1
ATTACH_DIST_PROX = 0.40

ARM_IDEAL_HD = 0.62
BASE_OBJECT_MIN_DIST = 0.70

LOW_GRASP_H1_MIN = 0.30
LOW_GRASP_H2_DELTA = 0.20

SETTLE_THRESHOLD = 0.008
SETTLE_INTERVAL  = 0.20
SETTLE_MAX_WAIT  = 5.0
SETTLE_STABLE    = 3

CONVERGE_MIN_WAIT  = 0.70
CONVERGE_THRESHOLD = 0.004
CONVERGE_INTERVAL  = 0.20
CONVERGE_STABLE    = 3
CONVERGE_TIMEOUT   = 12.0

T_STEP   = 0.04
T_CLOSE  = 2.0
MAX_RETRIES = 3
ARM_OFFSETS = {'left': (0.12, 0.15), 'right': (0.12, -0.15)}


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


class GraspController:

    def __init__(self, sim):
        self.sim          = sim
        self.policy       = GraspPolicy()
        self._cancel      = False
        self._thread      = None
        self._active_arm  = None
        self._close_j1    = J1_CLOSED

        self._held_obj_idx      = None
        self._held_obj_bid      = None
        self._held_dof_adr      = None
        self._held_orig_gravcomp = 0.0
        self._spring_active     = False
        self._spring_frozen     = False
        self._spring_target_pos  = None
        self._spring_local_pos   = None
        self._spring_local_quat  = None

        self._held_qpos_adr = None

        self.kill_base_vel = False
        self._base_frozen  = False
        self.on_base_moved = None

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

        self._palm_z_at_zero = PALM_Z_OFFSET_DEFAULT
        self._palm_z_slope   = 1.0
        self._palm_a1_scale  = PALM_A1_SCALE_DEFAULT
        self._calibrate_arm_offsets()

        print(f"[Grasp] SPRING K={SPRING_K} D={SPRING_D}  "
              f"A1_SAFE={A1_SAFE}  SIDE_STOP={SIDE_STOP_DIST}  "
              f"ARM_IDEAL_HD={ARM_IDEAL_HD}  "
              f"base_dof={self._base_dof}  "
              f"palm=({self._palm['left']},{self._palm['right']})  "
              f"z_at_zero={self._palm_z_at_zero:.3f}  "
              f"z_slope={self._palm_z_slope:.3f}")

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

    def _count_contacts(self, obj_geom_ids):
        count = 0
        for i in range(self.sim.data.ncon):
            c = self.sim.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if (g1 in obj_geom_ids or g2 in obj_geom_ids) and \
               (g1 in self._gripper_geoms or g2 in self._gripper_geoms):
                count += 1
        return count

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

    def _attach(self, obj_idx, arm):
        bid, qpa, dof = self._resolve_obj_joint(obj_idx)
        if bid is None or dof is None:
            print(f"[Grasp] ATTACH ERROR — body/joint not found")
            self._spring_active = False; return

        self._held_orig_gravcomp = float(self.sim.model.body_gravcomp[bid])
        self.sim.model.body_gravcomp[bid] = 1.0

        obj_pos = self.sim.data.xpos[bid].copy()

        self._held_obj_idx      = obj_idx
        self._held_obj_bid      = bid
        self._held_qpos_adr     = qpa
        self._held_dof_adr      = dof
        self._active_arm        = arm
        self._spring_target_pos  = obj_pos.copy()
        self._spring_local_pos   = None
        self._spring_local_quat  = None
        self._spring_frozen      = True
        self._spring_active      = True
        print(f"[Grasp] ATTACH obj_{obj_idx}  spring=ON  gravcomp=ON  "
              f"target={obj_pos.round(3)}")

    def _reattach(self, arm):
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
        self._spring_frozen     = False
        print(f"[Grasp] REATTACH  local_offset={self._spring_local_pos.round(3)}")

    def _detach(self):
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

    def update_held_object(self):
        if not self._spring_active or self._held_obj_bid is None:
            return
        arm = self._active_arm
        if arm is None:
            return

        bid = self._held_obj_bid
        dof = self._held_dof_adr
        if dof is None:
            return

        if self._spring_frozen:
            target = self._spring_target_pos
        else:
            palm_id  = self._palm[arm]
            palm_pos = self.sim.data.xpos[palm_id]
            palm_mat = self.sim.data.xmat[palm_id].reshape(3, 3)
            target   = palm_mat @ self._spring_local_pos + palm_pos

        obj_pos  = self.sim.data.xpos[bid]
        obj_vel  = self.sim.data.qvel[dof:dof+3]
        mass     = float(self.sim.model.body_mass[bid])

        F  = SPRING_K * (target - obj_pos)
        F -= SPRING_D * obj_vel
        F[2] += mass * 9.81

        self.sim.data.qfrc_applied[dof:dof+3] = F
        self.sim.data.qfrc_applied[dof+3:dof+6] = 0.0

        self._close_full(arm, self._close_j1)

    def grasp(self, obj_idx, obj_world_pos, on_complete=None,
              preferred_arm=None, skip_approach=False):
        self._cancel = True
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._cancel = False
        self._thread = threading.Thread(
            target=self._run,
            args=(int(obj_idx), np.array(obj_world_pos, dtype=float),
                  on_complete, preferred_arm, skip_approach),
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

    def _run(self, obj_idx, obj_world_init, on_complete,
             preferred_arm=None, skip_approach=False):
        print(f"\n[Grasp] ══  obj_{obj_idx} @ {obj_world_init.round(3)}")
        if preferred_arm:
            arm = preferred_arm
            print(f"[Grasp] Arm: {arm.upper()} (OMPL-selected, skip_approach={skip_approach})")
        else:
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

            if self._base_frozen:
                self.unfreeze_base()

            ax0, ay0 = self._arm_pivot_world(arm)
            hd0 = math.sqrt((live_pos[0]-ax0)**2 + (live_pos[1]-ay0)**2)

            if not skip_approach:
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
                        if db2 - step2 < BASE_OBJECT_MIN_DIST:
                            step2 = max(0.0, db2 - BASE_OBJECT_MIN_DIST)
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
            else:
                print(f"[Grasp] S0 SKIP (OMPL base positioned) — hd={hd0:.3f}m")

            self.freeze_base()
            self.wait_for_settle()

            live_pos = self._get_obj_live_pos(obj_idx)
            theta    = self._compute_theta(live_pos, arm)
            obj_z    = float(live_pos[2])
            ax, ay   = self._arm_pivot_world(arm)
            hd       = math.sqrt((live_pos[0]-ax)**2+(live_pos[1]-ay)**2)
            print(f"[Grasp] θ={math.degrees(theta):.1f}°  hd={hd:.3f}m  obj_z={obj_z:.3f}m")

            if not skip_approach:
                print(f"[Grasp] S1 RETRACT+RAISE — h={H_MID:.2f}  a1={A1_SAFE:.2f}  "
                      f"θ={math.degrees(theta):.1f}°")
                self._open(arm)
                self._cmd(arm, H_MID, H_MID, A1_SAFE, theta)
                self._wait_converge(arm, "raise")
                if self._cancel: break

                h_raw     = self._h_cmd_for_palm_z(obj_z)
                h1_locked = max(h_raw, LOW_GRASP_H1_MIN)
                h2_cur    = min(max(h1_locked + LOW_GRASP_H2_DELTA, h_raw), 1.40)
                print(f"[Grasp] S2 DESCEND — h_raw={h_raw:.3f}  "
                      f"h1={h1_locked:.3f}  h2={h2_cur:.3f}  "
                      f"(palm_z≈{obj_z:.3f}m)  a1={A1_SAFE:.2f}")
                self._cmd(arm, h1_locked, h2_cur, A1_SAFE, theta)
                self._wait_converge(arm, "descend")
                if self._cancel: break

                palm_z = float(self._palm_pos(arm)[2])
                Z_gap  = palm_z - obj_z
                print(f"[Grasp]   palm_z={palm_z:.3f}  obj_z={obj_z:.3f}  Δz={Z_gap:.3f}")

                H2_TILT_MAX  = 1.40
                H2_TILT_STEP = 0.12
                Z_TOL        = 0.06

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

            else:
                if arm == 'left':
                    h1_locked  = float(self.sim.direct_arm_commands[0])
                    h2_cur     = float(self.sim.direct_arm_commands[1])
                    a1_current = float(self.sim.direct_arm_commands[2])
                else:
                    h1_locked  = float(self.sim.direct_arm_commands[4])
                    h2_cur     = float(self.sim.direct_arm_commands[5])
                    a1_current = float(self.sim.direct_arm_commands[6])
                print(f"[Grasp] S1-S2b SKIP (OMPL pre-positioned) — "
                      f"h1={h1_locked:.3f}  h2={h2_cur:.3f}  a1={a1_current:.3f}")

            print("[Grasp] S3 OPEN")
            self._open(arm)
            time.sleep(0.40)
            if self._cancel: break

            live_pos     = self._get_obj_live_pos(obj_idx)
            theta        = self._compute_theta(live_pos, arm)
            obj_world_3d = live_pos.copy()

            if not skip_approach:
                print(f"[Grasp] S4 SERVO EXTEND — stop at palm_dist≤{SIDE_STOP_DIST:.2f}m  "
                      f"step={APPROACH_STEP:.2f}m")
                a1_servo = A1_SAFE

                while a1_servo < A1_MAX and not self._cancel:
                    a1_servo = min(a1_servo + APPROACH_STEP, A1_MAX)
                    self._cmd(arm, h1_locked, h2_cur, a1_servo, theta)
                    time.sleep(APPROACH_SETTLE)

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

            else:
                a1_servo = a1_current
                palm_now = self._palm_pos(arm)
                pd_now   = float(np.linalg.norm(palm_now - obj_world_3d))
                print(f"[Grasp] S4 SKIP (OMPL) — palm={palm_now.round(3)}  "
                      f"obj={obj_world_3d.round(3)}  pd={pd_now:.3f}m")
                if self._cancel: break

            palm_final = self._palm_pos(arm)
            pd_final   = float(np.linalg.norm(palm_final - obj_world_3d))
            print(f"[Grasp]   EXTEND DONE: palm={palm_final.round(3)}  "
                  f"obj={obj_world_3d.round(3)}  pd={pd_final:.3f}  a1={a1_servo:.3f}")

            if pd_final > ATTACH_DIST_PROX:
                print(f"[Grasp]   TOO FAR ({pd_final:.3f}>{ATTACH_DIST_PROX:.3f}) — retry")
                self._open(arm); self._cmd(arm, H_MID, H_MID, A1_SAFE, 0.0)
                self._wait_converge(arm, "retry-reset")
                self.unfreeze_base(); continue

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

            self._attach(obj_idx, arm)
            time.sleep(0.20)
            if self._cancel: self._detach(); break

            h_carry_cmd = self._h_cmd_for_palm_z(H_CARRY)
            print(f"[Grasp] S8 LIFT — h→{h_carry_cmd:.3f}  (spring holds object)")
            self._cmd(arm, h_carry_cmd, h_carry_cmd, a1_servo, theta)
            self._wait_converge(arm, "lift", min_wait=2.0)
            if self._cancel: self._detach(); break

            self._reattach(arm)

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

        print("[Grasp] All attempts failed")
        arm = self._active_arm or 'left'
        self._open(arm); self._set_wrist(arm, WRIST_NEUTRAL)
        with self.sim._target_lock: self.sim.use_ik = False
        self._detach(); self._reset_arm(arm); self._active_arm = None
        self.unfreeze_base()
        if on_complete: on_complete(False)
