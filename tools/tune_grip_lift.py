
import argparse
import os
import sys
import time
import json
import csv
import itertools
import threading
import traceback
from contextlib import contextmanager

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

from simulations.morph_i_free_move import ParallelRobot
from navigation.arm_planner import MORPHBridge
from navigation import grasp_executor as gx
from navigation.grasp_executor import (GraspExecutor,
                                        compute_grasp_targets,
                                        reset_plan_data_for_ik)

gx.FAST_PICKUP_MODE   = False
gx.STRICT_PICKUP_MODE = True
print(f"[Tuning] STRICT_PICKUP_MODE forced to True "
      f"(FAST_PICKUP_MODE={gx.FAST_PICKUP_MODE})")


KNOWN_OBJ_XY_LIST = [
    (5.0, -6.5),
    (3.5, -6.5),
    (4.0, -7.0),
    (5.5, -7.0),
]
KNOWN_OBJ_Z = 0.15



TUNABLE_RUNTIME = {
    "STRICT_GRIP_SAFETY":          {"default": 1.80, "candidates": [1.50, 1.80, 2.20, 2.60, 3.00]},
    "STRICT_FRICTION_MU":          {"default": 0.70, "candidates": [0.50, 0.70, 1.00]},
    "STRICT_FORCE_STOP_STABLE_TICKS": {"default": 1, "candidates": [1, 2]},
    "LIFT_TIGHTEN_PREVERIFY_RAD":   {"default": 0.05, "candidates": [0.03, 0.05, 0.10, 0.15]},
    "LIFT_TIGHTEN_PRELIFT_RAD":     {"default": 0.05, "candidates": [0.03, 0.05, 0.10, 0.15]},
    "LIFT_TIGHTEN_POSTLIFT_RAD":    {"default": 0.03, "candidates": [0.01, 0.03, 0.06, 0.10]},
    "LIFT_RETIGHTEN_PER_STEP_RAD": {"default": 0.012, "candidates": [0.005, 0.012, 0.020, 0.030]},
    "LIFT_RETIGHTEN_INTERVAL_STEPS": {"default": 5, "candidates": [3, 5, 10]},
    "LIFT_STEPS_MULTIPLIER":        {"default": 2, "candidates": [1, 2, 4]},
    "LIFT_PER_STEP_SETTLE_MULTIPLIER": {"default": 1.2, "candidates": [1.0, 1.2, 2.0]},
    "STRICT_SLIP_DISP_THRESH":     {"default": 0.012, "candidates": [0.008, 0.012, 0.020]},
    "STRICT_SLIP_VEL_THRESH":      {"default": 0.04,  "candidates": [0.03, 0.04, 0.06]},
    "STRICT_LIFT_OBSERVE_S":       {"default": 0.80,  "candidates": [0.60, 0.80, 1.20]},
    "STRICT_RETRY_MAX":            {"default": 2, "candidates": [0, 1, 2]},
}



class SimContext:

    def __init__(self, xml_path, verbose=True):
        self.xml_path = xml_path
        self.verbose = verbose
        if verbose:
            print(f"[SimContext] constructing sim with xml={xml_path}")
        self.sim = ParallelRobot(xml_path, run_mode="headless",
                                  record=False)
        if verbose:
            print("[SimContext] ParallelRobot constructed")
        self.arm_bridge = MORPHBridge(xml_path, arm=1,
                                       use_calibration=True,
                                       calib_wrist_mode="sidegrip")
        if verbose:
            print("[SimContext] MORPHBridge constructed")
        self.grasp_exec = GraspExecutor(self.sim, self.arm_bridge)
        if verbose:
            print("[SimContext] GraspExecutor constructed")
        self._sim_thread = None
        self._sim_running = False

    def _sim_loop(self):
        while self._sim_running:
            try:
                self.sim.step_simulation(render=False)
                time.sleep(0.005)
            except Exception as e:
                print(f"[SimContext] sim step error: {e}")
                time.sleep(0.05)

    def start_bg_sim(self):
        if self._sim_running:
            return
        self._sim_running = True
        self._sim_thread = threading.Thread(
            target=self._sim_loop, daemon=True)
        self._sim_thread.start()
        time.sleep(0.05)

    def stop_bg_sim(self):
        if not self._sim_running:
            return
        self._sim_running = False
        if self._sim_thread is not None:
            self._sim_thread.join(timeout=1.0)
            self._sim_thread = None

    def reset_state(self):
        try:
            self.grasp_exec._cancel = True
        except Exception:
            pass

        try:
            mujoco.mj_resetData(self.sim.model, self.sim.data)
        except Exception as e:
            print(f"[SimContext] mj_resetData failed: {e}")

        try:
            key_id = mujoco.mj_name2id(self.sim.model,
                                        mujoco.mjtObj.mjOBJ_KEY,
                                        "home")
            if key_id >= 0:
                mujoco.mj_resetDataKeyframe(self.sim.model,
                                             self.sim.data, key_id)
            else:
                print("[SimContext] no 'home' keyframe in model")
        except Exception as e:
            print(f"[SimContext] keyframe restore failed: {e}")

        try:
            mujoco.mj_resetData(self.arm_bridge.model,
                                self.arm_bridge.planning_data)
            key_id_pd = mujoco.mj_name2id(self.arm_bridge.model,
                                           mujoco.mjtObj.mjOBJ_KEY,
                                           "home")
            if key_id_pd >= 0:
                mujoco.mj_resetDataKeyframe(self.arm_bridge.model,
                                             self.arm_bridge.planning_data,
                                             key_id_pd)
            self.arm_bridge.planning_data.eq_active[:] = 0
            mujoco.mj_forward(self.arm_bridge.model,
                              self.arm_bridge.planning_data)
        except Exception as e:
            print(f"[SimContext] planning_data reset failed: {e}")

        try:
            self.sim.reset("home")
        except Exception as e:
            print(f"[SimContext] sim.reset 'home' failed: {e}")

        try:
            self.grasp_exec._clear_held_state()
        except Exception:
            pass
        for attr in ("_last_valid_pre_grasp_q",
                      "last_grasp_failure_info",
                      "_active_pin_fn"):
            try:
                setattr(self.grasp_exec, attr, None)
            except Exception:
                pass
        for attr, default in (("_strict_retry_count", 0),
                                ("_strict_force_multiplier", 1.0),
                                ("_strict_finger_attempts_used", 0),
                                ("_pre_close_nudges_used", 0),
                                ("_pre_close_backup_used", False),
                                ("_side_grip_active", False),
                                ("_cancel", False)):
            try:
                setattr(self.grasp_exec, attr, default)
            except Exception:
                pass

        try:
            with self.sim._pin_lock:
                self.sim._pin_callbacks.clear()
        except Exception:
            pass

        mujoco.mj_forward(self.sim.model, self.sim.data)

        try:
            self.sim.target_base = self.sim.localization()
        except Exception:
            pass

        self.trial_count = getattr(self, 'trial_count', 0) + 1

    def teardown(self):
        self.stop_bg_sim()



class Trial:

    def __init__(self, xml_path, obj_idx=0, obj_xy=(5.0, -6.5),
                 approach_yaw_deg=None, verbose=True,
                 sim_context=None, drift=None):
        self.xml_path = xml_path
        self.obj_idx = obj_idx
        self.obj_xy = obj_xy
        self.approach_yaw_deg = (approach_yaw_deg
                                  if approach_yaw_deg is not None
                                  else 122.6)
        self.verbose = verbose
        self.sim_context = sim_context
        self.drift = drift or {}

        if sim_context is not None:
            self.sim = sim_context.sim
            self.arm_bridge = sim_context.arm_bridge
            self.grasp_exec = sim_context.grasp_exec
            self._owns_context = False
        else:
            self.sim = None
            self.arm_bridge = None
            self.grasp_exec = None
            self._owns_context = True
        self._sim_thread = None
        self._sim_running = False
        self._grasp_done_event = threading.Event()
        self._grasp_result = {"success": None, "info": None}

        self.metrics = {
            "close_3finger": False,
            "force_stop_max_n": 0.0,
            "verify_passed": False,
            "verify_opposing": False,
            "obj_z_pre_lift": None,
            "obj_z_post_motion": None,
            "obj_followed_grip": False,
            "obj_z_after_hold": None,
            "obj_held_2s": False,
            "slip_retries": 0,
            "trial_duration_s": 0.0,
            "errors": [],
        }


    def setup(self):
        if self.sim_context is not None:
            if self.verbose:
                print("[Trial] reusing shared SimContext (fast path)")
            self.sim_context.stop_bg_sim()
            self.sim_context.reset_state()
            return
        if self.verbose:
            print(f"[Trial] setting up sim with xml={self.xml_path}")
        self.sim = ParallelRobot(self.xml_path, run_mode="headless",
                                  record=False)
        if self.verbose:
            print("[Trial] ParallelRobot constructed")
        self.arm_bridge = MORPHBridge(self.xml_path, arm=1,
                                       use_calibration=True,
                                       calib_wrist_mode="sidegrip")
        if self.verbose:
            print("[Trial] MORPHBridge constructed")
        self.grasp_exec = GraspExecutor(self.sim, self.arm_bridge)
        if self.verbose:
            print("[Trial] GraspExecutor constructed")

    def _sim_loop(self):
        while self._sim_running:
            try:
                self.sim.step_simulation(render=False)
                time.sleep(0.005)
            except Exception as e:
                print(f"[Trial] sim step error: {e}")
                time.sleep(0.05)

    def _start_bg_sim(self):
        if self.sim_context is not None:
            self.sim_context.start_bg_sim()
            return
        if self._sim_running:
            return
        self._sim_running = True
        self._sim_thread = threading.Thread(
            target=self._sim_loop, daemon=True)
        self._sim_thread.start()
        if self.verbose:
            print("[Trial] background sim thread started")
        time.sleep(0.1)

    def _stop_bg_sim(self):
        if self.sim_context is not None:
            self.sim_context.stop_bg_sim()
            return
        if not self._sim_running:
            return
        self._sim_running = False
        if self._sim_thread is not None:
            self._sim_thread.join(timeout=1.0)
            self._sim_thread = None

    def teardown(self):
        self._stop_bg_sim()


    def place_initial_state_simple(self):
        from math import cos, sin, atan2
        chassis_dx = float(self.drift.get("chassis_dx", 0.0))
        chassis_dy = float(self.drift.get("chassis_dy", 0.0))
        chassis_dyaw = float(self.drift.get("chassis_dyaw", 0.0))
        chassis_xy = (4.0 + chassis_dx, -6.5 + chassis_dy)
        chassis_yaw = 0.0 + chassis_dyaw
        if self.verbose and (chassis_dx or chassis_dy or chassis_dyaw):
            print(f"[Trial-DRIFT] chassis offset: "
                  f"dx={chassis_dx:+.3f}m dy={chassis_dy:+.3f}m "
                  f"dyaw={chassis_dyaw:+.3f}rad "
                  f"→ ({chassis_xy[0]:.3f},{chassis_xy[1]:.3f}) "
                  f"yaw={chassis_yaw:+.3f}rad")
        base_bid = mujoco.mj_name2id(self.sim.model,
                                      mujoco.mjtObj.mjOBJ_BODY, "robot")
        if base_bid < 0:
            base_bid = mujoco.mj_name2id(self.sim.model,
                                          mujoco.mjtObj.mjOBJ_BODY, "base")
        if base_bid >= 0:
            ja_b = int(self.sim.model.body_jntadr[base_bid])
            if ja_b >= 0:
                qa_b = int(self.sim.model.jnt_qposadr[ja_b])
                self.sim.data.qpos[qa_b] = chassis_xy[0]
                self.sim.data.qpos[qa_b + 1] = chassis_xy[1]
                self.sim.data.qpos[qa_b + 3] = cos(chassis_yaw / 2.0)
                self.sim.data.qpos[qa_b + 4] = 0.0
                self.sim.data.qpos[qa_b + 5] = 0.0
                self.sim.data.qpos[qa_b + 6] = sin(chassis_yaw / 2.0)
        self.sim.target_base = np.array([float(chassis_xy[0]),
                                          float(chassis_xy[1]),
                                          float(chassis_yaw)])

        wz_drift = float(self.drift.get("wz", 0.0))
        wx_drift = float(self.drift.get("wx", 0.0))
        if self.verbose and (wz_drift or wx_drift):
            print(f"[Trial-DRIFT] wrist offset: "
                  f"wz={wz_drift:+.3f}rad wx={wx_drift:+.3f}rad")
        grasp_q = [
            0.135,
            0.360,
            0.305,
            float(chassis_yaw),
            0.025,
           -1.88 + wz_drift,
            0.80 + wx_drift,
            0.00,
        ]
        for i, q in enumerate(grasp_q[:4]):
            self.sim.data.qpos[self.sim.qpos_indices[i]] = q
        wrist_names = ("HandBearingJoint_1", "gripper_z_rotation_1",
                       "gripper_x_rotation_1", "gripper_y_rotation_1")
        for i, name in enumerate(wrist_names):
            jid = mujoco.mj_name2id(self.sim.model,
                                     mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                qa = int(self.sim.model.jnt_qposadr[jid])
                self.sim.data.qpos[qa] = grasp_q[4 + i]
        for i, q in enumerate([1.20, 1.20, 0.10, 0.0]):
            self.sim.data.qpos[self.sim.qpos_indices[4 + i]] = q
        try:
            self.sim.data.ctrl[self.sim.actuator_ids[4 + i]] = q * 100
        except Exception:
            pass
        try:
            self.grasp_exec._set_arm_cmd(grasp_q)
        except Exception as _e:
            print(f"[Trial] WARNING _set_arm_cmd: {_e}")

        try:
            gids = self.sim.gripper_ids_left
            open_curl = self.grasp_exec._curl_targets(
                gx.GRIPPER_OPEN_POS)
            with self.sim._target_lock:
                for j_idx, val in enumerate(open_curl):
                    if j_idx < 9 and j_idx < len(gids):
                        self.sim.data.ctrl[gids[j_idx]] = float(val)
                        addrs = self.grasp_exec._ensure_finger_joint_qposadrs()
                        if addrs and j_idx < len(addrs) and addrs[j_idx] >= 0:
                            self.sim.data.qpos[addrs[j_idx]] = float(val)
        except Exception as _e:
            print(f"[Trial] WARNING finger open: {_e}")

        mujoco.mj_forward(self.sim.model, self.sim.data)

        try:
            pocket_xyz = self.grasp_exec._pinch_midpoint_xyz(
                self.sim.data).copy()
        except Exception:
            try:
                palm_bid = self.grasp_exec.gripper_body_id
                pocket_xyz = self.sim.data.xpos[palm_bid].copy()
            except Exception:
                pocket_xyz = np.array([4.5, -6.5, 0.20])

        obj_name = f"pickup_obj_{self.obj_idx}"
        obj_bid = mujoco.mj_name2id(self.sim.model,
                                     mujoco.mjtObj.mjOBJ_BODY, obj_name)
        if obj_bid < 0:
            raise RuntimeError(f"obj '{obj_name}' not found")
        jntadr = int(self.sim.model.body_jntadr[obj_bid])
        qadr = int(self.sim.model.jnt_qposadr[jntadr])
        self.sim.data.qpos[qadr] = float(pocket_xyz[0])
        self.sim.data.qpos[qadr + 1] = float(pocket_xyz[1])
        self.sim.data.qpos[qadr + 2] = float(pocket_xyz[2])
        self.sim.data.qpos[qadr + 3:qadr + 7] = [1.0, 0.0, 0.0, 0.0]
        for i in range(10):
            if i == self.obj_idx:
                continue
            name_i = f"pickup_obj_{i}"
            bid_i = mujoco.mj_name2id(self.sim.model,
                                       mujoco.mjtObj.mjOBJ_BODY, name_i)
            if bid_i < 0:
                continue
            ja_i = int(self.sim.model.body_jntadr[bid_i])
            if ja_i < 0:
                continue
            qa_i = int(self.sim.model.jnt_qposadr[ja_i])
            self.sim.data.qpos[qa_i:qa_i + 3] = [-10.0,
                                                  -10.0 + i, 0.5]
            self.sim.data.qpos[qa_i + 3:qa_i + 7] = [1.0, 0, 0, 0]
        mujoco.mj_forward(self.sim.model, self.sim.data)

        for _ in range(40):
            self.sim.step_simulation(render=False)

        self._obj_bid = obj_bid
        self._grasp_q = grasp_q
        if self.verbose:
            obj_now = self.sim.data.xpos[obj_bid].copy()
            grip_now = self.grasp_exec._carry_anchor_xyz(
                self.sim.data).copy()
            d_xy = float(np.hypot(obj_now[0] - grip_now[0],
                                   obj_now[1] - grip_now[1]))
            print(f"[Trial] SIMPLE setup done: "
                  f"chassis=({chassis_xy[0]:.2f},{chassis_xy[1]:.2f}) "
                  f"pocket={pocket_xyz.round(3).tolist()} "
                  f"obj={obj_now.round(3).tolist()} "
                  f"d_xy(obj-grip_centroid)={d_xy*100:.1f}cm")

    def place_initial_state(self):
        obj_name = f"pickup_obj_{self.obj_idx}"
        obj_bid = mujoco.mj_name2id(self.sim.model,
                                     mujoco.mjtObj.mjOBJ_BODY, obj_name)
        if obj_bid < 0:
            raise RuntimeError(f"obj body '{obj_name}' not found")
        jntadr = int(self.sim.model.body_jntadr[obj_bid])
        if jntadr < 0:
            raise RuntimeError(f"obj '{obj_name}' has no joint")
        qadr = int(self.sim.model.jnt_qposadr[jntadr])
        half_h = 0.15
        for g in range(self.sim.model.ngeom):
            if int(self.sim.model.geom_bodyid[g]) == obj_bid:
                half_h = float(self.sim.model.geom_size[g, 1])
                break
        obj_z = half_h + 0.005
        self.sim.data.qpos[qadr:qadr + 3] = [self.obj_xy[0],
                                              self.obj_xy[1], obj_z]
        self.sim.data.qpos[qadr + 3:qadr + 7] = [1.0, 0.0, 0.0, 0.0]

        for i in range(10):
            if i == self.obj_idx:
                continue
            name_i = f"pickup_obj_{i}"
            bid_i = mujoco.mj_name2id(self.sim.model,
                                       mujoco.mjtObj.mjOBJ_BODY, name_i)
            if bid_i < 0:
                continue
            ja_i = int(self.sim.model.body_jntadr[bid_i])
            if ja_i < 0:
                continue
            qa_i = int(self.sim.model.jnt_qposadr[ja_i])
            self.sim.data.qpos[qa_i:qa_i + 3] = [-10.0,
                                                  -10.0 + i, 0.5]
            self.sim.data.qpos[qa_i + 3:qa_i + 7] = [1.0, 0, 0, 0]

        standoff = 0.64
        from math import atan2, cos, sin, pi
        chassis_yaw = np.radians(self.approach_yaw_deg)
        approach_unit = np.array([cos(chassis_yaw), sin(chassis_yaw)])
        chassis_xy = (self.obj_xy[0] - standoff * approach_unit[0],
                      self.obj_xy[1] - standoff * approach_unit[1])
        base_bid = mujoco.mj_name2id(self.sim.model,
                                      mujoco.mjtObj.mjOBJ_BODY, "robot")
        if base_bid < 0:
            base_bid = mujoco.mj_name2id(self.sim.model,
                                          mujoco.mjtObj.mjOBJ_BODY, "base")
        if base_bid >= 0:
            ja_b = int(self.sim.model.body_jntadr[base_bid])
            if ja_b >= 0:
                qa_b = int(self.sim.model.jnt_qposadr[ja_b])
                self.sim.data.qpos[qa_b] = chassis_xy[0]
                self.sim.data.qpos[qa_b + 1] = chassis_xy[1]
                self.sim.data.qpos[qa_b + 3] = cos(chassis_yaw / 2.0)
                self.sim.data.qpos[qa_b + 4] = 0.0
                self.sim.data.qpos[qa_b + 5] = 0.0
                self.sim.data.qpos[qa_b + 6] = sin(chassis_yaw / 2.0)
        self.sim.target_base = np.array([float(chassis_xy[0]),
                                          float(chassis_xy[1]),
                                          float(chassis_yaw)])

        mujoco.mj_forward(self.sim.model, self.sim.data)

        try:
            obj_world_xyz = self.sim.data.xpos[obj_bid].copy()
            obj_radius = None
            for g in range(self.sim.model.ngeom):
                if int(self.sim.model.geom_bodyid[g]) == obj_bid:
                    obj_radius = float(self.sim.model.geom_size[g, 0])
                    break
            print(f"[Trial] IK setup: chassis_xy={tuple(np.round(chassis_xy, 3))}  "
                  f"chassis_yaw={np.degrees(chassis_yaw):.1f}°  "
                  f"obj_world={obj_world_xyz.round(3).tolist()}  "
                  f"obj_radius={obj_radius:.3f}")
            wrist_goal = (0.00, -1.88, 0.80, 0.00)
            _, pre_grasp_pos = compute_grasp_targets(
                np.array(chassis_xy, dtype=float),
                obj_world_xyz,
                obj_radius=obj_radius,
                side_approach=True)
            print(f"[Trial] IK setup: pre_grasp_pos="
                  f"{pre_grasp_pos.round(3).tolist()} (world frame)")
            reset_plan_data_for_ik(self.arm_bridge,
                                    base_xy=np.array(chassis_xy),
                                    base_yaw=chassis_yaw)
            try:
                base_bid_pd = mujoco.mj_name2id(
                    self.arm_bridge.model, mujoco.mjtObj.mjOBJ_BODY, "robot")
                if base_bid_pd < 0:
                    base_bid_pd = mujoco.mj_name2id(
                        self.arm_bridge.model, mujoco.mjtObj.mjOBJ_BODY, "base")
                if base_bid_pd >= 0:
                    pd_xpos = self.arm_bridge.planning_data.xpos[
                        base_bid_pd].copy()
                    print(f"[Trial] IK setup: planning_data chassis "
                          f"world pos={pd_xpos.round(3).tolist()}")
            except Exception:
                pass
            ik_target_body  = "Gripper_Link3_1"
            ik_wrist_weight = (0.10, 3.0, 3.0, 3.0)
            q_pre, actual_pre_target = \
                self.arm_bridge.solve_ik_with_z_lift(
                    pre_grasp_pos,
                    n_seeds=8,
                    wrist_goal=wrist_goal,
                    wrist_weight=ik_wrist_weight,
                    target_body=ik_target_body,
                    max_lift=0.40)
            grasp_q_known = list(q_pre)
            if self.verbose:
                print(f"[Trial] IK solved: q_pre="
                      f"{[round(v, 3) for v in grasp_q_known]}  "
                      f"target={pre_grasp_pos.round(3)}  "
                      f"actual={actual_pre_target.round(3)}")
        except Exception as _e_ik:
            print(f"[Trial] IK failed during state setup: {_e_ik}")
            traceback.print_exc()
            grasp_q_known = [
                0.135, 0.360, 0.305, float(chassis_yaw),
                0.025, -1.88, 0.80, 0.00,
            ]
        for i, q in enumerate(grasp_q_known[:4]):
            self.sim.data.qpos[self.sim.qpos_indices[i]] = q
        wrist_names = ("HandBearingJoint_1", "gripper_z_rotation_1",
                       "gripper_x_rotation_1", "gripper_y_rotation_1")
        for i, name in enumerate(wrist_names):
            jid = mujoco.mj_name2id(self.sim.model,
                                     mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                qa = int(self.sim.model.jnt_qposadr[jid])
                self.sim.data.qpos[qa] = grasp_q_known[4 + i]
        try:
            self.grasp_exec._set_arm_cmd(grasp_q_known)
            if self.verbose:
                print("[Trial] _set_arm_cmd(grasp_q) called — arm + "
                      "wrist ctrls now match grasp pose")
        except Exception as _e:
            print(f"[Trial] WARNING: _set_arm_cmd raised: {_e} "
                  f"(wrist will drift to neutral)")
        for i, q in enumerate([1.20, 1.20, 0.10, 0.0]):
            self.sim.data.qpos[self.sim.qpos_indices[4 + i]] = q
        try:
            for i, q in enumerate([1.20, 1.20, 0.10, 0.0]):
                self.sim.data.ctrl[self.sim.actuator_ids[4 + i]] = q * 100
        except Exception:
            pass
        if self.verbose:
            print("[Trial] place: qpos set; running mj_forward + settle ...")
        mujoco.mj_forward(self.sim.model, self.sim.data)
        for _ in range(60):
            self.sim.step_simulation(render=False)
        try:
            self.grasp_exec._last_valid_pre_grasp_q = list(grasp_q_known)
            if self.verbose:
                print(f"[Trial] pre-populated _last_valid_pre_grasp_q "
                      f"with known good 8-DOF GRASP_Q")
        except Exception as _e:
            print(f"[Trial] WARNING: failed to pre-populate "
                  f"_last_valid_pre_grasp_q: {_e}")
        if self.verbose:
            loc = self.sim.localization()
            obj_xyz = self.sim.data.xpos[obj_bid][:3]
            print(f"[Trial] placed: chassis=({loc[0]:.2f},{loc[1]:.2f}) "
                  f"yaw={np.degrees(loc[2]):.0f}°  "
                  f"obj=({obj_xyz[0]:.2f},{obj_xyz[1]:.2f},{obj_xyz[2]:.3f})  "
                  f"grasp_q[h1,h2,a1]={grasp_q_known[:3]}")


    def run_pickup_simple(self):
        obj_bid = self._obj_bid
        grasp_q = self._grasp_q

        obj_pre = self.sim.data.xpos[obj_bid].copy()
        self.metrics["obj_z_pre_lift"] = float(obj_pre[2])

        self.grasp_exec._held_obj_idx = self.obj_idx
        self.grasp_exec._held_obj_bid = int(obj_bid)
        try:
            jntadr = int(self.sim.model.body_jntadr[obj_bid])
            qpa = int(self.sim.model.jnt_qposadr[jntadr])
            dofadr = int(self.sim.model.jnt_dofadr[jntadr])
            self.grasp_exec._held_obj_qpa = qpa
            self.grasp_exec._held_obj_dofadr = dofadr
        except Exception:
            pass

        self.grasp_exec._side_grip_active = True

        self._start_bg_sim()
        time.sleep(0.1)

        t0 = time.time()
        try:
            CLOSE_POS = 0.20
            if self.verbose:
                print(f"[Trial-SIMPLE] running close stroke ...")
            self.grasp_exec._set_gripper(CLOSE_POS,
                                         hold_seconds=0.60)
            contacts = (self.grasp_exec._last_close_finger_contacts
                          or [False, False, False])
            self.metrics["close_3finger"] = (
                sum(bool(x) for x in contacts) >= 3)
            if self.verbose:
                print(f"[Trial-SIMPLE] close done: contacts={contacts}")

            time.sleep(0.3)
            try:
                live_c = bool(self.grasp_exec._finger_touches_obj(0, obj_bid))
                live_b = bool(self.grasp_exec._finger_touches_obj(1, obj_bid))
                live_a = bool(self.grasp_exec._finger_touches_obj(2, obj_bid))
            except Exception:
                live_c = live_b = live_a = False
            opposing_pinch_ok = live_a and (live_b or live_c)
            self.metrics["verify_opposing"] = opposing_pinch_ok
            self.metrics["verify_passed"] = opposing_pinch_ok
            if self.verbose:
                print(f"[Trial-SIMPLE] verify: a={live_a} b={live_b} "
                      f"c={live_c} opposing_pinch_ok={opposing_pinch_ok}")
            if not opposing_pinch_ok:
                if self.verbose:
                    print(f"[Trial-SIMPLE] verify FAILED — skipping lift")
                self.metrics["trial_duration_s"] = time.time() - t0
                return

            if self.verbose:
                print(f"[Trial-SIMPLE] running lift ...")
            lift_ok = self.grasp_exec._strict_lift_with_retry(
                obj_bid, CLOSE_POS, grasp_q)
            if self.verbose:
                print(f"[Trial-SIMPLE] lift done: ok={lift_ok}")
            self.metrics["slip_retries"] = int(
                getattr(self.grasp_exec,
                          '_strict_retry_count', 0))

            obj_after_lift = self.sim.data.xpos[obj_bid].copy()
            self.metrics["obj_z_post_motion"] = float(obj_after_lift[2])
            time.sleep(2.0)
            obj_hold = self.sim.data.xpos[obj_bid].copy()
            self.metrics["obj_z_after_hold"] = float(obj_hold[2])
            self.metrics["obj_followed_grip"] = (
                self.metrics["obj_z_post_motion"]
                > self.metrics["obj_z_pre_lift"] + 0.05)
            self.metrics["obj_held_2s"] = (
                self.metrics["obj_z_after_hold"]
                > self.metrics["obj_z_post_motion"] - 0.05)
        except Exception as e:
            self.metrics["errors"].append(f"run_pickup_simple raised: {e}")
            if self.verbose:
                traceback.print_exc()
        finally:
            self.metrics["trial_duration_s"] = time.time() - t0
            self._stop_bg_sim()

    def run_pickup(self):
        obj_name = f"pickup_obj_{self.obj_idx}"
        obj_bid = mujoco.mj_name2id(self.sim.model,
                                     mujoco.mjtObj.mjOBJ_BODY, obj_name)
        obj_world = self.sim.data.xpos[obj_bid].copy()
        self.metrics["obj_z_pre_lift"] = float(obj_world[2])

        def _on_complete(success, info=None):
            self._grasp_result["success"] = bool(success)
            self._grasp_result["info"] = info
            self._grasp_done_event.set()

        if self.verbose:
            print(f"[Trial] calling grasp_exec.pick(obj_idx={self.obj_idx}, "
                  f"obj_world={obj_world.round(3)}, is_local_retry=True) ...")
        self._start_bg_sim()
        t0 = time.time()
        self.grasp_exec.pick(self.obj_idx, obj_world,
                              on_complete=_on_complete,
                              is_local_retry=True)
        completed = self._grasp_done_event.wait(timeout=60.0)
        self.metrics["trial_duration_s"] = time.time() - t0
        if not completed:
            self.metrics["errors"].append("TIMEOUT: grasp_exec.pick did not "
                                          "complete within 60s")
            return
        self.metrics["verify_passed"] = bool(
            self._grasp_result.get("success"))

        obj_after = self.sim.data.xpos[obj_bid][:3]
        self.metrics["obj_z_post_motion"] = float(obj_after[2])
        time.sleep(2.0)
        obj_after_hold = self.sim.data.xpos[obj_bid][:3]
        self.metrics["obj_z_after_hold"] = float(obj_after_hold[2])
        if (self.metrics["obj_z_pre_lift"] is not None
                and self.metrics["obj_z_post_motion"] is not None):
            self.metrics["obj_followed_grip"] = (
                self.metrics["obj_z_post_motion"]
                > self.metrics["obj_z_pre_lift"] + 0.05)
        if (self.metrics["obj_z_post_motion"] is not None
                and self.metrics["obj_z_after_hold"] is not None):
            self.metrics["obj_held_2s"] = (
                self.metrics["obj_z_after_hold"]
                > self.metrics["obj_z_post_motion"] - 0.05)
        try:
            self.metrics["slip_retries"] = int(
                getattr(self.grasp_exec, '_strict_retry_count', 0))
        except Exception:
            self.metrics["slip_retries"] = 0



@contextmanager
def apply_runtime_params(overrides):
    originals = {}
    for k, v in overrides.items():
        if k not in TUNABLE_RUNTIME:
            print(f"[apply] WARNING: unknown runtime param '{k}', skipping")
            continue
        if not hasattr(gx, k):
            print(f"[apply] WARNING: grasp_executor has no attr '{k}', skipping")
            continue
        originals[k] = getattr(gx, k)
        setattr(gx, k, v)
    try:
        yield
    finally:
        for k, v in originals.items():
            setattr(gx, k, v)



def score_trial(metrics):
    s = 0.0
    if metrics.get("close_3finger"):
        s += 1.0
    if metrics.get("verify_passed"):
        s += 2.0
    if metrics.get("obj_followed_grip"):
        s += 5.0
    if metrics.get("obj_held_2s"):
        s += 5.0
    s -= metrics.get("slip_retries", 0) * 1.0
    fmax = metrics.get("force_stop_max_n", 0.0)
    if fmax > 500.0:
        s -= min(5.0, (fmax - 500.0) / 200.0)
    return float(s)



def _run_one_attempt(xml_path, params, obj_xy, approach_yaw_deg,
                       verbose, sim_context, drift=None):
    trial = Trial(xml_path, obj_idx=0, obj_xy=obj_xy,
                   approach_yaw_deg=approach_yaw_deg, verbose=verbose,
                   sim_context=sim_context, drift=drift)
    with apply_runtime_params(params):
        try:
            trial.setup()
            trial.place_initial_state_simple()
            trial.run_pickup_simple()
        except Exception as e:
            trial.metrics["errors"].append(f"trial raised: {e}")
            if verbose:
                traceback.print_exc()
        finally:
            trial.teardown()
    sc = score_trial(trial.metrics)
    return sc, trial.metrics


def run_one_trial(xml_path, params=None, obj_xy=None,
                   approach_yaw_deg=None, verbose=True,
                   sim_context=None, n_runs=1, drift=None):
    if params is None:
        params = {}
    if obj_xy is None:
        obj_xy = KNOWN_OBJ_XY_LIST[0]
    n_runs = max(1, int(n_runs))
    run_results = []
    best_score = -float("inf")
    best_metrics = None
    for r in range(n_runs):
        sc, m = _run_one_attempt(xml_path, params, obj_xy,
                                    approach_yaw_deg, verbose,
                                    sim_context, drift=drift)
        run_results.append((sc, m.get("verify_passed"),
                              m.get("obj_followed_grip"),
                              m.get("obj_held_2s")))
        if sc > best_score:
            best_score = sc
            best_metrics = m
        if verbose and n_runs > 1:
            print(f"[run_one_trial] attempt {r+1}/{n_runs}: "
                  f"score={sc:.1f} verify={m.get('verify_passed')} "
                  f"followed={m.get('obj_followed_grip')} "
                  f"held={m.get('obj_held_2s')}")
    if best_metrics is None:
        best_metrics = {}
    best_metrics["runs"] = run_results
    return best_score, best_metrics


def find_best_approach_yaw(xml_path, obj_xy, yaws_deg=None,
                              verbose=True, sim_context=None):
    if yaws_deg is None:
        yaws_deg = list(range(-180, 180, 30))
    own_context = sim_context is None
    if own_context:
        sim_context = SimContext(xml_path, verbose=verbose)
    try:
        results = []
        for yaw in yaws_deg:
            if verbose:
                print(f"\n[Find-Pose] testing approach yaw={yaw:+.0f}° ...")
            sc, m = run_one_trial(xml_path, params={}, obj_xy=obj_xy,
                                    approach_yaw_deg=yaw, verbose=False,
                                    sim_context=sim_context)
            results.append((yaw, sc, m))
            if verbose:
                print(f"[Find-Pose] yaw={yaw:+.0f}° score={sc:.1f}  "
                      f"verify={m.get('verify_passed')}  "
                      f"followed={m.get('obj_followed_grip')}  "
                      f"held={m.get('obj_held_2s')}  "
                      f"dur={m.get('trial_duration_s', 0):.1f}s")
        results.sort(key=lambda x: x[1], reverse=True)
        best_yaw, best_score, best_m = results[0]
        return best_yaw, best_score, results
    finally:
        if own_context:
            sim_context.teardown()



def find_best_obj_position_and_yaw(xml_path,
                                      obj_xy_candidates=None,
                                      yaws_deg=None,
                                      score_threshold=10.0,
                                      out_csv=None,
                                      verbose=True,
                                      sim_context=None,
                                      n_runs_per_config=2):
    if obj_xy_candidates is None:
        obj_xy_candidates = [
            (3.0, -6.5),
            (4.0, -6.5),
            (5.0, -6.5),
            (3.5, -7.0),
            (4.5, -7.0),
            (5.5, -7.0),
            (3.5, -6.0),
            (4.5, -6.0),
            (5.5, -6.0),
        ]
    if yaws_deg is None:
        yaws_deg = [-135, -90, -45, 0, 45, 90, 135, 180]
    total_trials = len(obj_xy_candidates) * len(yaws_deg)
    if verbose:
        print(f"[Phase3] sweep: {len(obj_xy_candidates)} obj positions "
              f"× {len(yaws_deg)} yaws = {total_trials} trials  "
              f"(early-stop at score ≥ {score_threshold})")
    results = []
    writer = None
    fcsv = None
    if out_csv is not None:
        fcsv = open(out_csv, "w", newline="")
        writer = csv.writer(fcsv)
        writer.writerow(["trial", "obj_x", "obj_y", "yaw_deg",
                            "score", "verify_passed", "obj_followed",
                            "obj_held_2s", "slip_retries",
                            "duration_s"])
    idx = 0
    best_so_far = None
    REBUILD_EVERY = 20
    own_context = sim_context is None
    if own_context:
        sim_context = SimContext(xml_path, verbose=verbose)
    try:
        for obj_xy in obj_xy_candidates:
            for yaw in yaws_deg:
                idx += 1
                if own_context and idx > 1 and (idx - 1) % REBUILD_EVERY == 0:
                    if verbose:
                        print(f"\n[Phase3] PERIODIC REBUILD at trial {idx} "
                              f"(every {REBUILD_EVERY} trials) — "
                              f"resetting MuJoCo from scratch to avoid "
                              f"state drift")
                    sim_context.teardown()
                    sim_context = SimContext(xml_path, verbose=False)
                if verbose:
                    print(f"\n[Phase3] trial {idx}/{total_trials}: "
                          f"obj=({obj_xy[0]:.2f},{obj_xy[1]:.2f}) "
                          f"yaw={yaw:+.0f}° ...")
                sc, m = run_one_trial(xml_path, params={},
                                        obj_xy=obj_xy,
                                        approach_yaw_deg=yaw,
                                        verbose=False,
                                        sim_context=sim_context,
                                        n_runs=n_runs_per_config)
                results.append((obj_xy, yaw, sc, m))
                if writer is not None:
                    writer.writerow([idx, f"{obj_xy[0]:.2f}",
                                      f"{obj_xy[1]:.2f}", yaw,
                                      f"{sc:.2f}",
                                      m.get("verify_passed"),
                                      m.get("obj_followed_grip"),
                                      m.get("obj_held_2s"),
                                      m.get("slip_retries"),
                                      f"{m.get('trial_duration_s', 0):.2f}"])
                    fcsv.flush()
                if verbose:
                    print(f"[Phase3] trial {idx} score={sc:.1f}  "
                          f"verify={m.get('verify_passed')}  "
                          f"followed={m.get('obj_followed_grip')}  "
                          f"held={m.get('obj_held_2s')}  "
                          f"dur={m.get('trial_duration_s', 0):.1f}s")
                if best_so_far is None or sc > best_so_far[2]:
                    best_so_far = (obj_xy, yaw, sc, m)
                if sc >= score_threshold:
                    if verbose:
                        print(f"\n[Phase3] EARLY-STOP: trial {idx} hit "
                              f"score {sc:.1f} ≥ threshold "
                              f"{score_threshold:.1f}  "
                              f"obj=({obj_xy[0]:.2f},{obj_xy[1]:.2f}) "
                              f"yaw={yaw:+.0f}°")
                    return obj_xy, yaw, sc, results
    finally:
        if fcsv is not None:
            fcsv.close()
        if own_context:
            sim_context.teardown()
    results_sorted = sorted(
        results, key=lambda r: r[2], reverse=True)
    best_obj_xy, best_yaw, best_score, _ = results_sorted[0]
    return best_obj_xy, best_yaw, best_score, results


def grid_search(xml_path, out_csv, approach_yaw_deg=None,
                  sim_context=None, n_runs_per_config=2):
    own_context = sim_context is None
    if own_context:
        sim_context = SimContext(xml_path, verbose=True)
    grid = {
        "LIFT_TIGHTEN_PRELIFT_RAD":     [0.03, 0.05, 0.10],
        "LIFT_RETIGHTEN_PER_STEP_RAD":  [0.005, 0.012, 0.030],
        "LIFT_STEPS_MULTIPLIER":        [1, 2, 4],
        "STRICT_GRIP_SAFETY":           [1.80, 2.50],
    }
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    print(f"[grid] {len(combos)} combos at approach_yaw="
          f"{approach_yaw_deg if approach_yaw_deg is not None else 'default'}")
    results = []
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trial"] + keys + ["score",
                                          "verify_passed",
                                          "obj_followed",
                                          "obj_held_2s",
                                          "slip_retries",
                                          "duration_s"])
        REBUILD_EVERY = 20
        for i, combo in enumerate(combos):
            params = dict(zip(keys, combo))
            if own_context and i > 0 and i % REBUILD_EVERY == 0:
                print(f"\n[grid] PERIODIC REBUILD at trial {i+1} "
                      f"(every {REBUILD_EVERY}) — fresh MuJoCo state")
                sim_context.teardown()
                sim_context = SimContext(xml_path, verbose=False)
            print(f"\n[grid] trial {i+1}/{len(combos)}: {params}")
            sc, m = run_one_trial(xml_path, params,
                                    approach_yaw_deg=approach_yaw_deg,
                                    verbose=False,
                                    sim_context=sim_context,
                                    n_runs=n_runs_per_config)
            results.append((sc, params, m))
            w.writerow([i + 1] + list(combo) + [
                f"{sc:.2f}",
                m.get("verify_passed"),
                m.get("obj_followed_grip"),
                m.get("obj_held_2s"),
                m.get("slip_retries"),
                f"{m.get('trial_duration_s', 0):.2f}",
            ])
            f.flush()
            print(f"[grid] trial {i+1} score={sc:.1f}  "
                  f"verify={m.get('verify_passed')}  "
                  f"held={m.get('obj_held_2s')}")
    results.sort(key=lambda r: r[0], reverse=True)
    print("\n=== TOP 5 ===")
    for sc, params, m in results[:5]:
        print(f"  score={sc:.2f}  params={params}  "
              f"verify={m.get('verify_passed')}  "
              f"held_2s={m.get('obj_held_2s')}")
    if results:
        best_sc, best_params, best_m = results[0]
        print(f"\n=== BEST PARAMS (apply to grasp_executor.py) ===")
        for k, v in best_params.items():
            print(f"  {k} = {v}")
    if own_context:
        sim_context.teardown()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--xml",
                   default=os.path.join(SRC, "env/market_world_m1.xml"))
    p.add_argument("--single", action="store_true",
                   help="Run one trial with current params and report metrics")
    p.add_argument("--find-pose", action="store_true",
                   help="Sweep approach yaws to find a baseline pose that works")
    p.add_argument("--find-baseline", action="store_true",
                   help="Phase 3: 2D sweep over (obj position × yaw) "
                        "to find a baseline geometry where close+verify "
                        "succeeds.  Required before --grid is productive.")
    p.add_argument("--auto-tune", action="store_true",
                   help="Phase 3+: full pipeline — find baseline, then "
                        "grid-search physics params at that baseline.")
    p.add_argument("--grid", action="store_true",
                   help="Grid search over high-impact runtime params")
    p.add_argument("--obj-xy", type=float, nargs=2, default=None,
                   help="obj XY position (default: first in KNOWN_OBJ_XY_LIST)")
    p.add_argument("--approach-yaw", type=float, default=None,
                   help="chassis approach yaw in degrees (world frame). "
                        "Default 122° (reference success geometry).")
    p.add_argument("--out-csv", default="tune_grip_lift_results.csv")
    p.add_argument("--wz-residual", type=float, default=0.0,
                   help="Inject wrist-Z residual drift (rad). play_m1 "
                        "wrist-settle typically times out at ~0.13 rad.")
    p.add_argument("--wx-residual", type=float, default=0.0,
                   help="Inject wrist-X residual drift (rad).")
    p.add_argument("--chassis-dx", type=float, default=0.0,
                   help="Inject chassis X offset (m). play_m1 ALIGN snap "
                        "typically leaves 5-10cm chassis offset.")
    p.add_argument("--chassis-dy", type=float, default=0.0,
                   help="Inject chassis Y offset (m).")
    p.add_argument("--chassis-dyaw", type=float, default=0.0,
                   help="Inject chassis yaw drift (rad).")
    args = p.parse_args()
    drift = {
        "wz":           args.wz_residual,
        "wx":           args.wx_residual,
        "chassis_dx":   args.chassis_dx,
        "chassis_dy":   args.chassis_dy,
        "chassis_dyaw": args.chassis_dyaw,
    }
    drift_active = any(v != 0.0 for v in drift.values())

    if not (args.single or args.grid or args.find_pose
            or args.find_baseline or args.auto_tune):
        args.single = True

    obj_xy = tuple(args.obj_xy) if args.obj_xy else KNOWN_OBJ_XY_LIST[0]

    if args.find_pose:
        print(f"=== Phase 2: find best approach yaw ===")
        print(f"xml: {args.xml}  obj_xy: {obj_xy}")
        best_yaw, best_score, all_results = find_best_approach_yaw(
            args.xml, obj_xy, verbose=True)
        print(f"\n=== Best: yaw={best_yaw:+.0f}° score={best_score:.1f} ===")
        print(f"Pass this to --grid via --approach-yaw {best_yaw}")
        for yaw, sc, m in sorted(all_results, key=lambda r: r[1], reverse=True)[:5]:
            print(f"  yaw={yaw:+.0f}°  score={sc:.1f}  "
                  f"verify={m.get('verify_passed')}  "
                  f"held={m.get('obj_held_2s')}")
        return

    if args.single:
        print(f"=== Single trial ===")
        print(f"xml: {args.xml}")
        print(f"obj_xy: {obj_xy}")
        print(f"approach_yaw: "
              f"{args.approach_yaw if args.approach_yaw is not None else 'default 122°'}")
        if drift_active:
            print(f"drift: {drift}")
        sc, m = run_one_trial(args.xml, params={}, obj_xy=obj_xy,
                                approach_yaw_deg=args.approach_yaw,
                                verbose=True,
                                drift=drift if drift_active else None)
        print("\n=== Metrics ===")
        for k, v in m.items():
            print(f"  {k}: {v}")
        print(f"\n=== Score: {sc:.2f} ===")
        return

    if args.find_baseline:
        print(f"=== Phase 3: find baseline (obj position × yaw sweep) ===")
        print(f"xml: {args.xml}")
        baseline_csv = args.out_csv.replace(".csv", "_baseline.csv")
        best_obj_xy, best_yaw, best_score, all_results = \
            find_best_obj_position_and_yaw(
                args.xml,
                score_threshold=10.0,
                out_csv=baseline_csv,
                verbose=True)
        print(f"\n=== BEST BASELINE ===")
        print(f"  obj_xy = ({best_obj_xy[0]:.2f}, {best_obj_xy[1]:.2f})")
        print(f"  approach_yaw = {best_yaw:+.0f}°")
        print(f"  score = {best_score:.1f}")
        print(f"\nTop 5 baselines:")
        for o, y, sc, m in sorted(all_results, key=lambda r: r[2],
                                    reverse=True)[:5]:
            print(f"  obj=({o[0]:.2f},{o[1]:.2f}) yaw={y:+.0f}° "
                  f"score={sc:.1f} verify={m.get('verify_passed')} "
                  f"held={m.get('obj_held_2s')}")
        print(f"\nUse the best with --grid:")
        print(f"  python3 tools/tune_grip_lift.py --grid "
              f"--obj-xy {best_obj_xy[0]:.2f} {best_obj_xy[1]:.2f} "
              f"--approach-yaw {best_yaw}")
        print(f"\nBaseline CSV: {baseline_csv}")
        return

    if args.auto_tune:
        print(f"=== Auto-tune: find baseline + grid search physics ===")
        shared_ctx = SimContext(args.xml, verbose=True)
        try:
            baseline_csv = args.out_csv.replace(".csv", "_baseline.csv")
            print(f"\n--- Step 1/2: finding baseline geometry ---")
            best_obj_xy, best_yaw, best_score, _ = \
                find_best_obj_position_and_yaw(
                    args.xml,
                    score_threshold=10.0,
                    out_csv=baseline_csv,
                    verbose=True,
                    sim_context=shared_ctx)
            print(f"\n--- Step 1 complete: baseline score = {best_score:.1f} ---")
            if best_score < 6.0:
                print(f"\n!!! WARNING: best baseline score {best_score:.1f} "
                      f"< 6.0 — no geometry produced a verify pass.  "
                      f"Physics tuning at this baseline will not be "
                      f"productive.  Consider tuning obj geometry instead "
                      f"(OBJ_RADIUS_RANGE / OBJ_HEIGHT_RANGE) before "
                      f"re-running --auto-tune.")
                return
            print(f"\n--- Step 2/2: grid search at baseline "
                  f"obj=({best_obj_xy[0]:.2f},{best_obj_xy[1]:.2f}) "
                  f"yaw={best_yaw:+.0f}° ---")
            KNOWN_OBJ_XY_LIST[0] = tuple(best_obj_xy)
            grid_search(args.xml, args.out_csv,
                         approach_yaw_deg=best_yaw,
                         sim_context=shared_ctx)
        finally:
            shared_ctx.teardown()
        return

    if args.grid:
        print(f"=== Grid search ===")
        print(f"obj_xy: {obj_xy}")
        print(f"approach_yaw: "
              f"{args.approach_yaw if args.approach_yaw is not None else 'default 122°'}")
        KNOWN_OBJ_XY_LIST[0] = obj_xy
        grid_search(args.xml, args.out_csv,
                     approach_yaw_deg=args.approach_yaw)
        return


if __name__ == "__main__":
    main()
