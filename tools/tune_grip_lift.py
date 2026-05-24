"""tools/tune_grip_lift.py — physics parameter tuning harness for the
STRICT-mode grip + lift pipeline.

PHASE 1 (this file): minimal harness that runs ONE trial end-to-end —
spawns obj, sets chassis + arm via real IK, calls grasp_exec.pick
with is_local_retry=True (skips arm motion), captures metrics.
Validated against play_m1's actual physics paths.

KNOWN LIMITATION: the harness's hardcoded chassis approach angle
(SE of obj, ~120° approaching from obj's SE) is just ONE specific
geometry.  Due to the 1+2 DELTO gripper's structural asymmetry,
not every chassis-obj relative pose produces a 3/3 contact close.
play_m1's orbit retry mechanism searches over multiple yaws to find
working geometries; this harness does NOT (yet).  If a trial shows
`asymmetric reach` rejection at the default pose, adjust `--obj-xy`
or the `approach_unit` constant to a known good geometry (recorded
from a previous successful play_m1 run log).

For productive PHYSICS tuning, you need a baseline pose where
default params produce a successful close.  Once you have that:
  1. Run --single → confirm baseline succeeds with score ≥ 13
  2. Then --grid varies params around the baseline → CSV of results
  3. Apply best params back to play_m1 / grasp_executor constants

PHASE 2 (not yet implemented): automated search for a good baseline
chassis pose (try multiple yaws like orbit retry), then physics
grid search on top.

GOAL: skip the expensive nav / scan / orbit-retry layers and run JUST
the close-stroke + verify + lift + slip-monitor stages, with overrideable
parameters.  Capture per-trial metrics and score the trial.  Loop over
parameter variations to find the best combination.  Apply best back to
play_m1.

ARCHITECTURE (matches play_m1 physics faithfully):

  ParallelRobot(xml, run_mode="headless")     # same sim class as play_m1
       │
  MORPHBridge(xml, arm=1, use_calibration=...) # same IK / LUT
       │
  GraspExecutor(sim, arm_bridge)               # same close/verify/lift code
       │
  manually place obj + chassis + arm at known-good grasp_q
       │
  grasp_exec.pick(idx, obj_world, is_local_retry=True)  ← skips arm motion;
                                                          runs close+verify+lift
       │
  capture metrics → score → write CSV

The "known good" starting state comes from a reference successful
pickup (orbit +35°): arm at GRASP_Q [0.135, 0.36, 0.30, -0.34, 0.03,
-1.88, 0.80, 0.00], chassis at obj-standoff (0.55-0.60 m).

USAGE:

    # Single trial with current params (Step 1+2 — minimal harness)
    python3 tools/tune_grip_lift.py --single

    # Grid search over parameters (Step 3 — search loop)
    python3 tools/tune_grip_lift.py --grid

    # Random search with N trials
    python3 tools/tune_grip_lift.py --random --n 50
"""

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

# Imports from the live codebase — same objects play_m1 uses, so any
# tuned param transfers cleanly.
from simulations.morph_i_free_move import ParallelRobot
from navigation.arm_planner import MORPHBridge
from navigation import grasp_executor as gx
from navigation.grasp_executor import (GraspExecutor,
                                        compute_grasp_targets,
                                        reset_plan_data_for_ik)

# Force STRICT mode for the entire harness — this is the mode we're
# tuning. play_m1 sets these via --fast-pickup 1; we set them
# directly at module-import time so all subsequent code paths see
# STRICT_PICKUP_MODE=True (including the is_local_retry skip-IK
# condition).
gx.FAST_PICKUP_MODE   = False
gx.STRICT_PICKUP_MODE = True
print(f"[Tuning] STRICT_PICKUP_MODE forced to True "
      f"(FAST_PICKUP_MODE={gx.FAST_PICKUP_MODE})")


# ─── Known good initial state ─────────────────────────────────────────────
# From successful pickup at +35° orbit (the first end-to-end
# verify-OK pickup). Used as the starting pose for all tuning trials.
# Arm joint angles (8-DOF: h1, h2, a1, th, hb, wz, wx, wy). GRASP_Q
# is read at runtime from grasp_executor — we use that directly so
# the harness tracks any future tweaks.
# obj position: spawn at a fixed XY where the gripper can reach.
KNOWN_OBJ_XY_LIST = [
    # Roughly where obj_0 spawned in successful runs. All are inside
    # the SPAWN_ZONE so the validator is happy.
    (5.0, -6.5),
    (3.5, -6.5),
    (4.0, -7.0),
    (5.5, -7.0),
]
KNOWN_OBJ_Z = 0.15   # cylinder half-height: obj sits at z = half_h on floor


# ─── Tunable parameter registry ──────────────────────────────────────────
# Each entry: name, where to set, default, candidate values.
# "runtime": setattr on grasp_executor module → effective immediately
# "xml": rewrite XML and reload model → slower (~1-2 s per change)
# For Step 1+2 we ONLY use runtime params. XML params come in Step 3+.

TUNABLE_RUNTIME = {
    # Force-stop threshold computation
    "STRICT_GRIP_SAFETY":          {"default": 1.80, "candidates": [1.50, 1.80, 2.20, 2.60, 3.00]},
    "STRICT_FRICTION_MU":          {"default": 0.70, "candidates": [0.50, 0.70, 1.00]},
    "STRICT_FORCE_STOP_STABLE_TICKS": {"default": 1, "candidates": [1, 2]},
    # grip-retention tuning constants (now exposed)
    "LIFT_TIGHTEN_PREVERIFY_RAD":   {"default": 0.05, "candidates": [0.03, 0.05, 0.10, 0.15]},
    "LIFT_TIGHTEN_PRELIFT_RAD":     {"default": 0.05, "candidates": [0.03, 0.05, 0.10, 0.15]},
    "LIFT_TIGHTEN_POSTLIFT_RAD":    {"default": 0.03, "candidates": [0.01, 0.03, 0.06, 0.10]},
    "LIFT_RETIGHTEN_PER_STEP_RAD": {"default": 0.012, "candidates": [0.005, 0.012, 0.020, 0.030]},
    "LIFT_RETIGHTEN_INTERVAL_STEPS": {"default": 5, "candidates": [3, 5, 10]},
    "LIFT_STEPS_MULTIPLIER":        {"default": 2, "candidates": [1, 2, 4]},
    "LIFT_PER_STEP_SETTLE_MULTIPLIER": {"default": 1.2, "candidates": [1.0, 1.2, 2.0]},
    # Slip detection
    "STRICT_SLIP_DISP_THRESH":     {"default": 0.012, "candidates": [0.008, 0.012, 0.020]},
    "STRICT_SLIP_VEL_THRESH":      {"default": 0.04,  "candidates": [0.03, 0.04, 0.06]},
    "STRICT_LIFT_OBSERVE_S":       {"default": 0.80,  "candidates": [0.60, 0.80, 1.20]},
    # Retry counts (we usually want 0 for tuning so failures fail-fast)
    "STRICT_RETRY_MAX":            {"default": 2, "candidates": [0, 1, 2]},
}


# ─── Persistent sim context ────────────────────────────────
# SimContext builds sim / arm_bridge / grasp_exec ONCE and trials
# reuse it, resetting state between runs.  Per-trial overhead drops
# from ~30-60 s (full rebuild) to <1 s of state reset — important for
# sweeps with 70+ trials.

class SimContext:
    """Holds the sim, arm_bridge, and grasp_exec persistently across
    multiple trials.  Constructed once per harness run; reset between
    trials via reset_state().

    State that needs resetting between trials:
    - sim.data.qpos / qvel / ctrl (via mj_resetData + keyframe reset)
    - sim.target_base
    - sim._pin_callbacks
    - All GraspExecutor stateful fields (last_grasp_failure_info,
      _strict_force_multiplier, _last_valid_pre_grasp_q, etc.)
    - The background sim thread (stop during reset, restart for trial)
    """

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
        """Reset sim + executor state to a clean per-trial baseline.
        Bg thread MUST be stopped before calling this.

        Aggressive reset to ensure determinism between trials — clears
        MuJoCo's contact cache, warm-start solution, equality constraint
        state, and other derived data so trial N produces the same
        result as trial 1 for identical input."""
        # 1. Cancel any in-flight grasp operation FIRST.
        try:
            self.grasp_exec._cancel = True
        except Exception:
            pass

        # 2. Clear MuJoCo's data structure completely. mj_resetData
        # zeros qpos/qvel/ctrl/qacc/qfrc_*/contact[]/efc_*/etc. This
        # is the "scorched earth" reset.
        try:
            mujoco.mj_resetData(self.sim.model, self.sim.data)
        except Exception as e:
            print(f"[SimContext] mj_resetData failed: {e}")

        # 3. Restore the home keyframe (qpos/qvel/ctrl/time). After
        # mj_resetData above, these are zero; the keyframe restores
        # them to the sim's startup values.
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

        # 4. ALSO reset arm_bridge.planning_data so any IK side-effects
        # from prior trial don't bleed into the next IK call.
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

        # 5. Reset sim PID controller state via the same reset path
        # the sim itself uses (zeros base_integral_*, etc.).
        try:
            self.sim.reset("home")
        except Exception as e:
            print(f"[SimContext] sim.reset 'home' failed: {e}")

        # 6. Clear executor state. _clear_held_state is idempotent
        # and handles the held-object cleanup.
        try:
            self.grasp_exec._clear_held_state()
        except Exception:
            pass
        # Clear non-held-object state fields that _clear_held_state
        # doesn't touch.
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

        # 7. Clear pin callbacks attached by prior trial.
        try:
            with self.sim._pin_lock:
                self.sim._pin_callbacks.clear()
        except Exception:
            pass

        # 8. Re-pump forward kinematics so xpos / xmat reflect the
        # reset state and the next trial reads correct values.
        mujoco.mj_forward(self.sim.model, self.sim.data)

        # 9. Reset chassis target to wherever localization reports
        # the chassis is (just-reset position). Without this the
        # chassis PID may chase a stale target.
        try:
            self.sim.target_base = self.sim.localization()
        except Exception:
            pass

        # 10. Track how many trials this context has handled for
        # the periodic-rebuild safety in the search functions.
        self.trial_count = getattr(self, 'trial_count', 0) + 1

    def teardown(self):
        self.stop_bg_sim()


# ─── Trial runner ────────────────────────────────────────────────────────

class Trial:
    """One pickup attempt: sets up, runs close+verify+lift, returns metrics.

    Reuses the EXACT same sim + executor classes as play_m1 so tuned
    params transfer.  Skips nav/scan/IK by hardcoding chassis + arm
    qpos to a known good pose, then calls grasp_exec.pick with
    is_local_retry=True (which skips the arm motion phases).

    Accepts an optional `sim_context`.  If provided, reuses that
    persistent context (state-reset between trials).  If None,
    constructs a fresh context (legacy slow path).
    """

    def __init__(self, xml_path, obj_idx=0, obj_xy=(5.0, -6.5),
                 approach_yaw_deg=None, verbose=True,
                 sim_context=None, drift=None):
        self.xml_path = xml_path
        self.obj_idx = obj_idx
        self.obj_xy = obj_xy
        # approach_yaw_deg: angle (in degrees, world frame) from
        # which chassis approaches obj. None → default 122° (SE
        # approach matching success). Used by the
        # find-pose sweep to try multiple yaws.
        self.approach_yaw_deg = (approach_yaw_deg
                                  if approach_yaw_deg is not None
                                  else 122.6)
        self.verbose = verbose
        self.sim_context = sim_context  #  optional shared context
        # drift injection. Simulates play_m1's runtime
        # state at close-time (wrist-settle residual ~0.13 rad on wz,
        # chassis off-canonical by 5-10 cm from ALIGN snap + push
        # quantization). Tuning under drift gives params that
        # transfer to play_m1's real geometry, not the canonical pose.
        # Format: {"wz": 0.13, "chassis_dx": 0.05, "chassis_dy": 0.0,
        #          "chassis_dyaw": 0.0}. None → no drift (legacy).
        self.drift = drift or {}

        # If using shared context, reuse its sim/bridge/exec
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

        # Captured trial metrics
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

    # ── Setup ────────────────────────────────────────────────────────

    def setup(self):
        if self.sim_context is not None:
            # SHARED context path (fast): reset state of existing sim.
            # bg thread MUST be stopped during state reset to avoid
            # MuJoCo data races.
            if self.verbose:
                print("[Trial] reusing shared SimContext (fast path)")
            self.sim_context.stop_bg_sim()
            self.sim_context.reset_state()
            return
        # LEGACY path (slow): construct fresh sim each trial.
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
        # NOTE: do NOT start the background sim thread yet. During
        # state setup (place_initial_state), we drive the sim
        # synchronously to avoid contention. The bg thread starts
        # JUST BEFORE calling grasp_exec.pick, since pick uses
        # time.sleep() and expects the sim to advance in parallel.

    def _sim_loop(self):
        while self._sim_running:
            try:
                self.sim.step_simulation(render=False)
                # Tiny sleep to avoid 100% CPU spin and let other
                # threads (the main thread driving grasp_exec) get
                # GIL ticks. MuJoCo step is fast (~1 ms at dt=0.002
                # × 10 substeps). 10ms cadence is fine for ~50 Hz
                # control loop in grasp_exec.
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
        # Brief warm-up so the bg thread is actually stepping before
        # we hand control to grasp_exec.
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
        # When using a shared context, DON'T teardown the context's
        # sim/bridge/exec — they belong to the caller. Just stop the
        # bg sim and let the next trial reuse them.
        self._stop_bg_sim()
        # Legacy path: nothing else to clean up; objects fall out of
        # scope when Trial is GC'd.

    # ── Place obj + chassis + arm at known good pose ─────────────────

    def place_initial_state_simple(self):
        """SIMPLIFIED scene.  Skip alignment / IK / closed-loop /
        chassis push entirely.  Just:
          1. Place chassis at a fixed pose (any reasonable spot).
          2. Set arm to GRASP_Q (the canonical low grasp config).
          3. Compute where the gripper palm-pocket ends up in world.
          4. Place obj DIRECTLY at that palm-pocket position.
          5. Settle physics briefly.
        After this, the obj is between the gripper fingers ready to
        close — no asymmetric reach to worry about.  Trial then just
        runs close + verify + lift via the executor's primitives
        (bypassing _pick_run's gate/align machinery).
        """
        from math import cos, sin, atan2
        # 1. Park chassis at a fixed clear spot (no need to be near
        #    obj-spawn area — chassis pose doesn't affect close/lift
        #    physics, only arm pose does).
        # optional drift on chassis pose. Mirrors
        # play_m1's runtime state where chassis ends 5-10 cm off
        # canonical due to ALIGN closed-loop snap + push quantization.
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

        # 2. Set arm to a known-good GRASP_Q (the low-side-grip pose
        #    play_m1 uses at the moment of close stroke). These values
        #    come from observed successful play_m1 runs.
        # Optional wrist drift. play_m1's wrist-settle routinely times
        # out with wz residual ~0.13 rad — injecting that residual here
        # lets harness tuning find params robust to the actual at-close
        # arm pose play_m1 produces.
        wz_drift = float(self.drift.get("wz", 0.0))
        wx_drift = float(self.drift.get("wx", 0.0))
        if self.verbose and (wz_drift or wx_drift):
            print(f"[Trial-DRIFT] wrist offset: "
                  f"wz={wz_drift:+.3f}rad wx={wx_drift:+.3f}rad")
        grasp_q = [
            0.135,   # h1
            0.360,   # h2
            0.305,   # a1
            float(chassis_yaw),   # th (base joint)
            0.025,   # HandBearing
           -1.88 + wz_drift,    # WristZ (+ optional residual drift)
            0.80 + wx_drift,    # WristX (+ optional residual drift)
            0.00,    # WristY
        ]
        # Apply via qpos AND ctrl (set_arm_cmd writes both).
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
        # ARM2 parked
        for i, q in enumerate([1.20, 1.20, 0.10, 0.0]):
            self.sim.data.qpos[self.sim.qpos_indices[4 + i]] = q
        try:
            self.sim.data.ctrl[self.sim.actuator_ids[4 + i]] = q * 100
        except Exception:
            pass
        # _set_arm_cmd handles arm ctrl + wrist ctrl with PD comp
        try:
            self.grasp_exec._set_arm_cmd(grasp_q)
        except Exception as _e:
            print(f"[Trial] WARNING _set_arm_cmd: {_e}")

        # Open the gripper (fingers ready to receive obj).
        # Use direct ctrl writes for the finger joints — avoid running
        # _set_gripper(OPEN) here because that takes ~0.4 s of sim.
        try:
            gids = self.sim.gripper_ids_left
            open_curl = self.grasp_exec._curl_targets(
                gx.GRIPPER_OPEN_POS)  # 9 finger joints
            with self.sim._target_lock:
                for j_idx, val in enumerate(open_curl):
                    if j_idx < 9 and j_idx < len(gids):
                        self.sim.data.ctrl[gids[j_idx]] = float(val)
                        # also write qpos so obj-placement sees fingers
                        # open before settling
                        addrs = self.grasp_exec._ensure_finger_joint_qposadrs()
                        if addrs and j_idx < len(addrs) and addrs[j_idx] >= 0:
                            self.sim.data.qpos[addrs[j_idx]] = float(val)
        except Exception as _e:
            print(f"[Trial] WARNING finger open: {_e}")

        mujoco.mj_forward(self.sim.model, self.sim.data)

        # 3. Compute the gripper's palm-pocket position in world frame.
        # The "pocket" = pinch midpoint between thumb tip and bc-tips
        # centroid. Use the existing _pinch_midpoint_xyz helper for
        # an accurate world XYZ.
        try:
            pocket_xyz = self.grasp_exec._pinch_midpoint_xyz(
                self.sim.data).copy()
        except Exception:
            # Fallback: use palm body position
            try:
                palm_bid = self.grasp_exec.gripper_body_id
                pocket_xyz = self.sim.data.xpos[palm_bid].copy()
            except Exception:
                pocket_xyz = np.array([4.5, -6.5, 0.20])

        # 4. Place obj DIRECTLY at the pocket. Other 9 objs parked
        # far away.
        obj_name = f"pickup_obj_{self.obj_idx}"
        obj_bid = mujoco.mj_name2id(self.sim.model,
                                     mujoco.mjtObj.mjOBJ_BODY, obj_name)
        if obj_bid < 0:
            raise RuntimeError(f"obj '{obj_name}' not found")
        jntadr = int(self.sim.model.body_jntadr[obj_bid])
        qadr = int(self.sim.model.jnt_qposadr[jntadr])
        # Obj sits with its CENTER at the pocket — fingers are
        # already wrapping around this position.
        # Read obj half-height for proper grounding (obj_bottom at
        # z=0 is unrealistic if pocket_z > obj_half_h; but for grip
        # tuning we don't need obj to rest on floor — let it float
        # at the pocket and the gripper holds it).
        self.sim.data.qpos[qadr] = float(pocket_xyz[0])
        self.sim.data.qpos[qadr + 1] = float(pocket_xyz[1])
        self.sim.data.qpos[qadr + 2] = float(pocket_xyz[2])
        self.sim.data.qpos[qadr + 3:qadr + 7] = [1.0, 0.0, 0.0, 0.0]
        # Park all other obj far away
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

        # 5. Settle briefly so obj rests in pocket (gravity does the work).
        for _ in range(40):
            self.sim.step_simulation(render=False)

        # Cache info for the run_pickup path
        self._obj_bid = obj_bid
        self._grasp_q = grasp_q
        # Re-anchor obj at the (now-slightly-moved-by-gravity) position
        # and update the cached obj_world used by grasp_exec.pick.
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
        """Set obj XY, chassis XY/yaw, arm qpos to a known good pose."""
        # 1. Place the target obj at our known XY
        obj_name = f"pickup_obj_{self.obj_idx}"
        obj_bid = mujoco.mj_name2id(self.sim.model,
                                     mujoco.mjtObj.mjOBJ_BODY, obj_name)
        if obj_bid < 0:
            raise RuntimeError(f"obj body '{obj_name}' not found")
        # Find the obj's free joint qposadr
        jntadr = int(self.sim.model.body_jntadr[obj_bid])
        if jntadr < 0:
            raise RuntimeError(f"obj '{obj_name}' has no joint")
        qadr = int(self.sim.model.jnt_qposadr[jntadr])
        # Get obj's geom half-height for proper z placement
        half_h = 0.15  # fallback
        for g in range(self.sim.model.ngeom):
            if int(self.sim.model.geom_bodyid[g]) == obj_bid:
                half_h = float(self.sim.model.geom_size[g, 1])
                break
        obj_z = half_h + 0.005  # tiny clearance, settles on first step
        self.sim.data.qpos[qadr:qadr + 3] = [self.obj_xy[0],
                                              self.obj_xy[1], obj_z]
        # Free-joint orientation: identity quaternion (no rotation)
        self.sim.data.qpos[qadr + 3:qadr + 7] = [1.0, 0.0, 0.0, 0.0]

        # Park all OTHER pickup objects far away so they don't interfere
        # with the scene. Same convention as arm_bridge.park_all_pickup_objects.
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
            # Park at (-10, -10 + i, 0.5) — well outside the demo scene
            self.sim.data.qpos[qa_i:qa_i + 3] = [-10.0,
                                                  -10.0 + i, 0.5]
            self.sim.data.qpos[qa_i + 3:qa_i + 7] = [1.0, 0, 0, 0]

        # 2. Place chassis at obj-standoff position with yaw facing obj.
        # Use the same standoff distance used by initial chassis push
        # in play_m1 (the SIDE_FORWARD_PUSH_TARGET). 0.64 m is the
        # actual post-push chassis-to-obj distance in STRICT mode.
        standoff = 0.64
        from math import atan2, cos, sin, pi
        # Approach direction: chassis approaches obj from a position
        # offset by approach_yaw_deg in world frame. approach_yaw_deg
        # is the WORLD direction from chassis to obj (chassis faces
        # this direction). Different yaws explore different gripper
        # orientations relative to obj for the 1+2 gripper's
        # structural asymmetry — the harness's --find-pose mode
        # sweeps over this to find a working one.
        chassis_yaw = np.radians(self.approach_yaw_deg)
        # approach_unit = unit vector FROM chassis TO obj
        # = (cos(yaw), sin(yaw))
        # chassis_xy = obj - standoff × approach_unit
        approach_unit = np.array([cos(chassis_yaw), sin(chassis_yaw)])
        chassis_xy = (self.obj_xy[0] - standoff * approach_unit[0],
                      self.obj_xy[1] - standoff * approach_unit[1])
        # Use the sim's chassis qpos setter convention. Look up the
        # base body to find its free joint.
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
                # qpos[qa_b + 2] is z; keep current floor height
                # Quaternion for yaw: (cos(yaw/2), 0, 0, sin(yaw/2))
                self.sim.data.qpos[qa_b + 3] = cos(chassis_yaw / 2.0)
                self.sim.data.qpos[qa_b + 4] = 0.0
                self.sim.data.qpos[qa_b + 5] = 0.0
                self.sim.data.qpos[qa_b + 6] = sin(chassis_yaw / 2.0)
        # Tell the chassis controller to hold this pose
        self.sim.target_base = np.array([float(chassis_xy[0]),
                                          float(chassis_xy[1]),
                                          float(chassis_yaw)])

        # CRITICAL: call mj_forward NOW so data.xpos reflects the
        # qpos values we just set for obj + chassis. Without this,
        # the IK below reads stale (zero) xpos values and computes
        # nonsense.
        mujoco.mj_forward(self.sim.model, self.sim.data)

        # 3. Compute a proper GRASP_Q via IK for THIS specific
        # chassis+obj scenario. Using a hardcoded GRASP_Q from a
        # different scenario produces a palm that's nowhere near
        # obj. Replicates what play_m1 does at line ~1230-1260
        # inside the candidate screening / scan.
        try:
            obj_world_xyz = self.sim.data.xpos[obj_bid].copy()
            # Get obj radius for proper standoff in compute_grasp_targets
            obj_radius = None
            for g in range(self.sim.model.ngeom):
                if int(self.sim.model.geom_bodyid[g]) == obj_bid:
                    obj_radius = float(self.sim.model.geom_size[g, 0])
                    break
            print(f"[Trial] IK setup: chassis_xy={tuple(np.round(chassis_xy, 3))}  "
                  f"chassis_yaw={np.degrees(chassis_yaw):.1f}°  "
                  f"obj_world={obj_world_xyz.round(3).tolist()}  "
                  f"obj_radius={obj_radius:.3f}")
            # Wrist goal for STRICT side-grip mode
            wrist_goal = (0.00, -1.88, 0.80, 0.00)  # hb, wz, wx, wy
            # Compute the IK target. side_approach=True → target is
            # at obj height (no above-Z offset).
            _, pre_grasp_pos = compute_grasp_targets(
                np.array(chassis_xy, dtype=float),
                obj_world_xyz,
                obj_radius=obj_radius,
                side_approach=True)
            print(f"[Trial] IK setup: pre_grasp_pos="
                  f"{pre_grasp_pos.round(3).tolist()} (world frame)")
            # Set the plan_data for IK
            reset_plan_data_for_ik(self.arm_bridge,
                                    base_xy=np.array(chassis_xy),
                                    base_yaw=chassis_yaw)
            # Sanity check: read chassis pose from planning_data
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
            # Run IK
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
            # Fallback to a default pose if IK fails
            grasp_q_known = [
                0.135, 0.360, 0.305, float(chassis_yaw),
                0.025, -1.88, 0.80, 0.00,
            ]
        # Set arm qpos via the sim's qpos indices for ARM1 (h1, h2, a1, th)
        for i, q in enumerate(grasp_q_known[:4]):
            self.sim.data.qpos[self.sim.qpos_indices[i]] = q
        # Wrist joints (indices 4-7 of grasp_q: hb, wz, wx, wy) — need
        # their joint qpos addresses.
        wrist_names = ("HandBearingJoint_1", "gripper_z_rotation_1",
                       "gripper_x_rotation_1", "gripper_y_rotation_1")
        for i, name in enumerate(wrist_names):
            jid = mujoco.mj_name2id(self.sim.model,
                                     mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                qa = int(self.sim.model.jnt_qposadr[jid])
                self.sim.data.qpos[qa] = grasp_q_known[4 + i]
        # CRITICAL: also set the CTRL values for the arm slides AND
        # the wrist actuators. Without this, the PD controllers
        # pull each joint back to ctrl=0 (neutral wrist) within a
        # few sim steps, and the side-grip wrist orientation
        # immediately collapses. Use grasp_exec's _set_arm_cmd
        # which handles BOTH the arm-slide PD ctrl values AND the
        # wrist actuator ctrl values (including the wrist_Z PD
        # pre-compensation factor that play_m1 uses).
        try:
            self.grasp_exec._set_arm_cmd(grasp_q_known)
            if self.verbose:
                print("[Trial] _set_arm_cmd(grasp_q) called — arm + "
                      "wrist ctrls now match grasp pose")
        except Exception as _e:
            print(f"[Trial] WARNING: _set_arm_cmd raised: {_e} "
                  f"(wrist will drift to neutral)")
        # Mirror to ARM2 idle pose (doesn't matter for tuning)
        for i, q in enumerate([1.20, 1.20, 0.10, 0.0]):
            self.sim.data.qpos[self.sim.qpos_indices[4 + i]] = q
        # ARM2 ctrl — keep it parked
        try:
            for i, q in enumerate([1.20, 1.20, 0.10, 0.0]):
                self.sim.data.ctrl[self.sim.actuator_ids[4 + i]] = q * 100
        except Exception:
            pass
        if self.verbose:
            print("[Trial] place: qpos set; running mj_forward + settle ...")
        # Forward kinematics so xpos reflects the placed pose
        mujoco.mj_forward(self.sim.model, self.sim.data)
        # Settle a few steps so obj rests on floor + chassis settles.
        # SYNCHRONOUS stepping here — bg thread is NOT running yet.
        for _ in range(60):
            self.sim.step_simulation(render=False)
        # Pre-populate the executor's cached PRE_GRASP_Q so that
        # is_local_retry=True skips the IK solve and uses our known
        # good arm config. Without this, the executor's local-retry
        # path falls back to running IK from the post-push chassis
        # position, which can fail because we didn't go through the
        # normal nav → IK → descent sequence first.
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

    # ── Run the pickup ───────────────────────────────────────────────

    def run_pickup_simple(self):
        """SIMPLIFIED pickup — call close + lift directly.

        Bypasses _pick_run entirely.  Obj is already in the pocket
        (set up by place_initial_state_simple).  We just need to:
          1. Tell the executor we're holding this obj (set _held_obj_bid)
          2. Run the close stroke (_set_gripper(CLOSE))
          3. Run the pre-verify TIGHTEN + sustained-contact check
          4. Run the lift (_strict_lift_with_retry)
          5. Wait 2 s, measure if obj stayed
        Pure physics test, no geometry alignment failures.
        """
        obj_bid = self._obj_bid
        grasp_q = self._grasp_q

        # Record pre-lift obj z
        obj_pre = self.sim.data.xpos[obj_bid].copy()
        self.metrics["obj_z_pre_lift"] = float(obj_pre[2])

        # Tell the executor what we're holding so its strict mode
        # code paths know which obj to monitor.
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

        # Mark side-grip mode so the executor's _curl_targets uses
        # the symmetric thumb open (matches play_m1's STRICT side-grip).
        self.grasp_exec._side_grip_active = True

        # Start the background sim thread (needed for time.sleep
        # coordination in _set_gripper and lift).
        self._start_bg_sim()
        time.sleep(0.1)

        t0 = time.time()
        try:
            # ── 1. CLOSE stroke ──
            # close_pos = FINGER_CLOSE_MAX (0.20); transition_secs
            # uses STRICT_CLOSE_TRANSITION_SECS (4.0) automatically
            # inside _set_gripper since obj is held.
            CLOSE_POS = 0.20
            if self.verbose:
                print(f"[Trial-SIMPLE] running close stroke ...")
            self.grasp_exec._set_gripper(CLOSE_POS,
                                         hold_seconds=0.60)
            # Read close result
            contacts = (self.grasp_exec._last_close_finger_contacts
                          or [False, False, False])
            self.metrics["close_3finger"] = (
                sum(bool(x) for x in contacts) >= 3)
            if self.verbose:
                print(f"[Trial-SIMPLE] close done: contacts={contacts}")

            # ── 2. Verify (sustained-contact) ──
            # Replicate the verify check from _pick_run [6.4].
            time.sleep(0.3)   # wait for grip to stabilize
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

            # ── 3. LIFT (with all the /36/37/41 tighten
            # and slip-monitor machinery) ──
            if self.verbose:
                print(f"[Trial-SIMPLE] running lift ...")
            lift_ok = self.grasp_exec._strict_lift_with_retry(
                obj_bid, CLOSE_POS, grasp_q)
            if self.verbose:
                print(f"[Trial-SIMPLE] lift done: ok={lift_ok}")
            self.metrics["slip_retries"] = int(
                getattr(self.grasp_exec,
                          '_strict_retry_count', 0))

            # ── 4. Measure obj follow + hold ──
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
        """Call grasp_exec.pick(is_local_retry=True) and wait for completion."""
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
        # Start the background sim thread NOW (before grasp_exec.pick).
        # grasp_exec uses time.sleep() and expects the sim to advance
        # in parallel. Synchronous stepping during state setup has
        # finished; from here on the bg thread drives physics.
        self._start_bg_sim()
        t0 = time.time()
        # is_local_retry=True skips arm motion phases (open/approach/descent)
        # and runs STEP 1 (cond back) + closed-loop alignment + gate + close
        # + verify + lift + slip-monitor. Exactly the stages we want to tune.
        self.grasp_exec.pick(self.obj_idx, obj_world,
                              on_complete=_on_complete,
                              is_local_retry=True)
        # Wait for completion (or timeout)
        completed = self._grasp_done_event.wait(timeout=60.0)
        self.metrics["trial_duration_s"] = time.time() - t0
        if not completed:
            self.metrics["errors"].append("TIMEOUT: grasp_exec.pick did not "
                                          "complete within 60s")
            return
        self.metrics["verify_passed"] = bool(
            self._grasp_result.get("success"))

        # Capture post-pickup state for scoring. If success: obj should
        # be in the grip and high up. If fail: obj is on floor.
        obj_after = self.sim.data.xpos[obj_bid][:3]
        self.metrics["obj_z_post_motion"] = float(obj_after[2])
        # Hold for 2 s and re-check obj position (did it stay in grip?)
        time.sleep(2.0)
        obj_after_hold = self.sim.data.xpos[obj_bid][:3]
        self.metrics["obj_z_after_hold"] = float(obj_after_hold[2])
        # "obj_followed_grip": did obj rise more than 5 cm above its
        # pre-lift z? (Lift target h1=0.60 corresponds to ~0.4-0.5 m
        # obj_z if grip held.)
        if (self.metrics["obj_z_pre_lift"] is not None
                and self.metrics["obj_z_post_motion"] is not None):
            self.metrics["obj_followed_grip"] = (
                self.metrics["obj_z_post_motion"]
                > self.metrics["obj_z_pre_lift"] + 0.05)
        # "obj_held_2s": did obj NOT fall during the 2-s hold?
        if (self.metrics["obj_z_post_motion"] is not None
                and self.metrics["obj_z_after_hold"] is not None):
            self.metrics["obj_held_2s"] = (
                self.metrics["obj_z_after_hold"]
                > self.metrics["obj_z_post_motion"] - 0.05)
        # Slip-retries: read from grasp_exec internal state. Available
        # as `_strict_retry_count` after a strict pickup.
        try:
            self.metrics["slip_retries"] = int(
                getattr(self.grasp_exec, '_strict_retry_count', 0))
        except Exception:
            self.metrics["slip_retries"] = 0


# ─── Parameter application ───────────────────────────────────────────────

@contextmanager
def apply_runtime_params(overrides):
    """Temporarily set grasp_executor module constants, restore on exit."""
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


# ─── Scoring ─────────────────────────────────────────────────────────────

def score_trial(metrics):
    """Higher = better.  Composite score over the metrics dict."""
    s = 0.0
    if metrics.get("close_3finger"):
        s += 1.0
    if metrics.get("verify_passed"):
        s += 2.0
    if metrics.get("obj_followed_grip"):
        s += 5.0
    if metrics.get("obj_held_2s"):
        s += 5.0
    # Penalty for slip-retries needed (means grip was marginal)
    s -= metrics.get("slip_retries", 0) * 1.0
    # Penalty for very high force-stop spikes
    fmax = metrics.get("force_stop_max_n", 0.0)
    if fmax > 500.0:
        s -= min(5.0, (fmax - 500.0) / 200.0)
    return float(s)


# ─── Main ────────────────────────────────────────────────────────────────

def _run_one_attempt(xml_path, params, obj_xy, approach_yaw_deg,
                       verbose, sim_context, drift=None):
    """One physical sim run.  Returns (score, metrics).

    Uses the SIMPLE setup — obj placed directly in gripper pocket,
    then close + verify + lift run via the executor's primitives.
    Skips _pick_run's gate / alignment / chassis-push machinery
    entirely.  Pure physics test of close+lift.

    Optional `drift` dict injects play_m1-realistic pose residuals
    (chassis offset, wz/wx drift) into the canonical setup so tuned
    params transfer to play_m1's real geometry.
    """
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
    """Run a trial (one or more attempts) and return aggregated result.

    Thread-timing non-determinism means a single attempt isn't a
    reliable signal — same input produces score 0/3/5 across runs.
    With `n_runs > 1`, the same config is run multiple times and the
    BEST score is kept (we want to know if a config CAN work, not the
    worst-case).

    Returns: (best_score, best_metrics).  Metrics dict gets an extra
    field `runs` = list of (score, verify, followed, held) tuples.

    If `sim_context` is provided, reuses it (fast path — state reset
    only).  If None, constructs a fresh sim each trial.
    """
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
    """Sweep over approach yaws, find the one with best score.
    Takes optional sim_context for fast reuse.
    Returns: (best_yaw_deg, best_score, all_results_list).
    """
    if yaws_deg is None:
        yaws_deg = list(range(-180, 180, 30))
    # Build shared context if not provided
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


# ─── Phase 3: 2D sweep over (obj_xy, yaw) ─────────────────────────────────

def find_best_obj_position_and_yaw(xml_path,
                                      obj_xy_candidates=None,
                                      yaws_deg=None,
                                      score_threshold=10.0,
                                      out_csv=None,
                                      verbose=True,
                                      sim_context=None,
                                      n_runs_per_config=2):
    """Phase 3: 2D sweep over (obj position, approach yaw) to find a
    geometry that produces 3/3 close + verify pass.  Once such a
    baseline exists, physics tuning becomes productive.

    Score interpretation:
      < 6      : verify failed (no 3-finger contact sustained)
      6-7      : close + verify passed, but obj didn't follow lift
      8-12     : close + verify + obj followed lift (good grasp)
      >= 13    : close + verify + obj followed + held 2s (full success)

    `score_threshold` = early-stop target.  If a combo scores ≥
    threshold, stop sweeping and return that combo immediately.

    Returns: (best_obj_xy, best_yaw_deg, best_score, all_results).
    """
    if obj_xy_candidates is None:
        # Default sweep: 9 obj positions across the floor zone
        # (avoiding edges and shelf-occupied areas). Slight variation
        # in X and Y to sample different chassis-obj relative
        # geometries that gripper might happen to align with.
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
        # 8 yaws, 45° apart — wider step than Phase 2's 30° but more
        # obj positions, so total trial count is manageable.
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
    # build a shared SimContext if not given. Trial
    # bodies will reuse it, dropping per-trial overhead from ~30 s
    # to <1 s.
    # rebuild context every REBUILD_EVERY trials as a
    # safety against accumulated MuJoCo state drift. Loses ~20 s per
    # rebuild but guarantees clean state across the sweep.
    REBUILD_EVERY = 20
    own_context = sim_context is None
    if own_context:
        sim_context = SimContext(xml_path, verbose=verbose)
    try:
        for obj_xy in obj_xy_candidates:
            for yaw in yaws_deg:
                idx += 1
                # Periodic rebuild for state hygiene
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
                # Early-stop: if score ≥ threshold, we've found a
                # productive baseline. No need to keep searching.
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
    # No combo hit threshold — return the best found.
    results_sorted = sorted(
        results, key=lambda r: r[2], reverse=True)
    best_obj_xy, best_yaw, best_score, _ = results_sorted[0]
    return best_obj_xy, best_yaw, best_score, results


def grid_search(xml_path, out_csv, approach_yaw_deg=None,
                  sim_context=None, n_runs_per_config=2):
    """Grid over the most-impactful runtime params.  Reports best.
    `approach_yaw_deg` is the geometry to test all param combos at.
    `n_runs_per_config` = run each combo N times, keep BEST score
    (mitigates the thread-timing non-determinism).
    """
    own_context = sim_context is None
    if own_context:
        sim_context = SimContext(xml_path, verbose=True)
    # expanded grid over the grip-retention tunables.
    # These are the params with highest theoretical impact on lift
    # success per our physics breakdown (+).
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
            # periodic context rebuild for state hygiene
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
    # drift injection — simulates play_m1's at-close
    # state so params tuned here transfer to play_m1.
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
        # Sorted summary
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
        # build SimContext ONCE; reuse across all trials
        # in both phases. Avoids ~30 s of init overhead per trial.
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
        # Override KNOWN_OBJ_XY_LIST[0] so the run_one_trial default
        # picks up the user-supplied obj_xy. (grid_search internally
        # calls run_one_trial which falls back to KNOWN_OBJ_XY_LIST[0]
        # if no obj_xy is explicitly passed.)
        KNOWN_OBJ_XY_LIST[0] = obj_xy
        grid_search(args.xml, args.out_csv,
                     approach_yaw_deg=args.approach_yaw)
        return


if __name__ == "__main__":
    main()
