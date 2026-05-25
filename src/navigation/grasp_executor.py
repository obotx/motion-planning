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

from navigation.arm_planner import (
    HOME_Q, PARK_Q, ARM_DOF, WRIST_NEUTRAL, JOINT_RANGES_ARM,
)
from navigation.finger_planner import FingerBridge, FINGER_DOF

# ── Wrist orientation goals (Phase 2: integrated with 8-DOF OMPL) ────────
# The 4 wrist joints (HandBearing, gripper_z/x/y_rotation) are part of the
# OMPL state vector at indices 4..7. These constants set goal poses per
# obj geometry.
# HandBearing axis = world-Y rotation:
# hb = 0 → fingers HORIZONTAL forward → side approach
# hb = -0.50 → fingers tilt ~29° down → diagonal approach
# hb = -π/2 → fingers straight DOWN → top-down approach
# Strategy (3 tiers by obj half-height):
# TALL obj (half_h ≥ 0.10 m): SIDE APPROACH, hb = 0. The wrist
# reaches the obj's mid-height with palm horizontal; the b/c side
# fingers oppose the thumb across the obj's horizontal middle —
# same gripper pose the free-move sim manually demonstrated for
# the 26 cm cylinders. IK target Z = obj_z (no above-offset);
# descent is column-down to obj height, then close.
# MED obj (0.05 ≤ half_h < 0.10): DIAGONAL hb = -0.50. Reach
# from above-front, fingers wrap upper rim. Compromise for obj
# too short to side-grip cleanly (palm would dip near floor) but
# tall enough that pure top-down hovers above the obj body.
# SHORT obj (half_h < 0.05): AGGRESSIVE TOP-DOWN hb = -0.90.
# Fingertip drop ~7 cm needed to reach the obj body at all.
# Why these thresholds.  Wrist Z floor for the IK is 0.10 m: the manual
# free-move sim confirms the arm physically reaches floor level with
# H1=H2≈0.6 and A1≈0.28.  At half_h = 0.10, obj middle z = 0.10 — the
# wrist floor (0.10) is exactly at obj mid for side approach.  Below
# that, side approach brings the palm too close to floor.
WRIST_PITCH_SIDE_APPROACH      = 0.0    # HandBearing — palm horizontal.
                                        # wrist_X=0.80 below keeps the
                                        # palm level relative to the
                                        # floor even when the arm boom
                                        # tilts.  Tilting HandBearing
                                        # breaks that palm-leveling
                                        # effect.
WRIST_PITCH_DEFAULT_DIAGONAL   = -0.50  # rad, fingers tilt 29° down
WRIST_PITCH_AGGRESSIVE_TOPDOWN = -0.90  # rad, fingers tilt 52° down

# Side-mode gripper-frame rotation goals. The 1+2 gripper has the thumb
# at a specific angle around the palm-Z axis in the default chain
# orientation. Rotating wrist_Z spins the entire gripper around palm-Z
# so the thumb opposes the 2 side fingers ACROSS the obj's diameter
# (rather than "thumb up, fingers down"). From the user's free-move
# slider experimentation:
# wrist_Z ≈ 20% slider position → qpos ≈ -1.88 rad (~ -108°)
# wrist_X ≈ 100% slider position → qpos ≈ +0.80 rad (~ +46°)
WRIST_Z_SIDE_APPROACH = -1.88   # gripper_z_rotation (rotate thumb/fingers
                                # to opposite sides for horizontal pinch)
WRIST_X_SIDE_APPROACH =  0.80   # gripper_x_rotation (face-forward palm tilt)
WRIST_Y_SIDE_APPROACH =  0.00   # gripper_y_rotation (no extra tilt)

# Wrist-Z PD pre-compensation. With wz target -1.88 rad and kp=100,
# the actuator's steady-state residual leaves qpos ~0.17 rad short of
# target due to gravity and finger inertia. Pre-multiplying the ctrl
# by (1 + RATIO) cancels the offset so qpos lands at the intended
# orientation. Joint range ±π keeps the overshot ctrl in bounds.
WRIST_Z_PD_COMPENSATION_RATIO = 0.09

# Realism mode for the pre-close phase. When enabled, the executor
# does not preemptively animate the object into the gripper; physical
# finger motion is what makes contact. The pre-close gate also
# enforces a Z-gap bound and a non-zero finger-object contact count.
REALISM_MODE_NO_SMOOTH_LIFT = True
REALISM_PRE_CLOSE_Z_GAP     = 0.05

# Tiered post-gate handling of the residual XY error between the
# gripper pocket and the object after the live chassis nudge converges:
# carry_gap ≤ NATURAL_CLOSE_DRAG_THRESHOLD → no animation; close
# stroke physically drags the object into the pocket
# NATURAL < carry_gap ≤ MICRO_LIFT → a small animated
# translation eases the object into the pocket
# carry_gap > MICRO_LIFT → gate rejects
REALISM_MICRO_LIFT_THRESHOLD = 0.12
NATURAL_CLOSE_DRAG_THRESHOLD = 0.05

# Asymmetric soft-assist tier: when one finger is already in contact
# but the opposite side is still outside the per-side reach, relax
# sides_ok IFF the residual is bridgeable by the micro-lift animation.
ASYM_SOFT_ASSIST_MAX_FAR = 0.15

# Graceful disengage before asymmetric rejection — back the chassis
# this far along the reverse-approach to let any displaced object
# re-settle before falling through to the rejection cascade. One-shot
# per pick (guarded by `_pre_close_backup_used`).
ASYM_BACKUP_DISTANCE = 0.06

# Palm-anchor accept tier. When the gripper palm is positioned
# correctly relative to the object — below obj_top in Z, pinch midpoint
# near object XY, no arm-structure clipping — the close stroke wraps
# the fingers naturally regardless of where each individual fingertip
# currently sits. Mimics a palm-first human grip.
PALM_ANCHOR_Z_MARGIN     = 0.02   # palm must be ≥ this far BELOW obj_top
PALM_ANCHOR_MAX_CARRY    = 0.08   # pinch_midpoint must be ≤ this far from obj
PALM_ANCHOR_MAX_FAR      = 0.18   # no finger > this from obj

# Wrist-Z axis refinement. After the side-grip IK produces a
# candidate, the realised thumb→bc world axis may not be perpendicular
# to the chassis→obj approach. Two passes correct this:
# * IK-time refine: re-solve IK once with `wz` adjusted by -err
# * Runtime correction: measure the live axis after descent and
# chassis push, command a wz step, settle, and adopt only when
# the reach metric improves AND the axis error did not worsen.
WZ_REFINE_AXIS_ERR_THRESHOLD     = math.radians(5.0)
WZ_RUNTIME_CORRECTION_SETTLE_S      = 0.35
WZ_RUNTIME_CORRECTION_MAX_DELTA     = math.radians(60.0)
WZ_RUNTIME_CORRECTION_MAX_ITERS     = 3
WZ_RUNTIME_CORRECTION_MAX_INITIAL   = math.radians(10.0)

# Pickup mode selection. The CLI arg `--fast-pickup` accepts 1/2/3
# and sets these two flags accordingly:
# Mode 1 (strict) → STRICT_PICKUP_MODE=True, FAST_PICKUP_MODE=False
# All rescue tiers disabled. Only weld constraint for transport
# and proximity stop for surface contact are active. Pre-close
# pin animation off; gate accept tiers (palm-anchor, asymmetric
# N-finger, soft-assist) disabled; MIN_CONTACTS required = 3.
# Honest physics-driven pickup; lower success rate.
# Mode 2 (balanced) → STRICT_PICKUP_MODE=False, FAST_PICKUP_MODE=False
# Production default. Pre-close gates strict, but rescue tiers
# (palm-anchor accept, 1-finger / 2-finger accept, smooth-lift
# MICRO) bridge marginal geometries via pin.
# Mode 3 (fast) → STRICT_PICKUP_MODE=False, FAST_PICKUP_MODE=True
# Demo path. Recoverable pre-close gates force-pass; pin closure
# carries marginal cases for guaranteed completion. Arm-vs-obj
# clip gate stays strict (unrecoverable by pin).
FAST_PICKUP_MODE   = False
STRICT_PICKUP_MODE = False
STRICT_PERFECT_FRICTION_ONLY = False

# STRICT mode physics-driven grip parameters. These only take effect
# when STRICT_PICKUP_MODE is True; mode 2 / mode 3 ignore them.
# Grip-force target uses the standard friction holding-force inequality:
# mu * sum(normal_force_per_finger) >= safety * mass * g
# Required per-finger normal force at 3 contact points:
# N_req = safety * mass * g / (mu * 3)
# The close stroke continues until each finger reaches N_req (force-stop)
# rather than stopping at proximity. Position ctrl is allowed to advance
# past the contact qpos by up to STRICT_OVERDRIVE_DELTA so the PD residual
# generates real contact force.
STRICT_FRICTION_MU              = 0.7    # finger pad vs cylinder surface
STRICT_GRIP_SAFETY              = 2.5  #  holding force margin over m*g (step 3 tried 3.25 — made avg score worse, same direction as 's 4.0 failure)
STRICT_MIN_NORMAL_PER_F         = 0.6    # N — floor when mass is tiny
# (tune_grip_lift harness exposes these): grip-retention
# bump magnitudes (radians on each finger's j1 ctrl). Increasing
# these deepens the squeeze; decreasing relaxes. Tuned via the
# tools/tune_grip_lift.py harness.
LIFT_TIGHTEN_PREVERIFY_RAD       = 0.05   # before sustained-contact verify
LIFT_TIGHTEN_PRELIFT_RAD         = 0.03  #  before lift motion starts (tried 0.10 — no improvement under drift)
LIFT_TIGHTEN_POSTLIFT_RAD        = 0.03   # after lift motion, before slip-monitor
LIFT_RETIGHTEN_PER_STEP_RAD      = 0.012  # baseline open-loop re-tighten
LIFT_RETIGHTEN_INTERVAL_STEPS    = 5      # baseline re-tighten interval
# closed-loop slip-feedback re-tighten. The baseline
# above is a time-periodic open-loop top-up; the values below convert
# the *actual* slip (grip_dz − obj_dz) into a proportional ctrl
# bump every tick, mimicking how a human tightens grip the instant
# they feel the obj starting to fall. Set GAIN=0 to disable.
LIFT_SLIP_BUMP_THRESHOLD_M       = 0.003  # m — slip below this is ignored (noise)
LIFT_SLIP_BUMP_GAIN              = 6.0    # rad per metre of slip
LIFT_SLIP_BUMP_MAX_PER_STEP_RAD  = 0.05   # cap on single-tick bump (safety)
# further closed-loop refinements.
# (1) Per-finger attribution: when slip fires, bump only the finger(s)
# whose normal force is below the per-finger floor, so the fingers
# already gripping firmly are not over-squeezed.
LIFT_PER_FINGER_FORCE_FLOOR_N    = 4.0    # N — finger below this is "loose" and gets the bump
# (2) Tangential / XY slip detection during lift.
LIFT_XY_DRIFT_ABORT_M            = 0.03   # m — abs XY drift > 3 cm aborts the lift
# (3) Force-decay early warning — pre-emptive bump before slip becomes
# visible in `rel_z`. Track per-finger normal force over a short
# sliding window; if it has dropped >FRAC, bump pre-emptively.
LIFT_FORCE_DECAY_WINDOW          = 6      # ticks for force-decay sample window
LIFT_FORCE_DECAY_FRAC            = 0.40   # 40 % drop triggers pre-emptive bump
LIFT_FORCE_DECAY_BUMP_RAD        = 0.020  # rad — pre-emptive bump magnitude
# (4) Gentle test-lift before committing to full lift.
LIFT_TEST_LIFT_ALPHA             = 0.06   # 6 % of full lift = ~6 mm rise
LIFT_TEST_LIFT_SLIP_TOLERANCE_M  = 0.004  #  m — slip > 4 mm during test = fail
LIFT_TEST_LIFT_RETRY_BUMP_RAD    = 0.04   # rad — j1 bump between test-lift retries
LIFT_TEST_LIFT_MAX_RETRIES       = 2      # number of retry+bump iters
# (5) Mass re-estimation during lift — only ALLOWED TO INCREASE N_req,
# never to decrease (so we never accidentally relax mid-lift).
LIFT_MASS_REESTIMATE_ENABLED     = True
LIFT_MASS_REESTIMATE_K           = 0.30   # fraction of shortfall to absorb per check
LIFT_MASS_REESTIMATE_INTERVAL    = 8      # ticks between re-estimations
# (8) Hysteresis: after any bump, wait K ticks before allowing another
# so the PD has time to settle.
LIFT_BUMP_HYSTERESIS_TICKS       = 2
# Triad-balance gating (#6 + promotion).
# balance_score = |Σ normal_i × N_i| / max(N_i).
# 0 ≈ perfect opposing 3-finger cage
# 1 ≈ all force on one side (no opposing pinch — obj will squirt out)
# Two-stage gate so we don't reject borderline-good grasps:
# * < DIAG_THRESHOLD → log only, "BALANCED"
# * < GATE_THRESHOLD → log "MARGINAL", still accept
# * ≥ GATE_THRESHOLD → log "ONE-SIDED" AND reject the verify
VERIFY_TRIAD_BALANCE_DIAG_THRESHOLD = 0.45
VERIFY_TRIAD_BALANCE_GATE_THRESHOLD = 0.75   # generous — only catches obvious failures
VERIFY_TRIAD_BALANCE_GATE_ENABLED   = True
# SUSTAINED-FORCE verify gate. After 's
# solimp width 0.010 fix established the lift physics work in
# clean conditions, play_m1 transfer still fails because the
# single-tick verify snapshot can catch a chatter-trough where
# instantaneous forces hit 0 even with geometric contact. Lift
# then fires on zero-force grip → slip.
# Fix: actively sample per-finger normal forces over a window, fail
# the gate if any sample shows the required (thumb + b OR c) below a
# noise floor. Forces opposing-pinch verify rule to be confirmed
# across multiple ticks rather than trusting one snapshot.
VERIFY_SUSTAINED_FORCE_ENABLED        = True
VERIFY_SUSTAINED_FORCE_WINDOW_S       = 0.50   # observe forces for this long
VERIFY_SUSTAINED_FORCE_SAMPLE_INT_S   = 0.05   # sample interval (10 samples)
VERIFY_SUSTAINED_FORCE_FLOOR_N        = 30.0   # per-finger floor — well above noise, below force-stop target
# relax from "all samples pass" to "≥60 % pass".
# Initial all-pass criterion was correct in theory but with the
# MuJoCo solver chatter (period ~50 ms), 0-force troughs are
# essentially guaranteed within a 500 ms window even on good
# grips. Allowing 40 % of samples below floor accepts grips where
# the majority of ticks have real force — average force still
# carries lift since the obj's inertia smooths through troughs as
# long as they're brief.
VERIFY_SUSTAINED_FORCE_MIN_PASS_FRAC  = 0.60
# spike-detect upper bound. Even when sustained-force
# below-floor check passes, individual ticks can have M-N spike
# values (observed up to 8.15 M N). Those spikes during lift launch
# the obj sideways. Reject the verify if any per-finger sample
# exceeds this absolute cap. Setting at 50 kN is well above any
# realistic finger contact force (typical max ~10 kN in good grips)
# but cleanly catches the M-N solver-resonance spikes.
VERIFY_SUSTAINED_FORCE_SPIKE_CAP_N    = 50000.0
# BC-RESCUE. When the close stroke yields a 2/3 grasp
# with thumb (a) firm and exactly ONE of (b, c) missing, instead of
# the expensive full re-grasp cycle (open, retract, re-approach,
# re-close — ~10 s + 50% chance of repeat), try a targeted bump on
# the missing finger's j1 ctrl. Runs of …61 showed
# `(2/3 fingers)` is the second-most-common close outcome (after
# 1/3 thumb-only) — when bc-pair started at ~0.5 cm apart but only
# one engaged during the close stroke (obj rolled / contact solver
# missed the other). A small extra curl re-engages the missing
# finger without disturbing the firm grip.
USE_BC_RESCUE                = True
BC_RESCUE_J1_BUMP_RAD        = 0.05  # rad — extra curl on the missing finger.
                                     # ~0.4 mm tip motion — gentle enough to
                                     # register contact without crushing.
BC_RESCUE_SETTLE_S           = 0.30  # s — PD settle window after bump
BC_RESCUE_SUCCESS_N          = 4.0   # N — force threshold for successful re-engage (matches LIFT_PER_FINGER_FORCE_FLOOR_N)
# Lift motion duration (LIFT_STEPS × per_step_settle). Halve the
# steps → faster lift, less time for grip decay but more dynamic
# load. Tune to find the sweet spot.
LIFT_STEPS_MULTIPLIER            = 3      # LIFT_STEPS = DESCENT_STEPS × this
                                          # (2 → 3 — more
                                          # steps, smaller jumps per step.
                                          # Test-lift uses 0.06× alpha and
                                          # passes; main lift jumps were ~2×
                                          # bigger per step and broke contact
                                          # within step 1.)
LIFT_PER_STEP_SETTLE_MULTIPLIER  = 2.0    # LIFT settle = PD_SETTLE × this
                                          # (1.2 → 2.0 — test
                                          # lift uses 1.5× settle and passes;
                                          # main lift was using only 1.2×
                                          # → PD residual not catching up
                                          # between steps → contact decays.
                                          # 2.0 makes main lift slightly
                                          # slower than test-lift, giving
                                          # PD enough time to maintain grip.)

STRICT_FORCE_STOP_STABLE_TICKS  = 1
# When the force-stop check fires, the actuator's ctrl was already
# commanding the finger past the contact qpos by the residual the
# check loop allowed to accumulate.  After freeze the PD residual
# continues to push because ctrl = current qpos > rest qpos.
# Solution: when force exceeds the threshold by more than
# RELIEF_TRIGGER_RATIO × target, back off the proximal (j1) ctrl by
# FORCE_STOP_RELIEF_RAD so PD has a small unloading distance to work
# with.  j2 / j3 still freeze at current qpos to preserve the wrap
# geometry.
STRICT_FORCE_STOP_RELIEF_TRIGGER = 2.0    # × target → activate relief
STRICT_FORCE_STOP_RELIEF_RAD     = 0.005  # rad — base back-off on relief
# Tiered relief: with close-overdrive disabled in STRICT, peak forces
# sit at ~40-100 N
# range.
# (20/50/80 mrad) showed all 3 fingers losing contact entirely
# within 600 ms of force-stop because the back-off moves the
# fingertip 1.5-4 mm off the rigid obj surface.
# reduces magnitudes ~4× so relief bleeds PD residual without
# disengaging contact (fingertip motion now ~0.4-2 mm — within
# obj rigid surface compliance).
STRICT_FORCE_STOP_RELIEF_HIGH_RATIO   = 5.0    # × target → mid relief
STRICT_FORCE_STOP_RELIEF_HIGH_RAD     = 0.012  # rad — 5-10× overshoot
STRICT_FORCE_STOP_RELIEF_EXTREME_RATIO = 10.0  # × target → max relief
STRICT_FORCE_STOP_RELIEF_EXTREME_RAD   = 0.025 # rad — > 10× overshoot
                                          # to 1. At 50 Hz tick rate,
                                          # 3 consecutive ticks = 60 ms
                                          # of force-above-threshold
                                          # before freeze. In that 60
                                          # ms the finger penetrates
                                          # obj significantly (PD-driven
                                          # close speed), producing
                                          # absurd force spikes
                                          # (run log: 880 kN, 34 kN
                                          # contact forces). 1-tick
                                          # stops on first force reading
                                          # → finger freezes at surface,
                                          # not embedded in obj.
STRICT_OVERDRIVE_DELTA_INIT     = 0.08   # rad initial push past contact qpos
STRICT_OVERDRIVE_DELTA_MAX      = 0.30   # cap (actuator ctrlrange clamps too)
# Slip detection during the lift phase. obj pose is tracked in the
# gripper-anchor frame; if it drifts more than DISP_THRESH or moves
# faster than VEL_THRESH after a settle window, the grip is reported
# slipping and a retry is triggered.
STRICT_SLIP_DISP_THRESH         = 0.030  #  m
STRICT_SLIP_VEL_THRESH          = 0.04   # m/s
STRICT_SLIP_SETTLE_S            = 0.30   # ignore early lift transient
STRICT_LIFT_OBSERVE_S           = 2.00  #  observe window before declaring OK
STRICT_RETRY_MAX                = 2      # max grip-bump retries before bail
STRICT_RETRY_GRIP_BUMP          = 1.30   # force target multiplier per retry
# cap the accumulated force multiplier. Without this,
# repeated slip-retries (each ×1.30) ratchet the multiplier up
# unboundedly across multiple cycles or candidates —
# showed N=14 kN single-finger crushes when multiplier had grown past
# 5. Capping at 2.0 keeps N_req/f under ~24 N which is plenty for a
# 1 kg obj (10 N gravity × 2.4 safety) while preventing the runaway
# PD residual blowups.
STRICT_FORCE_MAX_MULTIPLIER     = 2.0

# Gate for verbose per-joint / per-waypoint diagnostic dumps. Off by
# default; grasp-essential gate / contact / refine logs remain
# unconditional.
VERBOSE_GRASP_DEBUG = False

# Per-side reach check for the side-grip pre-close gate. The
# pinch_midpoint metric averages thumb and bc-centroid distances;
# asymmetric mismatches (e.g. thumb 5 cm one side, bc 17 cm the
# opposite side) can pass it on the average while the close stroke
# can't physically reach the far side (finger close arc moves the
# tip ~7 cm inward). This bar enforces that EACH side is within
# close-stroke reach of obj independently — rejects geometries where
# smooth-lift would have to teleport obj to the gripper rather than
# the gripper grasping a nearby obj.  0.13 m bar: tip is 6 cm from
# obj surface (open) → curl brings it to contact.  Looser bars rejected
# "looked like a grip" attempts at d_bc 10-13 cm; those should fire
# close and let the force-stop + opposing-pinch verify decide.
SIDE_FINGER_PRECLOSE_REACH = 0.13

SIDE_FINGER_PRECLOSE_REACH_PERFECT = 0.15

# (v3): tighter curl angle for the --perfect kinematic
# pre-close so fingers compress obj even when slightly offset from
# pocket center. The default size-aware close_ctrl is 0.15 which works
# when obj is at pocket (initial pad-obj penetration = contact margin
# generates force). With offset obj, 0.15 leaves fingers just touching
# obj without compression. 0.30 curls fingers ~2x further so they
# overlap obj surface and force the contact solver to generate
# compression impulses.
PERFECT_KINEMATIC_CLOSE_POS = 0.30

ENABLE_PERFECT_PIN_FORWARD_SHIFT = False
PERFECT_PIN_FORWARD_SHIFT_M      = 0.005

# (v3 dev aid): one-shot flag to snap held obj to the
# current pinch_midpoint right before the close stroke fires. Emulates
# the harness's floating-obj-at-pocket setup in play_m1 (--auto-skip-nav
# sets this True per cycle). Not a pin (one qpos write), not a weld
# (no constraint). Matches harness which the user accepted as valid
# initial-state setup. Cleared after first close fires.
SNAP_OBJ_TO_POCKET_PRE_CLOSE_ONCE = False

# Live-nudge loop safety guards. Each nudge translates the chassis by
# the carry_gap residual; three checks per iteration prevent the loop
# from clipping the chassis into the object or stalling:
# 1. MIN_CHASSIS_OBJ_DIST stops the chassis advancing closer than the
# arm needs for descent clearance.
# 2. NUDGE_MIN_CARRY_GAP_IMPROVEMENT aborts when residual gain stalls.
# 3. Increasing arm-vs-chassis contact count terminates the loop.
MIN_CHASSIS_OBJ_DIST            = 0.40
NUDGE_MIN_CARRY_GAP_IMPROVEMENT = 0.010   # 1 cm minimum per iteration

WRIST_PITCH_TALL_OBJ_THRESHOLD  = 0.10  # half-h ≥ this → side approach
WRIST_PITCH_SHORT_OBJ_THRESHOLD = 0.05  # half-h <  this → aggressive

# Legacy names retained — external modules may import them. All alias
# the active tier constant.
WRIST_PITCH_TOPDOWN_SHORT = WRIST_PITCH_AGGRESSIVE_TOPDOWN
WRIST_PITCH_TOPDOWN_MED   = WRIST_PITCH_DEFAULT_DIAGONAL
WRIST_PITCH_TOPDOWN_TALL  = WRIST_PITCH_SIDE_APPROACH

# Gripper-ids array indices for the 4 wrist actuators (per arm).
# Order in gripper_actuator_V2.xml: wrist_X, wrist_Y, wrist_Z, HandBearing.
GIDS_WRIST_X      = 11
GIDS_WRIST_Y      = 12
GIDS_WRIST_Z      = 13
GIDS_HANDBEARING  = 14


def object_half_xy(model, obj_bid):
    """Return (half_x, half_y) for a box-like geom on the given body,
    or None for cylinder/sphere geoms (rotationally symmetric in XY).
    Used by compute_wrist_goal_for_obj to decide whether to spin the
    gripper about its palm-Z axis so the b/c finger span pinches across
    the obj's shorter side.
    """
    try:
        for g in range(model.ngeom):
            if int(model.geom_bodyid[g]) != int(obj_bid):
                continue
            # mjGEOM_BOX = 6; geom_size[0,1] are half-extents in obj X,Y
            if int(model.geom_type[g]) == 6:
                return (float(model.geom_size[g, 0]),
                        float(model.geom_size[g, 1]))
    except Exception:
        pass
    return None


def object_half_height(model, obj_bid, default=0.075):
    """Module-level half-height read used by both play_m1's IK screening
    loop and the executor.  Mirrors GraspExecutor._object_half_height
    but takes the model directly so it can be called pre-instantiation."""
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
    """Pick a wrist pose (hb, wz, wx, wy) per obj geometry — 3-tier logic.

    The wrist_goal is the IK's PREFERRED orientation, NOT a hard target.
    With palm-target IK + wrist_weight=0.5 (side-grip), SLSQP uses
    HandBearing as a 5th reach DOF — the palm can swing 15-20 cm around
    the Hand_Bearing pivot, which is what gets the palm to floor-obj
    height even though the wrist (Link1) alone cannot.

    Tiers:
        half_h < SHORT (0.05 m) → AGGRESSIVE_TOPDOWN, hb=-0.90, wrist=0
        half_h < TALL  (0.10 m) → DEFAULT_DIAGONAL,   hb=-0.50, wrist=0
        half_h ≥ TALL           → SIDE_APPROACH,      hb= 0.00, wrist
                                                      rotated to put the
                                                      thumb opposite the
                                                      2 fingers across
                                                      the obj's diameter
                                                      (gripper-frame
                                                      spin around Z).
    """
    half_h = object_half_height(model, obj_bid)
    if half_h < WRIST_PITCH_SHORT_OBJ_THRESHOLD:
        # Aggressive top-down: HB tilted way down, no gripper spin needed
        return (float(WRIST_PITCH_AGGRESSIVE_TOPDOWN), 0.0, 0.0, 0.0)
    if half_h < WRIST_PITCH_TALL_OBJ_THRESHOLD:
        # Diagonal: HB tilted, no gripper spin needed
        return (float(WRIST_PITCH_DEFAULT_DIAGONAL), 0.0, 0.0, 0.0)

    # Side approach (tall obj): palm horizontal + gripper rotated so
    # thumb and 2 fingers oppose each other ACROSS the horizontal axis
    # of the obj. The wrist_Z rotation is the key piece — without it
    # the default chain orientation puts the thumb "above" the 2 fingers
    # (thumb-down / fingers-up in palm frame), which is top-down geometry
    # even with HB=0. With wrist_Z = -1.88 rad, the thumb rotates to
    # one side of the palm and the 2 fingers to the opposite side —
    # exactly the "fingers one side, thumb other side" config.
    # Keep HandBearing at zero for side-grip — combining a non-zero
    # `hb` pitch with the side-grip `wx` tilt couples through the
    # wrist chain in a way that lifts the fingertips farther above the
    # palm rather than angling them down toward the object.
    hb = WRIST_PITCH_SIDE_APPROACH
    wz = WRIST_Z_SIDE_APPROACH
    wx = WRIST_X_SIDE_APPROACH
    wy = WRIST_Y_SIDE_APPROACH

    # Box obj: optional second 90° spin to pinch across the SHORTER side
    # added to whatever wrist_Z is already at. For cylinders this
    # whole branch is skipped (rotational symmetry).
    half_xy = object_half_xy(model, obj_bid)
    if half_xy is not None and abs(half_xy[0] - half_xy[1]) >= 0.005:
        if half_xy[0] > half_xy[1]:
            wz = wz + (math.pi / 2.0)
            # Wrap into the joint's allowed range [-π, +π]
            if wz > math.pi:
                wz -= 2.0 * math.pi
            elif wz < -math.pi:
                wz += 2.0 * math.pi

    return (float(hb), float(wz), float(wx), float(wy))


def is_side_approach(wrist_goal):
    """True when the wrist_goal's HandBearing is near 0 (palm horizontal
    — the side-grip family).  Diagonal (-0.50) and aggressive top-down
    (-0.90) are both excluded.  Side mode triggers the post-descent
    chassis push, palm-target IK, and the per-DOF wrist weights that
    let HandBearing be a free reach DOF while wrist_X/Z stay locked.

    Threshold |hb| < 0.20 matches play_m1's screening check so both
    sides of the pipeline agree on the mode.
    """
    if wrist_goal is None:
        return False
    return abs(float(wrist_goal[0])) < 0.20


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

# Pre-close acceptance gate. IK runs against plan_data which does not
# include passive RotationLeftJoint deflection, so the runtime wrist may
# fall short of the IK target. We measure deviation from the IK target
# (not raw distance to the object) because the IK target is intentionally
# placed GRIPPER_STANDOFF_XY behind the object by design.
GRIP_DEVIATION_TOLERANCE = 0.20      # legacy Link1 ik_dev safety net only;
                                     # carry_gap is the real success metric.
                                     # Wider window accommodates the passive-
                                     # joint deflection between Link1 and the
                                     # fingertip pocket.
GRIP_OBJ_XY_TOLERANCE    = 0.10      # legacy obj_grip_xy safety net only.

# ── Path B Step 1: carry-anchor gate ───────────────────────────────────
# Primary success metric: the runtime carry_anchor (centroid of the 3
# fingertip-pocket bodies) must sit within CARRY_GAP_TOLERANCE of the
# object XY. Link1 sees passive-joint deflection in this parallel
# manipulator, so the fingertip centroid — not Link1 — is the proper
# reference for grasp quality. The base-XY retry chain translates the
# chassis by the measured residual to bridge any remaining gap.
CARRY_GAP_TOLERANCE      = 0.070   # top-down / diagonal: 3-tip centroid
CARRY_GAP_TOLERANCE_SIDE = 0.060   # side-grip: pinch midpoint

# Place verification: drop_xyz must be within PLACE_XY_TOLERANCE of the
# shelf target (XY) and within 5cm vertically.
PLACE_XY_TOLERANCE    = 0.12

# Geometry guards.
MIN_PICK_WRIST_Z      = 0.10    # minimum IK target Z for the pick descent.
                                # With palm targeting + wrist_weight=0.5
                                # (side-grip mode), SLSQP uses HandBearing
                                # as a 5th reach DOF, swinging the palm
                                # 15-20 cm around the Hand_Bearing pivot.
                                # Effective palm floor is much lower —
                                # 0.10 m gives the
                                # IK headroom. Validity checker still
                                # rejects unreachable poses.
MAX_PICK_H_DIFF       = 0.18    # max |h2-h1| for DIAGONAL/TOP-DOWN
                                # accepted IK solutions
MAX_PICK_H_DIFF_SIDE  = 0.35    # max |h2-h1| for SIDE-GRIP solutions.
                                # Wider window than top-down because
                                # the side-grip IK cost biases toward
                                # boom tilt ~0.30 to keep the palm at
                                # object middle. The validity checker
                                # (planning-time chassis clearance)
                                # still rejects actual clip.
MIN_PICK_A1           = 0.16    # visual guard: avoid folding gripper into body

# ── Path B Step 1.5: hover→descent split ──────────────────────────────
# OMPL plans the arm approach with pickup objects parked in
# planning_data (so the planner can reach them). At runtime the
# cylinder IS at its real position, so a single high-to-low OMPL arc
# can side-swipe it. Split the approach into two phases:
# 1. OMPL: current pose → PRE_HOVER_Q (wrist HOVER_LIFT above grasp)
# 2. Kinematic: PRE_HOVER_Q → GRASP_Q (column-only motion → vertical
# Cartesian descent, no horizontal sweep)
# PRE_HOVER_Q is built by adding HOVER_LIFT to h1 AND h2 of GRASP_Q
# (a1 and th held constant). Equal-column bump keeps the platform
# level so the wrist rises purely vertically — no need for a second
# IK solve (avoids the "IK jumps to a different branch with worse
# deflection" issue seen during fine-IK shift testing).
COLUMN_JOINT_MAX      = 1.43    # XML range max for ColumnLeft/Right joints
HOVER_LIFT            = 0.30    # m above GRASP_Q wrist Z for the hover phase
HOVER_LIFT_RETRY      = 0.08    # m for column lift on POST-CLOSE failure
                                # retry — short hop is enough when only
                                # the fingers need to re-open and re-close.
HOVER_DESCENT_STEPS   = 45      # waypoints for PRE_HOVER_Q → GRASP_Q
                                # descent (~0.9 cm per step).
USE_HOVER_DESCENT     = True    # one-line revert if regression appears

# ── Path B Step 2: 3-finger contact verification ───────────────────────
# After the close ramp, require ALL 3 fingers to have physically
# contacted the object. If fewer fingers contacted, retract + fail this
# attempt so the existing base-XY retry can correct the alignment.
# After MAX_STRICT_3FINGER_ATTEMPTS strict attempts in one pick cycle
# (across multiple play_m1 retries), the bar relaxes to
# MIN_CONTACTS_RELAXED so the demo doesn't get stuck looping forever.
# Below MIN_CONTACTS_RELAXED is always a fail (1-finger held isn't a
# real grasp regardless of attempt count).
USE_3FINGER_VERIFY        = True
# Contact-count bars. Three fingers on a 4-5 cm cylinder with this
# 1+2 gripper geometry is rarely achievable across all approach yaws;
# accept two-finger wraps (thumb + one side, or both sides on opposite
# faces) as a real grasp. The relaxed bar after the strict-window
# expires drops one further so a marginal pickup can still complete.
MIN_CONTACTS_STRICT         = 2
MIN_CONTACTS_RELAXED        = 1
MAX_STRICT_3FINGER_ATTEMPTS = 3

# ── Hover-retract on pick failure ──────────────────────────────────────
# On gate / 3-finger failures, the previous behavior was to retract
# fully to carry pose (h1=0.60, h2=0.65, a1=0.10) before reporting
# failure. That meant every local retry then had to OMPL-plan the
# whole approach again — large motion, large chance of error. Hover-
# retract goes only to the previous PRE_HOVER_Q height (column
# +HOVER_LIFT above GRASP_Q, a1 and th held), so the next local retry
# starts from a pose that's already nearly aligned. Safe because
# local retries hold base yaw fixed; large nav arcs (next-candidate
# transitions) still call retract_to_carry() before the nav starts.
USE_HOVER_RETRACT_ON_FAIL = True
HOVER_RETRACT_STEPS       = 15  # column-only motion, no need for full 25
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

GRIPPER_OPEN_POS      = -0.55   # open-hand pre-grasp ctrl (sides b/c)
# The thumb hangs ~10 cm lower than the side fingers from the wrist
# and so needs a bigger open swing to clear the cylinder during
# descent. Paired with the 8-DOF OMPL planner the wrist tilts the
# whole gripper, so a moderate open angle is sufficient.
THUMB_OPEN_POS        = -1.05   # finger_a j1 open ctrl (qpos ≈ -60°)
THUMB_OPEN_J2         =  0.0     # thumb j2 at joint rest
THUMB_OPEN_J3         = -0.0523  # thumb j3 at joint rest
GRIPPER_CLOSE_POS     =  0.30   # full close (upper bound for the size-aware close)
GRIPPER_PIN_HOLD_POS  =  0.00   # legacy relaxed hold; superseded by size-aware hold

# Size-aware finger close mapping. Smaller objects need fingers closed
# more; larger objects need fingers stopping earlier so they appear to
# wrap around the cylinder surface.
# close_ctrl = FINGER_CLOSE_MAX - FINGER_CLOSE_PER_M * radius
# clamped to [FINGER_CLOSE_FLOOR, FINGER_CLOSE_MAX].
FINGER_CLOSE_MAX      = 0.20  #  max close ctrl (smallest object) (tested 0.25 — 1 test-lift passed but 7.3M N force spikes launched obj on retries)
FINGER_CLOSE_FLOOR    = 0.15    # min close ctrl across all object sizes
FINGER_CLOSE_PER_M    = 4.0     # ctrl backs off this fast as radius grows

# Smooth-attach interpolation: when the object pin switches from
# "pinned at world pose" to "pinned at gripper offset", interpolate over
# this duration instead of teleporting.
SMOOTH_ATTACH_SECS    = 0.7   # animation duration for the obj sliding
                              # into the gripper pocket. Runs in
                              # parallel with the close stroke so the
                              # motion appears caused by the closing
                              # finger contacts.
# Hold time after the pin is installed so the smooth-attach animation
# completes before the arm starts the lift.
SMOOTH_ATTACH_SETTLE  = 0.45

# Finger actuator layout: gripper_ids_left has 9 finger joint actuators
# arranged as 3 fingers x 3 joints (proximal, middle, distal).
# 0,1,2 = finger_c (j1, j2, j3)
# 3,4,5 = finger_b (j1, j2, j3)
# 6,7,8 = finger_a (j1, j2, j3)
# Indices 9, 10 are palm spread joints (palm_finger_c, palm_finger_b).
FINGER_BASE_INDICES   = (0, 3, 6)   # base index of each finger (j1)

# Curl profile: distal joints curl more than proximal so the fingertip
# wraps around the object surface.
# The 1+2 DELTO gripper has asymmetric mounting: increasing j1 swings
# the thumb (finger_a) backward into the wrist body but swings the
# side fingers (b, c) downward toward the object. A single global
# CURL_J1_FACTOR cannot satisfy both ends. Per-finger factors handle
# this — a moderate factor for the thumb, the same for the sides
# (HandBearing already tilts the gripper toward the object).
CURL_J1_FACTOR_THUMB_PERFECT  = 0.70
CURL_J1_FACTOR_THUMB_SOFTWELD = 0.95
CURL_J1_FACTOR_THUMB  = CURL_J1_FACTOR_THUMB_PERFECT
CURL_J1_FACTOR_SIDE   = 0.70    # finger_b / finger_c j1
CURL_J1_FACTOR        = CURL_J1_FACTOR_THUMB
CURL_J2_FACTOR        = 0.90    # middle (joint_2)
CURL_J3_FACTOR        = 1.00    # distal (joint_3) — full close

# Smooth gripper close/open animation: interpolate ctrl values over this
# duration instead of snapping.
SMOOTH_GRIPPER_SECS   = 0.4
SMOOTH_GRIPPER_STEP_S = 0.02    # per-step sleep — ~50 Hz interpolation

# STRICT-mode slow close. The 0.4 s close in mode 2/3 is paired with
# a pin that holds the obj in place during the close stroke — contact
# forces don't accumulate because the pin absorbs them. In STRICT
# mode there is NO pin: any contact force spike pushes the obj for
# real. At 0.4 s the close stroke builds force too fast for
# force-stop to catch it before the constraint solver applies a
# multi-thousand-N impulse on first contact. 4.0 s gives the contact
# solver plenty of time to build force gradually, letting force-stop
# fire at near N_req instead of multi-kN spikes (which knock obj).
STRICT_CLOSE_TRANSITION_SECS = 4.0  # gives the contact solver time to build force gradually so force-stop fires near N_req instead of multi-kN spikes (which kick obj during chassis-push-while-finger-touches).

# Feature flag: contact-stop close. When True, _set_gripper monitors
# per-finger contacts with the held object during the close interpolation;
# each finger freezes its joint ctrls the moment its geoms touch the
# cylinder. When False, falls back to position-only close.
USE_CONTACT_STOP_CLOSE = True

# Push-past-contact compression: after a finger first touches the held
# object, allow this many extra contact ticks before freezing its
# ctrl. At SMOOTH_GRIPPER_STEP_S = 0.02s/tick, the compression duration
# is COMPRESS_TICKS * 0.02s. Two effects:
# 1. Physical: the finger continues to PD-push toward the close target
# for this many ticks after first touching the object, producing a
# visible firm wrap rather than freezing at first contact (soft
# graze).
# 2. Safety: with a deeply-negative OPEN ctrl (e.g. THUMB_OPEN_POS =
# 1.30), an early-tick freeze would lock the thumb in the
# self-fold zone (qpos ~ -1.3+). Longer compression delays the
# freeze so the close ramp has time to escape the deep-negative
# region. Example for thumb j1 at -1.30 OPEN / +0.595 CLOSE:
# tick 2: ctrl ≈ -1.13 (UNSAFE — self-fold zone)
# tick 5: ctrl ≈ -0.85 (safer)
# tick 8: ctrl ≈ -0.45 (clearly safe)
CONTACT_COMPRESS_TICKS      = 5
FAST_CONTACT_COMPRESS_TICKS = 1

# Overdrive close ctrl: when True and USE_CONTACT_STOP_CLOSE is True,
# the close target is FINGER_CLOSE_MAX (100% intensity) instead of the
# size-aware mapping. Reason: with 1-3cm carry_gap, the finger nearest
# the cylinder hits and freezes immediately, but fingers on the far
# side of the cylinder reach their size-matched target and stop in
# mid-air before reaching the cylinder surface. Overdriving forces
# them to keep curling — the contact-stop catches them safely when
# they touch. FINGER_CLOSE_FLOOR=0.15 clamp in _finger_close_for_radius
# silently blocked the more obvious "shrink the target radius" trick,
# which is why overdriving FINGER_CLOSE_MAX is the right path.
USE_OVERDRIVE_CLOSE = True

# Per-finger diagnostic logging at key checkpoints (post-descent,
# pre-close, post-close). Logs each fingertip XYZ + delta to the
# object, plus any active finger-vs-object contacts with positions.
# Use to identify which finger is geometrically off-target when
# 3-finger contact fails. Verbose; turn off after diagnosis.
USE_FINGER_DIAGNOSTIC_LOG = True

# Silence per-waypoint logs from approach + kinematic_descent. When
# False, only the start ("approach: current → PRE_HOVER_Q") and end
# ("[approach] done") lines print, plus the [Diag] blocks. Drops
# ~70 verbose lines per pick attempt. Set True for trajectory debug.
VERBOSE_PATH_WAYPOINT_LOG = False

# Clamp finger actuator ctrl values to each joint's ctrlrange inside
# _set_gripper before writing to sim.data.ctrl. Prevents OMPL finger
# waypoints from commanding values that would drive joints past the
# soft-limit envelope.
USE_FINGER_CTRL_CLAMP = True
# Per-finger actuator ctrlrange (mirrors gripper_actuator_V2.xml).
# Order matches gripper_ids_left[0:11]:
# 0..2 = finger_c (j1, j2, j3)
# 3..5 = finger_b (j1, j2, j3)
# 6..8 = finger_a (j1, j2, j3)
# 9 = palm_finger_c
# 10 = palm_finger_b
# The j1 floors for the side fingers (-0.6) and thumb (-1.5) cap the
# OPEN ctrl just below its target so OMPL waypoint excursions cannot
# wind a joint past its declared range.
FINGER_CTRL_RANGES = (
    (-0.6,    1.2218),    # finger_c j1  ← floor to cap windup
    ( 0.0,    1.5708),    # finger_c j2
    (-1.2217, -0.0523),   # finger_c j3
    (-0.6,    1.2218),    # finger_b j1  ← floor to cap windup
    ( 0.0,    1.5708),    # finger_b j2
    (-1.2217, -0.0523),   # finger_b j3
    (-1.50,   1.2218),    # finger_a j1  ← matches rolled-back joint range (THUMB_OPEN_POS=-1.05)
    ( 0.0,    1.5708),    # finger_a j2
    (-1.2217, -0.0523),   # finger_a j3
    (-0.1784, 0.192),     # palm_finger_c
    (-0.192,  0.1784),    # palm_finger_b
)

# Synchronous open with settle gate. Step [3] drives the gripper to
# the OPEN ctrl values and then polls joint qpos until every joint is
# within OPEN_SETTLE_TOL_RAD of its target or OPEN_SETTLE_TIMEOUT
# expires. Ensures the arm doesn't begin descent before the fingers
# are physically out of the cylinder's path.
USE_SYNC_OPEN         = True
OPEN_SETTLE_TOL_RAD   = 0.20    # per-joint tolerance (rad)
OPEN_SETTLE_TIMEOUT   = 0.4     # wall-clock cap (seconds) — proceed anyway
OPEN_SETTLE_POLL_S    = 0.02    # poll period during settle

# Feature flag (currently disabled): when enabled, finger_c's j1 OPEN
# and CLOSE ctrls are swapped to compensate for a suspected mounting
# asymmetry. Live testing showed the underlying issue is contact-
# solver-side rather than ctrl-side, so the flag stays off.
MIRROR_FINGER_C_J1    = False

# Palm spread ctrl values. Idx 9 = palm_finger_c_joint_1, idx 10 =
# palm_finger_b_joint_1. Driving them wider during approach and back to
# narrow at grasp produces an "open hand reaching, then closing on object"
# choreography.
# Both palms use the same sign (+) by default. An earlier attempt with
# Both palms driven with the same sign — symmetric spread keeps the
# thumb axis aligned with the cylinder; signed-mirroring (e.g. PALM_B
# negative) per the raw XML ctrlrange skewed the gripper geometry so
# the thumb stopped contacting.
PALM_SPREAD_OPEN      = 0.18    # magnitude when gripper is open
PALM_SPREAD_CLOSE     = 0.0     # ctrl when gripper is grasping (centered)
PALM_C_SIGN           = +1.0
PALM_B_SIGN           = +1.0

# Per-finger stagger between successive fingers starting their close motion.
FINGER_STAGGER_SECS   = 0.06

GRIPPER_HOLD_TIME     = 1.5     # s — let fingers settle after open/close
PD_SETTLE_PER_WAYPOINT = 0.05   # s — pause between waypoints
PD_SETTLE_AT_PATH_END = 0.4     # s — pause after final waypoint for PD
                                # convergence (residual typically
                                # under 0.02 rad at this duration)


# ── Helpers ──────────────────────────────────────────────────────────────

def compute_grasp_targets(base_xy, obj_world, obj_radius=None,
                          side_approach=False):
    """
    Compute (grasp_target, pre_grasp_target) using the approach-vector
    standoff pattern.  Both virtual screening and the runtime executor
    must agree on this — otherwise screening passes but execution fails.

    Args:
        base_xy:        (x, y) of robot base (or candidate base for screening)
        obj_world:      (3,) object world position
        obj_radius:     optional object cylinder radius.  Retained for API
                        symmetry; the fixed standoff aligns fingertips with
                        the object surface across the current radius range.
        side_approach:  When True, the pre-grasp target is at obj height
                        (no above-Z offset) so the wrist arrives at the
                        obj's middle for a horizontal-palm grip.  When
                        False (top-down / diagonal), the pre-grasp target
                        sits ABOVE_OBJ_HEIGHT above the grasp target so a
                        column-only descent then lowers fingers onto the
                        obj from above.

    Returns:
        grasp_target:     IK target for GRASP_Q (palm at obj height,
                          pulled back effective_standoff along approach)
        pre_grasp_target: IK target for PRE_GRASP_Q.  Same as grasp_target
                          when side_approach=True, else +ABOVE_OBJ_HEIGHT.
    """
    obj = np.asarray(obj_world, dtype=float)
    base_xy = np.asarray(base_xy, dtype=float)
    approach = obj[:2] - base_xy
    nrm = float(np.linalg.norm(approach))
    if nrm < 1e-6:
        approach_unit = np.array([1.0, 0.0])
    else:
        approach_unit = approach / nrm
    # Effective standoff scales with obj radius so palm never lands inside
    # the obj surface. For thin objs (radius ≤ 0.03), GRIPPER_STANDOFF_XY
    # dominates and behaviour matches the historical fixed-standoff path.
    # For thicker objs the standoff grows to keep the palm-to-surface
    # gap = STANDOFF_RADIUS_CLEAR.
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
        # initialize STRICT-mode state attributes here so
        # they exist before any code path that reads them — including
        # the tune_grip_lift harness which calls _strict_lift_with_retry
        # directly without first going through pick(). pick() resets
        # these per-cycle (at line ~4953-4961); these defaults cover
        # callers that bypass pick().
        self._strict_force_multiplier = 1.0
        self._strict_retry_count      = 0
        self._strict_grasp_q          = None
        # stage classification (an earlier review). Tracks
        # which pipeline stages the current cycle reached so play_m1
        # can emit a single [STAGE-CLASS] line per cycle classifying
        # A/B/C/D for fast triage of where cycles fail.
        # A = pre-close reject (geometry/asymmetry/arm-clip — close never fired)
        # B = verify fail (close fired, verify rejected)
        # C = lift slip (verify passed, test-lift failed)
        # D = lift success (obj followed grip)
        # Reset at pick() entry.
        self._cycle_stage_close_fired   = False
        self._cycle_stage_verify_passed = False
        self._cycle_stage_lift_fired    = False
        self._cycle_stage_obj_followed  = False

        # Resolve weld + body indices once
        self.weld_id = mujoco.mj_name2id(
            sim.model, mujoco.mjtObj.mjOBJ_EQUALITY, "grasp_left")
        if self.weld_id < 0:
            raise RuntimeError("grasp_left weld not found in model")

        # Gripper_Link3_1 is the palm body where the three fingers attach
        # via their palm_finger_*_joint_1 joints. Used as the pin
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

        # Per-finger body groups for contact-stop close. Order matches
        # gripper_ids_left:
        # index 0 = finger_c (gripper joints 0-2)
        # index 1 = finger_b (gripper joints 3-5)
        # index 2 = finger_a (gripper joints 6-8)
        # finger_a has no link_0_1 in this model (only b and c have a
        # palm-spread joint). We probe link_0_1 through link_3_1 and
        # accept whatever exists; the tip body (link_3_1) is the most
        # important and is present on all three fingers.
        self._finger_body_groups = []
        finger_labels = ['c', 'b', 'a']
        for label in finger_labels:
            bodies = set()
            # Contact-stop monitors the mid and tip links of each
            # finger (link_2 and link_3). Including the proximal link
            # caused asymmetric early-freeze of one finger while the
            # others over-closed past their joint maxes.
            link_indices = (2, 3)
            for ln in link_indices:
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

        # OMPL motion planner for the gripper's 11 finger DOFs. The grasp
        # closure / open trajectory is generated via this planner; see
        # navigation/finger_planner.py for the state space and planner
        # setup. _set_gripper walks the returned waypoints with
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
        self._pre_close_lift_done = False # [5.7] pre-close smooth-lift installed pin
        self._fast_fixed_close_pin_active = False
        self._grasp_offset_quat = (1.0, 0.0, 0.0, 0.0)
        # Gravity compensation: while held, set body_gravcomp[obj_bid] = 1.0
        # so MuJoCo cancels gravity for this body. Without this, gravity
        # drifts the object between pin snaps and causes visible jitter.
        self._held_obj_orig_gravcomp = None
        # While the object is pinned to the gripper, soften its contact
        # solver references (solref/solimp) so fingers stop at the
        # surface without fighting the pin solver. Saved as a list of
        # (geom_id, solref_copy, solimp_copy) for restoration on release.
        self._held_obj_solref_saved = None

        # Path B Step 2: 3-finger verification state.
        # `_last_close_finger_contacts` is a 3-bool snapshot of which
        # fingers physically contacted the object during the most recent
        # close (set by _set_gripper). `_strict_finger_attempts_used`
        # counts how many strict-3-finger pick attempts have happened in
        # the current pick cycle — after MAX_STRICT_3FINGER_ATTEMPTS, the
        # bar relaxes to MIN_CONTACTS_RELAXED so the cycle can finish
        # rather than loop forever. play_m1 must call
        # reset_finger_attempt_counter() at the start of each MOVE cycle.
        self._last_close_finger_contacts = None
        self._strict_finger_attempts_used = 0

        # Active pin callback (function reference, registered with sim)
        self._active_pin_fn = None
        self._cancel = False

        # Pre-close diagnostics captured on gate rejection; consumed by
        # play_m1's closed-loop base correction.
        self.last_grasp_failure_info = None

        # Warm-start seed for retry IK. Set when an IK call succeeds in
        # `_pick_run`; passed as `seed_q` on subsequent `_pick_run` calls
        # within the same retry cycle so SLSQP starts in a known-feasible
        # basin (avoids `best_err=inf` collapse seen with cold seeds on
        # marginal local-retry poses). Persists across `_pick_run`
        # invocations until reset by a successful FULL pick.
        self._last_valid_pre_grasp_q = None

        # Grip-mode flag read by `_curl_targets` to pick thumb open
        # angle: side-grip uses symmetric thumb (j1=GRIPPER_OPEN_POS),
        # top-down uses wider THUMB_OPEN_POS for descent clearance.
        # Set per-pick by `_pick_run` after `wrist_goal` is computed.
        self._side_grip_active = False

        # Local-retry tracking: when a side-grip attempt rejects and
        # the arm is held at GRASP_Q (skipping hover-retract), the next
        # `pick()` call's `is_local_retry=True` path performs a
        # chassis-only back-then-forward maneuver instead of replanning.
        self._arm_held_at_grasp_for_retry = False

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

        # One-shot diagnostic at startup: for each finger joint, print
        # the joint type, range, limited flag, and qpos address. This
        # tells us definitively whether MuJoCo is treating the thumb
        # joints differently from b/c joints (e.g., not limited, or a
        # different joint type that uses multiple qpos slots).
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

        # Read-only startup probe to verify gripper state immediately
        # after sim.reset("home") + keyframe load. Reports:
        # (a) current qpos for every finger + palm joint, right after
        # sim.reset("home") and the keyframe load. Reveals
        # whether the joint sits at 1.4 from the keyframe itself
        # or only ends up there after a ctrl write.
        # (b) current ctrl for every finger + palm actuator. Should
        # match the keyframe's `ctrl="…"` line (all zeros).
        # (c) any active contact pair whose body name contains one
        # of the finger_c link bodies — tells us if the parked
        # arm-2 or another robot body is touching finger_c at
        # rest, which would explain the joint-trapping torque.
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

    def _pinch_midpoint_xyz(self, data):
        """World XYZ of the pinch midpoint — halfway between the thumb
        tip (finger_a) and the centroid of the 2 side fingers (b, c).

        For SIDE-GRIP alignment: in a correct side-grip pose the thumb
        is on ONE side of the obj and the 2 fingers on the OPPOSITE
        side, with the obj between them.  The geometric midpoint of
        (thumb_tip, bc_centroid) is where the obj should sit.  The
        plain 3-fingertip centroid (`_carry_anchor_xyz`) gets pulled
        toward the thumb side when the gripper is wide-open and gives
        a misleading "I'm far from obj" reading even when the obj is
        correctly between the fingers.

        carry_anchor_body_ids order: [finger_a (thumb), finger_b, finger_c]
        """
        if len(self._carry_anchor_body_ids) != 3:
            return data.xpos[self.gripper_body_id]
        thumb = data.xpos[self._carry_anchor_body_ids[0]]
        bc_centroid = 0.5 * (data.xpos[self._carry_anchor_body_ids[1]]
                             + data.xpos[self._carry_anchor_body_ids[2]])
        return 0.5 * (thumb + bc_centroid)

    def _log_finger_geometry(self, obj_bid, label):
        """Diagnostic: log each fingertip's world XYZ, offset from the
        held object, AND per-joint qpos (proximal j1 / middle j2 /
        distal j3) for all 3 fingers.  Use at key checkpoints
        (post-descent, pre-close, post-close) to identify WHICH
        finger is geometrically off-target AND WHICH JOINT angle is
        moving in the unexpected direction.

        carry_anchor_body_ids order is (a, b, c) per init loop:
            ("finger_a_link_3_1", "finger_b_link_3_1", "finger_c_link_3_1")

        Joint qpos: per gripper_actuator_V2.xml, the actuators map to
        joints in order [c_j1, c_j2, c_j3, b_j1, b_j2, b_j3,
        a_j1, a_j2, a_j3, palm_c, palm_b].  Reading
        sim.data.qpos at these joint addresses gives the actual
        angle the joint is at (NOT the ctrl target).  If a finger's
        tip is moving the wrong direction, comparing j1 vs j2 vs j3
        actual angles across attempts shows which joint is the
        culprit.
        """
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
        # Per-joint qpos. Look up each joint by name, read its qposadr,
        # print the actual angle. Tells us if the curl is curling in
        # the expected direction per finger.
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
        # Palm spread joints
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
        # Wrist orientation diagnostics — Phase 3 recalibration of
        # GRASP_OFFSET_Z_MIN and target_z depends on knowing the actual
        # palm-vs-fingertip Z relationship at the new tilted pose. Log
        # the 4 wrist qpos along with palm body Z.
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
        # Tip Z vs palm Z — the key offset for re-tuning target_z.
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
        """Count current arm/gripper ↔ chassis contacts.  Used by the
        [5.5b] nudge loop as a safety abort: if a nudge produces MORE
        arm-vs-chassis contacts than before, the arm is starting to
        clip the base and further nudges in that direction will only
        make it worse.  Returns 0 if helpers are unavailable so the
        loop falls back to convergence-only checks.

        Mirrors the logic in `_log_filtered_contact_summary` but
        returns just the count (no print, no obj-related bookkeeping).
        """
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
            # Exclude pickup obj bodies (those are counted elsewhere)
            # and only count what's a chassis/base structural body.
            if other_name.startswith("pickup_obj_"):
                continue
            count += 1
        return count

    def _ensure_arm_subtree_body_ids(self):
        """Cache and return the set of body IDs for every descendant of
        the "Arm_1" body (the structural arm bodies plus the finger and
        palm chain).  Used by `_count_arm_obj_contacts` to detect when
        any part of the arm chain clips the target object at pre-close.
        """
        if getattr(self, '_arm_subtree_body_ids_cache', None) is not None:
            return self._arm_subtree_body_ids_cache
        model = self.sim.model
        root_bid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "Arm_1")
        ids = set()
        if root_bid >= 0:
            ids.add(int(root_bid))
            # BFS over descendants
            frontier = [int(root_bid)]
            while frontier:
                cur = frontier.pop(0)
                for bid in range(model.nbody):
                    if int(model.body_parentid[bid]) == cur:
                        if int(bid) not in ids:
                            ids.add(int(bid))
                            frontier.append(int(bid))
        else:
            # Fall back to the narrower gripper-only set as a safety
            # net; better than empty.
            try:
                ids = set(self._ensure_gripper_body_ids().keys())
            except Exception:
                pass
        self._arm_subtree_body_ids_cache = ids
        return ids

    def _runtime_wz_correction(self, obj_bid, ik_base_xy_2d):
        """Live wrist-Z correction at runtime (post-descent + chassis-push).

        Iterates UP TO `WZ_RUNTIME_CORRECTION_MAX_ITERS` times:
        each iteration measures the thumb→bc axis error vs the
        approach-perpendicular target, commands a wrist_Z step toward
        alignment, settles, and re-measures.  Continues if reach
        improved; reverts that step and exits if it regressed; exits
        early on success (axis err < threshold).

        Up to three passes typically converge to ≤ 3° axis error,
        putting both finger sides within close-stroke reach.  Returns
        a dict summarising the cumulative outcome.
        """
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

            # State accepted across iterations
            best_ctrl = float(data.ctrl[gids[GIDS_WRIST_Z]])
            best_err = err_o
            best_max_far = max_far_o
            best_carry = carry_o
            iters_applied = 0

            for iter_i in range(int(WZ_RUNTIME_CORRECTION_MAX_ITERS)):
                # Cap each step's rotation so we don't oversweep.
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
                # Axis-error guard: only accept a wz step that keeps
                # the axis error from growing past a settle-noise
                # margin (≈ 3°). Without this guard, the loop would
                # accept reach-improving steps that rotate the wrap
                # axis away from the desired direction.
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
                    # Early exit when axis err is now small enough.
                    if abs(best_err) <= WZ_REFINE_AXIS_ERR_THRESHOLD:
                        print(f"[Exec] [5.46] converged after "
                              f"{iters_applied} iter(s); axis err "
                              f"{math.degrees(best_err):+.1f}° "
                              f"≤ {math.degrees(WZ_REFINE_AXIS_ERR_THRESHOLD):.0f}° "
                              f"threshold")
                        break
                else:
                    # Revert THIS iteration only; keep best so far.
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
        """Count arm/gripper-structure ↔ TARGET obj contacts with
        penetration depth >= `min_penetration` metres, EXCLUDING
        finger bodies.

        Finger-obj contact is expected during the close stroke (and
        at pre-close is a sign close is about to grip).  ARM-body-obj
        contact (boom, palm, wrist) at pre-close is BAD — the arm
        structure is clipping obj before the gripper has a chance to
        grasp it.

        `data.contact[i].dist` is the signed distance: negative for
        overlap (penetration).  The `min_penetration` filter excludes
        shallow 1-3 mm grazes (which runtime physics absorbs without
        visual artifact) while still catching real structural clips.
        """
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
            # Filter out shallow grazing contacts.
            if cdist >= -float(min_penetration):
                continue
            if (b1 in non_finger_arm and b2 == int(obj_bid)) or \
               (b2 in non_finger_arm and b1 == int(obj_bid)):
                count += 1
        return count

    def _count_finger_obj_contacts(self, obj_bid):
        """Count how many distinct finger BODIES (out of c, b, a) are
        currently in contact with the obj body.  Returns 0-3.

        Used by the REALISM pre-close gate to verify that physical
        contact exists BEFORE pinning/lifting — separates "real grasp"
        from "magic pin from a distance"."""
        data = self.sim.data
        model = self.sim.model
        try:
            n = int(data.ncon)
        except Exception:
            return 0
        touched = [False, False, False]   # c, b, a
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
        """One-line breakdown of contact categories so the user can
        correlate the visual yellow contact dots with what's actually
        being touched.  Defaults of mjVIS_CONTACTPOINT show ALL
        contacts (47+ on this scene), mostly obj↔floor and wheel↔floor
        — confusing if you're looking for the grasp.  This summary
        names the only contacts that matter for the pick:

            finger_a/b/c ↔ obj   : N each
            arm-vs-chassis       : N
            arm-vs-other-obj     : N  (collisions with non-target objs)
            obj-vs-floor         : N
        """
        data = self.sim.data
        model = self.sim.model
        try:
            n = int(data.ncon)
        except Exception:
            return
        # Build sets we care about.
        finger_groups = self._finger_body_groups or []
        finger_names = ['c', 'b', 'a']
        # All arm/gripper body ids (for chassis-clip count) — reuse the
        # cache built by `_ensure_gripper_body_ids` if present.
        try:
            arm_bid_set = self._ensure_gripper_body_ids()
        except Exception:
            arm_bid_set = set()
        # World body id is 0; "floor" is anything in world geom group 0
        # whose Z position is near 0. For simplicity, count contacts
        # where one body is 0 (world).
        # Target obj is obj_bid; "other objs" are pickup_obj_* not equal
        # to obj_bid. Detect by body-name prefix.
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
            # finger ↔ obj
            for fi, fname in enumerate(finger_names):
                fb = finger_groups[fi] if fi < len(finger_groups) else set()
                if not fb:
                    continue
                if ((b1 in fb and b2 == int(obj_bid)) or
                        (b2 in fb and b1 == int(obj_bid))):
                    per_finger[fname] += 1
            # obj ↔ floor/world (one side is body 0)
            if (b1 == int(obj_bid) and b2 == 0) or \
               (b2 == int(obj_bid) and b1 == 0):
                n_obj_floor += 1
            # arm-vs-chassis: one side is arm/gripper body, other is
            # "base" or similar non-arm non-target.
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
                        pass  # already counted via finger or general
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
        """Diagnostic: scan sim.data.contact[] for any contact pair
        between a finger body and the held object body.  Logs each
        contact with the world position, normal, and which finger
        chain is touching.  Use to detect "finger nudges cylinder
        before close even starts" cases."""
        data = self.sim.data
        model = self.sim.model
        try:
            n = int(data.ncon)
        except Exception:
            return
        finger_names = ['c', 'b', 'a']  # _finger_body_groups order
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

    def _side_grip_chassis_push(self, target_xy_world, obj_xy_2d,
                                timeout=2.0, dist_tol=0.025,
                                abort_on_finger_obj_contact=False,
                                obj_bid=None):
        """Push the chassis to `target_xy_world` while the arm holds its
        current pose.  Used between [5] descent and [6] close in side-grip
        mode: the IK planned the arm for this future chassis position;
        this step makes the chassis catch up.

        The target XY encodes BOTH forward distance AND L/R alignment
        with the obj (target = obj − push_dist × approach_unit), so the
        gripper lands precisely on the obj's XY.

        Sets `sim.target_base` (the attribute the chassis PID controller
        reads) and polls localization until the chassis converges or the
        timeout expires.

        `abort_on_finger_obj_contact`: if True, monitor finger-vs-obj
        contacts during the push.  As soon as ANY finger touches obj,
        STOP the push and freeze chassis at current position.  This
        prevents the local-retry's forward push from ramming the arm
        deeper into obj after first contact (which knocks obj away).
        Caller must supply `obj_bid` when enabling this.
        """
        loc = self.sim.localization()
        base_xy = np.asarray([loc[0], loc[1]], dtype=float)
        target_xy = np.asarray(target_xy_world, dtype=float)
        obj_xy = np.asarray(obj_xy_2d, dtype=float)
        start_dist = float(np.linalg.norm(obj_xy - base_xy))
        # The chassis yaw target is the same direction it has now —
        # pointing toward the obj. No yaw change during the push.
        target_yaw = float(math.atan2(obj_xy[1] - target_xy[1],
                                      obj_xy[0] - target_xy[0]))
        contact_guard = bool(abort_on_finger_obj_contact and obj_bid is not None)
        print(f"[Exec] [5.4] side-grip chassis push: "
              f"({base_xy[0]:.3f},{base_xy[1]:.3f}) → "
              f"({target_xy[0]:.3f},{target_xy[1]:.3f})  "
              f"dist_to_obj {start_dist:.3f}m → "
              f"{float(np.linalg.norm(obj_xy - target_xy)):.3f}m"
              + (f"  [contact-guard ON]" if contact_guard else ""))
        with self.sim._target_lock:
            self.sim.target_base = np.array([float(target_xy[0]),
                                             float(target_xy[1]),
                                             target_yaw])
        import time as _time
        t0 = _time.time()
        last_diag = t0
        aborted_on_contact = False
        while _time.time() - t0 < timeout:
            cx, cy, _ = self.sim.localization()
            if math.hypot(cx - target_xy[0], cy - target_xy[1]) <= dist_tol:
                break
            if contact_guard:
                any_contact = False
                contact_reason = ""
                # Pre-compute obj geometry for the geometric guard
                _obj_r_guard = float(self._object_radius(obj_bid))
                _obj_hh_guard = float(self._object_half_height(obj_bid))
                _obj_xyz_guard = self.sim.data.xpos[obj_bid].copy()
                # tighter relief — only abort on REAL
                # penetration (≥ 1 cm inside the obj surface), not a
                # grazing surface touch. Combined with the
                # pinch-midpoint stop criterion below, the chassis
                # advances until the gripper is geometrically centred
                # around the obj — instead of stopping the instant
                # the forward-mounted thumb grazes the surface.
                PENETRATION_DEPTH_GUARD = 0.010   # m — abort if tip is ≥ 1 cm INSIDE obj surface
                if STRICT_PERFECT_FRICTION_ONLY:
                    BC_REACH_STOP_MARGIN    = 0.015  # 4cm -> 1.5cm: bc lands ~7.7cm from obj center
                    THUMB_SAFETY_MARGIN     = 0.015  # 2cm -> 1.5cm: thumb lands ~7.7cm; less penetration risk than 1cm
                else:
                    BC_REACH_STOP_MARGIN    = 0.040  # soft-weld baseline
                    THUMB_SAFETY_MARGIN     = 0.020  #  5mm caused 1.7cm thumb-in-obj; 20mm keeps thumb at surface
                bc_stop_dist    = (_obj_r_guard + BC_REACH_STOP_MARGIN)
                thumb_stop_dist = (_obj_r_guard + THUMB_SAFETY_MARGIN)
                try:
                    if (self._carry_anchor_body_ids
                            and len(self._carry_anchor_body_ids) == 3):
                        # carry_anchor_body_ids order: [a, b, c]
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
                # Safety backstops: real finger penetration ≥ 1 cm into
                # the obj. A grazing surface touch (was the old abort
                # trigger) is now allowed.
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
                    # Freeze chassis target at current position so it
                    # stops moving forward.
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
        fx, fy, _ = self.sim.localization()
        final_dist = float(np.linalg.norm(np.array([fx, fy]) - obj_xy))
        moved = math.hypot(fx - base_xy[0], fy - base_xy[1])
        print(f"[Exec] [5.4] chassis push done: base-obj dist={final_dist:.3f}m "
              f"(moved {moved*100:.1f}cm in {_time.time()-t0:.1f}s)"
              + (f"  [aborted on contact]" if aborted_on_contact else ""))

    def _wait_for_wrist_settle(self, tolerance=0.05, timeout=2.0,
                                label="wrist-settle",
                                intended_targets=None):
        """Poll wrist qpos until each of the 4 wrist joints is within
        `tolerance` (rad) of its INTENDED IK target, or `timeout`
        elapses.  Returns the final residuals as (hb, wz, wx, wy).

        Why this matters: descent and chassis push both happen while the
        wrist is still actively rotating to its commanded orientation
        (e.g. wz=-1.88 for side-grip).  The PD controller has finite
        bandwidth — at kp=100 the wrist takes ~1s to overcome gravity +
        finger inertia after a large ctrl step.  Without this settle,
        the pre-close gate measures pinch midpoint BEFORE the wrist has
        actually rotated to the commanded pose.

        Critical: compare qpos against the INTENDED IK target, not the
        ctrl value.  The wz PD pre-compensation overshoots ctrl by 9 %
        (e.g. target -1.88 → ctrl -2.05) so qpos lands at the target.
        Comparing qpos to ctrl would always report 9 % residual and
        spuriously time out even though the wrist is correctly placed.

        Args:
            intended_targets: (hb, wz, wx, wy) — the original IK-planned
                values BEFORE any PD pre-compensation.  When provided,
                this is used for the comparison.  When None, falls back
                to reading ctrl (legacy behavior).
        """
        import time as _time
        model = self.sim.model
        data = self.sim.data
        gids = self.sim.gripper_ids_left
        if len(gids) <= GIDS_HANDBEARING:
            return None

        # Map: (joint name, ctrl gids index, short label, intended-index).
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
            # Prefer the intended IK target (correct for PD pre-comp);
            # fall back to ctrl read if caller didn't supply it.
            if intended_targets is not None:
                targets.append(float(intended_targets[intended_idx]))
            else:
                targets.append(float(data.ctrl[gids[gidx]]))
            labels.append(slabel)

        t0 = _time.time()
        last_diag = t0
        residuals = [float('inf')] * len(qpas)
        # Steady-state detection: if worst residual hasn't moved by more
        # than `ss_delta` over `ss_window` seconds, the PD controller is
        # at equilibrium — waiting longer is pointless. This is the
        # common failure mode: kp=100 wrist can't overcome gravity +
        # finger inertia, settles at ~0.17 rad off target permanently.
        ss_history = []          # (timestamp, worst) samples
        ss_window  = 0.30        # seconds of "no progress" before break
        ss_delta   = 0.003       # rad — change considered "no progress"
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
            # Drop samples older than the window
            ss_history = [(t, w) for (t, w) in ss_history
                          if now - t <= ss_window]
            # Steady state: have samples spanning the full window AND
            # worst-vs-worst-back-then changed by less than ss_delta.
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


    def _post_descent_extra_tilt(self, q_now, h2_delta=0.25, a1_delta=0.15,
                                  n_steps=15):
        """Stage 2 of the tilt strategy: after the standard small-tilt
        descent + chassis push (palm at ~obj_top), ramp h2 and a1 to
        add MORE boom tilt + arm extension so palm drops toward obj
        middle.  Done as a kinematic interpolation — no OMPL planning
        required because we're already next to the obj and the only
        motion is local arm articulation.

        Why staged instead of "just make IK find the high-tilt pose
        directly": OMPL planning from HOME→PRE_GRASP needs valid
        path waypoints.  If PRE_GRASP itself is a high-tilt pose, the
        intermediate waypoints would pass through chassis-clipping
        configurations.  By splitting into [small-tilt approach] +
        [post-descent extra tilt], the OMPL path is safe and the final
        low palm pose is reached as a short local motion.

        Args:
            q_now:    Current 8-vec arm state (post-descent values).
                      h1 stays as-is; h2 increases by h2_delta; a1
                      increases by a1_delta.  Wrist (q[4:8]) unchanged.
            h2_delta: Additional column-right height (m).  +0.25 means
                      h2 ramps up 25 cm, boom tilts more.
            a1_delta: Additional arm extension (m).
            n_steps:  Interpolation waypoints for the ramp.

        Returns the final commanded q.  Caller should check geometry
        afterwards.
        """
        q_target = list(q_now)
        # h2 (index 1) — clamp to both ends of joint range
        h2_min, h2_max = JOINT_RANGES_ARM[1]
        q_target[1] = max(float(h2_min),
                          min(float(q_now[1]) + h2_delta, float(h2_max)))
        # a1 (index 2) — clamp to both ends. a1_delta can be NEGATIVE
        # to pull the arm in (counter the outward palm shift caused by
        # boom tilt + avoid Arm_Left clipping the chassis).
        a1_min, a1_max = JOINT_RANGES_ARM[2]
        q_target[2] = max(float(a1_min),
                          min(float(q_now[2]) + a1_delta, float(a1_max)))
        # h1, th, wrist unchanged

        print(f"[Exec] [5.46] extra-tilt: h2 {q_now[1]:.3f}→{q_target[1]:.3f} "
              f"(+{q_target[1]-q_now[1]:.3f}), "
              f"a1 {q_now[2]:.3f}→{q_target[2]:.3f} "
              f"(+{q_target[2]-q_now[2]:.3f})  "
              f"[h1, th, wrist held]")
        self._kinematic_descent(q_now, q_target, "extra-tilt",
                                n_steps=n_steps)
        return q_target


    def _log_gripper_floor_chassis_contacts(self, label):
        """Diagnostic: list every active contact between the gripper /
        arm chain and either (a) the floor (world geoms with z near 0)
        or (b) the chassis body.  Used to debug "gripper hits ground"
        and "arm hits robot body" symptoms — pinpoints which finger /
        arm geom is the culprit and where.

        Detection:
          - One side: a body whose name starts with 'finger_', 'Gripper_',
            'Arm_', 'Hand_Bearing', 'Rotation_Link', 'Column_',
            'Bearing_Column' (the full arm + gripper chain)
          - Other side: world (body 0) or 'robot' / 'base' / 'chassis'
            / 'Lifepo4' / 'summit' (any non-arm structural body)
        """
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
                # both arm or neither — not arm-vs-world
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
        """Cache the body ids for arm-1's full finger chain + palm so
        the contact scan can filter quickly without name lookups every
        call."""
        if self._gripper_body_ids_cache is not None:
            return self._gripper_body_ids_cache
        model = self.sim.model
        names = (
            # finger c chain (arm-1)
            'finger_c_link_0_1', 'finger_c_link_1_1',
            'finger_c_link_2_1', 'finger_c_link_3_1',
            # finger b chain
            'finger_b_link_0_1', 'finger_b_link_1_1',
            'finger_b_link_2_1', 'finger_b_link_3_1',
            # finger a (thumb) chain
            'finger_a_link_1_1', 'finger_a_link_2_1', 'finger_a_link_3_1',
            # palm body the fingers attach to
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
        """Scan sim.data.contact[] and print every contact pair where
        at least one body is in the arm-1 gripper chain (finger a/b/c
        links + palm links).  Used to identify which contacts are
        producing the sustained torques that fight PD during open /
        close.  Read-only; optionally calls mj_forward first to refresh
        the contact list from current qpos."""
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
            # Look up names — use cache for gripper bodies, fall back
            # to mj_id2name for the other side.
            n1 = gripper_ids.get(b1) or (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b1)
                or f"#{b1}")
            n2 = gripper_ids.get(b2) or (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b2)
                or f"#{b2}")
            pos = c.pos.copy() if hasattr(c, 'pos') else None
            pos_str = (f"  pos={pos.round(3)}"
                       if pos is not None else "")
            # Self-contacts inside the gripper are the interesting ones
            # (palm ↔ finger, finger ↔ finger). External contacts (arm,
            # cylinder) print too but with a less alarming prefix.
            kind = "SELF" if (in_gripper_1 and in_gripper_2) else "ext "
            print(f"  [{kind}] {n1}  ↔  {n2}{pos_str}")
            hits += 1
        if hits == 0:
            print("  (no gripper contacts)")

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

    def _strict_log(self, phase, msg):
        """Timestamped STRICT-mode log line.  No-op when STRICT mode is
        off so mode 2 / mode 3 see byte-identical stdout.  Phase is a
        short label (CLOSE / LIFT / RETRY / etc.); the timestamp is
        wall-clock seconds since the current STRICT pick attempt began."""
        if not STRICT_PICKUP_MODE:
            return
        try:
            t0 = getattr(self, '_strict_t0', None)
            t = (time.time() - t0) if t0 is not None else 0.0
        except Exception:
            t = 0.0
        print(f"[STRICT t={t:7.3f}s] {phase:<14s} {msg}")

    def _finger_contact_force(self, finger_idx, obj_bid):
        """Return (normal_force_sum_N, n_contacts) for contacts pairing
        finger `finger_idx`'s bodies with obj_bid.  Normal force is the
        first component of the 6-axis constraint force in contact frame
        (mj_contactForce convention)."""
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
        """return [N_a, N_b, N_c] — per-finger normal
        force (sum of normals across all finger↔obj contacts).
        Caller order: thumb (a), b, c — matches j1 ctrl indices
        (a_j1=6, b_j1=3, c_j1=0)."""
        n_a, _ = self._finger_contact_force(2, obj_bid)
        n_b, _ = self._finger_contact_force(1, obj_bid)
        n_c, _ = self._finger_contact_force(0, obj_bid)
        return [float(n_a), float(n_b), float(n_c)]

    def _contact_normal_triad(self, obj_bid):
        """compute how 'balanced'
        the per-finger contact forces are.

        For each finger, sum the contact-frame normals (geom1 → geom2)
        weighted by their normal-force magnitudes.  The vector sum
        across all 3 fingers measures opposing-pinch quality: a true
        wrap-around grasp produces forces that cancel (sum ≈ 0); a
        one-sided grasp produces a large net push.

        Returns dict:
          * balance_score: |Σ F_i n_i| / max(|F_i|)  ∈ [0, 1+]
              0   = perfect opposing cage (good)
              ~1  = all force pushes obj in one direction (bad)
          * per_finger_force: [N_a, N_b, N_c]
          * per_finger_dir:   [3-vec, 3-vec, 3-vec]  unit force-direction
            into the obj (None when no contact).
        """
        data = self.sim.data
        model = self.sim.model
        force6 = np.zeros(6, dtype=np.float64)
        per_finger_dirs = [None, None, None]
        per_finger_forces = [0.0, 0.0, 0.0]
        # Iterate in slot order: 0 → c, 1 → b, 2 → a (thumb). Output
        # order will be reordered to [a, b, c] at the end to match
        # the rest of the codebase.
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
                    # Contact normal points geom1 → geom2 (world frame).
                    # We want force direction acting from finger onto obj.
                    normal = np.array([
                        float(c.frame[0]),
                        float(c.frame[1]),
                        float(c.frame[2])
                    ])
                    if b1 == int(obj_bid):
                        normal = -normal   # flip so normal points into obj
                    net += Fn * normal
                    net_mag += Fn
                except Exception:
                    continue
            if net_mag > 1e-6:
                per_finger_dirs[slot] = net / net_mag
                per_finger_forces[slot] = net_mag
        # Reorder slot→finger as [a, b, c] for the dict output.
        # finger_body_groups slot 0=c, 1=b, 2=a (matches the rest of the
        # file's _finger_touches_obj indexing).
        order = (2, 1, 0)  # a, b, c
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
        """emit a one-line GRASP_DIAG capture summarising
        the full grasp state at verify-time.  Called from BOTH the
        verify_OK and verify_FAIL branches so we can compare working
        vs non-working (chassis × wrist × geometry) combos.

        Format (CSV-friendly, single line):
        [GRASP_DIAG] outcome=... obj_xy=(X,Y) obj_r=R chassis_xy=(X,Y)
         chassis_yaw_deg=D approach_yaw_deg=D yaw_offset_deg=D
         wrist=(hb,wz,wx,wy) d_thumb=X d_b=X d_c=X
         N=(a,b,c) contacts_abc=(T/F,T/F,T/F) balance=B mode=SIDE/TOP

        `outcome` is one of: "verify_OK", "verify_FAIL", or any other
        future label.
        """
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
            # Normalise yaw offset to [-180, 180]
            while yaw_offset_deg > 180.0:
                yaw_offset_deg -= 360.0
            while yaw_offset_deg < -180.0:
                yaw_offset_deg += 360.0
            # Wrist qpos
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
            # Per-finger distances to obj (XY). Anchor body order is
            # [a, b, c] per `_carry_anchor_body_ids`.
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
            # Per-finger normal forces [a, b, c]
            try:
                per_N = self._per_finger_normal_forces(obj_bid)
            except Exception:
                per_N = [float('nan')] * 3
            # Triad balance score
            try:
                _triad = self._contact_normal_triad(obj_bid)
                balance = float(_triad.get('balance_score', float('nan')))
            except Exception:
                balance = float('nan')
            # contacts_snapshot order is [c, b, a]
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
        """Apply arm joint vector `q` to planning_data with fingers
        OPEN, run mj_forward, and read the world-frame XY offset
        between the gripper palm (Gripper_Link3_1) and the pinch
        midpoint `(thumb_tip + (b_tip + c_tip)/2) / 2`.

        Returned offset = `pinch_midpoint_xy - palm_xy`.  The caller
        subtracts this from the desired pinch-midpoint-at-target
        world position to get the palm IK target.

        Unlike `_estimate_pinch_midpoint_palm_offset_xy`, this probe
        uses the CALLER-supplied converged arm pose `q` instead of
        a fixed seed.  Eliminates the seed-mismatch error that the
        fixed-seed probe suffered when SLSQP picked a different arm
        family than the probe assumed.

        Restores planning_data qpos after the probe.  Returns None on
        failure.
        """
        try:
            _model = self.arm_bridge.model
            _pdata = self.arm_bridge.planning_data
            qmap   = self.arm_bridge.qpos_map
            _qpos_save = _pdata.qpos.copy()
            _qvel_save = _pdata.qvel.copy()
            try:
                # Write the converged q into planning_data
                _keys = ("ColumnLeft", "ColumnRight", "ArmLeft", "Base",
                         "HandBearing", "WristZ", "WristX", "WristY")
                for _k, _v in zip(_keys, list(q)[:len(_keys)]):
                    qpa = qmap.get(_k)
                    if qpa is not None:
                        _pdata.qpos[qpa] = float(_v)
                # Fingers OPEN (must match the gripper geometry the
                # close-stroke will operate from).
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
        """Probe the world-frame XY offset between the gripper palm
        (`Gripper_Link3_1`) and the PINCH MIDPOINT — the average of
        thumb tip and b/c centroid — in the OPEN-finger pose at the
        given wrist goal and a representative side-grip arm seed.

        Returned offset = `pinch_midpoint_xy - palm_xy`.  Callers
        subtract this from a desired pinch-midpoint-at-target world
        position to get the palm IK target — so the IK still targets
        the palm body (the long-standing palm-target architecture),
        but the target POSITION is shifted so that the pinch midpoint
        — not the palm — lands at the object.

        Why pinch midpoint and not 3-finger centroid: the 3-finger
        centroid `(thumb + b + c) / 3` weights b and c twice as much
        as the thumb.  When the open-thumb pose (j1≈-1.05) extends
        the thumb much farther from the palm than b/c (j1≈-0.55), a
        3-finger centroid sits closer to b/c — putting the centroid
        at obj leaves the thumb at ~2× the b/c-to-obj distance.

        The pinch midpoint `(thumb + bc_centroid) / 2` weights the
        thumb side and the b/c side EQUALLY — placing the pinch
        midpoint at obj makes `d_thumb ≈ d_bc` (the metric the
        pre-close gate actually uses).

        Restores planning_data qpos after the probe.  Returns None
        on any failure — caller falls back to the legacy target
        position with no shift.
        """
        try:
            _model = self.arm_bridge.model
            _pdata = self.arm_bridge.planning_data
            qmap   = self.arm_bridge.qpos_map
            _qpos_save = _pdata.qpos.copy()
            _qvel_save = _pdata.qvel.copy()
            try:
                # Chassis at virtual IK base, wrist at goal, fingers
                # open, arm at a side-grip-representative seed. The
                # offset depends mainly on wrist orientation; arm
                # pose shifts it slightly via palm world-orientation.
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
                # _carry_anchor_body_ids order is (a=thumb, b, c).
                # Pinch midpoint = ((thumb) + ((b + c) / 2)) / 2.
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
        """Per-finger normal-force target so that
            mu * 3 * N >= safety * m * g
        with mass read from the obj body.  Falls back to a floor when
        mass is unavailable or tiny.  Multiplied by
        `_strict_force_multiplier` so slip retries can demand a
        stronger grip than the first attempt."""
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
        """Execute lift to carry pose and watch the obj's pose in the
        gripper-anchor frame.  Returns True if the obj stays put (held
        by friction); False if it drifts more than
        STRICT_SLIP_DISP_THRESH or moves faster than
        STRICT_SLIP_VEL_THRESH after the settle window."""
        grip0 = self._carry_anchor_xyz(self.sim.data).copy()
        obj0  = self.sim.data.xpos[obj_bid].copy()
        rel0  = obj0 - grip0
        self._strict_log(
            "LIFT",
            f"START  obj_xyz={obj0.round(3)}  "
            f"rel0={rel0.round(3)}  "
            f"|rel0|={float(np.linalg.norm(rel0)):.3f}m")

        # Option 2: PRE-LIFT TIGHTEN. After verify passes,
        # the fingers are at the position where they FIRST made contact
        # (force-stop froze ctrl at that qpos). But the squeeze force
        # is just at the minimum threshold (8.5 N target). Before
        # lifting, bump the close ctrl deeper so the fingers wrap
        # tighter into obj surface. Higher normal force = stronger
        # friction grip during the lift's acceleration phase.
        try:
            gids = self.sim.gripper_ids_left
            # Bump each finger's proximal joint ctrl by ~0.05 rad
            # (closes ~3°), keeping middle/distal frozen at force-stop
            # so the grip tightens rather than re-opens at the tip.
            FINGER_J1_INDICES = (0, 3, 6)   # c_j1, b_j1, a_j1
            TIGHTEN_BUMP_RAD = LIFT_TIGHTEN_PRELIFT_RAD
            with self.sim._target_lock:
                for j_idx in FINGER_J1_INDICES:
                    if j_idx < len(gids):
                        gid = int(gids[j_idx])
                        cur_ctrl = float(self.sim.data.ctrl[gid])
                        lo = float(self.sim.model.actuator_ctrlrange[gid, 0])
                        hi = float(self.sim.model.actuator_ctrlrange[gid, 1])
                        # All three j1 joints curl inward at POSITIVE
                        # ctrl, but the thumb's mirror-mount means its
                        # ctrl convention differs.  _curl_targets
                        # already handles the sign per finger; here we
                        # follow the same convention.  For all 3 j1's
                        # in this gripper, "closer" is +rad direction
                        # (j1 range is mostly positive for closing).
                        new_ctrl = min(hi, max(lo, cur_ctrl + TIGHTEN_BUMP_RAD))
                        self.sim.data.ctrl[gid] = new_ctrl
            time.sleep(0.2)   # let PD converge on the tighter target
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
        # Leave th (yaw) and wrist DOFs at their current values; the
        # weld-free transport doesn't tolerate a wrist reset because
        # the fingers must stay where they currently grip.
        # lift duration revised down from 10 s (which
        # was wasted entirely when obj didn't follow) to ~3 s. The
        # slower lift didn't actually help grip retention — diagnostic
        # log showed obj fell out at some unknown point during the
        # 10 s motion regardless of speed. Combined with the new
        # early-abort below, total wasted time on a failed lift drops
        # to ~0.5 s before the abort fires.
        LIFT_STEPS = DESCENT_STEPS * LIFT_STEPS_MULTIPLIER     # tunable
        LIFT_PER_STEP_SETTLE = PD_SETTLE_PER_WAYPOINT * LIFT_PER_STEP_SETTLE_MULTIPLIER
        # Log obj/grip position BEFORE lift so we can measure full lift trajectory.
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
        # EARLY-ABORT during lift motion.  Monitor
        # `rel_z = obj_z - grip_z` continuously and abort the lift
        # early if it drifts more than 2 cm from `rel0_z` (obj
        # falling behind grip's vertical motion).  The slip-monitor
        # at the end of the motion alone wastes the full ~10 s of
        # lift travel; this catches the failure within a few cm.
        # The slip-retry then re-grasps + re-lifts.
        EARLY_ABORT_REL_Z_DRIFT = 0.02  #  m — 2 cm drift = obj slip
        EARLY_ABORT_MIN_GRIP_DZ = 0.04     # m — wait until grip has
                                            # moved 4 cm before checking,
                                            # so initial PD settle noise
                                            # doesn't false-trip
        rel0_z = float(rel0[2])
        grip0_z = float(grip0[2])
        grip0_xy = grip0[:2].copy()
        obj0_xy = (rel0[:2] + grip0[:2]).copy()
        _grasp_self = self
        def _lift_obj_tracking_check(step_idx, alpha):
            # (soft-weld mode): skip SLIP-EARLY abort
            # when pin is active. Pin enforces obj-grip relative pose
            # every step, so any measured drift is just numerical
            # residual, not real slip. Return False = "do not abort"
            # (the descent runner treats truthy return = abort signal).
            if _grasp_self._active_pin_fn is not None:
                return False
            try:
                grip_now = _grasp_self._carry_anchor_xyz(
                    _grasp_self.sim.data)
                grip_now_z = float(grip_now[2])
                grip_dz = grip_now_z - grip0_z
                if grip_dz < EARLY_ABORT_MIN_GRIP_DZ:
                    return False   # grip hasn't moved enough yet
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
                # #2: tangential / XY-drift abort. Obj can
                # roll or shift sideways without rel_z changing — the
                # grip rises with the obj displaced laterally. Measure
                # XY separation between obj and gripper centroid; if it
                # has grown by more than LIFT_XY_DRIFT_ABORT_M from the
                # start-of-lift baseline, treat as slip.
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
            "force_history":       [],   # list of [Na, Nb, Nc] per tick
            "mass_reestimate_step": 0,
        }

        def _apply_bump_per_finger(bump_rad, mask_a_b_c):
            """Apply bump_rad to selected fingers' j1 ctrl.
            mask = (apply_to_a, apply_to_b, apply_to_c)."""
            gids_l = self.sim.gripper_ids_left
            # Map: (a, b, c) → j_idx (6, 3, 0).
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
            # (soft-weld mode): skip all retighten work
            # when a pin is active. The pin holds obj at grip+offset
            # each step; finger force is not what holds obj during
            # transport. Re-tightening builds up huge stored contact
            # forces (700 kN spikes, 35 kN at post-lift) which then
            # impulse-launch obj at lift-end. With pin active, fingers
            # sit at close pose; pin propagates lift motion to obj.
            if self._active_pin_fn is not None:
                return
            try:
                # measure -----------------------------------------
                grip_now_z = float(
                    self._carry_anchor_xyz(self.sim.data)[2])
                obj_now_z = float(self.sim.data.xpos[obj_bid][2])
                grip_dz = grip_now_z - grip0_z
                obj_dz_expected = grip_dz
                obj_dz_actual = obj_now_z - (rel0_z + grip0_z)
                slip = obj_dz_expected - obj_dz_actual   # +ve = obj behind

                per_finger_N = self._per_finger_normal_forces(obj_bid)
                # diagnostic: log per-finger force every
                # 5 ticks during lift so we can see exactly when forces
                # decay vs the obj slip metric. Tagged `[LIFT_FORCE]`.
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

                # decide trigger ----------------------------------
                # Priority order:
                # 1. Slip exceeds threshold → proportional bump.
                # 2. Force-decay early warning → pre-emptive bump.
                # 3. FORCE_TARGET: any finger below
                # the force target → small bump.
                # 4. Baseline periodic keepalive → light bump.
                trigger = None     # "SLIP" | "DECAY" | "FORCE_TGT" | "BASELINE"
                bump_rad = 0.0
                # FORCE_TARGET: active per-tick maintenance. If a
                # finger has dropped below the desired holding force,
                # gently push its j1 ctrl toward more close. Aims to
                # counter the MuJoCo soft-contact oscillation between
                # 0 and high-N states by maintaining a baseline force
                # whenever the measured N drops to zero.
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
                    # Compare current per-finger N to mean over window.
                    window_mean = np.mean(np.array(history), axis=0)
                    any_decayed = False
                    for fi in range(3):
                        if (window_mean[fi] > 1.0   # only flag when there was real force to begin with
                                and per_finger_N[fi]
                                    < (1.0 - LIFT_FORCE_DECAY_FRAC)
                                       * window_mean[fi]):
                            any_decayed = True
                            break
                    if any_decayed:
                        bump_rad = LIFT_FORCE_DECAY_BUMP_RAD
                        trigger = "DECAY"
                if trigger is None and hysteresis_open and _any_below_target:
                    # FORCE_TARGET maintenance. At least
                    # one finger has dropped well below the holding
                    # force target — give it a small bump to keep
                    # baseline grip alive across the contact oscillation
                    # phase. Small magnitude (3 mrad) to avoid adding
                    # to the kN oscillation amplitude.
                    bump_rad = 0.003
                    trigger = "FORCE_TGT"
                if trigger is None:
                    if (step_idx + 1) % max(
                            1, LIFT_RETIGHTEN_INTERVAL_STEPS) == 0:
                        bump_rad = LIFT_RETIGHTEN_PER_STEP_RAD
                        trigger = "BASELINE"
                    else:
                        return    # nothing to do this tick
                if bump_rad <= 0.0:
                    return

                # per-finger attribution --------------------------
                # Only bump fingers whose normal force is below floor.
                # If ALL fingers are above floor, bump all (catches the
                # case where forces are fine but obj is still slipping
                # for some other reason, e.g. friction underestimate).
                below_floor = [n < LIFT_PER_FINGER_FORCE_FLOOR_N
                               for n in per_finger_N]
                if any(below_floor):
                    mask = tuple(below_floor)
                else:
                    mask = (True, True, True)

                _apply_bump_per_finger(bump_rad, mask)
                _retighten_state["ticks_since_bump"] = 0

                # mass re-estimation (#5) ------------
                # If slip is persistent despite bumps, the initial N_req
                # is too low. Raise self._strict_force_multiplier so
                # future retries (and subsequent ticks via the per-tick
                # ceiling) target a higher holding force. Monotonic up
                # only — never lower.
                if (LIFT_MASS_REESTIMATE_ENABLED
                        and trigger == "SLIP"
                        and (step_idx - _retighten_state[
                            "mass_reestimate_step"])
                            >= LIFT_MASS_REESTIMATE_INTERVAL):
                    _retighten_state["mass_reestimate_step"] = step_idx
                    try:
                        cur_mult = float(getattr(
                            self, '_strict_force_multiplier', 1.0))
                        # Shortfall = how much more force we needed
                        # (slip > 0 means the existing force was insufficient).
                        # Bump multiplier proportionally — capped so a single
                        # outlier doesn't blow up the target.
                        bump_mult = 1.0 + LIFT_MASS_REESTIMATE_K * (
                            slip / max(LIFT_SLIP_BUMP_THRESHOLD_M, 1e-3))
                        bump_mult = min(bump_mult, 1.10)   # ≤ 10 % per event
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

                # throttled log -----------------------------------
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

        # #4: GENTLE TEST-LIFT. Before committing to the
        # full carry-pose lift, perform a small slow rise (~6 % of full
        # lift = ~6 mm vertical) and measure whether the obj follows.
        # If it doesn't, bump j1 ctrl and re-test up to N times. This
        # mirrors how a human lifts gingerly first to check grip
        # before committing to the full motion. When test-lift fails
        # all retries, abort to the outer slip-retry loop early —
        # before we've moved the arm noticeably, so re-grasping is
        # quick.
        _np_current = np.array(current_q, dtype=float)
        _np_carry   = np.array(carry_q,   dtype=float)
        q_test = (_np_current
                  + (_np_carry - _np_current) * LIFT_TEST_LIFT_ALPHA)
        test_n_steps = max(4, int(DESCENT_STEPS
                                  * LIFT_TEST_LIFT_ALPHA))
        # Slow per-step settle: 1.5× the main-lift rate so PD has
        # extra time to track during the test.
        test_per_step_settle = LIFT_PER_STEP_SETTLE * 1.5
        test_passed = False
        # stage tracking: lift is about to fire.
        self._cycle_stage_lift_fired = True
        # skip TEST-LIFT in --perfect mode. Test-lift
        # fires phantom-slip retries (obj+finger drag DOWN during PD
        # ramp = false negative slip). Incremental lift below replaces
        # test-lift as the slip detector (per-chunk obj_dz check).
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
            # Measure obj tracking during the test.
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
                    # asymmetric tolerance. Positive
                    # slip (grip rising, obj lagging) is failure. But
                    # NEGATIVE slip (obj following or moving ahead of
                    # grip) is good behaviour — it means obj is firmly
                    # carried by the pin/weld, possibly being pulled up
                    # by the pin ease-in animation. Don't fail on that.
                    and test_slip < LIFT_TEST_LIFT_SLIP_TOLERANCE_M):
                test_passed = True
                # Main lift starts from the test-end pose.
                current_q = list(q_test)
                break
            # Test failed. If retries left, bump j1 and try again.
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
                # Lower back to current_q baseline for the next test.
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
        # incremental lift in --perfect mode. Single
        # big-jump lift (h1 0.135→0.60 = 46cm) causes PD lag → initial
        # grip_dz NEGATIVE → slip-feedback fires phantom retries that
        # compound chassis drift. Break into chunks of 10cm Δh1 each;
        # between chunks settle 0.3s + verify obj_dz tracking. Abort
        # whole lift if obj fell > 5cm during any chunk.
        if STRICT_PERFECT_FRICTION_ONLY:
            # 10 cm chunks: smaller chunks let obj slip mid-chunk
            # before lift detects.
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
                    # interpolate h2/a1 proportionally
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
            # Early-abort fired — obj not following. Treat as slip
            # so the outer retry loop kicks in with a bumped force
            # multiplier. This is much faster than completing the
            # full lift then catching slip post-motion (~0.5 s vs
            # ~10 s wasted per failure).
            self._strict_log(
                "LIFT",
                f"SLIP-EARLY  motion aborted after "
                f"{_lift_motion_dur:.2f}s (obj fell behind grip during "
                f"lift) — propagating to slip-retry")
            return False
        # Log obj/grip position AFTER lift motion completes so we see
        # whether obj followed the gripper up.
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
            # stage tracking: obj followed grip during lift.
            if _obj_followed:
                self._cycle_stage_obj_followed = True
        except Exception:
            pass

        # Option 3: MAINTAIN SQUEEZE DURING LIFT. The
        # force-stop froze ctrl at first-contact qpos. During the lift
        # itself the fingers don't re-tighten if contact normal drops.
        # Re-bump j1 ctrls once after the lift motion completes (before
        # the slip-monitoring window starts) so the grip is at its
        # highest squeeze when we check for slip.
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

        # v2 (soft-weld mode): skip post-lift slip-monitor
        # when pin is active. Pin re-snaps obj to grip+offset every
        # step, so any measured disp/vel is just pin numerical residual
        # not real slip. Triggering slip-retry on that residual
        # caused a false-fail cascade: lift succeeded → slip-monitor
        # saw 12-22 mm residual → "SLIP" detected → chassis retract
        # for re-descend → obj fell during retract → obj_out_of_bounds.
        # Return True so cycle proceeds to place/drop phase normally.
        if self._active_pin_fn is not None:
            self._strict_log(
                "LIFT",
                "slip-monitor SKIPPED — pin active, obj-grip relative "
                "pose is enforced each step")
            return True

        # log per-finger contact state at start of slip-
        # monitoring so we can correlate slip events to which finger
        # lost grip first.
        try:
            _live_c = bool(self._finger_touches_obj(0, obj_bid))
            _live_b = bool(self._finger_touches_obj(1, obj_bid))
            _live_a = bool(self._finger_touches_obj(2, obj_bid))
            # Per-finger forces via mj_contactForce — diagnostic only.
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
        # periodic in-flight contact log (every 0.2 s)
        # so we can see which finger lets go and when.
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

            # Periodic contact-state log during slip monitor
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
                    # contact-state at slip moment for
                    # correlation with the disp/vel readings.
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
        # emit BASELINE capture on full lift success
        # (verify ✓ + lift stable). This is the GOLD baseline —
        # geometry where the whole pickup pipeline worked end-to-end.
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
        """Slip-monitored lift with up to STRICT_RETRY_MAX in-attempt
        retries.  On slip: check if obj has moved (Fix 1 — abort if
        it has since same-pose retry is useless); else open fingers,
        re-descend to grasp_q, re-close with bumped force target,
        lift again.  Returns True on stable lift, False on exhausted
        retries or obj-moved abort."""
        # Snapshot obj position at start (before any lift) — used by
        # the obj-moved guard to detect if obj fell/was knocked away
        # during a failed lift.
        try:
            obj_pos_at_close = np.asarray(
                self.sim.data.xpos[obj_bid][:2], dtype=float).copy()
        except Exception:
            obj_pos_at_close = None
        OBJ_MOVED_ABORT_THRESHOLD = 0.05   # 5 cm

        _max_attempts = 1 + STRICT_RETRY_MAX
        for attempt in range(_max_attempts):
            if attempt > 0:
                # Fix 1: obj-moved guard. If obj has shifted more
                # than 5 cm since the close stroke (typically because
                # it fell after a slip), re-descending to the cached
                # grasp_q targets the OLD obj location — wasted retry.
                # Abort to outer retry/candidate loop instead.
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
                # Open fingers to free obj, descend back to grasp_q,
                # re-close with the new (higher) force target. The
                # close inside _set_gripper reads
                # _strict_force_multiplier each call so the bump takes
                # effect transparently.
                self._set_gripper(GRIPPER_OPEN_POS,
                                  hold_seconds=GRIPPER_HOLD_TIME * 0.5)
                if self._cancel:
                    return False
                # Retract chassis ~10 cm along reverse-
                # approach before the re-descent. Operator observed
                # that the slip-retry's open+descend at the SAME
                # chassis XY caused the arm to swing past obj on the
                # way down (arm was up at carry pose h1=0.6, descend
                # back to h1=0.193 = ~40cm of vertical sweep). obj
                # has fallen to floor after the slip; the descending
                # arm tip can hit it. Retracting chassis first gives
                # the arm space to descend straight without crossing
                # obj's footprint.
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

                # After the chassis retract + re-descend,
                # the chassis is ~10 cm farther from obj than during
                # the original successful grasp (the retract was for
                # arm-clear-during-descent safety, but never undone).
                # If we close here, fingers are 14-19 cm from obj and
                # close fires in air (operator observed this in run
                # log: `close summary: NO finger contact during close`).
                # Re-approach chassis to obj-standoff so palm lands
                # near obj surface, then closed-loop-snap any residual.
                try:
                    obj_xy_reapp = self.sim.data.xpos[obj_bid][:2].copy()
                    cur_xy_reapp = np.asarray(
                        self.sim.localization()[:2], dtype=float)
                    appr_vec = obj_xy_reapp - cur_xy_reapp
                    appr_dist = float(np.linalg.norm(appr_vec))
                    # Target: same standoff as the original successful
                    # grasp. 0.55-0.60 m chassis-to-obj is typical.
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
                        # Contact-guarded so the re-approach stops at
                        # first finger touch (same as the initial
                        # chassis push). Geometric guard catches
                        # thumb penetration.
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
        """Return a closure that pins the held object at a fixed offset
        from the fingertip-pocket centroid (`_carry_anchor_xyz`).  The
        fingertip centroid is the right reference for the carry phase
        because it sits at obj height (~10 cm below the palm with the
        open-thumb pose), so the verify-grasp pose stays inside the
        finger cage."""
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

    def _object_half_xy(self, obj_bid):
        """Wrapper around the module-level helper.  Returns (half_x,
        half_y) for box geoms or None for cylinders/spheres."""
        return object_half_xy(self.sim.model, obj_bid)

    def _compute_wrist_goal_for_obj(self, obj_bid):
        """Wrapper so the executor method preserves its existing call
        site (`self._compute_wrist_goal_for_obj`).  Real logic lives in
        the module-level helper so play_m1 can call it during candidate
        screening without instantiating an executor."""
        return compute_wrist_goal_for_obj(self.sim.model, obj_bid)

    def _finger_close_for_radius(self, radius):
        """Map object radius to a finger ctrl value such that the closed
        fingers contact the object's outer surface.  Used by the [6]
        close-gripper command and by the post-pin carry hold pose so
        the fingers stay closed around the object during transport."""
        pos = FINGER_CLOSE_MAX - FINGER_CLOSE_PER_M * float(radius)
        return max(FINGER_CLOSE_FLOOR, min(FINGER_CLOSE_MAX, pos))

    def _pin_obj_to_gripper_animated(self, start_world_xyz,
                                      duration=SMOOTH_ATTACH_SECS,
                                      anchor_palm=False,
                                      anchor_pinch_midpoint=False,
                                      phased_xy_then_z=False,
                                      xy_phase_secs=0.6,
                                      z_phase_secs=0.6):
        """Pin closure that smoothly interpolates the object from its
        current world position to the gripper-relative carry offset over
        `duration` seconds using smoothstep easing (3t^2 - 2t^3).  After
        interpolation completes the closure behaves identically to the
        static `_pin_obj_to_gripper`.

        anchor_palm: pin target offset from `gripper_body_id` (palm —
        `Gripper_Link3_1`) rather than the fingertip centroid.  Palm
        is rigid in the arm chain; useful when pin must hold obj at a
        stable world position regardless of finger motion.

        anchor_pinch_midpoint: pin target is the thumb↔bc-centroid
        midpoint (`_pinch_midpoint_xyz`) instead of the 3-tip mean.
        For a 1+2 gripper, the 3-tip mean is biased ~30% toward the
        bc-side (two of three points are on one side); pinch_midpoint
        is the geometric centre between the thumb and the bc pair —
        the visual "centre between thumb and finger" the eye expects
        obj to sit at.

        Both flags default False (legacy carry_anchor behaviour).  If
        both are True, anchor_palm wins.
        """
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
                # Phase A: XY slides into pinch centre while Z stays.
                # Phase B: Z drops to grip height while XY at attached.
                # After both phases: pin holds at live attached pose.
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
        """Pin closure that eases the held object to a fixed world pose.

        FAST_PICKUP_MODE uses this during the close stroke.  The normal
        gripper-relative pin tracks the fingertip centroid, and that
        centroid moves while fingers close.  Holding a fixed world target
        prevents the object from dragging the wrist/arm sideways through
        hard contact forces during close; after close we reattach to the
        gripper with the current offset for lift/transport.
        """
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
        # pinned to the gripper). Only restores if a value was saved
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
        self._pre_close_lift_done = False
        self._fast_fixed_close_pin_active = False

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
        """Write an arm command.  Accepts 4-vec (arm slides only) or 8-vec
        (arm + wrist).  When q has 8+ elements the 4 wrist actuators
        (`wrist_X/Y/Z`, `HandBearing`) are commanded too — that lets the
        OMPL 8-DOF plan and the descent interpolation drive wrist
        orientation in lockstep with the arm slides.
        """
        with self.sim._target_lock:
            self.sim.use_ik = False
            c = self.sim.direct_arm_commands.copy()
            c[0] = q[0]; c[1] = q[1]; c[2] = q[2]; c[3] = q[3]
            self.sim.direct_arm_commands = c
        if len(q) >= 8:
            # State vector order: 4=hb, 5=wz, 6=wx, 7=wy
            # Actuator order: gids[11]=wrist_X, [12]=wrist_Y,
            # [13]=wrist_Z, [14]=HandBearing
            gids = self.sim.gripper_ids_left
            if len(gids) > GIDS_HANDBEARING:
                # Pre-compensate wrist_Z for PD steady-state residual:
                # at kp=100 the joint settles ~9% short of |target|.
                # Overshoot the ctrl by the same ratio so qpos lands
                # at the IK-planned value. Other wrist joints (hb,
                # wx, wy) settle within 1° of target — no compensation
                # needed for them.
                wz_target = float(q[5])
                wz_ctrl = wz_target * (1.0 + WRIST_Z_PD_COMPENSATION_RATIO)
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
        # Compact end-of-path summary: final joint state in one line.
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
        """Linear joint-space interpolation, executed via PD waypoints.
        Interpolates across all dims present in both endpoints — when
        both are 8-vec the wrist tilts in lockstep with the arm slides.

        optional `per_step_settle` overrides the default
        `PD_SETTLE_PER_WAYPOINT`.  Slower per-step time means lower
        joint velocity per waypoint → lower lift acceleration.  The
        `lift-strict` label uses this to halve lift velocity so
        friction grip can hold obj.

        optional `early_abort_check(step_idx, alpha)`
        callback evaluated every `early_abort_interval` steps.  If it
        returns True the descent ABORTS without finishing.  Used by
        the lift code to bail when obj isn't following the gripper
        upward (saves 7-10 s per failed lift).  Returns True if the
        descent completed normally, False if aborted (by cancel or
        early-abort).

        optional `per_step_action(step_idx, alpha)`
        callback evaluated every `per_step_action_interval` steps for
        SIDE EFFECTS (no return value).  Used by the lift code to
        periodically re-bump finger j1 ctrls so the squeeze pressure
        is maintained throughout the lift motion (force-stop ctrl
        freeze causes grip force to decay otherwise).
        """
        n_dims = min(len(q_start), len(q_end))
        step_sleep = (per_step_settle if per_step_settle is not None
                      else PD_SETTLE_PER_WAYPOINT)
        for i in range(n_steps):
            if self._cancel:
                return False
            alpha = (i + 1) / n_steps
            q = [q_start[j] + alpha * (q_end[j] - q_start[j])
                 for j in range(n_dims)]
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

    # ── Gripper ──────────────────────────────────────────────────────────

    def _curl_targets(self, pos):
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

        Thumb open angle depends on grip mode:
        - **Top-down approach**: thumb opens to `THUMB_OPEN_POS = -1.05`
          (~60° back from rest) so the thumb tip clears the obj_top
          plane during vertical descent.  Sides open less (-0.55).
        - **Side-grip approach**: gripper enters horizontally; thumb
          doesn't need to clear obj_top, so all three fingers open
          symmetrically at `GRIPPER_OPEN_POS = -0.55`.  Symmetric
          open ⇒ all three fingertips at the same radius from palm
          ⇒ when palm sits at obj-standoff, thumb and bc are
          equidistant from obj.

        Returns a list of 9 target ctrl values in gripper_ids_left order.
        """
        if pos >= 0.0:
            # CLOSE: all 3 fingers symmetric (thumb closes with sides).
            # `pos` is a close-ctrl value in roughly [0, 0.20]. Map to a
            # [0, 1] intensity for joint-range scaling.
            intensity = min(1.0, max(0.0, pos / 0.20))
            _thumb_factor = (CURL_J1_FACTOR_THUMB_PERFECT
                             if STRICT_PERFECT_FRICTION_ONLY
                             else CURL_J1_FACTOR_THUMB_SOFTWELD)
            j1_side  = intensity * CURL_J1_FACTOR_SIDE * 0.85
            j1_thumb = intensity * _thumb_factor       * 0.85
            j2 = intensity * CURL_J2_FACTOR * 0.95
            j3 = -0.052 - intensity * CURL_J3_FACTOR * 1.10
            # Indices 0..2 = finger_c, 3..5 = finger_b, 6..8 = finger_a.
            return [j1_side,  j2, j3,
                    j1_side,  j2, j3,
                    j1_thumb, j2, j3]
        else:
            # Opening.
            j1_side  = pos                # = GRIPPER_OPEN_POS = -0.55
            # In side-grip the thumb and bc-pair need different j1 open
            # angles.  The 1+2 DELTO M3 mounts the thumb 3.2 cm forward
            # of the bc bases in palm-local X, so at a matched j1 the
            # thumb tip ends up ~6 cm ahead of bc tips in the chassis-
            # approach direction.  Matching j1 stops chassis push on
            # thumb contact while the bc-pair
            # is still 4 cm short of the obj. Opening the thumb to the
            # wider top-down value pulls the tip back so the chassis
            # can push further before any finger hits the obj, and the
            # close stroke can then bridge a symmetric
            # remaining gap.
            j1_thumb = THUMB_OPEN_POS
            j2 = 0.0
            j3 = -0.0523
            return [j1_side,  j2, j3,
                    j1_side,  j2, j3,
                    j1_thumb, j2, j3]

    def _thumb_open_pos_for_mode(self):
        """Return the thumb open ctrl appropriate for the current grip
        mode.  Mirrors `_curl_targets`'s open branch so call sites that
        write planning_data directly (e.g. IK validity setup) match the
        runtime gripper geometry.  side-grip now uses the
        wider THUMB_OPEN_POS too (see comment in `_curl_targets`)."""
        return THUMB_OPEN_POS

    # Cache of finger/palm joint qposadrs, populated lazily on first call.
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
        """Poll finger joint qpos against ctrl until every commanded
        joint is within `tol` of its target, or `timeout` elapses.

        Returns a tuple (settled, elapsed_s, per_joint_residual) so the
        caller can log how the open phase actually landed.  This is
        called right after a synchronous OPEN _set_gripper to gate the
        descent on the gripper actually being open in physical space,
        not just at ctrl-target.
        """
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
        # STRICT-mode slow close: when closing onto a held obj without
        # a pin, the close stroke must run slowly enough for the
        # contact constraint solver to ramp force gradually. Otherwise
        # force-stop fires AFTER the impulse has shoved the obj.
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
        palm_mag = PALM_SPREAD_CLOSE if pos >= 0.0 else PALM_SPREAD_OPEN
        # Mirrored sign convention — see PALM_*_SIGN comment above.
        palm_c_target = PALM_C_SIGN * palm_mag
        palm_b_target = PALM_B_SIGN * palm_mag
        # Full 11-DOF target (9 finger joints + 2 palm spread joints)
        target = list(finger_targets) + [palm_c_target, palm_b_target]

        # one-shot obj-to-pocket snap for --auto-skip-nav.
        # Emulates harness's floating-obj setup so lift physics can
        # be tested in play_m1 without obj-on-floor interference.
        global SNAP_OBJ_TO_POCKET_PRE_CLOSE_ONCE
        # In --perfect mode always run the chassis-drive alignment
        # block (chassis-translate to obj; snap only as fallback if
        # chassis-drive doesn't converge).  Flag-gated path retained
        # for non-perfect callers.
        _perfect_align_fire = (STRICT_PERFECT_FRICTION_ONLY
                               and pos >= 0.0
                               and self._held_obj_bid is not None
                               and self._held_obj_qpa is not None)
        _kin_close_safe = bool(SNAP_OBJ_TO_POCKET_PRE_CLOSE_ONCE
                               or _perfect_align_fire)
        if ((SNAP_OBJ_TO_POCKET_PRE_CLOSE_ONCE or _perfect_align_fire)
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
                        # Locate base body free joint qposadr.
                        _base_bid = mujoco.mj_name2id(
                            self.sim.model, mujoco.mjtObj.mjOBJ_BODY, "robot")
                        if _base_bid < 0:
                            _base_bid = mujoco.mj_name2id(
                                self.sim.model, mujoco.mjtObj.mjOBJ_BODY, "base")
                        _ja = int(self.sim.model.body_jntadr[_base_bid])
                        _qa = int(self.sim.model.jnt_qposadr[_ja])
                        _da = int(self.sim.model.jnt_dofadr[_ja])
                        # 2-pass drive — first pass moves ~70%; re-
                        # measure and second pass closes the rest.
                        # Each pass is 30 steps × 50ms = 1.5s smooth.
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
                            time.sleep(0.30)  # settle
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
                            # Stop if pass didn't improve ≥0.5cm —
                            # further passes only worsen via oscillation.
                            if _d_after >= _d_pre - 0.005:
                                print(f"[Exec] chassis-align: "
                                      f"pass {_pass+1} oscillating "
                                      f"(Δ improvement <0.5cm); stop")
                                break
                    except Exception as _e_aa:
                        print(f"[Exec] WARN chassis-align: {_e_aa}")
                else:
                    _converged = True
                # No obj-snap fallback: if chassis-drive doesn't
                # converge to ≤2cm, RETIRE this candidate — set the
                # flag so the close stroke aborts and the caller
                # retries from a new pose.
                SNAP_OBJ_TO_POCKET_PRE_CLOSE_ONCE = False
                # Retire only if chassis-drive truly failed (>5cm).
                # 3-5cm is recoverable — pin still aligns obj, PD ramp
                # close still wraps fingers. Strict triad-gate may
                # still reject one-sided, but at least we try.
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

        # v2 (v3 TIER 4): per-finger asymmetric close in
        # perfect mode. Gripper has STRUCTURAL palm-depth offset (~6cm
        # between thumb-front +X and bc-rear -X anchors). With perfect
        # axis-aligned reach, thumb-bc are equidistant from obj, but
        # play_m1 chassis approach + descent makes thumb anchor 5-7cm
        # closer than bc. Original v1 measured live d_t/d_b/d_c but at
        # close-stroke entry fingers are still OPEN — anchors look
        # similar. Solution: ALWAYS apply structural-offset bc deepening
        # in --perfect; structural offset is geometry-fixed, not pose-
        # dependent. Each cm of expected post-close anchor asymmetry
        # → bc curl ramp +0.04 (each radian j1 ≈ 4cm tip travel).
        # Indices: c=[0,1,2], b=[3,4,5], a=[6,7,8] (a = thumb).
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
                    # animated pin snap. Interpolate
                    # obj from current to pocket over 0.3s with
                    # smoothstep easing BEFORE locking — looks like
                    # obj nudged in place, not teleport.
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
                            # ctrl-only write (no qpos snap) → PD
                            # ramps joints toward target smoothly.
                            self.sim.data.ctrl[_gid_k] = val_c
                # Plain sleep instead of force-stop polling: polled
                # variants admitted MN-scale solver spikes between
                # samples, leaving residual energy that launched obj
                # on the subsequent lift.
                time.sleep(0.80)
                # Populate _last_close_finger_contacts so verify works.
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

        # Stage-4 OMPL trajectory probe (verbose only — 6+ lines per
        # open/close, gated behind VERBOSE_GRASP_DEBUG to keep main
        # log focused on grasp events).
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

        # ── Execute the planned path with per-finger timing ───────────────
        # For an asymmetric 1+2 gripper closing on an obj that isn't
        # centred in the pinch midpoint, the three fingertips can be at
        # very different distances from obj surface at close-start
        # (logged as d_thumb / d_b / d_c at pre-close). With zero
        # offsets all three close at the same speed → the closest
        # finger (typically thumb) contacts seconds before the others
        # and the obj settles in a loose grip during the wait → b/c
        # contact a moving obj at a transient pose → ctrl freezes at
        # those poses → obj slips out of the resulting bad grip during
        # the sustained-contact verify.
        # Fix: stagger close-start per finger so all three fingertips
        # reach obj approximately simultaneously. Closest finger waits;
        # farthest starts at t=0. Real grippers do exactly this.
        joint_offsets = [0.0] * 11
        if (STRICT_PICKUP_MODE
                and pos >= 0.0
                and self._held_obj_bid is not None
                and getattr(self, '_proximity_finger_links', None) is not None
                and len(self._proximity_finger_links) == 3
                and all(len(g) >= 1 for g in self._proximity_finger_links)):
            try:
                obj_xy_close = self.sim.data.xpos[self._held_obj_bid][:2]
                # Closest link XY distance per finger group (0=c, 1=b, 2=a).
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
                # Empirical fingertip speed during the slow close stroke.
                # Run log evidence: with 4 s transition the tip travels
                # ~6 cm in ~2 s before force-stop trips → ~3 cm/s.
                _TIP_SPEED = 0.030
                # Per-finger start delay = (d_max - d_finger) / speed.
                # Closest finger gets the largest delay so all three
                # reach obj within ~150 ms of each other.
                _delays = [(_d_max - d) / _TIP_SPEED for d in _finger_d]
                # Cap each delay at 50% of the close transition so the
                # slowest finger still gets a full close stroke after
                # waiting (avoids the "finger never started" failure).
                _cap = STRICT_CLOSE_TRANSITION_SECS * 0.5
                _delays = [min(d, _cap) for d in _delays]
                # joint_i 0-2 = c, 3-5 = b, 6-8 = a; palm (9, 10) unstaggered.
                joint_offsets = [
                    _delays[0], _delays[0], _delays[0],   # finger_c
                    _delays[1], _delays[1], _delays[1],   # finger_b
                    _delays[2], _delays[2], _delays[2],   # finger_a (thumb)
                    0.0, 0.0,                             # palm spread
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
                # Defensive fallback to synchronous timing. Log so it's
                # visible but don't abort the close stroke.
                print(f"[Exec] per-finger stagger compute failed: "
                      f"{_e_stagger} — falling back to synchronous close")
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
        # close path. When a finger's bodies touch the held object,
        # freeze that finger's three joint ctrls at their current values
        # for the rest of this _set_gripper call. Other fingers continue
        # advancing toward their targets.
        contact_stop_enabled = (
            USE_CONTACT_STOP_CLOSE
            and pos >= 0.0                              # closing only
            and self._held_obj_bid is not None          # have an object
            and self._finger_body_groups is not None
            and all(len(g) >= 1 for g in self._finger_body_groups)
        )
        compress_ticks = (FAST_CONTACT_COMPRESS_TICKS
                          if FAST_PICKUP_MODE and pos >= 0.0
                          else CONTACT_COMPRESS_TICKS)
        finger_frozen = [False, False, False]   # [c, b, a]
        finger_contact_ticks = [0, 0, 0]        # For push-past-contact compression
        # Proximity stop: enabled in mode 2 / mode 3. STRICT (mode 1)
        # disables it so the finger PD can keep pushing past the surface
        # to build measurable contact force — STRICT freezes on force
        # threshold instead (see force_stop_enabled below).
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

        # Force-stop close (STRICT mode only). Per-finger normal force
        # is read each tick via mj_contactForce. When summed force on
        # a finger reaches the per-finger requirement for
        # STRICT_FORCE_STOP_STABLE_TICKS consecutive ticks, freeze that
        # finger's ctrl at its current qpos. N_req is derived from the
        # holding-force inequality mu * sum(N) >= safety * m * g.
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
        # joint_i (0-8) → finger group index (0=c, 1=b, 2=a):
        joint_to_finger = {
            0: 0, 1: 0, 2: 0,    # finger_c
            3: 1, 4: 1, 5: 1,    # finger_b
            6: 2, 7: 2, 8: 2,    # finger_a
        }

        last_k_executed = 0
        _set_gripper_t0 = time.time()
        for k in range(1, n_steps + 1):
            if self._cancel:
                return
            t = k * SMOOTH_GRIPPER_STEP_S
            last_k_executed = k

            # Each iteration: count compression ticks instead of
            # freezing on first contact. After CONTACT_COMPRESS_TICKS
            # ticks of continuous contact (~0.04 s at 50 Hz), freeze
            # the finger's ctrl at its current qpos. The interim
            # ticks let the PD push slightly into the cylinder surface
            # for a visible firm wrap. Contact-stop applies to all
            # three fingers; disabling it on the thumb produced
            # uncontrolled close-ramp wind-up across cycles.

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
                        # / 54: tiered relief. PD residual
                        # ramping can drive force orders of magnitude
                        # past the threshold before the check sample
                        # catches it. Back off j1 ctrl by an amount
                        # proportional to the overshoot:
                        # ≥ 10× target → 80 mrad (extreme)
                        # ≥ 5× target → 50 mrad (high)
                        # ≥ 2× target → 20 mrad (moderate, baseline)
                        # < 2× target → no relief (just freeze)
                        # j2 / j3 still snap to current qpos (preserve
                        # wrap geometry).
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
                                # Apply relief only on j1 (proximal).
                                if (relief_active
                                        and jidx == j1_idx_local):
                                    # All 3 fingers close at +rad on j1
                                    # (closing direction); relief = lower
                                    # ctrl by the tier amount to unload PD.
                                    # in --perfect mode,
                                    # zero relief. Even MOD relief (5mrad)
                                    # pulls bc fingers ~0.25mm off obj
                                    # surface (lever ~5cm), dropping bc
                                    # compression below contact margin
                                    # band (5mm) → force=0 → verify fails
                                    # one-sided. v3 pad μ=5 + dampers
                                    # already limit penetration without
                                    # needing relief.
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

            # STRICT mode bypasses contact-stop entirely — the freeze
            # criterion is force-stop only (above). Contact-stop would
            # latch on any contact after `compress_ticks`, which fires
            # before force has time to ramp to the required value and
            # would lock in a weak grip.
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
                            # Snap ctrl = current qpos for this finger's
                            # 3 joints so PD residual drops to zero and
                            # the finger no longer pushes into the obj
                            # (otherwise the un-followed PD residual
                            # produces an oscillation against the
                            # contact constraint).
                            _addrs_freeze = self._ensure_finger_joint_qposadrs()
                            _finger_joint_groups_loop = {
                                0: (0, 1, 2), 1: (3, 4, 5), 2: (6, 7, 8)}
                            for jidx in _finger_joint_groups_loop[fi]:
                                if (_addrs_freeze
                                        and jidx < len(_addrs_freeze)
                                        and _addrs_freeze[jidx] >= 0):
                                    qv = float(self.sim.data.qpos[
                                        _addrs_freeze[jidx]])
                                    # Clamp to actuator ctrlrange so a
                                    # corrupted qpos (out-of-range from
                                    # a missed-exclude collision push)
                                    # doesn't get locked in as ctrl.
                                    _gid = int(gids[jidx])
                                    _clo = float(self.sim.model.actuator_ctrlrange[_gid, 0])
                                    _chi = float(self.sim.model.actuator_ctrlrange[_gid, 1])
                                    qv = max(_clo, min(_chi, qv))
                                    self.sim.data.ctrl[_gid] = qv
                            fname = ['c', 'b', 'a'][fi]
                            # Include fingertip xyz + obj xyz in the
                            # contact log so we can correlate which
                            # finger touched and where it was.
                            tip_xyz = None
                            if (self._carry_anchor_body_ids
                                    and len(self._carry_anchor_body_ids) == 3):
                                # carry_anchor_body_ids order is (a, b, c)
                                # but finger_frozen order is (c, b, a) →
                                # reverse the index.
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
                            # Don't advance this finger's ctrl any
                            # further — its tip is on the surface.
                            continue
                    local_t = max(0.0, t - joint_offsets[joint_i])
                    alpha = min(1.0, local_t / transition_secs)
                    val = _wp_value(joint_i, alpha)
                    # Clamp to actuator ctrlrange to prevent OMPL path
                    # waypoints from driving a joint past its declared
                    # limit (which the soft-limit constraint would let
                    # accumulate across pick attempts).
                    if (USE_FINGER_CTRL_CLAMP
                            and joint_i < len(FINGER_CTRL_RANGES)):
                        lo, hi = FINGER_CTRL_RANGES[joint_i]
                        val = max(lo, min(hi, val))
                    self.sim.data.ctrl[gids[joint_i]] = val

            # Early exit if all three fingers are in contact — no
            # benefit to keeping the loop running.
            if contact_stop_enabled and all(finger_frozen):
                # Fall through to the hold below.
                break

            time.sleep(SMOOTH_GRIPPER_STEP_S)

        # Hold phase — keep monitoring contacts. Some fingers form
        # contact only after the OMPL trajectory ramp ends; polling
        # during the hold catches them and snaps ctrl = qpos so the
        # un-followed PD residual doesn't push the finger into the
        # object (which produces visible "dancing" via the contact
        # constraint feedback loop). STRICT mode skips this — the
        # close-loop force-stop is the sole freeze criterion, and any
        # under-gripped state is caught by slip detection during lift.
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
                            # Set ctrl = current qpos for this finger's 3
                            # joints so PD has zero driving force.
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
                    # All fingers now stable; finish remaining hold time
                    # as plain sleep (no need to keep polling).
                    remaining = hold_seconds - (time.time() - hold_t0)
                    if remaining > 0:
                        time.sleep(remaining)
                    break
                time.sleep(SMOOTH_GRIPPER_STEP_S)
        else:
            time.sleep(hold_seconds)

        # End-of-close summary — log which fingers physically contacted
        # the object. No contact ⇒ object held by pin only.
        if contact_stop_enabled and pos >= 0.0:
            labels = ['c', 'b', 'a']
            contacts = [lbl for lbl, frz in zip(labels, finger_frozen) if frz]
            # Path B Step 2: stash per-finger contact for the caller's
            # 3-finger verification. [c, b, a] order, matching label list.
            self._last_close_finger_contacts = list(finger_frozen)
            if contacts:
                print(f"[Exec] close summary: contacted={contacts} "
                      f"({len(contacts)}/3 fingers)")
            else:
                print("[Exec] close summary: NO finger contact during close — "
                      "fingers closed in air, object held by pin only "
                      "(visual grip will look loose)")
        else:
            # Path B Step 2: opening or contact-stop disabled — clear
            # any stale snapshot so the caller knows to skip verification.
            self._last_close_finger_contacts = None

        # (v3): global ctrl←qpos snap at end of close in
        # perfect mode. The per-finger force-stop freeze (above) only
        # snaps fingers that crossed `N_req` for `STABLE_TICKS` consecutive
        # ticks. Under the kN/MN chatter regime, some fingers may never
        # produce a stable-enough force trace to qualify (the force
        # samples spike between 0 and MN within 50 ms), so they stay PD-
        # tracked and continue overshooting the ctrl target — that PD
        # residual is the contact-bounce loop.
        # Mirror the harness's pure-friction state: at end of close,
        # force every finger joint's ctrl = current qpos so the PD
        # residual is zero across the whole gripper. Static Coulomb
        # friction with μ=25 (pad) × N=kN > m·g holds the obj for verify
        # and lift without any further PD push.
        # Gated to --perfect only because the weld path (non-perfect)
        # doesn't depend on a chatter-free grip — it relies on the post-
        # verify weld to bridge.
        # DISABLED in --perfect with asymmetric close:
        # snap kills PD push on bc fingers, which lose compression
        # (snapshot=True during close → live=False at verify, force=0).
        # Force-stop relief handles thumb overshoot; bc needs continued
        # PD residual to maintain compression.
        if (STRICT_PICKUP_MODE
                and STRICT_PERFECT_FRICTION_ONLY
                and pos >= 0.0
                and False):  # DISABLED — see comment above
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

        # End-of-close diagnostic: print final ctrl + qpos for every
        # finger and palm joint plus loop tick count. Useful for
        # determining whether a finger failed to curl because the PD
        # was overridden by contact (ctrl reached target but qpos did
        # not follow) versus an early exit (ctrl still at open value).
        try:
            elapsed = time.time() - _set_gripper_t0
            addrs = self._ensure_finger_joint_qposadrs()
            labels11 = ('c_j1','c_j2','c_j3',
                        'b_j1','b_j2','b_j3',
                        'a_j1','a_j2','a_j3',
                        'palm_c','palm_b')
            pos_kind = 'CLOSE' if pos >= 0.0 else 'OPEN'
            # End-state header line stays unconditional (cheap, useful).
            print(f"[Exec] _set_gripper({pos_kind} pos={pos:+.3f}) end-state: "
                  f"loop_ticks={last_k_executed}/{n_steps}  "
                  f"elapsed={elapsed:.2f}s  "
                  f"hold={hold_seconds:.2f}s")
            # Per-joint qpos/ctrl/resid dump (11 lines per open/close)
            # gated behind VERBOSE_GRASP_DEBUG.
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
            # Contact scan to identify what's fighting PD when a
            # PD-RESIDUAL flag is set. Only print on close (pos >= 0)
            # to avoid log spam during open — the open phase always
            # has parked-arm contacts on palm_c/palm_b that are
            # already visible at startup-probe time.
            if pos >= 0.0:
                self._log_gripper_contacts(
                    f"_set_gripper({pos_kind}) end", force_forward=False)
        except Exception as e:
            print(f"[Exec] _set_gripper end-state log warning: {e}")

    # ── Reading current arm config ──────────────────────────────────────

    def _current_arm_q(self):
        """Return the 8-DOF arm + wrist qpos snapshot.  Order matches
        JOINT_RANGES_ARM in arm_planner.py: h1, h2, a1, th, hb, wz, wx, wy."""
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

    # ── Public: Pick ─────────────────────────────────────────────────────

    def reset_finger_attempt_counter(self):
        """Reset the per-cycle 3-finger strict-attempts counter.  Call
        from play_m1 when a new MOVE pick cycle starts (not on internal
        base-XY retries — those should keep the counter so the strict
        window expires after MAX_STRICT_3FINGER_ATTEMPTS *total* attempts
        in the cycle, not per retry)."""
        self._strict_finger_attempts_used = 0

    def _retract_after_failure(self, tag, skip_lift=False,
                                side_grip_retry=False):
        """Retract the arm after a gate / 3-finger failure.

        Default: hover-retract (column-only, +HOVER_LIFT above grasp
        pose) so the next local base-XY retry can re-descend with
        minimal motion.

        Direction-aware mode (`skip_lift=True`): when the residual
        between the alignment point and the obj is mostly along the
        approach axis (gripper-forward), the upcoming base nudge will
        bring the obj forward into the gripper without needing to
        lift the arm — saves the entire lift/lower cycle.

        When `side_grip_retry=True` the non-FAST side-grip local retry
        will perform a chassis-only back-then-forward maneuver and so
        needs the arm held at GRASP_Q.  This branch keeps the arm in
        place and sets a flag the next `_pick_run` call reads.
        """
        if side_grip_retry:
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
            # Use the smaller retry lift for same-yaw retries — full
            # HOVER_LIFT (30 cm) is only needed for full re-orientation
            # retries reached via `retract_to_carry()`.
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
            # Legacy full retract to carry pose — wrist resets to neutral.
            retry_q = [CARRY_H1, CARRY_H2, CARRY_A1, float(current_q[3])] \
                      + list(WRIST_NEUTRAL)
            print(f"[Exec] {tag} retract arm before candidate retry")
            self._kinematic_descent(current_q, retry_q, "retry-retract",
                                    n_steps=DESCENT_STEPS)

    def retract_to_carry(self):
        """Move the arm to the safe carry pose (h1=CARRY_H1, h2=CARRY_H2,
        a1=CARRY_A1, current yaw, wrist=neutral).  play_m1 should call
        this before any next-candidate navigation so the chassis-yaw arc
        doesn't swing an extended arm through the rack or floor obstacles.

        Resetting the wrist DOFs to neutral here makes the *next* pickup
        cycle start from a clean orientation — important now that pickup
        commands a top-down wrist tilt that must not persist across
        navigation arcs.  Idempotent if already close to carry pose."""
        current_q = self._current_arm_q()
        carry_q = [CARRY_H1, CARRY_H2, CARRY_A1, float(current_q[3])] \
                  + list(WRIST_NEUTRAL)
        # Skip if already close enough (idempotent) — compare arm slides
        # and wrist together so a stale wrist tilt forces a re-pose.
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
        """Pick ARM1 to the object at obj_world.  Runs in a background
        thread.  Caller: play_m1.py after base nav has positioned the
        robot at a virtually-screened pick standoff.

        `side_grip_push_target`: when set (typically 0.52 m), enables
        the post-descent chassis-push flow:
          1. IK plans from a VIRTUAL chassis position at this distance
             from obj (so the arm picks a low-tilt comfortable pose)
          2. Approach + descent execute the arm
          3. Chassis pushes forward from current (~0.75 m) to the
             target distance — arm rides along, gripper lands on obj
          4. Close phase proceeds normally
        Without it, the executor uses the real (current) chassis
        position for IK and skips the chassis push — the legacy
        top-down / diagonal flow.
        """
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
        # Clear any stale diagnostics from a previous attempt so the
        # next failure (if any) reflects only this pick.
        self.last_grasp_failure_info = None
        # Reset live-nudge counter for this pick — counted per
        # _pick_run invocation, not across the full play_m1 retry loop.
        self._pre_close_nudges_used = 0
        # Reset graceful-disengage flag (one chance per pick).
        self._pre_close_backup_used = False
        # STRICT-mode per-attempt state. `_strict_t0` anchors the
        # [STRICT t=...s] log timestamps; `_strict_retry_count` counts
        # in-attempt slip retries (separate from play_m1's outer retry
        # loop — STRICT internally re-closes with a stronger grip target
        # before reporting failure). `_strict_overdrive_delta` is
        # bumped up by STRICT_RETRY_GRIP_BUMP on each slip retry so the
        # close stroke pushes deeper for higher contact force.
        self._strict_t0 = time.time()
        self._strict_retry_count = 0
        self._strict_overdrive_delta = float(STRICT_OVERDRIVE_DELTA_INIT)
        # Force multiplier — bumped per slip retry so the next close
        # demands more contact force. Default 1.0 keeps the initial
        # close at the model-derived per-finger target.
        self._strict_force_multiplier = 1.0
        # `_strict_grasp_q` is captured at the moment the close stroke
        # fires; used by slip-retry to re-descend to the exact pose.
        self._strict_grasp_q = None
        # stage flags are NOT reset here on purpose.
        # pick() can be called multiple times per cycle (orbit retry
        # path), and we want the cycle-level "highest stage reached"
        # to be preserved across attempts. play_m1's
        # `_emit_cycle_metrics` resets the flags after reading them,
        # giving each cycle a clean starting point.
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

            # ── Compute the per-object wrist goal FIRST ──
            # Selects side / diagonal / top-down per obj geometry. The
            # mode determines whether compute_grasp_targets adds an
            # above-Z offset to the pre-grasp target: side approach
            # lands the wrist at obj height directly, while
            # diagonal/top-down needs ABOVE_OBJ_HEIGHT so the descent
            # phase has somewhere to go.
            wrist_goal = self._compute_wrist_goal_for_obj(obj_bid)
            side_grip = is_side_approach(wrist_goal)
            mode_str = ("SIDE" if side_grip
                        else "DIAGONAL" if wrist_goal[0] > -0.70
                        else "AGGRESSIVE_TOPDOWN")
            # Publish grip mode to instance state so _curl_targets and
            # _thumb_open_pos_for_mode can adapt the open-finger pose:
            # side-grip uses symmetric thumb (same j1 as sides) so
            # thumb and bc tips are equidistant from palm; top-down
            # uses the deeper THUMB_OPEN_POS for descent clearance.
            self._side_grip_active = bool(side_grip)

            # Note: object contacts are NOT softened at the start of
            # the pick. Soft contacts during the close phase would let
            # fingers compress past the surface. Softening is applied
            # only after physical contact is established (at the
            # smooth-lift pin install or verify-grasp).
            print(f"[Exec] wrist_goal (hb,wz,wx,wy) = "
                  f"({wrist_goal[0]:+.2f},{wrist_goal[1]:+.2f},"
                  f"{wrist_goal[2]:+.2f},{wrist_goal[3]:+.2f})  mode={mode_str}")

            # ── Compute virtual chassis (post-push) FIRST ──
            # Side-grip + push-target mode: plan IK from the FUTURE
            # chassis position (post-push), not the current one. This
            # is the user-validated architecture — the arm picks a
            # comfortable low-tilt pose at the future position, then
            # the chassis pushes forward AFTER descent so the gripper
            # lands on obj. Without this, the IK would have to span
            # 22 cm of horizontal reach and pick an extreme tilt that
            # clips the chassis.
            # Compute the virtual chassis BEFORE deriving
            # pre_grasp_target so the standoff target is anchored at
            # the same chassis position the IK will use. Without this,
            # the approach_unit differs between real and virtual
            # chassis and the standoff lands off-axis.
            loc = self.sim.localization()
            obj_radius_for_standoff = self._object_radius(obj_bid)
            ik_base_xy = (float(loc[0]), float(loc[1]))
            ik_base_yaw = float(loc[2])
            push_dist_to_run = 0.0
            push_target_xy_world = None
            if side_grip and side_grip_push_target is not None:
                obj_xy_2d = np.asarray(obj_pos_snapshot[:2], dtype=float)
                real_xy = np.asarray([loc[0], loc[1]], dtype=float)
                dist_now = float(np.linalg.norm(obj_xy_2d - real_xy))
                if dist_now > float(side_grip_push_target):
                    push_dist_to_run = dist_now - float(side_grip_push_target)
                    approach_unit = (obj_xy_2d - real_xy) / dist_now
                    # The future chassis sits at (obj - target × approach_unit)
                    push_target_xy_world = (
                        obj_xy_2d - float(side_grip_push_target) * approach_unit)
                    ik_base_xy = (float(push_target_xy_world[0]),
                                  float(push_target_xy_world[1]))
                    # Plan the IK at the SAME yaw the chassis will have
                    # AFTER the side-grip push (the push re-aligns yaw
                    # to point at the object). Otherwise the arm —
                    # with its fixed joint angles — swings in the
                    # world frame when the chassis yaw corrects.
                    ik_base_yaw = float(math.atan2(
                        obj_xy_2d[1] - ik_base_xy[1],
                        obj_xy_2d[0] - ik_base_xy[0]))
                    yaw_delta_deg = math.degrees(
                        ((ik_base_yaw - float(loc[2]) + math.pi) % (2 * math.pi))
                        - math.pi)
                    print(f"[Exec] side-grip push planned: virtual chassis "
                          f"({ik_base_xy[0]:.3f},{ik_base_xy[1]:.3f}) "
                          f"= obj − {side_grip_push_target:.2f}m × approach_unit  "
                          f"(actual push after descent: {push_dist_to_run*100:.1f}cm; "
                          f"IK yaw {math.degrees(ik_base_yaw):+.1f}° vs current "
                          f"{math.degrees(loc[2]):+.1f}°, Δ={yaw_delta_deg:+.2f}°)")

            # ── Compute IK targets using the virtual chassis ──
            # For side approach: pre_grasp_target Z = obj_z (wrist lands
            # at obj middle, fingers reach horizontally).
            # For diagonal/top-down: pre_grasp_target Z = obj_z + 0.10
            # (column descent then lowers fingers onto obj top).
            # MIN_PICK_WRIST_Z = 0.10 is the soft IK lower bound; the
            # validity checker still rejects unreachable poses.
            # Anchor approach_unit at ik_base_xy so the standoff target
            # is colinear with the IK chassis → obj line.
            approach_base_xy = ik_base_xy if push_target_xy_world is not None \
                else (float(loc[0]), float(loc[1]))
            _grasp_unused, pre_grasp_target = compute_grasp_targets(
                approach_base_xy, obj_pos_snapshot,
                obj_radius=obj_radius_for_standoff,
                side_approach=side_grip)
            pre_grasp_target[2] = max(pre_grasp_target[2], MIN_PICK_WRIST_Z)

            reset_plan_data_for_ik(self.arm_bridge,
                                   base_xy=ik_base_xy,
                                   base_yaw=ik_base_yaw)
            # Write the OPEN finger qpos to planning_data before the
            # IK runs so the validity check sees the actual runtime
            # gripper geometry. Without this, the closed-by-default
            # fingers under-report chassis intrusion and the IK can
            # accept a pose whose open thumb will clip the chassis.
            try:
                # In side-grip mode, thumb opens to the same angle as
                # b/c (symmetric). In top-down, thumb opens deeper for
                # descent clearance. Must match `_curl_targets` so the
                # IK validity check sees the actual runtime geometry.
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

            # STRICT side-grip pinch-midpoint alignment is now handled
            # in the IK section below via TWO-PASS IK (probe at the
            # actual converged seed, not a fixed seed) — see the
            # solve_ik_with_z_lift block. Removed the fixed-seed
            # pre-shift here; the seed mismatch between the fixed
            # probe pose (h1=0.05, h2=0.45, a1=0.55) and the IK's
            # actual converged pose was causing a ~15 cm error in
            # the offset, defeating the whole alignment.
            print(f"[Exec] pre_grasp_target = {pre_grasp_target.round(3)}  "
                  f"IK base=({ik_base_xy[0]:.3f},{ik_base_xy[1]:.3f}) "
                  f"yaw={math.degrees(ik_base_yaw):.1f}deg  "
                  f"(IK validity uses OPEN-finger geometry)")

            # Strict z-lift IK returns (q, actual_target). If requested
            # z is unreachable, the helper lifts z until reachable and
            # reports the actual reached target.
            # IMPORTANT: when the side-grip chassis push is active
            # (push_target_xy_world is set), the screened q is INVALID
            # because it was computed by play_m1 from the candidate's
            # nav position (~0.75 m from obj), NOT from the virtual
            # post-push chassis position (~0.52 m from obj). Using
            # it would put the arm at a pose 22 cm off after the
            # push. Always recompute IK in side-grip-push mode.
            # STRICT side-grip local retry: skip IK entirely. The prior
            # attempt's IK + descent left the arm at its GRASP_Q pose;
            # the local retry's only job is the chassis back-then-forward
            # maneuver to close the carry_gap residual. Re-running IK
            # at the nudged-closer chassis position consistently fails
            # (the wrist orientation constraints + closer chassis put
            # the IK in infeasible territory), so the prior pose is
            # what we want anyway. Use the cached
            # `_last_valid_pre_grasp_q` from attempt 1.
            use_local_retry_skip_ik = (
                STRICT_PICKUP_MODE and side_grip and is_local_retry
                and self._last_valid_pre_grasp_q is not None)
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
                  and push_target_xy_world is None):
                use_screened = True
                PRE_GRASP_Q = [float(v) for v in screened_pre_grasp_q]
                # The screened q may have come from a 4-DOF caller in
                # play_m1; pad up to 8-DOF with the computed wrist_goal
                # so the rest of the pipeline gets the tilt.
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
                    # Side-grip mode: target PALM directly with per-DOF
                    # wrist weights so HandBearing stays FREE (a 5th reach
                    # DOF that swings the palm via the Hand_Bearing pivot)
                    # while wrist_Z (thumb-fingers spin), wrist_X (palm
                    # leveling), and wrist_Y stay LOCKED at the gripper-
                    # frame orientation goal. This lets SLSQP use both
                    # arm tilt (h2 > h1 brings boom down) AND HandBearing
                    # to reach low, with the wrist actuators holding the
                    # gripper level in world frame.
                    # Diagonal/top-down mode: target Link1 (legacy) and
                    # keep wrist_weight high so HandBearing sits at goal.
                    if side_grip:
                        ik_target_body  = "Gripper_Link3_1"
                        # (hb_w, wz_w, wx_w, wy_w): hb free, others held
                        ik_wrist_weight = (0.10, 3.0, 3.0, 3.0)
                    else:
                        ik_target_body  = "Gripper_Link1_1"
                        ik_wrist_weight = 5.0
                    # The high-tilt seed family plus the manual-pose
                    # seed reaches the target basin with four seeds.
                    # Warm-start: if a prior IK in this retry cycle
                    # succeeded, pass its q as seed-0 and bump n_seeds
                    # so cold-seed collapse on marginal poses is
                    # avoided.
                    _warm_seed = self._last_valid_pre_grasp_q
                    _ik_seeds  = 6 if _warm_seed is not None else 4
                    if _warm_seed is not None:
                        print(f"[Exec] IK warm-start: using last valid "
                              f"PRE_GRASP_Q as seed-0  n_seeds={_ik_seeds}")
                    # max_lift = 0.40 m allows the side-grip IK to lift
                    # the palm target up to 40 cm above the requested
                    # obj-middle z when the exact target is unreachable
                    # with side-grip wrist constraints. Tighter caps
                    # (0.05, 0.15) cause IK failure at most candidates
                    # because the side-grip wrist orientation can't be
                    # maintained with palm below ~0.30 m for this
                    # robot's parallel-mechanism kinematics.
                    _ik_max_lift = 0.40

                    # Two-pass IK was tried (probing pinch_midpoint
                    # offset at converged q1 then re-solving with the
                    # shifted target) but found that pass 1 already
                    # places the pinch midpoint at obj — the standoff
                    # baked into `compute_grasp_targets` (0.05 m in
                    # approach direction) is exactly compensated by
                    # the natural palm→pinch-midpoint offset in
                    # open-finger side-grip pose.  Subtracting the
                    # offset would double-count, putting palm at
                    # obj-10cm — past the side-grip kinematic limit.
                    # Single-pass IK at the original palm target is
                    # used instead.  The remaining ~14 cm IK-vs-runtime
                    # carry_gap is wrist actuator compliance
                    # (HB/wz/wx stiffness=10 sags ~4° under gripper
                    # weight) — addressed via the smaller side-grip
                    # push target in play_m1, not via IK.
                    PRE_GRASP_Q, actual_pre_target = \
                        self.arm_bridge.solve_ik_with_z_lift(
                            pre_grasp_target, n_seeds=_ik_seeds,
                            wrist_goal=wrist_goal,
                            wrist_weight=ik_wrist_weight,
                            target_body=ik_target_body,
                            seed_q=_warm_seed,
                            max_lift=_ik_max_lift)
                    # Cache for next retry within this pickup cycle.
                    self._last_valid_pre_grasp_q = list(PRE_GRASP_Q)

                    # ── Approach-dependent wz refinement (side-grip) ──
                    # Probe the actual thumb→bc world axis under the
                    # candidate q; if it's not perpendicular to the
                    # approach, adjust wz and re-solve once.
                    if side_grip:
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
                                # Apply candidate q to planning_data
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
                                # Reach quality for the ORIGINAL q —
                                # adopt the refined wz only when the
                                # measured reach (max_far / carry_gap)
                                # improves, not on axis-only.
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
                                        # Verify the refinement landed
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
                                        # Reach quality for the
                                        # REFINED q.
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
                                        # Adopt if WORST-side reach
                                        # improved by ≥ 5 mm OR pinch
                                        # midpoint improved by ≥ 1 cm
                                        # without worsening the worst
                                        # side by > 1 cm. Reach is the
                                        # primary metric (axis error
                                        # is a secondary diagnostic).
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

            # GRASP_Q == PRE_GRASP_Q: the arm cannot physically reach
            # obj_z for floor objects, so fingers close at the achievable
            # wrist height and the pin closure carries the object
            # thereafter at the recorded grasp offset.
            GRASP_Q = list(PRE_GRASP_Q)
            print(f"[Exec] PRE_GRASP_Q (actual_target z={actual_pre_target[2]:.3f}m) "
                  f"= {[round(x,3) for x in PRE_GRASP_Q]}")
            h_diff = abs(float(PRE_GRASP_Q[1]) - float(PRE_GRASP_Q[0]))
            # Mode-aware tilt cap: side-grip uses arm tilt as a reach DOF.
            h_diff_cap = MAX_PICK_H_DIFF_SIDE if side_grip else MAX_PICK_H_DIFF
            if h_diff > h_diff_cap or float(PRE_GRASP_Q[2]) < MIN_PICK_A1:
                print(f"[Exec] PRE_GRASP_Q rejected for visual safety: "
                      f"h2-h1={h_diff:.3f}m (max {h_diff_cap:.2f}, "
                      f"mode={'SIDE' if side_grip else 'TOPDOWN'}), "
                      f"a1={float(PRE_GRASP_Q[2]):.3f}m (min {MIN_PICK_A1:.2f})")
                self._clear_held_state()
                fire(False)
                return

            # ── Build PRE_HOVER_Q (gradient-tilt descent) ──
            # User-suggested architecture: instead of pre-tilting the arm
            # at hover (so the wide-open thumb projects over the chassis
            # for the whole approach), keep the arm MORE LEVEL and the
            # WRIST NEUTRAL at the hover pose. The descent then
            # progressively grows the arm tilt AND rotates the wrist to
            # the grasp orientation. Result:
            # During approach + most of descent: arm near-level,
            # wrist neutral → thumb in natural forward position →
            # no chassis-overlap geometry.
            # At end of descent: full tilt + full wrist orientation
            # → grasp pose, but now at obj height, well below the
            # chassis upper structure.
            # The linear-interp `_kinematic_descent` handles the
            # gradient automatically — just need PRE_HOVER to differ
            # from GRASP in the dims we want to interpolate.
            if USE_HOVER_DESCENT:
                # Column height: equal-column lift above the GRASP mid.
                # Average the two columns so PRE_HOVER is near-level.
                h_mid_grasp = 0.5 * (float(GRASP_Q[0]) + float(GRASP_Q[1]))
                # Keep a SMALL tilt at PRE_HOVER (≥ ALPHA_MIN_DEG=1.5°
                # ⇒ |h2-h1| ≥ 0.003 m) in the same direction the GRASP
                # pose has so the descent grows tilt monotonically.
                # 0.04 m gives ~21° alpha, comfortably valid.
                small_tilt   = 0.04
                diff_dir     = 1.0 if (GRASP_Q[1] >= GRASP_Q[0]) else -1.0
                h_lifted     = h_mid_grasp + HOVER_LIFT
                h1_hover     = min(h_lifted - 0.5 * small_tilt * diff_dir,
                                   COLUMN_JOINT_MAX)
                h2_hover     = min(h_lifted + 0.5 * small_tilt * diff_dir,
                                   COLUMN_JOINT_MAX)
                # Wrist orientation at pre-hover. If the current arm
                # wrist is closer to GRASP_Q's wrist than to neutral,
                # this is a retry — inherit the GRASP wrist at
                # pre-hover so the approach does not waste a 100° + 100°
                # round trip through neutral. Otherwise keep neutral
                # for the fresh-start case (thumb forward during the
                # long approach avoids chassis-overlap in transit).
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
                # OMPL state-validity check.
                hover_valid = False
                try:
                    hover_valid = bool(self.arm_bridge.is_valid(PRE_HOVER_Q))
                except Exception as e:
                    print(f"[Exec] PRE_HOVER_Q is_valid check raised: {e}")
                if not hover_valid and hover_wrist is not WRIST_NEUTRAL:
                    # If chosen hover_wrist makes PRE_HOVER invalid
                    # (rare, e.g. grasp-wrist + low column clips chassis),
                    # fall back to neutral wrist.
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
                    # Fall back: try the legacy same-orientation hover
                    # (inherits GRASP_Q's tilt + wrist). Logs the
                    # fallback so we know the gradient idea didn't apply.
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
            else:
                PRE_HOVER_Q = list(GRASP_Q)

            # ── [3] Open gripper ──
            # USE_SYNC_OPEN=True (default): drive ctrl to OPEN values and
            # wait for joints to physically settle BEFORE the approach
            # starts. Eliminates the race where the arm descended onto
            # the cylinder while the thumb hangs low.
            # USE_SYNC_OPEN=False (legacy): open runs in a daemon thread
            # concurrent with the approach; the race resurfaces if
            # the approach OMPL plan resolves quickly.
            open_thread = None
            # Local-retry path: when invoked from play_m1's chassis-only
            # retry the arm is already at GRASP_Q. Skip [3]/[4]/[5] and
            # run a chassis back-then-forward maneuver to release any
            # gripper-vs-object contact and re-engage with the corrected
            # alignment.
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

            # ── [4] OMPL approach current → PRE_HOVER_Q ──
            # Approach ends at PRE_HOVER_Q (HOVER_LIFT above the grasp
            # pose) so the swept arc stays well above the cylinder.
            # Final straight-down motion is in [5] descent.
            if is_local_retry:
                print("[Exec] [4] approach — SKIPPED "
                      "(local retry, arm at GRASP_Q already)")
                print("[Exec] [5] descent — SKIPPED "
                      "(local retry, arm at GRASP_Q already)")
            else:
                print("[Exec] [4] approach: current → PRE_HOVER_Q")
                q_now = self._current_arm_q()
                # 3.0 s — RRTConnect finds approach paths in
                # ~0.3-1.0 s typically; 3 s is plenty and keeps the
                # worst-case latency before the arm starts moving low.
                path1 = self.arm_bridge.plan(q_now, PRE_HOVER_Q, timeout=3.0)
                if path1 is None:
                    # OMPL rejected the start state (most commonly the
                    # h1==h2 singular PARK_Q on the very first pick, where
                    # arm_planner.is_valid requires alpha >= ALPHA_MIN_DEG).
                    # Nudge h1/h2 apart into a nearby valid pose and replan.
                    print("[Exec] OMPL approach start invalid/unavailable — "
                          "unlocking PARK_Q then replanning")
                    avg_h = 0.5 * (float(q_now[0]) + float(q_now[1]))
                    # Carry wrist from q_now so the unlock move doesn't
                    # snap the wrist to neutral mid-approach.
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
                        # Last-resort fallback: direct kinematic approach so
                        # the log is explicit that this was not OMPL-planned.
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

                # ── [5] Vertical descent PRE_HOVER_Q → GRASP_Q ──
                # Column-only motion (a1 and th held constant) → pure vertical
                # Cartesian descent. When USE_HOVER_DESCENT is False this is
                # a no-op (PRE_HOVER_Q == GRASP_Q), matching legacy behavior.
                # Step count scales with HOVER_LIFT (~0.9 cm per step).
                print("[Exec] [5] descent: PRE_HOVER_Q → GRASP_Q")
                self._kinematic_descent(PRE_HOVER_Q, GRASP_Q, "descent",
                                        n_steps=HOVER_DESCENT_STEPS)
                if self._cancel:
                    self._clear_held_state(); fire(False); return

            # ── [5.4] Side-grip chassis push ──
            # Arm now at its pre-grasp pose (IK planned for the future
            # chassis position). Push the chassis forward (with L/R
            # compensation included in the target XY) so the arm + gripper
            # translate into final position over the obj.
            # Local-retry path: the chassis is already at the
            # play_m1-aligned target but the arm is still at GRASP_Q,
            # so the gripper may be in contact with the object. A
            # back-then-forward chassis maneuver releases any contact
            # and re-engages with the corrected alignment. The
            # forward step targets `current + residual` where
            # `residual` is the obj-to-alignment-point delta saved
            # at the prior rejection.
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

                # Step 1: chassis BACK along reverse-approach by 8 cm,
                # but ONLY if a finger is currently touching obj. The
                # back-step exists to release pre-existing gripper↔obj
                # contact so the forward step doesn't slam. When the
                # gripper is still 10+ cm short of obj (typical
                # rejection case) the back-step is pure waste motion
                # that leaves the chassis farther from obj than where
                # it started — the contact-guard fires on STEP 2 well
                # before the chassis has recovered the lost ground.
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
                # Snapshot BEFORE the chassis push so we can see the
                # post-descent state with no other motion happening.
                # The user-observed "palm reached obj" moment likely
                # falls between this snapshot and the post-settle one.
                if USE_FINGER_DIAGNOSTIC_LOG:
                    self._log_finger_geometry(
                        obj_bid, "post-descent (pre-push)")
                    self._log_finger_object_contacts(
                        obj_bid, "post-descent (pre-push)")
                self._side_grip_chassis_push(push_target_xy_world,
                                             obj_pos_snapshot[:2])
                if self._cancel:
                    self._clear_held_state(); fire(False); return
                # Snapshot AFTER chassis push but BEFORE wrist-settle.
                # Wrist-settle takes ~0.5s and during that window the
                # wrist re-orients, which moves the palm via the chain.
                # The user's observed "palm in contact" moment is most
                # likely visible here.
                if USE_FINGER_DIAGNOSTIC_LOG:
                    self._log_finger_geometry(
                        obj_bid, "post-push (pre-settle)")
                    self._log_finger_object_contacts(
                        obj_bid, "post-push (pre-settle)")

            # ── [5.45] Wait for wrist to settle ──
            # The descent's kinematic interpolation finishes when the
            # column joints reach their targets, but the wrist still
            # has PD residual against gravity + finger inertia. Even a
            # few degrees of residual on wz rotates the closure axis
            # enough to drop b/c fingertips out of reach.
            # Wait for the wrist residuals before measuring
            # pinch midpoint. Side-grip needs tighter tolerance because
            # the wz residual directly steers the closure axis.
            # Pass the INTENDED IK target wrist values (GRASP_Q[4:8])
            # so the settle compares qpos to the planned orientation,
            # not the PD-pre-compensated ctrl values. Otherwise wz
            # always reports residual=0.17 rad (the pre-comp overshoot)
            # and the gate fails to detect a properly-settled wrist.
            intended_wrist = tuple(float(GRASP_Q[i]) for i in (4, 5, 6, 7))
            if side_grip:
                # 0.06 rad ≈ 3.4° — tight enough to keep the closure
                # axis aligned, loose enough to skip the last 1° of PD
                # residual.
                self._wait_for_wrist_settle(
                    tolerance=0.06, timeout=0.5,
                    label="wrist-settle (side-grip)",
                    intended_targets=intended_wrist)
            else:
                self._wait_for_wrist_settle(
                    tolerance=0.10, timeout=0.8,
                    label="wrist-settle (top-down)",
                    intended_targets=intended_wrist)
            if self._cancel:
                self._clear_held_state(); fire(False); return

            # ── [5.46] Runtime wz correction (side-grip only) ──
            # Closes the IK-vs-runtime axis drift left over from the
            # planning-time refine. Measures the live finger axis,
            # commands a small wrist_Z step to align it perpendicular
            # to the approach, and adopts only on reach improvement.
            if side_grip:
                self._runtime_wz_correction(
                    obj_bid,
                    ik_base_xy[0:2] if isinstance(ik_base_xy, (list, tuple))
                    else (float(ik_base_xy[0]), float(ik_base_xy[1])))

            # (Disabled staged-tilt branch. Rotating the boom around
            # h1 after the IK solve shifts the palm tangentially off
            # the object XY and breaks the chassis-push alignment.
            # `_post_descent_extra_tilt` is retained for reference.)

            if (STRICT_PICKUP_MODE and side_grip
                    and (push_target_xy_world is not None
                         or is_local_retry)):
                try:
                    obj_xy_align = self.sim.data.xpos[obj_bid][:2].copy()
                    carry_xy_align = self._carry_anchor_xyz(
                        self.sim.data)[:2].copy()
                    residual_align = obj_xy_align - carry_xy_align
                    residual_mag_align = float(np.linalg.norm(
                        residual_align))
                    # Pre-existing finger-obj contact: contact-guard
                    # would abort the snap immediately; skip with a log.
                    any_finger_touching_align = (
                        self._finger_body_groups is not None
                        and any(self._finger_touches_obj(fi, obj_bid)
                                for fi in range(3))
                    )
                    CLOSED_LOOP_ALIGN_MIN = 0.03   # below: no point
                    CLOSED_LOOP_ALIGN_CAP = 0.10   # max per-snap motion
                    CLOSED_LOOP_ALIGN_ITERS = 2     # multi-snap iterations
                    CLOSED_LOOP_ALIGN_OK = 0.06    # stop when residual ≤ this
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
                        # removed the 12 cm MAX skip. Even
                        # for residuals > 12 cm, a 10-cm cap snap closes
                        # part of the gap. Run up to 2 iterations to
                        # converge. Each iteration is contact-guarded;
                        # if thumb touches obj mid-snap, the iteration
                        # aborts safely.
                        last_residual = residual_mag_align
                        for _snap_iter in range(CLOSED_LOOP_ALIGN_ITERS):
                            # Re-observe each iteration so the snap
                            # vector points toward obj from the CURRENT
                            # carry_anchor, not the original one.
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
                            # Re-check finger-obj contact each iter —
                            # may have made contact after prev snap.
                            if (self._finger_body_groups is not None
                                    and any(self._finger_touches_obj(fi, obj_bid)
                                            for fi in range(3))):
                                self._strict_log(
                                    "ALIGN",
                                    f"closed-loop iter {_snap_iter+1} "
                                    f"STOPPED — finger now touching obj "
                                    f"after prev snap")
                                break
                            # Cap snap magnitude.
                            snap_vec = residual_iter.copy()
                            if residual_iter_mag > CLOSED_LOOP_ALIGN_CAP:
                                snap_vec = snap_vec * (
                                    CLOSED_LOOP_ALIGN_CAP / residual_iter_mag)
                            cur_xy_iter = np.asarray(
                                self.sim.localization()[:2], dtype=float)
                            snap_target = cur_xy_iter + snap_vec
                            snap_mag = float(np.linalg.norm(snap_vec))
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
                            # Track residual for the final summary log
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

            # Diagnostic: after descent (and after chassis push if any).
            # Logs per-finger XYZ + delta to cylinder + any pre-existing
            # contacts so we can see which finger(s) are out of position.
            if USE_FINGER_DIAGNOSTIC_LOG:
                self._log_finger_geometry(obj_bid, "post-descent (pre-close)")
                self._log_finger_object_contacts(obj_bid, "post-descent (pre-close)")

            # Floor / chassis collision audit — what (if anything) on the
            # gripper or arm chain is touching the floor or chassis?
            self._log_gripper_floor_chassis_contacts("post-descent (pre-close)")

            # Categorized contact summary — disambiguates the visual
            # contact dots (47+ total ncon includes wheels, joints,
            # obj-floor; only finger↔obj actually matters for the pick).
            self._log_filtered_contact_summary(obj_bid,
                                               "post-descent (pre-close)")

            # ── [5.5] Pre-close acceptance gate (Path B Step 1) ──
            # Primary metric: depends on grip mode.
            # TOP-DOWN / DIAGONAL: carry_anchor (centroid of 3 fingertip
            # pockets) sits OVER the obj — fingers converge from above.
            # SIDE-GRIP: thumb is INTENTIONALLY on the opposite side of
            # the obj from the 2 side fingers (b, c); the obj should sit
            # BETWEEN them. The 3-fingertip centroid gets pulled toward
            # the thumb side, giving a misleading large gap even when
            # the obj is correctly positioned for closure. Use the
            # PINCH MIDPOINT (halfway between thumb tip and bc-centroid)
            # instead — that's where the obj should physically end up.
            # Read directly from sim.data — do NOT call mj_forward from
            # this background thread (MuJoCo is not thread-safe for
            # concurrent operations on a single MjData).
            if side_grip:
                align_pt_xy = self._pinch_midpoint_xyz(self.sim.data)[:2].copy()
                align_metric_name = "pinch_midpoint"
            else:
                align_pt_xy = self._carry_anchor_xyz(self.sim.data)[:2].copy()
                align_metric_name = "carry_anchor"
            grip_xyz_pre = self.sim.data.xpos[self.gripper_body_id].copy()
            obj_xyz_pre  = self.sim.data.xpos[obj_bid].copy()
            carry_gap    = float(np.linalg.norm(align_pt_xy - obj_xyz_pre[:2]))
            # Per-side reach (side-grip only): distance from thumb tip
            # AND from bc-centroid to the obj. pinch_midpoint averages
            # these so an asymmetric mismatch (one side close, the other
            # far) passes the average gate but the close stroke can't
            # physically reach the far side.
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
            # obj-radius aware carry_gap tolerance.
            # The fixed CARRY_GAP_TOLERANCE_SIDE = 6 cm was tuned for
            # smaller obj radii (≤ ~5 cm), where palm at 6 cm from obj
            # centre = palm 1 cm OUTSIDE the obj surface. For current
            # obj radii (6.8-7.5 cm) the same 6 cm tolerance demands
            # palm be 0.5-1.5 cm INSIDE the obj — guaranteed arm-
            # clipping rejection. Scale the tolerance with obj radius
            # so it matches the chassis-push stop criterion
            # (obj_r + 1.5 cm). The arm-clip check still catches any
            # genuine over-penetration.
            try:
                _obj_r_gate = float(self._object_radius(obj_bid))
            except Exception:
                _obj_r_gate = 0.0
            CARRY_GAP_PALM_CLEARANCE = 0.040  #  matches BC_REACH_STOP_MARGIN — accepts direct-rotation poses that produce pinch_midpoint ~11cm from obj when fingers are equally spaced 11cm from obj. Looser than PINCH_MIDPOINT_STOP_MARGIN so visually-aligned poses no realistic chassis can hit on a 5mm margin are not rejected.
            if side_grip and _obj_r_gate > 0.0:
                carry_gap_tol = _obj_r_gate + CARRY_GAP_PALM_CLEARANCE
            else:
                carry_gap_tol = (CARRY_GAP_TOLERANCE_SIDE if side_grip
                                 else CARRY_GAP_TOLERANCE)

            # Realism-mode extras (side-grip only). Two extra checks
            # gate the close motion when realism is active:
            # * Z gap — palm must be within REALISM_PRE_CLOSE_Z_GAP
            # of obj_top, otherwise close can't reach.
            # * Contact count — at least 1 finger-obj contact OR
            # proximity within close-stroke reach.
            palm_z = float(grip_xyz_pre[2])
            obj_top_z = float(obj_xyz_pre[2] + self._object_half_height(obj_bid))
            z_gap = palm_z - obj_top_z       # +ve: palm above obj_top
            try:
                finger_obj_ncon = int(self._count_finger_obj_contacts(obj_bid))
            except Exception:
                finger_obj_ncon = 0
            # Hard reject signal: arm STRUCTURE (boom/palm/wrist body —
            # NOT fingers) touching obj at pre-close. Close fires
            # would intrude further, push obj, or fight contact forces.
            try:
                arm_obj_ncon = int(self._count_arm_obj_contacts(obj_bid))
            except Exception:
                arm_obj_ncon = 0
            realism_active = bool(REALISM_MODE_NO_SMOOTH_LIFT and side_grip)

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
            # Primary check: alignment point must be near object.
            # Graduated tolerance:
            # tight tol (≈ 6 cm): close fires directly
            # micro-lift threshold (≈ 12 cm): the smooth-lift
            # animation bridges the residual when realism is active
            # above threshold: reject, retry
            effective_carry_tol = (REALISM_MICRO_LIFT_THRESHOLD
                                   if realism_active
                                   else carry_gap_tol)
            carry_ok = (carry_gap <= effective_carry_tol)
            # Legacy Link1-based gate (ik_dev + obj_grip_xy). For
            # side-grip the gripper extends ~10 cm forward of Link1, so
            # obj_grip_xy is naturally 12-18 cm even on a clean pinch —
            # rejecting on the 0.15 m bar masks valid micro-lift poses.
            # Trust carry_gap (pinch-midpoint metric) in realism+side mode.
            if realism_active and side_grip:
                legacy_ok = True
            else:
                legacy_ok = (ik_dev <= GRIP_DEVIATION_TOLERANCE
                             and obj_grip_xy <= obj_xy_gate)
            # Per-side reach gate (side-grip only). Both thumb-to-obj
            # AND bc-centroid-to-obj must be within close-stroke reach.
            # Rejects asymmetric geometries where pinch_midpoint passes
            # on average but one side is too far to physically close.
            # STRICT mode keeps this gate strict: chassis translation
            # cannot fix orientation mismatches, so asymmetric reach
            # must REJECT here and let play_m1 retry with a different
            # base candidate (different yaw → different gripper
            # orientation in world frame → potentially aligned thumb-bc
            # axis with obj).  Relaxing the gate produces thumb-only
            # closes (1/3 verify fails) and large thumb penetration
            # spikes.
            if side_grip:
                # v2 (v3): small gate bump in --perfect.
                # Gripper-structural asymmetry adds 5-6cm to bc anchor
                # vs thumb (palm-depth offset). With nav-yaw filter
                # rejecting truly bad approaches and v3 dampers eliminating
                # PD spike risk, allowing d_bc up to 0.15m in --perfect
                # admits feasible-but-structurally-asymmetric configs.
                _reach_max = (SIDE_FINGER_PRECLOSE_REACH_PERFECT
                              if STRICT_PERFECT_FRICTION_ONLY
                              else SIDE_FINGER_PRECLOSE_REACH)
                sides_ok = (d_thumb_obj <= _reach_max
                            and d_bc_obj <= _reach_max)
            else:
                sides_ok = True
            # Realism checks (only enforced when realism_active).
            z_ok = (z_gap <= REALISM_PRE_CLOSE_Z_GAP) if realism_active else True
            # Pre-close contact is informational only: open fingers hover
            # AROUND the object without touching it; the close stroke
            # itself is what produces contact. The carry_gap + z_gap
            # geometry is the close-success signal.
            contact_reach_ok = True

            # ── [5.5b] Live chassis nudge ──
            # If the gate fails because the palm is at the right Z but
            # the wrong XY, nudge the chassis by the residual while the
            # arm stays at GRASP_Q. The gripper translates with the
            # chassis, so the residual closes without needing to lift
            # the arm, re-solve IK, or re-execute approach. Capped at
            # MAX_NUDGES iterations.
            MAX_NUDGES = 3   # cap on iterations within one gate call.
                             # Each nudge reduces residual by ~40% (PD
                             # undershoot + obj slip), so 3 nudges
                             # converge ~11.8→7→4→2 cm — under the 9 cm
                             # side-reach bar.
            NUDGE_MAX_RESIDUAL = 0.25   # m — bigger residuals need a full retry
            xy_residual_vec = obj_xyz_pre[:2] - align_pt_xy
            xy_residual_mag = float(np.linalg.norm(xy_residual_vec))
            xy_fail = (not carry_ok) or (not contact_reach_ok) or (not sides_ok)
            # ── Multi-nudge loop with per-iteration safety checks.
            # Each iteration: safety pre-check → nudge → re-measure →
            # convergence check + arm-chassis-contact check. Aborts
            # cleanly on any safety trip; final gate decision is made
            # by the existing rejection block after the loop.
            import time as _t
            nudge_iter = 0
            arm_chassis_ncon_pre = self._count_arm_chassis_contacts()
            # Skip the nudge loop in FAST mode — the gate override fires
            # on the initial geometry, the smooth-lift animation aligns
            # the object, and the close fires without a visible dance.
            # Skip in STRICT too — the nudge drags obj ahead of the
            # gripper (visible in run logs as d_bc getting WORSE after
            # each nudge). STRICT instead REJECTS asymmetric reach so
            # play_m1 retries with a different base candidate
            # (different yaw → different gripper orientation → may
            # align thumb-bc axis with obj).
            # ALWAYS fire axis-orbit in --perfect when
            # delta > 5°, regardless of sides_ok. Gate sides_ok=True
            # passed cases still produce one-sided grips because anchor
            # distance is OK but thumb-bc axis tilted relative to obj-grip
            # direction = uneven compression = bc force≈0 at verify.
            if (STRICT_PICKUP_MODE
                    and STRICT_PERFECT_FRICTION_ONLY
                    and side_grip
                    and len(self._carry_anchor_body_ids) == 3):
                if True:  # always-check block
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
                        # Desired axis: perpendicular to obj-grip direction.
                        # Try both perpendicular orientations, pick smaller delta.
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
                        # Only orbit if delta in 5° to 60° range.
                        if math.radians(5.0) < abs(_delta) < math.radians(60.0):
                            # 3-phase retract-rotate-
                            # approach to avoid sweeping arm into obj.
                            # 1) RETRACT 10cm away from obj
                            # 2) ROTATE yaw at retracted position
                            # 3) APPROACH back to original chassis-obj dist
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
                            # Above: rotate the chassis-to-obj direction
                            # vector by delta around obj, then approach to
                            # same radial distance. Net effect: chassis
                            # orbited around obj while preserving distance.
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
                            # Phase 1: retract
                            with self.sim._target_lock:
                                self.sim.target_base = np.array([
                                    float(_retract_xy[0]),
                                    float(_retract_xy[1]),
                                    float(_cur_yaw_o)])
                            time.sleep(1.2)
                            # Phase 2: rotate (still retracted)
                            with self.sim._target_lock:
                                self.sim.target_base = np.array([
                                    float(_retract_xy[0]),
                                    float(_retract_xy[1]),
                                    float(_new_yaw_o)])
                            time.sleep(1.2)
                            # Phase 3: approach
                            _new_xy_o = _approach_xy_after
                            with self.sim._target_lock:
                                self.sim.target_base = np.array([
                                    float(_new_xy_o[0]),
                                    float(_new_xy_o[1]),
                                    float(_new_yaw_o)])
                            time.sleep(1.5)
                            # Re-measure d_thumb / d_bc after orbit.
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
                # Safety pre-check: would this nudge bring chassis closer
                # than MIN_CHASSIS_OBJ_DIST to obj? If so, abort.
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
                # Persist across attempts so a re-entered gate would still
                # respect MAX_NUDGES across a full pick cycle.
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
                # Brief settle so contacts update after the push.
                _t.sleep(0.2)
                # Re-measure everything and re-evaluate gate.
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
                    sides_ok = (d_thumb_obj <= SIDE_FINGER_PRECLOSE_REACH
                                and d_bc_obj <= SIDE_FINGER_PRECLOSE_REACH)
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
                # Success: gate now passes — exit loop, let downstream
                # [5.7] smooth-lift handle the rest.
                if not xy_fail:
                    print(f"[Exec] [5.5b] gate passes after "
                          f"#{nudge_iter} nudge(s) — exiting loop")
                    break
                # Safety: arm-chassis contacts worsening — arm is
                # starting to clip the base. Stop nudging in this
                # direction.
                if arm_chassis_ncon_post > arm_chassis_ncon_pre:
                    print(f"[Exec] [5.5b] abort: arm-chassis contacts "
                          f"increased {arm_chassis_ncon_pre}"
                          f"→{arm_chassis_ncon_post} — arm starting to "
                          f"clip base; further nudges would worsen")
                    break
                # Safety: ASYMMETRIC IN-CONTACT pattern. If one finger
                # has already made contact with obj but the per-side gate
                # is still failing (the OTHER side is too far), translating
                # the chassis further will drive the in-contact finger
                # DEEPER into obj while the far side stays unreachable.
                # The asymmetry is a wrist-orientation mismatch, not a
                # position residual — chassis nudges cannot fix it.
                # Abort cleanly so play_m1 can try a new base candidate
                # (different yaw → different gripper orientation).
                if finger_obj_ncon >= 1 and not sides_ok and not STRICT_PICKUP_MODE:
                    # 2-finger accept: when ≥ 2 fingers are already in
                    # contact (e.g. b+c wrapping while thumb hovers
                    # farther out), the visible grip is essentially
                    # "fingers around object" — let the close stroke
                    # complete it instead of aborting.
                    if finger_obj_ncon >= 2:
                        print(f"[Exec] [5.5b] ASYMMETRIC 2-finger accept "
                              f"({finger_obj_ncon} finger(s) on obj, "
                              f"far side ="
                              f"{max(d_thumb_obj, d_bc_obj)*100:.1f}cm) — "
                              f"close stroke will complete the grip; "
                              f"pin closure holds for transport.  "
                              f"Skipping disengage + abort.")
                        # Override sides_ok so the rejection cascade
                        # doesn't reject on the original 9 cm bar.
                        # Other gates (carry_ok, arm_obj_ok, z_ok)
                        # still apply.
                        sides_ok = True
                        xy_fail = (not carry_ok) or (not contact_reach_ok)
                        if not xy_fail:
                            break
                        # If still failing on other gates, fall through
                        # to existing abort below.
                    # 1-finger + palm-anchor accept: when a single
                    # finger has just grazed the object (typically the
                    # thumb) AND all palm-anchor accept criteria are
                    # already satisfied, exit the [5.5b] loop cleanly
                    # and let the downstream palm-anchor tier handle
                    # the residual via the pin animation.
                    # Trigger requires ncon == 1 (single light contact);
                    # multi-point contact would indicate a deeper wrap
                    # and hits the 2-finger accept path above.
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
                    # GRACEFUL DISENGAGE recovery (one-shot per pick).
                    # Before truly aborting, back chassis 6 cm along
                    # reverse-approach to release the stuck finger and
                    # let obj re-settle. If asymmetry was caused by
                    # obj displacement during descent (case A), the
                    # geometry can recover. If it's pure orientation
                    # mismatch (case B), we'll fall through to abort
                    # but obj is now in a cleaner state for the next
                    # base candidate.
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
                            # Report whichever side is actually far.
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
                            # Re-measure after disengage
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
                            # Else: fall through to the abort below
                    # (Either backup already used, or backup didn't help)
                    far_side = ("bc" if d_bc_obj > d_thumb_obj else "thumb")
                    far_dist = max(d_thumb_obj, d_bc_obj)
                    # Phrasing depends on current contact state — after
                    # disengage, finger_obj_ncon may be 0.
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
                # Convergence: stop if a nudge didn't materially improve
                # carry_gap. PD slip + obj sliding can cancel each other.
                if improvement < NUDGE_MIN_CARRY_GAP_IMPROVEMENT:
                    print(f"[Exec] [5.5b] abort: no convergence "
                          f"(improvement {improvement*100:+.1f}cm < "
                          f"{NUDGE_MIN_CARRY_GAP_IMPROVEMENT*100:.1f}cm) "
                          f"— further nudges unlikely to help")
                    break
                # Update arm-chassis pre-state for next iteration.
                arm_chassis_ncon_pre = arm_chassis_ncon_post

            # ── ASYMMETRIC SOFT-ASSIST tier ──────────────────────────
            # Before rejecting on asymmetric reach, check whether the
            # case is bridgeable by smooth-lift MICRO: one finger
            # already in contact, pinch-midpoint near obj, far side
            # not extreme. If so, relax sides_ok so [5.7] smooth-lift
            # fires and animates obj into the pocket. Outside this
            # band the original rejection still fires.
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
                # Override sides_ok so the gate passes. [5.7]
                # smooth-lift MICRO mode (5 cm < carry_gap ≤ 12 cm)
                # will fire and animate obj toward the gripper centroid.
                sides_ok = True

            # ── PALM-ANCHOR ACCEPT tier ──────────────────────────────
            # When the gripper palm (Gripper_Link3_1) is positioned
            # correctly — below obj_top in Z, pinch midpoint near the
            # object XY, no arm-structure clipping — the close stroke
            # will wrap the fingers around the object from this scoop
            # pose. Fires only when the strict sides_ok gate would
            # otherwise reject AND the geometry sits inside the
            # palm-anchor accept band; outside the band the original
            # rejection still applies.
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

            # Hard reject: arm STRUCTURE clipping obj. Higher
            # priority than the alignment gates — if the arm body is
            # already inside obj, no close attempt can be valid.
            arm_obj_ok = (arm_obj_ncon == 0)

            # FAST_PICKUP_MODE override. When the flag is True for a
            # side-grip realism pickup, force the RECOVERABLE gates
            # (carry / sides / legacy) to pass so the pin closure can
            # rescue asymmetric reach or sub-threshold carry residuals.
            # `arm_obj_ok` stays strict — when the arm structure is
            # already clipping the object the close attempt cannot
            # produce a valid grasp, so we fall through to retry.
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
                # arm_obj_ok intentionally NOT forced — when False, the
                # rejection cascade still fires and triggers retry.

            if (not carry_ok or not legacy_ok or not z_ok
                    or not contact_reach_ok or not sides_ok
                    or not arm_obj_ok):
                if not arm_obj_ok:
                    reason = (f"arm structure clipping obj at pre-close "
                              f"({arm_obj_ncon} arm-vs-obj contact(s) — "
                              f"boom/palm/wrist intruding obj geom; "
                              f"close would damage or fail)")
                elif not sides_ok:
                    reason = (f"asymmetric reach: thumb={d_thumb_obj*100:.1f}cm "
                              f"bc={d_bc_obj*100:.1f}cm "
                              f"(both must be ≤ "
                              f"{SIDE_FINGER_PRECLOSE_REACH*100:.1f}cm — "
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
                # Per-side reach metrics passed to play_m1 so the
                # retry chain can detect worsening geometry and bail
                # to a different base candidate.
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
                # Direction-aware retract: decompose the residual
                # (obj - alignment_point) into "along approach axis"
                # vs "perpendicular". If mostly along approach
                # (within ±30° of the robot→obj direction), the next
                # base nudge moves the arm forward into the obj
                # without needing an arm lift — saves the entire
                # lift/lower cycle. Only lift for lateral errors.
                residual_xy = obj_xyz_pre[:2] - align_pt_xy
                robot_xy_now = self.sim.localization()[:2]
                approach_xy = obj_xyz_pre[:2] - np.asarray(robot_xy_now)
                a_n = float(np.linalg.norm(approach_xy))
                r_n = float(np.linalg.norm(residual_xy))
                skip_lift = False
                if a_n > 1e-6 and r_n > 1e-6:
                    cos_ang = float(np.dot(residual_xy, approach_xy)
                                    / (a_n * r_n))
                    if abs(cos_ang) > 0.85:  # within ±30° of approach
                        skip_lift = True
                        print(f"[Exec] residual is forward-aligned "
                              f"(cos={cos_ang:+.2f}, "
                              f"|r|={r_n*100:.1f}cm) — "
                              f"direction-aware retract: no arm lift")
                # Non-FAST side-grip rejection keeps the arm at GRASP_Q
                # so the next local retry can run a chassis back-then-
                # forward maneuver without re-planning the approach.
                # Other modes use the column-lift hover-retract.
                _use_side_grip_retry_mode = bool(
                    side_grip and not FAST_PICKUP_MODE)
                self._retract_after_failure(
                    "[5.6]",
                    skip_lift=skip_lift,
                    side_grip_retry=_use_side_grip_retry_mode)
                self._clear_held_state()
                fire(False)
                return

            # ── [5.7] Pre-close object alignment ──
            # Aligns the object into the gripper pocket before the close
            # stroke fires. Three-tier residual policy:
            # * Tiny residual (≤ NATURAL_CLOSE_DRAG_THRESHOLD): skip
            # animation; the close stroke drags the object via
            # physical contact.
            # * Medium residual (≤ REALISM_MICRO_LIFT_THRESHOLD): a
            # small animated translation bridges the gap.
            # * Above threshold: the gate has already rejected the
            # pose (defensive branch).
            # The natural-drag branch is suppressed in the palm-anchor
            # scoop tier — in that geometry the close arc would push
            # fingers into the object's upper body, so the pin
            # animation runs even at small residuals.
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
            # target_z = obj center 2.5cm above centroid → top sits at
            # finger-tip cage volume.
            target_z_pre = -(half_h_pre - 0.025)
            # Floor-aware Z clamp on the pin target. Without this,
            # short floor-resting cylinders can target an absolute Z
            # below the floor plane (the formula above is relative to
            # the pinch midpoint, which sits low for short objects).
            # The pin's kinematic override bypasses contact constraints
            # so the object would visually clip through the floor. The
            # clamp only kicks in when the original target would dip
            # below floor_z + a 2 mm safety margin.
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
            # XY offset for the smooth-lift target. Aligning the
            # gripper centroid with the NEAR edge of the object (the
            # side closer to the robot base) — instead of the centre
            # leaves the thumb clearance to close on the far surface
            # while the side fingers wrap the near surface.
            obj_r_pre = self._object_radius(obj_bid)
            try:
                _robot_xy = np.asarray(self.sim.localization()[:2], dtype=float)
                _centroid_xy = np.asarray(self._carry_anchor_xyz(self.sim.data)[:2], dtype=float)
                _away = _centroid_xy - _robot_xy
                _norm = float(np.linalg.norm(_away))
                if _norm > 1e-4:
                    _unit = _away / _norm
                    _shift_xy = _unit * obj_r_pre  # near-edge target offset
                else:
                    _shift_xy = np.zeros(2)
                # Blend 50/50 between the object's current position
                # and the near-edge snapped target. Letting the object
                # stay closer to where it naturally is keeps the
                # close-phase contacts visibly doing the alignment
                # work rather than producing a teleport.
                _raw_offset_xy = np.asarray(obj_xyz_pre[:2], dtype=float) - _centroid_xy
                _blend_xy = 0.5 * _raw_offset_xy + 0.5 * _shift_xy
            except Exception as _e:
                print(f"[Exec] [5.7] near-edge XY shift skipped: {_e}")
                _blend_xy = np.zeros(2)
                _shift_xy = np.zeros(2)
                _raw_offset_xy = np.zeros(2)
            # FAST mode pins to the near-edge offset — pinning to the
            # exact centroid pulls the object too deep into the palm/
            # finger cage.
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
            # The three actions below TELEPORT/PIN the obj before close.
            # Gated by `_smooth_lift_skipped` — in REALISM mode they're
            # all skipped so the obj stays put and the close must make
            # physical contact on its own.
            if not _smooth_lift_skipped:
                # Activate weld in plan_data so OMPL transport treats the
                # carried obj as robot geometry.
                self.arm_bridge.model.eq_obj2id[self.weld_id] = obj_bid
                self.arm_bridge.planning_data.eq_active[self.weld_id] = 1
                # Zero-g on the held obj so the pin closure doesn't fight
                # gravity during the smooth-lift.
                try:
                    self._held_obj_orig_gravcomp = float(
                        self.sim.model.body_gravcomp[obj_bid])
                    self.sim.model.body_gravcomp[obj_bid] = 1.0
                except Exception as e:
                    print(f"[Exec] gravcomp set warning: {e}")
                    self._held_obj_orig_gravcomp = None
                # Soften held-obj contact stiffness so the lift does
                # not impulse-shock nearby fingers. FAST mode skips
                # this so the parallel close stroke sees hard contacts
                # (hard contacts stop fingers AT the object surface;
                # soft contacts let them slide INTO the geometry).
                # The pin keeps the object in place regardless.
                if FAST_PICKUP_MODE and side_grip:
                    print(f"[Exec] [5.7] skipping contact soften "
                          f"(FAST_PICKUP_MODE: hard contacts keep "
                          f"fingers at obj surface during close)")
                else:
                    self._soften_held_obj_contacts(obj_bid)
                # Install animated pin. Normal mode keeps the historical
                # gripper-relative target. FAST mode uses a fixed world
                # target during close: anchoring to the fingertip centroid
                # while fingers close makes the target move sideways, and
                # hard contacts push that motion back into the wrist/arm.
                obj_xyz_snapshot = self.sim.data.xpos[obj_bid].copy()
                if FAST_PICKUP_MODE and side_grip:
                    # Live pinch_midpoint tracking with phased XY→Z
                    # animation. Phase A (0.6 s) slides the object
                    # laterally into the pinch centre; phase B (0.6 s)
                    # drops Z to grip height while XY tracks the live
                    # midpoint. Live tracking keeps the object between
                    # the closing thumb and bc fingers — all three
                    # converge cleanly without the object "passing
                    # over" the thumb.
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
                    # Non-FAST side-grip pin uses the same live
                    # pinch_midpoint anchor and phased XY→Z animation
                    # for visual consistency. It runs in parallel
                    # with the close stroke (no serialized wait).
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
                        # Non-side-grip (top-down/diagonal) keeps
                        # the legacy 3-tip mean — no asymmetry there.
                        self._install_pin(
                            self._pin_obj_to_gripper_animated(
                                obj_xyz_snapshot))
                    self._fast_fixed_close_pin_active = False
                # FAST mode serializes the pin animation and the close
                # stroke. Running them in parallel in FAST (with the
                # gate-bypass active) caused the fingers to chase a
                # moving target and overrun, ending inside the object.
                # Non-FAST keeps them in parallel — the geometry has
                # been validated by the strict gates upstream so the
                # close-stroke contact catches the object cleanly.
                if FAST_PICKUP_MODE and side_grip:
                    import time as _t_pin
                    # Phased pin = XY phase (0.6 s) + Z phase (0.6 s)
                    # + a small settle.
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
                # No pre-close sleep — the animated pin runs in
                # parallel with the close phase, so the object visibly
                # follows the closing fingers rather than appearing as
                # a teleport. The pin tracks `_carry_anchor_xyz` (the
                # fingertip centroid), which moves with the closing
                # fingers. Mark the pre-close lift done so the
                # post-close path skips the duplicate pin install.
                self._pre_close_lift_done = True

            # ── [6] Close gripper + activate weld + record grasp_offset ──
            # Size-aware close: smaller objects close more, larger
            # objects stop earlier so the fingers wrap around the
            # cylinder surface. Contacts are kept at default (hard)
            # stiffness during the close so MuJoCo stops the fingertips
            # at the object surface; softening is applied afterwards
            # so the carry phase does not fight the pin solver.
            obj_radius = self._object_radius(obj_bid)
            size_aware_close_pos = self._finger_close_for_radius(obj_radius)

            # OVERDRIVE: when contact-stop is on, override the size-aware
            # close target with FINGER_CLOSE_MAX (100% intensity). With
            # carry_gap up to 3 cm, the far fingers reach the size-matched
            # target and stop in mid-air before touching the cylinder.
            # Driving to FINGER_CLOSE_MAX forces them to keep curling
            # until contact, then CONTACT_COMPRESS_TICKS lets them
            # compress firmly into the surface before freezing.
            # Overdrive drives the close stroke to FINGER_CLOSE_MAX so
            # fingers reliably reach the object surface; the proximity
            # check in `_set_gripper` freezes each finger as its tip
            # crosses obj_radius, so overdrive does not produce
            # penetration in practice.
            # (an earlier review): in STRICT mode contact-stop
            # is bypassed (see line 4313-4314 `not STRICT_PICKUP_MODE`),
            # so the overdrive's safety assumption ("contact-stop will
            # freeze each finger at obj surface") does not hold. Without
            # that brake the close stroke drives ctrl past the contact
            # qpos before force-stop can react — observed to produce
            # 100+ N steady forces with occasional 10^5 N overshoot
            # spikes. In STRICT we therefore use the SIZE-AWARE close
            # target (0.15 for typical obj radii) instead of the 0.20
            # overdrive — force-stop + tiered relief still freeze
            # fingers, but the ramp ends at a lower ctrl so the
            # baseline PD residual is much smaller.
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
                # stage tracking: close-stroke about to fire.
                self._cycle_stage_close_fired = True

            # Make sure the async open from step [3] has completed
            # before we start the close (they share the finger
            # actuators). The join is cheap insurance.
            if open_thread is not None and open_thread.is_alive():
                open_thread.join(timeout=2.0)
            print(f"[Exec] [6] close gripper  "
                  f"radius={obj_radius:.3f}m  close_ctrl={close_pos:.3f}")
            # Snapshot the arm pose at the moment of close — STRICT
            # slip-retry re-descends to this exact pose if the first
            # lift slips.
            self._strict_grasp_q = list(self._current_arm_q())
            self._set_gripper(close_pos, hold_seconds=0.6)
            if self._cancel:
                self._clear_held_state(); fire(False); return

            # ── [6.3] BC-RESCUE ──
            # When close summary shows 2/3 (thumb + exactly one of bc
            # held, the other missing), try a targeted bump on the
            # missing finger before falling through to the full
            # verify-fail / re-grasp cycle. The bump is on the
            # missing finger's j1 ONLY, so the firm thumb + one bc
            # finger keep their grip undisturbed.
            try:
                if (STRICT_PICKUP_MODE and side_grip and USE_BC_RESCUE
                        and self._last_close_finger_contacts is not None):
                    # _last_close_finger_contacts is indexed [c, b, a].
                    _bc_contacts = self._last_close_finger_contacts
                    _a_held = bool(_bc_contacts[2])
                    _b_held = bool(_bc_contacts[1])
                    _c_held = bool(_bc_contacts[0])
                    # Trigger only on the 2/3 with thumb-firm,
                    # exactly-one-bc pattern (a + b XOR a + c).
                    if _a_held and (_b_held ^ _c_held):
                        missing_name = 'c' if not _c_held else 'b'
                        # j1 indices: c=0, b=3, a=6 in gripper_ids_left
                        _j1_idx = 0 if missing_name == 'c' else 3
                        _bc_gids = self.sim.gripper_ids_left
                        # Pre-bump force reading on the missing finger.
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
                            # ramp the BC-RESCUE bump
                            # over the existing settle window instead
                            # of a single-step ctrl jump. At kp=100 a
                            # 50-mrad step is a 5 Nm torque transient
                            # visible as a finger "twitch". 10 mini-
                            # steps over 300 ms = 0.5 Nm/step max =
                            # visually smooth.
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
                            # Post-bump force reading.
                            _bc_N_post = self._per_finger_normal_forces(obj_bid)
                            _missing_N_post = (_bc_N_post[2]
                                               if missing_name == 'c'
                                               else _bc_N_post[1])
                            if _missing_N_post >= BC_RESCUE_SUCCESS_N:
                                # Mark the missing slot as contacted so the
                                # downstream verify gate sees 3/3.
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

            # Diagnostic: post-close fingertip geometry + contacts.
            # Compare against the post-descent log above to see how
            # much the close moved each finger toward the cylinder
            # and which finger(s) ended up in/out of contact.
            if USE_FINGER_DIAGNOSTIC_LOG:
                self._log_finger_geometry(obj_bid, "post-close")
                self._log_finger_object_contacts(obj_bid, "post-close")

            # ── [6.4] 3-finger contact verification (Path B Step 2) ──
            # Strict: require 3-finger contact for the first
            # MAX_STRICT_3FINGER_ATTEMPTS attempts in this MOVE cycle.
            # If fewer fingers contacted, OPEN gripper, retract arm, and
            # report failure — the existing base-XY retry will adjust
            # alignment and try again. After the strict window expires,
            # accept MIN_CONTACTS_RELAXED (=2) so the demo doesn't loop
            # forever. Below MIN_CONTACTS_RELAXED is always a fail
            # (1-finger held looks like the "fake pickup" the client
            # explicitly called out).
            if USE_3FINGER_VERIFY and self._last_close_finger_contacts is not None:
                # Snapshot contacts BEFORE any further _set_gripper calls.
                # _set_gripper(OPEN) clears _last_close_finger_contacts
                # (the snapshot is only valid for the most recent close),
                # so we cache here to survive the open-before-retract.
                contacts_snapshot = list(self._last_close_finger_contacts)
                self._strict_finger_attempts_used += 1
                n_contacts = sum(bool(x) for x in contacts_snapshot)
                strict_window_open = (self._strict_finger_attempts_used
                                      <= MAX_STRICT_3FINGER_ATTEMPTS)
                # contacts_snapshot order is [c, b, a] (finger_frozen
                # in _set_gripper). thumb = finger_a = index 2;
                # finger_b = index 1; finger_c = index 0.
                thumb_contacted = bool(contacts_snapshot[2])
                b_contacted     = bool(contacts_snapshot[1])
                c_contacted     = bool(contacts_snapshot[0])
                if STRICT_PICKUP_MODE:
                    # Fix 2: SUSTAINED-CONTACT re-check.
                    # `contacts_snapshot` reflects force-stop firing
                    # during the close stroke — but contacts can release
                    # during the 0.6 s close-hold if the grip is
                    # marginal (finger_b on obj_top edge etc). Wait a
                    # brief settling window and re-check live contacts
                    # via mj_contactForce. Only accept a finger as
                    # "contacted" if it's STILL in contact post-hold.
                    # _finger_body_groups order is (c=0, b=1, a=2) per
                    # init. Same as contacts_snapshot indexing.
                    try:
                        # PRE-VERIFY TIGHTEN.
                        # After force-stop freezes individual finger
                        # ctrls, contact normal force can decay during
                        # the close-hold + settling window. Operator
                        # observed runs where snapshot=3/3 but live=1/3
                        # 100ms later — bc fingers separated as their
                        # frozen ctrls didn't maintain squeeze. Bump
                        # j1 ctrls inward by +0.05 rad to re-energize
                        # the squeeze before the live check. Same
                        # technique as 's pre-lift tighten,
                        # but BEFORE verify so verify reflects the
                        # tightened grip, not the marginal post-stop
                        # state.
                        try:
                            gids_v = self.sim.gripper_ids_left
                            # Slot→j1 indices (a, b, c) per the rest of
                            # the file. a=j_idx 6, b=3, c=0.
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
                            # Single-line summary so the log isn't flooded.
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
                        # Recompute n_contacts after sustained check.
                        n_contacts = (int(thumb_contacted)
                                      + int(b_contacted)
                                      + int(c_contacted))
                    except Exception as _e:
                        self._strict_log(
                            "VERIFY",
                            f"sustained-contact check failed: {_e} — "
                            f"falling back to close-time snapshot")
                    # #6 / CONTACT-NORMAL
                    # TRIAD GATE. balance_score = ‖Σ F_i n̂_i‖ /
                    # max(F_i). ≈ 0 = opposing cage (good); ≈ 1 = all
                    # force on one side (obj will squirt out). Two
                    # thresholds: DIAG (warn) and GATE (reject). Gate
                    # is generous (0.75) — only catches obviously
                    # one-sided grasps so we don't waste time on a lift
                    # that physics has already condemned.
                    # TRIAD-GATE catches one-sided grips
                    # that would slip during friction-only lift. In
                    # soft-weld mode (default — `--perfect` OFF), the
                    # weld holds the obj regardless of force balance,
                    # so one-sided grips are not a failure mode. We
                    # still RUN the diag (useful telemetry) but skip
                    # the REJECT branch. `--perfect` keeps the gate
                    # active for friction-only rigor.
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
                    # SUSTAINED-FORCE gate. Active force
                    # sampling over a time window to catch chatter-trough
                    # snapshots where N≈0 despite geometric contact.
                    # Without this, the single-tick verify above can
                    # let lift fire on a zero-force grip → guaranteed
                    # slip. Requires opposing-pinch (a + (b OR c))
                    # forces sustained above floor for EVERY sample in
                    # the window.
                    # Runs INDEPENDENTLY of triad gate — the triad
                    # passthrough at `required=3` lets `N=(0,0,0)` cases
                    # slip through when 3 fingers geometrically touch
                    # the obj. This gate catches that.
                    _sustained_force_failed = False
                    # the sustained-force gate exists to
                    # prevent lift firing on chatter-trough grips that
                    # would slip in PURE-FRICTION mode. In soft-weld
                    # mode (default — `--perfect` OFF), the weld bridges
                    # chatter so a transient 0-force tick during verify
                    # doesn't slip the obj on lift. Skip the gate when
                    # soft-weld is active. Still run TRIAD-GATE above
                    # so we reject genuinely one-sided grips (those the
                    # weld also can't save).
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
                                # Required: thumb (a) above floor AND at
                                # least one of (b, c) above floor.
                                _thumb_ok = _Na >= _N_floor
                                _side_ok = (_Nb >= _N_floor
                                            or _Nc >= _N_floor)
                                if _thumb_ok and _side_ok:
                                    _pass_count += 1
                                time.sleep(_interval)
                            _pass_frac = _pass_count / max(_n_samples, 1)
                            # spike-detect upper bound.
                            # Any per-finger sample exceeding the cap
                            # indicates solver-resonance spike that
                            # would launch obj during lift. Reject.
                            _spike_cap = VERIFY_SUSTAINED_FORCE_SPIKE_CAP_N
                            _max_overall = max(_max_a, _max_b, _max_c)
                            _spike_detected = _max_overall > _spike_cap
                            _all_passed = (
                                _pass_frac >= VERIFY_SUSTAINED_FORCE_MIN_PASS_FRAC
                                and not _spike_detected)
                            # Summarise the window for the log.
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
                                # Hard-fail: zero the geometric contact
                                # flags so the n_contacts vs required
                                # check below actually fails. Without
                                # this, the `required=3 / n_contacts=3`
                                # passthrough would let lift fire anyway
                                # despite our rejection.
                                thumb_contacted = False
                                b_contacted     = False
                                c_contacted     = False
                                n_contacts      = 0
                        except Exception as _e_susf:
                            self._strict_log(
                                "VERIFY",
                                f"sustained-force check raised: {_e_susf} "
                                f"— skipping gate (accept)")
                    # 1+2 gripper geometry: valid friction-pinch =
                    # thumb (finger_a, one side) + at least one of
                    # finger_b/c (opposing side). finger_b+c alone
                    # without thumb is NOT a pinch — both on same
                    # side, no opposing normal force.
                    # also gate on triad balance — if the
                    # force triad is too one-sided, force-fail even
                    # when geometric contact criteria pass.
                    # also gate on sustained force —
                    # opposing-pinch with chatter-trough is invalid.
                    opposing_pinch_ok = (
                        thumb_contacted
                        and (b_contacted or c_contacted)
                        and not _triad_gate_failed
                        and not _sustained_force_failed)
                    if opposing_pinch_ok:
                        # Force the gate to accept regardless of count
                        # (1 thumb + 1 side = 2 fingers, but it's a
                        # valid pinch). Downstream slip detection on
                        # lift confirms whether the grip actually holds.
                        required = n_contacts
                        bar_label = (
                            "strict (mode 1, 3/3 ideal)"
                            if n_contacts == 3
                            else f"strict (mode 1, opposing 2/3: "
                                 f"thumb+"
                                 f"{'b' if b_contacted else 'c'})")
                        # stage tracking: verify accepted
                        # opposing-pinch + triad + sustained-force.
                        self._cycle_stage_verify_passed = True
                    else:
                        required = 3   # force fail
                        bar_label = "strict (mode 1, opposing-pinch REQUIRED)"
                else:
                    required = (MIN_CONTACTS_STRICT if strict_window_open
                                else MIN_CONTACTS_RELAXED)
                    bar_label = "strict" if strict_window_open else "relaxed"
                # FAST mode bypass for the verify gate. When the
                # pre-close gate has already force-passed an
                # asymmetric/marginal pose, the pin holds the object at
                # the carry pose regardless of physical contact count,
                # so we accept the close as successful. Scope matches
                # the pre-close override (side-grip + realism).
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
                    # emit GRASP_DIAG on verify-FAIL too,
                    # so we can compare working vs non-working
                    # (chassis × wrist × geometry) combos. Only fires
                    # when the close actually produced ≥ 1 contact —
                    # the pure pre-close rejects don't reach here.
                    self._emit_grasp_diag(
                        outcome="verify_FAIL",
                        obj_bid=obj_bid,
                        n_contacts=n_contacts,
                        contacts_snapshot=contacts_snapshot,
                        side_grip=side_grip)
                    # Re-open before retract so fingers don't drag the
                    # cylinder. Short hold; we're abandoning this pose.
                    self._set_gripper(GRIPPER_OPEN_POS, hold_seconds=0.3)
                    if self._cancel:
                        self._clear_held_state(); fire(False); return
                    # Set failure info with a distinct reason so the
                    # base-XY retry can log it.
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
                    # STRICT mode: keep arm at GRASP_Q for the local
                    # retry's chassis nudge. The legacy hover-retract
                    # path (column +8cm small lift) breaks the local-
                    # retry-skip-arm assumption — arm rises 10 cm,
                    # then local retry skips descent and the close
                    # stroke fires with palm 6+ cm above obj_top.
                    # an earlier review
                    # attempt 1 had good Z (-3 cm vs obj_top), retry
                    # had +3.5 cm (above obj_top → close in air).
                    # `side_grip_retry=True` skips the column lift,
                    # opens gripper only, holds arm at GRASP_Q.
                    self._retract_after_failure(
                        "[6.4]",
                        side_grip_retry=(STRICT_PICKUP_MODE and side_grip))
                    self._clear_held_state(); fire(False); return
                else:
                    label = "3/3 fingers" if n_contacts == 3 else \
                            f"{n_contacts}/3 fingers (relaxed accept)"
                    print(f"[Exec] [6.4] 3-finger verify OK ({bar_label}): "
                          f"{label}")
                    # / 64: emit BASELINE log line on
                    # every successful verify. Capture full grasp
                    # state so we can study the (chassis × wrist ×
                    # geometry) combos that actually produce working
                    # grips. Grep this line: grep '\[GRASP_DIAG\]'
                    self._emit_grasp_diag(
                        outcome="verify_OK",
                        obj_bid=obj_bid,
                        n_contacts=n_contacts,
                        contacts_snapshot=contacts_snapshot,
                        side_grip=side_grip)

            # If [5.7] pre-close smooth-lift already installed the pin,
            # skip the post-close re-install. The obj is already in
            # the pocket, the fingers have closed around it, and the
            # pin is already tracking centroid + grasp_offset. Just
            # hold the close pose briefly and report success.
            if getattr(self, '_pre_close_lift_done', False):
                self._pre_close_lift_done = False
                self._set_gripper(close_pos, hold_seconds=SMOOTH_ATTACH_SETTLE)
                if getattr(self, '_fast_fixed_close_pin_active', False):
                    # The close stroke used a fixed world pin to avoid
                    # centroid-chasing side forces. Now that the fingers
                    # are stopped, capture the current gripper-relative
                    # offset and switch back to the normal dynamic carry
                    # pin before lift/transport.
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
                # Continue to [6.5] lift.
            else:
                # Legacy path: capture offset post-close and install pin now.
                # NOTE: do NOT call mj_forward(sim.data) from this background
                # thread — the main render thread runs mj_step concurrently
                # on the same MjData, and MuJoCo is not thread-safe for
                # concurrent ops on one MjData (SIGSEGV). sim.data.xpos is
                # already up-to-date from the main thread's last mj_step.
                grip_xyz = self._carry_anchor_xyz(self.sim.data).copy()
                obj_xyz  = self.sim.data.xpos[obj_bid].copy()
                raw_offset = obj_xyz - grip_xyz
                raw_dist = float(np.linalg.norm(raw_offset))
                raw_xy_dist = float(np.linalg.norm(raw_offset[:2]))
                half_h = self._object_half_height(obj_bid)
                target_z = -(half_h - 0.025)
                raw_z = float(raw_offset[2])
                # In realism mode the post-close pin captures the obj
                # wherever the close-stroke contact left it (no XY snap,
                # no Z lift). Demo mode snaps XY to zero and lifts Z
                # to the target so the obj appears cleanly centred in
                # the gripper during transport.
                if REALISM_MODE_NO_SMOOTH_LIFT and side_grip:
                    z_lift = 0.0
                    # Keep raw_offset.xy and raw_offset.z as captured.
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
                    # Physics-driven grip: NO weld, NO gravcomp, NO pin.
                    # The obj is held only by finger friction. The lift
                    # phase below monitors slip and triggers a retry with
                    # a stronger grip target if the obj drops away.
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
                    # Side-grip transport pin uses the pinch_midpoint
                    # anchor (visual centre between thumb and bc),
                    # matching the pre-close smooth-lift pin.
                    self._install_pin(
                        self._pin_obj_to_gripper_animated(
                            obj_xyz_now,
                            anchor_pinch_midpoint=bool(side_grip)))
                    self._set_gripper(close_pos, hold_seconds=SMOOTH_ATTACH_SETTLE)
                    print(f"[Exec] [verify-grasp] OK: object pinned at offset "
                          f"{self._grasp_offset_xyz.round(3)} from gripper, "
                          f"smooth-attach over {SMOOTH_ATTACH_SECS:.2f}s")

            # post-verify soft-weld activation (STRICT
            # mode, default). By this point all pure-physics gates have
            # passed:
            # close stroke fired and produced real contacts (LATE-87)
            # triad-balance check passed (balance < 0.75)
            # sustained-force check passed (≥60 % of 10 samples above
            # 30 N floor, no spike > 50 kN)
            # i.e. the fingers physically grip obj with realistic kN forces.
            # Activating a compliant weld here bridges MuJoCo's
            # contact-solver chatter during transport so obj follows the
            # gripper reliably. This is NOT FAST mode — FAST mode
            # bypasses the verify gates; this only activates after they
            # all pass.
            # The `--perfect` flag (STRICT_PERFECT_FRICTION_ONLY=True)
            # skips this block — pure-friction lift, no weld assist.
            # The `_cycle_stage_verify_passed` flag is set ONLY when
            # opposing_pinch_ok = True, i.e. when ALL of:
            # thumb_contacted AND (b_contacted OR c_contacted)
            # triad-balance < threshold (one-sided rejected)
            # sustained-force window passed (≥60% samples, no spike)
            # Using this flag makes the weld activation strictly verify-
            # gated. The relaxed-fallback path where required=3 and
            # n_contacts=3 but opposing_pinch_ok=False does NOT trigger
            # the weld — it falls to the existing pure-friction lift.
            if (STRICT_PICKUP_MODE
                    and not STRICT_PERFECT_FRICTION_ONLY
                    and side_grip
                    and not FAST_PICKUP_MODE
                    and self._cycle_stage_verify_passed):
                try:
                    # Two-part activation:
                    # 1. eq_active flag in planning_data — tells OMPL
                    # transport-phase planner that obj is robot-rigid.
                    # 2. Pin callback — the actual physical constraint
                    # that makes obj follow gripper centroid each
                    # tick. Without this, the eq_active flag alone
                    # does NOT constrain obj in sim physics (the
                    # XML weld constraint is solver-gated by
                    # eq_active in sim.data, but writing sim.data
                    # eq_active mid-sim triggers nefc mis-allocation
                    # see _clear_held_state docstring).
                    self.arm_bridge.model.eq_obj2id[self.weld_id] = obj_bid
                    self.arm_bridge.planning_data.eq_active[self.weld_id] = 1
                    # soften obj contacts in STRICT soft-
                    # weld too (FAST-mode branch already did this).
                    # Without softening, when pin pulls obj toward
                    # palm/wrist, the rigid cylinder-vs-palm contact
                    # produces an impulse that BOUNCES obj away (client
                    # reported "object touches the center of wrist or
                    # palm and bounces away"). Softening obj geoms
                    # (solref → 0.05, solimp → low impedance) lets the
                    # pin keep obj in contact with palm without bounce.
                    self._soften_held_obj_contacts(obj_bid)
                    # Install pin so obj physically follows gripper.
                    obj_xyz_now = self.sim.data.xpos[obj_bid].copy()
                    # pin obj at its ACTUAL current
                    # offset from pinch_midpoint (instead of canonical).
                    # Removes visible "obj dragged sideways" snap.
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

            # ── [6.5] LIFT — raise arm to carry pose for safe transport ──
            # Without this the gripper would stay at pick height during
            # base nav and drag the held object across the floor.
            print("[Exec] [6.5] lift to carry pose")
            if STRICT_PICKUP_MODE:
                # Friction-only lift with slip monitoring and up to
                # STRICT_RETRY_MAX grip-bump retries. No weld, no pin
                # the obj is held only by finger friction.
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
            # Reset warm-start cache on successful pick — next pickup
            # (likely a different obj_idx) should start with a cold seed
            # rather than reusing this obj's q (which is anchored to
            # this chassis pose and orientation).
            self._last_valid_pre_grasp_q = None
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
            # surface. Release/retract still completed cleanly, but
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
            # reset). Snapshot dofadr beforehand so we can zero qvel
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
