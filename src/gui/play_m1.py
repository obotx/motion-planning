"""
play_m1.py — Mobile Manipulator Pick & Place GUI (MuJoCo / GLFW / ImGui).

Provides OMPL RRT* base navigation and direct-joint grasping with
proximity-attach for the full pick-and-place cycle.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Tee stdout / stderr to RUN_LOG.txt at the project root. Truncated at
# startup, so the file always reflects only the latest run. Console
# output is unaffected.
_project_root = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
_run_log_path = os.path.join(_project_root, "RUN_LOG.txt")

class _Tee:
    """Write-through tee — duplicates output to multiple streams.

    Flushes only on line boundaries (or explicit `flush()`) so write-
    heavy logging doesn't bottleneck on per-write fsync.  Python's
    atexit cleanup guarantees the final flush on shutdown.
    """
    def __init__(self, *streams):
        self._streams = streams
    def write(self, data):
        ends_with_newline = bool(data) and data[-1] == '\n'
        for s in self._streams:
            try:
                s.write(data)
                if ends_with_newline:
                    s.flush()
            except Exception:
                pass
    def flush(self):
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass
    def isatty(self):
        # Some libs check isatty(); return the underlying console's
        # value so colors etc. work normally.
        try:
            return self._streams[0].isatty()
        except Exception:
            return False

try:
    _run_log_fh = open(_run_log_path, "w", buffering=1, encoding="utf-8")
    _orig_stdout = sys.stdout
    _orig_stderr = sys.stderr
    sys.stdout = _Tee(_orig_stdout, _run_log_fh)
    sys.stderr = _Tee(_orig_stderr, _run_log_fh)
    print(f"[RUN_LOG] tee-ing stdout/stderr to {_run_log_path}")
except Exception as _e:
    print(f"[RUN_LOG] failed to open log file ({_e}); console output only")

import argparse
import threading
import time
import math

import mujoco
import glfw
import numpy as np
import imgui
from imgui.integrations.glfw import GlfwRenderer

from simulations.morph_i_free_move import ParallelRobot
from modules.pubsub import IPCPubSub
from navigation.ompl_windows_bridge import InProcessNavigator
from navigation.grasp_executor import (
    GraspExecutor, compute_grasp_targets, reset_plan_data_for_ik,
    compute_wrist_goal_for_obj,
    MIN_PICK_WRIST_Z, GRIPPER_STANDOFF_XY,
    CARRY_H1, CARRY_H2, CARRY_A1,
    FAST_PICKUP_MODE, STRICT_PICKUP_MODE)
from navigation.arm_planner import MORPHBridge, HOME_Q
from navigation.plan import OBSTACLE_RECTS, ROBOT_RADIUS as NAV_ROBOT_RADIUS

# Constants

NUM_OBJECTS     = 10
NUM_SHELF_SLOTS = 10

OBJECT_LABELS = [
    "Obj-0  [Brisk Can]",
    "Obj-1  [Pepsi Max]",
    "Obj-2  [Nescafe Dolce]",
    "Obj-3  [Nescafe Taster]",
    "Obj-4  [Nestle Candy]",
    "Obj-5  [Pepsi Cherry]",
    "Obj-6  [Crunch Bar]",
    "Obj-7  [Skinny Cow]",
    "Obj-8  [Vans Cereal]",
    "Obj-9  [Pepsi Diet]",
]

OBJECT_COLORS_RGBA = [
    (0.9, 0.1, 0.1, 1.0),
    (0.9, 0.5, 0.0, 1.0),
    (0.9, 0.9, 0.0, 1.0),
    (0.1, 0.8, 0.1, 1.0),
    (0.0, 0.8, 0.9, 1.0),
    (0.1, 0.2, 0.9, 1.0),
    (0.6, 0.0, 0.8, 1.0),
    (0.9, 0.3, 0.6, 1.0),
    (0.95, 0.95, 0.95, 1.0),
    (0.5, 0.25, 0.0, 1.0),
]

SHELF_SLOT_POSITIONS = np.array([
    [2.5, -3.5, 0.50],
    [3.0, -3.5, 0.50],
    [3.5, -3.5, 0.50],
    [4.0, -3.5, 0.50],
    [4.5, -3.5, 0.50],
    [2.5, -3.5, 1.00],
    [3.0, -3.5, 1.00],
    [3.5, -3.5, 1.00],
    [4.0, -3.5, 1.50],
    [4.5, -3.5, 1.50],
], dtype=float)

SHELF_SLOT_LABELS = [
    f"Slot-{i}  [row={'low' if SHELF_SLOT_POSITIONS[i,2]<0.75 else ('mid' if SHELF_SLOT_POSITIONS[i,2]<1.25 else 'high')}"
    f"  z={SHELF_SLOT_POSITIONS[i,2]:.2f}m]"
    for i in range(NUM_SHELF_SLOTS)
]

FLOOR_X_RANGE      = (0.5, 7.5)
FLOOR_Y_RANGE      = (-7.8, -6.2)
FLOOR_X_RANGE_2    = (0.5, 7.5)
FLOOR_Y_RANGE_2    = (-3.2, -2.2)
OBJECT_Z           = 0.075
MIN_OBJ_SEPARATION = 0.45  #  tracks the OBJ_RADIUS_RANGE upper bound
                            # (2×max_r = 0.17).  25 cm surface-to-
                            # surface gap is adequate to prevent
                            # spawn-settle bumping.

OBJ_RADIUS_RANGE = (0.068, 0.075)  # cylinder radius range.
# Half-height 0.14-0.16 puts obj middle at z=0.14-0.16. Palm
# settles at z≈0.22 in side-grip, so gap is 6-8 cm (down from
# 10 cm with old 0.11-0.14 range). Aspect ratio 1.87:1 to 2.46:1
# within the spawn-stable band per -9112.
OBJ_HEIGHT_RANGE = (0.14, 0.16)    # cylinder half-height range
SPAWN_EDGE_MARGIN = OBJ_RADIUS_RANGE[1] + 0.05
SPAWN_FLOOR_CLEARANCE = 0.001
SPAWN_SETTLE_STEPS = 400
SPAWN_ROBOT_KEEP_CENTER = np.array([3.70, -6.00])
SPAWN_ROBOT_KEEP_RADIUS = 1.15
# Open-floor zones used for object spawning. Distributed across the scene
# instead of a single south-aisle band so objects appear in multiple regions
# of the market. Margins are conservative around the shelf rack
# (x=2.5..8.05, y=-5.2..-3.1) and the world walls.
SPAWN_ZONES = (
    ((0.75, 7.25), (-7.50, -5.50)),  # south aisle (below the rack)
    ((0.75, 7.25), (-3.00, -0.40)),  # north aisle (above the rack)
    ((0.40, 2.20), (-5.10, -3.40)),  # west strip (between west wall and rack)
)
SPAWN_EXTRA_KEEP_RECTS = (
    (0.65, 8.00, -8.00, -6.85),      # far-south rack/footer band
    (6.60, 8.00, -8.00, -5.20),      # right-side lower rack/overhang strip
)

WAYPOINT_REACH_DIST = 0.50
GOAL_REACH_DIST     = 0.55
WAYPOINT_TIMEOUT    = 180.0

PICK_BASE_STANDOFF       = 0.78      # base standoff for DIAGONAL/TOP-DOWN
                                     # picks — arm reaches mostly vertical.
PICK_BASE_STANDOFF_SIDE  = 0.55      # base standoff for SIDE-GRIP picks —
                                     # closer so the horizontal IK family
                                     # satisfying the wrist orientation goal
                                     # can also reach the object.
MIN_PICK_BASE_OBJ_DIST = NAV_ROBOT_RADIUS + OBJ_RADIUS_RANGE[1]  # chassis-safety floor
PICK_RETRY_OBJ_PATH_CLEARANCE = 0.40  # min path clearance from selected object on retry
PICK_NAV_OBJECT_CLEARANCE = 0.40      # selected-object corridor guard
PICK_MAX_H_DIFF          = 0.08      # max h2-h1 tilt for DIAGONAL/TOP-DOWN
                                     # picks (high tilt + top-down wrist
                                     # causes the thumb to sweep the obj).
PICK_MAX_H_DIFF_SIDE     = 0.35      # max h2-h1 tilt for SIDE-GRIP picks
                                     # (wider window — wrist_X compensates
                                     # for boom angle, so larger tilt is
                                     # geometrically usable).
PICK_MIN_A1         = 0.16           # visual guard: avoid gripper folding into body
PICK_A1_FIXED_REACH_OFFSET = 0.32    # base-to-wrist reach present even at a1≈0
PICK_GOAL_X_RANGE   = (0.40, 7.25)   # pick nav goal x bounds (full floor minus wall margin)
PICK_GOAL_Y_RANGE   = (-7.50, -0.40) # pick nav goal y bounds (full floor minus wall margin)
PICK_NAV_GOAL_TOL   = 0.06           # base nav goal tolerance for pick
PICK_CANDIDATE_STANDOFFS = (0.75, 0.78, 0.82)  # candidate standoff distances.
                                                # Shared across pick modes;
                                                # side-grip adds a post-nav
                                                # forward push (see
                                                # SIDE_FORWARD_PUSH_TARGET).
SIDE_FORWARD_PUSH_TARGET = 0.64  # final chassis-to-obj distance for
                                 # SIDE-GRIP mode after nav + fine-align.
                                 # 0.64 m gives IK enough room to converge
                                 # with the side-grip wrist orientation
                                 # locked (.
                                 # The ~14-23 cm IK-vs-runtime palm gap
                                 # (Rotation joint passive deflection +
                                 # wrist actuator compliance) is now
                                 # compensated via the calibration LUT
                                 # (enabled by default — see --use-calib).
PICK_CANDIDATE_DUP_TOL = 0.02        # dedup tolerance for candidate XY
PICK_FINE_ALIGN_VISUAL_DIST = 0.75   # max distance for inward fine-align trim
PICK_FINE_ALIGN_PRESERVE_H_DIFF = 0.08  # preserve screened radius if hΔ below this
PICK_FINE_ALIGN_MIN_SAFE_DIST = 0.70  # min base-object distance during fine-align
PICK_FINE_ALIGN_DIST_TOL = 0.025     # fine-align distance tolerance
PICK_FINE_ALIGN_MAX_STEP = 0.08      # max single-step nudge during fine-align
PICK_FINE_ALIGN_TIMEOUT = 1.5        # fine-align convergence timeout (s)
PICK_LOCAL_RETRY_MIN_DIST = PICK_FINE_ALIGN_MIN_SAFE_DIST
# During a pick attempt the chassis is intentionally close to the target
# (post-push at SIDE_FORWARD_PUSH_TARGET ≈ 0.64 m), so the 0.70 m nav min
# would reject every local-retry candidate. Use a smaller pick-specific
# floor that still leaves a reach buffer.
PICK_LOCAL_RETRY_MIN_DIST_PICK = 0.40
PICK_LOCAL_RETRY_TIMEOUT = 1.5       # local retry convergence timeout (s)
# Closed-loop base-correction retry: translate the base by the residual
# gripper-vs-IK-target error. Scales fall back to smaller corrections
# if the validator rejects the full move (e.g. would clip a wall).
PICK_LOCAL_RETRY_SCALES   = (1.0, 0.7, 0.5, 0.3)
PICK_LOCAL_RETRY_MIN_ERR  = 0.015    # skip retry if residual < 1.5cm
PICK_LOCAL_RETRY_MAX_ERR  = 0.25     # cap retry magnitude to 25cm
PICK_LOCAL_RETRY_MAX_ATTEMPTS = 5    # closed-loop retries per cycle
PICK_LOCAL_RETRY_MAX_ATTEMPTS_STRICT = 3
PICK_CANDIDATE_ANGLE_OFFSETS = (
    0.0,
    math.radians(35.0), math.radians(-35.0),
    math.radians(70.0), math.radians(-70.0),
    math.radians(110.0), math.radians(-110.0),
    math.pi,
)
PICK_VIRTUAL_PLAN_TIMEOUT = 0.3  #  short timeout so OMPL bails fast
                                   # on infeasible candidates during
                                   # screening.  Trade-off: a few
                                   # marginal candidates that need
                                   # >0.3 s to find a path get
                                   # rejected.  The runtime arm planner
                                   # re-tries with the full timeout
                                   # anyway.
# Cap on virtual screening successes before stopping the candidate scan.
# Three feasible candidates is plenty for the retry loop; the failure
# rate of the top-ranked candidate is well under 30%.
MAX_VIRTUAL_SCREEN_SUCCESSES = 3
FAST_PICK_MAX_VIRTUAL_SCREEN_SUCCESSES = 1

# Place-phase configuration: after grasp + lift, the robot navigates to a
# safe aisle point near the assigned shelf slot and parks the object on the
# floor (approximate drop) or runs the full place pipeline.
ENABLE_PLACE_PHASE      = True
USE_APPROXIMATE_DROP    = True
APPROX_DROP_AISLE_Y     = -6.30  # nominal aisle y-coordinate for drop standoff
APPROX_DROP_AISLE_Y_OPTIONS = (-6.45, -6.60, -6.30, -6.75)
APPROX_DROP_X_OFFSETS   = (0.0, 0.25, -0.25)
APPROX_DROP_OBJECT_Y    = -5.82  # floor drop line near the slot column
APPROX_DROP_GOAL_TOL    = 0.15   # base nav goal tolerance for approximate drop


# Object helpers

def _is_inside_rack(x, y, margin=0.10):
    """True if (x, y) falls inside any OBSTACLE_RECTS entry plus margin."""
    for (x0, x1, y0, y1) in OBSTACLE_RECTS:
        if (x0 - margin) <= x <= (x1 + margin) and \
           (y0 - margin) <= y <= (y1 + margin):
            return True
    for (x0, x1, y0, y1) in SPAWN_EXTRA_KEEP_RECTS:
        if (x0 - margin) <= x <= (x1 + margin) and \
           (y0 - margin) <= y <= (y1 + margin):
            return True
    return False


def _is_inside_spawn_keepout(x, y):
    """True for rack/floor/robot-start zones where objects should not spawn."""
    if _is_inside_rack(x, y, margin=SPAWN_EDGE_MARGIN):
        return True
    if np.linalg.norm(np.array([x, y]) - SPAWN_ROBOT_KEEP_CENTER) < SPAWN_ROBOT_KEEP_RADIUS:
        return True
    return False


def _sample_spawn_zone(zone, rng):
    xr, yr = zone
    return (
        rng.uniform(xr[0] + SPAWN_EDGE_MARGIN, xr[1] - SPAWN_EDGE_MARGIN),
        rng.uniform(yr[0] + SPAWN_EDGE_MARGIN, yr[1] - SPAWN_EDGE_MARGIN),
    )


def _is_in_spawn_zone(x, y, margin=0.0):
    for xr, yr in SPAWN_ZONES:
        if ((xr[0] - margin) <= x <= (xr[1] + margin)
                and (yr[0] - margin) <= y <= (yr[1] + margin)):
            return True
    return False


def random_floor_positions(n, rng):
    # Floor for the relaxed-separation fallback: two cylinders never touch
    # even at worst-case radii. Prevents two objects from spawning on top
    # of each other (which produces strong contact ejection and the
    # "objects shaking / half in the ground" effect on the first frame).
    min_safe_sep = 2.0 * OBJ_RADIUS_RANGE[1] + 0.03
    positions = []
    for idx in range(n):
        # Cycle through known-open floor zones so objects are distributed
        # across the scene, then rejection-sample inside that zone for rack
        # keepout, robot-start keepout, and inter-object spacing.
        placed = False
        zone_order = [(idx + k) % len(SPAWN_ZONES) for k in range(len(SPAWN_ZONES))]
        for zone_idx in zone_order:
            for _ in range(700):
                x, y = _sample_spawn_zone(SPAWN_ZONES[zone_idx], rng)
                if _is_inside_spawn_keepout(x, y):
                    continue
                candidate = np.array([x, y])
                if all(np.linalg.norm(candidate - p) >= MIN_OBJ_SEPARATION for p in positions):
                    positions.append(candidate)
                    placed = True
                    break
            if placed:
                break
        if placed:
            continue
        # Fallback: relax separation toward `min_safe_sep`, never below it,
        # so two objects can never start the simulation overlapping.
        candidate = None
        for _ in range(2000):
            zone = SPAWN_ZONES[int(rng.integers(0, len(SPAWN_ZONES)))]
            x, y = _sample_spawn_zone(zone, rng)
            if _is_inside_spawn_keepout(x, y):
                continue
            cand = np.array([x, y])
            if all(np.linalg.norm(cand - p) >= min_safe_sep for p in positions):
                positions.append(cand)
                placed = True
                break
            candidate = cand
        if not placed:
            print(f"[INIT] WARN: object {idx} could not satisfy spacing; "
                  f"using best safe-zone candidate (>=2 obj radii apart)")
            positions.append(candidate if candidate is not None
                              else np.array(_sample_spawn_zone(
                                  SPAWN_ZONES[idx % len(SPAWN_ZONES)], rng)))
    return np.array(positions)


def get_object_qpos_slice(model, body_name):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    jnt_id  = model.body_jntadr[body_id]
    if jnt_id < 0:
        return None
    return model.jnt_qposadr[jnt_id]


def snapshot_object_positions(model, data):
    """capture current obj positions + sizes so respawn
    can restore the EXACT same scene without re-randomising.  Used by
    auto-MOVE so each cycle starts from the same baseline (rather than
    a different lucky/unlucky obj layout per cycle)."""
    snap = []
    for i in range(NUM_OBJECTS):
        qs = get_object_qpos_slice(model, f"pickup_obj_{i}")
        if qs is None:
            snap.append(None)
            continue
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                     f"pickup_obj_{i}")
        radius = float('nan')
        half_height = float('nan')
        for g in range(model.ngeom):
            if int(model.geom_bodyid[g]) != int(body_id):
                continue
            radius = float(model.geom_size[g, 0])
            half_height = float(model.geom_size[g, 1])
            break
        snap.append({
            'qs': qs,
            'body_id': int(body_id),
            'xy': (float(data.qpos[qs]), float(data.qpos[qs+1])),
            'z':  float(data.qpos[qs+2]),
            'quat': (float(data.qpos[qs+3]), float(data.qpos[qs+4]),
                     float(data.qpos[qs+5]), float(data.qpos[qs+6])),
            'radius': radius,
            'half_height': half_height,
        })
    return snap


def restore_object_positions(model, data, snap):
    """restore objects to the positions / sizes captured
    by `snapshot_object_positions`.  Same settle behaviour as
    `randomize_object_positions` so post-restore physics matches the
    initial spawn state."""
    body_ids = []
    for entry in snap:
        if entry is None:
            continue
        qs = entry['qs']
        body_id = entry['body_id']
        body_ids.append(body_id)
        # Restore geom sizes (don't trust prior frame's runtime values
        # since other code may resize on the fly).
        if not (entry['radius'] != entry['radius']):  # not NaN
            for g in range(model.ngeom):
                if int(model.geom_bodyid[g]) != int(body_id):
                    continue
                model.geom_size[g, 0] = entry['radius']
                model.geom_size[g, 1] = entry['half_height']
        # Clamp z to half_height so the restored obj is never placed below
        # the floor surface (snapshot z can sit inside the soft-contact band).
        z_safe = max(float(entry['z']), float(entry['half_height']))
        data.qpos[qs:qs+3]   = [entry['xy'][0], entry['xy'][1], z_safe]
        data.qpos[qs+3:qs+7] = list(entry['quat'])
        jntadr = model.body_jntadr[body_id]
        if jntadr >= 0:
            dofadr = int(model.jnt_dofadr[jntadr])
            data.qvel[dofadr:dofadr + 6] = 0.0
    mujoco.mj_forward(model, data)
    for _ in range(SPAWN_SETTLE_STEPS):
        mujoco.mj_step(model, data)
    for body_id in body_ids:
        jntadr = model.body_jntadr[body_id]
        if jntadr >= 0:
            dofadr = int(model.jnt_dofadr[jntadr])
            data.qvel[dofadr:dofadr + 6] = 0.0
    mujoco.mj_forward(model, data)


def randomize_object_positions(model, data, rng):
    xy = random_floor_positions(NUM_OBJECTS, rng)
    body_ids = []
    for i in range(NUM_OBJECTS):
        qs = get_object_qpos_slice(model, f"pickup_obj_{i}")
        if qs is None:
            continue
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"pickup_obj_{i}")
        body_ids.append(body_id)
        # Sample size in the achievable range (see OBJ_*_RANGE). The
        # body_mass / body_inertia values are computed at XML load and
        # are not updated on resize; this is acceptable because the pin
        # closure owns held-object physics during transport.
        # obj radius range gated to mode.
        # Soft-weld (--perfect OFF) keeps the proven (0.068, 0.075) m
        # range — fingers wrap obj cleanly, pin holds during transport.
        # perfect ON uses (0.050, 0.060) m so the smaller obj lets
        # the forward-offset pin location coincide with the DELTO M3
        # finger-curl convergence point (which is at palm center, a
        # fixed mechanical constraint). With smaller obj, fingers
        # CAN maintain contact at the forward-shifted pin location
        # (resolving the historic LATE-110AL/AM dead-end where the
        # forward-shifted pin looked perfect but verify read 0 N).
        # _perfect_flag is a local variable in main(); inside
        # randomize_object_positions (module-scope helper) we must
        # read the mode flag from the grasp_executor module global
        # that main() sets at startup.
        try:
            import navigation.grasp_executor as _gm
            _perfect_now = bool(getattr(
                _gm, "STRICT_PERFECT_FRICTION_ONLY", False))
        except Exception:
            _perfect_now = False
        if _perfect_now:
            # 0.050-0.060 was too small (finger_c missed obj in 2/2 tries).
            # 0.060-0.065 keeps obj small enough for fingers-not-in-obj
            # but big enough for finger_c to reach reliably.
            radius = rng.uniform(0.060, 0.065)
        else:
            radius = rng.uniform(*OBJ_RADIUS_RANGE)
        half_height = rng.uniform(*OBJ_HEIGHT_RANGE)
        for g in range(model.ngeom):
            if int(model.geom_bodyid[g]) != int(body_id):
                continue
            # MuJoCo cylinders use [radius, half-height] for geom_size.
            model.geom_size[g, 0] = radius
            model.geom_size[g, 1] = half_height
        data.qpos[qs:qs+3]   = [xy[i, 0], xy[i, 1],
                                half_height + SPAWN_FLOOR_CLEARANCE]
        data.qpos[qs+3]      = 1.0
        data.qpos[qs+4:qs+7] = 0.0
        # Clear any residual velocity from previous holds/perturbations.
        jntadr = model.body_jntadr[body_id]
        if jntadr >= 0:
            dofadr = int(model.jnt_dofadr[jntadr])
            data.qvel[dofadr:dofadr + 6] = 0.0
    mujoco.mj_forward(model, data)

    # Settle the objects on the floor before the GUI starts rendering.
    # Without this, the user sees the SPAWN_FLOOR_CLEARANCE drop play out
    # in the first frames of any recording (visible as objects "shaking"
    # at t=0). Stepping with the existing actuator state (PD targets at
    # the home pose set by sim.reset("home")) keeps the robot still while
    # the cylinders drop the few mm and come to rest.
    for _ in range(SPAWN_SETTLE_STEPS):
        mujoco.mj_step(model, data)
    # Zero qvel on every pickup object after settling so any residual
    # micro-velocity from the contact impulse is gone.
    for body_id in body_ids:
        jntadr = model.body_jntadr[body_id]
        if jntadr >= 0:
            dofadr = int(model.jnt_dofadr[jntadr])
            data.qvel[dofadr:dofadr + 6] = 0.0
    mujoco.mj_forward(model, data)
    # Post-settle floor-clearance correction. MuJoCo's contact solver
    # can leave 1-5 mm of residual penetration after spawn settle which
    # reads as visible clipping at startup. For each pickup obj, snap
    # the body z back up to (half-height + floor) if it dipped below.
    fixed_count = 0
    for body_id in body_ids:
        jntadr = model.body_jntadr[body_id]
        if jntadr < 0:
            continue
        qs = int(model.jnt_qposadr[jntadr])
        # Find max geom half-height for this body.
        max_half_h = 0.0
        for g in range(model.ngeom):
            if int(model.geom_bodyid[g]) != int(body_id):
                continue
            # Cylinder size = (radius, half_height); also handles
            # the inner-tint cosmetic geom.
            _gh = float(model.geom_size[g, 1])
            if _gh > max_half_h:
                max_half_h = _gh
        if max_half_h <= 0.0:
            continue
        expected_z = max_half_h
        cur_z = float(data.qpos[qs + 2])
        if cur_z < expected_z - 0.001:
            data.qpos[qs + 2] = expected_z
            dofadr = int(model.jnt_dofadr[jntadr])
            data.qvel[dofadr:dofadr + 6] = 0.0
            fixed_count += 1
            print(f"[INIT] floor-clearance fix: obj at body {body_id} "
                  f"clipped (z={cur_z:.4f} < expected={expected_z:.4f}); "
                  f"snapped up by {(expected_z - cur_z)*1000:.1f} mm")
    if fixed_count > 0:
        mujoco.mj_forward(model, data)
    print(f"[INIT] Randomized {NUM_OBJECTS} objects (settled on floor)."
          + (f"  [{fixed_count} floor-clip corrections]"
             if fixed_count else ""))


def get_object_world_pos(model, data, obj_idx):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"pickup_obj_{obj_idx}")
    return data.xpos[body_id].copy()


def get_object_radius(model, obj_idx, default=0.05):
    """Return the cylinder radius for a pickup object (max across its geoms)."""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                 f"pickup_obj_{obj_idx}")
    if body_id < 0:
        return default
    radii = [
        float(model.geom_size[g, 0])
        for g in range(model.ngeom)
        if int(model.geom_bodyid[g]) == body_id
    ]
    return max(radii) if radii else default


# Collision overlay setup

# Body name prefixes that belong to the parallel arm chain (both ARM1 and ARM2).
_ARM_BODY_PREFIXES = (
    "Column_Left", "Column_Right",
    "Bearing_Column",
    "Arm_Left", "Hand_Bearing",
    "Gripper_Link", "Rotation",
)

# Bodies that hold the gripper-fingertip collision pads. Group 3 visualisation
# focuses on these (the actual contact surfaces between gripper and obj) — the
# big arm-chain proxies (long invisible slabs on Arm_Left for fast OMPL
# collision checks) get pushed to group 4 so they don't drown out the finger
# geometry when the user presses '3'.
_GRIPPER_BODY_PREFIXES = (
    "finger_a_link", "finger_b_link", "finger_c_link",
    "Gripper_Link3",   # palm
)

# MuJoCo geom-group convention used here:
# group 3 — collision overlay (toggled by pressing '3')
# group 4 — hidden non-arm collision proxies (toggled by pressing '4')
_OVERLAY_GROUP = 3
_HIDDEN_GROUP  = 4


def _is_arm_body(body_name):
    if not body_name:
        return False
    return any(body_name.startswith(p) for p in _ARM_BODY_PREFIXES)


def _is_gripper_body(body_name):
    """Bodies that host the fingertip pad collision geoms + the palm."""
    if not body_name:
        return False
    return any(body_name.startswith(p) for p in _GRIPPER_BODY_PREFIXES)


def _setup_pick_collision_overlay(sim_model):
    """Reorganize geom groups so 'press 3' toggles ONLY the small
    fingertip pad-box collision shapes (the high-friction grip pads).
    The visual gripper mesh stays in its default group so it's always
    visible — pressing 3 only adds/removes the pad-box overlay on top.

    Strategy:
      group 3 (toggle '3') = small pad-box collision shapes on the
                             finger links — these are the actual
                             grip surfaces; useful for verifying
                             where contact happens.
      group 4 (toggle '4') = big invisible arm-chain proxies (the
                             long slabs on Arm_Left used by OMPL for
                             fast collision checks).  Hidden by
                             default; press '4' to inspect.

    Detection of pad boxes: geom_type == mjGEOM_BOX (6) on a finger
    body, with mass=0 (the `pad_box1`/`pad_box2` default class sets
    mass=0).  Visual finger meshes (type=mesh) are LEFT in their
    original group so the gripper body stays visible.
    """
    MJGEOM_BOX = 6
    pad_overlay      = 0
    arm_proxy_hidden = 0
    moved_out        = 0
    for i in range(sim_model.ngeom):
        bid = int(sim_model.geom_bodyid[i])
        body_name = mujoco.mj_id2name(sim_model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        is_collision = int(sim_model.geom_contype[i]) != 0
        is_invisible = float(sim_model.geom_rgba[i, 3]) < 0.01
        in_overlay   = int(sim_model.geom_group[i]) == _OVERLAY_GROUP
        is_box       = int(sim_model.geom_type[i]) == MJGEOM_BOX
        # Pad boxes are small (max half-extent ~2.5 cm); the only other
        # boxes on the gripper are the arm-chain proxy slabs (half-extents
        # 0.41 × 0.02 × 0.04) which are large. Detect pad boxes by max
        # half-extent < 0.03 m.
        try:
            sz = sim_model.geom_size[i]
            max_half = float(max(abs(sz[0]), abs(sz[1]), abs(sz[2])))
        except Exception:
            max_half = 1.0
        is_pad_size = max_half < 0.03

        # 1) Pad-box collision shapes on finger bodies → group 3.
        if _is_gripper_body(body_name) and is_collision and is_box \
                and is_pad_size:
            sim_model.geom_group[i] = _OVERLAY_GROUP
            pad_overlay += 1
            continue

        # 2) Big arm-chain invisible proxies → group 4 (hidden by
        # default, user can press '4' to inspect).
        if _is_arm_body(body_name) and is_collision and is_invisible:
            sim_model.geom_group[i] = _HIDDEN_GROUP
            sim_model.geom_rgba[i] = [0.0, 1.0, 0.0, 0.35]
            arm_proxy_hidden += 1
            continue

        # 3) Non-arm, non-gripper geoms left over in group 3 → group 4.
        if in_overlay and not _is_arm_body(body_name) \
                and not _is_gripper_body(body_name):
            sim_model.geom_group[i] = _HIDDEN_GROUP
            moved_out += 1
            continue

    print(f"[overlay] Pick collision overlay ready: "
          f"{pad_overlay} fingertip pad-box collision shapes in group 3 "
          f"(press '3' to toggle); "
          f"{arm_proxy_hidden} arm-chain proxies moved to group 4 "
          f"(press '4' to inspect); "
          f"{moved_out} other non-arm geoms moved to group 4.")


def compute_pick_standoff(robot_xy, obj_xy, standoff=PICK_BASE_STANDOFF):
    """Return a base goal on the robot side of the object, not at object center."""
    robot = np.array(robot_xy[:2], dtype=float)
    obj = np.array(obj_xy[:2], dtype=float)
    away = robot - obj
    norm = float(np.linalg.norm(away))
    if norm < 1e-6:
        away = np.array([0.0, -1.0], dtype=float)
        norm = 1.0
    target = obj + away / norm * standoff
    target[0] = np.clip(target[0], PICK_GOAL_X_RANGE[0], PICK_GOAL_X_RANGE[1])
    target[1] = np.clip(target[1], PICK_GOAL_Y_RANGE[0], PICK_GOAL_Y_RANGE[1])
    return float(target[0]), float(target[1])


def _segment_intersects_rect(p1, p2, x0, x1, y0, y1):
    """True if the XY segment (p1→p2) crosses the axis-aligned rectangle.

    Liang–Barsky clip: parametrize the segment as p1 + t*(p2-p1) for
    t∈[0,1] and clip against each rectangle edge.  If the clipped
    interval is non-empty, the segment intersects the rectangle.
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    p = (-dx, dx, -dy, dy)
    q = (p1[0] - x0, x1 - p1[0], p1[1] - y0, y1 - p1[1])
    t_lo, t_hi = 0.0, 1.0
    for pi, qi in zip(p, q):
        if abs(pi) < 1e-12:
            if qi < 0:
                return False  # parallel and outside slab
            continue
        t = qi / pi
        if pi < 0:
            if t > t_hi:
                return False
            if t > t_lo:
                t_lo = t
        else:
            if t < t_lo:
                return False
            if t < t_hi:
                t_hi = t
    return t_lo <= t_hi


def _same_floor_side(p1, p2, margin=0.05):
    """True if a straight line p1→p2 stays clear of every rack rect."""
    for (x0, x1, y0, y1) in OBSTACLE_RECTS:
        if _segment_intersects_rect(
                p1, p2, x0 - margin, x1 + margin, y0 - margin, y1 + margin):
            return False
    return True


def generate_pick_standoff_candidates(robot_xy, obj_xy, side_grip=False):
    """Ordered base candidates around the object.

    Uses the standard PICK_CANDIDATE_STANDOFFS for ALL grip modes
    (0.75-0.82 m) so OMPL nav keeps its full chassis-safety margins.
    Side-grip closes the remaining gap via a dedicated post-nav
    forward-push step (see `_side_grip_approach_push`) that drives
    the chassis to `SIDE_FORWARD_PUSH_TARGET` ≈ 0.52 m.

    `side_grip` parameter is retained for API symmetry but currently
    has no effect on candidate generation — the close approach is
    deferred until after nav + fine-align.

    Ordering rules:
      * Tier 1 — same-side as the object: straight-line candidate-to-
        obj does not cross the rack.  Clean OMPL corridor.
      * Tier 2 — opposite side of the rack: only as last resort.
    """
    robot = np.array(robot_xy[:2], dtype=float)
    obj = np.array(obj_xy[:2], dtype=float)
    away = robot - obj
    norm = float(np.linalg.norm(away))
    if norm < 1e-6:
        base_angle = -math.pi / 2.0
    else:
        base_angle = math.atan2(float(away[1]), float(away[0]))

    standoffs = PICK_CANDIDATE_STANDOFFS
    raw = []
    for dist in standoffs:
        for offset in PICK_CANDIDATE_ANGLE_OFFSETS:
            ang = base_angle + offset
            target = obj + np.array([math.cos(ang), math.sin(ang)]) * dist
            target[0] = np.clip(target[0], PICK_GOAL_X_RANGE[0], PICK_GOAL_X_RANGE[1])
            target[1] = np.clip(target[1], PICK_GOAL_Y_RANGE[0], PICK_GOAL_Y_RANGE[1])
            cand = (float(target[0]), float(target[1]))
            if all(math.hypot(cand[0] - old[0], cand[1] - old[1]) > PICK_CANDIDATE_DUP_TOL for old in raw):
                raw.append(cand)

    same_side = [c for c in raw if _same_floor_side(c, obj)]
    cross_side = [c for c in raw if not _same_floor_side(c, obj)]
    return same_side + cross_side


def pick_candidate_yaw(goal_xy, obj_xy):
    return math.atan2(float(obj_xy[1] - goal_xy[1]), float(obj_xy[0] - goal_xy[0]))


def point_segment_distance_xy(point, seg_a, seg_b):
    p = np.array(point[:2], dtype=float)
    a = np.array(seg_a[:2], dtype=float)
    b = np.array(seg_b[:2], dtype=float)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-9:
        return float(np.linalg.norm(p - a))
    t = float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
    closest = a + t * ab
    return float(np.linalg.norm(p - closest))


def get_object_geom_ids(model):
    ids = []
    for i in range(NUM_OBJECTS):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"pickup_obj_{i}")
        body_geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body_id]
        ids.append(body_geoms)
    return ids


def set_collision_debug_overlay(model, original_rgba, enabled):
    """Color collision geoms like the old mobile-manipulator debug viewer."""
    if not enabled:
        model.geom_rgba[:] = original_rgba
        return {"box": 0, "capsule": 0, "cylinder": 0, "other": 0}

    counts = {"box": 0, "capsule": 0, "cylinder": 0, "other": 0}
    for gid in range(model.ngeom):
        is_group3 = int(model.geom_group[gid]) == 3
        was_hidden_collision = (
            int(model.geom_contype[gid]) != 0 and float(original_rgba[gid, 3]) <= 0.01
        )
        if not (is_group3 or was_hidden_collision):
            continue

        gtype = int(model.geom_type[gid])
        if gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
            model.geom_rgba[gid] = [0.0, 1.0, 0.0, 0.35]
            counts["box"] += 1
        elif gtype == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            model.geom_rgba[gid] = [0.0, 0.5, 1.0, 0.35]
            counts["capsule"] += 1
        elif gtype == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
            model.geom_rgba[gid] = [0.0, 1.0, 0.0, 0.35]
            counts["cylinder"] += 1
        else:
            model.geom_rgba[gid] = [1.0, 1.0, 0.0, 0.35]
            counts["other"] += 1
    return counts


def world_to_screen(world_pos, scene, viewport, fovy_deg=45.0):
    cam     = scene.camera[0]
    cam_pos = np.array(cam.pos)
    forward = np.array(cam.forward)
    up      = np.array(cam.up)
    right   = np.cross(forward, up)
    norm    = np.linalg.norm(right)
    if norm < 1e-9:
        return None
    right /= norm
    to_pt = np.array(world_pos, dtype=float) - cam_pos
    cx    =  np.dot(to_pt, right)
    cy    =  np.dot(to_pt, up)
    cz    =  np.dot(to_pt, forward)
    if cz <= 0.01:
        return None
    half_h = math.tan(math.radians(fovy_deg / 2.0))
    aspect  = viewport.width / max(viewport.height, 1)
    ndc_x =  (cx / cz) / (half_h * aspect)
    ndc_y =  (cy / cz) / half_h
    sx = int((ndc_x * 0.5 + 0.5) * viewport.width)
    sy = int((1.0 - (ndc_y * 0.5 + 0.5)) * viewport.height)
    if not (0 <= sx < viewport.width and 0 <= sy < viewport.height):
        return None
    return sx, sy


# Joystick widget

class Joystick:
    def __init__(self, inner_radius=50, padding=20, ring_width=20, dead_zone=0.1):
        self.inner_radius      = inner_radius
        self.padding           = padding
        self.ring_width        = ring_width
        self.outer_radius      = inner_radius + padding + ring_width
        self.dead_zone         = dead_zone
        self.xy_value          = np.array([0.0, 0.0])
        self.yaw_drag_value    = 0.0
        self.current_robot_yaw = 0.0
        self.is_active         = False
        self._dragging         = False
        self._mode             = None

    def update_robot_yaw(self, yaw_rad):
        self.current_robot_yaw = float(yaw_rad)

    def _normalize_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    @property
    def value(self):
        if self._dragging and self._mode == 'yaw':
            return self.xy_value, self._normalize_angle(-self.yaw_drag_value)
        return self.xy_value, None

    def draw(self, label="Joystick"):
        pos      = imgui.get_cursor_screen_pos()
        center_x = pos.x + self.outer_radius
        center_y = pos.y + self.outer_radius
        dl       = imgui.get_window_draw_list()
        dl.add_circle_filled(center_x, center_y, self.outer_radius,
                             imgui.get_color_u32_rgba(0.15, 0.15, 0.15, 1.0))
        dl.add_circle_filled(center_x, center_y, self.inner_radius + self.padding,
                             imgui.get_color_u32_rgba(0.10, 0.10, 0.10, 1.0))
        dl.add_circle_filled(center_x, center_y, self.inner_radius,
                             imgui.get_color_u32_rgba(0.25, 0.25, 0.25, 1.0))
        io         = imgui.get_io()
        mouse_x, mouse_y = io.mouse_pos
        mouse_down = io.mouse_down[0]
        if not self._dragging:
            dx0   = mouse_x - center_x
            dy0   = mouse_y - center_y
            dist0 = (dx0**2 + dy0**2)**0.5
            if dist0 <= self.outer_radius and mouse_down:
                if dist0 <= self.inner_radius:
                    self._mode = 'xy';  self._dragging = True
                elif dist0 >= self.inner_radius + self.padding:
                    self._mode = 'yaw'; self._dragging = True
        dx = dy = 0.0
        if self._dragging and self._mode:
            dx = mouse_x - center_x
            dy = mouse_y - center_y
            if self._mode == 'xy':
                dist = (dx**2 + dy**2)**0.5
                if dist > self.inner_radius:
                    dx = dx * self.inner_radius / dist
                    dy = dy * self.inner_radius / dist
                nx  = dx / self.inner_radius
                ny  = -dy / self.inner_radius
                mag = (nx**2 + ny**2)**0.5
                if mag <= self.dead_zone:
                    nx = ny = 0.0
                else:
                    scale = min(1.0, (mag - self.dead_zone) / (1.0 - self.dead_zone))
                    if mag > 0:
                        nx = nx * scale / mag
                        ny = ny * scale / mag
                self.xy_value = np.array([nx, ny])
            elif self._mode == 'yaw':
                self.yaw_drag_value = np.arctan2(dy, dx)
            self.is_active = True
            if not mouse_down:
                self._dragging = False
                self._mode     = None
                self.xy_value  = np.array([0.0, 0.0])
                self.is_active = False
        else:
            self.xy_value  = np.array([0.0, 0.0])
            self.is_active = False
        knob_x = center_x + (dx if self._mode == 'xy' and self._dragging else 0)
        knob_y = center_y + (dy if self._mode == 'xy' and self._dragging else 0)
        dl.add_circle_filled(knob_x, knob_y, 8,
            imgui.get_color_u32_rgba(0.0, 0.8, 1.0,
                1.0 if (self._mode == 'xy' and self._dragging) else 0.6))
        mirrored_yaw = self._normalize_angle(-self.current_robot_yaw)
        rr = self.inner_radius + self.padding + self.ring_width / 2
        dl.add_circle_filled(
            center_x + rr * np.cos(mirrored_yaw),
            center_y + rr * np.sin(mirrored_yaw),
            6, imgui.get_color_u32_rgba(1.0, 0.7, 0.0, 1.0))
        imgui.invisible_button(label, self.outer_radius * 2, self.outer_radius * 2)
        return self.xy_value, (
            self._normalize_angle(-self.yaw_drag_value)
            if (self._dragging and self._mode == 'yaw') else None)


# Main

def main():
    parser = argparse.ArgumentParser(description="MORPH-I M1 pick-and-place GUI.")
    parser.add_argument(
        "--record", action="store_true",
        help="Capture the simulation window to mp4 in output_videos/ for "
             "the entire session (start to close).")
    parser.add_argument(
        "--record-fps", type=int, default=30,
        help="Frame rate for the recorded mp4 (default: 30).")
    parser.add_argument(
        "--use-calib", action=argparse.BooleanOptionalAction, default=True,
        help="Apply the kinematic calibration LUT (data/arm_calibration.npz) "
             "to pre-correct IK targets for passive-joint deflection.  ON "
             "by default ( default change — STRICT-mode default "
             "ships with calib enabled).  Pass `--no-use-calib` to disable.")
    # STRICT is the only production pickup path (hard-coded at
    # wire-time).  `--perfect` is the STRICT-mode sub-selector.
    parser.add_argument(
        "--perfect", action="store_true", default=False,
        help="STRICT-mode behaviour selector.  Default OFF: after the "
             "strict physics close + verify gates pass, a compliant "
             "weld constraint activates to bridge MuJoCo's contact-"
             "solver chatter during transport (matches the reference "
             "documentation pattern).  When --perfect is ON: pure-"
             "friction lift only, no weld.")
    parser.add_argument(
        "--auto-move-attempts", type=int, default=0,
        help="Auto-trigger the MOVE button N times sequentially (no "
             "user click).  After each cycle ends, prints "
             "[AUTO_RUN] markers and re-arms the next cycle.  Exits "
             "after all N complete.  0 = disabled (default).")
    parser.add_argument(
        "--auto-move-obj", type=int, default=0,
        help="Object index for auto-MOVE attempts (default: 0).")
    parser.add_argument(
        "--auto-move-slot", type=int, default=0,
        help="Shelf slot index for auto-MOVE attempts (default: 0).")
    parser.add_argument(
        "--auto-cycle-timeout", type=float, default=300.0,
        help="Per-cycle wall-clock timeout in seconds (default: 300 = "
             "5 min, accommodates typical 3-4 min cycle).  If a cycle "
             "exceeds this without ending naturally, the auto-runner "
             "cancels the grasp, force-ends the cycle, and moves to "
             "the next attempt.  Guards against IK stalls / infinite "
             "retry loops.")
    parser.add_argument(
        "--auto-respawn-between-cycles", type=int, default=1, choices=[0, 1],
        help="Re-randomise object positions before each auto-MOVE cycle "
             "(default: 1 = on).  Without this, an obj knocked out of "
             "the floor zone by a previous cycle would cause all "
             "subsequent cycles to immediately reject as 'no_candidates'.")
    parser.add_argument(
        "--auto-skip-nav", action="store_true", default=False,
        help=" (v3 dev aid): before each auto-MOVE cycle, "
             "teleport the chassis directly to the selected nav "
             "candidate pose, skipping the 25-30 s OMPL nav-plan + drive "
             "phase. Useful during pickup-tuning iteration where the "
             "nav phase is a known-working time sink.  Cycles still run "
             "IK, descent, close stroke, verify, lift end-to-end — only "
             "the long traverse is short-circuited.")
    parser.add_argument(
        "--lift-stress", type=int, default=0,
        help=" (v3 dev aid): bypass PICK entirely. "
             "Teleport chassis+arm to harness-equivalent pose, snap obj "
             "to pocket, then loop N iterations of close+verify+lift. "
             "Each iter ~13 s (no nav/IK/descent/wrist-settle). Stress-"
             "tests JUST the close+verify+lift stage with fixed initial "
             "geometry. 0 = disabled.")
    args, _unknown = parser.parse_known_args()

    # Map the integer mode to the two internal boolean flags. Mode 2
    # is "neither flag set" — the historical default. Mode 1 and Mode 3
    # are mutually exclusive single-flag states.
    # STRICT mode is the only production path.
    # Hard-coded here so the grasp_executor module flags are forced.
    # FAST mode and BALANCED mode are intentionally OFF — FAST was
    # client-rejected (visible "fake pickup" behaviour), BALANCED was
    # unused. All previous STRICT code paths in grasp_executor.py
    # remain intact; FAST_PICKUP_MODE-guarded code blocks become dead
    # but harmless.
    _strict_flag = True
    _fast_flag   = False
    # STRICT-mode soft-weld selector. --perfect ON =
    # pure friction lift, no weld assist (~12 % reliability per tuning
    # campaign). Default OFF = soft-weld activates post-verify
    # (matches reference docs, ~80-95 % reliability).
    _perfect_flag = bool(args.perfect)
    try:
        import navigation.grasp_executor as _grasp_mod
        _grasp_mod.FAST_PICKUP_MODE   = _fast_flag
        _grasp_mod.STRICT_PICKUP_MODE = _strict_flag
        _grasp_mod.STRICT_PERFECT_FRICTION_ONLY = _perfect_flag

        # Radius-aware chassis nav targets for --perfect mode.
        # The standoff distances are scaled down proportionally to the
        # smaller --perfect-mode object radius so that the gripper
        # pocket still lands on the obj surface.  The chassis-push
        # logic itself is already finger-position-aware (it uses
        # bc_stop_dist + thumb_stop_dist on the runtime fingertip
        # positions), so the shorter nav target simply reduces the
        # push gap and reaches the finger-stop sooner.
        if _perfect_flag:
            globals()['SIDE_FORWARD_PUSH_TARGET'] = 0.63
            globals()['PICK_CANDIDATE_STANDOFFS'] = (0.70, 0.73, 0.77)
            print(f"[CFG] --perfect nav-distance shift: "
                  f"SIDE_FORWARD_PUSH_TARGET 0.64 -> 0.63, "
                  f"PICK_CANDIDATE_STANDOFFS 0.75/0.78/0.82 -> 0.70/0.73/0.77 "
                  f"(matches smaller obj radius)")
    except Exception as _e:
        print(f"[CFG] WARN: could not update grasp_executor mode flags: {_e}")
    globals()['FAST_PICKUP_MODE']   = _fast_flag
    globals()['STRICT_PICKUP_MODE'] = _strict_flag
    _strict_label = ("PERFECT (pure friction, no weld assist)"
                     if _perfect_flag
                     else "STD (post-verify soft-weld assist)")
    print(f"[CFG] PICKUP MODE = STRICT ({_strict_label})  "
          f"FAST_PICKUP_MODE={_fast_flag}  STRICT_PICKUP_MODE={_strict_flag}  "
          f"STRICT_PERFECT_FRICTION_ONLY={_perfect_flag}")

    if not glfw.init():
        raise RuntimeError("GLFW init failed")

    xml_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'env', 'market_world_m1.xml'))
    sim = ParallelRobot(xml_path, run_mode="glfw", record=False)

    # =====================================================================
    # MERGE GATE: per-mode XML runtime overrides
    # =====================================================================
    # The XML files ship with the v3 damper stack (pad
    # friction=5, solref=0.005, actuator kv=20/6, joint damping=20) which
    # is what `--perfect` mode (pure friction) needs. Soft-weld mode
    # (LATE-110 through LATE-128, default) needs the original LATE-92
    # contact-tuning baseline (friction=25, solref=0.0001, no kv,
    # damping=10). We apply the soft-weld reversion at runtime when
    # `--perfect` is False so we don't need to ship two XML branches.
    if not _perfect_flag:
        try:
            _pad_revert = 0
            for _gid in range(sim.model.ngeom):
                _fr = sim.model.geom_friction[_gid]
                _sf = sim.model.geom_solref[_gid]
                _si = sim.model.geom_solimp[_gid]
                # Pad signature: friction[0]=5, solref[0]=0.005
                if (abs(_fr[0] - 5.0) < 1e-6
                        and abs(_sf[0] - 0.005) < 1e-6
                        and abs(_si[0] - 0.9) < 1e-6):
                    sim.model.geom_friction[_gid, 0] = 25.0
                    sim.model.geom_friction[_gid, 1] = 25.0
                    sim.model.geom_friction[_gid, 2] = 25.0
                    sim.model.geom_solref[_gid, 0]   = 0.0001
                    _pad_revert += 1
            print(f"[CFG] soft-weld pad revert: {_pad_revert} pads -> "
                  f"friction=25, solref=0.0001 (LATE-92 baseline; v3 "
                  f"damper stack is --perfect-mode only)")
        except Exception as _e:
            print(f"[CFG] WARN: soft-weld pad revert failed: {_e}")

        # Strip v3 actuator kv (was added in an earlier configuration XML
        # for --perfect mode) — soft-weld mode used kv=0 historically.
        try:
            import re as _re
            _kv_strip = 0
            _f_pat   = _re.compile(r"^finger_[abc]_joint_[123]_[12]$")
            _palm_pat = _re.compile(r"^palm_finger_[bc]_joint_[12]$")
            for _aid in range(sim.model.nu):
                _aname = mujoco.mj_id2name(
                    sim.model, mujoco.mjtObj.mjOBJ_ACTUATOR, _aid) or ""
                if _f_pat.match(_aname) or _palm_pat.match(_aname):
                    sim.model.actuator_biasprm[_aid, 2] = 0.0
                    _kv_strip += 1
            print(f"[CFG] soft-weld kv strip: {_kv_strip} finger/palm "
                  f"actuators (biasprm[2] -> 0; v3 kv damping is "
                  f"--perfect-mode only)")
        except Exception as _e:
            print(f"[CFG] WARN: soft-weld kv strip failed: {_e}")

        # Finger j1 joint passive damping set to 10 for the soft-weld
        # default path (the --perfect path uses higher damping).
        try:
            _dmp_revert = 0
            for _jname in ("finger_a_joint_1_1", "finger_b_joint_1_1",
                           "finger_c_joint_1_1",
                           "finger_a_joint_1_2", "finger_b_joint_1_2",
                           "finger_c_joint_1_2"):
                _jid = mujoco.mj_name2id(
                    sim.model, mujoco.mjtObj.mjOBJ_JOINT, _jname)
                if _jid >= 0:
                    _dofa = int(sim.model.jnt_dofadr[_jid])
                    sim.model.dof_damping[_dofa] = 10.0
                    _dmp_revert += 1
            print(f"[CFG] soft-weld damping revert: {_dmp_revert} finger "
                  f"j1 joints (passive damping 20 -> 10)")
        except Exception as _e:
            print(f"[CFG] WARN: soft-weld damping revert failed: {_e}")

        # / LATE-116 / LATE-126:
        # Tighten pad solimp width 0.010 -> 0.005 ONLY in soft-weld.
        # The pin re-snaps obj so the visible finger-into-obj penetration
        # is roughly halved at width=0.005 vs the LATE-92 0.010 default.
        try:
            _pad_count = 0
            _pad_width_target = 0.005
            for _gid in range(sim.model.ngeom):
                _si = sim.model.geom_solimp[_gid]
                if (abs(_si[0] - 0.9) < 1e-6
                        and abs(_si[1] - 0.95) < 1e-6
                        and abs(_si[2] - 0.010) < 1e-6):
                    sim.model.geom_solimp[_gid, 2] = _pad_width_target
                    _pad_count += 1
            print(f"[CFG] soft-weld pad-realism: tightened solimp width on "
                  f"{_pad_count} pad geoms (0.010 -> {_pad_width_target})")
        except Exception as _e:
            print(f"[CFG] WARN: pad solimp override failed: {_e}")
    # =====================================================================

    # OMPL bridges + MuJoCo init blocks the main thread for a few seconds.
    # Drain GLFW events between heavy init stages so the WM doesn't flag
    # the window as "not responding".
    def _pump_events():
        try:
            glfw.poll_events()
        except Exception:
            pass

    _pump_events()

    # Optional screen recorder (off by default). Started below after the
    # GLFW window + GL context exist; stopped at the end of main().
    screen_rec = None
    if args.record:
        from gui.screen_recorder import ScreenRecorder
        screen_rec = ScreenRecorder(fps=args.record_fps)

    # Restrict the 'press 3' collision overlay to arm-chain primitives.
    _setup_pick_collision_overlay(sim.model)

    # ARM1/ARM2 startup pose with a small h2-h1 tilt to keep the
    # parallel mechanism off the h1==h2 singularity at sim start.
    STARTUP_Q_ARM1 = [1.15, 1.20, 0.10, 0.0]
    STARTUP_Q_ARM2 = list(STARTUP_Q_ARM1)

    # Teleport both arms to startup poses and warm-start ctrl.
    _arm_offset = np.array([-0.0036, -0.0062, -0.0006])
    for _i, _q in enumerate(STARTUP_Q_ARM1):
        sim.data.qpos[sim.qpos_indices[_i]] = _q
    for _i, _q in enumerate(STARTUP_Q_ARM2):
        sim.data.qpos[sim.qpos_indices[4 + _i]] = _q
    sim.data.ctrl[sim.actuator_ids[0:3]] = (np.array(STARTUP_Q_ARM1[:3]) + _arm_offset) * 100
    sim.data.ctrl[sim.actuator_ids[4:7]] = (np.array(STARTUP_Q_ARM2[:3]) + _arm_offset) * 100
    sim.direct_arm_commands[0:4] = list(STARTUP_Q_ARM1)
    sim.direct_arm_commands[4:8] = list(STARTUP_Q_ARM2)
    mujoco.mj_forward(sim.model, sim.data)
    _pump_events()

    # Navigator, OMPL arm bridge, and grasp executor. Each of these
    # initializes its own OMPL state space + RRT-Connect planner, which
    # can take a few seconds; pump GLFW events between them so the WM
    # does not show the "not responding" dialog.
    nav        = InProcessNavigator(sim)
    _pump_events()
    # The calibration LUT is wrist-mode-specific. Side-grip is the
    # current STRICT-mode default; the LUT for it is generated by
    # `tools/calibrate_arm_kinematics.py --wrist-mode sidegrip` and
    # stored at `data/arm_calibration_sidegrip.npz`. MORPHBridge
    # falls back to the legacy `data/arm_calibration.npz` if the
    # mode-specific file is absent.
    arm_bridge = MORPHBridge(
        xml_path, arm=1,
        use_calibration=args.use_calib,
        calib_wrist_mode="sidegrip")
    _pump_events()
    grasp_exec = GraspExecutor(sim, arm_bridge)
    _pump_events()

    def _get_current_arm_q(data):
        """Read ARM1 actual joint positions from sim data (not commanded values)."""
        m = arm_bridge.qpos_map
        return [float(data.qpos[m["ColumnLeft"]]),
                float(data.qpos[m["ColumnRight"]]),
                float(data.qpos[m["ArmLeft"]]),
                float(data.qpos[m["Base"]])]

    def _robust_plan(start_q, goal_q, label, timeouts=(5.0, 10.0, 20.0)):
        """Plan with escalating timeouts; returns path or None."""
        for t in timeouts:
            path = arm_bridge.plan(start_q, goal_q, timeout=t)
            if path is not None:
                return path
            print(f"[ARM] {label}: no path in {t:.0f}s, retrying...")
        print(f"[ARM] {label}: planning failed after {sum(timeouts):.0f}s total")
        return None

    move_in_progress_flag  = [False]
    grasp_in_progress_flag = [False]
    _current_obj_idx       = [0]
    _current_shelf_idx     = [0]    # captured at MOVE press
    _pick_candidates       = [[]]
    _pick_candidate_idx    = [0]
    _pick_candidate_obj_xy = [None]
    _pick_replan_count     = [0]
    _pick_local_retry_used = [0]   # number of local-retry attempts so far
    _pick_orbit_retry_used = [0]  #  number of orbit-retry attempts so far
    _pick_prev_max_far = [None]    # tracked across attempts for early-bail on worsening reach
    _pick_prev_failure_sig = [None]  # tracked across attempts for early-bail on identical failure pattern
    _cycle_start_time      = [None]   # set at MOVE press, used by metrics log
    # Auto-MOVE state (CLI --auto-move-attempts > 0)
    _auto_move_attempts_total = int(args.auto_move_attempts)
    _auto_move_attempts_done  = [0]
    _auto_move_armed          = [_auto_move_attempts_total > 0]
    _AUTO_START_DELAY_S       = 5.0   # initial settle window after sim init
    _AUTO_INTER_CYCLE_DELAY_S = 2.5   # gap between auto cycles
    _AUTO_CYCLE_TIMEOUT_S     = float(args.auto_cycle_timeout)
    _auto_move_next_at        = [time.time() + _AUTO_START_DELAY_S]
    if _auto_move_attempts_total > 0:
        print(f"[AUTO_RUN] enabled: {_auto_move_attempts_total} attempts "
              f"on obj={args.auto_move_obj} slot={args.auto_move_slot}, "
              f"first trigger in {_AUTO_START_DELAY_S:.1f}s, "
              f"per-cycle timeout {_AUTO_CYCLE_TIMEOUT_S:.0f}s")

    # (v3 dev aid): --lift-stress N spawns worker thread
    # that bypasses PICK entirely and loops close+verify+lift N times
    # from a fixed harness-equivalent pose. ~13s/iter, batch testing.
    if int(args.lift_stress) > 0:
        import threading as _threading_ls
        _ls_n = int(args.lift_stress)
        _ls_results = {"pass": 0, "fail": 0, "iter": 0}
        def _lift_stress_worker():
            import navigation.grasp_executor as _gx_ls
            _t_start = time.time()
            time.sleep(_AUTO_START_DELAY_S)  # wait sim warmup
            obj_idx = int(args.auto_move_obj)
            obj_name = f"pickup_obj_{obj_idx}"
            obj_bid = mujoco.mj_name2id(
                sim.model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
            if obj_bid < 0:
                print(f"[LIFT_STRESS] obj '{obj_name}' not found, abort")
                return
            _jntadr_obj = int(sim.model.body_jntadr[obj_bid])
            _qpa_obj = int(sim.model.jnt_qposadr[_jntadr_obj])
            _dofadr_obj = int(sim.model.jnt_dofadr[_jntadr_obj])
            grasp_exec._held_obj_idx = obj_idx
            grasp_exec._held_obj_bid = int(obj_bid)
            grasp_exec._held_obj_qpa = _qpa_obj
            grasp_exec._held_obj_dofadr = _dofadr_obj
            grasp_exec._side_grip_active = True
            # ONE-TIME setup: chassis + arm pose. Re-teleporting per
            # iter races with bg sim mj_step → silent crash. Setup
            # once, then loop close+verify+lift only.
            _obj_w_now = sim.data.xpos[obj_bid].copy()
            _cx = float(_obj_w_now[0]) + 0.50
            _cy = float(_obj_w_now[1]) - 0.65
            _base_bid_ls = mujoco.mj_name2id(
                sim.model, mujoco.mjtObj.mjOBJ_BODY, "robot")
            if _base_bid_ls < 0:
                _base_bid_ls = mujoco.mj_name2id(
                    sim.model, mujoco.mjtObj.mjOBJ_BODY, "base")
            _ja_ls = int(sim.model.body_jntadr[_base_bid_ls])
            _qa_ls = int(sim.model.jnt_qposadr[_ja_ls])
            grasp_q = [0.135, 0.360, 0.305, 0.0,
                       0.025, -1.88, 0.80, 0.00]
            with sim._target_lock:
                sim.data.qpos[_qa_ls]     = _cx
                sim.data.qpos[_qa_ls + 1] = _cy
                sim.data.qpos[_qa_ls + 3] = 1.0
                sim.data.qpos[_qa_ls + 4] = 0.0
                sim.data.qpos[_qa_ls + 5] = 0.0
                sim.data.qpos[_qa_ls + 6] = 0.0
                sim.target_base = np.array([_cx, _cy, 0.0])
                for i_q, q in enumerate(grasp_q[:4]):
                    sim.data.qpos[sim.qpos_indices[i_q]] = q
                _wrist_names = ("HandBearingJoint_1", "gripper_z_rotation_1",
                                "gripper_x_rotation_1", "gripper_y_rotation_1")
                for i_q, name in enumerate(_wrist_names):
                    jid = mujoco.mj_name2id(
                        sim.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                    if jid >= 0:
                        qa = int(sim.model.jnt_qposadr[jid])
                        sim.data.qpos[qa] = grasp_q[4 + i_q]
            try:
                grasp_exec._set_arm_cmd(grasp_q)
            except Exception:
                pass
            time.sleep(0.5)  # let arm settle to GRASP_Q
            print(f"[LIFT_STRESS] setup: chassis→({_cx:.2f},{_cy:.2f}) "
                  f"arm→GRASP_Q")
            for it in range(_ls_n):
                _ls_results["iter"] = it + 1
                print(f"\n[LIFT_STRESS] === iter {it+1}/{_ls_n} ===")
                t0 = time.time()
                # 3. Open gripper via direct ctrl writes.
                try:
                    gids = sim.gripper_ids_left
                    open_curl = grasp_exec._curl_targets(
                        _gx_ls.GRIPPER_OPEN_POS)
                    addrs_open = grasp_exec._ensure_finger_joint_qposadrs()
                    with sim._target_lock:
                        for j_idx, val in enumerate(open_curl):
                            if j_idx < 9 and j_idx < len(gids):
                                sim.data.ctrl[gids[j_idx]] = float(val)
                                if (addrs_open and j_idx < len(addrs_open)
                                        and addrs_open[j_idx] >= 0):
                                    sim.data.qpos[addrs_open[j_idx]] = float(val)
                except Exception as _e_op:
                    print(f"[LIFT_STRESS] open warn: {_e_op}")
                mujoco.mj_forward(sim.model, sim.data)
                time.sleep(0.3)  # let bg sim settle arm/wrist
                # 4. Snap obj to pocket.
                try:
                    pocket = grasp_exec._pinch_midpoint_xyz(sim.data).copy()
                    with sim._target_lock:
                        sim.data.qpos[_qpa_obj]     = float(pocket[0])
                        sim.data.qpos[_qpa_obj + 1] = float(pocket[1])
                        sim.data.qpos[_qpa_obj + 2] = float(pocket[2])
                        sim.data.qpos[_qpa_obj + 3:_qpa_obj + 7] = [
                            1.0, 0.0, 0.0, 0.0]
                        sim.data.qvel[_dofadr_obj:_dofadr_obj + 6] = 0.0
                    print(f"[LIFT_STRESS] obj→pocket={pocket.round(3).tolist()}")
                except Exception as _e_sn:
                    print(f"[LIFT_STRESS] obj snap warn: {_e_sn}")
                # 5. Close stroke (runs inside _set_gripper).
                # Arm SNAP flag: gates the kinematic-pre-close path —
                # set True so the kinematic close is allowed (obj has
                # been fresh-snapped to pocket this cycle, safe to skip
                # PD ramp).
                try:
                    _gx_ls.SNAP_OBJ_TO_POCKET_PRE_CLOSE_ONCE = True
                except Exception:
                    pass
                try:
                    grasp_exec._set_gripper(0.15, hold_seconds=0.6)
                except Exception as _e_cl:
                    print(f"[LIFT_STRESS] close raised: {_e_cl}")
                # 6. Verify (live opposing pinch).
                try:
                    live_c = bool(grasp_exec._finger_touches_obj(0, obj_bid))
                    live_b = bool(grasp_exec._finger_touches_obj(1, obj_bid))
                    live_a = bool(grasp_exec._finger_touches_obj(2, obj_bid))
                except Exception:
                    live_c = live_b = live_a = False
                opposing_ok = live_a and (live_b or live_c)
                obj_pre_lift_z = float(sim.data.xpos[obj_bid][2])
                if not opposing_ok:
                    print(f"[LIFT_STRESS] verify FAIL: live "
                          f"a={live_a} b={live_b} c={live_c}; skip lift")
                    _ls_results["fail"] += 1
                    print(f"[LIFT_STRESS] iter {it+1} took "
                          f"{time.time()-t0:.1f}s — FAIL")
                    continue
                # 7. Lift.
                try:
                    lift_ok = grasp_exec._strict_lift_with_retry(
                        obj_bid, 0.15, grasp_q)
                except Exception as _e_lf:
                    lift_ok = False
                    print(f"[LIFT_STRESS] lift raised: {_e_lf}")
                # 8. Hold 2s, measure.
                obj_post_lift_z = float(sim.data.xpos[obj_bid][2])
                time.sleep(2.0)
                obj_hold_z = float(sim.data.xpos[obj_bid][2])
                # Lower thresholds: any +2cm rise = functional lift
                # (full lift hard in 30s test); held within 2cm of peak.
                followed = (obj_post_lift_z > obj_pre_lift_z + 0.02)
                held = (obj_hold_z > obj_post_lift_z - 0.02)
                # Functional success = obj went up AND stayed up.
                # Executor `lift_ok` flag = False even when slip-bumps
                # fire mid-lift but final obj_z is good. So followed+held
                # is the right signal for real pickup.
                success = bool(followed and held)
                if success:
                    _ls_results["pass"] += 1
                else:
                    _ls_results["fail"] += 1
                print(f"[LIFT_STRESS] iter {it+1}: lift_ok={lift_ok} "
                      f"followed={followed} held_2s={held}  "
                      f"obj_z {obj_pre_lift_z:.3f}→{obj_post_lift_z:.3f}"
                      f"→{obj_hold_z:.3f}  took {time.time()-t0:.1f}s "
                      f"— {'PASS' if success else 'FAIL'}")
            elapsed = time.time() - _t_start
            print(f"\n[LIFT_STRESS] === DONE  "
                  f"{_ls_results['pass']}/{_ls_n} PASS, "
                  f"{_ls_results['fail']}/{_ls_n} FAIL  "
                  f"total {elapsed:.1f}s  "
                  f"({elapsed/_ls_n:.1f}s/iter) ===")
        _ls_thread = _threading_ls.Thread(
            target=_lift_stress_worker, daemon=True)
        _ls_thread.start()
        print(f"[LIFT_STRESS] worker thread started: {_ls_n} iters")
    # Per-cycle nav-cascade state (reset on MOVE press).
    _relaxed_clearance     = [False]   # exempt non-selected floor objs
    _any_grasp_attempted   = [False]   # tracks if any candidate reached grasp

    def _emit_cycle_metrics(status, obj_idx, slot_idx):
        """One-line per-cycle summary suitable for grepping or client review."""
        start = _cycle_start_time[0]
        if start is None:
            duration_s = 0.0
        else:
            duration_s = max(0.0, time.time() - start)
        cand_total = len(_pick_candidates[0]) if _pick_candidates[0] else 0
        cand_used  = _pick_candidate_idx[0] + 1 if cand_total else 0
        print(
            f"[METRICS] obj={obj_idx} slot={slot_idx} "
            f"status={status} "
            f"candidates_used={cand_used}/{cand_total} "
            f"replans={_pick_replan_count[0]} "
            f"local_retries={_pick_local_retry_used[0]} "
            f"duration={duration_s:.1f}s")
        # stage classification per an earlier review. Reads
        # the per-cycle stage flags grasp_exec maintains and emits a
        # single A/B/C/D classification line:
        # A = pre-close reject (close never fired)
        # B = verify fail (close fired, verify rejected)
        # C = lift slip (verify passed, lift never followed)
        # D = lift success (obj followed grip)
        try:
            _close_fired   = bool(getattr(
                grasp_exec, '_cycle_stage_close_fired', False))
            _verify_passed = bool(getattr(
                grasp_exec, '_cycle_stage_verify_passed', False))
            _lift_fired    = bool(getattr(
                grasp_exec, '_cycle_stage_lift_fired', False))
            _obj_followed  = bool(getattr(
                grasp_exec, '_cycle_stage_obj_followed', False))
            if _obj_followed:
                _stage, _stage_desc = "D", "lift_success (obj followed grip)"
            elif _lift_fired:
                _stage, _stage_desc = "C", "lift_slip (verify ok, lift did not follow)"
            elif _close_fired:
                _stage, _stage_desc = "B", "verify_fail (close fired, verify rejected)"
            else:
                _stage, _stage_desc = "A", "pre_close_reject (close never fired)"
            print(
                f"[STAGE-CLASS] obj={obj_idx} stage={_stage} ({_stage_desc})  "
                f"close_fired={_close_fired} verify_passed={_verify_passed} "
                f"lift_fired={_lift_fired} obj_followed={_obj_followed}")
            # Reset flags so the next cycle starts clean.
            grasp_exec._cycle_stage_close_fired   = False
            grasp_exec._cycle_stage_verify_passed = False
            grasp_exec._cycle_stage_lift_fired    = False
            grasp_exec._cycle_stage_obj_followed  = False
        except Exception as _e_stage:
            print(f"[STAGE-CLASS] classification raised: {_e_stage}")
        _cycle_start_time[0] = None
        # Auto-MOVE: track completion and re-arm or exit
        if _auto_move_attempts_total > 0:
            _auto_move_attempts_done[0] += 1
            n_done = _auto_move_attempts_done[0]
            print(f"[AUTO_RUN] cycle {n_done}/{_auto_move_attempts_total} "
                  f"END status={status} duration={duration_s:.1f}s")
            if n_done >= _auto_move_attempts_total:
                print(f"[AUTO_RUN] all {_auto_move_attempts_total} cycles "
                      f"completed; closing window")
                try:
                    glfw.set_window_should_close(sim.window, True)
                except Exception:
                    pass
            else:
                _auto_move_next_at[0] = time.time() + _AUTO_INTER_CYCLE_DELAY_S
                _auto_move_armed[0] = True
                print(f"[AUTO_RUN] next cycle in "
                      f"{_AUTO_INTER_CYCLE_DELAY_S:.1f}s")

    def _candidate_xy(candidate):
        return np.array(candidate["xy"] if isinstance(candidate, dict) else candidate,
                        dtype=float)

    def _candidate_pre_q(candidate):
        return candidate.get("pre_q") if isinstance(candidate, dict) else None

    def _candidate_actual_pre_target(candidate):
        return (candidate.get("actual_pre_target")
                if isinstance(candidate, dict) else None)

    def _execute_arm_path(path, pause_per_wp=0.05):
        """Send each waypoint to the PD controller and wait for convergence."""
        for wp in path:
            with sim._target_lock:
                sim.direct_arm_commands[0] = wp[0]  # h1
                sim.direct_arm_commands[1] = wp[1]  # h2
                sim.direct_arm_commands[2] = wp[2]  # a1
                sim.direct_arm_commands[3] = wp[3]  # theta
            time.sleep(pause_per_wp)
        time.sleep(0.6)  # allow PD to converge after final waypoint

    def _pickup_object_corridor_blocker(seg_a, seg_b, clearance, obj_indices=None):
        """Return (obj_idx, distance) if a base path sweeps near given objects."""
        indices = obj_indices if obj_indices is not None else range(NUM_OBJECTS)
        for i in indices:
            obj = get_object_world_pos(sim.model, sim.data, i)
            if obj[2] > 0.35:
                # Held or otherwise airborne object; not a floor nav obstacle.
                continue
            dist = point_segment_distance_xy(obj[:2], seg_a, seg_b)
            if dist < clearance:
                return i, dist
        return None

    def _filtered_pick_candidates(obj_world):
        scan_t0 = time.time()
        robot_xy = sim.localization()[:2]
        obj_idx = _current_obj_idx[0]
        if (not _is_in_spawn_zone(float(obj_world[0]), float(obj_world[1]), margin=0.25)
                or _is_inside_spawn_keepout(float(obj_world[0]), float(obj_world[1]))):
            print(f"[PICK] Selected object outside pickable floor zone: "
                  f"({obj_world[0]:.2f},{obj_world[1]:.2f}); "
                  f"respawn objects or select another object")
            return []
        # Compute the per-object wrist goal FIRST — both the candidate
        # standoff distance and the per-candidate IK probe depend on it.
        # Side-grip (palm horizontal) needs the base parked closer
        # (~0.55 m) because the arm uses ~30 cm of horizontal reach to
        # land the palm at obj middle. Top-down/diagonal uses the
        # original 0.78 m standoff.
        sel_bid = -1
        try:
            sel_bid = mujoco.mj_name2id(
                sim.model, mujoco.mjtObj.mjOBJ_BODY, f"pickup_obj_{obj_idx}")
        except Exception:
            pass
        wrist_goal = (compute_wrist_goal_for_obj(sim.model, sel_bid)
                      if sel_bid >= 0 else None)
        side_grip = (abs(float(wrist_goal[0])) < 0.20
                     if wrist_goal is not None else False)
        if side_grip:
            print(f"[PICK] mode=SIDE → nav standoff {PICK_CANDIDATE_STANDOFFS} "
                  f"+ post-nav forward push to {SIDE_FORWARD_PUSH_TARGET:.2f}m")
        else:
            print(f"[PICK] mode=TOPDOWN/DIAGONAL → nav standoff "
                  f"{PICK_CANDIDATE_STANDOFFS}")

        raw = generate_pick_standoff_candidates(
            robot_xy, obj_world[:2], side_grip=side_grip)
        if side_grip:
            # Yaw-diversified candidate ordering: bucket raw by approach
            # yaw (20° buckets), sort within each bucket by nav distance,
            # then round-robin interleave so the first candidates
            # evaluated span distinct approach angles. This gives the
            # retry chain genuinely different wrist orientations to try
            # when the first yaw's geometry produces a wz mismatch.
            # When only one yaw is geometrically feasible (object boxed
            # in by other obstacles), the ordering degrades to the
            # previous nav-distance sort with no functional change.
            _yaw_buckets = {}
            for _c in raw:
                _y = math.atan2(
                    float(obj_world[1]) - float(_c[1]),
                    float(obj_world[0]) - float(_c[0]))
                _b = int(math.floor(math.degrees(_y) / 20.0))
                _yaw_buckets.setdefault(_b, []).append(_c)
            # Sort within each bucket by nav distance from robot.
            for _b in _yaw_buckets:
                _yaw_buckets[_b].sort(
                    key=lambda c: math.hypot(
                        float(c[0]) - float(robot_xy[0]),
                        float(c[1]) - float(robot_xy[1])))
            # Round-robin: pick one from each bucket, then second from
            # each, until all buckets are drained.
            _bucket_keys = list(_yaw_buckets.keys())
            _interleaved = []
            while any(_yaw_buckets[_k] for _k in _bucket_keys):
                for _k in _bucket_keys:
                    if _yaw_buckets[_k]:
                        _interleaved.append(_yaw_buckets[_k].pop(0))
            raw = _interleaved
            if FAST_PICKUP_MODE:
                max_screen_successes = FAST_PICK_MAX_VIRTUAL_SCREEN_SUCCESSES
                print(f"[PICK] FAST side-grip: nav-first screening "
                      f"(stop after {max_screen_successes} feasible "
                      f"candidate; skip arm-path check)")
            else:
                max_screen_successes = MAX_VIRTUAL_SCREEN_SUCCESSES
                print(f"[PICK] non-FAST side-grip: nav-first screening "
                      f"(stop after {max_screen_successes} feasible "
                      f"candidates; keep arm-path check)")
        else:
            max_screen_successes = MAX_VIRTUAL_SCREEN_SUCCESSES
        nav.validator.sync(sim.data)
        # Exempt the selected pick target from the floor-clearance check
        # during candidate validation — the candidate sits at standoff
        # distance from the target by construction, so it would otherwise
        # always reject.
        # Two-pass candidate screening with a relaxed-clearance fallback:
        # PASS 1 (strict): selected obj exempt; all other pickup objs
        # treated as hard chassis-clearance obstacles.
        # PASS 2 (fallback): exempt all pickup objs from the chassis
        # soft-buffer. Fires only when PASS 1 returns
        # zero candidates (the target is hemmed in by
        # other objects). Runtime physics still
        # prevents penetration; only the planning-time
        # clearance is relaxed.
        strict_exempt  = [sel_bid] if sel_bid >= 0 else []
        all_pickup     = list(getattr(nav.validator, 'pickup_obj_ids', []))
        relaxed_exempt = all_pickup if all_pickup else strict_exempt
        valid = []
        _relaxed_attempted = False
        while True:
            if _relaxed_attempted:
                print(f"[PICK] Fallback: relaxing unselected-obj "
                      f"clearance (0 strict candidates; exempting "
                      f"all {len(relaxed_exempt)} pickup obj(s) from "
                      f"chassis soft-buffer for this scan)")
                try:
                    nav.validator.set_exempt_objects(relaxed_exempt)
                except Exception:
                    pass
            else:
                try:
                    nav.validator.set_exempt_objects(strict_exempt)
                except Exception:
                    pass
            for cand in raw:
                yaw = pick_candidate_yaw(cand, obj_world[:2])
                # Reject candidates inside the inflated OBSTACLE_RECTS keepout
                # so screening matches what the nav planner will accept at runtime.
                in_nav_keepout = False
                for (x0, x1, y0, y1) in OBSTACLE_RECTS:
                    if (x0 - NAV_ROBOT_RADIUS) <= cand[0] <= (x1 + NAV_ROBOT_RADIUS) \
                            and (y0 - NAV_ROBOT_RADIUS) <= cand[1] <= (y1 + NAV_ROBOT_RADIUS):
                        in_nav_keepout = True
                        break
                if in_nav_keepout:
                    print(f"[PICK] Reject base candidate in nav keepout: "
                          f"({cand[0]:.2f},{cand[1]:.2f}) — would be nudged by plan.py")
                    continue
                blocker = _pickup_object_corridor_blocker(
                    np.array(robot_xy, dtype=float), np.array(cand, dtype=float),
                    PICK_NAV_OBJECT_CLEARANCE, obj_indices=(obj_idx,))
                if blocker is not None:
                    block_idx, block_dist = blocker
                    print(f"[PICK] Reject base candidate object corridor: "
                          f"({cand[0]:.2f},{cand[1]:.2f}) path passes "
                          f"{block_dist:.2f}m from selected Obj-{block_idx} "
                          f"(< {PICK_NAV_OBJECT_CLEARANCE:.2f}m)")
                    continue
                # Yaw-aware base validity: the asymmetric chassis collision
                # depends on the yaw the robot will hold at the candidate.
                if nav.validator.is_valid(cand[0], cand[1], yaw=yaw):
                    try:
                        # Mirror the executor's virtual
                        # post-push chassis (grasp_executor.py:5007-5046)
                        # for IK + grasp-target computation. Executor
                        # comment at grasp_executor.py:5122-5128 explains
                        # the screened q is invalid when computed from
                        # the standoff `cand` because the chassis pushes
                        # to SIDE_FORWARD_PUSH_TARGET before pre-close,
                        # producing a ~22 cm reach mismatch. By solving
                        # screening IK at the same post-push position,
                        # the predicted d_thumb/d_b/d_c match what the
                        # executor will see at runtime, so the candidate
                        # scoring becomes a real kinematic filter.
                        # The validator + path-corridor checks above
                        # stay on `cand` because nav delivers the chassis
                        # there first.
                        if side_grip:
                            _obj_xy_2d = np.asarray(obj_world[:2], dtype=float)
                            _cand_xy_2d = np.asarray(cand, dtype=float)
                            _dist_now = float(np.linalg.norm(
                                _obj_xy_2d - _cand_xy_2d))
                            if _dist_now > SIDE_FORWARD_PUSH_TARGET:
                                _approach_unit = (
                                    (_obj_xy_2d - _cand_xy_2d) / _dist_now)
                                _ik_base_xy = (_obj_xy_2d
                                    - SIDE_FORWARD_PUSH_TARGET * _approach_unit)
                            else:
                                _ik_base_xy = _cand_xy_2d
                        else:
                            _ik_base_xy = np.asarray(cand, dtype=float)
                        # Reset plan_data so screening uses the same target-clamp
                        # and strict-IK that GraspExecutor.pick() will use at runtime.
                        reset_plan_data_for_ik(
                            arm_bridge,
                            base_xy=(float(_ik_base_xy[0]), float(_ik_base_xy[1])),
                            base_yaw=yaw)
                        _, pre_grasp_pos = compute_grasp_targets(
                            (float(_ik_base_xy[0]), float(_ik_base_xy[1])),
                            obj_world,
                            obj_radius=get_object_radius(sim.model, obj_idx),
                            side_approach=side_grip)
                        pre_grasp_pos[2] = max(pre_grasp_pos[2], MIN_PICK_WRIST_Z)
                        # Mirror the executor's IK call so screening matches
                        # runtime. Side-grip: palm target + per-DOF wrist
                        # weights (HB free, wx/wz/wy locked). Other modes:
                        # wrist (Link1) target + scalar wrist weight.
                        if side_grip:
                            ik_target_body  = "Gripper_Link3_1"
                            ik_wrist_weight = (0.10, 3.0, 3.0, 3.0)
                        else:
                            ik_target_body  = "Gripper_Link1_1"
                            ik_wrist_weight = 5.0
                        # Screening uses fewer seeds (4 vs default 8) to cut
                        # main-thread blocking from ~250ms/candidate to ~120ms.
                        # Side-grip-push mode IGNORES this screened q anyway
                        # the executor re-solves IK from the virtual post-
                        # push chassis position. So a less-precise screening
                        # solution is still good enough for go/no-go.
                        q_pre, actual_pre_target = \
                            arm_bridge.solve_ik_with_z_lift(
                                pre_grasp_pos, n_seeds=4,
                                wrist_goal=wrist_goal,
                                wrist_weight=ik_wrist_weight,
                                target_body=ik_target_body)
                        h_diff = abs(float(q_pre[1]) - float(q_pre[0]))
                        # Side-grip uses arm tilt as a reach DOF (boom drops
                        # to bring palm to obj height; wrist_X compensates).
                        # Diagonal/top-down keeps the legacy tight cap.
                        h_diff_cap = (PICK_MAX_H_DIFF_SIDE if side_grip
                                      else PICK_MAX_H_DIFF)
                        if h_diff > h_diff_cap:
                            print(f"[PICK] Reject base candidate high tilt: "
                                  f"({cand[0]:.2f},{cand[1]:.2f}) "
                                  f"h2-h1={h_diff:.3f}m > {h_diff_cap:.2f}m "
                                  f"(mode={'SIDE' if side_grip else 'TOPDOWN'})")
                            continue
                        if float(q_pre[2]) < PICK_MIN_A1:
                            print(f"[PICK] Reject base candidate arm too retracted: "
                                  f"({cand[0]:.2f},{cand[1]:.2f}) "
                                  f"a1={float(q_pre[2]):.3f}m < {PICK_MIN_A1:.2f}m")
                            continue
                        # Skip the screening arm-path check in STRICT
                        # and FAST modes for side-grip.  The runtime
                        # planner re-plans HOME → PRE_HOVER with full
                        # timeout anyway; the screening check costs
                        # ~1 s per candidate and rejects only a tiny
                        # fraction.  Top-down / diagonal modes keep
                        # the check.
                        if side_grip and (FAST_PICKUP_MODE or STRICT_PICKUP_MODE):
                            path_wps = "screen-skip"
                        else:
                            path = arm_bridge.plan(
                                HOME_Q, q_pre,
                                timeout=PICK_VIRTUAL_PLAN_TIMEOUT)
                            if path is None:
                                print(f"[PICK] Reject base candidate no arm path: ({cand[0]:.2f},{cand[1]:.2f})")
                                continue
                            path_wps = len(path)
                        a1 = float(q_pre[2])
                        base_obj_dist = float(np.linalg.norm(
                            np.array(cand, dtype=float) - obj_world[:2]))
                        target_reach = max(0.0, base_obj_dist - GRIPPER_STANDOFF_XY)
                        expected_a1 = float(np.clip(
                            target_reach - PICK_A1_FIXED_REACH_OFFSET,
                            PICK_MIN_A1, 0.60))
                        a1_under = max(0.0, expected_a1 - a1)
                        a1_over = max(0.0, a1 - expected_a1)
                        # Palm-orientation tiebreaker: world XY direction
                        # from palm to thumb (finger A) dotted with the
                        # approach direction. Higher align → thumb on the
                        # object side, fingers on the opposite side (1+2
                        # gripper grips cleanly this way).
                        palm_align = 0.0
                        try:
                            palm_bid = mujoco.mj_name2id(
                                arm_bridge.model, mujoco.mjtObj.mjOBJ_BODY,
                                "Gripper_Link3_1")
                            thumb_bid = mujoco.mj_name2id(
                                arm_bridge.model, mujoco.mjtObj.mjOBJ_BODY,
                                "finger_a_link_1_1")
                            if palm_bid >= 0 and thumb_bid >= 0:
                                mujoco.mj_forward(
                                    arm_bridge.model, arm_bridge.planning_data)
                                palm_xy = arm_bridge.planning_data.xpos[palm_bid][:2]
                                thumb_xy = arm_bridge.planning_data.xpos[thumb_bid][:2]
                                tx = float(thumb_xy[0] - palm_xy[0])
                                ty = float(thumb_xy[1] - palm_xy[1])
                                tn = (tx * tx + ty * ty) ** 0.5
                                if tn > 1e-6:
                                    ax = obj_world[0] - cand[0]
                                    ay = obj_world[1] - cand[1]
                                    an = (ax * ax + ay * ay) ** 0.5
                                    if an > 1e-6:
                                        palm_align = (
                                            (tx / tn) * (ax / an)
                                          + (ty / tn) * (ay / an))
                        except Exception:
                            pass
                        # Predictive per-finger reach scoring. Read
                        # fingertip world XY for a/b/c at the FK pose,
                        # compute per-finger distances and the predicted
                        # carry_gap, and score candidates so the one
                        # most likely to grasp cleanly is tried first.
                        d_thumb_pred = 0.0
                        d_b_pred = 0.0
                        d_c_pred = 0.0
                        max_finger_far = 0.0
                        carry_gap_pred = 0.0
                        arm_obj_predicted_overlap = False
                        try:
                            _m = arm_bridge.model
                            _pd = arm_bridge.planning_data
                            _t_bid = mujoco.mj_name2id(
                                _m, mujoco.mjtObj.mjOBJ_BODY,
                                "finger_a_link_3_1")
                            _b_bid_f = mujoco.mj_name2id(
                                _m, mujoco.mjtObj.mjOBJ_BODY,
                                "finger_b_link_3_1")
                            _c_bid_f = mujoco.mj_name2id(
                                _m, mujoco.mjtObj.mjOBJ_BODY,
                                "finger_c_link_3_1")
                            _obj_xy = np.asarray(
                                obj_world[:2], dtype=float)
                            if _t_bid >= 0 and _b_bid_f >= 0 and _c_bid_f >= 0:
                                _t_xy = _pd.xpos[_t_bid][:2].copy()
                                _b_xy = _pd.xpos[_b_bid_f][:2].copy()
                                _c_xy = _pd.xpos[_c_bid_f][:2].copy()
                                d_thumb_pred = float(np.linalg.norm(
                                    _t_xy - _obj_xy))
                                d_b_pred = float(np.linalg.norm(
                                    _b_xy - _obj_xy))
                                d_c_pred = float(np.linalg.norm(
                                    _c_xy - _obj_xy))
                                max_finger_far = max(
                                    d_thumb_pred, d_b_pred, d_c_pred)
                                _bc_xy = 0.5 * (_b_xy + _c_xy)
                                _pinch_xy = 0.5 * (_t_xy + _bc_xy)
                                carry_gap_pred = float(np.linalg.norm(
                                    _pinch_xy - _obj_xy))
                        except Exception:
                            pass
                        # Predicted arm-vs-obj clearance: check planning
                        # contacts after FK. If any arm body (non-finger)
                        # is overlapping obj at the IK pose, reject
                        # this candidate — close would damage obj.
                        try:
                            obj_bid_pred = mujoco.mj_name2id(
                                arm_bridge.model, mujoco.mjtObj.mjOBJ_BODY,
                                f"pickup_obj_{obj_idx}")
                            if obj_bid_pred >= 0:
                                _pd2 = arm_bridge.planning_data
                                for _i in range(int(_pd2.ncon)):
                                    _c_obj = _pd2.contact[_i]
                                    _g1 = int(_c_obj.geom1)
                                    _g2 = int(_c_obj.geom2)
                                    _bod1 = int(arm_bridge.model.geom_bodyid[_g1])
                                    _bod2 = int(arm_bridge.model.geom_bodyid[_g2])
                                    if ((_bod1 == obj_bid_pred
                                         and _bod2 != _t_bid
                                         and _bod2 != _b_bid_f
                                         and _bod2 != _c_bid_f)
                                            or (_bod2 == obj_bid_pred
                                                and _bod1 != _t_bid
                                                and _bod1 != _b_bid_f
                                                and _bod1 != _c_bid_f)):
                                        if float(_c_obj.dist) < -0.005:
                                            arm_obj_predicted_overlap = True
                                            break
                        except Exception:
                            pass
                        if arm_obj_predicted_overlap:
                            print(f"[PICK] Reject base candidate predicted "
                                  f"arm-vs-obj clip: "
                                  f"({cand[0]:.2f},{cand[1]:.2f})")
                            continue
                        # (v3): nav-yaw asymmetric-reach
                        # filter and penalty. v2 's documented
                        # root-cause fix ("the real knob
                        # is nav-candidate yaw selection — pick chassis
                        # approach yaws where the default wz=-1.88
                        # already produces axis-aligned reach"). Predicts
                        # post-push fingertip positions per
                        # screening and rejects candidates where the
                        # bc-pair would land >8 cm farther from obj than
                        # the thumb (the chronic asymmetric-reach pattern
                        # that v3's lift physics fix can't compensate for).
                        bc_thumb_gap = max(d_b_pred, d_c_pred) - d_thumb_pred
                        ASYMMETRY_REJECT_M = 0.08
                        if bc_thumb_gap > ASYMMETRY_REJECT_M:
                            print(f"[PICK] Reject base candidate "
                                  f"asymmetric reach: "
                                  f"({cand[0]:.2f},{cand[1]:.2f}) "
                                  f"d_th={d_thumb_pred*100:.1f}cm "
                                  f"d_b={d_b_pred*100:.1f}cm "
                                  f"d_c={d_c_pred*100:.1f}cm "
                                  f"bc_thumb_gap={bc_thumb_gap*100:.1f}cm "
                                  f"> {ASYMMETRY_REJECT_M*100:.0f}cm — "
                                  f"bc-pair would land far behind thumb")
                            continue
                        # Composite quality score (lower = better):
                        # max_finger_far × 10 ← worst-side reach
                        # (the deciding factor)
                        # carry_gap_pred × 5 ← pinch alignment
                        # bc_thumb_gap × 15 ← asymmetry penalty
                        # h_diff × 2 ← arm tilt cost
                        # a1_under × 4 ← arm too short
                        # a1_over × 0.5 ← arm too long (mild)
                        # −0.02 × palm_align ← orientation tiebreak
                        ORIENT_BONUS_K = 0.02
                        ASYMMETRY_PENALTY_K = 15.0
                        quality = (max_finger_far * 10.0
                                   + carry_gap_pred * 5.0
                                   + bc_thumb_gap * ASYMMETRY_PENALTY_K
                                   + h_diff * 2.0
                                   + a1_under * 4.0
                                   + a1_over * 0.5
                                   - ORIENT_BONUS_K * palm_align)
                        _pass_tag = " (relaxed)" if _relaxed_attempted else ""
                        print(f"[PICK] Candidate OK{_pass_tag}: ({cand[0]:.2f},{cand[1]:.2f}) "
                              f"yaw={math.degrees(yaw):.1f}° path_wps={path_wps} "
                              f"base_obj={base_obj_dist:.3f}m "
                              f"exp_a1={expected_a1:.3f} "
                              f"h2-h1={h_diff:.3f} a1={a1:.3f} "
                              f"d_th={d_thumb_pred*100:.1f}cm "
                              f"d_b={d_b_pred*100:.1f}cm "
                              f"d_c={d_c_pred*100:.1f}cm "
                              f"max_far={max_finger_far*100:.1f}cm "
                              f"carry={carry_gap_pred*100:.1f}cm "
                              f"align={palm_align:+.2f} "
                              f"score={quality:.3f}")
                        valid.append((
                            quality, cand, h_diff, a1, base_obj_dist, expected_a1,
                            [float(v) for v in q_pre],
                            np.array(actual_pre_target, dtype=float),
                        ))
                        if len(valid) >= max_screen_successes:
                            print(f"[PICK] {len(valid)} feasible candidates found — stopping scan")
                            break
                    except Exception as e:
                        print(f"[PICK] Reject base candidate no virtual reach: "
                              f"({cand[0]:.2f},{cand[1]:.2f}) reason={e}")
                else:
                    print(f"[PICK] Reject base candidate in collision: ({cand[0]:.2f},{cand[1]:.2f})")
            # End of inner for loop — decide whether to retry with relaxed.
            if valid or _relaxed_attempted:
                break
            _relaxed_attempted = True
        # Restore the strict exempt set for any subsequent nav-path
        # validation (`filter_path`, etc.) so the chassis still avoids
        # unselected objs during transit.
        try:
            nav.validator.set_exempt_objects(strict_exempt)
        except Exception:
            pass
        if not valid:
            print("[PICK] No virtually feasible base candidates "
                  "(both strict and relaxed clearance exhausted); "
                  f"not moving robot  scan={time.time() - scan_t0:.2f}s")
            return []
        valid.sort(key=lambda item: item[0])
        ordered = [
            {
                "xy": item[1],
                "pre_q": item[6],
                "actual_pre_target": item[7],
            }
            for item in valid
        ]
        print("[PICK] Candidate order by arm quality: " +
              " → ".join(f"({cand[0]:.2f},{cand[1]:.2f}) "
                         f"d={base_obj_dist:.2f} "
                         f"hΔ={h_diff:.3f} a1={a1:.3f}/{expected_a1:.3f}"
                         for _, cand, h_diff, a1, base_obj_dist, expected_a1, _, _ in valid)
              + f"  scan={time.time() - scan_t0:.2f}s")
        return ordered

    def _start_pick_nav(candidate, reason):
        goal_xy = _candidate_xy(candidate)
        move_in_progress_flag[0] = True
        grasp_in_progress_flag[0] = False
        _pick_local_retry_used[0] = 0
        _pick_orbit_retry_used[0] = 0  #  reset orbit budget per candidate
        # Reset the per-candidate early-bail trackers so they compare
        # only within this candidate's attempts.
        _pick_prev_max_far[0] = None
        _pick_prev_failure_sig[0] = None
        # Only park the arm before base motion after an actual grasp/arm
        # attempt. A pure "navigation failed" retry has not moved the arm yet;
        # parking there looked like the arm was moving while the base was stuck.
        should_park_arm = (
            "grasp failed" in reason
            or "object moved" in reason
            or "base too close" in reason
        )
        if should_park_arm:
            with sim._target_lock:
                theta = float(sim.direct_arm_commands[3])
                sim.direct_arm_commands[0:4] = [CARRY_H1, CARRY_H2,
                                                CARRY_A1, theta]
            # Brief idle so the carry-pose command is committed before
            # base motion resumes.
            time.sleep(0.2)
        print(f"[PICK] Navigate candidate {_pick_candidate_idx[0]+1}/"
              f"{len(_pick_candidates[0])}: ({goal_xy[0]:.2f},{goal_xy[1]:.2f})  "
              f"reason={reason}")
        obj_idx = _current_obj_idx[0]
        obj_world = get_object_world_pos(sim.model, sim.data, obj_idx)
        final_yaw = pick_candidate_yaw(goal_xy, obj_world[:2])
        # Exempt the selected pick target from floor-object clearance —
        # the chassis is supposed to approach it to standoff distance.
        # All other floor objects remain hard obstacles so the nav path
        # can't graze them.
        try:
            sel_bid = mujoco.mj_name2id(
                sim.model, mujoco.mjtObj.mjOBJ_BODY, f"pickup_obj_{obj_idx}")
            if sel_bid >= 0:
                if _relaxed_clearance[0]:
                    # Fallback mode: only the selected target stays as
                    # a hard floor-object obstacle; all others exempted.
                    nav.validator.set_exempt_all_except(sel_bid)
                else:
                    # Default mode: only the selected target is exempted
                    # (the chassis is supposed to approach it to standoff).
                    nav.validator.set_exempt_objects([sel_bid])
        except Exception:
            pass
        nav.navigate_to(goal_xy, on_complete=_on_nav_complete,
                        goal_tolerance=PICK_NAV_GOAL_TOL,
                        final_yaw=final_yaw,
                        allow_goal_nudge=False)

    def _try_next_pick_candidate(reason):
        next_idx = _pick_candidate_idx[0] + 1
        obj_idx = _current_obj_idx[0]
        obj_world = get_object_world_pos(sim.model, sim.data, obj_idx)
        current_xy = np.array(sim.localization()[:2], dtype=float)

        # SAFETY: side-grip leaves the chassis ~0.52 m from obj. Any
        # rotation / nav from this close pose risks the chassis clipping
        # the obj. Back the chassis straight away from obj to the safe
        # radius before continuing. Uses sim.target_base (the chassis
        # PID controller's input) — same mechanism as fine-align and
        # the side-grip forward push.
        cur_dist_to_obj = float(np.linalg.norm(
            current_xy - np.asarray(obj_world[:2])))
        if cur_dist_to_obj < MIN_PICK_BASE_OBJ_DIST - 0.02:
            obj_xy_2d = np.asarray(obj_world[:2], dtype=float)
            if cur_dist_to_obj > 1e-6:
                away_unit = (current_xy - obj_xy_2d) / cur_dist_to_obj
            else:
                away_unit = np.array([1.0, 0.0])
            # Target: chassis at MIN_PICK_BASE_OBJ_DIST from obj, on the
            # same approach line.
            safe_xy = obj_xy_2d + away_unit * MIN_PICK_BASE_OBJ_DIST
            safe_yaw = math.atan2(obj_xy_2d[1] - safe_xy[1],
                                  obj_xy_2d[0] - safe_xy[0])
            print(f"[PICK] Safe retract before retry: chassis at "
                  f"{cur_dist_to_obj:.2f}m → {MIN_PICK_BASE_OBJ_DIST:.2f}m "
                  f"(avoid obj-vs-chassis clip during nav rotation)")
            with sim._target_lock:
                sim.target_base = np.array([float(safe_xy[0]),
                                            float(safe_xy[1]),
                                            float(safe_yaw)])
            t0 = time.time()
            while time.time() - t0 < PICK_FINE_ALIGN_TIMEOUT:
                cx, cy, _ = sim.localization()
                if math.hypot(cx - safe_xy[0],
                              cy - safe_xy[1]) <= PICK_FINE_ALIGN_DIST_TOL:
                    break
                time.sleep(0.05)
            fx, fy, _ = sim.localization()
            current_xy = np.array([fx, fy], dtype=float)
            actual_dist = float(np.linalg.norm(current_xy - obj_xy_2d))
            print(f"[PICK] Safe retract done: base-obj dist={actual_dist:.3f}m")

        while next_idx < len(_pick_candidates[0]):
            cand = _pick_candidates[0][next_idx]
            cand_xy = _candidate_xy(cand)
            selected_dist = point_segment_distance_xy(obj_world[:2], current_xy, cand_xy)
            if selected_dist >= PICK_RETRY_OBJ_PATH_CLEARANCE:
                break
            if selected_dist < PICK_RETRY_OBJ_PATH_CLEARANCE:
                print(f"[PICK] Skip candidate {next_idx+1}/{len(_pick_candidates[0])}: "
                      f"retry path would pass {selected_dist:.2f}m from Obj-{obj_idx} "
                      f"(< {PICK_RETRY_OBJ_PATH_CLEARANCE:.2f}m clearance)")
            next_idx += 1
        if next_idx >= len(_pick_candidates[0]):
            # Strict replan from current pose: only fires if at least one
            # candidate reached grasp (uniformly nav-failed cycles skip
            # straight to relaxed fallback below).
            if _pick_replan_count[0] < 1 and _any_grasp_attempted[0]:
                _pick_replan_count[0] += 1
                print(f"[PICK] No more base candidates after {reason}; "
                      "recomputing once from current robot pose")
                _pick_candidates[0] = _filtered_pick_candidates(obj_world)
                _pick_candidate_obj_xy[0] = obj_world[:2].copy()
                _pick_candidate_idx[0] = 0
                if _pick_candidates[0]:
                    _start_pick_nav(_pick_candidates[0][0],
                                    f"{reason}; replanned from current pose")
                    return True
            # Relaxed-clearance fallback: exempt non-selected floor objects
            # and skip segment-sampling. Rack/wall waypoint contacts are
            # still rejected.
            if _pick_replan_count[0] < 2 and not _relaxed_clearance[0]:
                _pick_replan_count[0] += 1
                _relaxed_clearance[0] = True
                nav.validator.skip_segment_validation = True
                print(f"[PICK] Strict clearance failed all candidates after "
                      f"{reason}; falling back to relaxed clearance "
                      "(only selected object guards floor-object distance; "
                      "rack/walls still hard obstacles; "
                      "segment-sampling disabled)")
                _pick_candidates[0] = _filtered_pick_candidates(obj_world)
                _pick_candidate_obj_xy[0] = obj_world[:2].copy()
                _pick_candidate_idx[0] = 0
                if _pick_candidates[0]:
                    _start_pick_nav(_pick_candidates[0][0],
                                    f"{reason}; relaxed-clearance fallback")
                    return True
            print(f"[PICK] No more base candidates after {reason}; aborting pick")
            move_in_progress_flag[0] = False
            grasp_in_progress_flag[0] = False
            _emit_cycle_metrics(
                "aborted_no_candidates",
                _current_obj_idx[0], _current_shelf_idx[0])
            return False
        _pick_candidate_idx[0] = next_idx
        _start_pick_nav(_pick_candidates[0][next_idx], reason)
        return True

    def _fine_align_base_for_pick(candidate, obj_world):
        """Small post-nav radial trim so the base starts the grasp at a useful range."""
        cand_xy = _candidate_xy(candidate)
        obj_xy = np.array(obj_world[:2], dtype=float)
        screened_dist = float(np.linalg.norm(cand_xy - obj_xy))
        pre_q = _candidate_pre_q(candidate)
        pre_h_diff = (abs(float(pre_q[1]) - float(pre_q[0]))
                      if pre_q is not None else float("inf"))
        preserve_screened_radius = (
            screened_dist > PICK_FINE_ALIGN_VISUAL_DIST
            and pre_h_diff <= PICK_FINE_ALIGN_PRESERVE_H_DIFF
        )
        if preserve_screened_radius:
            desired_dist = screened_dist
        else:
            desired_dist = max(PICK_FINE_ALIGN_MIN_SAFE_DIST,
                               min(screened_dist, PICK_FINE_ALIGN_VISUAL_DIST))
        loc = sim.localization()
        base_xy = np.array(loc[:2], dtype=float)
        cur_dist = float(np.linalg.norm(base_xy - obj_xy))
        if cur_dist < 1e-6:
            return False

        if cur_dist < PICK_FINE_ALIGN_MIN_SAFE_DIST - PICK_FINE_ALIGN_DIST_TOL:
            step = min(PICK_FINE_ALIGN_MIN_SAFE_DIST - cur_dist,
                       PICK_FINE_ALIGN_MAX_STEP)
            target_dist = cur_dist + step
            reason = "too close"
        elif cur_dist > desired_dist + PICK_FINE_ALIGN_DIST_TOL:
            step = min(cur_dist - desired_dist, PICK_FINE_ALIGN_MAX_STEP)
            target_dist = cur_dist - step
            reason = "too far"
        else:
            return False

        if target_dist < MIN_PICK_BASE_OBJ_DIST and target_dist < cur_dist:
            print(f"[PICK] Fine-align skipped: target base-object dist "
                  f"{target_dist:.2f}m < {MIN_PICK_BASE_OBJ_DIST:.2f}m")
            return False
        direction = (base_xy - obj_xy) / cur_dist
        target_xy = obj_xy + direction * target_dist
        target_yaw = pick_candidate_yaw(target_xy, obj_xy)
        nav.validator.sync(sim.data)
        if not nav.validator.is_valid(float(target_xy[0]), float(target_xy[1]),
                                      yaw=target_yaw):
            print(f"[PICK] Fine-align skipped: target "
                  f"({target_xy[0]:.2f},{target_xy[1]:.2f}) invalid")
            return False
        print(f"[PICK] Fine-align base: dist {cur_dist:.3f}m → "
              f"{target_dist:.3f}m ({reason}; screened {screened_dist:.3f}m, "
              f"target {desired_dist:.3f}m, hΔ={pre_h_diff:.3f})")
        with sim._target_lock:
            sim.target_base = np.array([target_xy[0], target_xy[1], target_yaw])
        t0 = time.time()
        while time.time() - t0 < PICK_FINE_ALIGN_TIMEOUT:
            cx, cy, _ = sim.localization()
            if math.hypot(cx - target_xy[0], cy - target_xy[1]) <= PICK_FINE_ALIGN_DIST_TOL:
                break
            time.sleep(0.05)
        fx, fy, _ = sim.localization()
        final_dist = float(np.linalg.norm(np.array([fx, fy]) - obj_xy))
        print(f"[PICK] Fine-align done: base-object dist={final_dist:.3f}m")
        return True

    def _side_grip_approach_push(obj_world):
        """Drive the chassis straight forward (toward obj) to
        `SIDE_FORWARD_PUSH_TARGET` for side-grip mode.  The nav planner
        keeps its full chassis-safety margins (stops at ~0.78 m, fine-
        align trims to ~0.70 m); this step closes the remaining gap so
        the arm's horizontal reach lands on the obj.

        No OMPL needed — the obj is directly in front of the chassis
        at this stage and the corridor was already validated clean by
        nav + fine-align.  Uses the same nav-validator approach as
        fine-align: pose-driven setpoint + timed wait for the chassis
        to converge.
        """
        obj_xy = np.array(obj_world[:2], dtype=float)
        loc = sim.localization()
        base_xy = np.array(loc[:2], dtype=float)
        cur_dist = float(np.linalg.norm(base_xy - obj_xy))
        if cur_dist < 1e-6:
            return False
        if cur_dist <= SIDE_FORWARD_PUSH_TARGET + PICK_FINE_ALIGN_DIST_TOL:
            print(f"[PICK] Side-grip push skipped: already at "
                  f"{cur_dist:.3f}m ≤ target {SIDE_FORWARD_PUSH_TARGET:.2f}m")
            return False
        direction = (base_xy - obj_xy) / cur_dist  # unit vector AWAY from obj
        target_xy = obj_xy + direction * SIDE_FORWARD_PUSH_TARGET
        target_yaw = pick_candidate_yaw(target_xy, obj_xy)
        nav.validator.sync(sim.data)
        if not nav.validator.is_valid(float(target_xy[0]), float(target_xy[1]),
                                      yaw=float(target_yaw)):
            print(f"[PICK] Side-grip push aborted: target "
                  f"({target_xy[0]:.2f},{target_xy[1]:.2f}) fails validator")
            return False
        print(f"[PICK] Side-grip approach push: "
              f"{cur_dist:.3f}m → {SIDE_FORWARD_PUSH_TARGET:.2f}m "
              f"(chassis forward toward obj)")
        # Use sim.target_base — the attribute the chassis PID controller
        # actually reads. (sim.target_pos is a vestigial attribute the
        # sim does NOT consume.) Matches the fine-align convention.
        with sim._target_lock:
            sim.target_base = np.array([float(target_xy[0]),
                                        float(target_xy[1]),
                                        float(target_yaw)])
        t0 = time.time()
        last_diag = t0
        while time.time() - t0 < PICK_FINE_ALIGN_TIMEOUT:
            cx, cy, _ = sim.localization()
            if math.hypot(cx - target_xy[0],
                          cy - target_xy[1]) <= PICK_FINE_ALIGN_DIST_TOL:
                break
            # Diag every 0.3s so we can see whether the chassis is
            # making progress or stuck against something.
            now = time.time()
            if now - last_diag >= 0.3:
                last_diag = now
                cur_d = float(np.linalg.norm(np.array([cx, cy]) - obj_xy))
                print(f"[PICK]   push progress: t={now-t0:.1f}s "
                      f"base=({cx:.3f},{cy:.3f}) dist_to_obj={cur_d:.3f}m")
            time.sleep(0.05)
        fx, fy, _ = sim.localization()
        final_dist = float(np.linalg.norm(np.array([fx, fy]) - obj_xy))
        moved = math.hypot(fx - base_xy[0], fy - base_xy[1])
        print(f"[PICK] Side-grip push done: base-object dist={final_dist:.3f}m "
              f"(moved {moved*100:.1f}cm in {time.time()-t0:.1f}s)")
        return True

    def _local_radial_retry_after_grasp_fail(obj_world):
        """Closed-loop base correction after a pre-close gate rejection.

        Translates the base by the gripper-vs-IK-target residual error
        (chassis yaw held fixed so the gripper moves by the same vector).
        Up to PICK_LOCAL_RETRY_MAX_ATTEMPTS attempts per cycle.
        """
        attempts = _pick_local_retry_used[0]
        # STRICT mode caps retries at 1 to avoid arm/wrist drift
        # accumulation that degrades the gripper pose by retry #2+.
        _max_attempts = (PICK_LOCAL_RETRY_MAX_ATTEMPTS_STRICT
                         if STRICT_PICKUP_MODE
                         else PICK_LOCAL_RETRY_MAX_ATTEMPTS)
        if attempts >= _max_attempts:
            print(f"[PICK] Local retry exhausted "
                  f"({attempts}/{_max_attempts} attempts"
                  f"{' STRICT' if STRICT_PICKUP_MODE else ''})")
            return False
        info = getattr(grasp_exec, 'last_grasp_failure_info', None)
        if info is None:
            print("[PICK] Local retry skipped: no grasp_failure_info "
                  "(failure was not from pre-close gate)")
            return False

        # In STRICT mode, skip local retry entirely when the rejection
        # cause was asymmetric reach (max_far > 9 cm side-reach
        # threshold).  Local retry is just a chassis translation along
        # the current approach axis; it cannot fix orientation
        # mismatches between thumb-bc axis and obj — the same
        # gripper-frame asymmetry persists at any chassis XY.  Skipping
        # straight to the next nav candidate (different approach yaw →
        # different gripper world orientation) is the only path that
        # can resolve this; local retry just delays the cycle by ~5 s.
        SIDE_REACH_THRESHOLD_M = 0.09   # matches grasp_executor.SIDE_FINGER_PRECLOSE_REACH
        d_thumb_info = info.get('d_thumb', None)
        d_bc_info    = info.get('d_bc', None)
        max_far_info = info.get('max_far', None)
        if (STRICT_PICKUP_MODE
                and max_far_info is not None
                and max_far_info > SIDE_REACH_THRESHOLD_M):
            d_thumb_str = (f"{d_thumb_info*100:.1f}cm"
                           if d_thumb_info is not None else "?")
            d_bc_str    = (f"{d_bc_info*100:.1f}cm"
                           if d_bc_info is not None else "?")
            print(f"[PICK] Local retry SKIPPED (asymmetric reach in "
                  f"STRICT): d_thumb={d_thumb_str} d_bc={d_bc_str} "
                  f"max_far={max_far_info*100:.1f}cm > "
                  f"{SIDE_REACH_THRESHOLD_M*100:.0f}cm — chassis "
                  f"translation cannot fix orientation mismatches; "
                  f"jumping to next nav candidate for different approach yaw")
            _pick_prev_max_far[0] = None
            _pick_prev_failure_sig[0] = None
            return False

        # Early-bail when a local retry makes reach worse than the
        # previous attempt — same-yaw retries can flip the asymmetry
        # rather than fix it. Tolerate a +1 cm noise margin; bail only
        # when max_far definitely worsened. Falls through to the next
        # nav candidate (different approach yaw, different wrist result).
        cur_max_far = info.get('max_far', None)
        prev_max_far = _pick_prev_max_far[0]
        if (cur_max_far is not None and prev_max_far is not None
                and cur_max_far > prev_max_far + 0.01):
            print(f"[PICK] Local retry EARLY-BAIL: max_far worsened "
                  f"{prev_max_far*100:.1f}cm → {cur_max_far*100:.1f}cm "
                  f"(>1cm regress) — same-yaw retries won't escape "
                  f"this asymmetry pattern; jumping to next nav "
                  f"candidate for a different approach yaw")
            _pick_prev_max_far[0] = None
            _pick_prev_failure_sig[0] = None
            return False
        _pick_prev_max_far[0] = cur_max_far

        # Same-pattern early-bail: when two consecutive retries fail
        # with effectively identical geometry (same arm-clip state and
        # both finger distances within 2 cm of the prior attempt), the
        # local retry is not making progress — declare the candidate
        # dead and jump to the next.
        cur_sig = (
            int(info.get('arm_obj_ncon', 0) > 0),
            round(info.get('d_thumb', 0.0) * 50.0),   # 2 cm buckets
            round(info.get('d_bc',    0.0) * 50.0),
        )
        prev_sig = _pick_prev_failure_sig[0]
        if prev_sig is not None and cur_sig == prev_sig:
            print(f"[PICK] Local retry EARLY-BAIL: same failure pattern "
                  f"repeats (arm_clip={cur_sig[0]} d_thumb≈"
                  f"{info.get('d_thumb', 0.0)*100:.1f}cm d_bc≈"
                  f"{info.get('d_bc', 0.0)*100:.1f}cm) — local nudges "
                  f"cannot break this geometry; jumping to next nav "
                  f"candidate")
            _pick_prev_max_far[0] = None
            _pick_prev_failure_sig[0] = None
            return False
        _pick_prev_failure_sig[0] = cur_sig

        loc = sim.localization()
        base_xy = np.array(loc[:2], dtype=float)
        cur_yaw = float(loc[2])
        obj_xy  = np.array(obj_world[:2], dtype=float)
        gripper_actual_xy = np.array(info['gripper_xy'], dtype=float)
        ik_target_xy      = np.array(info['ik_target_xy'], dtype=float)

        # Residual = obj_xy - carry_anchor_xy (fingertip centroid).
        # Translating the base by this delta moves the carry_anchor
        # onto the cylinder because base→arm→carry_anchor is rigid in
        # the base frame when yaw is held fixed.
        carry_xy = info.get('carry_xy')
        if carry_xy is not None:
            carry_xy = np.array(carry_xy, dtype=float)
            err_vec = obj_xy - carry_xy
            err_source = "carry_anchor→obj"
        else:
            # Fallback for failure modes that didn't capture carry_xy.
            err_vec = ik_target_xy - gripper_actual_xy
            err_source = "ik_target→Link1"
        err_mag = float(np.linalg.norm(err_vec))
        if err_mag < PICK_LOCAL_RETRY_MIN_ERR:
            print(f"[PICK] Local retry skipped: residual {err_mag*100:.1f}cm "
                  f"below {PICK_LOCAL_RETRY_MIN_ERR*100:.1f}cm threshold "
                  f"(metric: {err_source})")
            return False
        if err_mag > PICK_LOCAL_RETRY_MAX_ERR:
            err_vec = err_vec * (PICK_LOCAL_RETRY_MAX_ERR / err_mag)
            err_mag = PICK_LOCAL_RETRY_MAX_ERR
            print(f"[PICK] Local retry: residual capped at "
                  f"{PICK_LOCAL_RETRY_MAX_ERR*100:.1f}cm")

        nav.validator.sync(sim.data)
        # During a pick the target obj is intentionally close to the
        # chassis — exempt it from the validator's floor-object
        # clearance check so candidate poses inside the nav-clearance
        # band (but on the way TO obj) aren't rejected. Restore after.
        obj_idx_for_exempt = _current_obj_idx[0]
        obj_bid_for_exempt = mujoco.mj_name2id(
            sim.model, mujoco.mjtObj.mjOBJ_BODY,
            f"pickup_obj_{obj_idx_for_exempt}")
        prev_exempt = set(nav.validator.exempt_obj_ids)
        if obj_bid_for_exempt >= 0:
            new_exempt = prev_exempt | {int(obj_bid_for_exempt)}
            nav.validator.set_exempt_objects(new_exempt)

        target_xy = None
        target_yaw = cur_yaw   # KEEP CURRENT YAW — rigid arm in base frame
        chosen_scale = None
        for scale in PICK_LOCAL_RETRY_SCALES:
            cand_xy = base_xy + err_vec * scale
            # Use the during-pick min-dist (PICK_LOCAL_RETRY_MIN_DIST_PICK
            # = 0.40 m), not the nav-distance min (0.70 m) which would
            # reject every local-retry candidate that moves toward obj
            # (chassis is at 0.64m post-push, already inside the 0.70m
            # nav clearance). The during-pick min is calibrated to
            # the side-grip push target minus a small reach buffer.
            cand_dist = float(np.linalg.norm(cand_xy - obj_xy))
            if cand_dist < PICK_LOCAL_RETRY_MIN_DIST_PICK:
                continue
            if nav.validator.is_valid(float(cand_xy[0]), float(cand_xy[1]),
                                      yaw=cur_yaw):
                target_xy = cand_xy
                chosen_scale = scale
                break

        # Restore the validator's exempt set so subsequent nav planning
        # (after a successful retry or after this retry's eventual
        # failure) treats obstacles correctly.
        nav.validator.set_exempt_objects(prev_exempt)

        if target_xy is None:
            print(f"[PICK] Local retry skipped: no valid pose at any scale "
                  f"of err={err_mag*100:.1f}cm "
                  f"(dir=({err_vec[0]:+.3f},{err_vec[1]:+.3f}))  "
                  f"[min_dist={PICK_LOCAL_RETRY_MIN_DIST_PICK}m, "
                  f"target exempt={obj_bid_for_exempt>=0}]")
            return False

        applied = err_mag * chosen_scale
        print(f"[PICK] Local retry #{attempts + 1}/"
              f"{PICK_LOCAL_RETRY_MAX_ATTEMPTS}: residual "
              f"{err_mag*100:.1f}cm ({err_source})  "
              f"applied {applied*100:.1f}cm "
              f"({chosen_scale*100:.0f}%)  "
              f"base ({base_xy[0]:.2f},{base_xy[1]:.2f}) → "
              f"({target_xy[0]:.2f},{target_xy[1]:.2f})  yaw fixed  "
              f"[executor closed-loop will move chassis]")
        _pick_local_retry_used[0] = attempts + 1
        # Chassis movement on local retry is handled exclusively by
        # the executor's closed-loop alignment block [5.47], which
        # observes carry_anchor and snaps with contact + geometric
        # guards.  The target_xy VALIDATION above is kept so we don't
        # local-retry when the residual would land in a blocked zone.
        grasp_in_progress_flag[0] = True
        _any_grasp_attempted[0] = True
        grasp_exec.pick(
            _current_obj_idx[0], obj_world,
            on_complete=_on_grasp_complete,
            is_local_retry=True)
        return True

    # Yaw-orbit retry — chassis orbits around obj at the
    # SAME arm height (GRASP_Q) to try a different approach angle.
    # Fires when:
    # The radial (XY-translation) local retry has already been skipped
    # or exhausted, AND
    # The grasp failure reason was asymmetric reach (orientation
    # mismatch — chassis translation can't fix it; chassis ROTATION
    # around obj changes the gripper orientation in world frame).
    # Skips the full retract+nav+lift+descend cycle (~12 s) — orbit
    # move + grasp retry is ~4-5 s.
    PICK_ORBIT_RETRY_MAX_ATTEMPTS = 5  # yaws sampled per candidate
                                         # before falling through to
                                         # the next nav candidate.
    # Orbit yaws: ±8°..±40° in 8° increments.  Cap at ±40° keeps the
    # max robot swing small; finer increments near zero make it
    # likely to find a workable yaw before a larger rotation.
    PICK_ORBIT_YAW_DELTAS_RAD = (
        +0.140, -0.140,   # ±8°
        +0.279, -0.279,   # ±16°
        +0.419, -0.419,   # ±24°
        +0.559, -0.559,   # ±32°
        +0.698, -0.698,   # ±40°
    )

    def _orbit_retry_after_grasp_fail(obj_world):
        if not STRICT_PICKUP_MODE:
            return False
        info = getattr(grasp_exec, 'last_grasp_failure_info', None)
        if info is None:
            return False
        max_far = info.get('max_far', None)
        # Only fire on asymmetric reach (same signal that gates local
        # retry). Symmetric-but-far cases already handled by closed-
        # loop alignment inside _pick_run.
        if max_far is None or max_far <= 0.09:
            return False
        used = _pick_orbit_retry_used[0]
        if used >= PICK_ORBIT_RETRY_MAX_ATTEMPTS:
            print(f"[PICK] Orbit retry exhausted "
                  f"({used}/{PICK_ORBIT_RETRY_MAX_ATTEMPTS} attempts)")
            return False

        loc = sim.localization()
        cur_xy = np.array([loc[0], loc[1]], dtype=float)
        obj_xy = np.array(obj_world[:2], dtype=float)
        offset = cur_xy - obj_xy
        d_to_obj = float(np.linalg.norm(offset))
        if d_to_obj < 0.30:
            print(f"[PICK] Orbit retry skipped: chassis already too "
                  f"close to obj ({d_to_obj:.2f}m) — would clip during "
                  f"rotation")
            return False
        cur_angle = float(math.atan2(offset[1], offset[0]))
        nav.validator.sync(sim.data)
        # Exempt the target obj from the validator's floor-clearance
        # check during orbit pose validation (chassis sits intentionally
        # close to it).
        obj_idx_orbit = _current_obj_idx[0]
        obj_bid_orbit = mujoco.mj_name2id(
            sim.model, mujoco.mjtObj.mjOBJ_BODY,
            f"pickup_obj_{obj_idx_orbit}")
        prev_exempt = set(nav.validator.exempt_obj_ids)
        if obj_bid_orbit >= 0:
            nav.validator.set_exempt_objects(
                prev_exempt | {int(obj_bid_orbit)})

        # Skip the yaw deltas we've already used this candidate so
        # consecutive orbit attempts try fresh angles.
        delta_idx_start = used * 2   # advance 2 deltas per attempt
        target_orbit_xy = None
        target_orbit_yaw = None
        chosen_delta = None
        for di in range(delta_idx_start,
                        len(PICK_ORBIT_YAW_DELTAS_RAD)):
            delta = PICK_ORBIT_YAW_DELTAS_RAD[di]
            new_angle = cur_angle + delta
            new_xy = obj_xy + d_to_obj * np.array(
                [math.cos(new_angle), math.sin(new_angle)])
            new_yaw = math.atan2(
                float(obj_xy[1] - new_xy[1]),
                float(obj_xy[0] - new_xy[0]))
            # Reject orbit poses inside the nav-keepout regions.
            in_keepout = False
            for (x0, x1, y0, y1) in OBSTACLE_RECTS:
                if ((x0 - NAV_ROBOT_RADIUS) <= new_xy[0]
                        <= (x1 + NAV_ROBOT_RADIUS)
                        and (y0 - NAV_ROBOT_RADIUS) <= new_xy[1]
                        <= (y1 + NAV_ROBOT_RADIUS)):
                    in_keepout = True
                    break
            if in_keepout:
                continue
            # Validate chassis pose at new yaw.
            if not nav.validator.is_valid(float(new_xy[0]),
                                          float(new_xy[1]),
                                          yaw=new_yaw):
                continue
            target_orbit_xy = new_xy
            target_orbit_yaw = new_yaw
            chosen_delta = delta
            break

        nav.validator.set_exempt_objects(prev_exempt)

        if target_orbit_xy is None:
            print(f"[PICK] Orbit retry skipped: no valid orbit pose "
                  f"at any tried yaw delta (chassis blocked by "
                  f"obstacles around obj)")
            return False

        _pick_orbit_retry_used[0] = used + 1
        print(f"[PICK] Orbit retry #{used + 1}/"
              f"{PICK_ORBIT_RETRY_MAX_ATTEMPTS}: rotating "
              f"{math.degrees(chosen_delta):+.0f}° around obj "
              f"(arm STAYS at GRASP_Q, no lift): "
              f"chassis ({cur_xy[0]:.2f},{cur_xy[1]:.2f}) → "
              f"({target_orbit_xy[0]:.2f},{target_orbit_xy[1]:.2f})  "
              f"yaw {math.degrees(loc[2]):+.0f}° → "
              f"{math.degrees(target_orbit_yaw):+.0f}°")
        ORBIT_DIRECT_ATTEMPTS_BUDGET = 2   # first N attempts use direct rotation
        use_direct_rotation = used <= ORBIT_DIRECT_ATTEMPTS_BUDGET
        ORBIT_PHASE_TIMEOUT = 2.0

        if use_direct_rotation:
            print(f"[PICK]   Orbit mode: DIRECT rotation (attempt "
                  f"{used}/{PICK_ORBIT_RETRY_MAX_ATTEMPTS}, "
                  f"≤{ORBIT_DIRECT_ATTEMPTS_BUDGET} direct budget) — "
                  f"chassis stays at current radius, arm rotates "
                  f"rigidly around obj; preserves alignment")
            # Single-phase: move chassis directly to target_orbit_xy
            # at target_orbit_yaw. Chassis stays at d_to_obj radius
            # because target_orbit_xy was computed at d_to_obj.
            with sim._target_lock:
                sim.target_base = np.array([float(target_orbit_xy[0]),
                                             float(target_orbit_xy[1]),
                                             float(target_orbit_yaw)])
            t0 = time.time()
            while time.time() - t0 < ORBIT_PHASE_TIMEOUT * 1.5:
                cx, cy, _ = sim.localization()
                if math.hypot(cx - target_orbit_xy[0],
                              cy - target_orbit_xy[1]) <= 0.04:
                    break
                time.sleep(0.05)
            time.sleep(0.2)
            fx, fy, fyaw = sim.localization()
            print(f"[PICK] Orbit retry (DIRECT): chassis arrived at "
                  f"({fx:.2f},{fy:.2f}) yaw={math.degrees(fyaw):+.0f}° "
                  f"(target ({target_orbit_xy[0]:.2f},"
                  f"{target_orbit_xy[1]:.2f}) yaw="
                  f"{math.degrees(target_orbit_yaw):+.0f}°)")
        else:
            print(f"[PICK]   Orbit mode: 3-PHASE retract-rotate-approach "
                  f"(attempt {used}/{PICK_ORBIT_RETRY_MAX_ATTEMPTS}, "
                  f"direct budget {ORBIT_DIRECT_ATTEMPTS_BUDGET} "
                  f"exhausted) — safety fallback")
            # Three-phase orbit move so the arm doesn't
            # sweep through obj during rotation.
            # PHASE 1 (retract): Back chassis OUT along the current
            # approach direction to a safe radius (0.8 m). This pulls
            # the palm/arm clear of obj before rotation.
            # PHASE 2 (rotate-at-safe-radius): Move to a point on the
            # orbit circle at the same safe radius but at the new yaw.
            # Arm rotates around obj at safe distance — no sweep contact.
            # PHASE 3 (re-approach): Push chassis forward toward obj
            # along the NEW approach direction until back at orbit
            # distance. Arm palm now faces obj from new angle.
            SAFE_ORBIT_RADIUS = 0.80   # m — chassis-to-obj during rotation
            # Phase 1: retract along current approach (radially outward
            # from obj along current chassis-to-obj vector).
            if d_to_obj > 1e-6:
                # offset = cur_xy - obj_xy; +offset/d points OUT from obj
                away_unit = offset / d_to_obj
            else:
                away_unit = np.array([1.0, 0.0])
            retract_xy = obj_xy + SAFE_ORBIT_RADIUS * away_unit
            retract_yaw = float(loc[2])   # keep current yaw during retract
            print(f"[PICK]   Orbit phase 1 (retract): chassis "
                  f"({cur_xy[0]:.2f},{cur_xy[1]:.2f}) → "
                  f"({retract_xy[0]:.2f},{retract_xy[1]:.2f})  "
                  f"d_to_obj {d_to_obj:.2f}m → {SAFE_ORBIT_RADIUS:.2f}m")
            move_in_progress_flag[0] = True
            with sim._target_lock:
                sim.target_base = np.array([float(retract_xy[0]),
                                             float(retract_xy[1]),
                                             retract_yaw])
            t0 = time.time()
            while time.time() - t0 < ORBIT_PHASE_TIMEOUT:
                cx, cy, _ = sim.localization()
                if math.hypot(cx - retract_xy[0],
                              cy - retract_xy[1]) <= 0.05:
                    break
                time.sleep(0.05)
            # Phase 2: orbit point at SAFE_ORBIT_RADIUS at new angle.
            orbit_xy_safe = obj_xy + SAFE_ORBIT_RADIUS * np.array(
                [math.cos(cur_angle + chosen_delta),
                 math.sin(cur_angle + chosen_delta)])
            orbit_yaw_safe = math.atan2(
                float(obj_xy[1] - orbit_xy_safe[1]),
                float(obj_xy[0] - orbit_xy_safe[0]))
            print(f"[PICK]   Orbit phase 2 (rotate at safe radius): "
                  f"chassis "
                  f"({retract_xy[0]:.2f},{retract_xy[1]:.2f}) → "
                  f"({orbit_xy_safe[0]:.2f},{orbit_xy_safe[1]:.2f})  "
                  f"yaw {math.degrees(retract_yaw):+.0f}° → "
                  f"{math.degrees(orbit_yaw_safe):+.0f}°")
            with sim._target_lock:
                sim.target_base = np.array([float(orbit_xy_safe[0]),
                                             float(orbit_xy_safe[1]),
                                             float(orbit_yaw_safe)])
            t0 = time.time()
            while time.time() - t0 < ORBIT_PHASE_TIMEOUT:
                cx, cy, _ = sim.localization()
                if math.hypot(cx - orbit_xy_safe[0],
                              cy - orbit_xy_safe[1]) <= 0.05:
                    break
                time.sleep(0.05)
            # Phase 3: approach toward obj along the new direction back
            # to orbit-target distance.
            print(f"[PICK]   Orbit phase 3 (re-approach): chassis "
                  f"({orbit_xy_safe[0]:.2f},{orbit_xy_safe[1]:.2f}) → "
                  f"({target_orbit_xy[0]:.2f},{target_orbit_xy[1]:.2f})  "
                  f"d_to_obj {SAFE_ORBIT_RADIUS:.2f}m → {d_to_obj:.2f}m")
            with sim._target_lock:
                sim.target_base = np.array([float(target_orbit_xy[0]),
                                             float(target_orbit_xy[1]),
                                             float(target_orbit_yaw)])
            t0 = time.time()
            while time.time() - t0 < ORBIT_PHASE_TIMEOUT:
                cx, cy, _ = sim.localization()
                if math.hypot(cx - target_orbit_xy[0],
                              cy - target_orbit_xy[1]) <= 0.04:
                    break
                time.sleep(0.05)
            # Brief settle so the arm tracks the new chassis yaw
            time.sleep(0.2)
            fx, fy, fyaw = sim.localization()
            print(f"[PICK] Orbit retry (3-PHASE): chassis arrived at "
                  f"({fx:.2f},{fy:.2f}) yaw={math.degrees(fyaw):+.0f}° "
                  f"(target ({target_orbit_xy[0]:.2f},"
                  f"{target_orbit_xy[1]:.2f}) yaw="
                  f"{math.degrees(target_orbit_yaw):+.0f}°)")
        move_in_progress_flag[0] = False
        grasp_in_progress_flag[0] = True
        _any_grasp_attempted[0] = True
        # Reuse the local-retry executor path: skips arm motion,
        # runs closed-loop alignment + gate from the new chassis pose.
        grasp_exec.pick(
            _current_obj_idx[0], obj_world,
            on_complete=_on_grasp_complete,
            is_local_retry=True)
        return True

    def _choose_approx_drop_standoff(shelf_pos):
        """Pick a validated aisle pose near the assigned slot column."""
        loc = sim.localization()
        sx = float(shelf_pos[0])
        for y in APPROX_DROP_AISLE_Y_OPTIONS:
            for dx in APPROX_DROP_X_OFFSETS:
                xy = np.array([
                    float(np.clip(sx + dx, PICK_GOAL_X_RANGE[0],
                                  PICK_GOAL_X_RANGE[1])),
                    float(y),
                ], dtype=float)
                yaw = math.atan2(float(xy[1] - loc[1]),
                                 float(xy[0] - loc[0]))
                nav.validator.sync(sim.data)
                if nav.validator.is_valid(float(xy[0]), float(xy[1]), yaw=yaw):
                    if abs(xy[0] - sx) > 1e-6 or abs(y - APPROX_DROP_AISLE_Y) > 1e-6:
                        print(f"[NAV] Adjusted approximate drop aisle point "
                              f"near slot column: ({sx:.2f},{APPROX_DROP_AISLE_Y:.2f}) "
                              f"→ ({xy[0]:.2f},{xy[1]:.2f})")
                    return (float(xy[0]), float(xy[1]))
        print("[NAV] No validated approximate drop aisle alternative found; "
              "using nominal slot-column point")
        return (sx, APPROX_DROP_AISLE_Y)

    def _on_nav_complete(success):
        print(f"[GUI] Navigation complete — success={success}")
        if not success:
            if _try_next_pick_candidate("navigation failed"):
                return
            move_in_progress_flag[0] = False
            print("[GUI] Navigation failed for all base candidates — skipping grasp")
            return

        obj_idx   = _current_obj_idx[0]
        obj_world = get_object_world_pos(sim.model, sim.data, obj_idx)

        # Bail early if obj has rolled / been knocked out of the
        # pickable floor zone since nav started.  Without this the
        # chassis would chase the obj across the world before the
        # eventual zone-reject — instead we abort the cycle the
        # moment we see the post-nav obj position is unreachable so
        # the auto-runner can advance to the next attempt.
        if (not _is_in_spawn_zone(float(obj_world[0]),
                                  float(obj_world[1]), margin=0.25)
                or _is_inside_spawn_keepout(float(obj_world[0]),
                                            float(obj_world[1]))):
            print(f"[PICK] obj_{obj_idx} drifted outside pickable zone "
                  f"after nav (now at "
                  f"({obj_world[0]:.2f},{obj_world[1]:.2f})) — "
                  f"aborting cycle before chassis chase")
            move_in_progress_flag[0] = False
            _emit_cycle_metrics("obj_out_of_bounds",
                                obj_idx, _current_shelf_idx[0])
            return

        candidate = _pick_candidates[0][_pick_candidate_idx[0]]
        fine_aligned = _fine_align_base_for_pick(candidate, obj_world)

        # Side-grip mode: compute the "virtual base position" the IK
        # should plan for. The chassis stays at the safe nav distance
        # for now (~0.70 m). AFTER the arm descent completes, a
        # forward chassis push (run by GraspExecutor between [5] descent
        # and [6] close) translates the arm into final position.
        # Planning the IK from this future-chassis position lets the arm
        # pick a comfortable low-tilt pose now (instead of reaching
        # across 22 cm with h2-h1≈0.36 and clipping the chassis).
        sel_bid_pickup = -1
        try:
            sel_bid_pickup = mujoco.mj_name2id(
                sim.model, mujoco.mjtObj.mjOBJ_BODY,
                f"pickup_obj_{obj_idx}")
        except Exception:
            pass
        wrist_goal_pickup = (compute_wrist_goal_for_obj(sim.model, sel_bid_pickup)
                             if sel_bid_pickup >= 0 else None)
        side_grip_pickup = (abs(float(wrist_goal_pickup[0])) < 0.20
                            if wrist_goal_pickup is not None else False)

        move_in_progress_flag[0] = False
        # Chassis-safety floor: only enforced for top-down/diagonal —
        # side-grip is at the safe nav distance now; the close-approach
        # push happens AFTER arm descent.
        base_xy = np.array(sim.localization()[:2], dtype=float)
        base_obj_dist = float(np.linalg.norm(base_xy - obj_world[:2]))
        if not side_grip_pickup and base_obj_dist < MIN_PICK_BASE_OBJ_DIST:
            print(f"[PICK] Reject reached pose: base-object dist={base_obj_dist:.2f}m "
                  f"< {MIN_PICK_BASE_OBJ_DIST:.2f}m (chassis-safety floor); "
                  f"Retrying next candidate.")
            if _try_next_pick_candidate("base too close to object"):
                return
            print("[GUI] No safe pick candidate left after base-object distance check")
            return
        grasp_in_progress_flag[0] = True
        _any_grasp_attempted[0] = True

        # Hand off to GraspExecutor (open -> approach -> descent ->
        # OPTIONAL chassis push -> close -> pin-to-gripper).
        # side_grip flag tells the executor to plan IK from the
        # post-push virtual base AND to run the chassis push after
        # descent.
        if fine_aligned:
            print("[PICK] Fine-align changed base pose; recomputing IK "
                  "from actual pose")
        grasp_exec.pick(
            obj_idx, obj_world, on_complete=_on_grasp_complete,
            pre_grasp_q=None if fine_aligned else _candidate_pre_q(candidate),
            pre_grasp_actual_target=None if fine_aligned
            else _candidate_actual_pre_target(candidate),
            side_grip_push_target=(
                SIDE_FORWARD_PUSH_TARGET if side_grip_pickup else None))

    def _on_grasp_complete(success):
        print(f"[GUI] Grasp complete — success={success}")
        if not success:
            grasp_in_progress_flag[0] = False
            obj_idx = _current_obj_idx[0]
            obj_world = get_object_world_pos(sim.model, sim.data, obj_idx)
            prev_xy = _pick_candidate_obj_xy[0]
            moved = (prev_xy is not None and
                     float(np.linalg.norm(obj_world[:2] - prev_xy)) > 0.15)
            if moved and _pick_replan_count[0] < 2:
                _pick_replan_count[0] += 1
                print(f"[PICK] Obj-{obj_idx} moved during failed grasp "
                      f"(old=({prev_xy[0]:.2f},{prev_xy[1]:.2f}) "
                      f"new=({obj_world[0]:.2f},{obj_world[1]:.2f})) — "
                      f"recomputing base candidates")
                _pick_candidates[0] = _filtered_pick_candidates(obj_world)
                _pick_candidate_obj_xy[0] = obj_world[:2].copy()
                _pick_candidate_idx[0] = 0
                if _pick_candidates[0]:
                    _start_pick_nav(_pick_candidates[0][0], "object moved after grasp failure")
                    return
                print(f"[PICK] Obj-{obj_idx} unreachable after movement; aborting pick")
                # emit cycle-end so the auto-runner
                # (and any external tracker) can advance. Without
                # this, the cycle stays "in progress" forever and the
                # auto-runner only escapes via wall-clock timeout.
                _emit_cycle_metrics("obj_out_of_bounds",
                                    obj_idx, _current_shelf_idx[0])
                return
            if _local_radial_retry_after_grasp_fail(obj_world):
                return
            # orbit retry — chassis rotates around obj
            # at SAME arm height (GRASP_Q). Tries different approach
            # yaws without the expensive lift+nav+descend cycle.
            # Fires only when grasp failure was asymmetric reach
            # (orientation mismatch); other failures fall through to
            # the full candidate retry.
            if _orbit_retry_after_grasp_fail(obj_world):
                return
            if _try_next_pick_candidate("grasp failed"):
                return
            return

        # Object is now pinned to the gripper (pin closure handles it).
        grasp_in_progress_flag[0] = False

        if not ENABLE_PLACE_PHASE:
            print("[GUI] HOLDING — place phase disabled (set ENABLE_PLACE_PHASE=True to run it)")
            return

        # Navigate base to shelf standoff; place/drop runs from there.
        shelf_idx = _current_shelf_idx[0]
        shelf_pos = SHELF_SLOT_POSITIONS[shelf_idx].copy()
        move_in_progress_flag[0]  = True

        if USE_APPROXIMATE_DROP:
            # Use a validated open-aisle pose near the slot column; do not
            # require a final yaw so the base can release without rotating.
            standoff = _choose_approx_drop_standoff(shelf_pos)
            goal_tol = APPROX_DROP_GOAL_TOL
            final_yaw = None
            print(f"[NAV] Navigating to approximate drop aisle point {standoff}")
        else:
            standoff = (float(shelf_pos[0]), float(shelf_pos[1]) - 0.6)
            goal_tol = None
            final_yaw = None
            print(f"[NAV] Navigating to shelf standoff {standoff}")
        # Exempt the held object from floor-clearance check during
        # transport — it's pinned to the gripper and moves with the
        # robot, so static-position distance to it is meaningless.
        # Other floor objects must still be cleared by the chassis.
        try:
            held_bid = grasp_exec._held_obj_bid
            if held_bid is not None:
                nav.validator.set_exempt_objects([int(held_bid)])
        except Exception:
            pass
        nav.navigate_to(standoff, on_complete=_on_shelf_nav_complete,
                        goal_tolerance=goal_tol, final_yaw=final_yaw,
                        allow_goal_nudge=not USE_APPROXIMATE_DROP)

    def _on_shelf_nav_complete(success):
        move_in_progress_flag[0] = False
        label = "approximate drop aisle point" if USE_APPROXIMATE_DROP else "shelf standoff"
        print(f"[SHELF] Base at {label} — success={success}")
        if not success:
            if USE_APPROXIMATE_DROP:
                print("[SHELF] Navigation to assigned shelf-side aisle failed; "
                      "keeping object held, not dropping away from the slot")
                grasp_in_progress_flag[0] = False
            else:
                print("[SHELF] Navigation to shelf failed; cancelling place")
                grasp_exec.cancel()
            return

        shelf_idx = _current_shelf_idx[0]
        shelf_pos = SHELF_SLOT_POSITIONS[shelf_idx].copy()
        grasp_in_progress_flag[0] = True

        def _on_place_complete(place_ok):
            grasp_in_progress_flag[0] = False
            label = "Approximate drop" if USE_APPROXIMATE_DROP else "Place"
            print(f"[GUI] {label} complete — success={place_ok}")
            if place_ok:
                print("[GUI] Pick-and-place cycle complete!")
                _emit_cycle_metrics(
                    "success", _current_obj_idx[0], _current_shelf_idx[0])
            else:
                _emit_cycle_metrics(
                    "place_failed", _current_obj_idx[0], _current_shelf_idx[0])

        if USE_APPROXIMATE_DROP:
            print("[GUI] Approximate drop — releasing near assigned slot")
            drop_xy = (float(shelf_pos[0]), APPROX_DROP_OBJECT_Y)
            grasp_exec.drop(on_complete=_on_place_complete, target_xy=drop_xy)
        else:
            # GraspExecutor handles transport, descent, release, retract.
            grasp_exec.place(shelf_idx, shelf_pos, on_complete=_on_place_complete)

    glfw.make_context_current(sim.window)
    glfw.swap_interval(1)
    glfw.poll_events()
    if not glfw.get_current_context():
        glfw.make_context_current(sim.window)
    imgui.create_context()
    impl = GlfwRenderer(sim.window, attach_callbacks=False)

    # Start the optional screen recorder once the GL context is live.
    if screen_rec is not None:
        screen_rec.start(label="play_m1_session")

    obj_geom_ids  = get_object_geom_ids(sim.model)
    obj_rgba_orig = {}
    for i, geom_list in enumerate(obj_geom_ids):
        obj_rgba_orig[i] = [sim.model.geom_rgba[gid].copy() for gid in geom_list]
    collision_rgba_orig = sim.model.geom_rgba.copy()

    sim.opt.label = mujoco.mjtLabel.mjLABEL_SITE
    rng = np.random.default_rng()
    randomize_object_positions(sim.model, sim.data, rng)
    # capture the initial spawn so auto-respawn can
    # restore to THIS exact layout each cycle (rather than re-
    # randomising and getting different lucky/unlucky scenes).
    _initial_obj_snapshot = snapshot_object_positions(sim.model, sim.data)

    initial_x, initial_y, initial_yaw = sim.localization()
    target_x   = initial_x
    target_y   = initial_y
    target_yaw = initial_yaw

    joystick = Joystick(inner_radius=50, padding=20, ring_width=20, dead_zone=0.1)
    joystick.update_robot_yaw(initial_yaw)

    paused = False
    dt     = sim.model.opt.timestep

    sim.camera.distance  = 6.0
    sim.camera.azimuth   = 90
    sim.camera.elevation = -45
    sim.camera.lookat[:] = [4.0, -5.0, 0.5]

    xpos, ypos = glfw.get_cursor_pos(sim.window)
    sim._last_mouse_x = xpos
    sim._last_mouse_y = ypos

    selected_object = 0
    selected_shelf  = 0
    move_status     = ""
    collision_debug = False
    key3_was_down   = False

    # Final event pump before the render loop — drains anything that
    # accumulated during the late ImGui / scene setup so the very first
    # iteration of the main loop is responsive immediately.
    _pump_events()

    while not glfw.window_should_close(sim.window):
        glfw.poll_events()
        impl.process_inputs()

        io = imgui.get_io()
        key3_down = glfw.get_key(sim.window, glfw.KEY_3) == glfw.PRESS
        if key3_down and not key3_was_down and not io.want_capture_keyboard:
            collision_debug = not collision_debug
            counts = set_collision_debug_overlay(sim.model, collision_rgba_orig, collision_debug)
            sim.opt.geomgroup[3] = 1 if collision_debug else 0
            # NOTE: do NOT touch mjVIS_CONTACTPOINT here. The yellow
            # contact-point markers are useful regardless of whether
            # the collision-overlay is on, and they're enabled at
            # startup (morph_i_free_move.py). Use the `C` key (in
            # the free-move sim's handler) to toggle them independently.
            sim.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 1 if collision_debug else 0
            print(f"[VIS] Collision debug {'ON' if collision_debug else 'OFF'} "
                  f"(geomgroup[3]={sim.opt.geomgroup[3]}) "
                  f"box={counts['box']} cyl={counts['cylinder']} "
                  f"capsule={counts['capsule']} other={counts['other']}")
        key3_was_down = key3_down

        if not io.want_capture_mouse:
            xpos, ypos = glfw.get_cursor_pos(sim.window)
            dx = xpos - sim._last_mouse_x
            dy = ypos - sim._last_mouse_y
            sim._last_mouse_x = xpos
            sim._last_mouse_y = ypos
            factor = 0.001
            left   = glfw.get_mouse_button(sim.window, glfw.MOUSE_BUTTON_LEFT)   == glfw.PRESS
            right  = glfw.get_mouse_button(sim.window, glfw.MOUSE_BUTTON_RIGHT)  == glfw.PRESS
            middle = glfw.get_mouse_button(sim.window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
            if left:
                mujoco.mjv_moveCamera(sim.model, mujoco.mjtMouse.mjMOUSE_ROTATE_H,
                                      dx*factor, dy*factor, sim.scene, sim.camera)
            elif right:
                mujoco.mjv_moveCamera(sim.model, mujoco.mjtMouse.mjMOUSE_MOVE_H,
                                      dx*factor, dy*factor, sim.scene, sim.camera)
            elif middle:
                mujoco.mjv_moveCamera(sim.model, mujoco.mjtMouse.mjMOUSE_ZOOM,
                                      dx*factor, dy*factor, sim.scene, sim.camera)
        else:
            xpos, ypos = glfw.get_cursor_pos(sim.window)
            sim._last_mouse_x = xpos
            sim._last_mouse_y = ypos

        actual_x, actual_y, actual_yaw = sim.localization()
        joystick.update_robot_yaw(actual_yaw)

        if not paused:
            (x_local, y_local), yaw_command = joystick.value
            manual_active = joystick.is_active or yaw_command is not None
            if manual_active:
                if yaw_command is not None:
                    target_yaw = yaw_command
                cos_a    = np.cos(actual_yaw)
                sin_a    = np.sin(actual_yaw)
                world_vx =  cos_a * y_local + sin_a * x_local
                world_vy =  sin_a * y_local - cos_a * x_local
                target_x += world_vx * 1.0 * dt
                target_y += world_vy * 1.0 * dt
            else:
                # Idle joystick: keep the manual target locked to the real
                # robot pose so it doesn't fight autonomous nav/drop.
                target_x, target_y, target_yaw = actual_x, actual_y, actual_yaw

            if not move_in_progress_flag[0] and not grasp_in_progress_flag[0]:
                with sim._target_lock:
                    sim.target_base = np.array([target_x, target_y, target_yaw])

            # Pin closures registered by GraspExecutor run inside step_simulation.
            sim.step_simulation(render=False)

        # Object highlight
        for i, geom_list in enumerate(obj_geom_ids):
            for j, gid in enumerate(geom_list):
                orig = obj_rgba_orig[i][j]
                if i == selected_object:
                    sim.model.geom_rgba[gid] = np.clip(
                        orig[:3] * 1.4 + np.array([0.25, 0.25, 0.25]), 0, 1
                    ).tolist() + [1.0]
                elif collision_debug:
                    sim.model.geom_rgba[gid] = orig.copy()
                else:
                    sim.model.geom_rgba[gid] = [0.0, 0.0, 0.0, 0.0]

        fb_width, fb_height = glfw.get_framebuffer_size(sim.window)
        sim.viewport.width  = fb_width
        sim.viewport.height = fb_height
        mujoco.mjv_updateScene(sim.model, sim.data, sim.opt, None,
                               sim.camera, mujoco.mjtCatBit.mjCAT_ALL, sim.scene)
        mujoco.mjr_render(sim.viewport, sim.scene, sim.ctx)

        imgui.new_frame()
        bg = imgui.get_background_draw_list()

        for i in range(NUM_OBJECTS):
            pos3d = get_object_world_pos(sim.model, sim.data, i)
            sc    = world_to_screen(pos3d + np.array([0, 0, 0.18]), sim.scene, sim.viewport)
            if sc is None: continue
            sx, sy = sc
            col    = OBJECT_COLORS_RGBA[i]
            is_sel = (i == selected_object)
            radius = 13 if is_sel else 9
            bg.add_circle_filled(sx, sy, radius,
                imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 0.95) if is_sel
                else imgui.get_color_u32_rgba(col[0]*0.7, col[1]*0.7, col[2]*0.7, 0.90))
            txt_col = imgui.get_color_u32_rgba(0.0, 0.0, 0.0, 1.0) if is_sel \
                      else imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 1.0)
            bg.add_text(sx - 4, sy - 7, txt_col, str(i))
            if is_sel:
                bg.add_circle(sx, sy, radius + 3,
                    imgui.get_color_u32_rgba(1.0, 0.9, 0.0, 1.0), thickness=2)

        for i in range(NUM_SHELF_SLOTS):
            sp  = SHELF_SLOT_POSITIONS[i]
            sc  = world_to_screen(sp + np.array([0, 0, 0.12]), sim.scene, sim.viewport)
            if sc is None: continue
            sx, sy  = sc
            is_sel  = (i == selected_shelf)
            radius  = 11 if is_sel else 7
            bg.add_circle_filled(sx, sy, radius,
                imgui.get_color_u32_rgba(0.2, 0.7, 0.2, 0.90) if is_sel
                else imgui.get_color_u32_rgba(0.15, 0.45, 0.15, 0.75))
            bg.add_text(sx - 4, sy - 7, imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 1.0), str(i))
            if is_sel:
                bg.add_circle(sx, sy, radius + 3,
                    imgui.get_color_u32_rgba(0.0, 1.0, 0.5, 1.0), thickness=2)

        # Bottom status bar
        bar_h = 32
        bar_y = fb_height - bar_h
        bar_w = fb_width
        bg.add_rect_filled(0, bar_y, bar_w, fb_height,
                           imgui.get_color_u32_rgba(0.05, 0.05, 0.15, 0.82))
        col_o = OBJECT_COLORS_RGBA[selected_object]
        if move_in_progress_flag[0]:
            status_txt = "● MOVING"
            status_col = imgui.get_color_u32_rgba(1.0, 0.8, 0.0, 1.0)
        elif grasp_in_progress_flag[0]:
            status_txt = "● GRASPING"
            status_col = imgui.get_color_u32_rgba(1.0, 0.5, 0.0, 1.0)
        elif grasp_exec.is_holding():
            status_txt = "● HOLDING"
            status_col = imgui.get_color_u32_rgba(0.3, 1.0, 0.5, 1.0)
        else:
            status_txt = "IDLE"
            status_col = imgui.get_color_u32_rgba(0.5, 0.9, 0.5, 1.0)
        bg.add_text(10,  bar_y + 8,
                    imgui.get_color_u32_rgba(col_o[0], col_o[1], col_o[2], 1.0),
                    f"Object: {OBJECT_LABELS[selected_object]}")
        bg.add_text(280, bar_y + 8,
                    imgui.get_color_u32_rgba(0.3, 0.9, 0.4, 1.0),
                    f"Shelf: {SHELF_SLOT_LABELS[selected_shelf]}")
        bg.add_text(bar_w - 110, bar_y + 8, status_col, status_txt)

        imgui.set_next_window_position(10, 10)
        imgui.set_next_window_size(465, 820)
        imgui.set_next_window_bg_alpha(0.95)
        flags = (imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE |
                 imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_COLLAPSE)
        imgui.begin("Robot Control", flags=flags)

        ik_enabled = bool(sim.use_ik)
        clicked, ik_enabled = imgui.checkbox("IK Control", ik_enabled)
        if clicked:
            with sim._target_lock:
                sim.use_ik = ik_enabled
                if ik_enabled:
                    l_enc, r_enc = sim.get_encoder()
                    fk_l = sim.fk(*l_enc)
                    fk_r = sim.fk(*r_enc)
                    sim.target_left  = np.array(fk_l) if fk_l is not None else np.array([ 0.5, 0.0, 0.5])
                    sim.target_right = np.array(fk_r) if fk_r is not None else np.array([-0.5, 0.0, 0.5])
                else:
                    l_enc, r_enc = sim.get_encoder()
                    sim.direct_arm_commands[0:4] = l_enc
                    sim.direct_arm_commands[4:8] = r_enc

        imgui.separator(); imgui.text("ARM1")
        imgui.push_item_width(180)
        if ik_enabled:
            tl = list(sim.target_left)
            _, x = imgui.slider_float("X##ARM1", tl[0], -1.0, 1.0, "%.3f")
            _, y = imgui.slider_float("Y##ARM1", tl[1], -1.0, 1.0, "%.3f")
            _, z = imgui.slider_float("Z##ARM1", tl[2], -1.0, 1.0, "%.3f")
            if x != tl[0] or y != tl[1] or z != tl[2]:
                with sim._target_lock:
                    sim.target_left = np.array([x, y, z])
        else:
            h1i,h2i,a1i,th = sim.direct_arm_commands[0:4]
            _,h1u = imgui.slider_float("H1##ARM1", np.interp(h1i,[0,1.5],[-75,75]), -75.0, 75.0, "%.1f")
            _,h2u = imgui.slider_float("H2##ARM1", np.interp(h2i,[0,1.5],[-75,75]), -75.0, 75.0, "%.1f")
            _,a1u = imgui.slider_float("A1##ARM1", np.interp(a1i,[0,0.7],[-45,45]), -45.0, 45.0, "%.1f")
            _,thd = imgui.slider_float("TH##ARM1", np.degrees(th), -90.0, 90.0, "%.1f")
            with sim._target_lock:
                sim.direct_arm_commands[0:4] = [np.interp(h1u,[-75,75],[0,1.5]),
                    np.interp(h2u,[-75,75],[0,1.5]), np.interp(a1u,[-45,45],[0,0.7]),
                    np.radians(thd)]
        imgui.pop_item_width()

        imgui.separator(); imgui.text("ARM2")
        imgui.push_item_width(180)
        if ik_enabled:
            tr = list(sim.target_right)
            _, x = imgui.slider_float("X##ARM2", tr[0], -1.0, 1.0, "%.3f")
            _, y = imgui.slider_float("Y##ARM2", tr[1], -1.0, 1.0, "%.3f")
            _, z = imgui.slider_float("Z##ARM2", tr[2], -1.0, 1.0, "%.3f")
            if x != tr[0] or y != tr[1] or z != tr[2]:
                with sim._target_lock:
                    sim.target_right = np.array([x, y, z])
        else:
            h1i,h2i,a1i,th = sim.direct_arm_commands[4:8]
            _,h1u = imgui.slider_float("H1##ARM2", np.interp(h1i,[0,1.5],[-75,75]), -75.0, 75.0, "%.1f")
            _,h2u = imgui.slider_float("H2##ARM2", np.interp(h2i,[0,1.5],[-75,75]), -75.0, 75.0, "%.1f")
            _,a1u = imgui.slider_float("A1##ARM2", np.interp(a1i,[0,0.7],[-45,45]), -45.0, 45.0, "%.1f")
            _,thd = imgui.slider_float("TH##ARM2", np.degrees(th), -90.0, 90.0, "%.1f")
            with sim._target_lock:
                sim.direct_arm_commands[4:8] = [np.interp(h1u,[-75,75],[0,1.5]),
                    np.interp(h2u,[-75,75],[0,1.5]), np.interp(a1u,[-45,45],[0,0.7]),
                    np.radians(thd)]
        imgui.pop_item_width()

        imgui.separator(); imgui.text("Grippers")
        imgui.text_disabled("(0% = Safe Open, 100% = Closed)")
        OPEN, CLOSED = -0.35, 0.8
        # Curl profile mirrors grasp_executor._curl_targets so the manual
        # slider matches autonomous-grasp visuals. j3 (distal) curls
        # inward (more negative). j1 (proximal) is decoupled per finger:
        # the thumb uses a shallower factor to avoid curling into the
        # wrist; the side fingers use a deeper factor to wrap the object.
        CURL_J1_F_THUMB = 0.70   # finger_a (thumb)
        CURL_J1_F_SIDE  = 0.85   # finger_b, finger_c
        CURL_J2_F, CURL_J3_F = 0.90, 1.00
        J1_EXTENT, J2_EXTENT, J3_EXTENT = 0.85, 0.95, 1.10
        J3_REST = -0.052
        # Palm spread joints (indices 9, 10 of gripper_ids_left).
        # See grasp_executor.PALM_*_SIGN comment — both palms driven
        # with the same sign by default (legacy convention).
        PALM_OPEN, PALM_CLOSE = 0.18, 0.0
        PALM_C_SIGN_MAN, PALM_B_SIGN_MAN = +1.0, +1.0

        def _apply_curl(gids, v):
            if len(gids) < 9:
                # Fallback for models missing the expected 9 finger actuators.
                for idx in [0, 3, 6]:
                    if idx < len(gids):
                        sim.data.ctrl[gids[idx]] = v
                return
            # CLOSE symmetric (thumb closes with sides). OPEN asymmetric
            # thumb j1 goes deeper than sides for descent clearance.
            THUMB_OPEN_MAN = -1.87   # matches grasp_executor.THUMB_OPEN_POS
            if v >= 0.0:
                intensity = min(1.0, max(0.0, v / 0.20))
                j1 = intensity * CURL_J1_F_SIDE * J1_EXTENT
                j2 = intensity * CURL_J2_F      * J2_EXTENT
                j3 = J3_REST - intensity * CURL_J3_F * J3_EXTENT
                palm_mag = PALM_CLOSE
                j1_side, j1_thumb = j1, j1
            else:
                j1 = v               # = OPEN slider value = -0.55
                j2 = 0.0
                j3 = -0.0523
                palm_mag = PALM_OPEN
                j1_side, j1_thumb = v, THUMB_OPEN_MAN
            # gids order: indices 0..2 = finger_c, 3..5 = finger_b,
            # 6..8 = finger_a. finger_a's j1 uses j1_thumb (deeper on
            # OPEN); j2/j3 same across all three.
            for base, j1_use in ((0, j1_side), (3, j1_side), (6, j1_thumb)):
                sim.data.ctrl[gids[base + 0]] = j1_use
                sim.data.ctrl[gids[base + 1]] = j2
                sim.data.ctrl[gids[base + 2]] = j3
            # Palm spread (palm_finger_c, palm_finger_b) — mirrored signs.
            if len(gids) >= 11:
                sim.data.ctrl[gids[9]]  = PALM_C_SIGN_MAN * palm_mag
                sim.data.ctrl[gids[10]] = PALM_B_SIGN_MAN * palm_mag

        imgui.push_item_width(180)
        # Slider display maps the proximal-joint (j1) ctrl back to an
        # equivalent slider percentage. Use the side-finger j1 since
        # that's what's visible on gids[0] (finger_c) — gives a more
        # useful read-out than the shallow thumb factor.
        J1_MAX_CLOSE = CURL_J1_F_SIDE * J1_EXTENT     # j1 ctrl when intensity=1
        if len(sim.gripper_ids_left) >= 9:
            cur_prox = float(sim.data.ctrl[sim.gripper_ids_left[0]])
            if cur_prox >= 0.0 and J1_MAX_CLOSE > 1e-6:
                equiv_close = (cur_prox / J1_MAX_CLOSE) * CLOSED
            else:
                equiv_close = cur_prox
            pct = np.clip((equiv_close - OPEN) / (CLOSED - OPEN) * 100.0, 0, 100)
            chg, pct = imgui.slider_float("Left##GRIP", pct, 0.0, 100.0, "%.0f%%")
            if chg:
                v = OPEN + (CLOSED - OPEN) * pct / 100.0
                _apply_curl(sim.gripper_ids_left, v)
        if len(sim.gripper_ids_right) >= 9:
            cur_prox = float(sim.data.ctrl[sim.gripper_ids_right[0]])
            if cur_prox >= 0.0 and J1_MAX_CLOSE > 1e-6:
                equiv_close = (cur_prox / J1_MAX_CLOSE) * CLOSED
            else:
                equiv_close = cur_prox
            pct = np.clip((equiv_close - OPEN) / (CLOSED - OPEN) * 100.0, 0, 100)
            chg, pct = imgui.slider_float("Right##GRIP", pct, 0.0, 100.0, "%.0f%%")
            if chg:
                v = OPEN + (CLOSED - OPEN) * pct / 100.0
                _apply_curl(sim.gripper_ids_right, v)
        imgui.pop_item_width()

        imgui.separator()
        _, paused = imgui.checkbox("Pause", paused)
        imgui.same_line()
        if imgui.button("Reset Robot"):
            sim.reset("home")
            grasp_exec.cancel()
            rx, ry, ryaw = sim.localization()
            target_x, target_y, target_yaw = rx, ry, ryaw
            joystick.update_robot_yaw(ryaw)
            move_in_progress_flag[0]  = False
            grasp_in_progress_flag[0] = False
            _arm_offset = np.array([-0.0036, -0.0062, -0.0006])
            for _i, _q in enumerate(STARTUP_Q_ARM1):
                sim.data.qpos[sim.qpos_indices[_i]] = _q
            for _i, _q in enumerate(STARTUP_Q_ARM2):
                sim.data.qpos[sim.qpos_indices[4 + _i]] = _q
            sim.data.ctrl[sim.actuator_ids[0:3]] = (np.array(STARTUP_Q_ARM1[:3]) + _arm_offset) * 100
            sim.data.ctrl[sim.actuator_ids[4:7]] = (np.array(STARTUP_Q_ARM2[:3]) + _arm_offset) * 100
            mujoco.mj_forward(sim.model, sim.data)
            with sim._target_lock:
                sim.target_base = np.array([rx, ry, ryaw])
                sim.direct_arm_commands[0:4] = list(STARTUP_Q_ARM1)
                sim.direct_arm_commands[4:8] = list(STARTUP_Q_ARM2)
        imgui.same_line()
        if imgui.button("Respawn Objects"):
            grasp_exec.cancel()
            randomize_object_positions(sim.model, sim.data, rng)
            move_status = "Objects respawned."

        imgui.separator()
        imgui.text("Joystick  (inner=translate, ring=rotate):")
        joystick.draw("MainJoystick")
        imgui.text(f"Base → X:{target_x:.2f}  Y:{target_y:.2f}  Yaw:{np.degrees(target_yaw):.1f}°")

        # PICK & PLACE PANEL
        imgui.separator(); imgui.separator()
        dl    = imgui.get_window_draw_list()
        p     = imgui.get_cursor_screen_pos()
        win_w = imgui.get_window_width() - imgui.get_style().window_padding.x * 2
        dl.add_rect_filled(p.x, p.y, p.x + win_w, p.y + 20,
                           imgui.get_color_u32_rgba(0.15, 0.35, 0.55, 1.0), rounding=3.0)
        imgui.set_cursor_screen_pos((p.x + 6, p.y + 3))
        imgui.text_colored("PICK & PLACE", 1.0, 1.0, 1.0, 1.0)
        imgui.dummy(0, 4)

        imgui.push_item_width(230)
        imgui.text("Object ID")
        imgui.same_line()
        col = OBJECT_COLORS_RGBA[selected_object]
        imgui.color_button(f"##sw{selected_object}", col[0], col[1], col[2], col[3],
                           flags=0, width=14, height=14)
        _, selected_object = imgui.combo("##ObjCombo", selected_object, OBJECT_LABELS)
        obj_pos = get_object_world_pos(sim.model, sim.data, selected_object)
        imgui.text_disabled(f"  Pos: ({obj_pos[0]:.2f}, {obj_pos[1]:.2f}, {obj_pos[2]:.2f})")
        imgui.spacing()
        imgui.text("Shelf Slot ID")
        _, selected_shelf = imgui.combo("##ShelfCombo", selected_shelf, SHELF_SLOT_LABELS)
        sp = SHELF_SLOT_POSITIONS[selected_shelf]
        imgui.text_disabled(f"  Pos: ({sp[0]:.2f}, {sp[1]:.2f}, {sp[2]:.2f})")
        imgui.pop_item_width()
        imgui.spacing()

        any_in_progress = move_in_progress_flag[0] or grasp_in_progress_flag[0]
        holding_object = grasp_exec.is_holding()

        # Auto-MOVE trigger (CLI --auto-move-attempts). Fires when:
        # 1. auto mode armed (next cycle ready to start)
        # 2. nothing in progress AND not holding
        # 3. inter-cycle delay elapsed
        # Replicates the MOVE button click logic programmatically.
        if (_auto_move_armed[0]
                and not any_in_progress
                and not holding_object
                and time.time() >= _auto_move_next_at[0]):
            _auto_move_armed[0] = False
            cycle_num = _auto_move_attempts_done[0] + 1
            selected_object = int(args.auto_move_obj)
            selected_shelf  = int(args.auto_move_slot)
            _is_first_cycle = (_auto_move_attempts_done[0] == 0)
            if (int(args.auto_respawn_between_cycles) == 1
                    and not _is_first_cycle):
                try:
                    # 1. Cancel any in-flight grasp state
                    grasp_exec.cancel()
                    # 2. Reset robot sim state to home
                    sim.reset("home")
                    rx, ry, ryaw = sim.localization()
                    target_x, target_y, target_yaw = rx, ry, ryaw
                    joystick.update_robot_yaw(ryaw)
                    move_in_progress_flag[0]  = False
                    grasp_in_progress_flag[0] = False
                    # 3. Re-apply STARTUP_Q arm pose
                    _arm_offset = np.array([-0.0036, -0.0062, -0.0006])
                    for _i, _q in enumerate(STARTUP_Q_ARM1):
                        sim.data.qpos[sim.qpos_indices[_i]] = _q
                    for _i, _q in enumerate(STARTUP_Q_ARM2):
                        sim.data.qpos[sim.qpos_indices[4 + _i]] = _q
                    sim.data.ctrl[sim.actuator_ids[0:3]] = (
                        np.array(STARTUP_Q_ARM1[:3]) + _arm_offset) * 100
                    sim.data.ctrl[sim.actuator_ids[4:7]] = (
                        np.array(STARTUP_Q_ARM2[:3]) + _arm_offset) * 100
                    mujoco.mj_forward(sim.model, sim.data)
                    with sim._target_lock:
                        sim.target_base = np.array([rx, ry, ryaw])
                        sim.direct_arm_commands[0:4] = list(STARTUP_Q_ARM1)
                        sim.direct_arm_commands[4:8] = list(STARTUP_Q_ARM2)
                    # also reset gripper to OPEN ctrl
                    # before next cycle. Without this, fingers stay at
                    # last-cycle's closed ctrl → next-cycle close stroke
                    # re-fires from already-closed = obj wedged/blocked.
                    try:
                        import navigation.grasp_executor as _gx_y
                        _gids_left_y = sim.gripper_ids_left
                        _open_curl_y = grasp_exec._curl_targets(
                            _gx_y.GRIPPER_OPEN_POS)
                        _addrs_y = grasp_exec._ensure_finger_joint_qposadrs()
                        with sim._target_lock:
                            for j_idx_y, val_y in enumerate(_open_curl_y):
                                if (j_idx_y < 9
                                        and j_idx_y < len(_gids_left_y)):
                                    sim.data.ctrl[_gids_left_y[j_idx_y]] = float(val_y)
                                    if (_addrs_y
                                            and j_idx_y < len(_addrs_y)
                                            and _addrs_y[j_idx_y] >= 0):
                                        sim.data.qpos[_addrs_y[j_idx_y]] = float(val_y)
                    except Exception as _e_gr:
                        print(f"[AUTO_RUN] gripper reset to open "
                              f"failed: {_e_gr}")
                    # 4. Restore objects to initial layout
                    restore_object_positions(sim.model, sim.data,
                                              _initial_obj_snapshot)
                    print(f"[AUTO_RUN] cycle {cycle_num}/"
                          f"{_auto_move_attempts_total}: "
                          f"robot reset to startup pose + objects "
                          f"restored to initial spawn positions")
                except Exception as _e_respawn:
                    print(f"[AUTO_RUN] respawn/reset raised: {_e_respawn}")
            _current_obj_idx[0]   = selected_object
            _current_shelf_idx[0] = selected_shelf
            obj_world = get_object_world_pos(
                sim.model, sim.data, selected_object)
            _pick_candidate_obj_xy[0] = obj_world[:2].copy()
            _pick_replan_count[0] = 0
            _pick_local_retry_used[0] = 0
            _relaxed_clearance[0] = False
            _any_grasp_attempted[0] = False
            nav.validator.skip_segment_validation = False
            try:
                grasp_exec.reset_finger_attempt_counter()
            except AttributeError:
                pass
            _pick_candidates[0] = _filtered_pick_candidates(obj_world)
            _cycle_start_time[0] = time.time()
            print(f"[AUTO_RUN] === cycle {cycle_num}/"
                  f"{_auto_move_attempts_total} START "
                  f"obj={selected_object} slot={selected_shelf} ===")
            if not _pick_candidates[0]:
                print(f"[AUTO_RUN] cycle {cycle_num}: no nav candidates")
                _emit_cycle_metrics("no_candidates",
                                    selected_object, selected_shelf)
            else:
                _pick_candidate_idx[0] = 0
                # auto-skip-nav teleports chassis
                # straight to the first ranked candidate's pose so the
                # OMPL nav-plan + drive phase (~25-30 s) is
                # short-circuited. With (nav-yaw
                # asymmetry filter), the first candidate is the one
                # with best predicted thumb-bc symmetric reach.
                if args.auto_skip_nav:
                    try:
                        _cand0 = _pick_candidates[0][0]
                        _cgxy = _candidate_xy(_cand0)
                        _obj_w = get_object_world_pos(
                            sim.model, sim.data, selected_object)
                        _cyaw = pick_candidate_yaw(_cgxy, _obj_w[:2])
                        _base_bid_sk = mujoco.mj_name2id(
                            sim.model, mujoco.mjtObj.mjOBJ_BODY, "robot")
                        if _base_bid_sk < 0:
                            _base_bid_sk = mujoco.mj_name2id(
                                sim.model, mujoco.mjtObj.mjOBJ_BODY, "base")
                        if _base_bid_sk >= 0:
                            _ja_sk = int(sim.model.body_jntadr[_base_bid_sk])
                            if _ja_sk >= 0:
                                _qa_sk = int(sim.model.jnt_qposadr[_ja_sk])
                                sim.data.qpos[_qa_sk]     = float(_cgxy[0])
                                sim.data.qpos[_qa_sk + 1] = float(_cgxy[1])
                                # free-joint quat: [w, x, y, z] for yaw
                                sim.data.qpos[_qa_sk + 3] = math.cos(_cyaw / 2.0)
                                sim.data.qpos[_qa_sk + 4] = 0.0
                                sim.data.qpos[_qa_sk + 5] = 0.0
                                sim.data.qpos[_qa_sk + 6] = math.sin(_cyaw / 2.0)
                        with sim._target_lock:
                            sim.target_base = np.array(
                                [float(_cgxy[0]),
                                 float(_cgxy[1]),
                                 float(_cyaw)])
                        mujoco.mj_forward(sim.model, sim.data)
                        print(f"[AUTO_RUN] --auto-skip-nav: teleported "
                              f"chassis to ({_cgxy[0]:.2f},{_cgxy[1]:.2f}) "
                              f"yaw={math.degrees(_cyaw):+.0f}°  "
                              f"(skipped OMPL nav)")
                        # arm obj-to-pocket snap one-shot, giving the
                        # lift a floating-obj setup (same as harness).
                        # Cleared by grasp_executor after the first
                        # close fires this cycle.
                        try:
                            import navigation.grasp_executor as _gx2
                            _gx2.SNAP_OBJ_TO_POCKET_PRE_CLOSE_ONCE = True
                            print(f"[AUTO_RUN] --auto-skip-nav: armed "
                                  f"obj-to-pocket snap for next close")
                        except Exception as _e_arm:
                            print(f"[AUTO_RUN] obj-snap arm failed: {_e_arm}")
                    except Exception as _e_sk:
                        print(f"[AUTO_RUN] --auto-skip-nav teleport "
                              f"failed: {_e_sk} — falling back to nav")
                _start_pick_nav(_pick_candidates[0][0], "auto_initial")

        # Auto-MOVE timeout safeguard: if a cycle is still running
        # past the wall-clock budget, cancel the grasp, FORCE-CLEAR
        # the PICK orchestrator's retry state, and force-end the
        # cycle so the next attempt can start. Catches stalls in
        # IK convergence and the multi-candidate-iteration overrun
        # we saw in.cancel()
        # stopped the active grasp but the outer PICK loop kept
        # iterating through nav candidates for another 5+ minutes.
        if (_auto_move_attempts_total > 0
                and not _auto_move_armed[0]
                and _cycle_start_time[0] is not None
                and (time.time() - _cycle_start_time[0]
                     > _AUTO_CYCLE_TIMEOUT_S)):
            cycle_num_to = _auto_move_attempts_done[0] + 1
            elapsed_to = time.time() - _cycle_start_time[0]
            print(f"[AUTO_RUN] cycle {cycle_num_to}/"
                  f"{_auto_move_attempts_total} TIMEOUT after "
                  f"{elapsed_to:.1f}s (budget {_AUTO_CYCLE_TIMEOUT_S:.0f}s) — "
                  f"cancelling grasp + clearing PICK state, force-ending cycle")
            try:
                grasp_exec.cancel()
            except Exception as _e_cancel:
                print(f"[AUTO_RUN] cancel raised: {_e_cancel}")
            # also wipe the PICK orchestrator's
            # retry/candidate state so it doesn't keep iterating
            # through remaining nav candidates after the timeout.
            _pick_candidates[0]       = []
            _pick_candidate_idx[0]    = 0
            _pick_replan_count[0]     = 0
            _pick_local_retry_used[0] = 0
            try:
                _pick_orbit_retry_used[0] = 0
            except Exception:
                pass
            try:
                _pick_prev_max_far[0] = None
                _pick_prev_failure_sig[0] = None
            except Exception:
                pass
            move_in_progress_flag[0] = False
            grasp_in_progress_flag[0] = False
            _emit_cycle_metrics("auto_timeout",
                                _current_obj_idx[0],
                                _current_shelf_idx[0])

        if move_in_progress_flag[0]:
            imgui.push_style_color(imgui.COLOR_BUTTON,         0.70, 0.55, 0.00, 1.0)
            imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.85, 0.65, 0.00, 1.0)
            imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE,  0.90, 0.70, 0.00, 1.0)
            btn_label = " Moving…   "
        elif grasp_in_progress_flag[0]:
            imgui.push_style_color(imgui.COLOR_BUTTON,         0.70, 0.30, 0.00, 1.0)
            imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.85, 0.40, 0.00, 1.0)
            imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE,  0.90, 0.45, 0.00, 1.0)
            btn_label = " Grasping… "
        elif holding_object:
            imgui.push_style_color(imgui.COLOR_BUTTON,         0.05, 0.35, 0.25, 1.0)
            imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.08, 0.45, 0.32, 1.0)
            imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE,  0.04, 0.30, 0.22, 1.0)
            btn_label = " Holding  "
        else:
            imgui.push_style_color(imgui.COLOR_BUTTON,         0.05, 0.55, 0.15, 1.0)
            imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.10, 0.70, 0.20, 1.0)
            imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE,  0.02, 0.45, 0.10, 1.0)
            btn_label = "  MOVE ▶   "

        if imgui.button(btn_label, width=win_w, height=28):
            if not any_in_progress and not holding_object:
                _current_obj_idx[0]   = selected_object
                _current_shelf_idx[0] = selected_shelf
                obj_world   = get_object_world_pos(sim.model, sim.data, selected_object)
                _pick_candidate_obj_xy[0] = obj_world[:2].copy()
                _pick_replan_count[0] = 0
                _pick_local_retry_used[0] = 0
                _relaxed_clearance[0] = False  # start each cycle in strict mode
                _any_grasp_attempted[0] = False
                nav.validator.skip_segment_validation = False
                # Path B Step 2: reset the 3-finger strict-attempts
                # counter at the start of each MOVE cycle. The counter
                # persists across the base-XY retries that happen inside
                # the cycle, so the strict→relaxed transition fires after
                # MAX_STRICT_3FINGER_ATTEMPTS total attempts in the
                # cycle (not per retry).
                try:
                    grasp_exec.reset_finger_attempt_counter()
                except AttributeError:
                    pass
                _pick_candidates[0] = _filtered_pick_candidates(obj_world)
                _cycle_start_time[0] = time.time()
                if not _pick_candidates[0]:
                    move_status = (f"Obj-{selected_object} unreachable from safe pick standoffs; "
                                   "no robot motion started")
                    print(f"[MOVE] {move_status}")
                    _emit_cycle_metrics("no_candidates", selected_object, selected_shelf)
                else:
                    _pick_candidate_idx[0] = 0
                    goal_xy = _candidate_xy(_pick_candidates[0][0])
                    move_status = (f"Obj-{selected_object} → Slot-{selected_shelf}  "
                                   f"src=({obj_world[0]:.2f},{obj_world[1]:.2f})  "
                                   f"nav=({goal_xy[0]:.2f},{goal_xy[1]:.2f})  "
                                   f"candidates={len(_pick_candidates[0])}")
                    print(f"[MOVE] {move_status}")
                    _start_pick_nav(_pick_candidates[0][0], "initial")

        imgui.pop_style_color(3)

        if any_in_progress:
            if move_in_progress_flag[0]:
                imgui.text_colored("● Navigating to object…", 1.0, 0.8, 0.0, 1.0)
            else:
                imgui.text_colored("● Grasping object…", 1.0, 0.5, 0.0, 1.0)
            if imgui.small_button("Cancel##mv"):
                nav.cancel()
                grasp_exec.cancel()
                move_in_progress_flag[0]  = False
                grasp_in_progress_flag[0] = False
                move_status = "Cancelled."
                _emit_cycle_metrics(
                    "cancelled",
                    _current_obj_idx[0], _current_shelf_idx[0])
        elif holding_object:
            held = grasp_exec._held_obj_idx
            imgui.text_colored(f"● Holding Obj-{held}", 0.3, 1.0, 0.5, 1.0)
            if imgui.small_button("Release##rel"):
                grasp_exec.drop()
                move_status = "Object released."
        elif move_status:
            imgui.push_text_wrap_pos(imgui.get_cursor_pos_x() + win_w)
            imgui.text_disabled(move_status[:120])
            imgui.pop_text_wrap_pos()

        imgui.end()
        imgui.render()
        impl.render(imgui.get_draw_data())
        # Capture before swap_buffers so the readback sees the just-rendered
        # frame. No-op when --record is not set.
        if screen_rec is not None:
            screen_rec.capture_frame(sim.window)
        glfw.swap_buffers(sim.window)

    if screen_rec is not None:
        screen_rec.stop()
    impl.shutdown()
    glfw.terminate()


if __name__ == "__main__":
    main()
