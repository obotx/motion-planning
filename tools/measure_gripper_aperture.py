"""Measure the actual gripper aperture (thumb-tip ↔ bc-centroid
distance) at a range of finger close-ctrl values, and derive a
geometry-based ``OBJ_RADIUS_RANGE`` for stable 3-point pinch.

Run:

    cd motion-planning
    python -m tools.measure_gripper_aperture

The script loads ``market_world_m1.xml``, parks the robot, and steps
the finger actuators across the close stroke.  At each step it
records the world-XY span between the thumb fingertip
(``finger_a_link_3_1``) and the centroid of the bc fingertips,
mirroring how ``_pinch_midpoint_xyz`` defines the pinch geometry in
``grasp_executor.py``.

For each obj radius candidate the script computes the closure margin
(``span − 2 × radius``):

* positive  → fingers can still close further around the obj
* near zero → fingers just kiss the surface, no wrap-around
* negative  → fingers would interpenetrate the obj (geometrically
  infeasible)

The recommended OBJ_RADIUS_RANGE is the band whose closure margin at
the *target* close ctrl stays in the productive zone — empirically
2-4 cm of margin gives the bc-pair enough curl to wrap the obj.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import mujoco


ROOT = Path(__file__).resolve().parents[1]
XML_PATH = ROOT / "src" / "env" / "market_world_m1.xml"

# Mirrors GraspExecutor._carry_anchor_body_ids order: thumb, b, c.
ANCHOR_BODY_NAMES = (
    "finger_a_link_3_1",
    "finger_b_link_3_1",
    "finger_c_link_3_1",
)

# gripper_actuator_V2.xml ctrl order (left arm only — index 0..10):
#   0:c_j1  1:c_j2  2:c_j3  3:b_j1  4:b_j2  5:b_j3
#   6:a_j1  7:a_j2  8:a_j3  9:palm_c 10:palm_b
# The close stroke drives j1 ctrl positive on all 3 fingers.
J1_INDICES = (0, 3, 6)   # c_j1, b_j1, a_j1
J2_INDICES = (1, 4, 7)
J3_INDICES = (2, 5, 8)


def _body_id(model: mujoco.MjModel, name: str) -> int:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise RuntimeError(f"body not found: {name}")
    return bid


def _resolve_left_actuator_offset(model: mujoco.MjModel) -> int:
    """Find the ctrl-array index of ``finger_c_joint_1_1`` (first
    finger actuator of the LEFT arm)."""
    target = "finger_c_joint_1_1"
    for aid in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
        if name == target:
            return aid
    raise RuntimeError(f"actuator not found: {target}")


def measure_aperture_at(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    anchor_bids: tuple[int, int, int],
    actuator_offset: int,
    j1_ctrl: float,
) -> dict:
    """Set j1 ctrl on all 3 fingers and read out fingertip geometry."""
    # Reset to keyframe state to avoid drift from prior runs.
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)

    # Drive j1 of all 3 left-arm fingers; leave j2, j3 at default
    # (they passively curl due to coupling springs in the model).
    for ji in J1_INDICES:
        data.ctrl[actuator_offset + ji] = j1_ctrl

    # Long settle so the finger PD reaches near-steady-state.
    for _ in range(800):
        mujoco.mj_step(model, data)

    thumb_xyz = data.xpos[anchor_bids[0]].copy()
    b_xyz     = data.xpos[anchor_bids[1]].copy()
    c_xyz     = data.xpos[anchor_bids[2]].copy()
    bc_mid    = 0.5 * (b_xyz + c_xyz)

    # Pinch span measured in the horizontal plane (the side-grip
    # close motion is XY-dominated).
    span_xy = float(np.linalg.norm((thumb_xyz - bc_mid)[:2]))
    span_3d = float(np.linalg.norm(thumb_xyz - bc_mid))
    bc_sep  = float(np.linalg.norm((b_xyz - c_xyz)[:2]))

    return {
        "j1_ctrl": j1_ctrl,
        "thumb_xyz": thumb_xyz,
        "b_xyz": b_xyz,
        "c_xyz": c_xyz,
        "bc_mid": bc_mid,
        "span_xy": span_xy,
        "span_3d": span_3d,
        "bc_sep": bc_sep,
    }


def main():
    if not XML_PATH.exists():
        print(f"[FAIL] missing scene: {XML_PATH}", file=sys.stderr)
        sys.exit(1)

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data  = mujoco.MjData(model)

    anchor_bids = tuple(_body_id(model, n) for n in ANCHOR_BODY_NAMES)
    act_offset  = _resolve_left_actuator_offset(model)
    print(f"[probe] left-arm gripper actuator offset = {act_offset}")
    print(f"[probe] anchor bodies: "
          f"thumb={anchor_bids[0]}  b={anchor_bids[1]}  c={anchor_bids[2]}")
    print()

    # Sweep j1 ctrl across the full close stroke.  The grasp_executor
    # close stroke targets j1 ≈ 1.0-1.2 rad on the bc-fingers and
    # similar on the thumb (after the +/- sign flip).  We probe from
    # fully open (0.05, joint floor) to fully closed (1.22, joint
    # ceiling).
    j1_values = np.linspace(0.05, 1.22, 14)
    rows = []
    for v in j1_values:
        r = measure_aperture_at(model, data, anchor_bids, act_offset, float(v))
        rows.append(r)
        print(f"  j1={r['j1_ctrl']:.3f} rad   "
              f"span_xy={r['span_xy']*100:6.2f}cm  "
              f"bc_sep={r['bc_sep']*100:5.2f}cm  "
              f"max_obj_dia={r['span_xy']*100:6.2f}cm")
    print()

    # Now derive a productive radius range.  Define:
    #   max_obj_radius_at(ctrl) = span_xy(ctrl) / 2
    # The gripper can wrap an obj only when there is *closure margin
    # left* — i.e., the fingers can still curl further after first
    # contact.  We require ≥ 2 cm of margin, meaning the obj diameter
    # must be ≥ 2 cm smaller than the FULLY-OPEN span.
    open_span = rows[0]["span_xy"]
    full_close_span = rows[-1]["span_xy"]
    print(f"[derive] fully-open span_xy      = {open_span*100:.2f} cm "
          f"→ obj diameter must be ≤ {open_span*100:.2f} cm to enter pocket")
    print(f"[derive] fully-closed span_xy    = {full_close_span*100:.2f} cm "
          f"→ obj diameter must be ≥ {full_close_span*100:.2f} cm "
          f"to prevent over-close")
    print()

    # Productive zone: max obj diameter where there is 2-4 cm of
    # remaining closure margin (so bc-pair can still wrap inward).
    margin_lo = 0.02   # 2 cm — minimum closure margin
    margin_hi = 0.04   # 4 cm — comfortable margin
    r_min = (full_close_span + margin_lo) / 2.0
    r_max = (open_span - margin_hi) / 2.0

    print(f"[derive] geometry-based OBJ_RADIUS_RANGE = "
          f"({r_min:.3f}, {r_max:.3f})  m")
    print(f"[derive] in cm: ({r_min*100:.1f}, {r_max*100:.1f})")
    print()

    # Sanity-check the current settings.
    current_lo, current_hi = 0.075, 0.085
    print(f"[check] current play_m1.OBJ_RADIUS_RANGE = "
          f"({current_lo:.3f}, {current_hi:.3f})")
    if current_hi > r_max:
        print(f"[check] !! current upper bound {current_hi:.3f} m "
              f"exceeds geometry-derived upper bound {r_max:.3f} m")
        print(f"[check] !! → at obj radius {current_hi:.3f} m the "
              f"bc-pair has < {margin_hi*100:.0f} cm closure margin "
              f"and cannot reliably form a 3-point wrap")
    if current_lo < r_min:
        print(f"[check] !! current lower bound {current_lo:.3f} m "
              f"below geometry-derived lower bound {r_min:.3f} m")


if __name__ == "__main__":
    main()
