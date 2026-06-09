
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import mujoco


ROOT = Path(__file__).resolve().parents[1]
XML_PATH = ROOT / "src" / "env" / "market_world_m1.xml"

ANCHOR_BODY_NAMES = (
    "finger_a_link_3_1",
    "finger_b_link_3_1",
    "finger_c_link_3_1",
)

J1_INDICES = (0, 3, 6)
J2_INDICES = (1, 4, 7)
J3_INDICES = (2, 5, 8)


def _body_id(model: mujoco.MjModel, name: str) -> int:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise RuntimeError(f"body not found: {name}")
    return bid


def _resolve_left_actuator_offset(model: mujoco.MjModel) -> int:
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
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)

    for ji in J1_INDICES:
        data.ctrl[actuator_offset + ji] = j1_ctrl

    for _ in range(800):
        mujoco.mj_step(model, data)

    thumb_xyz = data.xpos[anchor_bids[0]].copy()
    b_xyz     = data.xpos[anchor_bids[1]].copy()
    c_xyz     = data.xpos[anchor_bids[2]].copy()
    bc_mid    = 0.5 * (b_xyz + c_xyz)

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

    open_span = rows[0]["span_xy"]
    full_close_span = rows[-1]["span_xy"]
    print(f"[derive] fully-open span_xy      = {open_span*100:.2f} cm "
          f"→ obj diameter must be ≤ {open_span*100:.2f} cm to enter pocket")
    print(f"[derive] fully-closed span_xy    = {full_close_span*100:.2f} cm "
          f"→ obj diameter must be ≥ {full_close_span*100:.2f} cm "
          f"to prevent over-close")
    print()

    margin_lo = 0.02
    margin_hi = 0.04
    r_min = (full_close_span + margin_lo) / 2.0
    r_max = (open_span - margin_hi) / 2.0

    print(f"[derive] geometry-based OBJ_RADIUS_RANGE = "
          f"({r_min:.3f}, {r_max:.3f})  m")
    print(f"[derive] in cm: ({r_min*100:.1f}, {r_max*100:.1f})")
    print()

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
