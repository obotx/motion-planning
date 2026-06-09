
import os
import sys

import numpy as np
import mujoco

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from navigation.arm_planner import (MORPHBridge, compute_carry_anchor_xyz,  # noqa: E402
                                    CARRY_ANCHOR_FINGER_BODIES)

XML = os.path.join(ROOT, "src", "env", "market_world_m1.xml")

WRIST_GOAL = (0.0, -1.88, 0.80, 0.0)
WRIST_WEIGHT = (0.1, 3.0, 3.0, 3.0)
RACK_OPENING_Y = -3.41
STANDOFF_D = (0.55, 0.60, 0.65, 0.70, 0.75)
FACING_SOUTH_YAW = -np.pi / 2.0

OBJ_HALF_HEIGHTS = (0.14, 0.16)
Z_SAFETY_MARGIN = 0.005

LEVEL_STANDOFF_CACHE = {}

PLACE_XY_PASS = 0.03
PLACE_XY_GATE = 0.05
PLACE_XY_HARD_ABORT = 0.12


def grade(err):
    if err <= PLACE_XY_PASS:
        return "precise"
    if err <= PLACE_XY_GATE:
        return "marginal"
    if err <= PLACE_XY_HARD_ABORT:
        return "approx"
    return "FAILED"


def main():
    print(f"Loading MORPHBridge: {XML}")
    bridge = MORPHBridge(XML, arm=1, use_calibration=False)
    model = bridge.model
    pd = bridge.planning_data
    finger_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
                  for n in CARRY_ANCHOR_FINGER_BODIES]

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    slots = {}
    for i in range(10):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"shelf_slot_{i}")
        if sid >= 0:
            slots[i] = data.site_xpos[sid].copy()
    LEVEL = {0.247: "low", 0.687: "mid", 1.190: "high"}

    def _solve_at(target, d):
        bridge.set_base_pose_xy_yaw(float(target[0]),
                                    RACK_OPENING_Y + d, FACING_SOUTH_YAW)
        q, _ = bridge.solve_ik_with_z_lift_carry_anchor(
            tuple(target), n_seeds=10, max_iters=4,
            wrist_goal=WRIST_GOAL, wrist_weight=WRIST_WEIGHT)
        if q is None:
            return None
        for k, idx in (("ColumnLeft", 0), ("ColumnRight", 1), ("ArmLeft", 2),
                       ("Base", 3), ("HandBearing", 4), ("WristZ", 5),
                       ("WristX", 6), ("WristY", 7)):
            pd.qpos[bridge.qpos_map[k]] = q[idx]
        mujoco.mj_forward(model, pd)
        anchor = compute_carry_anchor_xyz(pd, finger_ids)
        err_xy = float(np.linalg.norm(anchor[:2] - target[:2]))
        err_z = float(abs(anchor[2] - target[2]))
        valid = bool(bridge.is_valid(q))
        return (err_xy, err_z, valid, d, list(q))

    def place_err_at(target, level):
        cached = LEVEL_STANDOFF_CACHE.get(level)
        if cached is not None:
            r = _solve_at(target, cached)
            if r is not None and r[0] <= PLACE_XY_GATE:
                return r
        best = None
        for d in STANDOFF_D:
            r = _solve_at(target, d)
            if r is None:
                continue
            if best is None or r[0] < best[0]:
                best = r
        if best is not None:
            LEVEL_STANDOFF_CACHE[level] = best[3]
        return best

    print()
    print("=" * 98)
    print("M2 PLACEMENT-FEASIBILITY SWEEP  (carry pocket at object centre, per "
          "slot x object-size)")
    print("=" * 98)
    print(f"{'slot':>4} {'level':>5} {'hh':>5} {'XYerr':>7} {'Zerr':>6} "
          f"{'valid':>6} {'grade':>9}  {'standoff':>8}")
    print("-" * 98)

    ready = []
    for i in sorted(slots):
        s = slots[i]
        lvl = LEVEL.get(round(float(s[2]), 3), "?")
        grades = []
        for hh in OBJ_HALF_HEIGHTS:
            target = s.copy()
            target[2] += hh + Z_SAFETY_MARGIN
            best = place_err_at(target, lvl)
            if best is None:
                print(f"{i:>4} {lvl:>5} {hh:5.2f} {'NO-IK':>7}")
                grades.append("FAILED")
                continue
            err_xy, err_z, valid, d, q = best
            g = grade(err_xy) if valid else "FAILED"
            grades.append(g)
            print(f"{i:>4} {lvl:>5} {hh:5.2f} {err_xy*100:6.1f} {err_z*100:5.1f} "
                  f"{str(valid):>6} {g:>9}  {RACK_OPENING_Y + d:+8.2f}")
        if all(g == "precise" for g in grades):
            ready.append(i)

    print("-" * 98)
    print(f"SUMMARY — slots PRECISE (<= {PLACE_XY_PASS*100:.0f} cm XY) for ALL "
          f"object sizes: {len(ready)}/{len(slots)}  {ready}")
    print()
    print("Grades (§5e): precise <=3cm | marginal 3-5cm (retry) | "
          "approx 5-12cm (drop fallback) | FAILED >12cm or invalid.")
    print("XYerr = carry-pocket XY vs slot; Zerr = pocket Z vs object-centre Z.")


if __name__ == "__main__":
    main()
