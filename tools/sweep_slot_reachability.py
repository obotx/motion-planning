
import os
import sys

import numpy as np
import mujoco

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from navigation.arm_planner import MORPHBridge  # noqa: E402

XML = os.path.join(ROOT, "src", "env", "market_world_m1.xml")

WRIST_GOAL = (0.0, -1.88, 0.80, 0.0)
WRIST_WEIGHT = (0.1, 3.0, 3.0, 3.0)
TARGET_BODY = "Gripper_Link3_1"

OBJ_HALF_HEIGHT_RANGE = (0.14, 0.16)
OBJ_HALF_HEIGHTS = (0.14, 0.16)

RACK_OPENING_Y = -3.41
STANDOFF_D = (0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 0.90)
FACING_SOUTH_YAW = -np.pi / 2.0

REACH_THRESHOLD = 0.05


def slot_world_positions(model, data):
    out = {}
    for i in range(10):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"shelf_slot_{i}")
        if sid >= 0:
            out[i] = data.site_xpos[sid].copy()
    return out


def palm_fk(bridge, q):
    pd = bridge.planning_data
    qmap = bridge.qpos_map
    for key, val in zip(
        ("ColumnLeft", "ColumnRight", "ArmLeft", "Base",
         "HandBearing", "WristZ", "WristX", "WristY"), q):
        pd.qpos[qmap[key]] = val
    mujoco.mj_forward(bridge.model, pd)
    bid = mujoco.mj_name2id(bridge.model, mujoco.mjtObj.mjOBJ_BODY, TARGET_BODY)
    return pd.xpos[bid].copy()


def sweep_slot(bridge, slot_xyz, half_height):
    target = np.array(slot_xyz, dtype=float)
    target[2] += half_height
    best = None
    for d in STANDOFF_D:
        chassis_x = float(target[0])
        chassis_y = RACK_OPENING_Y + d
        bridge.set_base_pose_xy_yaw(chassis_x, chassis_y, FACING_SOUTH_YAW)
        try:
            q = bridge.solve_ik(
                tuple(target),
                n_seeds=10,
                threshold=REACH_THRESHOLD,
                wrist_goal=WRIST_GOAL,
                wrist_weight=WRIST_WEIGHT,
                target_body=TARGET_BODY,
                validity_penalty_scale=50.0,
            )
        except Exception as exc:                # noqa: BLE001
            continue
        if q is None:
            continue
        palm = palm_fk(bridge, q)
        reach_err = float(np.linalg.norm(palm - target))
        try:
            valid = bool(bridge.is_valid(q))
        except Exception:                        # noqa: BLE001
            valid = False
        cand = dict(d=d, chassis_y=chassis_y, reach_err=reach_err,
                    valid=valid, q=list(q), palm=palm)
        key = (not valid, reach_err)
        if best is None or key < (not best["valid"], best["reach_err"]):
            best = cand
    return best


def main():
    print(f"Loading MORPHBridge: {XML}")
    bridge = MORPHBridge(XML, arm=1, use_calibration=False)
    data = mujoco.MjData(bridge.model)
    mujoco.mj_forward(bridge.model, data)
    slots = slot_world_positions(bridge.model, data)

    LEVEL = {0.247: "low", 0.687: "mid", 1.190: "high"}

    print()
    print("=" * 100)
    print("M2 SLOT REACHABILITY SWEEP  (object centre at slot, canonical side-grip "
          "wrist, north-aisle standoff)")
    print(f"Object half-height bracket tested per slot: {OBJ_HALF_HEIGHTS} m "
          f"(size-dynamic — §0.9)")
    print("=" * 100)
    hh_cols = "  ".join(f"hh={hh:.2f}" for hh in OBJ_HALF_HEIGHTS)
    print(f"{'slot':>4} {'level':>5} {'slot XYZ':>22}   {hh_cols:>22}   "
          f"{'verdict':>16}  best-config (smallest obj)")
    print("-" * 100)

    usable_all = {"low": [], "mid": [], "high": []}
    for i in sorted(slots):
        s = slots[i]
        lvl = LEVEL.get(round(float(s[2]), 3), "?")
        per_hh = {}
        for hh in OBJ_HALF_HEIGHTS:
            per_hh[hh] = sweep_slot(bridge, s, hh)
        def ok(b):
            return b is not None and b["valid"] and b["reach_err"] <= REACH_THRESHOLD
        flags = []
        for hh in OBJ_HALF_HEIGHTS:
            b = per_hh[hh]
            if ok(b):
                flags.append(f"{b['reach_err']*100:4.1f}cm")
            elif b is not None:
                flags.append(f"x{b['reach_err']*100:4.1f}cm")
            else:
                flags.append("  FAIL")
        all_ok = all(ok(per_hh[hh]) for hh in OBJ_HALF_HEIGHTS)
        any_ok = any(ok(per_hh[hh]) for hh in OBJ_HALF_HEIGHTS)
        if all_ok:
            verdict = "ALL SIZES"; usable_all[lvl].append(i)
        elif any_ok:
            verdict = "SOME SIZES"
        else:
            verdict = "NONE"
        ref = next((per_hh[hh] for hh in OBJ_HALF_HEIGHTS if ok(per_hh[hh])), per_hh[OBJ_HALF_HEIGHTS[0]])
        cfg = ""
        if ref is not None:
            q = ref["q"]
            cfg = (f"d={ref['chassis_y']:+.2f} a1={q[2]:.2f} "
                   f"h1={q[0]:.2f} h2={q[1]:.2f} hb={q[4]:+.2f}")
        print(f"{i:>4} {lvl:>5} ({s[0]:+.2f},{s[1]:+.2f},{s[2]:+.2f})   "
              f"{flags[0]:>10} {flags[1]:>10}   {verdict:>16}  {cfg}")

    print("-" * 100)
    print("SUMMARY — slots reachable for the WHOLE object-size bracket "
          f"(valid IK, err <= {REACH_THRESHOLD*100:.0f} cm at BOTH hh extremes):")
    for lvl in ("low", "mid", "high"):
        ids = usable_all[lvl]
        tot = sum(1 for i in slots if LEVEL.get(round(float(slots[i][2]),3)) == lvl)
        print(f"  {lvl:>4}: {len(ids)}/{tot} reachable for all sizes  {ids}")
    total = sum(len(v) for v in usable_all.values())
    print(f"  TOTAL: {total}/{len(slots)} slots usable for any picked object.")
    print()
    print("Notes:")
    print(" * IK target = object CENTRE (slot surface + half-height), not the")
    print("   shelf surface — the gripper holds the object at its mid-height.")
    print(" * 'x' prefix on an err = IK got that close but the config was invalid")
    print("   (self/chassis/shelf collision) or over threshold.")
    print(" * 'palm at object centre' is a reachability proxy; the place logic")
    print("   forward-biases for the pinch-to-palm offset + reads the real held")
    print("   object's half-height at runtime (size-dynamic).")


if __name__ == "__main__":
    main()
