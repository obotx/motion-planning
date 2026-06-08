
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

RACK_OPENING_Y = -3.41
STANDOFF_D = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.85)
FACING_SOUTH_YAW = -np.pi / 2.0

OBJ_HALF_HEIGHT = 0.16
APPROACH_RETRACT_A1 = 0.20
MIN_PICK_A1 = 0.16
INSERT_STEPS = 12
PROBE_MARGIN = 0.06

ARM_BODY_KEYS = ("Arm", "Column", "Gripper", "finger", "Wrist",
                 "HandBearing", "Rotation_Link", "Bearing_Column")


def is_arm_geom(model, g):
    nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or ""
    return any(k in nm for k in ARM_BODY_KEYS) and "_2" not in nm


def obstacle_geom_ids(model):
    ids = set()
    for g in range(model.ngeom):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        bnm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or ""
        if nm.startswith("m2_shelf_") or "rack" in bnm.lower() or "rack" in nm.lower():
            if int(model.geom_contype[g]) != 0 or int(model.geom_conaffinity[g]) != 0:
                ids.add(g)
    return ids


def set_arm(bridge, q):
    pd = bridge.planning_data
    qm = bridge.qpos_map
    for k, v in zip(("ColumnLeft", "ColumnRight", "ArmLeft", "Base",
                     "HandBearing", "WristZ", "WristX", "WristY"), q):
        pd.qpos[qm[k]] = v
    mujoco.mj_forward(bridge.model, pd)


def min_clearance(bridge, obstacle_ids):
    pd = bridge.planning_data
    m = bridge.model
    worst = None
    worst_pair = None
    for c in range(pd.ncon):
        g1, g2 = pd.contact[c].geom1, pd.contact[c].geom2
        a1 = g1 in obstacle_ids and is_arm_geom(m, g2)
        a2 = g2 in obstacle_ids and is_arm_geom(m, g1)
        if not (a1 or a2):
            continue
        dist = float(pd.contact[c].dist)
        if worst is None or dist < worst:
            worst = dist
            og = g1 if g1 in obstacle_ids else g2
            ag = g2 if g1 in obstacle_ids else g1
            worst_pair = (f"{mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, ag)} "
                          f"<-> {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, og)}")
    return worst, worst_pair


def solve_inserted(bridge, target):
    best = None
    for d in STANDOFF_D:
        bridge.set_base_pose_xy_yaw(float(target[0]), RACK_OPENING_Y + d, FACING_SOUTH_YAW)
        try:
            q = bridge.solve_ik(tuple(target), n_seeds=10, threshold=0.05,
                                wrist_goal=WRIST_GOAL, wrist_weight=WRIST_WEIGHT,
                                target_body=TARGET_BODY, validity_penalty_scale=50.0)
        except Exception:    # noqa: BLE001
            continue
        if q is None:
            continue
        set_arm(bridge, q)
        bid = mujoco.mj_name2id(bridge.model, mujoco.mjtObj.mjOBJ_BODY, TARGET_BODY)
        err = float(np.linalg.norm(bridge.planning_data.xpos[bid] - target))
        if err <= 0.05 and (best is None or err < best[2]):
            best = (list(q), d, err)
    if best is None:
        return None, None
    return best[0], best[1]


def main():
    print(f"Loading MORPHBridge: {XML}")
    bridge = MORPHBridge(XML, arm=1, use_calibration=False)
    model = bridge.model

    obstacles = obstacle_geom_ids(model)
    for g in obstacles:
        model.geom_margin[g] = max(float(model.geom_margin[g]), PROBE_MARGIN)
    print(f"Probing clearance against {len(obstacles)} shelf/rack geoms "
          f"(margin={PROBE_MARGIN*100:.0f} cm).")

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    slots = {}
    for i in range(10):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"shelf_slot_{i}")
        if sid >= 0:
            slots[i] = data.site_xpos[sid].copy()
    LEVEL = {0.247: "low", 0.687: "mid", 1.190: "high"}

    print()
    print("=" * 96)
    print(f"M2 INSERTION-CLEARANCE SWEEP  (Phase-L a1-extension, tallest object "
          f"hh={OBJ_HALF_HEIGHT} m)")
    print("=" * 96)
    print(f"{'slot':>4} {'level':>5} {'insert?':>8} {'min-clear':>10} "
          f"{'@step':>6}  worst arm<->obstacle pair")
    print("-" * 96)

    clear_ok = []
    for i in sorted(slots):
        s = slots[i]
        lvl = LEVEL.get(round(float(s[2]), 3), "?")
        target = s.copy()
        target[2] += OBJ_HALF_HEIGHT
        q_in, d = solve_inserted(bridge, target)
        if q_in is None:
            print(f"{i:>4} {lvl:>5} {'NO-IK':>8}")
            continue
        q_pre = list(q_in)
        q_pre[2] = max(MIN_PICK_A1, q_in[2] - APPROACH_RETRACT_A1)
        worst_gap = None
        worst_step = -1
        worst_pair = None
        for k in range(INSERT_STEPS + 1):
            alpha = k / INSERT_STEPS
            q = list(q_in)
            q[2] = q_pre[2] + alpha * (q_in[2] - q_pre[2])
            set_arm(bridge, q)
            gap, pair = min_clearance(bridge, obstacles)
            if gap is not None and (worst_gap is None or gap < worst_gap):
                worst_gap = gap
                worst_step = k
                worst_pair = pair
        if worst_gap is None:
            verdict = "CLEAR"
            clear_ok.append(i)
            print(f"{i:>4} {lvl:>5} {verdict:>8} {'>'+str(int(PROBE_MARGIN*100))+'cm':>10}")
        else:
            hit = worst_gap < 0.0
            verdict = "HIT" if hit else "tight"
            if not hit:
                clear_ok.append(i)
            print(f"{i:>4} {lvl:>5} {verdict:>8} {worst_gap*100:9.1f} "
                  f"{worst_step:6d}  {worst_pair}")

    print("-" * 96)
    print(f"SUMMARY — slots with a collision-free Phase-L insertion: "
          f"{len(clear_ok)}/{len(slots)}  {clear_ok}")
    print()
    print("Notes:")
    print(" * 'CLEAR'  = no arm/gripper geom within "
          f"{PROBE_MARGIN*100:.0f} cm of any shelf/rack geom along the insertion.")
    print(" * 'tight'  = closest approach positive but < "
          f"{PROBE_MARGIN*100:.0f} cm (usable, watch margin).")
    print(" * 'HIT'    = arm/gripper penetrates a shelf/rack geom (negative gap)"
          " — insertion path needs adjustment for that slot.")
    print(" * Tested at the TALLEST object (hh=0.16) = conservative for the")
    print("   arm-under-board clearance; smaller objects have more room.")


if __name__ == "__main__":
    main()
