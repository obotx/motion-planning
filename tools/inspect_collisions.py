"""inspect_collisions.py — print actual MuJoCo collision geometry for the scene.

Prints world-axis-aligned bounding boxes (AABBs) for the rack, walls,
chassis, and other key bodies, then compares them to the 2D obstacle
bounding boxes the OMPL base planner uses (`navigation/plan.py:OBSTACLE_RECTS`).

Why this matters: OMPL plans paths assuming the obstacles are exactly
the rectangles in `OBSTACLE_RECTS`, inflated by `ROBOT_RADIUS`.  But
MuJoCo's collision check uses the actual geom geometry, which can
extend past those rectangles (overhangs, structural members).  When
the two disagree, OMPL plans paths that MuJoCo then rejects — exactly
the rack-graze rejection failure mode we hit in M1.

Run from the project root:

    docker compose -f docker/docker-compose.yml run --rm motion-planning \
        python3 tools/inspect_collisions.py

Or natively (Python 3.10 venv with mujoco + numpy):

    PYTHONPATH=src python3 tools/inspect_collisions.py
"""

import os
import sys

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

# Same constants the OMPL base planner uses.
from navigation.plan import OBSTACLE_RECTS, ROBOT_RADIUS


# ---------------------------------------------------------------------------

def _box_corners(half):
    """8 corners of a box with half-extents `half` in local frame."""
    return np.array([
        [s * half[0], t * half[1], u * half[2]]
        for s in (-1.0, 1.0)
        for t in (-1.0, 1.0)
        for u in (-1.0, 1.0)
    ], dtype=float)


def geom_world_aabb(model, data, gid):
    """Return (min_xyz, max_xyz) world AABB for geom gid, or (None, None)
    if the geom type isn't supported (e.g. plane, hfield)."""
    gtype = model.geom_type[gid]
    pos = data.geom_xpos[gid].copy()
    R = data.geom_xmat[gid].reshape(3, 3)

    if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
        r = float(model.geom_size[gid, 0])
        return pos - r, pos + r

    if gtype == mujoco.mjtGeom.mjGEOM_BOX:
        half = model.geom_size[gid, :3].astype(float)
        corners = _box_corners(half) @ R.T + pos
        return corners.min(axis=0), corners.max(axis=0)

    if gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
        r = float(model.geom_size[gid, 0])
        h = float(model.geom_size[gid, 1])
        ep1 = pos + R @ np.array([0.0, 0.0, h])
        ep2 = pos - R @ np.array([0.0, 0.0, h])
        return np.minimum(ep1, ep2) - r, np.maximum(ep1, ep2) + r

    if gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
        # AABB approximated as the bounding box of an oriented cylinder
        # treated as a box (2r, 2r, 2h).  Conservative.
        r = float(model.geom_size[gid, 0])
        h = float(model.geom_size[gid, 1])
        corners = _box_corners(np.array([r, r, h])) @ R.T + pos
        return corners.min(axis=0), corners.max(axis=0)

    if gtype == mujoco.mjtGeom.mjGEOM_MESH:
        mesh_id = int(model.geom_dataid[gid])
        if mesh_id < 0:
            return None, None
        # mesh_aabb format: (center_x, center_y, center_z, half_x, half_y, half_z)
        aabb = model.mesh_aabb[mesh_id]
        center = np.asarray(aabb[:3], dtype=float)
        half = np.asarray(aabb[3:], dtype=float)
        # 8 mesh-local corners → world
        corners_local = _box_corners(half) + center
        corners_world = corners_local @ R.T + pos
        return corners_world.min(axis=0), corners_world.max(axis=0)

    return None, None


def body_world_aabb(model, data, body_id, include_descendants=True):
    """Union AABB over all geoms attached to body_id (and descendants if asked)."""
    body_ids = [body_id]
    if include_descendants:
        # Recursively collect descendant body IDs
        i = 0
        while i < len(body_ids):
            bid = body_ids[i]
            for j in range(model.nbody):
                if int(model.body_parentid[j]) == bid and j != bid:
                    body_ids.append(j)
            i += 1

    aabb_min = np.array([np.inf, np.inf, np.inf])
    aabb_max = np.array([-np.inf, -np.inf, -np.inf])
    geom_count = 0
    for bid in body_ids:
        for gid in range(model.ngeom):
            if int(model.geom_bodyid[gid]) != bid:
                continue
            # Skip non-collision geoms (visual-only, contype=0 AND conaffinity=0)
            if int(model.geom_contype[gid]) == 0 and int(model.geom_conaffinity[gid]) == 0:
                continue
            mn, mx = geom_world_aabb(model, data, gid)
            if mn is None:
                continue
            aabb_min = np.minimum(aabb_min, mn)
            aabb_max = np.maximum(aabb_max, mx)
            geom_count += 1
    if geom_count == 0:
        return None, None, 0
    return aabb_min, aabb_max, geom_count


def _fmt_aabb(mn, mx):
    if mn is None:
        return "(empty)"
    return (f"X=[{mn[0]:+.3f}, {mx[0]:+.3f}]  "
            f"Y=[{mn[1]:+.3f}, {mx[1]:+.3f}]  "
            f"Z=[{mn[2]:+.3f}, {mx[2]:+.3f}]")


def _name_of(model, kind, idx):
    return mujoco.mj_id2name(model, kind, idx) or f"<{kind}:{idx}>"


def _bid(model, name):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    return bid if bid >= 0 else None


# ---------------------------------------------------------------------------

def main(xml_path):
    print(f"Loading model: {xml_path}")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    print(f"Model: {model.nbody} bodies, {model.ngeom} geoms\n")

    # ---- Section 1: rack body, full breakdown ----
    print("=" * 72)
    print("RACK COLLISION GEOMETRY")
    print("=" * 72)
    rack_bid = _bid(model, "minimarket_racks")
    if rack_bid is None:
        print("[WARN] body 'minimarket_racks' not found")
    else:
        print(f"Body: minimarket_racks (id={rack_bid})")
        # Per-geom breakdown for the rack and descendants
        per_geom = []
        body_ids_under_rack = [rack_bid]
        i = 0
        while i < len(body_ids_under_rack):
            bid = body_ids_under_rack[i]
            for j in range(model.nbody):
                if int(model.body_parentid[j]) == bid and j != bid:
                    body_ids_under_rack.append(j)
            i += 1
        for bid in body_ids_under_rack:
            for gid in range(model.ngeom):
                if int(model.geom_bodyid[gid]) != bid:
                    continue
                if int(model.geom_contype[gid]) == 0 and int(model.geom_conaffinity[gid]) == 0:
                    continue
                mn, mx = geom_world_aabb(model, data, gid)
                if mn is None:
                    continue
                gname = _name_of(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
                gtype = model.geom_type[gid]
                type_name = {
                    mujoco.mjtGeom.mjGEOM_BOX:      "box",
                    mujoco.mjtGeom.mjGEOM_SPHERE:   "sphere",
                    mujoco.mjtGeom.mjGEOM_CAPSULE:  "capsule",
                    mujoco.mjtGeom.mjGEOM_CYLINDER: "cylinder",
                    mujoco.mjtGeom.mjGEOM_MESH:     "mesh",
                }.get(gtype, f"type{gtype}")
                per_geom.append((gid, gname, type_name, mn, mx))
        if per_geom:
            print(f"  {len(per_geom)} collision geoms:")
            for gid, gname, type_name, mn, mx in per_geom:
                print(f"    geom {gid:3d}  {type_name:8s}  '{gname}'  {_fmt_aabb(mn, mx)}")
        # Union
        rmn, rmx, n = body_world_aabb(model, data, rack_bid)
        print(f"\n  Union AABB ({n} geoms):  {_fmt_aabb(rmn, rmx)}")

    # ---- Section 2: OMPL OBSTACLE_RECTS comparison ----
    print()
    print("=" * 72)
    print("OMPL PLAN.PY OBSTACLE_RECTS (what the planner thinks the world is)")
    print("=" * 72)
    print(f"ROBOT_RADIUS (inflation):  {ROBOT_RADIUS:.3f} m\n")
    for i, (x0, x1, y0, y1) in enumerate(OBSTACLE_RECTS):
        print(f"  rect[{i}]:  X=[{x0:+.3f}, {x1:+.3f}]  Y=[{y0:+.3f}, {y1:+.3f}]")

    # ---- Section 3: rack mismatch report ----
    if rack_bid is not None and rmn is not None:
        print()
        print("=" * 72)
        print("RACK: ACTUAL vs OMPL")
        print("=" * 72)
        # Find the OMPL rect that best matches the rack (largest XY overlap)
        rack_xy_actual = (rmn[0], rmx[0], rmn[1], rmx[1])
        rack_rect_idx = None
        best_overlap = -1
        for i, rect in enumerate(OBSTACLE_RECTS):
            ox = max(0, min(rect[1], rack_xy_actual[1]) - max(rect[0], rack_xy_actual[0]))
            oy = max(0, min(rect[3], rack_xy_actual[3]) - max(rect[2], rack_xy_actual[2]))
            if ox * oy > best_overlap:
                best_overlap = ox * oy
                rack_rect_idx = i
        if rack_rect_idx is not None:
            ompl = OBSTACLE_RECTS[rack_rect_idx]
            actual_x0, actual_x1, actual_y0, actual_y1 = rack_xy_actual
            ompl_x0, ompl_x1, ompl_y0, ompl_y1 = ompl

            def diff(a, o, label):
                d = a - o
                sign = "+" if d > 0 else ""
                tag = (" — actual EXTENDS past OMPL" if abs(d) > 0.005 and (
                    (label.endswith("min") and d < 0) or
                    (label.endswith("max") and d > 0)
                ) else "")
                return f"  {label}: actual={a:+.3f}  ompl={o:+.3f}  Δ={sign}{d*100:.1f}cm{tag}"

            print(f"OMPL rect[{rack_rect_idx}] (best XY overlap with rack):")
            print(diff(actual_x0, ompl_x0, "X-min (west edge) "))
            print(diff(actual_x1, ompl_x1, "X-max (east edge) "))
            print(diff(actual_y0, ompl_y0, "Y-min (south edge)"))
            print(diff(actual_y1, ompl_y1, "Y-max (north edge)"))

            x_under = max(0.0, ompl_x0 - actual_x0) + max(0.0, actual_x1 - ompl_x1)
            y_under = max(0.0, ompl_y0 - actual_y0) + max(0.0, actual_y1 - ompl_y1)
            max_extension = max(
                ompl_x0 - actual_x0,
                actual_x1 - ompl_x1,
                ompl_y0 - actual_y0,
                actual_y1 - ompl_y1,
            )
            print(f"\n  Total extension past OMPL box: "
                  f"X=±{x_under * 100 / 2:.1f}cm  Y=±{y_under * 100 / 2:.1f}cm")
            if max_extension > 0.005:
                print(f"\n  [RECOMMENDATION] Either:")
                margin = max_extension + 0.02
                print(f"  - Bump ROBOT_RADIUS by ≥ {margin*100:.0f}cm  "
                      f"(currently {ROBOT_RADIUS*100:.0f}cm), OR")
                new_rect = (
                    min(actual_x0, ompl_x0) - 0.01,
                    max(actual_x1, ompl_x1) + 0.01,
                    min(actual_y0, ompl_y0) - 0.01,
                    max(actual_y1, ompl_y1) + 0.01,
                )
                print(f"  - Expand OBSTACLE_RECTS rack to "
                      f"({new_rect[0]:.2f}, {new_rect[1]:.2f}, "
                      f"{new_rect[2]:.2f}, {new_rect[3]:.2f}) "
                      "and keep ROBOT_RADIUS as-is.")
            else:
                print("\n  [OK] OMPL box matches actual rack within 0.5cm. No change needed.")

    # ---- Section 4: chassis (robot body) AABB ----
    print()
    print("=" * 72)
    print("CHASSIS COLLISION GEOMETRY")
    print("=" * 72)
    robot_bid = _bid(model, "robot")
    if robot_bid is None:
        print("[WARN] body 'robot' not found")
    else:
        # The robot body's direct geoms only (NOT descendants — those are
        # the arm chain, which the validator handles via allowed-pairs).
        # We want the chassis footprint specifically.
        cmn = np.array([np.inf, np.inf, np.inf])
        cmx = np.array([-np.inf, -np.inf, -np.inf])
        n = 0
        for gid in range(model.ngeom):
            if int(model.geom_bodyid[gid]) != robot_bid:
                continue
            if int(model.geom_contype[gid]) == 0 and int(model.geom_conaffinity[gid]) == 0:
                continue
            mn, mx = geom_world_aabb(model, data, gid)
            if mn is None:
                continue
            cmn = np.minimum(cmn, mn)
            cmx = np.maximum(cmx, mx)
            n += 1
        if n > 0:
            half_x = (cmx[0] - cmn[0]) / 2
            half_y = (cmx[1] - cmn[1]) / 2
            longest_radius = float(np.hypot(half_x, half_y))
            print(f"Chassis direct geoms ({n}):  {_fmt_aabb(cmn, cmx)}")
            print(f"  half-extent X={half_x*100:.1f}cm  half-extent Y={half_y*100:.1f}cm")
            print(f"  diagonal half-extent (worst yaw): {longest_radius*100:.1f}cm")
            print(f"\n  ROBOT_RADIUS={ROBOT_RADIUS*100:.0f}cm — "
                  f"{'OK (≥ diagonal)' if ROBOT_RADIUS >= longest_radius else 'INSUFFICIENT (< diagonal)'}")
        else:
            print("[WARN] no direct collision geoms on body 'robot'")

    # ---- Section 5: pickup objects (Z range) ----
    print()
    print("=" * 72)
    print("PICKUP OBJECT GEOMETRY (height range across spawned objects)")
    print("=" * 72)
    obj_zs = []
    obj_radii = []
    for i in range(20):
        bid = _bid(model, f"pickup_obj_{i}")
        if bid is None:
            continue
        mn, mx, _ = body_world_aabb(model, data, bid, include_descendants=False)
        if mn is None:
            continue
        h = mx[2] - mn[2]
        r = max(mx[0] - mn[0], mx[1] - mn[1]) / 2
        obj_zs.append(h)
        obj_radii.append(r)
    if obj_zs:
        print(f"Objects found: {len(obj_zs)}")
        print(f"  height range: {min(obj_zs)*100:.1f} - {max(obj_zs)*100:.1f}cm")
        print(f"  radius range: {min(obj_radii)*100:.1f} - {max(obj_radii)*100:.1f}cm")
    else:
        print("[INFO] no pickup_obj_* bodies found at this snapshot")

    print()
    print("Done.")


if __name__ == "__main__":
    xml_default = os.path.join(ROOT, "src", "env", "market_world_m1.xml")
    xml_path = sys.argv[1] if len(sys.argv) > 1 else xml_default
    if not os.path.exists(xml_path):
        sys.exit(f"[error] XML not found: {xml_path}")
    main(xml_path)
