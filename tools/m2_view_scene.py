
import os
import sys
import time

import numpy as np
import mujoco
import mujoco.viewer as mjv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from navigation.plan import OBSTACLE_RECTS, ROBOT_RADIUS  # noqa: E402

DEFAULT_XML = os.path.join(ROOT, "src", "env", "market_world_m1.xml")

_RECT_GOLD        = np.array([1.0, 0.84, 0.0, 0.30], dtype=np.float32)
_OVERLAY_Z_CENTER = 1.25
_OVERLAY_Z_HALF   = 1.25

SPAWN_ZONES = (
    ((0.25, 7.77), (-7.30, -4.70)),
    ((0.25, 7.77), (-3.30, -0.79)),
    ((0.25, 2.40), (-4.70, -3.30)),
    ((0.25, 2.40), (-0.79, -0.30)),
)
SPAWN_EXTRA_KEEP_RECTS = ()
SPAWN_ROBOT_KEEP_CENTER = (3.70, -6.00)
SPAWN_ROBOT_KEEP_RADIUS = 1.15
_SPAWN_ALLOW_GREEN = np.array([0.10, 0.80, 0.10, 0.28], dtype=np.float32)
_SPAWN_KEEP_RED    = np.array([0.85, 0.10, 0.10, 0.28], dtype=np.float32)
_SPAWN_Z_CENTER    = 0.006
_SPAWN_Z_HALF      = 0.004


def main(xml_path: str) -> None:
    print(f"Loading model: {xml_path}")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    print()
    print("=" * 72)
    print("M2 scene viewer — VISUAL vs COLLISION vs SLOT PLACEHOLDERS")
    print("=" * 72)
    print("Rack ACTUAL collision (red boxes, group 3):")
    print("  X = [+2.59, +4.28]  Y = [-7.93, -7.40]  Z = [+1.16, +1.64]")
    print("  (only 2 boxes — a solid block, NO shelf surfaces / NO slot openings)")
    print()
    print("Rack VISUAL footprint (gray meshes, group 2): spans most of the workspace")
    print("Slot placeholder sites (green dots, group 1): Y = -3.5  (~4 m off the rack)")
    print()
    print("Tip:  press 'V' to dim the visual mesh, '3' to toggle collision boxes,")
    print("      'T' for transparency, mouse-drag to orbit.")
    print("=" * 72)
    print()

    with mjv.launch_passive(model, data) as viewer:
        for i in range(6):
            viewer.opt.geomgroup[i] = 1
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False

        n_added = 0
        for x0, x1, y0, y1 in OBSTACLE_RECTS:
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            sx, sy = (x1 - x0) / 2.0, (y1 - y0) / 2.0
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[n_added],
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=np.array([sx, sy, _OVERLAY_Z_HALF], dtype=np.float64),
                pos=np.array([cx, cy, _OVERLAY_Z_CENTER], dtype=np.float64),
                mat=np.eye(3).flatten(),
                rgba=_RECT_GOLD,
            )
            n_added += 1
        for x0, x1, y0, y1 in (
                ((xr[0], xr[1], yr[0], yr[1]) for xr, yr in SPAWN_ZONES)):
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            sx, sy = (x1 - x0) / 2.0, (y1 - y0) / 2.0
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[n_added],
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=np.array([sx, sy, _SPAWN_Z_HALF], dtype=np.float64),
                pos=np.array([cx, cy, _SPAWN_Z_CENTER], dtype=np.float64),
                mat=np.eye(3).flatten(),
                rgba=_SPAWN_ALLOW_GREEN,
            )
            n_added += 1
        for x0, x1, y0, y1 in SPAWN_EXTRA_KEEP_RECTS:
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            sx, sy = (x1 - x0) / 2.0, (y1 - y0) / 2.0
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[n_added],
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=np.array([sx, sy, _SPAWN_Z_HALF], dtype=np.float64),
                pos=np.array([cx, cy, _SPAWN_Z_CENTER], dtype=np.float64),
                mat=np.eye(3).flatten(),
                rgba=_SPAWN_KEEP_RED,
            )
            n_added += 1
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[n_added],
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=np.array([SPAWN_ROBOT_KEEP_RADIUS, SPAWN_ROBOT_KEEP_RADIUS,
                           _SPAWN_Z_HALF], dtype=np.float64),
            pos=np.array([SPAWN_ROBOT_KEEP_CENTER[0], SPAWN_ROBOT_KEEP_CENTER[1],
                          _SPAWN_Z_CENTER], dtype=np.float64),
            mat=np.eye(3).flatten(),
            rgba=_SPAWN_KEEP_RED,
        )
        n_added += 1

        viewer.user_scn.ngeom = n_added
        print(f"Overlayed {len(OBSTACLE_RECTS)} OBSTACLE_RECTS (gold, chassis "
              f"clearance +{ROBOT_RADIUS:.2f} m implicit), "
              f"{len(SPAWN_ZONES)} spawn zones (green, allowed), "
              f"{len(SPAWN_EXTRA_KEEP_RECTS)+1} spawn keep-outs (red, including "
              f"robot-start circle r={SPAWN_ROBOT_KEEP_RADIUS:.2f} m at "
              f"{SPAWN_ROBOT_KEEP_CENTER}).")
        viewer.sync()

        while viewer.is_running():
            viewer.sync()
            time.sleep(1.0 / 60.0)


if __name__ == "__main__":
    xml = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_XML
    if not os.path.exists(xml):
        sys.exit(f"[error] XML not found: {xml}")
    main(xml)
