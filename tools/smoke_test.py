import importlib
import os
from pathlib import Path

import mujoco


ROOT = Path(__file__).resolve().parents[1]


def check_import(module_name):
    importlib.import_module(module_name)
    print(f"[OK] import {module_name}")


def check_xml(relative_path):
    path = ROOT / relative_path
    model = mujoco.MjModel.from_xml_path(str(path))
    print(f"[OK] load {relative_path}  bodies={model.nbody} geoms={model.ngeom}")


def main():
    for module_name in ("numpy", "scipy", "cv2", "glfw", "imgui", "zmq", "mujoco"):
        check_import(module_name)

    for relative_path in (
        "src/env/market_world_plain.xml",
        "src/env/market_world.xml",
        "src/env/market_world_m1.xml",
        "src/env/kitchen_world.xml",
    ):
        check_xml(relative_path)

    allow_missing_ompl = os.environ.get("ALLOW_MISSING_OMPL", "").lower() in {
        "1",
        "true",
        "yes",
    }
    try:
        check_import("ompl")
    except Exception as exc:
        if allow_missing_ompl:
            print(f"[WARN] import ompl failed: {exc}")
            print("[WARN] M1 OMPL object navigation requires OMPL.")
            return
        print(f"[FAIL] import ompl failed: {exc}")
        print("[FAIL] M1 OMPL object navigation requires OMPL.")
        print("[FAIL] For non-M1-only checks, run: ALLOW_MISSING_OMPL=1 make smoke")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
