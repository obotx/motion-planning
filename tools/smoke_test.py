import importlib
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
    for module_name in (
        "numpy",
        "scipy",
        "cv2",
        "glfw",
        "imgui",
        "zmq",
        "mujoco",
        "ompl",
    ):
        check_import(module_name)

    check_xml("src/env/market_world_m1.xml")

    print("[OK] M1 OMPL object-navigation environment is ready.")


if __name__ == "__main__":
    main()
