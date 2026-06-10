#!/usr/bin/env python3
import os
import sys
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))

COMMON = {
    "AH_DOCK_VEL": "9", "AH_DOCK_D_MAX": "0.78", "AH_DOCK_TWOLEG": "1",
    "AH_DOCK_YAW_PROG": "1", "AH_DOCK_SLEW": "0.10", "AH_DOCK_TOL": "0.08",
    "AH_DOCK_OVERSHOOT": "0.0", "AH_STALL_TIMEOUT": "16",
    "AH_FINAL_YAW_TOL": "0.06", "AH_PICKUP_YAW_TOL": "0.175",
    "AH_TRANSPORT_PIN": "1", "AH_PIN_SUBSTEP": "1", "AH_CARRY_NO_OBJ_COL": "1",
    "AH_CARRY_WRAP_FINGERS": "1",
    "AH_CARRY_LOW_IMPRATIO": "0", "AH_CARRY_NO_BATTERY_COL": "1",
    "AH_CARRY_NO_ARM2_COL": "1", "AH_NO_ARM2_COL_GLOBAL": "1", "AH_RIGID_WELD": "0",
    "AH_PLACE_NO_OBJ_COL": "1", "AH_PLACE_SERVO": "1", "AH_PLACE_RUNTIME_REACH": "1",
    "AH_PLACE_RAM_FORCE": "250", "AH_PLACE_LAT_STRAFE": "1", "AH_PLACE_SERVO_ITERS": "40",
    "AH_PLACE_COL_STEP": "0.05", "AH_PLACE_Z_KP": "0.9",
    "AH_PLACE_SMOOTH_LIFT": "1",
    "AH_PLACE_ARM_ONLY": "1",
    "AH_PLACE_A1_STEP_FAR": "0.015",
    "AH_PLACE_LOW_IMPRATIO": "1", "AH_PLACE_IMPRATIO": "3",
    "AH_SLOT_FWD_SHIFT": "0.11",
}


def _has(flag):
    return any(t == flag or t.startswith(flag + "=") for t in sys.argv[1:])


def main():
    env = dict(os.environ)
    env.setdefault("DISPLAY", ":1")
    env.setdefault("PYTHONPATH", "src")
    for k, v in COMMON.items():
        env.setdefault(k, v)

    args = list(sys.argv[1:])
    if not _has("--use-calib"):
        args = ["--use-calib"] + args

    py = os.path.join(HERE, ".venv", "bin", "python3")
    if not os.path.exists(py):
        py = sys.executable
    cmd = [py, "-u", os.path.join(HERE, "src", "gui", "play_m1.py")] + args

    print("[play] GUI ready — click MOVE, pick an object + a shelf slot. "
          "Per-level tuning auto-applies from the slot you choose.")
    return subprocess.call(cmd, cwd=HERE, env=env)


if __name__ == "__main__":
    sys.exit(main())
