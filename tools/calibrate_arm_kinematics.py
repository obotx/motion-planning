"""calibrate_arm_kinematics.py — generate the MORPH-I arm calibration LUT.

The MORPH-I arm has a passive `RotationLeftJoint` whose qpos is set
by physics dynamics under gravity + the parallel-mechanism constraint.
The IK solver in `arm_planner.solve_ik` uses `mj_forward` (kinematics
only), which leaves the passive joint at its initial qpos value and
therefore produces a Link3 (palm) world position that does NOT match
where Link3 lands once `mj_step` runs and the passive joint settles.
Empirically this gap is 4-8cm at typical floor-pick poses.

This script measures that gap on a grid of arm joint values and
writes the result to `data/arm_calibration.npz`, which the planner
loads at startup and uses to pre-correct IK targets.

Output schema (numpy npz):
    h1_grid:   1-D array of h1 values sampled
    h2_grid:   1-D array of h2 values sampled
    a1_grid:   1-D array of a1 values sampled
    error:     4-D array shape (len_h1, len_h2, len_a1, 3) — chassis-
               relative XYZ deflection (physics − kinematics) at theta=0.

Run from the project root:

    docker compose -f docker/docker-compose.yml run --rm motion-planning \
        python3 tools/calibrate_arm_kinematics.py

Takes ~5-15 minutes on a 7×7×7 grid (343 poses, ~2s each).  The LUT
file is small (~10 KB) and is loaded once at planner init.
"""

import os
import sys
import time

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

# Reuse joint-name conventions from the planner so we calibrate exactly
# the same arm the IK targets.
ARM = 1                                # ARM1 == left arm chain
SETTLE_STEPS = 200                     # mj_step iterations to settle physics
SETTLE_DT_TOL = 1e-5                   # velocity threshold for "settled"

# Grid resolution.  Higher = more accurate but slower (n³ poses).  7×7×7
# = 343 poses takes ~10 minutes on typical hardware; 5×5×5 = 125 poses
# in ~4 minutes is a quicker first pass.
GRID_H1 = np.linspace(0.05, 0.30, 7)   # column heights typical for floor pick
GRID_H2 = np.linspace(0.05, 0.30, 7)
GRID_A1 = np.linspace(0.10, 0.55, 7)   # boom extension range


def _qi(model, jname):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
    if jid < 0:
        raise RuntimeError(f"Joint '{jname}' not found")
    return int(model.jnt_qposadr[jid])


def _bid(model, bname):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
    if bid < 0:
        raise RuntimeError(f"Body '{bname}' not found")
    return bid


def calibrate(xml_path, out_path):
    print(f"[calib] Loading model: {xml_path}")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data_kin = mujoco.MjData(model)   # kinematics-only
    data_phys = mujoco.MjData(model)  # physics-stepped

    # Resolve joint qpos addresses for the configured arm.  Both arms
    # use a `_{N}` suffix in the model XML.
    suffix = f"_{ARM}"
    h1_q = _qi(model, f"ColumnLeftBearingJoint{suffix}")
    h2_q = _qi(model, f"ColumnRightBearingJoint{suffix}")
    a1_q = _qi(model, f"ArmLeftJoint{suffix}")
    th_q = _qi(model, f"BaseJoint{suffix}")
    palm_bid = _bid(model, "Gripper_Link3_1")
    chassis_bid = _bid(model, "robot")

    # Build a "passive joint qpos mask".  We pin EVERYTHING in qpos
    # except the one passive joint we want to settle (RotationLeftJoint
    # for the configured arm).  Without this, the model's other 140+
    # qpos values (other arm, 10 spawned objects, every other body's
    # freejoint) all evolve under physics during mj_step — they can
    # collide with the arm and corrupt our measurement.  By pinning
    # only the active arm and chassis is automatic since they're
    # included in the "everything except passive" mask.
    passive_q = _qi(model, f"RotationLeftJoint{suffix}")
    n_qpos = int(model.nq)
    pin_mask = np.ones(n_qpos, dtype=bool)
    pin_mask[passive_q] = False  # let only this one evolve
    print(f"[calib] passive joint RotationLeftJoint{suffix} qposadr={passive_q}")
    print(f"[calib] Pinning {int(pin_mask.sum())}/{n_qpos} qpos values "
          f"every step; only RotationLeftJoint settles freely.")

    print(f"[calib] qpos_addr h1={h1_q} h2={h2_q} a1={a1_q} theta={th_q}")
    print(f"[calib] palm_body_id={palm_bid}  chassis_body_id={chassis_bid}")
    print(f"[calib] Grid: {len(GRID_H1)}×{len(GRID_H2)}×{len(GRID_A1)} = "
          f"{len(GRID_H1)*len(GRID_H2)*len(GRID_A1)} poses")

    error = np.zeros(
        (len(GRID_H1), len(GRID_H2), len(GRID_A1), 3), dtype=np.float64)
    invalid = np.zeros(error.shape[:3], dtype=bool)

    t0 = time.time()
    n_total = error.shape[0] * error.shape[1] * error.shape[2]
    n_done = 0

    for i, h1 in enumerate(GRID_H1):
        for j, h2 in enumerate(GRID_H2):
            for k, a1 in enumerate(GRID_A1):
                # -- Kinematics-only prediction --
                # Project Link3 into the chassis local frame so chassis
                # world position AND orientation cancel.  Without the
                # rotation step, a chassis that spins under physics
                # produces "deflection" of up to ~1m purely from arm
                # length sweeping with the rotated chassis frame.
                mujoco.mj_resetData(model, data_kin)
                data_kin.qpos[h1_q] = h1
                data_kin.qpos[h2_q] = h2
                data_kin.qpos[a1_q] = a1
                data_kin.qpos[th_q] = 0.0
                mujoco.mj_forward(model, data_kin)
                R_kin = data_kin.xmat[chassis_bid].reshape(3, 3)
                rel_kin = (data_kin.xpos[palm_bid].copy()
                           - data_kin.xpos[chassis_bid].copy())
                p_kin = R_kin.T @ rel_kin

                # -- Physics-settled measurement --
                mujoco.mj_resetData(model, data_phys)
                data_phys.qpos[h1_q] = h1
                data_phys.qpos[h2_q] = h2
                data_phys.qpos[a1_q] = a1
                data_phys.qpos[th_q] = 0.0
                # Snapshot the FULL qpos vector now that we've placed
                # the arm at the test pose.  Every step we restore all
                # pinned qpos values (i.e. everything except the
                # RotationLeftJoint) and zero all qvel.  The passive
                # joint is the only thing physics is allowed to move.
                qpos_init = data_phys.qpos.copy()
                for _ in range(SETTLE_STEPS):
                    data_phys.qpos[pin_mask] = qpos_init[pin_mask]
                    data_phys.qvel[:] = 0.0
                    mujoco.mj_step(model, data_phys)
                R_phys = data_phys.xmat[chassis_bid].reshape(3, 3)
                rel_phys = (data_phys.xpos[palm_bid].copy()
                            - data_phys.xpos[chassis_bid].copy())
                p_phys = R_phys.T @ rel_phys

                err = p_phys - p_kin
                error[i, j, k] = err
                if not np.all(np.isfinite(err)):
                    invalid[i, j, k] = True
                    err[:] = 0.0

                n_done += 1
                if n_done % 25 == 0 or n_done == n_total:
                    elapsed = time.time() - t0
                    rate = n_done / max(1e-3, elapsed)
                    eta = (n_total - n_done) / max(1e-3, rate)
                    print(f"[calib] {n_done}/{n_total} "
                          f"({100*n_done/n_total:.0f}%) "
                          f"err=({err[0]:+.3f}, {err[1]:+.3f}, {err[2]:+.3f}) "
                          f"elapsed={elapsed:.0f}s eta={eta:.0f}s")

    n_invalid = int(invalid.sum())
    if n_invalid:
        print(f"[calib] WARNING: {n_invalid} grid points produced "
              "non-finite errors; replaced with zero.")

    print(f"[calib] Calibration done in {time.time() - t0:.1f}s")
    print(f"[calib] Error magnitude stats:")
    mag = np.linalg.norm(error, axis=-1)
    print(f"        min={mag.min()*100:.1f}cm  "
          f"mean={mag.mean()*100:.1f}cm  "
          f"max={mag.max()*100:.1f}cm")
    print(f"[calib] Error XY-magnitude (the part IK target correction "
          "needs):")
    mag_xy = np.linalg.norm(error[..., :2], axis=-1)
    print(f"        min={mag_xy.min()*100:.1f}cm  "
          f"mean={mag_xy.mean()*100:.1f}cm  "
          f"max={mag_xy.max()*100:.1f}cm")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path,
        h1_grid=GRID_H1,
        h2_grid=GRID_H2,
        a1_grid=GRID_A1,
        error=error,
    )
    print(f"[calib] Wrote LUT to {out_path}")


if __name__ == "__main__":
    xml_default = os.path.join(ROOT, "src", "env", "market_world_m1.xml")
    out_default = os.path.join(ROOT, "data", "arm_calibration.npz")

    xml_path = sys.argv[1] if len(sys.argv) > 1 else xml_default
    out_path = sys.argv[2] if len(sys.argv) > 2 else out_default

    if not os.path.exists(xml_path):
        sys.exit(f"[calib] XML not found: {xml_path}")
    calibrate(xml_path, out_path)
