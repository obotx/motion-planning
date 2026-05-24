"""calibrate_arm_kinematics.py — generate the MORPH-I arm calibration LUT.

The MORPH-I arm has a passive `RotationLeftJoint` whose qpos is set
by physics dynamics under gravity + the parallel-mechanism constraint.
The IK solver in `arm_planner.solve_ik` uses `mj_forward` (kinematics
only), which leaves the passive joint at its initial qpos value and
therefore produces a Link3 (palm) world position that does NOT match
where Link3 lands once `mj_step` runs and the passive joint settles.
Empirically this gap is 4-25cm depending on grip mode.

This script measures that gap on a grid of arm joint values for a
specific wrist orientation (grip mode) and writes the result to
`data/arm_calibration_<mode>.npz`.  The planner loads the appropriate
LUT at startup based on grip mode and uses it to pre-correct IK
targets.

Output schema (numpy npz):
    h1_grid:     1-D array of h1 values sampled
    h2_grid:     1-D array of h2 values sampled
    a1_grid:     1-D array of a1 values sampled
    error:       4-D array shape (len_h1, len_h2, len_a1, 3) — chassis-
                 relative XYZ deflection (physics − kinematics) at theta=0
    wrist_mode:  string indicating the wrist configuration used
    wrist_qpos:  4-vector (HandBearing, gripper_z, gripper_x, gripper_y)
                 used during measurement

Run from the project root:

    docker compose -f docker/docker-compose.yml run --rm motion-planning \\
        python3 tools/calibrate_arm_kinematics.py

The default mode is `sidegrip` (matches the current STRICT-mode side-grip
wrist orientation used by play_m1).  Override with --wrist-mode topdown
for the legacy 4-DOF top-down configuration.

Takes ~5-15 minutes on a 7×7×7 grid (343 poses, ~2s each).  The LUT
file is small (~10 KB) and is loaded once at planner init.
"""

import argparse
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

# Per-mode configuration.  Each entry sets the wrist orientation
# (HandBearing, gripper_z, gripper_x, gripper_y) used during BOTH
# kinematic and physics measurements, plus the arm-joint grid ranges
# appropriate for typical IK poses in that mode.
#
# Side-grip values mirror WRIST_*_SIDE_APPROACH constants in
# `src/navigation/grasp_executor.py`:
#   WRIST_X_SIDE_APPROACH = +0.80
#   WRIST_Y_SIDE_APPROACH =  0.00
#   WRIST_Z_SIDE_APPROACH = -1.88
#   WRIST_PITCH_SIDE_APPROACH (HandBearing) = 0.00
#
# Top-down keeps the legacy identity wrist orientation that matches
# the original 4-DOF calibration.
MODE_CONFIGS = {
    "sidegrip": {
        # (HandBearing, gripper_z_rotation, gripper_x_rotation, gripper_y_rotation)
        "wrist_qpos": (0.00, -1.88, +0.80, 0.00),
        # Side-grip typically lands at h1≈0.10-0.20, h2≈0.25-0.45,
        # a1≈0.25-0.40 (per recent run logs).  Extend ranges a bit
        # beyond the observed band for safe trilinear interpolation.
        "h1_grid": np.linspace(0.05, 0.30, 7),
        "h2_grid": np.linspace(0.05, 0.55, 7),
        "a1_grid": np.linspace(0.10, 0.60, 7),
    },
    "topdown": {
        "wrist_qpos": (0.00, 0.00, 0.00, 0.00),
        "h1_grid": np.linspace(0.05, 0.30, 7),
        "h2_grid": np.linspace(0.05, 0.30, 7),
        "a1_grid": np.linspace(0.10, 0.55, 7),
    },
}


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


def calibrate(xml_path, out_path, wrist_mode):
    """Run calibration sweep at the given wrist_mode and save LUT."""
    cfg = MODE_CONFIGS[wrist_mode]
    wrist_qpos = cfg["wrist_qpos"]
    GRID_H1 = cfg["h1_grid"]
    GRID_H2 = cfg["h2_grid"]
    GRID_A1 = cfg["a1_grid"]

    print(f"[calib] Loading model: {xml_path}")
    print(f"[calib] Wrist mode: {wrist_mode}")
    print(f"[calib] Wrist qpos (hb, wz, wx, wy) = "
          f"({wrist_qpos[0]:+.3f}, {wrist_qpos[1]:+.3f}, "
          f"{wrist_qpos[2]:+.3f}, {wrist_qpos[3]:+.3f})")
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
    # Wrist joints — order in wrist_qpos tuple is
    # (HandBearing, gripper_z, gripper_x, gripper_y).
    hb_q = _qi(model, f"HandBearingJoint{suffix}")
    wz_q = _qi(model, f"gripper_z_rotation{suffix}")
    wx_q = _qi(model, f"gripper_x_rotation{suffix}")
    wy_q = _qi(model, f"gripper_y_rotation{suffix}")
    palm_bid = _bid(model, "Gripper_Link3_1")
    chassis_bid = _bid(model, "robot")

    # Resolve actuator IDs for the joints we set.  The calibration
    # must set BOTH qpos AND the corresponding actuator ctrl —
    # otherwise the position actuator (ctrl defaulting to 0) generates
    # a restoring torque trying to drive qpos back to 0, propagating
    # phantom forces through the kinematic chain to the passive joint
    # we're trying to measure.
    #
    # MuJoCo's `actuator_trnid[:, 0]` gives the joint ID each actuator
    # drives (for joint-type actuators).  Find the actuator that
    # corresponds to each joint we care about by matching jnt_qposadr.
    def _find_actuator_for_joint(jname):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0:
            return -1
        for aid in range(model.nu):
            if int(model.actuator_trnid[aid, 0]) == jid:
                return aid
        return -1
    h1_a = _find_actuator_for_joint(f"ColumnLeftBearingJoint{suffix}")
    h2_a = _find_actuator_for_joint(f"ColumnRightBearingJoint{suffix}")
    a1_a = _find_actuator_for_joint(f"ArmLeftJoint{suffix}")
    th_a = _find_actuator_for_joint(f"BaseJoint{suffix}")
    hb_a = _find_actuator_for_joint(f"HandBearingJoint{suffix}")
    wz_a = _find_actuator_for_joint(f"gripper_z_rotation{suffix}")
    wx_a = _find_actuator_for_joint(f"gripper_x_rotation{suffix}")
    wy_a = _find_actuator_for_joint(f"gripper_y_rotation{suffix}")
    missing = [(name, aid) for name, aid in (
        ("h1", h1_a), ("h2", h2_a), ("a1", a1_a), ("th", th_a),
        ("hb", hb_a), ("wz", wz_a), ("wx", wx_a), ("wy", wy_a),
    ) if aid < 0]
    if missing:
        print(f"[calib] WARNING: actuators not found for: "
              f"{[m[0] for m in missing]}  "
              f"— their ctrl will not be set, may cause phantom torques")
    print(f"[calib] actuator IDs: h1={h1_a} h2={h2_a} a1={a1_a} th={th_a}  "
          f"hb={hb_a} wz={wz_a} wx={wx_a} wy={wy_a}")

    # Build a "passive joint qpos mask".  Free BOTH passive joints in
    # the parallel mechanism: RotationLeftJoint and ArmRightJoint.
    # Both must evolve to satisfy the closed-loop constraint when h1,
    # h2, a1 are pinned at arbitrary values.  Earlier 4-DOF calibration
    # freed only RotationLeftJoint — for the wider sidegrip grid range
    # (h2 up to 0.55) this leaves the constraint over-determined and
    # generates massive spurious forces.
    passive_q_rot = _qi(model, f"RotationLeftJoint{suffix}")
    passive_q_armR = _qi(model, f"ArmRightJoint{suffix}")
    n_qpos = int(model.nq)
    pin_mask = np.ones(n_qpos, dtype=bool)
    pin_mask[passive_q_rot]  = False  # let this evolve
    pin_mask[passive_q_armR] = False  # let this evolve too
    print(f"[calib] passive RotationLeftJoint{suffix} qposadr={passive_q_rot}")
    print(f"[calib] passive ArmRightJoint{suffix}    qposadr={passive_q_armR}")
    print(f"[calib] Pinning {int(pin_mask.sum())}/{n_qpos} qpos values "
          f"every step; both passive parallel-chain joints settle freely.")

    # Zero joint stiffness for the active arm joints we'll pin.
    # Without this, the passive spring (e.g. stiffness=50000 on
    # ColumnLeftBearingJoint with springref=0) generates a huge
    # restoring torque trying to pull the pinned qpos back to 0,
    # propagating spurious forces into the passive joints and giving
    # wildly inflated deflection estimates.  At runtime the actuator
    # kp dominates over the joint spring (e.g. kp=580 vs k=50000 —
    # OK the spring is bigger, but the pin during calibration is
    # IDEAL while at runtime the actuator+spring equilibrium produces
    # only a small residual error).  Zero stiffness during calibration
    # measures purely the gravity-driven deflection of the passive
    # chain, which is the signal the LUT needs to capture.
    active_joints_to_zero_stiffness = [
        f"ColumnLeftBearingJoint{suffix}",
        f"ColumnRightBearingJoint{suffix}",
        f"ArmLeftJoint{suffix}",
        f"HandBearingJoint{suffix}",
        f"gripper_z_rotation{suffix}",
        f"gripper_x_rotation{suffix}",
        f"gripper_y_rotation{suffix}",
    ]
    zeroed = []
    for jname in active_joints_to_zero_stiffness:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0:
            continue
        original_stiffness = float(model.jnt_stiffness[jid])
        if original_stiffness > 0:
            model.jnt_stiffness[jid] = 0.0
            zeroed.append((jname, original_stiffness))
    if zeroed:
        print(f"[calib] Zeroed joint stiffness on active joints "
              f"(was-vs-now): "
              + ", ".join(f"{n}={k:.0f}→0" for n, k in zeroed))

    print(f"[calib] qpos_addr h1={h1_q} h2={h2_q} a1={a1_q} theta={th_q}")
    print(f"[calib] wrist qpos_addrs hb={hb_q} wz={wz_q} wx={wx_q} wy={wy_q}")
    print(f"[calib] palm_body_id={palm_bid}  chassis_body_id={chassis_bid}")
    print(f"[calib] Grid: {len(GRID_H1)}×{len(GRID_H2)}×{len(GRID_A1)} = "
          f"{len(GRID_H1)*len(GRID_H2)*len(GRID_A1)} poses")
    print(f"[calib]   h1 range: [{GRID_H1[0]:.3f}, {GRID_H1[-1]:.3f}]")
    print(f"[calib]   h2 range: [{GRID_H2[0]:.3f}, {GRID_H2[-1]:.3f}]")
    print(f"[calib]   a1 range: [{GRID_A1[0]:.3f}, {GRID_A1[-1]:.3f}]")

    error = np.zeros(
        (len(GRID_H1), len(GRID_H2), len(GRID_A1), 3), dtype=np.float64)
    invalid = np.zeros(error.shape[:3], dtype=bool)

    t0 = time.time()
    n_total = error.shape[0] * error.shape[1] * error.shape[2]
    n_done = 0

    def _apply_pose(data, h1, h2, a1):
        """Write all 8 active-arm qpos values + matching ctrl values.

        Setting ctrl = qpos prevents the position actuators from
        generating restoring torques during the physics settle.  At
        runtime, ctrl IS set to the commanded position, so actuators
        sit at zero residual — the calibration must replicate this to
        measure ONLY the gravity-driven passive deflection."""
        # qpos
        data.qpos[h1_q] = h1
        data.qpos[h2_q] = h2
        data.qpos[a1_q] = a1
        data.qpos[th_q] = 0.0
        data.qpos[hb_q] = wrist_qpos[0]
        data.qpos[wz_q] = wrist_qpos[1]
        data.qpos[wx_q] = wrist_qpos[2]
        data.qpos[wy_q] = wrist_qpos[3]
        # ctrl — match each actuator's commanded position to qpos so
        # the position actuator's PD residual is zero at the start.
        # Without this, actuators with default ctrl=0 generate huge
        # restoring torques (e.g., wz with kp=500 and qpos=-1.88 →
        # ~940 N·m phantom torque) that deflect the passive joint
        # under physics — corrupting the LUT with phantom errors.
        if h1_a >= 0: data.ctrl[h1_a] = h1
        if h2_a >= 0: data.ctrl[h2_a] = h2
        if a1_a >= 0: data.ctrl[a1_a] = a1
        if th_a >= 0: data.ctrl[th_a] = 0.0
        if hb_a >= 0: data.ctrl[hb_a] = wrist_qpos[0]
        if wz_a >= 0: data.ctrl[wz_a] = wrist_qpos[1]
        if wx_a >= 0: data.ctrl[wx_a] = wrist_qpos[2]
        if wy_a >= 0: data.ctrl[wy_a] = wrist_qpos[3]

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
                _apply_pose(data_kin, h1, h2, a1)
                mujoco.mj_forward(model, data_kin)
                R_kin = data_kin.xmat[chassis_bid].reshape(3, 3)
                rel_kin = (data_kin.xpos[palm_bid].copy()
                           - data_kin.xpos[chassis_bid].copy())
                p_kin = R_kin.T @ rel_kin

                # -- Physics-settled measurement --
                mujoco.mj_resetData(model, data_phys)
                _apply_pose(data_phys, h1, h2, a1)
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
    print(f"[calib] Error Z-magnitude (vertical deflection):")
    mag_z = np.abs(error[..., 2])
    print(f"        min={mag_z.min()*100:.1f}cm  "
          f"mean={mag_z.mean()*100:.1f}cm  "
          f"max={mag_z.max()*100:.1f}cm")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path,
        h1_grid=GRID_H1,
        h2_grid=GRID_H2,
        a1_grid=GRID_A1,
        error=error,
        wrist_mode=wrist_mode,
        wrist_qpos=np.asarray(wrist_qpos, dtype=np.float64),
    )
    print(f"[calib] Wrote LUT to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate the MORPH-I arm calibration LUT for a "
                    "specific wrist orientation (grip mode).")
    parser.add_argument(
        "--xml", default=None,
        help="Path to MuJoCo XML world file.  Defaults to "
             "src/env/market_world_m1.xml.")
    parser.add_argument(
        "--out", default=None,
        help="Output .npz path.  Defaults to "
             "data/arm_calibration_<wrist-mode>.npz.")
    parser.add_argument(
        "--wrist-mode", default="sidegrip",
        choices=sorted(MODE_CONFIGS.keys()),
        help="Wrist orientation to calibrate for.  Default: sidegrip "
             "(the current STRICT-mode pickup configuration).  Use "
             "topdown for the legacy identity-wrist configuration.")
    args = parser.parse_args()

    xml_path = args.xml or os.path.join(
        ROOT, "src", "env", "market_world_m1.xml")
    out_path = args.out or os.path.join(
        ROOT, "data", f"arm_calibration_{args.wrist_mode}.npz")

    if not os.path.exists(xml_path):
        sys.exit(f"[calib] XML not found: {xml_path}")
    calibrate(xml_path, out_path, args.wrist_mode)
