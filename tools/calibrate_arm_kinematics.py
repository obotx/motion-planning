
import argparse
import os
import sys
import time

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

ARM = 1
SETTLE_STEPS = 200
SETTLE_DT_TOL = 1e-5

MODE_CONFIGS = {
    "sidegrip": {
        "wrist_qpos": (0.00, -1.88, +0.80, 0.00),
        "h1_grid": np.linspace(0.00, 0.30, 7),
        "h2_grid": np.linspace(0.05, 0.60, 7),
        "a1_grid": np.linspace(0.10, 0.625, 9),
        "hb_grid": np.linspace(-0.40, +0.40, 5),
        "wz_grid": np.linspace(-1.88 - 0.35, -1.88 + 0.35, 3),
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
    cfg = MODE_CONFIGS[wrist_mode]
    wrist_qpos = cfg["wrist_qpos"]
    GRID_H1 = cfg["h1_grid"]
    GRID_H2 = cfg["h2_grid"]
    GRID_A1 = cfg["a1_grid"]
    GRID_HB = cfg.get("hb_grid")
    GRID_WZ = cfg.get("wz_grid")
    is_5d = (GRID_HB is not None) and (GRID_WZ is not None)
    ndim = 5 if is_5d else 3

    print(f"[calib] Loading model: {xml_path}")
    print(f"[calib] Wrist mode: {wrist_mode}  (LUT ndim={ndim})")
    print(f"[calib] Centre wrist qpos (hb, wz, wx, wy) = "
          f"({wrist_qpos[0]:+.3f}, {wrist_qpos[1]:+.3f}, "
          f"{wrist_qpos[2]:+.3f}, {wrist_qpos[3]:+.3f})")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data_kin = mujoco.MjData(model)
    data_phys = mujoco.MjData(model)

    suffix = f"_{ARM}"
    h1_q = _qi(model, f"ColumnLeftBearingJoint{suffix}")
    h2_q = _qi(model, f"ColumnRightBearingJoint{suffix}")
    a1_q = _qi(model, f"ArmLeftJoint{suffix}")
    th_q = _qi(model, f"BaseJoint{suffix}")
    hb_q = _qi(model, f"HandBearingJoint{suffix}")
    wz_q = _qi(model, f"gripper_z_rotation{suffix}")
    wx_q = _qi(model, f"gripper_x_rotation{suffix}")
    wy_q = _qi(model, f"gripper_y_rotation{suffix}")
    palm_bid = _bid(model, "Gripper_Link3_1")
    chassis_bid = _bid(model, "robot")

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

    passive_q_rot = _qi(model, f"RotationLeftJoint{suffix}")
    passive_q_armR = _qi(model, f"ArmRightJoint{suffix}")
    n_qpos = int(model.nq)
    pin_mask = np.ones(n_qpos, dtype=bool)
    pin_mask[passive_q_rot]  = False
    pin_mask[passive_q_armR] = False
    print(f"[calib] passive RotationLeftJoint{suffix} qposadr={passive_q_rot}")
    print(f"[calib] passive ArmRightJoint{suffix}    qposadr={passive_q_armR}")
    print(f"[calib] Pinning {int(pin_mask.sum())}/{n_qpos} qpos values "
          f"every step; both passive parallel-chain joints settle freely.")

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
    if is_5d:
        n_poses_total = (len(GRID_H1) * len(GRID_H2) * len(GRID_A1)
                         * len(GRID_HB) * len(GRID_WZ))
        print(f"[calib] Grid: {len(GRID_H1)}×{len(GRID_H2)}×{len(GRID_A1)}"
              f"×{len(GRID_HB)}×{len(GRID_WZ)} = {n_poses_total} poses")
    else:
        n_poses_total = len(GRID_H1) * len(GRID_H2) * len(GRID_A1)
        print(f"[calib] Grid: {len(GRID_H1)}×{len(GRID_H2)}×{len(GRID_A1)} = "
              f"{n_poses_total} poses")
    print(f"[calib]   h1 range: [{GRID_H1[0]:.3f}, {GRID_H1[-1]:.3f}]")
    print(f"[calib]   h2 range: [{GRID_H2[0]:.3f}, {GRID_H2[-1]:.3f}]")
    print(f"[calib]   a1 range: [{GRID_A1[0]:.3f}, {GRID_A1[-1]:.3f}]")
    if is_5d:
        print(f"[calib]   hb range: [{GRID_HB[0]:+.3f}, {GRID_HB[-1]:+.3f}]"
              f"  ({len(GRID_HB)} samples)")
        print(f"[calib]   wz range: [{GRID_WZ[0]:+.3f}, {GRID_WZ[-1]:+.3f}]"
              f"  ({len(GRID_WZ)} samples)")

    if is_5d:
        error = np.zeros(
            (len(GRID_H1), len(GRID_H2), len(GRID_A1),
             len(GRID_HB), len(GRID_WZ), 3), dtype=np.float64)
        invalid = np.zeros(error.shape[:-1], dtype=bool)
    else:
        error = np.zeros(
            (len(GRID_H1), len(GRID_H2), len(GRID_A1), 3), dtype=np.float64)
        invalid = np.zeros(error.shape[:3], dtype=bool)

    t0 = time.time()
    n_done = 0

    def _apply_pose(data, h1, h2, a1, hb=None, wz=None):
        hb_use = wrist_qpos[0] if hb is None else hb
        wz_use = wrist_qpos[1] if wz is None else wz
        wx_use = wrist_qpos[2]
        wy_use = wrist_qpos[3]
        data.qpos[h1_q] = h1
        data.qpos[h2_q] = h2
        data.qpos[a1_q] = a1
        data.qpos[th_q] = 0.0
        data.qpos[hb_q] = hb_use
        data.qpos[wz_q] = wz_use
        data.qpos[wx_q] = wx_use
        data.qpos[wy_q] = wy_use
        if h1_a >= 0: data.ctrl[h1_a] = h1
        if h2_a >= 0: data.ctrl[h2_a] = h2
        if a1_a >= 0: data.ctrl[a1_a] = a1
        if th_a >= 0: data.ctrl[th_a] = 0.0
        if hb_a >= 0: data.ctrl[hb_a] = hb_use
        if wz_a >= 0: data.ctrl[wz_a] = wz_use
        if wx_a >= 0: data.ctrl[wx_a] = wx_use
        if wy_a >= 0: data.ctrl[wy_a] = wy_use

    def _measure(h1, h2, a1, hb=None, wz=None):
        mujoco.mj_resetData(model, data_kin)
        _apply_pose(data_kin, h1, h2, a1, hb, wz)
        mujoco.mj_forward(model, data_kin)
        R_kin = data_kin.xmat[chassis_bid].reshape(3, 3)
        rel_kin = (data_kin.xpos[palm_bid].copy()
                   - data_kin.xpos[chassis_bid].copy())
        p_kin = R_kin.T @ rel_kin

        mujoco.mj_resetData(model, data_phys)
        _apply_pose(data_phys, h1, h2, a1, hb, wz)
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
        return err, bool(np.all(np.isfinite(err)))

    n_total = n_poses_total

    if is_5d:
        for i, h1 in enumerate(GRID_H1):
            for j, h2 in enumerate(GRID_H2):
                for k, a1 in enumerate(GRID_A1):
                    for m, hb in enumerate(GRID_HB):
                        for n, wz in enumerate(GRID_WZ):
                            err, ok = _measure(h1, h2, a1, hb, wz)
                            if not ok:
                                invalid[i, j, k, m, n] = True
                                err = np.zeros(3, dtype=float)
                            error[i, j, k, m, n] = err
                            n_done += 1
                            if n_done % 50 == 0 or n_done == n_total:
                                elapsed = time.time() - t0
                                rate = n_done / max(1e-3, elapsed)
                                eta = (n_total - n_done) / max(1e-3, rate)
                                print(f"[calib] {n_done}/{n_total} "
                                      f"({100*n_done/n_total:.0f}%) "
                                      f"err=({err[0]:+.3f}, {err[1]:+.3f}, "
                                      f"{err[2]:+.3f}) "
                                      f"elapsed={elapsed:.0f}s "
                                      f"eta={eta:.0f}s")
    else:
        for i, h1 in enumerate(GRID_H1):
            for j, h2 in enumerate(GRID_H2):
                for k, a1 in enumerate(GRID_A1):
                    err, ok = _measure(h1, h2, a1)
                    if not ok:
                        invalid[i, j, k] = True
                        err = np.zeros(3, dtype=float)
                    error[i, j, k] = err
                    n_done += 1
                    if n_done % 25 == 0 or n_done == n_total:
                        elapsed = time.time() - t0
                        rate = n_done / max(1e-3, elapsed)
                        eta = (n_total - n_done) / max(1e-3, rate)
                        print(f"[calib] {n_done}/{n_total} "
                              f"({100*n_done/n_total:.0f}%) "
                              f"err=({err[0]:+.3f}, {err[1]:+.3f}, "
                              f"{err[2]:+.3f}) "
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
    save_kwargs = dict(
        ndim=np.asarray(ndim, dtype=np.int32),
        h1_grid=GRID_H1,
        h2_grid=GRID_H2,
        a1_grid=GRID_A1,
        error=error,
        wrist_mode=wrist_mode,
        wrist_qpos=np.asarray(wrist_qpos, dtype=np.float64),
    )
    if is_5d:
        save_kwargs["hb_grid"] = GRID_HB
        save_kwargs["wz_grid"] = GRID_WZ
    np.savez_compressed(out_path, **save_kwargs)
    print(f"[calib] Wrote LUT to {out_path}  (ndim={ndim})")


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
