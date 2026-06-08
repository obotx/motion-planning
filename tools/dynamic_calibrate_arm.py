
import argparse
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

ARM = 1

PRESTAB_STEPS = 200
RAMP_STEPS = 500
SETTLE_STEPS = 1000

WZ_PD_RATIO = 0.09

N_WORKERS_OVERRIDE = None
MEM_PER_WORKER_GB = 2.0

START_Q = {
    "h1":  0.20,
    "h2":  0.20,
    "a1":  0.10,
    "th":  0.00,
    "hb":  0.00,
    "wz": -1.88,
    "wx":  0.80,
    "wy":  0.00,
}

MODE_CONFIGS = {
    "sidegrip": {
        "wrist_qpos": (0.00, -1.88, +0.80, 0.00),
        "h1_grid": np.linspace(0.05, 0.25, 5),
        "h2_grid": np.linspace(0.10, 0.50, 5),
        "a1_grid": np.linspace(0.15, 0.60, 7),
        "hb_grid": np.linspace(-0.30, +0.30, 5),
        "wz_grid": np.linspace(-2.10, -1.66, 3),
    },
}

SMOKE_TEST_POSES = [
    (0.15, 0.30, 0.30, 0.00, -1.88),
    (0.05, 0.30, 0.30, 0.00, -1.88),
    (0.25, 0.30, 0.30, 0.00, -1.88),
    (0.15, 0.10, 0.30, 0.00, -1.88),
    (0.15, 0.50, 0.30, 0.00, -1.88),
    (0.15, 0.30, 0.15, 0.00, -1.88),
    (0.15, 0.30, 0.60, 0.00, -1.88),
    (0.15, 0.30, 0.30, -0.30, -1.88),
    (0.15, 0.30, 0.30, 0.00, -2.10),
    (0.13, 0.23, 0.30, 0.00, -1.88),
]


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


def _find_actuator_for_joint(model, jname):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
    if jid < 0:
        return -1
    for aid in range(model.nu):
        if int(model.actuator_trnid[aid, 0]) == jid:
            return aid
    return -1



_W = {}


def _worker_init(xml_path, wrist_mode):
    global _W
    model = mujoco.MjModel.from_xml_path(xml_path)
    data_kin = mujoco.MjData(model)
    data_phys = mujoco.MjData(model)
    suffix = f"_{ARM}"
    home_key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_key < 0:
        raise RuntimeError("worker: 'home' keyframe not found")
    _W.update({
        "model": model,
        "data_kin": data_kin,
        "data_phys": data_phys,
        "home_key": home_key,
        "wrist_qpos": MODE_CONFIGS[wrist_mode]["wrist_qpos"],
        "h1_q": _qi(model, f"ColumnLeftBearingJoint{suffix}"),
        "h2_q": _qi(model, f"ColumnRightBearingJoint{suffix}"),
        "a1_q": _qi(model, f"ArmLeftJoint{suffix}"),
        "th_q": _qi(model, f"BaseJoint{suffix}"),
        "hb_q": _qi(model, f"HandBearingJoint{suffix}"),
        "wz_q": _qi(model, f"gripper_z_rotation{suffix}"),
        "wx_q": _qi(model, f"gripper_x_rotation{suffix}"),
        "wy_q": _qi(model, f"gripper_y_rotation{suffix}"),
        "palm_bid": _bid(model, "Gripper_Link3_1"),
        "chassis_bid": _bid(model, "robot"),
        "h1_a": _find_actuator_for_joint(model, f"ColumnLeftBearingJoint{suffix}"),
        "h2_a": _find_actuator_for_joint(model, f"ColumnRightBearingJoint{suffix}"),
        "a1_a": _find_actuator_for_joint(model, f"ArmLeftJoint{suffix}"),
        "th_a": _find_actuator_for_joint(model, f"BaseJoint{suffix}"),
        "hb_a": _find_actuator_for_joint(model, f"HandBearingJoint{suffix}"),
        "wz_a": _find_actuator_for_joint(model, f"gripper_z_rotation{suffix}"),
        "wx_a": _find_actuator_for_joint(model, f"gripper_x_rotation{suffix}"),
        "wy_a": _find_actuator_for_joint(model, f"gripper_y_rotation{suffix}"),
        "arm2_h1_q": _qi(model, "ColumnLeftBearingJoint_2"),
        "arm2_h2_q": _qi(model, "ColumnRightBearingJoint_2"),
        "arm2_a1_q": _qi(model, "ArmLeftJoint_2"),
        "arm2_th_q": _qi(model, "BaseJoint_2"),
    })


def _w_park_arm2(data):
    data.qpos[_W["arm2_h1_q"]] = 0.60
    data.qpos[_W["arm2_h2_q"]] = 0.65
    data.qpos[_W["arm2_a1_q"]] = 0.10
    data.qpos[_W["arm2_th_q"]] = 0.0


def _w_park_pickup_objs(data, n=10):
    model = _W["model"]
    for i in range(n):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                f"pickup_obj_{i}")
        if bid < 0:
            continue
        ja = int(model.body_jntadr[bid])
        if ja < 0:
            continue
        qa = int(model.jnt_qposadr[ja])
        data.qpos[qa]     = float(i) * 3.0
        data.qpos[qa + 1] = 50.0
        data.qpos[qa + 2] = 100.0
        data.qpos[qa + 3] = 1.0
        data.qpos[qa + 4] = 0.0
        data.qpos[qa + 5] = 0.0
        data.qpos[qa + 6] = 0.0


def _w_set_qpos(data, h1, h2, a1, hb, wz):
    data.qpos[_W["h1_q"]] = h1
    data.qpos[_W["h2_q"]] = h2
    data.qpos[_W["a1_q"]] = a1
    data.qpos[_W["th_q"]] = 0.0
    data.qpos[_W["hb_q"]] = hb
    data.qpos[_W["wz_q"]] = wz
    data.qpos[_W["wx_q"]] = 0.80
    data.qpos[_W["wy_q"]] = 0.00


def _w_set_ctrl(data, h1, h2, a1, hb, wz):
    OFF_H1, OFF_H2, OFF_A1 = -0.0036, -0.0062, -0.0006
    if _W["h1_a"] >= 0: data.ctrl[_W["h1_a"]] = (h1 + OFF_H1) * 100.0
    if _W["h2_a"] >= 0: data.ctrl[_W["h2_a"]] = (h2 + OFF_H2) * 100.0
    if _W["a1_a"] >= 0: data.ctrl[_W["a1_a"]] = (a1 + OFF_A1) * 100.0
    if _W["th_a"] >= 0: data.ctrl[_W["th_a"]] = 0.0
    if _W["hb_a"] >= 0: data.ctrl[_W["hb_a"]] = hb
    if _W["wz_a"] >= 0: data.ctrl[_W["wz_a"]] = wz * (1.0 + WZ_PD_RATIO)
    if _W["wx_a"] >= 0: data.ctrl[_W["wx_a"]] = 0.80
    if _W["wy_a"] >= 0: data.ctrl[_W["wy_a"]] = 0.00


def _measure_pose(task):
    i, j, k, m, n, h1, h2, a1, hb, wz = task
    try:
        model = _W["model"]
        data_kin = _W["data_kin"]
        data_phys = _W["data_phys"]
        palm_bid = _W["palm_bid"]
        chassis_bid = _W["chassis_bid"]
        home_key = _W["home_key"]

        mujoco.mj_resetDataKeyframe(model, data_kin, home_key)
        _w_park_arm2(data_kin)
        _w_park_pickup_objs(data_kin)
        _w_set_qpos(data_kin, h1, h2, a1, hb, wz)
        mujoco.mj_forward(model, data_kin)
        R_kin = data_kin.xmat[chassis_bid].reshape(3, 3)
        rel_kin = (data_kin.xpos[palm_bid].copy()
                   - data_kin.xpos[chassis_bid].copy())
        p_kin = R_kin.T @ rel_kin

        mujoco.mj_resetDataKeyframe(model, data_phys, home_key)
        _w_park_arm2(data_phys)
        _w_park_pickup_objs(data_phys)
        _w_set_qpos(data_phys,
                    START_Q["h1"], START_Q["h2"], START_Q["a1"],
                    START_Q["hb"], START_Q["wz"])
        _w_set_ctrl(data_phys,
                    START_Q["h1"], START_Q["h2"], START_Q["a1"],
                    START_Q["hb"], START_Q["wz"])
        for _ in range(PRESTAB_STEPS):
            mujoco.mj_step(model, data_phys)
        _w_park_arm2(data_phys)
        _w_park_pickup_objs(data_phys)

        for s in range(RAMP_STEPS):
            alpha = (s + 1) / RAMP_STEPS
            alpha_h = 3.0 * alpha * alpha - 2.0 * alpha * alpha * alpha
            h1_c = (1.0 - alpha_h) * START_Q["h1"] + alpha_h * h1
            h2_c = (1.0 - alpha_h) * START_Q["h2"] + alpha_h * h2
            a1_c = (1.0 - alpha)   * START_Q["a1"] + alpha   * a1
            hb_c = (1.0 - alpha)   * START_Q["hb"] + alpha   * hb
            wz_c = (1.0 - alpha)   * START_Q["wz"] + alpha   * wz
            _w_set_ctrl(data_phys, h1_c, h2_c, a1_c, hb_c, wz_c)
            mujoco.mj_step(model, data_phys)
        _w_park_pickup_objs(data_phys)

        _w_set_ctrl(data_phys, h1, h2, a1, hb, wz)
        for _ in range(SETTLE_STEPS):
            mujoco.mj_step(model, data_phys)

        R_phys = data_phys.xmat[chassis_bid].reshape(3, 3)
        rel_phys = (data_phys.xpos[palm_bid].copy()
                    - data_phys.xpos[chassis_bid].copy())
        p_phys = R_phys.T @ rel_phys
        err = p_phys - p_kin
        if not bool(np.all(np.isfinite(err))):
            return (i, j, k, m, n, 0.0, 0.0, 0.0, False)
        return (i, j, k, m, n,
                float(err[0]), float(err[1]), float(err[2]), True)
    except Exception:
        return (i, j, k, m, n, 0.0, 0.0, 0.0, False)


def _smoothstep(alpha):
    return 3.0 * alpha * alpha - 2.0 * alpha * alpha * alpha


def _run_full_sweep(xml_path, out_path, wrist_mode, cfg):
    wrist_qpos_meta = cfg["wrist_qpos"]
    GRID_H1 = cfg["h1_grid"]
    GRID_H2 = cfg["h2_grid"]
    GRID_A1 = cfg["a1_grid"]
    GRID_HB = cfg["hb_grid"]
    GRID_WZ = cfg["wz_grid"]

    n_total = (len(GRID_H1) * len(GRID_H2) * len(GRID_A1)
               * len(GRID_HB) * len(GRID_WZ))

    def _avail_mem_gb():
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) / (1024.0 * 1024.0)
        except Exception:
            pass
        return 4.0
    if N_WORKERS_OVERRIDE:
        n_workers = N_WORKERS_OVERRIDE
        print(f"[calib-dyn-v2] Workers: {n_workers} (--workers override)",
              flush=True)
    else:
        avail = _avail_mem_gb()
        mem_cap = max(1, int((avail - 2.0) / MEM_PER_WORKER_GB))
        n_workers = max(1, min(mem_cap, mp.cpu_count() - 2))
        print(f"[calib-dyn-v2] Memory-bounded workers: "
              f"{avail:.1f}GB avail → {mem_cap} (mem cap), "
              f"{mp.cpu_count()-2} (core cap) → {n_workers}", flush=True)
    print(f"[calib-dyn-v2] Grid: {len(GRID_H1)}×{len(GRID_H2)}×{len(GRID_A1)}"
          f"×{len(GRID_HB)}×{len(GRID_WZ)} = {n_total} poses")
    print(f"[calib-dyn-v2] Parallel: {n_workers} workers (spawn), "
          f"parent loads NO model.  ETA ~{n_total*1.4/n_workers/60:.1f} min",
          flush=True)

    error = np.zeros(
        (len(GRID_H1), len(GRID_H2), len(GRID_A1),
         len(GRID_HB), len(GRID_WZ), 3), dtype=np.float64)
    invalid = np.zeros(error.shape[:-1], dtype=bool)

    tasks = []
    for i, h1 in enumerate(GRID_H1):
        for j, h2 in enumerate(GRID_H2):
            for k, a1 in enumerate(GRID_A1):
                for m, hb in enumerate(GRID_HB):
                    for n, wz in enumerate(GRID_WZ):
                        tasks.append((i, j, k, m, n,
                                      float(h1), float(h2), float(a1),
                                      float(hb), float(wz)))

    t0 = time.time()
    n_done = 0
    ctx = mp.get_context("spawn")
    with ctx.Pool(n_workers, initializer=_worker_init,
                  initargs=(xml_path, wrist_mode)) as pool:
        for result in pool.imap_unordered(_measure_pose, tasks, chunksize=8):
            i, j, k, m, n, ex, ey, ez, ok = result
            if not ok:
                invalid[i, j, k, m, n] = True
                error[i, j, k, m, n] = (0.0, 0.0, 0.0)
            else:
                error[i, j, k, m, n] = (ex, ey, ez)
            n_done += 1
            if n_done % 100 == 0 or n_done == n_total:
                elapsed = time.time() - t0
                rate = n_done / max(1e-3, elapsed)
                eta = (n_total - n_done) / max(1e-3, rate)
                mag = (float(np.sqrt(ex*ex + ey*ey + ez*ez)) if ok else 0.0)
                print(f"[calib-dyn-v2] {n_done}/{n_total} "
                      f"({100*n_done/n_total:.0f}%) "
                      f"last=({ex:+.3f},{ey:+.3f},{ez:+.3f}) "
                      f"|e|={mag*100:.1f}cm  "
                      f"elapsed={elapsed:.0f}s eta={eta:.0f}s  "
                      f"rate={rate:.1f} poses/s", flush=True)

    n_invalid = int(invalid.sum())
    if n_invalid:
        print(f"[calib-dyn-v2] WARNING: {n_invalid} grid points "
              f"produced non-finite errors; replaced with zero.")
    print(f"[calib-dyn-v2] Calibration done in {time.time() - t0:.1f}s")
    mag = np.linalg.norm(error, axis=-1)
    print(f"[calib-dyn-v2] Error magnitude: min={mag.min()*100:.1f}cm  "
          f"mean={mag.mean()*100:.1f}cm  max={mag.max()*100:.1f}cm")
    mag_xy = np.linalg.norm(error[..., :2], axis=-1)
    print(f"[calib-dyn-v2] Error XY: min={mag_xy.min()*100:.1f}cm  "
          f"mean={mag_xy.mean()*100:.1f}cm  max={mag_xy.max()*100:.1f}cm")
    mag_z = np.abs(error[..., 2])
    print(f"[calib-dyn-v2] Error Z: min={mag_z.min()*100:.1f}cm  "
          f"mean={mag_z.mean()*100:.1f}cm  max={mag_z.max()*100:.1f}cm")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path,
        ndim=np.asarray(5, dtype=np.int32),
        h1_grid=GRID_H1, h2_grid=GRID_H2, a1_grid=GRID_A1,
        hb_grid=GRID_HB, wz_grid=GRID_WZ,
        error=error,
        wrist_mode=f"{wrist_mode}-dynamic",
        wrist_qpos=np.asarray(wrist_qpos_meta, dtype=np.float64),
        prestab_steps=np.asarray(PRESTAB_STEPS, dtype=np.int32),
        ramp_steps=np.asarray(RAMP_STEPS, dtype=np.int32),
        settle_steps=np.asarray(SETTLE_STEPS, dtype=np.int32),
    )
    print(f"[calib-dyn-v2] Wrote dynamic LUT to {out_path}", flush=True)


def calibrate_dynamic_v2(xml_path, out_path, wrist_mode, smoke_test=False):
    cfg = MODE_CONFIGS[wrist_mode]
    wrist_qpos_meta = cfg["wrist_qpos"]
    GRID_H1 = cfg["h1_grid"]
    GRID_H2 = cfg["h2_grid"]
    GRID_A1 = cfg["a1_grid"]
    GRID_HB = cfg["hb_grid"]
    GRID_WZ = cfg["wz_grid"]

    if not smoke_test:
        _run_full_sweep(xml_path, out_path, wrist_mode, cfg)
        return

    print(f"[calib-dyn-v2] Loading model: {xml_path}")
    print(f"[calib-dyn-v2] Wrist mode: {wrist_mode} (5D dynamic, smooth-ramp)")
    print(f"[calib-dyn-v2] Protocol: prestab={PRESTAB_STEPS}, "
          f"ramp={RAMP_STEPS}, settle={SETTLE_STEPS}  "
          f"(total {PRESTAB_STEPS+RAMP_STEPS+SETTLE_STEPS} steps/pose)")
    print(f"[calib-dyn-v2] WZ_PD_RATIO={WZ_PD_RATIO} (matches runtime)")
    print(f"[calib-dyn-v2] START_Q (constraint-satisfied HOME): {START_Q}")

    model = mujoco.MjModel.from_xml_path(xml_path)
    data_kin = mujoco.MjData(model)
    data_phys = mujoco.MjData(model)

    home_key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_key < 0:
        raise RuntimeError("Required 'home' keyframe not found in model")

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

    h1_a = _find_actuator_for_joint(model, f"ColumnLeftBearingJoint{suffix}")
    h2_a = _find_actuator_for_joint(model, f"ColumnRightBearingJoint{suffix}")
    a1_a = _find_actuator_for_joint(model, f"ArmLeftJoint{suffix}")
    th_a = _find_actuator_for_joint(model, f"BaseJoint{suffix}")
    hb_a = _find_actuator_for_joint(model, f"HandBearingJoint{suffix}")
    wz_a = _find_actuator_for_joint(model, f"gripper_z_rotation{suffix}")
    wx_a = _find_actuator_for_joint(model, f"gripper_x_rotation{suffix}")
    wy_a = _find_actuator_for_joint(model, f"gripper_y_rotation{suffix}")

    arm2_h1_q = _qi(model, "ColumnLeftBearingJoint_2")
    arm2_h2_q = _qi(model, "ColumnRightBearingJoint_2")
    arm2_a1_q = _qi(model, "ArmLeftJoint_2")
    arm2_th_q = _qi(model, "BaseJoint_2")

    def _park_arm2(data):
        data.qpos[arm2_h1_q] = 0.60
        data.qpos[arm2_h2_q] = 0.65
        data.qpos[arm2_a1_q] = 0.10
        data.qpos[arm2_th_q] = 0.0

    def _park_pickup_objs(data, n=10):
        for i in range(n):
            bname = f"pickup_obj_{i}"
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
            if bid < 0:
                continue
            ja = int(model.body_jntadr[bid])
            if ja < 0:
                continue
            qa = int(model.jnt_qposadr[ja])
            data.qpos[qa]     = float(i) * 3.0
            data.qpos[qa + 1] = 50.0
            data.qpos[qa + 2] = 100.0
            data.qpos[qa + 3] = 1.0
            data.qpos[qa + 4] = 0.0
            data.qpos[qa + 5] = 0.0
            data.qpos[qa + 6] = 0.0

    def _set_qpos(data, h1, h2, a1, hb, wz):
        data.qpos[h1_q] = h1
        data.qpos[h2_q] = h2
        data.qpos[a1_q] = a1
        data.qpos[th_q] = 0.0
        data.qpos[hb_q] = hb
        data.qpos[wz_q] = wz
        data.qpos[wx_q] = 0.80
        data.qpos[wy_q] = 0.00

    def _set_ctrl(data, h1, h2, a1, hb, wz):
        OFF_H1 = -0.0036
        OFF_H2 = -0.0062
        OFF_A1 = -0.0006
        if h1_a >= 0: data.ctrl[h1_a] = (h1 + OFF_H1) * 100.0
        if h2_a >= 0: data.ctrl[h2_a] = (h2 + OFF_H2) * 100.0
        if a1_a >= 0: data.ctrl[a1_a] = (a1 + OFF_A1) * 100.0
        if th_a >= 0: data.ctrl[th_a] = 0.0
        if hb_a >= 0: data.ctrl[hb_a] = hb
        if wz_a >= 0: data.ctrl[wz_a] = wz * (1.0 + WZ_PD_RATIO)
        if wx_a >= 0: data.ctrl[wx_a] = 0.80
        if wy_a >= 0: data.ctrl[wy_a] = 0.00

    def _measure_kinematic(h1, h2, a1, hb, wz):
        mujoco.mj_resetDataKeyframe(model, data_kin, home_key)
        _park_arm2(data_kin)
        _park_pickup_objs(data_kin)
        _set_qpos(data_kin, h1, h2, a1, hb, wz)
        mujoco.mj_forward(model, data_kin)
        R = data_kin.xmat[chassis_bid].reshape(3, 3)
        rel = (data_kin.xpos[palm_bid].copy()
               - data_kin.xpos[chassis_bid].copy())
        return R.T @ rel

    def _measure_dynamic(h1_t, h2_t, a1_t, hb_t, wz_t):
        mujoco.mj_resetDataKeyframe(model, data_phys, home_key)
        _park_arm2(data_phys)
        _park_pickup_objs(data_phys)
        _set_qpos(data_phys,
                  START_Q["h1"], START_Q["h2"], START_Q["a1"],
                  START_Q["hb"], START_Q["wz"])
        _set_ctrl(data_phys,
                  START_Q["h1"], START_Q["h2"], START_Q["a1"],
                  START_Q["hb"], START_Q["wz"])
        for _ in range(PRESTAB_STEPS):
            mujoco.mj_step(model, data_phys)
        _park_arm2(data_phys)
        _park_pickup_objs(data_phys)

        for s in range(RAMP_STEPS):
            alpha = (s + 1) / RAMP_STEPS
            alpha_h = _smoothstep(alpha)
            h1_c = (1.0 - alpha_h) * START_Q["h1"] + alpha_h * h1_t
            h2_c = (1.0 - alpha_h) * START_Q["h2"] + alpha_h * h2_t
            a1_c = (1.0 - alpha)   * START_Q["a1"] + alpha   * a1_t
            hb_c = (1.0 - alpha)   * START_Q["hb"] + alpha   * hb_t
            wz_c = (1.0 - alpha)   * START_Q["wz"] + alpha   * wz_t
            _set_ctrl(data_phys, h1_c, h2_c, a1_c, hb_c, wz_c)
            mujoco.mj_step(model, data_phys)
        _park_pickup_objs(data_phys)

        _set_ctrl(data_phys, h1_t, h2_t, a1_t, hb_t, wz_t)
        for _ in range(SETTLE_STEPS):
            mujoco.mj_step(model, data_phys)

        R = data_phys.xmat[chassis_bid].reshape(3, 3)
        rel = (data_phys.xpos[palm_bid].copy()
               - data_phys.xpos[chassis_bid].copy())
        return R.T @ rel

    if smoke_test:
        print(f"[calib-dyn-v2] SMOKE TEST: {len(SMOKE_TEST_POSES)} hand-picked poses")
        t0 = time.time()
        results = []
        for idx, (h1, h2, a1, hb, wz) in enumerate(SMOKE_TEST_POSES):
            try:
                p_kin = _measure_kinematic(h1, h2, a1, hb, wz)
                p_phys = _measure_dynamic(h1, h2, a1, hb, wz)
                err = p_phys - p_kin
                ok = bool(np.all(np.isfinite(err)))
            except Exception as _e:
                err = np.array([float('nan')] * 3)
                ok = False
                print(f"  [pose {idx}] EXCEPTION: {_e}")
            results.append((h1, h2, a1, hb, wz, err, ok))
            print(f"  [pose {idx+1:2d}/{len(SMOKE_TEST_POSES)}] "
                  f"h1={h1:.2f} h2={h2:.2f} a1={h2:.2f} "
                  f"hb={hb:+.2f} wz={wz:+.2f}  →  "
                  f"err=({err[0]*100:+5.1f},{err[1]*100:+5.1f},"
                  f"{err[2]*100:+5.1f})cm  "
                  f"|err|={np.linalg.norm(err)*100:.1f}cm  ok={ok}")
        elapsed = time.time() - t0
        print(f"\n[calib-dyn-v2] Smoke test done in {elapsed:.1f}s  "
              f"({elapsed/len(SMOKE_TEST_POSES):.2f}s/pose)")
        mags = [np.linalg.norm(r[5]) for r in results if r[6]]
        if mags:
            print(f"[calib-dyn-v2] |err| stats: min={min(mags)*100:.1f}cm  "
                  f"mean={np.mean(mags)*100:.1f}cm  "
                  f"max={max(mags)*100:.1f}cm")
            pose4_drift = np.linalg.norm(results[3][5]) if len(results) > 3 else float('nan')
            spread = max(mags) - min(mags)
            print(f"[calib-dyn-v2] sanity: pose-4 (small reach) drift "
                  f"= {pose4_drift*100:.1f}cm; spread (max-min) = "
                  f"{spread*100:.1f}cm")
            if pose4_drift > 0.10:
                print(f"[calib-dyn-v2] ⚠️  pose-4 (small reach) > 10cm "
                      f"— sim may be unstable or unit conversion off.  "
                      f"Inspect individual poses.")
            elif spread < 0.05:
                print(f"[calib-dyn-v2] ⚠️  drift values too similar "
                      f"(spread < 5cm) — measurements may be chaotic "
                      f"noise, not config-dependent drift.")
            else:
                print(f"[calib-dyn-v2] ✓ Measurements vary with config "
                      f"AND small-reach pose has small drift — physics "
                      f"appears stable, drift signal is real.  Ready "
                      f"for full sweep.  Note: max drift (extended-reach "
                      f"configs) may reach 40-50cm; that matches runtime "
                      f"observation (Round 36: z drift -25cm at a1=0.37).")
        full_est = elapsed * (2625 / len(SMOKE_TEST_POSES)) / 60.0
        print(f"[calib-dyn-v2] Full-sweep ETA: ~{full_est:.0f} min")
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate MORPH-I arm DYNAMIC calibration LUT "
                    "via smooth-ramp protocol.  See module docstring.")
    parser.add_argument(
        "--xml", default=None,
        help="Path to MuJoCo XML world file.  "
             "Defaults to src/env/market_world_m1.xml.")
    parser.add_argument(
        "--out", default=None,
        help="Output .npz path.  Defaults to "
             "data/arm_calibration_<mode>_dynamic.npz.")
    parser.add_argument(
        "--wrist-mode", default="sidegrip",
        choices=sorted(MODE_CONFIGS.keys()),
        help="Wrist orientation to calibrate for.  "
             "Currently only sidegrip (the --no-chassis-push regime).")
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run 10 hand-picked poses (~15 s) to validate the script "
             "produces stable drift values before committing to the "
             "full sweep.")
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Parallel worker count (spawn processes).  Default: "
             "min(8, cpu_count-4).  Each worker loads its own ~200MB "
             "mujoco model; keep ≤ 8 unless you have lots of RAM.")
    args = parser.parse_args()

    if args.workers is not None:
        N_WORKERS_OVERRIDE = max(1, int(args.workers))

    xml_path = args.xml or os.path.join(
        ROOT, "src", "env", "market_world_m1.xml")
    out_path = args.out or os.path.join(
        ROOT, "data", f"arm_calibration_{args.wrist_mode}_dynamic.npz")

    if not os.path.exists(xml_path):
        sys.exit(f"[calib-dyn-v2] XML not found: {xml_path}")
    calibrate_dynamic_v2(xml_path, out_path, args.wrist_mode,
                         smoke_test=args.smoke_test)
