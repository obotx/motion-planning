
import argparse
import os
import sys
import time

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from navigation.arm_planner import JOINT_RANGES_ARM, MORPHBridge


def build_reachability(xml_path, out_path, wrist_mode, n_samples,
                       use_calib=True, seed=42):
    print(f"[reach] Loading model: {xml_path}")
    print(f"[reach] Wrist mode: {wrist_mode}")
    print(f"[reach] N samples: {n_samples:,}")
    print(f"[reach] Compose with CALIB LUT: {use_calib}")

    bridge = MORPHBridge(
        xml_path, arm=1,
        use_calibration=use_calib,
        calib_wrist_mode=wrist_mode,
    )
    model = bridge._model
    data = bridge._plan_data

    qmap = bridge._qpos_map
    qi = [qmap["ColumnLeft"], qmap["ColumnRight"], qmap["ArmLeft"],
          qmap["Base"], qmap["HandBearing"], qmap["WristZ"],
          qmap["WristX"], qmap["WristY"]]

    allowed_pairs = set(bridge._rest_pairs)
    min_clearance = bridge._checker._min_clearance
    alpha_sq = bridge._checker._alpha_sq
    d2 = bridge._checker._d2

    def _is_valid_post_forward(h1, h2):
        alpha_deg = float(np.degrees(np.arctan2(h2 - h1, d2)))
        if not (alpha_deg * alpha_deg >= alpha_sq):
            return False
        for i in range(data.ncon):
            c = data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if (g1, g2) in allowed_pairs or (g2, g1) in allowed_pairs:
                continue
            if c.dist < min_clearance:
                return False
        return True

    grip_bid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "Gripper_Link1_1")
    chassis_bid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "robot")
    if grip_bid < 0 or chassis_bid < 0:
        sys.exit(f"[reach] missing body: grip={grip_bid} chassis={chassis_bid}")

    rng = np.random.default_rng(seed)
    lows = np.array([r[0] for r in JOINT_RANGES_ARM], dtype=float)
    highs = np.array([r[1] for r in JOINT_RANGES_ARM], dtype=float)
    lows[0], highs[0] = 0.05, 0.35
    lows[1], highs[1] = 0.05, 0.65
    lows[2], highs[2] = 0.05, 0.65
    samples = rng.uniform(lows, highs, size=(n_samples, 8))

    sample_pos = np.zeros((n_samples, 3), dtype=np.float32)
    self_collision = np.zeros(n_samples, dtype=bool)

    t0 = time.time()
    print(f"[reach] Sweeping {n_samples:,} samples ...")
    for n in range(n_samples):
        q = samples[n]
        for i, addr in enumerate(qi):
            data.qpos[addr] = q[i]
        mujoco.mj_forward(model, data)
        R = data.xmat[chassis_bid].reshape(3, 3)
        rel = data.xpos[grip_bid] - data.xpos[chassis_bid]
        p_local_kin = R.T @ rel

        if use_calib and bridge._calib is not None:
            calib_corr_world = bridge._calib_error(
                q[0], q[1], q[2], q[3], hb=q[4], wz=q[5])
            calib_corr_local = R.T @ calib_corr_world
            p_local = p_local_kin + calib_corr_local
        else:
            p_local = p_local_kin

        sample_pos[n] = p_local.astype(np.float32)
        if not _is_valid_post_forward(q[0], q[1]):
            self_collision[n] = True

        if (n + 1) % 25000 == 0:
            elapsed = time.time() - t0
            rate = (n + 1) / max(1e-3, elapsed)
            eta = (n_samples - (n + 1)) / max(1e-3, rate)
            n_coll = int(self_collision[:n + 1].sum())
            print(f"[reach] {n + 1:,}/{n_samples:,} "
                  f"({100*(n+1)/n_samples:.0f}%) "
                  f"collisions={n_coll:,} ({100*n_coll/(n+1):.1f}%) "
                  f"elapsed={elapsed:.0f}s eta={eta:.0f}s")

    n_collision = int(self_collision.sum())
    n_free = n_samples - n_collision
    print(f"[reach] Done in {time.time() - t0:.1f}s")
    print(f"[reach]   Free samples: {n_free:,} ({100*n_free/n_samples:.1f}%)")
    print(f"[reach]   Self-colliding: {n_collision:,} "
          f"({100*n_collision/n_samples:.1f}%)")
    print(f"[reach] Bounding box of FREE samples (chassis-local):")
    free_pos = sample_pos[~self_collision]
    if len(free_pos):
        print(f"        x: [{free_pos[:, 0].min():+.3f}, "
              f"{free_pos[:, 0].max():+.3f}]")
        print(f"        y: [{free_pos[:, 1].min():+.3f}, "
              f"{free_pos[:, 1].max():+.3f}]")
        print(f"        z: [{free_pos[:, 2].min():+.3f}, "
              f"{free_pos[:, 2].max():+.3f}]")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path,
        sample_q=samples.astype(np.float32),
        sample_pos=sample_pos,
        self_collision=self_collision,
        wrist_mode=wrist_mode,
        n_samples=np.asarray(n_samples, dtype=np.int64),
    )
    file_mb = os.path.getsize(out_path) / 1e6
    print(f"[reach] Wrote LUT to {out_path}  ({file_mb:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a 3D Cartesian reachability LUT for the "
                    "MORPH-I arm — used as deterministic warm-start "
                    "for IK.")
    parser.add_argument("--xml", default=None,
                        help="Path to the MuJoCo scene XML. Defaults to "
                             "src/env/market_world_m1.xml.")
    parser.add_argument("--out", default=None,
                        help="Output .npz path. Defaults to "
                             "data/arm_reachability_<wrist-mode>.npz.")
    parser.add_argument("--wrist-mode", default="sidegrip",
                        choices=["sidegrip", "topdown"],
                        help="Wrist mode CALIB LUT to compose with. "
                             "Defaults to sidegrip.")
    parser.add_argument("--n-samples", type=int, default=500_000,
                        help="Number of joint-space samples (default 500K).")
    parser.add_argument("--no-calib", action="store_true",
                        help="Skip CALIB composition — store bare kinematic "
                             "FK positions (less accurate, faster).")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for sample reproducibility.")
    args = parser.parse_args()

    xml_path = args.xml or os.path.join(
        ROOT, "src", "env", "market_world_m1.xml")
    out_path = args.out or os.path.join(
        ROOT, "data", f"arm_reachability_{args.wrist_mode}.npz")
    if not os.path.exists(xml_path):
        sys.exit(f"[reach] XML not found: {xml_path}")
    build_reachability(
        xml_path, out_path, args.wrist_mode, args.n_samples,
        use_calib=not args.no_calib, seed=args.seed)
