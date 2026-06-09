import os, sys, time, math, threading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
import mujoco
from simulations.morph_i_free_move import ParallelRobot
from navigation.arm_planner import MORPHBridge, CARRY_ANCHOR_FINGER_BODIES
from navigation.grasp_executor import GraspExecutor, reset_plan_data_for_ik

XML = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src',
                                   'env', 'market_world_m1.xml'))
SIDE_GRIP_Q = [0.30, 0.35, 0.30, 0.0, 0.0, -1.88, 0.80, 0.0]

sim = ParallelRobot(XML, run_mode="headless", record=False)
arm_bridge = MORPHBridge(XML, arm=1, use_calibration=False,
                         calib_wrist_mode="sidegrip")
gx = GraspExecutor(sim, arm_bridge)
mujoco.mj_forward(sim.model, sim.data)

gx._carry_anchor_body_ids = [
    mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, nm)
    for nm in CARRY_ANCHOR_FINGER_BODIES]
print("carry-anchor body ids:", gx._carry_anchor_body_ids)

with sim._target_lock:
    sim.direct_arm_commands[:] = SIDE_GRIP_Q
for _ in range(200):
    sim.step_simulation(render=False)

_loc = sim.localization()
reset_plan_data_for_ik(arm_bridge, base_xy=(float(_loc[0]), float(_loc[1])),
                       base_yaw=float(_loc[2]))

_stop = threading.Event()
def _phys():
    while not _stop.is_set():
        sim.step_simulation(render=False)
        time.sleep(0.0)
threading.Thread(target=_phys, daemon=True).start()
time.sleep(0.3)

c0 = np.asarray(gx._carry_anchor_xyz(sim.data), dtype=float)
print(f"start centroid = ({c0[0]:.3f},{c0[1]:.3f},{c0[2]:.3f})")

OFFSETS = [(0.08, 0.0), (-0.08, 0.0), (0.0, 0.08), (0.06, 0.06), (-0.06, -0.06)]
print(f"\n{'offset(cm)':>14} {'final_err':>10} {'xy_err':>8} {'z_drift':>8}")
results = []
for dx, dy in OFFSETS:
    cur = np.asarray(gx._carry_anchor_xyz(sim.data), dtype=float)
    target = (float(cur[0] + dx), float(cur[1] + dy), float(cur[2]))
    err = gx._cartesian_move_closed_loop(target, label=f"cl({dx*100:+.0f},{dy*100:+.0f})")
    final = np.asarray(gx._carry_anchor_xyz(sim.data), dtype=float)
    xy_err = math.hypot(final[0] - target[0], final[1] - target[1])
    z_drift = final[2] - target[2]
    results.append((dx, dy, err, xy_err, z_drift))
    print(f"({dx*100:+.0f},{dy*100:+.0f})".rjust(14)
          + f" {err*100:9.1f}c {xy_err*100:7.1f}c {z_drift*100:+7.1f}c")
    time.sleep(0.3)

_stop.set(); time.sleep(0.2)
ok = sum(1 for r in results if r[3] < 0.015 and abs(r[4]) < 0.02)
print(f"\n=== {ok}/{len(results)} moves accurate (xy<1.5cm, |z_drift|<2cm) ===")
