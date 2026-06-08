import os, sys, time, math, threading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
import mujoco
from simulations.morph_i_free_move import ParallelRobot
from navigation.ompl_windows_bridge import InProcessNavigator

XML = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src',
                                   'env', 'market_world_m1.xml'))
RACK_OPENING_Y = -3.41
LEVEL_D = {"low": 0.75, "mid": 0.65, "high": 0.70}
FINAL_YAW = -math.pi / 2.0
START_XY_YAW = (float(os.environ.get("START_X", "1.5")),
                float(os.environ.get("START_Y", "-5.3")),
                math.pi / 2.0)
PER_SLOT_TIMEOUT = 45.0

sim = ParallelRobot(XML, run_mode="headless", record=False)
nav = InProcessNavigator(sim)
mujoco.mj_forward(sim.model, sim.data)

base_qadr = int(getattr(nav, "validator", nav).base_qposadr)
base_z = float(sim.data.qpos[base_qadr + 2])


def teleport_base(x, y, yaw):
    nav.cancel()
    time.sleep(0.3)
    with sim._target_lock:
        sim.data.qpos[base_qadr + 0] = x
        sim.data.qpos[base_qadr + 1] = y
        sim.data.qpos[base_qadr + 2] = base_z
        sim.data.qpos[base_qadr + 3] = math.cos(yaw / 2.0)
        sim.data.qpos[base_qadr + 4] = 0.0
        sim.data.qpos[base_qadr + 5] = 0.0
        sim.data.qpos[base_qadr + 6] = math.sin(yaw / 2.0)
        sim.data.qvel[:] = 0.0
        sim.target_base = np.array([x, y, yaw], dtype=float)
        sim.base_integral_1 = sim.base_integral_2 = 0.0
        sim.base_prev_error_1 = sim.base_prev_error_2 = 0.0
    mujoco.mj_forward(sim.model, sim.data)


def slot_standoff(i):
    sid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_SITE, f"shelf_slot_{i}")
    p = sim.data.site_xpos[sid].copy()
    z = float(p[2])
    lvl = "low" if z < 0.45 else ("mid" if z < 0.95 else "high")
    return (float(p[0]), RACK_OPENING_Y + LEVEL_D[lvl]), lvl, p


print(f"{'slot':>4} {'lvl':>4} {'standoff':>16} {'result':>8} "
      f"{'dist':>7} {'yaw_err':>8} {'t(s)':>6}")
results = []
for i in range(10):
    (sx, sy), lvl, p = slot_standoff(i)
    teleport_base(*START_XY_YAW)
    if os.environ.get("LOADED") == "1":
        _ch = float(os.environ.get("CARRY_H", "0.60"))
        with sim._target_lock:
            sim.direct_arm_commands[:] = [_ch, _ch + 0.05, 0.10, 0.0,
                                          0.0, -1.88, 0.80, 0.0]
        for _ in range(150):
            sim.step_simulation(render=False)
    for _ in range(30):
        sim.step_simulation(render=False)
    _sx0, _sy0, _ = sim.localization()
    if math.hypot(_sx0 - START_XY_YAW[0], _sy0 - START_XY_YAW[1]) > 0.2:
        print(f"  [warn] teleport drift: start=({_sx0:.2f},{_sy0:.2f}) "
              f"want=({START_XY_YAW[0]:.2f},{START_XY_YAW[1]:.2f})")
    done = {"ok": None}
    def _cb(ok, _d=done):
        _d["ok"] = bool(ok)
    t0 = time.time()
    nav.navigate_to((sx, sy), on_complete=_cb, final_yaw=FINAL_YAW,
                    allow_goal_nudge=True)
    while done["ok"] is None and time.time() - t0 < PER_SLOT_TIMEOUT:
        sim.step_simulation(render=False)
    dt = time.time() - t0
    if done["ok"] is None:
        nav.cancel()
        res = "TIMEOUT"
        dist = yaw_err = float('nan')
    else:
        cx, cy, cyaw = sim.localization()
        dist = math.hypot(cx - sx, cy - sy)
        yaw_err = abs(((cyaw - FINAL_YAW + math.pi) % (2 * math.pi)) - math.pi)
        res = "REACH" if done["ok"] else "FAIL"
    results.append((i, lvl, res, dist, math.degrees(yaw_err) if yaw_err == yaw_err else yaw_err, dt))
    print(f"{i:>4} {lvl:>4} ({sx:6.2f},{sy:6.2f}) {res:>8} "
          f"{dist*100:6.1f}c {math.degrees(yaw_err) if yaw_err==yaw_err else float('nan'):7.1f}d {dt:6.1f}")
    time.sleep(0.5)

n_reach = sum(1 for r in results if r[2] == "REACH")
print(f"\n=== {n_reach}/10 slots REACHED ===")
for i, lvl, res, dist, yd, dt in results:
    if res != "REACH":
        print(f"  slot {i} ({lvl}): {res}  dist={dist*100:.1f}cm yaw_err={yd:.1f}deg")
