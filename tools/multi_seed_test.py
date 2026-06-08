
import argparse
import csv
import os
import re
import subprocess
import sys
import time


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

STAGE_RE = re.compile(
    r"\[STAGE-CLASS\] obj=\d+ stage=(?P<stage>[A-D]) \((?P<desc>[^)]+)\)\s+"
    r"close_fired=(?P<close>True|False) "
    r"verify_passed=(?P<verify>True|False) "
    r"lift_fired=(?P<lift>True|False) "
    r"obj_followed=(?P<follow>True|False)"
)
END_RE = re.compile(
    r"\[AUTO_RUN\] cycle \d+/\d+ END status=(?P<status>\S+) "
    r"duration=(?P<dur>\d+\.\d+)s"
)
DIAG_TILT_RE = re.compile(
    r"\[DIAG-TILT\] pre-close-gate: h2-h1=([+-]?\d+\.\d+)cm\s+"
    r"hb=([+-]?\d+\.\d+)rad.*"
    r"wz=([+-]?\d+\.\d+)rad \(resid ([+-]?\d+\.\d+)"
)
PRECLOSE_GATE_RE = re.compile(
    r"\[Exec\] \[5\.5\] pre-close gate \([^)]+\): "
    r"carry_gap=(?P<gap>\d+\.\d+)cm.*"
    r"d_thumb=(?P<thumb>\d+\.\d+)cm\s+d_bc=(?P<bc>\d+\.\d+)cm"
)


def run_one_seed(seed, args, per_cycle_timeout):
    env = os.environ.copy()
    env["OMPL_BRIDGE_MODE"] = "native"
    env["PYTHONPATH"] = "src"
    env["RUN_LOG_PATH"] = f"/tmp/multi_seed_{seed}.log"

    cmd = [
        ".venv/bin/python", "src/gui/play_m1.py",
        "--seed", str(seed),
        "--auto-move-attempts", "1",
        "--auto-cycle-timeout", str(per_cycle_timeout),
    ]
    if args.no_chassis_push:
        cmd.append("--no-chassis-push")
    if args.extra_args:
        cmd.extend(args.extra_args.split())

    result = {
        "seed": seed,
        "stage": "?",
        "close_fired": False,
        "verify_passed": False,
        "lift_fired": False,
        "obj_followed": False,
        "duration_s": None,
        "carry_gap_cm": None,
        "d_thumb_cm": None,
        "d_bc_cm": None,
        "wz_resid_rad": None,
        "h2_h1_cm": None,
        "notes": "",
    }

    t0 = time.time()
    print(f"\n=== seed {seed} START (timeout {per_cycle_timeout + 60}s) ===")
    try:
        proc = subprocess.Popen(
            cmd, cwd=ROOT, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=1, universal_newlines=True,
        )
        try:
            for line in proc.stdout:
                if (STAGE_RE.search(line) or END_RE.search(line)
                        or DIAG_TILT_RE.search(line)
                        or PRECLOSE_GATE_RE.search(line)
                        or "[PICK]" in line or "[Exec] [5.5]" in line
                        or "Segmentation" in line):
                    sys.stdout.write(f"  [seed {seed}] {line.rstrip()}\n")
                    sys.stdout.flush()
                m = STAGE_RE.search(line)
                if m:
                    result["stage"] = m.group("stage")
                    result["close_fired"] = (m.group("close") == "True")
                    result["verify_passed"] = (m.group("verify") == "True")
                    result["lift_fired"] = (m.group("lift") == "True")
                    result["obj_followed"] = (m.group("follow") == "True")
                m = END_RE.search(line)
                if m:
                    result["duration_s"] = float(m.group("dur"))
                m = DIAG_TILT_RE.search(line)
                if m:
                    result["h2_h1_cm"] = float(m.group(1))
                    result["wz_resid_rad"] = float(m.group(4))
                m = PRECLOSE_GATE_RE.search(line)
                if m:
                    result["carry_gap_cm"] = float(m.group("gap"))
                    result["d_thumb_cm"] = float(m.group("thumb"))
                    result["d_bc_cm"] = float(m.group("bc"))
        except Exception as _e_stream:
            result["notes"] = f"stream error: {_e_stream}"
        rc = proc.wait(timeout=60)
        if rc != 0:
            result["notes"] = (result["notes"] + " " if result["notes"]
                               else "") + f"rc={rc}"
    except subprocess.TimeoutExpired:
        proc.kill()
        result["notes"] = "HUNG/TIMEOUT"
    except Exception as _e:
        result["notes"] = f"exception: {_e}"
    finally:
        result["wall_clock_s"] = round(time.time() - t0, 1)
    return result


def print_summary(results):
    print("\n" + "=" * 100)
    print("MULTI-SEED TEST SUMMARY")
    print("=" * 100)
    header = (f"{'seed':>4} | {'stage':>5} | {'close':>5} | {'verify':>6} | "
              f"{'lift':>4} | {'follow':>6} | {'gap':>5} | {'thumb':>5} | "
              f"{'bc':>5} | {'wz_r':>6} | {'h2-h1':>6} | {'dur':>5} | notes")
    print(header)
    print("-" * len(header))
    stage_counts = {"A": 0, "B": 0, "C": 0, "D": 0, "?": 0}
    durations = []
    carry_gaps = []
    wz_resids = []
    for r in results:
        stage = r["stage"]
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
        if r["duration_s"]:
            durations.append(r["duration_s"])
        if r["carry_gap_cm"] is not None:
            carry_gaps.append(r["carry_gap_cm"])
        if r["wz_resid_rad"] is not None:
            wz_resids.append(abs(r["wz_resid_rad"]))
        def fmt(v, w, ndig=1):
            if v is None:
                return f"{'-':>{w}}"
            if isinstance(v, bool):
                return f"{('y' if v else 'n'):>{w}}"
            if isinstance(v, float):
                return f"{v:{w}.{ndig}f}"
            return f"{v:>{w}}"
        print(f"{r['seed']:>4} | "
              f"{r['stage']:>5} | "
              f"{fmt(r['close_fired'], 5)} | "
              f"{fmt(r['verify_passed'], 6)} | "
              f"{fmt(r['lift_fired'], 4)} | "
              f"{fmt(r['obj_followed'], 6)} | "
              f"{fmt(r['carry_gap_cm'], 5)} | "
              f"{fmt(r['d_thumb_cm'], 5)} | "
              f"{fmt(r['d_bc_cm'], 5)} | "
              f"{fmt(r['wz_resid_rad'], 6, 3)} | "
              f"{fmt(r['h2_h1_cm'], 6)} | "
              f"{fmt(r['duration_s'], 5, 0)} | "
              f"{r['notes']}")
    print("-" * len(header))
    n = len(results)
    print(f"\nStage counts: {stage_counts}")
    pass_rate_d = stage_counts.get("D", 0) / max(1, n)
    print(f"Stage D pass rate: {stage_counts.get('D', 0)}/{n} ({100*pass_rate_d:.0f}%)")
    if durations:
        print(f"Mean cycle duration: {sum(durations)/len(durations):.1f}s  "
              f"(range {min(durations):.0f}-{max(durations):.0f}s)")
    if carry_gaps:
        print(f"Mean carry_gap at pre-close: {sum(carry_gaps)/len(carry_gaps):.1f}cm  "
              f"(range {min(carry_gaps):.1f}-{max(carry_gaps):.1f}cm)")
    if wz_resids:
        print(f"Mean |wz residual| at pre-close: "
              f"{sum(wz_resids)/len(wz_resids):.3f}rad  "
              f"(range {min(wz_resids):.3f}-{max(wz_resids):.3f})")
    print("=" * 100)


def save_csv(results, out_path):
    fieldnames = ["seed", "stage", "close_fired", "verify_passed",
                  "lift_fired", "obj_followed", "carry_gap_cm",
                  "d_thumb_cm", "d_bc_cm", "wz_resid_rad", "h2_h1_cm",
                  "duration_s", "wall_clock_s", "notes"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"\n[multi-seed] CSV written: {out_path}")


def parse_seeds(args):
    if args.seeds:
        return [int(s) for s in args.seeds.split(",")]
    if args.seed_range:
        parts = args.seed_range.split("-")
        lo, hi = int(parts[0]), int(parts[1])
        return list(range(lo, hi + 1))
    return [7, 8, 9, 10, 11]


def main():
    parser = argparse.ArgumentParser(
        description="Multi-seed validation harness for play_m1.  Runs the "
                    "cycle across a range of seeds, parses key alignment "
                    "metrics, and prints a summary.")
    parser.add_argument("--seeds", default=None,
                        help="Comma-separated seed list, e.g. 7,8,9,10,11")
    parser.add_argument("--seed-range", default=None,
                        help="Inclusive range like 7-16")
    parser.add_argument("--per-cycle-timeout", type=int, default=250,
                        help="play_m1's --auto-cycle-timeout per cycle (s)")
    parser.add_argument("--no-chassis-push", action="store_true",
                        default=True,
                        help="Pass --no-chassis-push (default: True)")
    parser.add_argument("--baseline", action="store_true",
                        help="Disable --no-chassis-push (baseline mode)")
    parser.add_argument("--extra-args", default="",
                        help="Extra args to pass to play_m1 verbatim")
    parser.add_argument("--out", default=None,
                        help="CSV output path (default: /tmp/multi_seed_results.csv)")
    args = parser.parse_args()
    if args.baseline:
        args.no_chassis_push = False

    seeds = parse_seeds(args)
    print(f"[multi-seed] Running {len(seeds)} cycles: seeds={seeds}")
    print(f"[multi-seed] no_chassis_push={args.no_chassis_push}  "
          f"per_cycle_timeout={args.per_cycle_timeout}s  "
          f"extra_args='{args.extra_args}'")

    t0 = time.time()
    results = []
    for seed in seeds:
        r = run_one_seed(seed, args, args.per_cycle_timeout)
        results.append(r)
        print(f"=== seed {seed} END: stage={r['stage']} "
              f"duration={r['duration_s']}s wall={r['wall_clock_s']}s ===")
    total = time.time() - t0
    print(f"\n[multi-seed] All {len(seeds)} cycles done in {total:.0f}s "
          f"({total/60:.1f} min)")
    print_summary(results)
    out_path = args.out or "/tmp/multi_seed_results.csv"
    save_csv(results, out_path)


if __name__ == "__main__":
    main()
