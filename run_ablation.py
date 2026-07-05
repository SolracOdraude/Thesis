#!/usr/bin/env python
"""
run_ablation.py
===============
Grid-search ablation over QEggRoll hyperparameters.

Supports two environments:
  ablation_cartpole  — CartPole-v1  (discrete, ~3 h)
  ablation_pendulum  — Pendulum-v1  (continuous, ~4 h)

Grid axes (same for both):
  pop_size              : 128, 256, 512
  rank                  : 1, 2, 4
  sigma_shift           : 1, 2, 4   (σ = 2^{-(4+s)} ≈ 0.031, 0.016, 0.004)
  n_parallel_evaluations: 1, 2, 4

3^4 = 81 runs per environment, seed 0.

Usage
-----
  python run_ablation.py ablation_cartpole           # CartPole grid (~3 h)
  python run_ablation.py ablation_pendulum           # Pendulum grid (~4 h)
  python run_ablation.py ablation_pendulum --dry_run
  python run_ablation.py ablation_pendulum --only p512
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from itertools import product
from typing import Dict, List

from metrics import RunLogger

HERE   = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "experiments.py")

NOISE_RE = re.compile(
    r"xla_cuda|cuInit|CUDA error|discover_pjrt|check_cuda|device_count|warp"
)

SEED           = 0
NOISE_SIZE_EXP = 24   # 16 MB noise table — no hash collisions at pop=512
LOG_EVERY      = 20

POP_SIZES    = [128, 256, 512]
RANKS        = [1, 2, 4]
SIGMA_SHIFTS = [1, 2, 4]
N_EVALS      = [1, 2, 4]


# ── Suite definitions ─────────────────────────────────────────────────────────

def suite_ablation_cartpole() -> List[Dict]:
    """81-run grid on CartPole-v1 (discrete control, 300 epochs)."""
    runs = []
    for pop, rank, sig, npar in product(POP_SIZES, RANKS, SIGMA_SHIFTS, N_EVALS):
        tag = f"abl_p{pop}_r{rank}_s{sig}_k{npar}"
        runs.append({
            "env":                    "CartPole-v1",
            "num_epochs":             300,
            "pop_size":               pop,
            "rank":                   rank,
            "sigma_shift":            sig,
            "n_parallel_evaluations": npar,
            "tag":                    tag,
        })
    return runs


def suite_ablation_pendulum() -> List[Dict]:
    """81-run grid on Pendulum-v1 (continuous control, 400 epochs).

    Pendulum is a better ablation target than CartPole for continuous control:
      - sigma_shift matters more (large σ overshoots the torque range)
      - rank affects how precisely the perturbation shapes the continuous output
      - pop_size affects gradient estimate quality on a harder fitness landscape
    400 epochs gives enough signal to distinguish failing from converging configs
    while keeping each run to ~2-3 min.
    """
    runs = []
    for pop, rank, sig, npar in product(POP_SIZES, RANKS, SIGMA_SHIFTS, N_EVALS):
        tag = f"abl_pendulum_p{pop}_r{rank}_s{sig}_k{npar}"
        runs.append({
            "env":                    "Pendulum-v1",
            "num_epochs":             400,
            "pop_size":               pop,
            "rank":                   rank,
            "sigma_shift":            sig,
            "n_parallel_evaluations": npar,
            "tag":                    tag,
        })
    return runs


SUITES = {
    "ablation_cartpole": suite_ablation_cartpole,
    "ablation_pendulum": suite_ablation_pendulum,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def result_path(cfg: Dict) -> str:
    return RunLogger.default_path("QEggRoll", cfg["env"], SEED, "results", cfg["tag"])


def run_label(cfg: Dict) -> str:
    return (f"QEggRoll  {cfg['env']}  seed{SEED}  "
            f"pop={cfg['pop_size']}  rank={cfg['rank']}  "
            f"sig={cfg['sigma_shift']}  k={cfg['n_parallel_evaluations']}")


def build_command(python: str, cfg: Dict, max_cpus: int) -> List[str]:
    cmd = [python, SCRIPT,
           "--env",                    cfg["env"],
           "--seed",                   str(SEED),
           "--num_epochs",             str(cfg["num_epochs"]),
           "--log_every",              str(LOG_EVERY),
           "--noise_size_exp",         str(NOISE_SIZE_EXP),
           "--pop_size",               str(cfg["pop_size"]),
           "--rank",                   str(cfg["rank"]),
           "--sigma_shift",            str(cfg["sigma_shift"]),
           "--n_parallel_evaluations", str(cfg["n_parallel_evaluations"]),
           "--run_tag",                cfg["tag"],
           "--save_results",
           ]
    if max_cpus > 0 and shutil.which("taskset"):
        cmd = ["taskset", "-c", f"0-{max_cpus-1}"] + cmd
    return cmd


def limited_env(max_cpus: int) -> Dict:
    env = os.environ.copy()
    if max_cpus > 0:
        env["OMP_NUM_THREADS"]      = str(max_cpus)
        env["OPENBLAS_NUM_THREADS"] = str(max_cpus)
        env["MKL_NUM_THREADS"]      = str(max_cpus)
        flags = env.get("XLA_FLAGS", "")
        if "xla_cpu_multi_thread_eigen" not in flags:
            env["XLA_FLAGS"] = (flags + " --xla_cpu_multi_thread_eigen=false").strip()
    return env


def stream(proc, logfile, filter_noise: bool):
    for line in proc.stdout:
        if filter_noise and NOISE_RE.search(line):
            continue
        sys.stdout.write(line)
        sys.stdout.flush()
        if logfile:
            logfile.write(line)
            logfile.flush()


def run_one(cfg: Dict, args, logfile) -> str:
    rpath = result_path(cfg)
    if not args.force and os.path.exists(rpath):
        print(f"  [skip] (exists): {rpath}")
        return "skipped"

    cmd = build_command(args.python, cfg, args.max_cpus)
    if args.dry_run:
        print("  $ " + " ".join(cmd))
        return "dry-run"

    t0 = time.time()
    try:
        proc = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1,
                                env=limited_env(args.max_cpus))
        try:
            stream(proc, logfile, not args.no_filter)
            proc.wait(timeout=args.timeout if args.timeout > 0 else None)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            print(f"  [TIMEOUT] after {args.timeout}s")
            return "timeout"
    except Exception as e:
        print(f"  [ERROR] launch error: {e}")
        return "error"

    dt = time.time() - t0
    if proc.returncode == 0:
        print(f"  [done] in {dt/60:.1f} min")
        return "done"
    print(f"  [FAILED] (exit {proc.returncode}) after {dt/60:.1f} min")
    return "failed"


# ── Runner ────────────────────────────────────────────────────────────────────

def hr(char="="):
    return char * 60


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("suite", choices=sorted(SUITES), help="which batch to run")
    p.add_argument("--python",
                   default=os.path.join(HERE, "thesisEnv", "bin", "python"),
                   help="Python interpreter to use (default: thesisEnv venv)")
    p.add_argument("--timeout",   type=int, default=0,
                   help="per-run timeout in seconds (0 = no limit)")
    p.add_argument("--max_cpus",  type=int, default=4,
                   help="pin each run to this many cores")
    p.add_argument("--force",     action="store_true",
                   help="re-run even if a result JSON already exists")
    p.add_argument("--dry_run",   action="store_true",
                   help="print commands without running them")
    p.add_argument("--only",      default="",
                   help="only run specs whose label contains this substring")
    p.add_argument("--no_filter", action="store_true",
                   help="do not filter harmless CUDA-probe lines")
    p.add_argument("--logfile",   default="",
                   help="tee all output to this file (default: <suite>.log)")
    args = p.parse_args()

    if not os.path.exists(args.python):
        print(f"WARNING: {args.python} not found; falling back to {sys.executable}")
        args.python = sys.executable

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)

    runs = SUITES[args.suite]()
    if args.only:
        runs = [r for r in runs if args.only in run_label(r)]

    envs  = sorted({r["env"] for r in runs})
    logpath = args.logfile or os.path.join(HERE, f"{args.suite}.log")
    logfile = None if args.dry_run else open(logpath, "a")

    print(hr())
    print(f" suite={args.suite}   runs={len(runs)}   env={envs}   seed={SEED}")
    print(f" grid: pop∈{POP_SIZES}  rank∈{RANKS}  sigma_shift∈{SIGMA_SHIFTS}  k∈{N_EVALS}")
    if not args.dry_run:
        print(f" logging to {logpath}")
    print(hr())

    summary: Dict[str, int] = {}
    batch_t0 = time.time()
    for i, cfg in enumerate(runs, 1):
        stamp = time.strftime("%H:%M:%S")
        print(f"\n{hr()}\n [{i}/{len(runs)}] {stamp}  {run_label(cfg)}\n{hr()}")
        status = run_one(cfg, args, logfile)
        summary[status] = summary.get(status, 0) + 1

    if logfile:
        logfile.close()

    print(f"\n{hr()}")
    print(f" SUITE '{args.suite}' COMPLETE in {(time.time()-batch_t0)/60:.1f} min")
    print(" " + "   ".join(f"{k}={v}" for k, v in sorted(summary.items())))
    print(hr())
    sys.exit(1 if (summary.get("failed", 0) or summary.get("timeout", 0)
                   or summary.get("error", 0)) else 0)


if __name__ == "__main__":
    main()
