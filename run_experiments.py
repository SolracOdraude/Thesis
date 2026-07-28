#!/usr/bin/env python
"""
run_experiments.py
==================
Launches QEggRoll and PPO experiments as separate subprocesses.

Each run is a fresh process so JAX memory is released between runs and a
single crash does not abort the whole batch.  Result files are skipped
automatically (use --force to overwrite).

Target environments:
  CartPole-v1, Pendulum-v1, kinetix/s/h1_thrust_over_ball

Suites
------
  thesis     — 20 seeds per method, all 3 envs (main thesis run)
  moreseeds  — seeds 3-19 only (used when thesis was first run with seeds 0-2)
  pendulum   — 20 seeds, Pendulum-v1 only (standalone focused run)
  kinetix    — 20 seeds, kinetix/s/h1_thrust_over_ball only

Usage
-----
  python run_experiments.py thesis --timeout 0 --python python --logfile thesis.log
  python run_experiments.py thesis --dry_run
  python run_experiments.py thesis --only pendulum
  python run_experiments.py thesis --force
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List

from metrics import RunLogger

HERE = os.path.dirname(os.path.abspath(__file__))

SCRIPTS = {
    "QEggRoll": "experiments.py",
    "PPO":      "ppo_baseline.py",
    "DQN":      "dqn_baseline.py",
}

NOISE_RE = re.compile(
    r"xla_cuda|cuInit|CUDA error|discover_pjrt|check_cuda|device_count|warp"
)


# -------------------------------------------------------------------------------
# Run specification
# -------------------------------------------------------------------------------

@dataclass
class RunSpec:
    method: str
    env:    str
    seed:   int
    tag:    str = ""
    flags:  Dict[str, str] = field(default_factory=dict)

    @property
    def result_path(self) -> str:
        return RunLogger.default_path(self.method, self.env, self.seed,
                                      results_dir="results", tag=self.tag)

    def label(self) -> str:
        bits = [self.method, self.env, f"seed{self.seed}"]
        if self.tag:
            bits.append(self.tag)
        return "  ".join(bits)

    def command(self, python: str, max_cpus: int) -> List[str]:
        cmd = [python, os.path.join(HERE, SCRIPTS[self.method]),
               "--env", self.env, "--seed", str(self.seed), "--save_results"]
        if self.tag:
            cmd += ["--run_tag", self.tag]
        for k, v in self.flags.items():
            cmd += [f"--{k}", str(v)]
        if max_cpus > 0 and shutil.which("taskset"):
            cmd = ["taskset", "-c", f"0-{max_cpus - 1}"] + cmd
        return cmd


def limited_env(max_cpus: int) -> Dict[str, str]:
    env = os.environ.copy()
    if max_cpus > 0:
        env["OMP_NUM_THREADS"] = str(max_cpus)
        env["OPENBLAS_NUM_THREADS"] = str(max_cpus)
        env["MKL_NUM_THREADS"] = str(max_cpus)
        flags = env.get("XLA_FLAGS", "")
        if "xla_cpu_multi_thread_eigen" not in flags:
            env["XLA_FLAGS"] = (flags + " --xla_cpu_multi_thread_eigen=false").strip()
    # Disable JAX GPU memory pre-allocation so CUDA memory is released between
    # subprocesses. Without this, each new subprocess hangs waiting for GPU RAM
    # that the previous process hasn't fully returned to the OS yet.
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    return env


# -------------------------------------------------------------------------------
# Suites
# -------------------------------------------------------------------------------

def suite_thesis() -> List[RunSpec]:
    """
    Full thesis comparison: 20 seeds per method across CartPole, Pendulum, and Kinetix.

    QEggRoll parameters match the EggRoll paper (Tables 17/18/29):
      CartPole : pop=2048, rank=4, sigma_shift=2 (sigma~0.2), K=1, 500 epochs
      Pendulum : pop=4096, rank=4, sigma_shift=4 (sigma~0.05), K=1, 500 epochs
      Kinetix  : pop=512,  rank=1, sigma_shift=1 (sigma~0.5),  K=1, 300 epochs

    PPO budgets sized to match QEggRoll env-step totals for a comparable x-axis:
      CartPole : 2048 x 250 x 500 ~= 256M steps  -> PPO 200M
      Pendulum : 4096 x 200 x 500 ~= 410M steps  -> PPO 400M
      Kinetix  :  512 x 150 x 300 ~=  23M steps  -> PPO  20M

    Rough time budget (post flat-vmap fix):
      CartPole  (20Q + 20P):   1 +  5 min/seed x 20  ~=  120 min
      Pendulum  (20Q + 20P):   2 +  4 min/seed x 20  ~=  120 min
      Kinetix   (20Q + 20P):   8 +  5 min/seed x 20  ~=  260 min
                                              TOTAL   ~=  500 min ~= 8-9h

    Existing result files are skipped automatically (safe to resume after interruption).
      python run_experiments.py thesis --timeout 0 --python python --logfile thesis.log
    """
    runs: List[RunSpec] = []
    tag = "thesis"

    EGG_BASE = {"noise_size_exp": 28, "log_every": 20}

    def egg(env, seeds, pop, epochs, **extra):
        flags = {**EGG_BASE, "pop_size": pop, "num_epochs": epochs, **extra}
        return [RunSpec("QEggRoll", env, s, tag=tag, flags=flags) for s in range(seeds)]

    def ppo(env, seeds, total_timesteps=None):
        flags = {"total_timesteps": total_timesteps} if total_timesteps else {}
        return [RunSpec("PPO", env, s, tag=tag, flags=flags) for s in range(seeds)]

    # CartPole: PPO batch_size=256×128=32768; 200M ≈ 6103 update iterations.
    runs += egg("CartPole-v1",  20, pop=2048, epochs=500, rank=4, sigma_shift=2)
    runs += ppo("CartPole-v1",  20, total_timesteps=200_000_000)

    # Pendulum: log_every=10 → 50 checkpoints/seed.
    # PPO batch_size=256×256=65536; 400M ≈ 6104 update iterations.
    runs += egg("Pendulum-v1",  20, pop=4096, epochs=500, rank=4, sigma_shift=4, log_every=10)
    runs += ppo("Pendulum-v1",  20, total_timesteps=400_000_000)

    # Kinetix: PPO batch_size=64×64=4096; 20M ≈ 4880 update iterations.
    runs += egg("kinetix/s/h1_thrust_over_ball", 10, pop=512, epochs=300, rank=1, sigma_shift=1)
    runs += ppo("kinetix/s/h1_thrust_over_ball", 10, total_timesteps=20_000_000)

    return runs


def suite_moreseeds() -> List[RunSpec]:
    """
    Expand CartPole, Pendulum, and Kinetix to 20 seeds each (seeds 3-19 are new).
    Parameters match suite_thesis exactly so result files are pooled correctly.

    Used when suite_thesis was initially run with seeds 0-2 only.
    Seeds 0-2 are skipped automatically (result files already exist).
    """
    runs: List[RunSpec] = []
    tag = "thesis"
    EGG_BASE = {"noise_size_exp": 28, "log_every": 20}

    def egg(env, seeds_range, pop, epochs, **extra):
        flags = {**EGG_BASE, "pop_size": pop, "num_epochs": epochs, **extra}
        return [RunSpec("QEggRoll", env, s, tag=tag, flags=flags) for s in seeds_range]

    def ppo(env, seeds_range, total_timesteps=None):
        flags = {"total_timesteps": total_timesteps} if total_timesteps else {}
        return [RunSpec("PPO", env, s, tag=tag, flags=flags) for s in seeds_range]

    NEW = range(3, 20)

    runs += egg("CartPole-v1", NEW, pop=2048, epochs=500, rank=4, sigma_shift=2)
    runs += ppo("CartPole-v1", NEW, total_timesteps=200_000_000)

    runs += egg("Pendulum-v1", NEW, pop=4096, epochs=500, rank=4, sigma_shift=4, log_every=10)
    runs += ppo("Pendulum-v1", NEW, total_timesteps=400_000_000)

    runs += egg("kinetix/s/h1_thrust_over_ball", NEW, pop=512, epochs=300, rank=1, sigma_shift=1)
    runs += ppo("kinetix/s/h1_thrust_over_ball", NEW, total_timesteps=20_000_000)

    return runs


def suite_pendulum() -> List[RunSpec]:
    """
    Focused Pendulum-v1 standalone run: 20 seeds per method.
    Parameters match suite_thesis exactly so all result files are pooled together.

    QEggRoll  pop=4096, 500 epochs, log_every=10, rank=4, sigma_shift=4.
    PPO       400M steps, batch_size=65536.
    """
    runs: List[RunSpec] = []
    tag = "thesis"

    for s in range(20):
        runs.append(RunSpec("QEggRoll", "Pendulum-v1", s, tag=tag,
                            flags={"noise_size_exp": 28, "log_every": 10,
                                   "pop_size": 4096, "num_epochs": 500,
                                   "rank": 4, "sigma_shift": 4}))
    for s in range(20):
        runs.append(RunSpec("PPO", "Pendulum-v1", s, tag=tag,
                            flags={"total_timesteps": 400_000_000}))

    return runs


def suite_kinetix() -> List[RunSpec]:
    """
    Focused kinetix/s/h1_thrust_over_ball standalone run: 20 seeds per method.
    Parameters match suite_thesis exactly so all result files are pooled together.

    QEggRoll  pop=512, 300 epochs, rank=1, sigma_shift=1.
    PPO       20M steps, batch_size=4096.
    """
    runs: List[RunSpec] = []
    tag = "thesis"
    EGG_BASE = {"noise_size_exp": 28, "log_every": 20}
    KENV = "kinetix/s/h1_thrust_over_ball"

    for s in range(10):
        runs.append(RunSpec("QEggRoll", KENV, s, tag=tag,
                            flags={**EGG_BASE, "pop_size": 512, "num_epochs": 300,
                                   "rank": 1, "sigma_shift": 1}))
    for s in range(10):
        runs.append(RunSpec("PPO", KENV, s, tag=tag,
                            flags={"total_timesteps": 20_000_000}))
    return runs


def suite_mountaincar() -> List[RunSpec]:
    """
    MountainCar-v0 and MountainCarContinuous-v0 — 10 seeds per method each.

    QEggRoll parameters follow the ablation-informed choices in hparams.py
    (pop=2048, rank=4).  sigma_shift differs per task: 1 for the hard-
    exploration discrete task, 2 for the shaped-reward continuous task.

    Step budgets (matching CartPole / Pendulum scale):
      MountainCar           : 2048 × 200 × 500 ≈ 205M  → PPO 200M
      MountainCarContinuous : 2048 × 999 × 200 ≈ 409M  → PPO 400M

    Existing result files are skipped automatically.
      python run_experiments.py mountaincar --timeout 0 --python python
    """
    runs: List[RunSpec] = []
    tag = "thesis"
    EGG_BASE = {"noise_size_exp": 28, "log_every": 20}

    # Discrete MountainCar: hard-exploration, sigma_shift=1
    # 500 epochs matches CartPole's epoch count (same max_steps=200, same pop).
    runs += [RunSpec("QEggRoll", "MountainCar-v0", s, tag=tag,
                     flags={**EGG_BASE, "pop_size": 2048, "num_epochs": 500,
                            "rank": 4, "sigma_shift": 1})
             for s in range(10)]
    runs += [RunSpec("PPO", "MountainCar-v0", s, tag=tag,
                     flags={"total_timesteps": 200_000_000})
             for s in range(10)]

    # Continuous MountainCar: shaped reward, sigma_shift=2.
    # 200 epochs matches Pendulum's step budget (pop=2048 vs 4096 offset by 2×).
    runs += [RunSpec("QEggRoll", "MountainCarContinuous-v0", s, tag=tag,
                     flags={**EGG_BASE, "pop_size": 2048, "num_epochs": 200,
                            "rank": 4, "sigma_shift": 2})
             for s in range(10)]
    runs += [RunSpec("PPO", "MountainCarContinuous-v0", s, tag=tag,
                     flags={"total_timesteps": 400_000_000})
             for s in range(10)]

    return runs


SUITES = {
    "thesis":      suite_thesis,
    "moreseeds":   suite_moreseeds,
    "pendulum":    suite_pendulum,
    "kinetix":     suite_kinetix,
    "mountaincar": suite_mountaincar,
}


# -------------------------------------------------------------------------------
# Runner
# -------------------------------------------------------------------------------

def hr(char="="):
    return char * 60


def stream(proc, logfile, filter_noise: bool):
    for line in proc.stdout:
        if filter_noise and NOISE_RE.search(line):
            continue
        sys.stdout.write(line)
        sys.stdout.flush()
        if logfile:
            logfile.write(line)
            logfile.flush()


def run_one(spec: RunSpec, args, logfile) -> str:
    if not args.force and os.path.exists(spec.result_path):
        print(f"  [skip] (exists): {spec.result_path}")
        return "skipped"

    cmd = spec.command(args.python, args.max_cpus)
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


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("suite", choices=sorted(SUITES), help="which batch to run")
    p.add_argument("--python", default=os.path.join(HERE, "thesisEnv", "bin", "python"),
                   help="interpreter to launch each run with (default: thesisEnv venv)")
    p.add_argument("--timeout", type=int, default=0,
                   help="per-run timeout in seconds (0 = no limit)")
    p.add_argument("--max_cpus", type=int, default=4,
                   help="pin each run to this many cores (0 = no limit)")
    p.add_argument("--force", action="store_true",
                   help="re-run even if a result JSON already exists")
    p.add_argument("--dry_run", action="store_true",
                   help="print the commands without running them")
    p.add_argument("--only", default="",
                   help="only run specs whose label contains this substring")
    p.add_argument("--no_filter", action="store_true",
                   help="do not filter harmless CUDA-probe lines from output")
    p.add_argument("--logfile", default="",
                   help="tee all output to this file (default: <suite>.log)")
    args = p.parse_args()

    if not os.path.exists(args.python):
        print(f"WARNING: interpreter {args.python} not found; falling back to {sys.executable}")
        args.python = sys.executable

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)

    specs = SUITES[args.suite]()
    if args.only:
        specs = [s for s in specs if args.only in s.label()]

    logpath = args.logfile or os.path.join(HERE, f"{args.suite}.log")
    logfile = None if args.dry_run else open(logpath, "a")

    print(hr())
    print(f" suite={args.suite}   runs={len(specs)}   python={args.python}")
    if not args.dry_run:
        print(f" logging to {logpath}")
    print(hr())

    summary: Dict[str, int] = {}
    batch_t0 = time.time()
    for i, spec in enumerate(specs, 1):
        stamp = time.strftime("%H:%M:%S")
        print(f"\n{hr()}\n [{i}/{len(specs)}] {stamp}  {spec.label()}\n{hr()}")
        status = run_one(spec, args, logfile)
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
