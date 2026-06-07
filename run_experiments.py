#!/usr/bin/env python
"""
run_experiments.py
==================
Python replacement for run_test_1h.sh / run_overnight.sh.

Each experiment is launched as a SEPARATE SUBPROCESS (just like the shell
scripts did). This is deliberate: JAX on CPU accumulates memory, and a
fresh process per run releases everything on exit. It also means a single
run that OOMs or segfaults only kills itself — the runner logs the failure
and moves on to the next run instead of aborting the whole batch.

What this adds over the shell scripts
-------------------------------------
  * resume / skip-existing : a run whose results/*.json already exists is
                             skipped (use --force to re-run). An interrupted
                             batch picks up where it left off.
  * crash isolation        : non-zero exit / timeout is recorded, batch continues.
  * per-run timeout        : --timeout SECONDS (0 = no limit).
  * one editable config    : SUITES below is a plain Python dict of run specs.
  * dry-run                : --dry_run prints the commands without running them.

Usage
-----
  python run_experiments.py test                  # ~45 min validation batch
  python run_experiments.py overnight             # full comparison + ablations
  python run_experiments.py test --dry_run        # show what would run
  python run_experiments.py overnight --only Pendulum   # filter by substring
  python run_experiments.py test --force          # ignore existing results

Add new environments by extending the suite builders below.
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

from metrics import RunLogger  # reuse the exact result-path convention

HERE = os.path.dirname(os.path.abspath(__file__))

# Script that implements each method's CLI.
SCRIPTS = {
    "QEggRoll": "experiments.py",
    "PPO":      "ppo_baseline.py",
    "DQN":      "dqn_baseline.py",
}

# Lines from JAX's CUDA probe that are harmless on a CPU-only box (mirrors the
# grep filter the shell scripts used). Toggle off with --no_filter.
NOISE_RE = re.compile(
    r"xla_cuda|cuInit|CUDA error|discover_pjrt|check_cuda|device_count|warp"
)


# ─────────────────────────────────────────────────────────────────────────────
# Run specification
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RunSpec:
    method: str                       # "QEggRoll" | "PPO" | "DQN"
    env:    str
    seed:   int
    tag:    str = ""                  # --run_tag (also distinguishes result files)
    flags:  Dict[str, str] = field(default_factory=dict)  # extra --key value flags

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
        # Pin to a few cores so XLA does not fan its LLVM compilation across all
        # CPUs at once — that peak is what OOMs this box (swap already full).
        if max_cpus > 0 and shutil.which("taskset"):
            cmd = ["taskset", "-c", f"0-{max_cpus - 1}"] + cmd
        return cmd


def limited_env(max_cpus: int) -> Dict[str, str]:
    """Env that caps thread-pool sizes to match the CPU pin (lowers compile memory)."""
    env = os.environ.copy()
    if max_cpus > 0:
        env["OMP_NUM_THREADS"] = str(max_cpus)
        env["OPENBLAS_NUM_THREADS"] = str(max_cpus)
        env["MKL_NUM_THREADS"] = str(max_cpus)
        # Disable XLA's multithreaded Eigen so compile/runtime stay within the pin.
        flags = env.get("XLA_FLAGS", "")
        if "xla_cpu_multi_thread_eigen" not in flags:
            env["XLA_FLAGS"] = (flags + " --xla_cpu_multi_thread_eigen=false").strip()
    return env


# ─────────────────────────────────────────────────────────────────────────────
# Suites  (CartPole + Pendulum scope for now; extend here for the full 16)
# ─────────────────────────────────────────────────────────────────────────────

def suite_test() -> List[RunSpec]:
    """~45 min validation batch (was run_test_1h.sh): QEggRoll + PPO, 3 seeds."""
    runs: List[RunSpec] = []
    for seed in (0, 1, 2):
        runs.append(RunSpec("QEggRoll", "CartPole-v1", seed, tag="test1h",
                            flags={"pop_size": 512, "num_epochs": 200,
                                   "log_every": 1, "eval_episodes": 5,
                                   "noise_size_exp": 24}))
        runs.append(RunSpec("PPO", "CartPole-v1", seed, tag="test1h",
                            flags={"num_updates": 200, "log_every": 1,
                                   "eval_episodes": 5}))
    return runs


def suite_overnight() -> List[RunSpec]:
    """Full comparison + ablations (was run_overnight.sh)."""
    runs: List[RunSpec] = []

    # CartPole-v1 — main comparison (QEggRoll vs PPO vs DQN)
    for seed in (0, 1, 2):
        runs.append(RunSpec("QEggRoll", "CartPole-v1", seed,
                            flags={"num_epochs": 500, "log_every": 20}))
        runs.append(RunSpec("PPO", "CartPole-v1", seed,
                            flags={"num_updates": 500, "log_every": 20}))
        runs.append(RunSpec("DQN", "CartPole-v1", seed,
                            flags={"total_timesteps": 250000, "log_every": 5000}))

    # Pendulum-v1 — continuous action test (DQN skipped: discrete only)
    for seed in (0, 1, 2):
        runs.append(RunSpec("QEggRoll", "Pendulum-v1", seed,
                            flags={"num_epochs": 500, "log_every": 20}))
        runs.append(RunSpec("PPO", "Pendulum-v1", seed,
                            flags={"num_updates": 500, "log_every": 20}))

    # Ablation 1: population size scaling (CartPole, seed 0)
    for pop in (128, 256, 512, 1024, 2048):
        runs.append(RunSpec("QEggRoll", "CartPole-v1", 0, tag=f"pop{pop}",
                            flags={"num_epochs": 300, "log_every": 20,
                                   "pop_size": pop}))

    # Ablation 2: LoRA rank scaling (CartPole, seed 0)
    for rank in (1, 2, 4):
        runs.append(RunSpec("QEggRoll", "CartPole-v1", 0, tag=f"rank{rank}",
                            flags={"num_epochs": 300, "log_every": 20,
                                   "rank": rank}))

    return runs


SUITES = {
    "test":      suite_test,
    "overnight": suite_overnight,
}


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def hr(char="═"):
    return char * 60


def stream(proc, logfile, filter_noise: bool):
    """Forward subprocess output to console + logfile, optionally filtering CUDA noise."""
    for line in proc.stdout:
        if filter_noise and NOISE_RE.search(line):
            continue
        sys.stdout.write(line)
        sys.stdout.flush()
        if logfile:
            logfile.write(line)
            logfile.flush()


def run_one(spec: RunSpec, args, logfile) -> str:
    """Run a single spec. Returns status string."""
    if not args.force and os.path.exists(spec.result_path):
        print(f"  ⏭  skip (exists): {spec.result_path}")
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
            print(f"  ⏱  TIMEOUT after {args.timeout}s")
            return "timeout"
    except Exception as e:  # noqa: BLE001 — runner must survive any launch error
        print(f"  ✖  launch error: {e}")
        return "error"

    dt = time.time() - t0
    if proc.returncode == 0:
        print(f"  ✓  done in {dt/60:.1f} min")
        return "done"
    print(f"  ✖  FAILED (exit {proc.returncode}) after {dt/60:.1f} min")
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
                   help="pin each run to this many cores so XLA's parallel LLVM "
                        "compilation does not OOM the box (0 = no limit)")
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
    # Non-zero exit if anything actually failed (useful for CI / nohup checks).
    sys.exit(1 if (summary.get("failed", 0) or summary.get("timeout", 0)
                   or summary.get("error", 0)) else 0)


if __name__ == "__main__":
    main()
