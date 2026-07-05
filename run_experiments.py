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
  python run_experiments.py test                  # CartPole smoke-test, 3 seeds
  python run_experiments.py test_all              # all 16 envs, 1 seed, minimal epochs
  python run_experiments.py thesis                # 8-env thesis run, ~12h on RTX 4060 Ti
  python run_experiments.py overnight             # full thesis runs, 3 seeds
  python run_experiments.py test_all --dry_run    # preview commands without running
  python run_experiments.py thesis --only brax    # filter by substring
  python run_experiments.py test_all --force      # ignore existing results

Method / environment support matrix
-------------------------------------
  QEggRoll : all 16 environments (gymnax, brax, craftax, jumanji, kinetix, navix)
  PPO      : gymnax-native only  (CartPole-v1, Pendulum-v1)
             PPO uses gymnax.make() directly; brax/jumanji/navix/kinetix are not
             gymnax-registered so they would crash.
  DQN      : standard gymnasium, discrete actions only (CartPole-v1)
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


# -------------------------------------------------------------------------------
# Run specification
# -------------------------------------------------------------------------------

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


# -------------------------------------------------------------------------------
# Environment lists (single source of truth)
# -------------------------------------------------------------------------------

# All 16 target environments, grouped by suite.
GYMNAX_ENVS   = ["CartPole-v1", "Pendulum-v1"]
BRAX_ENVS     = ["brax/ant", "brax/humanoid", "brax/inverted_double_pendulum"]
CRAFTAX_ENVS  = ["craftax/Craftax-Classic-Symbolic-AutoReset-v1",
                 "craftax/Craftax-Symbolic-AutoReset-v1"]
JUMANJI_ENVS  = ["jumanji/Game2048-v1", "jumanji/Knapsack-v1", "jumanji/Snake-v1"]
KINETIX_ENVS  = ["kinetix/l/hard_pinball",
                 "kinetix/m/h17_thrustcontrol_left",
                 "kinetix/s/h1_thrust_over_ball"]
NAVIX_ENVS    = ["navix/Navix-DoorKey-8x8-v0",
                 "navix/Navix-Dynamic-Obstacles-Random-6x6-v0",
                 "navix/Navix-FourRooms-v0"]

ALL_ENVS = GYMNAX_ENVS + BRAX_ENVS + CRAFTAX_ENVS + JUMANJI_ENVS + KINETIX_ENVS + NAVIX_ENVS

# PPO uses Rejax and supports all 16 environments.
PPO_ENVS = ALL_ENVS

# DQN uses gymnasium and requires discrete actions.
DQN_ENVS = ["CartPole-v1"]


# -------------------------------------------------------------------------------
# Suites
# -------------------------------------------------------------------------------

def suite_test() -> List[RunSpec]:
    """Original CartPole smoke-test: QEggRoll + PPO, 3 seeds, 200 epochs."""
    runs: List[RunSpec] = []
    for seed in (0, 1, 2):
        runs.append(RunSpec("QEggRoll", "CartPole-v1", seed, tag="test1h",
                            flags={"pop_size": 512, "num_epochs": 200,
                                   "log_every": 1, "eval_episodes": 5,
                                   "noise_size_exp": 24}))
        runs.append(RunSpec("PPO", "CartPole-v1", seed, tag="test1h",
                            flags={"total_timesteps": 2_000_000}))
    return runs


def suite_test_all() -> List[RunSpec]:
    """
    Smoke-test every target environment with a single seed and minimal params.
    Goal: surface import errors, API mismatches, shape bugs, OOM issues early.

    Parameters are intentionally tiny — results are not meaningful, only that
    the run completes without error.  Expect ~2-5 min per env on GPU, longer
    on CPU for craftax/brax with their larger observation spaces.

    Tag: 'smoke'
    """
    runs: List[RunSpec] = []
    seed = 0
    tag  = "smoke"

    # Minimal flags for each method.
    # noise_size_exp=16 uses a 64 KB noise table instead of the default 256 MB —
    # safe for a quick sanity check but would cause collisions in a real run.
    egg_flags = {"num_epochs": 5, "log_every": 1, "eval_episodes": 2,
                 "pop_size": 64, "noise_size_exp": 16}
    # total_timesteps=5000 is enough to verify the code runs end-to-end.
    # eval_freq=5000 ensures at least one evaluation checkpoint is produced.
    ppo_flags = {"total_timesteps": 5000, "eval_freq": 5000}
    dqn_flags = {"total_timesteps": 2000, "log_every": 500, "eval_episodes": 2}

    def egg(env): return RunSpec("QEggRoll", env, seed, tag=tag, flags=egg_flags)
    def ppo(env): return RunSpec("PPO",      env, seed, tag=tag, flags=ppo_flags)
    def dqn(env): return RunSpec("DQN",      env, seed, tag=tag, flags=dqn_flags)

    # gymnax — all three methods
    for env in GYMNAX_ENVS:
        runs.append(egg(env))
        runs.append(ppo(env))
    runs.append(dqn("CartPole-v1"))

    # brax — QEggRoll + PPO
    for env in BRAX_ENVS:
        runs.append(egg(env))
        runs.append(ppo(env))

    # craftax — QEggRoll + PPO
    for env in CRAFTAX_ENVS:
        runs.append(egg(env))
        runs.append(ppo(env))

    # jumanji — QEggRoll + PPO
    for env in JUMANJI_ENVS:
        runs.append(egg(env))
        runs.append(ppo(env))

    # kinetix — QEggRoll + PPO
    for env in KINETIX_ENVS:
        runs.append(egg(env))
        runs.append(ppo(env))

    # navix — QEggRoll + PPO
    for env in NAVIX_ENVS:
        runs.append(egg(env))
        runs.append(ppo(env))

    return runs


def suite_overnight() -> List[RunSpec]:
    """
    Full thesis runs: all 16 environments, 3 seeds, paper hyperparameters.

    Methods per environment match the support matrix in the module docstring.
    DQN is included for CartPole-v1 only (discrete gymnasium).
    """
    runs: List[RunSpec] = []

    # ── gymnax: CartPole-v1 — three-way comparison ────────────────────────────
    for seed in (0, 1, 2):
        runs.append(RunSpec("QEggRoll", "CartPole-v1", seed,
                            flags={"num_epochs": 1000, "log_every": 20}))
        runs.append(RunSpec("PPO",      "CartPole-v1", seed))
        runs.append(RunSpec("DQN",      "CartPole-v1", seed,
                            flags={"total_timesteps": 500_000, "log_every": 10_000}))

    # ── gymnax: Pendulum-v1 — continuous, no DQN ─────────────────────────────
    for seed in (0, 1, 2):
        runs.append(RunSpec("QEggRoll", "Pendulum-v1", seed,
                            flags={"num_epochs": 1000, "log_every": 20}))
        runs.append(RunSpec("PPO",      "Pendulum-v1", seed))

    # ── brax — QEggRoll + PPO ─────────────────────────────────────────────────
    for env in BRAX_ENVS:
        for seed in (0, 1, 2):
            runs.append(RunSpec("QEggRoll", env, seed,
                                flags={"num_epochs": 500, "log_every": 20}))
            runs.append(RunSpec("PPO", env, seed))

    # ── craftax — QEggRoll + PPO ──────────────────────────────────────────────
    for env in CRAFTAX_ENVS:
        for seed in (0, 1, 2):
            runs.append(RunSpec("QEggRoll", env, seed,
                                flags={"num_epochs": 500, "log_every": 20}))
            runs.append(RunSpec("PPO", env, seed))

    # ── jumanji — QEggRoll + PPO ──────────────────────────────────────────────
    for env in JUMANJI_ENVS:
        for seed in (0, 1, 2):
            runs.append(RunSpec("QEggRoll", env, seed,
                                flags={"num_epochs": 500, "log_every": 20}))
            runs.append(RunSpec("PPO", env, seed))

    # ── kinetix — QEggRoll + PPO ──────────────────────────────────────────────
    for env in KINETIX_ENVS:
        for seed in (0, 1, 2):
            runs.append(RunSpec("QEggRoll", env, seed,
                                flags={"num_epochs": 500, "log_every": 20}))
            runs.append(RunSpec("PPO", env, seed))

    # ── navix — QEggRoll + PPO ────────────────────────────────────────────────
    for env in NAVIX_ENVS:
        for seed in (0, 1, 2):
            runs.append(RunSpec("QEggRoll", env, seed,
                                flags={"num_epochs": 500, "log_every": 20}))
            runs.append(RunSpec("PPO", env, seed))

    # ── Ablations (CartPole, seed 0) ──────────────────────────────────────────
    for pop in (128, 256, 512, 1024, 2048):
        runs.append(RunSpec("QEggRoll", "CartPole-v1", 0, tag=f"pop{pop}",
                            flags={"num_epochs": 300, "log_every": 20,
                                   "pop_size": pop}))

    for rank in (1, 2, 4):
        runs.append(RunSpec("QEggRoll", "CartPole-v1", 0, tag=f"rank{rank}",
                            flags={"num_epochs": 300, "log_every": 20,
                                   "rank": rank}))

    return runs


def suite_thesis() -> List[RunSpec]:
    """
    Thesis-quality overnight run: 8 environments, balanced seeds and parameters
    for a 12-14h GPU window on an RTX 4060 Ti.

    Design choices vs suite_overnight:
      - pop_size capped at 512 (paper uses 2048-4096 on L40S; 512 is ~2.2x faster
        after accounting for GPU utilization, while still producing clear learning curves)
      - craftax: pop=128, n_par=1, 150 epochs  — obs_dim=8268 is very expensive;
        reduced to fit in budget while still producing a visible signal
      - jumanji/Snake: n_par=2 (paper:8) — halves evaluation cost per epoch
      - brax/ant and kinetix/s: 2 seeds only to save ~2h
      - Excluded: brax/humanoid (pop=4096, n_par=8 — too slow), craftax/Full
        (same obs_dim as Classic but larger network), Game2048/Knapsack
        (less narrative value than Snake), navix DynObs/FourRooms (redundant
        with DoorKey given constraints), kinetix/l and kinetix/m (complex
        physics, less stable convergence)

    Rough time budget (per-seed × seeds):
      CartPole     (3s):  ~6 + 5 + 3 min  ×3  ≈  42 min
      Pendulum     (3s):  ~5 + 5 min       ×3  ≈  30 min
      brax/IDP     (3s):  ~20 + 12 min     ×3  ≈  96 min
      brax/ant     (2s):  ~25 + 15 min     ×2  ≈  80 min
      craftax      (2s):  ~60 + 30 min     ×2  ≈ 180 min
      jumanji/Sn   (3s):  ~20 + 15 min     ×3  ≈ 105 min
      navix/DK     (3s):  ~12 + 10 min     ×3  ≈  66 min
      kinetix/s    (2s):  ~30 + 20 min     ×2  ≈ 100 min
                                              ≈ 699 min ≈ 11.7h

    Seed strategy for reproducibility and future expansion
    -------------------------------------------------------
    Each RunSpec has an explicit integer seed. The seed is passed directly to
    jax.random.key(seed) / torch.manual_seed(seed), so the full training
    trajectory is bit-for-bit reproducible on the same hardware. The result
    file name encodes the seed: e.g. QEggRoll__CartPole-v1__seed0__thesis.json.
    To add more seeds in a later session, simply extend the range() calls
    (0,1,2 -> 0,1,2,3,4) and re-run; existing files are skipped automatically.

    The runner already skips completed results (resume on failure) and logs
    [FAILED] / [TIMEOUT] per run without aborting the batch.
    Recommended command (no per-run timeout — safer for long craftax runs):
      python run_experiments.py thesis --timeout 0 --logfile thesis.log
    Results are written to results/ immediately when each run exits, so
    plot_results.py can be run at any point to inspect completed runs.
    """
    runs: List[RunSpec] = []
    tag = "thesis"

    EGG_BASE = {"noise_size_exp": 24, "log_every": 20}

    def egg(env, seeds, pop, epochs, **extra):
        flags = {**EGG_BASE, "pop_size": pop, "num_epochs": epochs, **extra}
        return [RunSpec("QEggRoll", env, s, tag=tag, flags=flags) for s in range(seeds)]

    def ppo(env, seeds, total_timesteps=None):
        flags = {"total_timesteps": total_timesteps} if total_timesteps else {}
        return [RunSpec("PPO", env, s, tag=tag, flags=flags) for s in range(seeds)]

    # ── gymnax: CartPole ──────────────────────────────────────────────────────
    runs += egg("CartPole-v1",  3, pop=512, epochs=1000)
    runs += ppo("CartPole-v1",  3, total_timesteps=2_000_000)

    # ── gymnax: Pendulum — continuous actions ─────────────────────────────────
    runs += egg("Pendulum-v1",  3, pop=512, epochs=800)
    runs += ppo("Pendulum-v1",  3, total_timesteps=4_000_000)

    # brax excluded: physics simulation is memory-bandwidth bound and OOMs
    # the RTX 4060 Ti (8 GB VRAM) under JAX's vmapped rollout.
    # Brax results will be sourced from the paper's L40S runs in the thesis.

    # ── craftax Classic — OOM ceiling on RTX 4060 Ti is ~epoch 100 at pop=128;
    # use 60 epochs so there is clear headroom, and log_every=10 for denser curves.
    CRAFTAX = "craftax/Craftax-Classic-Symbolic-AutoReset-v1"
    runs += egg(CRAFTAX, 2, pop=128, epochs=60, n_parallel_evaluations=1, log_every=10)
    runs += ppo(CRAFTAX, 2, total_timesteps=2_000_000)

    # ── jumanji Snake — combinatorial sequential task ─────────────────────────
    runs += egg("jumanji/Snake-v1", 3, pop=512, epochs=400, n_parallel_evaluations=2)
    runs += ppo("jumanji/Snake-v1", 3, total_timesteps=4_000_000)

    # ── navix DoorKey — sparse-reward navigation ──────────────────────────────
    runs += egg("navix/Navix-DoorKey-8x8-v0", 3, pop=512, epochs=500)
    runs += ppo("navix/Navix-DoorKey-8x8-v0", 3, total_timesteps=4_000_000)

    # ── kinetix/s — physics manipulation (2 seeds, simplest kinetix) ─────────
    runs += egg("kinetix/s/h1_thrust_over_ball", 2, pop=256, epochs=300)
    runs += ppo("kinetix/s/h1_thrust_over_ball", 2, total_timesteps=4_000_000)

    return runs


def suite_moreseeds() -> List[RunSpec]:
    """
    2-3h follow-up: expand to 10 seeds on the three fastest environments and
    fill the PPO jumanji/Snake gap (all 3 seeds failed overnight with a shape
    error that is now fixed in ppo_baseline.py).

    Timing basis (observed from thesis run):
      CartPole  QEggRoll ~3.5 min/seed,  PPO ~20 sec/seed
      Pendulum  QEggRoll ~5   min/seed,  PPO ~30 sec/seed
      navix     QEggRoll ~8   min/seed,  PPO ~55 sec/seed
      jumanji   PPO      ~unknown, estimated ~5 min/seed (post-fix)

    Budget breakdown (seeds 3-9 = 7 new seeds per env):
      CartPole  QE+PPO  x7  ≈  40 min
      Pendulum  QE+PPO  x7  ≈  40 min
      jumanji   PPO     x3  ≈   6 min  (gap fill, seeds 0-2)
                              ≈  86 min ≈ 1.5h
    """
    runs: List[RunSpec] = []
    tag = "thesis"
    EGG_BASE = {"noise_size_exp": 24, "log_every": 20}

    def egg(env, seeds_range, pop, epochs, **extra):
        flags = {**EGG_BASE, "pop_size": pop, "num_epochs": epochs, **extra}
        return [RunSpec("QEggRoll", env, s, tag=tag, flags=flags) for s in seeds_range]

    def ppo(env, seeds_range, total_timesteps=None):
        flags = {"total_timesteps": total_timesteps} if total_timesteps else {}
        return [RunSpec("PPO", env, s, tag=tag, flags=flags) for s in seeds_range]

    NEW = range(3, 10)  # seeds 3-9

    # ── Expand CartPole to 10 seeds ───────────────────────────────────────────
    runs += egg("CartPole-v1", NEW, pop=512, epochs=1000)
    runs += ppo("CartPole-v1", NEW, total_timesteps=2_000_000)

    # ── Expand Pendulum to 10 seeds ───────────────────────────────────────────
    runs += egg("Pendulum-v1", NEW, pop=512, epochs=800)
    runs += ppo("Pendulum-v1", NEW, total_timesteps=4_000_000)

    # ── Fill jumanji/Snake PPO gap (seeds 0-2 failed with shape error) ───────
    runs += ppo("jumanji/Snake-v1", range(3), total_timesteps=4_000_000)

    return runs


def suite_kinetix() -> List[RunSpec]:
    """
    2h follow-up focused on kinetix/s — the highest-variance result.
    QEggRoll n=2 std=6.68 is unreportable; 2 more seeds bring it to n=4.
    PPO is fast so expand to n=9 for completeness.

    Timing (observed): QEggRoll ~38 min/seed, PPO ~5 min/seed.
      QEggRoll seeds 2,3  (2 × 38) ≈  76 min
      PPO      seeds 2-8  (7 ×  5) ≈  35 min
                                    ≈ 111 min
    """
    runs: List[RunSpec] = []
    tag = "thesis"
    EGG_BASE = {"noise_size_exp": 24, "log_every": 20}
    KENV = "kinetix/s/h1_thrust_over_ball"

    for s in range(2, 4):
        runs.append(RunSpec("QEggRoll", KENV, s, tag=tag,
                            flags={**EGG_BASE, "pop_size": 256, "num_epochs": 300}))
    for s in range(2, 9):
        runs.append(RunSpec("PPO", KENV, s, tag=tag,
                            flags={"total_timesteps": 4_000_000}))
    return runs


def suite_rerun_continuous() -> List[RunSpec]:
    """
    Re-run QEggRoll on continuous environments after removing the accidental
    residual connection from IntMLPContinuous.  PPO results are unaffected.

    Pendulum : seeds 0-9  (params match thesis + moreseeds)
    Kinetix/s: seeds 0-3  (params match thesis + kinetix)

    Run with --force to overwrite the stale result files:
      python run_experiments.py rerun_continuous --force --timeout 0 --logfile rerun.log
    """
    runs: List[RunSpec] = []
    tag      = "thesis"
    EGG_BASE = {"noise_size_exp": 24, "log_every": 20}

    for s in range(10):
        runs.append(RunSpec("QEggRoll", "Pendulum-v1", s, tag=tag,
                            flags={**EGG_BASE, "pop_size": 512, "num_epochs": 800}))

    for s in range(4):
        runs.append(RunSpec("QEggRoll", "kinetix/s/h1_thrust_over_ball", s, tag=tag,
                            flags={**EGG_BASE, "pop_size": 256, "num_epochs": 300}))

    return runs


SUITES = {
    "test":               suite_test,
    "test_all":           suite_test_all,
    "overnight":          suite_overnight,
    "thesis":             suite_thesis,
    "moreseeds":          suite_moreseeds,
    "kinetix":            suite_kinetix,
    "rerun_continuous":   suite_rerun_continuous,
}


# -------------------------------------------------------------------------------
# Runner
# -------------------------------------------------------------------------------

def hr(char="="):
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
    except Exception as e:  # noqa: BLE001 — runner must survive any launch error
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
