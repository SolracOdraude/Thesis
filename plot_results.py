#!/usr/bin/env python
"""
plot_results.py
===============
Turn the JSON files in results/ into the figures the thesis needs.

Every run is a JSON written by metrics.RunLogger:
    {method, env, seed, config, log: [{env_steps, wall_time, eval_return,
                                       train_return, eval_std, ...}, ...]}
Filename convention (from RunLogger.default_path):
    {method}__{env}__seed{N}[__{tag}].json

Figures produced (into plots/)
------------------------------
  sample_efficiency__{env}.png  eval_return vs env_steps   (mean±std over seeds)
  wallclock__{env}.png          eval_return vs wall_time   (mean±std over seeds)
  ablation_{param}.png          learning curves + final-return-vs-param
                                for tagged runs (pop{N}, rank{N}, ...)

The "main comparison" plots use only runs whose tag is empty (or whose tag you
pass with --tag), so ablation runs don't pollute the per-method curves.

Usage
-----
  python plot_results.py                       # everything it can find
  python plot_results.py --tag test1h          # only the test1h batch
  python plot_results.py --env CartPole-v1
  python plot_results.py --show                # also open windows
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib
# Default to a headless backend; --show flips it to interactive.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# Stable color per method so figures are comparable across envs.
METHOD_COLORS = {"QEggRoll": "tab:blue", "PPO": "tab:orange", "DQN": "tab:green"}
ABLATION_RE = re.compile(r"^(pop|rank|hidden|sigma|noise)(\d+)$")


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Run:
    method: str
    env:    str
    seed:   int
    tag:    str
    config: Dict
    log:    List[Dict]
    path:   str

    def series(self, x_key: str, y_key: str = "eval_return") -> Tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        for row in self.log:
            if row.get(x_key) is None or row.get(y_key) is None:
                continue
            xs.append(row[x_key])
            ys.append(row[y_key])
        return np.asarray(xs, float), np.asarray(ys, float)


def parse_tag(path: str, method: str, env: str, seed: int) -> str:
    """Recover the optional tag from the filename."""
    base = os.path.basename(path)[:-len(".json")]
    prefix = f"{method}__{env.replace('/', '_')}__seed{seed}"
    rest = base[len(prefix):]
    return rest[2:] if rest.startswith("__") else ""


def load_runs(results_dir: str, tag_filter: Optional[str],
              env_filter: Optional[str], methods: Optional[List[str]]) -> List[Run]:
    runs: List[Run] = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        try:
            d = json.load(open(path))
        except Exception as e:  # noqa: BLE001
            print(f"  ! skip unreadable {path}: {e}")
            continue
        if not d.get("log"):
            print(f"  ! skip empty (no log rows, likely crashed mid-run): {path}")
            continue
        method, env, seed = d["method"], d["env"], int(d["seed"])
        tag = parse_tag(path, method, env, seed)
        if env_filter and env != env_filter:
            continue
        if methods and method not in methods:
            continue
        runs.append(Run(method, env, seed, tag, d.get("config", {}), d["log"], path))
    return runs


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation across seeds
# ─────────────────────────────────────────────────────────────────────────────

def aggregate(runs: List[Run], x_key: str,
              n_points: int = 200) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Interpolate each run's (x, eval_return) onto a shared grid over the range
    common to all seeds, then return mean and std across seeds.
    """
    series = [r.series(x_key) for r in runs]
    series = [(x, y) for x, y in series if len(x) >= 2]
    if not series:
        return np.array([]), np.array([]), np.array([]), 0

    lo = max(x.min() for x, _ in series)
    hi = min(x.max() for x, _ in series)
    if hi <= lo:  # no overlap (e.g. wildly different run lengths) → use longest
        x, y = max(series, key=lambda s: len(s[0]))
        return x, y, np.zeros_like(y), 1

    grid = np.linspace(lo, hi, n_points)
    stacked = np.vstack([np.interp(grid, x, y) for x, y in series])
    return grid, stacked.mean(0), stacked.std(0), len(series)


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def _curve(ax, x, mean, std, n, label, color):
    ax.plot(x, mean, label=f"{label} (n={n})", color=color, lw=2)
    if n > 1:
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)


def plot_comparison(runs: List[Run], x_key: str, x_label: str,
                    title_prefix: str, fname_prefix: str, outdir: str, show: bool):
    """One figure per env: a curve per method, shaded ±std over seeds."""
    by_env: Dict[str, Dict[str, List[Run]]] = defaultdict(lambda: defaultdict(list))
    for r in runs:
        by_env[r.env][r.method].append(r)

    for env, by_method in sorted(by_env.items()):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        any_data = False
        for method in sorted(by_method, key=lambda m: list(METHOD_COLORS).index(m)
                             if m in METHOD_COLORS else 99):
            x, mean, std, n = aggregate(by_method[method], x_key)
            if n == 0:
                continue
            any_data = True
            _curve(ax, x, mean, std, n, method,
                   METHOD_COLORS.get(method, None))
        if not any_data:
            plt.close(fig)
            continue
        ax.set_xlabel(x_label)
        ax.set_ylabel("eval return")
        ax.set_title(f"{title_prefix} — {env}")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out = os.path.join(outdir, f"{fname_prefix}__{env.replace('/', '_')}.png")
        fig.savefig(out, dpi=130)
        print(f"  → {out}")
        if not show:
            plt.close(fig)


def plot_ablations(runs: List[Run], outdir: str, show: bool):
    """Group tagged runs by ablation param (pop/rank/...) and plot two panels each."""
    groups: Dict[Tuple[str, str], Dict[int, List[Run]]] = defaultdict(lambda: defaultdict(list))
    for r in runs:
        m = ABLATION_RE.match(r.tag)
        if m:
            param, val = m.group(1), int(m.group(2))
            groups[(param, r.env)][val].append(r)

    for (param, env), by_val in sorted(groups.items()):
        fig, (axc, axf) = plt.subplots(1, 2, figsize=(12, 4.5))
        cmap = plt.cm.viridis(np.linspace(0, 0.9, len(by_val)))
        finals_x, finals_y = [], []
        for color, (val, rs) in zip(cmap, sorted(by_val.items())):
            x, mean, std, n = aggregate(rs, "env_steps")
            if n == 0:
                continue
            axc.plot(x, mean, color=color, lw=2, label=f"{param}={val}")
            if n > 1:
                axc.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)
            # "final" performance = mean of last 10% of points (robust to noise)
            tail = mean[max(1, int(0.9 * len(mean))):]
            finals_x.append(val)
            finals_y.append(tail.mean())
        axc.set(xlabel="env steps", ylabel="eval return",
                title=f"{param} scaling — {env}")
        axc.legend(); axc.grid(alpha=0.3)

        axf.plot(finals_x, finals_y, "o-", color="tab:purple")
        axf.set(xlabel=param, ylabel="final eval return (last 10%)",
                title=f"final return vs {param}")
        axf.set_xscale("log", base=2) if param in ("pop", "noise", "hidden") else None
        axf.grid(alpha=0.3)

        fig.tight_layout()
        out = os.path.join(outdir, f"ablation_{param}__{env.replace('/', '_')}.png")
        fig.savefig(out, dpi=130)
        print(f"  → {out}")
        if not show:
            plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results_dir", default=os.path.join(HERE, "results"))
    p.add_argument("--outdir", default=os.path.join(HERE, "plots"))
    p.add_argument("--tag", default=None,
                   help="only use main-comparison runs with this tag (e.g. test1h). "
                        "Default: untagged runs. Ablation runs are detected separately.")
    p.add_argument("--env", default=None, help="restrict to a single env")
    p.add_argument("--methods", nargs="*", default=None,
                   help="restrict to these methods (QEggRoll PPO DQN)")
    p.add_argument("--show", action="store_true", help="also open interactive windows")
    args = p.parse_args()

    if args.show:
        matplotlib.use("TkAgg", force=True)
    os.makedirs(args.outdir, exist_ok=True)

    all_runs = load_runs(args.results_dir, args.tag, args.env, args.methods)
    if not all_runs:
        print("No usable result JSONs found.")
        return
    print(f"Loaded {len(all_runs)} runs from {args.results_dir}")

    # Main comparison: untagged runs (or those matching --tag), excluding ablations.
    want_tag = args.tag or ""
    main_runs = [r for r in all_runs
                 if r.tag == want_tag and not ABLATION_RE.match(r.tag)]
    if main_runs:
        print("Main comparison plots:")
        plot_comparison(main_runs, "env_steps", "env steps",
                        "Sample efficiency", "sample_efficiency", args.outdir, args.show)
        plot_comparison(main_runs, "wall_time", "wall-clock time (s)",
                        "Wall-clock efficiency", "wallclock", args.outdir, args.show)
    else:
        print(f"(no main-comparison runs with tag='{want_tag}')")

    # Ablations: any tag like pop512 / rank2, regardless of --tag.
    if any(ABLATION_RE.match(r.tag) for r in all_runs):
        print("Ablation plots:")
        plot_ablations(all_runs, args.outdir, args.show)

    if args.show:
        plt.show()
    print(f"Done. Figures in {args.outdir}/")


if __name__ == "__main__":
    main()
