"""
plot_results.py
===============
Visualise QEggRoll / PPO / DQN results saved by metrics.RunLogger.

Usage
-----
  python plot_results.py                        # all results/
  python plot_results.py --tag test1h           # only tagged runs
  python plot_results.py --env CartPole-v1      # single environment
  python plot_results.py --out figures/         # save directory (default: figures/)

Output (per environment)
------------------------
  figures/<env>_learning_curves.{png,pdf}   - eval curves vs steps & wall-time
  figures/<env>_summary.{png,pdf}           - final perf, efficiency, QEggRoll diagnostics
  figures/summary_all_envs.{png,pdf}        - cross-env bar chart (if > 1 env)
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


# ── Appearance ────────────────────────────────────────────────────────────────

METHOD_COLOR = {"QEggRoll": "#2196F3", "PPO": "#E53935", "DQN": "#43A047"}
METHOD_LS    = {"QEggRoll": "-",       "PPO": "--",      "DQN": "-."}

def _color(method): return METHOD_COLOR.get(method, "#888888")
def _ls(method):    return METHOD_LS.get(method, ":")

def setup_style():
    plt.rcParams.update({
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "axes.grid":         True,
        "grid.alpha":        0.3,
        "grid.linestyle":    "--",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "font.size":         11,
        "axes.titlesize":    12,
        "axes.labelsize":    11,
        "legend.fontsize":   10,
        "figure.dpi":        120,
    })


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_runs(results_dir: str, tag: str = "", env_filter: str = "") -> Dict:
    """Return {(method, env): [run_dict, ...]}."""
    runs: Dict = defaultdict(list)
    for f in sorted(Path(results_dir).glob("*.json")):
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        if tag and tag not in f.name:
            continue
        if env_filter and env_filter not in data.get("env", ""):
            continue
        runs[(data["method"], data["env"])].append(data)
    return runs


def _interp(runs: List[dict], x_key: str, n: int = 300):
    """Interpolate multiple runs to a common x grid; return (xs, mean, std)."""
    xs_list  = [np.array([e[x_key]        for e in r["log"]]) for r in runs]
    ret_list = [np.array([e["eval_return"] for e in r["log"]]) for r in runs]
    x_lo = max(xs[0]  for xs in xs_list)
    x_hi = min(xs[-1] for xs in xs_list)
    if x_lo >= x_hi:
        x_lo = min(xs[0]  for xs in xs_list)
        x_hi = max(xs[-1] for xs in xs_list)
    grid = np.linspace(x_lo, x_hi, n)
    mat  = np.array([np.interp(grid, xs, ret) for xs, ret in zip(xs_list, ret_list)])
    return grid, mat.mean(0), mat.std(0)


def _steps_to(runs: List[dict], threshold: float) -> List[Optional[float]]:
    """First env_step where eval_return >= threshold; None if never reached."""
    out = []
    for r in runs:
        hit = next((e["env_steps"] for e in r["log"] if e["eval_return"] >= threshold), None)
        out.append(hit)
    return out


def _time_to(runs: List[dict], threshold: float) -> List[Optional[float]]:
    out = []
    for r in runs:
        hit = next((e["wall_time"] for e in r["log"] if e["eval_return"] >= threshold), None)
        out.append(hit)
    return out


def _finals(runs: List[dict]) -> np.ndarray:
    return np.array([r["log"][-1]["eval_return"] for r in runs])


def _solve_threshold(grouped: Dict[str, List[dict]]) -> float:
    """95% of the best observed eval return across all methods."""
    all_ret = [e["eval_return"] for runs in grouped.values()
               for r in runs for e in r["log"]]
    return max(all_ret) * 0.95


# ── Individual axis functions ─────────────────────────────────────────────────

def ax_learning_curve(ax, grouped, x_key="env_steps", log_x=False, title=""):
    """Mean +/- std eval curves; faint individual seeds behind."""
    for method, runs in grouped.items():
        c = _color(method)
        for r in runs:
            ax.plot([e[x_key] for e in r["log"]],
                    [e["eval_return"] for e in r["log"]],
                    color=c, alpha=0.18, linewidth=0.9, linestyle=_ls(method))
        xs, mean, std = _interp(runs, x_key)
        ax.plot(xs, mean, color=c, linewidth=2.2, linestyle=_ls(method), label=method)
        ax.fill_between(xs, mean - std, mean + std, color=c, alpha=0.15)

    if log_x:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{v/1e6:.1f}M" if v >= 1e6 else
                                           f"{v/1e3:.0f}K" if v >= 1e3 else str(int(v))))
    ax.set_xlabel("Env Steps" if x_key == "env_steps" else "Wall-clock Time (s)")
    ax.set_ylabel("Eval Return")
    ax.set_title(title)
    ax.legend(loc="lower right")


def ax_learning_curve_normalised(ax, grouped, title=""):
    """
    x-axis is training progress in [0, 100%], independent of how many env_steps
    or wall-clock seconds each method consumed. Both QEggRoll and PPO start at 0
    and end at 100 so their learning trajectories are directly comparable even
    when their absolute step counts differ by orders of magnitude.
    """
    GRID = np.linspace(0, 100, 300)
    for method, runs in grouped.items():
        c = _color(method)
        interp_runs = []
        for r in runs:
            n  = len(r["log"])
            xs = np.linspace(0, 100, n)
            ys = np.array([e["eval_return"] for e in r["log"]])
            ax.plot(xs, ys, color=c, alpha=0.18, linewidth=0.9, linestyle=_ls(method))
            interp_runs.append(np.interp(GRID, xs, ys))
        mat  = np.array(interp_runs)
        mean = mat.mean(0)
        std  = mat.std(0)
        ax.plot(GRID, mean, color=c, linewidth=2.2, linestyle=_ls(method), label=method)
        ax.fill_between(GRID, mean - std, mean + std, color=c, alpha=0.15)

    ax.set_xlabel("Training Progress (%)")
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_ylabel("Eval Return")
    ax.set_title(title)
    ax.legend(loc="lower right")


def ax_final_performance(ax, grouped):
    """Box + scatter of final eval returns per method."""
    methods = sorted(grouped)
    data    = [_finals(grouped[m]) for m in methods]
    colors  = [_color(m) for m in methods]

    bp = ax.boxplot(data, tick_labels=methods, patch_artist=True, widths=0.45,
                    medianprops=dict(color="black", linewidth=2),
                    flierprops=dict(marker="x", color="gray"))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.65)
    for i, (d, c) in enumerate(zip(data, colors), 1):
        ax.scatter([i] * len(d), d, color=c, zorder=5, s=55,
                   edgecolors="black", linewidths=0.8)

    ax.set_ylabel("Final Eval Return")
    ax.set_title("Final Performance (last checkpoint)")


def ax_bar_efficiency(ax, grouped, threshold, metric="steps"):
    """Bar chart: mean steps or seconds to first hit threshold (+/- std)."""
    fn      = _steps_to if metric == "steps" else _time_to
    methods = sorted(grouped)
    means, stds, n_solved = [], [], []
    for m in methods:
        vals = [v for v in fn(grouped[m], threshold) if v is not None]
        n_solved.append(f"{len(vals)}/{len(grouped[m])}")
        means.append(np.mean(vals) if vals else np.nan)
        stds.append(np.std(vals) if len(vals) > 1 else 0)

    x = np.arange(len(methods))
    ax.bar(x, means, yerr=stds, color=[_color(m) for m in methods],
           alpha=0.8, capsize=6, edgecolor="black", linewidth=0.8, width=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\n({s} solved)" for m, s in zip(methods, n_solved)])

    if metric == "steps":
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{v/1e6:.1f}M" if v >= 1e6 else
                                           f"{v/1e3:.0f}K"))
        ax.set_ylabel("Env Steps")
        ax.set_title(f"Steps to Reach Return >= {threshold:.0f}")
    else:
        ax.set_ylabel("Wall-clock Time (s)")
        ax.set_title(f"Time to Reach Return >= {threshold:.0f}")


def ax_fitness_signal(ax, qeggroll_runs):
    """QEggRoll: fraction of antithetic pairs with a clear fitness winner."""
    if not qeggroll_runs:
        ax.text(0.5, 0.5, "No QEggRoll data", ha="center", va="center",
                transform=ax.transAxes, color="gray")
        return

    c = _color("QEggRoll")
    all_fit, ref_steps = [], None
    for r in qeggroll_runs:
        steps = [e["env_steps"]      for e in r["log"] if "fitness_signal" in e]
        fit   = [e["fitness_signal"] for e in r["log"] if "fitness_signal" in e]
        ax.plot(steps, fit, color=c, alpha=0.25, linewidth=1)
        all_fit.append(fit)
        if ref_steps is None:
            ref_steps = steps

    if all_fit and ref_steps:
        min_len  = min(len(f) for f in all_fit)
        mean_fit = np.mean([f[:min_len] for f in all_fit], axis=0)
        ax.plot(ref_steps[:min_len], mean_fit, color=c, linewidth=2, label="mean")

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="maximum")
    ax.set_ylim(0, 1.12)
    ax.set_xlabel("Env Steps")
    ax.set_ylabel("|Fitness Signal|")
    ax.set_title("QEggRoll Fitness Signal\n(fraction of antithetic pairs with clear winner)")
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v/1e6:.1f}M" if v >= 1e6 else f"{v/1e3:.0f}K"))
    ax.legend()


def ax_eval_std(ax, qeggroll_runs):
    """QEggRoll: eval std across evaluation episodes (policy consistency)."""
    if not qeggroll_runs:
        ax.set_visible(False)
        return

    c = _color("QEggRoll")
    all_std, ref_steps = [], None
    for r in qeggroll_runs:
        steps = [e["env_steps"] for e in r["log"] if "eval_std" in e]
        stds  = [e["eval_std"]  for e in r["log"] if "eval_std" in e]
        ax.plot(steps, stds, color=c, alpha=0.25, linewidth=1)
        all_std.append(stds)
        if ref_steps is None:
            ref_steps = steps

    if all_std and ref_steps:
        min_len  = min(len(s) for s in all_std)
        mean_std = np.mean([s[:min_len] for s in all_std], axis=0)
        ax.plot(ref_steps[:min_len], mean_std, color=c, linewidth=2)

    ax.set_xlabel("Env Steps")
    ax.set_ylabel("Eval Return Std Dev")
    ax.set_title("QEggRoll Policy Consistency\n(std dev across eval episodes per checkpoint)")
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v/1e6:.1f}M" if v >= 1e6 else f"{v/1e3:.0f}K"))


def ax_train_eval_gap(ax, grouped):
    """QEggRoll eval-return minus train-return (PPO train_return is not comparable)."""
    qruns = grouped.get("QEggRoll", [])
    if not qruns:
        ax.set_visible(False)
        return

    c = _color("QEggRoll")
    all_gaps, ref_steps = [], None
    for r in qruns:
        steps = [e["env_steps"]   for e in r["log"]]
        gap   = [e["eval_return"] - e["train_return"] for e in r["log"]]
        ax.plot(steps, gap, color=c, alpha=0.25, linewidth=1)
        all_gaps.append(gap)
        if ref_steps is None:
            ref_steps = steps

    if all_gaps and ref_steps:
        min_len  = min(len(g) for g in all_gaps)
        mean_gap = np.mean([g[:min_len] for g in all_gaps], axis=0)
        ax.plot(ref_steps[:min_len], mean_gap, color=c, linewidth=2)

    ax.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Env Steps")
    ax.set_ylabel("Eval - Train Return")
    ax.set_title("QEggRoll Eval vs Train Gap\n(positive = greedy eval beats noisy training)")
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v/1e6:.1f}M" if v >= 1e6 else f"{v/1e3:.0f}K"))


# ── Figure builders ───────────────────────────────────────────────────────────

def _save(fig, path_no_ext):
    fig.savefig(path_no_ext + ".png", bbox_inches="tight", dpi=150)
    fig.savefig(path_no_ext + ".pdf", bbox_inches="tight")
    print(f"  -> {path_no_ext}.png + .pdf")
    plt.close(fig)


def figure_learning_curves(env, grouped, out_dir):
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(21, 5))
    fig.suptitle(f"{env} — Learning Curves", fontsize=14, fontweight="bold")

    ax_learning_curve(axes[0], grouped, x_key="env_steps", log_x=True,
                      title="Sample Efficiency  (log x-axis)")
    ax_learning_curve(axes[1], grouped, x_key="wall_time",
                      title="Wall-clock Time")
    ax_learning_curve_normalised(axes[2], grouped,
                      title="Training Progress  (methods aligned)")

    plt.tight_layout()
    _save(fig, os.path.join(out_dir, f"{env.replace('/', '_')}_learning_curves"))


def figure_summary(env, grouped, out_dir):
    setup_style()
    threshold = _solve_threshold(grouped)
    qruns     = grouped.get("QEggRoll", [])

    fig = plt.figure(figsize=(16, 11))
    fig.suptitle(f"{env} - Performance Summary", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.52, wspace=0.35)

    ax_final_performance(fig.add_subplot(gs[0, 0]), grouped)
    ax_bar_efficiency(fig.add_subplot(gs[0, 1]), grouped, threshold, metric="steps")
    ax_bar_efficiency(fig.add_subplot(gs[1, 0]), grouped, threshold, metric="time")
    ax_train_eval_gap(fig.add_subplot(gs[1, 1]), grouped)
    ax_fitness_signal(fig.add_subplot(gs[2, 0]), qruns)
    ax_eval_std(fig.add_subplot(gs[2, 1]), qruns)

    _save(fig, os.path.join(out_dir, f"{env.replace('/', '_')}_summary"))


def figure_cross_env(all_env_grouped, out_dir):
    """Final performance bar chart across environments (requires > 1 env)."""
    envs = sorted(all_env_grouped)
    if len(envs) < 2:
        return

    setup_style()
    methods   = sorted({m for g in all_env_grouped.values() for m in g})
    n_methods = len(methods)
    width = 0.7 / n_methods
    x = np.arange(len(envs))

    fig, ax = plt.subplots(figsize=(max(8, 3 * len(envs)), 5))
    for i, method in enumerate(methods):
        means = [_finals(all_env_grouped[e].get(method, [])).mean()
                 if all_env_grouped[e].get(method) else np.nan for e in envs]
        stds  = [_finals(all_env_grouped[e].get(method, [])).std()
                 if all_env_grouped[e].get(method) else 0 for e in envs]
        offset = (i - (n_methods - 1) / 2) * width
        ax.bar(x + offset, means, width * 0.9, yerr=stds, label=method,
               color=_color(method), alpha=0.85, capsize=5,
               edgecolor="black", linewidth=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels([e.split("/")[-1] for e in envs], rotation=20, ha="right")
    ax.set_ylabel("Final Eval Return (mean +/- std across seeds)")
    ax.set_title("Final Performance Across Environments")
    ax.legend()
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "summary_all_envs"))


# ── Console summary ───────────────────────────────────────────────────────────

def print_table(all_env_grouped):
    hdr = f"{'ENV':<38} {'METHOD':<12} {'MEAN':>8} {'STD':>7} {'MIN':>7} {'MAX':>7}  SEEDS"
    sep = "=" * len(hdr)
    print(f"\n{sep}\n{hdr}\n{sep}")
    for env in sorted(all_env_grouped):
        for method in sorted(all_env_grouped[env]):
            f = _finals(all_env_grouped[env][method])
            print(f"{env:<38} {method:<12} {f.mean():8.2f} {f.std():7.2f} "
                  f"{f.min():7.2f} {f.max():7.2f}  {len(f)}")
    print(sep + "\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results_dir", default="results")
    p.add_argument("--out",         default="figures", help="output directory")
    p.add_argument("--tag",         default="",  help="filter files containing this tag")
    p.add_argument("--env",         default="",  help="filter by env name substring")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    raw = load_runs(args.results_dir, tag=args.tag, env_filter=args.env)
    if not raw:
        print(f"No results found in '{args.results_dir}' "
              f"(tag='{args.tag}', env='{args.env}')")
        return

    all_env: Dict[str, Dict[str, List]] = defaultdict(lambda: defaultdict(list))
    for (method, env), run_list in raw.items():
        all_env[env][method].extend(run_list)

    print_table(all_env)

    for env, grouped in sorted(all_env.items()):
        print(f"Plotting {env} ...")
        figure_learning_curves(env, grouped, args.out)
        figure_summary(env, grouped, args.out)

    figure_cross_env(all_env, args.out)
    print(f"\nAll figures saved to '{args.out}/'")


if __name__ == "__main__":
    main()
