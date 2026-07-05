#!/usr/bin/env python
"""
plot_ablation.py
================
Visualise the 81-run QEggRoll hyperparameter ablation.

Supports CartPole-v1 (discrete) and Pendulum-v1 (continuous).

Produces 3 learning-curve figures (one per rank value) + 1 summary figure:
  figures/ablation/{env}/ablation_rank{r}.{png,pdf}
    3x3 grid: sigma_shift columns x pop_size rows.
    k = 1, 2, 4 overlaid as separate lines per subplot.
  figures/ablation/{env}/ablation_summary.{png,pdf}

Usage
-----
  python plot_ablation.py                    # CartPole-v1 (default)
  python plot_ablation.py --env Pendulum-v1
"""

import argparse
import json
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import scienceplots  # noqa: F401 — registers the styles

# ── Grid axes (must match run_ablation.py) ────────────────────────────────────

POP_SIZES    = [128, 256, 512]
RANKS        = [1, 2, 4]
SIGMA_SHIFTS = [1, 2, 4]
N_EVALS      = [1, 2, 4]

# Light → dark sequential blue keyed on k (same hue as QEggRoll in plot_results.py)
K_COLOR  = {1: "#90CAF9", 2: "#2196F3", 4: "#0D47A1"}
K_LS     = {1: "-",       2: "-",       4: "-"}
K_MARKER = {1: "o",       2: "s",       4: "^"}   # circle / square / triangle

ENV_CONFIG = {
    "CartPole-v1": {
        "min_return":  0.0,
        "max_return":  500.0,
        "tag_re":      re.compile(r"abl_p(\d+)_r(\d+)_s(\d+)_k(\d+)\.json$"),
    },
    "Pendulum-v1": {
        "min_return": -1800.0,
        "max_return": -900.0,
        "tag_re":      re.compile(r"abl_pendulum_p(\d+)_r(\d+)_s(\d+)_k(\d+)\.json$"),
    },
}


# ── Style (mirrors plot_results.py) ───────────────────────────────────────────

def setup_style():
    # Apply scienceplots first (sets closed box, grid, serif fonts, tight spacing),
    # then override only the settings that don't fit our use case.
    plt.style.use(['science', 'notebook', 'grid'])
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "figure.dpi":       120,
        "font.size":        11,
        "axes.titlesize":   12,
        "axes.labelsize":   11,
        "legend.fontsize":  10,
    })


def _save(fig, path_no_ext):
    os.makedirs(os.path.dirname(os.path.abspath(path_no_ext)), exist_ok=True)
    fig.savefig(f"{path_no_ext}.png", bbox_inches="tight", dpi=150)
    fig.savefig(f"{path_no_ext}.pdf", bbox_inches="tight")
    print(f"  -> {path_no_ext}.png + .pdf")
    plt.close(fig)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_ablation(results_dir: str, env: str):
    """Return dict keyed by (pop, rank, sigma_shift, k) → run dict."""
    tag_re = ENV_CONFIG[env]["tag_re"]
    data = {}
    for f in sorted(Path(results_dir).glob(f"QEggRoll__{env}__seed0__abl_*.json")):
        m = tag_re.search(f.name)
        if not m:
            continue
        pop, rank, sig, k = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
        try:
            run = json.loads(f.read_text())
        except Exception as e:
            print(f"  warning: could not load {f.name}: {e}")
            continue
        data[(pop, rank, sig, k)] = run
    return data


def _extract(run):
    """Return (epoch array, eval_return array)."""
    log = run["log"]
    le  = run["config"].get("log_every", 20)
    xs  = np.array([i * le for i in range(len(log))])
    ys  = np.array([e["eval_return"] for e in log])
    return xs, ys


# ── Per-rank learning-curve figure ────────────────────────────────────────────

def figure_rank(rank: int, data: dict, env: str, out_dir: str):
    """
    3x3 subplot grid (sigma_shift columns x pop_size rows).
    k = 1, 2, 4 overlaid as lines in each subplot.
    """
    y_lo  = ENV_CONFIG[env]["min_return"]
    y_hi  = ENV_CONFIG[env]["max_return"]
    y_pad = (y_hi - y_lo) * 0.05

    fig, axes = plt.subplots(
        3, 3,
        figsize=(15, 10),
        sharey=True,
        #gridspec_kw={"hspace": 0.50, "wspace": 0.15},
    )
    fig.suptitle(
        f"QEggRoll Ablation — {env}   (rank = {rank})",
        fontsize=13, fontweight="bold",
    )

    for ri, pop in enumerate(POP_SIZES):
        for ci, sig in enumerate(SIGMA_SHIFTS):
            ax = axes[ri, ci]

            if ri == 0:
                sigma_real = 2.0 ** -(4 + sig)
                ax.set_title(f"σ_shift = {sig}  (σ ≈ {sigma_real:.4f})", fontsize=10)

            if ci == 0:
                ax.set_ylabel(f"pop = {pop}\nEval Return", fontsize=10)

            # x-axis label + ticks on every row
            ax.set_xlabel("Epoch", fontsize=10)
            ax.tick_params(axis="x", labelbottom=True)

            plotted_any = False
            for k in N_EVALS:
                run = data.get((pop, rank, sig, k))
                if run is None:
                    continue
                xs, ys = _extract(run)
                ax.plot(xs, ys, color=K_COLOR[k], linewidth=2.2,
                        linestyle=K_LS[k], marker=K_MARKER[k],
                        markevery=max(1, len(xs) // 8), markersize=4,
                        label=f"k = {k}", zorder=3)
                plotted_any = True

            if not plotted_any:
                ax.text(0.5, 0.5, "missing", ha="center", va="center",
                        transform=ax.transAxes, color="red", fontsize=10)

            ax.set_ylim(y_lo - y_pad, y_hi + y_pad)
            ax.legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    _save(fig, os.path.join(out_dir, f"ablation_rank{rank}"))


# ── Summary: marginal effect of each parameter ────────────────────────────────

def figure_summary(data: dict, env: str, out_dir: str):
    """4-panel bar chart: mean peak return as each hyperparameter varies."""
    y_lo  = ENV_CONFIG[env]["min_return"]
    y_hi  = ENV_CONFIG[env]["max_return"]
    span  = y_hi - y_lo
    peaks = {key: float(_extract(run)[1].max()) for key, run in data.items()}

    param_specs = [
        ("Population size",            "pop",  POP_SIZES),
        ("LoRA rank (r)",              "rank", RANKS),
        ("σ_shift  (↑ = smaller σ)",  "sig",  SIGMA_SHIFTS),
        ("Evals / member (k)",         "k",    N_EVALS),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=True,
                             gridspec_kw={"wspace": 0.10})
    fig.suptitle(
        f"Marginal hyperparameter effect — {env}\n"
        "(mean peak eval return across 27 configurations per bar; error bars = std)",
        fontsize=12, fontweight="bold",
    )

    for ax, (xlabel, pname, values) in zip(axes, param_specs):
        means, stds = [], []
        for v in values:
            if pname == "pop":
                subset = [peaks[(v, r, s, k_)]
                          for r in RANKS for s in SIGMA_SHIFTS for k_ in N_EVALS
                          if (v, r, s, k_) in peaks]
            elif pname == "rank":
                subset = [peaks[(p, v, s, k_)]
                          for p in POP_SIZES for s in SIGMA_SHIFTS for k_ in N_EVALS
                          if (p, v, s, k_) in peaks]
            elif pname == "sig":
                subset = [peaks[(p, r, v, k_)]
                          for p in POP_SIZES for r in RANKS for k_ in N_EVALS
                          if (p, r, v, k_) in peaks]
            else:
                subset = [peaks[(p, r, s, v)]
                          for p in POP_SIZES for r in RANKS for s in SIGMA_SHIFTS
                          if (p, r, s, v) in peaks]
            means.append(np.mean(subset) if subset else 0.0)
            stds.append(np.std(subset)   if subset else 0.0)

        means = np.array(means)
        stds  = np.array(stds)
        xs    = np.arange(len(values))

        ax.bar(xs, means, color="#2196F3", alpha=0.80, zorder=3,
               yerr=stds, error_kw={"ecolor": "#444", "capsize": 6,
                                     "linewidth": 1.2, "zorder": 4},
               edgecolor="black", linewidth=0.7, width=0.5)

        label_offset = stds.max() * 0.15 + span * 0.01
        for x, m in zip(xs, means):
            ax.text(x, m + label_offset, f"{m:.0f}",
                    ha="center", va="bottom", fontsize=10)

        ax.set_xticks(xs)
        ax.set_xticklabels([str(v) for v in values], fontsize=10)
        ax.set_xlabel(xlabel, fontsize=11, labelpad=6)
        ax.set_ylim(y_lo - span * 0.05, y_hi + span * 0.18)
        ax.tick_params(axis="y", labelsize=10)

    axes[0].set_ylabel("Mean peak eval return", fontsize=11)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "ablation_summary"))


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", default="CartPole-v1", choices=list(ENV_CONFIG),
                    help="environment to plot (default: CartPole-v1)")
    ap.add_argument("--results", default="results",
                    help="directory with result JSON files (default: results/)")
    ap.add_argument("--out", default="",
                    help="output directory (default: figures/ablation/<env>/)")
    args = ap.parse_args()

    out_dir = args.out or os.path.join("figures", "ablation", args.env.replace("/", "_"))

    setup_style()
    data = load_ablation(args.results, args.env)
    print(f"Loaded {len(data)} / 81 ablation runs for {args.env} from '{args.results}/'.")

    if not data:
        print("No results found. Run:  python run_ablation.py ablation_pendulum")
        return

    os.makedirs(out_dir, exist_ok=True)

    for i, rank in enumerate(RANKS, 1):
        print(f"[{i}/{len(RANKS)}] rank={rank}")
        figure_rank(rank, data, args.env, out_dir)

    print("[summary]")
    figure_summary(data, args.env, out_dir)

    print(f"\nDone — {len(RANKS) + 1} figures written to {out_dir}/")


if __name__ == "__main__":
    main()
