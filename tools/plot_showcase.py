#!/usr/bin/env python3
"""
Showcase plot: prov=10 collapse story.

Three-panel figure showing:
  (A) Eval reward curve  – 3 selected lines
  (B) Truncated ratio    – collapse signal
  (C) Training raw reward

Lines:
  base_agg   – prov=10, no PPO clip, no M2PO  (unstable/collapses)
  base_clip  – prov=10, PPO clip eps=0.2       (performance drops permanently)
  m2po_best  – prov=10, no PPO clip + best M2PO topk (stable/maintained)

Usage:
  python tools/plot_showcase.py \
    --sweep-dir results/m2po-swp2-JOBID \
    --base-agg  /path/to/base-agg-run \
    --base-clip /path/to/base-clip-run \
    --m2po-runs "m2po_0.05:/path" "m2po_0.1:/path" ... \
    --max-rollout 49 \
    --out-dir results/m2po-swp2-JOBID
"""
import argparse
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# data loading (shared with plot_m2po.py logic)
# ---------------------------------------------------------------------------

def load_jsonl(path):
    rows = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    except FileNotFoundError:
        pass
    return rows


def get_eval(run_dir, label, max_rollout=None):
    rows = load_jsonl(pathlib.Path(run_dir) / "analysis" / "eval_metrics.jsonl")
    records = []
    for row in rows:
        rid = row["rollout_id"]
        if max_rollout is not None and rid > max_rollout:
            continue
        for key, val in row["metrics"].items():
            if key.startswith("eval/") and not key.endswith("-truncated_ratio"):
                records.append({"run": label, "rollout_id": rid, "eval_value": val})
    return pd.DataFrame(records).sort_values("rollout_id") if records else pd.DataFrame()


def get_rollout(run_dir, label, max_rollout=None):
    rows = load_jsonl(pathlib.Path(run_dir) / "analysis" / "rollout_metrics.jsonl")
    records = []
    for row in rows:
        rid = row["rollout_id"]
        if max_rollout is not None and rid > max_rollout:
            continue
        m = row["metrics"]
        records.append({
            "run": label,
            "rollout_id": rid,
            "raw_reward": m.get("rollout/raw_reward"),
            "truncated": m.get("rollout/truncated"),
            "response_length": m.get("rollout/response_lengths"),
        })
    return pd.DataFrame(records).sort_values("rollout_id") if records else pd.DataFrame()


# ---------------------------------------------------------------------------
# auto-select best M2PO run
# ---------------------------------------------------------------------------

def pick_best_m2po(m2po_specs, max_rollout):
    """
    Select the M2PO run with the highest mean eval reward AND lowest
    truncated ratio in the second half of training (rollouts > max_rollout/2).
    Returns (label, run_dir).
    """
    scores = []
    for label, run_dir in m2po_specs:
        ev = get_eval(run_dir, label, max_rollout)
        ro = get_rollout(run_dir, label, max_rollout)
        if ev.empty or ro.empty:
            continue
        half = max_rollout // 2
        late_eval = ev[ev["rollout_id"] > half]["eval_value"].mean()
        late_trunc = ro[ro["rollout_id"] > half]["truncated"].mean()
        # score: high eval is good, high truncation is bad
        score = late_eval - 0.5 * late_trunc
        scores.append((score, label, run_dir))
        print(f"  [{label}] late_eval={late_eval:.4f}, late_trunc={late_trunc:.3f}, score={score:.4f}")
    if not scores:
        return None, None
    scores.sort(reverse=True)
    _, best_label, best_dir = scores[0]
    print(f"  → best M2PO: {best_label}")
    return best_label, best_dir


# ---------------------------------------------------------------------------
# plotting helpers
# ---------------------------------------------------------------------------

COLORS = {
    "base_agg":  "#e74c3c",   # red   – unstable baseline
    "base_clip": "#e67e22",   # orange – clip hurts
    "m2po_best": "#2ecc71",   # green  – M2PO works
}
STYLES = {
    "base_agg":  dict(linestyle="--", linewidth=2.0, marker="o", markersize=4),
    "base_clip": dict(linestyle=":",  linewidth=2.0, marker="s", markersize=4),
    "m2po_best": dict(linestyle="-",  linewidth=2.5, marker="^", markersize=5),
}
LABELS = {
    "base_agg":  "Baseline (prov=10, no clip)",
    "base_clip": "Baseline (prov=10, PPO clip ε=0.2)",
    "m2po_best": None,  # filled in at runtime with topk value
}


def smooth(series, window=3):
    return series.rolling(window=window, min_periods=1, center=True).mean()


def add_panel(ax, df_dict, col, ylabel, title, ylim=None, legend=True):
    for key, df in df_dict.items():
        if df.empty:
            continue
        x = df["rollout_id"].values
        y = df[col].values
        lbl = LABELS.get(key, key)
        ax.plot(x, smooth(pd.Series(y)), color=COLORS[key],
                label=lbl, **STYLES[key])
    ax.set_xlabel("Rollout Step", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.3)
    if ylim:
        ax.set_ylim(ylim)
    if legend:
        ax.legend(fontsize=9)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir",  required=True)
    ap.add_argument("--base-agg",   required=True)
    ap.add_argument("--base-clip",  required=True)
    ap.add_argument("--m2po-runs",  nargs="+", required=True,
                    help="label:run_dir pairs for all M2PO sweep runs")
    ap.add_argument("--best-m2po",  default=None,
                    help="Force a specific label as best (skip auto-pick)")
    ap.add_argument("--max-rollout", type=int, default=49)
    ap.add_argument("--out-dir",    required=True)
    ap.add_argument("--lr-label",   default="5e-6",
                    help="LR string shown in plot titles (default: 5e-6)")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    max_r = args.max_rollout

    # parse m2po run specs
    m2po_specs = []
    for s in args.m2po_runs:
        label, path = s.split(":", 1)
        m2po_specs.append((label, path))

    # ---- pick best M2PO ----
    if args.best_m2po:
        best_label = args.best_m2po
        best_dir = dict(m2po_specs)[best_label]
    else:
        print("Auto-selecting best M2PO run...")
        best_label, best_dir = pick_best_m2po(m2po_specs, max_r)

    if best_dir is None:
        print("No M2PO run has data yet. Skipping showcase plot.")
        return

    # Update LABELS with topk value
    topk_str = best_label.replace("m2po_", "").replace("_", "")
    LABELS["m2po_best"] = f"M2PO (prov=10, topk={topk_str})"

    # ---- load data ----
    eval_dfs = {
        "base_agg":  get_eval(args.base_agg,  "base_agg",  max_r),
        "base_clip": get_eval(args.base_clip, "base_clip", max_r),
        "m2po_best": get_eval(best_dir,       "m2po_best", max_r),
    }
    roll_dfs = {
        "base_agg":  get_rollout(args.base_agg,  "base_agg",  max_r),
        "base_clip": get_rollout(args.base_clip, "base_clip", max_r),
        "m2po_best": get_rollout(best_dir,       "m2po_best", max_r),
    }

    # ---- 3-panel showcase figure ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"M2PO Stabilization: prov=10, LR={args.lr_label}  |  Qwen2.5-3B, Math-Level2",
        fontsize=13, fontweight="bold"
    )

    add_panel(axes[0], {k: eval_dfs[k] for k in eval_dfs if not eval_dfs[k].empty},
              "eval_value", "Eval Reward", "(A) Eval Reward Curve", ylim=(0, 0.75))

    add_panel(axes[1], {k: roll_dfs[k] for k in roll_dfs if not roll_dfs[k].empty},
              "truncated", "Truncated Ratio", "(B) Response Truncation\n(→1 = collapse)",
              ylim=(0, 1.05), legend=False)

    add_panel(axes[2], {k: roll_dfs[k] for k in roll_dfs if not roll_dfs[k].empty},
              "raw_reward", "Train Reward", "(C) Training Reward",
              ylim=(0, 1.05), legend=False)

    plt.tight_layout()
    out_path = out_dir / "showcase_3panel.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[showcase] saved: {out_path}")

    # ---- eval-only 2-line variant (cleaner for paper) ----
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    for key in ["base_agg", "base_clip", "m2po_best"]:
        df = eval_dfs[key]
        if df.empty:
            continue
        x, y = df["rollout_id"].values, df["eval_value"].values
        ax2.plot(x, smooth(pd.Series(y)), color=COLORS[key],
                 label=LABELS[key], **STYLES[key])
    ax2.set_xlabel("Rollout Step", fontsize=12)
    ax2.set_ylabel("Eval Reward", fontsize=12)
    ax2.set_title(f"M2PO vs Baselines (prov=10, LR={args.lr_label})", fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 0.75)
    ax2.legend(fontsize=10)
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(10))
    out_path2 = out_dir / "showcase_eval_only.png"
    plt.tight_layout()
    plt.savefig(out_path2, dpi=200)
    plt.close()
    print(f"[showcase] saved: {out_path2}")

    # ---- all M2PO sweep eval curves ----
    fig3, ax3 = plt.subplots(figsize=(9, 5))
    cmap = plt.cm.viridis(np.linspace(0.15, 0.9, len(m2po_specs)))
    for i, (lbl, rdir) in enumerate(m2po_specs):
        df = get_eval(rdir, lbl, max_r)
        if df.empty:
            continue
        x, y = df["rollout_id"].values, df["eval_value"].values
        ax3.plot(x, smooth(pd.Series(y)), color=cmap[i], marker="o",
                 markersize=3, linewidth=1.8, label=lbl)
    # overlay baselines
    for key, lbl, rdir in [("base_agg", "base_agg(no-clip)", args.base_agg),
                            ("base_clip", "base_clip(ε=0.2)", args.base_clip)]:
        df = get_eval(rdir, lbl, max_r)
        if not df.empty:
            x, y = df["rollout_id"].values, df["eval_value"].values
            ax3.plot(x, smooth(pd.Series(y)), color=COLORS[key],
                     linestyle="--", linewidth=2.2, label=lbl, **STYLES[key])
    ax3.set_xlabel("Rollout Step", fontsize=11)
    ax3.set_ylabel("Eval Reward", fontsize=11)
    ax3.set_title("M2PO TopK Sweep vs Baselines", fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8, ncol=2)
    ax3.xaxis.set_major_locator(ticker.MultipleLocator(10))
    out_path3 = out_dir / "showcase_full_sweep.png"
    plt.tight_layout()
    plt.savefig(out_path3, dpi=200)
    plt.close()
    print(f"[showcase] saved: {out_path3}")

    print(f"\nAll showcase plots in: {out_dir}/")
    print(f"Best M2PO: {best_label} ({best_dir})")


if __name__ == "__main__":
    main()
