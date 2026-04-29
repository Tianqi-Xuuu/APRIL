#!/usr/bin/env python3
"""
Plot M2PO vs baseline metrics, matching the style of
results/qwen2.5-3b-level2-shortprompt-provision-compare.

Usage:
  # one-shot, 2-way
  python tools/plot_m2po.py --runs baseline:<dir> m2po_01:<dir> --out-dir <dir>

  # watch mode
  python tools/plot_m2po.py --runs baseline:<dir> m2po_01:<dir> --out-dir <dir> --watch 30
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
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


def extract_eval_df(run_dir: Path, label: str) -> pd.DataFrame:
    rows = load_jsonl(Path(run_dir) / "analysis" / "eval_metrics.jsonl")
    records = []
    for row in rows:
        for key, value in row["metrics"].items():
            if key.startswith("eval/") and not key.endswith("-truncated_ratio"):
                records.append({
                    "run": label,
                    "rollout_id": row["rollout_id"],
                    "eval_metric": key,
                    "eval_value": value,
                })
    return pd.DataFrame(records).sort_values("rollout_id") if records else pd.DataFrame()


def extract_train_metrics_from_log(run_dir: Path) -> dict[int, dict]:
    """Parse job_output.log for per-step train metrics (step N → rollout N)."""
    import re
    log_path = Path(run_dir) / "job_output.log"
    step_re = re.compile(r"step (\d+): (\{.*\})")
    metrics_by_step: dict[int, dict] = {}
    try:
        with open(log_path) as f:
            for line in f:
                m = step_re.search(line)
                if m:
                    step_id = int(m.group(1))
                    try:
                        d = json.loads(m.group(2).replace("'", '"'))
                        metrics_by_step[step_id] = d
                    except Exception:
                        pass
    except FileNotFoundError:
        pass
    return metrics_by_step


def extract_rollout_df(run_dir: Path, label: str) -> pd.DataFrame:
    rows = load_jsonl(Path(run_dir) / "analysis" / "rollout_metrics.jsonl")
    train_metrics = extract_train_metrics_from_log(run_dir)
    records = []
    for row in rows:
        m = row["metrics"]
        rid = row["rollout_id"]
        tm = train_metrics.get(rid, {})
        records.append({
            "run": label,
            "rollout_id": rid,
            "raw_reward": m.get("rollout/raw_reward"),
            "response_length_mean": m.get("rollout/response_lengths"),
            "truncated_ratio": m.get("rollout/truncated"),
            "m2po_masked_ratio": tm.get("train/m2po_masked_ratio"),
            "m2po_eligible_tokens": tm.get("train/m2po_eligible_tokens"),
            "m2po_masked_tokens": tm.get("train/m2po_masked_tokens"),
            "grad_norm": tm.get("train/grad_norm"),
        })
    return pd.DataFrame(records).sort_values("rollout_id") if records else pd.DataFrame()


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------

def plot_eval_curve(eval_df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    for run_name, group in eval_df.groupby("run"):
        plt.plot(group["rollout_id"], group["eval_value"], marker="o", label=run_name)
    plt.xlabel("Rollout ID")
    plt.ylabel("Eval Reward")
    plt.title("Eval Curve: baseline vs M2PO variants")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_eval_curve_smoothed(eval_df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    for run_name, group in eval_df.groupby("run"):
        group = group.sort_values("rollout_id").copy()
        plt.plot(group["rollout_id"], group["eval_value"],
                 marker="o", alpha=0.35, linewidth=1.5, label=f"{run_name} raw")
        smooth = group["eval_value"].rolling(window=3, min_periods=1, center=True).mean()
        plt.plot(group["rollout_id"], smooth, linewidth=2.5, label=f"{run_name} smooth")
    plt.xlabel("Rollout ID")
    plt.ylabel("Eval Reward")
    plt.title("Smoothed Eval Curve: baseline vs M2PO variants")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_rollout_length(rollout_df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    for run_name, group in rollout_df.groupby("run"):
        plt.plot(group["rollout_id"], group["response_length_mean"], marker="o", label=run_name)
    plt.xlabel("Rollout ID")
    plt.ylabel("Mean Response Length")
    plt.title("Training Rollout Mean Length: baseline vs M2PO variants")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_raw_reward(rollout_df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    for run_name, group in rollout_df.groupby("run"):
        plt.plot(group["rollout_id"], group["raw_reward"], marker="o", label=run_name)
    plt.xlabel("Rollout ID")
    plt.ylabel("Raw Reward")
    plt.title("Training Raw Reward: baseline vs M2PO variants")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_m2po_mask_stats(rollout_df: pd.DataFrame, out_path: Path) -> None:
    m2po = rollout_df[~rollout_df["run"].str.startswith("baseline")].copy()
    if m2po.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for run_name, group in m2po.groupby("run"):
        axes[0].plot(group["rollout_id"], group["m2po_masked_ratio"], marker="o", label=run_name)
        axes[1].plot(group["rollout_id"], group["m2po_eligible_tokens"], marker="s", label=f"{run_name} eligible")

    axes[0].set_title("M2PO Masked Ratio")
    axes[0].set_xlabel("Rollout ID")
    axes[0].set_ylabel("Masked Ratio")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("M2PO Eligible Tokens")
    axes[1].set_xlabel("Rollout ID")
    axes[1].set_ylabel("Token Count")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("M2PO Mask Statistics", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def run_once(run_specs: list[tuple[str, str]], out_dir: Path) -> bool:
    rollout_parts, eval_parts = [], []
    for label, run_dir in run_specs:
        r = extract_rollout_df(run_dir, label)
        e = extract_eval_df(run_dir, label)
        if not r.empty:
            rollout_parts.append(r)
        if not e.empty:
            eval_parts.append(e)

    if not rollout_parts:
        print("[plot_m2po] no data yet, skipping")
        return False

    out_dir.mkdir(parents=True, exist_ok=True)
    rollout_df = pd.concat(rollout_parts, ignore_index=True)
    rollout_df.to_csv(out_dir / "rollout_compare.csv", index=False)
    plot_rollout_length(rollout_df, out_dir / "rollout_mean_length_compare.png")
    plot_raw_reward(rollout_df,     out_dir / "raw_reward_compare.png")
    plot_m2po_mask_stats(rollout_df, out_dir / "m2po_mask_stats.png")

    if eval_parts:
        eval_df = pd.concat(eval_parts, ignore_index=True)
        eval_df.to_csv(out_dir / "eval_curve_compare.csv", index=False)
        plot_eval_curve(eval_df,          out_dir / "eval_curve_compare.png")
        plot_eval_curve_smoothed(eval_df,  out_dir / "eval_curve_compare_smoothed.png")

    counts = {label: len(r) for label, r in zip(
        [s[0] for s in run_specs], rollout_parts)}
    print(f"[plot_m2po] saved to {out_dir}  steps={counts}")
    return True


def parse_run_spec(s: str) -> tuple[str, str]:
    if ":" not in s:
        raise argparse.ArgumentTypeError(f"--runs entry must be label:path, got: {s!r}")
    label, path = s.split(":", 1)
    return label, path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", type=parse_run_spec, required=True,
                    help="label:run_dir pairs, e.g. baseline:/path m2po_01:/path")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--watch", type=int, default=0,
                    help="re-plot every N seconds (0=one-shot)")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)

    if args.watch:
        print(f"[plot_m2po] watch mode, interval={args.watch}s, output={out_dir}")
        while True:
            run_once(args.runs, out_dir)
            time.sleep(args.watch)
    else:
        run_once(args.runs, out_dir)


if __name__ == "__main__":
    main()
