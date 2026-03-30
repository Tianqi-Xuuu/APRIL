#!/usr/bin/env python3

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


NONPROV_RUN = Path("/root/APRIL/runs/qwen2.5-3b-train-math-level2-shortprompt-bs8-n8-r30")
PROV_RUN = Path("/root/APRIL/runs/qwen2.5-3b-train-math-level2-shortprompt-prov20-bs8-n8-r30")
PROV3_RUN = Path("/root/APRIL/runs/qwen2.5-3b-train-math-level2-shortprompt-prov30-bs8-n8-r30")
PROV4_RUN = Path("/root/APRIL/runs/qwen2.5-3b-train-math-level2-shortprompt-prov40-bs8-n8-r30")
OUT_DIR = Path("/root/APRIL/results/qwen2.5-3b-level2-shortprompt-provision-compare")


def load_metrics(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def extract_eval_df(run_dir: Path, label: str) -> pd.DataFrame:
    rows = load_metrics(run_dir / "analysis" / "eval_metrics.jsonl")
    records = []
    for row in rows:
        metric_items = row["metrics"].items()
        for key, value in metric_items:
            if key.startswith("eval/") and not key.endswith("-truncated_ratio"):
                records.append(
                    {
                        "run": label,
                        "rollout_id": row["rollout_id"],
                        "eval_metric": key,
                        "eval_value": value,
                    }
                )
    return pd.DataFrame(records).sort_values("rollout_id")


def extract_rollout_df(run_dir: Path, label: str) -> pd.DataFrame:
    rows = load_metrics(run_dir / "analysis" / "rollout_metrics.jsonl")
    records = []
    for row in rows:
        metrics = row["metrics"]
        records.append(
            {
                "run": label,
                "rollout_id": row["rollout_id"],
                "raw_reward": metrics.get("rollout/raw_reward"),
                "response_length_mean": metrics.get("rollout/response_lengths"),
                "truncated_ratio": metrics.get("rollout/truncated"),
            }
        )
    return pd.DataFrame(records).sort_values("rollout_id")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    runs = [
        (NONPROV_RUN, "non_provision"),
        (PROV_RUN, "provision_2p0x"),
        (PROV3_RUN, "provision_3p0x"),
        (PROV4_RUN, "provision_4p0x"),
    ]

    eval_parts = []
    rollout_parts = []
    for run_dir, label in runs:
        if (run_dir / "analysis" / "eval_metrics.jsonl").exists():
            eval_parts.append(extract_eval_df(run_dir, label))
        if (run_dir / "analysis" / "rollout_metrics.jsonl").exists():
            rollout_parts.append(extract_rollout_df(run_dir, label))

    eval_df = pd.concat(eval_parts, ignore_index=True)
    rollout_df = pd.concat(rollout_parts, ignore_index=True)

    eval_df.to_csv(OUT_DIR / "eval_curve_compare.csv", index=False)
    rollout_df.to_csv(OUT_DIR / "rollout_length_compare.csv", index=False)

    plt.figure(figsize=(8, 5))
    for run_name, group in eval_df.groupby("run"):
        plt.plot(group["rollout_id"], group["eval_value"], marker="o", label=run_name)
    plt.xlabel("Rollout ID")
    plt.ylabel("Eval Reward")
    plt.title("Eval Curve: non-provision vs 2.0x/3.0x/4.0x provision")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "eval_curve_compare.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    for run_name, group in eval_df.groupby("run"):
        group = group.sort_values("rollout_id").copy()
        plt.plot(group["rollout_id"], group["eval_value"], marker="o", alpha=0.35, linewidth=1.5, label=f"{run_name} raw")
        smooth = group["eval_value"].rolling(window=3, min_periods=1, center=True).mean()
        plt.plot(group["rollout_id"], smooth, linewidth=2.5, label=f"{run_name} smooth")
    plt.xlabel("Rollout ID")
    plt.ylabel("Eval Reward")
    plt.title("Smoothed Eval Curve: non-provision vs provision")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "eval_curve_compare_smoothed.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    for run_name, group in rollout_df.groupby("run"):
        plt.plot(group["rollout_id"], group["response_length_mean"], marker="o", label=run_name)
    plt.xlabel("Rollout ID")
    plt.ylabel("Mean Response Length")
    plt.title("Training Rollout Mean Length: non-provision vs 2.0x/3.0x/4.0x provision")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "rollout_mean_length_compare.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
