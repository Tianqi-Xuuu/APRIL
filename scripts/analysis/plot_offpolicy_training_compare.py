#!/usr/bin/env python3

import argparse
import ast
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def parse_prefixed_metrics(log_path: Path, prefix: str) -> list[tuple[int, dict]]:
    rows = []
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            marker = f"{prefix} "
            if marker not in line or ": {" not in line:
                continue
            rest = line.split(marker, 1)[1]
            rollout_id = int(rest.split(":", 1)[0].strip())
            payload = rest.split(": ", 1)[1]
            rows.append((rollout_id, ast.literal_eval(payload)))
    return rows


def label_from_run_name(run_name: str) -> str:
    if "non-provision" in run_name:
        return "non_provision"
    marker = "provision-"
    idx = run_name.index(marker) + len(marker)
    suffix = run_name[idx:].split("-bs", 1)[0]
    return suffix.replace("p", ".").rstrip(".")


def get_value(metric: dict, key: str):
    if key in metric:
        return metric[key]
    for prefix in ("rollout/", "partial_rollout/", "eval/"):
        prefixed = f"{prefix}{key}"
        if prefixed in metric:
            return metric[prefixed]
    return None


def plot_metric(series: dict[str, list[tuple[int, float]]], title: str, ylabel: str, output_plot: Path) -> None:
    plt.figure(figsize=(9, 5))
    for label, rows in series.items():
        plt.plot([x for x, _ in rows], [y for _, y in rows], marker="o", linewidth=2, label=label)
    plt.title(title)
    plt.xlabel("Rollout")
    plt.ylabel(ylabel)
    plt.grid(True, axis="y", linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_plot, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root-base", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root_base)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    reward_series = {}
    off_policy_series = {}
    eval_series = {}

    csv_path = outdir / "training_compare_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "metric_type", "rollout_id", "value"])

        for run_dir in sorted(p for p in run_root.iterdir() if p.is_dir()):
            log_path = run_dir / "job_output.log"
            if not log_path.exists():
                continue
            label = label_from_run_name(run_dir.name)

            rollout_rows = []
            for rollout_id, metric in parse_prefixed_metrics(log_path, "rollout"):
                value = get_value(metric, "raw_reward")
                if value is not None:
                    rollout_rows.append((rollout_id, value))
                    writer.writerow([label, "raw_reward", rollout_id, value])
            if rollout_rows:
                reward_series[label] = rollout_rows

            partial_rows = []
            for rollout_id, metric in parse_prefixed_metrics(log_path, "partial_rollout"):
                value = get_value(metric, "off_policy_ratio")
                if value is not None:
                    partial_rows.append((rollout_id, value))
                    writer.writerow([label, "off_policy_ratio", rollout_id, value])
            if partial_rows:
                off_policy_series[label] = partial_rows

            eval_rows = []
            for rollout_id, metric in parse_prefixed_metrics(log_path, "eval"):
                # only one eval dataset in this experiment
                value = None
                for key, val in metric.items():
                    if key.startswith("eval/") and not key.endswith("truncated_ratio"):
                        value = val
                        break
                if value is not None:
                    eval_rows.append((rollout_id, value))
                    writer.writerow([label, "eval_reward", rollout_id, value])
            if eval_rows:
                eval_series[label] = eval_rows

    if reward_series:
        plot_metric(reward_series, "Train Raw Reward vs Rollout", "Mean raw reward", outdir / "train_reward_vs_rollout.png")
    if off_policy_series:
        plot_metric(
            off_policy_series,
            "Off-Policy Ratio vs Rollout",
            "Off-policy ratio",
            outdir / "off_policy_ratio_vs_rollout.png",
        )
    if eval_series:
        plot_metric(eval_series, "Eval Reward vs Rollout", "Mean eval reward", outdir / "eval_reward_vs_rollout.png")


if __name__ == "__main__":
    main()
