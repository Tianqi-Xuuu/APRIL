#!/usr/bin/env python3

import argparse
import ast
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def metric_value(metric: dict, key: str):
    if key in metric:
        return metric[key]
    prefixed = f"partial_rollout/{key}"
    return metric.get(prefixed)


def parse_rollout_metrics(log_path: Path) -> list[dict]:
    metrics = []
    if not log_path.exists():
        return metrics
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            marker = "partial_rollout "
            if marker in line and ": {" in line:
                payload = line.split(marker, 1)[1].split(": ", 1)[1]
                metrics.append(ast.literal_eval(payload))
    return metrics


def provision_order(label: str) -> float:
    if label == "non_provision":
        return 1.0
    return float(label.rstrip("x"))


def label_from_run_name(run_name: str) -> str:
    if "non-provision" in run_name:
        return "non_provision"
    marker = "provision-"
    idx = run_name.index(marker) + len(marker)
    suffix = run_name[idx:].split("-bs", 1)[0]
    return suffix.replace("p", ".").rstrip(".")


def summarize_run(run_dir: Path) -> dict | None:
    log_path = run_dir / "job_output.log"
    metrics = parse_rollout_metrics(log_path)
    if not metrics:
        return None

    tokens_per_sec = [v for m in metrics if (v := metric_value(m, "tokens_throughput")) is not None]
    rollout_time = [v for m in metrics if (v := metric_value(m, "rollout_time")) is not None]
    total_tokens = [v for m in metrics if (v := metric_value(m, "total_tokens")) is not None]
    off_policy_ratio = [v for m in metrics if (v := metric_value(m, "off_policy_ratio")) is not None]
    partial_samples = [v for m in metrics if (v := metric_value(m, "partial_samples")) is not None]

    if not tokens_per_sec:
        return None

    return {
        "run_name": run_dir.name,
        "provision_label": label_from_run_name(run_dir.name),
        "num_rollouts": len(tokens_per_sec),
        "tokens_per_sec_mean": sum(tokens_per_sec) / len(tokens_per_sec),
        "tokens_per_sec_min": min(tokens_per_sec),
        "tokens_per_sec_max": max(tokens_per_sec),
        "rollout_time_mean": sum(rollout_time) / len(rollout_time) if rollout_time else None,
        "total_tokens_mean": sum(total_tokens) / len(total_tokens) if total_tokens else None,
        "off_policy_ratio_mean": sum(off_policy_ratio) / len(off_policy_ratio) if off_policy_ratio else None,
        "partial_samples_mean": sum(partial_samples) / len(partial_samples) if partial_samples else None,
    }


def write_csv(rows: list[dict], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_name",
        "provision_label",
        "num_rollouts",
        "tokens_per_sec_mean",
        "tokens_per_sec_min",
        "tokens_per_sec_max",
        "rollout_time_mean",
        "total_tokens_mean",
        "off_policy_ratio_mean",
        "partial_samples_mean",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_rows(rows: list[dict], output_plot: Path) -> None:
    ordered = sorted(rows, key=lambda row: provision_order(row["provision_label"]))
    x_labels = [row["provision_label"] for row in ordered]
    y = [row["tokens_per_sec_mean"] for row in ordered]

    plt.figure(figsize=(9, 5))
    plt.plot(x_labels, y, marker="o", linewidth=2, color="#1f77b4")
    plt.title("Qwen3-1.7B Tokens/sec vs Provision Size")
    plt.xlabel("Provision Size")
    plt.ylabel("Mean tokens/sec")
    plt.grid(True, axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_plot, dpi=160)
    plt.close()


def plot_metric(rows: list[dict], key: str, title: str, ylabel: str, output_plot: Path, color: str) -> None:
    ordered = sorted(rows, key=lambda row: provision_order(row["provision_label"]))
    x_labels = [row["provision_label"] for row in ordered]
    y = [row[key] for row in ordered]

    plt.figure(figsize=(9, 5))
    plt.plot(x_labels, y, marker="o", linewidth=2, color=color)
    plt.title(title)
    plt.xlabel("Provision Size")
    plt.ylabel(ylabel)
    plt.grid(True, axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_plot, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root-base", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-plot", required=True)
    args = parser.parse_args()

    run_root_base = Path(args.run_root_base)
    rows = []
    for run_dir in sorted(p for p in run_root_base.iterdir() if p.is_dir()):
        row = summarize_run(run_dir)
        if row is not None:
            rows.append(row)

    if not rows:
        raise SystemExit(f"No rollout metrics found under {run_root_base}")

    write_csv(rows, Path(args.output_csv))
    plot_rows(rows, Path(args.output_plot))
    output_dir = Path(args.output_plot).parent
    plot_metric(
        rows,
        "off_policy_ratio_mean",
        "Qwen3-1.7B Off-Policy Ratio vs Provision Size",
        "Mean off-policy ratio",
        output_dir / "off_policy_ratio_vs_provision.png",
        "#d62728",
    )
    plot_metric(
        rows,
        "partial_samples_mean",
        "Qwen3-1.7B Partial Samples vs Provision Size",
        "Mean partial samples",
        output_dir / "partial_samples_vs_provision.png",
        "#2ca02c",
    )

    for row in sorted(rows, key=lambda item: provision_order(item["provision_label"])):
        print(
            {
                "provision_label": row["provision_label"],
                "tokens_per_sec_mean": round(row["tokens_per_sec_mean"], 4),
                "rollout_time_mean": round(row["rollout_time_mean"], 4) if row["rollout_time_mean"] is not None else None,
                "num_rollouts": row["num_rollouts"],
            }
        )


if __name__ == "__main__":
    main()
