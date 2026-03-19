#!/usr/bin/env python3

import argparse
import ast
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def parse_metrics(log_path: Path) -> list[dict]:
    metrics = []
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "partial_rollout " not in line or ": {" not in line:
                continue
            payload = line.split("partial_rollout ", 1)[1].split(": ", 1)[1]
            metrics.append(ast.literal_eval(payload))
    return metrics


def label_from_run_name(run_name: str) -> str:
    if "non-provision" in run_name:
        return "non_provision"
    marker = "provision-"
    idx = run_name.index(marker) + len(marker)
    suffix = run_name[idx:].split("-bs", 1)[0]
    return suffix.replace("p", ".").rstrip(".")


def provision_order(label: str) -> float:
    if label == "non_provision":
        return 1.0
    return float(label.rstrip("x"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root-base", required=True)
    parser.add_argument("--output-plot", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root_base)
    series = []
    for run_dir in sorted(p for p in run_root.iterdir() if p.is_dir()):
        log_path = run_dir / "job_output.log"
        if not log_path.exists():
            continue
        metrics = parse_metrics(log_path)
        y = [m["partial_rollout/tokens_throughput"] for m in metrics if "partial_rollout/tokens_throughput" in m]
        if not y:
            continue
        series.append({"run_name": run_dir.name, "label": label_from_run_name(run_dir.name), "y": y})

    if not series:
        raise SystemExit(f"No rollout metrics found under {run_root}")

    series.sort(key=lambda item: provision_order(item["label"]))
    min_len = min(len(item["y"]) for item in series)
    x = list(range(min_len))

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rollout_idx"] + [item["label"] for item in series])
        for idx in x:
            writer.writerow([idx] + [item["y"][idx] for item in series])

    plt.figure(figsize=(9, 5))
    for item in series:
        plt.plot(x, item["y"][:min_len], marker="o", linewidth=2, label=item["label"])

    plt.title("Qwen3-1.7B Tokens/sec per Rollout by Provision Size")
    plt.xlabel(f"Rollout Index (aligned to shortest run = {min_len})")
    plt.ylabel("Tokens/sec")
    plt.grid(True, axis="y", linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    output_plot = Path(args.output_plot)
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_plot, dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
