#!/usr/bin/env python3

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


METRICS = [
    ("group_goodput", "Completed Groups/sec vs Window Ratio", "Groups / sec", "goodput_compare.png"),
    (
        "tokens_throughput_mean",
        "Completion Tokens/sec vs Window Ratio",
        "Completion tokens / sec",
        "tokens_throughput_compare.png",
    ),
    (
        "mean_step_cost_sec_proxy",
        "Mean Step Cost Proxy vs Window Ratio",
        "rollout_time / decode_batches",
        "c_hat_compare.png",
    ),
    (
        "decode_batches_per_rollout_proxy",
        "Decode Batch Count Proxy vs Window Ratio",
        "Decode batches / rollout",
        "t_hat_compare.png",
    ),
    ("off_policy_ratio_mean", "Off-policy Ratio vs Window Ratio", "Off-policy ratio", "off_policy_compare.png"),
    (
        "carryover_fraction",
        "Carry-over Completion Fraction vs Window Ratio",
        "Carry-over completion fraction",
        "carryover_compare.png",
    ),
]


def parse_summary_arg(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("Each --summary must be in the form LABEL=/path/to/summary.csv")
    label, path = text.split("=", 1)
    if not label:
        raise argparse.ArgumentTypeError("Summary label cannot be empty")
    return label, Path(path)


def load_rows(summary_path: Path) -> list[dict]:
    with summary_path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def safe_float(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def plot_metric(labeled_rows: list[tuple[str, list[dict]]], metric_key: str, title: str, ylabel: str, output_path: Path) -> None:
    plt.figure(figsize=(9, 5))
    any_points = False

    for label, rows in labeled_rows:
        points = []
        for row in rows:
            x = safe_float(row.get("window_ratio"))
            y = safe_float(row.get(metric_key))
            if x is None or y is None:
                continue
            points.append((x, y))
        if not points:
            continue
        any_points = True
        points.sort(key=lambda item: item[0])
        plt.plot([x for x, _ in points], [y for _, y in points], marker="o", linewidth=2, label=label)

    if not any_points:
        plt.close()
        return

    plt.title(title)
    plt.xlabel("Window ratio N / B")
    plt.ylabel(ylabel)
    plt.grid(True, axis="y", linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


def write_peak_table(labeled_rows: list[tuple[str, list[dict]]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "best_window_ratio", "best_group_goodput"])
        for label, rows in labeled_rows:
            best_row = None
            best_goodput = None
            for row in rows:
                goodput = safe_float(row.get("group_goodput"))
                if goodput is None:
                    continue
                if best_goodput is None or goodput > best_goodput:
                    best_goodput = goodput
                    best_row = row
            writer.writerow(
                [
                    label,
                    best_row.get("window_ratio") if best_row is not None else "",
                    f"{best_goodput:.12f}" if best_goodput is not None else "",
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay multiple APRIL window-summary CSVs.")
    parser.add_argument(
        "--summary",
        action="append",
        required=True,
        type=parse_summary_arg,
        help="LABEL=/path/to/summary.csv. Pass multiple times.",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    labeled_rows = []
    for label, summary_path in args.summary:
        labeled_rows.append((label, load_rows(summary_path)))

    output_dir = Path(args.output_dir)
    for metric_key, title, ylabel, filename in METRICS:
        plot_metric(labeled_rows, metric_key, title, ylabel, output_dir / filename)
    write_peak_table(labeled_rows, output_dir / "best_window_by_summary.csv")


if __name__ == "__main__":
    main()
