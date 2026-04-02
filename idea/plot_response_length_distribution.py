#!/usr/bin/env python3

import argparse
import csv
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_response_lengths(debug_dir: Path) -> list[float]:
    response_lengths = []
    for pickle_path in sorted(debug_dir.glob("rollout_*.pkl")):
        with pickle_path.open("rb") as f:
            samples = pickle.load(f)
        for sample in samples:
            if not isinstance(sample, dict):
                continue
            value = sample.get("response_length")
            if value is None:
                continue
            response_lengths.append(float(value))
    return response_lengths


def reduce_grouped(values: list[float], group_size: int, reduce: str) -> list[float]:
    if group_size <= 1:
        return list(values)
    usable = (len(values) // group_size) * group_size
    if usable == 0:
        return []
    data = np.array(values[:usable], dtype=np.float64).reshape(-1, group_size)
    if reduce == "mean":
        return data.mean(axis=1).tolist()
    if reduce == "max":
        return data.max(axis=1).tolist()
    if reduce == "min":
        return data.min(axis=1).tolist()
    raise ValueError(f"Unsupported reduce mode: {reduce}")


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(values, pct))


def build_bins(max_value: float) -> int:
    if max_value <= 256:
        return 32
    if max_value <= 2048:
        return 48
    return 64


def plot_distribution(response_lengths: list[float], title: str, output_path: Path, xlabel: str) -> None:
    data = np.array(response_lengths, dtype=np.float64)
    max_len = float(np.max(data))
    bins = build_bins(max_len)

    fig, (ax_hist, ax_cdf) = plt.subplots(1, 2, figsize=(12, 4.8))

    ax_hist.hist(data, bins=bins, color="#3B82F6", alpha=0.85, edgecolor="white")
    ax_hist.set_title("Response Length Histogram")
    ax_hist.set_xlabel(xlabel)
    ax_hist.set_ylabel("Count")
    ax_hist.grid(True, axis="y", linestyle="--", alpha=0.35)

    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    ax_cdf.plot(sorted_data, cdf, color="#EF4444", linewidth=2)
    ax_cdf.set_title("Response Length CDF")
    ax_cdf.set_xlabel(xlabel)
    ax_cdf.set_ylabel("CDF")
    ax_cdf.set_ylim(0, 1.01)
    ax_cdf.grid(True, linestyle="--", alpha=0.35)

    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_summary(response_lengths: list[float], output_path: Path) -> None:
    if not response_lengths:
        raise ValueError("No response lengths found")

    values = {
        "count": len(response_lengths),
        "mean": float(np.mean(response_lengths)),
        "std": float(np.std(response_lengths)),
        "min": float(np.min(response_lengths)),
        "p50": percentile(response_lengths, 50),
        "p90": percentile(response_lengths, 90),
        "p95": percentile(response_lengths, 95),
        "p99": percentile(response_lengths, 99),
        "max": float(np.max(response_lengths)),
        "clip_ratio_at_max_len": None,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in values.items():
            writer.writerow([key, value])


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot response length distribution from APRIL debug_rollout pickles.")
    parser.add_argument("--run-dir", required=True, help="Run directory containing debug_rollout/")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--title", default=None)
    parser.add_argument("--group-size", type=int, default=1)
    parser.add_argument("--group-reduce", choices=["mean", "max", "min"], default="mean")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    debug_dir = run_dir / "debug_rollout"
    if not debug_dir.is_dir():
        raise FileNotFoundError(f"debug_rollout directory not found: {debug_dir}")

    response_lengths = load_response_lengths(debug_dir)
    if not response_lengths:
        raise ValueError(f"No response lengths found under {debug_dir}")

    plot_values = reduce_grouped(response_lengths, args.group_size, args.group_reduce)
    if not plot_values:
        raise ValueError("No grouped response lengths available after reduction")

    output_dir = Path(args.output_dir)
    title = args.title or f"Response Length Distribution: {run_dir.name}"
    if args.group_size > 1:
        xlabel = f"Group-{args.group_size} {args.group_reduce} response length (tokens)"
        stem = f"group{args.group_size}_{args.group_reduce}_response_length"
    else:
        xlabel = "Response length (tokens)"
        stem = "response_length"
    plot_distribution(plot_values, title, output_dir / f"{stem}_distribution.png", xlabel)
    write_summary(plot_values, output_dir / f"{stem}_summary.csv")


if __name__ == "__main__":
    main()
