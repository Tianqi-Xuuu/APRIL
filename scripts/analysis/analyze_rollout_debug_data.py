import argparse
import ast
import glob
import json
import os
import pickle
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np


def parse_log_metrics(log_path):
    perf_metrics = []
    rollout_metrics = []
    if log_path is None or not os.path.exists(log_path):
        return perf_metrics, rollout_metrics

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.startswith("partial_rollout ") and ": {" in line:
                payload = line.split(": ", 1)[1]
                rollout_metrics.append(ast.literal_eval(payload))
            elif line.startswith("perf ") and ": {" in line:
                payload = line.split(": ", 1)[1]
                perf_metrics.append(ast.literal_eval(payload))
    return perf_metrics, rollout_metrics


def flatten_samples(debug_dir):
    samples = []
    for path in sorted(glob.glob(os.path.join(debug_dir, "*.pkl"))):
        with open(path, "rb") as f:
            rollout_samples = pickle.load(f)
        for sample in rollout_samples:
            metadata = sample.get("metadata", {}) or {}
            samples.append(
                {
                    "sample_index": sample.get("index"),
                    "response_length": sample.get("response_length"),
                    "completion_tokens": sample.get("completion_tokens"),
                    "status": sample.get("status"),
                    "is_padding": metadata.get("is_padding", False),
                    "source_index": metadata.get("source_index"),
                    "reward": sample.get("reward"),
                }
            )
    return samples


def summarize(samples, perf_metrics, rollout_metrics):
    non_padding = [s for s in samples if not s["is_padding"]]
    response_lengths = [s["response_length"] for s in non_padding]
    completion_tokens = [s["completion_tokens"] for s in non_padding if s["completion_tokens"] is not None]

    summary = {
        "num_samples_total": len(samples),
        "num_samples_non_padding": len(non_padding),
        "response_length_mean": float(np.mean(response_lengths)) if response_lengths else None,
        "response_length_p50": float(np.percentile(response_lengths, 50)) if response_lengths else None,
        "response_length_p90": float(np.percentile(response_lengths, 90)) if response_lengths else None,
        "response_length_p99": float(np.percentile(response_lengths, 99)) if response_lengths else None,
        "response_length_max": int(np.max(response_lengths)) if response_lengths else None,
        "completion_tokens_mean": float(np.mean(completion_tokens)) if completion_tokens else None,
        "completion_tokens_max": int(np.max(completion_tokens)) if completion_tokens else None,
    }

    if rollout_metrics:
        keys = sorted({k for metric in rollout_metrics for k in metric.keys()})
        summary["rollout_metrics_mean"] = {
            key: mean([metric[key] for metric in rollout_metrics if key in metric]) for key in keys
        }
    if perf_metrics:
        keys = sorted({k for metric in perf_metrics for k in metric.keys()})
        summary["perf_metrics_mean"] = {key: mean([metric[key] for metric in perf_metrics if key in metric]) for key in keys}
    return summary, response_lengths


def plot_histogram(response_lengths, output_path):
    plt.figure(figsize=(10, 6))
    plt.hist(response_lengths, bins=50, color="#2c7fb8", edgecolor="black", alpha=0.85)
    plt.title("Response Length Distribution")
    plt.xlabel("Response Length (tokens)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze APRIL rollout debug data.")
    parser.add_argument("--debug-dir", required=True, help="Directory containing rollout debug pickle files.")
    parser.add_argument("--output-dir", required=True, help="Directory to write analysis outputs.")
    parser.add_argument("--log-path", default=None, help="Optional job output log path.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    samples = flatten_samples(args.debug_dir)
    perf_metrics, rollout_metrics = parse_log_metrics(args.log_path)
    summary, response_lengths = summarize(samples, perf_metrics, rollout_metrics)

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)

    histogram_path = os.path.join(args.output_dir, "response_length_hist.png")
    if response_lengths:
        plot_histogram(response_lengths, histogram_path)

    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
