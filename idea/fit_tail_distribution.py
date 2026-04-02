#!/usr/bin/env python3

import argparse
import csv
import json
import math
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


DISTRIBUTIONS = ("weibull", "gamma", "lognorm")


def load_response_lengths(debug_dir: Path) -> np.ndarray:
    values = []
    for pickle_path in sorted(debug_dir.glob("rollout_*.pkl")):
        with pickle_path.open("rb") as f:
            samples = pickle.load(f)
        for sample in samples:
            if not isinstance(sample, dict):
                continue
            value = sample.get("response_length")
            if value is None:
                continue
            values.append(float(value))
    if not values:
        raise ValueError(f"No response lengths found under {debug_dir}")
    return np.asarray(values, dtype=np.float64)


def reduce_grouped(values: np.ndarray, group_size: int, reduce: str) -> np.ndarray:
    if group_size <= 1:
        return values
    usable = (len(values) // group_size) * group_size
    if usable == 0:
        raise ValueError("No grouped samples available")
    grouped = values[:usable].reshape(-1, group_size)
    if reduce == "mean":
        return grouped.mean(axis=1)
    if reduce == "max":
        return grouped.max(axis=1)
    if reduce == "min":
        return grouped.min(axis=1)
    raise ValueError(f"Unsupported reduce mode: {reduce}")


def fit_distribution(name: str, tail_shifted: np.ndarray) -> dict:
    if name == "weibull":
        shape, loc, scale = stats.weibull_min.fit(tail_shifted, floc=0)
        logpdf = stats.weibull_min.logpdf(tail_shifted, shape, loc=0, scale=scale)
        cdf_fn = lambda x: stats.weibull_min.cdf(x, shape, loc=0, scale=scale)
        params = {"shape": float(shape), "scale": float(scale)}
    elif name == "gamma":
        shape, loc, scale = stats.gamma.fit(tail_shifted, floc=0)
        logpdf = stats.gamma.logpdf(tail_shifted, shape, loc=0, scale=scale)
        cdf_fn = lambda x: stats.gamma.cdf(x, shape, loc=0, scale=scale)
        params = {"shape": float(shape), "scale": float(scale)}
    elif name == "lognorm":
        sigma, loc, scale = stats.lognorm.fit(tail_shifted, floc=0)
        logpdf = stats.lognorm.logpdf(tail_shifted, sigma, loc=0, scale=scale)
        cdf_fn = lambda x: stats.lognorm.cdf(x, sigma, loc=0, scale=scale)
        params = {"sigma": float(sigma), "scale": float(scale)}
    else:
        raise ValueError(f"Unsupported distribution: {name}")

    ll = float(np.sum(logpdf))
    ks = stats.kstest(tail_shifted, cdf_fn)
    num_params = len(params)
    aic = 2 * num_params - 2 * ll
    return {
        "name": name,
        "log_likelihood": ll,
        "aic": float(aic),
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "params": params,
    }


def make_cdf_fn(name: str, params: dict):
    if name == "weibull":
        return lambda x: stats.weibull_min.cdf(x, params["shape"], loc=0, scale=params["scale"])
    if name == "gamma":
        return lambda x: stats.gamma.cdf(x, params["shape"], loc=0, scale=params["scale"])
    if name == "lognorm":
        return lambda x: stats.lognorm.cdf(x, params["sigma"], loc=0, scale=params["scale"])
    raise ValueError(f"Unsupported distribution: {name}")


def plot_tail_fit(
    full_values: np.ndarray,
    threshold: float,
    tail_values: np.ndarray,
    best_fit: dict,
    output_path: Path,
    title: str,
) -> None:
    shifted = tail_values - threshold
    cdf_fn = make_cdf_fn(best_fit["name"], best_fit["params"])

    fig, (ax_hist, ax_cdf) = plt.subplots(1, 2, figsize=(12, 4.8))

    bins = 48 if tail_values.max() <= 1024 else 64
    ax_hist.hist(tail_values, bins=bins, density=True, alpha=0.6, color="#60A5FA", edgecolor="white")
    x = np.linspace(threshold, tail_values.max(), 400)
    y = np.maximum(x - threshold, 1e-12)
    if best_fit["name"] == "weibull":
        pdf = stats.weibull_min.pdf(y, best_fit["params"]["shape"], loc=0, scale=best_fit["params"]["scale"])
    elif best_fit["name"] == "gamma":
        pdf = stats.gamma.pdf(y, best_fit["params"]["shape"], loc=0, scale=best_fit["params"]["scale"])
    else:
        pdf = stats.lognorm.pdf(y, best_fit["params"]["sigma"], loc=0, scale=best_fit["params"]["scale"])
    ax_hist.plot(x, pdf, color="#DC2626", linewidth=2.2, label=f"{best_fit['name']} fit")
    ax_hist.axvline(threshold, color="#111827", linestyle="--", linewidth=1.2, label=f"threshold={threshold:g}")
    ax_hist.set_title("Tail Histogram + Fitted PDF")
    ax_hist.set_xlabel("Response length (tokens)")
    ax_hist.set_ylabel("Density")
    ax_hist.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax_hist.legend()

    sorted_tail = np.sort(tail_values)
    empirical_cdf = np.arange(1, len(sorted_tail) + 1) / len(sorted_tail)
    fitted_cdf = cdf_fn(sorted_tail - threshold)
    ax_cdf.plot(sorted_tail, empirical_cdf, color="#2563EB", linewidth=2.0, label="Empirical tail CDF")
    ax_cdf.plot(sorted_tail, fitted_cdf, color="#DC2626", linewidth=2.0, linestyle="--", label="Fitted tail CDF")
    ax_cdf.axvline(threshold, color="#111827", linestyle="--", linewidth=1.2)
    ax_cdf.set_title("Tail CDF")
    ax_cdf.set_xlabel("Response length (tokens)")
    ax_cdf.set_ylabel("Conditional tail CDF")
    ax_cdf.set_ylim(0, 1.01)
    ax_cdf.grid(True, linestyle="--", alpha=0.35)
    ax_cdf.legend()

    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_summary(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["distribution", "aic", "ks_statistic", "ks_pvalue", "params_json"])
        for row in rows:
            writer.writerow(
                [
                    row["name"],
                    row["aic"],
                    row["ks_statistic"],
                    row["ks_pvalue"],
                    json.dumps(row["params"], ensure_ascii=True, sort_keys=True),
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit a simple parametric distribution to the long-tail body of APRIL response lengths.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threshold", type=float, required=True, help="Fit only lengths strictly greater than this threshold.")
    parser.add_argument("--dist", choices=DISTRIBUTIONS, default=None, help="Force a single distribution instead of selecting by AIC.")
    parser.add_argument("--group-size", type=int, default=1)
    parser.add_argument("--group-reduce", choices=["mean", "max", "min"], default="mean")
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    debug_dir = run_dir / "debug_rollout"
    values = load_response_lengths(debug_dir)
    values = reduce_grouped(values, args.group_size, args.group_reduce)

    tail = values[values > args.threshold]
    if len(tail) == 0:
        raise ValueError(f"No samples above threshold={args.threshold}")
    tail_shifted = tail - args.threshold

    dist_names = [args.dist] if args.dist else list(DISTRIBUTIONS)
    fit_rows = [fit_distribution(name, tail_shifted) for name in dist_names]
    fit_rows.sort(key=lambda row: row["aic"])
    best_fit = fit_rows[0]

    output_dir = Path(args.output_dir)
    summary = {
        "threshold": float(args.threshold),
        "total_count": int(len(values)),
        "tail_count": int(len(tail)),
        "tail_fraction": float(len(tail) / len(values)),
        "tail_min": float(np.min(tail)),
        "tail_p50": float(np.percentile(tail, 50)),
        "tail_p90": float(np.percentile(tail, 90)),
        "tail_p95": float(np.percentile(tail, 95)),
        "tail_p99": float(np.percentile(tail, 99)),
        "tail_max": float(np.max(tail)),
        "best_fit": best_fit,
        "candidates": fit_rows,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "tail_fit_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True, sort_keys=True)
    write_summary(output_dir / "tail_fit_candidates.csv", fit_rows)

    title = args.title or f"Tail Fit: {run_dir.name}"
    plot_tail_fit(values, args.threshold, tail, best_fit, output_dir / "tail_fit.png", title)


if __name__ == "__main__":
    main()
