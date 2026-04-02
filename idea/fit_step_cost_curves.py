#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def linear(x, a, b):
    return a + b * x


def quadratic(x, a, b, c):
    return a + b * x + c * x * x


def power(x, a, gamma):
    return a * (x**gamma)


def positive_decay(x, a, b, gamma):
    return a + b * (x ** (-gamma))


def offset_power(x, a, b, gamma):
    return a + b * np.maximum(x - 1.0, 0.0) ** gamma


MODELS = {
    "linear": {
        "fn": linear,
        "p0": [0.0, 1.0],
        "bounds": (-np.inf, np.inf),
        "formula": "a + b * rho",
    },
    "quadratic": {
        "fn": quadratic,
        "p0": [0.0, 0.0, 0.1],
        "bounds": (-np.inf, np.inf),
        "formula": "a + b * rho + c * rho^2",
    },
    "power": {
        "fn": power,
        "p0": [1.0, 1.0],
        "bounds": ([0.0, -5.0], [np.inf, 5.0]),
        "formula": "a * rho^gamma",
    },
    "positive_decay": {
        "fn": positive_decay,
        "p0": [1.0, 20.0, 1.0],
        "bounds": ([0.0, 0.0, 0.0], [np.inf, np.inf, 5.0]),
        "formula": "a + b * rho^(-gamma)",
    },
    "offset_power": {
        "fn": offset_power,
        "p0": [1.0, 1.0, 1.0],
        "bounds": ([0.0, -np.inf, 0.0], [np.inf, np.inf, 5.0]),
        "formula": "a + b * (rho - 1)^gamma",
    },
}


def fit_model(x: np.ndarray, y: np.ndarray, model_name: str) -> dict:
    model = MODELS[model_name]
    fn = model["fn"]
    popt, _ = curve_fit(fn, x, y, p0=model["p0"], bounds=model["bounds"], maxfev=20000)
    pred = fn(x, *popt)
    rss = float(np.sum((y - pred) ** 2))
    tss = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - rss / tss if tss > 0 else 1.0
    rmse = float(np.sqrt(rss / len(x)))
    aic = len(x) * np.log(rss / len(x)) + 2 * len(popt)
    return {
        "model": model_name,
        "params": [float(v) for v in popt],
        "rmse": rmse,
        "r2": float(r2),
        "aic": float(aic),
        "pred": pred,
    }


def format_params(model_name: str, params: list[float]) -> str:
    if model_name == "linear":
        a, b = params
        return f"{a:.4f} + {b:.4f} * rho"
    if model_name == "quadratic":
        a, b, c = params
        return f"{a:.4f} + {b:.4f} * rho + {c:.4f} * rho^2"
    if model_name == "power":
        a, gamma = params
        return f"{a:.4f} * rho^{gamma:.4f}"
    if model_name == "positive_decay":
        a, b, gamma = params
        return f"{a:.4f} + {b:.4f} * rho^(-{gamma:.4f})"
    if model_name == "offset_power":
        a, b, gamma = params
        return f"{a:.4f} + {b:.4f} * (rho - 1)^{gamma:.4f}"
    return str(params)


def load_summary(path: Path, metric_key: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[["window_ratio", metric_key]].dropna().sort_values("window_ratio")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit APRIL summary metrics vs window ratio curves.")
    parser.add_argument("--summary", action="append", required=True, help="LABEL=/path/to/summary.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--preferred-model", default="offset_power", choices=list(MODELS))
    parser.add_argument("--metric-key", default="mean_step_cost_sec_proxy")
    parser.add_argument("--plot-title", default=None)
    parser.add_argument("--y-label", default=None)
    parser.add_argument("--output-stem", default="fit")
    args = parser.parse_args()

    labeled = []
    for text in args.summary:
        label, path_text = text.split("=", 1)
        labeled.append((label, load_summary(Path(path_text), args.metric_key)))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    plt.figure(figsize=(9, 5))

    for label, df in labeled:
        x = df["window_ratio"].to_numpy(float)
        y = df[args.metric_key].to_numpy(float)

        fits = [fit_model(x, y, model_name) for model_name in MODELS]
        fits.sort(key=lambda item: item["aic"])
        best = fits[0]
        preferred = next(item for item in fits if item["model"] == args.preferred_model)

        x_dense = np.linspace(float(np.min(x)), float(np.max(x)), 300)
        y_dense = MODELS[preferred["model"]]["fn"](x_dense, *preferred["params"])

        plt.scatter(x, y, s=50, label=f"{label} data")
        plt.plot(x_dense, y_dense, linewidth=2, label=f"{label} {preferred['model']} fit")

        for fit in fits:
            rows.append(
                {
                    "label": label,
                    "model": fit["model"],
                    "formula": MODELS[fit["model"]]["formula"],
                    "formatted_formula": format_params(fit["model"], fit["params"]),
                    "params": str([round(v, 12) for v in fit["params"]]),
                    "rmse": fit["rmse"],
                    "r2": fit["r2"],
                    "aic": fit["aic"],
                    "is_best_aic": fit["model"] == best["model"],
                    "is_preferred_model": fit["model"] == preferred["model"],
                }
            )

    plt.title("Fitted c_hat vs Window Ratio")
    if args.plot_title:
        plt.title(args.plot_title)
    plt.xlabel("Window ratio rho = N / B")
    plt.ylabel(args.y_label or args.metric_key)
    plt.grid(True, axis="y", linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{args.output_stem}_compare.png", dpi=180)
    plt.close()

    with (output_dir / f"{args.output_stem}_table.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "label",
                "model",
                "formula",
                "formatted_formula",
                "params",
                "rmse",
                "r2",
                "aic",
                "is_best_aic",
                "is_preferred_model",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
