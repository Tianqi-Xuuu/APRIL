#!/usr/bin/env python3

import argparse
import csv
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt


LINE_RE = re.compile(
    r"\[(?P<ts>[\d\-: ]+)\] Decode batch\. "
    r"#running-req: (?P<running>\d+), "
    r"#token: (?P<tokens>\d+), "
    r"token usage: (?P<usage>[\d.]+), .*?"
    r"gen throughput \(token/s\): (?P<tps>[\d.]+), "
    r"#queue-req: (?P<queue>\d+)"
)


def label_from_run_name(run_name: str) -> str:
    if "non-provision" in run_name:
        return "non_provision"
    marker = "provision-"
    idx = run_name.index(marker) + len(marker)
    suffix = run_name[idx:].split("-bs", 1)[0]
    return suffix.replace("p", ".").rstrip(".")


def parse_trace(log_path: Path) -> list[dict]:
    rows = []
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = LINE_RE.search(line)
            if not match:
                continue
            ts = datetime.strptime(match.group("ts"), "%Y-%m-%d %H:%M:%S")
            rows.append(
                {
                    "timestamp": ts,
                    "throughput": float(match.group("tps")),
                    "running_req": int(match.group("running")),
                    "queue_req": int(match.group("queue")),
                    "token_usage": float(match.group("usage")),
                }
            )
    if not rows:
        return rows
    t0 = rows[0]["timestamp"]
    for idx, row in enumerate(rows):
        row["decode_idx"] = idx
        row["elapsed_sec"] = (row["timestamp"] - t0).total_seconds()
    return rows


def write_csv(series: dict[str, list[dict]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["label", "decode_idx", "elapsed_sec", "throughput", "running_req", "queue_req", "token_usage"]
        )
        for label, rows in series.items():
            for row in rows:
                writer.writerow(
                    [
                        label,
                        row["decode_idx"],
                        row["elapsed_sec"],
                        row["throughput"],
                        row["running_req"],
                        row["queue_req"],
                        row["token_usage"],
                    ]
                )


def plot(series: dict[str, list[dict]], output_plot: Path) -> None:
    plt.figure(figsize=(10, 5))
    for label, rows in series.items():
        plt.plot(
            [row["decode_idx"] for row in rows],
            [row["throughput"] for row in rows],
            marker="o",
            linewidth=2,
            markersize=3,
            label=label,
        )
    plt.title("Decode Throughput Trace")
    plt.xlabel("Decode Log Index")
    plt.ylabel("Token/s")
    plt.grid(True, axis="y", linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_plot, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root-base", required=True)
    parser.add_argument("--output-plot", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--include-labels", nargs="*", default=None)
    args = parser.parse_args()

    run_root = Path(args.run_root_base)
    series = {}
    for run_dir in sorted(p for p in run_root.iterdir() if p.is_dir()):
        log_path = run_dir / "job_output.log"
        if not log_path.exists():
            continue
        label = label_from_run_name(run_dir.name)
        if args.include_labels is not None and label not in args.include_labels:
            continue
        rows = parse_trace(log_path)
        if rows:
            series[label] = rows

    if not series:
        raise SystemExit(f"No decode throughput trace found under {run_root}")

    write_csv(series, Path(args.output_csv))
    plot(series, Path(args.output_plot))


if __name__ == "__main__":
    main()
