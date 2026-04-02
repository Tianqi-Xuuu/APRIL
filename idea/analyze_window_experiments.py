#!/usr/bin/env python3

import argparse
import ast
import csv
import json
import math
import pickle
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
CONFIG_RE = re.compile(r"^\s*(?P<key>[A-Za-z0-9_-]+)\s+\.{2,}\s+(?P<value>.+?)\s*$")
PARTIAL_RE = re.compile(r"partial_rollout (?P<rollout_id>\d+): (?P<payload>\{.*\})")
DECODE_RE = re.compile(
    r"\[(?P<ts>[\d\-: ]+)\] Decode batch\. "
    r"#running-req: (?P<running>\d+), "
    r"#token: (?P<tokens>\d+), "
    r"token usage: (?P<usage>[\d.]+), .*?"
    r"gen throughput \(token/s\): (?P<tps>[\d.]+), "
    r"#queue-req: (?P<queue>\d+)"
)


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def normalize_metric_keys(metric: dict) -> dict:
    normalized = {}
    for key, value in metric.items():
        if "/" in key:
            normalized[key.split("/", 1)[1]] = value
        else:
            normalized[key] = value
    return normalized


def safe_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def mean_or_none(values: list[float]) -> float | None:
    filtered = [value for value in values if value is not None]
    return mean(filtered) if filtered else None


def parse_bool(text: str | None) -> bool | None:
    if text is None:
        return None
    lowered = text.strip().lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    return None


def percentile_or_none(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(values, pct))


def scan_run_dirs(root: Path, recursive: bool) -> list[Path]:
    candidates = []
    if recursive:
        for log_path in root.rglob("job_output.log"):
            candidates.append(log_path.parent)
        for debug_dir in root.rglob("debug_rollout"):
            candidates.append(debug_dir.parent)
    else:
        for child in root.iterdir():
            if not child.is_dir():
                continue
            if (child / "job_output.log").exists() or (child / "debug_rollout").is_dir():
                candidates.append(child)
    unique = sorted(set(candidates))
    return unique


def parse_config(log_path: Path) -> dict:
    config = {}
    if not log_path.exists():
        return config
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = strip_ansi(raw_line.rstrip("\n"))
            match = CONFIG_RE.match(line)
            if match:
                config[match.group("key")] = match.group("value").strip()
    return config


def parse_partial_metrics(log_path: Path) -> list[tuple[int, dict]]:
    rows = []
    if not log_path.exists():
        return rows
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = strip_ansi(raw_line.rstrip("\n"))
            match = PARTIAL_RE.search(line)
            if not match:
                continue
            payload = ast.literal_eval(match.group("payload"))
            rows.append((int(match.group("rollout_id")), normalize_metric_keys(payload)))
    return rows


def parse_decode_trace_by_rollout(log_path: Path) -> dict[int, list[dict]]:
    decode_rows_by_rollout: dict[int, list[dict]] = {}
    pending_decode_rows: list[dict] = []
    if not log_path.exists():
        return decode_rows_by_rollout

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = strip_ansi(raw_line.rstrip("\n"))
            decode_match = DECODE_RE.search(line)
            if decode_match:
                pending_decode_rows.append(
                    {
                        "timestamp": datetime.strptime(decode_match.group("ts"), "%Y-%m-%d %H:%M:%S"),
                        "running_req": int(decode_match.group("running")),
                        "queue_req": int(decode_match.group("queue")),
                        "token_usage": float(decode_match.group("usage")),
                        "throughput": float(decode_match.group("tps")),
                    }
                )
                continue

            partial_match = PARTIAL_RE.search(line)
            if partial_match:
                rollout_id = int(partial_match.group("rollout_id"))
                decode_rows_by_rollout[rollout_id] = pending_decode_rows
                pending_decode_rows = []

    return decode_rows_by_rollout


def parse_run_name(run_name: str) -> dict:
    info = {
        "run_name": run_name,
        "tail_label": "default",
        "window_ratio": None,
        "rollout_batch_size": None,
        "over_sampling_batch_size": None,
        "rollout_max_response_len": None,
    }

    if match := re.search(r"bs(?P<bs>\d+)", run_name):
        info["rollout_batch_size"] = int(match.group("bs"))
    if match := re.search(r"len(?P<len>\d+)", run_name):
        info["rollout_max_response_len"] = int(match.group("len"))
        info["tail_label"] = f"len{match.group('len')}"
    if match := re.search(r"over(?P<over>\d+)", run_name):
        info["over_sampling_batch_size"] = int(match.group("over"))
    elif match := re.search(r"prov(?P<prov>\d+)", run_name):
        ratio = int(match.group("prov")) / 10.0
        info["window_ratio"] = ratio
        if info["rollout_batch_size"] is not None:
            info["over_sampling_batch_size"] = int(round(info["rollout_batch_size"] * ratio))
    elif "nonprovision" in run_name or "non-provision" in run_name:
        if info["rollout_batch_size"] is not None:
            info["over_sampling_batch_size"] = info["rollout_batch_size"]
            info["window_ratio"] = 1.0

    if info["window_ratio"] is None and info["rollout_batch_size"] and info["over_sampling_batch_size"]:
        info["window_ratio"] = info["over_sampling_batch_size"] / info["rollout_batch_size"]
    return info


def infer_run_settings(run_dir: Path, config: dict) -> dict:
    info = parse_run_name(run_dir.name)

    rollout_bs = info["rollout_batch_size"]
    if rollout_bs is None and config.get("rollout_batch_size"):
        rollout_bs = int(config["rollout_batch_size"])
    over_bs = info["over_sampling_batch_size"]
    if over_bs is None and config.get("over_sampling_batch_size"):
        over_bs = int(config["over_sampling_batch_size"])
    partial_flag = parse_bool(config.get("partial_rollout"))
    if partial_flag is False and rollout_bs is not None:
        over_bs = rollout_bs
    if over_bs is None and rollout_bs is not None and partial_flag is not True:
        over_bs = rollout_bs

    rollout_max_len = info["rollout_max_response_len"]
    if rollout_max_len is None and config.get("rollout_max_response_len"):
        rollout_max_len = int(config["rollout_max_response_len"])

    tail_label = info["tail_label"]
    if tail_label == "default" and rollout_max_len is not None:
        tail_label = f"len{rollout_max_len}"

    window_ratio = info["window_ratio"]
    if window_ratio is None and rollout_bs is not None and over_bs is not None:
        window_ratio = over_bs / rollout_bs

    n_samples_per_prompt = None
    if config.get("n_samples_per_prompt"):
        n_samples_per_prompt = int(config["n_samples_per_prompt"])

    return {
        "run_name": run_dir.name,
        "tail_label": tail_label,
        "rollout_batch_size": rollout_bs,
        "over_sampling_batch_size": over_bs,
        "window_ratio": window_ratio,
        "rollout_max_response_len": rollout_max_len,
        "n_samples_per_prompt": n_samples_per_prompt,
        "partial_rollout_enabled": partial_flag if partial_flag is not None else (window_ratio or 1.0) > 1.0,
    }


def load_debug_stats(debug_dir: Path) -> dict:
    if not debug_dir.is_dir():
        return {}

    response_lengths = []
    completion_tokens = []
    carryover_count = 0
    carryover_tokens = 0
    truncated_count = 0
    non_padding_count = 0
    per_rollout_metrics = []

    for pickle_path in sorted(debug_dir.glob("rollout_*.pkl")):
        with pickle_path.open("rb") as f:
            samples = pickle.load(f)
        if not samples:
            continue

        metadata0 = (samples[0].get("metadata") or {}) if isinstance(samples[0], dict) else {}
        completion_stats = metadata0.get("completion_tokens_stats") or {}
        rollout_time = safe_float(metadata0.get("rollout_time"))
        total_tokens = safe_float(completion_stats.get("total_completion_tokens"))
        total_off_policy_tokens = safe_float(metadata0.get("total_off_policy_tokens"))
        partial_samples = safe_float(metadata0.get("partial_samples"))
        tokens_throughput = None
        off_policy_ratio = None
        if rollout_time and total_tokens:
            tokens_throughput = total_tokens / rollout_time
        if total_tokens is not None and total_off_policy_tokens is not None and total_tokens + total_off_policy_tokens > 0:
            off_policy_ratio = total_off_policy_tokens / (total_tokens + total_off_policy_tokens)
        per_rollout_metrics.append(
            {
                "rollout_time": rollout_time,
                "total_tokens": total_tokens,
                "partial_samples": partial_samples,
                "total_off_policy_tokens": total_off_policy_tokens,
                "off_policy_ratio": off_policy_ratio,
                "tokens_throughput": tokens_throughput,
            }
        )

        for sample in samples:
            if not isinstance(sample, dict):
                continue
            metadata = sample.get("metadata") or {}
            if metadata.get("is_padding", False):
                continue
            non_padding_count += 1

            response_length = sample.get("response_length")
            if response_length is not None:
                response_lengths.append(float(response_length))

            comp_tokens = sample.get("completion_tokens")
            if comp_tokens is not None:
                completion_tokens.append(float(comp_tokens))

            if sample.get("status") == "truncated":
                truncated_count += 1

            start_rollout_id = metadata.get("start_rollout_id")
            final_rollout_id = metadata.get("final_rollout_id")
            if (
                start_rollout_id is not None
                and final_rollout_id is not None
                and int(start_rollout_id) < int(final_rollout_id)
            ):
                carryover_count += 1
                if comp_tokens is not None:
                    carryover_tokens += float(comp_tokens)

    total_completion_tokens = sum(completion_tokens) if completion_tokens else None
    carryover_fraction = None
    if non_padding_count > 0:
        carryover_fraction = carryover_count / non_padding_count
    carryover_token_fraction = None
    if total_completion_tokens and total_completion_tokens > 0:
        carryover_token_fraction = carryover_tokens / total_completion_tokens

    return {
        "debug_num_rollouts": len(per_rollout_metrics),
        "debug_metric_rows": per_rollout_metrics,
        "num_samples_non_padding": non_padding_count,
        "response_length_mean": mean_or_none(response_lengths),
        "response_length_p50": percentile_or_none(response_lengths, 50),
        "response_length_p90": percentile_or_none(response_lengths, 90),
        "response_length_p99": percentile_or_none(response_lengths, 99),
        "response_length_max": max(response_lengths) if response_lengths else None,
        "completion_tokens_mean": mean_or_none(completion_tokens),
        "completion_tokens_total": total_completion_tokens,
        "carryover_fraction": carryover_fraction,
        "carryover_token_fraction": carryover_token_fraction,
        "truncated_ratio": (truncated_count / non_padding_count) if non_padding_count > 0 else None,
    }


def summarize_metric_rows(metric_rows: list[dict]) -> dict:
    fields = [
        "rollout_time",
        "tokens_throughput",
        "total_tokens",
        "partial_samples",
        "off_policy_ratio",
        "total_off_policy_tokens",
        "response_length_mean",
        "response_length_p50",
        "response_length_p90",
        "response_length_p99",
        "response_length_max",
    ]
    summary = {"num_rollouts": len(metric_rows)}
    for field in fields:
        summary[f"{field}_mean"] = mean_or_none([safe_float(row.get(field)) for row in metric_rows])
    return summary


def summarize_trace_rows(trace_by_rollout: dict[int, list[dict]], num_rollouts: int) -> dict:
    all_rows = [row for rows in trace_by_rollout.values() for row in rows]
    decode_event_count = len(all_rows)
    decode_batches_per_rollout_proxy = None
    if num_rollouts > 0 and decode_event_count > 0:
        decode_batches_per_rollout_proxy = decode_event_count / num_rollouts

    return {
        "decode_event_count": decode_event_count,
        "decode_batches_per_rollout_proxy": decode_batches_per_rollout_proxy,
        "decode_throughput_mean": mean_or_none([row["throughput"] for row in all_rows]),
        "running_req_mean": mean_or_none([row["running_req"] for row in all_rows]),
        "running_req_p90": percentile_or_none([row["running_req"] for row in all_rows], 90),
        "queue_req_max": max((row["queue_req"] for row in all_rows), default=None),
        "token_usage_mean": mean_or_none([row["token_usage"] for row in all_rows]),
    }


def summarize_run(run_dir: Path, skip_decode_proxy: bool) -> dict | None:
    log_path = run_dir / "job_output.log"
    debug_dir = run_dir / "debug_rollout"

    config = parse_config(log_path)
    run_info = infer_run_settings(run_dir, config)

    partial_metric_rows = [metric for _, metric in parse_partial_metrics(log_path)]
    debug_stats = load_debug_stats(debug_dir)
    metric_rows = partial_metric_rows if partial_metric_rows else debug_stats.get("debug_metric_rows", [])
    if not metric_rows and not debug_stats:
        return None

    summary = dict(run_info)
    summary["metric_source"] = "partial_log" if partial_metric_rows else "debug_metadata"
    summary.update(summarize_metric_rows(metric_rows))
    summary.update(debug_stats)

    rollout_time_mean = summary.get("rollout_time_mean")
    rollout_batch_size = summary.get("rollout_batch_size")
    summary["group_goodput"] = None
    if rollout_time_mean and rollout_batch_size:
        summary["group_goodput"] = rollout_batch_size / rollout_time_mean

    if skip_decode_proxy:
        summary["decode_event_count"] = None
        summary["decode_batches_per_rollout_proxy"] = None
        summary["decode_throughput_mean"] = None
        summary["running_req_mean"] = None
        summary["running_req_p90"] = None
        summary["queue_req_max"] = None
        summary["token_usage_mean"] = None
        summary["mean_step_cost_sec_proxy"] = None
    else:
        trace_summary = summarize_trace_rows(parse_decode_trace_by_rollout(log_path), summary["num_rollouts"])
        summary.update(trace_summary)
        summary["mean_step_cost_sec_proxy"] = None
        decode_batches = summary.get("decode_batches_per_rollout_proxy")
        if rollout_time_mean and decode_batches and decode_batches > 0:
            summary["mean_step_cost_sec_proxy"] = rollout_time_mean / decode_batches

    return summary


def plot_metric(rows: list[dict], y_key: str, title: str, ylabel: str, output_path: Path) -> None:
    grouped = defaultdict(list)
    for row in rows:
        x = row.get("window_ratio")
        y = row.get(y_key)
        if x is None or y is None:
            continue
        grouped[row.get("tail_label", "default")].append((float(x), float(y)))

    if not grouped:
        return

    plt.figure(figsize=(9, 5))
    for label, points in sorted(grouped.items()):
        points = sorted(points, key=lambda item: item[0])
        plt.plot([x for x, _ in points], [y for _, y in points], marker="o", linewidth=2, label=label)

    plt.title(title)
    plt.xlabel("Window ratio N / B")
    plt.ylabel(ylabel)
    plt.grid(True, axis="y", linestyle="--", alpha=0.35)
    if len(grouped) > 1:
        plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


def write_csv(rows: list[dict], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(rows: list[dict], output_json: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize APRIL window-size experiments.")
    parser.add_argument("--run-root-base", required=True, help="Directory containing experiment runs.")
    parser.add_argument("--output-dir", required=True, help="Directory to store CSV and plots.")
    parser.add_argument("--recursive", action="store_true", help="Recursively discover run directories.")
    parser.add_argument("--run-name-regex", default=None, help="Only include run names matching this regex.")
    parser.add_argument(
        "--skip-decode-proxy",
        action="store_true",
        help="Skip decode-step proxy metrics. Recommended for train-mode runs with eval.",
    )
    args = parser.parse_args()

    root = Path(args.run_root_base)
    run_dirs = scan_run_dirs(root, recursive=args.recursive)
    if args.run_name_regex:
        pattern = re.compile(args.run_name_regex)
        run_dirs = [run_dir for run_dir in run_dirs if pattern.search(run_dir.name)]

    rows = []
    for run_dir in run_dirs:
        row = summarize_run(run_dir, skip_decode_proxy=args.skip_decode_proxy)
        if row is not None:
            rows.append(row)

    if not rows:
        raise SystemExit(f"No experiment runs found under {root}")

    rows = sorted(rows, key=lambda row: (row.get("tail_label") or "", row.get("window_ratio") or math.inf, row["run_name"]))

    output_dir = Path(args.output_dir)
    write_csv(rows, output_dir / "summary.csv")
    write_json(rows, output_dir / "summary.json")

    plot_metric(rows, "group_goodput", "Completed Groups/sec vs Window Ratio", "Groups / sec", output_dir / "goodput_vs_window.png")
    plot_metric(
        rows,
        "mean_step_cost_sec_proxy",
        "Mean Step Cost Proxy vs Window Ratio",
        "rollout_time / decode_batches",
        output_dir / "c_hat_vs_window.png",
    )
    plot_metric(
        rows,
        "decode_batches_per_rollout_proxy",
        "Decode Batch Count Proxy vs Window Ratio",
        "Decode batches / rollout",
        output_dir / "t_hat_vs_window.png",
    )
    plot_metric(
        rows,
        "off_policy_ratio_mean",
        "Off-policy Ratio vs Window Ratio",
        "Off-policy ratio",
        output_dir / "off_policy_vs_window.png",
    )
    plot_metric(
        rows,
        "carryover_fraction",
        "Carry-over Completion Fraction vs Window Ratio",
        "Carry-over completion fraction",
        output_dir / "carryover_vs_window.png",
    )
    plot_metric(
        rows,
        "tokens_throughput_mean",
        "Completion Tokens/sec vs Window Ratio",
        "Completion tokens / sec",
        output_dir / "tokens_per_sec_vs_window.png",
    )

    for row in rows:
        preview = {
            "run_name": row["run_name"],
            "window_ratio": row.get("window_ratio"),
            "group_goodput": row.get("group_goodput"),
            "mean_step_cost_sec_proxy": row.get("mean_step_cost_sec_proxy"),
            "decode_batches_per_rollout_proxy": row.get("decode_batches_per_rollout_proxy"),
            "off_policy_ratio_mean": row.get("off_policy_ratio_mean"),
            "carryover_fraction": row.get("carryover_fraction"),
        }
        print(json.dumps(preview, ensure_ascii=True))


if __name__ == "__main__":
    main()
