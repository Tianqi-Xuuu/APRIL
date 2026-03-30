#!/usr/bin/env python3

import argparse
import ast
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt


ROLLOUT_RE = re.compile(r"(?<!partial_)rollout (\d+): (\{.*\})")
EVAL_RE = re.compile(r"eval (\d+): (\{.*\})")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-png", required=True)
    parser.add_argument("--eval-key", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    log_path = Path(args.log_path)
    output_csv = Path(args.output_csv)
    output_png = Path(args.output_png)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    rollout_rows = []
    eval_rows = []

    for line in log_path.read_text(errors="ignore").splitlines():
        rollout_match = ROLLOUT_RE.search(line)
        if rollout_match:
            rollout_id = int(rollout_match.group(1))
            payload = ast.literal_eval(rollout_match.group(2))
            if "rollout/raw_reward" not in payload:
                continue
            rollout_rows.append(
                {
                    "step_type": "rollout",
                    "step_id": rollout_id,
                    "raw_reward": payload.get("rollout/raw_reward"),
                    "truncated": payload.get("rollout/truncated"),
                }
            )
            continue

        eval_match = EVAL_RE.search(line)
        if eval_match:
            eval_id = int(eval_match.group(1))
            payload = ast.literal_eval(eval_match.group(2))
            eval_key = args.eval_key
            if eval_key is None:
                eval_candidates = [k for k in payload.keys() if k.startswith("eval/") and not k.endswith("-truncated_ratio")]
                eval_key = eval_candidates[0] if eval_candidates else None
            eval_rows.append(
                {
                    "step_type": "eval",
                    "step_id": eval_id,
                    "eval_reward": payload.get(eval_key) if eval_key else None,
                    "eval_key": eval_key,
                }
            )

    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step_type", "step_id", "raw_reward", "truncated", "eval_reward", "eval_key"])
        writer.writeheader()
        for row in rollout_rows:
            writer.writerow({**row, "eval_reward": "", "eval_key": ""})
        for row in eval_rows:
            writer.writerow({**row, "raw_reward": "", "truncated": ""})

    plt.figure(figsize=(10, 5))
    if rollout_rows:
        plt.plot(
            [row["step_id"] for row in rollout_rows],
            [row["raw_reward"] for row in rollout_rows],
            marker="o",
            linewidth=2,
            label="rollout/raw_reward",
        )
    if eval_rows:
        plt.plot(
            [row["step_id"] for row in eval_rows],
            [row["eval_reward"] for row in eval_rows],
            marker="s",
            linewidth=2,
            label=eval_rows[0]["eval_key"] or "eval reward",
        )

    plt.xlabel("Rollout ID")
    plt.ylabel("Reward")
    plt.title("Non-provision Reward Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png, dpi=160)


if __name__ == "__main__":
    main()
