#!/usr/bin/env python3
"""Export eval metrics vs rollout_id and save CSV + PNG inside each run directory."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _eval_rows_to_frame(rows: list[dict]) -> pd.DataFrame:
    """Wide table: rollout_id + one column per eval/* score + matching truncated_ratio."""
    records: list[dict] = []
    for row in rows:
        rid = row["rollout_id"]
        metrics = row["metrics"]
        bases = [
            k
            for k in metrics
            if k.startswith("eval/") and not k.endswith("-truncated_ratio")
        ]
        rec: dict = {"rollout_id": rid}
        for k in sorted(bases):
            slug = re.sub(r"^eval/", "", k)
            slug = re.sub(r"[^\w]+", "_", slug).strip("_")
            rec[f"eval_{slug}"] = metrics[k]
            tr_key = f"{k}-truncated_ratio"
            if tr_key in metrics:
                rec[f"eval_{slug}_truncated_ratio"] = metrics[tr_key]
        records.append(rec)
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame.from_records(records).sort_values("rollout_id").reset_index(drop=True)
    return df


def _primary_eval_column(df: pd.DataFrame) -> str | None:
    cols = [c for c in df.columns if c.startswith("eval_") and not c.endswith("_truncated_ratio")]
    if not cols:
        return None
    preferred = [c for c in cols if "math_level2" in c and "subset64" in c]
    if preferred:
        return preferred[0]
    return sorted(cols)[0]


def plot_run(run_dir: Path, df: pd.DataFrame, y_col: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["rollout_id"], df[y_col], marker="o", linewidth=1.2, markersize=4)
    ax.set_xlabel("Rollout step (rollout_id at eval)")
    ax.set_ylabel(y_col.replace("eval_", "").replace("_", "/"))
    ax.set_title(run_dir.name)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png = run_dir / "eval_vs_rollout_step.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def process_run(run_dir: Path, *, dry_run: bool) -> bool:
    eval_path = run_dir / "analysis" / "eval_metrics.jsonl"
    if not eval_path.is_file():
        return False
    rows = _load_jsonl(eval_path)
    df = _eval_rows_to_frame(rows)
    if df.empty:
        return False
    y_col = _primary_eval_column(df)
    if y_col is None:
        return False
    csv_path = run_dir / "eval_vs_rollout_step.csv"
    if not dry_run:
        df.to_csv(csv_path, index=False)
        plot_run(run_dir, df, y_col)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("run_4_16"),
        help="Directory whose subfolders are run names (each with analysis/eval_metrics.jsonl).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print which runs would be processed.",
    )
    args = parser.parse_args()
    root: Path = args.run_root
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    n_ok = 0
    for child in sorted(root.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        if process_run(child, dry_run=args.dry_run):
            print(child.name)
            n_ok += 1
        elif (child / "analysis" / "eval_metrics.jsonl").exists():
            print(f"{child.name} (skipped: empty or no eval columns)", file=sys.stderr)

    if n_ok == 0:
        raise SystemExit(f"No runs with analysis/eval_metrics.jsonl under {root}")


if __name__ == "__main__":
    main()
