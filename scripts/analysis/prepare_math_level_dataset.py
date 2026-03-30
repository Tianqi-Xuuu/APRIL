#!/usr/bin/env python3

import argparse
import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset


SYSTEM_SUFFIX = (
    '\n\nRemember to put your final answer on its own line as '
    '"Answer: \\\\boxed{123}".'
)


def build_prompt(question: str) -> list[dict[str, str]]:
    prompt = (
        "Solve the following math problem step by step. "
        "The last line of your response should be of the form "
        "Answer: \\boxed{123} (without quotes), replacing 123 with the final answer "
        f"to the problem.\n\n{question.strip()}{SYSTEM_SUFFIX}"
    )
    return [{"role": "user", "content": prompt}]


def extract_final_answer(solution: str) -> str:
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", solution)
    if boxed:
        return boxed[-1].strip()

    answer_lines = re.findall(r"Answer:\s*(.+)", solution, flags=re.IGNORECASE)
    if answer_lines:
        return answer_lines[-1].strip().rstrip(".。")

    return solution.strip()


def normalize_levels(levels: list[str]) -> set[str]:
    normalized = set()
    for level in levels:
        level = level.strip()
        if not level:
            continue
        if level.lower().startswith("level "):
            normalized.add(level)
        else:
            normalized.add(f"Level {level}")
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--dataset", default="SuperSecureHuman/competition_math_hf_dataset")
    parser.add_argument("--levels", nargs="+", default=["1", "2"])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    levels = normalize_levels(args.levels)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.dataset, split=args.split)
    records = []
    for idx, sample in enumerate(dataset):
        if sample["level"] not in levels:
            continue
        records.append(
            {
                "source_prompt": build_prompt(sample["problem"]),
                "answer": extract_final_answer(sample["solution"]),
                "solution": sample["solution"],
                "metadata": {
                    "source_index": idx,
                    "dataset": "competition_math",
                    "split": args.split,
                    "level": sample["level"],
                    "type": sample["type"],
                },
            }
        )
        if args.limit is not None and len(records) >= args.limit:
            break

    df = pd.DataFrame(records)
    df.to_parquet(output_path, index=False)
    print(
        {
            "output": str(output_path),
            "rows": len(df),
            "levels": sorted(levels),
            "split": args.split,
            "dataset": args.dataset,
        }
    )


if __name__ == "__main__":
    main()
