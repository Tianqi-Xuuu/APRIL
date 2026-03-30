#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset


SYSTEM_SUFFIX = (
    '\n\nRemember to put your final answer on its own line as '
    '"Answer: \\\\boxed{$Answer}".'
)


def extract_final_answer(answer_text: str) -> str:
    if "####" in answer_text:
        return answer_text.split("####")[-1].strip()
    return answer_text.strip()


def build_prompt(question: str) -> list[dict[str, str]]:
    prompt = (
        "Solve the following math problem step by step. "
        "The last line of your response should be of the form "
        "Answer: \\boxed{$Answer} (without quotes) where $Answer is the answer "
        f"to the problem.\n\n{question.strip()}{SYSTEM_SUFFIX}"
    )
    return [{"role": "user", "content": prompt}]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--config", default="main")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("gsm8k", args.config, split=args.split)
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    records = []
    for idx, sample in enumerate(dataset):
        answer = sample["answer"]
        records.append(
            {
                "source_prompt": build_prompt(sample["question"]),
                "answer": extract_final_answer(answer),
                "solution": answer,
                "metadata": {
                    "source_index": idx,
                    "dataset": "gsm8k",
                    "split": args.split,
                    "config": args.config,
                },
            }
        )

    df = pd.DataFrame(records)
    df.to_parquet(output_path, index=False)
    print(
        {
            "output": str(output_path),
            "rows": len(df),
            "columns": list(df.columns),
        }
    )


if __name__ == "__main__":
    main()
