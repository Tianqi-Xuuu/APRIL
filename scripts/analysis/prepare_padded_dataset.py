import argparse
import math
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Pad a parquet dataset to a multiple of batch size.")
    parser.add_argument("--input", required=True, help="Input parquet path.")
    parser.add_argument("--output", required=True, help="Output parquet path.")
    parser.add_argument("--batch-size", type=int, required=True, help="Target batch size multiple.")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    df = df.copy()

    metadata = []
    for idx in range(len(df)):
        metadata.append(
            {
                "source_index": idx,
                "is_padding": False,
            }
        )
    df["metadata"] = metadata

    remainder = len(df) % args.batch_size
    padding_needed = (args.batch_size - remainder) % args.batch_size
    if padding_needed:
        padding_df = df.iloc[:padding_needed].copy()
        padding_metadata = []
        for pad_idx in range(padding_needed):
            padding_metadata.append(
                {
                    "source_index": int(padding_df.iloc[pad_idx]["metadata"]["source_index"]),
                    "is_padding": True,
                    "padding_copy_id": pad_idx,
                }
            )
        padding_df["metadata"] = padding_metadata
        df = pd.concat([df, padding_df], ignore_index=True)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_parquet(args.output, index=False)

    print(
        {
            "input_rows": len(pd.read_parquet(args.input)),
            "output_rows": len(df),
            "padding_needed": padding_needed,
            "num_rollouts": math.ceil(len(df) / args.batch_size),
        }
    )


if __name__ == "__main__":
    main()
