import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sglang.srt.utils import kill_process_tree
from transformers import AutoTokenizer

from slime.backends.sglang_utils.http_server_engine import HttpServerEngineAdapter
from slime.rollout.sglang_example import _extract_behavior_logprobs_from_output, _merge_behavior_logprobs


def _to_jsonable(value):
    if isinstance(value, np.ndarray):
        return _to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_jsonable(val) for key, val in value.items()}
    return value


def _normalize_text_value(value) -> str:
    value = _to_jsonable(value)
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, list):
        if len(value) == 1:
            return _normalize_text_value(value[0])
        if value and all(isinstance(item, str) for item in value):
            return "".join(value)
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, dict):
        for key in ("source_prompt", "prompt", "text", "content", "question"):
            if key in value:
                return _normalize_text_value(value[key])
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _build_sampling_params(max_new_tokens: int) -> dict:
    return {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "max_new_tokens": max_new_tokens,
        "skip_special_tokens": False,
        "no_stop_trim": True,
        "spaces_between_special_tokens": False,
    }


def _post_generate(
    url: str,
    text: str,
    sampling_params: dict,
    *,
    logprob_start_len: int = -1,
    return_text_in_logprobs: bool = False,
) -> dict:
    payload = {
        "text": text,
        "sampling_params": sampling_params,
        "return_logprob": True,
        "logprob_start_len": logprob_start_len,
        "top_logprobs_num": 0,
    }
    if return_text_in_logprobs:
        payload["return_text_in_logprobs"] = True
    response = requests.post(url, json=payload, timeout=600)
    response.raise_for_status()
    return response.json()


def _response_token_ids(tokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def _choose_segment_sizes(total_tokens: int) -> list[int]:
    if total_tokens <= 1:
        return [total_tokens]
    if total_tokens <= 3:
        return [1, total_tokens - 1]
    if total_tokens <= 5:
        return [2, total_tokens - 2]
    return [2, 2, total_tokens - 4]


def _max_abs_diff(lhs: list[float], rhs: list[float]) -> float | None:
    if len(lhs) != len(rhs) or not lhs:
        return None
    return max(abs(a - b) for a, b in zip(lhs, rhs))


def _mean_abs_diff(lhs: list[float], rhs: list[float]) -> float | None:
    if len(lhs) != len(rhs) or not lhs:
        return None
    return sum(abs(a - b) for a, b in zip(lhs, rhs)) / len(lhs)


def _launch_server(model_path: Path, sglang_mem_fraction: float, *, port: int) -> HttpServerEngineAdapter:
    return HttpServerEngineAdapter(
        model_path=str(model_path),
        tokenizer_path=str(model_path),
        trust_remote_code=True,
        host="127.0.0.1",
        port=port,
        nccl_port=port + 1,
        nnodes=1,
        node_rank=0,
        dist_init_addr=f"127.0.0.1:{port + 2}",
        base_gpu_id=0,
        gpu_id_step=1,
        tp_size=1,
        dp_size=1,
        pp_size=1,
        ep_size=1,
        skip_server_warmup=True,
        mem_fraction_static=sglang_mem_fraction,
        enable_memory_saver=True,
        disable_cuda_graph=True,
        weight_loader_disable_mmap=True,
    )


def _extract_input_logprobs_from_output(output: dict) -> list[float]:
    vals = output.get("meta_info", {}).get("input_token_logprobs")
    if vals is None:
        return []
    extracted = []
    for item in vals:
        if isinstance(item, (int, float)):
            extracted.append(float(item))
            continue
        if isinstance(item, dict):
            if isinstance(item.get("logprob"), (int, float)):
                extracted.append(float(item["logprob"]))
                continue
            return []
        if isinstance(item, (list, tuple)):
            found = None
            for value in item:
                if isinstance(value, (int, float)):
                    found = float(value)
                    break
            if found is None:
                return []
            extracted.append(found)
            continue
        return []
    return extracted


def _score_response_tokens(
    *,
    url: str,
    tokenizer,
    prompt: str,
    response: str,
) -> dict:
    prompt_ids = _response_token_ids(tokenizer, prompt)
    response_ids = _response_token_ids(tokenizer, response)
    output = _post_generate(
        url,
        prompt + response,
        _build_sampling_params(max_new_tokens=0),
        logprob_start_len=max(len(prompt_ids) - 1, 0),
        return_text_in_logprobs=True,
    )
    input_logprobs = _extract_input_logprobs_from_output(output)
    aligned = input_logprobs[-len(response_ids) :] if len(input_logprobs) >= len(response_ids) else []
    return {
        "raw_len": len(input_logprobs),
        "aligned_len": len(aligned),
        "aligned_logprobs": aligned,
    }


def run_probe(
    *,
    model_path: Path,
    dataset_path: Path,
    sample_index: int = 0,
    max_new_tokens: int = 64,
    sglang_mem_fraction: float = 0.5,
    score_model_path: Path | None = None,
) -> dict:
    df = pd.read_parquet(dataset_path)
    row = df.iloc[sample_index]
    prompt = _normalize_text_value(row["source_prompt"])
    answer = _to_jsonable(row.get("answer"))

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    adapter = _launch_server(model_path, sglang_mem_fraction, port=10000)
    url = "http://127.0.0.1:10000/generate"
    score_adapter = None
    score_url = url
    if score_model_path is not None and score_model_path != model_path:
        score_adapter = _launch_server(score_model_path, sglang_mem_fraction, port=11000)
        score_url = "http://127.0.0.1:11000/generate"

    try:
        full_output = _post_generate(url, prompt, _build_sampling_params(max_new_tokens=max_new_tokens))
        full_response = full_output["text"]
        full_response_ids = _response_token_ids(tokenizer, full_response)
        full_behavior_log_probs = _extract_behavior_logprobs_from_output(full_output) or []

        if len(full_response_ids) < 2:
            raise RuntimeError(
                f"Generated response too short for stitch probe: {len(full_response_ids)} token(s). "
                f"Sample index={sample_index}"
            )

        segment_sizes = _choose_segment_sizes(len(full_response_ids))
        accumulated_response = ""
        merged_behavior_log_probs = None
        segment_reports = []

        for segment_id, segment_len in enumerate(segment_sizes):
            output = _post_generate(url, prompt + accumulated_response, _build_sampling_params(max_new_tokens=segment_len))
            segment_text = output["text"]
            segment_behavior_log_probs = _extract_behavior_logprobs_from_output(output) or []
            old_response_len = len(_response_token_ids(tokenizer, accumulated_response))
            accumulated_response += segment_text
            total_response_len = len(_response_token_ids(tokenizer, accumulated_response))

            merged_behavior_log_probs = _merge_behavior_logprobs(
                behavior_lp=segment_behavior_log_probs,
                existing=merged_behavior_log_probs,
                target_total=total_response_len,
                old_response_len=old_response_len,
            )

            segment_reports.append(
                {
                    "segment_id": segment_id,
                    "requested_max_new_tokens": segment_len,
                    "segment_text_preview": segment_text[:200],
                    "segment_response_tokens": len(_response_token_ids(tokenizer, segment_text)),
                    "segment_logprob_len": len(segment_behavior_log_probs),
                    "old_response_len": old_response_len,
                    "merged_len_after_segment": len(merged_behavior_log_probs) if merged_behavior_log_probs else 0,
                }
            )

        stitched_response_ids = _response_token_ids(tokenizer, accumulated_response)
        stitched_behavior_log_probs = merged_behavior_log_probs or []
        scored = _score_response_tokens(url=score_url, tokenizer=tokenizer, prompt=prompt, response=full_response)

        return {
            "dataset_path": str(dataset_path),
            "generate_model_path": str(model_path),
            "score_model_path": str(score_model_path or model_path),
            "sample_index": sample_index,
            "ground_truth_answer": answer,
            "prompt_preview": prompt[:300],
            "full_response_preview": full_response[:300],
            "full_response_token_count": len(full_response_ids),
            "full_raw_logprob_len": len(full_behavior_log_probs),
            "segment_sizes": segment_sizes,
            "segments": segment_reports,
            "stitched_response_preview": accumulated_response[:300],
            "stitched_response_token_count": len(stitched_response_ids),
            "stitched_logprob_len": len(stitched_behavior_log_probs),
            "stitched_text_matches_full": accumulated_response == full_response,
            "stitched_tokens_match_full": stitched_response_ids == full_response_ids,
            "stitch_matches_current_merge_logic": len(stitched_behavior_log_probs) == len(stitched_response_ids),
            "raw_logprob_len_matches_full_tokens": len(full_behavior_log_probs) == len(full_response_ids),
            "stitched_vs_full_max_abs_diff": _max_abs_diff(stitched_behavior_log_probs, full_behavior_log_probs),
            "stitched_vs_full_mean_abs_diff": _mean_abs_diff(stitched_behavior_log_probs, full_behavior_log_probs),
            "score_input_logprob_raw_len": scored["raw_len"],
            "score_input_logprob_aligned_len": scored["aligned_len"],
            "score_vs_full_max_abs_diff": _max_abs_diff(scored["aligned_logprobs"], full_behavior_log_probs),
            "score_vs_stitched_max_abs_diff": _max_abs_diff(scored["aligned_logprobs"], stitched_behavior_log_probs),
            "full_raw_logprob_head": full_behavior_log_probs[:10],
            "stitched_logprob_head": stitched_behavior_log_probs[:10],
            "score_logprob_head": scored["aligned_logprobs"][:10],
        }
    finally:
        if score_adapter is not None:
            kill_process_tree(score_adapter.process.pid)
        kill_process_tree(adapter.process.pid)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Probe raw vs stitched SGLang behavior logprobs.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--sglang-mem-fraction", type=float, default=0.5)
    parser.add_argument("--score-model-path")
    args = parser.parse_args()

    result = run_probe(
        model_path=Path(args.model_path),
        dataset_path=Path(args.dataset_path),
        sample_index=args.sample_index,
        max_new_tokens=args.max_new_tokens,
        sglang_mem_fraction=args.sglang_mem_fraction,
        score_model_path=Path(args.score_model_path) if args.score_model_path else None,
    )
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
