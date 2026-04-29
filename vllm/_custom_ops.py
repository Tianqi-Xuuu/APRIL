def scaled_fp8_quant(*args, **kwargs):
    raise RuntimeError(
        "APRIL Modal shim: vllm._custom_ops.scaled_fp8_quant is unavailable in this runtime. "
        "This path should remain unused for the current Qwen2.5-3B training setup."
    )
