"""Single import surface for the organizers' benchmark file.

Every other module in this repo imports the organizers' classes and helpers
*through here* rather than reaching into `bench.torch_transformer_benchmark`
directly. Two reasons:

  1. It makes the dependency on the untouched organizers' file explicit and
     greppable — if this file is the only importer, nobody can quietly fork it.
  2. Our accuracy and timing numbers are then produced by exactly the functions
     the organizers' own script would call, so `bench/sweep.py` cannot drift
     from `python bench/torch_transformer_benchmark.py`.

Do not add wrappers or "fixed" versions of these here. Re-export only.
"""

from bench.torch_transformer_benchmark import (
    # ---- model ----
    TransformerConfig,
    BaselineTransformer,
    BaselineTransformerBlock,
    BaselineSelfAttention,
    # ---- setup helpers ----
    copy_model_weights,
    resolve_device,
    resolve_dtype,
    maybe_compile,
    # ---- data generation ----
    generate_random_case,
    # ---- accuracy ----
    AccuracyResult,
    compare_outputs,
    run_accuracy_tests,
    # ---- timing ----
    TimingResult,
    percentile,
    warmup_model,
    benchmark_once,
    benchmark_models,
)

__all__ = [
    "TransformerConfig",
    "BaselineTransformer",
    "BaselineTransformerBlock",
    "BaselineSelfAttention",
    "copy_model_weights",
    "resolve_device",
    "resolve_dtype",
    "maybe_compile",
    "generate_random_case",
    "AccuracyResult",
    "compare_outputs",
    "run_accuracy_tests",
    "TimingResult",
    "percentile",
    "warmup_model",
    "benchmark_once",
    "benchmark_models",
]
