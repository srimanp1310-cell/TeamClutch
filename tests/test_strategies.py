"""The accuracy oracle: every registered strategy vs the baseline, on CPU.

This is the test Person A runs before every push, on either machine. It exists
to answer one question precisely: *if this strategy is wrong, which branch is
wrong?* So the matrix deliberately crosses the two things that break first --
padding and causal masking -- with shapes chosen to break assumptions:

    (2, 16,  64, 4)   small and tidy
    (1, 33,  96, 3)   S=33 and d=96 are not powers of two, heads=3 does not
                      divide evenly into a warp, head_dim=32
    (3, 64, 128, 8)   batch > 1 with a wider model

and on failure it prints the exact fields the sprint plan says to paste at a
checkpoint: max_abs, max_rel, worst_index, both values at the worst element,
and which output feature dims failed.

`float32` on CPU is the contract. `float16`/`bfloat16` are only asserted where
there is a GPU: CPU reduced-precision kernels are a different implementation
with different rounding, so a failure there tells us nothing about the GPU path
we are actually shipping. bf16-on-CPU is still *run*, as xfail(strict=False),
so we see the number without gating the build on it.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import pytest
import torch

from src.baseline import (
    AccuracyResult,
    BaselineTransformer,
    TransformerConfig,
    compare_outputs,
    copy_model_weights,
    generate_random_case,
    resolve_dtype,
)
from src.strategies import (
    STRATEGIES, get_strategy, requires_cuda, supported_dtypes, supports_dtype,
)

# The tolerance we target: the torch script's defaults, which are stricter than
# the problem statement PDF's 0.002 / 0.02.
ATOL = 0.001
RTOL = 0.01

SEED = 1234

#: (batch, seq_len, d_model, heads)
SHAPES: Tuple[Tuple[int, int, int, int], ...] = (
    (2, 16, 64, 4),
    (1, 33, 96, 3),
    (3, 64, 128, 8),
)

STRATEGY_NAMES = sorted(STRATEGIES)
CUDA = torch.cuda.is_available()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def build_config(
    batch: int, seq_len: int, d_model: int, heads: int,
    layers: int = 2, causal: bool = False, ffn_dim: Optional[int] = None,
) -> TransformerConfig:
    config = TransformerConfig(
        batch_size=batch, seq_len=seq_len, d_model=d_model, num_heads=heads,
        ffn_dim=ffn_dim if ffn_dim is not None else 2 * d_model,
        num_layers=layers, causal=causal,
    )
    config.validate()
    return config


def compare_against_baseline(
    strategy_name: str,
    config: TransformerConfig,
    padding_ratio: float,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
    seed: int = SEED,
) -> AccuracyResult:
    """Run baseline and strategy on identical weights and input; compare.

    Mirrors the organizers' setup order exactly: construct, copy weights, move
    to device/dtype, eval. Weights are copied with `strict=True` -- a strategy
    that renamed a parameter fails here rather than silently comparing two
    differently-initialised models.
    """
    device = device or torch.device("cuda" if CUDA else "cpu")

    baseline = BaselineTransformer(config)
    optimized = get_strategy(strategy_name)(config)
    copy_model_weights(baseline, optimized, strict=True)

    baseline = baseline.to(device=device, dtype=dtype).eval()
    optimized = optimized.to(device=device, dtype=dtype).eval()

    x, valid_mask = generate_random_case(
        config=config, device=device, dtype=dtype, seed=seed,
        padding_ratio=padding_ratio, input_scale=1.0,
    )

    with torch.inference_mode():
        reference = baseline(x, valid_mask)
        candidate = optimized(x, valid_mask)

    assert candidate.shape == reference.shape, (
        f"{strategy_name}: shape mismatch, baseline={tuple(reference.shape)}, "
        f"optimized={tuple(candidate.shape)}"
    )
    return compare_outputs(reference, candidate, rtol=RTOL, atol=ATOL)


def describe(result: AccuracyResult, strategy_name: str, context: str) -> str:
    """The paste-ready failure report. These are the fields that identify
    *which* branch is wrong, not just that something is."""
    preview = result.failed_feature_dims[:16]
    suffix = "..." if len(result.failed_feature_dims) > len(preview) else ""

    # A NaN max_abs_err reads as a broken metric; it is not. It means the
    # optimized output itself contains non-finite values, and there is one
    # overwhelmingly common cause, so say it here rather than in a wiki.
    hint = ""
    if not math.isfinite(result.max_abs_error):
        hint = (
            "\n  NOTE                 : the optimized output contains NaN/Inf."
            "\n                         Usual cause: a mask leaves some row with"
            " every position masked,"
            "\n                         so softmax over all -inf is NaN. Check the"
            " causal diagonal"
            "\n                         (triu must be diagonal=1, not 0) and the"
            " mask polarity"
            "\n                         (valid_token_mask is True = KEEP)."
        )

    return (
        f"\n{strategy_name} FAILED at {context}"
        f"\n  criterion            : abs_err <= {ATOL:g} OR rel_err <= {RTOL:.2%}"
        f"\n  failed               : {result.failed_elements}/{result.total_elements}"
        f"\n  max_abs_err          : {result.max_abs_error:.6g}"
        f"\n  max_rel_err          : {result.max_relative_error:.6g}"
        f"\n  mean_abs_err         : {result.mean_abs_error:.6g}"
        f"\n  worst_index          : {result.worst_index}"
        f"\n  baseline   @ worst   : {result.reference_at_worst:.8g}"
        f"\n  optimized  @ worst   : {result.optimized_at_worst:.8g}"
        f"\n  failed feature dims  : {preview}{suffix}"
        + hint
    )


def skip_if_cuda_only(strategy_name: str) -> None:
    if requires_cuda(strategy_name) and not CUDA:
        pytest.skip(
            f"{strategy_name} declares REQUIRES_CUDA = True and this machine has "
            "no GPU — correctness for it must be verified on Person A's box"
        )


def skip_if_dtype_unsupported(strategy_name: str, dtype_name: str) -> None:
    """Honour a strategy's `SUPPORTED_DTYPES` declaration.

    A strategy may be correctly implemented and still be unable to meet the
    tolerance in a given precision — bf16 against this reference is the worked
    example (docs/INTERFACE.md §5.1). Declaring the dtype unsupported is not a
    way to hide a bug: `src/dispatch.py` reads the same declaration and will
    never select the strategy for that dtype, so the untested path is also
    unreachable in production.
    """
    if not supports_dtype(strategy_name, dtype_name):
        pytest.skip(
            f"{strategy_name} declares SUPPORTED_DTYPES = "
            f"{sorted(supported_dtypes(strategy_name))} and does not claim "
            f"{dtype_name}; dispatch will never select it for that dtype"
        )


# ---------------------------------------------------------------------------
# the main matrix: strategy x shape x padding x causal, float32
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strategy_name", STRATEGY_NAMES)
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"B{s[0]}_S{s[1]}_d{s[2]}_H{s[3]}")
@pytest.mark.parametrize("padding_ratio", (0.0, 0.3), ids=("nopad", "pad0.3"))
@pytest.mark.parametrize("causal", (False, True), ids=("noncausal", "causal"))
def test_strategy_matches_baseline(strategy_name, shape, padding_ratio, causal):
    skip_if_cuda_only(strategy_name)

    batch, seq_len, d_model, heads = shape
    config = build_config(batch, seq_len, d_model, heads, causal=causal)
    result = compare_against_baseline(strategy_name, config, padding_ratio)

    context = (
        f"B={batch} S={seq_len} d={d_model} H={heads} "
        f"causal={causal} padding_ratio={padding_ratio} dtype=float32"
    )
    assert result.passed, describe(result, strategy_name, context)


# ---------------------------------------------------------------------------
# the control
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"B{s[0]}_S{s[1]}_d{s[2]}_H{s[3]}")
@pytest.mark.parametrize("padding_ratio", (0.0, 0.3), ids=("nopad", "pad0.3"))
@pytest.mark.parametrize("causal", (False, True), ids=("noncausal", "causal"))
def test_baseline_control_is_bit_identical(shape, padding_ratio, causal):
    """The control is a copy of the reference, so "close enough" is not the bar.

    Anything other than exactly 0.0 here means the harness itself introduces a
    difference -- different weights, a stray dropout, a non-deterministic op --
    and every accuracy number we report would inherit it.
    """
    batch, seq_len, d_model, heads = shape
    config = build_config(batch, seq_len, d_model, heads, causal=causal)
    result = compare_against_baseline("baseline", config, padding_ratio)

    assert result.max_abs_error == 0.0, describe(
        result, "baseline",
        f"B={batch} S={seq_len} d={d_model} H={heads} causal={causal} "
        f"padding_ratio={padding_ratio} — the control must be bit-identical",
    )
    assert result.failed_elements == 0


# ---------------------------------------------------------------------------
# edge cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strategy_name", STRATEGY_NAMES)
def test_single_token_sequence(strategy_name):
    """S=1: attention is a 1x1 softmax, and a causal mask is entirely empty.

    Kernels that assume S > 1, or that special-case the diagonal, break here.
    """
    skip_if_cuda_only(strategy_name)
    for causal in (False, True):
        config = build_config(1, 1, 64, 4, causal=causal)
        result = compare_against_baseline(strategy_name, config, padding_ratio=0.0)
        assert result.passed, describe(result, strategy_name, f"S=1 causal={causal}")


@pytest.mark.parametrize("strategy_name", STRATEGY_NAMES)
def test_non_power_of_two_model_dim(strategy_name):
    """d_model=96, heads=3 -> head_dim=32. Tiling that assumes a power-of-two
    d_model, or that heads is a multiple of 4, fails here and nowhere else."""
    skip_if_cuda_only(strategy_name)
    config = build_config(2, 40, 96, 3)
    result = compare_against_baseline(strategy_name, config, padding_ratio=0.3)
    assert result.passed, describe(result, strategy_name, "d=96 H=3 head_dim=32")


@pytest.mark.parametrize("strategy_name", STRATEGY_NAMES)
def test_fully_padded_except_one_token(strategy_name):
    """The extreme of the padding branch: rows where almost everything is
    masked out. Zeroing the wrong axis shows up immediately."""
    skip_if_cuda_only(strategy_name)
    config = build_config(2, 32, 64, 4)
    result = compare_against_baseline(strategy_name, config, padding_ratio=0.97)
    assert result.passed, describe(result, strategy_name, "padding_ratio=0.97")


@pytest.mark.parametrize("strategy_name", STRATEGY_NAMES)
def test_deeper_stack_accumulates_no_extra_error(strategy_name):
    """Error compounds with depth. 6 layers is the config we actually report."""
    skip_if_cuda_only(strategy_name)
    config = build_config(2, 32, 64, 4, layers=6)
    result = compare_against_baseline(strategy_name, config, padding_ratio=0.3)
    assert result.passed, describe(result, strategy_name, "layers=6")


# ---------------------------------------------------------------------------
# reduced precision
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not CUDA, reason="fp16/bf16 are only asserted where we ship them: on the GPU")
@pytest.mark.parametrize("strategy_name", STRATEGY_NAMES)
@pytest.mark.parametrize("dtype_name", ("float16", "bfloat16"))
def test_reduced_precision_on_gpu(strategy_name, dtype_name):
    skip_if_dtype_unsupported(strategy_name, dtype_name)
    config = build_config(2, 64, 128, 8)
    result = compare_against_baseline(
        strategy_name, config, padding_ratio=0.3, dtype=resolve_dtype(dtype_name)
    )
    assert result.passed, describe(result, strategy_name, f"dtype={dtype_name}")


@pytest.mark.xfail(
    not CUDA,
    strict=False,
    reason="CPU bf16 kernels round differently from the CUDA ones we ship; "
           "run for visibility, not as a gate",
)
@pytest.mark.parametrize("strategy_name", STRATEGY_NAMES)
def test_bfloat16_on_cpu_for_visibility(strategy_name):
    skip_if_cuda_only(strategy_name)
    skip_if_dtype_unsupported(strategy_name, "bfloat16")
    config = build_config(2, 32, 64, 4)
    result = compare_against_baseline(
        strategy_name, config, padding_ratio=0.0, dtype=torch.bfloat16
    )
    assert result.passed, describe(result, strategy_name, "dtype=bfloat16 on CPU")


def test_declared_dtype_support_is_a_recognised_set():
    """A typo in SUPPORTED_DTYPES would silently exclude a dtype from testing."""
    from src.strategies import ALL_DTYPES

    for name in STRATEGY_NAMES:
        declared = supported_dtypes(name)
        assert declared <= ALL_DTYPES, (
            f"{name} declares unknown dtypes: {sorted(declared - ALL_DTYPES)}"
        )
        assert declared, f"{name} declares an empty SUPPORTED_DTYPES"
        assert "float32" in declared, (
            f"{name} does not claim float32. Every strategy must be correct in "
            "fp32 — that is the dtype the accuracy oracle runs in on CPU."
        )


# ---------------------------------------------------------------------------
# the oracle must be able to fail
# ---------------------------------------------------------------------------

def test_oracle_catches_a_deliberately_wrong_strategy():
    """A test suite that cannot fail proves nothing.

    Scaling the output by (1 + k) gives every element `rel_err == k` and
    `abs_err == k * |ref|`. The pass rule is an OR, so k must clear *both*
    thresholds to be caught: k = 0.02 is just past rtol = 0.01, and past
    atol = 0.001 everywhere |ref| > 0.05, which after the final LayerNorm is
    almost everywhere. That makes this a sensitivity check on the oracle, not
    just a smoke test -- a 2% error is the smallest thing it must never miss.
    """

    class SlightlyWrong(BaselineTransformer):
        def forward(self, x, valid_token_mask=None):
            return super().forward(x, valid_token_mask) * 1.02

    STRATEGIES["__slightly_wrong__"] = SlightlyWrong
    try:
        config = build_config(2, 32, 64, 4)
        result = compare_against_baseline("__slightly_wrong__", config, padding_ratio=0.0)
    finally:
        STRATEGIES.pop("__slightly_wrong__", None)

    assert not result.passed
    assert result.max_abs_error > 0
    report = describe(result, "__slightly_wrong__", "sanity")
    for field in ("worst_index", "baseline   @ worst", "optimized  @ worst"):
        assert field in report


def test_strategy_with_renamed_parameters_is_rejected():
    """copy_model_weights(strict=True) is what proves both models hold the same
    weights. A strategy that renames a parameter must fail loudly here, not
    quietly benchmark a differently-initialised model."""
    import torch.nn as nn

    class RenamedParameters(BaselineTransformer):
        def __init__(self, config):
            super().__init__(config)
            self.extra_projection = nn.Linear(config.d_model, config.d_model)

    STRATEGIES["__renamed__"] = RenamedParameters
    try:
        # The strategy declares a parameter the baseline's state_dict has no
        # entry for, so strict loading reports it as missing.
        with pytest.raises(RuntimeError, match="Missing key|Unexpected key"):
            compare_against_baseline("__renamed__", build_config(1, 8, 32, 4), 0.0)
    finally:
        STRATEGIES.pop("__renamed__", None)


# ---------------------------------------------------------------------------
# run_official injection
# ---------------------------------------------------------------------------

def test_run_official_injects_our_class_into_the_organizers_module():
    import bench.run_official as runner
    import bench.torch_transformer_benchmark as official

    original = official.UserOptimizedTransformer
    try:
        entry_point, origin = runner.resolve_entry_point()
        assert issubclass(entry_point, BaselineTransformer)
        assert origin

        official.UserOptimizedTransformer = entry_point
        assert official.UserOptimizedTransformer is entry_point
        # The injection must not be the organizers' own placeholder, or we would
        # be measuring their stub against itself and calling it our result.
        assert official.UserOptimizedTransformer is not original
    finally:
        official.UserOptimizedTransformer = original
