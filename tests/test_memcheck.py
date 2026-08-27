"""Task 1 acceptance: the memory estimate must be monotonic, quadratic in S,
land in the expected band for the known-good config, and refuse a tiny budget.
"""

from __future__ import annotations

import torch

from src.baseline import TransformerConfig
from src.memcheck import (
    available_device_memory_bytes,
    check_fits,
    dtype_nbytes,
    estimate_baseline_peak_bytes,
    format_gib,
)

GIB = 1024**3


def cfg(batch=8, seq_len=512, d_model=512, heads=8, ffn_dim=None, layers=6, causal=False):
    return TransformerConfig(
        batch_size=batch,
        seq_len=seq_len,
        d_model=d_model,
        num_heads=heads,
        ffn_dim=ffn_dim if ffn_dim is not None else 4 * d_model,
        num_layers=layers,
        causal=causal,
    )


def est(config, dtype=torch.float32):
    return estimate_baseline_peak_bytes(config, dtype)


# --------------------------------------------------------------------------
# monotonicity
# --------------------------------------------------------------------------

def test_monotonic_in_batch():
    values = [est(cfg(batch=b)) for b in (1, 2, 8, 32)]
    assert values == sorted(values)
    assert values[0] < values[-1]


def test_monotonic_in_seq_len():
    values = [est(cfg(seq_len=s)) for s in (128, 512, 1024, 2048)]
    assert values == sorted(values)
    assert values[0] < values[-1]


def test_monotonic_in_d_model():
    values = [est(cfg(d_model=d)) for d in (256, 512, 1024)]
    assert values == sorted(values)
    assert values[0] < values[-1]


def test_monotonic_in_layers():
    values = [est(cfg(layers=layers)) for layers in (1, 2, 4, 6)]
    assert values == sorted(values)


# --------------------------------------------------------------------------
# the S^2 term
# --------------------------------------------------------------------------

def test_score_term_is_quadratic_in_seq_len():
    """Doubling S must multiply the [B,H,S,S] term by exactly 4.

    Isolated by differencing two estimates that share every other axis: the
    model-weight term is independent of S and the token workspace is linear in
    S, so the quadratic growth has to be recovered from the score term itself.
    """
    b, h, s, e = 8, 8, 512, dtype_nbytes(torch.float32)

    def score_term(seq_len):
        return b * h * seq_len * seq_len * (2 * e + 8)

    assert score_term(2 * s) == 4 * score_term(s)

    # ...and the full estimate must grow by at least that much.
    small, large = est(cfg(seq_len=s)), est(cfg(seq_len=2 * s))
    assert large - small >= score_term(2 * s) - score_term(s)


def test_seq_len_dominates_at_long_sequences():
    """At S=4096 the score matrix must be the overwhelming term."""
    config = cfg(seq_len=4096)
    total = est(config)
    e = dtype_nbytes(torch.float32)
    score_bytes = config.batch_size * config.num_heads * 4096 * 4096 * (2 * e + 8)
    assert score_bytes / total > 0.9


# --------------------------------------------------------------------------
# absolute magnitudes (the numbers the team notes rely on)
# --------------------------------------------------------------------------

def test_known_small_config_lands_in_expected_band():
    """B=8, S=128, H=8, d=512, L=6, fp32 -> between 0.15 and 0.6 GiB."""
    gib = format_gib(est(cfg(batch=8, seq_len=128, d_model=512, heads=8, layers=6)))
    assert 0.15 < gib < 0.6, gib


def test_s4096_exceeds_four_gib():
    """The team notes say S=4096 will OOM on the 6 GB card. Confirm the
    estimate agrees, or the sweep would happily walk into it."""
    gib = format_gib(est(cfg(batch=8, seq_len=4096, d_model=512, heads=8, layers=6)))
    assert gib > 4.0, gib


def test_half_precision_is_cheaper_than_fp32():
    """fp16/bf16 shrink the weights and activations but NOT the 8 bytes of
    fp32 softmax intermediates per score element, so the saving is real but
    strictly less than half."""
    fp32 = est(cfg(), torch.float32)
    bf16 = est(cfg(), torch.bfloat16)
    assert bf16 < fp32
    assert bf16 > fp32 / 2


def test_causal_only_adds_the_mask():
    """A causal mask is one [S,S] bool buffer -- it must not change the order
    of magnitude, or the estimate is double-counting."""
    plain = est(cfg(causal=False))
    causal = est(cfg(causal=True))
    assert causal - plain == 512 * 512


# --------------------------------------------------------------------------
# check_fits
# --------------------------------------------------------------------------

def test_check_fits_refuses_a_tiny_hard_cap():
    fits, reason = check_fits(
        cfg(), torch.float32, torch.device("cpu"), hard_cap_gib=0.01
    )
    assert fits is False
    assert "will not fit" in reason
    # The cap must be quoted back at a readable scale: 0.01 GiB as "0.00 GiB"
    # would discard the one number the reader of a SKIPPED row needs.
    assert "10.2 MiB" in reason, reason


def test_check_fits_accepts_a_tiny_config():
    fits, reason = check_fits(
        cfg(batch=1, seq_len=16, d_model=64, heads=4, layers=1),
        torch.float32,
        torch.device("cpu"),
    )
    assert fits is True
    assert "GiB" in reason


def test_check_fits_honours_memory_fraction():
    """Same config, same machine: a stricter fraction can only tighten."""
    config = cfg(batch=8, seq_len=2048)
    generous, _ = check_fits(config, torch.float32, torch.device("cpu"), memory_fraction=1.0)
    strict, _ = check_fits(config, torch.float32, torch.device("cpu"), memory_fraction=0.01)
    assert generous >= strict  # bool ordering: True >= False


def test_available_memory_never_raises_and_is_positive():
    """Runs on a Mac with no CUDA: must degrade, not crash."""
    for device in (torch.device("cpu"), torch.device("cuda")):
        value = available_device_memory_bytes(device)
        assert isinstance(value, int) and value > 0


def test_memcheck_cli_runs_without_a_gpu():
    from src.memcheck import main

    assert main(["--batch", "1", "--seq-len", "16", "--d-model", "64", "--heads", "4"]) == 0
    assert main(["--batch", "8", "--seq-len", "4096", "--hard-cap-gib", "0.01"]) == 1
