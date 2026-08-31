"""Memory pre-check: will the *baseline* fit before we try to run it?

Why this exists
---------------
The sweep runs many configs unattended on a 6 GB laptop card. Without a
pre-check, one oversized config (S=4096 is the known cliff) raises
`torch.cuda.OutOfMemoryError` half-way through and takes the rest of the sweep
with it. With it, that config is recorded as `SKIPPED: <reason>` and the sweep
continues — a skipped row is data, a dead sweep is not.

The estimate is deliberately for the **baseline**, not for our optimized path.
The baseline is the one that materializes the full `[B, H, S, S]` score matrix,
so it is always the binding constraint; if the baseline fits, ours does too.
That asymmetry is itself a result — see the VRAM-ceiling figure, where the
baseline OOMs at a sequence length our implementation still handles.

The math is ported unchanged from the organizers' TensorFlow benchmark
(`estimate_baseline_peak_bytes` in `bench/tensorflow_transformer_benchmark.py`),
so the two benchmarks agree about which configs are impossible. Their comments
are kept. It is a conservative safety guard, not an allocator trace.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from typing import Optional, Tuple

import torch

from src.baseline import TransformerConfig, resolve_dtype

__all__ = [
    "dtype_nbytes",
    "estimate_baseline_peak_bytes",
    "available_device_memory_bytes",
    "check_fits",
    "format_gib",
    "human_bytes",
]

_FALLBACK_MEMORY_BYTES = 8 * 1024**3


def dtype_nbytes(dtype: torch.dtype) -> int:
    """Bytes per element of a torch dtype."""
    itemsize = getattr(dtype, "itemsize", None)
    if itemsize is not None:
        return int(itemsize)
    return torch.empty((), dtype=dtype).element_size()


def format_gib(value_bytes: float) -> float:
    return value_bytes / 1024**3


def human_bytes(value_bytes: float) -> str:
    """Size with a unit that keeps significant digits.

    The reason string goes verbatim into the CSV `notes` of a SKIPPED row, so
    "0.00 GiB" would throw away the very number someone is reading the row for.
    """
    gib = value_bytes / 1024**3
    if gib >= 0.1:
        return f"{gib:.2f} GiB"
    mib = value_bytes / 1024**2
    if mib >= 0.1:
        return f"{mib:.1f} MiB"
    return f"{value_bytes:,.0f} B"


def estimate_baseline_peak_bytes(
    config: TransformerConfig,
    dtype: torch.dtype,
) -> int:
    """
    Conservative estimate for this explicit-attention baseline.

    The dominant term is [B, H, S, S]. Because reference softmax is evaluated
    in fp32, several score/probability buffers may overlap. The estimate is not
    an exact allocator trace; it is a safety guard against obviously impossible
    configurations.
    """
    b = config.batch_size
    s = config.seq_len
    d = config.d_model
    h = config.num_heads
    f = config.ffn_dim
    e = dtype_nbytes(dtype)

    # Two models are resident simultaneously. Approximate parameters per layer:
    # Q/K/V/out = 4*d*d; FFN = 2*d*f; norms/biases are lower order.
    params_per_model = config.num_layers * (4 * d * d + 2 * d * f) + 2 * d
    model_bytes = 2 * params_per_model * e

    # Input/output/residual/QKV/FFN temporaries for one active model.
    token_elements = b * s * d
    ffn_elements = b * s * f
    token_workspace = (10 * token_elements + 2 * ffn_elements) * e

    # Explicit scores plus fp32 softmax intermediates and casted probabilities.
    # This is the term that grows as S^2 and decides every OOM in practice.
    score_elements = b * h * s * s
    attention_workspace = score_elements * (2 * e + 8)

    mask_bytes = b * s + (s * s if config.causal else 0)
    return int(model_bytes + token_workspace + attention_workspace + mask_bytes)


def _system_available_memory_bytes() -> int:
    """Free system RAM, or a conservative 8 GiB if the OS will not say.

    macOS does not expose `SC_AVPHYS_PAGES`, so this returns the fallback there.
    That is fine: on a CPU-only machine the number that matters is the *estimate*,
    which is device-independent. The real free-VRAM figure comes from the GPU
    machine via the CUDA branch below.
    """
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        available_pages = os.sysconf("SC_AVPHYS_PAGES")
        return int(page_size * available_pages)
    except (AttributeError, ValueError, OSError):
        return _FALLBACK_MEMORY_BYTES


def _nvidia_smi_free_memory_bytes(gpu_index: int = 0) -> Optional[int]:
    """Free VRAM per `nvidia-smi`, or None when the tool is unavailable."""
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    try:
        completed = subprocess.run(
            [
                executable,
                f"--id={gpu_index}",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        first_line = completed.stdout.strip().splitlines()[0]
        return int(first_line.strip()) * 1024**2
    except (subprocess.SubprocessError, ValueError, IndexError, OSError):
        return None


def available_device_memory_bytes(device: torch.device) -> int:
    """Free memory on `device`: VRAM for cuda, system RAM otherwise."""
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            free, _total = torch.cuda.mem_get_info(device)
            return int(free)
        except (RuntimeError, AssertionError, TypeError):
            pass  # fall through to nvidia-smi, then to the constant
        smi = _nvidia_smi_free_memory_bytes(device.index or 0)
        if smi is not None:
            return smi
        return _FALLBACK_MEMORY_BYTES
    return _system_available_memory_bytes()


def check_fits(
    config: TransformerConfig,
    dtype: torch.dtype,
    device: torch.device,
    memory_fraction: float = 0.75,
    hard_cap_gib: Optional[float] = None,
) -> Tuple[bool, str]:
    """Decide whether to attempt this config.

    The budget is `memory_fraction` of currently-free memory, further limited by
    `hard_cap_gib` when given. `memory_fraction` defaults to 0.75 because the
    estimate ignores allocator fragmentation, the CUDA context (~300 MB) and
    anything else already on the card.

    Returns `(fits, reason)`. The reason is written verbatim into the CSV
    `notes` column of a SKIPPED row, so it carries real numbers.
    """
    estimated = estimate_baseline_peak_bytes(config, dtype)
    free = available_device_memory_bytes(device)

    budget = int(free * memory_fraction)
    cap_applied = False
    if hard_cap_gib is not None:
        cap_bytes = int(hard_cap_gib * 1024**3)
        if cap_bytes < budget:
            budget = cap_bytes
            cap_applied = True

    detail = (
        f"estimated baseline peak {human_bytes(estimated)} vs budget "
        f"{human_bytes(budget)} "
        f"({memory_fraction:.0%} of {human_bytes(free)} free on {device}"
        + (f", capped at {human_bytes(hard_cap_gib * 1024**3)}" if cap_applied else "")
        + ")"
    )

    if estimated > budget:
        return False, f"{detail} — will not fit"
    return True, detail


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.memcheck",
        description="Estimate the baseline Transformer's peak memory and say whether it fits.",
    )
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument(
        "--ffn-dim", type=int, default=0, help="0 means 4 * d_model"
    )
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="float32"
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--memory-fraction", type=float, default=0.75)
    parser.add_argument(
        "--hard-cap-gib",
        type=float,
        default=None,
        help="additionally cap the budget at this many GiB",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    config = TransformerConfig(
        batch_size=args.batch,
        seq_len=args.seq_len,
        d_model=args.d_model,
        num_heads=args.heads,
        ffn_dim=args.ffn_dim or 4 * args.d_model,
        num_layers=args.layers,
        causal=args.causal,
    )
    config.validate()

    dtype = resolve_dtype(args.dtype)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    estimated = estimate_baseline_peak_bytes(config, dtype)
    fits, reason = check_fits(
        config,
        dtype,
        device,
        memory_fraction=args.memory_fraction,
        hard_cap_gib=args.hard_cap_gib,
    )

    print(config)
    print(f"dtype={dtype}, device={device}")
    print(f"estimated baseline peak : {format_gib(estimated):.3f} GiB ({estimated:,} bytes)")
    print(f"free memory             : {format_gib(available_device_memory_bytes(device)):.3f} GiB")
    print(f"verdict                 : {'FITS' if fits else 'WILL NOT FIT'}")
    print(f"reason                  : {reason}")
    return 0 if fits else 1


if __name__ == "__main__":
    raise SystemExit(main())
