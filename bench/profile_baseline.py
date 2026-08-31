#!/usr/bin/env python3
"""Rung 0: profile BaselineTransformer at three fixed shapes.

Answers one question before any kernel gets written: where is the time going,
and is the GPU actually busy? See docs/CONTRIBUTING.md section 9 — do this
before writing any optimized strategy.

For each shape this reports wall ms/iter, GPU ms/iter (via torch.cuda.Event,
which lives on the CUDA stream and is independent of the profiler backend),
GPU busy % (gpu/wall), CUDA kernel-launch count per forward, and peak VRAM.
Chrome traces are exported to logs/trace_{small,medium,large}_baseline.json.

Imports BaselineTransformer through src/baseline.py — never touches the
organizers' file directly (see CLAUDE.md rule 1).

Note on kernel counts: on this environment (WSL2), CUPTI does not populate
device-side kernel completion events in the profiler trace, so the exported
chrome trace has no "kernel" category entries and key_averages() reports no
Self CUDA time. The kernel count below instead counts `cudaLaunchKernel` calls
captured on the CPU-side "cuda_runtime" track, which is a 1:1 proxy for the
number of GPU kernels launched per forward. If you run this on a machine
where CUPTI works, the exported traces will additionally show per-kernel
device timings on the CUDA track.

Usage:
    python bench/profile_baseline.py
"""

from __future__ import annotations

import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch.profiler import ProfilerActivity, profile

from src.baseline import (
    BaselineTransformer,
    TransformerConfig,
    generate_random_case,
    resolve_device,
)

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"

NUM_LAYERS = 6
WARMUP_ITERS = 10
TIMED_ITERS = 30
PROFILED_ITERS = 3
SEED = 0

# (name, batch_size, seq_len, d_model, num_heads)
SHAPES = [
    ("small", 8, 128, 512, 8),
    ("medium", 8, 512, 512, 8),
    ("large", 4, 2048, 512, 8),
]


@dataclass
class ShapeResult:
    name: str
    config: TransformerConfig
    wall_ms: Optional[float] = None
    gpu_ms: Optional[float] = None
    gpu_busy_pct: Optional[float] = None
    kernel_count: Optional[int] = None
    peak_vram_mb: Optional[float] = None
    oom: bool = False
    error: Optional[str] = None


def build_config(batch_size: int, seq_len: int, d_model: int, num_heads: int) -> TransformerConfig:
    config = TransformerConfig(
        batch_size=batch_size,
        seq_len=seq_len,
        d_model=d_model,
        num_heads=num_heads,
        ffn_dim=4 * d_model,
        num_layers=NUM_LAYERS,
        causal=False,
    )
    config.validate()
    return config


def count_kernel_launches(trace_path: Path) -> int:
    """Count cudaLaunchKernel calls recorded in an exported chrome trace."""
    with open(trace_path) as f:
        trace = json.load(f)
    events = trace["traceEvents"] if isinstance(trace, dict) else trace
    return sum(
        1
        for e in events
        if e.get("cat") == "cuda_runtime" and e.get("name") == "cudaLaunchKernel"
    )


def profile_shape(
    name: str, batch_size: int, seq_len: int, d_model: int, num_heads: int, device: torch.device
) -> ShapeResult:
    config = build_config(batch_size, seq_len, d_model, num_heads)
    result = ShapeResult(name=name, config=config)
    dtype = torch.float32

    try:
        model = BaselineTransformer(config).to(device=device, dtype=dtype).eval()
        x, _ = generate_random_case(
            config, device, dtype, seed=SEED, padding_ratio=0.0, input_scale=1.0
        )

        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

        with torch.no_grad():
            for _ in range(WARMUP_ITERS):
                model(x, None)
            if device.type == "cuda":
                torch.cuda.synchronize(device)

            if device.type == "cuda":
                start_evt = torch.cuda.Event(enable_timing=True)
                end_evt = torch.cuda.Event(enable_timing=True)
                wall_start = time.perf_counter()
                start_evt.record()
                for _ in range(TIMED_ITERS):
                    model(x, None)
                end_evt.record()
                torch.cuda.synchronize(device)
                wall_end = time.perf_counter()

                gpu_ms = start_evt.elapsed_time(end_evt) / TIMED_ITERS
                wall_ms = (wall_end - wall_start) * 1000.0 / TIMED_ITERS
                result.gpu_ms = gpu_ms
                result.wall_ms = wall_ms
                result.gpu_busy_pct = 100.0 * gpu_ms / wall_ms if wall_ms > 0 else None
                result.peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
            else:
                wall_start = time.perf_counter()
                for _ in range(TIMED_ITERS):
                    model(x, None)
                wall_end = time.perf_counter()
                result.wall_ms = (wall_end - wall_start) * 1000.0 / TIMED_ITERS

            activities = [ProfilerActivity.CPU]
            if device.type == "cuda":
                activities.append(ProfilerActivity.CUDA)
            with profile(activities=activities) as prof:
                for _ in range(PROFILED_ITERS):
                    model(x, None)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)

            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            trace_path = LOGS_DIR / f"trace_{name}_baseline.json"
            prof.export_chrome_trace(str(trace_path))

            if device.type == "cuda":
                total_launches = count_kernel_launches(trace_path)
                result.kernel_count = round(total_launches / PROFILED_ITERS)

    except torch.cuda.OutOfMemoryError as e:
        result.oom = True
        result.error = str(e).splitlines()[0]
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            result.oom = True
            result.error = str(e).splitlines()[0]
        else:
            raise
    finally:
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return result


def fmt(value: Optional[float], spec: str = ".3f") -> str:
    return format(value, spec) if value is not None else "n/a"


def print_summary(results: list[ShapeResult]) -> None:
    headers = ["shape", "B", "S", "d", "wall ms", "gpu ms", "gpu busy %", "kernels", "peak VRAM MB"]
    rows = []
    for r in results:
        if r.oom:
            rows.append([r.name, str(r.config.batch_size), str(r.config.seq_len), str(r.config.d_model), "OOM", "OOM", "OOM", "OOM", "OOM"])
            continue
        rows.append(
            [
                r.name,
                str(r.config.batch_size),
                str(r.config.seq_len),
                str(r.config.d_model),
                fmt(r.wall_ms, ".3f"),
                fmt(r.gpu_ms, ".3f"),
                fmt(r.gpu_busy_pct, ".1f"),
                str(r.kernel_count) if r.kernel_count is not None else "n/a",
                fmt(r.peak_vram_mb, ".1f"),
            ]
        )

    widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)]
    line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
    sep = "-+-".join("-" * w for w in widths)
    print(line)
    print(sep)
    for row in rows:
        print(" | ".join(c.ljust(w) for c, w in zip(row, widths)))

    for r in results:
        if r.oom:
            print(f"\n[{r.name}] OOM: {r.error}")


def main() -> None:
    device = resolve_device("auto")

    # Match the organizers' defaults (torch_transformer_benchmark.py main()):
    # matmul_precision="high", allow_tf32=True. Without these, fp32 matmuls run
    # at ~5.7 TFLOPS instead of ~11.0 on this card, and the numbers here would
    # not be comparable to what bench/sweep.py measures.
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print(f"device: {device}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(device)}")
    print(f"matmul_precision: {torch.get_float32_matmul_precision()}", end="")
    if device.type == "cuda":
        print(f", allow_tf32(matmul/cudnn): {torch.backends.cuda.matmul.allow_tf32}/{torch.backends.cudnn.allow_tf32}")
    else:
        print()
    print(f"layers: {NUM_LAYERS}, dtype: float32, causal: False, padding: none\n")

    results = []
    for name, batch_size, seq_len, d_model, num_heads in SHAPES:
        print(f"profiling {name} (B={batch_size}, S={seq_len}, d={d_model}, H={num_heads})...")
        result = profile_shape(name, batch_size, seq_len, d_model, num_heads, device)
        results.append(result)
        if result.oom:
            print(f"  -> OOM: {result.error}")
        else:
            print(f"  -> wall {result.wall_ms:.3f} ms, gpu {fmt(result.gpu_ms)} ms, busy {fmt(result.gpu_busy_pct, '.1f')}%")

    print("\n=== summary ===")
    print_summary(results)


if __name__ == "__main__":
    main()
