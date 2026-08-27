"""Generate synthetic Chrome traces so `analysis/trace.py` can be built and
tested before Person A exports a real one.

Run:  python tests/fixtures/make_trace_fixtures.py

The format is what `torch.profiler`'s `export_chrome_trace` writes: a JSON
object with a `traceEvents` list of Chrome Trace Event objects. Only complete
events (`"ph": "X"`) carry a duration, and GPU kernels are distinguished by
their `cat`. Real traces also contain `cpu_op`, `cuda_runtime`, flow arrows
(`"ph": "s"` / `"f"`) and metadata (`"ph": "M"`) records, so the fixtures
include those too — a parser that only ever sees clean input is a parser that
breaks on the first real file.

Fixtures produced:

  trace_halfidle.json         exactly 50% GPU-busy, by construction. The
                              acceptance test for `gpu_busy_fraction`.
  trace_{small,medium,large}_{baseline,optimized}.json
                              the six traces Person A is asked to export. The
                              baselines are launch-bound at small shapes (many
                              short kernels, large gaps) and become compute-bound
                              at large ones; the optimized traces have fewer,
                              longer kernels and less idle — which is the whole
                              argument for fusion, drawn as a picture.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List

# Kernel names shaped like the real ones, so the family classifier is exercised.
KERNELS = {
    "matmul": ("ampere_sgemm_128x64_tn", "volta_sgemm_64x64_nn",
               "cutlass_80_tensorop_bf16_s16816gemm"),
    "softmax": ("at::native::softmax_warp_forward",
                "at::native::cunn_SoftMax_forward"),
    "layernorm": ("at::native::vectorized_layer_norm_kernel",),
    "elementwise": ("at::native::elementwise_kernel",
                    "at::native::vectorized_elementwise_kernel"),
    "masking": ("at::native::masked_fill_kernel",),
}


def _event(name: str, category: str, start_us: float, duration_us: float,
           tid: int = 7) -> dict:
    return {
        "ph": "X", "cat": category, "name": name,
        "pid": 0, "tid": tid,
        "ts": round(start_us, 3), "dur": round(duration_us, 3),
        "args": {"stream": 7 if category == "kernel" else 0},
    }


def _wrap(events: List[dict]) -> dict:
    """Add the non-kernel noise a real trace carries."""
    preamble = [
        {"ph": "M", "pid": 0, "tid": 0, "name": "process_name",
         "args": {"name": "python"}},
        {"ph": "M", "pid": 0, "tid": 7, "name": "thread_name",
         "args": {"name": "stream 7"}},
        {"ph": "i", "cat": "user_annotation", "name": "ProfilerStep#1",
         "pid": 0, "tid": 1, "ts": 0, "s": "g"},
    ]
    # A CPU-side launch record and a flow arrow per kernel, as torch emits.
    extras: List[dict] = []
    for index, event in enumerate(e for e in events if e["cat"] == "kernel"):
        extras.append(_event("cudaLaunchKernel", "cuda_runtime",
                             event["ts"] - 6, 4, tid=1))
        extras.append({"ph": "s", "cat": "ac2g", "name": "ac2g", "pid": 0,
                       "tid": 1, "ts": event["ts"] - 6, "id": index})
    return {
        "schemaVersion": 1,
        "deviceProperties": [{"id": 0, "name": "NVIDIA GeForce RTX 4050 Laptop GPU",
                              "totalGlobalMem": 6979321856}],
        "traceEvents": preamble + extras + events,
    }


def half_idle_trace(n_kernels: int = 20, duration_us: float = 1000.0) -> dict:
    """Exactly 50% busy under the default span definition.

    `gpu_busy_fraction` spans first-kernel-start to last-kernel-end, so the gaps
    must total exactly the busy time: with n kernels there are n-1 gaps, hence
    gap = n*duration / (n-1). Choosing the gap rather than asserting a round
    number keeps the fixture honest about which definition it is testing.
    """
    total_busy = n_kernels * duration_us
    gap = total_busy / (n_kernels - 1)
    events, cursor = [], 0.0
    for index in range(n_kernels):
        events.append(_event(KERNELS["matmul"][index % 3], "kernel", cursor, duration_us))
        cursor += duration_us + gap
    return _wrap(events)


def workload_trace(
    n_launches: int, kernel_us: float, gap_us: float, seed: int,
    families=("layernorm", "matmul", "matmul", "softmax", "masking",
              "matmul", "elementwise"),
) -> dict:
    """One transformer-shaped burst of kernels per layer, repeated."""
    rng = random.Random(seed)
    events, cursor = [], 0.0
    for _ in range(n_launches):
        for family in families:
            duration = kernel_us * rng.uniform(0.6, 1.4)
            if family == "matmul":
                duration *= 2.4  # GEMMs dominate the compute
            events.append(_event(rng.choice(KERNELS[family]), "kernel", cursor, duration))
            cursor += duration + gap_us * rng.uniform(0.8, 1.2)
    return _wrap(events)


def main() -> None:
    here = Path(__file__).parent
    written: Dict[str, dict] = {"trace_halfidle.json": half_idle_trace()}

    # (launches, kernel_us, gap_us) per shape. The gap is fixed launch overhead
    # and does not shrink with the shape, so small shapes are launch-bound and
    # large ones are not -- the point Rung 0 turns on.
    shapes = {
        "small":  dict(n_launches=12, kernel_us=6.0,   gap_us=9.0),
        "medium": dict(n_launches=12, kernel_us=60.0,  gap_us=9.0),
        "large":  dict(n_launches=12, kernel_us=480.0, gap_us=9.0),
    }
    for shape, params in shapes.items():
        written[f"trace_{shape}_baseline.json"] = workload_trace(seed=1, **params)
        # The optimized path fuses: fewer launches, longer kernels, same work.
        written[f"trace_{shape}_optimized.json"] = workload_trace(
            n_launches=params["n_launches"],
            kernel_us=params["kernel_us"] * 1.7,
            gap_us=params["gap_us"],
            seed=2,
            families=("layernorm", "matmul", "matmul", "elementwise"),
        )

    for name, payload in written.items():
        (here / name).write_text(json.dumps(payload))
    print(f"wrote {len(written)} traces to {here}")


if __name__ == "__main__":
    main()
