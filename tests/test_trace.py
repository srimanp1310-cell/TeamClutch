"""Task 7 acceptance: Chrome-trace parsing and the GPU-busy number.

The headline assertion is the busy fraction on a hand-built 50%-idle trace.
Everything else exists because a trace parser fails silently: if it decides no
event is a kernel, `gpu_busy_fraction` returns 0.0 rather than raising, and a
0% busy reading looks exactly like a catastrophically launch-bound model. So
the parser's assumptions are asserted individually.
"""

from __future__ import annotations

import gzip
import itertools
import json
import math
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from analysis.trace import (
    busy_table, gpu_active_span_us, gpu_busy_fraction, has_kernel_track,
    kernel_breakdown, kernel_family, launch_stats, load_trace, merged_intervals,
    plot_busy_vs_shape, plot_launch_overhead, plot_timeline,
)

FIXTURES = Path(__file__).parent / "fixtures"
HALF_IDLE = FIXTURES / "trace_halfidle.json"
SHAPES = ("small", "medium", "large")
MIN_PNG_BYTES = 10 * 1024


@pytest.fixture(scope="module")
def traces():
    return {
        f"{shape}_{variant}": load_trace(FIXTURES / f"trace_{shape}_{variant}.json")
        for shape in SHAPES for variant in ("baseline", "optimized")
    }


_SCRATCH = Path(tempfile.mkdtemp(prefix="trace_fixtures_"))
_COUNTER = itertools.count()


def synthetic(events) -> pd.DataFrame:
    """Build a frame from (name, cat, ts, dur) tuples via the real loader.

    Goes through a file on purpose: constructing the DataFrame directly would
    skip `load_trace`, which is the part under test.
    """
    payload = {"traceEvents": [
        {"ph": "X", "cat": cat, "name": name, "pid": 0, "tid": 7,
         "ts": ts, "dur": dur}
        for name, cat, ts, dur in events
    ]}
    path = _SCRATCH / f"synthetic_{next(_COUNTER)}.json"
    path.write_text(json.dumps(payload))
    return load_trace(path)


# ---------------------------------------------------------------------------
# the acceptance number
# ---------------------------------------------------------------------------

def test_half_idle_fixture_is_exactly_fifty_percent_busy():
    frame = load_trace(HALF_IDLE)
    assert gpu_busy_fraction(frame) == pytest.approx(0.5, abs=0.01)


def test_half_idle_fixture_has_twenty_kernels():
    frame = load_trace(HALF_IDLE)
    assert int(frame["is_kernel"].sum()) == 20


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------

def test_only_complete_events_are_counted():
    """Metadata, instant and flow records have no extent. Summing them would
    inflate every duration in the file."""
    raw = json.loads(HALF_IDLE.read_text())["traceEvents"]
    complete = [e for e in raw if e.get("ph") == "X"]
    assert len(raw) > len(complete), "fixture should contain non-X records"
    assert len(load_trace(HALF_IDLE)) == len(complete)


@pytest.mark.parametrize("category", ["kernel", "Kernel", "gpu_op", "GPU_OP"])
def test_every_known_kernel_category_spelling_is_recognised(category):
    """The category name has changed across PyTorch versions. Missing one
    yields a silent 0% busy, which reads as a real (catastrophic) result."""
    frame = synthetic([("ampere_sgemm", category, 0, 100)])
    assert bool(frame["is_kernel"].iloc[0]), category


def test_cpu_side_events_are_not_kernels():
    frame = synthetic([
        ("aten::matmul", "cpu_op", 0, 500),
        ("cudaLaunchKernel", "cuda_runtime", 10, 5),
        ("ampere_sgemm", "kernel", 20, 100),
    ])
    assert int(frame["is_kernel"].sum()) == 1
    assert gpu_busy_fraction(frame) == pytest.approx(1.0)


def test_memcpy_is_gpu_work_but_not_a_kernel():
    frame = synthetic([("Memcpy DtoH", "gpu_memcpy", 0, 100)])
    assert not frame["is_kernel"].iloc[0]
    assert frame["is_gpu"].iloc[0]
    # No kernel records at all -> unmeasurable, not 0%.
    assert math.isnan(gpu_busy_fraction(frame))
    assert gpu_busy_fraction(frame, include_memory=True) == pytest.approx(1.0)


def test_gzipped_traces_load(tmp_path):
    path = tmp_path / "trace.json.gz"
    path.write_bytes(gzip.compress(HALF_IDLE.read_bytes()))
    assert gpu_busy_fraction(load_trace(path)) == pytest.approx(0.5, abs=0.01)


def test_empty_trace_degrades_without_crashing():
    frame = synthetic([])
    assert frame.empty
    assert math.isnan(gpu_busy_fraction(frame))
    assert gpu_active_span_us(frame) == 0.0
    assert kernel_breakdown(frame).empty
    assert not has_kernel_track(frame)


def test_a_trace_with_no_kernel_track_is_unmeasurable_not_zero():
    """The single most dangerous confusion in this module.

    Under WSL2, CUPTI does not populate device-side kernel records, so a real
    trace can contain the whole CPU-side story and no kernels. Reporting 0.0%
    busy for that would read as "the GPU did nothing" — a catastrophic result —
    when the truth is "this platform did not report it".
    """
    frame = synthetic([
        ("aten::linear", "cpu_op", 0, 5000),
        ("cudaLaunchKernel", "cuda_runtime", 10, 12),
        ("cuLaunchKernel", "cuda_driver", 40, 9),
    ])
    assert not has_kernel_track(frame)
    assert math.isnan(gpu_busy_fraction(frame)), "must be NaN, never 0.0"

    # ...but the launch-side analysis still works, which is the whole point.
    stats = launch_stats(frame, forwards=1)
    assert stats["launches"] == 2
    assert stats["launch_fraction"] > 0
    assert stats["kernels_per_forward"] == 2


def test_driver_api_launches_are_counted_too():
    """cuBLAS submits via cuLaunchKernel, not cudaLaunchKernel, and those
    records are not nested inside the runtime-API ones. Counting only
    cudaLaunchKernel undercounts launches by ~40% on this workload."""
    frame = synthetic([
        ("cudaLaunchKernel", "cuda_runtime", 0, 10),
        ("cudaLaunchKernel", "cuda_runtime", 20, 10),
        ("cuLaunchKernel", "cuda_driver", 40, 10),
    ])
    assert launch_stats(frame)["launches"] == 3


def test_launch_stats_on_the_real_wsl2_traces():
    """Guards the platform assumption against a future PyTorch or driver that
    starts populating the kernel track — at which point this test flips and we
    should notice rather than keep quoting the fallback."""
    real = Path(__file__).resolve().parents[1] / "logs" / "trace_small_baseline.json"
    if not real.exists():
        pytest.skip("no real trace committed yet")
    frame = load_trace(real)
    stats = launch_stats(frame)
    assert stats["launches"] > 0
    assert 0 < stats["launch_fraction"] < 1
    if has_kernel_track(frame):
        pytest.skip("this trace HAS a device kernel track — the WSL2 limitation "
                    "no longer applies and the report's §3 needs revisiting")


def test_a_trace_with_no_traceevents_key_raises_clearly(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"nothing": "here"}))
    with pytest.raises(ValueError, match="traceEvents"):
        load_trace(path)


# ---------------------------------------------------------------------------
# busy fraction semantics
# ---------------------------------------------------------------------------

def test_overlapping_kernels_are_merged_not_summed():
    """Kernels on different streams overlap. Summing durations would report a
    GPU more than 100% busy, which is not a thing."""
    frame = synthetic([
        ("a", "kernel", 0, 100),
        ("b", "kernel", 50, 100),   # overlaps a
    ])
    assert merged_intervals(frame) == [(0.0, 150.0)]
    assert gpu_busy_fraction(frame) == pytest.approx(1.0)
    assert frame["dur_us"].sum() == 200  # the naive number, deliberately not used


def test_adjacent_kernels_merge_into_one_interval():
    frame = synthetic([("a", "kernel", 0, 100), ("b", "kernel", 100, 100)])
    assert merged_intervals(frame) == [(0.0, 200.0)]


def test_gap_between_kernels_is_idle():
    frame = synthetic([("a", "kernel", 0, 100), ("b", "kernel", 300, 100)])
    assert gpu_active_span_us(frame) == 400.0
    assert gpu_busy_fraction(frame) == pytest.approx(200 / 400)


def test_explicit_window_clips_the_intervals():
    frame = synthetic([("a", "kernel", 0, 1000)])
    assert gpu_busy_fraction(frame, window=(0, 2000)) == pytest.approx(0.5)
    assert gpu_busy_fraction(frame, window=(500, 1500)) == pytest.approx(0.5)
    assert gpu_busy_fraction(frame, window=(2000, 3000)) == 0.0


def test_busy_fraction_never_exceeds_one(traces):
    for label, frame in traces.items():
        assert 0.0 <= gpu_busy_fraction(frame) <= 1.0, label


# ---------------------------------------------------------------------------
# breakdown and classification
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,expected", [
    ("ampere_sgemm_128x64_tn", "matmul"),
    ("cutlass_80_tensorop_bf16_s16816gemm", "matmul"),
    ("at::native::softmax_warp_forward", "softmax"),
    ("at::native::vectorized_layer_norm_kernel", "layernorm"),
    ("at::native::masked_fill_kernel", "masking"),
    ("at::native::elementwise_kernel", "elementwise"),
    ("Memcpy DtoH", "memory"),
    ("some_kernel_nobody_has_seen", "other"),
])
def test_kernel_family_classification(name, expected):
    assert kernel_family(name) == expected


def test_breakdown_shares_sum_to_one(traces):
    breakdown = kernel_breakdown(traces["small_baseline"], by="family", top=99)
    assert breakdown["share"].sum() == pytest.approx(1.0)
    assert (breakdown["total_us"] > 0).all()
    assert breakdown["total_us"].is_monotonic_decreasing


def test_breakdown_respects_top_n(traces):
    assert len(kernel_breakdown(traces["small_baseline"], top=3)) == 3


def test_matmul_dominates_a_transformer_trace(traces):
    """A transformer that is not mostly GEMM means the fixture is wrong."""
    breakdown = kernel_breakdown(traces["large_baseline"], by="family", top=99)
    top_family = breakdown.iloc[0]
    assert top_family["family"] == "matmul"
    assert top_family["share"] > 0.5


# ---------------------------------------------------------------------------
# the Rung 0 classification
# ---------------------------------------------------------------------------

def test_busy_fraction_still_separates_small_from_large(traces):
    """The whole point of Rung 0: fixed launch overhead does not shrink with
    the shape, so it dominates at small shapes and vanishes at large ones."""
    table = busy_table(traces).set_index("trace")
    assert table.loc["small_baseline", "gpu_busy"] < 0.70
    assert table.loc["large_baseline", "gpu_busy"] > 0.90


def test_verdict_uses_launch_arithmetic_not_the_busy_fraction(traces):
    """The verdict must survive a platform with no device track, so it is
    derived from launch counts — available everywhere — rather than busy %."""
    table = busy_table(traces).set_index("trace")
    assert set(table["verdict"]) <= {
        "launch-bound", "borderline", "not launch-bound", "no data"
    }
    # The fixtures carry CPU-side launch records alongside their kernels, so the
    # verdict is derived even though a busy fraction is also available.
    assert (table["launches"] > 0).all()
    assert table["launch_fraction"].notna().all()


def test_fusing_raises_the_busy_fraction_at_every_shape(traces):
    """Fewer, longer kernels over the same work means less idle."""
    table = busy_table(traces).set_index("trace")
    for shape in SHAPES:
        assert (table.loc[f"{shape}_optimized", "gpu_busy"]
                >= table.loc[f"{shape}_baseline", "gpu_busy"]), shape


def test_busy_table_columns(traces):
    table = busy_table(traces)
    for column in ("trace", "kernel_track", "gpu_busy", "launches",
                   "launch_fraction", "mean_launch_us", "span_ms", "kernels",
                   "mean_kernel_us", "verdict"):
        assert column in table.columns
    assert len(table) == len(traces)


def test_launch_overhead_figure_writes_a_real_png(traces, tmp_path):
    path = plot_launch_overhead(traces, out_dir=tmp_path)
    assert path.stat().st_size > MIN_PNG_BYTES


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def test_timeline_writes_a_real_png(traces, tmp_path):
    path = plot_timeline(traces["small_baseline"], traces["small_optimized"],
                         out_dir=tmp_path, window_ms=1.0)
    assert path.stat().st_size > MIN_PNG_BYTES


def test_busy_vs_shape_writes_a_real_png(traces, tmp_path):
    path = plot_busy_vs_shape(traces, out_dir=tmp_path)
    assert path.stat().st_size > MIN_PNG_BYTES


def test_figures_degrade_when_there_are_no_traces(tmp_path):
    empty = synthetic([])
    assert plot_timeline(empty, empty, out_dir=tmp_path).exists()
    assert plot_busy_vs_shape({}, out_dir=tmp_path).exists()
