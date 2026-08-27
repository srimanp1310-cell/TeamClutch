"""Task 4 acceptance: the analysis layer, on the synthetic fixture.

The fixture exists so this whole layer is testable before Person A has produced
a single real measurement. What is checked here is not "does it plot" but
"does it plot the truth": that non-measurements never reach a mean, that a
re-run supersedes its predecessor, that the geometric mean is the geometric
mean, and that a one-factor-at-a-time figure actually holds the other factors
fixed.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import pandas as pd
import pytest

from analysis import figures, load, make_all
from analysis.roofline import (
    RTX_4050_LAPTOP, achieved_tflops, arithmetic_intensity,
    explicit_attention_bytes, forward_bytes, forward_flops, plot_roofline,
    ridge_point, roofline_frame,
)
from src.baseline import TransformerConfig

FIXTURES = Path(__file__).parent / "fixtures"
RESULTS = FIXTURES / "results_synthetic.csv"
CLOCKS = FIXTURES / "clocks_synthetic.csv"
FLAT_CLOCKS = FIXTURES / "clocks_flat.csv"

MIN_PNG_BYTES = 10 * 1024


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return load.load_results(RESULTS)


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------

def test_fixture_exercises_every_status(frame):
    """If the fixture stops covering a status, the code path for it stops
    being tested -- silently. Assert the coverage, not just the parsing."""
    counts = load.status_counts(frame)
    for status in ("PASS", "FAIL", "SKIPPED", "OOM_BASELINE", "DISCARDED"):
        assert counts.get(status, 0) > 0, f"fixture has no {status} rows"


def test_status_derivation_matches_the_notes_prefixes(frame):
    for _, row in frame.iterrows():
        notes = str(row["notes"])
        if notes.startswith("SKIPPED:"):
            assert row["status"] == "SKIPPED"
        elif notes.startswith("FAIL:"):
            assert row["status"] == "FAIL"
        elif "baseline OOM" in notes:
            assert row["status"] == "OOM_BASELINE"


def test_dtypes_are_numeric_not_strings(frame):
    for column in ("speedup", "max_abs_err", "peak_vram_mb"):
        assert frame[column].dtype.kind == "f", column
    for column in ("batch", "seq_len", "layers"):
        assert str(frame[column].dtype) == "Int64", column
    assert frame["causal"].dtype == "boolean"


def test_non_measurements_never_reach_the_usable_set(frame):
    usable = load.usable(frame)
    assert (usable["status"] == "PASS").all()
    assert usable["speedup"].notna().all()
    assert not usable["discarded"].any()


def test_compiled_baseline_rows_are_excluded_by_default(frame):
    """A --compile-baseline run has a different denominator. It must not be
    averaged in with, or supersede, the eager runs."""
    default = load.usable(frame)
    both = load.usable(frame, include_compiled_baseline=True)
    assert len(both) > len(default)
    assert not (default["compile_baseline"] == True).any()  # noqa: E712


def test_latest_per_config_keeps_the_newest_row_only(frame):
    latest = load.latest_per_config(frame)
    assert not latest.duplicated(["strategy_name", "config_key"]).any()

    key = latest.iloc[0]["config_key"]
    strategy = latest.iloc[0]["strategy_name"]
    candidates = load.usable(frame)
    candidates = candidates[
        (candidates["config_key"] == key) & (candidates["strategy_name"] == strategy)
    ]
    kept = latest[
        (latest["config_key"] == key) & (latest["strategy_name"] == strategy)
    ].iloc[0]
    assert kept["timestamp"] == candidates["timestamp"].max()


def test_config_key_distinguishes_the_compile_baseline_condition(frame):
    """Otherwise a later compiled-baseline run silently replaces the eager one."""
    assert "compile_baseline" in load.CONFIG_FIELDS
    both = load.usable(frame, include_compiled_baseline=True)
    for _, group in both.groupby(["strategy_name", "config_key"]):
        assert group["compile_baseline"].nunique() == 1


# ---------------------------------------------------------------------------
# geometric mean
# ---------------------------------------------------------------------------

def test_geometric_mean_on_a_hand_computed_toy_frame():
    """2.0, 4.0, 8.0 -> exactly 4.0. The arithmetic mean would say 4.67."""
    toy = pd.DataFrame({
        "strategy_name": ["s", "s", "s"],
        "speedup": [2.0, 4.0, 8.0],
        "status": ["PASS"] * 3,
        "compile_baseline": [False] * 3,
    })
    assert load.geometric_mean_speedup(toy, "s") == pytest.approx(4.0)
    assert sum([2.0, 4.0, 8.0]) / 3 == pytest.approx(4.6667, abs=1e-4)


def test_geometric_mean_punishes_a_regression_the_way_it_should():
    """2x on one shape and 0.5x on another is 1.00x, not 1.25x."""
    toy = pd.DataFrame({
        "strategy_name": ["s", "s"], "speedup": [2.0, 0.5],
        "status": ["PASS"] * 2, "compile_baseline": [False] * 2,
    })
    assert load.geometric_mean_speedup(toy, "s") == pytest.approx(1.0)


def test_geometric_mean_is_nan_for_an_unmeasured_strategy(frame):
    assert math.isnan(load.geometric_mean_speedup(frame, "does-not-exist"))


def test_control_measures_one_x(frame):
    """The fixture's control must land in the same 0.97-1.03 band the real
    Day-1 gate uses, or the fixture is not representative."""
    assert load.geometric_mean_speedup(frame, "baseline") == pytest.approx(1.0, abs=0.03)


def test_speedup_summary_is_ordered_best_first(frame):
    summaries = load.speedup_summary(frame)
    assert [s.geomean for s in summaries] == sorted(
        (s.geomean for s in summaries), reverse=True
    )
    for item in summaries:
        assert item.minimum <= item.geomean <= item.maximum
        assert item.n > 0


# ---------------------------------------------------------------------------
# crossover and dispatch
# ---------------------------------------------------------------------------

def test_crossover_table_names_a_winner_and_a_runner_up(frame):
    table = load.crossover_table(frame)
    assert not table.empty
    assert "baseline" not in set(table["best_strategy"])
    for _, row in table.iterrows():
        if row["runner_up"] is not None:
            assert row["best_speedup"] >= row["runner_up_speedup"]
            assert row["margin"] == pytest.approx(
                row["best_speedup"] - row["runner_up_speedup"]
            )


def test_dispatch_keys_are_unique_after_configs_collapse(frame):
    """Several configs share one dispatch key because it carries no layer
    count. The winner among them must be decided, not taken at random."""
    choices = load.dispatch_choices(frame)
    assert choices["dispatch_key"].is_unique
    assert (choices["n_configs"] >= 1).all()
    assert (choices["n_configs"] > 1).any(), "fixture should exercise the collapse"


def test_write_dispatch_table_is_valid_json_with_a_default(tmp_path, frame):
    path = load.write_dispatch_table(frame, tmp_path / "dispatch.json")
    table = json.loads(path.read_text())

    assert "default" in table
    assert isinstance(table["default"], str) and table["default"]
    assert "sm_89" in table
    assert table["sm_89"], "no dispatch entries"
    for key, strategy in table["sm_89"].items():
        assert len(key.split(",")) == len(load.DISPATCH_FIELDS)
        assert isinstance(strategy, str)
    assert "geometric_mean_speedup" in table["meta"]
    assert table["meta"]["n_dispatch_keys"] == len(table["sm_89"])


def test_dispatch_table_capability_is_explicit(tmp_path, frame):
    """Performance is only ever claimed for a capability we measured on."""
    path = load.write_dispatch_table(frame, tmp_path / "d.json", capability="sm_75")
    table = json.loads(path.read_text())
    assert "sm_75" in table and "sm_89" not in table


def test_results_markdown_table_renders(frame):
    text = load.results_markdown_table(frame)
    assert text.startswith("| seq_len |")
    assert "baseline" in text and "x |" in text


# ---------------------------------------------------------------------------
# roofline
# ---------------------------------------------------------------------------

def test_ridge_point_matches_the_hand_computed_value():
    """12.0 TFLOP/s over 192 GB/s = 62.5 FLOP/byte."""
    assert ridge_point(12.0, 192.0) == pytest.approx(62.5)
    assert ridge_point(48.0, 192.0) == pytest.approx(250.0)


def test_flops_scale_as_documented():
    small = TransformerConfig(1, 128, 256, 8, 1024, 2, False)
    doubled_layers = TransformerConfig(1, 128, 256, 8, 1024, 4, False)
    assert forward_flops(doubled_layers) == 2 * forward_flops(small)

    # The S^2 attention term means doubling S more than doubles the FLOPs.
    doubled_seq = TransformerConfig(1, 256, 256, 8, 1024, 2, False)
    assert forward_flops(doubled_seq) > 2 * forward_flops(small)


def test_explicit_attention_is_the_whole_difference():
    config = TransformerConfig(8, 512, 512, 8, 2048, 6, False)
    with_scores = forward_bytes(config, "float32", include_attention=True)
    without = forward_bytes(config, "float32", include_attention=False)
    assert with_scores - without == explicit_attention_bytes(config, "float32")
    # ...and it is what pushes the baseline toward the bandwidth-bound side.
    assert arithmetic_intensity(config, "float32", True) < arithmetic_intensity(
        config, "float32", False
    )


def test_explicit_attention_bytes_are_quadratic_in_seq_len():
    small = TransformerConfig(1, 128, 256, 8, 1024, 1, False)
    large = TransformerConfig(1, 256, 256, 8, 1024, 1, False)
    assert explicit_attention_bytes(large) == 4 * explicit_attention_bytes(small)


def test_achieved_tflops_is_flops_over_time():
    config = TransformerConfig(1, 128, 256, 8, 1024, 2, False)
    assert achieved_tflops(config, 10.0) == pytest.approx(
        forward_flops(config) / 0.01 / 1e12
    )
    assert math.isnan(achieved_tflops(config, 0.0))


def test_no_measurement_exceeds_the_machine_peak(frame):
    """A point above its own roof is physically impossible, so it means either
    the FLOP model or the fixture is wrong. Either way, catch it here."""
    data = roofline_frame(frame)
    for _, row in data.iterrows():
        peak = (RTX_4050_LAPTOP.peak_bf16_tflops
                if row["dtype"] in ("float16", "bfloat16")
                else RTX_4050_LAPTOP.peak_fp32_tflops)
        assert row["achieved_tflops"] <= peak, (
            f"{row['strategy_name']} at {row['dtype']} achieves "
            f"{row['achieved_tflops']:.1f} TFLOP/s against a {peak} peak"
        )


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("function", figures.ALL_FIGURES, ids=lambda f: f.__name__)
def test_every_figure_writes_a_real_png(function, frame, tmp_path):
    path = function(frame, out_dir=tmp_path)
    assert path.exists()
    assert path.stat().st_size > MIN_PNG_BYTES, f"{path} is suspiciously small"
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_roofline_writes_a_real_png(frame, tmp_path):
    path = plot_roofline(frame, out_dir=tmp_path)
    assert path.stat().st_size > MIN_PNG_BYTES


def test_thermal_trace_writes_a_real_png(tmp_path):
    path = figures.thermal_trace(CLOCKS, out_dir=tmp_path)
    assert path.stat().st_size > MIN_PNG_BYTES


@pytest.mark.parametrize("function", figures.ALL_FIGURES, ids=lambda f: f.__name__)
def test_figures_degrade_to_a_labelled_panel_on_empty_data(function, tmp_path):
    """A half-finished sweep must produce a figure that says what is missing,
    not a traceback -- otherwise the pipeline cannot be run until the very end."""
    empty = load.load_results(RESULTS).iloc[0:0]
    path = function(empty, out_dir=tmp_path)
    assert path.exists() and path.stat().st_size > 0


def test_ofat_slice_holds_every_other_axis_fixed(frame):
    """The bug this prevents: a 'peak memory vs sequence length' line that
    mixes dtypes and depths, and comes out falling as S grows."""
    sliced = figures._ofat_slice(load.usable(frame), "seq_len")
    for field in load.CONFIG_FIELDS:
        if field != "seq_len":
            assert sliced[field].nunique() == 1, f"{field} varies within the slice"
    assert sliced["seq_len"].nunique() > 1


def test_vram_line_is_monotonic_in_seq_len(frame):
    """Peak memory cannot fall as the sequence grows."""
    sliced = figures._ofat_slice(load.usable(frame), "seq_len")
    series = sliced.groupby("seq_len")["baseline_peak_vram_mb"].max().dropna().sort_index()
    assert len(series) > 1
    assert list(series.values) == sorted(series.values)


def test_palette_is_never_cycled():
    from analysis.style import SERIES, color_for

    assert len({color_for(i) for i in range(len(SERIES))}) == len(SERIES)
    with pytest.raises(IndexError):
        color_for(len(SERIES))


# ---------------------------------------------------------------------------
# make_all
# ---------------------------------------------------------------------------

def test_make_all_runs_end_to_end(tmp_path):
    started = time.perf_counter()
    code = make_all.main([
        "--results", str(RESULTS),
        "--logs", str(FIXTURES),
        "--figures", str(tmp_path / "figures"),
        "--summary", str(tmp_path / "summary.md"),
        "--dispatch", str(tmp_path / "dispatch.json"),
    ])
    elapsed = time.perf_counter() - started

    assert code == 0
    assert elapsed < 30, f"make_all took {elapsed:.1f}s"

    pngs = sorted((tmp_path / "figures").glob("*.png"))
    assert len(pngs) >= len(figures.ALL_FIGURES) + 1  # + roofline
    assert all(p.stat().st_size > MIN_PNG_BYTES for p in pngs)

    summary = (tmp_path / "summary.md").read_text()
    for heading in ("Row counts by status", "Speedup by strategy", "Dispatch",
                    "Roofline reference"):
        assert heading in summary
    assert "Control check" in summary and "PASS" in summary

    json.loads((tmp_path / "dispatch.json").read_text())


def test_make_all_reports_a_missing_results_file(tmp_path, capsys):
    assert make_all.main(["--results", str(tmp_path / "nope.csv")]) == 1
    assert "no results" in capsys.readouterr().err


def test_make_all_writes_one_thermal_figure_per_clock_log(tmp_path):
    make_all.main([
        "--results", str(RESULTS), "--logs", str(FIXTURES),
        "--figures", str(tmp_path / "f"), "--summary", str(tmp_path / "s.md"),
        "--dispatch", str(tmp_path / "d.json"),
    ])
    thermal = sorted((tmp_path / "f").glob("thermal_*.png"))
    assert len(thermal) == len(sorted(FIXTURES.glob("clocks_*.csv")))
