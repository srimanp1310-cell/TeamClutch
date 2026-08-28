"""Task 2 acceptance: the sweep harness, exercised entirely on CPU.

Every test here runs in seconds with no GPU. The point is not to measure
anything interesting -- it is to prove the harness records the truth in every
branch it has: pass, fail, skip, matrix expansion, append-only, and the
thermal discard rule.

A note on the timing settings used below. The plan's example command uses
`--repeats 3 --benchmark-rounds 1`, which on CPU at S=16 gives a speedup that
swings between 0.7x and 1.5x run to run -- it is measuring scheduler noise, not
the model. `--warmup 5 --repeats 25 --benchmark-rounds 3` costs ~0.7 s and
holds the control strategy inside 1.00 +/- 0.02, which is what makes the
"baseline must measure 1.00x" gate meaningful.
"""

from __future__ import annotations

import contextlib
import csv
from pathlib import Path
from typing import Dict, List, Optional

import pytest
import torch

from bench import sweep
from bench.thermal import parse_clock_log, summarize
from src.baseline import BaselineTransformer
from src.strategies import STRATEGIES

FIXTURES = Path(__file__).parent / "fixtures"

# Small, fast, and stable enough that a 1.00x control reads as 1.00x.
TINY = [
    "--device", "cpu", "--batch", "2", "--seq-len", "32", "--d-model", "64",
    "--heads", "4", "--layers", "2",
    "--warmup", "5", "--repeats", "25", "--benchmark-rounds", "3",
    "--accuracy-trials", "2",
    "--no-thermal", "--no-cooldown", "--allow-dirty",
]


def run_sweep(results: Path, *extra: str, strategy: str = "baseline") -> int:
    return sweep.main(["--strategy", strategy, "--results", str(results), *TINY, *extra])


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_header(path: Path) -> List[str]:
    with path.open(newline="") as handle:
        return next(csv.reader(handle))


@contextlib.contextmanager
def temporarily_registered(name: str, cls: type):
    """Add a strategy for one test and remove it again.

    STRATEGIES is module-global; a leaked entry would make every other test
    that parametrizes over the registry fail in a confusing way.
    """
    STRATEGIES[name] = cls
    try:
        yield
    finally:
        STRATEGIES.pop(name, None)


# ---------------------------------------------------------------------------
# the happy path
# ---------------------------------------------------------------------------

def test_single_config_writes_one_passing_row(tmp_path):
    results = tmp_path / "r.csv"
    assert run_sweep(results) == 0

    rows = read_rows(results)
    assert len(rows) == 1
    row = rows[0]

    assert row["accuracy_pass"] == "True"
    assert row["strategy_name"] == "baseline"
    assert float(row["max_abs_err"]) == 0.0, "the control must be bit-identical"
    assert 0.7 <= float(row["speedup"]) <= 1.4
    assert float(row["baseline_median_ms"]) > 0
    assert float(row["optimized_median_ms"]) > 0
    assert row["git_sha"] and row["timestamp"]


def test_header_starts_with_the_21_agreed_columns_in_order(tmp_path):
    """The contract fixes the first 21 columns. Extras may only be appended."""
    results = tmp_path / "r.csv"
    run_sweep(results)

    header = read_header(results)
    assert header[:21] == list(sweep.CSV_COLUMNS_AGREED)
    assert len(sweep.CSV_COLUMNS_AGREED) == 21
    assert header[21:] == list(sweep.CSV_COLUMNS_PROPOSED)


def test_mask_branches_pass(tmp_path):
    """Padding and causal masking are where a wrong implementation shows up."""
    results = tmp_path / "r.csv"
    assert run_sweep(results, "--causal", "--padding-ratio", "0.3") == 0

    row = read_rows(results)[0]
    assert row["accuracy_pass"] == "True"
    assert float(row["max_abs_err"]) == 0.0
    assert row["causal"] == "True"
    assert float(row["padding_ratio"]) == 0.3


def test_thermal_columns_are_empty_without_a_gpu(tmp_path):
    """No nvidia-smi means no clock data -- empty cells, not zeros or crashes."""
    results = tmp_path / "r.csv"
    run_sweep(results)
    row = read_rows(results)[0]
    assert row["mean_sm_clock_mhz"] == ""
    assert row["max_temp_c"] == ""
    assert row["peak_vram_mb"] == ""  # CUDA-only metric


# ---------------------------------------------------------------------------
# matrices and append-only behaviour
# ---------------------------------------------------------------------------

def test_matrix_quick_appends_and_never_overwrites(tmp_path):
    results = tmp_path / "r.csv"

    run_sweep(results, "--matrix", "quick")
    assert len(read_rows(results)) == 2

    run_sweep(results, "--matrix", "quick")
    assert len(read_rows(results)) == 4, "results.csv must be append-only"

    header_count = sum(1 for line in results.read_text().splitlines() if line.startswith("timestamp"))
    assert header_count == 1, "the header must not be rewritten on append"


def test_matrix_quick_varies_only_seq_len(tmp_path):
    results = tmp_path / "r.csv"
    run_sweep(results, "--matrix", "quick")

    rows = read_rows(results)
    assert sorted(int(row["seq_len"]) for row in rows) == [128, 512]
    assert {row["dtype"] for row in rows} == {"float32"}
    assert {row["causal"] for row in rows} == {"False"}


@pytest.mark.parametrize(
    "name,expected",
    [("seq", 4), ("batch", 3), ("dmodel", 3), ("dtype", 3), ("layers", 4),
     ("mask", 4), ("quick", 2), ("accuracy", 12), ("long", 3)],
)
def test_matrix_sizes(name, expected):
    assert len(sweep.build_matrix(name, sweep.RunSpec())) == expected


def test_long_matrix_targets_the_vram_ceiling():
    """Batch 1, because at B=8 the baseline exceeds the memory budget at
    S=2048 and every interesting row would be SKIPPED before it ran."""
    specs = sweep.build_matrix("long", sweep.RunSpec())
    assert [s.seq_len for s in specs] == [2048, 4096, 8192]
    assert all(s.batch == 1 for s in specs)


def test_default_matrix_is_deduplicated():
    """Every sub-matrix includes the base point; `default` must keep it once."""
    specs = sweep.build_matrix("default", sweep.RunSpec())
    assert len(specs) == len(set(specs))
    assert specs.count(sweep.RunSpec()) == 1


def test_matrix_respects_non_varied_cli_overrides(tmp_path):
    """`--matrix quick --d-model 64` must keep d_model=64 on both rows."""
    results = tmp_path / "r.csv"
    run_sweep(results, "--matrix", "quick")
    assert {row["d_model"] for row in read_rows(results)} == {"64"}
    assert {row["layers"] for row in read_rows(results)} == {"2"}


# ---------------------------------------------------------------------------
# a strategy that is wrong
# ---------------------------------------------------------------------------

class DeliberatelyWrong(BaselineTransformer):
    """Correct shapes, wrong numbers -- 10% high everywhere."""

    def forward(self, x: torch.Tensor, valid_token_mask: Optional[torch.Tensor] = None):
        return super().forward(x, valid_token_mask) * 1.1


def test_wrong_strategy_is_recorded_as_a_failure_with_no_timing(tmp_path):
    results = tmp_path / "r.csv"
    with temporarily_registered("wrong", DeliberatelyWrong):
        assert run_sweep(results, strategy="wrong") == 0

    row = read_rows(results)[0]
    assert row["accuracy_pass"] == "False"
    assert row["notes"].startswith("FAIL:")
    assert float(row["max_abs_err"]) > 0
    # Timing an implementation that is wrong would put a meaningless speedup in
    # the results table, so it must be left empty.
    assert row["baseline_median_ms"] == ""
    assert row["optimized_median_ms"] == ""
    assert row["speedup"] == ""


def test_failure_note_carries_the_debugging_fields(tmp_path):
    """The note must be paste-ready: which trial, which element, both values."""
    results = tmp_path / "r.csv"
    with temporarily_registered("wrong", DeliberatelyWrong):
        run_sweep(results, strategy="wrong")

    note = read_rows(results)[0]["notes"]
    for field in ("trial=", "worst_index=", "base=", "opt=", "failed_feature_dims="):
        assert field in note, f"{field!r} missing from {note!r}"


def test_benchmark_on_failure_times_it_anyway(tmp_path):
    results = tmp_path / "r.csv"
    with temporarily_registered("wrong", DeliberatelyWrong):
        run_sweep(results, "--benchmark-on-failure", strategy="wrong")

    row = read_rows(results)[0]
    assert row["accuracy_pass"] == "False"
    assert float(row["speedup"]) > 0


# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------

def test_oversized_config_is_skipped_not_crashed(tmp_path):
    results = tmp_path / "r.csv"
    assert run_sweep(results, "--hard-cap-gib", "0.000001") == 0

    row = read_rows(results)[0]
    assert row["notes"].startswith("SKIPPED:")
    assert row["accuracy_pass"] == ""
    assert row["speedup"] == ""
    # The reason must carry real numbers, not just "too big".
    assert "estimated baseline peak" in row["notes"]


def test_force_overrides_the_memory_precheck(tmp_path):
    results = tmp_path / "r.csv"
    assert run_sweep(results, "--hard-cap-gib", "0.000001", "--force") == 0
    row = read_rows(results)[0]
    assert row["accuracy_pass"] == "True"
    assert not row["notes"].startswith("SKIPPED:")


def test_unknown_strategy_fails_loudly(tmp_path):
    with pytest.raises(KeyError, match="baseline"):
        run_sweep(tmp_path / "r.csv", strategy="no-such-strategy")


def test_allow_dirty_is_recorded_in_the_row(tmp_path):
    """A row measured from an uncommitted tree must say so."""
    results = tmp_path / "r.csv"
    run_sweep(results)
    row = read_rows(results)[0]
    if sweep.git_is_dirty():
        assert "dirty" in row["notes"]


def test_notes_flag_is_copied_into_every_row(tmp_path):
    results = tmp_path / "r.csv"
    run_sweep(results, "--matrix", "quick", "--notes", "rung-3 sdpa+bf16")
    for row in read_rows(results):
        assert "rung-3 sdpa+bf16" in row["notes"]


# ---------------------------------------------------------------------------
# append_row against an older, narrower file
# ---------------------------------------------------------------------------

def test_append_row_respects_an_existing_narrower_header(tmp_path, capsys):
    """A results.csv written before the extra columns existed must still work."""
    results = tmp_path / "old.csv"
    with results.open("w", newline="") as handle:
        csv.writer(handle).writerow(sweep.CSV_COLUMNS_AGREED)

    header = sweep.append_row(
        results, {"strategy_name": "baseline", "speedup": 1.0, "ffn_dim": 2048}
    )
    assert header == list(sweep.CSV_COLUMNS_AGREED)

    rows = read_rows(results)
    assert len(rows) == 1 and rows[0]["strategy_name"] == "baseline"
    assert "ffn_dim" not in rows[0]
    # ...and it must say out loud that a field was dropped.
    assert "ffn_dim" in capsys.readouterr().err


def test_append_row_creates_the_file_with_the_full_header(tmp_path):
    results = tmp_path / "nested" / "new.csv"
    header = sweep.append_row(results, {"strategy_name": "baseline"})
    assert header == list(sweep.CSV_COLUMNS)
    assert results.exists()


# ---------------------------------------------------------------------------
# thermal parsing and the discard rule
# ---------------------------------------------------------------------------

def test_throttling_log_is_discarded():
    stats = summarize(parse_clock_log(FIXTURES / "clocks_synthetic.csv"))
    assert stats["discard"] is True
    assert stats["mean_sm_clock_mhz"] < 0.85 * stats["opening_sm_mhz"]
    assert stats["max_temp_c"] > 70


def test_flat_log_is_kept():
    stats = summarize(parse_clock_log(FIXTURES / "clocks_flat.csv"))
    assert stats["discard"] is False
    assert 2300 < stats["mean_sm_clock_mhz"] < 2500


def test_clock_log_columns_and_units_are_stripped():
    frame = parse_clock_log(FIXTURES / "clocks_flat.csv")
    assert list(frame.columns) == ["t", "sm_mhz", "temp_c", "power_w", "util_pct"]
    assert frame["t"].iloc[0] == 0.0
    assert frame["t"].is_monotonic_increasing
    for column in ("sm_mhz", "temp_c", "power_w", "util_pct"):
        # units stripped and parsed as numbers; int64 for whole-number columns
        assert frame[column].dtype.kind in "if", (column, frame[column].dtype)


def test_summarize_window_falls_back_when_too_few_samples():
    """A window narrower than a few polls would decide on noise; use it all."""
    frame = parse_clock_log(FIXTURES / "clocks_synthetic.csv")
    windowed = summarize(frame, window=(0.0, 0.5))
    assert windowed["n_samples"] == len(frame)


def test_summarize_window_selects_a_subrange():
    frame = parse_clock_log(FIXTURES / "clocks_synthetic.csv")
    tail = summarize(frame, window=(15.0, 29.0))
    assert tail["n_samples"] < len(frame)
    assert tail["mean_sm_clock_mhz"] < 1800  # the throttled plateau only


def test_thermal_logger_is_a_no_op_without_nvidia_smi(tmp_path):
    from bench.thermal import ThermalLogger, nvidia_smi_path

    if nvidia_smi_path() is not None:
        pytest.skip("this machine has nvidia-smi; the no-op path is not exercised")

    with ThermalLogger(log_dir=tmp_path, strategy="baseline") as logger:
        logger.mark("timing_start")
        logger.mark("timing_end")
    assert logger.available is False
    assert logger.path is None
    assert logger.summarize() is None
    assert logger.window("timing_start", "timing_end") is None
