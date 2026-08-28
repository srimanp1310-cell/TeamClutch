"""Task 8: the submission documents contain what the problem statement requires.

These are cheap structural checks, not prose review. They exist because the
required sections come from the *problem statement* — a missing "Datasets used"
field or README "Limitations" section costs marks directly, and is exactly the
kind of thing that gets deleted during a late edit and never noticed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

REPORT = DOCS / "TECH_REPORT.md"
DEVPOST = DOCS / "DEVPOST.md"
VIDEO = DOCS / "VIDEO_SCRIPT.md"
README = ROOT / "README.md"


def text(path: Path) -> str:
    assert path.exists(), f"{path.relative_to(ROOT)} is missing"
    return path.read_text()


# ---------------------------------------------------------------------------
# tech report — the section order from PLAN_PERSON_B.md Task 8
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("heading", [
    "What the problem actually is",
    "Environment",
    "Rung 0",
    "The optimizations",
    "Shape and device dispatch",
    "Roofline",
    "Accuracy budget",
    "Thermal methodology",
    "The VRAM ceiling",
    "honesty check",
    "Limitations",
    "AI tools used",
])
def test_report_has_every_required_section(heading):
    assert heading in text(REPORT), f"TECH_REPORT.md is missing a section on: {heading}"


def test_report_states_the_tolerance_we_target():
    """The stricter pair, and the fact that it is stricter, must both be said."""
    body = text(REPORT)
    assert "0.001" in body and "0.01" in body
    assert "0.002" in body, "the looser PDF tolerance should be named for contrast"


def test_report_pins_the_organizers_file_hash():
    body = text(REPORT)
    pinned = re.search(r"[0-9a-f]{64}", body)
    assert pinned, "the report should quote the organizers' file SHA-256"

    import hashlib

    actual = hashlib.sha256(
        (ROOT / "bench" / "torch_transformer_benchmark.py").read_bytes()
    ).hexdigest()
    assert pinned.group(0) == actual, "the hash quoted in the report is stale"


def test_report_is_explicit_about_untested_architectures():
    """Claiming performance on hardware we never ran on is the easiest way to
    lose credibility. The report has to say which claims are structural."""
    body = text(REPORT).lower()
    assert "correctness-tested by forced dispatch" in body
    assert "performance is measured only on" in body


def test_report_links_the_figures_it_discusses():
    body = text(REPORT)
    for figure in ("speedup_vs_seq_len", "accuracy_budget", "vram_ceiling",
                   "roofline", "compile_baseline_survival"):
        assert figure in body, f"no reference to {figure}.png"
    # Rung 0 must point at whichever figure the platform can actually produce.
    assert ("gpu_launch_overhead" in body) or ("gpu_busy_vs_shape" in body), (
        "the report discusses Rung 0 but references neither the launch-overhead "
        "figure nor the GPU-busy figure"
    )


def test_report_does_not_claim_an_unmeasurable_gpu_busy_number():
    """CUPTI does not report device kernels under WSL2. If the report ever
    quotes a busy percentage, either the platform changed or someone pasted a
    0% reading as if it were a result."""
    body = text(REPORT)
    assert "unmeasurable" in body.lower()
    assert "CUPTI" in body


# ---------------------------------------------------------------------------
# devpost — the fields the problem statement enumerates
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("field,pattern", [
    ("how it addresses the problem statement", r"addresses the problem"),
    ("development tools", r"[Dd]evelopment tools"),
    ("APIs used", r"APIs"),
    ("libraries and frameworks", r"[Ff]rameworks and libraries"),
    ("datasets used", r"[Dd]atasets"),
    ("repository link", r"[Rr]epository"),
    ("YouTube link", r"YouTube"),
])
def test_devpost_covers_every_required_field(field, pattern):
    assert re.search(pattern, text(DEVPOST)), f"DEVPOST.md has no {field} field"


def test_devpost_states_there_is_no_dataset():
    """"None" is a valid answer and has to be given explicitly — the field is
    required whether or not we used data."""
    body = text(DEVPOST)
    datasets = body[body.index("**Datasets**"):][:400]
    assert "None" in datasets
    assert "synthetic" in datasets.lower()


# ---------------------------------------------------------------------------
# video script
# ---------------------------------------------------------------------------

def test_video_script_is_three_minutes_and_has_timestamps():
    body = text(VIDEO)
    assert "3 minute" in body or "3-minute" in body
    stamps = re.findall(r"\d:\d\d–\d:\d\d", body)
    assert len(stamps) >= 5, f"expected a timed shot list, found {stamps}"


def test_video_script_names_the_exact_commands_to_type():
    body = text(VIDEO)
    for command in ("run_official.py", "pytest -q", "analysis.make_all"):
        assert command in body, f"script never shows {command} on screen"


def test_video_script_carries_the_submission_rules():
    body = text(VIDEO).lower()
    assert "public" in body and "youtube" in body
    assert "trademark" in body or "logo" in body


# ---------------------------------------------------------------------------
# readme — the sections the problem statement requires of the repo
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("heading", [
    "Project overview", "Setup", "Reproduce", "Results",
    "Limitations", "Team contributions", "AI usage",
])
def test_readme_has_every_required_section(heading):
    assert heading in text(README), f"README.md is missing: {heading}"


# ---------------------------------------------------------------------------
# the readiness checker
# ---------------------------------------------------------------------------

def test_check_ready_finds_the_outstanding_placeholders():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "check_ready", DOCS / "check_ready.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    found = module.scan(ROOT)
    assert found, "no documents scanned"
    # Non-zero while placeholders remain, so it can gate a submission checklist.
    assert module.main(["--root", str(ROOT)]) == 1

    owners = {owner for entries in found.values() for _, owner, _ in entries}
    assert {"A", "B"} <= owners, "placeholders should be attributed to a person"


def test_check_ready_reports_success_when_nothing_is_outstanding(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "check_ready", DOCS / "check_ready.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "TECH_REPORT.md").write_text("# Done\n\nAll filled in.\n")
    (tmp_path / "README.md").write_text("# Done\n")
    assert module.main(["--root", str(tmp_path)]) == 0
