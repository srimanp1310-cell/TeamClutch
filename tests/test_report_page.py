"""The single-page HTML report: structure, self-containment, honest empty states."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from analysis.report_page import build_page, write_page

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
SYNTHETIC = FIXTURES / "results_synthetic.csv"
FIGURES = ROOT / "results" / "figures"


@pytest.fixture(scope="module")
def page() -> str:
    return build_page(SYNTHETIC, FIGURES)


def test_page_is_self_contained(page):
    """No relative paths and no external hosts except the font stylesheet:
    the file has to survive being copied anywhere and screen-recorded."""
    assert "data:image/png;base64," in page
    external = re.findall(r'(?:src|href)="(https?://[^"]+)"', page)
    for url in external:
        assert url.startswith("https://fonts.g"), f"external dependency: {url}"
    assert 'src="results/' not in page and 'src="../' not in page


def test_every_tag_is_closed(page):
    for tag in ("div", "section", "table", "figure", "header", "footer", "tbody"):
        opened = len(re.findall(rf"<{tag}[\s>]", page))
        assert opened == page.count(f"</{tag}>"), f"<{tag}> is unbalanced"


def test_no_template_leakage(page):
    """Check the markup, not the payload.

    Embedded base64 is arbitrary alphanumeric data and will contain any short
    token you search for by coincidence — "FILL" and "nan" both occur naturally.
    Strip the data URIs before looking, or this test fails at random depending
    on what the figures happen to encode to.
    """
    markup = re.sub(r"data:image/png;base64,[A-Za-z0-9+/=]+", "", page)
    for marker in ("{r[", "__wrapped__", "None×", "<FILL", "TODO"):
        assert marker not in markup, f"template leakage: {marker}"


def test_numbers_come_from_the_results_file(page):
    """The page must reflect the data it was given, not hard-coded results."""
    from analysis.load import load_results, speedup_summary

    best = [s for s in speedup_summary(load_results(SYNTHETIC))
            if s.strategy != "baseline"][0]
    assert f"{best.geomean:.3f}×" in page


def test_amdahl_section_flags_an_overshoot(page):
    """The strongest claim in the report: a speedup above the FLOP-share
    ceiling is evidence the bottleneck was memory, not arithmetic."""
    assert "ceiling if time ∝ FLOPs" in page
    assert "implied share of runtime" in page


def test_unreachable_roofs_are_labelled(page):
    assert "unreachable" in page


def test_empty_data_degrades_to_an_instruction_not_a_crash(tmp_path):
    """A half-finished sweep must produce a page that says which command would
    fill the gap, rather than an exception or a blank panel."""
    empty = tmp_path / "empty.csv"
    empty.write_text("timestamp,git_sha,strategy_name,batch,seq_len,d_model,heads,"
                     "layers,dtype,causal,padding_ratio,accuracy_pass,max_abs_err,"
                     "max_rel_err,baseline_median_ms,optimized_median_ms,speedup,"
                     "peak_vram_mb,mean_sm_clock_mhz,max_temp_c,notes,"
                     "baseline_peak_vram_mb,compile_baseline,ffn_dim\n")
    rendered = build_page(empty, tmp_path)
    assert 'class="empty"' in rendered
    assert "bench/sweep.py" in rendered


def test_missing_results_file_still_renders(tmp_path):
    rendered = build_page(tmp_path / "nope.csv", tmp_path)
    assert "<section>" in rendered


def test_write_page_emits_a_complete_document(tmp_path):
    path = write_page(tmp_path / "report.html", SYNTHETIC, FIGURES)
    text = path.read_text()
    assert text.startswith("<!doctype html>")
    assert "<title>" in text and text.rstrip().endswith("</html>")
    assert path.stat().st_size > 20_000
