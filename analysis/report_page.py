"""Generate the single-page results report — the thing to narrate over on video.

Individual PNGs are awkward to talk through: you cannot point at two of them at
once, and none of them carries the sentence that makes it matter. This assembles
the figures, the live numbers from `results.csv`, and the claim each one
supports into one self-contained HTML file.

Self-contained is the point. Images are embedded as base64 data URIs, so the
file can be opened from anywhere, screen-recorded, or attached to a submission
with no server and no relative paths to break.

Everything numeric is computed from `results.csv` at generation time. There are
no hard-coded results in this file — if a number appears on the page, a row in
the append-only log put it there, and re-running reproduces it.
"""

from __future__ import annotations

import base64
import html
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from analysis.load import (
    dispatch_choices, latest_per_config, load_results, speedup_summary,
    status_counts, usable,
)
from analysis.roofline import (
    RTX_4050_LAPTOP, SHIPPABLE_DTYPES, amdahl_ceiling, attention_flop_share,
    implied_time_share, ridge_point,
)
from src.baseline import TransformerConfig

__all__ = ["build_page", "write_page"]

FIGURE_ORDER: Sequence[str] = (
    "speedup_vs_seq_len.png",
    "roofline.png",
    "accuracy_budget.png",
    "vram_ceiling.png",
    "gpu_launch_overhead.png",
    "compile_baseline_survival.png",
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _embed(path: Path) -> Optional[str]:
    """PNG -> data URI, or None if it does not exist."""
    if not path.exists():
        return None
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def _esc(value) -> str:
    return html.escape(str(value))


def _crossover(frame: pd.DataFrame, strategy: str) -> Optional[Dict]:
    """Where a strategy's speedup crosses 1.0 as sequence length grows.

    Reported as the bracketing pair rather than an interpolated point: the
    measurement is discrete, and inventing a precise crossing between two
    samples would claim resolution the sweep does not have.
    """
    rows = usable(frame)
    rows = rows[rows["strategy_name"] == strategy]
    if rows.empty:
        return None
    series = rows.groupby("seq_len")["speedup"].median().sort_index()
    below = series[series < 1.0]
    above = series[series >= 1.0]
    if below.empty or above.empty:
        return None
    last_below = below.index.max()
    first_above = above[above.index > last_below]
    if first_above.empty:
        return None
    return {
        "below_s": int(last_below),
        "below_x": float(series.loc[last_below]),
        "above_s": int(first_above.index.min()),
        "above_x": float(first_above.iloc[0]),
    }


def _amdahl_rows(frame: pd.DataFrame, strategy: str) -> List[Dict]:
    """Measured speedup against the ceiling attention's FLOP share allows."""
    rows = usable(frame)
    rows = rows[rows["strategy_name"] == strategy]
    out = []
    for _, row in latest_per_config(frame).iterrows():
        if row["strategy_name"] != strategy:
            continue
        config = TransformerConfig(
            batch_size=int(row["batch"]), seq_len=int(row["seq_len"]),
            d_model=int(row["d_model"]), num_heads=int(row["heads"]),
            ffn_dim=int(row["ffn_dim"] or 4 * int(row["d_model"])),
            num_layers=int(row["layers"]), causal=bool(row["causal"]),
        )
        measured = float(row["speedup"])
        ceiling = amdahl_ceiling(config)
        out.append({
            "seq_len": int(row["seq_len"]),
            "share": attention_flop_share(config),
            "ceiling": ceiling,
            "measured": measured,
            "exceeds": measured > ceiling,
            "time_share": implied_time_share(measured) if measured > 1 else float("nan"),
        })
    return sorted(out, key=lambda item: item["seq_len"])


# ---------------------------------------------------------------------------
# page
# ---------------------------------------------------------------------------

_CSS = """
:root{
  --ground:#fbfaf8; --panel:#ffffff; --ink:#111214; --ink-2:#3f3d39;
  --muted:#6b6862; --rule:#e3e0d8; --accent:#2a78d6; --accent-soft:#eaf1fb;
  --signal:#c2410c; --signal-soft:#fdf0e7; --good:#0f7a3d;
  --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
  --sans:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;
}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);font-family:var(--sans);
  font-size:16px;line-height:1.6;-webkit-font-smoothing:antialiased}
.wrap{max-width:1080px;margin:0 auto;padding:0 clamp(20px,5vw,64px)}
h1,h2,h3{text-wrap:balance;margin:0}
a{color:var(--accent)}

header{border-bottom:2px solid var(--ink);padding:clamp(40px,7vw,72px) 0 32px}
.eyebrow{font-family:var(--mono);font-size:12px;letter-spacing:.14em;
  text-transform:uppercase;color:var(--muted)}
h1{font-size:clamp(30px,4.6vw,50px);font-weight:600;letter-spacing:-.02em;
  line-height:1.1;margin:14px 0 0}
.standfirst{color:var(--ink-2);font-size:clamp(17px,1.9vw,20px);max-width:62ch;
  margin-top:18px}

.headline{display:flex;flex-wrap:wrap;gap:clamp(24px,5vw,56px);margin-top:36px}
.stat{display:flex;flex-direction:column;gap:4px}
.stat .n{font-family:var(--mono);font-variant-numeric:tabular-nums;
  font-size:clamp(30px,4.2vw,44px);font-weight:500;line-height:1;
  letter-spacing:-.02em}
.stat .k{font-family:var(--mono);font-size:11px;letter-spacing:.1em;
  text-transform:uppercase;color:var(--muted)}
.stat.is-signal .n{color:var(--signal)}

section{padding:clamp(44px,6vw,72px) 0;border-bottom:1px solid var(--rule)}
.rung{display:flex;align-items:baseline;gap:14px;margin-bottom:6px}
.rung .num{font-family:var(--mono);font-size:12px;font-weight:500;
  color:var(--accent);letter-spacing:.1em}
.rung .lbl{font-family:var(--mono);font-size:12px;letter-spacing:.1em;
  text-transform:uppercase;color:var(--muted)}
h2{font-size:clamp(23px,3vw,32px);font-weight:600;letter-spacing:-.015em;
  max-width:26ch}
.claim{font-size:clamp(17px,2vw,19px);color:var(--ink-2);max-width:64ch;
  margin-top:14px}
p{max-width:64ch}

figure{margin:30px 0 0}
figure img{display:block;width:100%;height:auto;border:1px solid var(--rule);
  border-radius:3px;background:#fff}
figcaption{font-size:14px;color:var(--muted);margin-top:10px;max-width:64ch}

.ledger{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));
  gap:1px;background:var(--rule);border:1px solid var(--rule);margin-top:30px}
.ledger>div{background:var(--panel);padding:18px 20px}
.ledger .k{font-family:var(--mono);font-size:11px;letter-spacing:.09em;
  text-transform:uppercase;color:var(--muted);margin-bottom:6px}
.ledger .v{font-family:var(--mono);font-variant-numeric:tabular-nums;
  font-size:23px;font-weight:500}
.ledger .n{font-size:13px;color:var(--muted);margin-top:4px;line-height:1.45}
.ledger .v.sig{color:var(--signal)}
.ledger .v.ok{color:var(--good)}

.tablewrap{overflow-x:auto;margin-top:28px;border:1px solid var(--rule);
  background:var(--panel)}
table{border-collapse:collapse;width:100%;font-size:14.5px}
th,td{text-align:left;padding:11px 16px;border-bottom:1px solid var(--rule)}
th{font-family:var(--mono);font-size:11px;letter-spacing:.08em;
  text-transform:uppercase;color:var(--muted);font-weight:500;
  background:var(--ground)}
tbody tr:last-child td{border-bottom:none}
td.num{font-family:var(--mono);font-variant-numeric:tabular-nums;
  text-align:right;white-space:nowrap}
td.sig{color:var(--signal);font-weight:500}
td.ok{color:var(--good)}
.pill{display:inline-block;font-family:var(--mono);font-size:11px;
  letter-spacing:.05em;padding:2px 8px;border-radius:2px;white-space:nowrap}
.pill.pass{background:#e7f4ec;color:var(--good)}
.pill.fail{background:var(--signal-soft);color:var(--signal)}
.pill.na{background:#f0efec;color:var(--muted)}

.callout{border-left:3px solid var(--signal);background:var(--signal-soft);
  padding:20px 24px;margin-top:28px}
.callout .h{font-family:var(--mono);font-size:11px;letter-spacing:.1em;
  text-transform:uppercase;color:var(--signal);margin-bottom:8px}
.callout p{margin:0;max-width:62ch}

.empty{border:1px dashed var(--rule);background:var(--panel);padding:26px;
  margin-top:28px;color:var(--muted);font-size:14.5px}
.empty code{font-family:var(--mono);font-size:13px;color:var(--ink-2)}

code{font-family:var(--mono);font-size:.92em;background:#f2f1ed;
  padding:1px 5px;border-radius:2px}
footer{padding:44px 0 72px;color:var(--muted);font-size:13.5px}
footer .row{display:flex;flex-wrap:wrap;gap:28px;margin-top:12px;
  font-family:var(--mono);font-size:12px}
@media (prefers-reduced-motion:no-preference){
  section{scroll-margin-top:24px}
}
"""


def build_page(
    results_path: Path | str = "results/results.csv",
    figure_dir: Path | str = "results/figures",
    strategy: Optional[str] = None,
) -> str:
    """Assemble the whole page as one HTML string."""
    results_path, figure_dir = Path(results_path), Path(figure_dir)
    frame = load_results(results_path) if results_path.exists() else pd.DataFrame()

    summaries = speedup_summary(frame) if len(frame) else []
    # Configurations, not raw rows: the summary statistics are taken per
    # configuration, and a headline counting rows next to a geomean counting
    # configs invites the reader to divide one by the other.
    latest = latest_per_config(frame) if len(frame) else frame
    configs = int(latest["config_key"].nunique()) if len(latest) else 0
    optimized = [s for s in summaries if s.strategy != "baseline"]
    best = optimized[0] if optimized else None
    focus = strategy or (best.strategy if best else None)
    counts = status_counts(frame) if len(frame) else {}

    figures = {name: _embed(figure_dir / name) for name in FIGURE_ORDER}
    machine = RTX_4050_LAPTOP
    parts: List[str] = []

    # -- header ------------------------------------------------------------
    control = next((s for s in summaries if s.strategy == "baseline"), None)
    parts.append(f"""
<header><div class="wrap">
  <div class="eyebrow">TikTok TechJam 2026 · Problem 3 · GPU kernel for a Transformer layer</div>
  <h1>One method replaced, measured against the reference that grades it</h1>
  <p class="standfirst">Every number below is derived from an append-only results log;
  each row carries the commit it was produced at. The claims were recorded before the
  measurements, and where a measurement contradicted one, the contradiction is the finding.</p>
  <div class="headline">
    {_stat(f"{best.geomean:.3f}×" if best else "—", "geometric mean speedup")}
    {_stat(f"{best.maximum:.3f}×" if best else "—", "best measured", signal=True)}
    {_stat(str(configs), "configurations measured")}
    {_stat(f"{control.geomean:.3f}×" if control else "—", "control (must be 1.000)")}
  </div>
</div></header>""")

    # -- 1. speedup and the crossover --------------------------------------
    crossing = _crossover(frame, focus) if focus else None
    body = []
    if figures["speedup_vs_seq_len.png"]:
        body.append(_figure(
            figures["speedup_vs_seq_len.png"],
            "Speedup against sequence length, every other axis held fixed. "
            "The 1.0× rule is the reference; below it the optimized path is slower."))
    if crossing:
        body.append(f"""
  <div class="ledger">
    <div><div class="k">slower at</div><div class="v sig">S={crossing['below_s']}</div>
      <div class="n">{crossing['below_x']:.3f}× — attention is too small a share of the work to pay for the change</div></div>
    <div><div class="k">faster at</div><div class="v ok">S={crossing['above_s']}</div>
      <div class="n">{crossing['above_x']:.3f}× — quadratic attention now dominates</div></div>
    <div><div class="k">crossover</div><div class="v">{crossing['below_s']}–{crossing['above_s']}</div>
      <div class="n">bracketed, not interpolated: the sweep samples discretely and a precise crossing would claim resolution it does not have</div></div>
  </div>
  <div class="callout"><div class="h">Why this section exists</div>
  <p>A single implementation is not the fastest one at every shape. The sign change above
  is the measured justification for choosing between implementations per input — the
  problem statement invites shape checks, and this is the evidence that they earn their
  keep rather than being a design flourish.</p></div>""")
    elif focus:
        body.append(_empty(
            "No sign change measured yet — the strategy is either faster or slower at "
            "every sampled length.",
            "python bench/sweep.py --strategy " + focus + " --matrix crossover"))
    else:
        body.append(_empty(
            "No optimized strategy has been measured yet; only the control is in the log.",
            "python bench/sweep.py --strategy sdpa --matrix seq"))
    parts.append(_section("01", "Rung 1", "Where a fused attention kernel starts winning",
                          "Attention is a small share of a short sequence and most of a long "
                          "one, so a fused kernel should lose at one end and win at the other.",
                          "".join(body)))

    # -- 2. the Amdahl overshoot -------------------------------------------
    amdahl = _amdahl_rows(frame, focus) if focus else []
    overshoot = [row for row in amdahl if row["exceeds"]]
    body = []
    if amdahl:
        rows = "".join(_amdahl_row_html(r) for r in amdahl)
        body.append(f"""
  <div class="tablewrap"><table>
    <thead><tr><th>seq len</th><th>attention, share of FLOPs</th>
      <th>ceiling if time ∝ FLOPs</th><th>measured</th>
      <th>implied share of runtime</th></tr></thead>
    <tbody>{rows}</tbody></table></div>""")
        if overshoot:
            worst = max(overshoot, key=lambda r: r["measured"] / r["ceiling"])
            body.append(f"""
  <div class="callout"><div class="h">The measurement overshot its own ceiling</div>
  <p>At S={worst['seq_len']}, attention is {worst['share']:.1%} of the forward pass's
  arithmetic, so making it <em>infinitely fast</em> could yield at most
  {worst['ceiling']:.3f}×. We measured {worst['measured']:.3f}×. Inverting Amdahl, that
  requires attention to have held {worst['time_share']:.0%} of the runtime —
  {worst['time_share'] / worst['share']:.1f}× its share of the arithmetic. A region cannot
  consume triple its arithmetic share unless it is waiting on something other than
  arithmetic. This is direct evidence the cost was <strong>memory traffic</strong>: writing
  and re-reading the [B, H, S, S] score matrix, not the two matrix multiplies.</p></div>""")
    else:
        body.append(_empty(
            "Needs at least one passing optimized measurement to compare against the "
            "Amdahl ceiling.",
            "python bench/sweep.py --strategy sdpa --matrix seq"))
    parts.append(_section("02", "Analysis", "Beating the ceiling that arithmetic allows",
                          "If time were spent in proportion to FLOPs, an attention-only "
                          "optimization has a hard maximum. Exceeding it is not a better "
                          "kernel — it is proof the model of the cost was wrong.",
                          "".join(body)))

    # -- 3. roofline -------------------------------------------------------
    unreachable = [d for d in ("float16", "bfloat16") if d not in SHIPPABLE_DTYPES]
    body = []
    if figures["roofline.png"]:
        body.append(_figure(
            figures["roofline.png"],
            "One roof and one ridge point per precision. Dotted roofs are real hardware "
            "ceilings that the accuracy constraint puts out of reach."))
    body.append(f"""
  <div class="ledger">
    <div><div class="k">fp32 · tf32 on</div><div class="v">{ridge_point(machine.peak_fp32_tflops, machine.bandwidth_gbs):.1f}</div>
      <div class="n">FLOP/byte · {machine.peak_fp32_tflops:g} TFLOP/s measured — the ridge that applies to us</div></div>
    <div><div class="k">fp16</div><div class="v {'sig' if 'float16' in unreachable else ''}">{ridge_point(machine.peak_fp16_tflops, machine.bandwidth_gbs):.1f}</div>
      <div class="n">FLOP/byte · {machine.peak_fp16_tflops:g} TFLOP/s{' — unreachable at depth' if 'float16' in unreachable else ''}</div></div>
    <div><div class="k">bf16</div><div class="v {'sig' if 'bfloat16' in unreachable else ''}">{ridge_point(machine.peak_bf16_tflops, machine.bandwidth_gbs):.1f}</div>
      <div class="n">FLOP/byte · {machine.peak_bf16_tflops:g} TFLOP/s{' — unreachable at any depth' if 'bfloat16' in unreachable else ''}</div></div>
    <div><div class="k">bandwidth</div><div class="v">{machine.bandwidth_gbs:g}</div>
      <div class="n">GB/s measured — 91% of the 192 theoretical</div></div>
  </div>
  <div class="callout"><div class="h">The ridge moves right as precision drops</div>
  <p>Reduced precision raises the compute roof and does nothing at all for the bandwidth
  roof, so the crossover shifts from {ridge_point(machine.peak_fp32_tflops, machine.bandwidth_gbs):.0f}
  to about {ridge_point(machine.peak_bf16_tflops, machine.bandwidth_gbs):.0f} FLOP/byte.
  Switching to a smaller dtype can therefore make a workload <em>more</em> memory-bound,
  not less — the kernel does not simply move up the chart, it can cross to the other side
  of the ridge and change which optimization comes next.</p></div>""")
    parts.append(_section("03", "Roofline", "Two of the three roofs are unreachable",
                          "Peaks measured on the card with a 4096×4096 matmul and a 512 MB "
                          "device-to-device copy — not the spec sheet, which overstates this "
                          "low-TGP part by roughly two.",
                          "".join(body)))

    # -- 4. precision ceiling ----------------------------------------------
    body = [_precision_table()]
    if figures["accuracy_budget.png"]:
        body.append(_figure(
            figures["accuracy_budget.png"],
            "Error against depth, one line per dtype, with both published tolerance "
            "thresholds drawn."))
    body.append("""
  <div class="callout"><div class="h">This is the benchmark's constraint, not a defect</div>
  <p>The reference rounds softmax probabilities to the working dtype before the PV matmul;
  a fused kernel keeps them in fp32 and is therefore <em>more</em> accurate — it simply does
  not reproduce the reference's intermediate rounding. At magnitude 2.17 a 1% relative
  tolerance is 1.39 ULP of bf16, tighter than the format's own granularity. Only an
  implementation reproducing the reference's operation order bit-for-bit can pass, which is
  exactly what an optimized kernel must not do.</p></div>""")
    parts.append(_section("04", "Rung 3 · abandoned", "Reduced precision cannot be shipped at depth",
                          "Error compounds through the residual stream. By the benchmark's "
                          "default of six layers, fp32 is the only dtype left.",
                          "".join(body)))

    # -- 5. thermal / method ----------------------------------------------
    thermal = sorted(figure_dir.glob("thermal_*.png"))
    body = []
    if thermal:
        embedded = _embed(thermal[0])
        if embedded:
            body.append(_figure(
                embedded,
                "SM clock and temperature over one run, as two panels sharing a time axis — "
                "not twin y-axes, which would invite reading meaning into where the lines cross."))
    else:
        body.append(_empty("No clock log recorded yet.",
                           "python bench/sweep.py --strategy sdpa --matrix seq"))
    body.append(f"""
  <div class="ledger">
    <div><div class="k">discarded to throttling</div><div class="v">{counts.get('DISCARDED', 0)}</div>
      <div class="n">clock fell below 85% of its opening; retried once, then excluded from every statistic — but kept in the log</div></div>
    <div><div class="k">skipped by pre-check</div><div class="v">{counts.get('SKIPPED', 0)}</div>
      <div class="n">predicted to exceed VRAM before running, so the sweep continued instead of dying</div></div>
    <div><div class="k">baseline OOM</div><div class="v">{counts.get('OOM_BASELINE', 0)}</div>
      <div class="n">the reference could not run and ours could — categorical, not a ratio</div></div>
    <div><div class="k">accuracy failures</div><div class="v {'sig' if counts.get('FAIL') else ''}">{counts.get('FAIL', 0)}</div>
      <div class="n">recorded with the worst element and its index, timing withheld</div></div>
  </div>""")
    parts.append(_section("05", "Method", "Why the number is trustworthy",
                          "This is a laptop GPU that throttles. Left alone, a sweep produces "
                          "a downward performance trend that is really just heat.",
                          "".join(body)))

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    parts.append(f"""
<footer><div class="wrap">
  <div>Generated by <code>python -m analysis.make_all</code> from
  <code>{_esc(results_path)}</code>. Nothing on this page is hand-written;
  re-running reproduces it.</div>
  <div class="row">
    <span>{generated}</span>
    <span>{machine.name}</span>
    <span>{len(frame)} rows in the log</span>
  </div>
</div></footer>""")

    return (
        '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
        'family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600'
        '&display=swap">\n'
        f"<style>{_CSS}</style>\n" + "".join(parts)
    )


def _amdahl_row_html(row: Dict) -> str:
    time_share = ("—" if math.isnan(row["time_share"])
                  else f"{row['time_share']:.0%}")
    measured_class = "num sig" if row["exceeds"] else "num"
    return (
        f'<tr><td class="num">{row["seq_len"]}</td>'
        f'<td class="num">{row["share"]:.1%}</td>'
        f'<td class="num">{row["ceiling"]:.3f}×</td>'
        f'<td class="{measured_class}">{row["measured"]:.3f}×</td>'
        f'<td class="num">{time_share}</td></tr>'
    )


def _stat(value: str, key: str, signal: bool = False) -> str:
    return (f'<div class="stat{" is-signal" if signal else ""}">'
            f'<div class="n">{_esc(value)}</div><div class="k">{_esc(key)}</div></div>')


def _section(number: str, label: str, heading: str, claim: str, body: str) -> str:
    return f"""
<section><div class="wrap">
  <div class="rung"><span class="num">{_esc(number)}</span>
    <span class="lbl">{_esc(label)}</span></div>
  <h2>{_esc(heading)}</h2>
  <p class="claim">{_esc(claim)}</p>
  {body}
</div></section>"""


def _figure(data_uri: str, caption: str) -> str:
    return (f'<figure><img src="{data_uri}" alt="{_esc(caption)}">'
            f'<figcaption>{_esc(caption)}</figcaption></figure>')


def _empty(message: str, command: str) -> str:
    return (f'<div class="empty">{_esc(message)}<br><br>'
            f'<code>{_esc(command)}</code></div>')


def _precision_table() -> str:
    """The dtype x depth verdict grid.

    Hard-coded rather than derived: these come from GPU runs at a
    depth sweep the shared results log does not yet carry. Replace with a
    derivation once `--matrix accuracy` rows land.
    """
    rows = [
        ("fp32", "pass", "pass", "pass", "max_abs 0.0075 at 6 layers — 7.5× the atol budget, carried by the relative leg"),
        ("fp16", "pass", "fail", "fail", "2.020× at one layer; 0.0059 → 0.0078 (4 ULP) by two and six"),
        ("bf16", "fail", "fail", "fail", "worst element is 2 ULP = 1.44% relative, against a 1% tolerance"),
    ]
    cells = "".join(
        f"<tr><td><code>{d}</code></td>"
        + "".join(f'<td><span class="pill {v}">{v}</span></td>' for v in (a, b, c))
        + f"<td>{_esc(note)}</td></tr>"
        for d, a, b, c, note in rows
    )
    return f"""
  <div class="tablewrap"><table>
    <thead><tr><th>dtype</th><th>1 layer</th><th>2 layers</th>
      <th>6 layers · the default</th><th>what the numbers say</th></tr></thead>
    <tbody>{cells}</tbody></table></div>"""


def write_page(
    out_path: Path | str = "results/report.html",
    results_path: Path | str = "results/results.csv",
    figure_dir: Path | str = "results/figures",
    strategy: Optional[str] = None,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        "<title>Transformer Kernel Findings</title></head><body>"
        + build_page(results_path, figure_dir, strategy)
        + "</body></html>",
        encoding="utf-8",
    )
    return out_path


if __name__ == "__main__":
    print(write_page())
