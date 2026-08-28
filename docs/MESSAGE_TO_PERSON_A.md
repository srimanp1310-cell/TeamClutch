# To Person A — replies, and two bugs your data found

All pushed to `main`. `pytest -q` green: 236 passed, 2 skipped, 1 xpassed.

---

## 1. `SUPPORTED_DTYPES` — done

Declare it on the class and it is honoured in **two** places:

```python
@register("sdpa")
class SdpaTransformer(BaselineTransformer):
    SUPPORTED_DTYPES = (torch.float32, torch.float16)
```

- `tests/test_strategies.py` skips that dtype with a visible reason, naming what
  the strategy does claim.
- **`src/dispatch.py` will never select the strategy for that dtype** — it falls
  through to the next allowed candidate, then to `baseline`.

Untested *and* unreachable is the only safe combination. If it were only skipped
in tests, the dispatcher could still route a bf16 tensor to it in production and
return quietly wrong numbers.

Undeclared still means all three. Every strategy must claim `float32` — that's
the dtype the CPU oracle runs in, and a test enforces it.

**This closed a real hole.** With no `results/` directory, the hard-coded
fallback default was selected for *every* dtype. A bf16 input would have gone
straight to a strategy that can't meet the tolerance in bf16 — not a crash, not
a slowdown, a plausible wrong answer. That's the worst failure mode available
and it's now gated.

## 2. Your bf16 analysis — checked, and it's exactly right

I verified it rather than taking it on trust, both analytically and with
`torch.nextafter`:

| | bf16 | fp16 |
|---|---|---|
| mantissa bits | 7 | 10 |
| 1 ULP at magnitude 2.17 | 0.015625 | 0.001953 |
| 2 ULP, relative | **1.440%** | 0.180% |

And the sharpest part of it: `rtol = 0.01` at that magnitude is **1.389 ULP of
bf16** — tighter than the format's own granularity. So the only bf16
implementation that can pass is one reproducing the reference's operation order
bit-for-bit, which is precisely what an optimized kernel must not do.

It's written up as **§7.1 of `TECH_REPORT.md`**, framed as a property of the
benchmark rather than of our kernel, including that we're *more* accurate and
that it would pass at the PDF's `rtol < 0.02`. Agreed on shipping fp16 and
giving up the 3%.

## 3. Two bugs in my code that your real data exposed

**(a) `gpu_busy_fraction` returned 0.0 on your traces.** You warned me and you
were right — no `cat: kernel` entries at all. My parser found no kernels, summed
no busy time, and reported **0.0% GPU busy**, which reads as the most
catastrophic result imaginable rather than as a missing measurement. Now returns
`NaN`, every consumer prints "unmeasurable", and a test asserts it is never 0.0
for a trace with no kernel track. Slightly galling: I'd written three tests for
this failure mode, one per category *spelling* — I never imagined the whole
track being absent.

**(b) Your launch count is low by ~40%, which moves your conclusion.** cuBLAS
submits through the **driver** API (`cuLaunchKernel`), not `cudaLaunchKernel`,
and I checked that those 144 records are *not* nested inside the runtime-API
ones. So:

| | yours | measured |
|---|---|---|
| launches per forward | 67 | **115** |
| per-launch cost | ~5 µs assumed | **15.5 µs measured** |
| small-shape launch share | 2.5% | **8.2%** |

Medium (2.9%) and large (1.3%) are unaffected — definitively not launch-bound,
your conclusion holds. **Small is now "borderline", not settled.** I've banded
the verdict: <5% rules it out, 5–15% is undecided, >15% is launch-bound. And I
flagged in the report that launch share is an *upper bound* — launches are
async, so CPU time inside a launch call doesn't prove the GPU was idle.

Fusion-first still stands, since the shapes where speedup matters are clean.
CUDA graphs stay on the list for the small shape only.

New figure that works on this platform: `results/figures/gpu_launch_overhead.png`.
Busy-% and timeline figures are skipped automatically when there's no kernel
track, rather than emitting an empty chart.

## 4. Roofline — your numbers are in

Placeholders replaced. `MachineSpec` now carries 11.0 (TF32 on) / 5.7 (TF32 off)
/ 22.5 fp16 / 23.2 bf16 / 174.8 GB/s, and my computed ridge points reproduce
yours exactly: **62.9 / 32.6 / 128.7 / 132.7**.

Took your point about one ceiling per dtype — the plot now draws a roof and a
ridge marker per precision, and the report states the counterintuitive
consequence plainly: reduced precision raises the compute roof and does nothing
for bandwidth, so a workload can become **more** memory-bound in bf16.

Worth noting the fp32 ridge barely moved (62.5 placeholder → 62.9 measured), so
the prediction I'd recorded in §1.1 — that the baseline crosses below the ridge
around S=1024 — stands as written, against real numbers.

## 5. Long sequences — yes, and here's exactly what fits

Estimator against a 6 GB card at 75% budget (d=512, H=8, L=6):

| S | B=8 | B=4 | B=2 | B=1 |
|---|---|---|---|---|
| 2048 | 4.70 GiB ✗ | 2.42 ✓ | 1.28 ✓ | 0.71 ✓ |
| 4096 | — | 6.35 ✗ | 4.42 ✓ | 2.28 ✓ |
| 8192 | — | — | — | **9.1 GiB ✗** |

So S=4096 is reachable at B≤2. **S=8192 is where it gets interesting**: the
baseline needs ~9.1 GiB and cannot run at any batch size, but a fused path never
materializes `[B, H, S, S]` at all — so it may well complete. That's a
categorical result and a stronger claim than any ratio: not "ours is faster" but
"ours runs where the reference cannot".

Added `--matrix long` (S ∈ {2048, 4096, 8192} at **batch 1**):

```bash
python bench/sweep.py --strategy sdpa --matrix long
```

Batch 1 deliberately — at B=8 the baseline already exceeds budget at S=2048 and
every interesting row would be SKIPPED before running. The harness already
handles the asymmetric case: if the baseline OOMs and yours completes, it writes
the row with `notes="baseline OOM; optimized completed"` and an empty
`baseline_median_ms`.

## 6. Still outstanding

- **The combined HTML page for the video** — not done yet. It's the next thing I
  pick up unless you'd rather I did something else first. Tell me if the three
  panels you listed are still the right priority order.
- `--matrix accuracy` (12 configs) — needed for the accuracy-vs-depth figure.
- Optimized-path traces, when you have them:
  `logs/trace_{small,medium,large}_optimized.json`. Given (3a) they'll have no
  kernel track either, so the timeline figure stays unavailable — but the launch
  comparison baseline-vs-optimized is still worth having.
- `python docs/check_ready.py --owner A` — 28 report slots still yours, mostly
  §4 (one per rung: hypothesis → before/after → surprise).

## 7. One thing I'd flag

Your `results.csv` has 2 rows, both `baseline`. When you commit the SDPA rows,
`.gitattributes` will union-merge the file automatically — don't hand-resolve it
if git ever does ask.

Also: the TF32 correction from your §4 is written up as **§4 Rung 0.5** in the
report. It's a genuinely good story — a flag that looks like a precision knob
turning out to be a kernel-selection knob, caught only because two tools
disagreed — and it makes the measurement-discipline argument better than
anything I could have written about the harness in the abstract.


---

# Round 2 — Rung 1 and the precision ceiling

## Your Amdahl argument, sharpened

Verified it independently and it's stronger than you put it. At S=512 attention
is **14.3%** of forward FLOPs, ceiling `1/(1-0.143) = 1.167×`. You measured
1.693×. Inverting Amdahl, that requires attention to have held **41% of the
runtime** — **2.9× its share of the arithmetic**.

That ratio is the headline. A region can't consume triple its arithmetic share
unless it's waiting on something other than arithmetic, so it's direct evidence
the cost was the `[B,H,S,S]` memory traffic. It's in the report as Rung 1's
"surprise", and there's now `analysis/roofline.py::amdahl_ceiling()` and
`implied_time_share()` so the numbers regenerate rather than being quoted.

Your S=128 result is the same argument inverted: ceiling there is only 1.042×,
so there's almost nothing to win and SDPA's own overhead makes it a net loss.
Perfectly consistent.

## Crossover — S=1024 was already in `--matrix seq`

You ran `--matrix quick`, which is only {128, 512}. `seq` has always had 1024
and 2048. I've added **256** to it, and added a dedicated matrix that samples
the region densely:

```bash
python bench/sweep.py --strategy sdpa --matrix crossover
```

S ∈ {128, 192, 256, 320, 384, 512, 768, 1024} — 8 configs. That locates the sign
change properly instead of bracketing it.

## Your fp32 numbers look inconsistent — worth a check

You reported both:
- "fp32 max_abs is 0.0075, 7.5× the atol budget"
- "fp32 error compounds 0.00024 → 0.00084 across 1→6 layers"

0.00084 and 0.0075 differ by 9×. Different shapes, presumably (the depth sweep
at a smaller S?). I've written both into the report attributed to their own
context, but confirm which config each came from so §7.2 says the right thing.

## Your √L is closer to L^0.70

0.00084/0.00024 = 3.5 over 6 layers. √L would predict 2.45; linear would predict
6.0. The exponent is `log(3.5)/log(6) = 0.70`.

So errors are **partially correlated** — not the independent random walk √L
assumes, but well short of worst-case accumulation. Worth having precisely,
because it changes extrapolation: a 12-layer model projects to ~0.0014 rather
than the 0.0012 a √L rule gives. Written up that way in §7.2.

## Declare fp32-only

Given fp16 fails at 2+ layers and the benchmark defaults to 6:

```python
SUPPORTED_DTYPES = (torch.float32,)
```

One caveat worth knowing: `SUPPORTED_DTYPES` is a static class attribute, so it
can't express "fp16 is fine at 1 layer". Admissibility turns out to depend on
**depth**, which the dispatcher deliberately doesn't key on — I excluded layers
from `DispatchKey` on the reasoning that depth changes how long a forward takes
but not which kernel suits a shape. That reasoning holds for performance and
fails for correctness.

Rather than adding depth to the dispatch key, declare conservatively for the
deepest config we ship. A safe answer at every depth beats a fast one that
depends on an axis the dispatcher can't see. Explained in §7.2.

## Roofline — unreachable roofs, as you asked

fp16 and bf16 roofs now draw dotted and muted, labelled
"unreachable: fails tolerance at depth", with their ridge markers greyed.
`SHIPPABLE_DTYPES = ("float32",)` in `analysis/roofline.py` drives it — change
that constant if anything becomes shippable again.

Also flagged in §11 that this is the single largest piece of performance left on
the table: the card offers ~2× in fp16/bf16 and the tolerance means we can't
touch any of it.

## The HTML page — built

`python -m analysis.make_all` now also writes **`results/report.html`**. Fully
self-contained (figures embedded as base64), so it opens anywhere and
screen-records cleanly. Five sections in your priority order, plus the Amdahl
comparison because it's the strongest thing we have:

1. Speedup vs sequence length + the crossover, with the "why dispatch exists" line
2. Amdahl: measured vs ceiling, and the implied runtime share
3. Roofline, unreachable roofs marked
4. The precision ceiling — dtype × depth grid
5. Thermal + the discard/skip/OOM counts

Everything numeric is computed from `results.csv` at generation time; nothing is
hard-coded. Right now it renders mostly empty states, because **your Rung 1 rows
aren't pushed yet** — the message arrived but not the data. Push
`results/results.csv` and the page fills in on the next `make_all`.

One exception I want to flag: the dtype × depth grid **is** currently hard-coded
from your message, because those runs aren't in the shared log. It's marked as
such in the source. Push the `--matrix accuracy` rows and I'll derive it.

## What I need

1. **Push your Rung 1 results.** Everything downstream is waiting on the rows.
2. `--matrix crossover` and `--matrix accuracy`.
3. Confirm the fp32 max_abs discrepancy above.
4. `python docs/check_ready.py --owner A` — 20 slots left, mostly §4 per-rung.
