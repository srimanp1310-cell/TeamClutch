# Devpost submission text

> Copy each section into the matching Devpost field. Slots marked `<FILL …>`
> need a real value before submitting — run `python docs/check_ready.py`.

---

## Project name

Shapeshift

## Elevator pitch (one line)

A Transformer layer that picks its own kernel — 1.32x average across 15 shapes,
and nothing slower than baseline.

---

## How the solution addresses the problem statement

The task gives a fixed Transformer, one replaceable method
(`UserOptimizedTransformer.forward()`), a correctness rule — every output
element within `abs_err ≤ 0.001` **or** `rel_err ≤ 1%` — and a score that is the
ratio of median latencies.

We treated it as two problems rather than one.

**Making it fast.** The quadratic cost of attention is not intrinsic:
`softmax(QKᵀ)V` cannot be re-associated into `Q(KᵀV)` only because the softmax
sits between them, which forces the `S × S` score matrix to exist. The reference
implementation materializes that matrix explicitly, and in fp32 for stability,
so it moves roughly `2e + 8` bytes per score element per layer. That single fact
predicts the whole optimization ladder, and the analytic numbers bear it out:
as sequence length grows from 512 to 4096, the reference's arithmetic intensity
*falls* from 64.6 to 32.3 FLOP/byte — sliding below the card's 62.5 FLOP/byte
ridge point into the bandwidth-bound region — while a fused path's intensity
*rises* from 113.8 to 237.4. So we profiled first, confirmed which bottleneck we
actually had at each shape, and optimized in that order: we fixed a baseline that
was itself mismeasured (TF32 was off in the profiler and on in the harness, worth
18–27%), replaced the four-kernel attention chain with a single fused
`scaled_dot_product_attention` call, tried reduced precision and abandoned it
when the accuracy oracle refused it at depth, declined CUDA graphs because the
launch budget said the ceiling was too low to be worth a day, and then routed
per shape once we could see that no single kernel wins everywhere.

**Making the number believable.** The problem statement says test cases will
span "large/small batchsize, large/small sequence length, large/small
dimensions", and explicitly invites choosing different implementations per shape.
That makes the measurement apparatus part of the deliverable, so we built it:

- an append-only results log where every row carries the git SHA it was produced
  at, and the harness refuses to run against a dirty tree;
- a memory pre-check that predicts an OOM *before* it happens, so an impossible
  configuration is recorded as `SKIPPED` with numbers rather than killing an
  unattended sweep;
- GPU clock and temperature logging with a mechanical discard rule — a run whose
  clock falls below 85% of its opening clock is retried once and then excluded
  from every statistic, because this is a laptop GPU that throttles;
- a correctness oracle that runs on a *different machine with no GPU*, crossing
  every implementation against non-power-of-two shapes and both mask branches;
- and a shape-and-capability dispatch layer built from the measurements, which
  can only ever pick a strategy that is registered and supported by the card.

The harness also carries the check that would make us look worse: how much of
the speedup survives when the *baseline* is compiled too. Some of any gain over
an eager baseline is just `torch.compile` doing what it does to any model, and
reporting only the flattering number is the easiest way to overstate a result.
The `--compile-baseline` path is built, tested and wired into the results
schema — but we ran out of GPU time before producing rows with it, so **we have
no number to report there yet**, and we would rather say so than quote the eager
comparison as though it were the whole story.

## Results

**On the organizers' own script, unmodified.** `bench/run_official.py
--batch-size 8 --seq-len 1024`, fp32, 6 layers:

```
baseline : median=135.4051 ms | throughput= 60,500 token/s
optimized: median= 72.7521 ms | throughput=112,602 token/s
speedup  : 1.861x    accuracy: PASS — 0 of 20,971,520 elements outside tolerance
```

**Across the measured matrix**, the shipped router averages **1.322x** over
**15 configurations** — geometric mean, because speedups are ratios — with a
minimum of **0.984x** and **zero accuracy failures**. Best single shape:
**2.487x** at `d_model=256`.

| seq_len | baseline (control) | sdpa | optimized (shipped router) |
|---|---|---|---|
| 128 | 0.99x | 0.91x | 0.98x |
| 512 | 1.00x | 1.58x | 1.45x |
| 1024 | — | 2.08x | 1.88x |

**Why 1.322x and not 1.531x.** Fused attention on its own averages 1.531x, and
that is the more flattering number, but it is computed over only the 12
configurations where fused attention is *correct*. It is wrong on 9 more — bf16
at every depth, fp16 from two layers up, and causal fp32 at six layers — and
those simply leave the denominator. The router's 1.322x covers all 15 with
nothing dropped, because where the fast path is unsafe it returns the baseline's
own answer at ~1.00x rather than a wrong answer quickly. One number is a
complete result; the other is an average over the survivors.

Correctness: every reported configuration passes the elementwise oracle at
`atol = 0.001, rtol = 0.01` — the stricter of the two published tolerance pairs,
not the problem statement's `0.002 / 0.02`. Across all 15 shipped
configurations the worst element is **0.000798** absolute, **79.8% of the atol
budget** — a 20% margin against the tighter rule, and a 2.5x margin against the
one the problem statement actually specifies.

**Memory.** At S=1024 the fused path's peak is **2.46x lower** — 328.4 MB
against the reference's 808.5 MB — and the gap widens with sequence length
(1.02x at S=128, 1.47x at S=512) because the reference materializes the
`[B, H, S, S]` score matrix and a fused kernel never does. At S=2048 that
becomes categorical rather than incremental: the memory pre-check estimates the
reference needs **4.70 GiB** and refuses the configuration on this 6 GB card,
recording it as `SKIPPED` with the arithmetic attached rather than crashing an
unattended sweep.

## What we learned

**The baseline was wrong before we optimized anything.** Our profiler reported
18.6 / 64.6 / 215.2 ms; the harness reported 13.5 / 52.1 / 176.5 ms for the same
code. The profiling script never set `allow_tf32=True`, which the organizers'
`main()` sets by default. Worth 18–27% — and the kernel count dropped 42% at the
same time, because TF32 is a *kernel-selection* knob, not just a precision knob.
We had two contradictory baselines and, for a while, no idea which was real. It
was caught only because two tools disagreed.

**Being more accurate is what makes you fail.** bf16 misses the tolerance by
*exactly* 2 units in the last place — 1.44% relative against a 1% bound. The
reference rounds softmax probabilities to bf16 before the PV matmul and a fused
kernel does not, so we are closer to the true answer and fail anyway. At that
magnitude `rtol = 0.01` is 1.389 ULP of bf16, tighter than the format's own
granularity, so the only bf16 implementation that could pass is one reproducing
the reference's operation order bit-for-bit — precisely what an optimized kernel
must not do. The card offers roughly 2x in fp16/bf16 and the accuracy rule puts
all of it out of reach.

**We caught ourselves shipping a false positive.** A fix for a marginal causal
case looked like it worked. It was numerically inert — monkeypatching it out
produced bit-identical output across 40 seeds. It had "passed" on seed luck:
21 of 40 seeds, a coin flip on the judge's RNG. That configuration now routes to
the baseline, and the episode is why the accuracy figures here are quoted per
configuration with failures included rather than averaged over survivors.

**Two questions we could not close, stated as such.** The slot this section
started from asked which shape turned out to be launch-bound, and how much of
the speedup was the compiler rather than the kernels. Neither has an honest
answer yet. Launch overhead is 3.45% and 0.80% at the medium and large shapes —
ruled out — but **13.75%** at the smallest, which sits in the undecided band;
settling it needs a device-side kernel timeline, and WSL2's CUPTI does not
produce one. The compiled-baseline comparison is built and wired into the
harness but has **no rows yet**, so we report no number for it rather than a
flattering guess.

## Challenges

One GPU between two people, and a laptop one that throttles under sustained
load. That shaped the whole project: one person owned the card and the kernels,
the other built everything that had to run without it — harness, correctness
suite, dispatch, analysis — and the thermal methodology exists because the
alternative was publishing a downward trend that was really just heat.

---

## Built with

**Languages** — Python

**Frameworks and libraries** — PyTorch (`torch.nn.functional.scaled_dot_product_attention`,
`torch.compile`, `torch.profiler`, `torch.cuda` memory and event APIs)
· Triton 3.2.0 (installed, not used — see the report's §11.1) · NumPy · pandas · Matplotlib · pytest

**Platforms** — NVIDIA CUDA · WSL2 · macOS (Apple Silicon, for the CPU
correctness suite and all analysis)

**Development tools** — VS Code · Git · Jupyter · `nvidia-smi` ·
`torch.profiler` (Nsight Systems not used — WSL2's CUPTI does not populate device-side kernel events)

**APIs / AI services** — Claude, via Claude Code (Opus 5) and Claude chat. No
other AI tools were used. Per-session usage log with corrections:
`docs/AI_USAGE.md` in the repository.

**Datasets** — None. All inputs are synthetic tensors from the organizers'
own `generate_random_case()`, seeded for reproducibility. No external data,
no pretrained weights.

---

## Links

- **Repository:** <https://github.com/srimanp1310-cell/TeamClutch>
- **Demo video (YouTube, public):** <https://youtu.be/FwJuWa1SSwo>

## Try it yourself

```bash
git clone https://github.com/srimanp1310-cell/TeamClutch && cd <repo>
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

pytest -q                                                   # correctness, no GPU needed
python bench/run_official.py --batch-size 8 --seq-len 1024   # the organizers' own script
python -m analysis.make_all                                  # regenerate every figure
```

`bench/run_official.py` runs the organizers' benchmark **unmodified** with our
implementation injected — the scoring code stays theirs. Their file's SHA-256 is
pinned in `docs/INTERFACE.md` and a test fails if it ever changes.

## Team

- **De Xun** — optimized kernel implementations, GPU measurement, profiling.
- **K Sriman Preeth** — measurement harness, correctness suite, dispatch layer,
  analysis and figures, report.
