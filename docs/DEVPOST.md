# Devpost submission text

> Copy each section into the matching Devpost field. Slots marked `<FILL …>`
> need a real value before submitting — run `python docs/check_ready.py`.

---

## Project name

`<FILL A/B: short, descriptive, no third-party trademarks. e.g. "Shape-Aware
Transformer Inference">`

## Elevator pitch (one line)

A measured, shape-aware Transformer forward pass: `<FILL B: geometric-mean
speedup>`x faster than the reference at identical numerics, with the
measurement apparatus in the repo so every number can be re-derived.

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
actually had at each shape, and optimized in that order: `<FILL A: the rungs, in
one sentence>`.

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

We also report the number that makes us look worse: how much of the speedup
survives when the *baseline* is compiled too. Some of any gain over an eager
baseline is just `torch.compile` doing what it does to any model, and reporting
only the flattering number is the easiest way to overstate a result.

## Results

`<FILL B: geometric-mean speedup, min, max, n configurations, and the headline
per-shape table from results/summary.md.>`

Correctness: every reported configuration passes the elementwise oracle at
`atol = 0.001, rtol = 0.01` — the stricter of the two published tolerance pairs,
not the problem statement's `0.002 / 0.02`. `<FILL B: worst observed error and
the margin.>`

`<FILL B: the VRAM result — the sequence length at which the reference stops
fitting on the card and ours does not.>`

## What we learned

`<FILL A: the genuine surprises. The ones from the measurement side: which shape
turned out to be launch-bound rather than compute-bound, and how much of the
apparent speedup was the compiler rather than the kernels.>`

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
· `<FILL A: Triton, if used>` · NumPy · pandas · Matplotlib · pytest

**Platforms** — NVIDIA CUDA · WSL2 · macOS (Apple Silicon, for the CPU
correctness suite and all analysis)

**Development tools** — VS Code · Git · Jupyter · `nvidia-smi` ·
`<FILL A: Nsight Systems, if used>`

**APIs / AI services** — Claude (via Claude Code) · `<FILL A: Codex or others,
if used>`. Per-session usage log with corrections:
`docs/AI_USAGE.md` in the repository.

**Datasets** — None. All inputs are synthetic tensors from the organizers'
own `generate_random_case()`, seeded for reproducibility. No external data,
no pretrained weights.

---

## Links

- **Repository:** `<FILL A/B: public GitHub URL>`
- **Demo video (YouTube, public):** `<FILL A: URL>`

## Try it yourself

```bash
git clone <FILL A/B: repo URL> && cd <repo>
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

- `<FILL A: name>` — optimized implementations, GPU measurement, profiling.
- `<FILL B: name>` — measurement harness, correctness suite, dispatch layer,
  analysis and figures, report.
