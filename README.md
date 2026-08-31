# Transformer GPU Kernel Optimization — TikTok TechJam 2026, Task 3

> **Status: complete and reproducible from a clean clone.** Verified end to end
> on 2026-08-31: install, `pytest` (286 passed, 5 skipped), the control sweep,
> the organizers' own script, and full figure regeneration. Measured numbers
> live in `results/results.csv` (append-only) and are summarised in
> `results/summary.md` and `results/report.html`.

## Project overview

The organizers provide a fixed Transformer and a correct-but-slow reference
implementation of its forward pass (`bench/torch_transformer_benchmark.py`).
The task is to replace one method — `UserOptimizedTransformer.forward()` — with
a faster implementation that produces the same numbers, where "the same" means
every output element satisfies `abs_err <= 0.001` **or** `rel_err <= 1%`. The
score is `speedup = baseline_median_ms / optimized_median_ms`.

This repo contains both halves of that work: the optimized implementations, and
the measurement apparatus that makes their speedups believable — an append-only
results log, a CPU-runnable correctness oracle, a memory pre-check, thermal
monitoring, and a shape-and-capability dispatch layer that picks the best
implementation per input shape (which the problem statement explicitly invites).

## Environment

All measured results come from one machine:

| | |
|---|---|
| CPU | Intel Core i7-12650HX (12th gen) |
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU (Ada Lovelace, low-TGP) — `sm_89` |
| VRAM | 6.0 GB GDDR6 |
| Driver / CUDA | GeForce Game Ready 616.56, CUDA 12.4 |
| OS | Windows 11 + WSL2, Ubuntu 24.04 LTS |
| PyTorch | 2.6.0+cu124 |
| Triton | 3.2.0 installed, not used |

Measured peaks used for the roofline — **measured on the card, not spec sheet**,
which for this low-TGP part is roughly half the published figures: fp32 with
TF32 on **11.0 TFLOP/s** (the benchmark defaults to `--allow-tf32`), fp32 TF32
off 5.7, fp16 22.5, bf16 23.2, memory bandwidth **174.8 GB/s**. Full table and
the reasoning in [docs/TECH_REPORT.md](docs/TECH_REPORT.md) §2.

The harness, CPU test suite and all analysis also run on macOS with no GPU
(`torch.cuda.is_available() == False`); GPU-only paths degrade rather than
crash.

## Setup

**macOS / CPU (tests, analysis, harness development):**

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Python 3.11 is not incidental — at time of writing there is no PyTorch wheel for
the 3.14 that ships as `python3` on a current macOS, and the install fails with
a source build. Use 3.10–3.12.

**Linux / WSL2 with an NVIDIA GPU (all measured results):**

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124   # match your driver
pip install -r requirements.txt
pip install -e .
```

The cu124 index URL above is the one all measured results were produced with;
it installs `torch 2.6.0+cu124` and pulls `triton 3.2.0` as a dependency. Run
`pip install -r requirements.txt` **after** the CUDA torch — it lists a bare
`torch`, which pip then leaves alone because the requirement is already
satisfied. Installing it first would give you a CPU or default-CUDA build.

## Reproduce

```bash
pytest                             # correctness oracle — CPU, no GPU needed, ~25 s
python bench/sweep.py --strategy baseline --matrix quick     # GPU. control: every speedup ~1.00x
python bench/run_official.py --batch-size 8 --seq-len 1024   # GPU. organizers' script, our class injected
python -m src.memcheck --batch 8 --seq-len 4096              # will this config OOM? (no GPU needed)
python -m analysis.make_all                                  # regenerate every figure, summary.md, report.html
                                                             # (picks up logs/trace_*.json and logs/clocks_*.csv)
```

**Run `pytest`, not `pytest -q`.** `pyproject.toml` already sets `addopts = "-q"`,
so adding another `-q` becomes `-qq` and suppresses the summary line — you get a
wall of dots and no verdict. Plain `pytest` prints the count. Expected on a clean
clone: **286 passed, 5 skipped**, no failures. Four of the skips are the bfloat16
cases, skipped with a visible reason because the strategies declare
`SUPPORTED_DTYPES` without bf16 (see [docs/INTERFACE.md](docs/INTERFACE.md) §5.1);
the fifth is a thermal-logging path that only runs on a machine without
`nvidia-smi`.

> **`bench/sweep.py` requires a completely clean tree — including untracked
> files.** It refuses to run otherwise, because a row it cannot tie to a commit
> is a row nobody can reproduce. `git_is_dirty()` shells out to a bare
> `git status --porcelain`, so **untracked files count**. Two consequences bite
> in the documented order above:
>
> 1. **`pytest` blocks the next sweep.** The suite regenerates the *tracked* file
>    `results/report.html` (`tests/test_analysis.py` writes to the real
>    `results/` rather than a temp directory).
> 2. **A sweep blocks the next sweep.** Each run appends to `results/results.csv`
>    and drops untracked `logs/clocks_*.csv` thermal traces.
>
> So before each sweep, either commit what the previous step produced — which is
> the intended workflow, since the logs are evidence for the rows — or reset:
>
> ```bash
> git checkout -- results/report.html          # undo what pytest regenerated
> git status --porcelain                       # must print nothing before sweeping
> ```
>
> `--allow-dirty` runs anyway and tags the rows `dirty`. Use it for a throwaway
> measurement, never for one you intend to quote.
>
> Both are open defects rather than intended design: a test should not write into
> `results/`, and untracked output should not count toward reproducibility of the
> code that produced a row.

`python -m src.memcheck` **exits non-zero when a config will not fit** — that is
the answer, not a failure. Do not run the block above under `set -e`.

To see what the figures look like before any GPU run exists, render them from the
synthetic fixture into a scratch directory. The summary it writes is stamped
SYNTHETIC:

```bash
python -m analysis.make_all --results tests/fixtures/results_synthetic.csv \
  --logs tests/fixtures --figures /tmp/preview --summary /tmp/preview/summary.md \
  --dispatch /tmp/preview/dispatch.json
```

> **As written, this command overwrites `results/report.html`.** It redirects
> `--figures`, `--summary` and `--dispatch` but not the HTML page, so the page
> is written to its default path and the real results page ends up showing
> invented fixture numbers. `make_all` does have a `--page` flag — pass it too:
>
> ```bash
> python -m analysis.make_all --results tests/fixtures/results_synthetic.csv \
>   --logs tests/fixtures --figures /tmp/preview --summary /tmp/preview/summary.md \
>   --dispatch /tmp/preview/dispatch.json --page /tmp/preview/report.html
> ```
>
> If you already ran it without `--page`: `git checkout -- results/report.html`.
>
> `tests/test_analysis.py` has the same omission on all four of its `make_all`
> calls, so **running the test suite also overwrites the page with fixture
> data**. That one is an open defect rather than a documentation gap.

Everything above except the two commands marked GPU runs on macOS CPU.

Verify the organizers' file was never modified — the hash must match the one
pinned in [docs/INTERFACE.md](docs/INTERFACE.md):

```bash
sha256sum bench/torch_transformer_benchmark.py    # Linux / WSL2
shasum -a 256 bench/torch_transformer_benchmark.py  # macOS
```

## Viewing the results page

`python -m analysis.make_all` writes **`results/report.html`** — one page with
every figure and every headline number, for reading, screen-recording or
attaching to a submission.

It is a **static file**, not a served app. There is no port and nothing to keep
running: every figure is embedded in the file itself, so it opens straight from
disk and works offline.

```bash
# macOS
open results/report.html

# WSL2 — opens in the default Windows browser.
# `open` does not exist on Linux and `xdg-open` usually fails under WSL;
# this converts the WSL path to a Windows one, which is what actually works.
explorer.exe "$(wslpath -w results/report.html)"

# plain Linux
xdg-open results/report.html
```

**The numbers do not update by themselves.** They are baked in when the file is
written, so a page open in a tab keeps showing whatever the log said at
generation time. To refresh it:

```bash
git pull                      # pick up the other person's measurements
python -m analysis.make_all   # regenerate — about a second
```

…then reload the browser tab. That is deliberate: the page is a snapshot tied to
a commit, not a live view. Every number traces to append-only rows that each
carry the git SHA they were produced at, and the footer stamps the generation
time and row count — so what is on screen during a recording is exactly what is
in the log.

If the page looks mostly empty, that is not a bug: it is honestly reporting that
`results/results.csv` has no rows for those figures yet. Each empty panel names
the sweep command that would fill it.

## Results

The single most direct number, from the organizers' own script on a clean clone
(`run_official.py --batch-size 8 --seq-len 1024`, fp32, 6 layers):

```
baseline : median=135.4051 ms | throughput= 60,500 token/s
optimized: median= 72.7521 ms | throughput=112,602 token/s
speedup  : 1.861x    accuracy: PASS, max_abs=0.00068 over 5 trials
```

Across the measured shape matrix, by sequence length (median speedup per
configuration, from `results/summary.md`):

| seq_len | baseline (control) | sdpa | optimized (shipped router) |
|---|---|---|---|
| 128 | 0.99x | 0.91x | 0.98x |
| 512 | 1.00x | 1.65x | 1.45x |
| 1024 | — | 2.08x | 1.88x |

The `baseline` row is the control: an unmodified copy of the reference model,
which must measure ~1.00x or the measuring rig itself is wrong.

The shipped `optimized` router is deliberately *not* the fastest row everywhere —
it falls back to the baseline at shapes where fusion loses (S=128, batch 1) and
wherever a dtype or depth would breach the accuracy budget. That is why its
geometric mean sits below `sdpa`'s while its worst case does not regress.

**How the aggregates are computed.** Every summary statistic takes one row per
configuration — the most recent verdict for each — rather than every passing row
in the log. `results.csv` is append-only, so averaging raw rows would weight a
configuration by how many times it happened to be re-swept, which is a fact
about our development history rather than about the code. A configuration whose
latest verdict is a FAIL drops out entirely rather than reverting to its last
passing number. `summary.md`, `report.html` and the figures are all computed
this way and agree with each other.
[docs/TECH_REPORT.md](docs/TECH_REPORT.md) §4.1 has the detail, including why
the router's 1.322x is a more honest number than `sdpa`'s 1.531x.

## Limitations and what we would do with more time

Written up in full, with the measurements behind each one, in
[docs/TECH_REPORT.md](docs/TECH_REPORT.md) §11. The short version:

- **Reduced precision is unreachable, not unexplored.** bf16 fails the tolerance
  at every depth and fp16 from two layers up, because the reference rounds
  softmax probabilities to the model dtype before `probs @ v` and a fused kernel
  does not — we are *more* accurate and therefore fail to reproduce its rounding.
  That puts the card's 22.5 / 23.2 TFLOP/s tensor-core peaks out of reach by
  construction. The largest single piece of performance left on the table.
- **No device-side kernel timeline.** WSL2's CUPTI populates no kernel-completion
  events, so GPU-busy percentages are reported as unmeasurable rather than
  guessed, and the launch-overhead argument rests on a CPU-side launch count.
- **CUDA graphs are undecided at the smallest shape**, not ruled out — 13.75% of
  trace span goes into launch calls at B=8/S=128. Deprioritized on opportunity
  cost against a 2.30x win already measured elsewhere, not on proof of absence.
- **S=2048 is skipped by the memory pre-check** on a 6 GB card, so it drops out
  of every aggregate — the score is a ratio and the baseline cannot produce a
  denominator there.

## Team contributions

- **De Xun** — optimized kernel implementations (`src/strategies/`,
  `src/optimized.py`), all GPU measurement and profiling.
- **K Sriman Preeth** — measurement harness and thermal logging, memory
  pre-check, CPU correctness suite, shape/capability dispatch, analysis and
  figures, report and submission materials.

## AI usage

See [docs/AI_USAGE.md](docs/AI_USAGE.md) for a contemporaneous, per-session log
of which AI tools were used, what they produced, and what had to be corrected.

## Documents

| | |
|---|---|
| [docs/TECH_REPORT.md](docs/TECH_REPORT.md) | The full write-up: problem framing, profile, each optimization, dispatch, roofline, accuracy budget, thermal methodology, limitations. |
| [docs/DEVPOST.md](docs/DEVPOST.md) | Devpost submission text, one section per required field. |
| [docs/VIDEO_SCRIPT.md](docs/VIDEO_SCRIPT.md) | 3-minute demo shot list with the exact commands to run on screen. |
| [docs/INTERFACE.md](docs/INTERFACE.md) | Internal contract: strategy signature, masking semantics, registry, `results.csv` schema, tolerance. |
| [docs/AI_USAGE.md](docs/AI_USAGE.md) | Per-session log of AI tool usage, written contemporaneously, corrections included. |
| [docs/DECISIONS.md](docs/DECISIONS.md) | Open decisions and artefacts still owed between the two of us. |
| [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) | Onboarding for the second contributor: git workflow, file ownership, naming rules, and what already exists so it doesn't get rebuilt. |

Before submitting, check nothing is still a placeholder:

```bash
python docs/check_ready.py
```

It exits non-zero while any `<FILL …>` marker remains, so it can gate the final
pre-submission check.
