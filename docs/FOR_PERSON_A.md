# Read this before you write any code

This is everything you need to work on this repo without the two halves
colliding. It takes about ten minutes to read and will save hours.

**The short version:** the entire measurement, testing, analysis and reporting
half of the project is already built and tested. You own exactly two things —
the optimized implementations and the entry point that selects between them.
Everything else already exists, so if you find yourself about to write a
benchmark script, a plotting script, or a results file, **stop and read section
2** — it is already there and yours will double-count.

---

## 1. How we work together (git)

### The setup

1. I create the GitHub repository and push the current code.
2. **You clone it from GitHub.** Not from a zip, not from a USB stick — clone
   the GitHub URL. If we both start from GitHub there is exactly one history and
   nothing to reconcile later.

```bash
git clone <REPO_URL>
cd <repo>
# choose the command according to your system
python -m venv .venv && source .venv/bin/activate #for mac
python -m venv .venv && .venv\Scripts\activate.bat #for windows
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124   # match your driver
pip install -r requirements.txt
pip install -e .
pytest -q          # should be all green before you change anything
```

If `pytest -q` is not green on a fresh clone, **tell me before doing anything
else**. That is a broken checkout, not a you problem, and everything you build
on top of it would inherit the problem.

### The daily loop

Work on a branch, never directly on `main`:

```bash
git checkout main
git pull                                  # ALWAYS pull before starting
git checkout -b rung1-sdpa                # one branch per optimization
# ... work ...
pytest -q                                 # must be green
git add -A
git commit -m "Rung 1: SDPA attention"
git push -u origin rung1-sdpa
```

Then open a pull request on GitHub, or tell me and I will merge it.

### Four git rules that matter

1. **Always `git pull` before you start work.** Most conflicts are just someone
   working from a stale copy.
2. **Never `git push --force`** on `main` or on any branch I might have pulled.
   Force-pushing rewrites history and can delete work that is already merged.
3. **Never commit `.venv/`.** It is in `.gitignore` already — just don't
   override it. It is hundreds of megabytes and it is machine-specific.
4. **Never resolve a conflict in `results/results.csv` by choosing one side.**
   That file is append-only and both of us add rows to it. I have configured git
   to merge it by keeping *both* sets of rows automatically
   (`.gitattributes`, `merge=union`), so this should never come up — but if it
   somehow does, the answer is always "keep both", never "keep mine".

### Big files

Profiler traces can be tens of megabytes. Please gzip them before committing:

```python
prof.export_chrome_trace("logs/trace_small_baseline.json")
```
```bash
gzip logs/trace_*.json     # the analysis reads .json.gz directly
```

If a single file is over ~50 MB, don't commit it — send it to me instead and
we'll decide.

---

## 2. What already exists — please don't rebuild any of this

This is the section that prevents duplicated work. **All of the following is
written, tested and working.** If you need one of these things, use the existing
one. If it doesn't do what you need, tell me and I'll extend it — that keeps one
version of the truth instead of two that disagree.

| I need to… | Use this — it already exists | Do NOT write |
|---|---|---|
| Run a benchmark and record the result | `python bench/sweep.py --strategy <name> ...` | your own benchmark loop or timing script |
| Run the organizers' script with our code | `python bench/run_official.py <their flags>` | a modified copy of their script |
| Check my implementation is numerically correct | `pytest -q` | your own comparison script |
| Know whether a config will fit in VRAM | `python -m src.memcheck --batch 8 --seq-len 4096 ...` | your own memory estimate |
| Log GPU clocks / detect throttling | `bench/thermal.py` (sweep.py calls it automatically) | your own nvidia-smi logger |
| Make any chart or table | `python -m analysis.make_all` | your own matplotlib code |
| Store results | `results/results.csv` — written only by `sweep.py` | a second CSV, a spreadsheet, notes in a text file |
| Decide which strategy to use for a shape | `src/dispatch.py` → `select_strategy()` | your own if/else shape checks |
| Analyse a profiler trace | `analysis/trace.py` (make_all picks traces up automatically) | your own trace parser |

**Why this matters concretely:** if you write a second timing loop, it will not
set the same random seeds, the same matmul precision, or the same TF32 flags as
the organizers' script does — and it will produce a different number for the
same code. Then we have two speedups and no way to tell which one is real. The
existing harness imports the organizers' own functions specifically so that
cannot happen.

---

## 3. What you own, and what you must not touch

### Yours — edit freely

| File | What it is |
|---|---|
| `src/strategies/*.py` | Your optimized implementations, one file per strategy |
| `src/optimized.py` | The entry point (see section 5) — **you need to create this** |
| `bench/profile_baseline.py` | Your profiling script |
| Your entries in `docs/AI_USAGE.md` | Append only; don't edit mine |
| Your `<FILL A: …>` slots in `docs/TECH_REPORT.md` and `docs/DEVPOST.md` | See section 8 |

### Mine — please don't edit; tell me instead

`bench/sweep.py` · `bench/thermal.py` · `bench/run_official.py` ·
`src/memcheck.py` · `src/dispatch.py` · `src/baseline.py` ·
`src/strategies/__init__.py` · everything in `analysis/` · everything in
`tests/` · `pyproject.toml` · `.gitignore` · `.gitattributes`

This isn't territorialism — it's that these files have 224 tests pinned to their
behaviour, and a change in one of them can silently alter numbers that are
already in the report. **If one of them is blocking you, message me and I'll
change it in a minute.** That is much faster than us both discovering later that
our results were produced by two different versions of the harness.

### Nobody's — frozen

**`bench/torch_transformer_benchmark.py`** and
**`bench/tensorflow_transformer_benchmark.py`** are the organizers' files. They
must stay byte-identical to the download. Their SHA-256 is pinned in
`docs/INTERFACE.md` and **a test fails if either changes**. If we edit them, our
speedup is not comparable to what the judges would measure, and the whole
submission is void.

---

## 4. Naming rules — this is where consistency actually breaks

Names in this project are not cosmetic. A strategy's name is written into
`results/results.csv`, keyed into `results/dispatch_table.json`, printed in the
report tables, and used at runtime by the dispatcher. If a name changes, old
rows in the results file orphan and the dispatch table stops matching.

### Strategy names

- **Lowercase, underscores, no spaces, no hyphens, no capitals.**
  Good: `sdpa`, `sdpa_bf16`, `fused_qkv`, `triton_layernorm`, `compiled`
  Bad: `SDPA`, `sdpa-bf16`, `SDPA attention`, `myKernel`, `test2`, `final_v3`
- **The filename should match the strategy name.** Strategy `fused_qkv` lives in
  `src/strategies/fused_qkv.py`. One primary strategy per file.
- **Register each name exactly once.** Registering a duplicate raises an error
  immediately — that's deliberate, so two different implementations can never
  quietly share a name.
- **Never rename a strategy once you have collected results for it.**
  `results.csv` is append-only and its old rows would refer to a name that no
  longer exists. If you genuinely need a rename, tell me — it needs a migration
  note, not a find-and-replace.
- **Don't use version numbers in names.** `sdpa_v2` replacing `sdpa` means the
  results table has two rows that look like different optimizations but are the
  same one at different times. Git history is the version history. If the new
  one is genuinely a different approach, give it a descriptive name
  (`sdpa_flash`, not `sdpa_v2`).

### Combined strategies

If a strategy is a combination, name it in a consistent order —
**attention, then precision, then compilation**:

```
sdpa                    SDPA only
sdpa_bf16               SDPA + bf16
sdpa_bf16_compiled      SDPA + bf16 + torch.compile
fused_qkv_bf16          fused QKV + bf16
```

Not `bf16_sdpa` for one and `sdpa_bf16` for another — those look like two
strategies in every table we produce.

### Branch names

`rung1-sdpa`, `rung2-bf16`, `rung3-compile` — matching the rung numbering in
the report so a branch is traceable to a report section.

### Everything else

- Python files: `lower_snake_case.py`
- Functions and variables: `lower_snake_case`
- Classes: `CamelCase`
- Follow the style of the file you're in. Match its comment density and naming.

---

## 5. Your two coding jobs, concretely

### Job 1 — write strategies (`src/strategies/*.py`)

Each strategy is one file. Adding a file is all it takes — the registry
auto-imports everything in that folder, so you never edit
`src/strategies/__init__.py`.

```python
# src/strategies/sdpa.py
"""SDPA attention: let PyTorch pick a fused attention kernel."""

from typing import Optional

import torch
import torch.nn.functional as F

from src.baseline import BaselineTransformer
from src.strategies import register


@register("sdpa")
class SdpaTransformer(BaselineTransformer):
    # Set these only if they apply:
    # REQUIRES_CUDA = True        # cannot run on CPU at all (Triton, custom CUDA)
    # MIN_CAPABILITY = (8, 0)     # needs Ampere or newer (e.g. any bf16 path)

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        ...
```

**Five rules the tests enforce.** Each one exists because breaking it produces a
wrong number rather than an error:

1. **Subclass `BaselineTransformer`**, and keep the signature exactly
   `forward(self, x, valid_token_mask=None)`. No extra required arguments — the
   organizers' script only ever passes those two.
2. **Never rename or remove a parameter.** The organizers'
   `copy_model_weights(baseline, optimized, strict=True)` must succeed, and that
   is the only thing proving both models hold the *same weights*. If you fuse
   QKV, keep `q_proj` / `k_proj` / `v_proj` registered under those names and
   build the fused view from them. Don't add new parameters either — a new
   `nn.Linear` in `__init__` fails the strict load.
3. **Set `REQUIRES_CUDA = True`** if it can't run on CPU. My machine has no GPU;
   without this the test suite reports a fake failure instead of a clean skip.
4. **Set `MIN_CAPABILITY`** if it needs specific hardware — `(8, 0)` for a bf16
   path, `(7, 5)` for Triton or flash-style kernels. The dispatcher uses it so
   an older card never gets handed a kernel it can't run.
5. **Run `pytest -q` before every push.** It's CPU-only and takes about ten
   seconds.

**The four masking rules that will bite you.** These are all in the baseline and
all of them are places where an optimized version silently diverges:

- `valid_token_mask` is `[B, S]`, `torch.bool`, and **True means KEEP**
  (not "mask out" — the polarity catches people).
- Invalid **key** positions are masked inside attention:
  `~valid_token_mask[:, None, None, :]`.
- Output is zeroed at invalid **query** positions in **four** places: after
  attention, after each block, and after the final norm. Miss one and only the
  padded branch fails.
- Causal mask is `triu(diagonal=1)` — strictly above the diagonal. Using
  `diagonal=0` masks the diagonal too, which makes the first row entirely
  masked, which makes softmax return NaN.

**And the one that catches everyone:** the reference computes
**softmax in fp32 and casts back**, even when the model is fp16 or bf16
(`torch.softmax(scores.float(), dim=-1).to(x.dtype)`). If you do the softmax in
reduced precision, your `max_abs` will land just over budget. That's a
correctness bug, not a tolerance problem — don't "fix" it by loosening the
tolerance.

When a test fails it prints exactly what you need: `max_abs`, `max_rel`,
`worst_index`, both values at the worst element, which output dims failed, and —
if the output contains NaN — a note naming the two usual causes. The failing
test's *name* tells you which branch is wrong, e.g.
`[causal-nopad-B2_S16_d64_H4-sdpa]` means the causal branch, unpadded.

### Job 2 — write the entry point (`src/optimized.py`)

**This file does not exist yet and it is yours.** It is what the organizers'
script actually instantiates. Its job is to pick a strategy for the incoming
shape and delegate to it:

```python
# src/optimized.py
"""The entry point the organizers' benchmark instantiates."""

from typing import Optional

import torch

from src.baseline import BaselineTransformer
from src.dispatch import DispatchKey, select_strategy
from src.strategies import get_strategy


class UserOptimizedTransformer(BaselineTransformer):
    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        key = DispatchKey.from_forward(x, valid_token_mask, self.config)
        name = select_strategy(key)
        # ... delegate to the chosen strategy ...
```

Two performance notes:

- `select_strategy()` is memoised — calling it per forward costs a dict lookup.
- `DispatchKey.from_forward()` can cost a **GPU synchronisation**, because
  working out whether the mask is all-True means reading a reduction back to the
  host. Pass `padded=` explicitly when you already know it. In the benchmark
  harness, padding is a property of the configuration and is known before the
  call.

Exactly how you delegate (holding one instance per strategy, sharing
parameters, etc.) is your design call — talk to me if you want a second opinion,
since it's the one file where our two halves meet.

---

## 6. Answering your question: "who writes the code that shows the optimized output?"

Two different things, and they have different owners:

- **The optimized implementation itself** — the code that computes the faster
  forward pass. **That's you**, in `src/strategies/` and `src/optimized.py`.
- **The code that runs it, measures it, and displays the result** — **already
  built.** You don't need to write any of it:

```bash
# The organizers' own script, with your class injected. This prints the
# accuracy check and the speedup, and is what we show in the demo video.
python bench/run_official.py --batch-size 8 --seq-len 1024 --dtype bfloat16

# The sweep harness: runs many shapes, appends rows to results/results.csv.
python bench/sweep.py --strategy sdpa --matrix default

# Turns those rows into every figure and results/summary.md.
python -m analysis.make_all
```

`run_official.py` prints which implementation it injected, so a run can never
leave you guessing what was measured. Right now it falls back to the baseline
control, because `src/optimized.py` doesn't exist yet — the moment you create
it, it picks it up automatically.

---

## 7. The results file — three rules

`results/results.csv` is the evidence behind every number in the report.

1. **Only ever write to it via `bench/sweep.py`.** Never open it in Excel, never
   edit a cell, never re-sort it.
2. **It is append-only.** Old rows are never deleted, even wrong ones — a run
   that failed, was skipped, or was thrown out for thermal throttling is *data*,
   and the report counts them.
3. **Every row records the git SHA it was produced at.** The harness refuses to
   run against uncommitted changes for that reason. If you really need to run
   dirty, pass `--allow-dirty` and the rows get tagged `dirty` so we know not to
   quote them.

The very first thing to run, before writing any kernel:

```bash
python bench/sweep.py --strategy baseline --matrix quick
```

`baseline` is a control — an unmodified copy of the reference model. **Every row
must show `max_abs_err` exactly 0 and a speedup of ~1.00 (0.97–1.03).** If it
doesn't, the measuring rig is lying and no other number either of us produces
means anything. Please paste me those rows.

---

## 8. Things I need from you

### Right now: fill in `docs/APPROVALS_NEEDED.md` yourself

Open it, answer the questions inline, commit, push. It has five decisions in it
that are baked into code. **If you don't answer, these defaults apply** — they
are all the safe choice, but some are probably wrong for your machine:

| Question | Default if you don't answer |
|---|---|
| Three extra columns in `results.csv` | Kept |
| Is SDPA safe on CPU? | No — CPU falls back to `baseline` |
| Tolerance target | The stricter pair, `atol=0.001 / rtol=0.01` |
| Your card's peak TFLOPS and bandwidth | Placeholders 12.0 / 48.0 / 192 — **these are almost certainly wrong and every roofline number scales with them** |
| Who creates the GitHub repo | I do |

### When you get to it

1. **Environment details** for the report — CPU, GPU, VRAM, driver, CUDA, WSL2,
   PyTorch, Triton versions, and the exact `pip install torch --index-url …`
   line you used. Section 2 of `docs/TECH_REPORT.md` has a table with a slot for
   each.
2. **Six profiler traces**, with these exact filenames (the analysis finds them
   by name, nothing to configure):
   ```
   logs/trace_small_baseline.json     logs/trace_small_optimized.json
   logs/trace_medium_baseline.json    logs/trace_medium_optimized.json
   logs/trace_large_baseline.json     logs/trace_large_optimized.json
   ```
3. **One run of `--matrix accuracy`** (12 configs) — it's the only sweep that
   produces the accuracy-vs-depth figure.
4. **Your AI usage log entries**, appended to `docs/AI_USAGE.md` as you go. The
   problem statement gives *bonus points* for this and it has to be written
   contemporaneously, not reconstructed the night before. One entry per session:
   what you asked, whether it worked, what you had to fix. The corrections are
   the valuable part.
5. **Your report slots.** Run this to see exactly what's outstanding and who
   owes it:
   ```bash
   python docs/check_ready.py --owner A
   ```
   It exits non-zero while anything is unfilled, so we can use it as the
   final pre-submission check.

---

## 9. Your first hour

```bash
git clone <REPO_URL> && cd <repo>
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt && pip install -e .

pytest -q                                                 # 1. must be green
python bench/sweep.py --strategy baseline --matrix quick   # 2. must be ~1.00x, max_abs 0
python bench/profile_baseline.py                           # 3. your profiling — where is the time?
```

Then:

4. Read `docs/INTERFACE.md` (the contract) — 10 minutes.
5. Fill in `docs/APPROVALS_NEEDED.md`, commit, push.
6. Paste me the rows from step 2 and the GPU-busy number from step 3.

**Do step 3 before writing any kernel.** If the GPU turns out to be idle most of
the time at small shapes, the fix is fewer/bigger kernels and making any single
kernel faster buys almost nothing. Guessing that wrong costs a day.

---

## 10. Things that break quietly

The tests catch most of these — but here's what they mean when they fire.

| What you did | What happens | How you'll know |
|---|---|---|
| Edited the organizers' file | Our speedup is no longer comparable to theirs; submission is void | `test_organizers_file_is_unmodified` fails |
| Added or renamed a parameter in a strategy | The two models no longer provably hold the same weights | Strict weight-copy raises "Missing key(s)" |
| Did the softmax in fp16/bf16 | Error lands just over budget | `max_abs` slightly over 0.001 in the accuracy check |
| Used `triu(diagonal=0)` | First row fully masked → softmax over all `-inf` → NaN | `max_abs_err: nan`, and the failure note names this cause |
| Missed one of the four masking sites | Only the padded branch fails | Test id contains `pad0.3` |
| Named a strategy `sdpa-v2` (hyphen/caps) | Ugly in every table; may break the dispatch key | Nothing fails — please just don't |
| Renamed a strategy after collecting results | Old rows orphan; the dispatch table stops matching | Nothing fails — the numbers just quietly stop lining up |
| Wrote your own timing loop | Different seeds/precision flags → a different speedup for the same code | Nothing fails, and we can't tell which number is real |
| Committed `.venv/` | Hundreds of MB in the repo | Clone gets slow; `git status` is a mess |
| Hand-resolved a `results.csv` conflict | Someone's measurements silently deleted | Nothing fails |

Notice how many of those say "nothing fails". That's the reason this document
exists.

---

## 11. When in doubt

**Tell me instead of changing it.** Specifically:

- You need a change in one of my files → message me, I'll do it in a minute.
- You disagree with a decision in `docs/INTERFACE.md` → say so. It's a draft
  waiting for your approval; it is meant to be argued with, just not silently
  contradicted in code.
- A test fails and you think the test is wrong → quite possible! Send me the
  output. Don't delete or skip the test, because it's probably pinning something
  the report already claims.
- Something in the harness doesn't do what you need → it's easier to extend one
  harness than to reconcile two.

The whole point of the split is that neither of us has to hold the other half in
our head. That only works if we don't quietly reach into each other's half.

---

## 11b. Viewing the results page (you'll need this for the video)

`python -m analysis.make_all` writes `results/report.html` — every figure and
every headline number on one page, with the claim each figure supports. That is
what to narrate over, rather than opening individual PNGs.

It is a **static file**. No server, no port, nothing to keep running — all the
figures are embedded inside it, so it opens from disk and works offline.

```bash
# WSL2 — this is the one that works. `open` is macOS-only, and `xdg-open`
# usually fails under WSL. wslpath converts to a Windows path so the default
# Windows browser can find it.
explorer.exe "$(wslpath -w results/report.html)"
```

**It does not update on its own.** The numbers are written into the file when
it is generated, so a tab you left open keeps showing the old ones. After any
sweep:

```bash
git pull && python -m analysis.make_all
```

…then reload the tab. Takes about a second.

That is on purpose: the page is a snapshot tied to a commit, not a live view.
The footer stamps the generation time and row count, and every number traces to
rows carrying their own git SHA — so what is on screen while you are recording
is provably what is in the log.

If a section looks empty, it is telling you the truth: no rows in
`results/results.csv` support that figure yet, and the panel names the sweep
command that would fill it.

## 12. Command cheat sheet

```bash
# Before you start work
git pull && pytest -q

# Correctness (CPU, ~10 s) — run before every push
pytest -q
pytest -q tests/test_strategies.py -k sdpa        # just your strategy

# The control gate — must be ~1.00x with max_abs exactly 0
python bench/sweep.py --strategy baseline --matrix quick

# Measure your strategy
python bench/sweep.py --strategy sdpa --matrix quick      # fast check
python bench/sweep.py --strategy sdpa --matrix default    # the full sweep
python bench/sweep.py --strategy sdpa --matrix accuracy   # layers x dtype
python bench/sweep.py --strategy sdpa --compile-baseline  # the honesty check

# The organizers' own script with your class injected
python bench/run_official.py --batch-size 8 --seq-len 1024 --dtype bfloat16

# Will this config OOM before I try it?
python -m src.memcheck --batch 8 --seq-len 4096 --d-model 512 --heads 8

# Regenerate every figure, the dispatch table, results/summary.md and report.html
python -m analysis.make_all

# Open the results page (WSL2 -> Windows browser)
explorer.exe "$(wslpath -w results/report.html)"

# What's still unfilled in the submission docs?
python docs/check_ready.py --owner A

# Verify the organizers' file is untouched
shasum -a 256 bench/torch_transformer_benchmark.py   # must match docs/INTERFACE.md
```

---

Anything unclear, ask — a five-minute message now is much cheaper than
discovering on submission day that our two halves measured different things.
