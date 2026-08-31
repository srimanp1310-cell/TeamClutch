# Decision record

Open questions raised while building, and the answers that settled them. Each
answer is baked into code somewhere, so this is the trail behind choices that
would otherwise look arbitrary.

I've built the harness half of the repo (Tasks 0–5 of `docs/PROJECT_PLAN.md`):
skeleton, memory pre-check, sweep harness, correctness oracle, analysis and
figures, and the dispatch layer. Before Day 1 I need a few decisions from you,
because they're baked into files you'll be using and they're expensive to change
later.

Reply inline in this file, or just answer in chat — whatever's faster.

---

## Part 1 — Decisions I need a yes/no on (these block me)

### 1.1 Three extra columns in `results.csv` — OK to add?

The agreed schema has 21 columns. I want to append **3 more at the end**
(appending is safe; nothing already written breaks):

| column | why I want it |
|---|---|
| `baseline_peak_vram_mb` | The VRAM-ceiling figure needs baseline and optimized peaks **separately**. They cannot be separated after the fact, so if we don't record it from row one, it's gone. |
| `compile_baseline` | We're doing the "how much speedup survives when the baseline is compiled too" honesty check. Without this column we can't tell those runs apart later. |
| `ffn_dim` | It's a free axis nothing else pins down. Two rows can look identical and have different FFN widths. |

**Default if you don't reply: I keep them.** They're already written and an older
21-column file still loads fine. Say the word and I'll drop them.

- [ ] Approve · [ ] Drop them · [ ] Something else: ______

---

### 1.2 Is an SDPA strategy safe to run on CPU?

`torch.nn.functional.scaled_dot_product_attention` exists on CPU, but I don't
know whether your version will match the baseline within tolerance there.

This decides one line of the dispatch logic: **when there's no GPU, do we fall
back to `"baseline"` or to `"sdpa"`?**

- [ ] Use `baseline` on CPU (safe default) · [ ] `sdpa` is CPU-safe, use it

---

### 1.3 Tolerance — we target the stricter pair, agreed?

Two different numbers are floating around:

- the problem statement PDF says `abs < 0.002`, `rel < 0.02`
- the torch script's own defaults are `atol=0.001`, `rtol=0.01`

**I'm targeting the stricter one (0.001 / 0.01).** Passing there passes the PDF
with room to spare, and the margin is a good number for the report.

- [ ] Agreed

---

### 1.4 Roofline numbers for your card

For the roofline plot I need three numbers off the RTX 4050 Laptop spec sheet.
My placeholders are:

- peak FP32: **12.0 TFLOPS**
- peak BF16 tensor core: **48.0 TFLOPS**
- memory bandwidth: **192 GB/s**

Please confirm or correct — these go in the report, so I'd rather not guess.

---

### 1.5 Do you already have a repo pushed?

I built the repo fresh at `/Users/srima/tiktoktechjam2026` because I couldn't
see one from you. **If you already started a skeleton, tell me now** and I'll
merge into yours instead of us ending up with two repos.

Also: there's no git remote yet. Are you creating the GitHub repo, or should I?

---

## Part 2 — Things I need you to send me (not blocking today)

1. **Environment details** for the README and report: CPU, GPU, VRAM, driver
   version, WSL2 version, PyTorch version, CUDA version, Triton version, and
   the exact `pip install torch --index-url ...` line you used.
2. **6 profiler traces** (Task 7), whenever you get to profiling — one Chrome
   trace per shape (small / medium / large) for baseline and for the final
   optimized path. **The filenames matter**: `analysis/make_all.py` picks them
   up by name and needs no configuration, so please use exactly

   ```
   logs/trace_small_baseline.json     logs/trace_small_optimized.json
   logs/trace_medium_baseline.json    logs/trace_medium_optimized.json
   logs/trace_large_baseline.json     logs/trace_large_optimized.json
   ```

   ```python
   prof.export_chrome_trace("logs/trace_small_baseline.json")
   ```

   `.json.gz` works too. Everything downstream is built and tested already
   against synthetic traces — the moment these land, the GPU-busy table, the
   kernel-family breakdown and the timeline figures appear in
   `results/summary.md` with no further work.
3. **Your AI usage log entries.** The problem statement gives *bonus points* for
   this and it has to be written as you go, not reconstructed at the end. Just
   append to `docs/AI_USAGE.md` — there's a template at the top. One entry per
   session: what you asked, whether it worked, what you had to fix.
4. **Your PR descriptions** — I'm writing report section 4 (one subsection per
   optimization: hypothesis → measured before/after → surprise) straight from
   them, so the more you write there, the less I have to invent.
5. **One run of `--matrix accuracy`** (12 configs, layers × dtype crossed), when
   you have the time. It's the only sweep that produces the accuracy-budget
   figure: error vs depth needs a line per dtype, and a one-factor-at-a-time
   sweep only varies depth at the base dtype, so fp16 and bf16 come out as
   single points rather than lines.

   ```bash
   python bench/sweep.py --strategy <yours> --matrix accuracy
   ```

---

## Part 3 — How to write a strategy (no reply needed, but please read)

Full detail is in `docs/INTERFACE.md`. The short version:

**Adding a strategy is one new file.** Drop it in `src/strategies/` — the
registry auto-imports everything in that folder, so you never edit
`__init__.py`:

```python
# src/strategies/sdpa.py
from src.baseline import BaselineTransformer
from src.strategies import register

@register("sdpa")
class SdpaTransformer(BaselineTransformer):
    def forward(self, x, valid_token_mask=None):
        ...
```

**Four rules the tests enforce:**

1. Subclass `BaselineTransformer`. Signature exactly
   `forward(self, x, valid_token_mask=None)` — no extra required arguments,
   because the organizers' script only ever passes those two.
2. **Don't rename parameters.** `copy_model_weights(..., strict=True)` has to
   succeed. If you fuse QKV, keep `q_proj` / `k_proj` / `v_proj` registered
   under those names and build the fused view from them. Renaming means we
   can't prove both models have the same weights.
3. If it can't run on CPU (Triton, custom CUDA), set `REQUIRES_CUDA = True` on
   the class. My tests then print a visible SKIP on my Mac instead of a fake
   failure.
   **Also** set `MIN_CAPABILITY` if the strategy needs specific hardware —
   `(8, 0)` for a bf16 path, `(7, 5)` for Triton/flash. The dispatcher uses it
   so an sm_75 card never gets handed a bf16 kernel. It guesses from the name
   too, but a declaration beats the guess.
4. Run `pytest -q` before every push. It's CPU-only and takes ~6 seconds.

**Masking details that will bite you** — these are where a wrong implementation
shows up, and all four are in the baseline:

- `valid_token_mask` is `[B, S]`, bool, **True = keep**.
- Invalid *key* positions get masked inside attention.
- Output is zeroed at invalid *query* positions — after attention, after each
  block, **and** after the final norm.
- Causal mask is `triu(diagonal=1)`.
- **Softmax runs in fp32 and casts back**, even in fp16/bf16. If you do the
  softmax in reduced precision, expect `max_abs` to land just over budget.

**Never edit `bench/torch_transformer_benchmark.py`.** Its SHA-256 is pinned and
a test fails if it changes — if we edit it, our numbers aren't comparable to the
organizers'.

**`results/results.csv` is append-only.** Always write to it via
`bench/sweep.py`; never edit it by hand, never re-sort it.

---

## Part 4 — Your first job when you pull

Run this and paste me the rows:

```bash
python bench/sweep.py --strategy baseline --matrix quick
```

`baseline` is the control — it's an unmodified copy of the reference model.
**Every row must show `max_abs_err` exactly 0 and a speedup of ~1.00 (0.97–1.03).**

If it doesn't, the measuring rig is lying and no other number we produce means
anything. That's the Day-1 gate — worth doing before you write a single kernel.

---

# ANSWERS FROM A — 2026-08-28

## 1.1 Three extra CSV columns
**YES** — add `baseline_peak_vram_mb`, `compile_baseline`, `ffn_dim` at the end.

## 1.2 Is SDPA safe on CPU?
**NO** — CPU falls back to `baseline`. I can't test the CPU path and it isn't
what we're being scored on.

## 1.3 Tolerance target
**Stricter pair: atol=0.001, rtol=0.01** (the torch script's defaults, not the
PDF's 0.002/0.02). If we pass at 0.001 we pass at 0.002 either way.

## 1.4 Peak TFLOPS and bandwidth — MEASURED, replace the placeholders

This is a **low-TGP RTX 4050 Laptop**. Real numbers are roughly half the
spec sheet. Measured on 2026-08-28, mains power, Windows "Best Performance",
4096x4096 matmul / 512 MB device-to-device copy:

| metric | measured |
|---|---|
| fp32 (TF32 off) | 5.7 TFLOPS |
| fp32 (TF32 on) | **11.0 TFLOPS**  <- use this; benchmark defaults --allow-tf32 |
| fp16 tensor | 22.5 TFLOPS |
| bf16 tensor | **23.2 TFLOPS** |
| memory bandwidth | **174.8 GB/s** (91% of 192 theoretical) |

Ridge points (TFLOPS / bandwidth):
- fp32 TF32 on: **62.9 FLOP/byte**  <- the one for the roofline
- fp32 TF32 off: 32.6
- fp16: 128.7
- bf16: 132.7

Note for the report: bf16 is marginally FASTER than fp16 here (23.2 vs 22.5),
which is convenient since bf16 is also the numerically safer choice.
TF32 nearly doubles fp32 throughput.

## 1.5 Who creates the GitHub repo
**You do** — already done, I cloned TeamClutch.

---

## Environment (for TECH_REPORT.md section 2)

| item | value |
|---|---|
| CPU | Intel Core i7-12650HX (12th gen) |
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU |
| Compute capability | sm_89 (8, 9) — Ada Lovelace |
| VRAM | 6.0 GB GDDR6 |
| OS | Windows 11 + WSL2, Ubuntu 24.04 |
| PyTorch | 2.6.0+cu124 |
| Triton | 3.2.0 |
| Install | `pip install torch --index-url https://download.pytorch.org/whl/cu124` |

## Gate results
- `pytest -q` green on fresh clone (1 skip)
- Control: `sweep.py --strategy baseline --matrix quick`
  - B=8 S=128 d=512 H=8 fp32: **0.986x**, max_abs=0
  - B=8 S=512 d=512 H=8 fp32: **1.001x**, max_abs=0

---

# REQUESTS TO B FROM A — 2026-08-28 (Rung 1 causal accuracy audit)

Four things, in priority order. R1 and R2 are correctness-of-the-record issues,
R3 is a provenance gap that cost us a session, R4 is a dead declaration.

## R1 — `latest_per_config` cannot express a PASS -> FAIL downgrade (please fix)

**This one actually misreports our results right now.**

`usable()` filters to `status == "PASS"` *before* `latest_per_config()` does its
`groupby(...).tail(1)`. So a newer row that corrects an earlier PASS down to a
FAIL is dropped by the filter and **can never supersede the row it corrects**.

Concretely, after I appended two correcting rows today:

```
sdpa float32 causal L=6:
  10:20:29  b96670e  pad=0.0  PASS  max_abs=0.001235   <- superseded, still winning
  10:22:03  b96670e  pad=0.3  PASS  max_abs=0.001235   <- superseded, still winning
  15:20:29  b96670e  pad=0.0  FAIL  max_abs=0.001358   <- the correction, invisible
  15:20:29  b96670e  pad=0.3  FAIL  max_abs=0.001358   <- the correction, invisible

latest_per_config() still returns the 10:20 / 10:22 rows, contributing
2.1241x and 1.7784x to the speedup geomean.
```

Those two speedups are currently inflating our headline number with a config we
know fails 48% of seeds. Suggested fix: do the `groupby().tail(1)` on the full
frame first, *then* apply the PASS filter — so the newest row per config wins on
its own merits rather than the newest *passing* row.

## R2 — Please route `float32 + causal + layers >= 6` to `baseline` in dispatch

I can't do this myself: `src/dispatch.py` is yours under both rule blocks in
`CLAUDE.md`, and I'd rather ask than reach across the line.

Measured with your sweep's exact global state (`manual_seed`,
`matmul_precision="high"`, `allow_tf32=True`), 5 trials per seed:

| config | pass rate | median max_abs | worst max_abs |
|---|---|---|---|
| `causal=False L=6` | 20/20 (100%) | 0.000735 | 0.000895 |
| `causal=True L=1` | 20/20 (100%) | 0.000484 | 0.000566 |
| `causal=True L=2` | 20/20 (100%) | 0.000705 | 0.000783 |
| `causal=True L=6` | **21/40 (52%)** | **0.00117** | **0.00136** |

Budget is `atol=0.001`. At L=6 causal we are a coin flip on the judge's seed.
Padding ratio makes no difference (the worst elements sit at sequence positions
1–2, where causal masking already restricts attention to positions 0–2).

This is not a bug I can fix in `sdpa.py`: the error is TF32 noise in the
*reference's* `q @ k^T` and `probs @ v`. With `allow_tf32=False` on both sides it
collapses to 1.9e-06 (650x smaller). Our fp32 path runs on
`EFFICIENT_ATTENTION`, which doesn't use TF32 reductions at all — so we are the
accurate one, and closing the gap would mean making the organizers' file worse.
Full write-up in the `sdpa.py` module docstring.

## R3 — Three more `results.csv` columns: `seed`, `allow_tf32`, `input_scale`

Same "append at the end, nothing already written breaks" argument you used for
`baseline_peak_vram_mb` / `compile_baseline` / `ffn_dim` in 1.1, so I assume this
is uncontroversial.

**Why it matters:** I spent a session unable to reconstruct why the b7f6312
causal row recorded `max_abs=0.00113016 FAIL` while the b96670e row recorded
`max_abs=0.00123502 PASS`. `git diff b7f6312 b96670e -- src/strategies/sdpa.py`
is **empty** — the strategy code is byte-identical between them. Neither row
records the seed, the TF32 flag, or the input scale, and all three change the
number. I could reproduce the b96670e value exactly but never the b7f6312 one,
and with the current schema there is no way to tell whether that run used a
different seed, a different `--allow-tf32`, or something else. Two of those
rows are now un-diagnosable forever.

## R4 — `SUPPORTED_DTYPES` is read by nothing

```
$ grep -rn "SUPPORTED_DTYPES" .
src/strategies/sdpa.py:21   (docstring)
src/strategies/sdpa.py:96   (the declaration)
```

No test, no sweep, no dispatch consumes it — which is why `results.csv` still
contains failing bfloat16 rows for a strategy that "excludes" bfloat16. Either
wire it into the sweep's dtype gate (the way `REQUIRES_CUDA` is honoured at
`bench/sweep.py:727`), or tell me it's decorative and I'll stop implying
otherwise in docstrings. I've added `UNSUPPORTED_CAUSAL_MIN_LAYERS = 6`
alongside it for R2, marked equally declarative.

## FYI — one flaky test, not yet chased

`tests/test_strategies.py::test_reduced_precision_on_gpu[float16-sdpa]` failed
once in a full-suite run and passed on the next two, and passes in isolation. It
looks like cross-test global-precision pollution landing on a marginal fp16
case. Not chasing it this session; flagging so it isn't mistaken for a
regression. The two bfloat16 failures in that file are the known, documented
limit and predate today.

## R5 — `make_all.py` and `figures.py` aggregate differently (please standardise)

`analysis/make_all.py:85` builds the speedup-by-strategy table with
`usable(frame)` — every passing row. `analysis/figures.py:118` builds the
speedup plots with `latest_per_config(frame)` — one row per configuration. So
the table in `results/summary.md` and the figures printed beside it are not
computed the same way, and they disagree:

| | raw rows (`summary.md`) | per-config (figures, report §4.1) |
|---|---|---|
| `sdpa` | 1.530x (n=27) | 1.584x (n=14) |
| `optimized` | 1.351x (n=24) | 1.322x (n=15) |

`results.csv` is append-only and accumulates repeated measurements of the same
configuration, so the raw-row version weights each configuration by how many
times it happened to be measured. `sdpa` at B=8/S=512/d=512 appears five times
and is counted five times; a shape measured once counts once. That is a fact
about our development history, not about the code.

`optimized`'s n=24 additionally mixes an interrupted first sweep — which
included the fp16 config that failed before the routing fix — with the
completed one.

**Request: switch `make_all.py` to `latest_per_config` so summary.md, the
figures and the report agree.** I did not change it myself; `analysis/*` is
yours. TECH_REPORT §4.1 currently states the per-configuration numbers, says
why, and notes in one line that `summary.md` differs — so the report is
internally consistent either way, but the two artefacts will keep disagreeing
until this lands.

While you are in there: `usable()` is also what makes R1 unfixable from the
outside, since it filters to PASS before `latest_per_config` groups. Same
function, same fix window.
