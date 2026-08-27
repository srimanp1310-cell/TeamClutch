# PLAN_PERSON_B.md — Measurement, Infrastructure & Deliverables (Mac, no GPU)

> Drop this file in the repo root. Then, in Claude Code, say:
> **"Read PLAN_PERSON_B.md. Implement Task N exactly as specified, then run its acceptance test."**
> Do one task per session. Do not let Claude Code skip ahead.

---

## 0. Context Claude Code must know before touching anything

### 0.1 What the project is (one paragraph)

The organizers give a correct-but-slow PyTorch Transformer (`torch_transformer_benchmark.py`). The
team must fill in one method — `UserOptimizedTransformer.forward()` — with a faster implementation
that produces the *same numbers* (every output element within `abs_err <= 0.001` OR `rel_err <= 1%`).
The organizers' script measures both and prints `speedup = baseline_median / optimized_median`.
Person A writes the fast implementations ("strategies": SDPA attention, bf16, torch.compile, fused
QKV, maybe a Triton LayerNorm) and runs them on the only GPU we have (RTX 4050 Laptop, sm_89, 6 GB,
inside WSL2). **Person B (this plan) builds everything that turns A's kernels into a defensible,
reproducible, judged submission**: the sweep harness, memory pre-check, correctness test suite,
the strategy-dispatch table, all figures, the report, README, AI-usage log, Devpost text, video script.

### 0.2 Hard rules (from the problem statement + team notes)

1. **Never edit `bench/torch_transformer_benchmark.py`.** It is the organizers' file, byte-identical.
   We import from it; we never modify it.
2. **Everything B writes must run on a Mac with no GPU.** The organizers' script already falls back
   to CPU (`--device cpu`). All acceptance tests below run on CPU with tiny shapes. GPU-only paths
   (nvidia-smi, torch.cuda.*) must be guarded with `if torch.cuda.is_available()` / `shutil.which("nvidia-smi")`
   and degrade gracefully, never crash.
3. **No implementations in notebooks.** All logic lives in `src/`, `bench/`, `analysis/`. Notebooks
   (if any) only import and call.
4. **Correctness before speed.** The tolerance we target is the *stricter* one: `atol=0.001, rtol=0.01`
   (the torch script's defaults), not the PDF's `0.002 / 0.02`.
5. **`results/results.csv` is append-only.** Never regenerate or overwrite it. Every row carries `git_sha`.
6. **Interface contract (agreed with A, do not change without telling A):**
   - Every strategy is a class subclassing `BaselineTransformer` with
     `forward(self, x: torch.Tensor, valid_token_mask: Optional[torch.Tensor] = None) -> torch.Tensor`
     and *identical parameter names* (so `load_state_dict(strict=True)` works).
   - Strategies are registered in `src/strategies/__init__.py` as `STRATEGIES: dict[str, type[nn.Module]]`.
     The name `"baseline"` is always registered and maps to an unmodified copy (the ≈1.000x control).
   - CSV columns, in this exact order:
     ```
     timestamp, git_sha, strategy_name, batch, seq_len, d_model, heads, layers,
     dtype, causal, padding_ratio,
     accuracy_pass, max_abs_err, max_rel_err,
     baseline_median_ms, optimized_median_ms, speedup,
     peak_vram_mb, mean_sm_clock_mhz, max_temp_c, notes
     ```
     Proposed additions (append at the END only, confirm with A in `docs/INTERFACE.md`):
     `baseline_peak_vram_mb, compile_baseline, ffn_dim`.

### 0.3 Repo layout (B owns the files marked ★; A owns `src/strategies/` and `src/optimized.py`)

```
transformer-gpu-opt/
├── README.md                     ★ required deliverable
├── PLAN_PERSON_B.md              ★ this file
├── CLAUDE.md                     ★ short rules file for Claude Code (Task 0)
├── pyproject.toml                ★ makes `src`, `bench`, `analysis` importable; pytest config
├── requirements.txt              ★
├── .gitignore                    ★ *.pyc, .ipynb_checkpoints, logs/*.csv, results/figures/*.png (keep figures? see Task 6)
├── src/
│   ├── __init__.py
│   ├── baseline.py               ★ re-exports from bench/torch_transformer_benchmark.py (imported, not copied)
│   ├── memcheck.py               ★ Task 1 — memory pre-check
│   ├── dispatch.py               ★ Task 5 — shape + capability → strategy name
│   ├── strategies/               (A) sdpa.py, fused_qkv.py, compiled.py, triton_kernels/ ...
│   │   └── __init__.py           ★ skeleton with STRATEGIES registry (Task 0); A adds entries
│   └── optimized.py              (A) UserOptimizedTransformer = the entry point, uses dispatch.py
├── bench/
│   ├── torch_transformer_benchmark.py   ORGANIZERS' FILE — UNMODIFIED
│   ├── run_official.py           ★ Task 3 — runs organizers' main() with our class injected
│   ├── sweep.py                  ★ Task 2 — strategy in → CSV rows out
│   ├── thermal.py                ★ Task 2 — nvidia-smi logger + parser + discard rule
│   └── profile_baseline.py       (A) Rung 0 — already drafted by A
├── analysis/
│   ├── __init__.py
│   ├── load.py                   ★ Task 4 — load/clean results.csv, geometric mean, crossover table
│   ├── figures.py                ★ Task 4 — all plots
│   ├── roofline.py               ★ Task 4 — analytic FLOPs/bytes per config + roofline plot
│   ├── trace.py                  ★ Task 7 — torch.profiler chrome-trace → GPU busy % + timeline figure
│   └── make_all.py               ★ regenerates every figure from results/ and logs/
├── tests/
│   ├── test_strategies.py        ★ Task 3 — every registered strategy vs baseline on CPU
│   ├── test_memcheck.py          ★ Task 1
│   ├── test_sweep_cpu.py         ★ Task 2
│   ├── test_dispatch.py          ★ Task 5
│   ├── test_analysis.py          ★ Task 4
│   └── fixtures/results_synthetic.csv   ★ Task 4 — fake data so figures render before A has real data
├── results/
│   ├── results.csv               append-only (A appends via sweep.py)
│   ├── dispatch_table.json       generated by analysis/load.py (Task 4 → Task 5)
│   └── figures/
├── logs/                         nvidia-smi clock traces, profiler traces (A pushes)
└── docs/
    ├── INTERFACE.md              ★ Task 0 — the contract in 0.2, in writing
    ├── AI_USAGE.md               ★ Task 6 — contemporaneous prompt log (bonus points)
    ├── TECH_REPORT.md            ★ Task 8
    ├── DEVPOST.md                ★ Task 8 — Devpost description text
    └── VIDEO_SCRIPT.md           ★ Task 8 — 3-minute shot list for A to record
```

### 0.4 How B tests locally (the key unlock)

The organizers' script runs fine on CPU. Smoke-test shape for everything below:
```
--device cpu --batch-size 2 --seq-len 16 --d-model 64 --heads 4 --ffn-dim 128 --layers 2 \
--accuracy-trials 2 --warmup 1 --repeats 3 --benchmark-rounds 1
```
Takes ~1 s. Same flags, `--padding-ratio 0.3` and `--causal`, exercise the mask branches.

### 0.5 Environment setup on the Mac (do once, before Task 0)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch                     # macOS wheel: CPU (+MPS). No cu124 here — that's A's machine.
pip install pandas matplotlib numpy pytest jupyter ipykernel
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # expect: x.y.z False
```

---

## Task 0 — Repo skeleton, contract, Claude Code rules  (pre-sprint, ~1 h)

**Goal:** the repo has the structure above, `import src`, `import bench`, `import analysis` work from
anywhere, the organizers' file is in place and untouched, and the interface is written down.

**Do:**
1. If A already pushed a skeleton, clone it and *add* what is missing; do not restructure his files.
   If not, create the tree in 0.3 (empty `__init__.py` files, empty `results/figures/`, `logs/`).
2. Copy the organizers' `torch_transformer_benchmark.py` into `bench/` unchanged.
   Record its SHA-256 in `docs/INTERFACE.md` so anyone can verify it was never edited:
   `shasum -a 256 bench/torch_transformer_benchmark.py`.
3. `pyproject.toml`: minimal `[project]`, `[tool.setuptools] packages = ["src","bench","analysis"]`,
   `[tool.pytest.ini_options] testpaths = ["tests"]`. Then `pip install -e .`.
4. `src/baseline.py`:
   ```python
   from bench.torch_transformer_benchmark import (
       TransformerConfig, BaselineTransformer, BaselineTransformerBlock, BaselineSelfAttention,
       copy_model_weights, generate_random_case, compare_outputs, run_accuracy_tests,
       benchmark_models, resolve_device, resolve_dtype, maybe_compile,
   )
   __all__ = [...]
   ```
5. `src/strategies/__init__.py` skeleton:
   ```python
   from src.baseline import BaselineTransformer
   class BaselineCopy(BaselineTransformer):
       """Control strategy: identical to baseline. Must measure ≈1.000x."""
   STRATEGIES = {"baseline": BaselineCopy}
   def register(name):
       def deco(cls): STRATEGIES[name] = cls; return cls
       return deco
   def get_strategy(name): ...  # KeyError with the list of known names
   ```
   Tell A: each of his files does `from src.strategies import register` and decorates his class.
6. `docs/INTERFACE.md`: paste section 0.2 items 6 (signature, registry, CSV schema) verbatim, plus
   the smoke-test command in 0.4, plus the benchmark file SHA-256. Ask A to approve in chat before Day 1.
7. `CLAUDE.md` (repo root, ≤ 25 lines): rules 1–6 from 0.2, the smoke-test command, "run `pytest -q`
   before every commit", "never touch `src/strategies/*` or `src/optimized.py` — those are A's".
8. `.gitignore`, `requirements.txt` (torch, pandas, matplotlib, numpy, pytest; A adds triton on his side).
9. `docs/AI_USAGE.md` with the log template (see Task 6) and your first entry: this planning session.

**Acceptance:**
```bash
pip install -e . && python -c "import src.baseline, src.strategies, bench, analysis"
python bench/torch_transformer_benchmark.py --device cpu --batch-size 2 --seq-len 16 --d-model 64 \
  --heads 4 --ffn-dim 128 --layers 2 --accuracy-trials 2 --warmup 1 --repeats 3 --benchmark-rounds 1
# expect: summary: PASS ... speedup : ~1.0x (anything 0.8–1.2 on CPU is fine)
git add -A && git commit -m "skeleton + interface" && git push
```

---

## Task 1 — Memory pre-check `src/memcheck.py`  (Day 1, 11:00–13:00 slot)

**Goal:** before A runs a config on the 6 GB card, predict whether the *baseline* will OOM, so
`sweep.py` can skip (and record SKIPPED) instead of crashing the sweep half-way.

**Spec:** port `estimate_baseline_peak_bytes()` from the TensorFlow benchmark (same math) to torch:
```python
def estimate_baseline_peak_bytes(config: TransformerConfig, dtype: torch.dtype) -> int
def available_device_memory_bytes(device: torch.device) -> int
    # cuda: torch.cuda.mem_get_info(device)[0]; cpu: os.sysconf pages; fallback 8 GiB
def check_fits(config, dtype, device, memory_fraction=0.75, hard_cap_gib=None) -> tuple[bool, str]
    # returns (fits, human-readable reason with GiB numbers)
```
Model: two models resident (`2 * params * elem_size`), token workspace `(10*B*S*d + 2*B*S*f) * e`,
score workspace `B*H*S*S * (2*e + 8)` (fp32 softmax intermediates), masks. Keep the TF comments.
Also expose `--memcheck` CLI: `python -m src.memcheck --batch 8 --seq-len 2048 --d-model 512 --heads 8 --dtype float32`
prints the estimate and the verdict. On a Mac the "free memory" will be system RAM — that's fine; the
number that matters is the estimate, and A's machine supplies the real free VRAM.

**Acceptance (`tests/test_memcheck.py`):**
- estimate is monotonic in B, S, d, and quadratic in S (S doubled → score term ×4);
- B=8,S=128,H=8,d=512,L=6 fp32 → between 0.15 and 0.6 GiB; B=8,S=4096 → > 4 GiB
  (matches the team notes: "S=4096 will OOM");
- `check_fits` returns `(False, reason)` when `hard_cap_gib=0.01`.

---

## Task 2 — Sweep harness `bench/sweep.py` + `bench/thermal.py`  (Day 1, 09:00–11:00)

This is B's most important code. A runs it; B writes and CPU-tests it.

**Goal:** `strategy name in → CSV rows out`, reusing the organizers' accuracy and timing functions
*unchanged* so the numbers are exactly what their script would print.

**CLI:**
```
python bench/sweep.py --strategy sdpa [--batch 8] [--seq-len 512] [--d-model 512] [--heads 8]
    [--ffn-dim 0 → 4*d_model] [--layers 6] [--dtype float32|float16|bfloat16] [--causal]
    [--padding-ratio 0.0] [--device auto|cpu|cuda] [--compile-baseline]
    [--matrix default|quick|seq|batch|dmodel|dtype|mask]   # run a predefined OFAT set instead of one config
    [--cooldown 30] [--no-cooldown] [--no-thermal] [--memory-fraction 0.75] [--force]
    [--results results/results.csv] [--notes "free text"]
    [--warmup 20] [--repeats 100] [--benchmark-rounds 3] [--accuracy-trials 5] [--rtol 0.01] [--atol 0.001] [--seed 1234]
```

**Per-config procedure (one function `run_config(cfg, strategy_name, args) -> Row`):**
1. `git_sha = subprocess.run(["git","rev-parse","--short","HEAD"])`; refuse to run if the working tree
   is dirty unless `--allow-dirty` (dirty runs are unreproducible; put `dirty` in notes if allowed).
2. Memory pre-check (Task 1). If it fails and not `--force` → append a row with `notes="SKIPPED: <reason>"`,
   empty numeric fields, and continue.
3. Start thermal logger (Task 2b) if cuda and nvidia-smi exist and not `--no-thermal`.
4. Build `baseline = BaselineTransformer(cfg)`, `optimized = STRATEGIES[name](cfg)`, `copy_model_weights`,
   `.to(device, dtype).eval()`, `maybe_compile(baseline, args.compile_baseline, "default")`.
   **Same order as the organizers' `main()`** — construct → copy weights → to(device,dtype) → eval → compile.
5. Accuracy: call the organizers' `run_accuracy_tests(...)` — but it only returns a bool and prints.
   To get `max_abs_err` / `max_rel_err` into the CSV, do the loop yourself using their
   `generate_random_case` + `compare_outputs` (identical semantics), aggregating `max_abs`, `max_rel`,
   `failed/total`. Keep their seed scheme (`seed + trial`).
6. If accuracy fails → row with `accuracy_pass=False`, no timing (unless `--benchmark-on-failure`),
   `notes="FAIL: <failed>/<total> worst_index=... base=... opt=..."` — this is exactly what A needs to paste
   when debugging mask polarity.
7. Timing: call the organizers' `benchmark_models(...)` verbatim? It prints but returns nothing.
   So re-implement the *same* alternating-rounds loop with their `benchmark_once` and `warmup_model`
   (import them) and compute medians. Wrap baseline and optimized separately in try/except
   `torch.cuda.OutOfMemoryError`: if **baseline** OOMs, still run optimized and write
   `notes="baseline OOM; optimized completed"` with `baseline_median_ms` empty — that is a *result* (VRAM ceiling figure).
8. Peak VRAM: `torch.cuda.reset_peak_memory_stats()` before each model's timing; `max_memory_allocated()/2**20`
   after. `peak_vram_mb` = optimized's; baseline's goes to `baseline_peak_vram_mb` (proposed column) or notes.
9. Stop thermal logger; parse it → `mean_sm_clock_mhz`, `max_temp_c`, discard flag (2b).
   If discarded and not already retried: cooldown 60 s and rerun once; if it fails again, keep the row
   with `notes+="DISCARD:thermal (retried)"`.
10. Append row (csv module, `newline=""`), flush, print a one-line summary.
11. Cooldown `--cooldown` seconds (skip on cpu / `--no-cooldown`), `gc.collect()`, `torch.cuda.empty_cache()`.

**`--matrix` sets (one-factor-at-a-time, defaults B=8, S=512, d=512, H=8, L=6, fp32, no causal, pad 0):**
- `seq`: S ∈ {128, 512, 1024, 2048}
- `batch`: B ∈ {1, 8, 32}
- `dmodel`: d ∈ {256, 512, 1024} (H=8 always divides these)
- `dtype`: {float32, float16, bfloat16}
- `mask`: (pad 0.0, causal off), (0.3, off), (0.0, on), (0.3, on)
- `default` = all of the above, de-duplicated; `quick` = S∈{128,512}, dtype fp32, mask (0,off) — for A to run after every rung in < 5 min.
- Always run `--strategy baseline --matrix quick` first on Day 1: **every row must show speedup ≈ 1.00** (0.97–1.03). This is the hard gate in the sprint plan.

**2b. `bench/thermal.py`:**
```python
class ThermalLogger:  # context manager
    # spawns: nvidia-smi --query-gpu=timestamp,clocks.sm,temperature.gpu,power.draw,utilization.gpu --format=csv -l 1
    # writes logs/clocks_<git_sha>_<HHMM>_<strategy>_<B>x<S>x<d>.csv ; on Mac: no-op, returns None stats
def parse_clock_log(path) -> pandas.DataFrame   # columns: t (s from start), sm_mhz, temp_c, power_w, util_pct
def summarize(df, window=(t_start, t_end)) -> dict(mean_sm_clock_mhz, max_temp_c, opening_sm_mhz, discard: bool)
    # opening = mean of first 3 samples in window; discard = mean < 0.85 * opening   (sprint plan rule)
def wait_until_cool(max_temp_c=45, timeout_s=120)  # polls nvidia-smi; no-op without it
```
Write a small synthetic clock CSV fixture (`tests/fixtures/clocks_synthetic.csv`) that starts at 2400 MHz
and decays to 1600 MHz so the discard rule can be unit-tested.

**Acceptance (`tests/test_sweep_cpu.py`, all on CPU, < 10 s):**
- `python bench/sweep.py --strategy baseline --device cpu --batch 2 --seq-len 16 --d-model 64 --heads 4 --layers 2 --warmup 1 --repeats 3 --benchmark-rounds 1 --accuracy-trials 2 --no-thermal --no-cooldown --results /tmp/r.csv --allow-dirty`
  → one row, `accuracy_pass=True`, speedup 0.7–1.4, all 21 header columns in the agreed order.
- Same with `--causal --padding-ratio 0.3` → passes.
- `--matrix quick --device cpu` with tiny overrides → 2 rows appended, file not overwritten (run twice → 4 rows).
- A deliberately wrong strategy (test registers `class Wrong(BaselineTransformer): forward = lambda self,x,m=None: super().forward(x,m)*1.1`)
  → `accuracy_pass=False`, notes start with `FAIL:`, timing columns empty.
- `parse_clock_log` + `summarize` on the synthetic fixture → `discard == True`; on a flat fixture → `False`.

**Hand-off to A:** one message: "`python bench/sweep.py --strategy baseline --matrix quick` — paste the CSV rows; every speedup must be ≈1.00".

---

## Task 3 — Official-runner wrapper + CPU correctness suite  (Day 1, 11:00–13:00, with Task 1)

**3a. `bench/run_official.py`** — runs the organizers' script *unmodified* with our class injected:
```python
import sys, importlib
import bench.torch_transformer_benchmark as official
from src.optimized import UserOptimizedTransformer   # A's entry point
official.UserOptimizedTransformer = UserOptimizedTransformer   # monkeypatch, file untouched
if __name__ == "__main__":
    # forward all CLI args to the official parser
    raise SystemExit(official.main())
```
This is what the demo video runs and what the README's "reproduce" section points to.
Until A's `src/optimized.py` exists, fall back to `STRATEGIES["baseline"]` so the file is runnable.

**3b. `tests/test_strategies.py`** — the accuracy oracle A runs before every push, on *either* machine:
- Parametrize over: every name in `STRATEGIES` × shapes `[(2,16,64,4), (1,33,96,3), (3,64,128,8)]`
  (batch, seq, d_model, heads — includes non-power-of-2 S and d) × padding `{0.0, 0.3}` × causal `{False, True}`
  × dtype `{float32}` on CPU (+ `float16/bfloat16` only if `torch.cuda.is_available()`; on CPU bf16
  is allowed but mark `xfail(strict=False)` because CPU bf16 kernels can differ).
- Build baseline + strategy, `copy_model_weights`, run, `compare_outputs(ref, opt, rtol=0.01, atol=0.001)`.
  Assert `result.passed`; on failure print `max_abs, max_rel, worst_index, ref@worst, opt@worst, failed_feature_dims` —
  the exact fields the sprint plan says to paste at a checkpoint.
- Strategies that *cannot* run on CPU (Triton, CUDA-only) must expose a class attribute
  `REQUIRES_CUDA = True`; the test skips them on CPU with a visible SKIP reason. Tell A about this attribute.
- Also test the **control**: `STRATEGIES["baseline"]` must pass with `max_abs == 0.0` exactly.
- Extra edge-case test: `seq_len=1`, `batch=1`; `d_model=96, heads=3` (head_dim 32 but non-pow2 d).

**Acceptance:** `pytest -q tests/test_strategies.py` green on the Mac with only `"baseline"` registered.
Then A pulls, registers `sdpa`, runs the same command, and it either passes or tells him which
(padding, causal) branch is wrong. That is the whole point.

---

## Task 4 — Analysis library + figures (skeleton Day 1 09:45; fill in through Day 1/2)

Build against `tests/fixtures/results_synthetic.csv` **now** so every plot works before real data exists.
Generate the fixture with a script `tests/fixtures/make_synthetic.py` (≈ 60 rows: strategies baseline/sdpa/bf16/compiled/fused_qkv,
the OFAT matrix, plausible speedups that grow with S, small errors that grow with layers and are larger for fp16, a couple of
`SKIPPED`/`baseline OOM` rows, one `DISCARD:thermal` row).

**`analysis/load.py`:**
- `load_results(path) -> DataFrame` with dtypes fixed, `discarded` bool derived from notes, `status`
  ∈ {PASS, FAIL, SKIPPED, OOM_BASELINE, DISCARDED}, and a `config_key` column `(B,S,d,H,L,dtype,causal,pad)`.
- `latest_per_config(df)` — keep the newest row per (strategy, config_key) among non-discarded PASS rows.
- `geometric_mean_speedup(df, strategy, only_pass=True) -> float` plus min/max and n.
- `crossover_table(df) -> DataFrame`: for each config_key, the strategy with the highest speedup, its speedup,
  and the runner-up — this becomes the dispatch table.
- `write_dispatch_table(df, path="results/dispatch_table.json")` — JSON of
  `{"sm_89": {"<B>,<S>,<d>,<H>,<dtype>,<causal>,<pad>": "<strategy>", ...}, "default": "<best overall strategy>"}`
  plus a `"meta"` block with git_sha of the newest row and the geometric mean.
- `results_markdown_table(df)` — the speedup table for README/report (rows = config, cols = strategy).

**`analysis/figures.py`** (each function takes a DataFrame, saves `results/figures/<name>.png`, returns the path; matplotlib, no seaborn):
1. `speedup_vs_seq_len` — one line per strategy, horizontal 1.0x reference, log-x.
2. `speedup_vs_batch`
3. `speedup_vs_dmodel`
4. `accuracy_budget` — `max_abs_err` (and a second panel `max_rel_err`) vs `layers`, one line per dtype;
   horizontal lines at atol=0.001 and the PDF's 0.002. Needs rows with layers ∈ {1,2,4,6} — add a `--matrix layers`
   set to sweep.py (layers ∈ {1,2,4,6} at S=512) once A has time; the fixture already contains them.
5. `vram_ceiling` — `peak_vram_mb` vs `seq_len` for baseline and best strategy, with an "OOM" marker
   (red ✕ at the top) where `status == OOM_BASELINE`. Horizontal line at 6144 MB.
6. `thermal_trace(log_csv)` — SM clock (left axis) and temperature (right axis) vs time for one run,
   shaded bands for cooldowns if timestamps are given; used in the report's methodology section.
7. `compile_baseline_survival` — bar chart of speedup with vs without `--compile-baseline`, per strategy (from rows whose `compile_baseline` column / note says so).

**`analysis/roofline.py`:**
- `forward_flops(cfg) -> int`: per layer `8*B*S*d²` (Q,K,V,out projections) `+ 4*B*S²*d` (QKᵀ and PV) `+ 4*B*S*d*f` (FFN); × layers.
- `forward_bytes(cfg, dtype) -> int`: weights read once `params*e` + activations written/read ≈ `layers * (14*B*S*d + 4*B*S*f) * e`
  (state the approximation in a docstring; baseline also materializes `B*H*S²` scores ×~3 passes — include as `explicit_attention_bytes` so baseline and fused strategies get different intensities).
- `arithmetic_intensity = flops / bytes`; `achieved_tflops = flops / (median_ms/1e3) / 1e12`.
- `plot_roofline(df, peak_fp32_tflops=12.0, peak_bf16_tflops=48.0, bandwidth_gbs=192.0)`: log-log, roof lines for fp32 and bf16
  tensor cores, ridge points annotated (~63 FLOP/byte fp32), each (strategy, config) as a point. Parameters have defaults for the RTX 4050 Laptop; A confirms the peak numbers from the spec sheet and records them in the report.

**`analysis/make_all.py`:** `python -m analysis.make_all [--results results/results.csv] [--logs logs/]` → regenerates every figure, the dispatch table, and `results/summary.md` (geomean, min, max, n, table). This is what the README's "reproduce figures" line calls.

**Acceptance (`tests/test_analysis.py`):** on the synthetic fixture, every figure function returns an existing PNG > 10 kB; `geometric_mean_speedup` equals a hand-computed value on a 3-row toy frame; `write_dispatch_table` produces valid JSON with a `"default"` key; `make_all` runs end-to-end in < 30 s.

---

## Task 5 — `src/dispatch.py`  (Day 1, 15:15–16:45; A integrates 16:45–18:30)

**Goal:** turn measurements into the shape/capability → strategy decision the problem statement explicitly asks for ("participants can choose different implementations for different shapes by adding shape checks").

```python
@dataclass(frozen=True)
class DispatchKey: batch: int; seq_len: int; d_model: int; heads: int; dtype: str; causal: bool; padded: bool
def device_capability() -> tuple[int,int] | None      # torch.cuda.get_device_capability() or None on cpu
def select_strategy(key: DispatchKey, capability=None, table_path="results/dispatch_table.json") -> str
```
Rules, in order:
1. If no CUDA (capability None) → `"baseline"` (or `"sdpa"` if A says it is CPU-safe — confirm).
2. Load the JSON table (cached at import; falls back to a small **hard-coded** table if the file is missing so the submission never depends on a generated artifact).
3. Exact key match under `sm_<major><minor>` → that strategy.
4. Nearest-neighbour fallback: match on (dtype, causal, padded) then pick the row with the closest `seq_len` (log distance), then closest batch.
5. Capability gating (Rung 8): if capability < (8,0) → never select `bf16*` strategies; if < (7,5) → never select Triton/flash strategies. Keep this as an *extra axis on the same table*, not a separate system (team notes §Rung 8).
6. `explain(key) -> str` returns a one-line reason ("exact match sm_89 S=1024 → sdpa_bf16_compiled (2.7x)") for logging and for the video.

A's `src/optimized.py` calls `select_strategy(...)` once per forward (cheap: dict lookup; cache by key).

**Acceptance (`tests/test_dispatch.py`):** on the synthetic dispatch table: exact match, nearest-neighbour, capability gating (fake capability (7,5) never returns a bf16 name), missing file → hard-coded fallback, CPU → baseline.

---

## Task 6 — Documentation that accrues during the sprint (start Day 1 11:00, never stop)

**`docs/AI_USAGE.md`** — the problem statement gives *bonus points* for this. Template per entry:
```markdown
## 2026-08-27 14:10 — <what was being built>
**Tool:** Claude Code / Claude chat / Codex
**Prompt:** <paste verbatim or summarize in ≤3 lines>
**Output:** worked / failed / needed correction
**What I had to fix:** <specifics>
**Verification:** pytest green / accuracy PASS max_abs=… / plot inspected
```
Log *every* Claude Code session from this plan. Also ask A to log his (he will not unless nagged — nag).

**`README.md`** skeleton on Day 1 (sections required by the problem statement): Project overview · Environment (A fills: CPU, GPU, VRAM, driver, WSL2, PyTorch/Triton versions) · Setup (Linux/WSL2 with cu124; macOS CPU-only for tests and analysis) · Reproduce (`pytest -q`; `python bench/sweep.py --strategy baseline --matrix quick`; `python bench/run_official.py …`; `python -m analysis.make_all`) · Results (paste `results/summary.md` table + headline figure) · Limitations & what we'd do with more time · Team contributions (A: strategies, GPU measurement; B: harness, tests, dispatch, analysis, report) · AI usage (link to docs/AI_USAGE.md).

---

## Task 7 — Profiler-trace figures  (Day 2, 09:00–11:30, parallel with A's Triton attempt)

A's `bench/profile_baseline.py` uses `torch.profiler` and can export Chrome traces (`prof.export_chrome_trace("logs/trace_<name>.json")`). Ask A to export one trace per shape for baseline and for the final optimized path (6 JSON files). Nsight Systems is optional; the chrome trace is enough and needs no Mac-side NVIDIA tools.

**`analysis/trace.py`:**
- `load_trace(path) -> DataFrame` of events with `name, cat, ts_us, dur_us, stream/tid`; GPU kernels are events whose `cat` is `kernel` (or `"Kernel"`/`gpu_op` depending on torch version — handle both) .
- `gpu_busy_fraction(df)`: union of kernel intervals ÷ total span. This is the "GPU busy %" number that Rung 0 hinges on (low → launch-bound → CUDA graphs first).
- `kernel_breakdown(df, top=12)`: time share per kernel name (matmul / softmax / masked_fill / layernorm / elementwise) — the "where time goes at each shape" table for the report.
- `plot_timeline(df_baseline, df_optimized, window_ms=5)`: two stacked strips, one bar per kernel, white = idle. This is the report's and video's strongest visual ("gap-filled baseline beside dense optimized").
- `plot_busy_vs_shape(traces: dict)`: bar chart of busy % for small/medium/large × baseline/optimized.

**Acceptance:** works on a synthetic trace JSON fixture with 20 fake kernel events; busy fraction of a hand-built 50 %-idle fixture equals 0.5 ± 0.01.

---

## Task 8 — Report, Devpost, video script  (Day 2, 13:15–15:30 draft; A reviews)

**`docs/TECH_REPORT.md`** — sections in this order (from the sprint plan + problem statement):
1. Problem framing: one fixed Transformer, one function to replace, correctness oracle, speedup = the score.
   Include the "softmax is the only reason attention is quadratic" paragraph and the three-bottleneck table (launch / bandwidth / compute).
2. Environment: CPU, GPU, VRAM, driver, WSL2, PyTorch, Triton (A supplies), plus the peak FLOPS/bandwidth used for the roofline.
3. Rung 0 profile: GPU busy % per shape, classification of each shape's bottleneck (Task 7).
4. Each optimization (rungs 1,2,3,4,5,[6]) with hypothesis → measured before/after → surprise. Pull from A's PR descriptions.
5. Shape & device dispatch: crossover table, dispatch rules, explicit "paths for sm_75/sm_80 are correctness-tested by forced dispatch; performance measured only on sm_89".
6. Roofline analysis against the ridge point.
7. Accuracy budget: error vs dtype vs depth; which tolerance we targeted and margin.
8. Thermal methodology: why median, why alternating order, why cooldowns, which runs were discarded (count them from `results.csv`), clock trace figure.
9. VRAM ceiling: where baseline OOMs and ours doesn't.
10. `--compile-baseline` honesty check: how much speedup survives.
11. Limitations, plainly. Negative results (e.g., Triton LayerNorm if abandoned) with the analysis of why.
12. AI tools used (summary + link to AI_USAGE.md).

**`docs/DEVPOST.md`** — every field the problem statement lists: how the solution addresses the statement; dev tools (VS Code, WSL2, Claude Code, Codex, Jupyter); APIs (Claude, Codex); libraries (PyTorch, Triton, pandas, matplotlib); datasets (none — synthetic random inputs from the organizers' generator); repo link; YouTube link (public).

**`docs/VIDEO_SCRIPT.md`** — the 3-minute table from the sprint plan expanded into: exact terminal commands A types on screen (`nvidia-smi`, `python bench/run_official.py --batch-size 8 --seq-len 1024 --dtype bfloat16`, `pytest -q`), which figure to show at which timestamp, the three sentences on dispatch/thermal/limitations. Reminder: no third-party logos/trademarks on screen beyond what's unavoidable.

---

## Task 9 — Fresh-clone verification on the Mac  (Day 2, 17:00–18:00)

While A does the GPU fresh-clone test, B does the *other OS* one: new directory, `git clone`, follow README exactly,
`pip install -e .`, `pytest -q`, `python bench/sweep.py --strategy baseline --matrix quick --device cpu` with tiny overrides,
`python -m analysis.make_all`. Every command in README must work verbatim on macOS-CPU or be clearly marked "GPU/WSL2 only".
Fix the README, not the reader. Then submit Devpost (Task 8 text, repo link, public video link).

---

## Schedule (B only — measurement lines from the original plan are now A's)

| When | B does | Hands to A |
|---|---|---|
| Tonight | Env (0.5), Task 0, read Horace He "Go Brrr" + FlashAttention §3 | INTERFACE.md for approval |
| Day 1 09:00–11:00 | Task 2 (sweep.py + thermal.py) | "run `--strategy baseline --matrix quick`, paste rows" |
| Day 1 09:45–11:00 (overlap) | Task 4 skeleton: fixture + load.py + thermal_trace figure | — |
| Day 1 11:00–13:00 | Task 1 (memcheck) + Task 3 (run_official + test_strategies) + AI_USAGE.md | "run `pytest -q` before every push" |
| Day 1 13:45–15:15 | Task 4: accuracy_budget, speedup_vs_* figures; README skeleton | — |
| Day 1 15:15–16:45 | Task 4: crossover table → dispatch_table.json; Task 5 dispatch.py | dispatch.py + JSON |
| Day 1 16:45–18:30 | Geomean + results_markdown_table; draft report §4 from A's rows | summary.md |
| Day 1 18:30–19:30 | README complete; review A's PR; `git tag v1-day1` | — |
| Day 2 09:00–11:30 | Task 7 trace figures; roofline plot | ask A for 6 trace JSONs |
| Day 2 11:30–12:30 | vram_ceiling figure; compile_baseline_survival figure | — |
| Day 2 13:15–15:30 | Task 8 report + Devpost + video script | A reviews |
| Day 2 15:30–17:00 | Insert final figures/numbers into report and README while A records | — |
| Day 2 17:00–18:00 | Task 9 fresh-clone on Mac | — |
| Day 2 18:00–19:00 | Devpost submission | — |

**Cut list if behind (in order):** Task 7 timeline plot (keep busy-% numbers in text) → roofline figure (keep the ridge-point number in text) → accuracy_budget figure → `--matrix layers`. **Never cut:** Tasks 0, 2, 3, README, AI_USAGE.md, Devpost.

---

## Claude Code prompts (copy/paste, one per session)

- **Task 0:** "Read PLAN_PERSON_B.md sections 0 and Task 0. Create the repo skeleton exactly as in 0.3, write pyproject.toml, src/baseline.py, src/strategies/__init__.py with the STRATEGIES registry, CLAUDE.md, docs/INTERFACE.md, .gitignore, requirements.txt. Do not modify bench/torch_transformer_benchmark.py. Run the acceptance commands and show me the output."
- **Task 2:** "Read PLAN_PERSON_B.md Task 2. Implement bench/sweep.py and bench/thermal.py reusing the organizers' functions from src/baseline.py without editing the organizers' file. Everything must run on CPU when CUDA/nvidia-smi are absent. Write tests/test_sweep_cpu.py and the synthetic clock fixture, run pytest, and show me the resulting /tmp/r.csv header and rows."
- **Task 3:** "Read PLAN_PERSON_B.md Task 3. Write bench/run_official.py and tests/test_strategies.py. Confirm the 'baseline' strategy passes with max_abs == 0.0 across all mask/causal/shape combinations on CPU."
- **Task 4:** "Read PLAN_PERSON_B.md Task 4. First write tests/fixtures/make_synthetic.py and generate the fixture. Then implement analysis/load.py, figures.py, roofline.py, make_all.py and tests/test_analysis.py. Run make_all on the fixture and list the PNGs produced."
- **Task 5 / 7 / 8:** same pattern: "Read PLAN_PERSON_B.md Task N, implement, run acceptance, show output."

After every session: append an entry to `docs/AI_USAGE.md`, run `pytest -q`, commit, push.
