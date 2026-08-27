# AI usage log

The problem statement awards bonus points for a clear account of the AI
skills/tools used. This file is written **contemporaneously** — one entry per
working session, added at the end of that session, not reconstructed afterwards.
That is the point: it records what the tools got wrong as well as what they got
right.

Template:

```markdown
## YYYY-MM-DD HH:MM — <what was being built>
**Tool:** Claude Code / Claude chat / Codex
**Prompt:** <paste verbatim or summarize in <=3 lines>
**Output:** worked / failed / needed correction
**What I had to fix:** <specifics>
**Verification:** pytest green / accuracy PASS max_abs=... / plot inspected
```

---

## 2026-08-27 — Planning: scoping Person B's half of the project
**Tool:** Claude chat
**Prompt:** Given the organizers' `torch_transformer_benchmark.py`, the problem
statement, and our two-person / one-GPU constraint, produce a task-by-task plan
for the non-kernel half of the work (harness, correctness suite, dispatch,
analysis, deliverables), with acceptance tests that run on a Mac with no GPU.
**Output:** worked — produced `PLAN_PERSON_B.md` (Tasks 0-9, schedule, cut list).
**What I had to fix:** nothing structural at this stage; the plan's open
questions (three proposed extra CSV columns, whether an SDPA strategy is
CPU-safe) were deliberately left for Person A to confirm rather than guessed at.
**Verification:** n/a — planning artefact. Its assumptions get tested by Task 0's
acceptance run.

## 2026-08-27 — Task 0: repo skeleton, contract, Claude Code rules
**Tool:** Claude Code (Opus 5)
**Prompt:** Read `PLAN_PERSON_B.md` sections 0 and Task 0. Create the repo
skeleton exactly as in 0.3, write `pyproject.toml`, `src/baseline.py`,
`src/strategies/__init__.py` with the STRATEGIES registry, `CLAUDE.md`,
`docs/INTERFACE.md`, `.gitignore`, `requirements.txt`. Do not modify the
organizers' file. Run the acceptance commands.
**Output:** worked, with two corrections needed (below).
**What I had to fix:**
- The machine's default Python was 3.14; the venv was rebuilt on Python 3.11
  (arm64) to get a PyTorch wheel. Recorded in the README setup section, since
  anyone cloning this on a Mac hits the same wall.
- `[tool.setuptools] packages = ["src","bench","analysis"]` as written in the
  plan does **not** pick up `src.strategies` — setuptools does not descend into
  subpackages of an explicitly listed package, so `import src.strategies` failed
  from outside the repo root until `"src.strategies"` was listed separately.
- The plan's registry skeleton has no way for Person A's strategy files to be
  *imported*, so their `@register` decorators would never run and `STRATEGIES`
  would stay `{"baseline": ...}` forever. Added `_autodiscover()` over
  `pkgutil.iter_modules`, with failed imports collected into `UNAVAILABLE`
  (so `import triton` on macOS degrades to a skip reason instead of taking the
  whole registry down).
**Verification:** `pip install -e .` then `python -c "import src.baseline,
src.strategies, bench, analysis"` succeeds from a directory outside the repo.
Organizers' script on CPU: `summary: PASS | max_abs=0 | max_rel=0`,
`speedup 1.047x`; with `--causal --padding-ratio 0.3`: `PASS`, `0.964x`.
`shasum -a 256 bench/torch_transformer_benchmark.py` matches the value pinned in
`docs/INTERFACE.md`.

## 2026-08-27 — Tasks 1 and 2: memory pre-check, sweep harness, thermal logging
**Tool:** Claude Code (Opus 5)
**Prompt:** Continue with Task 2 of `PLAN_PERSON_B.md`: implement `bench/sweep.py`
and `bench/thermal.py` reusing the organizers' functions via `src/baseline.py`
without editing the organizers' file; everything must run on CPU when CUDA and
nvidia-smi are absent. Write `tests/test_sweep_cpu.py` and the synthetic clock
fixture, run pytest, show the resulting CSV.
**Output:** worked, after resolving one ordering problem in the plan and three
defects found by the tests.
**What I had to fix:**
- **Task ordering.** `run_config` step 2 calls the Task 1 memory pre-check, so
  Task 2 could not be finished before Task 1. Implemented `src/memcheck.py`
  first (the math is ported unchanged from the organizers' TensorFlow file, so
  both benchmarks agree on which configs are impossible).
- **`notes` prefix ordering.** The `dirty` provenance marker was being written
  before the status prefix, producing `"dirty; FAIL: ..."`. The contract says
  the analysis layer parses `notes` on its *prefix*, so a status that is not at
  position 0 is unparseable. Status now always leads; caught by the test that
  asserts `notes.startswith("FAIL:")`.
- **Peak-VRAM measurement.** The plan says to reset peak-memory stats "before
  each model's timing", but the timing loop deliberately *interleaves* the two
  models to cancel clock drift, and `max_memory_allocated` is global — during
  that loop the two peaks are not separable, so the two CSV columns could not
  both be filled. Moved peak measurement into a dedicated single-forward pass
  per model, before the timed loop.
- **Noisy acceptance threshold.** The plan's example timing settings
  (`--repeats 3 --benchmark-rounds 1`) give a CPU speedup that swings 0.7x-1.5x
  run to run; a control asserted to be ~1.00x would fail intermittently. Tests
  use `--warmup 5 --repeats 25 --benchmark-rounds 3`, which costs 0.7 s and
  holds the control inside 1.00 +/- 0.02.
- Added a `--matrix layers` set (L in {1,2,4,6}) now rather than later, because
  Task 4's accuracy-budget figure needs those rows.
**Verification:** `pytest -q` green, 53 tests, 5.6 s, no GPU. Fidelity checked
directly against the organizers' script at B=4 S=64 d=128 H=4 L=3 pad=0.3 seed
1234: both report `max_abs=0, max_rel=0, PASS`; baseline medians 1.3407 ms
(theirs) vs 1.3503 ms (ours), within CPU timing noise. Discard rule verified on
the synthetic clock fixtures: throttling log mean 1815 MHz vs opening 2396 MHz
-> `discard=True`; flat log 2398 vs 2395 -> `discard=False`.

## 2026-08-27 — Task 3: official-runner wrapper and the CPU correctness oracle
**Tool:** Claude Code (Opus 5)
**Prompt:** Read `PLAN_PERSON_B.md` Task 3. Write `bench/run_official.py` and
`tests/test_strategies.py`. Confirm the "baseline" strategy passes with
`max_abs == 0.0` across all mask/causal/shape combinations on CPU.
**Output:** worked; two of my own test expectations were wrong and one probe
found a false assumption about the baseline.
**What I had to fix:**
- **A "wrong" strategy that wasn't wrong.** The oracle's self-test scaled the
  output by 1.001 to prove the comparison can fail — but the pass rule is an
  OR, and 0.1% is comfortably inside `rtol=0.01`, so it correctly passed.
  Reworked to 1.02, which is just past rtol and past atol wherever `|ref| >
  0.05`. That makes it a sensitivity bound (the smallest error the oracle must
  never miss) instead of a smoke test.
- **A renamed-parameter test expecting the wrong exception text.** Adding a
  parameter to a strategy raises "Missing key(s)", not "Unexpected key(s)" —
  the extra key is missing *from the baseline's* state_dict.
- **A probe bug that turned out not to be a bug.** I first tested the
  diagnostics by deleting the final `masked_fill` after `final_norm`, expecting
  a padding failure. It passed, correctly: each block already zeroes padded
  positions, and `nn.LayerNorm` initialises `bias=0`, so `final_norm(0) == 0`.
  Replaced it with a causal off-by-one (`triu(diagonal=0)`), which fails only
  the causal branch — the behaviour Task 3 is actually for.
- **Added a NaN hint to the failure report.** The off-by-one probe produced
  `max_abs_err: nan`, which reads like a broken metric. It isn't: a fully-masked
  row makes softmax over all `-inf` return NaN. The report now says so and names
  the two usual causes (causal diagonal, mask polarity).
**Verification:** `pytest -q` green — 84 passed, 2 skipped (fp16/bf16 are only
asserted where there is a GPU), 1 xpassed (bf16 on CPU, non-strict), 5.5 s.
Diagnostics verified by injecting the causal off-by-one: `causal=False` passes
and `causal=True` fails, with the failing branch named in the test id
(`[causal-nopad-B2_S16_d64_H4-...]`). `run_official.py` runs the organizers'
`main()` with our class injected and their file's SHA-256 unchanged.
