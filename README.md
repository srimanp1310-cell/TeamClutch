# Transformer GPU Kernel Optimization — TikTok TechJam 2026, Task 3

> **Status: skeleton (Task 0 complete).** Sections marked _TODO_ are filled in
> during the sprint — measured numbers land here from `results/results.csv`, and
> the environment block is supplied by Person A from the GPU machine.

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

_TODO (Person A): CPU, GPU, VRAM, driver version, WSL2 version, PyTorch, Triton,
CUDA. Plus peak FP32/BF16 TFLOPS and memory bandwidth for the roofline analysis._

Person B's machine (harness development, CPU tests, all analysis):
macOS on Apple Silicon, PyTorch 2.13.0, CPU-only (`torch.cuda.is_available() == False`).

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

_TODO (Person A): confirm the exact index URL / Triton version used._

## Reproduce

```bash
pytest -q                                                    # correctness oracle, CPU, no GPU needed
python bench/sweep.py --strategy baseline --matrix quick     # control: every speedup must be ~1.00x
python bench/run_official.py --batch-size 8 --seq-len 1024   # the organizers' own script, our class injected
python -m src.memcheck --batch 8 --seq-len 4096              # will this config OOM? (no GPU needed)
python -m analysis.make_all                                  # regenerate every figure and results/summary.md
```

To see what the figures look like before any GPU run exists, render them from
the synthetic fixture into a scratch directory (the summary it writes is stamped
SYNTHETIC, and nothing lands in `results/`):

```bash
python -m analysis.make_all --results tests/fixtures/results_synthetic.csv \
  --logs tests/fixtures --figures /tmp/preview --summary /tmp/preview/summary.md \
  --dispatch /tmp/preview/dispatch.json
```

Commands that require a GPU are marked _GPU/WSL2 only_ where they appear.
Everything above except `run_official.py` at large shapes runs on macOS CPU.

Verify the organizers' file was never modified:

```bash
shasum -a 256 bench/torch_transformer_benchmark.py   # must match docs/INTERFACE.md
```

## Results

_TODO: `results/summary.md` table (geometric-mean speedup, min, max, n) and the
headline figure, pasted here once the sweep has real rows._

## Limitations and what we would do with more time

_TODO — written from measured evidence, including negative results._

## Team contributions

- **Person A** — optimized implementations (`src/strategies/`, `src/optimized.py`),
  all GPU measurement, profiler traces.
- **Person B** — sweep harness and thermal logging, memory pre-check, CPU
  correctness suite, shape/capability dispatch, analysis and figures, report and
  submission materials.

## AI usage

See [docs/AI_USAGE.md](docs/AI_USAGE.md) for a contemporaneous, per-session log
of which AI tools were used, what they produced, and what had to be corrected.

## Repo map

See `docs/INTERFACE.md` for the A↔B contract: strategy signature, masking
semantics, registry, the `results.csv` schema, and the tolerance we target.
