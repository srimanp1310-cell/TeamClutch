# Rules for Claude Code in this repo

1. **Never edit `bench/torch_transformer_benchmark.py`** (or the tensorflow one) — organizers' files.
   Import them via `src/baseline.py`; their SHA-256 is pinned in `docs/INTERFACE.md`.
2. **Two areas, kept separate.** Kernel implementations live in `src/strategies/*` and
   `src/optimized.py`; the measurement harness is `bench/*`, `src/memcheck.py`, `src/dispatch.py`,
   `analysis/*` and `tests/*`. Changing one while measuring with the other invalidates the numbers.
3. **Everything B writes must run on macOS with no GPU.** Guard GPU paths with `torch.cuda.is_available()` /
   `shutil.which("nvidia-smi")`; degrade gracefully, never crash, never skip silently.
4. **Correctness before speed.** Tolerance is the stricter pair `atol=0.001, rtol=0.01`, per element, OR-combined.
5. **`results/results.csv` is append-only** — never regenerate or reorder. Column order is fixed in `docs/INTERFACE.md`.
6. **No implementations in notebooks.** Logic lives in `src/`, `bench/`, `analysis/`.
7. Follow `docs/PROJECT_PLAN.md`: one task per session, do not skip ahead.

Smoke test (~1 s, CPU; add `--padding-ratio 0.3 --causal` for the mask branches):

```bash
python bench/torch_transformer_benchmark.py --device cpu --batch-size 2 --seq-len 16 --d-model 64 --heads 4 --ffn-dim 128 --layers 2 --accuracy-trials 2 --warmup 1 --repeats 3 --benchmark-rounds 1
```

Before every commit: `pytest -q`. After every session: append an entry to `docs/AI_USAGE.md` (bonus points).

## Measured hardware (RTX 4050 Laptop, low-TGP — NOT spec sheet)

fp32 TF32-off 5.7 · fp32 TF32-on 11.0 · fp16 22.5 · bf16 23.2 TFLOPS
Bandwidth 174.8 GB/s · VRAM 6.0 GB · capability (8, 9)
Ridge points: 62.9 FLOP/byte fp32(TF32), 128.7 fp16, 132.7 bf16 — the ridge
moves RIGHT as precision drops, so reduced precision can make a workload MORE
memory-bound, not less.

**At the benchmark's default depth (6 layers), fp32 is the only shippable
dtype.** bf16 fails at every depth: the worst element is 2 ULP of bf16 (1.44%
relative) against a 1% rtol, because the reference rounds softmax probabilities
to bf16 before the PV matmul and a fused kernel does not. fp16 passes at **one**
layer (2.020x) and fails from two, once error has compounded through the
residual stream. See docs/INTERFACE.md §5.1 and TECH_REPORT §7.2.

`sdpa` therefore declares `SUPPORTED_DTYPES = (torch.float32, torch.float16)` --
correct, because fp16 genuinely passes at L=1 -- and `src/dispatch.py` never
routes bf16 to it.

**`select_strategy()` is depth-blind, on purpose.** `DispatchKey` carries no
layer count, because depth changes how long a forward takes but not which kernel
suits a shape -- true for performance, false for correctness. So dispatch alone
will hand `sdpa` an fp16 tensor at any depth. The fp16 depth rule lives in
`src/optimized.py` (rule 3), which is the entry point the organizers' script
instantiates, so **the shipped path is safe**. Calling `select_strategy()`
directly, or `sweep.py --strategy sdpa --dtype float16 --layers 6`, is not --
that combination is a known-failing config and only useful for reproducing the
failure.

## Non-negotiable rules

- Never edit bench/torch_transformer_benchmark.py — SHA-256 pinned, a test fails.
- Harness files are pinned by the test suite: bench/sweep.py, bench/thermal.py,
  bench/run_official.py, src/memcheck.py, src/dispatch.py, src/baseline.py,
  src/strategies/__init__.py, analysis/*, tests/*, pyproject.toml. Change one and
  re-run the suite before trusting any number produced afterwards.
- Strategies subclass BaselineTransformer, decorate @register("name"),
  signature forward(self, x, valid_token_mask=None).
- Never rename or add parameters — copy_model_weights(strict=True) must pass.
- Run `pytest -q` before every push. Ten seconds, CPU only.

## Masking rules (usual source of failures)

- valid_token_mask is [B,S] bool, True means KEEP (opposite of masked_fill).
- Invalid KEY positions masked in attention: ~mask[:, None, None, :]
- Output zeroed at invalid QUERY positions in several places.
- Causal is triu(diagonal=1). diagonal=0 masks the diagonal -> NaN.
- Softmax runs in fp32 and casts back even in bf16. Reduced-precision softmax
  puts max_abs just over budget.

## Context

48-hour hackathon. Submittable by end of Day 1. Breadth over depth.
Current rung: 0 (profiling). Next: SDPA.
