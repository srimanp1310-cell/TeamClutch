# Rules for Claude Code in this repo

1. **Never edit `bench/torch_transformer_benchmark.py`** (or the tensorflow one) — organizers' files.
   Import them via `src/baseline.py`; their SHA-256 is pinned in `docs/INTERFACE.md`.
2. **Never touch `src/strategies/*` or `src/optimized.py`** — Person A's. B owns `bench/sweep.py`,
   `bench/thermal.py`, `bench/run_official.py`, `src/memcheck.py`, `src/dispatch.py`, `analysis/*`, `tests/*`, `docs/*`.
3. **Everything B writes must run on macOS with no GPU.** Guard GPU paths with `torch.cuda.is_available()` /
   `shutil.which("nvidia-smi")`; degrade gracefully, never crash, never skip silently.
4. **Correctness before speed.** Tolerance is the stricter pair `atol=0.001, rtol=0.01`, per element, OR-combined.
5. **`results/results.csv` is append-only** — never regenerate or reorder. Column order is fixed in `docs/INTERFACE.md`.
6. **No implementations in notebooks.** Logic lives in `src/`, `bench/`, `analysis/`.
7. Follow `PLAN_PERSON_B.md`: one task per session, do not skip ahead.

Smoke test (~1 s, CPU; add `--padding-ratio 0.3 --causal` for the mask branches):

```bash
python bench/torch_transformer_benchmark.py --device cpu --batch-size 2 --seq-len 16 --d-model 64 --heads 4 --ffn-dim 128 --layers 2 --accuracy-trials 2 --warmup 1 --repeats 3 --benchmark-rounds 1
```

Before every commit: `pytest -q`. After every session: append an entry to `docs/AI_USAGE.md` (bonus points).
