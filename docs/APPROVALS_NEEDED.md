# For Person A — things I need from you

I've built the harness half of the repo (Tasks 0–2 of `PLAN_PERSON_B.md`).
Before Day 1 I need a few decisions from you, because they're baked into files
you'll be using and they're expensive to change later.

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
2. **6 profiler traces** (Task 7), whenever you get to profiling — one chrome
   trace per shape (small / medium / large) for baseline and for the final
   optimized path: `prof.export_chrome_trace("logs/trace_<name>.json")`.
3. **Your AI usage log entries.** The problem statement gives *bonus points* for
   this and it has to be written as you go, not reconstructed at the end. Just
   append to `docs/AI_USAGE.md` — there's a template at the top. One entry per
   session: what you asked, whether it worked, what you had to fix.
4. **Your PR descriptions** — I'm writing report section 4 (one subsection per
   optimization: hypothesis → measured before/after → surprise) straight from
   them, so the more you write there, the less I have to invent.

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
