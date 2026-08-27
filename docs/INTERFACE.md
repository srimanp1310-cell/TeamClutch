# INTERFACE.md — the A↔B contract

Status: **DRAFT, awaiting Person A's approval in chat before Day 1.**
Nothing below may change without telling the other person; these are the seams
where our two halves of the repo meet.

---

## 1. The organizers' file is frozen

`bench/torch_transformer_benchmark.py` is the organizers' script, byte-identical
to the download. We import from it; we never edit it.

```
SHA-256  1bd12523657f338c09b53f0bb9052d9d16f728a71bd22bc8298567e1a4d78c22
```

Verify at any time:

```bash
shasum -a 256 bench/torch_transformer_benchmark.py
```

If that hash ever changes, the run is not comparable to the organizers' own and
the result is void. `bench/tensorflow_transformer_benchmark.py` is also kept
unmodified — we do not run it, we only ported its memory-estimation math
(Task 1) and cite it there.

Everything imports the organizers' symbols through **`src/baseline.py`**, which
is a pure re-export module. That keeps one greppable importer, and guarantees
our accuracy/timing numbers come from the same functions the organizers' own
`main()` calls.

---

## 2. Strategy signature

Every optimized implementation ("strategy") is a class that:

* subclasses `BaselineTransformer`;
* overrides exactly this method, with **no extra required arguments**:

  ```python
  def forward(
      self,
      x: torch.Tensor,
      valid_token_mask: Optional[torch.Tensor] = None,
  ) -> torch.Tensor
  ```

* returns a tensor of shape `[batch_size, seq_len, d_model]` in the same dtype
  as `x`;
* **keeps the baseline's parameter names identical**, so that the organizers'
  `copy_model_weights(baseline, optimized, strict=True)` succeeds untouched.
  A fused-QKV strategy must still register `q_proj/k_proj/v_proj` under those
  names and build the fused view from them — never rename a parameter.
  (`--non-strict-weight-copy` exists in the organizers' script, but using it
  means our weights are not provably the same weights. Do not rely on it.)
* sets the class attribute `REQUIRES_CUDA = True` if it cannot run on CPU
  (Triton kernels, CUDA extensions). `tests/test_strategies.py` then SKIPs it
  on Person B's Mac with a visible reason instead of reporting a false failure.

### Masking semantics that must be reproduced exactly

Read these off the baseline before optimizing — they are the usual source of a
failing accuracy check:

* `valid_token_mask` is `[B, S]`, `torch.bool`, **True = keep**.
* Invalid *key* positions are masked inside attention: `~mask[:, None, None, :]`.
* The attention output is zeroed at invalid *query* positions
  (`output.masked_fill(~valid_token_mask[..., None], 0)`), and so is each block
  output and the final output after `final_norm`.
* Causal masking is `triu(diagonal=1)` — strictly upper triangular is masked.
* Softmax is computed **in fp32** and cast back (`torch.softmax(scores.float(), -1).to(x.dtype)`).
  A reduced-precision softmax is the most likely cause of a `max_abs` just over
  budget in fp16/bf16.

---

## 3. Registry

`src/strategies/__init__.py` owns:

```python
STRATEGIES: dict[str, type[nn.Module]]   # always contains "baseline"
def register(name): ...                  # class decorator
def get_strategy(name): ...              # KeyError lists the known names
```

`"baseline"` maps to `BaselineCopy`, an unmodified subclass. It is the control:
**every `--strategy baseline` run must report `max_abs_err == 0.0` and a speedup
of ~1.00x (0.97–1.03).** If it does not, the harness is wrong and no other row
in `results.csv` can be trusted. This is the Day-1 hard gate.

Submodules of `src/strategies/` are auto-imported, so adding a file is enough —
no edit to `__init__.py`. A submodule that fails to import on this machine
(e.g. `import triton` on macOS) lands in `UNAVAILABLE` instead of breaking the
registry for everyone.

**Person A owns** `src/strategies/*` and `src/optimized.py`.
**Person B owns** everything else. `src/optimized.py` defines
`UserOptimizedTransformer`, which calls `src.dispatch.select_strategy(...)`
(Task 5) and delegates to the chosen strategy.

---

## 4. results.csv schema

`results/results.csv` is **append-only**. Never regenerate it, never sort it in
place, never overwrite it. Every row carries the `git_sha` it was produced at.
Columns, in this exact order:

```
timestamp, git_sha, strategy_name, batch, seq_len, d_model, heads, layers,
dtype, causal, padding_ratio,
accuracy_pass, max_abs_err, max_rel_err,
baseline_median_ms, optimized_median_ms, speedup,
peak_vram_mb, mean_sm_clock_mhz, max_temp_c, notes
```

**Proposed additions — appended at the END only, A to confirm:**

```
baseline_peak_vram_mb, compile_baseline, ffn_dim
```

Rationale: `baseline_peak_vram_mb` is needed for the VRAM-ceiling figure (where
baseline OOMs and ours does not); `compile_baseline` for the honesty check
("how much of our speedup survives when the baseline is compiled too");
`ffn_dim` because it is a free config axis that the other columns do not pin
down. Appending keeps every already-written row parseable.

`notes` is free text but uses these reserved prefixes, which the analysis layer
parses into a `status` column:

| prefix | meaning |
|---|---|
| `SKIPPED: <reason>` | memory pre-check refused to run the config |
| `FAIL: <n>/<total> worst_index=…` | accuracy check failed; timing columns empty |
| `baseline OOM` | baseline could not run; optimized did. This is a *result*. |
| `DISCARD:thermal` | clocks dropped >15% mid-run; row kept but excluded from summaries |

A status prefix always comes **first** in `notes`; free text (`--notes`) and the
`dirty` provenance marker are appended after it, separated by `; `. A prefix
that is not at the start is not a prefix, and `analysis/load.py` derives the
`status` column by matching on the start of the field.

---

## 5. Tolerance

We target the **stricter** of the two published tolerances: the torch script's
defaults, `atol = 0.001` and `rtol = 0.01` — not the problem statement PDF's
`abs < 0.002 / rel < 0.02`. The rule is per element, and it is an **OR**:

```
abs(user - ref) <= atol   OR   abs(user - ref) <= rtol * abs(ref)
```

Passing at 0.001/0.01 passes at 0.002/0.02 with margin to spare, and the margin
is a number worth putting in the report.

---

## 6. Smoke test (runs anywhere, ~1 s, no GPU)

```bash
python bench/torch_transformer_benchmark.py --device cpu --batch-size 2 --seq-len 16 \
  --d-model 64 --heads 4 --ffn-dim 128 --layers 2 --accuracy-trials 2 \
  --warmup 1 --repeats 3 --benchmark-rounds 1
```

Add `--padding-ratio 0.3 --causal` to exercise both mask branches.
Expect `summary: PASS` and a speedup near 1.0x (CPU timing is noisy; 0.8–1.2 is fine).

Run `pytest -q` before every commit.