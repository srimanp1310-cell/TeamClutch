# Optimizing a Transformer forward pass for one GPU

**TikTok TechJam 2026 — Problem 3: Implement a GPU Kernel for a Transformer Layer**

> **Status: draft.** Everything that follows from the problem definition, the
> reference implementation or our methodology is final. Slots marked
> `<FILL A: …>` need a number from the GPU machine; `<FILL B: …>` are filled
> from `results/summary.md` once the sweep has run. Run
> `python docs/check_ready.py` to list what is still outstanding.

---

## 1. What the problem actually is

The organizers supply a fixed Transformer and a correct-but-slow reference
implementation of its forward pass. Exactly one method may be replaced —
`UserOptimizedTransformer.forward()`. Everything else, including the code that
grades the result, stays theirs and untouched.

Two constraints define the whole exercise:

**The correctness oracle is elementwise and unforgiving.** Every output element
must satisfy

```
abs(user − ref) <= atol   OR   abs(user − ref) <= rtol · abs(ref)
```

It is an OR, not an AND, and it is per element, not an aggregate — a single bad
element fails the run no matter how good the mean error is. We target the
stricter of the two published tolerance pairs: the torch script's own defaults
`atol = 0.001, rtol = 0.01`, not the problem statement's `0.002 / 0.02`.

**The score is a ratio of medians**, `baseline_median_ms / optimized_median_ms`,
measured on the same machine in the same process. That makes the *measurement
apparatus* part of the deliverable: a speedup that moves when the room warms up
is not a result. Section 8 is about what we did to make the number stable.

### 1.1 Why attention is quadratic, and what that costs

The quadratic term is not intrinsic to attention. Writing it out:

```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V
```

Without the softmax, this is a product of three matrices, and matrix
multiplication is associative: `(QKᵀ)V` costs `O(S²·d)` but `Q(KᵀV)` costs
`O(S·d²)` — linear in sequence length. **The softmax is the only reason the
regrouping is illegal.** It is a nonlinearity sitting between `QKᵀ` and `V`, so
the `S × S` matrix must exist before `V` can be applied.

That is why every serious attention optimization is an argument about the
softmax: not removing it, but avoiding *materializing* the matrix it operates
on. Flash-attention-style kernels compute the softmax in tiles with a running
maximum and a running normalizer, so the `S × S` matrix exists only in registers
and shared memory, never in HBM.

The reference implementation materializes it explicitly — and, for numerical
stability, in fp32 even when the model runs in fp16 or bf16. So each score
element costs 2 bytes of scores plus 8 bytes of fp32 softmax intermediates, and
the traffic is `B·H·S²` elements written and re-read roughly three times per
layer. This is the term that decides everything downstream:

| S | analytic FLOPs | est. peak memory | arithmetic intensity, explicit | arithmetic intensity, fused |
|---|---|---|---|---|
| 512 | 180 GFLOP | 0.53 GiB | 64.6 FLOP/byte | 113.8 FLOP/byte |
| 1024 | 412 GFLOP | 1.42 GiB | 52.0 FLOP/byte | 133.2 FLOP/byte |
| 2048 | 1031 GFLOP | 4.70 GiB | 40.5 FLOP/byte | 168.6 FLOP/byte |
| 4096 | 2886 GFLOP | 17.27 GiB | 32.3 FLOP/byte | 237.4 FLOP/byte |

_(B=8, d=512, H=8, L=6, fp32. FLOPs and bytes from `analysis/roofline.py`;
memory from `src/memcheck.py`. Derivations in §6.)_

The two intensity columns move in **opposite directions**. As the sequence
grows, the explicit implementation reads and writes more bytes per FLOP and
slides *down* toward the bandwidth-bound region; a fused implementation does the
same arithmetic while moving less memory and climbs *away* from it. With this
card's fp32 ridge at 62.5 FLOP/byte (§6), the baseline crosses from
compute-bound into bandwidth-bound somewhere around S = 1024 — and that crossing
is where a fused kernel should start winning by a lot rather than a little. The
measured speedup curve in §4 either shows that shape or contradicts it, and both
outcomes are informative.

### 1.2 Three bottlenecks, three different fixes

Optimizing the wrong one wastes a day, so the first thing we measured was which
one we had.

| Bottleneck | Symptom | How we diagnose it | What actually helps |
|---|---|---|---|
| **Launch overhead** | GPU idle between kernels; runtime barely responds to shape | GPU busy % from a profiler trace (§3) | Fewer, larger kernels: fusion, CUDA graphs. A faster kernel changes almost nothing. |
| **Memory bandwidth** | Arithmetic intensity left of the ridge; time scales with bytes moved, not FLOPs | Roofline position (§6) | Stop moving the same bytes: fuse, avoid materializing intermediates, use a smaller dtype. |
| **Compute** | Near peak for the dtype; right of the ridge | Achieved TFLOP/s vs peak (§6) | Better math or better tensor-core utilization. Everything else is noise. |

---

## 2. Environment

| | |
|---|---|
| CPU | Intel Core i7-12650HX (12th gen) |
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU — compute capability sm_89 (Ada Lovelace), **low-TGP** |
| VRAM | 6.0 GB GDDR6 |
| OS | Windows 11 + WSL2, Ubuntu 24.04 |
| PyTorch | 2.6.0+cu124 |
| Triton | 3.2.0 |
| Install | `pip install torch --index-url https://download.pytorch.org/whl/cu124` |

Peak figures used for the roofline in §6. These are **measured on the card**,
not taken from the spec sheet — with a 4096×4096 matmul and a 512 MB
device-to-device copy:

| | measured | note |
|---|---|---|
| fp32, TF32 **on** | **11.0 TFLOP/s** | the benchmark defaults to `--allow-tf32`, so this is the fp32 number that applies |
| fp32, TF32 off | 5.7 TFLOP/s | for contrast; nearly halves the ridge point |
| fp16 tensor | 22.5 TFLOP/s | |
| bf16 tensor | 23.2 TFLOP/s | |
| memory bandwidth | **174.8 GB/s** | 91% of the 192 GB/s theoretical |

Measuring rather than quoting the spec sheet was not pedantry: this is a low-TGP
part and the achieved figures are roughly **half** the published ones. A
spec-sheet roofline would place every operating point at half its true height
against the roof, and would have told us we were leaving twice as much on the
table as we actually were.

Every ridge point in this report scales directly with these numbers.

A second machine — macOS on Apple Silicon, CPU only — runs the correctness
suite, the analysis and every figure. Nothing in this report's *measurements*
comes from it; it exists so that correctness is verified somewhere other than
the machine that is also being optimized.

### 2.1 Provenance

The organizers' benchmark file is byte-identical to the download:

```
SHA-256  1bd12523657f338c09b53f0bb9052d9d16f728a71bd22bc8298567e1a4d78c22
```

A test asserts this on every run. We import from that file rather than copying
out of it, so our accuracy and timing numbers come from the organizers' own
functions — `generate_random_case`, `compare_outputs`, `warmup_model`,
`benchmark_once` — with their seed scheme and their alternating round order.
Global state (`manual_seed`, matmul precision, the TF32 flags) is set exactly as
their `main()` sets it, because all three change the result.

Every row in `results/results.csv` carries the `git_sha` it was produced at, and
the harness refuses to run against a dirty working tree unless explicitly
overridden (in which case the row is tagged `dirty`).

---

## 3. Rung 0 — where the time actually goes

Before optimizing anything we profiled the baseline at three shapes and measured
what fraction of the wall-clock window the GPU spent with a kernel running.

### 3.1 A measurement we could not take, and what we used instead

The intended metric was GPU busy % — the fraction of the wall-clock window with
a kernel actually running. **It is not available on this platform.** Under WSL2,
CUPTI does not populate device-side kernel completion records: the Chrome traces
contain the complete CPU-side story (`cpu_op`, `cuda_runtime`, `cuda_driver`,
flow arrows) and *zero* kernel events. Verified on the traces in `logs/`, not
assumed.

This is worth dwelling on because of how the failure presents. A naive parser
finds no kernels, sums zero busy time, and reports **0.0% GPU busy** — which
reads as the most catastrophic result imaginable rather than as a missing
measurement. Our analysis returns `NaN` and the word "unmeasurable" instead, and
a test asserts it never returns 0.0 for a trace with no kernel track.

The fallback is *launch arithmetic*, which needs no device timing: count the
kernel launches on the CPU side, multiply by their measured cost, compare to
wall time. It is the sounder argument anyway, for a reason specific to this
benchmark: `cuda.Event` pairs are enqueued **on the stream**, so a CPU-side
stall falls inside the measurement window and busy % reads near 100% almost
regardless of what the GPU was doing.

### 3.2 Launch overhead, measured

Baseline, fp32 with TF32 on, 6 layers, no causal, no padding:

| shape | launches | launch share of wall time | mean launch | verdict |
|---|---|---|---|---|
| small (B=8, S=128) | 345 | **8.2%** | 15.5 µs | borderline |
| medium (B=8, S=512) | 345 | 2.9% | 11.6 µs | not launch-bound |
| large (B=4, S=2048) | 345 | 1.3% | 19.7 µs | not launch-bound |

![Launch overhead by shape](../results/figures/gpu_launch_overhead.png)

Two corrections to a first pass at these numbers, both of which moved the
answer:

1. **Driver-API launches were being missed.** cuBLAS submits through
   `cuLaunchKernel`, not `cudaLaunchKernel`, and those records are *not* nested
   inside the runtime-API ones — we checked. Counting only `cudaLaunchKernel`
   gives 67 launches per forward; including the driver path gives **115**, a 42%
   undercount.
2. **Per-launch cost is measured, not assumed.** The mean is 15.5 µs at the
   small shape, not the ~5 µs a back-of-envelope estimate would use.

Together those take the small shape's launch share from ~2.5% to **8.2%** — the
difference between "definitively not launch-bound" and "borderline".

**Reading it honestly:** launch share is an *upper bound*. Launches are
asynchronous, so CPU time inside a launch call does not prove the GPU was idle —
the CPU may simply be running ahead. Below 5% rules launch-boundedness out; 5–15%
leaves it open. So: the medium and large shapes are definitively not
launch-bound, and the small shape is undecided by this evidence.

**What that implied:** fusion before CUDA graphs. At the shapes where the
speedup matters most, there is no launch overhead worth eliminating, so
kernel-level work is where the return is. CUDA graphs stay on the list for the
small shape only, and only if it becomes the binding case.

---

## 4. The optimizations

One subsection per rung. Each states the hypothesis *before* the measurement,
the measured before/after, and the surprise — because the surprises are the part
worth reading.

### Rung 0.5 — a baseline that was wrong before we optimized anything

Not an optimization, but the most instructive measurement of the project.

The first profiling run reported 18.6 / 64.6 / 215.2 ms across the three shapes,
with **115** kernels per forward. Those numbers were wrong. The standalone
profiling script never set `matmul_precision="high"` or `allow_tf32=True`, both
of which the organizers' `main()` sets by default — so it was running fp32
matmuls at 5.7 TFLOP/s while `sweep.py` ran the same code at 11.0. We had two
contradictory baselines and, for a while, no idea which was real.

After matching the organizers' global state: **13.5 / 52.1 / 176.5 ms**, and the
kernel count dropped to 67 runtime-API launches per forward. 18–27% faster, 42%
fewer kernels.

The kernel-count drop is the interesting part. TF32 is a *math mode*, not a
storage format — the naive expectation is that the same kernels run faster.
Instead PyTorch selected entirely different, tensor-core kernels. A flag that
looks like a precision knob is really a kernel-selection knob.

Two things this justifies. First, the harness sets global state exactly as the
organizers' `main()` does, and that is not optional bookkeeping — it is worth
18–27% and would have contaminated every subsequent comparison. Second, a
baseline is a measurement like any other and can be wrong; ours was, and it was
caught by two tools disagreeing rather than by either one looking suspicious on
its own.

### Rung 1 — `scaled_dot_product_attention`

- **Hypothesis (recorded before measuring):** attention is ~4% of FLOPs at
  S=128 and ~40% at S=2048, so a fused attention kernel should barely move the
  small shape and help substantially at the large one.
- **What changed:** the explicit `QKᵀ → mask → fp32 softmax → PV` sequence
  replaced by `F.scaled_dot_product_attention`, which never materializes the
  `[B, H, S, S]` score matrix in HBM.
- **Measured:** **0.899× at S=128** (slower than baseline) and **1.693× at
  S=512**. `<FILL B: fill in S=256/1024/2048 from --matrix crossover.>`
- **Accuracy:** fp32 passes. `max_abs = 0.0075` at the default depth — 7.5× the
  `atol` budget, passing on the relative leg alone. See §7.2.
- **Surprise, and the best result in the project so far:** 1.693× is *above the
  Amdahl ceiling*.

  Attention is 14.3% of the forward pass's FLOPs at S=512, so making it
  infinitely fast can yield at most `1/(1 − 0.143) = 1.167×` — **if time were
  spent in proportion to arithmetic**. We measured 1.693×. Inverting Amdahl,
  that speedup requires attention to have been **41% of the runtime**, which is
  **2.9× its share of the arithmetic**.

  A region cannot consume triple its arithmetic share unless it is waiting on
  something other than arithmetic. This is direct evidence that the baseline's
  attention was **memory-bound** — the cost was writing and re-reading the score
  matrix, exactly as §1.1 predicted from the byte counts, and not the two GEMMs.
  The prediction was recorded before the measurement and the measurement
  overshot it in the direction the mechanism implies.

  The S=128 result is the same argument from the other side: the ceiling there
  is only 1.042×, so there is almost nothing to win, and SDPA's own dispatch
  overhead makes it a net loss at 0.899×. **A crossover exists between S=128 and
  S=512, and it is the empirical justification for the dispatch layer** — not a
  design flourish but a measured sign change. `--matrix crossover` samples
  S ∈ {128, 192, 256, 320, 384, 512, 768, 1024} to locate it.

- **Saturation:** speedup settles at 1.65–1.67× beyond 2 layers — the
  steady-state attention share, once per-call overheads are amortized.

### Rung 2 — `<FILL A: e.g. bf16>`

- **Hypothesis:** `<FILL A>`
- **What changed:** `<FILL A>`
- **Before → after:** `<FILL B>`
- **Accuracy:** `<FILL B>` — see §7 for why reduced precision costs error budget
  and how much margin remained.
- **Surprise:** `<FILL A>`

### Rung 3 — reduced precision: **abandoned, with a measurement**

This rung was planned as fp16/bf16 execution and is **dead**, not deprioritized.
Neither format can be shipped at the benchmark's default depth. The full
analysis is §7.1 and §7.2; the short version:

| dtype | 1 layer | 2+ layers | 6 layers (the default) |
|---|---|---|---|
| fp32 | passes | passes | passes (on the relative leg) |
| fp16 | **passes**, 2.020× | fails | fails |
| bf16 | fails | fails | fails |

The constraint is the tolerance against the format's granularity, not a defect
in the kernel. Recording it as a negative result rather than omitting the rung:
the measurement that closes a direction is worth as much as one that opens it,
and it is what redirected the remaining effort.

**Consequence for the rest of the project.** With reduced precision unavailable,
the ~2× throughput the hardware offers in fp16/bf16 is unreachable, and fusion
plus shape dispatch have to carry the result on their own. That raises the value
of a fused LayerNorm (Rung 6) considerably — it is now one of the few remaining
sources of gain rather than a nice-to-have.

### Rung 4 — `<FILL A: e.g. fused QKV projection>`

- **Hypothesis:** `<FILL A>`
- **What changed:** `<FILL A>` — note the parameter-naming constraint: the
  organizers' `copy_model_weights(strict=True)` must still succeed, so the
  fused view is built from `q_proj`/`k_proj`/`v_proj` rather than replacing them.
- **Before → after:** `<FILL B>`
- **Surprise:** `<FILL A>`

### Rung 5 — `<FILL A>`

`<FILL A: same structure.>`

### Rung 6 — `<FILL A: attempted, or dropped>`

`<FILL A: if dropped, say so here and move the analysis to §11. A negative
result with a reason is worth more than an omission.>`

### 4.1 Aggregate

`<FILL B: paste the per-strategy geometric-mean table from results/summary.md.>`

Geometric mean, not arithmetic: speedups are ratios. A strategy that is 2x on
one shape and 0.5x on another has achieved nothing on average, and only the
geometric mean says so — the arithmetic mean would call it 1.25x.

![Speedup against sequence length](../results/figures/speedup_vs_seq_len.png)
![Speedup against batch size](../results/figures/speedup_vs_batch.png)
![Speedup against model dimension](../results/figures/speedup_vs_dmodel.png)

---

## 5. Shape and device dispatch

The problem statement invites this explicitly: *"participants can choose
different implementations for different shapes by adding shape checks in the
implementation of layers."* We do, and the choice comes from the measurements
rather than from intuition.

`analysis/load.py` reduces `results.csv` to a winner per configuration and
writes `results/dispatch_table.json`; `src/dispatch.py` reads it at run time.
Selection order:

1. **No CUDA** → the CPU fallback strategy.
2. **Exact match** in this card's capability block (`sm_89`, `sm_80`, …).
3. **Nearest measured neighbour** sharing `(dtype, causal, padded)`, by log
   distance on sequence length, then batch. Distance is logarithmic because the
   axes are swept multiplicatively — S=1536 is nearer 2048 than 1024, and a
   linear metric picks the wrong neighbour every time. Neighbours never cross a
   causal/padded/dtype boundary: borrowing across one would recommend a kernel
   for a branch it was never measured on.
4. **The table's default.**
5. **`baseline`**, which is always registered and always correct.

Two gates filter every step, and they can only *remove* candidates:

- **Registration.** The table is generated on the GPU machine and may name a
  strategy a given checkout does not have. Selecting one is an error inside
  `forward()`, which is strictly worse than being slower.
- **Capability.** bf16 paths require `(8, 0)`; Triton and flash-style kernels
  require `(7, 5)`. A strategy may declare `MIN_CAPABILITY` itself, which is
  believed over the name-based heuristic.

### 5.1 Crossover

`<FILL B: paste the dispatch table from results/summary.md §Dispatch, including
the margin column.>`

The margin column is the honest part. A key won by 0.002x is a tie: the choice
there is inside run-to-run noise and should not be read as a result. `<FILL B:
how many keys are won by ≥ 0.05x, and how many are ties.>`

### 5.2 What we did and did not measure

**Performance is measured only on `<FILL A: sm_XX>`.** We have one GPU.

The `sm_75` and `sm_80` paths are **correctness-tested by forced dispatch** —
`select_strategy` accepts an explicit capability, so the selection logic for
those cards is exercised in CI on a machine with no GPU at all, and the
strategies those paths select are verified against the baseline by the CPU
correctness suite. What is *not* verified is that they are faster on that
hardware. We do not claim it.

The dispatch table is a generated artefact, but the submission does not depend
on it: `src/dispatch.py` falls back to a small hard-coded table if the file is
missing, so a fresh clone with no `results/` directory still runs.

---

## 6. Roofline

Analytic counts, per layer, from `analysis/roofline.py`:

```
FLOPs  = 8·B·S·d²        Q, K, V and output projections
       + 4·B·S²·d        QKᵀ and PV
       + 4·B·S·d·f       the two FFN GEMMs
```

Bytes are weights read once, plus activation traffic approximated as
`L·(14·B·S·d + 4·B·S·f)·e`, plus — for the explicit implementation only —
`L·3·B·H·S²·e` for the score matrix written and re-read. Cache reuse is not
modelled, so this *overestimates* bytes and therefore *underestimates*
arithmetic intensity: points sit slightly left of the truth, which errs toward
calling ourselves more bandwidth-bound rather than less.

Ridge points for this card, from the measured peaks in §2:

| precision | measured peak | ridge point |
|---|---|---|
| fp32, TF32 on | 11.0 TFLOP/s | **62.9 FLOP/byte** |
| fp32, TF32 off | 5.7 TFLOP/s | 32.6 FLOP/byte |
| fp16 | 22.5 TFLOP/s | **128.7 FLOP/byte** |
| bf16 | 23.2 TFLOP/s | **132.7 FLOP/byte** |

**The ridge point moves with precision, and it moves right.** Reduced precision
raises the compute roof and does nothing at all for the bandwidth roof, so the
crossover shifts from 63 to ~130 FLOP/byte. The consequence is counterintuitive
and worth stating plainly: **switching to reduced precision can make a workload
*more* memory-bound, not less.** A kernel that was comfortably compute-bound in
fp32 can cross to the other side of the ridge in fp16 — it does not simply move
up the chart, and the right next optimization changes with it.

This is why the roofline figure draws one ceiling per dtype rather than one for
the chart.

![Roofline](../results/figures/roofline.png)

`<FILL B: which configurations land left of the ridge, which land right, and how
close the best strategy gets to its roof.>`

The prediction from §1.1 is testable here: the baseline should walk *left* along
the x-axis as S grows, crossing the fp32 ridge near S=1024, while the fused path
walks right. The measured fp32 ridge (62.9 FLOP/byte) is within half a percent
of the placeholder the prediction was made against (62.5), so the prediction
stands as recorded. `<FILL B: does the measured data follow it?>`

---

## 7. Accuracy budget

![Accuracy budget](../results/figures/accuracy_budget.png)

Error compounds with depth and grows as precision shrinks, so the interesting
question is not "does it pass" but "with how much room".

- Target: `atol = 0.001`, `rtol = 0.01` (the torch script's defaults).
- The problem statement's own bar is looser: `0.002 / 0.02`.
- `<FILL B: worst max_abs_err and max_rel_err across all passing runs, and the
  margin to each threshold.>`

Two properties of the reference implementation drive the numbers:

1. **Softmax runs in fp32 and casts back**, even when the model is fp16 or
   bf16. An optimized path that performs the softmax in reduced precision will
   land just over budget — this is the single most likely cause of a failing
   accuracy check, and it is a correctness bug, not a tolerance problem.
2. **Masking is applied in four places** — invalid key positions inside
   attention, and invalid query positions after attention, after each block and
   after the final norm. Any of them missed shows up only in the padded branch.

### 7.1 bf16 cannot pass against this reference, and that is a property of the benchmark

The most interesting accuracy result is a negative one, and it is not about our
implementation.

The SDPA strategy passes at fp32 and fp16 and **fails at bf16**. The failure is
not a bug. Its worst element is exactly **2 ULP of bf16**: an absolute error of
0.03125 at magnitude 2.17, where 1 ULP at that magnitude is 0.015625. That is
**1.44% relative**, against a 1% tolerance. Mean absolute error across the
tensor is 0.0009 — about 0.06 ULP — so roughly 95% of elements match the
reference *exactly*; only a small fraction drift by two representable steps. But
the oracle is per element, so a small fraction is enough.

The same code path in fp16 has 10 mantissa bits instead of 7. Two ULP there is
0.0039, or **0.18% relative** — comfortably inside tolerance. Identical
algorithm, identical kernel, opposite verdict, decided entirely by the width of
the mantissa.

**Where the two steps come from.** The reference rounds the softmax
probabilities to bf16 *before* the PV matmul; SDPA keeps them in fp32
internally. Our path is therefore **more** accurate than the reference — it just
does not reproduce the reference's intermediate rounding.

**Why this is a statement about the benchmark.** At magnitude 2.17, `rtol =
0.01` is 1.39 ULP of bf16 — *tighter than the format's own granularity*. Any
implementation that reorders the computation, or declines to round at the same
intermediate points, will land two representable steps away somewhere in a
tensor of two million elements. Two steps is already over budget. So at
`rtol = 0.01` the only bf16 implementation that can pass is one that reproduces
the reference's operation order bit-for-bit — which is precisely what an
optimized kernel must not do.

At the problem statement's own looser bar (`rel < 0.02`) this passes with room
to spare. The gap between the two published tolerances is the difference between
bf16 being shippable and not.

**What we did about it.** The strategy declares
`SUPPORTED_DTYPES = (torch.float32, torch.float16)`. That declaration is
honoured in two places at once: the test matrix skips bf16 with a visible
reason, and `src/dispatch.py` will never select the strategy for a bf16 tensor —
falling through to the next allowed candidate and ultimately to `baseline`. An
unsupported dtype is therefore both untested *and unreachable*, which is the
only combination that is safe. Declaring a dtype unsupported is not a way to
hide a failure.

**The cost.** We ship fp16 rather than bf16 and give up about 3% of peak
throughput (22.5 vs 23.2 TFLOP/s measured). This reverses an earlier judgement
of ours that bf16 was the better target — it is faster on this card and it is
the numerically more robust format in general, but it cannot clear this
benchmark's bar, and correctness is not negotiable against 3%.

### 7.2 The precision ceiling: error compounds with depth

bf16 (§7.1) fails at every depth. fp16 is the more interesting case, because it
fails *conditionally*:

| depth | fp16 max_abs | verdict |
|---|---|---|
| 1 layer | — | **passes**, 2.020× |
| 2 layers | 0.0059 | fails |
| 6 layers (default) | 0.0078 | fails — 4 ULP of fp16 |

Error accumulates through the residual stream: each layer's output is the next
layer's input, so a rounding difference introduced in layer 1 is carried and
re-perturbed five more times.

**How fast it accumulates is itself a measurement.** In fp32 the worst-case
error grows from 0.00024 at 1 layer to 0.00084 at 6 — a factor of 3.5. Two
reference behaviours bracket that:

- if per-layer errors were independent and random, they would add in quadrature
  and grow as `√L` — a factor of 2.45 over six layers;
- if they were perfectly correlated, they would add linearly as `L` — a factor
  of 6.

The observed 3.5 is `L^0.70`, sitting between the two and closer to the random
walk. The errors are therefore **partially correlated** — not the independent
noise a `√L` model assumes, but nowhere near worst-case accumulation. This
matters for extrapolation: a 12-layer model would be expected around `0.00084 ×
2^0.70 ≈ 0.0014`, not the 0.0012 a `√L` rule would predict.

**The consequence.** fp32 at 6 layers reaches `max_abs = 0.0075`, which is 7.5×
the `atol` budget of 0.001; it passes only because the *relative* leg of the OR
rule carries it. There is roughly one order of magnitude of headroom left before
fp32 itself would be at risk, and every reduced-precision format has already
spent it. **At the benchmark's default depth, fp32 is the only shippable
dtype** — which is the constraint the rest of the optimization had to work
inside.

**A note on how this interacts with dispatch.** Numerical admissibility turns
out to depend on *depth*, which the dispatcher deliberately does not key on
(§5 — depth changes how long a forward takes but not which kernel suits a
shape). That reasoning holds for performance and fails for correctness. Rather
than adding depth to the dispatch key, strategies declare `SUPPORTED_DTYPES`
conservatively — for the deepest configuration we ship — so a dtype that is
admissible only at 1 layer is simply not offered. A safe answer at every depth
is preferable to a fast one that depends on a config axis the dispatcher cannot
see.

The correctness suite crosses every registered strategy against three shapes
(including `S=33`, `d=96`, `heads=3` — non-power-of-two on both axes), padding
`{0, 0.3}`, and causal `{off, on}`, plus edge cases at `S=1` and
`padding_ratio=0.97`. Reduced-precision dtypes are asserted only where there is
a GPU: CPU fp16/bf16 kernels round differently from the CUDA ones being shipped,
so a failure there would say nothing about the real path.

---

## 8. Thermal methodology, and why the number is trustworthy

The measurements come from a laptop GPU in a chassis that cannot hold its boost
clock. Left alone, a sweep produces a beautiful downward performance trend that
is really just the card getting hot. Four defences:

1. **Alternating measurement order.** The organizers' own `benchmark_models`
   interleaves baseline and optimized rounds, so a drifting clock hits both
   sides of the ratio roughly equally and largely cancels. We reuse their loop
   rather than writing our own.
2. **Median, not mean.** A single scheduling hiccup or a background process
   moves a mean and does not move a median.
3. **Cooldowns between configurations**, and a temperature poll before retrying.
4. **A mechanical discard rule.** `bench/thermal.py` logs SM clock and
   temperature at 1 Hz for the duration of each run. If the mean clock over the
   *timed window* falls below 85% of the opening clock (the mean of the first
   three samples), the card throttled mid-run: the row is cooled down and re-run
   once, and if it throttles again the row is kept but tagged
   `DISCARD:thermal (retried)` and excluded from every summary statistic.

Discarded rows are kept rather than deleted, and counted:
`<FILL B: number of DISCARD:thermal rows out of the total, from
results/summary.md.>`

![Clock and temperature during one run](../results/figures/thermal_clocks_synthetic.png)

_(Replace with a real clock trace once one exists —
`results/figures/thermal_clocks_<sha>_<time>_<strategy>_<B>x<S>x<d>.png`.)_

The clock and temperature are drawn as two stacked panels sharing a time axis
rather than on twin y-axes. Two scales on one plot invite the reader to read
meaning into where the lines cross, and that crossing is an artefact of the two
arbitrary scalings, not a fact about the GPU.

---

## 9. The VRAM ceiling

![VRAM ceiling](../results/figures/vram_ceiling.png)

The baseline materializes `[B, H, S, S]` in fp16/bf16 *plus* fp32 softmax
intermediates — about `2e + 8` bytes per score element. On a `<FILL A: GB>` card
that becomes the binding constraint long before compute does. Estimated peak for
the baseline at B=8, d=512, H=8, L=6, fp32:

| S | estimated peak |
|---|---|
| 512 | 0.53 GiB |
| 1024 | 1.42 GiB |
| 2048 | 4.70 GiB |
| 4096 | **17.27 GiB** |

`src/memcheck.py` computes this *before* running a configuration, so an
impossible one is recorded as `SKIPPED: <reason with numbers>` and the sweep
continues rather than dying half-way. The estimate is ported unchanged from the
organizers' own TensorFlow benchmark, so both benchmarks agree about which
configurations are impossible.

Where the baseline OOMs and our implementation completes, the harness records
`baseline OOM; optimized completed` and writes the optimized timing with an
empty `baseline_median_ms`. There is no speedup ratio for those rows — the
result is categorical, and the speedup column has no way to express it.

`<FILL B: at which shape does the baseline stop fitting, and how far past that
does the optimized path go?>`

---

## 10. The `--compile-baseline` honesty check

![Speedup survival](../results/figures/compile_baseline_survival.png)

Some of any speedup over an *eager* baseline is just `torch.compile` doing what
it does to any PyTorch model. Reporting only that number is the easiest way to
overstate a result, so we also measured every strategy against a **compiled**
baseline and report both.

`<FILL B: per strategy, speedup vs eager, speedup vs compiled, and the
percentage that survives.>`

The compiled-baseline runs are stored as a separate experiment, not as a newer
run of the same one: the denominator of the ratio is a different model, so
mixing them into one average would silently understate every speedup.

---

## 11. Limitations

Stated plainly, because a report that claims no limitations is not credible.

1. **One GPU, one architecture.** Every performance number is from
   `<FILL A: sm_XX>`. The `sm_75`/`sm_80` paths are correctness-tested by forced
   dispatch and never performance-measured. We do not claim they are faster
   there.
2. **A laptop GPU in a thermally constrained chassis.** Section 8 describes what
   we did about it. Absolute latencies would differ on a desktop card; the
   ratios should be more portable than the absolute numbers, but we have not
   verified that.
3. **Inference only, no backward pass.** The problem asks for forward; a fused
   attention kernel that is correct forward is not automatically correct or fast
   backward.
4. **The roofline byte model ignores cache reuse**, so it underestimates
   arithmetic intensity. §6 states the direction of the error.
5. **Synthetic inputs.** The organizers' generator produces random normal
   tensors. Real activations have different distributions, and a reduced-
   precision path's error budget could behave differently on real data.
6. **The dispatch table is only as dense as the sweep.** Unmeasured shapes fall
   through to a nearest neighbour, which is a guess — a well-founded one, but
   still a guess.
7. **No device-side kernel attribution.** CUPTI does not populate kernel
   completion records under WSL2, so GPU busy % and per-kernel time breakdowns
   are unavailable on this setup (§3.1). `cuda.Event` timing is unaffected, so
   every latency and speedup number is sound; what is lost is the ability to say
   *which kernel* owned the time. The launch-count argument substitutes for the
   launch-bound question, but a per-kernel breakdown would have been better
   evidence and we do not have it.
8. **No reduced precision at all.** bf16 fails at every depth (§7.1) and fp16
   fails at two layers and beyond (§7.2), so at the benchmark's default of 6
   layers fp32 is the only shippable dtype. The card offers roughly 2× the
   throughput in fp16/bf16 and we cannot use any of it. This is a constraint
   imposed by the tolerance against the formats' granularity, not a defect in
   the kernels — and it is the single largest piece of performance left on the
   table.
9. `<FILL A: anything else you hit and worked around.>`

### 11.1 Negative results

`<FILL A: what was tried and abandoned, and the analysis of why. Candidates from
the plan: a Triton LayerNorm, CUDA graphs, max-autotune. If a rung was dropped
because the measurement said it was not worth it, that measurement belongs
here — it is evidence, not a gap.>`

### 11.2 What we would do with more time

`<FILL A/B: ranked. Suggested starting points: CUDA graphs if any shape is still
launch-bound after fusion; a real flash-attention kernel rather than SDPA's
dispatch; a wider sweep to densify the dispatch table; sm_80 hardware to make
the multi-architecture claim measurable rather than structural.>`

---

## 12. AI tools used

Full per-session log, written contemporaneously: [AI_USAGE.md](AI_USAGE.md).

Summary: `<FILL A/B: which tools, for which parts, and — more usefully — the
specific things they got wrong that had to be corrected. The log already
contains these; lift the substantive ones.>`

The log records corrections as prominently as successes, including several
cases where generated code was plausible and wrong: a memoisation that silently
discarded capability requirements, a figure that plotted peak memory decreasing
with sequence length, a "wrong" test strategy that was inside tolerance and
correctly passed, and a trace parser that returned 0% GPU-busy on empty input
instead of raising.

---

## Appendix — reproducing every number

```bash
pytest -q                                                 # correctness, CPU, no GPU
python bench/sweep.py --strategy baseline --matrix quick   # control: must be ~1.00x
python bench/sweep.py --strategy <name> --matrix default   # the full sweep
python bench/sweep.py --strategy <name> --matrix accuracy  # layers x dtype, for §7
python bench/run_official.py --batch-size 8 --seq-len 1024  # the organizers' script
python -m analysis.make_all                                # every figure + summary.md
```

`results/results.csv` is append-only and every row carries its `git_sha`.
`results/summary.md` is generated, never hand-edited.
