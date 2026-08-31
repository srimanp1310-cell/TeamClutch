# Demo video — 3 minute shot list

**Format:** screen recording with voiceover. Terminal + the results page.
**Upload:** YouTube, **public** visibility, linked in the Devpost description.

**Two hard rules.** No third-party logos, trademarks or copyrighted content
beyond what is unavoidable on screen (a terminal showing `nvidia-smi` output is
fine; a browser tab full of branding is not). And **run every command once
before recording** — a live failure costs a re-shoot.

---

## Shot list — run in this order, straight through

Everything below is verified working from a clean clone. Set up once, then
record without stopping.

**Before you hit record:**

```bash
git status --porcelain          # must print nothing — sweep.py refuses on a dirty tree
python -m analysis.make_all     # regenerate figures + results/report.html
git checkout -- results/report.html   # make_all/pytest dirty this; reset before any sweep
explorer.exe "$(wslpath -w results/report.html)"   # WSL2 -> Windows browser
```

Have exactly **two windows**: a terminal, and the browser on `results/report.html`.
Alternate between them. Don't open individual PNGs — the page carries the
sentence that makes each figure matter, which is what you're narrating.

| # | Beat | Window | What to have ready |
|---|---|---|---|
| 1 | 0:00–0:22 | editor | `bench/torch_transformer_benchmark.py`, scrolled to `# your codes here` |
| 2 | 0:22–0:55 | browser | report page, **roofline** panel |
| 3 | 0:22–0:55 | terminal | control-gate rows (command below) |
| 4 | 0:55–1:43 | terminal | **live** `run_official.py` — the money shot |
| 5 | 0:55–1:43 | browser | **speedup vs sequence length** panel |
| 6 | 1:43–2:16 | browser | **dispatch** panel |
| 7 | 2:16–3:00 | browser | **precision ceiling** panel |
| 8 | end card | — | repository URL, nothing else |

**The two commands you run live:**

```bash
# Beat 3 — the control gate (fast, ~40 s)
python bench/sweep.py --strategy baseline --matrix quick

# Beat 4 — the money shot (~90 s, let it scroll)
python bench/run_official.py --batch-size 8 --seq-len 1024
```

> **Do not add `--dtype bfloat16` to the money shot.** bf16 fails the accuracy
> gate at this depth by design (see beat 7) — it would record a live FAIL. fp32
> is the default and the only shipped dtype.
>
> **Do not run `pytest -q`.** `pyproject.toml` already sets `-q`, so a second one
> suppresses the summary line and you get dots with no verdict. Use plain
> `pytest` if you show it: **286 passed, 5 skipped**.
>
> **Two figures the script used to call for no longer exist**:
> `trace_timeline_small.png` and `gpu_busy_vs_shape.png`. WSL2's CUPTI populates
> no device-side kernel records, so the GPU-busy metric is reported as
> unmeasurable rather than guessed. Don't show the stale committed PNG. The
> honest version of that story is in beat 2.

---

## 0:00–0:22 — The task

**On screen:** the organizers' `UserOptimizedTransformer.forward()`, scrolled to
the `# your codes here` block.

> "We're given a fixed Transformer and one method we're allowed to replace.
> It's readable, and it's slow.
>
> The catch is the grading rule. Every output element has to stay within one
> thousandth absolute, or one percent relative, of the reference. So it's not
> 'make it fast.' It's 'make it fast while proving it still computes the same
> thing.'"

---

## 0:22–0:55 — Measure first

**On screen:** the roofline panel. Then cut to the terminal for the control gate.

> "Before optimizing anything, we measured the card we actually have.
>
> This is a low-TGP laptop 4050 — about half its datasheet numbers. Eleven
> teraflops, not twenty-two. That puts the ridge point at sixty-three flops per
> byte. Below that you're memory-bound, and the fix is fewer trips to memory,
> not more arithmetic.
>
> Then the control gate."

**Terminal:**

```bash
python bench/sweep.py --strategy baseline --matrix quick
```

> "This runs the reference against itself. Nought-nine-eight-six, and
> one-oh-oh-one. Error exactly zero — not small, zero.
>
> If the baseline doesn't measure as itself, the rig is lying and nothing else
> we report means anything."

---

## 0:55–1:43 — The result, and why one kernel isn't enough

**Terminal — run this live:**

```bash
python bench/run_official.py --batch-size 8 --seq-len 1024
```

**Let it scroll and land on the last two lines.**

> "This is the organizers' own script. We inject our class and let their code
> grade it. Their file's SHA-256 is pinned, and a test fails if it ever changes.
>
> One-point-eight-six times faster. Accuracy: pass — worst element about seven
> ten-thousandths, against a budget of one thousandth.
>
> Now the part that surprised us."

**On screen:** speedup vs sequence length panel.

> "Fused attention is not uniformly a win. At d-model two-fifty-six it's
> two-point-four-nine. At sequence length one-oh-two-four, two-point-oh-eight.
>
> But at one-twenty-eight it's nought-point-nine-one. Slower than the code we
> replaced.
>
> There's a clean reason. At a hundred and twenty-eight tokens, attention is
> four percent of the arithmetic. At two thousand, forty. You can't win much by
> optimizing four percent, and the fused kernel's setup cost eats what's left.
>
> So there's no single best kernel. That's the argument for dispatch."

---

## 1:43–2:16 — The dispatcher, and an honest average

**On screen:** the dispatch panel.

> "So we pick per shape, and the choice comes from the measurements.
>
> Across fifteen configurations the router averages one-point-three-two,
> geometric mean. Zero accuracy failures. Worst case nought-nine-eight-four —
> nothing we ship is meaningfully slower than the baseline.
>
> Now, fused attention alone averages one-point-five-eight. That looks better.
> It isn't. That average only counts the configurations where fused attention is
> *correct* — drop the ones it gets wrong and of course the survivors look good.
>
> One-point-three-two includes the cases we had to route around. That's the
> number we'll stand behind."

---

## 2:16–3:00 — Two findings, and the limits

**On screen:** the precision ceiling panel.

> "Two findings.
>
> The fastest kernel here is structurally unreachable. Flash attention needs
> fp16 or bf16, and in bf16 our worst element is out by exactly two units in the
> last place — one-point-four-four percent, against a one percent tolerance.
> Exactly two. That's the format's granularity, not our bug: the reference rounds
> inside the softmax and a fused kernel doesn't, so we're *more* accurate and
> fail anyway.
>
> Second — we caught ourselves with a false positive. A fix that looked like it
> worked was numerically inert. Patching it out gave bit-identical output. It had
> passed on seed luck: twenty-one of forty seeds. It routes to baseline now.
>
> One GPU, forward pass only. Full write-up in the repo."

**End card:** repository URL. Nothing else.

---

## Shot checklist

- [ ] `git status --porcelain` prints nothing before recording
- [ ] `python -m analysis.make_all` run, then `git checkout -- results/report.html`
- [ ] Both live commands run once already (cold caches warmed)
- [ ] `run_official.py` has **no** `--dtype bfloat16` flag
- [ ] Browser on `results/report.html`, zoomed ~110% if recording at 720p
- [ ] Terminal font large enough to read at 720p
- [ ] GPU cool before the live `run_official.py` take
- [ ] No third-party logos or branding visible
- [ ] Uploaded to YouTube as **public**, link pasted into the Devpost description

---

## Number provenance

Every figure spoken above, and where it comes from. Re-derive before a re-shoot
if the log has grown.

| spoken | value | source |
|---|---|---|
| tolerance | `atol=0.001`, `rtol=0.01` | `docs/INTERFACE.md` |
| peak fp32 (TF32 on) | 11.0 TFLOP/s measured | `docs/TECH_REPORT.md` §2 |
| ridge point | 62.9 FLOP/byte | `docs/TECH_REPORT.md` §2, §6 |
| control gate | 0.986x, 1.001x, `max_abs_err` 0.0 | `results/results.csv`, baseline rows |
| official script | 1.861x, PASS, max_abs 0.00068 | `run_official.py`, clean clone, B=8 S=1024 fp32 |
| sdpa at d=256 | 2.487x | `results.csv` via `latest_per_config` |
| sdpa at S=1024 | 2.077x | same |
| sdpa at S=128 | 0.910x | same |
| attention FLOP share | 4.0% at S=128, 40.0% at S=2048 | `analysis/roofline.py::attention_flop_share` |
| router geomean | 1.322x, n=15, min 0.984x | `latest_per_config`, strategy `optimized` |
| sdpa geomean | 1.584x, n=14 | same, strategy `sdpa` |
| bf16 error | exactly 2 ULP = 1.44% vs 1% rtol | `docs/TECH_REPORT.md` §7.1 |
| causal false positive | 21/40 seeds (52%) | `docs/APPROVALS_NEEDED.md` R2 |
| test suite | 286 passed, 5 skipped | `pytest` |

**Not spoken, deliberately.** Thermal discards: **zero** runs were discarded —
the four thermal defences are real and logged, but no run tripped them, so
claiming otherwise on camera would be inventing a result. Compile-baseline
survival: `results.csv` contains **no** `compile_baseline` rows yet, so there is
no number to quote; the figure exists but nothing is behind it.
