# Demo video — 3 minute shot list

**Format:** screen recording with voiceover. Terminal + the figure PNGs.
**Upload:** YouTube, **public** visibility, linked in the Devpost description.

**Two hard rules.** No third-party logos, trademarks or copyrighted content
beyond what is unavoidable on screen (a terminal showing `nvidia-smi` output is
fine; a browser tab full of branding is not). And **run every command once
before recording** — a live failure costs a re-shoot, and `torch.compile` takes
tens of seconds on a cold cache.

Prepare beforehand: a clean terminal at a readable font size, the repo at a
committed SHA, `results/figures/` already generated, and the machine cool.

**Have the results page open in a browser tab before you start recording.**
Regenerate it first so it reflects the committed state:

```bash
git pull && python -m analysis.make_all
explorer.exe "$(wslpath -w results/report.html)"    # WSL2 -> Windows browser
```

It is a static file — no server, nothing to crash mid-take, works offline. It
does not auto-update, so if you re-run a sweep between takes, regenerate and
reload the tab. The page is laid out to about 1080px wide, which reads cleanly
at 1080p; zoom the browser to ~110% if you are recording at 720p.

Alternate between the terminal and that tab rather than opening individual PNGs
— the page carries the sentence that makes each figure matter, which is what you
are narrating.

---

## 0:00–0:20 — The problem, stated once

**On screen:** the organizers' `UserOptimizedTransformer.forward()` in the
editor, scrolled to the `# your codes here` block.

> "We're given a fixed Transformer and exactly one method we're allowed to
> replace. The catch is the grading rule: every single output element has to
> stay within one thousandth absolute or one percent relative of the reference.
> So this is not 'make it fast' — it's 'make it fast while proving it still
> computes the same thing.'"

---

## 0:20–0:45 — Why attention is quadratic

**On screen:** the arithmetic-intensity table from the report (§1.1), or the
roofline figure.

> "Attention is quadratic for one reason: the softmax sits between Q-K-transpose
> and V, so you can't re-associate the matrix product. The S-by-S score matrix
> has to exist. The reference materializes it explicitly — and in fp32 for
> stability — so as the sequence gets longer it moves more bytes per FLOP, not
> fewer. Its arithmetic intensity *falls* from 65 to 32 FLOP per byte, straight
> through this card's ridge point at 62. That's the whole optimization plan in
> one number."

---

## 0:45–1:15 — Rung 0: measure before optimizing

**On screen:** `results/figures/gpu_busy_vs_shape.png`, then
`results/figures/trace_timeline_small.png`.

**Terminal:**
```bash
python -m analysis.make_all
```

> "Before touching a kernel we profiled. White space in this timeline is a GPU
> doing nothing. At small shapes we were `<FILL B: N>`% busy — launch-bound, so
> fusing kernels was worth more than making any one of them faster. At large
> shapes we were `<FILL B: N>`% busy, which is a completely different problem.
> Guessing wrong here costs a day."

---

## 1:15–2:00 — The result, from the organizers' own script

**Terminal — this is the money shot, run it live:**
```bash
python bench/run_official.py --batch-size 8 --seq-len 1024 --dtype bfloat16
```

**Let the output scroll and land on:**
```
summary: PASS | max_abs=... | max_rel=...
speedup  : ...x based on median latency
```

> "That's the organizers' script, unmodified — we inject our class and let their
> code grade it. Their file's SHA-256 is pinned in the repo and a test fails if
> it ever changes. `<FILL B: speedup>`x at this shape, and the accuracy check
> passes with `<FILL B: margin>` of headroom against the tolerance."

Then the aggregate:

**On screen:** `results/figures/speedup_vs_seq_len.png`

> "Across `<FILL B: N>` configurations, geometric mean `<FILL B>`x. Geometric,
> because these are ratios — a strategy that's 2x on one shape and half-speed on
> another has achieved nothing, and only the geometric mean says so."

---

## 2:00–2:30 — The three things that make it defensible

Roughly ten seconds each. One sentence, one visual.

**Dispatch** — on screen: the dispatch table in `results/summary.md`.

> "The problem statement says test shapes will vary and invites choosing
> different implementations per shape. Ours does, and the choice comes from the
> measurements — with two gates that can only remove options: the strategy has
> to be registered, and the card has to support it. A Turing card never gets
> handed a bf16 kernel."

**Thermal** — on screen: the clock/temperature trace figure.

> "This is a laptop GPU. It throttles. So we log the clock during every run, and
> any run whose clock drops below 85% of its opening is retried once and then
> thrown out. `<FILL B: N>` runs were discarded that way — they're still in the
> results file, just excluded from the statistics."

**Honesty check** — on screen: `results/figures/compile_baseline_survival.png`.

> "And we report the number that makes us look worse: how much speedup survives
> when the *baseline* is compiled too. `<FILL B: N>`% does. Some of any gain
> over an eager baseline is just the compiler."

---

## 2:30–2:50 — Correctness, on a machine with no GPU

**Terminal:**
```bash
pytest -q
```

> "The correctness suite runs on a second machine with no GPU at all — every
> implementation against non-power-of-two shapes, both mask branches, and the
> edge cases. `<FILL B: N>` tests. The control strategy is a copy of the
> reference, and it has to measure exactly zero error and exactly 1.00x; if it
> doesn't, the measuring rig is lying and nothing else we report means anything."

---

## 2:50–3:00 — Limitations, said out loud

> "One GPU, one architecture. The other capability paths are correctness-tested
> by forced dispatch but never performance-measured, and we don't claim they're
> faster there. Forward pass only. Full write-up and the per-session log of what
> the AI tools got wrong are both in the repo."

**End card:** repository URL. Nothing else.

---

## Shot checklist

- [ ] Every command run once before recording (`torch.compile` cold cache)
- [ ] `results/figures/` regenerated at the recorded SHA
- [ ] Terminal font large enough to read at 720p
- [ ] GPU cool before the live `run_official.py` take
- [ ] No third-party logos or branding visible
- [ ] All `<FILL B: …>` numbers in this script replaced with real ones
- [ ] Uploaded to YouTube as **public**, link pasted into the Devpost description
