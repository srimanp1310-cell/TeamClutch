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
