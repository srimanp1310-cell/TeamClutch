#!/usr/bin/env python3
"""Run the organizers' script, unmodified, with our implementation injected.

This is the file the demo video runs and the one the README's "reproduce"
section points at. The point is that the *scoring* code is theirs, untouched:
we do not re-implement their accuracy check or their timing loop and then claim
a speedup — we hand them our class and let their `main()` grade it.

The injection is a single attribute assignment. `main()` looks up
`UserOptimizedTransformer` as a module global, so rebinding that name on the
module is enough; the file on disk is never opened for writing, and its
SHA-256 still matches the one pinned in docs/INTERFACE.md.

Every command-line flag belongs to the organizers' parser, so anything their
script accepts works here unchanged::

    python bench/run_official.py --batch-size 8 --seq-len 1024 --dtype bfloat16
    python bench/run_official.py --device cpu --seq-len 128 --causal --padding-ratio 0.3

Until Person A's `src/optimized.py` exists this falls back to the "baseline"
control strategy, so the file is runnable from day one — it just measures
~1.00x, which is the correct answer for an implementation that is a copy of the
reference.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bench.torch_transformer_benchmark as official  # noqa: E402


def resolve_entry_point() -> tuple[type, str]:
    """Person A's entry point if it exists, else the baseline control.

    Returns the class and a human-readable description of where it came from,
    so a run can never leave you guessing which implementation was measured.
    """
    try:
        from src.optimized import UserOptimizedTransformer  # noqa: WPS433

        return UserOptimizedTransformer, "src.optimized.UserOptimizedTransformer"
    except ImportError:
        from src.strategies import STRATEGIES  # noqa: WPS433

        return (
            STRATEGIES["baseline"],
            'src.strategies.STRATEGIES["baseline"] '
            "(fallback: src/optimized.py does not exist yet — expect ~1.00x)",
        )


def main() -> int:
    entry_point, origin = resolve_entry_point()
    official.UserOptimizedTransformer = entry_point
    print(f"[run_official] injected {origin}")
    return official.main()


if __name__ == "__main__":
    raise SystemExit(main())
