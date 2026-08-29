"""Registry shim: makes `src/optimized.py`'s entry point visible to the sweep.

`src.strategies._autodiscover` only walks this package, so the entry point --
which lives at `src/optimized.py` because that is where `docs/INTERFACE.md`
says the organizers' script looks for it -- would otherwise never register and
`bench/sweep.py --strategy optimized` would fail with "unknown strategy".

Registering here rather than decorating the class in `src/optimized.py` keeps
that module importable on its own, without pulling in the whole registry as a
side effect of `import src.optimized`.
"""

from __future__ import annotations

from src.optimized import UserOptimizedTransformer
from src.strategies import register

register("optimized")(UserOptimizedTransformer)
