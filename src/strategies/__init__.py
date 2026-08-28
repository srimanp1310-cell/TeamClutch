"""Strategy registry — the single contract point between Person A and Person B.

A *strategy* is one optimized implementation of the Transformer forward pass.
Every strategy:

  * subclasses `BaselineTransformer` (from the organizers' file, via `src.baseline`);
  * overrides only
        forward(self, x: torch.Tensor,
                valid_token_mask: Optional[torch.Tensor] = None) -> torch.Tensor
    with *no* extra required arguments;
  * keeps the baseline's parameter names untouched, so that the organizers'
    `copy_model_weights(baseline, optimized, strict=True)` succeeds. If a
    strategy fuses parameters (e.g. one `qkv_proj` in place of `q/k/v_proj`), it
    must keep the original `nn.Parameter`s registered under the original names
    and build the fused view in `__init__` / lazily — never rename them;
  * sets `REQUIRES_CUDA = True` if it cannot run on CPU (Triton, CUDA kernels).
    `tests/test_strategies.py` then SKIPs it on the Mac instead of failing.
  * sets `SUPPORTED_DTYPES` if it is only numerically correct in some dtypes,
    e.g. `SUPPORTED_DTYPES = (torch.float32, torch.float16)`. A strategy may be
    perfectly implemented and still fail the tolerance in a given precision --
    see `docs/INTERFACE.md` on why bf16 cannot match this reference at
    `rtol = 0.01`. Declaring it is not an excuse: it removes the dtype from the
    test matrix *and* stops the dispatcher ever selecting the strategy for it,
    so an unsupported dtype can never reach production and be silently wrong.
  * sets `MIN_CAPABILITY` if it needs particular hardware, e.g. `(8, 0)`.

Person A registers a strategy like this::

    # src/strategies/sdpa.py
    from src.baseline import BaselineTransformer
    from src.strategies import register

    @register("sdpa")
    class SdpaTransformer(BaselineTransformer):
        def forward(self, x, valid_token_mask=None):
            ...

Submodules of this package are imported automatically at import time, so a new
file in `src/strategies/` is picked up with no edit here. A submodule whose
imports are unavailable on this machine (e.g. `triton` on macOS) is recorded in
`UNAVAILABLE` rather than crashing the whole registry.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Dict, Type

import torch
import torch.nn as nn

from src.baseline import BaselineTransformer

__all__ = [
    "STRATEGIES",
    "UNAVAILABLE",
    "ALL_DTYPES",
    "BaselineCopy",
    "register",
    "get_strategy",
    "available_strategies",
    "requires_cuda",
    "supported_dtypes",
    "supports_dtype",
]

#: The three dtypes the organizers' script accepts. A strategy that does not
#: declare `SUPPORTED_DTYPES` is assumed to handle all of them.
ALL_DTYPES: frozenset = frozenset({"float32", "float16", "bfloat16"})


class BaselineCopy(BaselineTransformer):
    """Control strategy: byte-for-byte the baseline behaviour.

    This exists to validate the *harness*, not the model. Any run of
    `--strategy baseline` must report `max_abs_err == 0.0` and a speedup of
    ~1.00x. If it does not, the measurement rig is lying and every other number
    in results.csv is suspect. This is the Day-1 hard gate.
    """

    REQUIRES_CUDA = False


#: name -> class. Populated by `register`; always contains "baseline".
STRATEGIES: Dict[str, Type[nn.Module]] = {"baseline": BaselineCopy}

#: submodule name -> import error, for strategies that cannot load here
#: (e.g. Triton kernels on a Mac). Tests use this to print a real SKIP reason.
UNAVAILABLE: Dict[str, str] = {}


def register(name: str):
    """Class decorator: add a strategy to the registry under `name`."""

    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        if name in STRATEGIES and STRATEGIES[name] is not cls:
            raise ValueError(
                f"strategy {name!r} is already registered to "
                f"{STRATEGIES[name].__module__}.{STRATEGIES[name].__qualname__}; "
                "pick a different name"
            )
        if not issubclass(cls, BaselineTransformer):
            raise TypeError(
                f"strategy {name!r} ({cls.__qualname__}) must subclass "
                "BaselineTransformer so that copy_model_weights(strict=True) works"
            )
        STRATEGIES[name] = cls
        return cls

    return decorator


def get_strategy(name: str) -> Type[nn.Module]:
    """Look up a strategy, with a useful error listing what *is* registered."""
    try:
        return STRATEGIES[name]
    except KeyError:
        known = ", ".join(sorted(STRATEGIES))
        extra = ""
        if UNAVAILABLE:
            broken = "; ".join(f"{m}: {e}" for m, e in sorted(UNAVAILABLE.items()))
            extra = f" (modules that failed to import here: {broken})"
        raise KeyError(
            f"unknown strategy {name!r}. Registered: {known}{extra}"
        ) from None


def requires_cuda(name: str) -> bool:
    """True if this strategy declares it cannot run on CPU."""
    return bool(getattr(get_strategy(name), "REQUIRES_CUDA", False))


def dtype_name(dtype) -> str:
    """`torch.float16` or `"float16"` -> `"float16"`."""
    if isinstance(dtype, torch.dtype):
        return str(dtype).replace("torch.", "")
    return str(dtype).replace("torch.", "")


def supported_dtypes(name_or_class) -> frozenset:
    """Dtypes this strategy is numerically correct in, as names.

    Undeclared means all three. Declaring a subset is a statement that the
    strategy is *wrong* in the others -- not slow, wrong -- so both the test
    matrix and the dispatcher must honour it.
    """
    cls = name_or_class if isinstance(name_or_class, type) else get_strategy(name_or_class)
    declared = getattr(cls, "SUPPORTED_DTYPES", None)
    if declared is None:
        return ALL_DTYPES
    return frozenset(dtype_name(d) for d in declared)


def supports_dtype(name_or_class, dtype) -> bool:
    return dtype_name(dtype) in supported_dtypes(name_or_class)


def available_strategies(cuda_available: bool) -> Dict[str, Type[nn.Module]]:
    """Registry filtered to what can actually run on the current device."""
    if cuda_available:
        return dict(STRATEGIES)
    return {n: c for n, c in STRATEGIES.items() if not requires_cuda(n)}


def _autodiscover() -> None:
    """Import every submodule so their @register decorators run."""
    for module_info in pkgutil.iter_modules(__path__):
        if module_info.name.startswith("_"):
            continue
        full_name = f"{__name__}.{module_info.name}"
        try:
            importlib.import_module(full_name)
        except Exception as exc:  # noqa: BLE001 - a bad strategy must not break the rest
            UNAVAILABLE[module_info.name] = f"{type(exc).__name__}: {exc}"


_autodiscover()
