"""Guards for the Task 0 deliverables themselves.

These do not test any optimization. They test that the two things the whole
project rests on are still true: the organizers' file is unmodified, and the
strategy registry honours the contract in docs/INTERFACE.md.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest
import torch

from src.baseline import BaselineTransformer, TransformerConfig, copy_model_weights
from src.strategies import STRATEGIES, UNAVAILABLE, get_strategy, register

REPO_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = REPO_ROOT / "bench" / "torch_transformer_benchmark.py"
INTERFACE = REPO_ROOT / "docs" / "INTERFACE.md"


def test_organizers_file_is_unmodified():
    """The SHA-256 in INTERFACE.md must still match the file on disk.

    If this fails, either the organizers' file was edited (rule 1 broken, the
    run is not comparable to theirs) or the pinned hash is stale. Fix the file,
    not the hash.
    """
    actual = hashlib.sha256(OFFICIAL.read_bytes()).hexdigest()
    pinned = re.search(r"SHA-256\s+([0-9a-f]{64})", INTERFACE.read_text())
    assert pinned is not None, "no SHA-256 pinned in docs/INTERFACE.md"
    assert actual == pinned.group(1), (
        f"bench/torch_transformer_benchmark.py has changed.\n"
        f"  pinned in INTERFACE.md: {pinned.group(1)}\n"
        f"  on disk:                {actual}"
    )


def test_baseline_control_is_registered():
    assert "baseline" in STRATEGIES
    assert issubclass(STRATEGIES["baseline"], BaselineTransformer)


def test_every_strategy_subclasses_baseline_and_accepts_the_signature():
    """Contract: subclass of BaselineTransformer, mask arg optional."""
    import inspect

    for name, cls in STRATEGIES.items():
        assert issubclass(cls, BaselineTransformer), f"{name} must subclass BaselineTransformer"
        params = list(inspect.signature(cls.forward).parameters.values())
        names = [p.name for p in params]
        assert names[:3] == ["self", "x", "valid_token_mask"], (
            f"{name}.forward must be (self, x, valid_token_mask=None), got {names}"
        )
        assert params[2].default is None, f"{name}: valid_token_mask must default to None"
        for extra in params[3:]:
            assert extra.default is not inspect.Parameter.empty, (
                f"{name}.forward has a required extra argument {extra.name!r}; "
                "the organizers' script only ever passes (x, valid_mask)"
            )


def test_strategies_accept_baseline_weights_strictly():
    """copy_model_weights(..., strict=True) must succeed for every strategy."""
    config = TransformerConfig(
        batch_size=1, seq_len=8, d_model=32, num_heads=4, ffn_dim=64,
        num_layers=1, causal=False,
    )
    baseline = BaselineTransformer(config)
    for name, cls in STRATEGIES.items():
        if getattr(cls, "REQUIRES_CUDA", False) and not torch.cuda.is_available():
            pytest.skip(f"{name} declares REQUIRES_CUDA and there is no GPU here")
        copy_model_weights(baseline, cls(config), strict=True)


def test_get_strategy_error_lists_known_names():
    with pytest.raises(KeyError, match="baseline"):
        get_strategy("definitely-not-a-strategy")


def test_register_rejects_a_non_subclass():
    with pytest.raises(TypeError):
        register("bogus")(type("NotATransformer", (torch.nn.Module,), {}))
    assert "bogus" not in STRATEGIES


def test_no_strategy_module_failed_to_import_silently():
    """Import failures are recorded, not swallowed. Report them loudly here."""
    if UNAVAILABLE:
        pytest.skip("strategy modules unavailable on this machine: " + str(UNAVAILABLE))
