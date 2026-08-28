"""Task 5 acceptance: shape and capability -> strategy selection.

The table under test is generated from the synthetic results fixture, so these
tests exercise the same JSON shape `analysis/load.py` actually writes rather
than a hand-written approximation of it.

The theme throughout: dispatch may only ever pick something that will *run*.
A selection that is faster on paper but is not registered here, or needs
hardware this card does not have, converts a performance decision into a crash,
which is a strictly worse outcome than running the baseline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from analysis.load import load_results, write_dispatch_table
from src import dispatch
from src.baseline import BaselineTransformer, TransformerConfig
from src.dispatch import (
    CPU_FALLBACK_STRATEGY, DispatchKey, capability_name, clear_caches,
    device_capability, explain, select_strategy,
)

RESULTS_FIXTURE = Path(__file__).parent / "fixtures" / "results_synthetic.csv"

#: What the synthetic sweep "registered". Passed explicitly so these tests do
#: not depend on which strategies Person A happens to have written yet.
AVAILABLE = {name: None for name in
             ("baseline", "sdpa", "bf16", "compiled", "fused_qkv")}

SM_89 = (8, 9)
SM_80 = (8, 0)
SM_75 = (7, 5)


@pytest.fixture(scope="module")
def table_path(tmp_path_factory) -> str:
    path = tmp_path_factory.mktemp("dispatch") / "dispatch_table.json"
    write_dispatch_table(load_results(RESULTS_FIXTURE), path)
    return str(path)


@pytest.fixture(autouse=True)
def _fresh_caches():
    """Selection is memoised; a stale entry would mask a real change."""
    clear_caches()
    yield
    clear_caches()


def key(seq_len=512, batch=8, d_model=512, heads=8, dtype="float32",
        causal=False, padded=False) -> DispatchKey:
    return DispatchKey(batch=batch, seq_len=seq_len, d_model=d_model, heads=heads,
                       dtype=dtype, causal=causal, padded=padded)


# ---------------------------------------------------------------------------
# rule 1 — no CUDA
# ---------------------------------------------------------------------------

def test_no_cuda_selects_the_cpu_fallback(table_path):
    assert select_strategy(key(), capability=None, table_path=table_path,
                           available=AVAILABLE) == CPU_FALLBACK_STRATEGY
    assert "no CUDA" in explain(key(), capability=None, table_path=table_path,
                                available=AVAILABLE)


def test_cpu_fallback_is_registered():
    """Whatever the CPU fallback is set to, it has to exist."""
    from src.strategies import STRATEGIES

    assert CPU_FALLBACK_STRATEGY in STRATEGIES


def test_device_capability_is_none_without_a_gpu():
    if torch.cuda.is_available():
        assert device_capability() is not None
    else:
        assert device_capability() is None


def test_capability_name_formatting():
    assert capability_name((8, 9)) == "sm_89"
    assert capability_name((7, 5)) == "sm_75"


# ---------------------------------------------------------------------------
# rule 3 — exact match
# ---------------------------------------------------------------------------

def test_exact_match_returns_the_measured_winner(table_path):
    table = json.loads(Path(table_path).read_text())
    raw, expected = next(iter(table["sm_89"].items()))
    parts = raw.split(",")
    exact = DispatchKey(
        batch=int(parts[0]), seq_len=int(parts[1]), d_model=int(parts[2]),
        heads=int(parts[3]), dtype=parts[4],
        causal=parts[5] == "True", padded=parts[6] == "True",
    )
    assert select_strategy(exact, SM_89, table_path, AVAILABLE) == expected
    assert "exact match" in explain(exact, SM_89, table_path, AVAILABLE)


def test_long_and_short_sequences_can_choose_differently(table_path):
    """If every shape picked the same strategy the dispatch layer would be
    theatre. On this fixture the crossover is real."""
    long_seq = select_strategy(key(seq_len=2048), SM_89, table_path, AVAILABLE)
    short_seq = select_strategy(key(seq_len=128), SM_89, table_path, AVAILABLE)
    assert long_seq != short_seq


# ---------------------------------------------------------------------------
# rule 4 — nearest neighbour
# ---------------------------------------------------------------------------

def test_unmeasured_shape_falls_back_to_the_nearest_neighbour(table_path):
    reason = explain(key(seq_len=1536), SM_89, table_path, AVAILABLE)
    assert "nearest measured neighbour" in reason
    assert select_strategy(key(seq_len=1536), SM_89, table_path, AVAILABLE) in AVAILABLE


def test_neighbour_distance_is_logarithmic_not_linear():
    """S=1536 is nearer 2048 than 1024 in log space (0.415 vs 0.585), and the
    axes are swept multiplicatively, so a linear metric picks the wrong one."""
    entries = {
        "8,1024,512,8,float32,False,False": "compiled",
        "8,2048,512,8,float32,False,False": "sdpa",
    }
    found = dispatch._nearest(key(seq_len=1536), entries)
    assert found is not None and found[1] == "sdpa"


def test_neighbour_never_crosses_a_mask_or_dtype_boundary():
    """Borrowing a measurement across the causal/padded boundary would
    recommend a kernel for a branch it was never measured on."""
    entries = {"8,512,512,8,float32,False,False": "sdpa"}
    assert dispatch._nearest(key(causal=True), entries) is None
    assert dispatch._nearest(key(padded=True), entries) is None
    assert dispatch._nearest(key(dtype="bfloat16"), entries) is None
    assert dispatch._nearest(key(), entries) is not None


# ---------------------------------------------------------------------------
# rule 5 — capability gating
# ---------------------------------------------------------------------------

def test_a_pre_ampere_card_never_gets_a_bf16_strategy(tmp_path):
    """bf16 arithmetic needs Ampere. Even when the table says otherwise."""
    path = tmp_path / "bf16_everywhere.json"
    path.write_text(json.dumps({
        "sm_75": {"8,512,512,8,float32,False,False": "bf16"},
        "default": "bf16",
    }))
    chosen = select_strategy(key(), SM_75, str(path), AVAILABLE)
    assert "bf16" not in chosen
    assert chosen == "baseline"
    assert "rejected" in explain(key(), SM_75, str(path), AVAILABLE)


def test_the_same_table_gives_bf16_on_ampere(tmp_path):
    """The gate must be the capability, not a blanket ban on the name."""
    path = tmp_path / "bf16.json"
    path.write_text(json.dumps({
        "sm_80": {"8,512,512,8,float32,False,False": "bf16"}, "default": "bf16",
    }))
    assert select_strategy(key(), SM_80, str(path), AVAILABLE) == "bf16"


def test_triton_strategies_are_gated_below_turing(tmp_path):
    path = tmp_path / "triton.json"
    path.write_text(json.dumps({"sm_70": {}, "default": "triton_layernorm"}))
    available = {**AVAILABLE, "triton_layernorm": None}
    assert select_strategy(key(), (7, 0), str(path), available) == "baseline"
    assert select_strategy(key(), (7, 5), str(path), available) == "triton_layernorm"


def test_a_declared_min_capability_beats_the_name_heuristic(tmp_path):
    """A strategy that says what it needs is believed over its name."""

    class NeedsHopper(BaselineTransformer):
        MIN_CAPABILITY = (9, 0)

    path = tmp_path / "t.json"
    path.write_text(json.dumps({"default": "innocuous_name"}))
    available = {"baseline": None, "innocuous_name": NeedsHopper}

    assert select_strategy(key(), SM_89, str(path), available) == "baseline"
    assert select_strategy(key(), (9, 0), str(path), available) == "innocuous_name"


def test_gating_falls_through_to_a_permitted_neighbour(tmp_path):
    """A rejected exact match must not end the search."""
    path = tmp_path / "mixed.json"
    path.write_text(json.dumps({
        "sm_75": {
            "8,512,512,8,float32,False,False": "bf16",       # rejected on sm_75
            "8,2048,512,8,float32,False,False": "compiled",  # allowed
        },
        "default": "baseline",
    }))
    chosen = select_strategy(key(seq_len=512), SM_75, str(path), AVAILABLE)
    assert chosen == "compiled"
    assert "rejected" in explain(key(seq_len=512), SM_75, str(path), AVAILABLE)


# ---------------------------------------------------------------------------
# dtype gate
# ---------------------------------------------------------------------------

class Fp32AndFp16Only(BaselineTransformer):
    """Like the real SDPA strategy: correct in fp32/fp16, cannot meet the
    tolerance in bf16 (see docs/INTERFACE.md 5.1)."""

    SUPPORTED_DTYPES = (torch.float32, torch.float16)


def test_a_strategy_is_never_selected_for_a_dtype_it_disclaims(tmp_path):
    """The worst failure mode available is a plausible wrong answer. A strategy
    that cannot meet the tolerance in bf16 must never be handed a bf16 tensor,
    even when the table's default names it."""
    path = tmp_path / "t.json"
    path.write_text(json.dumps({
        "sm_89": {"8,512,512,8,bfloat16,False,False": "sdpa"},
        "default": "sdpa",
    }))
    available = {"baseline": BaselineTransformer, "sdpa": Fp32AndFp16Only}

    assert select_strategy(key(dtype="bfloat16"), SM_89, str(path), available) == "baseline"
    # ...and the same strategy is still selected for the dtypes it does claim.
    assert select_strategy(key(dtype="float32"), SM_89, str(path), available) == "sdpa"
    assert select_strategy(key(dtype="float16"), SM_89, str(path), available) == "sdpa"


def test_the_dtype_rejection_reason_names_the_gate(tmp_path):
    """"Rejected" alone is not actionable; the reason must say which gate."""
    path = tmp_path / "t.json"
    path.write_text(json.dumps({
        "sm_89": {"8,512,512,8,bfloat16,False,False": "sdpa"}, "default": "baseline",
    }))
    available = {"baseline": BaselineTransformer, "sdpa": Fp32AndFp16Only}
    reason = explain(key(dtype="bfloat16"), SM_89, str(path), available)
    assert "does not support bfloat16" in reason
    assert "float16" in reason and "float32" in reason


def test_the_hard_coded_fallback_respects_the_dtype_gate():
    """The regression this exists for: with no results/ directory at all, the
    fallback default was selected for every dtype including unsupported ones."""
    available = {"baseline": BaselineTransformer, "sdpa": Fp32AndFp16Only}
    assert select_strategy(key(dtype="bfloat16"), SM_89,
                           "/definitely/not/a/file.json", available) == "baseline"
    assert select_strategy(key(dtype="float32"), SM_89,
                           "/definitely/not/a/file.json", available) == "sdpa"


def test_an_undeclared_strategy_is_assumed_to_support_everything():
    """Not declaring SUPPORTED_DTYPES must not silently disable a strategy."""
    available = {"baseline": BaselineTransformer, "anything": BaselineTransformer}
    for dtype in ("float32", "float16", "bfloat16"):
        assert select_strategy(key(dtype=dtype), SM_89,
                               "/definitely/not/a/file.json", available) in available


def test_dtype_and_capability_gates_compose(tmp_path):
    """Both gates apply; failing either removes the candidate."""
    class Bf16OnlyAmpere(BaselineTransformer):
        SUPPORTED_DTYPES = (torch.bfloat16,)
        MIN_CAPABILITY = (8, 0)

    path = tmp_path / "t.json"
    path.write_text(json.dumps({"default": "bf16_path"}))
    available = {"baseline": BaselineTransformer, "bf16_path": Bf16OnlyAmpere}

    assert select_strategy(key(dtype="bfloat16"), SM_80, str(path), available) == "bf16_path"
    assert select_strategy(key(dtype="bfloat16"), SM_75, str(path), available) == "baseline"
    assert select_strategy(key(dtype="float32"), SM_80, str(path), available) == "baseline"


# ---------------------------------------------------------------------------
# registration gate
# ---------------------------------------------------------------------------

def test_an_unregistered_strategy_is_never_selected(table_path):
    """The table is generated from Person A's machine; this one may not have
    every strategy yet. Recommending a class that does not exist here would be
    an AttributeError inside forward()."""
    only_baseline = {"baseline": None}
    for seq_len in (128, 512, 1024, 2048):
        assert select_strategy(key(seq_len=seq_len), SM_89, table_path,
                               only_baseline) == "baseline"


def test_selection_is_always_something_that_can_run(table_path):
    from src.strategies import STRATEGIES

    for capability in (SM_75, SM_80, SM_89, None):
        for seq_len in (1, 128, 777, 4096):
            chosen = select_strategy(key(seq_len=seq_len), capability, table_path)
            assert chosen in STRATEGIES, chosen


# ---------------------------------------------------------------------------
# missing / broken table
# ---------------------------------------------------------------------------

def test_missing_table_uses_the_hard_coded_fallback():
    """The submission must not depend on a generated artefact."""
    reason = explain(key(), SM_89, "/definitely/not/a/file.json", AVAILABLE)
    assert "hard-coded fallback" in reason
    assert select_strategy(key(), SM_89, "/definitely/not/a/file.json",
                           AVAILABLE) == dispatch.FALLBACK_TABLE["default"]


def test_malformed_table_degrades_instead_of_raising(tmp_path):
    """This runs inside forward(). A model that refuses to run because a JSON
    file is corrupt is worse than one that runs a slower kernel."""
    for content in ("{not json", "[]", ""):
        path = tmp_path / f"broken_{abs(hash(content))}.json"
        path.write_text(content)
        clear_caches()
        assert select_strategy(key(), SM_89, str(path), AVAILABLE) in AVAILABLE


def test_unmeasured_capability_block_says_so(table_path):
    reason = explain(key(), (8, 6), table_path, AVAILABLE)
    assert "no measured data for sm_86" in reason


# ---------------------------------------------------------------------------
# key construction and caching
# ---------------------------------------------------------------------------

def test_table_key_matches_what_the_analysis_layer_writes(table_path):
    """If these two ever disagree, every exact match silently becomes a
    nearest-neighbour and nobody notices."""
    table = json.loads(Path(table_path).read_text())
    for raw in table["sm_89"]:
        parsed = dispatch._parse_key(raw)
        assert parsed is not None
        assert parsed.table_key() == raw


def test_from_forward_reads_the_shape_off_the_tensor():
    config = TransformerConfig(2, 32, 64, 4, 128, 2, causal=True)
    x = torch.randn(2, 32, 64)
    mask = torch.ones(2, 32, dtype=torch.bool)

    unpadded = DispatchKey.from_forward(x, mask, config)
    assert (unpadded.batch, unpadded.seq_len, unpadded.d_model) == (2, 32, 64)
    assert unpadded.heads == 4 and unpadded.causal is True
    assert unpadded.dtype == "float32"
    assert unpadded.padded is False

    mask[0, 20:] = False
    assert DispatchKey.from_forward(x, mask, config).padded is True
    assert DispatchKey.from_forward(x, None, config).padded is False


def test_from_forward_accepts_an_explicit_padded_flag():
    """Deriving `padded` costs a device sync; callers who already know skip it."""
    config = TransformerConfig(1, 8, 32, 4, 64, 1, causal=False)
    x = torch.randn(1, 8, 32)
    mask = torch.ones(1, 8, dtype=torch.bool)
    assert DispatchKey.from_forward(x, mask, config, padded=True).padded is True


def test_repeated_selection_is_memoised(table_path):
    before = dispatch._select_cached.cache_info()
    for _ in range(50):
        select_strategy(key(), SM_89, table_path, AVAILABLE)
    after = dispatch._select_cached.cache_info()
    assert after.hits - before.hits >= 49


def test_clear_caches_actually_clears(table_path):
    select_strategy(key(), SM_89, table_path, AVAILABLE)
    assert dispatch._select_cached.cache_info().currsize > 0
    clear_caches()
    assert dispatch._select_cached.cache_info().currsize == 0
