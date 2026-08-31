"""The entry point the organizers' benchmark instantiates.

Its whole job is to answer one question per forward pass -- *is the fused SDPA
path actually better for this shape?* -- and delegate accordingly. Both answers
are correct implementations; only one of them is faster, and which one that is
depends on the shape in a way that measurement, not intuition, decides.

Why a router at all
-------------------

The fused path is not uniformly a win. `results/results.csv` (fp32, L=6) says:

    seq_len   128 -> 0.910x    512 -> 1.664x   1024 -> 2.077x
    d_model   256 -> 2.487x    512 -> 1.664x   1024 -> 1.219x
    batch       1 -> 0.853x      8 -> 1.664x     32 -> 1.780x

Two of those are **below 1.0x** -- at S=128 and at B=1 the fused kernel is a
*regression*. The pattern is the one the roofline predicts: SDPA's advantage is
avoiding the [B, H, S, S] round trip to HBM, and that matrix only dominates once
it is large. At S=128 it is 4 MB and the launch overhead of the fused kernel is
not amortised; at B=1 there is not enough parallel work to fill the SMs.
d_model moving the other way (2.487x at 256, falling to 1.219x at 1024) is the
same story from the other side: as d_model grows the projections and FFN take
over the runtime, so the attention term we optimize shrinks as a fraction of it.

Rules
-----

Correctness gates first, then performance:

1. no CUDA               -> baseline  (per docs/DECISIONS.md 1.2)
2. bfloat16              -> baseline  (fails at every depth; see sdpa.py)
3. float16 and L >= 2    -> baseline  (fp16 passes at L=1 only; see below)
4. causal and L >= 6     -> baseline  (52% of seeds; see sdpa.py)
5. batch == 1            -> baseline  (measured 0.853x -- a regression)
6. seq_len <= 128        -> baseline  (measured 0.910x -- a regression)
7. otherwise             -> sdpa

Rules 1-4 are correctness gates rather than performance tuning, and they are
marked as such below so they are easy to find if the underlying limits move.

The float16 depth limit is measured, consistent across three commits
(763ac93, b7f6312, b96670e) and recorded in results.csv:

    fp16 L=1  PASS  max_abs 0.003906
    fp16 L=2  FAIL  max_abs 0.005859
    fp16 L=4  FAIL  max_abs 0.007812
    fp16 L=6  FAIL  max_abs 0.007812

Note the tolerance here is `rtol=0.01` doing the work, not `atol`: fp16 error
at these magnitudes is far above 0.001, so every element depends on the
relative bound, and by L=2 enough of them exceed 1% to fail. Routing fp16 to
the fused path at any depth past 1 ships a known-failing config.

Two notes on cost
-----------------

* **The routing decision never touches tensor data.** It reads `x.shape`,
  `x.dtype` and `x.is_cuda` plus fields of `self.config` -- all host-side. In
  particular it never reduces `valid_token_mask`, because `mask.all()` on CUDA
  is a device-to-host read that stalls the pipeline on every forward. Padding
  is not an input to any rule here, so the question never has to be asked.

* **No second module is held.** `UserOptimizedTransformer` subclasses
  `SdpaTransformer`, so it inherits both forward implementations and *one*
  parameter set under the baseline's exact names. Holding an instance of each
  strategy would duplicate every weight and break
  `copy_model_weights(..., strict=True)`.

`src.dispatch.select_strategy` reads the measured table generated from
results.csv and applies the registration, capability and dtype gates. It is
deliberately not used for the rules below: it is keyed on shape alone and
carries no layer count, so it cannot express the causal and float16 *depth*
limits that rules 3 and 4 encode. A router that silently disagreed with the
measurements would be worse than an explicit one. See docs/TECH_REPORT.md 7.2
for why admissibility turned out to be depth-dependent.
"""

from __future__ import annotations

from typing import Optional

import torch

from src.baseline import BaselineTransformer
from src.strategies.sdpa import SdpaTransformer

__all__ = ["UserOptimizedTransformer", "MIN_SEQ_LEN_FOR_SDPA", "MIN_BATCH_FOR_SDPA",
           "MAX_CAUSAL_LAYERS_FOR_SDPA"]

#: Below this, the fused kernel measured 0.910x -- slower than the baseline.
MIN_SEQ_LEN_FOR_SDPA = 256

#: At batch 1 the fused kernel measured 0.853x. Not enough parallel work.
MIN_BATCH_FOR_SDPA = 2

#: Causal at this depth or deeper passes only 52% of seeds. See sdpa.py.
MAX_CAUSAL_LAYERS_FOR_SDPA = 5

#: float16 passes at one layer and fails at two or more. See the table above.
MAX_FP16_LAYERS_FOR_SDPA = 1


class UserOptimizedTransformer(SdpaTransformer):
    """Routes each forward to whichever of baseline / sdpa is right for the shape.

    Subclasses `SdpaTransformer` to inherit its `forward` and `_attend` while
    sharing a single set of parameters registered under the baseline's names.
    `BaselineTransformer.forward` is still reachable and still correct on the
    same `self`, since neither implementation adds or renames a parameter.
    """

    # Inherited from SdpaTransformer and deliberately not narrowed: this class
    # routes *around* those limits rather than declaring itself subject to them.
    # Both attributes are declarative only -- nothing in the repo reads either
    # (see R4 in docs/DECISIONS.md), which is exactly why the limits are
    # enforced by the rules in `use_sdpa` instead of by declaration.

    def use_sdpa(self, x: torch.Tensor) -> bool:
        """The routing decision. Host-side only -- never syncs with the device.

        Split out from `forward` so tests can assert the routing table directly
        without running a model.
        """
        batch, seq_len, _ = x.shape

        # -- correctness gates -----------------------------------------------
        if not x.is_cuda:
            return False                                    # rule 1
        if x.dtype is torch.bfloat16:
            return False                                    # rule 2
        if (x.dtype is torch.float16
                and self.config.num_layers > MAX_FP16_LAYERS_FOR_SDPA):
            return False                                    # rule 3
        if self.config.causal and self.config.num_layers > MAX_CAUSAL_LAYERS_FOR_SDPA:
            return False                                    # rule 4

        # -- performance rules -----------------------------------------------
        if batch < MIN_BATCH_FOR_SDPA:
            return False                                    # rule 5
        if seq_len < MIN_SEQ_LEN_FOR_SDPA:
            return False                                    # rule 6
        return True                                         # rule 7

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_sdpa(x):
            return SdpaTransformer.forward(self, x, valid_token_mask)
        return BaselineTransformer.forward(self, x, valid_token_mask)
