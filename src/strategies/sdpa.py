"""SDPA attention: replace the four-kernel score chain with one fused kernel.

The baseline's attention (`BaselineSelfAttention.forward` in
`bench/torch_transformer_benchmark.py`) computes `scores = q @ k^T`, masks it,
softmaxes it, then does `probs @ v` as four separate kernels, each reading and
writing the full `[B, H, S, S]` score/probability matrix to HBM. At
B=8, S=128 that matrix is 8*8*128*128*4 bytes = 4 MB, crossing the memory bus
on every one of those kernel boundaries. The arithmetic intensity of that
chain is roughly 0.125 FLOP/byte -- far below our measured ridge point of 62.9
FLOP/byte at TF32 (132.7 at bf16, see CLAUDE.md) -- so the baseline is deep in
the memory-bound region of the roofline: the GPU is mostly waiting on HBM
traffic for a matrix it never needed to write out at all.

`F.scaled_dot_product_attention` fuses that whole chain into one kernel that
keeps the score tile in SRAM and never materializes the full `[B, H, S, S]`
matrix in HBM. Same math, same fp32 softmax accumulation (SDPA's fused
backends accumulate the softmax in fp32 internally, matching
`torch.softmax(scores.float(), dim=-1)` in the reference) -- just without the
round trips.

Known precision limits
----------------------

Both limits below share one mechanism: this strategy is *more* accurate than
the reference, and the gap is the reference's own rounding. Neither is a
masking or causal bug, and neither can be closed from this file -- closing it
would mean making `bench/torch_transformer_benchmark.py` less accurate, and
that file must not be edited.

**bfloat16, all depths.** The reference casts softmax probabilities back to the
model dtype -- rounding them to bf16 -- *before* the `probs @ v` matmul. SDPA's
fused kernels keep those probabilities in fp32 all the way to that matmul.
Measured worst case is exactly 2 ULP of bf16: 1 ULP at magnitude ~2.17 is
0.015625 (bf16 has an 8-bit mantissa: 2^-7 * 2^exponent, and
2^floor(log2(2.17)) = 2^1, so 2^-7 * 2 = 0.015625), and the observed
max_abs_error there is 0.03125 -- 2 ULP -- a 1.44% relative error against the
accuracy check's 1% `rtol`.

**float16 at num_layers >= 2.** Not a passing config. Measured consistently
across 763ac93, b7f6312 and b96670e::

    fp16 L=1  PASS  max_abs 0.003906
    fp16 L=2  FAIL  max_abs 0.005859
    fp16 L=4  FAIL  max_abs 0.007812
    fp16 L=6  FAIL  max_abs 0.007812

Here it is `rtol=0.01` carrying the check, not `atol` -- fp16 error at these
magnitudes is well above 0.001, so every element rests on the relative bound,
and by two layers enough elements exceed 1% to fail. Same mechanism as bf16,
one rounding step finer: the reference rounds softmax probabilities to fp16
before `probs @ v` and we do not, so the two diverge by a rounding step per
layer and the divergence compounds with depth.

**float32 + causal at num_layers >= 6, with TF32 on.** Not a passing config.
Measured pass rate is **21/40 seeds (52%)** -- a coin flip, not a margin --
with median max_abs 0.00117 and worst 0.00136 against the 0.001 `atol` budget.
The error is entirely TF32: with `allow_tf32=False` on both implementations it
collapses to 1.9e-06 (650x smaller). With TF32 on -- the organizers' default --
the reference's `q @ k^T` and `probs @ v` are TF32 matmuls carrying ~1e-3
relative noise, while SDPA's fp32 path runs on EFFICIENT_ATTENTION, which does
not use TF32 reductions (Flash and cuDNN both reject fp32 inputs). The
divergence is the reference's noise, not ours.

It is depth that crosses the budget, not causality alone. Measured over 20
seeds each, fp32, TF32 on::

    causal=False L=6         20/20 (100%)   median max_abs 0.000735
    causal=True  L=1         20/20 (100%)   median max_abs 0.000484
    causal=True  L=2         20/20 (100%)   median max_abs 0.000705
    causal=True  L=6         11/20 ( 55%)   median max_abs 0.00120

Error grows ~sqrt(L) (4.8e-4 * sqrt(6) ~ 1.18e-3), and causal is ~2x
non-causal at equal depth -- fewer keys per softmax means less averaging of
the per-element noise. At L=6 that walks into the atol wall. Padding ratio is
irrelevant to this: the worst elements sit at sequence positions 1-2, where
causal masking already restricts attention to positions 0-2, so padding (which
only masks keys at position >= 358 at ratio 0.3) never touches them.

Two dead ends, recorded so they are not retried: forcing `SDPBackend.MATH`
does not help (10/20 seeds), and SDPA's habit of folding `scale` into `q`
before the matmul rather than scaling the product after contributes exactly
zero, because `head_dim ** -0.5 = 0.125` here is a power of two and commutes
exactly under any rounding.

.. warning::
   `SUPPORTED_DTYPES` below is **declarative only** -- as of this writing no
   module in the repo reads it, so it excludes nothing at runtime. That is why
   `results/results.csv` still contains failing bfloat16 rows. Enforcement
   lives in `src/dispatch.py` (Person B's file); see the request in
   `docs/APPROVALS_NEEDED.md`.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from src.baseline import BaselineSelfAttention, BaselineTransformer
from src.strategies import register


@register("sdpa")
class SdpaTransformer(BaselineTransformer):
    # SDPA has a math-backend CPU fallback, so this runs (slowly) on B's Mac too.
    REQUIRES_CUDA = False
    # bfloat16 excluded: see module docstring for the 2-ULP rounding-point
    # argument (baseline rounds softmax probs to bf16 before probs @ v; SDPA
    # keeps fp32 all the way through).
    #
    # DECLARATIVE ONLY -- nothing in the repo reads this attribute yet, so it
    # does not stop a bf16 run. See the warning in the module docstring.
    #
    # float16 is listed but is only safe at num_layers == 1; it fails from two
    # layers up. A flat dtype tuple cannot express that, which is the second
    # reason enforcement has to live in the router rather than here.
    SUPPORTED_DTYPES = (torch.float32, torch.float16)
    # float32 + causal + num_layers >= 6 passes only 52% of seeds with TF32 on.
    # Also declarative; dispatch must route this shape to "baseline".
    UNSUPPORTED_CAUSAL_MIN_LAYERS = 6

    def _attend(
        self,
        attn: BaselineSelfAttention,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
        causal: bool,
        has_padding: bool,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        q = attn._split_heads(attn.q_proj(x))
        k = attn._split_heads(attn.k_proj(x))
        v = attn._split_heads(attn.v_proj(x))

        # valid_token_mask is [B, S] bool, True = KEEP. SDPA's bool attn_mask
        # uses the same polarity (True = attend), which is the *opposite* of
        # masked_fill's "True = mask out" -- so this is used as-is, never
        # inverted.
        attn_mask = None
        is_causal = False
        if has_padding and causal:
            # SDPA rejects is_causal=True together with an explicit attn_mask,
            # so when both apply they are folded into one bool [B, 1, S, S]
            # mask up front.
            key_keep = valid_token_mask[:, None, None, :]  # [B, 1, 1, S]
            causal_keep = ~torch.ones(
                (seq_len, seq_len), device=x.device, dtype=torch.bool
            ).triu(diagonal=1)  # [S, S], True where attend is allowed
            attn_mask = key_keep & causal_keep  # broadcasts to [B, 1, S, S]
        elif has_padding:
            attn_mask = valid_token_mask[:, None, None, :]  # broadcasts over H, Sq
        elif causal:
            is_causal = True

        context = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=is_causal, scale=attn.scale
        )
        context = (
            context.transpose(1, 2).contiguous().view(batch, seq_len, attn.d_model)
        )
        output = attn.out_proj(context)

        # Output zeroed at invalid query positions -- site 1 of 3 (repeated
        # after each block, and once after the final norm, in forward()).
        if has_padding:
            output = output.masked_fill(~valid_token_mask[..., None], 0)
        return output

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        causal = self.config.causal
        # Fast path: an all-True mask is equivalent to no mask at all, and
        # checking it once here (instead of masking at every site below) skips
        # every masked_fill and lets the unpadded fast kernels run.
        has_padding = valid_token_mask is not None and not bool(valid_token_mask.all())

        for block in self.layers:
            attn_out = self._attend(
                block.attention, block.norm1(x), valid_token_mask, causal, has_padding
            )
            x = x + attn_out
            x = x + block.ffn_out(
                F.gelu(block.ffn_in(block.norm2(x)), approximate="none")
            )
            # Output zeroed at invalid query positions -- site 2 of 3.
            if has_padding:
                x = x.masked_fill(~valid_token_mask[..., None], 0)

        x = self.final_norm(x)
        # Output zeroed at invalid query positions -- site 3 of 3.
        if has_padding:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        return x
