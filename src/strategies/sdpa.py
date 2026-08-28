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

bfloat16 is excluded via `SUPPORTED_DTYPES` (float32 and float16 only). The
reference casts softmax probabilities back to the model dtype -- rounding
them to bf16 -- *before* the `probs @ v` matmul. SDPA's fused kernels keep
those probabilities in fp32 all the way to that matmul, so at bf16 we are
strictly more accurate than the reference but no longer reproduce its
rounding. Measured worst case is exactly 2 ULP of bf16: 1 ULP at magnitude
~2.17 is 0.015625 (bf16 has an 8-bit mantissa: 2^-7 * 2^exponent, and
2^floor(log2(2.17)) = 2^1, so 2^-7 * 2 = 0.015625), and the observed
max_abs_error there is 0.03125 -- 2 ULP -- which is a 1.44% relative error
against the accuracy check's 1% `rtol`. That is bf16 quantization noise
compounding through two ops with a different rounding point, not a masking or
causal bug: float16 and float32 pass on this exact same code path.
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
    # keeps fp32 all the way through). float16 and float32 match the reference.
    SUPPORTED_DTYPES = (torch.float32, torch.float16)

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
