import torch
import torch.nn as nn
import torch.nn.functional as F

# Both imported from transformer/ unchanged, for the reference path only.
# Run from repo root so `transformer` resolves.
from transformer.models.modules.multi_head_attention import MultiHeadAttention
from transformer.utils.mask_utils import create_causal_mask


class CausalSelfAttention(nn.Module):
    """
    GPT-2 masked multi-head self-attention (GPT-1 §3.1; GPT-2 §2.3 "largely follows" GPT-1)

        q, k, v = split(x·W_attn + b_attn)               one Linear(d, 3d)
        y       = softmax(q·kᵀ/√d_k + causal mask)·v     per head
        out     = Dropout( y·W_proj + b_proj )

    Masked: token t sees tokens 0..t, never t+1.., so predicting t+1 can't peek
    at the answer. Names c_attn / c_proj match OpenAI's checkpoint.

    vs transformer/'s MultiHeadAttention:
      - one fused c_attn (d → 3d) instead of W_q / W_k / W_v: one matmul, not three.
        Same params either way: 3·(d²+d).
      - self-attention only: q, k, v all come from x (no encoder memory).
      - F.scaled_dot_product_attention(is_causal=True): no mask tensor is built.

    attention_impl (config.model.attention_impl):
      "sdpa":      the fast path, used for training.
      "reference": transformer/'s MHA + create_causal_mask. Its four Linears are
                   swapped for nn.Identity, so it runs on THIS module's c_attn / c_proj:
                   same weights, no copy, no extra params. Only the attention math
                   in between differs, and that is what the equality check tests.
    Both paths are always built, so state_dict is the same under either flag.

    Dropout, both 0.1 (GPT-1 §4.1 "residual, embedding, and attention dropouts"):
      - attention: on the softmax weights, inside sdpa (dropout_p) / MHA (self.dropout)
      - residual:  after c_proj, before decoder.py adds it to x

    Init is NOT done here: gpt.py scales c_proj by 1/√(2·num_layers), same as the MLP's.

    Args:
        d_model (int): base.yaml 256.
        num_heads (int): base.yaml 8 → d_k = 256/8 = 32.
            c_attn: 256×768 + 768 = 197,376 params
            c_proj: 256×256 + 256 =  65,792 params  → 263,168 per block
        dropout (float): GPT-1 §4.1: 0.1.
        attention_impl (str): "sdpa" | "reference".
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float, attention_impl: str):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.attention_impl = attention_impl
        self.attn_dropout_p = dropout                    # a float: F.sdpa takes p, not a module

        self.c_attn = nn.Linear(d_model, 3 * d_model)    # x → [q | k | v]: 256 → 768
        self.c_proj = nn.Linear(d_model, d_model)        # heads joined → 256
        self.dropout = nn.Dropout(dropout)               # residual dropout

        # transformer/'s MHA with its projections turned into pass-throughs:
        # q, k, v go straight into its split_heads → masked softmax → combine_heads.
        self.reference = MultiHeadAttention(d_model, num_heads, dropout)
        for name in ("W_q", "W_k", "W_v", "W_o"):
            setattr(self.reference, name, nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, T, d_model), output of ln1 inside a decoder block
        Returns:
            (batch_size, T, d_model)
        """
        B, T, _ = x.shape
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)    # each (B, T, 256)

        if self.attention_impl == "sdpa":
            # (B, T, 256) → (B, T, 8, 32) → (B, 8, T, 32): one (T, 32) slice per head
            q = q.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
            k = k.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
            v = v.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
            y = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_dropout_p if self.training else 0.0,   # F.* doesn't see .eval()
                is_causal=True,
            )                                                               # (B, 8, T, 32)
            y = y.transpose(1, 2).contiguous().view(B, T, self.d_model)     # (B, T, 256)
        else:
            mask = create_causal_mask(T).to(x.device)    # (1, 1, T, T) bool, lower triangle
            y = self.reference(q, k, v, mask)            # (B, T, 256)

        return self.dropout(self.c_proj(y))
