import torch
import torch.nn as nn

# LayerNorm is imported from transformer/ unchanged: GPT-2's eps is 1e-5
# (OpenAI's model.py, `norm`), the same as its default. Run from repo root.
from transformer.models.modules.layer_norm import LayerNorm

from .modules.causal_self_attention import CausalSelfAttention
from .modules.feed_forward import FeedForward


class DecoderLayer(nn.Module):
    """
    One GPT-2 decoder block (GPT-2 §2.3)

        x = x + attn(ln1(x))     masked self-attention
        x = x + mlp(ln2(x))      position-wise MLP

    GPT-2 §2.3: "Layer normalization ... was moved to the input of each sub-block".
    This is pre-LN: the LayerNorm feeds the sub-layer and the add goes straight
    onto x, so the x → x path through all 6 blocks never passes through a LayerNorm.

    vs transformer/'s DecoderLayer, two changes:
      1. no cross_attn and no norm2: there is no encoder to attend to
      2. post-LN → pre-LN:
           transformer/:  x = norm1(x + dropout1(self_attn(x)))
           GPT:           x = x + attn(ln1(x))
    No dropout here: attn and mlp already drop their own output (after c_proj),
    so the 0.1 residual dropout happens once per sub-layer, not twice.

    Args:
        d_model (int): base.yaml 256.
        num_heads (int): base.yaml 8.
        d_ff (int): base.yaml 1024.
        dropout (float): GPT-1 §4.1: 0.1, passed to attn and mlp.
        attention_impl (str): "sdpa" | "reference", passed to attn.
            ln1:  256 + 256 =     512 params (gamma + beta)
            attn:             263,168
            ln2:                  512
            mlp:              525,568  → 789,760 per block
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float, attention_impl: str):
        super().__init__()
        self.ln1 = LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads, dropout, attention_impl)
        self.ln2 = LayerNorm(d_model)
        self.mlp = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, T, d_model). Block 0: output of GPTEmbeddings.
        Returns:
            (batch_size, T, d_model)
        """
        x = x + self.attn(self.ln1(x))    # tokens look at earlier tokens
        x = x + self.mlp(self.ln2(x))     # each token on its own: 256 → 1024 → 256
        return x


class Decoder(nn.Module):
    """
    Stack of N identical DecoderLayers. base.yaml: 6 × 789,760 = 4,738,560 params.

    Embeddings run BEFORE this; ln_f and lm_head run AFTER it, both in gpt.py.
    ln_f is GPT-2 §2.3's "additional layer normalization ... after the final
    self-attention block": with pre-LN, the last block's add is never normalized.

    Args:
        d_model, num_heads, d_ff, dropout, attention_impl: as DecoderLayer.
        num_layers (int): base.yaml 6. Paper: 12 (GPT-2 Table 2, smallest).
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, num_layers: int,
                 dropout: float, attention_impl: str):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout, attention_impl)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, T, d_model), output of GPTEmbeddings
        Returns:
            (batch_size, T, d_model), into ln_f in gpt.py
        """
        for layer in self.layers:
            x = layer(x)
        return x
