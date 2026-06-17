import torch
import torch.nn as nn

# MultiHeadAttention and LayerNorm are reused UNCHANGED from transformer/ —
# BERT's encoder is the same Transformer encoder block. The one BERT-specific
# swap is the feed-forward network (GELU instead of ReLU), imported locally.
# Run from repo root so `transformer` resolves as a package.
from transformer.models.modules.multi_head_attention import MultiHeadAttention
from transformer.models.modules.layer_norm import LayerNorm

from .modules.feed_forward import FeedForward


class EncoderLayer(nn.Module):
    """
    Single BERT Encoder Layer.

    Architecturally identical to the original Transformer encoder layer
    ("Attention Is All You Need", Section 3.1) — two sub-layers, each wrapped
    in a residual connection followed by LayerNorm (post-LN):

        output = LayerNorm(x + Sublayer(x))

    The two sub-layers are:
        1. Multi-Head Self-Attention  (reused from transformer/)
        2. Position-wise Feed-Forward Network  (BERT's GELU variant)

    The ONLY difference from our transformer/ encoder layer is sub-layer 2:
    BERT uses GELU in the FFN (§A.2), so we import the local FeedForward.

    Args:
        d_model (int): Hidden dimension. Example (BERT-base): 768
        num_heads (int): Number of attention heads. Example (BERT-base): 12
        d_ff (int): Inner feed-forward dimension. Example (BERT-base): 3072
        dropout (float): Dropout probability. Default: 0.1
        layer_norm_eps (float): ε for LayerNorm. Default: 1e-12 (BERT's value;
            matches BERTEmbeddings, not the transformer's 1e-5).
    """
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_ff: int,
            dropout: float = 0.1,
            layer_norm_eps: float = 1e-12
    ):
        super().__init__()

        # Sub-layer 1: Multi-Head Self-Attention
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)

        # Sub-layer 2: Position-wise Feed-Forward Network
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            src: Input tensor, shape (batch_size, seq_len, d_model)
            src_mask: Optional padding mask, shape (batch_size, 1, 1, seq_len)

        Returns:
            Output tensor, shape (batch_size, seq_len, d_model)
        """
        # Sub-layer 1: Self-Attention + Add & Norm
        # Self-attention: Q = K = V = src. Same bidirectional block as the
        # transformer ENCODER (padding mask only — no causal mask). BERT is
        # "bidirectional" because it's encoder-only: unlike GPT / the transformer
        # decoder, no layer ever masks future tokens.
        attn_output = self.self_attn(src, src, src, src_mask)
        src = self.norm1(src + self.dropout1(attn_output))

        # Sub-layer 2: Feed-Forward + Add & Norm
        ff_output = self.feed_forward(src)
        src = self.norm2(src + self.dropout2(ff_output))

        return src
    

class Encoder(nn.Module):
    """
    BERT Encoder: a stack of N identical EncoderLayers.

    BERT is an encoder-only Transformer — this stack IS the model body. The
    paper's two sizes:
        BERT-base:  num_layers=12, d_model=768,  num_heads=12  (d_ff=3072)
        BERT-large: num_layers=24, d_model=1024, num_heads=16  (d_ff=4096)

    Note: embeddings (token + segment + position) are applied BEFORE the
    encoder. This module only handles the N stacked layers.

    Args:
        d_model (int): Hidden dimension. Example (BERT-base): 768
        num_heads (int): Number of attention heads. Example (BERT-base): 12
        d_ff (int): Inner feed-forward dimension. Example (BERT-base): 3072
        num_layers (int): Number of encoder layers (N). Example (BERT-base): 12
        dropout (float): Dropout probability. Default: 0.1
        layer_norm_eps (float): ε for LayerNorm. Default: 1e-12 (BERT's value).
    """
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_ff: int,
            num_layers: int,
            dropout: float = 0.1,
            layer_norm_eps: float = 1e-12
    ):
        super().__init__()

        # N identical encoder layers
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout, layer_norm_eps)
            for _ in range(num_layers)
        ])

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            src: Embedded input (token + segment + position),
                 shape (batch_size, seq_len, d_model)
            src_mask: Optional padding mask, shape (batch_size, 1, 1, seq_len)

        Returns:
            Encoder output, shape (batch_size, seq_len, d_model)
        """
        for layer in self.layers:
            src = layer(src, src_mask)

        return src