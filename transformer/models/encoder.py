import torch
import torch.nn as nn

from .modules.feed_forward import FeedForward
from .modules.layer_norm import LayerNorm
from .modules.multi_head_attention import MultiHeadAttention


class EncoderLayer(nn.Module):
    """
    Single Encoder Layer (Section 3.1)

    Each layer has two sub-layers:
        1. Multi-Head Self-Attention
        2. Position-wise Feed-Forward Network

    Both sub-layers use a residual connection followed by layer normalization:
        output = LayerNorm(x + Sublayer(x))

    Dropout is applied to the output of each sub-layer before the residual addition.

    Args:
        d_model (int): Dimension of the model. Example (base): 512
        num_heads (int): Number of attention heads. Example (base): 8
        d_ff (int): Inner dimension of the feed-forward network. Example (base): 2048
        dropout (float): Dropout probability. Default: 0.1
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        # Sub-layer 1: Multi-Head Self-Attention
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # Sub-layer 2: Position-wise Feed-Forward Network
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            src: Source tensor, shape (batch_size, seq_len, d_model)
            src_mask: Optional padding mask, shape (batch_size, 1, 1, seq_len)

        Returns:
            Output tensor, shape (batch_size, seq_len, d_model)
        """
        # Sub-layer 1: Self-Attention + Add & Norm
        # In self-attention, Q = K = V = src (the encoder attends to itself)
        attn_output = self.self_attn(src, src, src, src_mask)
        src = self.norm1(src + self.dropout1(attn_output))

        # Sub-layer 2: Feed-Forward + Add & Norm
        ff_output = self.feed_forward(src)
        src = self.norm2(src + self.dropout2(ff_output))

        return src
    

class Encoder(nn.Module):
    """
    Encoder Stack (Section 3.1)

    Stack of N identical EncoderLayers (Section 3.1).

    The paper uses N=6 for the base model.

    Note: Embeddings and positional encoding are applied BEFORE the encoder.
          This module only handles the N encoder layers.

    Args:
        d_model (int): Dimension of the model. Example (base): 512
        num_heads (int): Number of attention heads. Example (base): 8
        d_ff (int): Inner dimension of the feed-forward network. Example (base): 2048
        num_layers (int): Number of encoder layers (N). Example (base): 6
        dropout (float): Dropout probability. Default: 0.1
    """
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_ff: int,
            num_layers: int,
            dropout: float = 0.1
    ):
        super().__init__()

        # N identical encoder layers
        self.layers = nn.ModuleList(
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        )

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            src: Source tensor (after embedding + positional encoding),
                 shape (batch_size, seq_len, d_model)
            src_mask: Optional padding mask, shape (batch_size, 1, 1, seq_len)

        Returns:
            Encoder output, shape (batch_size, seq_len, d_model)
        """
        for layer in self.layers:
            src = layer(src, src_mask)

        return src