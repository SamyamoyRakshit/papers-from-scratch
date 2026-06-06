import torch
import torch.nn as nn

from .modules.feed_forward import FeedForward
from .modules.layer_norm import LayerNorm
from .modules.multi_head_attention import MultiHeadAttention


class DecoderLayer(nn.Module):
    """
    Single Decoder Layer (Section 3.1)

    Each layer has three sub-layers:
        1. Masked Multi-Head Self-Attention
        2. Multi-Head Cross-Attention (encoder-decoder attention)
        3. Position-wise Feed-Forward Network

    All three sub-layers use a residual connection followed by layer normalization:
        output = LayerNorm(x + Sublayer(x))

    Dropout is applied to the output of each sub-layer before the residual addition.

    The self-attention sub-layer uses a causal mask to prevent positions from
    attending to subsequent positions (Section 3.1).

    Args:
        d_model (int): Dimension of the model. Example (base): 512
        num_heads (int): Number of attention heads. Example (base): 8
        d_ff (int): Inner dimension of the feed-forward network. Example (base): 2048
        dropout (float): Dropout probability. Default: 0.1
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        # Sub-layer 1: Masked Multi-Head Self-Attention
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # Sub-layer 2: Multi-Head Cross-Attention (encoder-decoder)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        # Sub-layer 3: Position-wise Feed-Forward Network
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm3 = LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
            self,
            tgt: torch.Tensor,
            encoder_output: torch.Tensor,
            tgt_mask: torch.Tensor = None,
            memory_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            tgt: Target tensor, shape (batch_size, tgt_seq_len, d_model)
            encoder_output: Encoder output, shape (batch_size, src_seq_len, d_model)
            tgt_mask: Causal + padding mask for self-attention,
                      shape (batch_size, 1, tgt_seq_len, tgt_seq_len)
            memory_mask: Padding mask for cross-attention,
                         shape (batch_size, 1, 1, src_seq_len)

        Returns:
            Output tensor, shape (batch_size, tgt_seq_len, d_model)
        """
        # Sub-layer 1: Masked Self-Attention + Add & Norm
        # Q = K = V = tgt (decoder attends to itself)
        # tgt_mask prevents attending to future positions
        attn_output = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = self.norm1(tgt + self.dropout1(attn_output))

        # Sub-layer 2: Cross-Attention + Add & Norm
        # Q from decoder, K and V from encoder output
        cross_attn_output = self.cross_attn(tgt, encoder_output, encoder_output, memory_mask)
        tgt = self.norm2(tgt + self.dropout2(cross_attn_output))

        # Sub-layer 3: Feed-Forward + Add & Norm
        ff_output = self.feed_forward(tgt)
        tgt = self.norm3(tgt + self.dropout3(ff_output))

        return tgt
    

class Decoder(nn.Module):
    """
    Decoder Stack (Section 3.1)

    Stack of N identical DecoderLayers.

    The paper uses N=6 for the base model.

    Note: Embeddings and positional encoding are applied BEFORE the decoder.
          This module only handles the N decoder layers.

    Args:
        d_model (int): Dimension of the model. Example (base): 512
        num_heads (int): Number of attention heads. Example (base): 8
        d_ff (int): Inner dimension of the feed-forward network. Example (base): 2048
        num_layers (int): Number of decoder layers (N). Example (base): 6
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

        # N identical decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(
            self,
            tgt: torch.Tensor,
            encoder_output: torch.Tensor,
            tgt_mask: torch.Tensor = None,
            memory_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            tgt: Target tensor (after embedding + positional encoding),
                 shape (batch_size, tgt_seq_len, d_model)
            encoder_output: Encoder output,
                            shape (batch_size, src_seq_len, d_model)
            tgt_mask: Causal + padding mask for self-attention,
                      shape (batch_size, 1, tgt_seq_len, tgt_seq_len)
            memory_mask: Padding mask for cross-attention,
                         shape (batch_size, 1, 1, src_seq_len)

        Returns:
            Decoder output, shape (batch_size, tgt_seq_len, d_model)
        """
        for layer in self.layers:
            tgt = layer(tgt, encoder_output, tgt_mask, memory_mask)
        return tgt