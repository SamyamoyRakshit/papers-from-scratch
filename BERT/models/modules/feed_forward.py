import math

import torch
import torch.nn as nn


def gelu(x: torch.Tensor) -> torch.Tensor:
    """
    Gaussian Error Linear Unit (Hendrycks & Gimpel, 2016).
    Paper: https://arxiv.org/abs/1606.08415

    GELU weights each input by the probability that a standard-normal variable
    is ≤ x, i.e. gelu(x) = x · Φ(x), where Φ is the Gaussian CDF.

    Exact (erf) form:
        gelu(x) = x · Φ(x) = 0.5 · x · (1 + erf( x / √2 ))

    Tanh approximation (what this function computes, matching Google's TF BERT
    in modeling.py; it's much faster than computing `erf()`):
        gelu(x) ≈ 0.5 · x · (1 + tanh[ √(2/π) · (x + 0.044715·x³) ])

    BERT §A.2 mandates a gelu activation; the original code uses this tanh
    approximation rather than the exact erf form (nn.GELU's default).

    Args:
        x: Input tensor of any shape, typically (batch_size, seq_len, d_ff).

    Returns:
        Tensor of the same shape as x, with GELU applied element-wise.
    """
    return 0.5 * x * (1.0 + torch.tanh(
        math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
    ))


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.

    Same two-linear structure as the original Transformer (Section 3.3), but
    with GELU instead of ReLU:

        FFN(x) = GELU(xW₁ + b₁)W₂ + b₂

    The GELU swap is paper-stated — BERT §A.2: "We use a gelu activation
    (Hendrycks and Gimpel, 2016) rather than the standard relu, following
    OpenAI GPT." We use the tanh approximation written out above, matching
    Google's original TF BERT implementation.

    Args:
        d_model: Embedding / hidden dimension. Example (BERT-base): 768
        d_ff: Inner feed-forward dimension. Example (BERT-base): 3072 (= 4·d_model)
        dropout: Dropout probability (default: 0.1)
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor, shape (batch_size, seq_len, d_model)

        Returns:
            Output tensor, shape (batch_size, seq_len, d_model)
        """
        # x: (batch_size, seq_len, d_model)
        x = self.linear1(x)     # (batch_size, seq_len, d_ff)
        x = gelu(x)             # GELU, tanh approx (BERT §A.2)
        x = self.dropout(x)     # Dropout
        x = self.linear2(x)     # (batch_size, seq_len, d_model)
        return x