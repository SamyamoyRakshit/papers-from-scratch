import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding as described in "Attention is All You Need" (Section 3.5).
    
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    where:
        pos = position in the sequence
        i = dimension index
    
    Args:
        d_model: Dimension of the model (must be even)
        dropout: Dropout probability (default: 0.1, as per paper Section 5.4)
        max_len: Maximum sequence length (default: 5000)
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, n: int=10000):
        super().__init__()

        # Ensure d_model is even for proper sin/cos pairing
        assert d_model % 2 == 0, f"d_model must be even, got {d_model}"

        self.dropout = nn.Dropout(p = dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)

        # Create position indices [0, 1, 2, ..., max_len-1]
        # `torch.arange` ranges [start, end); means start to end-1
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Create division term: 10000^(2i/d_model)
        # Using exp(log) for numerical stability
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (- math.log(n) / d_model)
        )

        # Apply sine to even indices
        # '0::2' means [start,end,step]-> so start from 0 and at every 2 steps
        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension: [1, max_len, d_model]
        pe = pe.unsqueeze(0)

        # Register as buffer (not a parameter, but part of state_dict)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input embeddings from Embeddings layer, 
               shape (batch_size, seq_len, d_model)
            
        Returns:
            Embeddings + positional encoding with dropout applied,
            shape (batch_size, seq_len, d_model)
        """
        # Add positional encoding (slice to match input sequence length)
        x = x + self.pe[:, :x.size(1)]

        # Apply dropout to the sum (as per paper Section 5.4)
        return self.dropout(x)