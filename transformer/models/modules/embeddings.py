import torch
import torch.nn as nn
import math

class Embeddings(nn.Module):
    """
    Token Embeddings with scaling as described in "Attention is All You Need" (Section 3.4).
    
    The paper states:
    "In the embedding layers, we multiply those weights by sqrt(d_model)."
    
    This scaling helps balance the magnitude of embeddings with positional encodings.
    
    Args:
        vocab_size: Size of the vocabulary
        d_model: Dimension of the model (embedding dimension)
    """
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert token IDs to embeddings.
        
        Args:
            x: Input tensor of token IDs, shape (batch_size, seq_len)
            
        Returns:
            Embeddings scaled by sqrt(d_model), shape (batch_size, seq_len, d_model)
        """
        # Scale embeddings by sqrt(d_model) as per the paper
        return self.embeddings(x) * math.sqrt(self.d_model)