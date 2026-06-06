import torch
import torch.nn as nn

class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network (Section 3.3)
    
    FFN(x) = max(0, xW₁ + b₁)W₂ + b₂

    Args:
        d_model: Dimension of the model (512 for base model)
        d_ff: Dimension of feed-forward network (2048 for base model)
        dropout: Dropout probability (default: 0.1)
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply position-wise feed-forward network.
        
        Args:
            x: Input tensor, shape (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor, shape (batch_size, seq_len, d_model)
        """
        # x: (batch_size, seq_len, d_model)
        x = self.linear1(x)     # → (batch_size, seq_len, d_ff)
        x = self.relu(x)        # ReLU activation
        x = self.dropout(x)     # Dropout
        x = self.linear2(x)     # → (batch_size, seq_len, d_model)
        return x