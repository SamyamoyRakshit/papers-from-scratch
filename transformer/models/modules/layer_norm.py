import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    """
    Layer Normalization as described in "Attention is All You Need".
    
    Normalizes the input across the feature dimension (d_model),
    then applies learnable scale (gamma) and shift (beta) parameters.
    
    Formula: 
        output = gamma * (x - mean) / sqrt(variance + eps) + beta
    
    Args:
        d_model: The dimension of the model (feature dimension)
        eps: Small constant for numerical stability (default: 1e-6; for TensorFlow)
    """
    def __init__(self, d_model: int, eps: float = 1e-5): # took eps from pytorch default
        super().__init__()
        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(d_model)) # Scale parameter
        self.beta = nn.Parameter(torch.zeros(d_model)) # Shift parameter
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply layer normalization.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Normalized tensor of same shape as input
        """
        # Compute mean and std across the last dimension (d_model)
        # d_model is reduced via mean; keepdim=True preserves the dimension for broadcasting
        mean = x.mean(dim=-1, keepdim=True) # Shape: (batch_size, seq_len, 1) ## 1 because the mean of d_model a.k.a embedding dimension is 1
        
        # Use population variance (unbiased=False) to match Original Layer Normalization paper (https://arxiv.org/abs/1607.06450)
        var = x.var(dim=-1, keepdim=True, unbiased=False) # Shape: (batch_size, seq_len, 1 )
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta