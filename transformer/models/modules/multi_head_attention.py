import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention (Section 3.2.2)

    MultiHead(Q, K, V) = Concat(head₁, ..., headₕ) W^O
    where headᵢ = Attention(QW^Q_i, KW^K_i, VW^V_i)

    Scaled Dot-Product Attention (Section 3.2.1):
    Attention(Q, K, V) = softmax(QK^T / √d_k) V

    Args:
        d_model (int): Dimension of the model.
            Example (Transformer base): 512
        num_heads (int): Number of attention heads.
            Must divide d_model evenly.
            Example (Transformer base): 8
        dropout (float, optional): Dropout probability. Default is 0.1.

    Derived:
        d_k (int): Dimension per head (key/query size).
            d_k = d_model // num_heads
            Example (Transformer base): 512 // 8 = 64
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float=0.1):
        super().__init__()

        # Ensure d_model is divisible by num_heads
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head

        # Linear projections for Q, K, V (initialized via _reset_parameters)
        # The nn.Linear layer will randomly initialize its weight matrix (in_features x out_features) and bias vector (out_features)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Xavier uniform init for projections; zero bias."""
        for module in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def scaled_dot_product_attention(
            self,
            Q: torch.Tensor,
            K: torch.Tensor,
            V: torch.Tensor,
            mask: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Scaled Dot-Product Attention (Section 3.2.1)
        
        Attention(Q, K, V) = softmax(QK^T / √d_k)V
        
        Args:
            Q: Query tensor, shape (batch_size, num_heads, seq_len, d_k)
            K: Key tensor, shape (batch_size, num_heads, seq_len, d_k)
            V: Value tensor, shape (batch_size, num_heads, seq_len, d_k)
            mask: Optional mask tensor, shape (batch_size, 1, seq_len, seq_len) or (batch_size, 1, 1, seq_len)
            
        Returns:
            output: Attention output, shape (batch_size, num_heads, seq_len, d_k)
            attention_weights: Attention scores, shape (batch_size, num_heads, seq_len, seq_len)
        """
        # Calculate attention scores
        # QK^T: (batch_size, num_heads, seq_len, d_k) @ (batch_size, num_heads, d_k, seq_len)
        #     = (batch_size, num_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.d_k)

        # Apply mask (if provided)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim = -1) # (Last dim = columns); Softmax applied across each row (along columns)

        # --- Corner case: fully-masked rows produce NaN after softmax(-inf…-inf).
        #     Replace NaN with 0 so those query positions produce zero-vector output.
        attention_weights = attention_weights.nan_to_num(0.0)

        # Apply dropout
        attention_weights = self.dropout(attention_weights)

        # Multiply by values
        # (batch_size, num_heads, seq_len, seq_len) @ (batch_size, num_heads, seq_len, d_k)
        # = (batch_size, num_heads, seq_len, d_k)
        output = torch.matmul(attention_weights, V)

        return output, attention_weights
    
    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split the last dimension into (num_heads, d_k) and transpose.
        
        Args:
            x: Input tensor, shape (batch_size, seq_len, d_model)
            
        Returns:
            Reshaped tensor, shape (batch_size, num_heads, seq_len, d_k)
        """
        batch_size, seq_len, d_model = x.shape

        # Reshape: (batch_size, seq_len, d_model) → (batch_size, seq_len, num_heads, d_k)
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)

        # Transpose: (batch_size, seq_len, num_heads, d_k) → (batch_size, num_heads, seq_len, d_k)
        return x.transpose(1,2)
    
    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inverse of split_heads: combine heads back to original dimension.
        
        Args:
            x: Input tensor, shape (batch_size, num_heads, seq_len, d_k)
            
        Returns:
            Reshaped tensor, shape (batch_size, seq_len, d_model)
        """
        batch_size, num_heads, seq_len, d_k = x.shape

        # Transpose: (batch_size, num_heads, seq_len, d_k) → (batch_size, seq_len, num_heads, d_k)
        x = x.transpose(1,2)

        # Reshape: (batch_size, seq_len, num_heads, d_k) → (batch_size, seq_len, d_model)
        return x.contiguous().view(batch_size, seq_len, self.d_model)
    
    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            mask: torch.Tensor = None,
            return_attention_weights: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Apply multi-head attention.
        
        Args:
            query: Query tensor, shape (batch_size, seq_len_q, d_model)
            key: Key tensor, shape (batch_size, seq_len_k, d_model)
            value: Value tensor, shape (batch_size, seq_len_v, d_model)
            mask: Optional mask tensor

            return_attention_weights (bool): If True, return the
                        per-head attention weight tensor for interpretability.
                        Default: False.
            
        Returns:
            output: (batch, seq_len_q, d_model)
            [attention_weights]: (batch, num_heads, seq_len_q, seq_len_k)
                        Only returned when return_attention_weights=True.
        """
        # Linear projections
        Q = self.W_q(query)     # (batch_size, seq_len_q, d_model)
        K = self.W_k(key)       # (batch_size, seq_len_k, d_model)
        V = self.W_v(value)     # (batch_size, seq_len_v, d_model)

        # Split into multiple heads
        Q = self.split_heads(Q)     # (batch_size, num_heads, seq_len_q, d_k)
        K = self.split_heads(K)     # (batch_size, num_heads, seq_len_k, d_k)
        V = self.split_heads(V)     # (batch_size, num_heads, seq_len_v, d_k)

        # Apply scaled dot-product attention
        attn_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        # attn_output: (batch_size, num_heads, seq_len_q, d_k)

        # Combine heads
        attn_output = self.combine_heads(attn_output)     # (batch_size, seq_len_q, d_model)

        # Final linear projection
        output = self.W_o(attn_output)      # (batch_size, seq_len_q, d_model)

        if return_attention_weights:
            return output, attention_weights
        return output