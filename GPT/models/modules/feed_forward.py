import torch
import torch.nn as nn

# gelu is imported from BERT/ unchanged: OpenAI's GPT-2 code (model.py, `gelu`)
# uses the same tanh approximation. Run from repo root so `BERT` resolves.
from BERT.models.modules.feed_forward import gelu


class FeedForward(nn.Module):
    """
    GPT-2 MLP (GPT-1 §4.1; GPT-2 §2.3 "largely follows" GPT-1)

        MLP(x) = Dropout( GELU(x·W_fc + b_fc)·W_proj + b_proj )

    GPT-1 §4.1: "For the activation function, we used the Gaussian Error
    Linear Unit (GELU)." Names c_fc / c_proj match OpenAI's checkpoint.

    vs BERT/'s FeedForward: same two linears, same gelu. Only the dropout moves:
        BERT/:  linear1 → gelu → DROPOUT → linear2
        GPT:    c_fc    → gelu → c_proj  → DROPOUT
    GPT-1 §4.1's "residual dropout" (0.1) is dropout on a sub-layer's output
    before it is added back to x (Vaswani §5.4). The `x + ...` itself is in decoder.py.

    Init is NOT done here: gpt.py scales c_proj by 1/√(2·num_layers) (GPT-2 §2.3),
    because c_proj writes into the residual stream, 2 writes per block.

    Args:
        d_model (int): base.yaml 256.
        d_ff (int): base.yaml 1024 (= 4·d_model).
            c_fc:   256×1024 + 1024 = 263,168 params
            c_proj: 1024×256 + 256  = 262,400 params  → 525,568 per block
        dropout (float): GPT-1 §4.1: 0.1 residual dropout.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.c_fc = nn.Linear(d_model, d_ff)      # expand:   256 → 1024
        self.c_proj = nn.Linear(d_ff, d_model)    # contract: 1024 → 256
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, T, d_model), output of ln2 inside a decoder block
        Returns:
            (batch_size, T, d_model)
        """
        x = self.c_fc(x)          # (B, T, d_ff)     each token: 256 → 1024
        x = gelu(x)               # same shape, element-wise
        x = self.c_proj(x)        # (B, T, d_model)  each token: 1024 → 256
        return self.dropout(x)    # residual dropout, then decoder.py adds it to x
