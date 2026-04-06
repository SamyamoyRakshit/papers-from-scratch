import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

class TransformerScheduler:
    """
    Transformer Learning Rate Schedule (Section 5.3)

    lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))

    Two phases:
        1. Warmup (steps 1 to warmup_steps): lr increases linearly
        2. Decay  (steps > warmup_steps):    lr decreases as step^(-0.5)

    Args:
        d_model (int): Model dimension. Controls the peak learning rate.
            Larger model → smaller peak lr.
            Example (base): 512
        warmup_steps (int): Number of warmup steps. Default is 4000.
    """
    def __init__(self, d_model: int, warmup_steps: int = 4000):
        self.d_model = d_model
        self.warmup_steps = warmup_steps

    def __call__(self, step: int) -> float:
        # Avoid division by zero at step 0
        step = max(step, 1)
        return self.d_model ** (-0.5) * min(step ** (-0.5), step * self.warmup_steps ** (-1.5))
    

def build_optimizer(model: nn.Module, d_model: int, warmup_steps: int = 4000) -> tuple:
    """
    Build Adam optimizer with Transformer learning rate schedule (Section 5.3).

    Adam hyperparameters from the paper:
        β₁ = 0.9, β₂ = 0.98, ε = 10⁻⁹

    Args:
        model (nn.Module): The transformer model.
        d_model (int): Model dimension.
        warmup_steps (int): Number of warmup steps. Default is 4000.

    Returns:
        tuple: (optimizer, scheduler)
    """
    optimizer = Adam(
        params=model.parameters(),
        lr=1.0,             # placeholder — scheduler controls actual lr
        betas=(0.9, 0.98),
        eps=1e-9
    )

    scheduler = LambdaLR(optimizer, lr_lambda=TransformerScheduler(d_model, warmup_steps))

    return optimizer, scheduler