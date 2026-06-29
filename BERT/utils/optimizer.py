import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


class LinearWarmupScheduler:
    """
    BERT learning rate schedule (Appendix A.2).

    Linear warmup, then linear decay to zero:

        warmup (step < warmup_steps):  factor = step / warmup_steps
        decay  (step ≥ warmup_steps):  factor = (total_steps - step) / (total_steps - warmup_steps)

    The factor multiplies AdamW's base lr (the peak lr from config), so it
    stays in [0, 1], hits 1.0 right at the end of warmup, and clamps at 0 if
    training runs past total_steps. Unlike the transformer's Noam schedule,
    there's no d_model^(-0.5) term — the peak lr is set directly.

    Args:
        warmup_steps (int): steps to linearly ramp lr up. Paper: 10,000.
        total_steps (int):  num_epochs * steps_per_epoch (known once the
            dataloader exists, so it's passed in by pretrain.py).
    """
    def __init__(self, warmup_steps: int, total_steps: int):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def __call__(self, step: int) -> float:
        if step < self.warmup_steps:
            return step / max(1, self.warmup_steps)
        return max(0.0, (self.total_steps - step) / max(1, self.total_steps - self.warmup_steps))
    

def _decay_groups(model: nn.Module, weight_decay: float) -> list:
    """
    Split params into decay / no-decay groups. Bias and LayerNorm weights
    (any 1-D tensor) are excluded from weight decay — the canonical BERT
    setup; decaying those hurts. Everything else (the 2-D weight matrices)
    keeps weight_decay.
    """
    decay, no_decay = [], []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        (no_decay if param.ndim < 2 else decay).append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_optimizer(
    model: nn.Module,
    total_steps: int,
    lr: float = 1e-4,
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-6,
    weight_decay: float = 0.01,
    warmup_steps: int = 10000,
) -> tuple:
    """
    Build AdamW with the BERT warmup→linear-decay schedule (Appendix A.2).

    AdamW hyperparameters — lr/β/decay from the paper (A.2), ε from the
    released code (optimization.py; paper omits it, PyTorch default is 1e-8):
        β₁ = 0.9, β₂ = 0.999, ε = 10⁻⁶, weight decay = 0.01

    Args:
        model (nn.Module): The BERT model.
        total_steps (int): Total training steps (for the decay phase).
        lr (float): Peak learning rate. Paper: 1e-4.
        betas (tuple): AdamW (β₁, β₂). Paper: (0.9, 0.999).
        eps (float): AdamW ε. Released code: 1e-6 (not in paper).
        weight_decay (float): Decoupled weight decay. Paper: 0.01.
        warmup_steps (int): Linear warmup steps. Paper: 10,000.

    Returns:
        tuple: (optimizer, scheduler)
    """
    optimizer = AdamW(
        params=_decay_groups(model, weight_decay),
        lr=lr,              # peak lr — scheduler scales it by a factor in [0, 1]
        betas=tuple(betas),
        eps=eps,
    )

    scheduler = LambdaLR(optimizer, lr_lambda=LinearWarmupScheduler(warmup_steps, total_steps))

    return optimizer, scheduler