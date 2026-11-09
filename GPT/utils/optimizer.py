import math

import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from BERT.utils.optimizer import _decay_groups   # 1-D params (bias, LayerNorm) get no decay — imported unchanged


class CosineWarmupScheduler:
    """
    GPT-3 learning rate schedule (Appendix B). GPT-2 does not state its own.

    Linear warmup, then half a cosine from 1.0 down to min_lr_ratio:

        warmup (step < warmup_steps):  factor = step / warmup_steps
        decay  (step ≥ warmup_steps):  p      = (step - warmup_steps) / (total_steps - warmup_steps)
                                       factor = min_lr_ratio + (1 - min_lr_ratio) · ½(1 + cos(π·p))

    The factor multiplies AdamW's base lr (the peak lr from config). Past
    total_steps it stays at min_lr_ratio — GPT-3 §B: "training continues at 10%
    of the original learning rate". Never 0: BERT's linear-to-0 schedule left
    --resume training at lr 0.

    Example, peak lr 6e-4, min_lr_ratio 0.1, warmup 1,000, total 10,000:
        step 0 → 0,  500 → 3e-4,  1,000 → 6e-4 (peak),  5,500 → 3.3e-4,  10,000+ → 6e-5

    Args:
        warmup_steps (int): steps to ramp lr up linearly from 0.
        total_steps (int):  step where the cosine reaches min_lr_ratio.
        min_lr_ratio (float): floor, as a fraction of the peak lr. GPT-3: 0.1.
    """
    def __init__(self, warmup_steps: int, total_steps: int, min_lr_ratio: float):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio

    def __call__(self, step: int) -> float:
        if step < self.warmup_steps:
            return step / max(1, self.warmup_steps)
        progress = min(1.0, (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def build_optimizer(
    model: nn.Module,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
    lr: float,
    betas: list,
    eps: float,
    weight_decay: float,
) -> tuple:
    """
    Build AdamW with the GPT-3 warmup → cosine schedule (Appendix B).

    No defaults: every value comes from the YAML (lr/betas/eps/weight_decay from
    `optimizer:`, the three schedule values from `training:`).

    Weight decay goes on every 2-D weight, including wte (= the tied lm_head) and
    wpe; biases and LayerNorm gains get none. That is GPT-1 §4.1's rule ("w = 0.01
    on all non bias or gain weights") and exactly BERT's _decay_groups split.

    Args:
        model (nn.Module): The GPT model.
        total_steps (int): Steps until lr reaches its floor.
        warmup_steps (int): Linear warmup steps.
        min_lr_ratio (float): lr floor as a fraction of peak. GPT-3: 0.1.
        lr (float): Peak learning rate. GPT-3 (125M): 6e-4.
        betas (list): AdamW (β₁, β₂). GPT-3: [0.9, 0.95].
        eps (float): AdamW ε. GPT-3: 1e-8.
        weight_decay (float): Decoupled weight decay. GPT-3: 0.1.

    Returns:
        tuple: (optimizer, scheduler)
    """
    optimizer = AdamW(
        params=_decay_groups(model, weight_decay),
        lr=lr,              # peak lr — scheduler scales it by a factor in [min_lr_ratio, 1]
        betas=tuple(betas),
        eps=eps,
    )

    scheduler = LambdaLR(optimizer, lr_lambda=CosineWarmupScheduler(warmup_steps, total_steps, min_lr_ratio))

    return optimizer, scheduler
