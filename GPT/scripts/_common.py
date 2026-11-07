"""
Shared builders for GPT scripts (pretrain, evaluate, generate, app).

Only factor out what's genuinely reused across scripts — anything single-use
stays inline in its caller.
"""
import torch
import torch.nn as nn

from ..models.gpt import GPT
from ..utils.config import Config


def build_model(config: Config, device: torch.device) -> nn.Module:
    """Construct GPT from config (GPT-2 §2.3). .to() moves wte in place, so the lm_head tie survives."""
    return GPT(config.data.vocab_size, **config.model.model_dump()).to(device)


def load_checkpoint(path: str, device: torch.device) -> dict:
    """
    Load a checkpoint dict from disk.

    weights_only=True   -> safe-load mode; blocks arbitrary pickle code execution.
    map_location=device -> remap tensors to `device` regardless of where they
                           were saved (mps/cuda/cpu) — needed when moving
                           checkpoints across machines.
    """
    return torch.load(path, map_location=device, weights_only=True)
