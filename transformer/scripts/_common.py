"""
Shared builders for transformer scripts (train, evaluate, inference).

Only factor out what's genuinely reused across scripts — anything single-use
stays inline in its caller.
"""
import torch
import torch.nn as nn

from ..models.transformer import Transformer
from ..utils.config import Config


def build_model(config: Config, vocab_size: int, device: torch.device) -> nn.Module:
    """Construct Transformer with shared src/tgt vocabulary (Section 3.4)."""
    return Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=config.model.d_model,
        num_heads=config.model.num_heads,
        d_ff=config.model.d_ff,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout,
        max_len=config.model.max_len,
    ).to(device)


def load_checkpoint(path: str, device: torch.device) -> dict:
    """
    Load a checkpoint dict from disk.

    weights_only=True   -> safe-load mode; blocks arbitrary pickle code execution.
    map_location=device -> remap tensors to `device` regardless of where they
                           were saved (mps/cuda/cpu) — needed when moving
                           checkpoints across machines.
    """
    return torch.load(path, map_location=device, weights_only=True)
