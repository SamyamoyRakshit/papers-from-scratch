"""
Shared builders for BERT scripts (pretrain, finetune, evaluate, inference).

Only factor out what's genuinely reused across scripts — anything single-use
stays inline in its caller.
"""
import torch
import torch.nn as nn

from ..models.bert_for_pretraining import BERTForPreTraining
from ..utils.config import Config


def build_model(config: Config, vocab_size: int, device: torch.device) -> nn.Module:
    """Construct BERTForPreTraining from config (Devlin et al. 2019, §3.1)."""
    return BERTForPreTraining(
        vocab_size=vocab_size,
        d_model=config.model.d_model,
        num_heads=config.model.num_heads,
        d_ff=config.model.d_ff,
        num_layers=config.model.num_layers,
        max_position_embeddings=config.model.max_position_embeddings,
        num_segments=config.model.num_segments,
        pad_idx=config.tokens.pad_idx,
        dropout=config.model.dropout,
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
