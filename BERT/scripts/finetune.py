"""
Fine-tuning entrypoint for BERT single-sentence classification (Devlin et al. 2019, §4.1).

Loads a pre-trained encoder, discards the MLM/NSP heads, attaches a fresh
classification head, and fine-tunes end-to-end on a HuggingFace text-classification
dataset (default: ai4bharat/indic_glue "sna.bn", Bengali news-topic).

Usage (from repo root):
    python -m BERT.scripts.finetune
    python -m BERT.scripts.finetune --config BERT/configs/finetune.yaml
"""
import argparse
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from common.run_utils import get_device, get_git_hash, setup_logging, sha256_file
from ..models.bert_for_classification import BERTForSequenceClassification
from ..utils.config import Config
from ..utils.data_utils import load_tokenizer
from ..utils.finetune_config import FinetuneConfig
from ..utils.finetune_data import create_finetune_dataloaders
from ..utils.finetune_utils import train
from ..utils.optimizer import build_optimizer
from ..utils.train_utils import set_seed
from ._common import load_checkpoint

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune BERT for classification")
    default_config = Path(__file__).parent.parent / "configs" / "finetune.yaml"
    parser.add_argument("--config", type=str, default=str(default_config))
    args = parser.parse_args()

    config = FinetuneConfig.from_yaml(args.config)
    seed = config.seed

    # --- Per-run output dirs (run_<ts>/ under checkpoint_dir and log_dir) ---
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_subdir = f"run_{timestamp}"
    run_checkpoint_dir = os.path.join(config.paths.checkpoint_dir, run_subdir)
    run_log_dir = os.path.join(config.paths.log_dir, run_subdir)
    os.makedirs(run_checkpoint_dir, exist_ok=True)
    shutil.copy(args.config, os.path.join(run_checkpoint_dir, "config.yaml"))

    setup_logging(run_log_dir)

    try:
        git_hash = get_git_hash()
        logger.info(f"Git commit: {git_hash}")

        set_seed(seed)
        device = get_device(config.device)
        logger.info(f"Using device: {device}")

        # --- Pre-trained checkpoint + its architecture snapshot ---
        if not os.path.exists(config.pretrained.checkpoint):
            raise FileNotFoundError(f"Pre-trained checkpoint not found: {config.pretrained.checkpoint}")
        if not os.path.exists(config.pretrained.config):
            raise FileNotFoundError(
                f"Pre-trained config snapshot not found: {config.pretrained.config}. "
                f"Point pretrained.config at the run dir's config.yaml (rebuilds the encoder)."
            )
        pretrained_cfg = Config.from_yaml(config.pretrained.config)

        # Fine-tune seqs can't outrun the encoder's learned position table.
        if config.training.max_seq_len > pretrained_cfg.model.max_position_embeddings:
            raise ValueError(
                f"max_seq_len ({config.training.max_seq_len}) exceeds the pre-trained "
                f"max_position_embeddings ({pretrained_cfg.model.max_position_embeddings})."
            )

        # --- Tokenizer (MUST be the one the encoder was pre-trained with) ---
        vocab_path = os.path.join(config.paths.tokenizer_dir, "vocab.txt")
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Tokenizer not found at {vocab_path}")
        tokenizer = load_tokenizer(vocab_path)
        tokenizer_sha256 = sha256_file(vocab_path)
        vocab_size = tokenizer.get_vocab_size()
        logger.info(f"Tokenizer sha256: {tokenizer_sha256[:12]}... (vocab {vocab_size})")

        # --- Load checkpoint + verify tokenizer integrity ---
        # The encoder's embedding rows are indexed by THIS vocab; a different
        # tokenizer would silently misalign every token id.
        checkpoint = load_checkpoint(config.pretrained.checkpoint, device)
        ckpt_tok_hash = checkpoint.get("tokenizer_sha256")
        if ckpt_tok_hash in (None, "unknown"):
            logger.warning("Checkpoint has no tokenizer_sha256 — cannot verify tokenizer. Continuing.")
        elif ckpt_tok_hash != tokenizer_sha256:
            raise RuntimeError(
                f"Tokenizer mismatch:\n"
                f"  checkpoint expects sha256={ckpt_tok_hash}\n"
                f"  current   tokenizer  sha256={tokenizer_sha256}\n"
                f"The pre-trained embeddings are indexed by a different vocab. "
                f"Use the tokenizer that produced this checkpoint."
            )

        # --- Data (num_labels emerges from the dataset) ---
        logger.info("Loading dataset...")
        train_loader, val_loader, num_labels = create_finetune_dataloaders(config, tokenizer)
        logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

        # --- Model: encoder dims from the snapshot + a fresh num_labels head ---
        logger.info("Building model...")
        model = BERTForSequenceClassification(
            vocab_size=vocab_size,
            d_model=pretrained_cfg.model.d_model,
            num_heads=pretrained_cfg.model.num_heads,
            d_ff=pretrained_cfg.model.d_ff,
            num_layers=pretrained_cfg.model.num_layers,
            num_labels=num_labels,
            max_position_embeddings=pretrained_cfg.model.max_position_embeddings,
            num_segments=pretrained_cfg.model.num_segments,
            pad_idx=pretrained_cfg.tokens.pad_idx,
            dropout=pretrained_cfg.model.dropout,
        ).to(device)

        # --- Transplant the pre-trained encoder body ---
        # Keep only `bert.*` (embeddings + encoder + pooler); drop the MLM/NSP heads;
        # the fresh classifier keeps its init. strict=False → missing classifier.*
        # keys don't raise.
        body_state = {k: v for k, v in checkpoint["model_state_dict"].items() if k.startswith("bert.")}
        dropped = len(checkpoint["model_state_dict"]) - len(body_state)   # MLM/NSP head keys the filter removed
        missing, _ = model.load_state_dict(body_state, strict=False)      # unexpected is always empty — we pre-filter
        logger.info(f"Loaded pre-trained encoder: {len(body_state)} tensors | "
                    f"new (untrained): {missing} | head keys dropped: {dropped}")

        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {total_params:,} total ({num_labels}-way head)")

        # --- Loss / optimizer (warmup as a fraction of total steps) ---
        criterion = nn.CrossEntropyLoss()
        total_steps = len(train_loader) * config.training.num_epochs
        warmup_steps = int(config.training.warmup_ratio * total_steps)
        optimizer, scheduler = build_optimizer(
            model=model, total_steps=total_steps,
            lr=config.optimizer.lr, betas=config.optimizer.betas,
            eps=config.optimizer.eps, weight_decay=config.optimizer.weight_decay,
            warmup_steps=warmup_steps,
        )

        writer = SummaryWriter(log_dir=run_log_dir)
        try:
            train(
                model=model, train_loader=train_loader, val_loader=val_loader,
                criterion=criterion, optimizer=optimizer, scheduler=scheduler,
                clip_grad_norm=config.training.clip_grad_norm, device=device,
                num_epochs=config.training.num_epochs, checkpoint_dir=run_checkpoint_dir,
                seed=seed, git_hash=git_hash, tokenizer_sha256=tokenizer_sha256,
                pretrained_checkpoint=config.pretrained.checkpoint, writer=writer,
            )
        finally:
            writer.close()
    except Exception:
        logger.exception("Fine-tuning failed with unhandled exception")
        raise


if __name__ == "__main__":
    main()