"""
Training script for the Transformer model.

Usage:
    python -m transformer.scripts.train                          # uses base.yaml
    python -m transformer.scripts.train --config configs/tiny.yaml
    python -m transformer.scripts.train --resume checkpoints/last.pt
"""
from dotenv import load_dotenv
load_dotenv()

import argparse
import logging
import os
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

import torch
from datasets import load_dataset
from torch.utils.tensorboard import SummaryWriter

from ..utils.config import Config
from ..utils.data_utils import (
    train_tokenizer,
    load_tokenizer,
    create_dataloaders
)
from ..utils.logging_setup import setup_logging
from ..utils.loss import LabelSmoothedLoss
from ..utils.optimizer import build_optimizer
from ..utils.train_utils import train, set_seed
from ._common import build_model, load_checkpoint

logger = logging.getLogger(__name__)


def get_git_hash() -> str:
    """Return current git commit hash for run provenance, or 'unknown' if not in a repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def get_device(device_config: str) -> torch.device:
    """Resolve device string from config to torch.device."""
    if device_config == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    return torch.device(device_config)


# Keys allowed to change between the snapshotted config and the resumed config.
# Anything outside this set means the resumed run is a different experiment than
# the one that produced the checkpoint — we warn but don't block.
_RESUME_SAFE_KEYS = {"training.num_epochs", "device", "paths.log_dir"}


def warn_if_config_diverges(snapshot_path: str, current: Config) -> None:
    """Print warnings for fields that changed since the checkpoint was written."""
    if not os.path.exists(snapshot_path):
        return  # nothing to compare against — silently skip
    snapshot = Config.from_yaml(snapshot_path).model_dump()
    risky: list[str] = []

    def walk(a, b, prefix: str = "") -> None:
        if isinstance(a, dict) and isinstance(b, dict):
            for k in set(a) | set(b):
                walk(a.get(k), b.get(k), f"{prefix}.{k}" if prefix else k)
        elif a != b and prefix not in _RESUME_SAFE_KEYS:
            risky.append(f"  {prefix}: {a!r} -> {b!r}")

    walk(snapshot, current.model_dump())
    if risky:
        logger.warning("resumed config differs from checkpoint snapshot:")
        for r in risky:
            logger.warning(r)
        logger.warning("  (continuing — trajectory may differ from the original run)")


def main():
    # --- CLI args ---
    parser = argparse.ArgumentParser(description="Train the Transformer model")
    # Default resolved relative to this file, not CWD — so `python -m ...` works from anywhere.
    default_config = Path(__file__).parent.parent / "configs" / "base.yaml"
    parser.add_argument("--config", type=str, default=str(default_config),
                        help="Path to config YAML file")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from (e.g. checkpoints/base/last.pt)")
    args = parser.parse_args()

    # --- Load config ---
    # Loaded before logging is set up so we know where to put the log file.
    config = Config.from_yaml(args.config)

    # Values reused across multiple call sites — extract once
    seed = config.seed
    pad_idx = config.tokens.pad_idx

    # --- Per-run output dirs ---
    # Each invocation writes to its own run_<timestamp>/ subdir under both
    # checkpoint_dir and log_dir. Resuming creates a new run dir, so prior
    # runs' best.pt / config.yaml / train.log are never overwritten.
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_subdir = f"run_{timestamp}"
    run_checkpoint_dir = os.path.join(config.paths.checkpoint_dir, run_subdir)
    run_log_dir = os.path.join(config.paths.log_dir, run_subdir)
    os.makedirs(run_checkpoint_dir, exist_ok=True)

    # Snapshot config inside the run dir — a year from now this run is still
    # reproducible even if the source YAML has changed or been deleted.
    shutil.copy(args.config, os.path.join(run_checkpoint_dir, "config.yaml"))

    # --- Logging ---
    # Set up first so every subsequent step also lands in train.log on disk.
    # Same timestamped subdir as the checkpoints so logs + tfevents + weights
    # live together per run.
    setup_logging(run_log_dir)

    # Wrap the run body so any unhandled exception lands in train.log with a
    # full traceback (Python's default excepthook only writes to stderr).
    # `except Exception` deliberately skips KeyboardInterrupt so Ctrl+C stays clean.
    try:
        # --- Run provenance ---
        # Logged + saved in checkpoint so a year from now you know exactly which code produced this run.
        git_hash = get_git_hash()
        logger.info(f"Git commit: {git_hash}")

        # --- Seed ---
        set_seed(seed)

        # --- Device ---
        device = get_device(config.device)
        logger.info(f"Using device: {device}")

        # --- Load dataset (reused by tokenizer + dataloaders) ---
        logger.info("Loading dataset...")
        raw_dataset = load_dataset(
            path=config.data.dataset,
            name=config.data.tgt_lang,
            split="train"
        )

        # Cap dataset size — e.g. 500K out of 8.5M pairs for practical training time on M1
        if config.data.max_rows is not None:
            raw_dataset = raw_dataset.select(
                range(min(config.data.max_rows, len(raw_dataset)))
            )
        logger.info(f"Dataset size: {len(raw_dataset)} pairs")

        # --- Tokenizer (train if missing, else load) ---
        tokenizer_path = config.paths.tokenizer_path                 # e.g. "transformer/tokenizer/base/sp.model"
        model_prefix = tokenizer_path.removesuffix(".model")         # e.g. "transformer/tokenizer/base/sp"
        if os.path.exists(tokenizer_path):
            logger.info("Loading existing tokenizer...")
            sp = load_tokenizer(tokenizer_path)
        elif args.resume:
            # Resume must use the original tokenizer — a fresh one would have
            # different vocab IDs, silently mismatching the saved embeddings.
            raise FileNotFoundError(
                f"Resume requires existing tokenizer at {tokenizer_path}"
            )
        else:
            logger.info("Training tokenizer...")
            sp = train_tokenizer(
                dataset=raw_dataset,
                vocab_size=config.data.vocab_size,
                pad_id=pad_idx,
                sos_id=config.tokens.sos_idx,
                eos_id=config.tokens.eos_idx,
                unk_id=config.tokens.unk_idx,
                model_prefix=model_prefix
            )
        logger.info(f"Vocabulary size: {sp.vocab_size()}")

        # --- DataLoaders ---
        logger.info("Creating dataloaders...")
        train_loader, val_loader = create_dataloaders(
            raw_dataset=raw_dataset,
            sp=sp,
            max_seq_len=config.training.max_seq_len,
            max_tokens=config.training.max_tokens_per_batch,
            pad_idx=pad_idx,
            seed=seed,
            num_workers=config.data.num_workers,
            val_split=config.training.val_split,
            filter_max_ratio=config.data.filter_max_ratio,
            filter_min_words=config.data.filter_min_words,
        )
        logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

        # --- Model ---
        logger.info("Building model...")
        vocab_size = sp.vocab_size()
        model = build_model(config, vocab_size, device)

        # p.numel() = number of elements in tensor p (e.g. Linear(256, 1024) -> 262,144 + 1,024 bias)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

        # --- Loss ---
        criterion = LabelSmoothedLoss(
            pad_idx=pad_idx,
            smoothing=config.training.label_smoothing,
        )

        # --- Optimizer + Scheduler ---
        optimizer, scheduler = build_optimizer(
            model=model,
            d_model=config.model.d_model,
            betas=config.optimizer.betas,
            eps=config.optimizer.eps,
            warmup_steps=config.training.warmup_steps,
        )

        # --- Resume from checkpoint ---
        # Defaults for a fresh run; overwritten below if --resume is passed.
        start_epoch = 1
        best_val_loss = float("inf")
        if args.resume:
            if not os.path.exists(args.resume):
                raise FileNotFoundError(f"Checkpoint not found: {args.resume}")
            logger.info(f"Resuming from {args.resume}...")

            # Snapshot lives next to the checkpoint being resumed (in its run_<ts>/ dir),
            # not at the parent — each run owns its own config.yaml.
            snapshot_path = os.path.join(os.path.dirname(args.resume), "config.yaml")
            warn_if_config_diverges(snapshot_path, config)

            checkpoint = load_checkpoint(args.resume, device)

            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            # Restore only state the next epoch DEPENDS on:
            #   start_epoch    -> so we don't redo finished epochs
            #   best_val_loss  -> so the first resumed epoch doesn't trivially overwrite best.pt
            # train_loss / val_loss are recomputed each epoch — only printed below for context.

            # NOTE: RNG state (torch/numpy/random) is NOT saved — would require weights_only=False
            # on load (arbitrary pickle execution). Instead, train() re-seeds with (seed + epoch)
            # at the start of each epoch, so uninterrupted and resumed runs are bit-identical
            # at epoch boundaries (mid-epoch crashes still diverge until the next epoch).
            start_epoch = checkpoint["epoch"] + 1
            best_val_loss = checkpoint["best_val_loss"]
            logger.info(f"Resumed from epoch {checkpoint['epoch']} "
                  f"(train_loss: {checkpoint['train_loss']:.4f}, val_loss: {checkpoint['val_loss']:.4f})")

            if start_epoch > config.training.num_epochs:
                raise ValueError(
                    f"Checkpoint already at epoch {checkpoint['epoch']} but config has "
                    f"num_epochs={config.training.num_epochs}. Increase num_epochs to continue training."
                )

        # --- TensorBoard ---
        # Reuses run_log_dir from the logging setup above so train.log and
        # the tfevents file land in the same per-run directory.
        writer = SummaryWriter(log_dir=run_log_dir)

        # --- Train ---
        try:
            train(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                pad_idx=pad_idx,
                clip_grad_norm=config.training.clip_grad_norm,
                device=device,
                num_epochs=config.training.num_epochs,
                checkpoint_dir=run_checkpoint_dir,
                seed=seed,
                start_epoch=start_epoch,
                best_val_loss=best_val_loss,
                git_hash=git_hash,
                writer=writer,
            )
        finally:
            writer.close()
    except Exception:
        logger.exception("Training failed with unhandled exception")
        raise


if __name__ == "__main__":
    main()