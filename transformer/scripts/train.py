"""
Training script for the Transformer model.

Usage:
    python -m transformer.scripts.train                          # uses base.yaml
    python -m transformer.scripts.train --config configs/tiny.yaml
    python -m transformer.scripts.train --resume checkpoints/tiny/run_<ts>/last.pt
"""
import argparse
import hashlib
import logging
import os
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv

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
    """
    Return current git commit hash for run provenance, or 'unknown' if not in a repo.

    Appends '-dirty' if the working tree has uncommitted changes — otherwise
    a clean-looking hash would lie about which code actually produced the run.
    """
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        # `git status --porcelain` prints one line per modified/untracked file;
        # empty output ⇒ clean working tree.
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode().strip()
        return f"{commit}-dirty" if dirty else commit
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def sha256_file(path: str) -> str:
    """SHA-256 of a file's bytes — used to pin the tokenizer to a checkpoint."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_data_fingerprint(dataset, dataset_name: str) -> str:
    """
    Deterministic fingerprint of a HuggingFace dataset slice — pins training
    data identity to a checkpoint.

    Hashes (dataset name, length, content of first / middle / last rows). Any
    change to shuffle, max_rows, seed, or the underlying dataset version flips
    at least one of these, so a stale resume against re-sliced data fails fast.
    Sampling 3 rows is cheap and catches the common "I changed the slice"
    mistake without iterating millions of pairs.
    """
    import json
    h = hashlib.sha256()
    h.update(dataset_name.encode())
    h.update(str(len(dataset)).encode())
    indices = [0, len(dataset) // 2, len(dataset) - 1]
    for i in indices:
        row = dataset[i]
        h.update(json.dumps(row, sort_keys=True, ensure_ascii=False).encode())
    return h.hexdigest()


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
            # sorted() so warnings appear in stable, alphabetical order across runs —
            # plain set iteration is hash-randomized.
            for k in sorted(set(a) | set(b)):
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
    # Load .env (e.g. HF_TOKEN for gated datasets) — kept inside main() so
    # importing this module doesn't trigger filesystem reads as a side effect.
    load_dotenv()

    # --- CLI args ---
    parser = argparse.ArgumentParser(description="Train the Transformer model")
    # Default resolved relative to this file, not CWD — so `python -m ...` works from anywhere.
    default_config = Path(__file__).parent.parent / "configs" / "base.yaml"
    parser.add_argument("--config", type=str, default=str(default_config),
                        help="Path to config YAML file")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from (e.g. checkpoints/base/run_<ts>/last.pt)")
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

        # --- Tokenizer path (resolved up front so preflight + tokenizer block share it) ---
        tokenizer_path = config.paths.tokenizer_path                 # e.g. "transformer/tokenizer/base/sp.model"
        # SentencePiece always writes {prefix}.model, so tokenizer_path must end in .model
        # — otherwise os.path.exists below misses it and we retrain every run.
        assert tokenizer_path.endswith(".model"), f"tokenizer_path must end in .model, got {tokenizer_path}"
        model_prefix = tokenizer_path.removesuffix(".model")         # e.g. "transformer/tokenizer/base/sp"

        # --- Load dataset (reused by tokenizer + dataloaders + resume preflight) ---
        logger.info("Loading dataset...")
        raw_dataset = load_dataset(
            path=config.data.dataset,
            name=config.data.tgt_lang,
            split="train"
        )

        # Cap dataset size — e.g. 500K out of 8.5M pairs for practical training time on M1.
        # Shuffle before slicing: Samanantar concatenates per-source corpora, so the first N rows
        # would be one domain (PMIndia, Wikipedia, ...). Seeded shuffle keeps the slice
        # representative AND reproducible — same seed → same N rows.
        if config.data.max_rows is not None:
            raw_dataset = raw_dataset.shuffle(seed=seed).select(
                range(min(config.data.max_rows, len(raw_dataset)))
            )
        logger.info(f"Dataset size: {len(raw_dataset)} pairs")

        # Fingerprint the resolved data slice. Pinned to the checkpoint so a resume
        # against a shifted slice (shuffle added, max_rows changed, dataset re-uploaded)
        # fails fast — tokenizer hash alone can't catch this.
        data_fingerprint = compute_data_fingerprint(raw_dataset, config.data.dataset)
        logger.info(f"Data fingerprint: {data_fingerprint[:12]}...")

        # --- Resume preflight (fail fast on tokenizer / data / config mismatch) ---
        # All resume validation lives here, BEFORE tokenizer-train / dataloader-build /
        # model-alloc. Mismatches surface in seconds (after dataset load, which is cached
        # after the first run anyway).
        checkpoint = None         # populated if resuming; reused in restore block below
        tokenizer_sha256 = None   # hashed here on resume, after train_tokenizer otherwise
        if args.resume:
            if not os.path.exists(args.resume):
                raise FileNotFoundError(f"Checkpoint not found: {args.resume}")
            if not os.path.exists(tokenizer_path):
                # A fresh tokenizer would have different vocab IDs and silently
                # mismatch the saved embeddings.
                raise FileNotFoundError(f"Resume requires existing tokenizer at {tokenizer_path}")

            logger.info(f"Resuming from {args.resume}...")

            tokenizer_sha256 = sha256_file(tokenizer_path)
            checkpoint = load_checkpoint(args.resume, device)

            # Tokenizer integrity — same path may now hold a retrained tokenizer
            # with different vocab IDs. Pre-hash checkpoints get a warning, not an error.
            ckpt_tok_hash = checkpoint.get("tokenizer_sha256")
            if ckpt_tok_hash is None:
                logger.warning(
                    "Checkpoint has no tokenizer_sha256 (pre-hash run) — "
                    "cannot verify tokenizer integrity. Continuing."
                )
            elif ckpt_tok_hash != tokenizer_sha256:
                raise RuntimeError(
                    f"Tokenizer mismatch on resume:\n"
                    f"  checkpoint expects sha256={ckpt_tok_hash}\n"
                    f"  current   tokenizer  sha256={tokenizer_sha256}\n"
                    f"  path: {tokenizer_path}\n"
                    f"Embeddings would be loaded against a different vocab. "
                    f"Restore the original tokenizer or train a fresh model."
                )

            # Data continuity — same tokenizer file is fine against any data, so we
            # also pin the resolved data slice. Catches shuffle/max_rows/dataset drift.
            ckpt_data_fp = checkpoint.get("data_fingerprint")
            if ckpt_data_fp is None:
                logger.warning(
                    "Checkpoint has no data_fingerprint (pre-fingerprint run) — "
                    "cannot verify data continuity. Continuing."
                )
            elif ckpt_data_fp != data_fingerprint:
                raise RuntimeError(
                    f"Data slice mismatch on resume:\n"
                    f"  checkpoint trained against fingerprint={ckpt_data_fp}\n"
                    f"  current  data slice    fingerprint={data_fingerprint}\n"
                    f"The data slice has changed since this checkpoint was saved "
                    f"(shuffle, max_rows, seed, or upstream dataset version differs). "
                    f"Resuming would fine-tune on a different distribution. "
                    f"Either revert your data-selection code, or train fresh without --resume."
                )

            # Snapshot lives next to the checkpoint being resumed (in its run_<ts>/ dir),
            # not at the parent — each run owns its own config.yaml.
            snapshot_path = os.path.join(os.path.dirname(args.resume), "config.yaml")
            warn_if_config_diverges(snapshot_path, config)

        # --- Tokenizer (train if missing, else load) ---
        # Resume case is filtered out by the preflight above — only fresh runs hit train_tokenizer.
        if os.path.exists(tokenizer_path):
            logger.info("Loading existing tokenizer...")
            sp = load_tokenizer(tokenizer_path)
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

        # Hash the tokenizer file so future runs can verify resumes against it.
        # Skipped on resume — preflight already computed it above.
        if tokenizer_sha256 is None:
            tokenizer_sha256 = sha256_file(tokenizer_path)
        logger.info(f"Tokenizer sha256: {tokenizer_sha256[:12]}...")

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

        # --- Resume from checkpoint (state restoration) ---
        # Validation + checkpoint loading happened in the preflight above.
        # Here we just restore the actual training state into the freshly built
        # model/optimizer/scheduler.
        start_epoch = 1
        best_val_loss = float("inf")
        if args.resume:
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
                tokenizer_sha256=tokenizer_sha256,
                data_fingerprint=data_fingerprint,
                writer=writer,
            )
        finally:
            writer.close()
    except Exception:
        logger.exception("Training failed with unhandled exception")
        raise


if __name__ == "__main__":
    main()