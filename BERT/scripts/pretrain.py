"""
Pre-training entrypoint for BERT (Devlin et al. 2019).

Usage:
    python -m BERT.scripts.pretrain                                  # uses base.yaml
    python -m BERT.scripts.pretrain --config BERT/configs/tiny.yaml
    python -m BERT.scripts.pretrain --resume BERT/checkpoints/base/run_<ts>/last.pt

Expects the corpus at config.data.corpus_path — run prepare_corpus.py first.
"""
import argparse
import logging
import os
import random
import shutil
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from common.run_utils import (
    get_device, get_git_hash, setup_logging, sha256_file, warn_if_config_diverges,
)
from ..utils.config import Config
from ..utils.data_utils import (
    build_documents,
    create_dataloader,
    load_tokenizer,
    train_tokenizer,
)
from ..utils.loss import BERTPreTrainingLoss
from ..utils.optimizer import build_optimizer
from ..utils.train_utils import set_seed, train
from ._common import build_model, load_checkpoint

logger = logging.getLogger(__name__)


def split_documents(all_documents, val_split: float, seed: int):
    """
    Split documents into (train, val) at the DOCUMENT level.

    Splitting by document — not by example — keeps NSP honest: a document's
    sentences never straddle the train/val boundary. Shuffled with a seeded,
    LOCAL RNG so the slice is representative AND reproducible, without disturbing
    the global RNG that masking.py / nsp.py draw from.
    """
    docs = list(all_documents)
    random.Random(seed).shuffle(docs)
    val_count = int(len(docs) * val_split)
    return docs[val_count:], docs[:val_count]   # train, val


# Keys allowed to change between the snapshotted config and the resumed config.
# Anything outside this set means the resumed run is a different experiment than
# the one that produced the checkpoint — we warn but don't block.
_RESUME_SAFE_KEYS = {"training.num_epochs", "device", "paths.log_dir"}


def main():
    # --- CLI args ---
    parser = argparse.ArgumentParser(description="Pre-train BERT")
    # Default resolved relative to this file, not CWD — so `python -m ...` works from anywhere.
    default_config = Path(__file__).parent.parent / "configs" / "base.yaml"
    parser.add_argument("--config", type=str, default=str(default_config),
                        help="Path to config YAML file")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a last.pt to resume from (e.g. checkpoints/base/run_<ts>/last.pt)")
    args = parser.parse_args()

    # --- Config (loaded before logging so we know where the log file goes) ---
    config = Config.from_yaml(args.config)
    seed = config.seed

    # --- Per-run output dirs (run_<ts>/ under checkpoint_dir and log_dir) ---
    # Each invocation owns its own dir, so a resume never overwrites a prior
    # run's best.pt / config.yaml / train.log. The leaderboard + best.pt symlink
    # that train() writes live one level up, in checkpoint_dir itself, ranking
    # all runs. (Same scheme as transformer/scripts/train.py.)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_subdir = f"run_{timestamp}"
    run_checkpoint_dir = os.path.join(config.paths.checkpoint_dir, run_subdir)
    run_log_dir = os.path.join(config.paths.log_dir, run_subdir)
    os.makedirs(run_checkpoint_dir, exist_ok=True)

    # Snapshot the config inside the run dir — reproducible even if the source
    # YAML later changes or is deleted.
    shutil.copy(args.config, os.path.join(run_checkpoint_dir, "config.yaml"))

    # --- Logging (set up first so everything below also lands in train.log) ---
    setup_logging(run_log_dir)

    # Wrap the run body so any unhandled exception lands in train.log with a full
    # traceback. `except Exception` skips KeyboardInterrupt so Ctrl+C stays clean.
    try:
        # --- Run provenance ---
        # Logged + saved in checkpoint so a year from now you know exactly which
        # code produced this run.
        git_hash = get_git_hash()
        logger.info(f"Git commit: {git_hash}")

        # --- Seed + device ---
        set_seed(seed)
        device = get_device(config.device)
        logger.info(f"Using device: {device}")

        # --- Corpus must exist (prepare_corpus.py writes it) ---
        corpus_path = config.data.corpus_path
        if not os.path.exists(corpus_path):
            raise FileNotFoundError(
                f"Corpus not found at {corpus_path}. "
                f"Run: python -m BERT.scripts.prepare_corpus --config {args.config}"
            )

        # --- Data fingerprint ---
        # BERT's corpus is a single .txt file, so we hash the file directly
        # (the transformer hashes an HF dataset slice). Pinned to the checkpoint
        # so a resume against a changed corpus fails fast.
        data_fingerprint = sha256_file(corpus_path)
        logger.info(f"Data fingerprint: {data_fingerprint[:12]}...")

        vocab_path = os.path.join(config.paths.tokenizer_dir, "vocab.txt")

        # --- Resume preflight (fail fast on tokenizer / data mismatch) ---
        # All resume validation lives here, BEFORE tokenizer-load / dataloader-build /
        # model-alloc — so mismatches surface in seconds.
        checkpoint = None         # populated if resuming; reused in restore block below
        tokenizer_sha256 = None   # hashed here on resume, after train_tokenizer otherwise
        if args.resume:
            if not os.path.exists(args.resume):
                raise FileNotFoundError(f"Checkpoint not found: {args.resume}")
            if not os.path.exists(vocab_path):
                # A fresh tokenizer would have different vocab IDs and silently
                # mismatch the saved embeddings.
                raise FileNotFoundError(f"Resume requires existing tokenizer at {vocab_path}")

            logger.info(f"Resuming from {args.resume}...")
            tokenizer_sha256 = sha256_file(vocab_path)
            checkpoint = load_checkpoint(args.resume, device)

            # Tokenizer integrity — same path may now hold a retrained tokenizer
            # with different vocab IDs. Pre-hash checkpoints get a warning, not an error.
            ckpt_tok_hash = checkpoint.get("tokenizer_sha256")
            if ckpt_tok_hash in (None, "unknown"):
                logger.warning(
                    "Checkpoint has no tokenizer_sha256 — cannot verify tokenizer "
                    "integrity. Continuing."
                )
            elif ckpt_tok_hash != tokenizer_sha256:
                raise RuntimeError(
                    f"Tokenizer mismatch on resume:\n"
                    f"  checkpoint expects sha256={ckpt_tok_hash}\n"
                    f"  current   tokenizer  sha256={tokenizer_sha256}\n"
                    f"  path: {vocab_path}\n"
                    f"Embeddings would be loaded against a different vocab. "
                    f"Restore the original tokenizer or train a fresh model."
                )

            # Data continuity — pin the corpus so shuffle/regen drift is caught.
            ckpt_data_fp = checkpoint.get("data_fingerprint")
            if ckpt_data_fp in (None, "unknown"):
                logger.warning(
                    "Checkpoint has no data_fingerprint — cannot verify data "
                    "continuity. Continuing."
                )
            elif ckpt_data_fp != data_fingerprint:
                raise RuntimeError(
                    f"Data slice mismatch on resume:\n"
                    f"  checkpoint trained against fingerprint={ckpt_data_fp}\n"
                    f"  current  corpus      fingerprint={data_fingerprint}\n"
                    f"The corpus has changed since this checkpoint was saved. "
                    f"Either revert the corpus, or train fresh without --resume."
                )

            # Snapshot lives next to the checkpoint being resumed (in its run_<ts>/ dir).
            # common's dict-diff is Config-agnostic, so we load + dump here.
            snapshot_path = os.path.join(os.path.dirname(args.resume), "config.yaml")
            if os.path.exists(snapshot_path):
                snapshot = Config.from_yaml(snapshot_path).model_dump()
                warn_if_config_diverges(snapshot, config.model_dump(), _RESUME_SAFE_KEYS)

        # --- Tokenizer (train on the corpus if missing, else load) ---
        # Resume case is filtered out by the preflight above — only fresh runs train.
        if os.path.exists(vocab_path):
            logger.info("Loading existing tokenizer...")
            tokenizer = load_tokenizer(vocab_path)
        else:
            logger.info("Training tokenizer...")
            tokenizer = train_tokenizer(
                corpus_files=corpus_path,
                vocab_size=config.data.vocab_size,
                save_dir=config.paths.tokenizer_dir,
            )

        # Hash the tokenizer so future runs can verify resumes against it.
        # Skipped on resume — preflight already computed it above.
        if tokenizer_sha256 is None:
            tokenizer_sha256 = sha256_file(vocab_path)
        logger.info(f"Tokenizer sha256: {tokenizer_sha256[:12]}...")
        vocab_size = tokenizer.get_vocab_size()
        logger.info(f"Vocabulary size: {vocab_size}")

        # --- Documents -> train/val split -> dataloaders ---
        logger.info("Building documents...")
        all_documents = build_documents(corpus_path, tokenizer)
        train_docs, val_docs = split_documents(all_documents, config.training.val_split, seed)
        logger.info(f"Documents: {len(train_docs)} train, {len(val_docs)} val")

        train_loader = create_dataloader(
            train_docs, tokenizer,
            max_seq_len=config.training.max_seq_len,
            batch_size=config.training.batch_size,
            mlm_probability=config.training.mlm_probability,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
        val_loader = create_dataloader(
            val_docs, tokenizer,
            max_seq_len=config.training.max_seq_len,
            batch_size=config.training.batch_size,
            mlm_probability=config.training.mlm_probability,
            shuffle=False,
            num_workers=config.data.num_workers,
        )
        logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

        # --- Model ---
        logger.info("Building model...")
        model = build_model(config, vocab_size, device)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

        # --- Loss ---
        criterion = BERTPreTrainingLoss()

        # --- Optimizer + scheduler ---
        # total_steps drives the linear decay; known only now that train_loader exists.
        total_steps = len(train_loader) * config.training.num_epochs
        optimizer, scheduler = build_optimizer(
            model=model,
            total_steps=total_steps,
            lr=config.optimizer.lr,
            betas=config.optimizer.betas,
            eps=config.optimizer.eps,
            weight_decay=config.optimizer.weight_decay,
            warmup_steps=config.training.warmup_steps,
        )

        # --- Resume (restore state into the freshly built model/opt/sched) ---
        # Validation happened in the preflight above; here we just restore state.
        start_epoch = 1
        best_val_loss = float("inf")
        if args.resume:
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            # Restore only what the next epoch depends on:
            #   start_epoch   -> don't redo finished epochs
            #   best_val_loss -> first resumed epoch can't trivially overwrite best.pt
            # RNG state isn't saved; train() re-seeds (seed + epoch) each epoch, so
            # uninterrupted and resumed runs match at epoch boundaries.
            start_epoch = checkpoint["epoch"] + 1
            best_val_loss = checkpoint["best_val_loss"]
            logger.info(
                f"Resumed from epoch {checkpoint['epoch']} "
                f"(train_loss: {checkpoint['train_loss']:.4f}, val_loss: {checkpoint['val_loss']:.4f})"
            )
            if start_epoch > config.training.num_epochs:
                raise ValueError(
                    f"Checkpoint already at epoch {checkpoint['epoch']} but config has "
                    f"num_epochs={config.training.num_epochs}. Increase num_epochs to continue."
                )

        # --- TensorBoard (same run dir as logs + checkpoints) ---
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
        logger.exception("Pre-training failed with unhandled exception")
        raise


if __name__ == "__main__":
    main()