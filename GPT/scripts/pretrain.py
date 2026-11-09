"""
Pre-training entrypoint for GPT (GPT-2 §2.3; optimizer and schedule from GPT-3 Appendix B).

Usage:
    python -m GPT.scripts.pretrain                                  # uses base.yaml
    python -m GPT.scripts.pretrain --config GPT/configs/tiny.yaml
    python -m GPT.scripts.pretrain --resume GPT/checkpoints/base/run_<ts>/last.pt

Expects train.bin / val.bin — run prepare_corpus.py, train_tokenizer.py and
prepare_data.py first. Unlike BERT's, this script never tokenizes: the ids are
already on disk.
"""
import argparse
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from common.run_utils import (
    get_device, get_git_hash, setup_logging, sha256_file, warn_if_config_diverges,
)
from ..utils.config import Config
from ..utils.data_utils import load_ids
from ..utils.loss import GPTLoss
from ..utils.optimizer import build_optimizer
from ..utils.train_utils import set_seed, train
from ._common import build_model, load_checkpoint

logger = logging.getLogger(__name__)


# Keys allowed to change between the snapshotted config and the resumed config.
# Anything outside this set means the resumed run is a different experiment than
# the one that produced the checkpoint — we warn but don't block.
# A larger max_steps does NOT stretch the cosine: scheduler.load_state_dict restores
# the old warmup/total (LambdaLR saves a callable object's __dict__), so the extra
# steps run at the floor lr — GPT-3 §B: "training continues at 10%".
_RESUME_SAFE_KEYS = {"training.max_steps", "training.eval_interval", "device", "paths.log_dir"}


def main():
    # --- CLI args ---
    parser = argparse.ArgumentParser(description="Pre-train GPT")
    # Default resolved relative to this file, not CWD — so `python -m ...` works from anywhere.
    default_config = Path(__file__).parent.parent / "configs" / "base.yaml"
    parser.add_argument("--config", type=str, default=str(default_config),
                        help="Path to config YAML file")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a last.pt to resume from (e.g. GPT/checkpoints/base/run_<ts>/last.pt)")
    args = parser.parse_args()

    # --- Config (loaded before logging so we know where the log file goes) ---
    config = Config.from_yaml(args.config)
    seed = config.seed

    # --- Per-run output dirs (run_<ts>/ under checkpoint_dir and log_dir) ---
    # Each invocation owns its own dir, so a resume never overwrites a prior
    # run's best.pt / config.yaml / train.log. (Same scheme as BERT.)
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
        git_hash = get_git_hash()
        logger.info(f"Git commit: {git_hash}")

        # --- Seed + device ---
        set_seed(seed)                               # the weight init draws from it
        device = get_device(config.device)
        logger.info(f"Using device: {device}")

        # --- Token files must exist (prepare_data.py writes them) ---
        for path in (config.data.train_path, config.data.val_path):
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"{path} not found. Run: python -m GPT.scripts.prepare_data {args.config}"
                )

        # --- Fingerprints ---
        # train.bin, not the corpus .txt: the ids depend on the corpus, the tokenizer
        # AND the split, and train.bin changes when any of them does.
        # merges.txt alone fixes every id (vocab.json is only its readable form).
        data_fingerprint = sha256_file(config.data.train_path)
        tokenizer_sha256 = sha256_file(os.path.join(config.paths.tokenizer_dir, "merges.txt"))
        logger.info(f"Data fingerprint: {data_fingerprint[:12]}... | Tokenizer sha256: {tokenizer_sha256[:12]}...")

        # --- Resume preflight (fail fast, before the model is built) ---
        checkpoint = None
        if args.resume:
            if not os.path.exists(args.resume):
                raise FileNotFoundError(f"Checkpoint not found: {args.resume}")
            logger.info(f"Resuming from {args.resume}...")
            checkpoint = load_checkpoint(args.resume, device)

            # Same path, different file: a retrained tokenizer or a re-run
            # prepare_data.py would feed the saved weights ids they never saw.
            for key, current in (("tokenizer_sha256", tokenizer_sha256),
                                 ("data_fingerprint", data_fingerprint)):
                if checkpoint[key] != current:
                    raise RuntimeError(
                        f"{key} mismatch on resume: checkpoint {checkpoint[key][:12]}..., "
                        f"current {current[:12]}... Restore the original files or train without --resume."
                    )

            # Snapshot lives next to the checkpoint being resumed (in its run_<ts>/ dir).
            snapshot_path = os.path.join(os.path.dirname(args.resume), "config.yaml")
            if os.path.exists(snapshot_path):
                snapshot = Config.from_yaml(snapshot_path).model_dump()
                warn_if_config_diverges(snapshot, config.model_dump(), _RESUME_SAFE_KEYS)

        # --- Data (memmaps: ids are read from disk per batch) ---
        train_ids = load_ids(config.data.train_path)
        val_ids = load_ids(config.data.val_path)
        tokens_per_step = config.training.batch_size * config.training.grad_accum_steps * config.model.block_size
        logger.info(
            f"Tokens: {len(train_ids):,} train, {len(val_ids):,} val | {tokens_per_step:,} per step → "
            f"{config.training.max_steps * tokens_per_step / len(train_ids):.1f} passes over train.bin"
        )

        # --- Model ---
        logger.info("Building model...")
        model = build_model(config, device)
        # parameters() yields the tied wte / lm_head tensor once.
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # --- Loss ---
        criterion = GPTLoss()

        # --- Optimizer + scheduler ---
        optimizer, scheduler = build_optimizer(
            model=model,
            total_steps=config.training.max_steps,
            warmup_steps=config.training.warmup_steps,
            min_lr_ratio=config.training.min_lr_ratio,
            lr=config.optimizer.lr,
            betas=config.optimizer.betas,
            eps=config.optimizer.eps,
            weight_decay=config.optimizer.weight_decay,
        )

        # --- Resume (restore state into the freshly built model/opt/sched) ---
        start_step = 0
        best_val_loss = float("inf")
        if checkpoint is not None:
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_step = checkpoint["step"]
            best_val_loss = checkpoint["best_val_loss"]
            logger.info(
                f"Resumed from step {start_step} "
                f"(train_loss: {checkpoint['train_loss']:.4f}, val_loss: {checkpoint['val_loss']:.4f})"
            )
            if start_step >= config.training.max_steps:
                raise ValueError(
                    f"Checkpoint already at step {start_step} but config has "
                    f"max_steps={config.training.max_steps}. Increase max_steps to continue."
                )

        # --- TensorBoard (same run dir as logs + checkpoints) ---
        writer = SummaryWriter(log_dir=run_log_dir)

        # --- Train ---
        try:
            train(
                model=model,
                train_ids=train_ids,
                val_ids=val_ids,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                batch_size=config.training.batch_size,
                block_size=config.model.block_size,
                grad_accum_steps=config.training.grad_accum_steps,
                clip_grad_norm=config.training.clip_grad_norm,
                max_steps=config.training.max_steps,
                eval_interval=config.training.eval_interval,
                device=device,
                checkpoint_dir=run_checkpoint_dir,
                seed=seed,
                start_step=start_step,
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
