import json
import logging
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

logger = logging.getLogger(__name__)


# NOTE: canonical copy is common.run_utils.update_leaderboard (finetune uses it).
# Kept local here to avoid churning already-shipped pre-training code — if you fix
# a bug in one, fix the other. Migrate this call when pretrain.py is next touched.
def _update_leaderboard(parent_dir: str, run_name: str, val_loss: float) -> None:
    """
    Record this run's best val_loss in {parent_dir}/leaderboard.json and repoint
    the {parent_dir}/best.pt symlink at the global best across all runs.

    parent_dir holds run_<timestamp>/ subdirs; run_name is the basename of the
    current run dir. Symlink target is relative ("run_X/best.pt") so the
    parent dir stays portable if moved.
    """
    leaderboard_path = os.path.join(parent_dir, "leaderboard.json")
    board: dict[str, float] = {}
    if os.path.exists(leaderboard_path):
        with open(leaderboard_path) as f:
            board = json.load(f)

    board[run_name] = val_loss
    # Sort ascending by val_loss so the file reads top-down as a ranking.
    board = dict(sorted(board.items(), key=lambda kv: kv[1]))
    with open(leaderboard_path, "w") as f:
        json.dump(board, f, indent=2)

    best_run = next(iter(board))
    symlink = os.path.join(parent_dir, "best.pt")
    target = os.path.join(best_run, "best.pt")  # relative -> portable across moves
    if os.path.islink(symlink) or os.path.exists(symlink):
        os.unlink(symlink)
    os.symlink(target, symlink)


def set_seed(seed: int) -> None:
    """Seed random/numpy/torch (CPU + CUDA + MPS) for reproducibility."""
    random.seed(seed)           # nsp.py: IsNext coin-flip + random next-sentence pick
    np.random.seed(seed)        # nothing draws from it here — defensive carryover
    torch.manual_seed(seed)     # masking.py randperm/rand/randint + dropout + DataLoader shuffle (CPU/CUDA/MPS)


def train_on_epoch(
        model: nn.Module,
        train_loader,
        criterion: nn.Module,
        optimizer,
        scheduler,
        clip_grad_norm: float,
        device: torch.device,
        epoch: int,
        num_epochs: int,
        writer: SummaryWriter | None = None
) -> tuple[float, float, float]:
    """
    Train the model for one epoch.

    One epoch = one full pass through all training batches.
    Each batch: forward → loss → backward → clip gradients → optimizer step → scheduler step.

    The batch is the dict from data_utils.collate_fn. The model derives its own
    pad mask from input_ids internally, so no attention_mask is needed.

    Args:
        model: BERTForPreTraining.
        train_loader: yields dict batches (input_ids, token_type_ids,
            mlm_labels, nsp_labels).
        criterion: BERTPreTrainingLoss — returns (total, mlm, nsp).
        optimizer: AdamW.
        scheduler: LambdaLR with LinearWarmupScheduler.
        clip_grad_norm: Max gradient norm — clips if exceeded.
        device: "mps", "cuda", or "cpu".
        epoch, num_epochs: for the progress-bar label.
        writer: optional TensorBoard SummaryWriter — logs per-step total/mlm/nsp
            loss and lr (captures the warmup→decay curve, Appendix A.2).

    Returns:
        (avg_total, avg_mlm, avg_nsp) over all sequences in this epoch.
    """
    model.train()                                    # enable dropout for training
    total_loss = total_mlm = total_nsp = 0.0
    total_seqs = 0

    pbar = tqdm(train_loader, desc=f"{epoch}/{num_epochs}")

    for batch in pbar:
        input_ids = batch["input_ids"].to(device)            # (B, S)
        token_type_ids = batch["token_type_ids"].to(device)  # (B, S)
        mlm_labels = batch["mlm_labels"].to(device)          # (B, S)
        nsp_labels = batch["nsp_labels"].to(device)          # (B,)

        # Model builds its own pad mask from input_ids — no mask arg.
        mlm_logits, nsp_logits = model(input_ids, token_type_ids)
        loss, mlm_loss, nsp_loss = criterion(mlm_logits, nsp_logits, mlm_labels, nsp_labels)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=clip_grad_norm)
        optimizer.step()
        scheduler.step()                              # advance lr (warmup → decay)

        # Release MPS cached blocks each step — on M1 unified memory the cache
        # holds freed tensors and surfaces as "other allocations" in OOM errors.
        if device.type == "mps":
            torch.mps.empty_cache()

        # Weight each batch's loss by its sequence count so the epoch average is
        # a true mean even when the last batch is short.
        n = input_ids.size(0)
        total_loss += loss.item() * n
        total_mlm += mlm_loss.item() * n
        total_nsp += nsp_loss.item() * n
        total_seqs += n

        pbar.set_postfix(
            loss=f"{total_loss / total_seqs:.4f}",
            mlm=f"{total_mlm / total_seqs:.4f}",
            nsp=f"{total_nsp / total_seqs:.4f}",
            lr=f"{scheduler.get_last_lr()[0]:.2e}",
        )

        if writer is not None:
            step = scheduler.last_epoch  # LambdaLR's global step count
            writer.add_scalar("train/loss_step", loss.item(), step)
            writer.add_scalar("train/mlm_step", mlm_loss.item(), step)
            writer.add_scalar("train/nsp_step", nsp_loss.item(), step)
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], step)

    return total_loss / total_seqs, total_mlm / total_seqs, total_nsp / total_seqs


@torch.no_grad()                                     # no grads — saves memory and speed
def validate(
        model: nn.Module,
        val_loader,
        criterion: nn.Module,
        device: torch.device,
) -> tuple[float, float, float]:
    """
    Evaluate on the validation set. Same forward as training, but model.eval()
    (dropout off), no backward, no optimizer/scheduler step.

    Returns:
        (avg_total, avg_mlm, avg_nsp).
    """
    model.eval()                                     # disable dropout for validation
    total_loss = total_mlm = total_nsp = 0.0
    total_seqs = 0

    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        mlm_labels = batch["mlm_labels"].to(device)
        nsp_labels = batch["nsp_labels"].to(device)

        mlm_logits, nsp_logits = model(input_ids, token_type_ids)
        loss, mlm_loss, nsp_loss = criterion(mlm_logits, nsp_logits, mlm_labels, nsp_labels)

        n = input_ids.size(0)
        total_loss += loss.item() * n
        total_mlm += mlm_loss.item() * n
        total_nsp += nsp_loss.item() * n
        total_seqs += n

        if device.type == "mps":
            torch.mps.empty_cache()

    return total_loss / total_seqs, total_mlm / total_seqs, total_nsp / total_seqs


def train(
        model: nn.Module,
        train_loader,
        val_loader,
        criterion: nn.Module,
        optimizer,
        scheduler,
        clip_grad_norm: float,
        device: torch.device,
        num_epochs: int,
        checkpoint_dir: str,
        seed: int,
        start_epoch: int = 1,
        best_val_loss: float = float("inf"),
        git_hash: str = "unknown",
        tokenizer_sha256: str = "unknown",
        data_fingerprint: str = "unknown",
        writer: SummaryWriter | None = None,
) -> None:
    """
    Full pre-training loop — train num_epochs, validate each epoch, checkpoint.

    Args:
        model: BERTForPreTraining.
        train_loader, val_loader: dict-batch DataLoaders.
        criterion: BERTPreTrainingLoss.
        optimizer: AdamW.
        scheduler: LambdaLR with LinearWarmupScheduler.
        clip_grad_norm: Max gradient norm for clipping.
        device: "mps", "cuda", or "cpu".
        num_epochs: Total epochs.
        checkpoint_dir: Directory for best.pt / last.pt.
        seed: Base seed. Re-seeded each epoch as (seed + epoch) so uninterrupted
            and resumed runs are bit-identical at epoch boundaries.
        start_epoch: First epoch to run (for resume). Defaults to 1.
        best_val_loss: Best val loss so far (for resume). Defaults to +inf.
        git_hash: Commit that produced these weights — saved in the checkpoint so
            a run can be traced back to its source code.
        tokenizer_sha256: SHA-256 of the tokenizer file — pins these embeddings
            to a specific vocab, so a resume against a retrained tokenizer fails.
        data_fingerprint: SHA-256 of the corpus — pins these weights to a specific
            data slice, so a resume against a changed corpus fails.
        writer: optional TensorBoard SummaryWriter — per-epoch train/val loss.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(start_epoch, num_epochs + 1):
        # Re-seed each epoch so resumes are bit-identical at epoch boundaries.
        set_seed(seed + epoch)

        train_loss, train_mlm, train_nsp = train_on_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            clip_grad_norm, device, epoch, num_epochs, writer=writer,
        )

        if device.type == "mps":
            torch.mps.empty_cache()

        val_loss, val_mlm, val_nsp = validate(model, val_loader, criterion, device)

        if device.type == "mps":
            torch.mps.empty_cache()

        logger.info(
            f"  Train {train_loss:.4f} (mlm {train_mlm:.4f} / nsp {train_nsp:.4f}) | "
            f"Val {val_loss:.4f} (mlm {val_mlm:.4f} / nsp {val_nsp:.4f})"
        )

        if writer is not None:
            writer.add_scalar("train/loss_epoch", train_loss, epoch)
            writer.add_scalar("val/loss_epoch", val_loss, epoch)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": train_loss,
            "train_mlm": train_mlm,
            "train_nsp": train_nsp,
            "val_loss": val_loss,
            "val_mlm": val_mlm,
            "val_nsp": val_nsp,
            "best_val_loss": best_val_loss,
            "git_hash": git_hash,                    # commit that produced these weights
            "tokenizer_sha256": tokenizer_sha256,    # pins these embeddings to a specific tokenizer
            "data_fingerprint": data_fingerprint,    # pins these weights to a specific data slice
        }

        if improved:
            best_path = os.path.join(checkpoint_dir, "best.pt")
            torch.save(obj=checkpoint, f=best_path)
            logger.info(f"  Saved best model (val_loss: {val_loss:.4f}) -> {best_path}")

            # Update parent-level leaderboard.json + best.pt symlink so the
            # global best across all runs is always one fixed path away.
            parent_dir = os.path.dirname(checkpoint_dir.rstrip(os.sep))
            run_name = os.path.basename(checkpoint_dir.rstrip(os.sep))
            _update_leaderboard(parent_dir, run_name, val_loss)

        # last.pt every epoch for resume — but skip on NaN so corrupted weights
        # don't poison auto-resume.
        last_path = os.path.join(checkpoint_dir, "last.pt")
        if math.isfinite(train_loss) and math.isfinite(val_loss):
            torch.save(obj=checkpoint, f=last_path)
        else:
            logger.warning(f"Skipping last.pt — NaN loss (train={train_loss}, val={val_loss})")

    logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
    logger.info(f"Latest weights: {last_path}")
