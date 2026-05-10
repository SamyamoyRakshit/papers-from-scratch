import json
import logging
import os
import random
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from .mask_utils import create_src_mask, create_tgt_mask, create_memory_mask

logger = logging.getLogger(__name__)


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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)     # handles CPU + CUDA (all) + MPS

def train_on_epoch(
        model: nn.Module,
        train_loader,
        criterion: nn.Module,
        optimizer,
        scheduler,
        pad_idx: int,
        clip_grad_norm: float,
        device: torch.device,
        epoch: int,
        num_epochs: int,
        writer: SummaryWriter | None = None,
) -> float:
    """
    Train the model for one epoch.

    One epoch = one full pass through all training batches.
    Each batch: forward → loss → backward → clip gradients → optimizer step → scheduler step.

    Args:
        model: Transformer model.
        train_loader: Training DataLoader (yields src, tgt batches).
        criterion: LabelSmoothedLoss instance.
        optimizer: Adam optimizer.
        scheduler: LambdaLR with TransformerScheduler.
        pad_idx: Padding token index (for masking and loss).
        clip_grad_norm: Max gradient norm — clips if exceeded.
        device: "mps", "cuda", or "cpu".
        epoch: Current epoch number (for logging).
        num_epochs: Total number of epochs (for logging).
        writer: Optional TensorBoard SummaryWriter. If provided, logs
            per-step train loss and learning rate (under tags
            "train/loss_step" and "train/lr") — useful for visualizing
            the warmup → decay LR curve from Section 5.3.

    Returns:
        float: Average loss over all batches in this epoch.
    """
    model.train()                                    # enable dropout for training
    total_loss = 0.0
    total_tokens = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")  # progress bar: "Epoch 1/30: 40%|████..."

    for src, tgt in pbar:
        src = src.to(device)                        # (batch_size, src_seq_len)
        tgt = tgt.to(device)                         # (batch_size, tgt_seq_len)
        
        # Teacher forcing: decoder sees tgt[:-1], predicts tgt[1:]
        # Input:  [<sos>, আমি, AI, ভালোবাসি]     → what the decoder sees
        # Target: [আমি, AI, ভালোবাসি, <eos>]      → what it should predict
        tgt_input = tgt[:, :-1]                      # remove last token  — decoder input
        tgt_output = tgt[:, 1:]                      # remove first token — ground truth

        # Create masks
        src_mask = create_src_mask(src, pad_idx).to(device)            # (batch, 1, 1, src_len)
        tgt_mask = create_tgt_mask(tgt_input, pad_idx).to(device)           # (batch, 1, tgt_len, tgt_len)
        memory_mask = create_memory_mask(src, pad_idx).to(device)     # (batch, 1, 1, src_len)

        # Forward pass
        logits = model(src, tgt_input, src_mask, tgt_mask, memory_mask)
        # logits: (batch_size, tgt_seq_len-1, vocab_size); tgt_seq_len-1 = len(tgt_input)

        # Flatten for loss — LabelSmoothedLoss expects 2D logits and 1D targets
        vocab_size = logits.size(-1)
        logits = logits.reshape(-1, vocab_size)      # (batch * seq_len, vocab_size)
        tgt_output = tgt_output.reshape(-1)           # (batch * seq_len,)

        # Compute loss
        loss = criterion(logits, tgt_output)

        # Backward pass
        optimizer.zero_grad()                         # clear previous gradients
        loss.backward()                               # compute gradients

        # Gradient clipping — prevents exploding gradients
        # Not in the paper, but standard practice for transformers
        nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=clip_grad_norm)

        # Update weights and learning rate
        optimizer.step()                              # update parameters
        scheduler.step()                              # update lr (warmup -> decay)

        # Track loss — count non-pad tokens for accurate average
        n_tokens = (tgt_output != pad_idx).sum().item() # .item(): Converts a one-element tensor into a Python scalar, removing the tensor wrapper.
        total_loss += loss.item() * n_tokens          # loss is already per-token, multiply back for logging
        total_tokens += n_tokens

        # Update progress bar with current loss and learning rate
        pbar.set_postfix(loss=f"{total_loss / total_tokens:.4f}",
                         lr=f"{scheduler.get_last_lr()[0]:.2e}")

        # Per-step TensorBoard logging — captures the warmup ramp (paper Section 5.3)
        if writer is not None:
            step = scheduler.last_epoch  # LambdaLR's global step count, incremented on .step()
            writer.add_scalar("train/loss_step", loss.item(), step)
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], step)

    return total_loss / total_tokens
    

@torch.no_grad()                                     # no gradient computation (requires_grad=False) — saves memory and speed
def validate(
    model: nn.Module,
    val_loader,
    criterion: nn.Module,
    pad_idx: int,
    device: torch.device
) -> float:
    """
    Evaluate model on validation set.

    Same as training forward pass, but:
        - No backward pass (no gradients)
        - No optimizer/scheduler step
        - model.eval() disables dropout

    Args:
        model: Transformer model.
        val_loader: Validation DataLoader.
        criterion: LabelSmoothedLoss instance.
        pad_idx: Padding token index.
        device: "mps", "cuda", or "cpu".

    Returns:
        float: Average validation loss.
    """
    model.eval()                                     # disable dropout for validation
    total_loss = 0.0
    total_tokens = 0

    for src, tgt in val_loader:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        src_mask = create_src_mask(src, pad_idx).to(device)
        tgt_mask = create_tgt_mask(tgt_input, pad_idx).to(device)
        memory_mask = create_memory_mask(src, pad_idx).to(device)

        logits = model(src, tgt_input, src_mask, tgt_mask, memory_mask)

        vocab_size = logits.size(-1)
        logits = logits.reshape(-1, vocab_size)
        tgt_output = tgt_output.reshape(-1)

        loss = criterion(logits, tgt_output)

        n_tokens = (tgt_output != pad_idx).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    return total_loss / total_tokens


def train(
        model: nn.Module,
        train_loader,
        val_loader,
        criterion: nn.Module,
        optimizer,
        scheduler,
        pad_idx: int,
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
    Full training loop — train for num_epochs, validate each epoch, save best model.

    Args:
        model: Transformer model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        criterion: LabelSmoothedLoss instance.
        optimizer: Adam optimizer.
        scheduler: LambdaLR with TransformerScheduler.
        pad_idx: Padding token index.
        clip_grad_norm: Max gradient norm for clipping.
        device: "mps", "cuda", or "cpu".
        num_epochs: Total number of epochs.
        checkpoint_dir: Directory to save checkpoints.
        seed: Base seed. Re-seeded each epoch as (seed + epoch) so uninterrupted
            and resumed runs are bit-identical at epoch boundaries.
        start_epoch: First epoch to run (for resume). Defaults to 1.
        best_val_loss: Best val loss seen so far (for resume). Defaults to +inf.
        git_hash: Commit hash that produced these weights — saved in checkpoint
            so a year-old run can be traced back to its source code.
        writer: Optional TensorBoard SummaryWriter. If provided, logs per-epoch
            train and val loss (under "train/loss_epoch" and "val/loss_epoch")
            and forwards to train_on_epoch for per-step lr logging.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(start_epoch, num_epochs+1):
        # Re-seed each epoch so resumes are bit-identical at epoch boundaries:
        # uninterrupted and resumed runs both hit set_seed(seed + epoch) here,
        # giving identical dropout masks / shuffle order from this point on.
        set_seed(seed + epoch)

        # Train
        train_loss = train_on_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            pad_idx, clip_grad_norm, device, epoch, num_epochs,
            writer=writer,
        )

        # Validate
        val_loss = validate(model, val_loader, criterion, pad_idx, device)

        logger.info(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Per-epoch TensorBoard logging — easy comparison across runs
        if writer is not None:
            writer.add_scalar("train/loss_epoch", train_loss, epoch)
            writer.add_scalar("val/loss_epoch", val_loss, epoch)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,
            'git_hash': git_hash,                    # commit that produced these weights
            'tokenizer_sha256': tokenizer_sha256,    # pins these embeddings to a specific tokenizer
            'data_fingerprint': data_fingerprint,    # pins these weights to a specific data slice
        }

        # Save best model
        if improved:
            checkpoint_path = os.path.join(checkpoint_dir, "best.pt")
            torch.save(obj=checkpoint, f=checkpoint_path)
            logger.info(f"  Saved best model (val_loss: {val_loss:.4f}) -> {checkpoint_path}")

            # Update parent-level leaderboard.json + best.pt symlink so the
            # global best across all runs is always one fixed path away.
            parent_dir = os.path.dirname(checkpoint_dir.rstrip(os.sep))
            run_name = os.path.basename(checkpoint_dir.rstrip(os.sep))
            _update_leaderboard(parent_dir, run_name, val_loss)

        # Save latest checkpoint every epoch (for resuming) — silent to avoid log noise.
        latest_path = os.path.join(checkpoint_dir, "last.pt")
        torch.save(obj=checkpoint, f=latest_path)

    logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
    logger.info(f"Latest weights: {latest_path}")