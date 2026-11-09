"""
Step-based pre-training loop for GPT: train → validate → checkpoint.

Steps, not epochs: train.bin is one token stream and get_batch draws random
windows from it, so there is no "end of the data" to count epochs by.
One step = grad_accum_steps forward/backward passes, then one optimizer step:

    tokens per step = batch_size · grad_accum_steps · block_size
    base.yaml:        32 · 8 · 256 = 65,536
"""
import logging
import math
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from BERT.utils.train_utils import set_seed   # random / numpy / torch seeds — imported unchanged
from common.run_utils import update_leaderboard
from .data_utils import get_batch

logger = logging.getLogger(__name__)


def train_step(
        model: nn.Module,
        train_ids: np.memmap,
        criterion: nn.Module,
        optimizer,
        scheduler,
        batch_size: int,
        block_size: int,
        grad_accum_steps: int,
        clip_grad_norm: float,
        device: torch.device,
) -> tuple[float, float]:
    """
    One optimizer step from grad_accum_steps batches.

    Each batch's loss is divided by grad_accum_steps before backward. backward()
    ADDS into .grad, so after the loop .grad is the gradient of the mean loss over
    all batch_size · grad_accum_steps sequences: the step one big batch would take,
    without holding its activations in memory at once.

    Returns:
        (loss, grad_norm): mean loss of the step, and the gradient norm BEFORE
        clipping (above clip_grad_norm → this step was clipped).
    """
    model.train()                                    # dropout on (validate turns it off)
    optimizer.zero_grad(set_to_none=True)
    loss_sum = torch.zeros((), device=device)        # on device: one .item() sync per step, not per batch
    for _ in range(grad_accum_steps):
        x, y = get_batch(train_ids, batch_size, block_size, device)   # (B, T), (B, T)
        loss = criterion(model(x), y) / grad_accum_steps
        loss.backward()
        loss_sum += loss.detach()
    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
    optimizer.step()
    scheduler.step()                                 # advance lr: warmup → cosine → floor
    return loss_sum.item(), grad_norm.item()


@torch.no_grad()                                     # no grads — saves memory and speed
def validate(
        model: nn.Module,
        val_ids: np.memmap,
        criterion: nn.Module,
        batch_size: int,
        block_size: int,
        device: torch.device,
) -> float:
    """
    Mean next-token loss over ALL of val.bin, in non-overlapping windows from id 0.

    Not get_batch's random windows: these are the same windows every call, so two
    val losses differ only because the weights changed. base: 1,200,709 ids →
    4,690 windows of 256.
    """
    model.eval()                                     # dropout off
    # The last start s must leave room for y: s + block_size + 1 ≤ len(val_ids).
    starts = range(0, len(val_ids) - block_size, block_size)
    total_loss, total_seqs = 0.0, 0

    for i in range(0, len(starts), batch_size):
        windows = torch.stack([torch.from_numpy(val_ids[s:s + block_size + 1].astype(np.int64))
                               for s in starts[i:i + batch_size]])
        x, y = windows[:, :-1].to(device), windows[:, 1:].to(device)
        loss = criterion(model(x), y)

        # Weight by sequence count: the last batch is short.
        n = x.size(0)
        total_loss += loss.item() * n
        total_seqs += n

    return total_loss / total_seqs


def train(
        model: nn.Module,
        train_ids: np.memmap,
        val_ids: np.memmap,
        criterion: nn.Module,
        optimizer,
        scheduler,
        batch_size: int,
        block_size: int,
        grad_accum_steps: int,
        clip_grad_norm: float,
        max_steps: int,
        eval_interval: int,
        device: torch.device,
        checkpoint_dir: str,
        seed: int,
        start_step: int = 0,
        best_val_loss: float = float("inf"),
        git_hash: str = "unknown",
        tokenizer_sha256: str = "unknown",
        data_fingerprint: str = "unknown",
        writer: SummaryWriter | None = None,
) -> None:
    """
    Full pre-training loop — max_steps optimizer steps; every eval_interval steps
    (and at the last one) a full val pass and a checkpoint.

    Args:
        model: GPT.
        train_ids, val_ids: memmaps from data_utils.load_ids.
        criterion: GPTLoss.
        optimizer: AdamW.
        scheduler: LambdaLR with CosineWarmupScheduler.
        batch_size, block_size, grad_accum_steps: one step = batch_size ·
            grad_accum_steps windows of block_size tokens.
        clip_grad_norm: Max gradient norm for clipping.
        max_steps: Last step to run.
        eval_interval: Steps between val passes + checkpoints.
        device: "mps", "cuda", or "cpu".
        checkpoint_dir: Directory for best.pt / last.pt.
        seed: Base seed. Re-seeded every step as (seed + step), so step k draws the
            same batches and dropout masks whether or not the run was resumed —
            BERT's (seed + epoch) rule, per step because a step is GPT's unit.
        start_step: Steps already done (resume). Defaults to 0.
        best_val_loss: Best val loss so far (for resume). Defaults to +inf.
        git_hash: Commit that produced these weights.
        tokenizer_sha256: SHA-256 of merges.txt — pins the weights to these ids.
        data_fingerprint: SHA-256 of train.bin — pins the weights to this data.
        writer: optional TensorBoard SummaryWriter — per-step loss, lr, grad norm
            and per-eval train/val loss.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    tokens_per_step = batch_size * grad_accum_steps * block_size
    running_loss, running_steps = 0.0, 0             # train loss since the last eval

    # logging_redirect_tqdm: the eval log lines print above the bar instead of breaking it.
    with logging_redirect_tqdm():
        pbar = tqdm(range(start_step + 1, max_steps + 1), initial=start_step, total=max_steps)
        for step in pbar:
            set_seed(seed + step)
            lr = scheduler.get_last_lr()[0]          # the lr THIS step uses; train_step moves it on
            loss, grad_norm = train_step(
                model, train_ids, criterion, optimizer, scheduler,
                batch_size, block_size, grad_accum_steps, clip_grad_norm, device,
            )
            running_loss += loss
            running_steps += 1

            pbar.set_postfix(loss=f"{loss:.4f}", lr=f"{lr:.2e}", norm=f"{grad_norm:.2f}")
            if writer is not None:
                writer.add_scalar("train/loss_step", loss, step)
                writer.add_scalar("train/lr", lr, step)
                writer.add_scalar("train/grad_norm", grad_norm, step)

            if step % eval_interval != 0 and step != max_steps:
                continue

            # --- Eval + checkpoint ---
            val_loss = validate(model, val_ids, criterion, batch_size, block_size, device)
            if device.type == "mps":
                torch.mps.empty_cache()
            train_loss = running_loss / running_steps
            running_loss, running_steps = 0.0, 0

            logger.info(
                f"  Step {step}/{max_steps} ({step * tokens_per_step / 1e6:.1f}M tokens) | "
                f"Train {train_loss:.4f} | Val {val_loss:.4f} (ppl {math.exp(val_loss):.1f})"
            )
            if writer is not None:
                writer.add_scalar("train/loss_eval", train_loss, step)
                writer.add_scalar("val/loss", val_loss, step)

            improved = val_loss < best_val_loss
            if improved:
                best_val_loss = val_loss

            checkpoint = {
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_val_loss": best_val_loss,
                "git_hash": git_hash,                    # commit that produced these weights
                "tokenizer_sha256": tokenizer_sha256,    # pins these embeddings to a specific tokenizer
                "data_fingerprint": data_fingerprint,    # pins these weights to a specific train.bin
            }

            if improved:
                best_path = os.path.join(checkpoint_dir, "best.pt")
                torch.save(obj=checkpoint, f=best_path)
                logger.info(f"  Saved best model (val_loss: {val_loss:.4f}) -> {best_path}")

                # Parent-level leaderboard.json + best.pt symlink: the global best
                # across all runs is always one fixed path away.
                parent_dir = os.path.dirname(checkpoint_dir.rstrip(os.sep))
                run_name = os.path.basename(checkpoint_dir.rstrip(os.sep))
                update_leaderboard(parent_dir, run_name, val_loss)

            # last.pt at every eval for resume — but skip on NaN so corrupted weights
            # don't poison --resume.
            last_path = os.path.join(checkpoint_dir, "last.pt")
            if math.isfinite(train_loss) and math.isfinite(val_loss):
                torch.save(obj=checkpoint, f=last_path)
            else:
                logger.warning(f"Skipping last.pt — NaN loss (train={train_loss}, val={val_loss})")

    logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
    logger.info(f"Latest weights: {last_path}")
