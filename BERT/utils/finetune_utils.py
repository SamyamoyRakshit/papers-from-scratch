import logging
import math
import os

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from common.run_utils import update_leaderboard
from .train_utils import set_seed

logger = logging.getLogger(__name__)


def train_on_epoch(model, loader, criterion, optimizer, scheduler,
                   clip_grad_norm, device, epoch, num_epochs, writer=None):
    """One training epoch: forward → CE loss → backward → clip → step → sched step."""
    model.train()
    total_loss = 0.0
    correct = total = 0

    pbar = tqdm(loader, desc=f"{epoch}/{num_epochs}")
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, token_type_ids)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()
        scheduler.step()

        if device.type == "mps":
            torch.mps.empty_cache()

        # Weighted running mean: criterion already averaged over the batch, so
        # multiply back by n to recover the batch's summed loss before accumulating.
        # Otherwise the short trailing batch (11284/32 → last batch = 20, not 32)
        # would count as much as a full one. argmax(-1) = predicted class per row.
        n = labels.size(0)
        total_loss += loss.item() * n
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += n

        pbar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{correct/total:.4f}",
                         lr=f"{scheduler.get_last_lr()[0]:.2e}")

        if writer is not None:
            step = scheduler.last_epoch
            writer.add_scalar("train/loss_step", loss.item(), step)
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], step)

    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Evaluate: same forward, dropout off, no backward. Returns (loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = total = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, token_type_ids)
        loss = criterion(logits, labels)

        n = labels.size(0)
        total_loss += loss.item() * n
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += n

        if device.type == "mps":
            torch.mps.empty_cache()

    return total_loss / total, correct / total


def train(model, train_loader, val_loader, criterion, optimizer, scheduler,
          clip_grad_norm, device, num_epochs, checkpoint_dir, seed,
          git_hash="unknown", tokenizer_sha256="unknown",
          pretrained_checkpoint="unknown", writer=None):
    """
    Fine-tuning loop — train num_epochs, validate each, checkpoint the best by
    val accuracy (classification's target metric). last.pt every epoch for resume;
    provenance (git_hash, tokenizer_sha256, source checkpoint) saved in each.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        set_seed(seed + epoch)

        train_loss, train_acc = train_on_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            clip_grad_norm, device, epoch, num_epochs, writer=writer)

        val_loss, val_acc = validate(model, val_loader, criterion, device)

        logger.info(f"  Train loss {train_loss:.4f} acc {train_acc:.4f} | "
                    f"Val loss {val_loss:.4f} acc {val_acc:.4f}")

        if writer is not None:
            writer.add_scalar("train/loss_epoch", train_loss, epoch)
            writer.add_scalar("train/acc_epoch", train_acc, epoch)
            writer.add_scalar("val/loss_epoch", val_loss, epoch)
            writer.add_scalar("val/acc_epoch", val_acc, epoch)

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            # Adam moments + lr schedule position — saved so this run COULD be
            # resumed, but no resume path is wired for fine-tuning (see pretrain.py
            # --resume): a 9-min run is cheaper to just re-run. Note these are NEVER
            # reused when starting a fresh fine-tune anyway — that loads only bert.*
            # weights and starts the optimizer/scheduler clean.
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
            "best_val_acc": best_val_acc,
            "git_hash": git_hash,
            "tokenizer_sha256": tokenizer_sha256,
            "pretrained_checkpoint": pretrained_checkpoint,   # which encoder this was fine-tuned from
        }

        if improved:
            best_path = os.path.join(checkpoint_dir, "best.pt")
            torch.save(checkpoint, best_path)
            logger.info(f"  Saved best model (val_acc: {val_acc:.4f}) -> {best_path}")
            parent_dir = os.path.dirname(checkpoint_dir.rstrip(os.sep))
            run_name = os.path.basename(checkpoint_dir.rstrip(os.sep))
            # higher_is_better=True: rank the parent-level board by val_acc DESCENDING
            # (mirror of pre-training's ascending val_loss) so best.pt points at the top run.
            update_leaderboard(parent_dir, run_name, val_acc, higher_is_better=True)

        last_path = os.path.join(checkpoint_dir, "last.pt")
        if math.isfinite(train_loss) and math.isfinite(val_loss):
            torch.save(checkpoint, last_path)
        else:
            logger.warning(f"Skipping last.pt — NaN loss (train={train_loss}, val={val_loss})")

    logger.info(f"Fine-tuning complete. Best val accuracy: {best_val_acc:.4f}")
    logger.info(f"Latest weights: {last_path}")