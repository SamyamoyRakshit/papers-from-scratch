import os
from tqdm import tqdm

import torch
import torch.nn as nn

from .mask_utils import create_src_mask, create_tgt_mask, create_memory_mask

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
        num_epochs: int
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
        scheduler.step()                              # update lr (warmup → decay)

        # Track loss — count non-pad tokens for accurate average
        n_tokens = (tgt_output != pad_idx).sum().item() # .item(): Converts a one-element tensor into a Python scalar, removing the tensor wrapper.
        total_loss += loss.item() * n_tokens          # loss is already per-token, multiply back for logging
        total_tokens += n_tokens

        # Update progress bar with current loss and learning rate
        pbar.set_postfix(loss=f"{total_loss / total_tokens:.4f}",
                         lr=f"{scheduler.get_last_lr()[0]:.2e}")
            
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
        checkpoint_dir: str
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
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(1, num_epochs+1):
        # Train
        train_loss = train_on_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            pad_idx, clip_grad_norm, device, epoch, num_epochs
        )

        # Validate
        val_loss = validate(model, val_loader, criterion, pad_idx, device)

        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss
        }
    
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(checkpoint_dir, "best.pt")
            torch.save(obj=checkpoint, f=checkpoint_path)
            print(f"  Saved best model (val_loss: {val_loss:.4f}) -> {checkpoint_path}")

        # Save latest checkpoint every epoch (for resuming)
        latest_path = os.path.join(checkpoint_dir, "last.pt")
        torch.save(obj=checkpoint, f=latest_path)
        
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")