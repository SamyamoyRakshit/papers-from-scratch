import torch
import torch.nn as nn
import torch.nn.functional as F     # fused cross_entropy


class GPTLoss(nn.Module):
    """
    Next-token loss (GPT-1 §3.1, Eq. 1: maximize Σ log P(u_i | u_{i-k}, …, u_{i-1})).

    Plain cross-entropy over the vocab at EVERY position:

        logits (B, T, 8192) ─view─►    (B·T, 8192) ┐
                                                   ├─► F.cross_entropy ─► loss ()
        y      (B, T)       ─reshape─► (B·T,)      ┘

    No shift here: get_batch already returns y[:, t] = the token after x[:, t].
    Example "আমি কলকাতা যাব": x = [844, 3300, 1636], y = [3300, 1636, 6464];
    position 1 (after "আমি") is scored on how high it put id 1636 " কলকাতা".

    Differences from the siblings:
      - no ignore_index: train.bin is one long stream, every window is full, no <pad>.
        All B·T positions count (BERT scores only the ~15% masked ones).
      - no label smoothing: neither GPT paper uses it (transformer/ uses 0.1, §5.4).
      - one scalar back, not BERT's (total, mlm, nsp) tuple: one objective.
      
    """
    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch_size, T, vocab_size), from GPT.forward.
            y: (batch_size, T) int64 next-token ids, from get_batch.
        Returns:
            loss: scalar, mean over all batch_size·T positions.
        """
        # reshape, not view: y = windows[:, 1:] is a non-contiguous slice on CPU.
        return F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))
