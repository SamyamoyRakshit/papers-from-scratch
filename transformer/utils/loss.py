import torch
import torch.nn as nn


class LabelSmoothedLoss(nn.Module):
    """
    Label-Smoothed Cross-Entropy Loss (Section 5.4)

    Instead of one-hot targets, distributes a small probability mass (ε)
    across all tokens, preventing the model from becoming overconfident.
    
    Target distribution:
        p(correct)    = 1 - ε
        p(each other) = ε / (V - 2)
        p(pad)        = 0

    Implemented as KL divergence since the target is a soft distribution:
        KL(p || q) = Σ p · (log p - log q)

    Reference: "Rethinking the Inception Architecture" (Szegedy et al., 2016)

    Args:
        pad_idx (int): Index of the <pad> token. Padding positions are
            excluded from both the target distribution and loss normalization.
        smoothing (float, optional): Label smoothing value (ε). Default is 0.1.

    Derived:
        confidence (float): Probability assigned to the correct token.
            confidence = 1 - smoothing
            Example: 1 - 0.1 = 0.9
    """
    def __init__(self, pad_idx: int, smoothing: float = 0.1):
        super().__init__()
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1 - smoothing # 0.9

        # KL divergence: KL(target || predicted)
        # reduction='sum' because we'll divide by token count manually
        # (to exclude padding tokens from the average)
        self.criterion = nn.KLDivLoss(reduction='sum')

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute label-smoothed loss.

        Args:
            logits (torch.Tensor): Raw model output (before softmax).
                Shape: (batch_size * seq_len, vocab_size)
            target (torch.Tensor): Ground truth token IDs.
                Shape: (batch_size * seq_len,)

        Returns:
            torch.Tensor: Scalar loss, normalized by the number of non-padding tokens.
        """
        vocab_size = logits.size(-1)    # 16000 in our case (from vocab_size: 16000 in config).

        # log_softmax for KL divergence (KLDivLoss expects log-probabilities)
        log_probs = torch.log_softmax(logits, dim=-1)

        # Build smoothed target distribution
        # Start with smoothing / (vocab_size - 2):
        #   -2 because we exclude pad_idx (always 0) and correct token (gets confidence)
        smooth_value = self.smoothing / (vocab_size - 2)
        smoothed = torch.full_like(log_probs, smooth_value)

        # Set correct token positions to confidence value (0.9)
        smoothed.scatter_(1, target.unsqueeze(1), self.confidence)

        # Zero out padding positions — pad tokens shouldn't contribute to loss
        smoothed[:, self.pad_idx] = 0

        # Also zero out entire rows where target is padding
        pad_mask = target == self.pad_idx
        smoothed[pad_mask] = 0

        # Count non-padding tokens for normalization
        n_tokens = (~pad_mask).sum().item()

        # KL divergence
        loss = self.criterion(log_probs, smoothed)

        # Normalize by number of real tokens (not padding)
        return loss / n_tokens