import torch
import torch.nn as nn
import torch.nn.functional as F     # fused cross_entropy with ignore_index


class BERTPreTrainingLoss(nn.Module):
    """
    Combined pre-training loss (Devlin et al. 2019, §3.1).

    BERT pre-trains on two objectives at once, and the paper sums them:

        L = L_MLM + L_NSP

    Both are plain cross-entropy — there is NO label smoothing here (unlike the
    transformer's translation loss). The two heads emit logits; this module turns
    (logits, labels) into the scalar that gets backprop'd.

        L_MLM : CE over the vocab, ONLY at masked positions
        L_NSP : CE over 2 classes (IsNext / NotNext), one per sequence

    The MLM "only at masked positions" is handled by the label convention, NOT by
    gathering: the data pipeline sets every NON-masked label to -100, and
    `ignore_index=-100` makes cross_entropy skip those — they contribute nothing to
    the loss and nothing to the mean's denominator. So ~85% of positions (the
    unmasked ones) are free supervision-wise; only the ~15% masked tokens are scored.

    NSP has no ignore_index: every sequence in the batch has a real 0/1 label.

    Args:
        ignore_index (int): Label value marking MLM positions to skip.
            Default: -100 (PyTorch's cross_entropy default; what the masker writes
            at non-masked positions).

    Returns (from forward): (total_loss, mlm_loss, nsp_loss) — the two parts are
    returned alongside the sum so the training loop can log them separately.
    """
    def __init__(self, ignore_index: int=-100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(
            self,
            mlm_logits: torch.Tensor,
            nsp_logits: torch.Tensor,
            mlm_labels: torch.Tensor,
            nsp_labels: torch.Tensor
    ):
        """
        Args:
            mlm_logits: (batch_size, seq_len, vocab_size) — from MLMHead.
            nsp_logits: (batch_size, 2)                   — from NSPHead.
            mlm_labels: (batch_size, seq_len) — original token id at masked
                positions, `ignore_index` (-100) everywhere else.
            nsp_labels: (batch_size,)         — 0 = IsNext, 1 = NotNext.

        Returns:
            total_loss (scalar), mlm_loss (scalar), nsp_loss (scalar)
        """
        vocab_size = mlm_logits.size(-1)

        # Flatten the per-token axis: CE wants (N, C) logits vs (N,) targets.
        # (B, S, V) -> (B*S, V),  (B, S) -> (B*S,)
        mlm_loss = F.cross_entropy(
            mlm_logits.view(-1, vocab_size),
            mlm_labels.view(-1),
            ignore_index=self.ignore_index
        )

        # NSP is already (B, 2) vs (B,) — no reshape needed.
        nsp_loss = F.cross_entropy(nsp_logits, nsp_labels)

        total_loss = mlm_loss + nsp_loss

        return total_loss, mlm_loss, nsp_loss