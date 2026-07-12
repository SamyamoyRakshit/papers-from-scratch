import torch
import torch.nn as nn

from .bert import BERTModel


class BERTForSequenceClassification(nn.Module):
    """
    BERT with a single-sentence classification head (Devlin et al. 2019, §4.1 / §3.5).

    The pre-training heads (MLM / NSP) are DISCARDED — only the encoder body +
    pooler carry over. On top of the pooled [CLS] state:

        pooled_output ─► dropout ─► Linear(d_model → num_labels) ─► logits

    §3.5: "the only new parameters introduced during fine-tuning are classification
    layer weights W ∈ R^(K×H)" — that's exactly this classifier.

    Args mirror BERTModel (so the encoder matches the pre-trained checkpoint),
    plus num_labels.
    """
    def __init__(
            self,
            vocab_size: int,
            d_model: int,
            num_heads: int,
            d_ff: int,
            num_layers: int,
            num_labels: int,
            max_position_embeddings: int = 512,
            num_segments: int = 2,
            pad_idx: int = 0,
            dropout: float = 0.1,
            layer_norm_eps: float = 1e-12,
            initial_range: float = 0.02,
    ):
        super().__init__()
        self.bert = BERTModel(
            vocab_size, d_model, num_heads, d_ff, num_layers,
            max_position_embeddings, num_segments, pad_idx, dropout,
            layer_norm_eps, initial_range,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_labels)

        # BERTModel.__init__ already ran its truncated-normal init over the body.
        # The classifier was built after that, so it still sits at PyTorch defaults —
        # apply BERT's scheme to it too (single-sourced via _init_weights).
        self.bert._init_weights(self.classifier)

    def forward(
            self,
            input_ids: torch.Tensor,
            token_type_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: WordPiece ids, (batch_size, seq_len).
            token_type_ids: Segment ids — all zeros for single-sentence input.

        Returns:
            logits: (batch_size, num_labels).
        """
        _, pooled_output = self.bert(input_ids, token_type_ids)
        return self.classifier(self.dropout(pooled_output))