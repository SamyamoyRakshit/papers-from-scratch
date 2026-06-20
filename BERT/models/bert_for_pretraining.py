import torch
import torch.nn as nn

from .bert import BERTModel
from .heads import BERTPreTrainingHeads


class BERTForPreTraining(nn.Module):
    """
    BERT with both pre-training heads attached (Devlin et al. 2019, §3.1).

    Composition only — owns the body + heads and wires them:

        input_ids ─► BERTModel ─► (sequence_output, pooled_output)
                                       │              │
                                  MLM head        NSP head
                                       │              │
                                  mlm_logits     nsp_logits

    Weight tying: the MLM head's un-embedding matrix is tied to the token
    embedding table. The tie is made HERE, after BERTModel.__init__ has already
    run its own `self.apply(self._init_weights)` — so we hand the *already-init'd*
    embedding weight to the head. Tying after init is what keeps the two layers
    sharing one tensor (re-applying init at this level would break it).

    The heads (~624k params) are DISCARDED after pre-training; only the body
    (the famous ~110M) is kept for fine-tuning.

    Args mirror BERTModel; see `bert.py` for the full list.
    """
    def __init__(
            self,
            vocab_size: int,
            d_model: int,
            num_heads: int,
            d_ff: int,
            num_layers: int,
            max_position_embeddings: int = 512,
            num_segments: int = 2,
            pad_idx: int = 0,
            dropout: float = 0.1,
            layer_norm_eps: float = 1e-12,
            initial_range: float = 0.02
    ):
        super().__init__()
        self.bert = BERTModel(
            vocab_size,
            d_model,
            num_heads,
            d_ff,
            num_layers,
            max_position_embeddings,
            num_segments,
            pad_idx,
            dropout,
            layer_norm_eps,
            initial_range
        )

        # The MLM decoder weight is tied to the token table inside the heads'
        # constructor (BERTModel already ran its own init, so we hand it the
        # already-init'd embedding weight).
        self.heads = BERTPreTrainingHeads(
            d_model=d_model,
            embedding_weight=self.bert.embeddings.token_embedding.weight,
            layer_norm_eps=layer_norm_eps
        )

        # BERTModel's init ran before the heads existed, so the heads' OWN params
        # still sit at PyTorch defaults. Apply BERT's scheme to just the transform
        # dense and the NSP classifier — reusing bert._init_weights so the formula
        # stays single-sourced. The tied MLM decoder is deliberately NOT touched
        # (re-init would re-randomize the shared table and un-zero its [PAD] row);
        # the MLM head's separate bias keeps its zero init.
        self.bert._init_weights(self.heads.mlm.dense)
        self.bert._init_weights(self.heads.nsp.classifier)

    def forward(
            self,
            input_ids: torch.Tensor,
            token_type_ids: torch.Tensor = None
    ):
        """
        Args:
            input_ids: WordPiece ids, (batch_size, seq_len)
            token_type_ids: Segment ids (0=A, 1=B), (batch_size, seq_len)

        Returns:
            mlm_logits: (batch_size, seq_len, vocab_size)
            nsp_logits: (batch_size, 2)
        """
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids)
        mlm_logits, nsp_logits = self.heads(sequence_output, pooled_output)

        return mlm_logits, nsp_logits