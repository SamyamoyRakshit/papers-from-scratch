import torch
import torch.nn as nn

# LayerNorm is imported from transformer/ — apple-to-apple, no change.
# Run from repo root so `transformer` resolves as a package.
from transformer.models.modules.layer_norm import LayerNorm


class BERTEmbeddings(nn.Module):
    """
    BERT Input Embeddings (Section 3.1, Figure 2)

    The input representation is the SUM of three embeddings:

        E = TokenEmbedding(ids) + SegmentEmbedding(seg) + PositionEmbedding(pos)

    Only the three-way SUM is from the paper (Section 3.1, Figure 2). The
    LayerNorm + dropout applied afterward are an IMPLEMENTATION detail, not in
    the paper text — they come from Google's TF BERT (the `embedding_postprocessor`
    function in modeling.py) and are mirrored in HF's `BertEmbeddings`.

    There is also NO √d_model scaling here:
      - The scaling exists in the original Transformer — "Attention Is All You
        Need", Section 3.4: "we multiply those [embedding] weights by √d_model"
        (our transformer/models/modules/embeddings.py does this).
      - BERT omits it: Google's TF BERT (`embedding_lookup` in modeling.py) and
        HF's `BertEmbeddings.forward` both feed the raw summed embeddings into
        LayerNorm with no √d_model factor.
    So the three tables are learned and used directly, unscaled.

    Three tables:
        1. Token     — WordPiece id  → vector   (the word itself)
        2. Segment   — 0 / 1          → vector   (sentence A vs sentence B, for NSP)
        3. Position  — 0 .. seq_len-1 → vector   (LEARNED, not sinusoidal)

    Args:
        vocab_size (int): WordPiece vocabulary size.
            Example (BERT-base): 30522
        d_model (int): Embedding / hidden dimension.
            Example (BERT-base): 768
        max_position_embeddings (int): Longest sequence the model can handle.
            Position table has this many rows. Example (BERT-base): 512
        num_segments (int): Number of segment types. BERT uses 2 (sentence A, B).
            Default: 2
        pad_idx (int): Token id of [PAD]. Its token-embedding row is held at zero
            and excluded from gradient updates via padding_idx.
        dropout (float, optional): Dropout on the summed embeddings. Default: 0.1.
        layer_norm_eps (float, optional): ε for LayerNorm. Default: 1e-12.
            Source: Google's TF BERT used tf.contrib.layers.layer_norm, whose
            default epsilon is 1e-12; HF kept it as BertConfig.layer_norm_eps.
            (For contrast: our transformer/ replication uses 1e-5 = PyTorch's
            nn.LayerNorm default; Tensor2Tensor's original-Transformer layer_norm
            used 1e-6. Three conventions, none specified by any paper.)

    References:
        - Paper: Devlin et al. 2019, "BERT: Pre-training of Deep Bidirectional
          Transformers for Language Understanding" — https://arxiv.org/abs/1810.04805
        - Official Google BERT (TF) — embedding_lookup / embedding_postprocessor:
          https://github.com/google-research/bert/blob/master/modeling.py
        - HF Transformers — BertEmbeddings (PyTorch), pinned to v5.12.0:
          https://github.com/huggingface/transformers/blob/v5.12.0/src/transformers/models/bert/modeling_bert.py
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_position_embeddings: int = 512,
        num_segments: int = 2,
        pad_idx: int = 0,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-12
    ):
        super().__init__()

        # padding_idx keeps the [PAD] row at zero and frozen (no gradient).
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=pad_idx
        )

        self.segment_embedding = nn.Embedding(
            num_embeddings=num_segments,
            embedding_dim=d_model
        )

        self.positional_embedding = nn.Embedding(
            num_embeddings=max_position_embeddings,
            embedding_dim=d_model
        )

        self.layer_norm = LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

        # position_ids = [0, 1, 2, ..., max_position_embeddings-1], shape (1, max_pos).
        # Registered as a buffer (persistent=False): moves with .to(device) and
        # is NOT a learnable parameter — we just slice the first seq_len of it
        # each forward, so we never rebuild torch.arange on every step.
        self.register_buffer(
            name="position_ids",
            tensor=torch.arange(max_position_embeddings).unsqueeze(0),
            persistent=False
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: WordPiece token ids, shape (batch_size, seq_len)
            token_type_ids: Segment ids (0 for sentence A, 1 for sentence B),
                shape (batch_size, seq_len). If None (single-sentence tasks),
                defaults to all zeros → every token treated as segment A.

        Returns:
            Embeddings, shape (batch_size, seq_len, d_model)
        """
        seq_len = input_ids.size(1)

        # Single-sentence input → all segment 0.
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # Slice the first seq_len positions: (1, seq_len) — broadcasts over batch.
        position_ids = self.position_ids[:, :seq_len]

        # Look up all three tables and sum.
        # Each: (batch_size, seq_len, d_model)
        token_emb = self.token_embedding(input_ids)
        segment_emb = self.segment_embedding(token_type_ids)
        position_emb = self.positional_embedding(position_ids)

        embeddings = token_emb + segment_emb + position_emb

        # LayerNorm then dropout, exactly as in BertEmbeddings.
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings