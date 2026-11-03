import torch
import torch.nn as nn


class GPTEmbeddings(nn.Module):
    """
    GPT Input Embeddings (GPT-1 §3.1, Eq. 2; GPT-2 §2.3 "largely follows" GPT-1)

        h_0 = U·W_e + W_p        (GPT-1 Eq. 2)

    Token embedding + learned position embedding, summed, then dropout.
    The two tables are named wte / wpe, as in OpenAI's checkpoint.

    vs the two siblings:
      - no ×√d_model: transformer/ scales by it (Vaswani §3.4); GPT does not.
      - LEARNED positions, not sinusoidal. GPT-1 §4.1: "We used learned position
        embeddings instead of the sinusoidal version proposed in the original work."
      - no segment table: one stream of text, no sentence A/B (BERT's 3-way sum → 2-way).
      - no LayerNorm after the sum: BERT has one. In GPT-2's pre-LN, the first LN
        is ln1 inside block 0.
      - no padding_idx: get_batch cuts fixed windows from one long id stream,
        so there is no [PAD] row to freeze.

    Init is NOT done here. nn.Embedding defaults to N(0, 1); gpt.py re-inits both
    tables to N(0, 0.02) and ties lm_head.weight to wte.weight (Eq. 2's
    P(u) = softmax(h_n·W_eᵀ) reuses W_e).

    Args:
        vocab_size (int): base.yaml 8192 → wte is 8192 × 256 = 2,097,152 params.
        d_model (int): base.yaml 256.
        block_size (int): base.yaml 256 → wpe is 256 × 256 = 65,536 params.
            wpe has rows 0..255, so position 256 has no vector.
        dropout (float): GPT-1 §4.1: 0.1 on embeddings.
    """

    def __init__(self, vocab_size: int, d_model: int, block_size: int, dropout: float):
        super().__init__()
        self.block_size = block_size
        self.wte = nn.Embedding(vocab_size, d_model)    # token id → vector
        self.wpe = nn.Embedding(block_size, d_model)    # position 0..block_size-1 → vector
        self.dropout = nn.Dropout(dropout)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            idx: token ids x from get_batch, shape (batch_size, T), int64
        Returns:
            (batch_size, T, d_model)
        """
        T = idx.size(1)
        # Training always has T = block_size. Generation grows T and must crop to the
        # last block_size ids; at T=257, wpe would be asked for row 256, which doesn't exist.
        assert T <= self.block_size, f"sequence length {T} > block_size {self.block_size}"

        # pos = [0, 1, ..., T-1]. Built each call on idx's device (microseconds),
        # instead of BERT's position_ids buffer.
        pos = torch.arange(T, device=idx.device)

        tok_emb = self.wte(idx)     # (B, T, d_model)
        pos_emb = self.wpe(pos)     # (T, d_model), broadcasts over B
        return self.dropout(tok_emb + pos_emb)
