import torch
import torch.nn as nn

from .modules.embeddings import Embeddings
from .modules.positional_encoding import PositionalEncoding
from .encoder import Encoder
from .decoder import Decoder


class Transformer(nn.Module):
    """
    Full Transformer Model (Section 3)

    Combines:
        - Shared token embeddings (Section 3.4)
        - Positional encoding (Section 3.5)
        - Encoder stack (Section 3.1)
        - Decoder stack (Section 3.1)
        - Final linear projection to vocabulary (Section 3.4)

    Weight sharing (Section 3.4):
        "We share the same weight matrix between the two embedding layers
        and the pre-softmax linear transformation."
        Source embedding, target embedding, and output projection all share weights.

    Args:
        src_vocab_size (int): Source vocabulary size
        tgt_vocab_size (int): Target vocabulary size
        d_model (int): Dimension of the model. Example (base): 512
        num_heads (int): Number of attention heads. Example (base): 8
        d_ff (int): Inner dimension of the feed-forward network. Example (base): 2048
        num_layers (int): Number of encoder/decoder layers (N). Example (base): 6
        dropout (float): Dropout probability. Default: 0.1
        max_len (int): Maximum sequence length for positional encoding. Default: 5000
    """
    def __init__(
            self,
            src_vocab_size: int,
            tgt_vocab_size: int,
            d_model: int,
            num_heads: int,
            d_ff: int,
            num_layers: int,
            dropout: float = 0.1,
            max_len: int = 5000
    ):
        super().__init__()

        # Embedding layers (scaled by sqrt(d_model) inside Embeddings)
        self.src_embedding = Embeddings(src_vocab_size, d_model)
        self.tgt_embedding = Embeddings(tgt_vocab_size, d_model)

        # Positional encoding (includes dropout on embedding + PE sum)
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len)

        # Encoder and Decoder stacks
        self.encoder = Encoder(d_model, num_heads, d_ff, num_layers, dropout)
        self.decoder = Decoder(d_model, num_heads, d_ff, num_layers, dropout)

        # Output projection: d_model → tgt_vocab_size
        self.output_projection = nn.Linear(d_model, tgt_vocab_size, bias=False)

        # Weight sharing (Section 3.4):
        # Tie target embedding weights with output projection weights
        self.output_projection.weight = self.tgt_embedding.embeddings.weight

        # If src and tgt share the same vocabulary, tie source embedding too
        if src_vocab_size == tgt_vocab_size:
            self.src_embedding.embeddings.weight = self.tgt_embedding.embeddings.weight

    def forward(
            self,
            src: torch.Tensor,
            tgt: torch.Tensor,
            src_mask: torch.Tensor = None,
            tgt_mask: torch.Tensor = None,
            memory_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            src: Source token IDs, shape (batch_size, src_seq_len)
            tgt: Target token IDs, shape (batch_size, tgt_seq_len)
            src_mask: Padding mask for encoder self-attention,
                      shape (batch_size, 1, 1, src_seq_len)
            tgt_mask: Causal + padding mask for decoder self-attention,
                      shape (batch_size, 1, tgt_seq_len, tgt_seq_len)
            memory_mask: Padding mask for decoder cross-attention,
                         shape (batch_size, 1, 1, src_seq_len)

        Returns:
            Logits over target vocabulary,
            shape (batch_size, tgt_seq_len, tgt_vocab_size)
        """
        # Encode: token IDs → embeddings → + PE → encoder stack
        src_embedded = self.positional_encoding(self.src_embedding(src))
        encoder_output = self.encoder(src_embedded, src_mask)

        # Decode: token IDs → embeddings → + PE → decoder stack
        tgt_embedded = self.positional_encoding(self.tgt_embedding(tgt))
        decoder_output = self.decoder(tgt_embedded, encoder_output, tgt_mask, memory_mask)

        # Project to vocabulary: (batch, tgt_seq_len, d_model) → (batch, tgt_seq_len, tgt_vocab_size)
        logits = self.output_projection(decoder_output)

        return logits

    def run_encoder_stack(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Encoder-only forward — exposed for inference (encode once, reuse `memory` across decode steps).

        Flow: src ids → embed → +PE → encoder stack
        Args:
            src: (batch_size, src_seq_len)
            src_mask: (batch_size, 1, 1, src_seq_len)
        Returns:
            memory: (batch_size, src_seq_len, d_model)
        """
        # embed + positional encoding, then run the encoder stack
        src_embedded = self.positional_encoding(self.src_embedding(src))
        memory = self.encoder(src_embedded, src_mask)
        return memory

    def run_decoder_stack(
            self,
            tgt: torch.Tensor,
            memory: torch.Tensor,
            tgt_mask: torch.Tensor = None,
            memory_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Decoder + output projection — called once per step during autoregressive decoding.

        Flow: tgt ids → embed → +PE → decoder stack (cross-attends to `memory`) → linear → logits
        Args:
            tgt: (batch_size, tgt_seq_len)               — tokens generated so far
            memory: (batch_size, src_seq_len, d_model)   — encoder output, precomputed once
            tgt_mask: (batch_size, 1, tgt_seq_len, tgt_seq_len)  — causal + padding
            memory_mask: (batch_size, 1, 1, src_seq_len) — source padding for cross-attention
        Returns:
            logits: (batch_size, tgt_seq_len, tgt_vocab_size)
        """
        # embed + positional encoding, then decode against the precomputed memory
        tgt_embedded = self.positional_encoding(self.tgt_embedding(tgt))
        decoder_output = self.decoder(tgt_embedded, memory, tgt_mask, memory_mask)
        # project to vocabulary logits (weights tied with target embedding)
        logits = self.output_projection(decoder_output)
        return logits