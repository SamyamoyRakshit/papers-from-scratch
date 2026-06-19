import torch
import torch.nn as nn

# Reuse the same GELU (tanh form) used in the encoder FFN, and the transformer
# LayerNorm — the MLM transform uses both. Run from repo root so `transformer`
# resolves as a package.
from transformer.models.modules.layer_norm import LayerNorm
from .modules.feed_forward import gelu


class MLMHead(nn.Module):
    """
    Masked Language Model head (BERT §3.1, "Task #1: Masked LM").

    Reads the per-token `sequence_output` and predicts, at each masked position,
    a distribution over the whole WordPiece vocabulary. Two stages:

        1. Transform:    Linear(d_model→d_model) → GELU → LayerNorm
        2. Un-embedding: Linear(d_model→vocab_size) — produces logits per token

    Note: BERT is encoder-only — there is no Transformer decoder here. Stage 2 is
    just a single linear that "un-embeds": it maps a 768-vector back to vocab logits,
    the reverse of the embedding lookup.

    Weight tying: this un-embedding (vocab_size × d_model) matrix is the SAME tensor
    as the token-embedding table, so the model uses one set of weights to map
    ids→vectors and vectors→ids. This is in the original Transformer (§3.4,
    "we share the same weight matrix between the two embedding layers and the
    pre-softmax linear transformation") and kept by Google's TF BERT
    (`get_masked_lm_output`) and HF's `BertLMPredictionHead`. The output bias is
    a SEPARATE learned vector (one per vocab token), not tied to anything.

    The transform's activation/LayerNorm mirror the encoder: GELU (§A.2) and
    ε = 1e-12 (matching BERTEmbeddings / the encoder layers).

    Args:
        d_model (int): Hidden dimension. Example (BERT-base): 768
        embedding_weight (torch.Tensor): The token-embedding weight to tie the
            decoder (un-embedding projection) to, shape (vocab_size, d_model) —
            pass `model.embeddings.token_embedding.weight`.
        layer_norm_eps (float): ε for the transform LayerNorm. Default: 1e-12
    """
    def __init__(
            self,
            d_model: int,
            embedding_weight: torch.Tensor,
            layer_norm_eps: float = 1e-12
    ):
        super().__init__()
        vocab_size = embedding_weight.size(0)

        # Stage 1: transform (dense → GELU → LayerNorm)
        self.dense = nn.Linear(d_model, d_model)
        self.layer_norm = LayerNorm(d_model, layer_norm_eps)

        # Stage 2: decode to vocab logits. bias=False because the tied weight
        # supplies the matrix; we add our own (separate) output bias below.
        self.decoder = nn.Linear(d_model, vocab_size, bias=False)
        self.decoder.weight = embedding_weight          # tie to token embeddings
        self.bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sequence_output: Per-token encoder states, (batch_size, seq_len, d_model).
                Typically only the masked positions are gathered before/after this,
                but the head runs on the full sequence.

        Returns:
            Vocabulary logits, shape (batch_size, seq_len, vocab_size).
        """
        x = self.dense(sequence_output)
        x = gelu(x)
        x = self.layer_norm(x)
        logits = self.decoder(x) + self.bias
        return logits
    

class NSPHead(nn.Module):
    """
    Next Sentence Prediction head (BERT §3.1, "Task #2: Next Sentence Prediction").

    Reads the single `pooled_output` ([CLS] after the pooler) and does a 2-way
    classification: IsNext (0) vs NotNext (1). Just one Linear(d_model→2) — the
    pooler already did the Linear+Tanh, so the head itself is a bare projection.
    Matches Google's `get_next_sentence_output` and HF's `BertOnlyNSPHead`.

    Args:
        d_model (int): Hidden dimension. Example (BERT-base): 768
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.classifier = nn.Linear(d_model, 2)

    def forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pooled_output: [CLS] summary vector, shape (batch_size, d_model).

        Returns:
            NSP logits, shape (batch_size, 2)  — [IsNext, NotNext].
        """
        return self.classifier(pooled_output)
    

class BERTPreTrainingHeads(nn.Module):
    """
    Both pre-training heads bundled together (matches HF's `BertPreTrainingHeads`).

    Wraps MLMHead + NSPHead so the pre-training model can call them in one go:
    MLM reads `sequence_output`, NSP reads `pooled_output`.

    Args:
        d_model (int): Hidden dimension. Example (BERT-base): 768
        embedding_weight (torch.Tensor): Token-embedding weight to tie the MLM
            decoder to (see MLMHead).
        layer_norm_eps (float): ε for the MLM transform LayerNorm. Default: 1e-12
    """
    def __init__(
            self,
            d_model: int,
            embedding_weight: torch.Tensor,
            layer_norm_eps: float = 1e-12
    ):
        super().__init__()
        self.mlm = MLMHead(d_model, embedding_weight, layer_norm_eps)
        self.nsp = NSPHead(d_model)

    def forward(
            self,
            sequence_output: torch.Tensor,
            pooled_output: torch.Tensor
    ):
        """
        Returns:
            mlm_logits: (batch_size, seq_len, vocab_size)
            nsp_logits: (batch_size, 2)
        """
        mlm_logits = self.mlm(sequence_output)
        nsp_logits = self.nsp(pooled_output)
        return mlm_logits, nsp_logits