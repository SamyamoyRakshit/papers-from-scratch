import torch
import torch.nn as nn

# Encoder reuses transformer's MHA + LayerNorm; embeddings are BERT-local.
# Run from repo root so `transformer` resolves as a package.
# create_padding_mask is reused from transformer/ too — it already returns the
# (B, 1, 1, S) mask we need. Run from repo root so `transformer` resolves.
from transformer.utils.mask_utils import create_padding_mask
from .modules.embeddings import BERTEmbeddings
from .encoder import Encoder


class BERTModel(nn.Module):
    """
    BERT base model (Devlin et al. 2019) — the encoder-only Transformer body.

    Pipeline:
        input_ids ─┐
        token_type ┼─► BERTEmbeddings ─► Encoder (N layers) ─► sequence_output
        positions ─┘                                             │
                                                    [CLS] token ─┴─► Pooler ─► pooled_output

    Returns BOTH outputs, matching HF's BertModel:
        - sequence_output: (B, S, d_model) — per-token states; the MLM head reads this.
        - pooled_output:   (B, d_model)    — [CLS] state pushed through a Linear+Tanh
          "pooler"; the NSP / sentence-classification head reads this.

    Weight init: all weights drawn from a Truncated Normal(0, 0.02) cut at ±2σ.
    This `initializer_range = 0.02` is NOT in the paper text — it's the default in
    Google's TF BERT `modeling.py` (BertConfig), used there via a
    `truncated_normal_initializer(stddev=0.02)`. So it's an implementation detail.
    (HF instead uses a plain, untruncated normal; we follow Google's original.)

    Args:
        vocab_size (int): WordPiece vocab. Example (BERT-base): 30522
        d_model (int): Hidden dimension. Example (BERT-base): 768
        num_heads (int): Attention heads. Example (BERT-base): 12
        d_ff (int): FFN inner dimension. Example (BERT-base): 3072
        num_layers (int): Number of encoder layers (N). Example (BERT-base): 12
        max_position_embeddings (int): Longest sequence. Default: 512
        num_segments (int): Segment types (sentence A/B). Default: 2
        pad_idx (int): [PAD] token id. Used to build the padding mask. Default: 0
        dropout (float): Dropout probability. Default: 0.1
        layer_norm_eps (float): ε for LayerNorm. Default: 1e-12 (BERT's value)
        initializer_range (float): std for the truncated-normal weight init. Default: 0.02
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
        self.pad_idx = pad_idx
        self.initial_range = initial_range

        self.embeddings = BERTEmbeddings(
            vocab_size,
            d_model,
            max_position_embeddings,
            num_segments,
            pad_idx,
            dropout,
            layer_norm_eps
        )

        self.encoder = Encoder(
            d_model,
            num_heads,
            d_ff,
            num_layers,
            dropout,
            layer_norm_eps
        )

        # Pooler: take the [CLS] state, project, squash with tanh. Used for
        # sentence-level tasks (NSP / classification), not per-token tasks (MLM).
        self.pooler = nn.Linear(d_model, d_model)
        self.pooler_activation = nn.Tanh()

        # Apply truncated-normal init to every submodule. MUST stay the last line
        # of __init__: self.apply() recursively overrides every submodule's own
        # init (e.g. the MHA's xavier_uniform) with this one. Any module built
        # after this line would keep its default init and miss the 0.02 override.
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Truncated Normal(0, initializer_range), cut at ±2σ — matching Google's
        TF `truncated_normal_initializer(stddev=0.02)`; biases zeroed.

        Note: PyTorch's `trunc_normal_` defaults to a=-2, b=2 as ABSOLUTE values,
        so we pass a=-2σ, b=2σ explicitly to truncate at two standard deviations.
        The transformer LayerNorm already constructs gamma=1, beta=0, so it needs
        no special handling here.
        """
        std = self.initial_range
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(tensor=module.weight, mean=0.0, std=std, a=-2*std, b=2*std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(tensor=module.weight, mean=0.0, std=std, a=-2*std, b=2*std)
            # Keep the [PAD] row at zero (padding_idx is frozen anyway).
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(
            self,
            input_ids: torch.Tensor,
            token_type_ids: torch.Tensor = None
    ):
        """
        Args:
            input_ids: WordPiece ids, shape (batch_size, seq_len)
            token_type_ids: Segment ids (0=A, 1=B), shape (batch_size, seq_len).
                None → all zeros (single-sentence input).

        Returns:
            sequence_output: (batch_size, seq_len, d_model)
            pooled_output:   (batch_size, d_model)
        """
        # Padding mask (B, 1, 1, S) — reused from transformer/. 1 = keep, 0 = pad.
        attn_mask = create_padding_mask(input_ids, self.pad_idx)

        embeddings = self.embeddings(input_ids, token_type_ids)
        sequence_output = self.encoder(embeddings, attn_mask)

        # Pool the [CLS] token (position 0).
        pooled_output = self.pooler_activation(self.pooler(sequence_output[:, 0]))

        return sequence_output, pooled_output