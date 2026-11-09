import math

import torch
import torch.nn as nn

# LayerNorm is imported from transformer/ unchanged, as in decoder.py.
# Run from repo root so `transformer` resolves.
from transformer.models.modules.layer_norm import LayerNorm

from .modules.embeddings import GPTEmbeddings
from .decoder import Decoder


class GPT(nn.Module):
    """
    GPT-2 language model (GPT-2 §2.3)

        idx ─► GPTEmbeddings ─► Decoder (6 pre-LN blocks) ─► ln_f ─► lm_head ─► logits
       (B, T)   (B, T, 256)         (B, T, 256)          (B, T, 256)    (B, T, 8192)

    logits[b, t] scores all 8192 ids as the token AFTER idx[b, t]; loss.py compares
    it to y[b, t] (get_batch already shifted y by one).

    One model, one head: no BERT-style body / heads / for_pretraining split.

    ln_f: GPT-2 §2.3 "an additional layer normalization was added after the final
    self-attention block". In pre-LN the last block's add is never normalized; ln_f does it.

    lm_head: Linear(256 → 8192), no bias, weight TIED to wte (GPT-1 Eq. 2:
    P(u) = softmax(h_n·W_eᵀ)). One 8192 × 256 table reads tokens in and scores
    them out → saves 2,097,152 params.

    Init is done here, not in the modules:
      - every Linear / Embedding weight: N(0, init_std). GPT-1 §4.1: "a simple weight
        initialization of N(0, 0.02) was sufficient". Plain normal, not BERT's
        truncated one. Biases: 0. LayerNorm keeps gamma=1, beta=0.
      - attn.c_proj and mlp.c_proj: N(0, init_std / √(2·num_layers)). GPT-2 §2.3:
        "We scale the weights of residual layers at initialization by a factor of
        1/√N where N is the number of residual layers". 2 adds per block → N = 12
        for base: 0.02/√12 = 0.0058, so 12 adds onto x sum to roughly the size of one.

    Args (names match ModelConfig, so: GPT(config.data.vocab_size, **config.model.model_dump())):
        vocab_size (int): config.data.vocab_size, base.yaml 8192.
        num_layers, d_model, num_heads, d_ff, block_size, dropout, attention_impl: config.model.
        init_std (float): config.model.init_std, GPT-1 §4.1: 0.02.

    Params (tied lm_head counts 0):
        base.yaml 6,901,760    tiny.yaml 169,728    GPT-2 124M 124,439,808
    """

    def __init__(self, vocab_size: int, num_layers: int, d_model: int, num_heads: int, d_ff: int,
                 block_size: int, dropout: float, attention_impl: str, init_std: float):
        super().__init__()
        self.init_std = init_std
        self.embeddings = GPTEmbeddings(vocab_size, d_model, block_size, dropout)
        self.decoder = Decoder(d_model, num_heads, d_ff, num_layers, dropout, attention_impl)
        self.ln_f = LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embeddings.wte.weight    # tie: one tensor, two names

        # Init MUST come after every module is built (BERT's rule): apply() only reaches
        # modules that exist. The tie survives: nn.init writes in place.
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):    # attn.c_proj + mlp.c_proj, all 6 blocks
                nn.init.normal_(p, mean=0.0, std=init_std / math.sqrt(2 * num_layers))

    def _init_weights(self, module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.init_std)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            idx: token ids x from get_batch, shape (batch_size, T), int64
        Returns:
            logits: (batch_size, T, vocab_size), raw scores before softmax.
                No loss here: loss.py takes (logits, y), as in BERT.
        """
        x = self.embeddings(idx)    # (B, T, 256)
        x = self.decoder(x)         # (B, T, 256)  6 pre-LN blocks
        x = self.ln_f(x)            # (B, T, 256)
        return self.lm_head(x)      # (B, T, 8192) = x · wteᵀ
