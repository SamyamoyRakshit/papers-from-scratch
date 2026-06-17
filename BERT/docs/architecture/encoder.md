# BERT Encoder

> Module: [`BERT/models/encoder.py`](../../models/encoder.py) — `EncoderLayer`, `Encoder`
> Paper: Devlin et al. 2019, [*BERT*](https://arxiv.org/abs/1810.04805), §3 (Model Architecture)

BERT's body **is** the original Transformer encoder stack. The full block mechanics
(scaled dot-product attention, multi-head split, residual + post-LN, the Add & Norm
pattern) are already documented for our Transformer — see
[`transformer/docs/architecture/encoder.md`](../../../transformer/docs/architecture/encoder.md).
This page only covers **what BERT changes**.

## Contents

- [What's reused vs. what changes](#whats-reused-vs-what-changes)
- [One layer (post-LN)](#one-layer-post-ln)
- [Bidirectional ≠ special block](#bidirectional--special-block)
- [Sizes](#sizes)
- [References](#references)

---

## What's reused vs. what changes

| Component | Source | BERT-specific? |
|---|---|---|
| `MultiHeadAttention` | imported from `transformer/` | reused unchanged |
| `LayerNorm` | imported from `transformer/` | reused, but **ε = 1e-12** (not 1e-5) |
| Feed-forward network | local [`feed_forward.py`](../../models/modules/feed_forward.py) | **GELU** instead of ReLU |
| Block structure (post-LN) | same as Transformer | no |
| Masking | padding mask only | no — see below |

So inside the encoder there is exactly **one architectural change** — ReLU → GELU in the
FFN ([§A.2](../modules/feed_forward.md#2-the-only-real-change-relu--gelu)) — plus one
implementation detail, the LayerNorm ε of `1e-12` threaded through to match
[`BERTEmbeddings`](../../models/modules/embeddings.py).

## One layer (post-LN)

```mermaid
flowchart TB
    x([x : B, S, d_model]) --> attn[Multi-Head Self-Attention<br/>Q = K = V = x]
    attn --> d1[Dropout] --> add1((+))
    x --> add1
    add1 --> ln1[LayerNorm ε=1e-12]
    ln1 --> ff[FeedForward<br/>GELU] --> d2[Dropout] --> add2((+))
    ln1 --> add2
    add2 --> ln2[LayerNorm ε=1e-12]
    ln2 --> out([out : B, S, d_model])

    style x fill:#eef,stroke:#99d,color:#000
    style out fill:#dfd,stroke:#9d9,color:#000
    style attn fill:#ffd,stroke:#dd9,color:#000
    style ff fill:#ffd,stroke:#dd9,color:#000
    style ln1 fill:#fdd,stroke:#f99,color:#000
    style ln2 fill:#fdd,stroke:#f99,color:#000
    style d1 fill:#fff,stroke:#999,color:#000
    style d2 fill:#fff,stroke:#999,color:#000
    style add1 fill:#fff,stroke:#999,color:#000
    style add2 fill:#fff,stroke:#999,color:#000
```

where **B** = batch size, **S** = sequence length, **d_model** = hidden dimension (768 for
BERT-base). Each sub-layer is `LayerNorm(x + Dropout(Sublayer(x)))`. Shape `(B, S, d_model)`
is preserved end to end, which is what lets the residual connections work and what lets `N`
of these stack.

## Bidirectional ≠ special block

The self-attention here is `Q = K = V = src` with a **padding mask only** — every token
attends to every other token in both directions. This is identical to the Transformer
*encoder*; it is **not** a BERT invention.

What makes BERT "bidirectional" is that it is **encoder-only**: no layer anywhere applies
a causal mask. The contrast is with **GPT** / the Transformer *decoder*, which mask future
tokens so each position only sees the past. BERT drops the decoder entirely, so context
flows both ways at every layer — the property the MLM objective is built to exploit.

## Sizes

| | layers (N) | d_model | heads | d_ff |
|---|---|---|---|---|
| BERT-base | 12 | 768 | 12 | 3072 |
| BERT-large | 24 | 1024 | 16 | 4096 |

Embeddings (token + segment + position) are applied **before** the stack; this module only
runs the `N` layers.

### Where BERT-base's ~110M parameters live

`encoder.py` builds **only the encoder body**, so its param count won't match the "110M"
usually quoted for BERT-base. That figure is the body **plus** the embedding tables, which
live in a separate module:

| Part | Built in | Params (BERT-base) |
|---|---|---|
| Encoder body — 12 layers (attention + FFN + LayerNorms) | [`encoder.py`](../../models/encoder.py) | ~85M |
| Token embedding table — vocab × d_model = 30522 × 768 | [`embeddings.py`](../../models/modules/embeddings.py) | ~23.4M |
| Position (512 × 768) + segment (2 × 768) tables | [`embeddings.py`](../../models/modules/embeddings.py) | ~0.4M |
| **Total** | assembled in `bert.py` | **~110M** |

So almost the entire embedding cost is the token table; the encoder body is the larger
share. The two are combined in `bert.py`.

**How the ~85M body is derived** — per layer (d_model=768, d_ff=3072):

| Sub-part | Formula | Params |
|---|---|---|
| Attention — Q, K, V, output projections | 4 × (768×768 + 768) | 2,362,368 |
| FFN — two linears (768→3072, 3072→768) | (768×3072 + 3072) + (3072×768 + 768) | 4,722,432 |
| 2 × LayerNorm (γ, β) | 2 × (768 + 768) | 3,072 |
| **Per layer** | | **7,087,872** |

× 12 layers = **85,054,464** — exactly the count the smoke test prints.

## References

- **Paper:** Devlin et al. 2019, [*BERT*](https://arxiv.org/abs/1810.04805) — §3 (architecture), §A.2 (GELU)
- **Transformer encoder (full mechanics):** [`transformer/docs/architecture/encoder.md`](../../../transformer/docs/architecture/encoder.md)
- **Feed-forward (the GELU change):** [`feed_forward.md`](../modules/feed_forward.md)
- **Embeddings (the ε=1e-12 source):** [`embeddings.md`](../modules/embeddings.md)
