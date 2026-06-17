# BERT Feed-Forward Network

> Module: [`BERT/models/modules/feed_forward.py`](../../models/modules/feed_forward.py) — `FeedForward`, `gelu`
> Paper: Devlin et al. 2019, [*BERT: Pre-training of Deep Bidirectional Transformers*](https://arxiv.org/abs/1810.04805), §A.2 (Pre-training Procedure)

Inside every encoder layer, after attention has mixed information *across* positions,
the feed-forward network transforms each position *independently*. It's the second of
the two sublayers in a Transformer block, and structurally it's the simplest module in
BERT — two linear layers with a non-linearity between them.

## Contents

1. [What it computes](#1-what-it-computes)
2. [The only real change: ReLU → GELU](#2-the-only-real-change-relu--gelu)
3. [Two faces of GELU: exact vs tanh](#3-two-faces-of-gelu-exact-vs-tanh)
4. [Why d_ff is 4·d_model](#4-why-d_ff-is-4d_model)
5. [A worked example](#5-a-worked-example)
- [References](#references)

---

## 1. What it computes

For each position independently:

```
FFN(x) = GELU(x·W₁ + b₁)·W₂ + b₂

         x:  (B, S, d_model)
              │  linear1: d_model → d_ff
              ▼
             (B, S, d_ff)        ← widened
              │  GELU, then dropout
              ▼
             (B, S, d_model)     ← projected back  (linear2: d_ff → d_model)
```

"Position-wise" means the **same** two linear layers are applied to every token vector
separately — there is no mixing across positions here (that already happened in
attention). It's just a per-token MLP: widen to `d_ff`, apply the non-linearity, project
back to `d_model`.

This is the **identical structure** to the original Transformer's FFN
([*Attention Is All You Need*](https://arxiv.org/abs/1706.03762), Section 3.3) — and to
our own [`transformer/models/modules/feed_forward.py`](../../../transformer/models/modules/feed_forward.py).
BERT changes exactly one thing.

---

## 2. The only real change: ReLU → GELU

The Transformer uses ReLU: `FFN(x) = max(0, xW₁+b₁)W₂+b₂`. BERT swaps in **GELU**.

Unlike most of the embeddings-module details (LayerNorm, no √d_model, ε = 1e-12 — all
*implementation* choices the paper never mentions), this swap is **stated in the paper**.
§A.2, Pre-training Procedure:

> *"We use a gelu activation (Hendrycks and Gimpel, 2016) rather than the standard relu,
> following OpenAI GPT."*

So this is a rare case where the docstring can cite the paper directly rather than an
implementation.

**What GELU does.** Where ReLU is a hard gate (`max(0, x)` — kill everything negative),
GELU is a *soft* gate. It weights each input by the probability that a standard-normal
variable is ≤ x:

```
gelu(x) = x · Φ(x)        Φ = the Gaussian CDF
```

Small negatives get partially passed through instead of zeroed, and the function is
smooth everywhere (ReLU has a kink at 0). `gelu(0) = 0`, like ReLU, but the transition
is gradual.

---

## 3. Two faces of GELU: exact vs tanh

There are two ways to compute GELU, and which one you pick is a genuine
faithfulness decision — not a cosmetic one.

**Exact (erf) form** — what `Φ` actually is:

```
gelu(x) = 0.5 · x · (1 + erf( x / √2 ))
```

**Tanh approximation** — what Google's original TF BERT wrote in `modeling.py`:

```
gelu(x) ≈ 0.5 · x · (1 + tanh[ √(2/π) · (x + 0.044715·x³) ])
```

Our module computes the **tanh form**, by hand, to match the original BERT code:

```python
return 0.5 * x * (1.0 + torch.tanh(
    math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
))
```

These two functions are *close but not identical*. That matters because of a subtle
mismatch in the reference implementations:

| Choice | Function | Faithful to |
|---|---|---|
| `gelu` (this module) / `nn.GELU(approximate='tanh')` | tanh approximation | **original Google TF BERT weights** |
| `nn.GELU()` (PyTorch default) | exact erf | **HF**, which maps `hidden_act="gelu"` to exact erf |

So HF's `"gelu"` is *not* the same function as Google's original `gelu` — in HF the tanh
version is called `"gelu_new"`. We chose the tanh form to stay true to the paper's
original implementation; if you were loading HF's pretrained checkpoints you'd want
`nn.GELU()` instead. Either is defensible — the choice just has to be deliberate.

> **Verification:** our hand-written `gelu` matches `torch.nn.functional.gelu(x,
> approximate='tanh')` to within floating-point tolerance (`torch.allclose` → True), so
> the manual formula is provably the same function PyTorch ships as the tanh variant.

### A note on the float literals

The constants are written with a trailing `.0` (`1.0`, `2.0`, `3.0`). The `.0` is
**purely stylistic** — `x` is a float tensor, so `math.sqrt(2)` and `torch.pow(x, 3)`
promote to float and give identical results. It just signals "real-valued math
constant." (Google's original wasn't this uniform — they wrote `1.0` but left `2 / np.pi`
and `pow(x, 3)` as bare ints; we keep all of them floats for consistency.) Drop the
`.0`s and nothing changes numerically.

---

## 4. Why d_ff is 4·d_model

BERT-base uses `d_model = 768` and `d_ff = 3072` — exactly **4×**. This 4:1 ratio is
inherited from the original Transformer (512 → 2048) and is the standard FFN expansion
ratio across the whole Transformer family.

The intuition: attention is relatively cheap at mixing information but limited in
*per-token* transformation capacity. The FFN is where most of a Transformer's
parameters and "thinking" live — widening to 4× gives the non-linearity room to compute
richer per-position features before compressing back down to `d_model` for the next
layer.

Like every other dimension in this replication, `d_model` and `d_ff` are plain `int`
arguments here and get supplied from YAML → config, not hard-coded in the module.

---

## 5. A worked example

```python
ff = FeedForward(d_model=768, d_ff=3072, dropout=0.1)

x   = torch.randn(2, 8, 768)   # (batch=2, seq_len=8, d_model=768)
out = ff(x)                    # (2, 8, 768)  ← same shape in and out
```

Step by step inside `forward`:

```python
x = self.linear1(x)   # (2, 8, 768) → (2, 8, 3072)   widen
x = gelu(x)           # (2, 8, 3072)                  soft non-linearity
x = self.dropout(x)   # (2, 8, 3072)                  regularize
x = self.linear2(x)   # (2, 8, 3072) → (2, 8, 768)    project back
```

The module is **shape-preserving**: `(B, S, d_model)` in, `(B, S, d_model)` out — which
is exactly what lets it slot into a residual connection (`x + FFN(x)`) inside the
encoder layer. Note dropout sits *after* the activation and *before* the second linear,
matching the transformer FFN.

---

## References

- **Paper:** Devlin et al. 2019, [*BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*](https://arxiv.org/abs/1810.04805) — §A.2 names the GELU activation
- **GELU paper:** Hendrycks & Gimpel 2016, [*Gaussian Error Linear Units*](https://arxiv.org/abs/1606.08415)
- **Official Google BERT (TF):** [`modeling.py`](https://github.com/google-research/bert/blob/master/modeling.py) — the hand-written `gelu` (tanh form)
- **HF Transformers (PyTorch), pinned to v5.12.0:** [`modeling_bert.py`](https://github.com/huggingface/transformers/blob/v5.12.0/src/transformers/models/bert/modeling_bert.py) — `BertIntermediate` (`"gelu"` → exact erf; tanh is `"gelu_new"`)
