## Table of Contents

1. [Why Cross-Attention Only Needs Padding Mask](#why-cross-attention-only-needs-padding-mask)
2. [Why Output Shape Follows Query, Not Key/Value](#why-output-shape-follows-query-not-keyvalue)

---

# Why Cross-Attention Only Needs Padding Mask

## The Decoder's Two Attention Masks

The decoder has two attention sub-layers, each with a different mask:

- **`tgt_mask`** — for self-attention (sub-layer 1)
- **`memory_mask`** — for cross-attention (sub-layer 2)

## Concrete Example

Translating English → Bengali:

```
Source (English):  ["We", "are", "friends", "<pad>", "<pad>"]   # src_seq_len = 5
Target (Bengali):  ["আমরা", "বন্ধু", "হই"]                       # tgt_seq_len = 3
```

## Sub-layer 1: Masked Self-Attention (`tgt_mask`)

The decoder attends to **itself** — target looks at target.

```
            Keys (target)
            আমরা  বন্ধু  হই
Queries     ┌─────────────────┐
আমরা        │ ✅    ✗    ✗    │  ← can only see itself
বন্ধু          │ ✅    ✅   ✗    │  ← can see আমরা + itself
হই          │ ✅    ✅   ✅    │   ← can see all previous
            └─────────────────┘
```

This is the **causal mask** — lower triangular. Position `i` can only attend to positions `≤ i`. If target had padding, `tgt_mask` would combine causal + padding.

**No source padding involved** — target is only looking at itself.

## Sub-layer 2: Cross-Attention (`memory_mask`)

The decoder attends to the **encoder output** — target looks at source.

```
         Keys (source/encoder output)
         "We"  "are"  "friends"  <pad>  <pad>
Queries  ┌────────────────────────────────────┐
আমরা     │  ✅    ✅     ✅       ✗      ✗     │
বন্ধু       │  ✅    ✅     ✅       ✗      ✗     │
হই       │  ✅    ✅     ✅       ✗      ✗     │
         └────────────────────────────────────┘
```

Every decoder position can attend to **all real source tokens** — no causal restriction. But `<pad>` positions must be masked out. That's `memory_mask`.

## Why No Causal Mask in Cross-Attention?

- **Self-attention** needs causal because the decoder generates left-to-right — position `i` can't peek at future token `i+1`
- **Cross-attention** doesn't need causal because the source sentence is **already complete** — it was fully encoded before decoding started. There's nothing to "hide". The only thing to mask is padding.

---

# Why Output Shape Follows Query, Not Key/Value

## The Question

`DecoderLayer.forward()` takes two different sequence lengths:

```python
tgt:             (batch_size, tgt_seq_len, d_model)   # e.g., 2 tokens
encoder_output:  (batch_size, src_seq_len, d_model)   # e.g., 3 tokens
```

Yet the output is `(batch_size, tgt_seq_len, d_model)`. How does `src_seq_len` disappear?

## Trace Through Cross-Attention Math

```
Q from tgt:              (batch, tgt_seq_len, d_model)
K from encoder_output:   (batch, src_seq_len, d_model)
V from encoder_output:   (batch, src_seq_len, d_model)

# After split_heads:
Q: (batch, heads, tgt_seq_len, d_k)
K: (batch, heads, src_seq_len, d_k)
V: (batch, heads, src_seq_len, d_k)

# Attention scores: Q @ K^T
(batch, heads, tgt_seq_len, d_k) @ (batch, heads, d_k, src_seq_len)
= (batch, heads, tgt_seq_len, src_seq_len)    ← the rectangular grid

# Multiply by V: scores @ V
(batch, heads, tgt_seq_len, src_seq_len) @ (batch, heads, src_seq_len, d_k)
= (batch, heads, tgt_seq_len, d_k)            ← src_seq_len cancels out!

# After combine_heads + W_o:
= (batch, tgt_seq_len, d_model)
```

## Why `src_seq_len` Disappears

The `src_seq_len` dimension gets **summed away** during `scores @ V`. Each query position produces a weighted sum over all source values — that weighted sum collapses `src_seq_len` into a single `d_k` vector per query position.

No matter how long the source is (3 tokens, 100 tokens, 1000 tokens), the output always matches the **query's** length: `(batch, tgt_seq_len, d_model)`.

This is exactly why `MultiHeadAttention.forward()` takes separate `query`, `key`, `value` arguments with independent `seq_len_q` and `seq_len_k` — the attention score matrix is `(seq_len_q x seq_len_k)`, which doesn't have to be square.
