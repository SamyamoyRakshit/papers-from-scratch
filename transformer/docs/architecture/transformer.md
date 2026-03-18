## Table of Contents

1. [What is Output Projection?](#what-is-output-projection)
2. [Weight Sharing — One Matrix, Two Jobs (Section 3.4)](#weight-sharing--one-matrix-two-jobs-section-34)
3. [Full Encoder-Decoder Trace — Weight Sharing in Action](#full-encoder-decoder-trace--weight-sharing-in-action)
4. [What if `src_vocab_size != tgt_vocab_size`?](#what-if-src_vocab_size--tgt_vocab_size)
5. [Why Shared PositionalEncoding Works](#why-shared-positionalencoding-works)
6. [Why Return Logits, Not Softmax](#why-return-logits-not-softmax)

---

# What is Output Projection?

## The Problem

The decoder outputs a vector of shape `(d_model,)` for each position — a rich, context-aware representation. But we need **a word**, not a vector. Specifically, we need a probability distribution over the entire vocabulary: "how likely is each word to be the next token?"

## The Solution

`output_projection` is a linear layer that converts each decoder output vector into vocabulary-sized logits:

```python
self.output_projection = nn.Linear(d_model, tgt_vocab_size, bias=False)
```

For one decoder position:

```
decoder_output:     (1, 512)          one vector from decoder
W^T:                (512, 37000)      transposed weight matrix

logits = decoder_output @ W^T
         (1, 512) @ (512, 37000) = (1, 37000)    one score per vocab token
```

Each of the 37,000 scores is a **dot product** — measuring how similar the decoder's output vector is to each vocabulary token's embedding. The highest score is the predicted token.

For a full sequence:

```
decoder_output:     (tgt_seq_len, 512)
W^T:                (512, 37000)

logits = (tgt_seq_len, 512) @ (512, 37000) = (tgt_seq_len, 37000)
```

Each position independently predicts the next token.

## Why `bias=False`?

This is not from the paper — it's an implementation detail of weight sharing. `nn.Embedding` has no bias (it's a pure row lookup). Since we tie the Linear's weight to the Embedding's weight, we set `bias=False` so both operations are symmetric:

- **Embedding:** `W[token_id]` — pure row lookup, no bias
- **Projection:** `x @ W^T` — pure matrix multiply, no bias

If we had bias, the projection would compute `x @ W^T + b` — an extra term that the embedding side doesn't have, breaking the symmetry of "same matrix, opposite directions."

---

# Weight Sharing — One Matrix, Two Jobs (Section 3.4)

## What the Paper Says

From Section 3.4:

> "We share the same weight matrix between the two embedding layers and the pre-softmax linear transformation."

This means three components share **one** weight matrix `W` of shape `(vocab_size, d_model)`:

1. **Source embedding** (`src_embedding`)
2. **Target embedding** (`tgt_embedding`)
3. **Output projection** (`output_projection`)

## How It Works in Code

```python
# Weight sharing (Section 3.4)
self.output_projection.weight = self.tgt_embedding.embeddings.weight

if src_vocab_size == tgt_vocab_size:
    self.src_embedding.embeddings.weight = self.tgt_embedding.embeddings.weight
```

This is **not** copying values — it's making them point to the **same tensor in memory**. When the optimizer updates the weights during training, all three see the update.

## The Two Jobs of One Matrix

**Job 1 — Embedding (row lookup):**

Given a token ID, look up its row in `W` to get a vector representation.

```
Token ID 42 → W[42] = [0.5, 0.6, 0.7, ...]    shape: (d_model,)
```

**Job 2 — Output Projection (dot product similarity):**

Given a decoder output vector, compute dot product with **every** row in `W` to find which token it's most similar to.

```
decoder_output @ W^T = [score_0, score_1, ..., score_vocab]    shape: (vocab_size,)
```

Same matrix, opposite directions. If token "আমরা" maps **to** `[0.5, 0.6, 0.7, 0.8]` during embedding, then a decoder output **near** `[0.5, 0.6, 0.7, 0.8]` should map **back** to "আমরা" during projection. Sharing forces this consistency.

## Why Share?

1. **Consistency** — embedding and projection agree on what each token "looks like" in vector space
2. **Fewer parameters** — the embedding matrix is one of the largest components (`vocab_size x d_model` can be 37000 x 512 = ~19M parameters). Sharing cuts this by up to 2/3
3. **Better generalization** — fewer parameters means less overfitting

---

# Full Encoder-Decoder Trace — Weight Sharing in Action

A complete trace of English → Bengali translation showing where the shared weight matrix `W` is used.

## Setup

```
Source vocabulary (English):  {<pad>: 0, we: 1, are: 2, friends: 3}    src_vocab_size = 4
Target vocabulary (Bengali):  {<pad>: 0, আমরা: 1, বন্ধু: 2, হই: 3}       tgt_vocab_size = 4
d_model = 4 (tiny for illustration)
```

Since `src_vocab_size == tgt_vocab_size`, all three weights are shared — one matrix `W` of shape `(4, 4)`:

```
W = [[0.1, 0.2, 0.3, 0.4],   ← row 0: <pad>
     [0.5, 0.6, 0.7, 0.8],   ← row 1: "we" / "আমরা"
     [0.9, 1.0, 1.1, 1.2],   ← row 2: "are" / "বন্ধু"
     [1.3, 1.4, 1.5, 1.6]]   ← row 3: "friends" / "হই"
```

## Step 1: Source Embedding (Job 1 of W)

Input: `src = [1, 2, 3]` → "we are friends"

```python
src_embedded = self.src_embedding(src)    # row lookup from W
```

```
token 1 ("we")      → W[1] = [0.5, 0.6, 0.7, 0.8]
token 2 ("are")     → W[2] = [0.9, 1.0, 1.1, 1.2]
token 3 ("friends") → W[3] = [1.3, 1.4, 1.5, 1.6]
```

Then `* sqrt(d_model)` scaling + positional encoding → feeds into **Encoder**.

## Step 2: Encoder

```python
src_embedded = self.positional_encoding(self.src_embedding(src))
encoder_output = self.encoder(src_embedded, src_mask)
# shape: (1, 3, 4) — 3 source tokens, each now a rich 4-dim vector
```

The encoder processes through N layers of self-attention + FFN. The output is **no longer the original embeddings** — it's a deeply processed, context-aware representation of the entire English sentence.

```
encoder_output ≈ [[0.3, -0.1, 0.8, 0.2],     "we" context-aware representation
                   [0.7,  0.4, 0.1, 0.9],     "are" context-aware representation
                   [0.2,  0.6, 0.5, 0.3]]     "friends" context-aware representation
```

## Step 3: Target Embedding (Job 1 of W again)

During training, we feed the target shifted right: `tgt = [0, 1, 2]` → `<pad>, আমরা, বন্ধু`

The decoder sees the previous tokens and predicts the next one.

```python
tgt_embedded = self.tgt_embedding(tgt)    # same W, row lookup
```

```
token 0 (<pad>)  → W[0] = [0.1, 0.2, 0.3, 0.4]
token 1 (আমরা)   → W[1] = [0.5, 0.6, 0.7, 0.8]
token 2 (বন্ধু)    → W[2] = [0.9, 1.0, 1.1, 1.2]
```

Same matrix `W`, same row lookup — just using Bengali vocabulary indices.

## Step 4: Decoder

```python
tgt_embedded = self.positional_encoding(self.tgt_embedding(tgt))
decoder_output = self.decoder(tgt_embedded, encoder_output, tgt_mask, memory_mask)
# shape: (1, 3, 4) — 3 target positions, each a 4-dim vector
```

The decoder runs 3 sub-layers per layer:

1. **Masked self-attention** — target attends to itself (with causal mask)
2. **Cross-attention** — target attends to encoder_output (Q from decoder, K/V from encoder)
3. **FFN** — position-wise transformation

```
decoder_output ≈ [[0.4, 0.7, 0.6, 0.8],     position 0: should predict "আমরা"
                   [0.8, 0.9, 1.0, 1.1],     position 1: should predict "বন্ধু"
                   [1.2, 1.3, 1.4, 1.5]]     position 2: should predict "হই"
```

## Step 5: Output Projection (Job 2 of W)

The decoder gave us vectors of shape `(d_model,)` — but we need **probabilities over the vocabulary** (which token comes next?).

```python
logits = self.output_projection(decoder_output)
# This is: decoder_output @ W^T
```

For each decoder position, compute a dot product with **every** vocabulary row in W:

```
Position 0 output: [0.4, 0.7, 0.6, 0.8]

dot with W[0] (<pad>):   0.4*0.1 + 0.7*0.2 + 0.6*0.3 + 0.8*0.4 = 0.68
dot with W[1] (আমরা):    0.4*0.5 + 0.7*0.6 + 0.6*0.7 + 0.8*0.8 = 1.68   ← predict this
dot with W[2] (বন্ধু):     0.4*0.9 + 0.7*1.0 + 0.6*1.1 + 0.8*1.2 = 2.68
dot with W[3] (হই):       0.4*1.3 + 0.7*1.4 + 0.6*1.5 + 0.8*1.6 = 3.68
```

```python
logits = self.output_projection(decoder_output)
# shape: (1, 3, 4) — 3 positions x 4 vocab tokens
```

Then `nn.CrossEntropyLoss` takes these logits, applies softmax internally, and computes loss against the true targets `[1, 2, 3]` → আমরা, বন্ধু, হই.

## Full Data Flow Summary

```
src token IDs                        tgt token IDs (shifted right)
     |                                       |
     v                                       v
[src_embedding] ← W (Job 1)        [tgt_embedding] ← W (Job 1)
     |                                       |
     v                                       v
[positional_encoding]               [positional_encoding]
     |                                       |
     v                                       v
  Encoder                              Decoder ←── encoder_output
     |                                       |
     |                                       v
     |                              [output_projection] ← W^T (Job 2)
     |                                       |
     |                                       v
     └──────────────────────────────     logits (vocab_size scores per position)
```

---

# What if `src_vocab_size != tgt_vocab_size`?

## Different Vocab Sizes = Separate Source Embedding

When languages have **separate vocabularies** (no shared tokenizer), `src_vocab_size` and `tgt_vocab_size` differ. Then the source embedding matrix has a different number of rows than the target — sharing is impossible.

```python
# Always shared: tgt_embedding <-> output_projection
self.output_projection.weight = self.tgt_embedding.embeddings.weight

# Only shared if vocab sizes match:
if src_vocab_size == tgt_vocab_size:
    self.src_embedding.embeddings.weight = self.tgt_embedding.embeddings.weight
```

With different sizes:

```
src_embedding:      (src_vocab_size, d_model) = (10000, 512)    own matrix
tgt_embedding:      (tgt_vocab_size, d_model) = (8000, 512)     shared with
output_projection:  (8000, d_model) = (8000, 512)               ← this
```

Target embedding and output projection **always** share — they operate on the same vocabulary.

## When Vocab Sizes Are Equal

This typically means both languages use a **shared tokenizer** (like BPE trained on combined data). The vocabulary contains tokens from both languages:

```
Shared BPE Vocabulary (single tokenizer):
Row 0:     <pad>
Row 1:     <sos>
Row 2:     <eos>
Row 3:     "the"
Row 4:     "আম"          Bengali subword
Row 5:     "রা"           Bengali subword
Row 6:     "we"
Row 7:     "friend"
Row 8:     "##s"          subword suffix
...
Row 4999:  "বন্ধু"
```

Both languages' tokens are in **one big vocabulary**. English tokens and Bengali tokens each occupy their own rows — they don't overlap (except shared symbols like punctuation, numbers, special tokens).

Even though "we" (row 6) and "আমরা" (composed of rows 4+5) are in the same matrix, each row learns an **independent** 512-dimensional vector. The model learns to place semantically similar words **near each other** in this vector space — that's actually a **benefit** of sharing. It forces the model to learn a shared multilingual representation.

---

# Why Shared PositionalEncoding Works

## One Instance, Two Users

```python
# One PositionalEncoding instance
self.positional_encoding = PositionalEncoding(d_model, dropout, max_len)

# Used by both encoder and decoder
src_embedded = self.positional_encoding(self.src_embedding(src))
tgt_embedded = self.positional_encoding(self.tgt_embedding(tgt))
```

This works because `PositionalEncoding` is **stateless** — it has no learned parameters that depend on which sequence (source or target) is being processed.

## What PositionalEncoding Does

```python
def forward(self, x):
    x = x + self.pe[:, :x.size(1)]    # add pre-computed sin/cos values
    return self.dropout(x)             # apply dropout
```

Two operations, neither depends on "who" calls it:

1. **Add positional signals** — `self.pe` is a fixed buffer of sin/cos values, pre-computed for all positions up to `max_len`. It just slices to match the input's sequence length.
2. **Apply dropout** — random zeroing, independent each call.

Position 0 gets the same sin/cos values whether it's the first English token or the first Bengali token. That's exactly what we want — position encoding captures **where** a token is, not **what** it is.

---

# Why Return Logits, Not Softmax

## What the Model Returns

```python
def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
    ...
    logits = self.output_projection(decoder_output)
    return logits    # raw scores, NOT probabilities
```

## Why Not Apply Softmax?

`nn.CrossEntropyLoss` — the standard loss function for classification tasks — expects **raw logits** and applies softmax internally:

```python
# CrossEntropyLoss = LogSoftmax + NLLLoss (combined for numerical stability)
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(logits, targets)    # softmax happens INSIDE here
```

If we applied softmax in the model **and** CrossEntropyLoss applied it again, we'd get `softmax(softmax(logits))` — mathematically wrong.

## Numerical Stability

PyTorch's `CrossEntropyLoss` uses the [log-sum-exp trick](https://en.wikipedia.org/wiki/LogSumExp) internally, which avoids overflow/underflow that can happen when computing `exp(logits)` for large values. Applying softmax manually first would lose this benefit.

## During Inference

When generating translations (not training), you **do** apply softmax yourself:

```python
# Training: let CrossEntropyLoss handle it
loss = loss_fn(model(src, tgt), targets)

# Inference: apply softmax to get probabilities
logits = model(src, tgt)
probs = torch.softmax(logits[:, -1, :], dim=-1)    # last position
next_token = probs.argmax(dim=-1)                    # greedy decoding
```
