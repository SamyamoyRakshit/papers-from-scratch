## 📋 Table of Contents

1. [References](#references)
2. [What is Q, K, V?](#what-is-q-k-v)
3. [What is QW^Q_i, KW^K_i, VW^V_i?](#what-is-wq_i-wk_i-wv_i)
4. [QK^T Matrix Multiplication Example](#qkt-matrix-multiplication-example)
5. [`//` vs `/` in Python](#-vs--in-python)
6. [Negative Indexing in `transpose(-2, -1)`](#negative-indexing-in-transpose-2--1)
7. [How `masked_fill` Works](#how-masked_fill-works)
8. [Causal Mask vs Padding Mask](#causal-mask-vs-padding-mask)
9. [Can a Row Be Fully Masked? (NaN Risk)](#can-a-row-be-fully-masked-nan-risk-from-masked_fill)
10. [Where Do Masks Come From?](#where-do-masks-come-from)
11. [Why `dim=-1` in Softmax?](#why-dim-1-in-softmax)
12. [Why Dropout on Attention Weights?](#why-dropout-on-attention-weights)
13. [Where Do Query, Key, Value Come From?](#where-do-query-key-value-come-from)
14. [Understanding view, contiguous, reshape & transpose](#understanding-view-contiguous-reshape--transpose)
15. [Xavier Initialization: Why and How](#xavier-initialization-why-and-how)
16. [The `_` Prefix Convention: `_reset_parameters`](#the-_-prefix-convention-_reset_parameters)
17. [`nn.init.zeros_` vs `nn.init.constant_` vs `torch.zeros`](#nninitzeros_-vs-nninitconstant_-vs-torchzeros)

---

# References:

1. [Attention Is All You Need (Paper)](https://arxiv.org/abs/1706.03762) — Section 3.2
2. [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
3. [Visualizing A Neural Machine Translation Model](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)

---

# What is Q, K, V?

Each word's embedding is multiplied by **three different learned weight matrices** ($W^Q$, $W^K$, $W^V$) to create three versions:

| Symbol | Name | Intuition |
|--------|------|-----------|
| **Q** (Query) | "What am I looking for?" | Like a search query |
| **K** (Key) | "What do I contain?" | Like a search index/tag |
| **V** (Value) | "What is my actual content?" | The real information |

Example for "We are friends":
```
e-we × Wq → q-we     (Query for "we")
e-we × Wk → k-we     (Key for "we")
e-we × Wv → v-we     (Value for "we")
```

> **In self-attention, Q, K, V all come from the same input X.** The weight matrices make them different.

---

# What is $W^Q_i$, $W^K_i$, $W^V_i$?

The subscript **i** refers to which **head** in Multi-Head Attention.

Each head has **its own** set of projection weights:

```
Head 1: uses Wq₁, Wk₁, Wv₁ → might focus on adjective relationships
Head 2: uses Wq₂, Wk₂, Wv₂ → might focus on verb relationships
...
Head 8: uses Wq₈, Wk₈, Wv₈ → might focus on prepositions
```

### The Q Naming Confusion

The paper uses **Q** at two levels:

| Symbol | Meaning | What it actually is |
|--------|---------|-------------------|
| **Q** in `MultiHead(Q, K, V)` | **Input** to the multi-head function | The raw embeddings |
| **$QW^Q_i$** | **Projected** query for head $i$ | The result after multiplying by weights |

> **The Q in $QW^Q_i$ is the raw input (embeddings), not the projected query — it's just the paper's variable name for the input that will *become* the query after projection.**

### Three Types of Attention Using the Same Class

| Type | Q comes from | K comes from | V comes from |
|------|-------------|-------------|-------------|
| **Encoder self-attention** | X (same) | X (same) | X (same) |
| **Decoder masked self-attention** | X (same) | X (same) | X (same) |
| **Cross-attention** | Decoder | **Encoder** | **Encoder** |

---

# QK^T Matrix Multiplication Example

**Setup**: `batch_size=1`, `num_heads=2`, `seq_len=3` ("We are friends"), `d_k=2`

```
Q shape: (1, 2, 3, 2)  →  1 batch, 2 heads, 3 words, 2 dims per head
K shape: (1, 2, 3, 2)

K.transpose(-2, -1) shape: (1, 2, 2, 3)  ← swap last two dims
```

**For Head 1:**

```
Q (3 words × 2 dims):          K^T (2 dims × 3 words):
┌─────────────────┐            ┌──────────────────┐
│ q-we:  [1, 2]   │            │ [1, 3, 5]        │  ← k-we, k-are, k-fri dim0
│ q-are: [3, 4]   │    @       │ [2, 4, 6]        │  ← k-we, k-are, k-fri dim1
│ q-fri: [5, 6]   │            └──────────────────┘
└─────────────────┘

Result (3 × 3 = word-to-word scores):
┌─────────────────────────────────────────────┐
│ q-we·k-we    q-we·k-are    q-we·k-fri       │  = [5,  11, 17]
│ q-are·k-we   q-are·k-are   q-are·k-fri      │  = [11, 25, 39]
│ q-fri·k-we   q-fri·k-are   q-fri·k-fri      │  = [17, 39, 61]
└─────────────────────────────────────────────┘
```

**Each value = how much one word "attends to" another word!**

### After Scaling (÷ √d_k)

```
√d_k = √2 ≈ 1.414

Scaled scores:
[5/1.414,  11/1.414, 17/1.414]   = [3.54,  7.78, 12.02]
[11/1.414, 25/1.414, 39/1.414]   = [7.78,  17.68, 27.58]
[17/1.414, 39/1.414, 61/1.414]   = [12.02, 27.58, 43.13]
```

### After Softmax (per row → probabilities)

```
Row 1: softmax([3.54, 7.78, 12.02])  ≈ [0.00, 0.01, 0.99]  ← "we" mostly attends to "friends"
Row 2: softmax([7.78, 17.68, 27.58]) ≈ [0.00, 0.00, 1.00]  ← "are" mostly attends to "friends"
Row 3: softmax([12.02, 27.58, 43.13])≈ [0.00, 0.00, 1.00]  ← "friends" mostly attends to itself
```

### Final Output (Attention Weights × V)

```
output-we     = 0.00 × v-we + 0.01 × v-are + 0.99 × v-friends
output-are    = 0.00 × v-we + 0.00 × v-are + 1.00 × v-friends
output-friends = 0.00 × v-we + 0.00 × v-are + 1.00 × v-friends
```

> Each word's output is a **weighted combination** of all Value vectors!

---

# `//` vs `/` in Python

## ❌ `//` is WRONG for attention!

```python
# ❌ WRONG — integer (floor) division, loses decimal precision
scores = torch.matmul(Q, K.transpose(-2,-1)) // math.sqrt(self.d_k)

# ✅ CORRECT — float (true) division, preserves precision
scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.d_k)
```

## Why this matters:

```python
# With d_k = 64
math.sqrt(64) = 8.0

# Score = 17.5
17.5 // 8.0 = 2.0     # ❌ Loses decimal information!
17.5 / 8.0  = 2.1875  # ✅ Correct, precise value
```

## Quick Reference:

| Operator | Name | Example | Result |
|----------|------|---------|--------|
| `/` | True division | `7 / 2` | `3.5` |
| `//` | Floor division | `7 // 2` | `3` |

> **`//` truncates decimals. Softmax needs precise float values to compute correct attention weights!**

---

# Negative Indexing in `transpose(-2, -1)`

## What `-2` and `-1` Mean

Negative indices count **from the end** of the shape:

```
K.shape = (batch_size, num_heads, seq_len, d_k)
              dim 0      dim 1     dim 2   dim 3
             dim -4     dim -3    dim -2  dim -1
```

So:

```python
K.transpose(-2, -1)  # Swap dim -2 (seq_len) with dim -1 (d_k)
```

## Before and After:

```
Before: (batch_size, num_heads, seq_len, d_k) → (1, 2, 3, 2)
After:  (batch_size, num_heads, d_k, seq_len) → (1, 2, 2, 3)
                                ↑        ↑
                              swapped!
```

## Why Use Negative Indices?

Because it works **regardless of how many dimensions come before**. The batch and head dimensions stay untouched — we only swap the last two!

```python
# These are equivalent:
K.transpose(-2, -1)  # ✅ Works for any number of leading dims
K.transpose(2, 3)    # ⚠️ Only works if exactly 4 dims
```

---

# How `masked_fill` Works

```python
scores.masked_fill(mask == 0, float('-inf'))
```

This is **two operations in one line**:

### Step 1: `mask == 0` → Boolean matrix

```python
mask = tensor([
    [1, 0, 0],
    [1, 1, 0],
    [1, 1, 1]
])

mask == 0  →  tensor([
    [False, True,  True ],
    [False, False, True ],
    [False, False, False]
])
```

### Step 2: Replace `True` positions with `-inf`

```
scores:                bool_mask (mask == 0):         result:
┌──────────────┐       ┌───────────────────┐       ┌──────────────────┐
│ 2.0  3.0  5.0│       │ F     T     T     │       │ 2.0  -inf  -inf  │
│ 1.0  4.0  2.0│   +   │ F     F     T     │  =    │ 1.0   4.0  -inf  │
│ 3.0  1.0  6.0│       │ F     F     F     │       │ 3.0   1.0   6.0  │
└──────────────┘       └───────────────────┘       └──────────────────┘
```

### Why `-inf`?

```python
softmax([2.0, -inf, -inf]) = [1.00, 0.00, 0.00]
# e^(-inf) = 0 → masked positions get zero attention weight!
```

---

# Causal Mask vs Padding Mask

| Mask Type | Purpose | Blocks | Used In |
|-----------|---------|--------|--------|
| **Causal** | Prevent seeing future words | Upper triangle | Decoder self-attention |
| **Padding** | Ignore `<PAD>` tokens | Entire columns | Encoder + Decoder |
| **Combined** | Both at once | Both | Decoder |

### Causal Mask (lower triangular)

```
"I love AI" (decoder predicting word by word):

         I    love   AI
I      [ 1,    0,    0 ]   ← can only see itself
love   [ 1,    1,    0 ]   ← can see "I" and itself
AI     [ 1,    1,    1 ]   ← can see everything before it
```

### Padding Mask

```
Batch: ["I love AI", "Hi <PAD> <PAD>"]

Sentence 1: [1, 1, 1]   ← all real tokens
Sentence 2: [1, 0, 0]   ← only "Hi" is real, block pad columns
```

### Can a Row Be Fully Masked? (NaN Risk from `masked_fill`)

After `masked_fill`, what happens if **every** key in a row is masked?

**Setup:** batch of 2 sentences, `seq_len=4`

```
Sentence A: "I love cats [PAD]"   → tokens: [1, 2, 3, 0]
Sentence B: "[PAD] [PAD] [PAD] [PAD]"  → tokens: [0, 0, 0, 0]  ← fully padded (data bug)
```

Padding mask (1=real, 0=pad):
```
Sentence A mask: [1, 1, 1, 0]
Sentence B mask: [0, 0, 0, 0]   ← all zeros
```

**Step 1:** Raw attention scores from `QK^T / √d_k`:
```
Sentence A (one query row):  [ 0.8,  0.3,  0.6, -0.1]
Sentence B (one query row):  [ 0.5, -0.2,  0.4,  0.1]
```

**Step 2:** After `masked_fill(mask == 0, float('-inf'))`:
```
Sentence A:  [ 0.8,  0.3,  0.6, -inf]   ← last token masked, 3 valid keys remain
Sentence B:  [-inf, -inf, -inf, -inf]    ← ALL masked, zero valid keys
```

**Step 3:** `F.softmax(scores, dim=-1)`:
```
Sentence A:
  exp([ 0.8,  0.3,  0.6, -inf]) = [2.23, 1.35, 1.82, 0.0]
  sum = 5.40
  weights = [0.41, 0.25, 0.34, 0.00]  ✓ sums to 1

Sentence B:
  exp([-inf, -inf, -inf, -inf]) = [0, 0, 0, 0]
  sum = 0
  weights = [0/0, 0/0, 0/0, 0/0] = [NaN, NaN, NaN, NaN]  ✗ 0/0 = NaN!
```

**Step 4:** `matmul(attention_weights, V)`:
```
Sentence B output:
  [NaN, NaN, NaN, NaN] @ V = [NaN, NaN, NaN, NaN]
```

NaN propagates forward through `W_o` → `LayerNorm` → `FFN` → loss — **the entire training step is poisoned.**

#### Is This Realistic?

**In standard training with proper data loading — no.** Every real sentence has at least one non-pad token, so there's always a valid key to attend to. Even when combining causal + padding masks:

```
Decoder input: ["the", "cat", "[PAD]", "[PAD]"]
Padding mask:  [  1,     1,     0,       0   ]

Combined (causal AND padding):
  [ 1  0  0  0 ]   ← position 0: attends to "the". Fine.
  [ 1  1  0  0 ]   ← position 1: attends to "the", "cat". Fine.
  [ 1  1  0  0 ]   ← position 2 ([PAD]): still has valid keys at 0,1. Fine.
  [ 1  1  0  0 ]   ← position 3 ([PAD]): still has valid keys at 0,1. Fine.
```

The real tokens (positions 0, 1) are always reachable, so no row ends up fully zero. You'd need a truly degenerate case — like an empty source sentence from a data bug — to trigger this.

#### The Fix: `nan_to_num(0.0)`

Replace NaN with 0 so those query positions produce a zero-vector instead of poisoning the batch:

```python
attention_weights = F.softmax(scores, dim=-1)
attention_weights = attention_weights.nan_to_num(0.0)  # NaN rows → zero weights
```

**Why zero?** A zero-weight row produces:
```
[0, 0, 0, 0] @ V = [0, 0, 0, 0]   ← zero output, no NaN
```

Zero is neutral — it contributes nothing to the output, gradients pass through cleanly, and downstream layers (LayerNorm, FFN) handle zero vectors without issue. It's cheap defensive coding for an edge case that shouldn't happen but could silently destroy training if it did.

---

# Where Do Masks Come From?

Masks are **created in a utility function**, not inside the attention module.

### Causal mask:
```python
def create_causal_mask(seq_len):
    return torch.tril(torch.ones(seq_len, seq_len))
```

### Padding mask:
```python
def create_padding_mask(input_ids, pad_token_id=0):
    return (input_ids != pad_token_id).int()
```

### Where they live:

```
transformer/
├── models/
│   ├── modules/
│   │   └── multi_head_attention.py   ← USES mask (receives it)
│   └── utils/
│       └── mask_utils.py             ← CREATES masks
└── train.py                          ← Calls mask creation before feeding to model
```

> **The attention module just receives and applies masks — it doesn't create them.**

---

# Why `dim=-1` in Softmax?

> **Why document this?** `dim` in softmax is a subtle but critical parameter — choosing the wrong dimension silently produces incorrect attention weights without any error. This section clarifies which dimension to use and why.

```python
attention_weights = F.softmax(scores, dim=-1)
```

`dim=-1` means **apply softmax along the last dimension** (columns/keys).

## Example

```python
scores = tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
])
# Shape: (2, 3) → dim -2 is rows, dim -1 is columns
```

### `dim=-1`: softmax across columns (→→→)

```
┌─────────────────┐
│ 0.09  0.24  0.67│ →→→ sums to 1
│ 0.09  0.24  0.67│ →→→ sums to 1
└─────────────────┘
```

Each **row** becomes a probability distribution.

### `dim=-2`: softmax across rows (↓↓↓) — WRONG for attention

```
┌─────────────────┐
│ 0.05  0.05  0.05│
│ 0.95  0.95  0.95│
└─────────────────┘
  ↓↓↓   ↓↓↓   ↓↓↓
  1.0   1.0   1.0
```

Each **column** sums to 1 — wrong meaning!

## Why `dim=-1` for Attention?

In the scores matrix: rows = **queries**, columns = **keys**.

```
         k-we   k-are  k-friends
q-we   [ 2.0,   3.0,   5.0  ]    ← "How much does 'we' attend to each word?"
q-are  [ 1.0,   4.0,   2.0  ]    ← "How much does 'are' attend to each word?"
```

We want **each query's attention weights to sum to 1**, so softmax runs across keys (`dim=-1`).

> **`dim=-1` = each row becomes a probability distribution over which keys to attend to.**

---

# Why Dropout on Attention Weights?

```python
attention_weights = self.dropout(attention_weights)
```

From paper Section 5.4: *"We apply dropout to the output of each sub-layer..."*

| Reason | Explanation |
|--------|-------------|
| **Prevents over-reliance** | Forces the model to NOT always depend on the same word |
| **Regularization** | Reduces overfitting by adding randomness during training |
| **Robustness** | Model learns to attend to multiple words, not just one |

### Example Intuition

**Without dropout:**
```
"The cat sat on the mat"
"cat" always attends 90% to "sat" → memorizes this pattern
```

**With dropout:**
```
"The cat sat on the mat"
Sometimes "cat"→"sat" is dropped → model also learns to attend to "the", "on", etc.
→ more robust representations!
```

### How PyTorch Dropout Really Works (Inverted Dropout)

PyTorch doesn't just zero values — it also **scales up** the survivors:

```python
# dropout p = 0.1 (10% chance of dropping)
# Scale factor = 1/(1-p) = 1/0.9 ≈ 1.111

# Step 1: Softmax output (sums to 1.0)
attention_weights = [0.05, 0.70, 0.15, 0.10]  # sums to 1.0

# Step 2: Randomly zero some values
                    [0.00, 0.70, 0.00, 0.10]

# Step 3: Scale survivors by 1/(1-p) = 1.111
                    [0.00, 0.778, 0.00, 0.111] # sums to ~0.889, NOT 1.0
```

### Why Scale by `1/(1-p)`?

To keep the **expected sum the same** between training and inference.

**Without scaling (wrong):**
```
Original:     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
Sum = 10.0

After dropout (randomly drop 1 out of 10):
              [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
Sum = 9.0  ← Expected sum during training

During inference (no dropout):
              [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
Sum = 10.0 ← Expected sum during inference

Training sum (9.0) ≠ Inference sum (10.0) ❌
```

**With scaling by `1/(1-p)` (correct):**
```
Scale = 1/(1-0.1) = 1/0.9 ≈ 1.111

After dropout + scaling:
              [1.111, 1.111, 0.000, 1.111, 1.111, 1.111, 1.111, 1.111, 1.111, 1.111]
Sum = 9 × 1.111 = 10.0 ✅

During inference (no dropout, no scaling):
              [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
Sum = 10.0 ✅

Training sum (10.0) = Inference sum (10.0) ✅
```

**The general math:**
```
Without scaling:
  Expected sum = Original sum × (1-p)          ← reduced!

With scaling by 1/(1-p):
  Expected sum = Original sum × (1-p) × 1/(1-p)
               = Original sum × 1
               = Original sum                   ← preserved! ✅
```

The `(1-p)` and `1/(1-p)` cancel each other out.

### Does Softmax Always Sum to 1?

| Step | Sums to 1? | When? |
|------|-----------|-------|
| After **softmax** | ✅ Always | Always |
| After **dropout** (training) | ❌ Not necessarily | Training only |
| After **dropout** (inference) | ✅ Yes | Dropout is OFF |

The flow:
```
scores → softmax → sums to 1.0 (ALWAYS)
                 ↓
              dropout → may NOT sum to 1.0 (ONLY during training)
                 ↓
              matmul with V → output
```

**Softmax ALWAYS sums to 1. Dropout (applied after) can break that — but only during training, and that's intentional!**

### Important: Only During Training!

```python
# Training mode:  randomly zeros some weights + scales survivors
# Inference mode (model.eval()):  does NOTHING, passes through unchanged
```

> **Dropout on attention weights prevents the model from always attending to the same words, forcing it to learn more diverse attention patterns.**

---

# Where Do Query, Key, Value Come From?

The input is the **output of the previous step** (embeddings + positional encoding for layer 1, or previous layer's output for deeper layers):

```
Tokens → Embedding → + PosEnc → x (batch_size, seq_len, 512)
                                  │
                          ┌───────┼───────┐
                          ↓       ↓       ↓
                        query    key    value
                          └───────┼───────┘
                                  ↓
                        MultiHeadAttention
```

| Type | query | key | value |
|------|-------|-----|-------|
| **Encoder self-attention** | x | x | x |
| **Decoder self-attention** | x | x | x |
| **Cross-attention** | decoder_out | encoder_out | encoder_out |

### How `nn.Linear` Preserves Shape

`nn.Linear(512, 512)` only transforms the **last dimension** — `batch_size` and `seq_len` pass through untouched:

```python
query.shape = (32, 10, 512)   # 32 batches, 10 words, 512-dim
Q = W_q(query)                # Linear(512, 512) applied to EACH word independently
Q.shape     = (32, 10, 512)   # same shape, different values
```

### Why Only Last Dimension?

Because the linear transform is a **per-vector** operation:

```
(2, 3, 4) @ (4, 4).T = (2, 3, 4)
      ↑                      ↑
  last dim matches     last dim transformed
  batch & seq untouched
```
Think of it like: **the same function is applied to every word independently**, regardless of which batch or position it's in. `nn.Linear` doesn't know about batches or sequences — it just sees individual vectors and transforms them.

> **`nn.Linear` treats input as a collection of vectors (last dim) and applies the same matrix multiplication to each one — batch and sequence dims are just "loops" that PyTorch handled automatically.**

---

# Understanding view, contiguous, reshape & transpose

## Step 1: How PyTorch Stores Tensors in Memory

A 2D tensor looks like a grid, but PyTorch stores it as a **flat 1D array** in memory:

```python
t = torch.tensor([[1, 2, 3],
                   [4, 5, 6]])
# You see:
# [[1, 2, 3],
#  [4, 5, 6]]
#
# PyTorch stores:
# Memory: [1, 2, 3, 4, 5, 6]
#          ← row 0 →  ← row 1 →
```

The tensor is stored row-by-row: first all elements of row 0, then all elements of row 1.

## Step 2: What is Stride?

Stride answers: **"How many slots do I skip to get to the next element in each dimension?"**

Think of memory as a **row of lockers**:

```
Locker #:  [0]  [1]  [2]  [3]  [4]  [5]
Value:      1    2    3    4    5    6
```

Your 2D tensor `[[1,2,3], [4,5,6]]` is stored in these lockers. PyTorch needs a rule to find any element.

```python
t.stride()  # → (3, 1)
```

This means:
- **To go to the next ROW** → skip **3** lockers
- **To go to the next COLUMN** → skip **1** locker

### Why Is Row Stride = 3?

```
Row 0 starts at position 0  → [1, 2, 3]
Row 1 starts at position 3  → [4, 5, 6]
```

Row 0 → Row 1 = jump 3 positions. Because each row has **3 columns**!

**Simple rule:** Row stride = number of columns. Column stride = 1.

### Walking Through With the Locker Analogy

**Start at locker 0 (that's `t[0][0]` = 1)**

```
t[0][0] → Start at locker 0              → value = 1 ✅

Move to next COLUMN (skip 1 locker):
t[0][1] → Locker 0 + 1 = locker 1        → value = 2 ✅

Move to next COLUMN (skip 1 locker):  
t[0][2] → Locker 0 + 1 + 1 = locker 2    → value = 3 ✅

Now go to next ROW (skip 3 lockers from start):
t[1][0] → Locker 0 + 3 = locker 3        → value = 4 ✅

Move to next COLUMN (skip 1):
t[1][1] → Locker 3 + 1 = locker 4        → value = 5 ✅

Move to next COLUMN (skip 1):
t[1][2] → Locker 3 + 1 + 1 = locker 5    → value = 6 ✅
```

### Visual Map

```
Memory:  [1]  [2]  [3]  [4]  [5]  [6]
          ↑              ↑
        t[0][0]        t[1][0]
        
        ←── 3 lockers ──→  (row stride = 3)
        
        ←1→              
        t[0][0] to t[0][1]  (col stride = 1)
```

### The Formula

```
To access t[row][col]:
  memory_position = row × stride[0] + col × stride[1]
                  = row × 3 + col × 1

t[0][0] = memory[0×3 + 0×1] = memory[0] = 1
t[0][2] = memory[0×3 + 2×1] = memory[2] = 3
t[1][1] = memory[1×3 + 1×1] = memory[4] = 5
```

### Another Example — Shape (3, 4)

```
[[a, b, c, d],     ← 4 items per row
 [e, f, g, h],
 [i, j, k, l]]

Memory: [a, b, c, d, e, f, g, h, i, j, k, l]

Stride: (4, 1)
         ↑  ↑
   skip 4 to next row (because 4 columns)
   skip 1 to next column
```

## Step 3: What is `view`?

`view` changes shape by **recalculating strides** — no data moves, just new reading rules:

```python
t = torch.tensor([[1, 2, 3],
                   [4, 5, 6]])
# Memory: [1, 2, 3, 4, 5, 6]
# Shape: (2, 3), Stride: (3, 1)

v = t.view(3, 2)
# tensor([[1, 2],
#         [3, 4],
#         [5, 6]])
#
# Memory: [1, 2, 3, 4, 5, 6]   ← SAME memory, no copy!
# Shape: (3, 2), Stride: (2, 1)  ← new strides
#
# Reads as:
# [1, 2]  → memory[0], memory[1]   ✅ sequential
# [3, 4]  → memory[2], memory[3]   ✅ sequential
# [5, 6]  → memory[4], memory[5]   ✅ sequential

v = t.view(6)
# Memory: [1, 2, 3, 4, 5, 6]  ← still same memory
# Interpreted as: [1, 2, 3, 4, 5, 6]
```

**No copy = fast!** But `view` can ONLY produce strides where reading is sequential (contiguous).

## Step 4: What Does `transpose` Do?

`transpose` literally just **swaps the two stride numbers** (and shape numbers). That's ALL it does:

```python
t = torch.tensor([[1, 2, 3],
                   [4, 5, 6]])
# Shape:  (2, 3)
# Stride: (3, 1)
#          ↑  ↑
#        dim0 dim1

t2 = t.transpose(0, 1)
# tensor([[1, 4],
#         [2, 5],
#         [3, 6]])
#
# Shape:  (3, 2)    ← swapped shape
# Stride: (1, 3)    ← swapped strides
#          ↑  ↑
#   was dim1  was dim0
# Memory: [1, 2, 3, 4, 5, 6]  ← SAME, nothing moved!
```

No data moves. No computation. Just **swap two numbers in shape and two numbers in stride**.

### Why Does Swapping Strides Work?

**Before:** "rows skip 3, columns skip 1"

```
t[row][col] → memory[row × 3 + col × 1]

t[0][0] = memory[0]  = 1
t[0][1] = memory[1]  = 2
t[1][0] = memory[3]  = 4
```

**After swap:** "rows skip 1, columns skip 3"

```
t2[row][col] → memory[row × 1 + col × 3]

t2[0][0] = memory[0×1 + 0×3] = memory[0] = 1
t2[0][1] = memory[0×1 + 1×3] = memory[3] = 4   ← was t[1][0], now t2[0][1]
t2[1][0] = memory[1×1 + 0×3] = memory[1] = 2   ← was t[0][1], now t2[1][0]
t2[1][1] = memory[1×1 + 1×3] = memory[4] = 5
t2[2][0] = memory[2×1 + 0×3] = memory[2] = 3
t2[2][1] = memory[2×1 + 1×3] = memory[5] = 6

Result:
[[1, 4],
 [2, 5],
 [3, 6]]
```

By swapping the skip rules, **rows become columns and columns become rows!** The data doesn't move — we just changed how we navigate it.

But look at the **memory access order**: 0, 3, 1, 4, 2, 5 — it's **scattered**, not sequential!

## Step 5: What is "Contiguous"?

Contiguous means: **reading left→right, top→bottom matches the memory order.**

Both `view` and `transpose` can give you a `(3, 2)` shaped tensor. But they're different:

```python
# view result — Shape (3, 2), Stride (2, 1):
# [1, 2]    reads memory: 0, 1, 2, 3, 4, 5
# [3, 4]    → SEQUENTIAL ✅ → CONTIGUOUS ✅
# [5, 6]

# transpose result — Shape (3, 2), Stride (1, 3):
# [1, 4]    reads memory: 0, 3, 1, 4, 2, 5
# [2, 5]    → SCATTERED ❌ → NOT CONTIGUOUS ❌
# [3, 6]
```

**Contiguous rule:** Strides must be **decreasing** (outer > inner).
- `(2, 1)` → 2 > 1 ✅ contiguous
- `(1, 3)` → 1 < 3 ❌ NOT contiguous

**Why must strides be decreasing?** The outer dimension must jump further than the inner dimension.

**Contiguous: Stride `(2, 1)` — 2 > 1 ✅**
```
Memory: [1, 2, 3, 4, 5, 6]

Row stride = 2 (jump 2 to next row — BIG jump)
Col stride = 1 (jump 1 to next col — small jump)

Memory positions read: 0, 1, 2, 3, 4, 5  → SEQUENTIAL ✅
```

**NOT Contiguous: Stride `(1, 3)` — 1 < 3 ❌**
```
Memory: [1, 2, 3, 4, 5, 6]

Row stride = 1 (jump 1 to next row — TINY jump)
Col stride = 3 (jump 3 to next col — BIG jump)

Memory positions read: 0, 3, 1, 4, 2, 5  → SCATTERED ❌
```

**"Decreasing strides" means the first dimension jumps the farthest, the last dimension jumps the least — this guarantees you read memory in order.**


### Why Does `transpose` Break Contiguity But `view` Doesn't?

Because they have **different jobs and different contracts**:

- **`view`** — "Reinterpret the same sequential data with a new shape." It recalculates strides to keep sequential reading. It's **NOT ALLOWED** to create non-contiguous results. That's its contract.

- **`transpose`** — "Swap how you navigate dimensions." It swaps strides, which can break the decreasing order. It's **ALLOWED** to create non-contiguous results. That's by design.

**Analogy:**
- `view` = reformat the same book into different page sizes. The text order stays the same.
- `transpose` = read the book column-by-column instead of row-by-row. The text order changes, but you don't reprint the book.

**`view` must keep memory-sequential reading (contiguous). `transpose` is free to rearrange navigation (can break contiguity) because their contracts are different.**

## Step 6: Why `view` Fails After `transpose`

```python
t2 = t.transpose(0, 1)   # Stride: (1, 3) — NOT contiguous

t2.view(6)   # ❌ CRASHES!
# RuntimeError: view size is not compatible with input tensor's
# size and stride
```

`view` needs to find strides that read memory **sequentially**. But the logical data order `[1,4,2,5,3,6]` doesn't exist sequentially in memory `[1,2,3,4,5,6]`. No stride can fix that.

## Step 7: What Does `.contiguous()` Do?

It **copies data into new memory** so the logical order matches memory order:

```python
t2 = t.transpose(0, 1)
# Logical order: [1, 4, 2, 5, 3, 6]
# Memory:        [1, 2, 3, 4, 5, 6]  ← doesn't match!

t3 = t2.contiguous()
# Memory is NOW:  [1, 4, 2, 5, 3, 6]  ← NEW copy, matches logical order!
# Stride: (2, 1) ← contiguous again ✅

t3.view(6)  # ✅ Works! → [1, 4, 2, 5, 3, 6]
```

## Step 8: What is `reshape`?

`reshape` = tries `view` first, falls back to `contiguous().view()` if needed:

```python
def reshape(tensor, new_shape):
    if tensor.is_contiguous():
        return tensor.view(new_shape)              # fast path, no copy
    else:
        return tensor.contiguous().view(new_shape)  # copy + view
```

That's literally it!

```python
t2 = t.transpose(0, 1)   # not contiguous

t2.view(6)      # ❌ CRASHES
t2.reshape(6)   # ✅ Works (internally does contiguous + view)
```

## Step 9: Why Use `contiguous().view()` Instead of `reshape`?

Both work. The reasons to prefer `contiguous().view()`:

**1. Explicitness:**
```python
# contiguous().view() — makes it OBVIOUS:
# "I know this tensor is non-contiguous, I'm explicitly fixing it"
x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

# reshape — hides this detail
x = x.transpose(1, 2).reshape(batch_size, seq_len, d_model)
```

**2. Debugging — `view` acts as a health check:**
If some upstream code **unexpectedly** gives you a non-contiguous tensor when you expected a contiguous one:

```python
# With view:
x.view(new_shape)   # ❌ CRASHES → "Hey, something is wrong upstream!"
# You investigate and find the bug

# With reshape:
x.reshape(new_shape) # ✅ Silently works → you never notice the upstream bug
```

**Upstream code: Code that runs earlier and produces the input or data used by the current code.*

`view` crashes and alerts you. `reshape` quietly fixes it, so you might never know something was wrong upstream.

**3. Convention:** It's the standard convention in PyTorch Transformer code.

**Both are correct!** It's a style preference, not a correctness issue.

## Summary Table

| Operation | Moves data? | Needs contiguous? | Changes strides? |
|-----------|------------|-------------------|------------------|
| `view` | ❌ Never | ✅ Yes (crashes otherwise) | Recalculates (keeps sequential) |
| `transpose` | ❌ Never | ❌ No | Swaps (can break sequential) |
| `contiguous()` | ✅ Copies | N/A | Resets to sequential |
| `reshape` | Maybe | ❌ Handles it | Auto |

## Why This Matters for Multi-Head Attention

In `combine_heads`:

```python
x = x.transpose(1, 2)              # swaps strides → NOT contiguous
x = x.contiguous().view(...)        # fix memory → then reshape
```

We use `contiguous().view()` instead of `reshape` because it's **explicit** — it tells the reader "I know transpose broke contiguity, I'm fixing it here."

---

# Xavier Initialization: Why and How

## The Problem with `nn.Linear`'s Default Init

When you create `nn.Linear(512, 512)`, PyTorch initializes the weight matrix using **Kaiming uniform** by default. Kaiming was designed for layers followed by **ReLU** activations — it accounts for the fact that ReLU kills ~50% of values (all negatives become zero), so it makes weights ~√2 larger to compensate.

But in Multi-Head Attention, the projection layers (`W_q`, `W_k`, `W_v`, `W_o`) are **not** followed by ReLU. They feed into dot products, softmax, and linear combinations. Using Kaiming here means the initial weights are slightly too large for our use case — the variance assumption doesn't match reality.

## What Xavier Does

Xavier initialization (also called Glorot initialization) assumes **no activation function** (or a linear/symmetric one like tanh). It draws weights so that the variance of the output equals the variance of the input, preventing signals from exploding or vanishing as they pass through layers.

The formula balances both the forward pass and backward pass:

```
Variance = 2 / (fan_in + fan_out)
```

Where `fan_in` = number of input features, `fan_out` = number of output features.

For our `W_q = nn.Linear(512, 512)`:
```
fan_in = 512, fan_out = 512
Variance = 2 / (512 + 512) = 1/512 ≈ 0.00195
```

## Uniform vs Normal

Xavier comes in two flavors:

- **Xavier uniform**: draws from `U(-a, a)` where `a = √(6 / (fan_in + fan_out))`
- **Xavier normal**: draws from `N(0, σ²)` where `σ = √(2 / (fan_in + fan_out))`

Both produce the **same variance** — `2 / (fan_in + fan_out)`. The difference is only in the shape of the distribution:

- **Uniform**: bounded values, no extreme outliers
- **Normal**: bell curve, most values near zero with rare large ones

In practice, the difference is negligible. Training converges to essentially the same result with either. We use **uniform** because that's what PyTorch's official `nn.MultiheadAttention` uses in their `_reset_parameters` — it's a codebase convention, not a theoretical requirement.

The original paper ("Attention Is All You Need") doesn't specify uniform vs normal — just that they used Xavier.

## Why This Matters

Will your model break without Xavier? **No.** Kaiming (the default) will still work — the model will converge. But Xavier gives theoretically correct variance scaling for these projection layers, which can mean:

- More stable gradients in early training
- Slightly faster convergence (fewer wasted early steps)

It's a small correctness detail, not a make-or-break decision.

```python
def _reset_parameters(self):
    """Xavier uniform init for projections; zero bias."""
    for module in [self.W_q, self.W_k, self.W_v, self.W_o]:
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
```

---

# The `_` Prefix Convention: `_reset_parameters`

## What `_` Means in Python

In Python, a single underscore prefix (`_method_name`) is a **naming convention** that communicates: "this is an internal helper — not part of the public API." It's a signal to other developers (and your future self): "you shouldn't need to call this directly."

It is **not enforced** by Python. There's no access control like `private` in Java or C++. You can still call `obj._reset_parameters()` from outside the class — Python won't stop you. It's purely about **communicating intent**.

## Why `_reset_parameters` Uses `_`

`_reset_parameters` is called **once**, inside `__init__`, to set up initial weight values. No external code should ever need to call it:

```python
def __init__(self, d_model, num_heads, dropout=0.1):
    super().__init__()
    # ... create layers ...
    self._reset_parameters()   # ← called once, internally
```

The `_` prefix tells the reader: "this exists to keep `__init__` clean — it's an implementation detail, not a feature of the class."

This also follows **PyTorch's own convention** — their official `nn.MultiheadAttention` names it `_reset_parameters` too.

## Why `split_heads` and `combine_heads` Don't Use `_`

By the same logic, `split_heads` and `combine_heads` are also internal helpers — no external code calls them directly. So `_split_heads` and `_combine_heads` would be equally valid.

We kept them **without** `_` because:

1. **They're meaningful, self-contained operations** — someone debugging attention might want to call `split_heads` directly to inspect intermediate tensor shapes
2. **Readability in `forward()`** — `self.split_heads(Q)` reads more naturally than `self._split_heads(Q)`
3. **A subclass might override them** — if someone extends `MultiHeadAttention` with a different head-splitting strategy, having them "public" makes that cleaner

It's a judgment call, not a rule. Both conventions are correct. The key principle: use `_` when a method is a one-time setup/utility that nobody should call from outside. Skip `_` when the method is a meaningful operation that could reasonably be inspected or overridden.

---

# `nn.init.zeros_` vs `nn.init.constant_` vs `torch.zeros`

## The Core Difference: In-Place vs Replacement

These three look similar but do fundamentally different things:

### `nn.init.zeros_(tensor)` and `nn.init.constant_(tensor, value)`

Both modify an **existing tensor in-place** — they overwrite the values inside the tensor without creating a new object. The `_` suffix means "in-place operation" (a PyTorch convention).

```python
# These two are equivalent:
nn.init.zeros_(self.W_q.bias)           # fills with 0.0
nn.init.constant_(self.W_q.bias, 0.0)   # fills with 0.0

# zeros_ is just shorthand for constant_(tensor, 0.)
```

The critical part: the `nn.Parameter` object **stays the same**. It's still registered with the module, still tracked by autograd, still visible to the optimizer. You're just changing what numbers are stored inside it.

### `torch.zeros(...)` — Creates a New Tensor

```python
self.W_q.bias = torch.zeros(512)   # ❌ DON'T DO THIS
```

This **replaces** the `nn.Parameter` with a plain `torch.Tensor`. The consequences:

- **Autograd breaks** — a plain tensor doesn't track gradients by default
- **Optimizer breaks** — the optimizer holds a reference to the old `nn.Parameter`, not the new tensor
- **`model.parameters()` breaks** — the new tensor isn't registered as a parameter

It's like replacing a tracked package with an untracked one — the system loses visibility of it.

### Why It Matters

```python
# ✅ CORRECT: modifies values in-place, parameter stays registered
nn.init.zeros_(self.W_q.bias)
# Before: Parameter([0.012, -0.034, 0.007, ...])
# After:  Parameter([0.0,    0.0,   0.0,   ...])  ← same Parameter object

# ❌ WRONG: replaces the Parameter entirely
self.W_q.bias = torch.zeros(512)
# Before: Parameter([0.012, -0.034, 0.007, ...])
# After:  Tensor([0.0, 0.0, 0.0, ...])  ← different object, NOT a Parameter
```

### Summary

| Function | What it does | Creates new tensor? | Safe for `nn.Parameter`? |
|----------|-------------|--------------------|-----------------------|
| `nn.init.zeros_(t)` | Fills `t` with zeros in-place | No | Yes |
| `nn.init.constant_(t, val)` | Fills `t` with `val` in-place | No | Yes |
| `torch.zeros(...)` | Creates a brand new zero tensor | Yes | No (breaks registration) |

**Rule of thumb:** When initializing weights/biases of `nn.Module` layers, always use `nn.init.*` functions. They modify values in-place without breaking the parameter system.

