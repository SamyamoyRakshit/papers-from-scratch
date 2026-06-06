## Table of Contents

1. [Why Masks?](#why-masks)
2. [Mask Convention](#mask-convention)
3. [The Three Masks](#the-three-masks)
4. [Code Explanation with Example](#code-explanation-with-example)
   - [`create_padding_mask`](#create_padding_mask)
   - [`create_causal_mask`](#create_causal_mask)
   - [`create_src_mask`](#create_src_mask)
   - [`create_tgt_mask`](#create_tgt_mask)
   - [`create_memory_mask`](#create_memory_mask)
5. [How `masked_fill` Uses These Masks](#how-masked_fill-uses-these-masks)
6. [Where Masks Are Created — Outside the Model](#where-masks-are-created--outside-the-model)
7. [Shape Summary](#shape-summary)

---

# Why Masks?

Two problems masks solve:

1. **Padding** — Sentences have different lengths, so shorter ones get `<pad>` tokens. The model should **ignore** these padding positions.
2. **Causality** — In the decoder, token at position `t` should only see positions `0` to `t` (not the future). Otherwise the model cheats during training by looking ahead.

---

# Mask Convention

Throughout the codebase, masks use:

```
1 = attend (keep this position)
0 = ignore (block this position)
```

This convention comes from `multi_head_attention.py` line 89:

```python
scores = scores.masked_fill(mask == 0, float('-inf'))
```

Where `mask == 0` → position is blocked → score becomes `-inf` → softmax turns it to `0` → zero attention weight.

---

# The Three Masks

| Mask | Used In | What It Blocks | Shape |
|---|---|---|---|
| `src_mask` | Encoder self-attention | Source padding tokens | `(batch, 1, 1, src_seq_len)` |
| `tgt_mask` | Decoder self-attention | Target padding + future tokens | `(batch, 1, tgt_seq_len, tgt_seq_len)` |
| `memory_mask` | Decoder cross-attention | Source padding tokens | `(batch, 1, 1, src_seq_len)` |

**Why does `memory_mask` use `src`, not `tgt`?**

Because in cross-attention, Q comes from the decoder but **K and V come from the encoder output** (source). So we need to mask the **source** padding positions — those are the positions being attended *to*.

---

# Code Explanation with Example

Input sentence: `"I love AI <pad> <pad>"` with `pad_idx = 0`

```python
src = [[14, 87, 3, 0, 0]]   # shape: (1, 5) — raw token IDs, NOT embeddings
```

**Important:** Mask functions receive **raw token IDs** `(batch_size, seq_len)`, not embedded tensors `(batch_size, seq_len, d_model)`. Masks only need to know *which positions are padding* — a yes/no per token. They don't care about the 512-dim embedding values.

---

## `create_padding_mask`

```python
def create_padding_mask(seq: torch.Tensor, pad_idx: int) -> torch.Tensor:
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)
    # shape: (batch_size, 1, 1, seq_len)
```

Step by step:

```
seq = [[14, 87, 3, 0, 0]]              # (1, 5)

(seq != pad_idx)
→ [[True, True, True, False, False]]    # (1, 5)

.unsqueeze(1)
→ [[[True, True, True, False, False]]]  # (1, 1, 5)

.unsqueeze(2)
→ [[[[True, True, True, False, False]]]]  # (1, 1, 1, 5)
```

Result (as integers):

```
mask = [[[[1, 1, 1, 0, 0]]]]   # shape: (1, 1, 1, 5)
```

**Why two unsqueezes?**

Attention scores shape is `(batch_size, num_heads, seq_len_q, seq_len_k)`. The mask needs to broadcast across:
- `dim=1` → all attention heads (unsqueeze(1))
- `dim=2` → all query positions (unsqueeze(2))

So `(batch, 1, 1, seq_len)` broadcasts to `(batch, num_heads, seq_len_q, seq_len_k)`.

---

## `create_causal_mask`

```python
def create_causal_mask(size: int) -> torch.Tensor:
    return torch.tril(torch.ones(size, size)).unsqueeze(0).unsqueeze(0)
    # shape: (1, 1, size, size)
```

Step by step (size = 4):

**1. `torch.ones(4, 4)`** — all ones:

```
[[1, 1, 1, 1],
 [1, 1, 1, 1],
 [1, 1, 1, 1],
 [1, 1, 1, 1]]
```

**2. `torch.tril(...)`** — keep only lower triangle (`tril` = **tri**angle **l**ower):

```
[[1, 0, 0, 0],
 [1, 1, 0, 0],
 [1, 1, 1, 0],
 [1, 1, 1, 1]]
```

Reading each row: token 0 can see only itself, token 1 can see tokens 0-1, token 2 can see 0-2, etc. **No token can see the future.**

**3. `.unsqueeze(0).unsqueeze(0)`** — add batch and head dimensions:

```
(4, 4) → (1, 4, 4) → (1, 1, 4, 4)
```

The `(1, 1, ...)` broadcasts across all batches and all heads.

### Visual

```
Token:    "I"    "love"   "AI"   "too"

"I"     [  1      0       0       0  ]   <- can only see "I"
"love"  [  1      1       0       0  ]   <- can see "I", "love"
"AI"    [  1      1       1       0  ]   <- can see "I", "love", "AI"
"too"   [  1      1       1       1  ]   <- can see everything before it
```

This is why it's called a **causal** mask — it enforces causality (can't peek at the future) in the decoder's self-attention.

### What Does "Causal" Mean?

**Causal** = relating to cause and effect — things happen in order, the past causes the future.

In the decoder, a token at position `t` was **generated from** tokens at positions `0` to `t-1`. So it should only see the past, not the future. The lower-triangular mask enforces this natural cause → effect ordering:

```
Position 0 → generated first  (sees nothing before it)
Position 1 → generated second (sees position 0)
Position 2 → generated third  (sees positions 0, 1)
```

If position 2 could see position 3, it would be using the **effect** to predict the **cause** — that's anti-causal, and it's cheating.

---

## `create_src_mask`

```python
def create_src_mask(src: torch.Tensor, pad_idx: int) -> torch.Tensor:
    return create_padding_mask(src, pad_idx)
    # shape: (batch_size, 1, 1, src_seq_len)
```

Just padding mask. The encoder has no causal constraint — every source token can attend to every other source token (except padding).

---

## `create_tgt_mask`

```python
def create_tgt_mask(tgt: torch.Tensor, pad_idx: int) -> torch.Tensor:
    padding_mask = create_padding_mask(tgt, pad_idx)
    causal_mask = create_causal_mask(tgt.size(1)).to(tgt.device)
    return padding_mask & causal_mask
    # shape: (batch_size, 1, tgt_seq_len, tgt_seq_len)
```

**What is `tgt.size(1)`?**

`tgt.size(1)` returns the length of dimension 1 — which is `seq_len`:

```python
tgt = [[14, 87, 3, 0]]
#       ↑ dim=0 (batch)
#           ↑ dim=1 (seq_len)

tgt.size(0)  →  1       # batch_size
tgt.size(1)  →  4       # seq_len
```

We pass it to `create_causal_mask(tgt.size(1))` because the causal mask needs to be `(seq_len x seq_len)` — a square matrix where each row says "which positions can this token attend to?" So `create_causal_mask(4)` creates a `(4, 4)` lower-triangle matrix.

This is the only mask that combines **two** constraints:

**Example:** `tgt = [[14, 87, 3, 0]]` (4 tokens, last is padding)

**Padding mask** — shape `(1, 1, 1, 4)`:

```
[[[[1, 1, 1, 0]]]]
```

**Causal mask** — shape `(1, 1, 4, 4)`:

```
[[[[1, 0, 0, 0],
   [1, 1, 0, 0],
   [1, 1, 1, 0],
   [1, 1, 1, 1]]]]
```

**`padding_mask & causal_mask`** — broadcasting `(1,1,1,4)` & `(1,1,4,4)` → `(1,1,4,4)`:

`&` is bitwise AND — compares element by element. A position is `1` (attend) **only if both masks say `1`**:

```
1 & 1 = 1   (both say attend → attend)
1 & 0 = 0   (one says block → block)
0 & 1 = 0   (one says block → block)
0 & 0 = 0   (both say block → block)
```

Element-by-element:

```
     col0  col1  col2  col3(pad)
row0 [ 1&1,  0&1,  0&1,  0&0 ]     [[1, 0, 0, 0],
row1 [ 1&1,  1&1,  0&1,  0&0 ]  →   [1, 1, 0, 0],
row2 [ 1&1,  1&1,  1&1,  0&0 ]      [1, 1, 1, 0],
row3 [ 1&1,  1&1,  1&1,  1&0 ]      [1, 1, 1, 0]]
                                               ↑
                                     column 3 is ALL zeros
                                     (nobody can attend to pad)
```

### Why Row 3 Column 3 Matters

Look at row 3 (last row) in the **causal mask alone**:

```
row3 [1, 1, 1, 1]   ← causal mask says: "token 3 can see ALL positions"
```

The causal mask allowed this — token 3 seeing itself is not "future", it's the present. But token 3 is `<pad>`, and column 3 represents "can anyone attend **to** token 3?"

Without the padding mask, the causal mask would let tokens attend to the padding position. The causal mask only cares about time — it doesn't know which tokens are padding.

The `&` fixes this:

```
Causal row3:   [1, 1, 1, 1]    ← "future is fine, you can see everything"
Padding:       [1, 1, 1, 0]    ← "but position 3 is padding — block it"
                         &
Result row3:   [1, 1, 1, 0]    ← "see positions 0-2, but NOT position 3 (pad)"
```

And for **column 3** across all rows:

```
              col3
              ----
Causal:   row0 [ 0 ]     Padding: [ 0 ]     Result: [ 0 ]
          row1 [ 0 ]              [ 0 ]              [ 0 ]
          row2 [ 0 ]              [ 0 ]              [ 0 ]
          row3 [ 1 ] ← allowed!  [ 0 ] ← blocked!   [ 0 ] ← blocked by &
```

Row 3 col 3 was `1` in the causal mask (token 3 seeing itself — not "future"). But the padding mask says `0` (it's padding — nobody should attend to it). The `&` takes the stricter rule.

### One-Line Summary

**Causal mask handles time (no future), padding mask handles content (no `<pad>`). The `&` enforces both rules simultaneously.**

**Why `.to(tgt.device)`?**

The causal mask is created from `torch.tril(torch.ones(...))` which defaults to CPU. If the target tensor is on GPU, we need to move the mask there too, otherwise `&` would crash (CPU & GPU = RuntimeError).

---

## `create_memory_mask`

```python
def create_memory_mask(src: torch.Tensor, pad_idx: int) -> torch.Tensor:
    return create_padding_mask(src, pad_idx)
    # shape: (batch_size, 1, 1, src_seq_len)
```

Same as `src_mask` — just source padding. Used in decoder's cross-attention where K and V come from encoder output (source sequence).

**Why the parameter is `src`, not `tgt`?**

Cross-attention: Q is from decoder (target), but K and V are from encoder (source). The mask blocks positions in K/V — which are source positions. So we mask source padding.

**Important: `src` here means raw token IDs, NOT encoder output.**

The mask is created **before** the model runs — from raw token IDs where we can still see which positions are `pad_idx`. By the time the encoder transforms them into 512-dim vectors, padding is indistinguishable from real tokens. The mask remembers what the model forgot.

```
1. Training loop: src = [14, 87, 3, 0, 0]   ← can see pad_idx → create mask
2. Embedding: [vec, vec, vec, vec, vec]       ← can't tell which are padding anymore
3. Encoder output: [vec, vec, vec, vec, vec]  ← same problem
4. Cross-attention uses mask from step 1      ← mask tells us what vectors can't
```

---

# How `masked_fill` Uses These Masks

In `multi_head_attention.py` line 88-89:

```python
if mask is not None:
    scores = scores.masked_fill(mask == 0, float('-inf'))
```

Full trace with padding mask:

```
scores = [[[[0.5, 0.3, 0.8, 0.2, 0.1]]]]    # (1, 1, 1, 5)
mask   = [[[[  1,   1,   1,   0,   0]]]]      # (1, 1, 1, 5)

mask == 0 → [[[[False, False, False, True, True]]]]

After masked_fill:
scores = [[[[0.5, 0.3, 0.8, -inf, -inf]]]]

After softmax:
weights = [[[[0.27, 0.23, 0.50, 0.0, 0.0]]]]
```

`e^(-inf) = 0` — padding tokens get **zero attention weight**. The model completely ignores them.

**One line of code handles all three mask types** because they all use the same convention (1 = attend, 0 = ignore):
- **Padding mask** `(1,1,1,5)`: zeros at pad positions → broadcasts across all query positions
- **Causal mask** `(1,1,4,4)`: zeros in upper triangle → each row blocks future tokens
- **Combined** `(padding & causal)`: zeros at both pad AND future positions

---

# Where Masks Are Created — Outside the Model

Masks are created in the **training loop**, not inside the model:

```python
# In training loop (before calling model.forward):
src_mask = create_src_mask(src, pad_idx)        # src is (batch, seq_len) — token IDs
tgt_mask = create_tgt_mask(tgt, pad_idx)        # tgt is (batch, seq_len) — token IDs
memory_mask = create_memory_mask(src, pad_idx)

# Then pass everything in:
output = model(src, tgt, src_mask, tgt_mask, memory_mask)
```

The flow:

```
Token IDs (batch, seq_len)
    |-- mask_utils sees THIS <- integer IDs, 2D
    |
    v
Embedding + PE -> (batch, seq_len, d_model) <- 3D
    |
    v
Attention receives the 3D tensor + the mask from above
```

---

# Shape Summary

| Function | Input Shape | Output Shape | Broadcasts To |
|---|---|---|---|
| `create_padding_mask` | `(batch, seq_len)` | `(batch, 1, 1, seq_len)` | `(batch, heads, any_q, seq_len)` |
| `create_causal_mask` | `int` (size) | `(1, 1, size, size)` | `(batch, heads, size, size)` |
| `create_src_mask` | `(batch, src_len)` | `(batch, 1, 1, src_len)` | `(batch, heads, src_len, src_len)` |
| `create_tgt_mask` | `(batch, tgt_len)` | `(batch, 1, tgt_len, tgt_len)` | `(batch, heads, tgt_len, tgt_len)` |
| `create_memory_mask` | `(batch, src_len)` | `(batch, 1, 1, src_len)` | `(batch, heads, tgt_len, src_len)` |
