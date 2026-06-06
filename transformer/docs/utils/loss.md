## Table of Contents

1. [What the Paper Says](#what-the-paper-says)
2. [Label Smoothing — The Idea](#label-smoothing--the-idea)
3. [Why KL Divergence Instead of Cross-Entropy](#why-kl-divergence-instead-of-cross-entropy)
4. [Code Explanation](#code-explanation)
   - [`__init__`](#__init__)
   - [`forward`](#forward)
5. [End-to-End: One Sentence Through Label-Smoothed Loss](#end-to-end-one-sentence-through-label-smoothed-loss)
   - [Setup](#setup)
   - [Step 1 — Prepare Target for Training](#step-1--prepare-target-for-training)
   - [Step 2 — Batching with Padding](#step-2--batching-with-padding)
   - [Step 3 — Model Predicts Logits](#step-3--model-predicts-logits)
   - [Step 4 — Flatten Before Loss](#step-4--flatten-before-loss)
   - [Step 5 — Forward Through LabelSmoothedLoss](#step-5--forward-through-labelsmoothedloss)
6. [The Smoothed Distribution — Why V-2](#the-smoothed-distribution--why-v-2)
7. [PyTorch Alternative](#pytorch-alternative)
8. [KL Divergence — Extra Computation Cost](#kl-divergence--extra-computation-cost)
   - [Without Label Smoothing — CE Is Cheap](#without-label-smoothing--ce-is-cheap)
   - [With Label Smoothing — Every Entry Matters](#with-label-smoothing--every-entry-matters)
   - [Concrete Example — vocab_size=8](#concrete-example--vocab_size8-position-0-correct--index-3)
   - [GPU Parallelism — Why It's Still Fast](#gpu-parallelism--why-its-still-fast)
   - [Compare With Attention — The Real Bottleneck](#compare-with-attention--the-real-bottleneck)
9. [The Full Picture](#the-full-picture)
10. [References](#references)

---

# What the Paper Says

From **"Attention Is All You Need"** (Vaswani et al., 2017), Section 5.4 — Regularization:

> "During training, we employed label smoothing of value ε_ls = 0.1 [36]. This hurt perplexity, as the model learns to be more unsure, but improved accuracy and BLEU score."

The `[36]` citation points to: **"Rethinking the Inception Architecture"** (Szegedy et al., 2016) — the paper that **invented** label smoothing.

---

# Label Smoothing — The Idea

## Standard Cross-Entropy (no label smoothing)

The target is a **one-hot** vector — all probability on the correct token:

```
Target for correct token "আমি" (index 3, vocab_size=8):
[0, 0, 0, 1, 0, 0, 0, 0]
            ↑ index 3 = 1.0, everything else = 0.0
```

The loss is: `-log(P(correct_token))`. The model is pushed to make the correct token's probability as close to 1.0 as possible.

**Problem:** The model becomes **overconfident** — it learns to output extreme probabilities (0.99+) instead of useful probability distributions. This hurts generalization.

## With Label Smoothing (ε = 0.1)

Instead of one-hot `[0, 0, 0, 1, 0, 0, 0, 0]`, we "smooth" the target:

```
ε = 0.1, vocab_size = 8

Without smoothing: [0,      0,      0,      1.0,    0,      0,      0,      0     ]
With smoothing:    [0,      0.0167, 0.0167, 0.9,    0.0167, 0.0167, 0.0167, 0.0167]
                    ↑ pad=0                  ↑ correct token
```

Target distribution:

```
p(correct)    = 1 - ε           = 0.9
p(each other) = ε / (V - 2)    = 0.0167
p(pad)        = 0
```

Instead of "be 100% sure", we say "be ~90% sure and spread 10% across everything else."

---

# Why KL Divergence Instead of Cross-Entropy

The paper describes **cross-entropy with a smoothed target distribution**. The math is:

```
loss = -Σ p(k) × log q(k)       ← this is cross-entropy
```

But `nn.CrossEntropyLoss` before PyTorch 1.10 (2021) only accepted **integer targets** (token IDs like `[3, 4, 5]`), not soft distributions like `[0.0167, 0.0167, 0.9, ...]`. (PyTorch 1.10+ and 2.x now support `label_smoothing=` natively — see [Section 7](#pytorch-alternative)).

`nn.KLDivLoss` accepts **soft distributions**. And mathematically:

```
KL(p || q) = Σ p × (log p - log q)
           = Σ p × log p  -  Σ p × log q
             ^^^^^^^^^^^     ^^^^^^^^^^^^^
             -H(p)           CE(p, q)
             constant!       what actually matters
```

Since the smoothed target `p` is fixed, `H(p)` is a constant — it disappears in gradients:

```
∂KL/∂θ = ∂CE/∂θ     (identical training, same weight updates)
```

**The KLDivLoss approach was not what the paper intended — it was a workaround** because PyTorch's `CrossEntropyLoss` couldn't accept soft targets. Early implementations (Harvard's Annotated Transformer, fairseq, tensor2tensor) all used this workaround.

We use KLDivLoss in our implementation to **replicate from scratch** — building it manually teaches exactly what label smoothing does (build soft distribution, compare with KL). This is the standard approach in paper replications and open-source transformer implementations.

---

# Code Explanation

## `__init__`

```python
def __init__(self, pad_idx: int, smoothing: float = 0.1):
    super().__init__()
    self.pad_idx = pad_idx
    self.smoothing = smoothing
    self.confidence = 1 - smoothing  # 0.9
    self.criterion = nn.KLDivLoss(reduction='sum')
```

**Why `reduction='sum'`?**

```python
# reduction='mean' would do:
loss = total_loss / ALL_elements     # ← includes padding rows (wrong!)

# reduction='sum' + manual division:
loss = total_loss / non_pad_tokens   # ← only real tokens (correct!)
```

If we used `'mean'`, PyTorch divides by every element including zeroed-out padding rows — diluting the loss. With `'sum'`, we control the denominator ourselves.

## `forward`

```python
def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    vocab_size = logits.size(-1)
    log_probs = torch.log_softmax(logits, dim=-1)

    smooth_value = self.smoothing / (vocab_size - 2)
    smoothed = torch.full_like(log_probs, smooth_value)
    smoothed.scatter_(1, target.unsqueeze(1), self.confidence)
    smoothed[:, self.pad_idx] = 0

    pad_mask = target == self.pad_idx
    smoothed[pad_mask] = 0

    n_tokens = (~pad_mask).sum().item()
    loss = self.criterion(log_probs, smoothed)
    return loss / n_tokens
```

**Two formulas in the code:**

**Formula 1 — Label smoothing builds the target distribution `p`:**

```
loss = (1 - ε) × loss(correct_class) + (ε/N) × Σ loss(other_classes)
```

The code builds this directly:

```python
smooth_value = self.smoothing / (vocab_size - 2)           # ε/N for each wrong class
smoothed = torch.full_like(log_probs, smooth_value)         # fill ALL with ε/N
smoothed.scatter_(1, target.unsqueeze(1), self.confidence)  # put 0.9 at correct class
smoothed[:, self.pad_idx] = 0                                # pad gets 0
```

**Formula 2 — KL divergence measures distance:**

```
KL(p || q) = Σ p × (log p - log q)
```

Where `p` = `smoothed` (our target), `q` = model's predictions, `log q` = `log_probs`.

Expanding for one row:

```
KL = 0.9 × (log 0.9 - log q_correct)          ← (1-ε) × loss(correct)
   + (ε/N) × (log(ε/N) - log q_1)             ←
   + (ε/N) × (log(ε/N) - log q_2)             ← (ε/N) × Σ loss(others)
   + ...                                        ←
```

The `log p` parts (`log 0.9`, `log(ε/N)`) are **constants** — they don't depend on the model. Gradients ignore them. Only `-p × log q` drives learning.

---

# End-to-End: One Sentence Through Label-Smoothed Loss

## Setup

```
Source (English):  "I love AI"
Target (Bengali):  "আমি AI ভালোবাসি"

Vocabulary (tiny, size V=8):
  0: <pad>
  1: <sos>
  2: <eos>
  3: আমি
  4: AI
  5: ভালোবাসি
  6: হ্যালো
  7: বিড়াল
```

## Step 1 — Prepare Target for Training

Teacher forcing: model gets `<sos> + target` as input and predicts `target + <eos>`:

```
Decoder input:    [<sos>, আমি, AI, ভালোবাসি]     → what the model sees
Expected output:  [আমি,  AI,  ভালোবাসি, <eos>]   → what the model should predict
```

As token IDs:

```
decoder_input = [1, 3, 4, 5]
target        = [3, 4, 5, 2]
```

## Step 2 — Batching with Padding

Another sentence in the batch is shorter, so we pad:

```
target = [[3, 4, 5, 2],      ← "আমি AI ভালোবাসি <eos>"
          [3, 2, 0, 0]]      ← "আমি <eos> <pad> <pad>"

Shape: (batch=2, seq_len=4)
```

## Step 3 — Model Predicts Logits

The model (encoder + decoder) outputs raw scores for every vocab token at every position:

```
logits shape: (2, 4, 8)  →  (batch, seq_len, vocab_size)
```

Where do logits come from? From `transformer.py` line 103:

```python
logits = self.output_projection(decoder_output)    # (batch, seq_len, vocab_size)
```

For sentence 1, position 0 (should predict আমি = idx 3):

```
logits[0][0] = [0.1, -0.5, 0.3, 2.8, 0.2, -0.1, 0.4, -0.3]
                                  ↑ highest score at idx 3 (good!)
```

The model doesn't "know" it's আমি — it just outputs scores for all 8 vocab indices. **We** know index 3 is আমি because **we** built the vocabulary mapping.

## Step 4 — Flatten Before Loss

The loss function expects **already-flattened** inputs. Flattening is the caller's responsibility — it happens in the **training loop**, not inside the loss function. This will be implemented in `train_utils.py`:

```python
# In train_utils.py:
logits = model(src, tgt)                 # (batch, seq_len, vocab_size)
logits = logits.view(-1, vocab_size)     # (batch * seq_len, vocab_size)
target = target.view(-1)                 # (batch * seq_len,)

loss = criterion(logits, target)         # loss expects flattened inputs
```

For our example:

```
target = [3, 4, 5, 2, 3, 2, 0, 0]
          ↑              ↑        ↑  ↑
          real tokens     real   pad  pad
```

After flattening, there's no concept of "which batch" or "which sentence." Every row is just: **here's 8 scores, the right answer is this index.**

The flattened logits look like:

```
target (before flatten): [[3, 4, 5, 2],     ← sentence 1
                          [3, 2, 0, 0]]     ← sentence 2

target (after flatten):  [3, 4, 5, 2, 3, 2, 0, 0]     ← (8,)

logits (after flatten):  (8, 8) — 8 rows, each with 8 vocab scores:

         <pad>  <sos>  <eos>  আমি   AI    ভালো   হ্যালো  বিড়াল
         idx0   idx1   idx2   idx3  idx4  idx5   idx6   idx7
row 0: [ 0.1,  -0.5,   0.3,  2.8,  0.2, -0.1,   0.4,  -0.3]  target=3 (আমি)
row 1: [-0.2,   0.1,   0.4,  0.3,  3.1,  0.2,  -0.5,   0.1]  target=4 (AI)
row 2: [ 0.0,  -0.3,   0.1,  0.2,  0.3,  2.9,   0.1,  -0.2]  target=5 (ভালো)
row 3: [ 0.1,   0.2,   3.0,  0.1, -0.1,  0.3,  -0.4,   0.2]  target=2 (<eos>)
row 4: [ 0.3,  -0.4,   0.2,  2.5,  0.1,  0.4,  -0.2,   0.1]  target=3 (আমি)
row 5: [ 0.1,   0.3,   2.7, -0.1,  0.2,  0.1,   0.3,  -0.5]  target=2 (<eos>)
row 6: [ 1.2,   0.3,   0.1,  0.4,  0.2, -0.1,   0.5,   0.3]  target=0 (pad) ← ignored
row 7: [ 0.8,   0.1,   0.5,  0.2,  0.3,  0.1,  -0.2,   0.4]  target=0 (pad) ← ignored
```

Each row: model's 8 guesses. Loss checks if the highest score matches the target index.

Rows 6-7 get zeroed out — padding positions, don't count.

## Step 5 — Forward Through LabelSmoothedLoss

Let's trace **row 0** (target = 3, আমি) and **row 7** (target = 0, `<pad>`):

### 5a. log_softmax

```python
log_probs = torch.log_softmax(logits, dim=-1)   # (8, 8)
```

Row 0: `[0.1, -0.5, 0.3, 2.8, ...]` → `[-2.8, -3.4, -2.6, -0.1, ...]`

`log_softmax` = `log(softmax(x))`. Converts raw scores into log-probabilities. KLDivLoss expects log-probabilities as input.

### 5b. Build smoothed target

```python
smooth_value = 0.1 / (8 - 2) = 0.01667
smoothed = torch.full_like(log_probs, 0.01667)   # (8, 8) all 0.01667
```

Row 0 after fill:

```
[0.01667, 0.01667, 0.01667, 0.01667, 0.01667, 0.01667, 0.01667, 0.01667]
 idx0     idx1     idx2     idx3     idx4     idx5     idx6     idx7
```

### 5c. Scatter confidence at correct token

```python
smoothed.scatter_(1, target.unsqueeze(1), 0.9)
```

`scatter_` places a value at a specific index in each row.

```python
target = [3, 4, 5, 2, 3, 2, 0, 0]    # shape: (8,)

target.unsqueeze(1)                     # shape: (8, 1)
= [[3],
   [4],
   [5],
   [2],
   [3],
   [2],
   [0],
   [0]]
```

`scatter_(1, target.unsqueeze(1), 0.9)` means:

- `1` → operate along dimension 1 (columns)
- `target.unsqueeze(1)` → which column in each row
- `0.9` → the value to place

```
Row 0: put 0.9 at column 3  → [0.017, 0.017, 0.017, 0.9, 0.017, 0.017, 0.017, 0.017]
Row 1: put 0.9 at column 4  → [0.017, 0.017, 0.017, 0.017, 0.9, 0.017, 0.017, 0.017]
Row 2: put 0.9 at column 5  → [0.017, 0.017, 0.017, 0.017, 0.017, 0.9, 0.017, 0.017]
Row 3: put 0.9 at column 2  → [0.017, 0.017, 0.9, 0.017, 0.017, 0.017, 0.017, 0.017]
...
```

It's like saying: **"for each row, I know which column is the correct answer — put 0.9 there."**

**Why `unsqueeze(1)`?** Because `scatter_` needs the index tensor to have the same number of dimensions as the source. `(8,)` → `(8, 1)` — one index per row.

Row 0 — target is 3:

```
[0.01667, 0.01667, 0.01667, 0.9, 0.01667, 0.01667, 0.01667, 0.01667]
                              ↑ idx 3 = আমি gets 0.9
```

### 5d. Zero out pad column

```python
smoothed[:, 0] = 0    # column 0 (pad_idx) → 0 for ALL rows
```

Row 0:

```
[0, 0.01667, 0.01667, 0.9, 0.01667, 0.01667, 0.01667, 0.01667]
 ↑ pad column zeroed

Count positions with 0.01667:
idx1 ✓  idx2 ✓  idx4 ✓  idx5 ✓  idx6 ✓  idx7 ✓  = 6 positions

Sum = 0 + 6 × 0.01667 + 0.9
    = 0 + 0.1 + 0.9
    = 1.0 ✓
```

### 5e. Zero out padding rows

```python
pad_mask = target == 0     # [F, F, F, F, F, F, T, T]
smoothed[pad_mask] = 0     # rows 6, 7 → all zeros
```

Row 7 (pad):

```
[0, 0, 0, 0, 0, 0, 0, 0]   ← contributes nothing to loss
```

**Two different things happening:**

- **Column zeroing** (`pad_idx` column) → "don't reward predicting `<pad>`" → affects the distribution
- **Row zeroing** (padding positions) → "don't compute loss for `<pad>` positions" → affects `n_tokens`

### 5f. Count real tokens

```python
n_tokens = (~pad_mask).sum() = 6    # rows 0-5 are real
```

### 5g. KL divergence

```python
loss = self.criterion(log_probs, smoothed)   # sum over all elements
```

For row 0 (আমি), the dominant term:

```
KL = 0.9 × (log(0.9) - (-0.1)) + 0.01667 × (log(0.01667) - (-2.8)) + ...
     ↑ correct token                ↑ wrong token
     big weight, small gap          tiny weight, big gap
```

Rows 6-7 (pad): smoothed is all zeros → contribute 0 to KL.

### 5h. Normalize

```python
return loss / 6    # divide by real tokens only
```

**Why divide by `n_tokens`?** Not to make computation less — to make **gradients consistent** across different batch sizes.

Without normalizing, a batch with 200 real tokens would produce a much larger total loss than a batch with 50 real tokens — simply because there are more terms being summed. The model would learn faster from big batches and slower from small batches. That's not what we want — we want every batch to contribute **equally per token**.

```
Without normalizing (reduction='sum' only):
  Batch A: 200 tokens → total loss = 180.0 → big gradients
  Batch B:  50 tokens → total loss =  45.0 → small gradients
  
  Same model, same quality predictions, but Batch A pushes
  weights 4× harder just because it has more tokens.

With normalizing (loss / n_tokens):
  Batch A: 180.0 / 200 = 0.90 per token → consistent gradients
  Batch B:  45.0 /  50 = 0.90 per token → consistent gradients
  
  Both batches contribute equally per token.
```

"Normalize" here means "make comparable", not "make smaller."

**What about `total_loss` in training?** In `train_utils.py`, we multiply back by `n_tokens`:

```python
total_loss += loss.item() * n_tokens
```

This is **not** undoing the normalization. The division was for **gradients** (so the model learns consistently). The multiplication is for **logging** — to reconstruct the correct weighted epoch average:

```
Epoch average = Σ(loss × n_tokens) / Σ(n_tokens)
              = total weighted loss / total tokens across all batches
```

If we just averaged per-batch losses (`Σ loss / num_batches`), a batch with 50 tokens would count the same as a batch with 200 tokens — giving too much weight to small batches.

---

# The Smoothed Distribution — Why V-2

Two positions get special treatment, so the remaining tokens that receive `smooth_value` = `V - 2`:

```
Total: V positions
Minus correct token: -1  (gets confidence = 0.9)
Minus pad_idx:       -1  (gets 0)
= V - 2 positions that keep smooth_value
```

Proof with V=8:

```
Step 1: smooth_value = 0.1 / (8 - 2) = 0.1 / 6 = 0.01667

Step 2: fill all
[0.01667, 0.01667, 0.01667, 0.01667, 0.01667, 0.01667, 0.01667, 0.01667]
 idx0     idx1     idx2     idx3     idx4     idx5     idx6     idx7

Step 3: scatter 0.9 at target index 3
[0.01667, 0.01667, 0.01667, 0.9, 0.01667, 0.01667, 0.01667, 0.01667]
                              ↑

Step 4: zero out pad column (idx 0)
[0, 0.01667, 0.01667, 0.9, 0.01667, 0.01667, 0.01667, 0.01667]
 ↑

Count positions with 0.01667:
idx1 ✓  idx2 ✓  idx4 ✓  idx5 ✓  idx6 ✓  idx7 ✓  = 6 positions = V - 2

Sum = 0 + 6 × 0.01667 + 0.9
    = 0 + 0.1 + 0.9
    = 1.0 ✓
```

General proof:

```
Sum = confidence + (V - 2) × ε/(V - 2) + 0
    = (1 - ε)   + ε                     + 0
    = 0.9        + 0.1
    = 1.0 ✓
```

If we had used `V - 1` instead:

```
smooth_value = 0.1 / 7 = 0.01429

After all steps:
[0, 0.01429, 0.01429, 0.9, 0.01429, 0.01429, 0.01429, 0.01429]

Sum = 0 + 6 × 0.01429 + 0.9 = 0.0857 + 0.9 = 0.9857 ✗  (doesn't sum to 1)
```

The missing probability is what **would have gone to `pad_idx`** but got zeroed out — breaking the distribution.

**Note on pad_idx:** There is always exactly **one** `<pad>` token in the vocabulary. A vocabulary is a Python dictionary — each key is unique. No standard tokenizer (SentencePiece, BPE, HuggingFace, torchtext) will produce duplicate tokens. The code assumes one `pad_idx` — this is a design contract.

---

# PyTorch Alternative

Since PyTorch 1.10 (2021), `nn.CrossEntropyLoss` supports label smoothing natively:

```python
class LabelSmoothedLoss(nn.Module):
    """
    Label-Smoothed Cross-Entropy Loss (Section 5.4)

    Uses PyTorch's built-in label smoothing with CrossEntropyLoss.
    Padding tokens are excluded via ignore_index.

    Reference: "Rethinking the Inception Architecture" (Szegedy et al., 2016)

    Args:
        pad_idx (int): Index of the <pad> token. Excluded from loss.
        smoothing (float, optional): Label smoothing value (ε). Default is 0.1.
    """

    def __init__(self, pad_idx: int, smoothing: float = 0.1):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=smoothing,
            ignore_index=pad_idx
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits (torch.Tensor): Shape (batch_size * seq_len, vocab_size)
            target (torch.Tensor): Shape (batch_size * seq_len,)

        Returns:
            torch.Tensor: Scalar loss value.
        """
        return self.criterion(logits, target)
```

No manual KLDivLoss, no scatter, no manual padding handling. `CrossEntropyLoss` does it all.

**Timeline:**

```
2016 — Szegedy et al. invent label smoothing
2017 — "Attention Is All You Need" uses it (ε = 0.1)
2017-2021 — everyone uses KLDivLoss workaround (PyTorch can't do it natively)
2021 — PyTorch 1.10 adds label_smoothing param to CrossEntropyLoss
```

**We use the manual KLDivLoss approach** because we're replicating from scratch — building it manually teaches exactly what label smoothing does. For production code, `CrossEntropyLoss(label_smoothing=0.1, ignore_index=pad_idx)` is the cleaner one-line choice.

---

# KL Divergence — Extra Computation Cost

## Without Label Smoothing — CE Is Cheap

Standard cross-entropy with one-hot targets only needs **one** lookup:

```
Target (one-hot):  [0, 0, 0, 0.9, 0, 0, 0, 0]    ← but wait, that's smoothed
Target (true one-hot): [0, 0, 0, 1.0, 0, 0, 0, 0] ← only index 3 matters

CE = -log(q[3])       ← one log, done
   = -log(0.85)
   = 0.163

All other entries are 0 × log(q[i]) = 0 → skip them entirely
```

## With Label Smoothing — Every Entry Matters

Once you smooth the target, **every** vocab entry has non-zero probability. You can't skip any:

```
Smoothed target: [0, 0.0167, 0.0167, 0.9, 0.0167, 0.0167, 0.0167, 0.0167]
                  ↑pad=0     ↑non-zero everywhere!  ↑correct=0.9
```

Now both KL and smoothed CE must process ALL entries.

## Concrete Example — vocab_size=8, position 0, correct = index 3

Model predictions after softmax:

```
q = [0.02, 0.05, 0.03, 0.70, 0.08, 0.04, 0.06, 0.02]
                        ↑ model is fairly confident about index 3
```

Smoothed target:

```
p = [0.00, 0.0167, 0.0167, 0.90, 0.0167, 0.0167, 0.0167, 0.0167]
     ↑pad    ↑ε/(V-2)        ↑1-ε
```

### Smoothed Cross-Entropy — What It Computes

```
Smoothed CE = -Σ p[i] × log(q[i])

i=0: -0.00   × log(0.02) = -0.00   × (-3.91) = 0.000     ← pad, skipped
i=1: -0.0167 × log(0.05) = -0.0167 × (-3.00) = 0.050
i=2: -0.0167 × log(0.03) = -0.0167 × (-3.51) = 0.059
i=3: -0.90   × log(0.70) = -0.90   × (-0.36) = 0.321     ← biggest term
i=4: -0.0167 × log(0.08) = -0.0167 × (-2.53) = 0.042
i=5: -0.0167 × log(0.04) = -0.0167 × (-3.22) = 0.054
i=6: -0.0167 × log(0.06) = -0.0167 × (-2.81) = 0.047
i=7: -0.0167 × log(0.02) = -0.0167 × (-3.91) = 0.065
─────────────────────────────────────────────────────
Total = 0.638

Operations per entry: 1 log + 1 multiply = 2 ops
Total: 8 entries × 2 ops = 16 ops
```

### KL Divergence — What It Computes

```
KL = Σ p[i] × (log(p[i]) - log(q[i]))

i=0: 0.00   × (log(0.00)   - log(0.02)) = 0.000           ← pad, skipped
i=1: 0.0167 × (log(0.0167) - log(0.05)) = 0.0167 × (-4.09 - (-3.00)) = 0.0167 × (-1.09) = -0.018
i=2: 0.0167 × (log(0.0167) - log(0.03)) = 0.0167 × (-4.09 - (-3.51)) = 0.0167 × (-0.58) = -0.010
i=3: 0.90   × (log(0.90)   - log(0.70)) = 0.90   × (-0.11 - (-0.36)) = 0.90   × (0.25)  =  0.225
i=4: 0.0167 × (log(0.0167) - log(0.08)) = 0.0167 × (-4.09 - (-2.53)) = 0.0167 × (-1.56) = -0.026
i=5: 0.0167 × (log(0.0167) - log(0.04)) = 0.0167 × (-4.09 - (-3.22)) = 0.0167 × (-0.87) = -0.015
i=6: 0.0167 × (log(0.0167) - log(0.06)) = 0.0167 × (-4.09 - (-2.81)) = 0.0167 × (-1.28) = -0.021
i=7: 0.0167 × (log(0.0167) - log(0.02)) = 0.0167 × (-4.09 - (-3.91)) = 0.0167 × (-0.18) = -0.003
─────────────────────────────────────────────────────────────────────────────────────────────
Total = 0.132

Operations per entry: 2 logs + 1 subtract + 1 multiply = 4 ops
Total: 8 entries × 4 ops = 32 ops
```

### The Extra Work — Side by Side

```
               Smoothed CE          KL Divergence         Extra in KL
per entry:     1 log + 1 multiply   2 logs + 1 sub + 1 mul   1 log + 1 subtract
total (V=8):   16 ops               32 ops                    16 extra ops
total (V=16K): 32K ops              64K ops                   32K extra ops
```

For a batch of 400 tokens × 16000 vocab:

```
Smoothed CE:  400 × 16000 × 2 = 12.8M ops
KL:           400 × 16000 × 4 = 25.6M ops
Extra:                           12.8M ops (log + subtract per entry)
```

Note: `log(p[i])` is a **constant** — the smoothed target `p` is always 0.9 for correct, 0.0167 for others. It doesn't change during training. So `log(0.9) = -0.11` and `log(0.0167) = -4.09` are the same every single batch. PyTorch recomputes them each time, but in theory they could be precomputed once.

## GPU Parallelism — Why It's Still Fast

Each step is **element-wise** — every element is independent. GPU processes all 16000 simultaneously:

```
Step 1: log_softmax(logits)        ← 16000 elements in parallel
Step 2: log(p[i])                  ← 16000 elements in parallel
Step 3: log(p[i]) - log(q[i])     ← 16000 elements in parallel
Step 4: p[i] × result             ← 16000 elements in parallel
Step 5: sum across 16000          ← reduction (partially sequential)
```

5 sequential steps, but **within** each step all 16000 run at once. The steps **must** wait for the previous one:

```
Can't do step 3 until step 2 (log(p)) AND step 1 (log(q)) are done
Can't do step 4 until step 3 (subtraction) is done
Can't do step 5 until step 4 (multiplication) is done
```

But that's only 5 steps of waiting. Not 16000 steps.

## Compare With Attention — The Real Bottleneck

Loss ops are **element-wise** — each of 16000 entries does `log`, `subtract`, `multiply` independently. GPU runs all 16000 in parallel, so 25.6M ops finish in microseconds.

Attention and FFN ops are **matrix multiplications** — each output element requires a **dot product** (sequential multiply-adds). These dominate training time.

### Step 1 — What One Attention Score Costs

In our model (`d_k = d_model / num_heads = 256 / 8 = 32`), each Q and K vector has 32 numbers. Computing **one** attention score = dot product of two 32-dimensional vectors:

```
score[i][j] = Σ Q[i][k] × K[j][k]     for k = 0 to 31

i = query position (which token is asking)
j = key position (which token is being looked at)
k = dimension index within d_k (0 to 31)

Example: "How much should token 2 attend to token 4?"

Q[2] = [0.3, -0.1, 0.7, ..., 0.2]     ← 32 numbers (d_k = 32)
K[4] = [0.5,  0.4, 0.1, ..., 0.8]     ← 32 numbers (d_k = 32)

score[2][4] = Q[2][0]×K[4][0] + Q[2][1]×K[4][1] + ... + Q[2][31]×K[4][31]
            = 0.3×0.5          + (-0.1)×0.4       + ... + 0.2×0.8
              ↑ step 1           ↑ step 2                  ↑ step 32

= 32 multiply-adds, computed sequentially (each needs the running sum)
```

Note: `32` here is `d_k` (dimension per head), **not** `num_heads`. `num_heads = 8` means we have 8 separate score matrices, each using `d_k = 32`.

The 32 multiply-adds **within one dot product** can't be parallelized — each step depends on the previous step's running sum:

```
score[2][4]:
  step 1:  sum = 0.3×0.5                    = 0.15
  step 2:  sum = 0.15 + (-0.1)×0.4          = 0.11      ← needs 0.15 from step 1
  step 3:  sum = 0.11 + 0.7×0.1             = 0.18      ← needs 0.11 from step 2
  ...
  step 32: sum = ... + 0.2×0.8              = final     ← needs all previous sums

Can't jump to step 32 without steps 1-31 — it's a chain.
```

**But** — all `seq_len × seq_len` dot products are independent and run **in parallel** across GPU cores.

Where does `seq_len = 22` come from? Our config has `max_tokens_per_batch = 8000`. If the batch has 363 sentences, each averages ~22 tokens:

```
363 sentences × 22 tokens ≈ 8000 tokens per batch
```

22 isn't fixed — some sentences are 10 tokens, some are 50 (up to `max_seq_len = 128`). It's the average for this example.

The attention score matrix for one sentence is `(seq_len × seq_len)` = `(22 × 22)` — every token computes a score against every other token in the **same sentence**:

```
Example sentence: "I love AI" → tokens [I, love, AI, ...] (22 tokens after BPE)

                  key positions (which token is being looked at)
                  k=0    k=1     k=2    ...  k=21
query      q=0  [sc00,  sc01,  sc02,  ..., sc0_21]   ← "I" attends to all 22
positions  q=1  [sc10,  sc11,  sc12,  ..., sc1_21]   ← "love" attends to all 22
(which     q=2  [sc20,  sc21,  sc22,  ..., sc2_21]   ← "AI" attends to all 22
token is   ...
asking)    q=21 [sc210, sc211, sc212, ..., sc21_21]  ← token 21 attends to all 22

= 22 × 22 = 484 scores total
```

Each of these 484 scores is an independent dot product — GPU runs all 484 simultaneously, each doing its own 32 sequential multiply-adds:

```
GPU core 1:   score[0][0] = Q[0]·K[0] → 32 steps
GPU core 2:   score[0][1] = Q[0]·K[1] → 32 steps      ← all at the
GPU core 3:   score[0][2] = Q[0]·K[2] → 32 steps         same time
...
GPU core 484: score[21][21] = Q[21]·K[21] → 32 steps

→ 484 cores fire at once, each waits 32 steps → done
```

### Step 2 — One Encoder Layer's Attention Cost

Our config: `max_tokens_per_batch = 8000`, `max_seq_len = 128`.

363 sentences averaging ~22 tokens each (363 × 22 ≈ 8000 tokens). Each sentence has 8 attention heads.

For **one head** on **one sentence** (seq_len ≈ 22):

```
Q:   (22, 32)    ← 22 tokens, each 32 dims (d_k = 256/8 = 32)
K^T: (32, 22)    ← transposed K

Q @ K^T = (22, 32) @ (32, 22) = (22, 22)
                ↑↑ 32 dims collapse into one score per pair

score[2][4] = Q[2]·K[4] = 32 multiply-adds → ONE number
              ↑ 32 dimensions gone, collapsed into a single attention score
```

The output is a `(22, 22)` matrix = 484 scores. Each score required 32 multiply-adds to compute (the dot product). So total ops:

```
484 scores      ×    32 multiply-adds each = 15,488 ops per head
↑scores(22 x 22)↑    ↑d_k (dot product depth)
```

For **all heads and all sentences** in one layer:

```
363 sentences × 8 heads × 22 × 22 × 32 = 363 × 8 × 15,488
                                        ≈ 45 MILLION multiply-adds
```

### Step 3 — Scores @ V (Same Cost Again)

After computing attention scores and applying softmax, we multiply the attention weights by V to get the final output — "blend the value vectors according to how much each token should attend to each other token":

```
attention_output = softmax(scores) @ V

scores after softmax: (22, 22)     ← attention weights (how much each token attends to others)
V:                    (22, 32)     ← value vectors (what information each token carries)

output:               (22, 32)    ← 22 positions × 32 dims
```

**Scores after softmax: `(22, 22)`** — 22 query tokens × 22 key tokens. Each row is a probability distribution (sums to 1.0) — "how much does this token attend to each other token?"

```
              k=0   k=1   k=2   ...  k=21
token 0:   [0.05, 0.10, 0.60, ..., 0.01]  ← token 0 attends mostly to token 2 (0.60)
token 1:   [0.02, 0.70, 0.08, ..., 0.03]  ← token 1 attends mostly to itself (0.70)
...
token 21:  [0.01, 0.04, 0.03, ..., 0.50]  ← token 21 attends mostly to itself (0.50)

Each row sums to 1.0 after softmax
```

**V: `(22, 32)`** — 22 tokens, each carrying a 32-dimensional value vector (`d_k = d_model / num_heads = 256 / 8 = 32`). The value vector is "what information this token carries" for other tokens to blend:

```
              dim0  dim1  dim2  ...  dim31
token 0:   [0.3,  0.7,  0.1,  ..., 0.5]   ← token 0's information
token 1:   [0.9,  0.2,  0.4,  ..., 0.8]   ← token 1's information
...
token 21:  [0.1,  0.6,  0.3,  ..., 0.2]   ← token 21's information
```

**Output: `(22, 32)`** — matrix multiply, inner dimensions (22) match and collapse:

```
(22, 22) @ (22, 32) = (22, 32)
      ↑↑ match & collapse

Each of 22 tokens gets a new 32-dim representation
= weighted blend of ALL tokens' value vectors
```

`output[i][j]` means: "for token `i`, what is dimension `j` of its new representation?"

Each output element blends ALL tokens' values, weighted by attention scores:

```
output[i][j] = Σ weights[i][k] × V[k][j]     for k = 0 to 21

i = which token's output (0 to 21)
j = which dimension of that token's new vector (0 to 31, d_k = 32)
k = loop over ALL 22 tokens, blending their values
```

**Example: `output[2][5]`** — "token 2's new vector, dimension 5" (just one of 32 dims, picked as example)

Token 2's attention weights (from softmax of scores, row 2 of the `(22, 22)` matrix):

```
weights[2] = [0.05, 0.10, 0.60, 0.02, ..., 0.01]    ← 22 weights (sum to 1.0)
              ↑     ↑     ↑
              token0 token1 token2(self)

Token 2 attends mostly to itself (0.60)
```

Each token carries a value vector (32 dims). We only need dimension 5 from each:

```
V[0][5]  = 0.3    ← token 0's value at dim 5
V[1][5]  = 0.7    ← token 1's value at dim 5
V[2][5]  = 0.9    ← token 2's value at dim 5 (self)
...
V[21][5] = 0.1    ← token 21's value at dim 5
```

Multiply each weight × each value and sum:

```
output[2][5] = 0.05 × 0.3     ← token 0 contributes little (low weight 0.05)
             + 0.10 × 0.7     ← token 1 contributes a bit
             + 0.60 × 0.9     ← token 2 (self) dominates (highest weight × its value)
             + ...
             + 0.01 × 0.1     ← token 21 contributes almost nothing
             = 0.015 + 0.07 + 0.54 + ... + 0.001
                               ↑ 0.54 alone is most of the sum
                                 because weight 0.60 × value 0.9

Token 2's dim 5 is mostly its OWN value (0.9 × 0.60 = 0.54)
because it attends mostly to itself.

= 22 multiply-adds, sequential (same chain dependency as Q @ K^T)
```

Repeat the same calculation for all 32 dimensions → token 2 gets a full new 32-dim vector:

```
output[2][0]  = Σ weights[2][k] × V[k][0]   ← dim 0
output[2][1]  = Σ weights[2][k] × V[k][1]   ← dim 1
...
output[2][5]  = Σ weights[2][k] × V[k][5]   ← dim 5 (our example above)
...
output[2][31] = Σ weights[2][k] × V[k][31]  ← dim 31

= 32 dot products, one per dimension → token 2's full new vector
```

Repeat for all 22 tokens → `(22, 32)` output matrix.

Total:

```
22 × 32 = 704 output elements
 ↑    ↑
 tokens dims    (output[0][0] to output[21][31])

Each element = 22 multiply-adds (dot product over all tokens in the sentence)

22  ×  32  ×  22  = 15,488 ops per head  (same as Q @ K^T)
 ↑      ↑      ↑
tokens  dims   dot product depth (k = 0 to 21)
 ↑output↑ ↑dot product depth

→ 363 × 8 × 15,488 ≈ 45M ops per layer
```

### Step 4 — FFN Cost Per Layer

Each layer has a position-wise FFN: `Linear(256→1024)` then `Linear(1024→256)`:

```
FFN layer 1: input (8000, 256) @ W₁ (256, 1024) → (8000, 1024)
  Each of 8000 × 1024 = 8.2M output elements needs a 256-dim dot product
  → 8.2M × 256 ≈ 2,097M ops

FFN layer 2: input (8000, 1024) @ W₂ (1024, 256) → (8000, 256)
  Each of 8000 × 256 = 2.05M output elements needs a 1024-dim dot product
  → 2.05M × 1024 ≈ 2,097M ops

Total FFN per layer ≈ 4,194M ops
```

Wait — that's much larger than attention! This is correct: **FFN is the biggest cost** because it applies the same linear layer to every token independently — all 8000 tokens in one big matrix multiply `(8000, 256) @ (256, 1024)`. Attention, on the other hand, operates per-sentence — each sentence has its own small `(seq_len × seq_len)` score matrix (tokens from different sentences don't attend to each other).

### Step 5 — Total Model Cost Per Batch (All Layers)

All numbers below are for **one batch** (~8000 tokens, ~363 sentences) — one forward pass through the model. Training repeats this for every batch in every epoch.

```
Per encoder layer:
  Q @ K^T:        ~45M ops
  scores @ V:     ~45M ops
  FFN:            ~4,194M ops
  ─────────────────────────
  Subtotal:       ~4,284M ops

Per decoder layer (3 attention sub-layers + FFN):
  Masked self-attention Q@K^T + scores@V:  ~90M ops
  Cross-attention Q@K^T + scores@V:        ~90M ops
  FFN:                                     ~4,194M ops
  ─────────────────────────────────────────
  Subtotal:                                ~4,374M ops

Total model (4 encoder + 4 decoder layers):
  4 × 4,284M + 4 × 4,374M
  = 17,136M + 17,496M
  ≈ 34,632M
  ≈ 34.6 BILLION ops per batch
```

### The Comparison

```
Loss (KL):    25.6M ops     ← element-wise, fully parallel, ~microseconds
Model:        34,600M ops   ← matrix multiplications, ~milliseconds

Loss is 0.07% of total computation
```

**Result:** loss computation is a rounding error compared to the model's forward pass. KL's extra `log + subtract` per entry adds ~12.8M ops — invisible next to 34.6 billion matrix multiply-adds in attention and FFN.

---

# The Full Picture

```
"আমি AI ভালোবাসি <eos>"  →  [3, 4, 5, 2]  →  pad  →  flatten
                                                           ↓
                              model predicts logits    (8, 8)
                                                           ↓
                              log_softmax              (8, 8)
                                                           ↓
                              build smoothed target:
                              - fill ε/(V-2) everywhere
                              - put (1-ε) at correct token
                              - zero pad column
                              - zero pad rows
                                                           ↓
                              KL(smoothed || log_probs)
                                                           ↓
                              divide by non-pad token count
                                                           ↓
                              scalar loss → backward → update weights
```

---

# References

### Paper

1. [Rethinking the Inception Architecture](https://arxiv.org/abs/1512.00567) — Szegedy et al., 2016 (introduced label smoothing)

### Label Smoothing

2. [CS 152 NN—8: Regularization—Label Smoothing](https://www.youtube.com/watch?v=wmUiOAra_-M) — Harvey Mudd College Neural Networks class (Day 8)
3. [What is Label Smoothing?](https://towardsdatascience.com/what-is-label-smoothing-108debd7ef06/) — Towards Data Science

### Entropy, Cross-Entropy & KL Divergence

4. [The Key Equation Behind Probability](https://www.youtube.com/watch?v=KHVR587oW8I) — Artem Kirsanov
5. [A Short Introduction to Entropy, Cross-Entropy and KL-Divergence](https://www.youtube.com/watch?v=ErfnhcEV1O8) — Aurélien Géron
6. [The KL Divergence: Data Science Basics](https://www.youtube.com/watch?v=q0AkK8aYbLY) — ritvikmath
