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
7. [PyTorch Alternative — `CrossEntropyLoss(label_smoothing=)`](#pytorch-alternative--crossentropyloss-label_smoothing)
8. [KL Divergence — Extra Computation Cost](#kl-divergence--extra-computation-cost)
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

But `nn.CrossEntropyLoss` before PyTorch 1.10 (2021) only accepted **integer targets** (token IDs like `[3, 4, 5]`), not soft distributions like `[0.0167, 0.0167, 0.9, ...]`. (PyTorch 1.10+ and 2.x now support `label_smoothing=` natively — see [Section 7](#pytorch-alternative--crossentropyloss-label_smoothing).)

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

Flattening happens in the **training loop** (not inside the loss function):

```python
logits = logits.view(-1, vocab_size)   # (2*4, 8) = (8, 8)
target = target.view(-1)               # (8,)

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

# PyTorch Alternative — `CrossEntropyLoss(label_smoothing=)`

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

KL computes `p × (log p - log q)` while CE computes just `-p × log q`. The extra `log p` means:

```
KL:  multiply + log + subtract + multiply    per element
CE:  multiply + negate                        per element
```

KL does **one extra log and one extra subtract** per element. For `vocab_size=30000` and `batch × seq_len = 4096`:

```
Extra operations = 30000 × 4096 = ~120M extra log + subtract
```

Sounds big, but on a GPU these are **element-wise ops** — massively parallelized, takes microseconds. The real bottleneck is matrix multiplications in attention (`Q×K`, `scores×V`), which are orders of magnitude heavier.

**Practically:** unnoticeable. The `log p` constant adds near-zero cost.

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
