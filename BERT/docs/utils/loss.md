# BERT Pre-training Loss (`loss.py`)

> Module: [`BERT/utils/loss.py`](../../utils/loss.py) — `BERTPreTrainingLoss`
> Paper: Devlin et al. 2019, [*BERT*](https://arxiv.org/abs/1810.04805), §3.1 (Pre-training BERT)

The model emits **logits**; this module turns `(logits, labels)` into the single scalar
that gets backprop'd. It's the **training-only** counterpart to the architecture — the
wrapper ([`bert_for_pretraining.py`](../architecture/bert_for_pretraining.md)) deliberately
returns logits and nothing else, so loss lives here, separate.

BERT trains on **two objectives at once**, and the paper simply **sums** them:

```
L = L_MLM + L_NSP
```

Both halves are **plain cross-entropy** — no label smoothing (unlike the
[transformer's translation loss](../../../transformer/utils/loss.py), which used ε=0.1).

Throughout, **B** = batch size, **S** = sequence length, **V** = vocab size
(30522 for BERT-base).

## Contents

- [The two losses at a glance](#the-two-losses-at-a-glance)
- [A worked example (B=2, S=4, V=6)](#a-worked-example-b2-s4-v6)
  - [MLM: logits, labels, and the -100 convention](#mlm-logits-labels-and-the--100-convention)
  - [NSP: binary classification from `[CLS]`](#nsp-binary-classification-from-cls)
- [Why `cross_entropy` takes raw logits](#why-cross_entropy-takes-raw-logits)
- [The flatten: `view(-1)` — and `contiguous` / `view` / `reshape`](#the-flatten-view-1--and-contiguous--view--reshape)
- [What `forward` returns](#what-forward-returns)
- [References](#references)

---

## The two losses at a glance

The two objectives differ in **what they predict** — and that drives every shape in the file:

| | MLM | NSP |
|---|---|---|
| predicts per... | **token** (`B*S` of them) | **sequence** (`B` of them) |
| logits shape | `(B, S, V)` → flatten → `(B*S, V)` | `(B, 2)` already |
| classes `C` | `V` = whole vocab (e.g. 30522) | `2` (IsNext / NotNext) |
| labels shape | `(B, S)`, mostly `-100` | `(B,)`, all real (0/1) |
| skip mechanism | `ignore_index=-100` (only ~15% scored) | none — every sequence scored |
| reshape needed? | **yes**, `view(-1, V)` | no |

MLM scores **many positions inside each sequence**; NSP scores **each sequence once**. That
single difference is why MLM gets flattened and NSP doesn't.

## A worked example (B=2, S=4, V=6)

A toy batch of **2 sequences**, each **4 tokens**, over a **6-word vocab**:

```
id:    0      1      2      3      4      5
word:  the    cat    dog    sat    ran    big
```

```
Sequence 0:  "the   [MASK]   ran   big"     ← we hid the word "cat"
Sequence 1:  "[MASK]   dog   sat   [MASK]"  ← we hid "big" and "cat"
```

### MLM: logits, labels, and the -100 convention

**`mlm_logits` `(B, S, V) = (2, 4, 6)`** — at **every** position the model outputs `V=6`
scores ("how much do I believe this slot is each vocab word"):

```
Sentence 0:                the   cat   dog   sat   ran   big
  pos0 "the"        →   [  4.0,  0.1,  0.2,  0.0,  0.1,  0.3 ]   leans "the"
  pos1 [MASK]       →   [  0.2,  3.5,  0.4,  0.1,  0.9,  0.2 ]   leans "cat"
  pos2 "ran"        →   [  0.1,  0.3,  0.2,  0.4,  3.8,  0.1 ]
  pos3 "big"        →   [  0.2,  0.1,  0.0,  0.3,  0.2,  3.9 ]

Sentence 1:                the   cat   dog   sat   ran   big
  pos0 [MASK]       →   [  0.3,  0.4,  0.2,  0.1,  0.5,  3.1 ]   leans "big"
  pos1 "dog"        →   [  0.1,  0.5,  3.6,  0.0,  0.2,  0.1 ]
  pos2 "sat"        →   [  0.2,  0.1,  0.3,  3.7,  0.4,  0.0 ]
  pos3 [MASK]       →   [  0.1,  3.2,  0.6,  0.2,  0.3,  0.4 ]   leans "cat"
```

**`mlm_labels` `(B, S) = (2, 4)`** — the **answer key**. We only know the truth where we
made a blank; every other slot is `-100` ("don't grade this"):

```
Sentence 0:  [ -100,   1,  -100,  -100 ]    ← only pos1, true = "cat" (id 1)
Sentence 1:  [   5,  -100,  -100,   1  ]    ← pos0 true = "big"(5), pos3 true = "cat"(1)
```

> **The `-100` is the whole masking mechanism on the loss side.** The MLM head runs on the
> *full* sequence (all `S` positions), but the data pipeline writes `-100` at every
> position we did **not** select for prediction. `cross_entropy(ignore_index=-100)` then
> skips those rows entirely — they add nothing to the loss **and** nothing to the mean's
> denominator. So ~85% of positions are free supervision-wise; only the ~15% selected
> tokens are scored. (See [`mlm.md`](../objectives/mlm.md) for the 80/10/10 split — note
> the scored 15% includes the *kept* and *random* tokens, not just the `[MASK]` ones.)

**Flatten + grade.** `cross_entropy` can't read a 3-D tensor with sentences nested inside;
it wants a flat exam — *one row of `V` scores, one answer*. So we stack both sentences'
positions into `(B*S, V) = (8, 6)` and the labels into `(8,)`, **in the same order**:

```
row   what it is      answer    graded?
 0    seq0 "the"      -100      skip
 1    seq0 [MASK]       1  cat  ✅ GRADE  (did "cat" get the top score? 3.5 → yes)
 2    seq0 "ran"      -100      skip
 3    seq0 "big"      -100      skip
 4    seq1 [MASK]       5  big  ✅ GRADE  (3.1 on "big" → yes)
 5    seq1 "dog"      -100      skip
 6    seq1 "sat"      -100      skip
 7    seq1 [MASK]       1  cat  ✅ GRADE  (3.2 on "cat" → yes)
```

Only rows **1, 4, 7** survive `-100`. `cross_entropy` grades those **3** and **averages
them into one number** = `mlm_loss`. Note they came from **both** sequences (1 from seq0,
2 from seq1) — flattening is exactly what lets them be pooled and averaged in one call,
regardless of which sequence each token belonged to.

```python
mlm_loss = F.cross_entropy(
    mlm_logits.view(-1, vocab_size),   # (B, S, V) -> (B*S, V)
    mlm_labels.view(-1),               # (B, S)    -> (B*S,)
    ignore_index=-100,
)
```

### NSP: binary classification from `[CLS]`

NSP is a **binary** task: is sentence B the real next sentence (`0 = IsNext`) or a random
one (`1 = NotNext`)? One prediction per sequence, read from the `[CLS]` vector.

**`nsp_logits` `(B, 2) = (2, 2)`** — 2 scores per sequence `[IsNext, NotNext]`:

```
                  IsNext   NotNext
  seq0   →     [   3.1  ,   -0.4  ]   leans IsNext
  seq1   →     [  -1.2  ,    2.8  ]   leans NotNext
```

**`nsp_labels` `(B,) = (2,)`** — one id per sequence:

```
  nsp_labels = [ 0,  1 ]   ← seq0 IsNext, seq1 NotNext
```

Already the `(N, C)` vs `(N,)` shape `cross_entropy` wants, so **no reshape**:

```python
nsp_loss = F.cross_entropy(nsp_logits, nsp_labels)   # grades both rows, averages → scalar
```

No `ignore_index` — every sequence carries a real 0/1 label (the coin-flip made when the
pair was built; see [`nsp.md`](../objectives/nsp.md)).

## Why `cross_entropy` takes raw logits

`F.cross_entropy` does the **softmax internally** — it is `log_softmax` + `nll_loss` fused:

```
F.cross_entropy(logits, labels)  ==  log_softmax(logits)  then  nll_loss(…, labels)
```

So for `seq0 = [3.1, -0.4]`, label `0`:

```
1. softmax([3.1, -0.4])    → [0.97, 0.03]     ← cross_entropy does this
2. take true class (0)     → 0.97
3. loss = -log(0.97)       → 0.03             ← small: confident AND right
```

Confident but **wrong** (label `1`) would be `-log(0.03) = 3.5` → large loss.

> **Practical rule:** because softmax is built in, the heads must output **raw logits** —
> never apply softmax yourself first, or you'd softmax twice and the loss would be wrong.
> That's why [`heads.py`](../architecture/heads.md) ends each head at a bare `nn.Linear`
> with no final activation. MLM is the same mechanism over `V` classes instead of 2.

## The flatten: `view(-1)` — and `contiguous` / `view` / `reshape`

The MLM flatten uses `view(-1, vocab_size)`. Worth knowing **why `view` and not `reshape`**,
because it comes down to how a tensor sits in memory.

A tensor is two things: a **flat 1-D block of memory** + **metadata** (shape + strides)
saying how to read it. **Contiguous** means the elements sit in memory **one after another,
in the same order you'd read them** logically (row by row).

```
contiguous (2,3):        logical             memory (back-to-back, in order)
                         [ 1  2  3 ]         1  2  3  4  5  6
                         [ 4  5  6 ]         └─row0─┘└─row1─┘
```

Reading left-to-right, top-to-bottom = walking straight through memory. So `view(-1)` →
`[1,2,3,4,5,6]` is **free**: same memory, just "forget the old shape." Nothing moves.

Now `.transpose(0,1)` → `(3,2)` moves **no data** — it only swaps the strides, so the *same*
memory is now read in a jumping order:

```
transposed (3,2):        logical             memory (UNCHANGED)
                         [ 1  4 ]            1  2  3  4  5  6
                         [ 2  5 ]            reading the transpose walks it as
                         [ 3  6 ]            1, 4, 2, 5, 3, 6 — NOT sequential
```

Logical order (`1,4,2,5,3,6`) ≠ memory order (`1,2,3,4,5,6`) → the tensor is
**non-contiguous**; the elements no longer "sit after each other" in reading order.

The operations involved:

| op | moves data? | behaviour |
|---|---|---|
| `transpose` / `permute` | no — only strides change | *makes* a tensor non-contiguous |
| `view` | **no** — pure reinterpret | needs contiguous; else **errors loudly** |
| `.contiguous()` | **yes — copies** | physically reorders memory to match logical order |
| `reshape` | maybe — copies *only if needed* | tries `view`; falls back to a silent copy |

So:

```
view:     free reinterpret  OR  loud error
reshape:  free reinterpret  OR  silent copy (extra memory + time)
```

**Why prefer `view` when we know it's safe:**

1. **Guaranteed zero-cost.** `view` *promises* no copy. With `reshape` you can't tell from
   the code whether you just silently duplicated a `(32, 128, 30522)` logits tensor
   (~500MB) on a per-step hot path.
2. **The error is information.** If `view` throws "use `.reshape()` or `.contiguous()`", an
   upstream `transpose`/`permute`/slice left the tensor non-contiguous — usually worth
   investigating, not papering over.
3. **Intent.** `view` documents "I know this is contiguous, this is free."

Here `mlm_logits` comes straight out of a `Linear` (the MLM decoder), whose output is
**always contiguous** — so `view` is correct, free, and the honest choice. Reach for
`reshape` only when you genuinely have a non-contiguous tensor and accept the copy.

| situation | use |
|---|---|
| right after `Linear`, `+`, elementwise ops (contiguous) | `view` |
| right after `transpose` / `permute` / some slices | `reshape` (or `.contiguous().view()`) |
| unsure / library boundary | `reshape` (safe default) |

## What `forward` returns

```python
total_loss = mlm_loss + nsp_loss
return total_loss, mlm_loss, nsp_loss
```

`total_loss` is what `.backward()` is called on (the paper's `L = L_MLM + L_NSP`). The two
parts are returned **alongside** the sum so the training loop can log MLM and NSP
separately — they typically converge at different rates (MLM is the harder, slower signal;
NSP saturates early), so watching them apart is how you tell pre-training is healthy.

```python
loss, mlm_loss, nsp_loss = criterion(mlm_logits, nsp_logits, mlm_labels, nsp_labels)
loss.backward()
```

## References

- **Paper:** Devlin et al. 2019, [*BERT*](https://arxiv.org/abs/1810.04805) — §3.1 (the two pre-training objectives)
- **The heads that emit these logits:** [`heads.md`](../architecture/heads.md)
- **The wrapper (why loss stays out of the model):** [`bert_for_pretraining.md`](../architecture/bert_for_pretraining.md)
- **MLM masking (the 80/10/10 and the `-100` labels):** [`mlm.md`](../objectives/mlm.md)
- **NSP pair sampling (the 50/50 IsNext):** [`nsp.md`](../objectives/nsp.md)
- **The label-smoothed translation loss we adapted from:** [`transformer/utils/loss.py`](../../../transformer/utils/loss.py)
- **PyTorch:** [`F.cross_entropy`](https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html) (softmax built in, `ignore_index` skips targets), [`Tensor.view`](https://pytorch.org/docs/stable/generated/torch.Tensor.view.html) vs [`Tensor.reshape`](https://pytorch.org/docs/stable/generated/torch.Tensor.reshape.html)
