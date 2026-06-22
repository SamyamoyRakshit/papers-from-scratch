# Masked Language Modeling — the masking step (`masking.py`)

> Module: [`BERT/utils/masking.py`](../../utils/masking.py) — `mask_tokens`
> Paper: Devlin et al. 2019, [*BERT*](https://arxiv.org/abs/1810.04805), §3.1 (Task #1: Masked LM)

This is **Stage 1** of the data pipeline — the step that turns one clean tokenized
sentence-pair into the `(masked input, answer key)` pair the MLM head and
[`loss.py`](loss.md) consume. It implements the famous **15% / 80-10-10** rule.

It does **not** tokenize text and it does **not** build batches. It receives one already-
tokenized example (from [`nsp.py`](../objectives/nsp.md), which assembles `[CLS] A [SEP] B [SEP]`),
masks it, and hands back two tensors. Tokenizing and batching happen around it in
[`data_utils.py`](data_utils.md).

Throughout, **S** = sequence length, **V** = vocab size (30522 for BERT-base).

## Contents

- [Where this sits in the pipeline](#where-this-sits-in-the-pipeline)
- [What goes in, what comes out](#what-goes-in-what-comes-out)
- [The 15% / 80-10-10 rule](#the-15--80-10-10-rule)
- [Walking the code, line by line](#walking-the-code-line-by-line)
  - [1. The blank answer key](#1-the-blank-answer-key)
  - [2. Finding the maskable positions](#2-finding-the-maskable-positions)
  - [3. Picking 15% — the permutation trick](#3-picking-15--the-permutation-trick)
  - [4. Writing the answers](#4-writing-the-answers)
  - [5. Corrupting the input — 80/10/10](#5-corrupting-the-input--801010)
- [Why `.clone()` — don't mutate the caller's tensor](#why-clone--dont-mutate-the-callers-tensor)
- [How a random number becomes a probability](#how-a-random-number-becomes-a-probability)
- [Why return `labels` if it's mostly -100](#why-return-labels-if-its-mostly--100)
- [A full worked example](#a-full-worked-example)
- [References](#references)

---

## Where this sits in the pipeline

```
raw text
  │  WordPiece tokenizer (data_utils.py) — owns the vocab: string ↔ id
  ▼
nsp.py  — glue two sentences:  [CLS] A [SEP] B [SEP]  → token_ids
  │
  ▼
masking.py  ──►  masked_ids  (the question — words hidden)
                 labels      (the answer key — original ids at hidden slots)
  │
  ▼
data_utils.py — pad + batch many examples → (B, S) tensors
  │
  ▼
model → loss.py
```

The key seam: `masking.py` produces `mlm_labels` with `-100` at every position the loss
should **ignore**, and the original token id at every position it should **score**. That
`-100` convention is read on the other side by `F.cross_entropy(..., ignore_index=-100)`
in [`loss.py`](loss.md).

## What goes in, what comes out

`token_ids` is **one sentence-pair already converted from words → integer ids** — not
words, not vectors, just row-indices into the vocab. It is a function **argument**;
`masking.py` neither creates nor owns it.

```python
masked_ids, labels = mask_tokens(
    token_ids,            # 1-D LongTensor (S,) — e.g. [2, 11, 12, 13, 3]
    vocab_size,           # e.g. 30522 — the size of the vocab, NOT the vocab itself
    mask_token_id,        # id of [MASK], e.g. 4
    special_token_ids,    # ids never to mask, e.g. {0, 2, 3} = [PAD]/[CLS]/[SEP]
    mlm_probability=0.15,
    ignore_index=-100,
)
```

Note `vocab_size` is just a number. Masking never sees the whole vocab table — it only
needs its **size** so it can draw a random replacement id in `[0, vocab_size)`.

| name | what it is |
|---|---|
| vocab (lives in the tokenizer) | the lookup table: `[PAD]→0  [CLS]→2  [SEP]→3  [MASK]→4 … the→11 cat→12` |
| `token_ids` | the **output** of applying that table to one example — a handful of ids |
| `vocab_size` | the table's **row count** — the range to draw random tokens from |

## The 15% / 80-10-10 rule

Of the **non-special** tokens, **15%** are picked to be **predicted**. Of those picked
positions:

| share of the picked 15% | what happens to the INPUT | still scored? |
|---|---|---|
| 80% | replaced with `[MASK]` | ✅ yes |
| 10% | replaced with a **random** token | ✅ yes |
| 10% | left **unchanged** (original word) | ✅ yes |

Two points that trip people up:

- **All 15% are scored**, not just the 80% shown as `[MASK]`. "Selected for prediction"
  and "turned into `[MASK]`" are *different* things — the label is recorded for every
  selected token regardless of what the input ends up showing.
- **The 10%-kept case is the whole trick.** At fine-tune time there is no `[MASK]` token.
  If the model only ever had to predict slots that literally show `[MASK]`, it would learn
  "only the `[MASK]` slot can be wrong." By sometimes showing the real (or a random) word
  and grading it anyway, BERT forces a real representation for *every* token.

These percentages are **per-token probabilities, not quotas** — they only emerge as
80/10/10 when averaged over the whole corpus (see
[below](#how-a-random-number-becomes-a-probability)).

## Walking the code, line by line

We'll use a running example. Vocab ids: `[PAD]=0, [CLS]=2, [SEP]=3, [MASK]=4`, words
`the=11, cat=12, sat=13`:

```python
token_ids        = [2, 11, 12, 13, 3]      # [CLS] the cat sat [SEP]
special_token_ids = {0, 2, 3}              # [PAD], [CLS], [SEP]
```

### 1. The blank answer key

```python
labels = torch.full_like(token_ids, ignore_index)   # all -100
```

`torch.full_like(token_ids, -100)` makes a **brand-new tensor the same shape as
`token_ids`, filled with -100**. It copies the *shape*, not the values — `token_ids`
is untouched. Starting all-`-100` means "score nothing yet"; we punch the real answers in
later, at only the picked positions.

```
token_ids = [ 2, 11, 12, 13,  3]    ← untouched
labels    = [-100,-100,-100,-100,-100]    ← fresh, all ignore
```

### 2. Finding the maskable positions

`special_token_ids` are the structural tokens — `[CLS]`, `[SEP]`, `[PAD]` — that aren't
real words and must **never** be masked. We build a boolean flag tensor marking where they
sit:

```python
special_mask = torch.zeros_like(token_ids, dtype=torch.bool)   # all False
for sid in special_token_ids:
    special_mask |= token_ids == sid
```

- `torch.zeros(..., dtype=torch.bool)` = **all `False`** — "nothing special yet."
- `token_ids == sid` is **element-wise** on a tensor (gives a True/False per position) —
  this is *why* the inputs must be tensors, not Python lists. On a plain list,
  `[2,11,12,3,0] == 2` is a single `False` and `list |= bool` throws `TypeError`.
- `|=` is **element-wise OR**, accumulating: it only ever turns `False`→`True`, never back.
  Each pass marks one special id's positions without erasing earlier passes' marks.

OR behaves like **add-with-a-cap** (`F`=0, `T`=1): `0|0=0, 0|1=1, 1|1=1`. (For contrast,
`&` / AND behaves like **multiply**: `1&1=1`, everything else 0 — used elsewhere in the
file for the `(decision >= 0.8) & (decision < 0.9)` band.)

Tracing the loop on our example:

```
start:            [F, F, F, F, F]
sid=0: ids==0  →  [F,F,F,F,F]  (no [PAD] here)
sid=2: ids==2  →  [T,F,F,F,F]
sid=3: ids==3  →  [T,F,F,T,F]
                   ↑[CLS]   ↑[SEP]
```

Then invert and read off the indices we're *allowed* to mask:

```python
candidates = (~special_mask).nonzero(as_tuple=True)[0]
```

- `~special_mask` flips every bool → `True` now marks the **real words**.
- `.nonzero()` returns the **indices** where it's `True`.
- `as_tuple=True ... [0]` returns those indices as a flat 1-D tensor (instead of the
  default 2-D column), because `token_ids` is 1-D.

```
~special_mask = [F, T, T, F, F]   →   candidates = [1, 2]    (positions of "the", "cat")
```

`candidates` holds **positions**, not token ids.

### 3. Picking 15% — the permutation trick

```python
num_to_predict = max(1, int(round(len(candidates) * mlm_probability)))
perm = torch.randperm(len(candidates))
selected = candidates[perm[:num_to_predict]]
```

**The `max(1, ...)` guard matters for short sequences.** With 2 candidate words,
`round(2 * 0.15) = round(0.3) = 0` — without the guard, *nothing* would be masked and the
example would give the loss zero signal (all labels -100 = wasted example). `max(1, 0)`
forces **at least one** prediction. 15% only naturally clears 1 once a sequence has ≳4 real
words; short toy sequences therefore get over-masked, which is fine and matches Google's
original BERT (`max(1, int(round(...)))`). In real pre-training, sequences are 128–512
tokens, so the guard almost never fires.

**Why a permutation?** `torch.randperm(n)` is a **random shuffle of `0..n-1`** — like
shuffling a deck and dealing the top `k` cards:

```
candidates = [2, 3, 5, 6, 7, 8, 9, 10, 12, 13]   # 10 real-word positions
num_to_predict = round(10 * 0.15) = 2

perm        = [7, 2, 9, 0, 4, 1, 8, 3, 6, 5]      # random ordering
perm[:2]    = [7, 2]
selected    = candidates[[7, 2]] = [10, 5]         # mask positions 10 and 5
```

Shuffle-then-take-front-`k` gives **sampling without replacement**: the picked positions
are guaranteed distinct (no duplicates) and every candidate has an equal shot. Using
`torch.randint` instead could draw the same position twice and mask fewer than intended.

### 4. Writing the answers

```python
labels[selected] = token_ids[selected]
```

This punches the **original ids** into the answer key at the selected positions — the only
positions the loss will score. Everything else stays `-100`. After this, on our example
(say `selected = [2]`, the "cat" position):

```
labels = [-100, -100, 12, -100, -100]
                       ↑ "the hidden word at slot 2 was 12 (cat)"
```

### 5. Corrupting the input — 80/10/10

```python
masked_ids = token_ids.clone()
decision = torch.rand(num_to_predict)      # one uniform draw per selected position

mask_bucket   = selected[decision < 0.8]                        # 80% → [MASK]
random_bucket = selected[(decision >= 0.8) & (decision < 0.9)]  # 10% → random
# decision >= 0.9 → remaining 10%, left unchanged

masked_ids[mask_bucket] = mask_token_id
masked_ids[random_bucket] = torch.randint(0, vocab_size, (len(random_bucket),))
```

`decision` has **exactly one random number per selected token** (`num_to_predict ==
len(selected)`). `selected[decision < 0.8]` is **boolean-mask indexing** — keep only the
selected positions whose draw landed in the 80% band:

```
selected = [2,    3,    4,    5,    6]
decision = [0.12, 0.95, 0.40, 0.83, 0.07]

decision < 0.8         → [T, F, T, F, T] → mask_bucket   = [2, 4, 6]
(>=0.8) & (<0.9)       → [F, F, F, T, F] → random_bucket = [5]
decision >= 0.9        → [F, T, F, F, F] → kept (untouched) = [3]
```

The three bands `[0, 0.8) / [0.8, 0.9) / [0.9, 1)` **don't overlap**, so each selected
token lands in exactly one bucket — that's how a single uniform draw cleanly splits the
selected tokens 80/10/10.

> Note: a random replacement can, rarely, draw a special-token id (e.g. land on `[CLS]`'s
> id). Google's original BERT allows this too — it's rare and harmless, so the code stays
> simple rather than filtering.

## Why `.clone()` — don't mutate the caller's tensor

`token_ids` is **borrowed**. It's a tensor that lives in the calling file
([`nsp.py`](../objectives/nsp.md) / [`data_utils.py`](data_utils.md)) and is *passed in*.
Python passes tensors **by reference** — the parameter is the *same object*, not a copy.
So writing `[MASK]` directly into `token_ids` would reach back out and silently corrupt the
caller's sentence:

```python
# WITHOUT clone — mutate the argument
def bad_mask(token_ids):
    token_ids[2] = 4
    return token_ids

sentence = torch.tensor([2, 11, 12, 13, 3])
bad_mask(sentence)
print(sentence)   # [2, 11, 4, 13, 3]  ← 💥 caller's sentence wrecked too

# WITH clone — work on a private copy
def good_mask(token_ids):
    masked_ids = token_ids.clone()
    masked_ids[2] = 4
    return masked_ids

sentence = torch.tensor([2, 11, 12, 13, 3])
good_mask(sentence)
print(sentence)   # [2, 11, 12, 13, 3]  ← ✅ untouched
```

The point is **not** "we read `token_ids` again later inside the function" (we don't) — it's
that the value belongs to whoever called us, and they might still need it clean (logging,
re-masking a fresh copy next epoch, …). Mutating an argument is a hidden side effect — a
nasty action-at-a-distance bug that doesn't even error, the data just quietly changes.
`.clone()` is the photocopy: scribble on the copy, hand the original back spotless.

> PyTorch's copy verb is **`.clone()`**, not `.copy()` (numpy/pandas use `.copy()`). The
> different name flags that a tensor copy is **autograd-aware** — the copy stays attached to
> the computation graph. (Irrelevant for these integer id tensors, but `.clone()` is still
> the correct call.)

## How a random number becomes a probability

`torch.rand()` draws from the **uniform distribution** on `[0, 1)`: every value is equally
likely, no clumping. Its density is flat:

```
f(x) = 1   for 0 ≤ x < 1,   else 0
```

Because probability = **area under the curve** and the height is a constant `1`, the
probability of any sub-range is just its **width**:

```
f(x)
 1 │■■■■■■■■■■■■■■│
   │  area = 0.8  │ .1 │ .1 │
 0 └──────────────┴────┴────
   0            0.8  0.9    1

P(x < 0.8)      = 0.8 × 1 = 0.8   → 80% → [MASK]
P(0.8 ≤ x < 0.9)= 0.1 × 1 = 0.1   → 10% → random
P(0.9 ≤ x < 1)  = 0.1 × 1 = 0.1   → 10% → keep
```

So `decision < 0.8` doesn't *force* 80% — it gives each token an **80% chance**. You don't
add the random numbers; you **compare each one to the 0.8 cutoff**. A short run can look
lopsided (5 draws might all fall below 0.8), but that's small-sample noise. Over the
millions of tokens in real pre-training, the *law of large numbers* pulls the split to
≈80/10/10:

| tokens masked | typical [MASK] share |
|---|---|
| 5 | anything — 40%, 100%, … |
| 100 | roughly 75–85% |
| 1,000,000 | ≈ 80.0% |

This is exactly how Google's BERT does it — a per-token draw, not a counted quota.

## Why return `labels` if it's mostly -100

`labels` is **not** just a list of -100s — it's mostly -100 but holds the **real answer**
at every selected position. `masked_ids` is the **question**; `labels` is the **answer
key**. The loss needs both:

```python
mlm_logits = model(masked_ids)                  # model's guess for each slot
loss = cross_entropy(mlm_logits, labels)        # grade guess vs answer; -100 = skip
```

Return only `masked_ids` and you'd have a quiz with no answer key. The -100s aren't waste —
they tell the loss *"ignore these ~85% positions, only grade where a real id sits."* That
is the seam [`loss.py`](loss.md) reads with `ignore_index=-100`.

## A full worked example

```python
token_ids         = [2, 11, 12, 13, 3]     # [CLS] the cat sat [SEP]
special_token_ids = {0, 2, 3}              # [PAD], [CLS], [SEP]

# 1. labels start all -100
labels   = [-100, -100, -100, -100, -100]

# 2. specials marked, candidates = real words
special_mask = [T, F, F, F, T]   →  candidates = [1, 2, 3]   # the, cat, sat

# 3. pick 15%:  round(3 * 0.15)=0 → max(1,0)=1 → say slot 2 ("cat") selected
selected = [2]

# 4. write the answer
labels   = [-100, -100, 12, -100, -100]

# 5. corrupt input — say decision=0.3 (<0.8) → [MASK]
masked_ids = [2, 11, 4, 13, 3]             # [CLS] the [MASK] sat [SEP]

return masked_ids, labels
```

- `masked_ids` → what the model sees: `[CLS] the [MASK] sat [SEP]`
- `labels` → grade only slot 2, answer `12` (cat); ignore everything else

## References

- **Paper:** Devlin et al. 2019, [*BERT*](https://arxiv.org/abs/1810.04805) — §3.1, Task #1 (Masked LM); the 80/10/10 split is described there and ablated in Appendix C.2.
- **Google BERT (TF)** — `create_masked_lm_predictions` in [`create_pretraining_data.py`](https://github.com/google-research/bert/blob/master/create_pretraining_data.py): the original implementation (same `max(1, round(...))` guard and per-token 80/10/10 draw).
- **The consumer of these labels:** [`loss.md`](loss.md) — the `-100` / `ignore_index` convention.
- **The producer of `token_ids`:** [`nsp.md`](../objectives/nsp.md) — sentence-pair assembly.
