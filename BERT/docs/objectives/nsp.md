# Next Sentence Prediction — the pair builder (`nsp.py`)

> Module: [`BERT/utils/nsp.py`](../../utils/nsp.py) — `build_nsp_example`
> Paper: Devlin et al. 2019, [*BERT*](https://arxiv.org/abs/1810.04805), §3.1 (Task #2: Next Sentence Prediction)

This is the step that **produces** the `token_ids` everything downstream consumes. It takes
a tokenized corpus, picks a sentence A and a sentence B, glues them into the single packed
sequence BERT eats — `[CLS] A [SEP] B [SEP]` — and labels whether B really followed A.

It does **not** tokenize text (the WordPiece tokenizer in
[`data_utils.py`](../utils/data_utils.md) does that, upstream) and it does **not** mask
(that's [`masking.py`](../utils/masking.md), downstream). It sits in the middle: assemble
the pair, emit the segment ids and the IsNext/NotNext label.

Throughout, **A** = first sentence, **B** = second sentence, **doc** = one contiguous
document.

## Contents

- [Where this sits in the pipeline](#where-this-sits-in-the-pipeline)
- [The corpus is nested three levels deep](#the-corpus-is-nested-three-levels-deep)
- [`a_index` vs `sentence_a` — the one that confuses readers](#a_index-vs-sentence_a--the-one-that-confuses-readers)
- [The 50/50 IsNext / NotNext rule](#the-5050-isnext--notnext-rule)
- [Walking the code, line by line](#walking-the-code-line-by-line)
  - [1. Find A and ask "is there a real next?"](#1-find-a-and-ask-is-there-a-real-next)
  - [2. The coin flip](#2-the-coin-flip)
  - [3. Assemble the sequence and segment ids](#3-assemble-the-sequence-and-segment-ids)
  - [4. Drawing a random B — `is not`, not `!=`](#4-drawing-a-random-b--is-not-not-)
- [How a random number becomes 50%](#how-a-random-number-becomes-50)
- [Paper text vs reference code](#paper-text-vs-reference-code)
- [A full worked example](#a-full-worked-example)
- [References](#references)

---

## Where this sits in the pipeline

```
raw text  (blank line = document boundary)
  │  WordPiece tokenizer (data_utils.py) — string ↔ id
  ▼
all_documents  — corpus as nested lists of ids (see below)
  │
  ▼
nsp.py  ──►  token_ids       [CLS] A [SEP] B [SEP]   ← the packed pair
             token_type_ids  0…0 over A, 1…1 over B  ← segment ids
             nsp_label        0 = IsNext / 1 = NotNext
  │
  ▼
masking.py  — hide 15% of token_ids → masked_ids + labels
  │
  ▼
data_utils.py — pad + batch many examples → (B, S) tensors
  │
  ▼
model → loss.py   (nsp_label feeds the NSP head's cross-entropy)
```

Two seams worth naming:

- **`token_ids` → [`masking.py`](../utils/masking.md):** the pair this builds is the exact
  input `mask_tokens` masks. Because `[CLS]`/`[SEP]` are stuffed in *here*, masking has to
  be told to skip them — that's why it takes `special_token_ids`.
- **`nsp_label` → [`loss.py`](../utils/loss.md):** a 2-class label, `0 = IsNext`,
  `1 = NotNext`, matching `nsp_labels` in `BERTPreTrainingLoss`.

## The corpus is nested three levels deep

The three list arguments aren't three unrelated things — they're the **same corpus peeled
one layer at a time**. Each level strips off one set of brackets:

```
all_documents   [ [ [5,6,7], [8,9] ],  [ [20,21] ] ]     ← the whole corpus
                  └────── doc0 ──────┘  └── doc1 ──┘

document          [ [5,6,7], [8,9] ]                      ← one document (doc0)
                    └ s0 ─┘  └ s1┘

sentence_a          [5,6,7]                               ← one sentence
                     │ │ │
                     └─┴─┴── token ids ("the temple stood")
```

| arg | type | depth | what it holds |
|---|---|---|---|
| `all_documents` | `list[list[list[int]]]` | 3 | every document — the pool a random B is drawn from |
| `document` | `list[list[int]]` | 2 | the one document A came from — holds A's real next sentence |
| `sentence_b` (derived) | `list[int]` | 1 | a sentence — a list of **token ids**, *not* an index |

A "document" is one contiguous piece of writing (a Wikipedia article, a book, one temple's
description). The reader splits the corpus on **blank lines** — blank line = new document.
That boundary is what makes NSP meaningful (see [the rule](#the-5050-isnext--notnext-rule)).

## `a_index` vs `sentence_a` — the one that confuses readers

The function is handed **`a_index` (a number)**, and *derives* **`sentence_a` (the content)**
from it on the first line:

```python
def build_nsp_example(a_index, document, all_documents, cls_id, sep_id):
    sentence_a = document[a_index]      # number → content
```

They are **two views of the same sentence**:

| name | example value | what it is | how to read it |
|---|---|---|---|
| `a_index` | `3` | a **position** — *where* A sits in the document | "sentence #3" |
| `sentence_a` | `[7, 8, 9]` | the **content** — the actual token ids at that position | "the words of sentence #3" |

`document[a_index]` is the bridge: feed it the *position*, it returns the *content*.

```
a_index = 3
            │  document[3]
            ▼
sentence_a = [7, 8, 9]      # = document[3]
```

**Why the function takes the index, not the sentence.** An earlier version took
`sentence_a` directly and then searched for its position with `document.index(sentence_a)`.
That search **breaks when a document repeats a sentence** — `.index()` returns the *first*
match, so it can grab the wrong "next" sentence:

```python
doc = [
    [5, 6],       # 0  "temple stood"
    [7, 8, 9],    # 1  "built in stone"
    [10, 11],     # 2  "people prayed"
    [7, 8, 9],    # 3  "built in stone"   ← identical tokens to sentence 1
    [12, 13],     # 4  "river flowed"
]

# We mean sentence 3. But searching by content:
doc.index([7, 8, 9])   # → 1, NOT 3   (returns the FIRST match)
# → "real next" becomes doc[2] "people prayed", not the true doc[4] "river flowed"
#   → an IsNext pair that is silently WRONG
```

`.index()` matches by **value**, and two sentences with identical tokens are indistinguishable
to it. Passing `a_index` in sidesteps this entirely: the caller already knows the position
(it's looping by index), so there's nothing to search for. It's also O(1) instead of O(n).

```python
# the caller (data_utils.py) hands the position straight in:
for a_index in range(len(document)):          # the index is free here
    build_nsp_example(a_index, document, all_documents, cls_id, sep_id)
```

So: **`a_index` comes in, `sentence_a` is `document[a_index]`.** One is the address, the
other is what lives there.

## The 50/50 IsNext / NotNext rule

NSP teaches the model: *"does sentence B actually follow sentence A?"* That question only
has a real answer **inside a document**, which is why document boundaries matter:

| label | value | how B is chosen | meaning |
|---|---|---|---|
| **IsNext** | `0` | the **real next sentence** in A's own document | a genuine succession |
| **NotNext** | `1` | a **random sentence from a different document** | guaranteed unrelated |

- **IsNext only exists within one document** — sentence 2 of an article genuinely follows
  sentence 1; coherence (pronouns, topic) lives inside a document.
- **NotNext draws from a *different* document** so the negative is reliably unrelated. (The
  paper says "a random sentence from the corpus"; the *different-document* refinement comes
  from Google's reference code — see [below](#paper-text-vs-reference-code).)

Each example is a 50/50 coin flip between the two, so the NSP head sees a balanced 0/1
target and can't win by always guessing one class.

## Walking the code, line by line

Running corpus:

```python
doc0 = [[5, 6, 7], [8, 9], [10, 11, 12]]   # 3 sentences
doc1 = [[20, 21], [22, 23, 24]]            # 2 sentences
all_documents = [doc0, doc1]
cls_id, sep_id = 1, 2
```

### 1. Find A and ask "is there a real next?"

```python
sentence_a = document[a_index]
has_real_next = a_index + 1 < len(document)
```

`has_real_next` answers *"is there a sentence **after** A in this document?"* — **not** "does
A exist" (it always does). With `document = doc0` (indices 0,1,2):

| A is sentence… | `a_index` | `a_index + 1 < 3` | `has_real_next` |
|---|---|---|---|
| `[5,6,7]` (first) | 0 | `1 < 3` | **True** |
| `[8,9]` (middle) | 1 | `2 < 3` | **True** |
| `[10,11,12]` (**last**) | 2 | `3 < 3` | **False** — nothing follows |

The last sentence of a document has no successor, so it can never be an IsNext A.

### 2. The coin flip

```python
if has_real_next and random.random() < 0.5:
    sentence_b = document[a_index + 1]     # the real next
    nsp_label = 0                          # IsNext
else:
    sentence_b = _random_sentence(all_documents, document)
    nsp_label = 1                          # NotNext
```

Two gates joined by `and`, **both** must be True to pick IsNext:

1. `has_real_next` — *can* we do IsNext? (Is there a next sentence to point at?)
2. `random.random() < 0.5` — the fair coin (see [below](#how-a-random-number-becomes-50)).

`and` **short-circuits**: if `has_real_next` is `False`, Python never rolls the coin — it
drops straight to `else` (forced NotNext). The `else` therefore has **two doors**: the coin
came up ≥ 0.5, *or* A was the last sentence.

```
has_real_next   coin < 0.5        result
   True          True (50%)   →   IsNext   (real next, label 0)
   True          False (50%)  →   NotNext  (random B, label 1)
   False         (not rolled) →   NotNext  (random B, label 1)
```

### 3. Assemble the sequence and segment ids

```python
token_ids = [cls_id] + sentence_a + [sep_id] + sentence_b + [sep_id]
token_type_ids = (
    [0] * (len(sentence_a) + 2)     # [CLS] A [SEP]
    + [1] * (len(sentence_b) + 1)   # B [SEP]
)
```

`sentence_b` here is the **content** (a list of ids like `[8, 9]`), not the index — so it
splices in as real tokens. With A = `[5,6,7]`, B = `[8,9]`:

```
token_ids       = [ 1,  5, 6, 7,  2,  8, 9,  2 ]
                    │   └─ A ─┘   │  └ B ┘   │
                  [CLS]        [SEP]       [SEP]

token_type_ids  = [ 0,  0, 0, 0,  0,  1, 1,  1 ]
                   └──── segment A (len A + 2) ─┘ └ segment B (len B + 1) ┘
```

`token_type_ids` is **new here** — masking never touches it (segment ids aren't masked), but
[`embeddings.py`](../modules/embeddings.md) needs it for the **segment embedding** (the
middle of token + segment + position). The `+2`/`+1` count the special tokens: A's segment
covers `[CLS]` + A + `[SEP]` (two extras); B's covers B + `[SEP]` (one extra).

### 4. Drawing a random B — `is not`, not `!=`

```python
def _random_sentence(all_documents, exclude_document):
    if len(all_documents) < 2:
        raise ValueError("NSP needs at least 2 documents — …")
    while True:
        doc = random.choice(all_documents)
        if doc is not exclude_document:    # different document — by IDENTITY
            return random.choice(doc)
```

Two deliberate choices:

- **`is not` (identity), not `!=` (equality).** We want "a different document *object*", not
  "a document whose contents differ." If two documents happened to be identical token-for-token,
  `!=` would wrongly reject the second one; `is not` only asks *"is this literally the same
  object as A's document?"* — which is both correct and O(1) (a pointer check, no deep
  comparison).

  ```python
  doc0 = [[5, 6], [7, 8]]
  doc1 = [[5, 6], [7, 8]]    # different object, same contents
  doc1 != doc0     # → False  (contents match → "same") ✗
  doc1 is not doc0 # → True   (different object)        ✓
  ```

- **The `< 2` guard.** With only one document there *is* no other document to draw a NotNext
  from — the `while True` would spin forever. A 1-document corpus is fundamentally broken for
  NSP, so we **raise loudly** rather than hang or silently mislabel. In real pre-training
  (thousands of docs) it never fires; in a too-small test it tells you immediately.

## How a random number becomes 50%

`random.random()` draws from the **uniform distribution** on `[0, 1)` — every value equally
likely. So the chance a draw lands below a cutoff is just the **width** of that slice:

```
0.0 ───────────────── 0.5 ───────────────── 1.0
│      < 0.5 (True)     │     >= 0.5 (False)   │
│<──────── 50% ────────>│<──────── 50% ───────>│
```

`random.random() < 0.5` is therefore a **fair coin** — `True` ≈ half the time. The cutoff
*is* the probability: `< 0.5` → 50%, `< 0.7` → 70%. It's the same trick
[`masking.py`](../utils/masking.md) uses for its `decision < 0.8` (the 80% `[MASK]` band),
just a different cutoff.

A short run can look lopsided (a few draws might all fall below 0.5), but the *law of large
numbers* pulls it to ≈50/50 over the corpus. **There is no quota** — each example flips its
own coin independently; the balance emerges in aggregate.

> Note: `random.random()` is the *whole* left-hand side of the comparison — it's a function
> **call** that returns a number, and that returned number is what gets compared to `0.5`.
> Equivalent to `draw = random.random(); if draw < 0.5:` — just inlined.

## Paper text vs reference code

The paper (§3.1) literally says B is *"a random sentence from the corpus"* — it does **not**
state "from a different document." That constraint comes from Google's reference
implementation:

| Source | NotNext B is… |
|---|---|
| **Paper** (Devlin §3.1) | "a random sentence from the corpus" — no document constraint |
| **Google's code** (`create_pretraining_data.py`) | a random sentence from a **different document** |

The code adds the constraint because a "random sentence from the corpus" could land inside
A's own document and accidentally be A's *real* next sentence (or a topically coherent one) —
a mislabeled negative. Restricting to a different document guarantees a clean negative. This
module follows the **code** (the de-facto standard for "replicating BERT").

This builder also **simplifies** two things the *full* Google pipeline does, both of which
belong downstream in [`data_utils.py`](../utils/data_utils.md):

- **One sentence per segment.** Google packs *multiple* sentences into A and B up to a target
  length (~512). This puts a single sentence in each.
- **No truncation to `max_seq_len`.** Google trims the longer segment token-by-token when the
  pair is too long.

As a standalone NSP-pair builder the sampling logic is faithful; the length-packing/truncation
lives next door.

## A full worked example

```python
doc0 = [[5, 6, 7], [8, 9], [10, 11, 12]]
doc1 = [[20, 21], [22, 23, 24]]
all_documents = [doc0, doc1]
cls_id, sep_id = 1, 2

# Call for sentence 0 of doc0:
build_nsp_example(a_index=0, document=doc0, all_documents=all_documents,
                  cls_id=1, sep_id=2)

# 1. sentence_a = doc0[0] = [5, 6, 7];  has_real_next = (1 < 3) = True
# 2. coin < 0.5 → IsNext → sentence_b = doc0[1] = [8, 9], nsp_label = 0
# 3. assemble:
token_ids      = [1, 5, 6, 7, 2, 8, 9, 2]     # [CLS] A [SEP] B [SEP]
token_type_ids = [0, 0, 0, 0, 0, 1, 1, 1]
nsp_label      = 0                              # IsNext

# --- alternate: coin >= 0.5 (NotNext) ---
# sentence_b = a random sentence from doc1 (a DIFFERENT doc), e.g. [22, 23, 24]
token_ids      = [1, 5, 6, 7, 2, 22, 23, 24, 2]
token_type_ids = [0, 0, 0, 0, 0,  1,  1,  1, 1]
nsp_label      = 1                              # NotNext
```

`token_ids` then flows into [`masking.py`](../utils/masking.md) (which hides 15% of it), and
`nsp_label` flows into [`loss.py`](../utils/loss.md) (the NSP head's cross-entropy).

## References

- **Paper:** Devlin et al. 2019, [*BERT*](https://arxiv.org/abs/1810.04805) — §3.1, Task #2 (Next Sentence Prediction).
- **Google BERT (TF)** — `create_instances_from_document` / `create_pretraining_data.py`: [the reference implementation](https://github.com/google-research/bert/blob/master/create_pretraining_data.py) (different-document negatives, sentence packing, truncation).
- **The consumer of `token_ids`:** [`masking.md`](../utils/masking.md) — the 15% / 80-10-10 masking step.
- **The consumer of `nsp_label`:** [`loss.md`](../utils/loss.md) — combined MLM + NSP loss.
- **The consumer of `token_type_ids`:** [`embeddings.md`](../modules/embeddings.md) — token + segment + position embeddings.
