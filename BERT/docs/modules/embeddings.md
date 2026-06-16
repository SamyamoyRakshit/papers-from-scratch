# BERT Input Embeddings

> Module: [`BERT/models/modules/embeddings.py`](../../models/modules/embeddings.py) — `BERTEmbeddings`
> Paper: Devlin et al. 2019, [*BERT: Pre-training of Deep Bidirectional Transformers*](https://arxiv.org/abs/1810.04805), Section 3.1, Figure 2

This is the very first layer of BERT — it turns a row of integer token IDs into a
matrix of vectors the encoder stack can actually work with. Everything downstream
(attention, feed-forward, the whole 12-layer tower) consumes the output of this module.

## Contents

1. [What it computes](#1-what-it-computes)
2. [The three tables](#2-the-three-tables)
3. [Why no √d_model scaling](#3-why-no-d_model-scaling)
4. [Position: embedding, not encoding](#4-position-embedding-not-encoding)
5. [LayerNorm and its ε](#5-layernorm-and-its-ε)
6. [Two small but deliberate details](#6-two-small-but-deliberate-details)
7. [A full worked example](#7-a-full-worked-example)
- [References](#references)

---

## 1. What it computes

For every token position, BERT sums **three** learned embeddings, then normalizes:

```
E = TokenEmbedding(input_ids)
  + SegmentEmbedding(token_type_ids)
  + PositionEmbedding(position_ids)

output = Dropout(LayerNorm(E))
```

Shape-wise, the whole module is a transformation:

```
input_ids: (batch_size, seq_len)  ── integers
                  │
                  ▼
output:    (batch_size, seq_len, d_model)  ── floats
```

That trailing `d_model` (768 for BERT-base) is the dimension every later layer expects.

### What's from the paper vs what's an implementation detail

This distinction matters for a faithful replication — the paper is **silent** on
several things this code does, so those parts are sourced from the reference
implementations, not invented:

| Element | Source |
|---|---|
| The three-way **sum** | **Paper** — Section 3.1, Figure 2 |
| **LayerNorm + dropout** after the sum | *Implementation* — Google's TF BERT `embedding_postprocessor`, mirrored in HF `BertEmbeddings` |
| **No √d_model scaling** | *Implementation* — Google's `embedding_lookup` and HF both feed raw summed embeddings into LayerNorm |
| ε = `1e-12` for LayerNorm | *Implementation* — `tf.contrib.layers.layer_norm` default, kept as `BertConfig.layer_norm_eps` |

When a value comes from an implementation rather than the paper, the docstring
says so explicitly — so anyone reading can trace exactly what is paper-mandated
versus convention.

---

## 2. The three tables

Each table is an `nn.Embedding`, which is just a **lookup table**: it maps an integer
index to a row vector. The three differ only in how many rows they have and what
those rows mean.

| Table | Rows (`num_embeddings`) | Width (`embedding_dim`) | Answers the question |
|---|---|---|---|
| `token_embedding` | `vocab_size` (30522) | `d_model` (768) | *Which word is this?* |
| `segment_embedding` | `num_segments` (2) | `d_model` (768) | *Which sentence — A or B?* |
| `positional_embedding` | `max_position_embeddings` (512) | `d_model` (768) | *Which position in the sequence?* |

All three have width `d_model` for one reason: **they get summed element-wise**, and
addition requires identical shapes. The *row count* differs per table (it answers a
different question); the *column count* is forced to `d_model` so the three line up.

### How a lookup actually works (the part that trips people up)

`nn.Embedding` does **not** do any math — it indexes. Take the segment table as the
simplest example (only 2 rows):

```
segment_embedding table  (2 × 768):
         col0   col1   col2   ...  col767
row 0:  [0.12, -0.30,  0.88,  ..., 0.04]   ← "I belong to sentence A"
row 1:  [0.55,  0.10, -0.20,  ..., 0.71]   ← "I belong to sentence B"
```

Feed it the integer sequence `[0, 0, 0, 0, 1, 1, 1, 1]` (shape `(1, 8)`), and it
**replaces each integer with that row**:

```
[0, 0, 0, 0, 1, 1, 1, 1]          (1, 8)     ── integers in
        │  lookup
        ▼
[[row0, row0, row0, row0,         (1, 8, 768) ── vectors out
  row1, row1, row1, row1]]
```

So the `768` does **not** live in the input IDs — it appears *because* the table's
columns are 768 wide. The integers are just **keys**; the table holds the numbers.
All four `0`s get the *exact same* row (segment A's vector), all four `1`s get the
*exact same* row. Same idea applies to `token_embedding` (30522 keys) and
`positional_embedding` (512 keys).

### Where the row values come from

Random at first, **learned by backprop** after. At `__init__`, PyTorch fills every
table with random floats. During training, gradients flow back through the lookups
and update those rows like any other weight. After training, row 0 of the segment
table has *learned* to be a useful "sentence A" signal, row 1 a "sentence B" signal,
and so on. You never hand-craft the 768 numbers — gradient descent finds them.

> This is the key contrast with the original Transformer's **position** handling
> (see §4): there it's a fixed sin/cos formula that is never updated.

---

## 3. Why no √d_model scaling

The original Transformer multiplies its token embeddings by √d_model
("Attention Is All You Need", Section 3.4: *"we multiply those weights by √d_model"*) —
and our [`transformer/models/modules/embeddings.py`](../../../transformer/models/modules/embeddings.py)
does exactly that.

**BERT does not.** Both Google's TF BERT (`embedding_lookup`) and HF's
`BertEmbeddings.forward` feed the raw summed embeddings straight into LayerNorm
with no scaling factor. The paper never mentions the scaling either way — so the
"BERT omits it" fact comes from reading the implementations, not the paper.

Intuitively the scaling is less necessary here because BERT immediately applies
**LayerNorm** to the sum, which re-normalizes the magnitude regardless — so a
hand-tuned √d_model factor would be redundant.

---

## 4. Position: embedding, not encoding

The naming in this repo is precise and worth preserving:

| | File | Mechanism | Learnable? |
|---|---|---|---|
| Transformer | `positional_encoding.py` | sin/cos **formula** | No — fixed |
| BERT | `embeddings.py` → `positional_embedding` | `nn.Embedding` **lookup** | Yes — trained |

- **Encoding** = computed by a deterministic transformation (the sinusoidal formula).
  No parameters; the same position always maps to the same vector.
- **Embedding** = a learned lookup table, exactly like token embeddings.

BERT chose learned position embeddings. The original Transformer authors tested both
and found them nearly identical — Table 3 row (E): the sinusoidal base scores
4.92 PPL / 25.8 BLEU and the learned variant 4.92 / 25.7 on newstest2013. In their
own words (Section 3.5):

> *"We also experimented with using learned positional embeddings [9] instead, and
> found that the two versions produced nearly identical results (see Table 3 row (E)).
> We chose the sinusoidal version because it may allow the model to extrapolate to
> sequence lengths longer than the ones encountered during training."*

Note the irony: the Transformer authors picked **sinusoidal** *specifically* for
length extrapolation. BERT went the **opposite** way — learned, for simplicity (it's
the same machinery as the other two tables) — and in doing so accepted exactly the
hard length cap (512, see below) that the extrapolation argument was meant to avoid.

**What "extrapolation" means here:** can the model handle a sequence *longer* than
anything it saw during training?

- **Sinusoidal = a formula.** Position → vector is computed by `sin/cos(pos / …)`.
  A formula has no size limit — plug in position 5000 even if training never went past
  512 and it returns a valid vector. So the model *might* generalize to longer
  sequences than it was trained on.
- **Learned = a lookup table.** Position → vector is `nn.Embedding(512, 768)` — literally
  512 stored rows. There is **no row 513**. Ask for position 512 and it raises an
  index error. A table can only answer for positions it physically has rows for, so
  BERT is **hard-capped at 512** with zero extrapolation.

| | Transformer | BERT |
|---|---|---|
| Choice | sinusoidal (formula) | learned (table) |
| Why | *to enable* extrapolation | for simplicity |
| Result | can go beyond training lengths | hard cap at 512 |

The Transformer authors went out of their way to pick the option that *preserves* the
ability to handle longer sequences; BERT picked the simpler, uniform learned-table
design and threw that capability away. This is exactly why later models (Llama, etc.)
adopted **RoPE** — a formula-based scheme that brings extrapolation back *without*
returning to fixed sinusoids.

### Table size vs the 128/512 training schedule

Given the 512 cap above, one more distinction is easy to mix up — don't confuse two
separate things:

- **`max_position_embeddings = 512`** — the *table size* (architecture). The position
  table is `nn.Embedding(512, 768)`, always 512 rows. Never changes during training.
- **"128 for 90% of steps, then 512 for 10%"** (paper, training section) — a *training
  schedule*, a compute trick. Attention is O(seq_len²), so most pre-training runs at
  length 128 to save compute. The catch: during the 128 phase only table rows 0–127
  ever get looked up, so only those receive gradients — rows 128–511 stay at their
  random init. The final 10% at length 512 exists precisely to **train those upper
  rows**. ("...to learn the positional embeddings.")

For a from-scratch replication on a single machine, picking one length (128 is plenty)
and noting the two-phase schedule as a faithful-but-simplified detail is fine — the
curriculum is a scaling optimization, not a correctness requirement.

---

## 5. LayerNorm and its ε

After the sum, the module applies LayerNorm then dropout. The `LayerNorm` itself is
**imported** from the transformer replication (it's identical math — single source of
truth):

```python
from transformer.models.modules.layer_norm import LayerNorm
```

> This is an absolute import, not relative, because `BERT/` and `transformer/` are
> two **separate top-level packages** — a relative import can't cross from one into a
> sibling. It only resolves when you **run from the repo root** (e.g.
> `python -m BERT.scripts.pretrain`), so both packages are visible on `sys.path`.

### The ε value: three conventions, none from a paper

`layer_norm_eps` defaults to `1e-12`. No paper specifies an ε for LayerNorm — these
are all implementation choices:

| ε | Origin | Used by |
|---|---|---|
| `1e-12` | `tf.contrib.layers.layer_norm` default → `BertConfig.layer_norm_eps` | **BERT** (this module) |
| `1e-5` | PyTorch `nn.LayerNorm` default | our `transformer/` replication |
| `1e-6` | Tensor2Tensor's original-Transformer `layer_norm` | the original Transformer's TF code |

We use `1e-12` here to match BERT. (Our transformer's `LayerNorm` accepts an `eps`
kwarg, so passing `1e-12` correctly overrides its `1e-5` default.)

---

## 6. Two small but deliberate details

### `padding_idx` on the token table

```python
self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
```

`padding_idx` does two things: it initializes the `[PAD]` row to **zeros** and
**freezes** it (no gradient), so it stays the zero vector forever.

This is **optional**, not a BERT-specific requirement — our transformer omits it
entirely and is still correct. The reason it doesn't matter functionally: the pad
embedding's value never reaches an output anyway. Padding positions are masked to
−∞ in attention (`create_padding_mask`) and dropped from the loss via
`ignore_index=pad_idx`. So whether the pad row is a clean zero or learned junk, the
result is identical.

We keep it in BERT for two minor reasons: it **matches HF's `BertEmbeddings`**, and
a clean zero vector is tidier than untracked random values that waste a sliver of
gradient compute getting updated to no effect.

### The `position_ids` buffer

```python
self.register_buffer(
    "position_ids",
    torch.arange(max_position_embeddings).unsqueeze(0),  # (1, 512)
    persistent=False,
)
```

This precomputes the position indices `[[0, 1, 2, ..., 511]]` **once** at init, instead
of rebuilding `torch.arange` on every forward. Three design points:

1. **Why a registered buffer (the real reason):** a buffer automatically moves with
   `model.to(device)`. A plain `self.position_ids = torch.arange(...)` attribute would
   stay on CPU when you move the model to MPS/GPU, and the lookup would crash with a
   device-mismatch. (Verified: after `.to("mps")` the forward runs on `mps:0`.)
2. **Shape `(1, 512)` — the leading `1`:** it's a batch axis. Every example in a batch
   shares the identical position sequence, so we store one row and let broadcasting
   replicate it across the batch in the final sum, rather than materializing a full
   `(batch, seq)` tensor.
3. **`persistent=False`:** excludes `position_ids` from `state_dict`. It's trivially
   reconstructable from `torch.arange` and never learned, so there's no reason to bloat
   every checkpoint with it. (`persistent=False` affects *saving only* — device-move
   still works.)

In `forward`, it's sliced to the current sequence length:

```python
position_ids = self.position_ids[:, :seq_len]   # (1, 512) → (1, seq_len)
```

`[:, :seq_len]` is 2D indexing: `:` keeps the single batch row, `:seq_len` takes
columns `0 .. seq_len-1`. A 128-token batch grabs `[0..127]`; a 512-token batch grabs
`[0..511]` — same buffer, different view, nothing recomputed.

### `token_type_ids` defaulting to zeros

```python
if token_type_ids is None:
    token_type_ids = torch.zeros_like(input_ids)
```

Two-sentence inputs (NSP pre-training) pass explicit segment IDs. Single-sentence
tasks (most fine-tuning) don't have a sentence B, so passing `None` is a convenience —
the module fills in all-zeros, treating every token as segment A, so callers don't
have to build a zero tensor themselves.

---

## 7. A full worked example

Batch of 2 sentences, padded to length 6:

```python
input_ids = [
    [101, 2023, 2003, 1996, 3793, 102],   # [CLS] this is the text [SEP]
    [101, 2748,  102,    0,    0,   0],   # [CLS] yes [SEP] [PAD][PAD][PAD]
]                                          # shape (2, 6)

token_type_ids = [
    [0, 0, 0, 0, 1, 1],   # (illustrative — sentence A then B)
    [0, 0, 0, 0, 0, 0],
]
```

Inside `forward` (`seq_len = 6`):

```python
position_ids = self.position_ids[:, :6]          # [[0, 1, 2, 3, 4, 5]]    (1, 6)

token_emb    = token_embedding(input_ids)        # (2, 6, 768)
segment_emb  = segment_embedding(token_type_ids) # (2, 6, 768)
position_emb = positional_embedding(position_ids)# (1, 6, 768)

E = token_emb + segment_emb + position_emb
#   (2,6,768) + (2,6,768)   + (1,6,768)
#                              └── broadcasts the batch-1 over both rows
#   → (2, 6, 768)

output = dropout(layer_norm(E))                  # (2, 6, 768)
```

The `(1, 6, 768)` position tensor is **broadcast** into both sentences — both get the
same per-position vectors, while their token and segment vectors differ.

---

## References

- **Paper:** Devlin et al. 2019, [*BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*](https://arxiv.org/abs/1810.04805)
- **Official Google BERT (TF):** [`modeling.py`](https://github.com/google-research/bert/blob/master/modeling.py) — `embedding_lookup`, `embedding_postprocessor`
- **HF Transformers (PyTorch), pinned to v5.12.0:** [`modeling_bert.py`](https://github.com/huggingface/transformers/blob/v5.12.0/src/transformers/models/bert/modeling_bert.py) — `BertEmbeddings`
