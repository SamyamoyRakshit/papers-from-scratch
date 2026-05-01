## Table of Contents

1. [What the Paper Says](#what-the-paper-says)
2. [Configuration — `base.yaml` vs `tiny.yaml`](#configuration--baseyaml-vs-tinyyaml)
3. [The Data Pipeline — Big Picture](#the-data-pipeline--big-picture)
4. [Step 1 — Filter Bad Pairs (`is_valid_pair`)](#step-1--filter-bad-pairs-is_valid_pair)
5. [Step 2 — Train Tokenizer (`train_tokenizer`)](#step-2--train-tokenizer-train_tokenizer)
   - [SentencePieceTrainer vs SentencePieceProcessor](#sentencepiecetrainer-vs-sentencepieceprocessor)
   - [Why Shared Vocabulary?](#why-shared-vocabulary)
6. [Step 3 — Encode Text (`encode` / `decode`)](#step-3--encode-text-encode--decode)
   - [`max_seq_len` vs `max_len` — Two Different Things](#max_seq_len-vs-max_len--two-different-things)
7. [Step 4 — Build Dataset (`TranslationDataset`)](#step-4--build-dataset-translationdataset)
   - [What PyTorch's `Dataset` Base Class Does](#what-pytorchs-dataset-base-class-does)
     - [Why DataLoader needs `__len__`](#why-dataloader-needs-__len__)
     - [How OUR DataLoader works (with `batch_sampler`)](#how-our-dataloader-works-with-batch_sampler)
   - [End-to-End Example](#end-to-end-example)
8. [Step 5 — Token-Based Batching (`TokenBatchSampler`)](#step-5--token-based-batching-tokenbatchsampler)
   - [Why Not Fixed Batch Size?](#why-not-fixed-batch-size)
   - [The Algorithm — Step by Step](#the-algorithm--step-by-step)
   - [Why Padding Still Exists](#why-padding-still-exists)
   - [`yield` vs `return`](#yield-vs-return)
   - [Batch Size — You Can't Know It in Advance](#batch-size--you-cant-know-it-in-advance)
9. [Step 6 — Collate & DataLoader (`collate_fn`, `create_dataloaders`)](#step-6--collate--dataloader-collate_fn-create_dataloaders)
   - [Why Custom `collate_fn`?](#why-custom-collate_fn)
   - [`batch_sampler` vs `batch_size` — Mutually Exclusive](#batch_sampler-vs-batch_size--mutually-exclusive)
   - [`pin_memory` — Why We Don't Use It](#pin_memory--why-we-dont-use-it)
   - [Train / Val Split — No Test Set](#train--val-split--no-test-set)
10. [`num_workers` — Why 0 on macOS](#num_workers--why-0-on-macos)
    - [What Workers Do](#what-workers-do)
    - [Why macOS Is Different](#why-macos-is-different)
    - [When Workers Help vs Hurt](#when-workers-help-vs-hurt)
11. [Data Quality — Noisy Pairs](#data-quality--noisy-pairs)
12. [Vocab Size — How 16K Subwords Cover 1M Sentences](#vocab-size--how-16k-subwords-cover-1m-sentences)
    - [Shared Vocab — Same Table, Different IDs](#shared-vocab--same-table-different-ids)
    - [Why 16K and Not 37K?](#why-16k-and-not-37k)
13. [The Full Journey — Raw Text to Model Input](#the-full-journey--raw-text-to-model-input)
    - [Batch Shape Through Each Step](#batch-shape-through-each-step)
    - [What's Shared vs Different Between Encoder and Decoder](#whats-shared-vs-different-between-encoder-and-decoder)
    - [Why 5K PE Rows When max_seq_len = 128?](#why-5k-pe-rows-when-max_seq_len--128)
14. [Integration — How `train.py` Wires Everything Together](#integration--how-trainpy-wires-everything-together)
15. [All the Numbers — Cheat Sheet](#all-the-numbers--cheat-sheet)
16. [References](#references)

---

# What the Paper Says

From **"Attention Is All You Need"** (Vaswani et al., 2017), Section 5.1 — Training Data and Batching:

> "Sentences were encoded using byte-pair encoding, which has a shared source-target vocabulary of about 37000 tokens."

> "Sentence pairs were batched together by approximate sequence length. Each training batch contained a set of sentence pairs containing approximately 25,000 source tokens and 25,000 target tokens."

Our adaptation:
- **Dataset:** AI4Bharat Samanantar English-Bengali (8.5M pairs) instead of WMT 2014 English-German (4.5M pairs)
- **Vocab size:** 16,000 (reduced for 500K pairs — 37K would be too large)
- **Batch tokens:** ~8K per side (reduced for M1 memory — paper used ~25K on 8x P100)

---

# Configuration — `base.yaml` vs `tiny.yaml`

`data_utils.py` is config-agnostic — it takes plain arguments. Values come from two YAML presets in `configs/`. `tiny.yaml` is a debug preset that runs the whole pipeline in ~2 minutes so you can catch bugs fast. `base.yaml` is the real training config.

Every argument to the data_utils functions maps to one YAML key:

| YAML key | `base.yaml` | `tiny.yaml` | Consumed by |
|---|---|---|---|
| `data.dataset` | `"ai4bharat/samanantar"` | `"ai4bharat/samanantar"` | `load_dataset(path=...)` in `train.py` |
| `data.tgt_lang` | `"bn"` | `"bn"` | `load_dataset(name=...)` in `train.py` |
| `data.max_rows` | `500000` | `1000` | `raw_dataset.select(range(...))` in `train.py` |
| `data.vocab_size` | `16000` | `4000` | `train_tokenizer(vocab_size=...)` |
| `data.num_workers` | `0` | `0` | `DataLoader(num_workers=...)` |
| `data.filter_max_ratio` | `5.0` | `5.0` | `is_valid_pair(max_ratio=...)` via `TranslationDataset` |
| `data.filter_min_words` | `1` | `1` | `is_valid_pair(min_words=...)` via `TranslationDataset` |
| `training.max_seq_len` | `128` | `64` | `encode(max_seq_len=...)` |
| `training.max_tokens_per_batch` | `8000` | `2000` | `TokenBatchSampler(max_tokens=...)` |
| `training.val_split` | `0.1` | `0.1` | `train_test_split(test_size=...)` inside `create_dataloaders` |
| `tokens.pad_idx` | `0` | `0` | `train_tokenizer(pad_id=...)`, `collate_fn(pad_idx=...)` |
| `tokens.sos_idx` | `1` | `1` | `train_tokenizer(sos_id=...)`, prepended in `encode` |
| `tokens.eos_idx` | `2` | `2` | `train_tokenizer(eos_id=...)`, appended in `encode` |
| `tokens.unk_idx` | `3` | `3` | `train_tokenizer(unk_id=...)` |
| `seed` | `42` | `42` | `train_test_split(seed=...)` |
| `paths.tokenizer_path` | `"transformer/tokenizer/base/sp.model"` | `"transformer/tokenizer/tiny/sp.model"` | `load_tokenizer(model_path=...)` or `train_tokenizer(model_prefix=...)` |

**What changes between the two presets (and why):**

- `vocab_size` 16K → 4K — with 1000 rows of text you can't learn 16K meaningful subword merges; 4K is enough to verify encoding works.
- `max_seq_len` 128 → 64 — shorter sequences mean smaller tensors, faster forward/backward pass.
- `max_tokens_per_batch` 8K → 2K — smaller batches mean fewer tokens per step; pipeline still exercises the full batch sampler.
- `max_rows` 500K → 1000 — enough data to build one batch and run a few steps, not enough to actually learn translation.

Everything else (pad/sos/eos/unk indices, seed, val split, filter thresholds, num_workers) stays identical — those are pipeline invariants, not tuning knobs.

**Workflow:** run `tiny.yaml` first — if loss doesn't move across 5 epochs, there's a bug. Only switch to `base.yaml` once the pipeline is verified.

---

# The Data Pipeline — Big Picture

```
HuggingFace Dataset (8.5M total en-bn pairs, we load 500K — config: max_rows=500000)
           │
           ▼
    ┌──────────────┐
    │ is_valid_pair│ ← Filter bad pairs (empty, wrong script, extreme ratios)
    └──────┬───────┘
           │ clean text pairs
           ▼
    ┌──────────────────┐
    │ train_tokenizer  │ ← Train SentencePiece BPE on all clean text
    └──────┬───────────┘   (one-time, saves .model + .vocab files)
           │ tokenizer (.model file)
           ▼
    ┌──────────────────┐
    │ encode           │ ← Text → token IDs + `<sos>` / `<eos>`
    └──────┬───────────┘   "I love AI" → [1, 14, 87, 3, 2]
           │
           ▼
    ┌──────────────────────┐
    │ TranslationDataset   │ ← Stores all (src_ids, tgt_ids) pairs
    └──────┬───────────────┘   Filters again + logs how many kept/dropped
           │
           ▼
    ┌──────────────────────┐
    │ TokenBatchSampler    │ ← Groups by ~8K tokens per batch (not fixed count)
    └──────┬───────────────┘   Sort by length → pack → shuffle batches
           │
           ▼
    ┌──────────────────────┐
    │ collate_fn           │ ← Pad variable-length sequences → rectangular tensors
    └──────┬───────────────┘
           │
           ▼
    ┌──────────────────────┐
    │ DataLoader           │ ← Yields (src_tensor, tgt_tensor) batches for training
    └──────────────────────┘
```

The pipeline runs in two phases:
1. **Once** — `train_tokenizer` learns BPE merges and saves the model
2. **Every run** — `create_dataloaders` loads the tokenizer, builds datasets, creates batches

---

# Step 1 — Filter Bad Pairs (`is_valid_pair`)

The Samanantar dataset contains noisy pairs — empty strings, English-only targets, extreme length mismatches. `is_valid_pair` catches these before they corrupt training.

```python
def is_valid_pair(src: str, tgt: str, max_ratio: float = 5.0, min_words: int = 1) -> bool:
```

```
is_valid_pair(src, tgt)
        │
        ▼
┌───────────────────┐    Yes
│ src or tgt empty? │────────→ return False
└───────┬───────────┘
        │ No
        ▼
┌───────────────────────┐    Yes
│ No Bengali in tgt?    │────────→ return False
└───────┬───────────────┘
        │ No
        ▼
┌────────────────────────┐    Yes
│ src or tgt < min_words?│────────→ return False
└───────┬────────────────┘
        │ No
        ▼
┌────────────────────────┐    Yes
│ length ratio>max_ratio?│────────→ return False
└───────┬────────────────┘
        │ No
        ▼
   return True
```

**Checks performed:**

| Check | Why | Example of bad pair |
|---|---|---|
| Empty string | Can't train on nothing | `src=""`, `tgt="হ্যালো"` |
| No Bengali in target | Target should be Bengali | `src="hello"`, `tgt="hello"` |
| Too few words | Single-character pairs are noise | `src="a"`, `tgt="ক"` |
| Extreme length ratio | Misaligned pairs | `src="The cat sat"` (3 words), `tgt="ক"` (1 word) → ratio 3.0 OK. `src="The quick brown fox..."` (30 words), `tgt="ক"` (1 word) → ratio 30.0, filtered |

The `_bengali_pattern` regex checks Unicode range `\u0980-\u09FF` — the Bengali/Assamese block.

**Why the underscore?** `_bengali_pattern` — the leading underscore means "private to this module." It's a Python convention: other files shouldn't import or rely on this variable. It's an implementation detail.

---

# Step 2 — Train Tokenizer (`train_tokenizer`)

```python
def train_tokenizer(dataset, vocab_size, pad_id, sos_id, eos_id, unk_id, model_prefix):
```

```
train_tokenizer(dataset, vocab_size=16000, model_prefix="tokenizer/sp")
        │
        ▼
┌──────────────────────────────┐
│ os.makedirs("tokenizer/")    │ ← create parent dirs if missing
└───────┬──────────────────────┘
        │
        ▼
┌──────────────────────────────────────┐
│ Write temp file:                     │
│ "tokenizer/sp_train_text.txt"        │  ← temp_file (deleted after training)
│                                      │
│ for each example in dataset:         │
│   if is_valid_pair(src, tgt):        │
│     write src + "\n"                 │
│     write tgt + "\n"                 │
└───────┬──────────────────────────────┘
        │ temp file with all clean sentences
        ▼
┌──────────────────────────────────────┐
│ SentencePieceTrainer.train(          │
│   input=temp_file,                   │  ← reads the temp file
│   model_prefix="tokenizer/sp",       │
│   vocab_size=16000,                  │
│   model_type="bpe"                   │
│ )                                    │
│                                      │
│ Creates (permanent, NOT temp):       │
│   tokenizer/sp.model  ← trained BPE  │
│   tokenizer/sp.vocab  ← vocab list   │
└───────┬──────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────┐
│ os.remove(temp_file)                 │ ← delete "tokenizer/sp_train_text.txt"
└───────┬──────────────────────────────┘   (only the temp file, NOT .model/.vocab)
        │
        ▼
┌──────────────────────────────────────┐
│ load_tokenizer("tokenizer/sp.model") │ ← load the permanent .model file
└───────┬──────────────────────────────┘
        │
        ▼
  return SentencePieceProcessor
```

**What it does:**

1. Writes all valid sentences to a temp file (SentencePiece reads from file, not memory)
2. Trains a BPE tokenizer — learns which character sequences to merge
3. Saves `{model_prefix}.model` and `{model_prefix}.vocab` (e.g., `tokenizer/sp.model`)
4. Deletes the temp file
5. Loads and returns the trained tokenizer

**The `os.makedirs` call:**

```python
os.makedirs(os.path.dirname(temp_file), exist_ok=True)
```

Creates all intermediate-level parent directories if they are missing (e.g. `"a/b/c/sp"` → creates `"a/"`, `"a/b/"`, `"a/b/c/"`). This is different from `os.mkdir` which only creates the last directory and fails if parents don't exist.

## SentencePieceTrainer vs SentencePieceProcessor

These are two completely different classes:

| | SentencePieceTrainer | SentencePieceProcessor |
|---|---|---|
| **Purpose** | Trains a new tokenizer | Uses a trained tokenizer |
| **When used** | Once, during setup | Every time you tokenize text |
| **Input** | Raw text file | `.model` file |
| **Output** | `.model` + `.vocab` files | Token IDs |
| **Analogy** | Teacher writing the textbook | Student using the textbook |

```python
# Phase 1 — Train (one-time)
spm.SentencePieceTrainer.train(input="text.txt", model_prefix="tokenizer/sp", ...)
# Creates: tokenizer/sp.model, tokenizer/sp.vocab

# Phase 2 — Use (every time)
sp = spm.SentencePieceProcessor()    # empty processor
sp.load("tokenizer/sp.model")        # load trained model
ids = sp.encode("I love AI")         # → [14, 87, 3]
```

**Why lowercase methods?** SentencePiece is a C++ library with a Python wrapper. The Python wrapper follows PEP 8 (lowercase with underscores), even though the underlying C++ uses CamelCase.

## Why Shared Vocabulary?

> "shared source-target vocabulary of about 37000 tokens" — Section 5.1

Both English and Bengali text go into the same tokenizer. The tokenizer learns subword units for both languages in a single vocabulary. This means:
- Words that appear in both languages (like "AI", "DNA", numbers) share the same token
- The encoder and decoder operate over the same vocabulary space
- The output projection layer maps to the same vocabulary for both languages

---

# Step 3 — Encode Text (`encode` / `decode`)

```python
def encode(sp, text, max_seq_len):
    token_ids = sp.encode(text.strip())       # "I love AI" → [14, 87, 3]
    token_ids = token_ids[: max_seq_len - 2]  # truncate, leaving room for <sos> and <eos>
    return [sp.bos_id()] + token_ids + [sp.eos_id()]  # [1, 14, 87, 3, 2]
```

```
encode(sp, "I love AI", max_seq_len=128)
        │
        ▼
┌─────────────────────────────────┐
│ sp.encode("I love AI")          │
│ → [14, 87, 3]                   │ ← SentencePiece BPE tokenization
└───────┬─────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│ truncate to max_seq_len - 2     │
│ [14, 87, 3][:126]               │ ← reserve 2 slots for <sos> and <eos>
│ → [14, 87, 3]  (no change)      │
└───────┬─────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│ prepend <sos> (1), append <eos> │
│ (2)                             │
│ → [1, 14, 87, 3, 2]             │
│    ↑              ↑             │
│  <sos>          <eos>           │
└───────┬─────────────────────────┘
        │
        ▼
  return [1, 14, 87, 3, 2]
```

**End-to-end example:**

```
Input:  "I love AI"
                    ↓ sp.encode()
Tokens: [14, 87, 3]
                    ↓ truncate to max_seq_len - 2
Tokens: [14, 87, 3]     (no truncation needed here — only 3 tokens)
                    ↓ add <sos> and <eos>
Output: [1, 14, 87, 3, 2]
         ↑              ↑
        <sos>          <eos>
```

If `max_seq_len = 5`:

```
Tokens: [14, 87, 3]  → truncate to 5-2=3 → [14, 87, 3] → [1, 14, 87, 3, 2]  (fits exactly)
```

If `max_seq_len = 4`:

```
Tokens: [14, 87, 3]  → truncate to 4-2=2 → [14, 87] → [1, 14, 87, 2]  (lost token 3)
```

**Why `-2`?** We need to reserve 2 slots — one for `<sos>` at the start and one for `<eos>` at the end. Without truncation, a 128-token sentence + `<sos>` + `<eos>` = 130 tokens, exceeding `max_seq_len=128`.

## `max_seq_len` vs `max_len` — Two Different Things

| Config key | Value | What it controls |
|---|---|---|
| `training.max_seq_len` | 128 | Actual sequence truncation — how long your encoded sentences can be |
| `model.max_len` | 5000 | Positional encoding table size — pre-computed sin/cos for 5000 positions |

`max_seq_len` is the practical limit: sentences longer than 128 tokens get truncated.

`max_len` is the theoretical capacity: the PE table has 5000 rows, but we never use more than 128 (because `max_seq_len` truncates first). The table is larger than needed because it's pre-computed once and costs almost nothing in memory.

See [positional_encoding.md Section 9](../modules/positional_encoding.md) for a deeper dive on `max_len`.

---

# Step 4 — Build Dataset (`TranslationDataset`)

```python
class TranslationDataset(Dataset):
    def __init__(
            self,
            dataset,
            sp: spm.SentencePieceProcessor,
            max_seq_len: int,
            filter_max_ratio: float = 5.0,
            filter_min_words: int = 1,
    ):
        self.pairs = []                                # (src_ids, tgt_ids) tuples
        skipped = 0                                    # count of filtered pairs

        for example in dataset:
            src = example["src"].strip()               # HuggingFace Samanantar uses "src"/"tgt" keys
            tgt = example["tgt"].strip()

            if not is_valid_pair(src, tgt, max_ratio=filter_max_ratio, min_words=filter_min_words):
                skipped += 1
                continue

            src_ids = encode(sp, src, max_seq_len)     # text → token IDs with <sos>/<eos>
            tgt_ids = encode(sp, tgt, max_seq_len)

            if len(src_ids) > 2 and len(tgt_ids) > 2:  # must have at least 1 real token
                self.pairs.append((src_ids, tgt_ids))
            else:
                skipped += 1                           # empty after encoding

        total = len(self.pairs) + skipped              # base for percentages
        print(f"[TranslationDataset] Kept {len(self.pairs)} pairs ({len(self.pairs)/total*100:.1f}%), "
              f"filtered {skipped} ({skipped/total*100:.1f}%)")
```

```
TranslationDataset(raw_dataset, sp, max_seq_len=128, filter_max_ratio, filter_min_words)
        │
        ▼
┌───────────────────────────────────────────┐
│ for each example in raw_dataset:          │
│                                           │
│   example = {"src": "I love AI",          │
│              "tgt": "আমি AI ভালোবাসি"}     │
│        │                                  │
│        ▼                                  │
│   ┌─────────────────────┐                 │
│   │ is_valid_pair?      │──No──→ skip++   │
│   └───────┬─────────────┘                 │
│           │ Yes                           │
│           ▼                               │
│   ┌─────────────────────────────────┐     │
│   │ src_ids = encode(sp, src, 128)  │     │
│   │ → [1, 14, 87, 3, 2]             │     │
│   │ tgt_ids = encode(sp, tgt, 128)  │     │
│   │ → [1, 45, 3, 92, 2]             │     │
│   └───────┬─────────────────────────┘     │
│           │                               │
│           ▼                               │
│   ┌──────────────────────┐                │
│   │ both len > 2?        │──No──→ skip++  │
│   │ (has real tokens?)   │                │
│   └───────┬──────────────┘                │
│           │ Yes                           │
│           ▼                               │
│   self.pairs.append(                      │
│     (src_ids, tgt_ids)                    │
│   )                                       │
└───────────────────────────────────────────┘
        │
        ▼
print("[TranslationDataset] Kept 448312 pairs (99.6%), filtered 1688 (0.4%)")
        │
        ▼
  self.pairs = [
    ([1, 14, 87, 3, 2],  [1, 45, 3, 92, 2]),     # pair 0
    ([1, 28, 56, 2],     [1, 45, 71, 2]),          # pair 1
    ...                                             # 448K+ pairs
  ]
```

**Two filter stages:**

1. `is_valid_pair` — catches bad text (empty, wrong script, extreme ratio)
2. `len > 2` check — catches pairs where tokenization produced only `<sos>` + `<eos>` (no real tokens)

The `print` at the end tells you how many pairs survived. If `filtered` is high (e.g., 30%+), something is wrong with the data.

## What PyTorch's `Dataset` Base Class Does

`Dataset` is a template (abstract class). It requires you to implement two methods:

```python
__len__()          # "How many samples do you have?"
__getitem__(idx)   # "Give me sample number idx"
```

`Dataset` itself has **no** `__len__` — that's why you **must** define it. If you don't, DataLoader crashes with `TypeError: object of type 'TranslationDataset' has no len()`.

### Why DataLoader needs `__len__`

DataLoader needs to know the total number of samples so it can **create indices** and **know when an epoch ends**:

```python
# What a standard DataLoader does internally:
indices = list(range(len(dataset)))         # ← needs __len__ to build this
# → [0, 1, 2, ..., 448311]

if shuffle:
    random.shuffle(indices)                  # randomize order

for i in range(0, len(indices), batch_size): # iterate in chunks
    batch_indices = indices[i:i+batch_size]  # e.g. [42, 7801, 102, ...]
    batch = [dataset[idx] for idx in batch_indices]  # ← needs __getitem__
    yield collate_fn(batch)
```

Without `__len__`, it can't even create the list of indices to iterate over.

### How OUR DataLoader works (with `batch_sampler`)

Our setup is different — we use `batch_sampler=TokenBatchSampler` instead of `batch_size`. The sampler takes over index management:

```python
# What our DataLoader does internally:

# Step 1: TokenBatchSampler already created batches in __init__:
#   batches = [[0, 3, 5, 2], [4, 1], ...]  ← groups of indices

# Step 2: Each epoch, DataLoader asks sampler for batches:
for batch_indices in batch_sampler:        # calls TokenBatchSampler.__iter__()
                                            # yields [0, 3, 5, 2], then [4, 1], ...

    # Step 3: Fetch each pair using __getitem__:
    batch = []
    for idx in batch_indices:               # e.g. [0, 3, 5, 2]
        pair = dataset[idx]                 # calls __getitem__(0), __getitem__(3), ...
        batch.append(pair)                  # → [([1,14,87,3,2], [1,45,3,92,2]), ...]

    # Step 4: Pad to same length using our collate_fn:
    src_padded, tgt_padded = collate_fn(batch, pad_idx=0)

    # Step 5: Give to training loop:
    yield src_padded, tgt_padded
```

```
Standard DataLoader:                    Our DataLoader:
┌──────────────────────┐               ┌──────────────────────────────┐
│ len(dataset) = 448K  │               │ len(dataset) = 448K          │
│ → indices [0..448K]  │               │ (PyTorch still checks this)  │
│ → chunk by batch_size│               │                              │
│ → fetch + collate    │               │ TokenBatchSampler handles:   │
└──────────────────────┘               │ → sort by length             │
                                       │ → pack by ~8K tokens         │
                                       │ → yield batch indices        │
                                       │                              │
                                       │ DataLoader handles:          │
                                       │ → fetch pairs via __getitem__│
                                       │ → pad via collate_fn         │
                                       └──────────────────────────────┘
```

**Key point:** Even though `TokenBatchSampler` handles all the batching logic, PyTorch still requires `__len__` on the Dataset — it's part of the contract. DataLoader checks for it even when batch_sampler handles everything.

You never call `__len__` or `__getitem__` directly. DataLoader calls them for you internally.

## End-to-End Example

```
Raw dataset (3 pairs):
  {"src": "I love AI",       "tgt": "আমি AI ভালোবাসি"}
  {"src": "",                "tgt": "হ্যালো"}                ← empty src, filtered
  {"src": "Hello world",     "tgt": "হ্যালো বিশ্ব"}

After filtering + encoding (max_seq_len=64):
  self.pairs = [
      ([1, 14, 87, 3, 2],       [1, 45, 3, 92, 2]),       # pair 0
      ([1, 28, 56, 2],          [1, 45, 71, 2])            # pair 1
  ]
  # pair 0: src has 5 tokens, tgt has 5 tokens
  # pair 1: src has 4 tokens, tgt has 4 tokens
  # 1 pair filtered (empty src)

dataset[0]  →  ([1, 14, 87, 3, 2], [1, 45, 3, 92, 2])
dataset[1]  →  ([1, 28, 56, 2], [1, 45, 71, 2])
len(dataset) →  2
```

**Why `dataset[0]` works even though the field is `self.pairs`?**

Because `dataset[0]` calls `__getitem__(0)`, which returns `self.pairs[0]`. The `Dataset` base class routes `dataset[idx]` → `self.__getitem__(idx)`. Python's `[]` operator calls `__getitem__` on any object.

---

# Step 5 — Token-Based Batching (`TokenBatchSampler`)

```
TokenBatchSampler(dataset, max_tokens=8000, shuffle=True)
        │
        ▼
┌───────────────────────────────────────────────┐
│ _create_batches()                             │
│                                               │
│   ┌─────────────────────────────────────┐     │
│   │ Step 1: Sort indices by src length  │     │
│   │ [5,2,8,3,5,4] → [3,3,4,5,5,8]       │     │
│   └───────┬─────────────────────────────┘     │
│           │                                   │
│           ▼                                   │
│   ┌─────────────────────────────────────┐     │
│   │ Step 2: Pack into batches           │     │
│   │                                     │     │
│   │ for each idx (sorted):              │     │
│   │   pair_len = max(src, tgt)          │     │
│   │   new_max = max(max_in_batch,       │     │
│   │                  pair_len)          │     │
│   │   would_be = (batch_count+1)        │     │
│   │              × new_max              │     │
│   │                                     │     │
│   │   would_be > 8000?                  │     │
│   │     Yes → save batch, start new     │     │
│   │     No  → add to current batch      │     │
│   └───────┬─────────────────────────────┘     │
│           │                                   │
│           ▼                                   │
│   batches = [[0,3,5,2], [4,1], ...]           │
└───────┬───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────┐
│ __iter__() — called each epoch                │
│                                               │
│   ┌─────────────────────────────────────┐     │
│   │ Step 3: Shuffle batch ORDER         │     │
│   │ (not sentences within batches)      │     │
│   │ [[4,1], [0,3,5,2], ...]             │     │
│   └───────┬─────────────────────────────┘     │
│           │                                   │
│           ▼                                   │
│   yield [4,1]        → DataLoader gets batch  │
│   yield [0,3,5,2]   → DataLoader gets batch   │
│   ...                                         │
└───────────────────────────────────────────────┘
```

## Why Not Fixed Batch Size?

With a fixed batch size (e.g., 32 sentences):

```
Batch 1: 32 short sentences (5 tokens each)  → 32 × 5 = 160 tokens   ← GPU underutilized
Batch 2: 32 long sentences (100 tokens each)  → 32 × 100 = 3200 tokens ← GPU overloaded
```

The GPU gets wildly different workloads per batch. Token-based batching (Section 5.1) keeps the total tokens per batch roughly constant:

```
Batch 1: 200 short sentences × 40 tokens = 8000 tokens  ← consistent
Batch 2: 20 long sentences × 400 tokens  = 8000 tokens  ← consistent
```

## The Algorithm — Step by Step

**Input:** 6 pairs with these src lengths: `[3, 8, 5, 3, 5, 4]`, `max_tokens = 20`

**Step 1 — Sort by source length:**

```
Original indices: [0, 1, 2, 3, 4, 5]
Sorted by src len: indices = [0, 3, 5, 2, 4, 1]
                   src lens: [3, 3, 4, 5, 5, 8]
```

Sorting groups similar-length sentences together → less padding waste.

**Step 2 — Pack into batches:**

For each pair, `pair_len = max(len(src_ids), len(tgt_ids))` — the longer side determines how much padding the pair needs.

The key tracking variable is `max_len_in_batch` — the longest sequence seen so far in the current batch. This is critical because when we pad, **all** sequences in the batch pad to the longest one.

```
Processing idx=0 (pair_len=3):
  max_len_in_batch = 0
  new_max = max(0, 3) = 3
  would_be = (0 + 1) × 3 = 3    ← 3 ≤ 20, fits
  current_batch = [0], max_len_in_batch = 3

Processing idx=3 (pair_len=3):
  new_max = max(3, 3) = 3
  would_be = (1 + 1) × 3 = 6    ← 6 ≤ 20, fits
  current_batch = [0, 3], max_len_in_batch = 3

Processing idx=5 (pair_len=4):
  new_max = max(3, 4) = 4
  would_be = (2 + 1) × 4 = 12   ← 12 ≤ 20, fits
  current_batch = [0, 3, 5], max_len_in_batch = 4

Processing idx=2 (pair_len=5):
  new_max = max(4, 5) = 5
  would_be = (3 + 1) × 5 = 20   ← 20 ≤ 20, fits (exactly!)
  current_batch = [0, 3, 5, 2], max_len_in_batch = 5

Processing idx=4 (pair_len=5):
  new_max = max(5, 5) = 5
  would_be = (4 + 1) × 5 = 25   ← 25 > 20, doesn't fit!
  → Save batch [0, 3, 5, 2], start new batch
  current_batch = [4], max_len_in_batch = 5

Processing idx=1 (pair_len=8):
  new_max = max(5, 8) = 8
  would_be = (1 + 1) × 8 = 16   ← 16 ≤ 20, fits
  current_batch = [4, 1], max_len_in_batch = 8

End → save last batch [4, 1]

Result: batches = [[0, 3, 5, 2], [4, 1]]
```

**Why `(len(current_batch) + 1) * new_max`?**

This calculates the **exact tensor size** if we add this pair. After padding, every sequence in the batch will be `new_max` long. So total tokens = `num_sequences × max_length`. The `+1` accounts for the pair we're considering adding.

**Why track `max_len_in_batch`?**

Without it, the code would use `pair_len` of the current pair — but a previous pair in the batch might have a longer target. Example:

```
Batch so far: pair A (src=3, tgt=7) → pair_len=7
New pair B:   (src=4, tgt=4) → pair_len=4

Without max_len_in_batch: would_be = 2 × 4 = 8   ← WRONG (actual tensor: 2 × 7 = 14)
With max_len_in_batch:    would_be = 2 × 7 = 14   ← CORRECT
```

**Step 3 — Shuffle batches (not sentences):**

```python
def __iter__(self):
    if self.shuffle:
        random.shuffle(self.batches)      # shuffles [[0,3,5,2], [4,1]] order
    for batch in self.batches:
        yield batch
```

Sentences within a batch stay together (they're similar length). Only the batch order changes each epoch.

## Why Padding Still Exists

Token-based batching **reduces** padding but doesn't **eliminate** it. Within a batch, sequences still vary slightly:

```
Batch: [pair_0 (len=3), pair_3 (len=3), pair_5 (len=4), pair_2 (len=5)]

After padding to max_len_in_batch=5:
  pair_0: [1, 14, 87, 2, 0]         ← 2 pad tokens
  pair_3: [1, 45, 71, 2, 0]         ← 2 pad tokens
  pair_5: [1, 28, 56, 92, 2]        ← 1 pad token
  pair_2: [1, 33, 44, 55, 2]        ← 0 pad tokens
```

Without sorting, you might batch a 3-token and a 100-token sentence together → 97 wasted pad tokens. Sorting keeps lengths close, minimizing waste.

## `yield` vs `return`

```python
# yield — returns one item at a time, pauses, resumes
def __iter__(self):
    for batch in self.batches:
        yield batch    # gives batch [0,3,5,2], pauses, gives [4,1], pauses, ...

# return — exits immediately, gives everything at once
def __iter__(self):
    return self.batches    # gives ALL batches at once (DataLoader can't iterate this)
```

DataLoader needs `yield` because it processes one batch at a time — load batch → train → load next batch. It doesn't want all batches in memory at once.

## Batch Size — You Can't Know It in Advance

With `max_tokens=8000`, the number of sentences per batch depends on sentence lengths:

```
Short sentences (avg 20 tokens): ~8000 / 20 = ~400 sentences per batch
Long sentences (avg 100 tokens):  ~8000 / 100 = ~80 sentences per batch
```

This is why we use `batch_sampler` instead of `batch_size` in DataLoader — the batch size varies.

The number of batches per epoch also varies slightly because `TokenBatchSampler` packs greedily. With 500K pairs and `max_tokens=8000`, you might get ~2000-5000 batches depending on length distribution.

---

# Step 6 — Collate & DataLoader (`collate_fn`, `create_dataloaders`)

```
collate_fn(batch, pad_idx=0)
        │
        │  batch = [([1,14,87,3,2], [1,45,3,92,2]),    ← pair 0
        │           ([1,28,56,2],   [1,45,71,2])]       ← pair 1
        ▼
┌───────────────────────────────────────┐
│ Split into src and tgt lists          │
│                                       │
│ src_seqs = [tensor([1,14,87,3,2]),    │
│             tensor([1,28,56,2])]      │
│                                       │
│ tgt_seqs = [tensor([1,45,3,92,2]),    │
│             tensor([1,45,71,2])]      │
└───────┬───────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│ pad_sequence (pad to longest in batch)│
│                                       │
│ src_padded:          tgt_padded:      │
│ [[1,14,87,3,2],     [[1,45,3,92,2],   │
│  [1,28,56,2,0]]      [1,45,71,2,0]]   │
│             ↑                    ↑    │
│           pad=0                pad=0  │
│                                       │
│ Shape: (2, 5)        Shape: (2, 5)    │
└───────┬───────────────────────────────┘
        │
        ▼
  return (src_padded, tgt_padded)
         → training loop unpacks: for src, tgt in loader
```

```
create_dataloaders(raw_dataset, sp, max_seq_len, max_tokens, pad_idx, seed, num_workers,
                   val_split, filter_max_ratio, filter_min_words)
        │
        │  raw_dataset is loaded and size-capped in scripts/train.py:
        │    raw_dataset = load_dataset("ai4bharat/samanantar", "bn", split="train")
        │    raw_dataset = raw_dataset.select(range(500000))   # cap to 500K
        │  then passed in here (so the tokenizer and dataloaders share one load)
        ▼
┌───────────────────────────────────────────────────┐
│ train_test_split(test_size=0.1, seed=42)          │
│ → train_raw: 450K pairs (90%)                     │
│ → val_raw:   50K pairs  (10%)                     │
└───────┬───────────────────────────────────────────┘
        │
        ├──────────────────────┐
        ▼                      ▼
┌────────────────────────┐  ┌────────────────────────┐
│ TranslationDataset     │  │ TranslationDataset     │
│ (train_raw, sp,        │  │ (val_raw, sp,          │
│  max_seq_len,          │  │  max_seq_len,          │
│  filter_max_ratio,     │  │  filter_max_ratio,     │
│  filter_min_words)     │  │  filter_min_words)     │
│ → train_dataset        │  │ → val_dataset          │
└───────┬────────────────┘  └───────┬────────────────┘
        │                      │
        ▼                      ▼
┌──────────────────┐  ┌──────────────────┐
│ TokenBatchSampler│  │ TokenBatchSampler│
│ (train, 8000,    │  │ (val, 8000,      │
│  shuffle=True)   │  │  shuffle=False)  │
│ → train_sampler  │  │ → val_sampler    │
└───────┬──────────┘  └───────┬──────────┘
        │                     │
        ▼                     ▼
┌───────────────────┐  ┌──────────────────┐
│ DataLoader(       │  │ DataLoader(      │
│  batch_sampler,   │  │  batch_sampler,  │
│  collate_fn,      │  │  collate_fn,     │
│  num_workers=0)   │  │  num_workers=0)  │
│ → train_loader    │  │ → val_loader     │
└───────┬───────────┘  └───────┬──────────┘
        │                      │
        ▼                      ▼
  return (train_loader, val_loader)
```

## Why Custom `collate_fn`?

PyTorch's default collate does `torch.stack()` — which requires all tensors to have the **same shape**. Our sequences have variable lengths:

```python
# Default collate tries:
torch.stack([
    tensor([1, 14, 87, 3, 2]),      # length 5
    tensor([1, 28, 56, 2]),          # length 4  ← CRASH! different size
])
# RuntimeError: stack expects each tensor to be equal size
```

Our custom `collate_fn` pads sequences to the longest in the batch:

```python
def collate_fn(batch: List[Tuple[List[int], List[int]]], pad_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    # List of token IDs → list of long tensors (embedding layers require int64 indices)
    src_seqs = [torch.tensor(pair[0], dtype=torch.long) for pair in batch]   # pair[0] = src_ids
    tgt_seqs = [torch.tensor(pair[1], dtype=torch.long) for pair in batch]   # pair[1] = tgt_ids

    # pad_sequence pads to the longest seq in the batch → rectangular (batch, seq_len) tensor
    src_padded = torch.nn.utils.rnn.pad_sequence(
        sequences=src_seqs, batch_first=True, padding_value=pad_idx
    )
    tgt_padded = torch.nn.utils.rnn.pad_sequence(
        sequences=tgt_seqs, batch_first=True, padding_value=pad_idx
    )

    return src_padded, tgt_padded
```

**Example:**

```
Input batch (3 pairs):
  src: [1, 14, 87, 3, 2]    tgt: [1, 45, 3, 92, 2]
  src: [1, 28, 56, 2]       tgt: [1, 45, 71, 2]
  src: [1, 33, 2]           tgt: [1, 55, 66, 77, 2]

After padding (pad_idx=0):
  src_padded:                        tgt_padded:
  [[1, 14, 87, 3, 2],               [[1, 45,  3, 92, 2],
   [1, 28, 56, 2, 0],                [1, 45, 71,  2, 0],
   [1, 33,  2, 0, 0]]                [1, 55, 66, 77, 2]]

  Shape: (3, 5)                      Shape: (3, 5)
```

**`batch_first=True`** → shape is `(batch_size, seq_len)`. Without it, shape would be `(seq_len, batch_size)` — we want batch first because it's more intuitive and matches what the model expects.

**Why does `collate_fn` return a tuple of 2 tensors, not 1?** Because DataLoader doesn't care about the return type — it just passes through whatever `collate_fn` returns. The training loop unpacks it:

```python
for src, tgt in train_loader:    # unpacks the (src_tensor, tgt_tensor) tuple
    ...
```

**Why the `lambda` wrapper?** In the DataLoader we write:

```python
# DataLoader only passes ONE argument (batch) to collate_fn.
# But our collate_fn needs TWO arguments (batch, pad_idx).
# So we wrap it:

collate_fn=lambda batch: collate_fn(batch, pad_idx)
#          ↑                        ↑       ↑
#     DataLoader passes         goes through  we pre-fill
#     this automatically                      this ourselves
```

Here's what happens step by step when DataLoader processes one batch:

```python
# Inside DataLoader:
batch = []
for idx in [0, 3, 5, 2]:                    # indices from TokenBatchSampler
    pair = dataset[idx]                       # __getitem__ → (src_ids, tgt_ids)
    batch.append(pair)

# batch is now:
# [([1,14,87,3,2], [1,45,3,92,2]),          ← pair 0
#  ([1,33,71,2],   [1,88,2]),                ← pair 3
#  ([1,28,56,92,2],[1,45,71,2]),             ← pair 5
#  ([1,33,44,55,2],[1,92,55,44,33,2])]       ← pair 2

# DataLoader calls:
result = collate_fn(batch)                    # ← DataLoader passes only batch
#         ↓
# lambda batch: collate_fn(batch, pad_idx)
#                ↓
# our collate_fn(batch, pad_idx=0)            # ← pad_idx pre-filled by us
#                ↓
# returns (src_padded, tgt_padded)
```

Without the lambda, DataLoader would call `collate_fn(batch)` — missing `pad_idx`, crash:
`TypeError: collate_fn() missing 1 required positional argument: 'pad_idx'`

## `batch_sampler` vs `batch_size` — Mutually Exclusive

```python
# Standard DataLoader — fixed batch size:
DataLoader(dataset, batch_size=32)    # always 32 sentences per batch

# Our DataLoader — variable batch size from TokenBatchSampler:
DataLoader(dataset, batch_sampler=train_sampler)    # sampler decides batch contents
```

You **cannot** use both. If you pass `batch_sampler`, PyTorch ignores `batch_size`, `shuffle`, `sampler`, and `drop_last` — the batch sampler controls everything.

## `pin_memory` — Why We Don't Use It

`pin_memory` is a DataLoader option that pre-loads data into **pinned (page-locked) CPU memory** so that transferring to GPU is faster.

```
Normal (pageable) memory → GPU:
  CPU RAM → copy to pinned memory → transfer to GPU
  [=====step 1=====][=====step 2=====]

Pinned memory → GPU:
  Pinned CPU RAM → transfer to GPU
  [=====step 2=====]                    ← skips step 1
```

Pinned memory can't be swapped to disk by the OS — it stays in physical RAM. This lets the GPU do a **direct memory access (DMA)** transfer without waiting for the CPU to copy it first.

**Why we don't use it:**

| Hardware | `pin_memory` useful? | Why |
|---|---|---|
| NVIDIA CUDA GPU | Yes — noticeable speedup | Separate CPU and GPU memory, transfer is a bottleneck |
| Apple M1 (MPS) | No benefit | **Unified memory** — CPU and GPU share the same RAM, no "transfer" to skip |
| CPU only | No benefit | No GPU transfer at all |

```python
# Our DataLoader — no pin_memory (default is False):
DataLoader(dataset=train_dataset, batch_sampler=train_sampler, ...)

# If you were on NVIDIA GPU, you'd add:
DataLoader(dataset=train_dataset, batch_sampler=train_sampler, pin_memory=True, ...)
```

## Train / Val Split — No Test Set

```python
split = raw_dataset.train_test_split(test_size=val_split, seed=seed)
train_raw = split["train"]     # 90% — train on this
val_raw = split["test"]        # 10% — validate on this
```

HuggingFace's `train_test_split` only creates two keys: `"train"` and `"test"`. We use `"test"` as our **validation** set — the naming is misleading but that's HuggingFace's API.

**Why no separate test set?**

For machine translation, the "test set" is typically a **standard benchmark** (like WMT newstest2014) — not a random split from training data. We evaluate with BLEU score on the validation set during training to monitor progress.

**Why `seed`?** Without a fixed seed, every run gets a different random split. With `seed=42`, the same pairs always go to train vs val — making experiments reproducible.

---

# `num_workers` — Why 0 on macOS

```python
# In both base.yaml and tiny.yaml:
num_workers: 0    # Data loading workers — 0 = main process only (safe for macOS/MPS)
```

## What Workers Do

With `num_workers > 0`, PyTorch prepares the **next batch in parallel** while the GPU trains on the current batch:

```
Without workers (num_workers=0):
  load batch → train → load batch → train → ...
  [====load====][====train====][====load====][====train====]

With workers (num_workers=4):
  [====train====][====train====][====train====]
  [==load next==][==load next==][==load next==]    ← happens in parallel
```

The idea: if data loading is slow, workers can pre-fetch batches so the GPU never waits.

## Why macOS Is Different

In multiprocessing, there are different ways to create new processes:

| Method | How it works | Used on |
|---|---|---|
| **fork** (fast) | Copies the parent process's memory | Linux |
| **spawn** (slow) | Starts a completely new Python interpreter from scratch | macOS, Windows |

PyTorch uses **spawn** on macOS. Every time a worker is created:

1. A **new Python interpreter** starts
2. Your script and all modules are **re-imported**
3. Dataset objects are **reconstructed**
4. Data must be **serialized (pickled)** and passed between processes

This has high startup + communication cost.

## When Workers Help vs Hurt

The key inequality:

```
work per batch  vs  spawn overhead
```

**Case 1: Light work (our case — tokenized text):**

Our data is already tokenized integers in memory (`self.pairs`). Loading a batch is just indexing a list — microseconds.

```
work per batch ≈ 0.1 ms
spawn overhead ≈ 20 ms
```

More workers → more overhead → **slower** training.

**Case 2: Heavy work (e.g., image augmentation):**

Loading large images from disk, resizing, cropping, normalizing, random augmentations:

```
work per batch ≈ 200 ms
spawn overhead ≈ 20 ms
```

Workers process batches in parallel. The 20ms overhead is negligible compared to 200ms of real work.

**Practical intuition:** If the task is "bring me a glass of water" — doing it yourself is faster than hiring 4 people, explaining the task, and coordinating them. If the task is "build a wall" — the coordination overhead is worth it.

**Rule of thumb:**

| Situation | `num_workers` |
|---|---|
| Small dataset / pre-tokenized / text | `0` |
| Large images / heavy augmentations | `2-4` (Linux) or `0-2` (macOS) |
| GPU is waiting on data (loading is the bottleneck) | Increase workers |

---

# Data Quality — Noisy Pairs

The Samanantar dataset contains noisy translation pairs that `is_valid_pair` **cannot** catch:

```
src: "The Supreme Court ruled in favor of the petitioner."
tgt: "সুপ্রিম কোর্ট আবেদনকারীর পক্ষে রায় দিয়েছে।"    ← correct translation

src: "International trade agreements require bilateral consent."
tgt: "আন্তর্জাতিক বাণিজ্য চুক্তি প্রয়োজন।"            ← partial/wrong translation
```

The second pair passes all filters (non-empty, has Bengali, reasonable length ratio) but the translation is incomplete. Our simple filters catch structural problems, not semantic ones.

Fixing this would require:
- A trained translation quality estimation model
- Back-translation filtering (translate back to English, compare)
- Manual human review

These are beyond the scope of this replication. The model will learn despite some noise — neural networks are robust to moderate label noise. The paper doesn't mention data cleaning either.

---

# Vocab Size — How 16K Subwords Cover 1M Sentences

We have 500K pairs = 500K English + 500K Bengali = **1M sentences** fed into SentencePiece. It picks the best 16K subwords across both languages.

```
tokenizer/sp_train_text.txt (the temp file):
  I love AI                    ← English sentence 1
  আমি AI ভালোবাসি              ← Bengali sentence 1
  The court ruled              ← English sentence 2
  আদালত রায় দিয়েছে             ← Bengali sentence 2
  ...
  (1M lines total — 500K English + 500K Bengali)
```

16K is more than enough because BPE stores **subwords**, not whole words. It can represent any word by combining pieces:

```
16K subwords can represent unlimited words:

"understanding"  → [▁under, stand, ing]       ← 3 known subwords
"misunderstand"  → [▁mis, under, stand]        ← 3 known subwords
"understandable" → [▁under, stand, able]       ← 3 known subwords

Three different words, same subword pieces reused.
```

**What's in the 16K — subwords or full words?** Depends on frequency. The 16K vocab stores **subwords**, but a very common full word can also get its own entry:

```
If "understanding" appears rarely:
  Vocab has: ▁under (idx 234), stand (idx 567), ing (idx 89)
  sp.encode("understanding") → [234, 567, 89]     ← 3 tokens

If "understanding" appears very frequently (thousands of times):
  Vocab has: ▁understanding (idx 1042)              ← got its own slot
  sp.encode("understanding") → [1042]               ← 1 token
```

SentencePiece decides this automatically during training — frequent sequences get merged into single tokens, rare ones stay as pieces.

**What is `▁` (the prefix symbol)?** That's NOT an underscore `_`. It's SentencePiece's **space marker** (Unicode `U+2581`, "Lower One Eighth Block"). It marks where a space was in the original text:

```
Original text:  "I love AI"
                 ↑ ↑    ↑
               space  space

SentencePiece tokenizes:
  ["▁I", "▁love", "▁AI"]
    ↑      ↑       ↑
    ▁ means "there was a space before me"
```

This is critical for decoding — `▁` tells SentencePiece how to reconstruct spacing:

```
Tokens:  ["▁I", "▁love", "▁AI"]
Decode:  "I love AI"              ← each ▁ becomes a space

Tokens:  ["▁under", "stand", "ing"]
Decode:  "understanding"          ← only first piece has ▁
                                     "stand" and "ing" have NO ▁, so they
                                     merge directly with "under" (no space)
```

Without `▁`, the decoder wouldn't know if `stand` is:
- Part of `understanding` (no space) → `["▁under", "stand", "ing"]`
- A separate word `stand` (space) → `["▁under", "▁stand", "▁ing"]`

How SentencePiece fills the 16K slots — BPE starts with individual characters and repeatedly merges the most frequent pair:

```
Step 0: All individual characters — a, b, c, ..., আ, মি, ...   (~500 chars)
Step 1: Most frequent pair "t"+"h" → merge into "th"
Step 2: Most frequent pair "th"+"e" → merge into "the"
Step 3: Most frequent pair "আ"+"মি" → merge into "আমি"
...
Step N: Stop when vocab reaches 16,000
```

Common words get their own token. Rare words get split:

```
"the"                    → [▁the]                             ← 1 token (common)
"আমি"                    → [▁আমি]                             ← 1 token (common)
"electroencephalography" → [▁electro, enceph, alo, graphy]   ← 4 tokens (rare)
```

## Shared Vocab — Same Table, Different IDs

"Shared vocabulary" means both languages share the **same table** — NOT the same IDs. English and Bengali words get **different IDs** in **one** dictionary:

```
Shared vocabulary (16K entries):
  Index 0:      <pad>
  Index 1:      <sos>
  Index 2:      <eos>
  Index 3:      <unk>
  Index 4:      ▁the          ← English
  Index 5:      ▁আমি          ← Bengali
  Index 6:      ▁of           ← English
  Index 7:      ▁করে          ← Bengali
  Index 8:      ▁AI           ← shared! (appears in both languages)
  Index 9:      tion          ← English subword
  Index 10:     ▁বাংলা        ← Bengali
  ...
  Index 15999:  (last token)
```

The split depends on the data — roughly:

```
16,000 total slots:
  4 special tokens:    <pad>, <sos>, <eos>, <unk>
  ~7-8K English subwords
  ~7-8K Bengali subwords
  + some shared tokens (numbers, "AI", "DNA", punctuation)
```

Why share? Without sharing:

```
Without shared vocab (two separate tokenizers):
  English tokenizer: "AI" → ID 50
  Bengali tokenizer: "AI" → ID 73     ← different IDs for same word!

  Encoder uses English vocab (16K)
  Decoder uses Bengali vocab (16K)
  Total: 32K entries, 2 embedding tables

With shared vocab (one tokenizer):
  Shared tokenizer:  "AI" → ID 8      ← same ID everywhere

  Encoder and decoder use same vocab (16K)
  Total: 16K entries, 1 embedding table
```

## Why 16K and Not 37K?

Paper used 37K vocab for 4.5M pairs (9M sentences). We have 500K pairs (1M sentences) — much less data.

A larger vocab with less data means:
- Many tokens appear only a few times → model can't learn good embeddings for them
- Wastes embedding parameters on rare tokens

Rule of thumb: smaller data → smaller vocab.

---

# The Full Journey — Raw Text to Model Input

Tracing one sentence through the entire pipeline:

```
═══════════════════════════════════════════════════════════════════════════════
STEP 1 — TOKENIZE (vocab_size = 16K)
═══════════════════════════════════════════════════════════════════════════════

Raw text: "I love AI."

sp.encode("I love AI.") → [47, 823, 156, 9]      ← look up in 16K vocab
                            ↑    ↑     ↑   ↑
                           "▁I" "▁love" "▁AI" "."

add <sos>/<eos> → [1, 47, 823, 156, 9, 2]         ← 6 tokens (< 128, no truncation)
                   ↑                     ↑
                 <sos>                 <eos>


═══════════════════════════════════════════════════════════════════════════════
STEP 2 — BATCHING (max_tokens = 8K)
═══════════════════════════════════════════════════════════════════════════════

TokenBatchSampler packs ~363 sentences (longest=22) into one batch.
collate_fn pads all to longest:

  src_padded: (363, 22)                    tgt_padded: (363, 22)
  ┌──────────────────────────┐             ┌──────────────────────────┐
  │[1, 47,823,156, 9, 2,0..] │             │[1,205, 44,823,87, 2,0..] │
  │[1, 33, 71,  2, 0, 0,0..] │             │[1, 88,  2,  0, 0, 0,0..] │
  │...361 more rows...       │             │...361 more rows...       │
  └──────────────────────────┘             └──────────────────────────┘
          │                                        │
          │ ENCODER PATH                           │ DECODER PATH
          ▼                                        ▼

═══════════════════════════════════════════════════════════════════════════════
STEP 3 — EMBEDDING (shared 16K × 256 table)
═══════════════════════════════════════════════════════════════════════════════

The embedding table has 16,000 rows, each 256D.
Each token ID → look up that row → get a 256D vector.

  Embedding table (16000 × 256):
    Row 0   (<pad>)  → [0.01, -0.03, 0.07, ..., 0.02]
    Row 1   (<sos>)  → [0.12,  0.05,-0.08, ..., 0.11]
    Row 47  (▁I)     → [0.15, -0.22, 0.08, ..., 0.31]
    Row 205 (▁আমি)   → [0.33,  0.08, 0.14, ..., 0.27]
    Row 823 (▁love)  → [0.42,  0.11,-0.19, ..., 0.07]
    ...15,994 more rows...


  HOW THE SHAPE CHANGES — (363, 22) → (363, 22, 256):

  Each integer ID gets REPLACED by its 256D row from the table.
  This is NOT matrix multiplication — it's a table lookup.

  Before — one sentence from the batch:
  ┌──┬───┬───┬───┬──┬──┬──┬─────┬──┐
  │1 │47 │823│156│ 9│ 2│ 0│ ... │ 0│    ← 22 integers
  └──┴───┴───┴───┴──┴──┴──┴─────┴──┘
    shape: (22,)

  After — each integer swapped for 256 floats:
  ┌──────────────────────────────────────────┐
  │ ID 1   → [0.12, 0.05, -0.08, ..., 0.11]  │  ← 256 floats
  │ ID 47  → [0.15,-0.22,  0.08, ..., 0.31]  │  ← 256 floats
  │ ID 823 → [0.42, 0.11, -0.19, ..., 0.07]  │  ← 256 floats
  │ ID 156 → [0.09,-0.15,  0.33, ..., 0.18]  │  ← 256 floats
  │ ID 9   → [-0.11,0.27,  0.06, ...,-0.14]  │  ← 256 floats
  │ ID 2   → [-0.04,0.09,  0.03, ...,-0.06]  │  ← 256 floats
  │ ID 0   → [0.01,-0.03,  0.07, ..., 0.02]  │  ← 256 floats (pad)
  │ ...22 rows total...                      │
  └──────────────────────────────────────────┘
    shape: (22, 256)

  For ALL 363 sentences in the batch:
  (363, 22)  →  (363, 22, 256)
       ↑              ↑
    22 IDs per     22 vectors of 256D
    sentence       per sentence

  Under the hood, PyTorch does this with a single indexing operation:
    output = embedding_table[token_ids]    ← fancy indexing, no multiplication


  ENCODER — src token IDs:              DECODER — tgt token IDs:
  [1, 47, 823, 156, 9, 2, 0...]        [1, 205, 44, 823, 87, 2, 0...]
   ↓   ↓    ↓    ↓  ↓  ↓                ↓    ↓   ↓    ↓   ↓  ↓
  Row1 Row47 Row823...                  Row1 Row205 Row44...

  src_embedded: (363, 22, 256)          tgt_embedded: (363, 22, 256)
  ┌──────────────────────────┐          ┌────────────────────────────┐
  │<sos>: [0.12, 0.05, ...]  │          │<sos>: [0.12, 0.05, ...]    │
  │"I":   [0.15,-0.22, ...]  │          │"আমি": [0.33, 0.08, ...]    │
  │"love":[0.42, 0.11, ...]  │          │"AI":  [0.09,-0.15, ...]    │
  │"AI":  [0.09,-0.15, ...]  │          │"ভালো": [0.42, 0.11, ...]   │
  │".":   [-0.11,0.27, ...]  │          │"।":   [-0.07,0.19, ...]    │
  │<eos>: [-0.04,0.09, ...]  │          │<eos>: [-0.04,0.09, ...]    │
  │<pad>: [0.01,-0.03, ...]  │          │<pad>: [0.01,-0.03, ...]    │
  │...padded to 22           │          │...padded to 22             │
  └──────────────────────────┘          └────────────────────────────┘

  22 token IDs in → 22 vectors out. Embedding just adds the 256D dimension.
  If longest in batch was 87, shape would be (363, 87, 256). It's NOT fixed at 128.

           ↓ × sqrt(256) = × 16                  ↓ × sqrt(256) = × 16
       (scale embeddings — Section 3.4)        (same scaling)


═══════════════════════════════════════════════════════════════════════════════
STEP 4 — POSITIONAL ENCODING (same PE table — 5000 × 256)
═══════════════════════════════════════════════════════════════════════════════

  PE table (5000, 256) — pre-computed sin/cos, FIXED, never learned:
    Position 0:    [sin(0), cos(0), sin(0), cos(0), ...]
    Position 1:    [sin(1/10000^0), cos(1/10000^0), ...]
    Position 2:    [sin(2/10000^0), cos(2/10000^0), ...]
    ...
    Position 21:   [sin(21/...), cos(21/...), ...]
    Position 22-4999: exist but UNUSED for this batch

  We slice PE[0:22] → (22, 256)          We slice PE[0:22] → (22, 256)
  Same PE for both sides — position 3 means "3rd token" regardless of language.

  ENCODER:                                DECODER:
  src_embedded + PE[0:22]                 tgt_embedded + PE[0:22]

  ┌──────────────────────────┐            ┌──────────────────────────┐
  │<sos>: [0.12+0.00, ...]   │            │<sos>: [0.12+0.00, ...]   │
  │"I":   [0.15+0.84, ...]   │            │"আমি": [0.33+0.84, ...]   │
  │"love":[0.42+0.91, ...]   │            │"AI":  [0.09+0.91, ...]   │
  │"AI":  [0.09+0.14, ...]   │            │"ভালো": [0.42+0.14, ...]  │
  │".":   [-0.11-0.76, ...]  │            │"।":   [-0.07-0.76, ...]  │
  │<eos>: [-0.04-0.96, ...]  │            │<eos>: [-0.04-0.96, ...]  │
  │<pad>: [0.01+0.66, ...]   │            │<pad>: [0.01+0.66, ...]   │
  └──────────────────────────┘            └──────────────────────────┘

  Now "love" at position 2 and "love" at position 5 would have
  DIFFERENT vectors — same embedding + different PE = model knows WHERE each word is.

  Shape: (363, 22, 256) for both         Shape: (363, 22, 256) for both


═══════════════════════════════════════════════════════════════════════════════
STEP 5 — INTO THE MODEL
═══════════════════════════════════════════════════════════════════════════════

  src_final: (363, 22, 256)              tgt_final: (363, 22, 256)
       │                                       │
       ▼                                       ▼
  ┌───────────┐                            ┌───────────┐
  │ ENCODER   │                            │ DECODER   │
  │ 4 layers  │                            │ 4 layers  │
  │           │                            │           │
  │ Self-Attn │                            │ Self-Attn │ ← causal mask (can't see future)
  │ + FFN     │                            │ Cross-Attn│ ← looks at encoder output
  │           │                            │ + FFN     │
  └─────┬─────┘                            └─────┬─────┘
        │                                        │
        │ encoder_output: (363, 22, 256)         │
        │                                        │
        └────────→ cross-attention ──────────────┘
                   (decoder Q attends to encoder K,V)
                              │
                              ▼
                   Output Projection (256 → 16K)
                   logits: (363, 22, 16000)
                              │
                              ▼
                   Each position: scores for all 16K vocab tokens
                   argmax → predicted token IDs
                   sp.decode() → readable Bengali text
```

## Batch Shape Through Each Step

For a batch with 363 sentences, longest = 22:

| Step | Encoder (src) shape | Decoder (tgt) shape |
|---|---|---|
| Token IDs (after collate) | `(363, 22)` | `(363, 22)` |
| Embedding lookup | `(363, 22, 256)` | `(363, 22, 256)` |
| × sqrt(d_model) | `(363, 22, 256)` | `(363, 22, 256)` |
| + Positional Encoding | `(363, 22, 256)` | `(363, 22, 256)` |
| After encoder/decoder | `(363, 22, 256)` | `(363, 22, 256)` |
| Output projection | — | `(363, 22, 16000)` |

The `22` is **not** `max_seq_len=128`. It's the longest sentence **in this batch**. Each batch can have a different seq_len dimension:

```
Batch 1 (short sentences): (363, 22, 256)
Batch 2 (medium sentences): (80, 65, 256)
Batch 3 (long sentences):   (20, 118, 256)
```

## What's Shared vs Different Between Encoder and Decoder

**Three things shared:**

| What | Size | Why shared |
|---|---|---|
| Embedding table | (16000, 256) | Shared vocab — same tokenizer for both languages |
| PE table | (5000, 256) | Position 3 means "3rd token" regardless of language |
| Pad token | ID 0 | Same padding convention for both sides |

**Three things different:**

| | Encoder | Decoder |
|---|---|---|
| Input | src (English) | tgt (Bengali) |
| Self-attention mask | No causal mask — sees all positions | Causal mask — can't see future tokens |
| Cross-attention | None | Attends to encoder output (Q from decoder, K/V from encoder) |

## Why 5K PE Rows When max_seq_len = 128?

The PE table is 5000 × 256. We only ever use rows 0-127 (at most). The other 4,872 rows sit unused.

Memory cost:

```
PE table:    5000 × 256 = 1,280,000 floats × 4 bytes = ~5 MB  (once, no gradients)
Embedding:  16000 × 256 = 4,096,000 floats × 4 bytes = ~16 MB (+ optimizer = ~48 MB)
Attention:  363 × 8 × 128 × 128                       = ~150 MB per layer (activations)
```

5 MB is noise compared to what actually eats memory.

**Why not just set max_len=128?** You could. But then if you change `max_seq_len` to 200 for an experiment, PE crashes because it only has 128 rows. With 5000, you never have to think about it — change `max_seq_len` to anything up to 5000 and it just works.

---

# Integration — How `train.py` Wires Everything Together

`scripts/train.py` is the only caller of these functions. It loads the YAML config, then invokes each data_utils function in order:

```python
# --- 1. Load config (base.yaml or --config configs/tiny.yaml) ---
with open(args.config, "r") as f:
    config = yaml.safe_load(f)

# --- 2. Load HuggingFace dataset ONCE — reused by tokenizer + dataloaders ---
raw_dataset = load_dataset(
    path=config["data"]["dataset"],      # "ai4bharat/samanantar"
    name=config["data"]["tgt_lang"],     # "bn"
    split="train",
)
if config["data"]["max_rows"]:
    raw_dataset = raw_dataset.select(
        range(min(config["data"]["max_rows"], len(raw_dataset)))
    )                                    # cap to 500K (base) or 1000 (tiny)

# --- 3. Tokenizer: train if missing, else reuse saved .model file ---
tokenizer_path = config["paths"]["tokenizer_path"]    # e.g. "transformer/tokenizer/base/sp.model"
model_prefix = tokenizer_path.removesuffix(".model")  # e.g. "transformer/tokenizer/base/sp"
if os.path.exists(tokenizer_path):
    sp = load_tokenizer(tokenizer_path)
else:
    sp = train_tokenizer(
        dataset=raw_dataset,
        vocab_size=config["data"]["vocab_size"],       # 16000 (base) or 4000 (tiny)
        pad_id=config["tokens"]["pad_idx"],            # 0
        sos_id=config["tokens"]["sos_idx"],            # 1
        eos_id=config["tokens"]["eos_idx"],            # 2
        unk_id=config["tokens"]["unk_idx"],            # 3
        model_prefix=model_prefix,
    )

# --- 4. Build train + val DataLoaders from the same raw_dataset ---
train_loader, val_loader = create_dataloaders(
    raw_dataset=raw_dataset,
    sp=sp,
    max_seq_len=config["training"]["max_seq_len"],            # 128 (base) or 64 (tiny)
    max_tokens=config["training"]["max_tokens_per_batch"],    # 8000 (base) or 2000 (tiny)
    pad_idx=config["tokens"]["pad_idx"],                      # 0
    seed=config["seed"],                                      # 42
    num_workers=config["data"]["num_workers"],                # 0
    val_split=config["training"]["val_split"],                # 0.1
    filter_max_ratio=config["data"]["filter_max_ratio"],      # 5.0
    filter_min_words=config["data"]["filter_min_words"],      # 1
)
```

```
          YAML config
               │
               ▼
     ┌─────────────────────┐
     │ yaml.safe_load      │
     └─────────┬───────────┘
               │
               ▼
     ┌─────────────────────┐
     │ load_dataset + cap  │ ← HuggingFace raw_dataset (loaded ONCE)
     └─────────┬───────────┘
               │ raw_dataset
               ├─────────────────────────────────────┐
               ▼                                     │
     ┌──────────────────────────────┐                │
     │ Tokenizer branch             │                │
     │  ──────────────────────      │                │
     │  train_tokenizer  ← first run│                │
     │         OR                   │                │
     │  load_tokenizer   ← reruns   │                │
     └──────────┬───────────────────┘                │
                │ sp                                 │ raw_dataset
                ▼                                    ▼
     ┌──────────────────────────────────────────────────┐
     │ create_dataloaders (uses sp + raw_dataset)       │ ← always runs the same way
     └──────────┬───────────────────────────────────────┘   (not affected by first/rerun)
                │
                ▼
       (train_loader, val_loader)
                │
                ▼
          training loop
```

**Two design decisions worth calling out:**

1. **`raw_dataset` is loaded once and passed in** — `train_tokenizer` needs it for BPE training, `create_dataloaders` needs it to build the Dataset. Loading the HF dataset twice would download/prepare 500K pairs twice. That's why neither function calls `load_dataset` internally — the caller owns it.

2. **Tokenizer is cached on disk** — `.model` lives at the path in `paths.tokenizer_path`. First run trains + saves. Subsequent runs skip the BPE training step entirely (`os.path.exists` check). Delete the `.model` file to force a retrain (e.g. after changing `vocab_size` or `max_rows`).

**Running:**

```bash
python -m transformer.scripts.train                           # uses configs/base.yaml by default
python -m transformer.scripts.train --config configs/tiny.yaml   # debug run
python -m transformer.scripts.train --resume checkpoints/last.pt # resume from checkpoint
```

---

# All the Numbers — Cheat Sheet

```
16K  = vocab_size           How many unique subwords in the dictionary
                            (16,000 entries for both English + Bengali)

256  = d_model              Embedding dimension — how rich each token's vector is
                            (each token becomes a 256-dimensional vector)

5K   = max_len (PE table)   Pre-computed positional encoding rows
                            (way more than needed — "set it and forget it")

128  = max_seq_len          Actual max sentence length after truncation
                            (any sentence > 128 tokens gets cut)

8K   = max_tokens/batch     Token budget per training batch
                            (TokenBatchSampler packs sentences until ~8K)

500K = max_rows             Pairs loaded from HuggingFace
                            (500K out of 8.5M total in Samanantar)

~448K = pairs after filter  Valid pairs that survive is_valid_pair + len > 2 check
                            (NOT fixed - depend on is_valid_pair)

450K = train pairs          90% of 500K (after train_test_split)
                            (NOT fixed - 90% 0f Data after is_valid_pair and train_test_split)

50K  = val pairs            10% of 500K
                            (NOT fixed - 10% 0f Data after is_valid_pair and train_test_split)

~363 = sentences/batch      Varies! Short sentences → more per batch, long → fewer
                            (NOT fixed — depends on sentence lengths in that batch)

22   = seq_len in batch     Varies! Longest sentence in THIS batch
                            (NOT max_seq_len=128 — collate_fn pads to actual longest)

4    = num_layers           Encoder and decoder each have 4 layers

8    = num_heads            Multi-head attention splits 256 into 8 heads of 32 each
```

**Same numbers, both presets side-by-side:**

| Number | `base.yaml` | `tiny.yaml` | Meaning |
|---|---|---|---|
| `vocab_size` | 16000 | 4000 | Subword dictionary size |
| `d_model` | 256 | 64 | Embedding / hidden dim |
| `max_len` (PE) | 5000 | 512 | Pre-computed PE rows |
| `max_seq_len` | 128 | 64 | Hard cap after truncation |
| `max_tokens_per_batch` | 8000 | 2000 | Token budget per batch |
| `max_rows` | 500000 | 1000 | HF rows loaded |
| `num_layers` | 4 | 2 | Encoder / decoder depth |
| `num_heads` | 8 | 4 | Attention heads |

`d_model`, `max_len`, `num_layers`, `num_heads` live in `config["model"]` and belong to the model modules — shown here only for completeness. The rest are the values `data_utils.py` actually consumes.

---

# References

### Paper

1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017 (Section 5.1 — Training Data and Batching)

### Dataset

2. [AI4Bharat Samanantar](https://huggingface.co/datasets/ai4bharat/samanantar) — Multilingual parallel corpus for Indian languages

### SentencePiece

3. [SentencePiece GitHub](https://github.com/google/sentencepiece) — Google's unsupervised text tokenizer
4. [SentencePiece Python Wrapper](https://github.com/google/sentencepiece/blob/master/python/README.md) — Python API reference

### num_workers

5. [Efficient Data Pipelines in PyTorch: Lessons from num_workers](https://medium.com/@allam.satyanarayana/efficient-data-pipelines-in-pytorch-lessons-from-num-workers-4d49eb6b384d) — macOS spawn overhead, benchmark results
6. [PyTorch DataLoader Documentation](https://pytorch.org/docs/stable/data.html#single-and-multi-process-data-loading) — Official docs on single vs multi-process loading

### Unicode

7. [Bengali Unicode Block (U+0980 – U+09FF)](https://www.unicode.org/charts/PDF/U0980.pdf) — Unicode Consortium
