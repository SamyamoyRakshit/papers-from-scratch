import os
import re
from typing import List, Tuple
import random

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from datasets import load_dataset
import sentencepiece as spm


# ============================================================
# Sanity Check — Filter Bad Translation Pairs
# ============================================================

# Pre-compiled regex for Bengali Unicode range (ঀ - ঽ)
# Source: https://www.unicode.org/charts/PDF/U0980.pdf
_bengali_pattern = re.compile(r'[\u0980-\u09FF]')


def is_valid_pair(src: str, tgt: str, max_ratio: float = 5.0, min_words: int = 1) -> bool:
    """
    Filter bad translation pairs.

    Checks:
        - Neither side is empty
        - Target contains at least one Bengali character
        - Neither side is shorter than min_words
        - Length ratio between src and tgt is not extreme

    Args:
        src (str): Source sentence (English).
        tgt (str): Target sentence (Bengali).
        max_ratio (float): Max allowed word count ratio. Default: 5.0.
        min_words (int): Minimum words per sentence. Default: 1.

    Returns:
        bool: True if the pair is valid.
    """
    if not src or not tgt:                                          # empty string check
        return False
    if not _bengali_pattern.search(tgt):                           # target must have Bengali text
        return False
    src_len = len(src.split())                                     # word count
    tgt_len = len(tgt.split())
    if src_len < min_words or tgt_len < min_words:                 # too short
        return False
    ratio = max(src_len, tgt_len) / max(min(src_len, tgt_len), 1) # length ratio (e.g. 6:1 = 6.0)
    if ratio > max_ratio:                                          # too imbalanced
        return False
    return True


# ============================================================
# 1. Train SentencePiece Tokenizer
# ============================================================

def train_tokenizer(
        dataset,
        vocab_size: int,
        pad_id: int,
        sos_id: int,
        eos_id: int,
        unk_id: int,
        model_prefix: str = "tokenizer/sp"
) -> spm.SentencePieceProcessor:
    """
    Train a shared SentencePiece BPE tokenizer on both source and target sentences.

    "Sentences were encoded using byte-pair encoding, which has a shared
    source-target vocabulary of about 37000 tokens." — Section 5.1

    Args:
        dataset: HuggingFace dataset with 'src' and 'tgt' fields.
        vocab_size (int): Vocabulary size. Paper: ~37000. We're using 16000.
        pad_id (int): Padding token index.
        sos_id (int): Start-of-sentence token index.
        eos_id (int): End-of-sentence token index.
        unk_id (int): Unknown token index.
        model_prefix (str): Path prefix for saving the model files.

    Returns:
        spm.SentencePieceProcessor: Trained tokenizer.
    """
    # Write all valid sentences to a temp file — SentencePiece reads from file, not memory
    temp_file = f"{model_prefix}_train_text.txt"
    # Creates all intermediate-level parent directories if they are missing (e.g. "a/b/c/sp" → creates "a/", "a/b/", "a/b/c/")
    os.makedirs(os.path.dirname(temp_file), exist_ok=True)

    with open(temp_file, "w", encoding="utf-8") as f:
        for example in dataset:
            src = example["src"].strip()     # keys "src"/"tgt" are from AI4Bharat dataset
            tgt = example["tgt"].strip()

            if not is_valid_pair(src, tgt):
                continue                     # skip bad pair, go to next example

            f.write(src + "\n")              # one sentence per line — SentencePiece expects this
            f.write(tgt + "\n")

    # Train BPE tokenizer — learns subword merges from the temp file
    spm.SentencePieceTrainer.train(
        input=temp_file,                     # text file to learn from
        model_prefix=model_prefix,           # output: {prefix}.model and {prefix}.vocab
        vocab_size=vocab_size,
        model_type="bpe",                    # byte-pair encoding (Section 5.1)
        pad_id=pad_id,                       # special token indices — must match config
        bos_id=sos_id,
        eos_id=eos_id,
        unk_id=unk_id,
        pad_piece="<pad>",                   # special token strings
        bos_piece="<sos>",
        eos_piece="<eos>",
        unk_piece="<unk>"
    )

    os.remove(temp_file)                     # clean up — no longer needed after training
    return load_tokenizer(f"{model_prefix}.model")


# ============================================================
# 2. Load Tokenizer
# ============================================================

def load_tokenizer(model_path: str) -> spm.SentencePieceProcessor:
    """
    Load a pre-trained SentencePiece model.

    Args:
        model_path (str): Path to the .model file.

    Returns:
        spm.SentencePieceProcessor: Loaded tokenizer.
    """
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    return sp


# ============================================================
# 3. Encode / Decode
# ============================================================

def encode(sp: spm.SentencePieceProcessor, text: str, max_seq_len: int) -> List[int]:
    """
    Tokenize text and add `<sos>` and `<eos>` tokens.

    E.g., "I love AI" → [`<sos>`, 14, 87, 3, `<eos>`] → [1, 14, 87, 3, 2]

    Args:
        sp: SentencePiece tokenizer.
        text (str): Raw text to encode.
        max_seq_len (int): Max sequence length (including `<sos>` and `<eos>`).

    Returns:
        List[int]: Token IDs with `<sos>` prepended and `<eos>` appended.
    """
    token_ids = sp.encode(text.strip())       # "I love AI" → [14, 87, 3]
    token_ids = token_ids[: max_seq_len - 2]  # truncate, leaving room for `<sos>` and `<eos>`
    return [sp.bos_id()] + token_ids + [sp.eos_id()]  # [1, 14, 87, 3, 2]


def decode(sp: spm.SentencePieceProcessor, token_ids: List[int]) -> str:
    """
    Decode token IDs back to text.

    Args:
        sp: SentencePiece tokenizer.
        token_ids (List[int]): Token IDs.

    Returns:
        str: Decoded text.
    """
    return sp.decode(token_ids)


# ============================================================
# 4. Dataset
# ============================================================

class TranslationDataset(Dataset):
    """
    PyTorch Dataset for parallel translation pairs.

    Each item is a tuple of (src_ids, tgt_ids) — both as token ID lists
    with `<sos>` and `<eos>` included.
    """
    def __init__(
            self,
            dataset,
            sp: spm.SentencePieceProcessor,
            max_seq_len: int
    ):
        self.pairs = []  # stores all valid (src_ids, tgt_ids) tuples
        skipped = 0      # count filtered pairs — so we know if data quality is bad

        for example in dataset:
            src = example["src"].strip()
            tgt = example["tgt"].strip()

            if not is_valid_pair(src, tgt):
                skipped += 1
                continue                        # skip bad pairs (same filter as tokenizer training)

            src_ids = encode(sp, src, max_seq_len)  # text → token IDs with `<sos>`/`<eos>`
            tgt_ids = encode(sp, tgt, max_seq_len)

            if len(src_ids) > 2 and len(tgt_ids) > 2:  # must have at least 1 real token (not just `<sos>`+`<eos>`)
                self.pairs.append((src_ids, tgt_ids))
            else:
                skipped += 1                     # empty after encoding — tokenizer produced no real tokens

        total = len(self.pairs) + skipped
        print(f"[TranslationDataset] Kept {len(self.pairs)} pairs ({len(self.pairs)/total*100:.1f}%), filtered {skipped} ({skipped/total*100:.1f}%)")

    def __len__(self):                                              # DataLoader calls this to know total pairs
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]: # DataLoader calls this with different idx values
        return self.pairs[idx]
    

# ============================================================
# 5. Token-Based Batch Sampler
# ============================================================

class TokenBatchSampler(Sampler):
    """
    Groups sentences into batches by total token count — Section 5.1:

    "Sentence pairs were batched together by approximate sequence length.
    Each training batch contained a set of sentence pairs containing
    approximately 25,000 source tokens and 25,000 target tokens."

    Steps:
        1. Sort sentence pairs by source length (groups similar lengths)
        2. Pack pairs into batches until token count reaches max_tokens
        3. Shuffle batches (not individual sentences)
    """
    def __init__(
            self,
            dataset: TranslationDataset,
            max_tokens: int,
            shuffle: bool = True
    ):
        self.dataset = dataset
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.batches = self._create_batches()

    def _create_batches(self) -> List[List[int]]:
        # Step 1: Sort indices by src length — groups similar lengths together, reduces padding
        indices = sorted(
            range(len(self.dataset)),
            key=lambda i: len(self.dataset.pairs[i][0])  # [0] = src_ids
        )

        batches = []
        current_batch = []
        max_len_in_batch = 0                                   # tracks true longest sequence in current batch

        # Step 2: Pack pairs into batches until token count reaches max_tokens
        for idx in indices:
            src_ids, tgt_ids = self.dataset.pairs[idx]
            pair_len = max(len(src_ids), len(tgt_ids))           # longest side determines padding
            new_max = max(max_len_in_batch, pair_len)            # true max if we add this pair
            would_be = (len(current_batch) + 1) * new_max        # exact tensor size after padding

            if would_be > self.max_tokens and current_batch:     # exceeds limit and batch not empty
                batches.append(current_batch)                     # save finished batch
                current_batch = []                                # start new batch
                max_len_in_batch = 0                              # reset for new batch

            current_batch.append(idx)
            max_len_in_batch = max(max_len_in_batch, pair_len)   # update actual max

        if current_batch:                                        # don't lose the last batch
            batches.append(current_batch)

        return batches

    def __iter__(self):
        # Step 3: Shuffle batch order each epoch (not sentences within batches)
        if self.shuffle:
            random.shuffle(self.batches)
        for batch in self.batches:
            yield batch                   # give one batch of indices at a time to DataLoader

    def __len__(self):
        return len(self.batches)          # number of batches, not number of sentences
    

# ============================================================
# 6. Collate Function & DataLoader
# ============================================================

def collate_fn(batch: List[Tuple[List[int], List[int]]], pad_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pad all sequences in the batch to the same length.

    Args:
        batch: List of (src_ids, tgt_ids) pairs.
        pad_idx (int): Padding token index.

    Returns:
        (src_tensor, tgt_tensor): Both shape (batch_size, max_seq_len_in_batch)
    """
    # Convert list of token IDs → list of tensors
    src_seqs = [torch.tensor(pair[0], dtype=torch.long) for pair in batch]  # pair[0] = src_ids
    tgt_seqs = [torch.tensor(pair[1], dtype=torch.long) for pair in batch]  # pair[1] = tgt_ids

    # Pad all sequences to the longest in this batch — creates a rectangular tensor
    # batch_first=True → shape (batch_size, seq_len), not (seq_len, batch_size)
    src_padded = torch.nn.utils.rnn.pad_sequence(sequences=src_seqs, batch_first=True, padding_value=pad_idx)
    tgt_padded = torch.nn.utils.rnn.pad_sequence(sequences=tgt_seqs, batch_first=True, padding_value=pad_idx)

    return src_padded, tgt_padded


def create_dataloaders(
        dataset_name: str,
        tgt_lang: str,
        max_rows: int,
        sp: spm.SentencePieceProcessor,
        max_seq_len: int,
        max_tokens: int,
        pad_idx: int,
        seed: int,
        num_workers: int,
        val_split: float = 0.1
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation DataLoaders.

    Args:
        dataset_name (str): HuggingFace dataset name.
        tgt_lang (str): Target language code (e.g. "bn"). Used as dataset subset.
        max_rows (int): Maximum number of rows to load.
        sp: Trained SentencePiece tokenizer.
        max_seq_len (int): Max sequence length.
        max_tokens (int): Max tokens per batch.
        pad_idx (int): Padding token index.
        seed (int): Random seed for reproducible train/val splits.
        num_workers (int): Number of data loading workers. 0 = main process only.
        val_split (float): Fraction of data for validation. Default: 0.1.

    Returns:
        (train_loader, val_loader): DataLoader tuple.
    """
    # Load from HuggingFace — e.g. load_dataset("ai4bharat/samanantar", "bn", split="train")
    raw_dataset = load_dataset(path=dataset_name, name=tgt_lang, split="train")

    # Cap dataset size — e.g. 500K out of 8.5M pairs for practical training time on M1
    if max_rows:
        raw_dataset = raw_dataset.select(range(min(max_rows, len(raw_dataset))))

    # Split into train (90%) and val (10%) — seed ensures same split every run
    split = raw_dataset.train_test_split(test_size=val_split, seed=seed)
    train_raw = split["train"]       # HuggingFace only creates "train" and "test" keys
    val_raw = split["test"]          # we use "test" as our validation set

    # Tokenize all pairs — filters bad pairs, encodes text → token IDs with `<sos>`/`<eos>`
    train_dataset = TranslationDataset(train_raw, sp, max_seq_len)
    val_dataset = TranslationDataset(val_raw, sp, max_seq_len)

    # Token-based batching (Section 5.1) — groups by ~8K tokens per batch, not fixed sentence count
    train_sampler = TokenBatchSampler(train_dataset, max_tokens, shuffle=True)
    val_sampler = TokenBatchSampler(val_dataset, max_tokens, shuffle=False)

    # batch_sampler replaces batch_size — they're mutually exclusive in DataLoader
    # collate_fn pads variable-length sequences to longest in batch → rectangular tensors
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_sampler=train_sampler,
        collate_fn=lambda batch: collate_fn(batch, pad_idx),
        num_workers=num_workers
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_sampler=val_sampler,
        collate_fn=lambda batch: collate_fn(batch, pad_idx),
        num_workers=num_workers
    )

    return train_loader, val_loader