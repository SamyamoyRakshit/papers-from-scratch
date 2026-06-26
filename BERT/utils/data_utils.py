import logging
import os
import re
from functools import partial

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tokenizers import BertWordPieceTokenizer

from .nsp import build_nsp_example
from .masking import mask_tokens

logger = logging.getLogger(__name__)

# Special tokens — the ORDER fixes their ids: [PAD]=0, [UNK]=1, [CLS]=2, [SEP]=3, [MASK]=4.
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


# ============================================================
# 1. Train / Load WordPiece Tokenizer
# ============================================================
def train_tokenizer(corpus_files, vocab_size: int, save_dir: str, lowercase: bool = False):
    """
    Train a WordPiece tokenizer (Devlin et al. 2019, §3 — "We use WordPiece embeddings (Wu et al.,
    2016) with a 30,000 token vocabulary.").

    BERT uses WordPiece, not BPE (that's the transformer's SentencePiece). The
    special-token ORDER above is what pins [CLS]=2 / [SEP]=3 / [MASK]=4 — nsp.py
    and masking.py are handed these ids, they don't hard-code them.

    Args:
        corpus_files: path (or list of paths) of raw .txt files to learn from.
        vocab_size: target vocab. Paper: 30000. Scale down for the temple corpus.
        save_dir: where `vocab.txt` is written (repo.txt reserves `tokenizer/`).
        lowercase: False keeps case. Bengali is caseless; English names keep case.

    Returns:
        BertWordPieceTokenizer — trained, ready to encode.
    """
    if isinstance(corpus_files, str):
        corpus_files = [corpus_files]

    tokenizer = BertWordPieceTokenizer(lowercase=lowercase)
    tokenizer.train(files=corpus_files, vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS)

    os.makedirs(save_dir, exist_ok=True)
    tokenizer.save_model(save_dir)          # writes {save_dir}/vocab.txt
    return tokenizer


def load_tokenizer(vocab_path: str, lowercase: bool = False):
    """Load a trained tokenizer from its `vocab.txt`."""
    return BertWordPieceTokenizer(vocab_path, lowercase=lowercase)


# ============================================================
# 2. Raw corpus  →  all_documents  (the 3-level nest nsp.py wants)
# ============================================================

# Sentence boundary: punctuation (Bengali purnawchchhed ।, or . ! ?) followed by space.
_SENT_BOUNDARY = re.compile(r"(?<=[।.!?])\s+")


def _split_sentences(block:str):
    """One document block → its sentences (whitespace flattened first)."""
    block = block.replace("\n", " ").strip()
    return [s for s in _SENT_BOUNDARY.split(block) if s.strip()]


def build_documents(corpus_path: str, tokenizer):
    """
    Read the corpus and tokenize it into the nest nsp.py consumes:

        all_documents  list[list[list[int]]]   the corpus
          └ document   list[list[int]]         one article, etc.
              └ sentence  list[int]            WordPiece ids (NO [CLS]/[SEP] —
                                               nsp.py adds those when it packs)

    A BLANK LINE separates documents (paper §3.1 needs a *document-level* corpus,
    not shuffled sentences — NSP's "real next sentence" is meaningless otherwise).

    Args:
        corpus_path: path to the raw .txt corpus.
        tokenizer: trained BertWordPieceTokenizer.

    Returns:
        all_documents — ready to hand to build_nsp_example.
    """
    with open(corpus_path, encoding="utf-8") as f:
        raw = f.read()

    all_documents = []
    for block in re.split(r"\n\s*\n", raw.strip()):     # blank line(s) = document boundary
        document = [
            tokenizer.encode(s, add_special_tokens=False).ids
            for s in _split_sentences(block)
        ]
        document = [ids for ids in document if ids]      # drop sentences that tokenized to nothing
        if document:
            all_documents.append(document)
    
    logger.info(f"Built {len(all_documents)} documents from {corpus_path}")
    return all_documents


# ============================================================
# 3. Truncation (deferred out of nsp.py — it lives here)
# ============================================================

def _truncate(token_ids, token_type_ids, max_seq_len: int):
    """
    Trim a packed [CLS] A [SEP] B [SEP] down to max_seq_len.

    Pops one token at a time from the LONGER segment (Google's truncate_seq_pair),
    always the content token just before that segment's [SEP], so the
    [CLS]/[SEP] skeleton — and the token_type_ids alignment — stay intact.
    """
    while len(token_ids) > max_seq_len:
        seg_a_len = token_type_ids.count(0)             # [CLS] A [SEP]
        seg_b_len = token_type_ids.count(1)             # B [SEP]
        if seg_a_len >= seg_b_len:
            drop = seg_a_len - 2                        # last content token of A (before A's [SEP])
        else:
            drop = len(token_ids) - 2                   # last content token of B (before B's [SEP])
        del token_ids[drop]
        del token_type_ids[drop]
    return token_ids, token_type_ids


# ============================================================
# 4. Dataset — NSP pairs built once, MLM masked fresh each access
# ============================================================
class BERTPretrainingDataset(Dataset):
    """
    Turns a corpus into (masked input, MLM labels, segment ids, NSP label) examples.

    NSP pairing is built ONCE in __init__ (static — the IsNext/NotNext choice is
    fixed per example). MLM masking happens in __getitem__, so every fetch re-rolls
    a FRESH 80/10/10 mask over the same pair — this is "dynamic masking".

    Contrast with original BERT: it masks in a SEPARATE preprocessing pass
    (create_masked_lm_predictions in create_pretraining_data.py), writes the masked
    examples to disk, and run_pretraining.py just reads them — so each example's mask
    is frozen for the whole run. To add variety that script writes every document
    dupe_factor=10 times with different masks (the "10×"; default in Google's code,
    described in RoBERTa §4.1 — not a number stated in the BERT paper). We need none
    of that: re-masking in __getitem__ gives unlimited variety with no stored copies.

    Concretely, the SAME example sees a new mask each time it's fetched:

        epoch 1, fetch this example → mask_tokens() rolls a NEW random mask
        epoch 2, fetch this example → rolls ANOTHER new mask
        ...

    So we get unlimited mask variety for free — no 10 duplicate copies of the corpus
    to store. RoBERTa §4.1 measured exactly this static-vs-dynamic choice and dynamic
    won, which is why we mask here in __getitem__ rather than once up front.
    """
    def __init__(self, all_documents, tokenizer, max_seq_len: int, mlm_probability: float = 0.15):
        self.vocab_size = tokenizer.get_vocab_size()
        self.mlm_probability = mlm_probability

        self.pad_id = tokenizer.token_to_id("[PAD]")
        self.cls_id = tokenizer.token_to_id("[CLS]")
        self.sep_id = tokenizer.token_to_id("[SEP]")
        self.mask_id = tokenizer.token_to_id("[MASK]")
        self.special_token_ids = [self.pad_id, self.cls_id, self.sep_id]

        # THE CALLER LOOP — every sentence in every document becomes one A.
        # a_index is passed straight in (not searched for) — see nsp.py's docstring.
        self.examples = []
        for document in all_documents:
            for a_index in range(len(document)):
                token_ids, token_type_ids, nsp_label = build_nsp_example(
                    a_index, document, all_documents, self.cls_id, self.sep_id
                )
                token_ids, token_type_ids = _truncate(token_ids, token_type_ids, max_seq_len)
                self.examples.append((token_ids, token_type_ids, nsp_label))

        logger.info(f"Built {len(self.examples)} pre-training examples")

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        token_ids, token_type_ids, nsp_label = self.examples[idx]

        masked_ids, mlm_labels = mask_tokens(
            token_ids=torch.tensor(token_ids, dtype=torch.long),
            vocab_size=self.vocab_size,
            mask_token_id=self.mask_id,
            special_token_ids=self.special_token_ids,
            mlm_probability=self.mlm_probability
        )
        return {
            "input_ids": masked_ids,                                           # (S,) — corrupted
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),  # (S,) — 0/1 segments
            "mlm_labels": mlm_labels,                                          # (S,) — orig id at masked, -100 else
            "nsp_label": torch.tensor(nsp_label, dtype=torch.long)             # () — 0 IsNext / 1 NotNext
        }
    

# ============================================================
# 5. Collate + DataLoader — pad ragged examples to (B, S)
# ============================================================

def collate_fn(batch, pad_id: int, ignore_index: int = -100):
    """
    Pad a list of examples to the batch's longest sequence → rectangular tensors.

    Each field pads with a DIFFERENT value:
      input_ids      → pad_id        (a real token slot the model must ignore)
      token_type_ids → 0             (padding sits in segment 0, harmless)
      mlm_labels     → ignore_index  (-100 → cross_entropy skips it, like masking.py)
    attention_mask is derived: 1 on real tokens, 0 on padding.
    """
    input_ids = pad_sequence([b["input_ids"] for b in batch], batch_first=True, padding_value=pad_id)
    token_type_ids = pad_sequence([b["token_type_ids"] for b in batch], batch_first=True, padding_value=0)
    mlm_labels = pad_sequence([b["mlm_labels"] for b in batch], batch_first=True, padding_value=ignore_index)
    nsp_labels = torch.stack([b["nsp_label"] for b in batch])

    attention_mask = (input_ids != pad_id).long()

    return {
        "input_ids": input_ids,            # (B, S)
        "token_type_ids": token_type_ids,  # (B, S)
        "attention_mask": attention_mask,  # (B, S) — 1 keep, 0 pad
        "mlm_labels": mlm_labels,          # (B, S) → loss.py
        "nsp_labels": nsp_labels          # (B,)   → loss.py
    }


def create_dataloader(all_documents, tokenizer, max_seq_len, batch_size,
                      mlm_probability: float = 0.15, shuffle=True, num_workers=0):
    """Build the pre-training DataLoader end-to-end."""
    dataset = BERTPretrainingDataset(all_documents, tokenizer, max_seq_len, mlm_probability)
    pad_id = tokenizer.token_to_id("[PAD]")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=partial(collate_fn, pad_id=pad_id)
    )