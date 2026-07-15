import logging
from functools import partial

import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)


class ClassificationDataset(Dataset):
    """
    Wraps a HuggingFace split as single-sentence classification examples
    (Devlin et al., 2019, §4.1 — the SST-2 / single-sequence input format).

    Each text is WordPiece-tokenized and packed as [CLS] tokens [SEP], truncated
    to max_seq_len. token_type_ids are all-zero (segment A only — one sentence, no
    segment B), and the label is the dataset's topic class. Note what is ABSENT vs
    pretraining: no MLM masking (the model sees the full clean sentence) and no NSP
    label (single sentence, not a pair) — the only supervision is the class label.

    All tokenization happens once, up front, in __init__ (called once per split);
    __getitem__ just tensorizes a pre-built example (called once per fetch).

    Args:
        hf_split:    a HuggingFace dataset split (e.g. dataset["train"]) — iterable
                     of dict rows like {"text": "...", "label": 3}.
        tokenizer:   the SAME WordPiece tokenizer the encoder was pretrained with
                     (its vocab indexes the encoder's embedding rows).
        text_field:  column holding the raw sentence (config.data.text_field, "text").
        label_field: column holding the integer class (config.data.label_field, "label").
        max_seq_len: max tokens incl. [CLS]/[SEP]; the body is cut to max_seq_len - 2.

    Example (max_seq_len=6, "[CLS]"→2, "[SEP]"→3, body "রাজনীতি"→[41, 88]):
        row  = {"text": "রাজনীতি", "label": 0}
        ids  = [41, 88]                    # add_special_tokens=False → no auto [CLS]/[SEP]
        ids  = [41, 88][:4]  = [41, 88]    # room left for the two we add by hand
        ids  = [2, 41, 88, 3]              # [CLS] body [SEP]
        stored: ([2, 41, 88, 3], 0)
    """
    def __init__(self, hf_split, tokenizer, text_field, label_field, max_seq_len):
        # Local (not self.) — consumed entirely here in __init__; __getitem__ never needs them.
        cls_id = tokenizer.token_to_id("[CLS]")
        sep_id = tokenizer.token_to_id("[SEP]")

        self.examples = []
        for row in hf_split:
            # add_special_tokens=False: suppress the tokenizer's auto [CLS]/[SEP] so we can
            # truncate the BODY first, then add them by hand (avoids chopping a trailing [SEP]).
            ids = tokenizer.encode(row[text_field], add_special_tokens=False).ids
            ids = ids[: max_seq_len - 2]                 # leave room for [CLS] and [SEP]
            ids = [cls_id] + ids + [sep_id]              # [CLS] tokens [SEP]
            self.examples.append((ids, int(row[label_field])))

        logger.info(f"Built {len(self.examples)} classification examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ids, label = self.examples[idx]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "token_type_ids": torch.zeros(len(ids), dtype=torch.long),  # single sentence → all segment 0
            "label": torch.tensor(label, dtype=torch.long),
        }


def _collate(batch, pad_id):
    """Pad ragged examples to the batch's longest sequence → (B, S) tensors."""
    input_ids = pad_sequence([b["input_ids"] for b in batch], batch_first=True, padding_value=pad_id)
    token_type_ids = pad_sequence([b["token_type_ids"] for b in batch], batch_first=True, padding_value=0)
    labels = torch.stack([b["label"] for b in batch])
    return {"input_ids": input_ids, "token_type_ids": token_type_ids, "labels": labels}


def create_finetune_dataloaders(config, tokenizer):
    """
    Build the train/val DataLoaders for fine-tuning from a HuggingFace dataset.

    Downloads the dataset, wraps each split in a ClassificationDataset (tokenize +
    [CLS]/[SEP] pack), and returns loaders that dynamically pad each batch to its
    own longest sequence via _collate.

    Args:
        config:    the loaded FinetuneConfig. Reads config.data (dataset_id, subset,
                   text_field, label_field, num_labels) and config.training
                   (max_seq_len, batch_size).
        tokenizer: the WordPiece tokenizer the encoder was pretrained with. Used
                   both to tokenize text and to look up the [PAD] id for padding.

    Returns:
        train_loader: DataLoader over the train split, shuffled.
        val_loader:   DataLoader over the val split, not shuffled.
        num_labels:   number of classes for the classification head.

    Notes:
        - val split: uses "validation" if the dataset ships one, else falls back to
          "test". sna.bn ships train + validation + test, so this resolves to
          "validation" — leaving "test" untouched as a held-out set for evaluate.py.
        - num_labels: inferred as the count of distinct train labels unless
          config.data.num_labels pins an explicit value (see the `or` on that line).
          Assumes labels are contiguous 0..N-1 (true for sna.bn's 6-class ClassLabel).

    Example:
        train_loader, val_loader, num_labels = create_finetune_dataloaders(cfg, tok)
        # sna.bn → 11284 train, 1411 validation (test held out), num_labels = 6
        batch = next(iter(train_loader))
        # batch["input_ids"]      → LongTensor (32, S)   S = batch's longest seq
        # batch["token_type_ids"] → LongTensor (32, S)   all zeros (single sentence)
        # batch["labels"]         → LongTensor (32,)     one class id per example
    """
    d = config.data
    dataset = load_dataset(d.dataset_id, d.subset) if d.subset else load_dataset(d.dataset_id)

    train_split = dataset["train"]
    val_key = "validation" if "validation" in dataset else "test"
    val_split = dataset[val_key]

    num_labels = d.num_labels or len(set(train_split[d.label_field]))
    logger.info(f"Dataset {d.dataset_id}/{d.subset}: "
                f"{len(train_split)} train, {len(val_split)} {val_key}, {num_labels} labels")

    collate = partial(_collate, pad_id=tokenizer.token_to_id("[PAD]"))
    max_seq_len = config.training.max_seq_len
    train_ds = ClassificationDataset(train_split, tokenizer, d.text_field, d.label_field, max_seq_len)
    val_ds = ClassificationDataset(val_split, tokenizer, d.text_field, d.label_field, max_seq_len)

    train_loader = DataLoader(train_ds, batch_size=config.training.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=config.training.batch_size, shuffle=False, collate_fn=collate)
    return train_loader, val_loader, num_labels


def create_test_dataloader(config, tokenizer):
    """
    Build the held-out TEST DataLoader for evaluation (evaluate.py).

    The eval-time mirror of create_finetune_dataloaders' val branch: same tokenize +
    [CLS]/[SEP] pack + dynamic padding, but pinned to the "test" split. This is the
    split training NEVER sees — create_finetune_dataloaders resolves its val to
    "validation" (sna.bn ships train + validation + test), leaving "test" untouched
    precisely so it can be scored here once, as the reportable number (val accuracy
    only *picked* the fine-tune winner; test is what you'd cite).

    Args:
        config:    the loaded FinetuneConfig. Same fields as create_finetune_dataloaders
                   — config.data (dataset_id, subset, text_field, label_field, num_labels)
                   and config.training (max_seq_len, batch_size).
        tokenizer: the WordPiece tokenizer the encoder was pretrained (and fine-tuned)
                   with — tokenizes text and supplies the [PAD] id.

    Returns:
        test_loader: DataLoader over the test split, NOT shuffled (order is irrelevant
                     for scoring; keeping it stable makes any per-row debugging repeatable).
        num_labels:  number of classes — sizes/validates the classifier head.
        label_names: human-readable class names for the report (see below). Training
                     never needs this; only evaluate.py's confusion matrix / P-R-F1
                     table use it, purely to print "sports" instead of an anonymous "3".

    Note — num_labels vs label_names: distinct concerns.
        - num_labels is the MATH (how many output logits) — feeds the model.
        - label_names is the LABELLING (what each index is called) — feeds the report.
          Positions align: label_names[3] names class 3 in the confusion matrix.

    Example:
        test_loader, num_labels, label_names = create_test_dataloader(cfg, tok)
        # sna.bn → 1411 test examples, num_labels = 6
        # label_names = ['kolkata', 'state', 'national', 'sports', 'entertainment', 'international']
    """
    from datasets import ClassLabel   # local import: only the eval path needs the type check

    d = config.data
    dataset = load_dataset(d.dataset_id, d.subset) if d.subset else load_dataset(d.dataset_id)
    test_split = dataset["test"]       # the held-out split — deliberately NOT "validation"

    # num_labels: config wins if pinned, else count distinct labels (assumes contiguous
    # 0..N-1, true for sna.bn's ClassLabel) — same `or` fallback as create_finetune_dataloaders.
    num_labels = d.num_labels or len(set(test_split[d.label_field]))

    # feat is the column's SCHEMA object (one per split), not the data: for a ClassLabel
    # column it carries the int->name table. sna.bn -> ClassLabel(names=['kolkata', ...]).
    feat = test_split.features[d.label_field]
    # Prefer the dataset's real topic names; fall back to str indices ['0'..'N-1'] if the
    # label column isn't a ClassLabel (e.g. a plain int/Value column) so the report never crashes.
    label_names = feat.names if isinstance(feat, ClassLabel) else [str(i) for i in range(num_labels)]
    logger.info(f"Test split: {len(test_split)} examples, {num_labels} labels")

    collate = partial(_collate, pad_id=tokenizer.token_to_id("[PAD]"))
    test_ds = ClassificationDataset(test_split, tokenizer, d.text_field, d.label_field,
                                    config.training.max_seq_len)
    test_loader = DataLoader(test_ds, batch_size=config.training.batch_size,
                             shuffle=False, collate_fn=collate)   # no shuffle — scoring is order-invariant
    return test_loader, num_labels, label_names