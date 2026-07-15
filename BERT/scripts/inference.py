"""
Inference entrypoint for a fine-tuned BERT classifier (Devlin et al. 2019, §4.1).

Takes a raw Bengali sentence and predicts its news topic. Rebuilds the exact
fine-tuned model from a finetune run's best.pt (same recipe as evaluate.py),
packs the text exactly like ClassificationDataset does at train/eval time
([CLS] body [SEP], truncated to max_seq_len, all-zero segments), and prints
the predicted class plus the full softmax distribution.

Config comes from the checkpoint's sibling config.yaml (the FinetuneConfig
snapshot) — just point --checkpoint at a best.pt and pass --text.

Usage (from repo root):
    python -m BERT.scripts.inference --text "কলকাতায় আজ বৃষ্টি হবে"
    python -m BERT.scripts.inference --checkpoint BERT/checkpoints/finetune/sna_bn/best.pt --text "..."
"""
import argparse
import logging
import os
from pathlib import Path

import torch
from datasets import ClassLabel, load_dataset

from common.run_utils import get_device, sha256_file
from ..models.bert_for_classification import BERTForSequenceClassification
from ..utils.config import Config
from ..utils.data_utils import load_tokenizer
from ..utils.finetune_config import FinetuneConfig
from ._common import load_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def encode_text(text, tokenizer, max_seq_len):
    """
    Pack ONE raw sentence the same way ClassificationDataset packs every training
    example — [CLS] body [SEP], body truncated first — so the model sees inference
    input in exactly the format it was fine-tuned on.

    Returns (input_ids, token_type_ids), both (1, S): a batch of one, no padding
    needed since there's nothing to align it with.
    """
    cls_id = tokenizer.token_to_id("[CLS]")
    sep_id = tokenizer.token_to_id("[SEP]")
    ids = tokenizer.encode(text, add_special_tokens=False).ids   # suppress auto [CLS]/[SEP]
    ids = ids[: max_seq_len - 2]                                 # leave room for the two we add
    ids = [cls_id] + ids + [sep_id]
    input_ids = torch.tensor([ids], dtype=torch.long)            # (1, S)
    token_type_ids = torch.zeros_like(input_ids)                 # single sentence → all segment 0
    return input_ids, token_type_ids


@torch.no_grad()
def predict(model, input_ids, token_type_ids, device):
    """Forward one packed sentence → (K,) softmax probabilities on CPU."""
    model.eval()                                                 # dropout OFF — deterministic
    logits = model(input_ids.to(device), token_type_ids.to(device))   # (1, K)
    # argmax alone would give the class; softmax also gives a confidence to report.
    return torch.softmax(logits, dim=-1)[0].cpu()                # (K,)


def load_finetuned_classifier(checkpoint_arg):
    """
    Resolve a finetune best.pt into a ready-to-serve model. Shared by this script's
    main() and app.py — call once, then feed encode_text/predict per sentence.

    Returns:
        model:       fine-tuned BERTForSequenceClassification on `device`, weights loaded.
        tokenizer:   the WordPiece tokenizer the model was fine-tuned with.
        label_names: int→name table for the K classes (from the dataset's ClassLabel schema).
        config:      the run's FinetuneConfig (callers need config.training.max_seq_len).
        device:      the resolved torch.device.
    """
    # Same preflight as evaluate.py: resolve the symlink, load the sibling
    # FinetuneConfig snapshot, and verify the tokenizer hash before trusting output.
    ckpt_path = os.path.realpath(checkpoint_arg)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    config_path = os.path.join(os.path.dirname(ckpt_path), "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config snapshot not found beside checkpoint: {config_path}")

    config = FinetuneConfig.from_yaml(config_path)
    pretrained_cfg = Config.from_yaml(config.pretrained.config)   # encoder dims
    device = get_device(config.device)
    logger.info(f"Checkpoint: {ckpt_path}")
    logger.info(f"Using device: {device}")

    # --- Tokenizer (MUST match the one the model was fine-tuned with) ---
    vocab_path = os.path.join(config.paths.tokenizer_dir, "vocab.txt")
    tokenizer = load_tokenizer(vocab_path)
    tokenizer_sha256 = sha256_file(vocab_path)
    vocab_size = tokenizer.get_vocab_size()

    checkpoint = load_checkpoint(ckpt_path, device)
    ckpt_tok_hash = checkpoint.get("tokenizer_sha256")
    if ckpt_tok_hash in (None, "unknown"):
        logger.warning("Checkpoint has no tokenizer_sha256 — cannot verify tokenizer. Continuing.")
    elif ckpt_tok_hash != tokenizer_sha256:
        raise RuntimeError(
            f"Tokenizer mismatch: checkpoint expects {ckpt_tok_hash}, current is {tokenizer_sha256}."
        )

    # num_labels: read K straight off the saved classifier — W ∈ R^(K×H) (§3.5), so
    # shape[0] = K. The checkpoint is the source of truth; no dataset needed for this.
    num_labels = checkpoint["model_state_dict"]["classifier.weight"].shape[0]

    # label_names: the int→name table lives in the dataset's ClassLabel schema (same
    # lookup as create_test_dataloader) — only needed to print "sports" instead of "3".
    d = config.data
    dataset = load_dataset(d.dataset_id, d.subset) if d.subset else load_dataset(d.dataset_id)
    feat = dataset["train"].features[d.label_field]
    label_names = feat.names if isinstance(feat, ClassLabel) else [str(i) for i in range(num_labels)]

    # --- Rebuild the exact model and load ALL fine-tuned weights (body + classifier) ---
    model = BERTForSequenceClassification(
        vocab_size=vocab_size,
        d_model=pretrained_cfg.model.d_model,
        num_heads=pretrained_cfg.model.num_heads,
        d_ff=pretrained_cfg.model.d_ff,
        num_layers=pretrained_cfg.model.num_layers,
        num_labels=num_labels,
        max_position_embeddings=pretrained_cfg.model.max_position_embeddings,
        num_segments=pretrained_cfg.model.num_segments,
        pad_idx=pretrained_cfg.tokens.pad_idx,
        dropout=pretrained_cfg.model.dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])   # strict=True — full fine-tuned state
    return model, tokenizer, label_names, config, device


def main():
    parser = argparse.ArgumentParser(description="Classify a sentence with a fine-tuned BERT")
    default_ckpt = Path(__file__).parent.parent / "checkpoints" / "finetune" / "sna_bn" / "best.pt"
    parser.add_argument("--checkpoint", type=str, default=str(default_ckpt),
                        help="best.pt from a finetune run (or the leaderboard best.pt symlink)")
    parser.add_argument("--text", type=str, required=True,
                        help="raw sentence to classify")
    args = parser.parse_args()

    model, tokenizer, label_names, config, device = load_finetuned_classifier(args.checkpoint)

    input_ids, token_type_ids = encode_text(args.text, tokenizer, config.training.max_seq_len)
    probs = predict(model, input_ids, token_type_ids, device)

    pred = int(probs.argmax())
    logger.info(f"Text: {args.text}")
    logger.info(f"Predicted topic: {label_names[pred]}  (p = {probs[pred]:.3f})")
    for i in probs.argsort(descending=True):                 # all K classes, most→least likely
        logger.info(f"  {label_names[i]:<15} {probs[i]:.3f}")


if __name__ == "__main__":
    main()
