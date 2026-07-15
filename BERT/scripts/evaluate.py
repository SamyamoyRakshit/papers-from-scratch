"""
Evaluation entrypoint for a fine-tuned BERT classifier (Devlin et al. 2019, §4.1).

Rebuilds the exact fine-tuned model from a finetune run's best.pt and scores it on
the HELD-OUT test split — the split create_finetune_dataloaders never touches, so this
number is the reportable one (val accuracy only picked the winner). Reports overall
accuracy, a per-class precision/recall/F1 table, and a confusion matrix.

Config comes from the checkpoint's sibling config.yaml (the FinetuneConfig snapshot),
so evaluate needs no config flag of its own — just point --checkpoint at a best.pt.

Usage (from repo root):
    python -m BERT.scripts.evaluate
    python -m BERT.scripts.evaluate --checkpoint BERT/checkpoints/finetune/sna_bn/best.pt
"""
import argparse
import logging
import os
from pathlib import Path

import numpy as np
import torch

from common.run_utils import get_device, sha256_file
from ..models.bert_for_classification import BERTForSequenceClassification
from ..utils.config import Config
from ..utils.data_utils import load_tokenizer
from ..utils.finetune_config import FinetuneConfig
from ..utils.finetune_data import create_test_dataloader
from ._common import load_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate(model, loader, device, num_labels):
    """
    Run the fine-tuned model over the test set, tallying a (K, K) confusion matrix.

    confusion[t, p] = # examples whose TRUE class was t but were PREDICTED as p. Every
    example lands in exactly one cell: correct ones on the diagonal (t == p), mistakes
    off-diagonal. So the whole matrix sums to len(test), the diagonal sums to #correct.
    This single matrix is all print_report needs — nothing else is accumulated here.
    """
    model.eval()                                                   # dropout OFF → fixed, deterministic weights
    confusion = torch.zeros(num_labels, num_labels, dtype=torch.long)   # no device= → CPU tensor
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["labels"]                                   # left on CPU — only used to index confusion (also CPU)
        preds = model(input_ids, token_type_ids).argmax(dim=-1).cpu()   # (B,) top-scoring class/row; .cpu() to match confusion
        for t, p in zip(labels, preds):
            confusion[t, p] += 1                                   # bump [true, pred]; a wrong pred bumps an OFF-diagonal cell
    return confusion


def print_report(confusion, label_names):
    """
    Turn the confusion matrix into the printed report: per-class precision/recall/F1,
    overall accuracy, macro + weighted averages, and the raw matrix.

    Everything is read off the same C (rows = true, cols = pred) along three axes:
        tp       = C.diagonal()  → correct per class
        support  = C.sum(1)      → # TRUE per class  (row totals)
        pred_tot = C.sum(0)      → # PREDICTED per class  (column totals)
    then  recall = tp/support     (of the true X, how many caught),
          precision = tp/pred_tot (of what we called X, how many right).

    Worked on the toy 3-class matrix C = [[1,1,0],[0,1,0],[1,0,1]] (5 examples):
        accuracy       = diag/total = 3/5           = 0.600
        precision      = [1/2, 1/2, 1/1]            = [.500, .500, 1.000]
        weighted-prec  = (2·.5 + 1·.5 + 2·1)/5      = 0.700   (support-weighted)
    """
    C = confusion.numpy().astype(np.int64)   # .numpy() needs a CPU tensor (confusion already is; else .cpu() first)
    support  = C.sum(axis=1)          # row totals = TRUE count per class
    tp       = np.diag(C)             # diagonal   = correct per class
    pred_tot = C.sum(axis=0)          # col totals = PREDICTED count per class
    # np.where guards 0/0 → 0.0 (not NaN): a class never predicted has pred_tot=0 (precision),
    # a class absent from test has support=0 (recall). errstate silences the divide warning.
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(pred_tot > 0, tp / pred_tot, 0.0)
        recall    = np.where(support  > 0, tp / support,  0.0)
        f1        = np.where((precision + recall) > 0,
                             2 * precision * recall / (precision + recall), 0.0)
    total = C.sum()
    accuracy = tp.sum() / total       # diagonal sum / all examples — THE reported metric for sna.bn

    w = max(len(n) for n in list(label_names) + ["weighted avg"])   # column width = longest row label
    logger.info(f"{'class':<{w}}  {'prec':>6} {'recall':>6} {'f1':>6} {'support':>8}")
    for i, name in enumerate(label_names):
        logger.info(f"{name:<{w}}  {precision[i]:>6.3f} {recall[i]:>6.3f} {f1[i]:>6.3f} {support[i]:>8}")
    logger.info("-" * (w + 32))
    logger.info(f"{'accuracy':<{w}}  {'':>6} {'':>6} {accuracy:>6.3f} {total:>8}")
    # macro = plain mean over classes (each class equal); weighted = mean weighted by support
    # (bigger classes count more, so it tracks accuracy). They differ when classes are imbalanced.
    macro = (precision.mean(), recall.mean(), f1.mean())
    wavg  = tuple(float((m * support).sum() / total) for m in (precision, recall, f1))
    logger.info(f"{'macro avg':<{w}}  {macro[0]:>6.3f} {macro[1]:>6.3f} {macro[2]:>6.3f} {total:>8}")
    logger.info(f"{'weighted avg':<{w}}  {wavg[0]:>6.3f} {wavg[1]:>6.3f} {wavg[2]:>6.3f} {total:>8}")

    logger.info("Confusion matrix (rows = true, cols = pred):")
    for i, name in enumerate(label_names):
        logger.info(f"{name:<{w}}  " + " ".join(f"{c:>5}" for c in C[i]))


def main():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned BERT classifier")
    default_ckpt = Path(__file__).parent.parent / "checkpoints" / "finetune" / "sna_bn" / "best.pt"
    parser.add_argument("--checkpoint", type=str, default=str(default_ckpt),
                        help="best.pt from a finetune run (or the leaderboard best.pt symlink)")
    args = parser.parse_args()

    # Resolve the symlink: parent-level best.pt → run_<ts>/best.pt, whose sibling is the
    # FinetuneConfig snapshot that produced it (dataset + tokenizer + pretrained dims).
    ckpt_path = os.path.realpath(args.checkpoint)
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

    # --- Held-out test data ---
    test_loader, num_labels, label_names = create_test_dataloader(config, tokenizer)

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
    model.load_state_dict(checkpoint["model_state_dict"])   # strict=True — full fine-tuned state (body + classifier)
    # val_acc is provenance only: the val score that PICKED this checkpoint (should match the
    # leaderboard winner). The reported number is the TEST accuracy from print_report below.
    logger.info(f"Loaded fine-tuned model | val_acc at save: {checkpoint.get('val_acc', 'n/a')}")

    confusion = evaluate(model, test_loader, device, num_labels)
    print_report(confusion, label_names)


if __name__ == "__main__":
    main()
