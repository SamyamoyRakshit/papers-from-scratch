"""
Held-out evaluation for a pre-trained GPT: loss → perplexity AND bits-per-byte.

Data: val.bin — the 10% of articles prepare_data.py held out (seed 42: the same
1,621 articles BERT held out). The model never trained on them. There is no
separate test split: val only picked best.pt, and for base that was the last step.

Why two numbers:
  - perplexity = exp(loss) is per TOKEN, so it depends on the tokenizer: a bigger
    vocab means fewer, harder tokens. Not comparable across tokenizers.
  - bits-per-byte = total loss in bits / UTF-8 bytes of the text. The bytes are the
    same whatever the tokenizer, so this one is (GPT-2 Table 3 reports it for
    enwik8 / text8). Still not comparable with English: a Bengali char is 3 bytes.

Config comes from the checkpoint's sibling config.yaml snapshot (via generate.load_model),
so evaluate needs no config flag of its own — just point --checkpoint at a best.pt.

Usage (from repo root):
    python -m GPT.scripts.evaluate
    python -m GPT.scripts.evaluate --checkpoint GPT/checkpoints/base/run_<ts>/best.pt
"""
import argparse
import logging
import math
import os
from pathlib import Path

import numpy as np

from common.run_utils import sha256_file
from ..utils.data_utils import load_ids
from ..utils.loss import GPTLoss
from ..utils.train_utils import validate
from ._common import load_checkpoint
from .generate import load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a pre-trained GPT on val.bin")
    default_ckpt = Path(__file__).parent.parent / "checkpoints" / "base" / "best.pt"
    parser.add_argument("--checkpoint", type=str, default=str(default_ckpt),
                        help="best.pt from a pretrain run (or the leaderboard best.pt symlink)")
    args = parser.parse_args()

    model, tokenizer, config, device = load_model(args.checkpoint)

    # val.bin has no fingerprint of its own, but prepare_data.py writes train.bin and
    # val.bin from ONE split: a changed train.bin means a changed split, and val.bin
    # could now hold articles the model trained on.
    checkpoint = load_checkpoint(os.path.realpath(args.checkpoint), "cpu")
    if sha256_file(config.data.train_path) != checkpoint["data_fingerprint"]:
        raise RuntimeError(
            "train.bin changed since training, so val.bin may hold training articles. "
            "Re-run prepare_data.py with the run's config."
        )

    # --- Loss: the same full pass pretrain.py logged as Val ---
    val_ids = load_ids(config.data.val_path)
    T = config.model.block_size
    loss = validate(model, val_ids, GPTLoss(), config.training.batch_size, T, device)   # nats per token

    # --- Bytes of exactly the targets validate() scored ---
    # Windows start at 0, T, 2T, ...; each y is its x shifted by one.
    n_windows = len(range(0, len(val_ids) - T, T))
    targets = val_ids[1:n_windows * T + 1]
    # <|endoftext|> counts 0 bytes, as in prepare_data.py's bytes/token: predicting
    # where an article ends costs bits, but the 13-byte marker is not text.
    id_bytes = np.array([len(tokenizer.vocab[i]) for i in range(len(tokenizer.vocab))])
    id_bytes[tokenizer.eot_id] = 0
    n_bytes = int(id_bytes[targets].sum())

    total_bits = loss * len(targets) / math.log(2)
    logger.info(f"Val: {len(targets):,} tokens scored ({n_windows:,} windows of {T}) = {n_bytes:,} bytes "
                f"({n_bytes / len(targets):.2f} bytes/token)")
    logger.info(f"Loss {loss:.4f} nats/token | perplexity {math.exp(loss):.1f} | "
                f"{loss / math.log(2):.3f} bits/token | {total_bits / n_bytes:.4f} bits/byte")


if __name__ == "__main__":
    main()
