"""
GPT corpus → train.bin / val.bin: the token ids as raw uint16, read with np.memmap.

Splits by whole article, as BERT/scripts/pretrain.py does: on the same corpus
with the same seed and val_split, val is the same articles BERT held out.
"""
import random
import sys
import time

import numpy as np

from ..bpe.bpe import EOT, BPETokenizer
from ..utils.config import Config


def split_articles(articles, val_split: float, seed: int):
    """Shuffle with a seeded, local RNG, then the first val_split go to val."""
    articles = list(articles)
    random.Random(seed).shuffle(articles)
    val_count = int(len(articles) * val_split)
    return articles[val_count:], articles[:val_count]   # train, val


def prepare_data(config):
    with open(config.data.corpus_path, encoding="utf-8") as f:
        articles = f.read().split(EOT)[:-1]    # the file ends with <|endoftext|>, so the last piece is ''

    tokenizer = BPETokenizer.load(config.paths.tokenizer_dir)
    train, val = split_articles(articles, config.data.val_split, config.seed)

    for name, part, path in (("train", train, config.data.train_path),
                             ("val", val, config.data.val_path)):
        start = time.time()
        # np.array raises on an id >= 2**16; .astype(np.uint16) would wrap it silently.
        ids = np.array(tokenizer.encode("".join(a + EOT for a in part)), dtype=np.uint16)
        ids.tofile(path)
        # bytes/token measures the text alone: not the 13-byte marker, not its one id per article.
        n_bytes = sum(len(a.encode("utf-8")) for a in part)
        n_text = len(ids) - len(part)
        print(f"{name}: {len(part):,} articles → {len(ids):,} tokens, "
              f"{n_bytes / n_text:.2f} bytes/token, {time.time() - start:.0f} s → {path}")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "GPT/configs/base.yaml"
    prepare_data(Config.from_yaml(config_path))
