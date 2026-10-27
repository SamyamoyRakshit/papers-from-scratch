"""
GPT corpus → byte-level BPE: merges.txt + vocab.json in paths.tokenizer_dir.

A separate step because learning the merges is the slow part; prepare_data.py
loads the saved merges and never relearns them.
"""
import sys
import time

from ..bpe.bpe import BPETokenizer
from ..utils.config import Config


def train_tokenizer(config):
    with open(config.data.corpus_path, encoding="utf-8") as f:
        text = f.read()

    start = time.time()
    tokenizer = BPETokenizer.train(text, config.data.vocab_size)
    print(f"Learned {len(tokenizer.merges):,} merges in {time.time() - start:.0f} s")

    tokenizer.save(config.paths.tokenizer_dir)
    print(f"→ {config.paths.tokenizer_dir}")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "GPT/configs/base.yaml"
    train_tokenizer(Config.from_yaml(config_path))
