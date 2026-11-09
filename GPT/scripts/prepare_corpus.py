"""
BERT's corpus → GPT's corpus: the same Bengali Wikipedia articles, with
<|endoftext|> between them.

BERT/scripts/prepare_corpus.py already streamed, filtered and cleaned the dump and
wrote one article per line, blank line between. Reusing that file gives GPT
exactly the text BERT trained on, offline. Only the separator changes: each
article ends with <|endoftext|>, as GPT-2 separates documents. No newline follows
the marker: it would become the first token of every article.

The price: BERT collapsed every newline inside an article, so the model never
sees a paragraph break.
"""
import os
import sys

from ..bpe.bpe import EOT
from ..utils.config import Config


def prepare_corpus(config):
    data = config.data
    assert os.path.exists(data.source_path), \
        f"{data.source_path} not found; run `python -m BERT.scripts.prepare_corpus` first"

    kept = 0
    os.makedirs(os.path.dirname(data.corpus_path), exist_ok=True)   # data/ is gitignored, so a fresh clone has none
    with open(data.source_path, encoding="utf-8") as src, \
         open(data.corpus_path, "w", encoding="utf-8") as out:
        for line in src:
            article = line.strip()
            if article:                          # skip the blank line between articles
                out.write(article + EOT)
                kept += 1

    print(f"Kept {kept} articles")
    print(f"→ {data.corpus_path}")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "GPT/configs/base.yaml"
    prepare_corpus(Config.from_yaml(config_path))
