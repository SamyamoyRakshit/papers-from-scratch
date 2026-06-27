import sys
from datasets import load_dataset

from ..utils.config import Config

# Optional private filter
try:
    from ._corpus_filter import keep_article
except ImportError:
    def keep_article(text):          # public default: no filtering
        return True
    

def prepare_corpus(config):
    data = config.data
    # streaming=True: pull rows over HTTP on demand instead of rebuilding the whole
    # ~143k-article dump to local Arrow first (multi-GB; we only want a slice of it).
    ds = load_dataset(path=data.dataset, name=data.wiki_dump, split="train", streaming=True)
    if data.max_articles:
        ds = ds.take(data.max_articles)              # IterableDataset → .take, not .select

    kept = dropped_short = dropped_filter = 0
    with open(data.corpus_path, "w", encoding="utf-8") as f:
        for article in ds:
            text = article["text"].strip()
            if len(text) < data.min_chars:
                dropped_short += 1
                continue
            if not keep_article(text):           # ← private filter (or always-True fallback)
                dropped_filter += 1
                continue
            text = " ".join(text.split())        # one article = one line
            f.write(text + "\n\n")
            kept += 1

    print(f"Kept {kept} | dropped (filter) {dropped_filter} | dropped (short) {dropped_short}")
    print(f"→ {data.corpus_path}")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "BERT/configs/base.yaml"
    prepare_corpus(Config.from_yaml(config_path))
