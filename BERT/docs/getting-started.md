# Getting Started

Install the project and prove the pre-training pipeline runs end to end on the
**tiny** config — a 2-layer model on a few hundred articles — in about 15 minutes.
This is a smoke test, not a real model; when it passes, move on to
[Training](training.md) for the full run.

## Prerequisites

- **Python 3.12**
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- A few GB of free disk (HuggingFace caches the Wikipedia dump)
- A GPU is optional — the code runs on Apple MPS, CUDA, or CPU (`device: "auto"`
  in the config picks the best available)

## 1. Install

From the **repo root** (the folder above `BERT/`):

```bash
uv sync        # installs torch, datasets, tokenizers, gradio, … from pyproject.toml
```

BERT-only subset, if you don't want the whole portfolio's dependencies:

```bash
pip install -r BERT/requirements.txt   # from the repo root, in your own venv
```

## 2. Sanity-check the pipeline

Everything runs from the **repo root** as a module — the leading `BERT.` matters.
The tiny config wires up the whole path (corpus → WordPiece → MLM/NSP examples →
batches → loss) so bugs surface in seconds instead of hours.

```bash
# Build a tiny corpus (~500 articles → BERT/data/bn_wiki_tiny.txt)
python -m BERT.scripts.prepare_corpus BERT/configs/tiny.yaml

# Pre-train the tiny model for 3 epochs — you only want to see the loss go down
python -m BERT.scripts.pretrain --config BERT/configs/tiny.yaml
```

If the loss falls and a checkpoint lands under `BERT/checkpoints/tiny/`, the
pipeline is healthy.

## 3. Where things land

The first two commands create everything else; nothing here is committed to git.

| Path | What |
|---|---|
| `BERT/data/` | The text corpus (`bn_wiki.txt`, `bn_wiki_tiny.txt`) |
| `BERT/tokenizer/<name>/` | Trained WordPiece vocab (`vocab.txt`) |
| `BERT/checkpoints/<task>/run_<ts>/` | `best.pt`, `last.pt`, and a frozen `config.yaml` snapshot |
| `BERT/logs/` | TensorBoard event files (`tensorboard --logdir BERT/logs`) |

## Next

- **[Training](training.md)** — reproduce the real 86.5% result on the `base` config.
- **[Architecture](architecture.md)** — model spec, config field reference, checkpoint convention.
- **The story (the *why*)** — the full write-up lives on the blog:
  [BERT, from scratch](https://www.samyamoyrakshit.com/blog/bert-from-scratch/).
