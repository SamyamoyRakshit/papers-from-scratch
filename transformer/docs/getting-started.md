# Getting Started

Install the project and prove the training pipeline runs end to end on the
**tiny** config — a 2-layer model on 1,000 sentence pairs — in a couple of minutes.
This is a smoke test, not a real model; when the loss falls, move on to
[Training](training.md) for the full run.

## Prerequisites

- **Python 3.12**
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- A few GB of free disk (HuggingFace caches the Samanantar dataset)
- A GPU is optional — the code runs on Apple MPS, CUDA, or CPU (`device: "auto"`
  in the config picks the best available)

## 1. Install

From the **repo root** (the folder above `transformer/`):

```bash
uv sync        # installs torch, sentencepiece, sacrebleu, gradio, … from pyproject.toml
```

For higher HuggingFace rate limits when downloading the dataset, copy
`.env.example` → `.env` and set `HF_TOKEN`.

## 2. Sanity-check the pipeline

Everything runs from the **repo root** as a module — the leading `transformer.`
matters. There's no separate data-prep step: the first run downloads the dataset,
trains a SentencePiece tokenizer, and caches both.

```bash
# Tiny model, 1,000 pairs, ~2 min — you only want to see the loss go down
python -m transformer.scripts.train --config transformer/configs/tiny.yaml
```

If the loss falls and a checkpoint lands under `transformer/checkpoints/tiny/`,
the pipeline is healthy.

## 3. Where things land

The first run creates everything else; nothing here is committed to git.

| Path | What |
|---|---|
| `transformer/tokenizer/<name>/sp.model` | Trained SentencePiece tokenizer (shared En/Bn vocab) |
| `transformer/checkpoints/<config>/run_<ts>/` | `best.pt`, `last.pt`, and a frozen `config.yaml` snapshot |
| `transformer/logs/` | TensorBoard event files (`tensorboard --logdir transformer/logs`) |

## Next

- **[Training](training.md)** — reproduce the real `base` run (val loss ≈6.0, BLEU 0.17).
- **[Architecture](architecture.md)** — model spec, config field reference, checkpoint convention.
- **The story (the *why*)** — the full write-up lives on the blog:
  [Attention Is All You Need, from scratch](https://www.samyamoyrakshit.com/blog/attention-from-scratch/).
