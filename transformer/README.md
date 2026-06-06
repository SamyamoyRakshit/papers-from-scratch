# Transformer — "Attention Is All You Need", from scratch (English → Bengali)

A complete Transformer from
[**"Attention Is All You Need"** (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762),
rebuilt tensor-by-tensor in plain PyTorch — **no `nn.Transformer`, no `nn.MultiheadAttention`** —
and trained to translate **English into Bengali** on a single **Apple M1 with 16 GB of RAM**.

Part of the *Replicating-Papers-On-CustomData* portfolio — SOTA architectures rebuilt from
scratch on a custom dataset.

## Read the full story

- 📖 **[ARTICLE.md](ARTICLE.md)** — the long-form write-up: every component explained with
  worked numerical examples, the 16 GB war stories (the NaN born in attention, the
  slow-motion OOM), and the results told honestly.
- 📐 **[docs/](docs/)** — per-module deep dives (the math + implementation) for
  [attention](docs/modules/multi_head_attention.md), [positional encoding](docs/modules/positional_encoding.md),
  [layer norm](docs/modules/layer_norm.md), the [encoder](docs/architecture/encoder.md) /
  [decoder](docs/architecture/decoder.md) / [full model](docs/architecture/transformer.md),
  and the scripts ([train](docs/scripts/train.md) / [inference](docs/scripts/inference.md) /
  [evaluate](docs/scripts/evaluate.md)).

## What's hand-written

Multi-head scaled dot-product attention, sinusoidal positional encoding (with the
relative-position property), the position-wise FFN, layer normalization, the label-smoothed
loss, the Noam LR schedule, weight tying, and pooled-top-k beam search. The constraint is the
hardware: the paper used 8× P100 GPUs; this fits in 16 GB by scaling the model down
(`d_model` 512→256, layers 6→4, ~65M→**~11M** params) while keeping every *method* from the
paper exact (optimizer betas, warmup, label smoothing, dropout, √d_model scaling).

## Results

Headline model: the 30-epoch `base` run — best val loss **6.004** (epoch 24) → perplexity ≈ **405**.

| Decode setting | BLEU (sacreBLEU `intl`) | Length ratio |
|---|---|---|
| `beam=2, α=0.6` (paper defaults) | 0.07 | 0.349 |
| `beam=4, α=1.0` (tuned) | **0.17** | 0.565 |

BLEU is humble (production En–Bn scores in the 20s–30s) — and that's the honest point: this is
a *correct, from-scratch* Transformer learning to translate on a laptop, with the ≈6.6× fewer
training tokens / ~31× smaller batch gap reported openly rather than hidden. See
[ARTICLE.md → Part 7](ARTICLE.md) for the full breakdown.

## Architecture / Configs

| Config | d_model | layers | heads | d_ff | vocab | Purpose |
|--------|---------|--------|-------|------|-------|---------|
| `base.yaml` | 256 | 4 | 8 | 1024 | 16k | Main runs (paper "base", scaled for 16 GB) |
| `tiny.yaml` | 64 | 2 | 4 | 256 | 4k | Fast pipeline sanity checks (~2 min) |

Optimizer, label smoothing (0.1), warmup (4000 steps), and dropout (0.1) follow the paper exactly.
Every knob is documented with its reason in [`configs/base.yaml`](configs/base.yaml).

## Setup

Python 3.12, managed with [`uv`](https://docs.astral.sh/uv/). From the **repo root** (the
folder above `transformer/`):

```bash
uv sync        # installs torch, sentencepiece, sacrebleu, gradio, … from pyproject.toml
```

For higher HuggingFace rate limits when downloading the dataset, copy `.env.example` → `.env`
and set `HF_TOKEN`.

> **Weights are not in this repo.** Checkpoints, logs, and the tokenizer are gitignored to keep
> the repo light. **Train them yourself** (below) — the first run also trains and caches the
> SentencePiece tokenizer.

## Usage

Run everything from the **repo root** as modules (the leading `transformer.` matters):

```bash
# Train (paper "base", scaled for 16 GB)
python -m transformer.scripts.train --config transformer/configs/base.yaml

# Quick sanity run (~2 min) on the tiny model
python -m transformer.scripts.train --config transformer/configs/tiny.yaml

# Crash-resilient wrapper — re-launches after MPS OOM, resumes latest (config, target_epoch)
bash transformer/scripts/auto_resume_train.sh transformer/configs/base.yaml 30

# Translate a sentence (beam search; REPL if --text omitted)
python -m transformer.scripts.inference \
    --config transformer/configs/base.yaml \
    --checkpoint transformer/checkpoints/base/run_<timestamp>/best.pt \
    --text "What are you saying?"

# BLEU + perplexity on the validation split
python -m transformer.scripts.evaluate \
    --config transformer/configs/base.yaml \
    --checkpoint transformer/checkpoints/base/run_<timestamp>/best.pt \
    --max_samples 500

# Gradio demo (browser UI with live beam / α sliders)
uv run python -m transformer.scripts.app   # serves http://127.0.0.1:7860

# Watch training
tensorboard --logdir transformer/logs/base
```

## Layout

See [`repo.txt`](repo.txt) for the full annotated file tree. The short version:

```
models/     # Transformer + encoder/decoder + modules (attention, PE, FFN, layer norm)
utils/      # config, data, masks, optimizer (Noam), loss, trainer, logging
scripts/    # train / evaluate / inference / app (Gradio) + auto_resume
configs/    # base.yaml, tiny.yaml
docs/       # per-module math + implementation write-ups
ARTICLE.md  # the full long-form article
```

## Citation

Replicates the original paper (citation is for the authors, not the implementer):

```bibtex
@inproceedings{vaswani2017attention,
  title     = {Attention Is All You Need},
  author    = {Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob
               and Jones, Llion and Gomez, Aidan N. and Kaiser, {\L}ukasz and Polosukhin, Illia},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2017},
  url       = {https://arxiv.org/abs/1706.03762}
}
```
