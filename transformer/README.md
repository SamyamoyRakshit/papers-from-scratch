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

## How the pieces fit together

### Paper → code

| Paper section | What it implements | File |
|---|---|---|
| §3.4 — embedding scale (√d_model) | scale token vectors so PE is a gentle perturbation | [`models/modules/embeddings.py`](models/modules/embeddings.py) |
| §3.5 — positional encoding | inject word order via sinusoids | [`models/modules/positional_encoding.py`](models/modules/positional_encoding.py) |
| §3.2 — scaled dot-product + multi-head attention | `softmax(QKᵀ/√dₖ)V`, split into 8 parallel heads | [`models/modules/multi_head_attention.py`](models/modules/multi_head_attention.py) |
| §3.3 — position-wise FFN | per-token 2-layer MLP (expand → ReLU → compress) | [`models/modules/feed_forward.py`](models/modules/feed_forward.py) |
| Ba et al. 2016 — layer normalization | normalize each token independently across features | [`models/modules/layer_norm.py`](models/modules/layer_norm.py) |
| §3.1 — encoder stack | N × (self-attn + FFN) with residuals + post-LayerNorm | [`models/encoder.py`](models/encoder.py) |
| §3.1 — decoder stack | N × (masked-self-attn + cross-attn + FFN) | [`models/decoder.py`](models/decoder.py) |
| §3.4 — full model + weight tying | wire encoder + decoder + share one embedding matrix | [`models/transformer.py`](models/transformer.py) |
| §3.2.3 — masking | padding mask + causal mask, combined for the decoder | [`utils/mask_utils.py`](utils/mask_utils.py) |
| §5.4 — label-smoothed loss | fused cross-entropy with ε=0.1 smoothing | [`utils/loss.py`](utils/loss.py) |
| §5.3 — Noam LR schedule | Adam + linear warmup + step^-0.5 decay | [`utils/optimizer.py`](utils/optimizer.py) |
| §5.1 — data pipeline | SentencePiece tokenizer + token-based batching | [`utils/data_utils.py`](utils/data_utils.py) |
| §6.1 — beam search | pooled top-k beams + length penalty | [`scripts/inference.py`](scripts/inference.py) |

### Training flow

```
AI4Bharat Samanantar (8.5M En-Bn pairs, 500K subset)
             │
             │  utils/data_utils.py
             ▼
   SentencePiece tokenizer — 16K shared vocab (English + Bengali)
             │
             ▼
   Token-batched DataLoader — ≤800 tokens/batch, sorted by length
             │
     ┌───────┴───────────────────────────────────────────┐
     │                                                   │
     ▼                                                   ▼
  src tokens                                        tgt tokens (shifted right)
  embeddings + positional_encoding                  embeddings + positional_encoding
     │                                                   │
     ▼                                                   │
  encoder.py                                             │
  N × (self-attn + FFN)                                  │
     │                                                   │
     │  memory (batch, src_len, d_model)                 │
     └──────────────────────────────────────▶  decoder.py
                                               N × (masked-self-attn
                                                  + cross-attn + FFN)
                                                         │
                                                         ▼
                                              logits (batch, tgt_len, vocab)
                                                         │
                                                         │  utils/loss.py
                                                         ▼
                                              label-smoothed cross-entropy
                                                         │
                                                         │  utils/optimizer.py
                                                         ▼
                                              Adam + Noam LR (step^-0.5)
                                                         │
                                                         │  utils/train_utils.py
                                                         ▼
                                              checkpoint + leaderboard.json
```

### Inference flow

```
"What are you saying?"  (English)
             │
             │  scripts/inference.py
             ▼
   SentencePiece tokenize  →  token IDs
             │
             │  models/transformer.py → run_encoder_stack()
             ▼
   Encoder  →  memory          ← computed ONCE per source sentence
             │
             ▼
   beam search — generate the Bengali one token at a time (beam_size=4, α=1.0):

      start: <sos>
        │
        ▼
      ┌──────────────────────────────────────────────────┐
      │  LOOP (one pass = one new token):                │
      │    1. Decoder(so-far, memory)  →  logits         │
      │    2. pick the top few next-word candidates      │
      │    3. keep the 4 best partial sentences (beams)  │
      └──────────────────────────────────────────────────┘
        │
        ▼  repeat the loop until each sentence hits <eos> (or 64 tokens)
        │
        ▼
   finished sentences  →  pick the best (score ÷ length^α)
             │
             ▼
   SentencePiece detokenize  →  "তুমি কি বলছ?"  (Bengali)
```

> The Gradio demo ([`scripts/app.py`](scripts/app.py)) is this same flow wrapped in a browser UI — it loads the model once at startup and calls the same `translate()`; the sliders just set `beam_size` and `α` per request.

### Evaluation flow

```
   config + tokenizer + checkpoint
             │
             │  scripts/evaluate.py
             ▼
   load val split  ──  SAME seed / max_rows / val_split as training
             │           → the exact pairs the model never trained on
             ▼
   cap to --max_samples (default 500; full val is too slow under beam search)
             │
             ▼
   for each (English src, Bengali ref):
        translate(src) via beam search   ← reuses scripts/inference.py
        collect  hypothesis + reference
             │
     ┌───────┴─────────────────────────────────────┐
     ▼                                             ▼
  Perplexity = exp(checkpoint val_loss)      Corpus BLEU = sacrebleu(hyps, refs)
  (intrinsic — full val set, stable)         (extrinsic — `intl` tokenizer for Bengali)
     │                                             │
     └───────┬─────────────────────────────────────┘
             ▼
   print BLEU + brevity penalty + sample translations
```

### Build order

Written strictly bottom-up — each layer only depends on what's below it:

```
1. models/modules/
   embeddings → positional_encoding → multi_head_attention → feed_forward → layer_norm

2. models/encoder.py + models/decoder.py
   stack the modules with residuals; decoder adds a cross-attention sub-layer

3. models/transformer.py
   wire encoder + decoder + output projection + weight tying

4. utils/
   config → mask_utils → loss → optimizer → data_utils → train_utils → logging_setup

5. scripts/
   train → inference → evaluate + app  (both reuse inference's translate())
   auto_resume_train.sh wraps train
```

---

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
