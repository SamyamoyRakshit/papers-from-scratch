# Training

Reproduce the result from scratch: train the `base` model to translate English →
Bengali on AI4Bharat Samanantar. The 30-epoch run reaches **val loss 6.004**
(epoch 24) → perplexity ≈ **405**, and **BLEU 0.17** with tuned decoding — on a
single 16 GB Apple M1.

New here? Install first — see [Getting Started](getting-started.md). All commands
run from the **repo root** as modules (the leading `transformer.` matters).

## 1. Train

`base` is the paper's "base", scaled for 16 GB (~11M params, 4 layers, ~15K
steps/epoch). The first run downloads Samanantar and trains the SentencePiece
tokenizer inline — no separate prep step.

```bash
python -m transformer.scripts.train --config transformer/configs/base.yaml
```

Each invocation owns a fresh `checkpoints/base/run_<ts>/` with its own `best.pt`,
`last.pt`, and frozen `config.yaml`. Resume an interrupted run from its `last.pt`:

```bash
python -m transformer.scripts.train --config transformer/configs/base.yaml \
    --resume transformer/checkpoints/base/run_<ts>/last.pt
```

MPS cache pressure builds over a long run and can OOM. The crash-resilient wrapper
restarts the process (freeing the cache) and resumes the latest run automatically,
up to a target epoch:

```bash
bash transformer/scripts/auto_resume_train.sh transformer/configs/base.yaml 30
```

Watch it train in another terminal:

```bash
tensorboard --logdir transformer/logs/base
```

## 2. Evaluate

Unlike a self-describing checkpoint, `evaluate` needs **both** `--config` and
`--checkpoint` (pass the run's `config.yaml` snapshot, or the source config —
same content). Beam search over the full val set is slow, so it caps samples.

```bash
python -m transformer.scripts.evaluate \
    --config transformer/checkpoints/base/run_<ts>/config.yaml \
    --checkpoint transformer/checkpoints/base/run_<ts>/best.pt \
    --max_samples 500
```

Prints corpus BLEU (sacreBLEU `intl`), perplexity, and a few sample translations.

## 3. Translate & serve

```bash
# One sentence (REPL if --text is omitted)
python -m transformer.scripts.inference \
    --config transformer/checkpoints/base/run_<ts>/config.yaml \
    --checkpoint transformer/checkpoints/base/run_<ts>/best.pt \
    --text "What are you saying?"

# Gradio demo with live beam / α sliders (loads the model once)
uv run python -m transformer.scripts.app     # http://127.0.0.1:7860
```

## Decoding matters

BLEU depends heavily on the decode settings in the config's `inference:` block.
The under-trained model emits `<eos>` early, so rewarding longer output helps:

| Decode setting | BLEU (sacreBLEU `intl`) | Length ratio |
|---|---|---|
| `beam=2, α=0.6` (paper defaults) | 0.07 | 0.349 |
| `beam=4, α=1.0` (tuned) | **0.17** | 0.565 |

Change `inference.beam_size` and `inference.length_penalty` in the config to
re-decode without retraining.

## What to expect

| Stage | Command | Result |
|---|---|---|
| Train | `train` (30 epochs) | best val loss **6.004** @ epoch 24 → PPL ≈ 405 |
| Evaluate | `evaluate --max_samples 500` | **BLEU 0.17** (beam=4, α=1.0) |

BLEU is humble by design — production En–Bn scores land in the 20s–30s. This is a
*correct, from-scratch* Transformer on a laptop, with the ≈6.6× fewer training
tokens and ~31× smaller batch reported openly. The full breakdown is on the blog:
[Attention Is All You Need, from scratch](https://www.samyamoyrakshit.com/blog/attention-from-scratch/).

Config field reference and checkpoint layout: [Architecture](architecture.md).
