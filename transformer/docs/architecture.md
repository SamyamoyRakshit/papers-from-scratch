# Architecture & Configuration

Reference for the model dimensions, the config fields you'll actually touch, and
the checkpoint layout. For the paper-section → source-file map and the
train/inference/evaluate flow diagrams, see the [README](../README.md); for *why*
each choice was made, the blog:
[Rebuilding "Attention Is All You Need" From Scratch on a Mac M1 with 16GB of RAM](https://www.samyamoyrakshit.com/blog/attention-from-scratch/).

## Model specification

Every **method** matches the paper (Vaswani et al., 2017); only the **scale** is
reduced to fit 16 GB of unified memory.

| | Transformer base (paper) | This replication | Config field |
|---|---|---|---|
| Layers (N) | 6 | **4** | `model.num_layers` |
| Model dim (d_model) | 512 | **256** | `model.d_model` |
| Attention heads (h) | 8 | 8 (32 dims/head) | `model.num_heads` |
| Feed-forward (d_ff) | 2048 | **1024** | `model.d_ff` |
| Vocab (SentencePiece) | ~37k | **16k** | `data.vocab_size` |
| Parameters | ~65M | **~11M** | — |

Same everywhere: dropout 0.1, label smoothing 0.1, warmup 4000 steps, Adam
`betas [0.9, 0.98]`, `eps 1e-9`, √d_model embedding scaling, and weight tying
between the input embedding and the output projection.

**Deviations from the paper**, all forced by the single-GPU budget and flagged
inline in [`configs/base.yaml`](../configs/base.yaml):

- **Hardware** — 1× Apple M1 (16 GB) vs 8× P100 GPUs.
- **Data** — AI4Bharat Samanantar En→Bn (500K-pair subset) vs WMT'14 En→De (4.5M).
- **Batch** — ~800 tokens/batch vs ~25,000.
- **Sequence length** — capped at 64 tokens.

## Config reference

Two strict, repo-root-relative YAML files drive everything. Both are heavily
annotated with the paper section behind each value — read the files for the full
list; this covers the fields you'd change and the non-obvious ones.

### `configs/base.yaml` — main runs

| Group | Key fields | Notes |
|---|---|---|
| `model` | `d_model`, `num_heads`, `d_ff`, `num_layers` | The scale knobs (see table above). |
| `training` | `num_epochs` (30), `max_tokens_per_batch` (800), `max_seq_len` (64), `warmup_steps` (4000), `label_smoothing` (0.1) | Batches are token-based, not fixed row counts. |
| `optimizer` | `betas` `[0.9, 0.98]`, `eps` (1e-9) | **No `lr` field** — the Noam schedule derives the learning rate from `d_model` and `warmup_steps` (§5.3). |
| `data` | `dataset` (`ai4bharat/samanantar`), `src_lang`/`tgt_lang` (`en`/`bn`), `max_rows` (500000), `vocab_size` (16000) | `filter_max_ratio` / `filter_min_words` drop skewed or empty pairs. |
| `inference` | `beam_size` (4), `length_penalty` (1.0) | Decode knobs — change and re-decode without retraining (see [Training](training.md)). |
| `paths` | `checkpoint_dir`, `log_dir`, `tokenizer_path` | Per-config subdirs (`base/`, `tiny/`). |

`configs/tiny.yaml` is the same shape at toy scale (2 layers, 1,000 pairs) for
pipeline smoke tests — see [Getting Started](getting-started.md).

## Checkpoint & run layout

Each run writes its own timestamped directory, so a resume never clobbers a prior
run's outputs:

```
checkpoints/<config>/
  run_<ts>/
    best.pt         # best-val checkpoint
    last.pt         # latest — resume from here
    config.yaml     # frozen snapshot of the config that produced this run
    train.log
  leaderboard.json  # ranks all runs under this config
```

`evaluate.py` and `inference.py` require **both** `--config` and `--checkpoint` —
pass the run's `config.yaml` snapshot (or the source config; same content)
alongside its `best.pt`.
