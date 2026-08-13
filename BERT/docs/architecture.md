# Architecture & Configuration

Reference for the model dimensions, the config fields you'll actually touch, and
the checkpoint layout. For the paper-section → source-file map and the two-stage
pipeline diagram, see the [README](../README.md); for *why* each choice was made,
the blog: [BERT, from scratch](https://www.samyamoyrakshit.com/blog/bert-from-scratch/).

## Model specification

Every **method** matches the paper (Devlin et al., 2019); only the **scale** is
reduced to fit 16 GB of unified memory.

| | BERT-base (paper) | This replication | Config field |
|---|---|---|---|
| Layers (L) | 12 | **6** | `model.num_layers` |
| Hidden size (H) | 768 | **256** | `model.d_model` |
| Attention heads (A) | 12 | **8** (32 dims/head) | `model.num_heads` |
| Feed-forward (intermediate) | 3072 | **1024** (4·H) | `model.d_ff` |
| Vocab (WordPiece) | 30k | **10k** | `data.vocab_size` |
| Max positions | 512 | 512 | `model.max_position_embeddings` |
| Parameters | 110M | **7,573,266 (~7.5M)** | — |

Same everywhere: dropout 0.1, learned position embeddings, [CLS]/[SEP] packing,
80/10/10 MLM masking, 50/50 NSP pairs, AdamW with linear warmup → decay.

**Deviations from the paper**, all forced by the single-GPU budget and flagged
inline in [`configs/base.yaml`](../configs/base.yaml):

- **Hardware** — 1× Apple M1 (16 GB) vs 16 Cloud TPUs / 4 days.
- **Data** — Bengali Wikipedia (`20231101.bn`) vs BooksCorpus + English Wikipedia.
- **Sequence length** — fixed 128 vs the paper's staged 128 → 512.
- **Batch size** — 32 vs 256.
- **Vocab** — 10k (smaller corpus → rare-token rows actually get trained).

## Config reference

Two strict, repo-root-relative YAML files drive everything. Both are heavily
annotated with the paper section behind each value — read the files for the full
list; this covers the fields you'd change and the non-obvious ones.

### `configs/base.yaml` — pre-training

| Group | Key fields | Notes |
|---|---|---|
| `model` | `d_model`, `num_heads`, `d_ff`, `num_layers` | The scale knobs (see table above). |
| `training` | `num_epochs` (10), `batch_size` (32), `max_seq_len` (128), `warmup_steps` (10000) | `val_split: 0.1` holds out 10% for validation. |
| `optimizer` | `lr` (1e-4), `weight_decay` (0.01), `betas`, `eps` | AdamW, §A.2. |
| `data` | `wiki_dump` (`20231101.bn`), `max_articles` (20000), `vocab_size` (10000) | `max_articles: null` uses the full dump. |
| `paths` | `tokenizer_dir`, `checkpoint_dir`, `log_dir` | Per-config subdirs (`base/`, `tiny/`). |

`configs/tiny.yaml` is the same shape at toy scale (2 layers, ~500 articles) for
pipeline smoke tests — see [Getting Started](getting-started.md).

### `configs/finetune.yaml` — fine-tuning

| Group | Key fields | Notes |
|---|---|---|
| `pretrained` | `checkpoint`, `config` | **You must set these** to your pre-train run (see [Training](training.md)). |
| `data` | `dataset_id` (`ai4bharat/indic_glue`), `subset` (`sna.bn`) | Config-driven — no dataset IDs hardcoded in code. `num_labels: null` → inferred from the train split. |
| `training` | `num_epochs` (3), `batch_size` (32), `warmup_ratio` (0.1), `class_weighting` (false) | 2–4 epochs per §A.3. `warmup_ratio` (not fixed steps) because fine-tune runs are short. |
| `optimizer` | `lr` (5e-5) | Sweep `{5e-5, 3e-5, 2e-5}` by editing this and rerunning. |

## Checkpoint & run layout

Each run writes its own timestamped directory, so a resume or a new sweep value
never clobbers a prior run:

```
checkpoints/<task>/
  run_<ts>/
    best.pt         # best-val checkpoint
    last.pt         # latest — resume from here
    config.yaml     # frozen snapshot of the config that produced this run
  leaderboard.json  # ranks all runs under this task
  best.pt           # symlink → the winning run's best.pt
```

`evaluate.py` and `inference.py` take only `--checkpoint` and read the
`config.yaml` snapshot sitting beside it — the checkpoint is self-describing, so
there's no `--config` flag to get out of sync.
