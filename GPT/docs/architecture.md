# Architecture & Configuration

Reference for the model, the config fields you'll actually touch, and the checkpoint
layout. For the paper-section → source-file map and the pipeline diagram, see the
[README](../README.md).

## Model specification

Every **method** follows the papers (GPT-2, with the training recipe from GPT-3
Appendix B); only the **scale** is reduced to fit a 16 GB M1. The width, depth and heads
are BERT's `base` exactly, so the two replications are the same size on the same text.

| | GPT-2 small (paper) | This replication | Config field |
|---|---|---|---|
| Layers (L) | 12 | **6** | `model.num_layers` |
| Hidden size (d) | 768 | **256** | `model.d_model` |
| Attention heads | 12 | **8** (32 dims/head) | `model.num_heads` |
| MLP inner size | 3072 (4·d) | **1024** (4·d) | `model.d_ff` |
| Context (`block_size`) | 1024 | **256** | `model.block_size` |
| Vocab (byte-level BPE) | 50,257 | **8,192** | `data.vocab_size` |
| Parameters | 124M | **6,901,760 (~6.9M)** | — |
| Training tokens | WebText, 40 GB | **111.4M** (11.0M-token corpus, ~10 passes) | `training.*` |

The forward pass:

```
ids (B, 256) → token + learned position embeddings → 6 × [ x + attn(LN(x)) ; x + mlp(LN(x)) ]
             → final LayerNorm → x · wteᵀ → logits (B, 256, 8192)
```

Same as the paper:

- **Pre-LN blocks**: LayerNorm at the input of each sub-layer (GPT-2 §2.3), plus a final
  LayerNorm after the last block.
- **Residual init**: the two output projections per block start at
  N(0, 0.02/√(2·L)) (GPT-2 §2.3); every other weight at N(0, 0.02) (GPT-1 §4.1).
- **Weight tying**: the output layer is the token embedding table, transposed.
- **Causal masked self-attention**, GELU MLP, learned positions, dropout 0.1.
- **Byte-level BPE** (GPT-2 §2.2), trained from scratch; never an unknown token.
- **AdamW, β (0.9, 0.95), weight decay 0.1, clip 1.0, warmup then cosine to 10%**
  (GPT-3 Appendix B; GPT-2 does not state its own).

**Deviations from the paper**, all forced by one 16 GB laptop and flagged inline in
[`configs/base.yaml`](../configs/base.yaml):

- **Hardware**: 1 × Apple M1 (16 GB), about 4 h 15 m.
- **Data**: Bengali Wikipedia (BERT's corpus, 16,213 articles), not 40 GB of English
  WebText.
- **Scale**: 6 layers × 256 wide instead of 12 × 768; context 256 instead of 1024
  (attention memory on MPS still grows with T², see [Performance](performance.md)).
- **Vocab**: 8,192 instead of 50,257, sized to the corpus.
- **Batch**: 65,536 tokens per step instead of 512 × 1024 = 524,288. At GPT-2's size one
  pass over the corpus would be 21 steps.
- **Warmup**: 100 steps. GPT-3's 375M-token warmup is longer than this whole run.

## Config reference

Three strict YAML files drive everything. Unknown keys, wrong types and out-of-range
values fail when the file loads, before any work starts. Paths are relative to the repo
root. The files are annotated with the paper section behind each value; this covers the
fields you'd change and the non-obvious ones.

### `configs/base.yaml`: pre-training

| Group | Key fields | Notes |
|---|---|---|
| `data` | `vocab_size` (8192), `val_split` (0.1), `source_path`, `corpus_path`, `train_path`, `val_path` | `vocab_size` must be ≤ 65,536: ids are stored as uint16. |
| `model` | `num_layers` (6), `d_model` (256), `num_heads` (8), `d_ff` (1024), `block_size` (256), `dropout` (0.1), `init_std` (0.02) | `block_size` is both the context length and the size of the position table, one field, so they cannot differ. |
| `model` | `attention_impl` (`"sdpa"`) | `"reference"` swaps in the hand-written attention from `transformer/` on the same weights, to check SDPA against it. |
| `training` | `batch_size` (32), `grad_accum_steps` (8), `max_steps` (1700), `eval_interval` (100) | Tokens per step = `batch_size × grad_accum_steps × block_size`. `batch_size` is what fits in memory; their product is the real batch. |
| `training` | `warmup_steps` (100), `min_lr_ratio` (0.1), `clip_grad_norm` (1.0) | `min_lr_ratio` must be > 0, so a resume with more steps always trains. |
| `optimizer` | `lr` (6.0e-4), `betas` ([0.9, 0.95]), `eps` (1.0e-8), `weight_decay` (0.1) | Write `6.0e-4`, not `6e-4`: YAML reads the second as a string. |
| `paths` | `tokenizer_dir`, `checkpoint_dir`, `log_dir` | Per-config subfolders (`base/`, `tiny/`). |
| top level | `seed` (42), `device` (`"auto"`) | The seed fixes the val split, the init and every batch. |

`configs/tiny.yaml` is the same shape at toy scale (2 layers, d 64, vocab 1,024, 200
steps) for pipeline smoke tests; see [Getting Started](getting-started.md).
`configs/gpt2-124m.yaml` is the paper's real size (124,439,808 parameters with this
code), kept as a reference for what `base` scales down from. It loads and builds, but
is far too slow to train on an M1.

### `configs/generate.yaml`: sampling

| Field | Notes |
|---|---|
| `max_new_tokens` | tokens to add at most; generation also stops when the model ends the article |
| `temperature` | ≥ 0. Below 1 is more predictable, above 1 more random; 0 = greedy, which repeats itself |
| `top_k` | keep the k most likely tokens (GPT-2's samples use 40); `null` = off |
| `top_p` | nucleus sampling: keep the smallest set with probability ≥ p; `null` = off |
| `num_samples` | continuations of the same prompt |
| `seed` | an integer repeats the same samples; `null` differs each run |

It is separate from the training config on purpose: the model is rebuilt from the
snapshot saved beside the checkpoint, and sampling values change how text is drawn,
not the model.

## Checkpoint & run layout

Each run writes its own timestamped folder, so a resume or a new run never overwrites
an earlier one:

```
checkpoints/<config>/
  run_<ts>/
    best.pt         # lowest-val checkpoint
    last.pt         # latest; resume from here
    config.yaml     # frozen copy of the config that produced this run
  leaderboard.json  # val loss of every run under this config
  best.pt           # symlink → the winning run's best.pt
logs/<config>/
  run_<ts>/
    train.log       # everything printed, plus any crash's traceback
    events.out.*    # TensorBoard: loss, lr, grad norm per step; train/val per eval
```

A checkpoint (82.9 MB for `base`) holds the weights, the optimizer and scheduler state,
the losses, the git commit, and two fingerprints: the sha256 of the tokenizer's
`merges.txt` and of `train.bin`. A resume refuses to continue if either file changed;
`evaluate` and `generate` refuse a changed tokenizer.

`evaluate.py` and `generate.py` take `--checkpoint` and read the `config.yaml` beside
it. The checkpoint describes itself, so there is no model config flag to get out of
sync.
