# BERT — "Pre-training of Deep Bidirectional Transformers", from scratch (Bengali)

A complete BERT from
[**"BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"** (Devlin et al., 2019)](https://arxiv.org/abs/1810.04805),
rebuilt in plain PyTorch — no HuggingFace models, no `nn.TransformerEncoder` — **pre-trained on
Bengali Wikipedia** and fine-tuned for Bengali news-topic classification, all on a single
**Apple M1 with 16 GB of RAM**.

Part of the *papers-from-scratch* portfolio — and a deliberate sibling of the
[`transformer/`](../transformer/) replication: BERT's encoder layer *is* the Transformer encoder
layer, so `MultiHeadAttention`, `LayerNorm`, and the padding mask are **imported from
`transformer/` unchanged** (the full ledger is in [What's hand-written vs reused](#whats-hand-written-vs-reused) below).

## Results

**86.5% test accuracy** on IndicGLUE `sna.bn` (Soham Bengali news, 6 classes) — directly comparable
to the published numbers on the same dataset and splits (Kakwani et al. 2020):

| approach | params | `sna.bn` test acc |
|---|---|---|
| IndicFT word embeddings + k-NN (no fine-tuning) | — | 71.82 |
| IndicBERT base, fine-tuned (ALBERT, 11 langs) | 12M | 78.45 |
| mBERT, fine-tuned (104 langs) | 110M | 80.23 |
| **this replication (from scratch, Bengali only)** | **~7.5M** | **86.5** |
| XLM-R base, fine-tuned (100 langs) | 125M | **87.60** |

A 7.5M-parameter model pre-trained in ~28 laptop-hours beats mBERT by ~6 points and IndicBERT by
~8, landing ~1.1 under XLM-R (~17× its size). Full analysis, confusion matrices, and honest
caveats: the [write-up](https://www.samyamoyrakshit.com/blog/bert-from-scratch/).

## Documentation

- **[Getting Started](docs/getting-started.md)** — install and verify the pipeline on the tiny config (~15 min).
- **[Training](docs/training.md)** — reproduce the 86.5% result end to end (pre-train → fine-tune → evaluate).
- **[Architecture & Configuration](docs/architecture.md)** — model spec, config field reference, checkpoint layout.
- **The story** — the full write-up, with the *why* behind every choice, lives on the blog:
  [BERT, from scratch](https://www.samyamoyrakshit.com/blog/bert-from-scratch/).

## What's hand-written vs reused

Hand-written here: the three-table input embeddings (token + segment + **learned** position, no
√d_model scaling), the GELU feed-forward (tanh approximation, matching Google's TF code), the
MLM + NSP heads, dynamic 80/10/10 masking, NSP pair construction, the joint MLM+NSP loss,
AdamW with linear warmup→decay, the WordPiece tokenizer training, both data pipelines, and the
full pretrain / finetune / evaluate / inference scripts.

Reused unchanged from [`transformer/`](../transformer/): `MultiHeadAttention`, `LayerNorm`,
`create_padding_mask` — because the paper reuses them too (§3: "based on the original
implementation described in Vaswani et al. (2017)").

The constraint is the hardware. The paper pre-trained BERT-base (L=12, H=768, A=12, 110M params)
on 16 Cloud TPUs for 4 days; this fits in 16 GB by scaling down — **L=6, H=256, A=8, d_ff=1024**
— while keeping every *method* exact (objectives, 80/10/10 masking, AdamW schedule, dropout 0.1,
learned positions, [CLS]/[SEP] packing).

## Architecture / Configs

| Config | d_model | layers | heads | d_ff | vocab | Purpose |
|--------|---------|--------|-------|------|-------|---------|
| `base.yaml` | 256 | 6 | 8 | 1024 | 10k | Main pre-training run (paper "base", scaled for 16 GB) |
| `tiny.yaml` | 64 | 2 | 4 | 256 | 4k | Fast pipeline sanity checks |
| `finetune.yaml` | (from the pretrain snapshot) | | | | | Fine-tuning on `sna.bn` — §A.3 hyperparameters (3 epochs, batch 32, lr sweep {5e-5, 3e-5, 2e-5}) |

Every knob is documented with its paper section (and every deviation flagged) in
[`configs/base.yaml`](configs/base.yaml).

## Setup

Python 3.12, managed with [`uv`](https://docs.astral.sh/uv/). From the **repo root** (the folder
above `BERT/`):

```bash
uv sync        # installs torch, datasets, tokenizers, gradio, … from pyproject.toml
```

Don't want the whole portfolio's dependencies? [`requirements.txt`](requirements.txt) is the
BERT-only subset — it covers every script here (pretrain → finetune → evaluate → inference →
app), and the modules reused from `transformer/` are pure torch, so nothing else is needed:

```bash
pip install -r BERT/requirements.txt   # BERT only — from the repo root, in your own venv
```

> **Weights are not in this repo.** Checkpoints, logs, the corpus, and the tokenizer are
> gitignored. Train them yourself (below) — the corpus and tokenizer are built by the first two
> commands.

## Usage

Run everything from the **repo root** as modules (the leading `BERT.` matters):

```bash
# 1. Build the corpus (streams Bengali Wikipedia → BERT/data/bn_wiki.txt, ~114 MB)
python -m BERT.scripts.prepare_corpus

# 2. Pre-train (MLM + NSP; ~28 h on an M1 — caffeinate keeps the Mac awake)
caffeinate -s uv run python -m BERT.scripts.pretrain
#    resume an interrupted run:
#    python -m BERT.scripts.pretrain --resume BERT/checkpoints/base/run_<ts>/last.pt

# 3. Fine-tune on IndicGLUE sna.bn (~9.5 min; rerun per lr for the sweep)
python -m BERT.scripts.finetune

# 4. Score the best checkpoint on the held-out test split
python -m BERT.scripts.evaluate

# 5. Classify one sentence (predicted topic + full softmax distribution)
python -m BERT.scripts.inference --text "কলকাতায় আজ বৃষ্টি হবে"

# 6. Gradio demo (browser UI, loads the model once)
uv run python -m BERT.scripts.app   # serves http://127.0.0.1:7860

# Watch training
tensorboard --logdir BERT/logs
```

Sanity-check the whole pipeline first on the tiny config:
`python -m BERT.scripts.pretrain --config BERT/configs/tiny.yaml`.

## Layout

```
models/          # bert.py (body) + encoder + heads + modules (embeddings, feed_forward)
utils/           # config, data_utils, masking (80/10/10), nsp, loss, optimizer, train loops,
                 # finetune_{config,data,utils}
scripts/         # prepare_corpus / pretrain / finetune / evaluate / inference / app (Gradio)
configs/         # base.yaml, tiny.yaml, finetune.yaml
docs/            # getting-started, training, architecture (see "Documentation")
```

## How the pieces fit together

### Paper → code

| Paper section | What it implements | File |
|---|---|---|
| §3.1 — input representation | token + segment + learned-position sum, LayerNorm + dropout | [`models/modules/embeddings.py`](models/modules/embeddings.py) |
| §A.2 — gelu activation | position-wise FFN with tanh-GELU | [`models/modules/feed_forward.py`](models/modules/feed_forward.py) |
| §3 — Transformer encoder | N × (self-attn + FFN), post-LayerNorm — attention/LN **reused from `transformer/`** | [`models/encoder.py`](models/encoder.py) |
| §3.1 — the body + pooler | embeddings → encoder → (sequence_output, pooled [CLS]) | [`models/bert.py`](models/bert.py) |
| §3.1 — MLM + NSP heads | tied-embedding MLM head, 2-way NSP head | [`models/heads.py`](models/heads.py) |
| §3.1 — MLM masking | dynamic 15% selection, 80% [MASK] / 10% random / 10% keep | [`utils/masking.py`](utils/masking.py) |
| §3.1 — NSP pairs | 50/50 IsNext/NotNext from document structure | [`utils/nsp.py`](utils/nsp.py) |
| §3.1 — joint loss | MLM cross-entropy (ignore_index) + NSP cross-entropy | [`utils/loss.py`](utils/loss.py) |
| §A.2 — optimizer | AdamW, linear warmup → linear decay, decay-group handling | [`utils/optimizer.py`](utils/optimizer.py) |
| §4.1 / §A.3 — fine-tuning | fresh `Linear(H, K)` on pooled [CLS], 2–4 epochs, small lr | [`models/bert_for_classification.py`](models/bert_for_classification.py) + [`scripts/finetune.py`](scripts/finetune.py) |

### The two stages

```
STAGE 1 — PRE-TRAIN (self-supervised, no labels)          ~28 h, once

  Bengali Wikipedia (wikimedia/wikipedia 20231101.bn)
        │  scripts/prepare_corpus.py → data/bn_wiki.txt (114 MB, article per block)
        ▼
  WordPiece tokenizer (10k vocab) + document/sentence structure
        │  utils/data_utils.py
        ▼
  [CLS] sentA [SEP] sentB [SEP]  pairs  (50% NotNext)  →  dynamic 80/10/10 masking
        │
        ▼
  BERTForPreTraining = body + MLM head + NSP head
        │  loss = MLM CE + NSP CE  →  AdamW, warmup→decay
        ▼
  checkpoints/base/run_<ts>/best.pt  (+ config.yaml snapshot, tokenizer sha256)

STAGE 2 — FINE-TUNE (supervised, ~9.5 min per run)

  best.pt ── transplant bert.* body, drop MLM/NSP heads ──►  + fresh Linear(256, 6)
        │
        ▼
  IndicGLUE sna.bn (11,284 train / 1,411 val / 1,411 test)
        │  [CLS] article [SEP], max_seq_len 128 — utils/finetune_data.py
        ▼
  3 epochs, lr sweep {5e-5, 3e-5, 2e-5} → winner 5e-5 (val 0.8533)
        │
        ▼
  checkpoints/finetune/sna_bn/best.pt symlink + leaderboard.json
        │
        ├──► scripts/evaluate.py   → test accuracy 0.865 + per-class report
        ├──► scripts/inference.py  → one sentence → topic + softmax
        └──► scripts/app.py        → the same, in a browser
```

### Build order

Written bottom-up, mirroring `transformer/`:

```
1. models/modules/   embeddings → feed_forward        (attention/LN come from transformer/)
2. models/           encoder → bert → heads → bert_for_pretraining
3. utils/            config → data_utils → masking → nsp → loss → optimizer → train_utils
4. scripts/          prepare_corpus → pretrain
5. fine-tune layer   bert_for_classification + finetune_{config,data,utils} → finetune
6. serving           evaluate → inference → app (both entrypoints share one loader)
```

---

## Citation

Replicates the original paper (citation is for the authors, not the implementer):

```bibtex
@inproceedings{devlin2019bert,
  title     = {BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  author    = {Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  booktitle = {Proceedings of NAACL-HLT},
  year      = {2019},
  url       = {https://arxiv.org/abs/1810.04805}
}
```
