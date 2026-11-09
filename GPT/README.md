# GPT — "Language Models are Unsupervised Multitask Learners", from scratch (Bengali)

A decoder-only GPT-2 from
[**"Language Models are Unsupervised Multitask Learners"** (Radford et al., 2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf),
rebuilt in plain PyTorch (no HuggingFace models, no `nn.TransformerDecoder`, no
tokenizer library) and **pre-trained on Bengali Wikipedia** on a single **Apple M1 with
16 GB of RAM**.

The third replication in this repo, and a deliberate sibling of the other two. It
trains on the **same articles** as [`BERT/`](../BERT/), at the **same size** (6 layers,
256 wide, 8 heads), so the encoder and the decoder are one controlled comparison. What
the siblings already built and GPT-2 did not change is **imported unchanged**:
`LayerNorm`, the reference attention and the causal mask from
[`transformer/`](../transformer/); config classes, `gelu` and the weight-decay split
from BERT.

## Results

Held-out loss on BERT's 1,621 validation articles, after 1,700 steps (111.4M tokens,
4 h 15 m on the M1):

| model | bits per token | **bits per byte** | perplexity |
|---|---|---|---|
| uniform over the 8,192 tokens | 13.000 | 1.3353 | 8,192 |
| unigram (token frequencies, no context) | 11.285 | 1.1592 | — |
| **this replication, 6.9M params** | **6.372** | **0.6545** | **82.8** |

Bits per byte is the comparable number: it divides by the size of the text, not by the
number of tokens, so it does not depend on the tokenizer (GPT-2 Table 3 reports the
same unit). It is not comparable with English results: a Bengali letter is 3 bytes in
UTF-8.

The model writes fluent, Wikipedia-shaped Bengali: names, dates in parentheses,
"was a Bengali poet", sentences that end in `।`. The facts in it are invented, and at
low temperature it falls into repetition loops. That is the expected result for 6.9M
parameters trained on 11M tokens; GPT-2's smallest model is 18 × larger and read 40 GB.

## Documentation

- **[Getting Started](docs/getting-started.md)**: install and verify the whole pipeline on the tiny config (~35 s).
- **[Training](docs/training.md)**: reproduce the result end to end (corpus → tokenizer → token files → pre-train → evaluate → generate).
- **[Architecture & Configuration](docs/architecture.md)**: model spec, config field reference, checkpoint layout.
- **[Performance](docs/performance.md)**: build-nanoGPT's speed techniques measured on an M1: what works on MPS, and what is slower there.

## What's hand-written vs reused

Hand-written here: the **byte-level BPE tokenizer** (pre-tokenizer regex fixed for
Bengali, byte↔unicode map, merge training, encode/decode), the token + learned-position
embeddings, causal self-attention with a fused QKV and SDPA, the GELU MLP, the **pre-LN**
decoder block, the full model with its final LayerNorm, tied output layer and 1/√N
residual init, the memmap token stream, the next-token loss, AdamW with warmup → cosine
to 10%, the step-based training loop with gradient accumulation, sampling (temperature,
top-k, top-p), and every script.

Reused unchanged from [`transformer/`](../transformer/): `LayerNorm`, `MultiHeadAttention`
and `create_causal_mask` (the reference attention path, checked against SDPA to
1.67e-06), and the strict config base. From [`BERT/`](../BERT/): the corpus file, the
`PathsConfig` / `OptimizerConfig` classes, `gelu`, `_decay_groups` and `set_seed`.

The constraint is the hardware. GPT-2's smallest model (L=12, d=768, 124M params) read
40 GB of WebText; this fits in 16 GB by scaling down to **L=6, d=256, 8 heads,
context 256, vocab 8,192** while keeping every *method*: pre-LN, the final LayerNorm,
the residual init, weight tying, byte-level BPE, and GPT-3's optimizer recipe (GPT-2
does not state one).

## Architecture / Configs

| Config | d_model | layers | heads | d_ff | context | vocab | Purpose |
|---|---|---|---|---|---|---|---|
| `base.yaml` | 256 | 6 | 8 | 1024 | 256 | 8,192 | The run: GPT-2's method at M1 scale, BERT's dimensions |
| `tiny.yaml` | 64 | 2 | 4 | 256 | 64 | 1,024 | Pipeline smoke test in seconds |
| `gpt2-124m.yaml` | 768 | 12 | 12 | 3072 | 1024 | 50,257 | Reference only: the paper's size (124,439,808 params), too slow for an M1 |
| `generate.yaml` | | | | | | | Sampling: temperature, top-k, top-p, samples, seed |

Every knob is documented with its paper section (and every deviation flagged) in
[`configs/base.yaml`](configs/base.yaml).

## Setup

Python 3.12, managed with [`uv`](https://docs.astral.sh/uv/). From the **repo root** (the
folder above `GPT/`):

```bash
uv sync        # installs torch, numpy, regex, pydantic, … from pyproject.toml
```

Don't want the whole portfolio's dependencies? [`requirements.txt`](requirements.txt) is
the GPT-only subset, with versions identical to `pyproject.toml`. It is enough to run
every script here, but GPT still imports from `transformer/`, `BERT/` and `common/`, so
check out the whole repo:

```bash
pip install -r GPT/requirements.txt   # GPT only — from the repo root, in your own venv
```

> **Weights are not in this repo.** Checkpoints, logs, the corpus, the token files and
> the tokenizer are gitignored. The commands below build all of them.

## Usage

Run everything from the **repo root** as modules (the leading `GPT.` matters):

```bash
# 0. BERT's corpus, if it isn't built yet (streams Bengali Wikipedia → BERT/data/bn_wiki.txt)
python -m BERT.scripts.prepare_corpus

# 1. Add <|endoftext|> between articles → GPT/data/bn_wiki.txt
python -m GPT.scripts.prepare_corpus

# 2. Train the byte-level BPE (8,192 tokens, ~5 min)
python -m GPT.scripts.train_tokenizer

# 3. Encode once → GPT/data/train.bin, val.bin (2-byte ids, ~25 s)
python -m GPT.scripts.prepare_data

# 4. Pre-train (~4 h 15 m on an M1; caffeinate keeps the Mac awake)
caffeinate -s python -m GPT.scripts.pretrain
#    resume an interrupted run:
#    python -m GPT.scripts.pretrain --resume GPT/checkpoints/base/run_<ts>/last.pt

# 5. Loss, perplexity and bits per byte on the held-out articles
python -m GPT.scripts.evaluate

# 6. Continue a prompt (sampling settings in configs/generate.yaml)
python -m GPT.scripts.generate --prompt "রবীন্দ্রনাথ ঠাকুর"

# Watch training
tensorboard --logdir GPT/logs
```

Sanity-check the whole pipeline first on the tiny config: see
[Getting Started](docs/getting-started.md).

## Layout

```
bpe/             # byte-level BPE from scratch: pretokenize (Bengali-fixed regex), byte_coder, bpe
models/          # gpt.py (full model) + decoder.py (pre-LN blocks) + modules (embeddings,
                 # causal_self_attention, feed_forward)
utils/           # config, generate_config, data_utils (memmap), loss, optimizer, train_utils, sampling
scripts/         # prepare_corpus / train_tokenizer / prepare_data / pretrain / evaluate / generate
configs/         # base.yaml, tiny.yaml, gpt2-124m.yaml, generate.yaml
docs/            # getting-started, training, architecture, performance (see "Documentation")
```

## How the pieces fit together

### Paper → code

| Paper section | What it implements | File |
|---|---|---|
| GPT-2 §2.2: input representation | byte-level BPE, 256 bytes + merges + `<|endoftext|>`, never an unknown token | [`bpe/`](bpe/) |
| GPT-1 §3.1 Eq. 2, §4.1 | token + learned position embeddings, summed, dropout | [`models/modules/embeddings.py`](models/modules/embeddings.py) |
| GPT-1 §3.1: masked self-attention | fused QKV, causal SDPA; transformer/'s attention as the reference path | [`models/modules/causal_self_attention.py`](models/modules/causal_self_attention.py) |
| GPT-1 §4.1: GELU | position-wise MLP, 4·d inner size | [`models/modules/feed_forward.py`](models/modules/feed_forward.py) |
| GPT-2 §2.3: LayerNorm moved to the sub-block input | pre-LN block: `x + attn(LN(x))`, `x + mlp(LN(x))` | [`models/decoder.py`](models/decoder.py) |
| GPT-2 §2.3; GPT-1 Eq. 2 | final LayerNorm, 1/√N residual init, output tied to the token table | [`models/gpt.py`](models/gpt.py) |
| GPT-1 §3.1 Eq. 1 | next-token cross-entropy at every position | [`utils/loss.py`](utils/loss.py) |
| GPT-3 Appendix B | AdamW β (0.9, 0.95), wd 0.1, warmup → cosine to 10% | [`utils/optimizer.py`](utils/optimizer.py) |
| GPT-2 §2.3: batch 512 | gradient accumulation, step-based loop, clip 1.0 | [`utils/train_utils.py`](utils/train_utils.py) |
| GPT-2 §3: top-k samples | temperature, top-k, top-p, stop at `<|endoftext|>` | [`utils/sampling.py`](utils/sampling.py) |
| GPT-2 Table 3: bits per byte | tokenizer-independent evaluation | [`scripts/evaluate.py`](scripts/evaluate.py) |

### The pipeline

```
  BERT/data/bn_wiki.txt   (Bengali Wikipedia, 16,213 articles — shared with BERT)
        │  scripts/prepare_corpus.py   → <|endoftext|> after each article
        ▼
  GPT/data/bn_wiki.txt
        │  scripts/train_tokenizer.py  → byte-level BPE, 8,192 tokens (merges.txt)
        │  scripts/prepare_data.py     → 90/10 split by article (BERT's val articles), encode once
        ▼
  train.bin (11.0M ids) · val.bin (1.2M ids)       uint16, read through np.memmap
        │  random 256-token windows; y = x shifted by one
        ▼
  GPT = embeddings → 6 pre-LN blocks → LN → tied head        6.9M params
        │  next-token loss · 8 micro-batches per step · AdamW, warmup → cosine to 10%
        ▼
  checkpoints/base/run_<ts>/best.pt  (+ config.yaml snapshot, tokenizer + data sha256)
        │
        ├──► scripts/evaluate.py   → 4.4166 nats/token · perplexity 82.8 · 0.6545 bits/byte
        └──► scripts/generate.py   → prompt → continuation (temperature, top-k, top-p)
```

### Build order

Written bottom-up, mirroring `transformer/` and `BERT/`:

```
1. data           prepare_corpus → bpe/ (pretokenize → byte_coder → bpe) → train_tokenizer → prepare_data
2. config         utils/config → data_utils
3. models/        modules (embeddings → feed_forward → causal_self_attention) → decoder → gpt
4. training       loss → optimizer → train_utils → pretrain
5. using it       evaluate → sampling + generate_config → generate
```

---

## Citation

Replicates the original paper (citation is for the authors, not the implementer):

```bibtex
@techreport{radford2019language,
  title       = {Language Models are Unsupervised Multitask Learners},
  author      = {Radford, Alec and Wu, Jeffrey and Child, Rewon and Luan, David and Amodei, Dario and Sutskever, Ilya},
  institution = {OpenAI},
  year        = {2019},
  url         = {https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf}
}
```
