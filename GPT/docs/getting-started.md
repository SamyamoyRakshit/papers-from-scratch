# Getting Started

Install the project and prove the whole GPT pipeline runs end to end on the **tiny**
config: a 2-layer, 170K-parameter model on 355 articles. Six commands, about 35 seconds
in total once the corpus exists. This is a smoke test, not a real model; when it
passes, move on to [Training](training.md) for the full run.

## Prerequisites

- **Python 3.12**
- [`uv`](https://docs.astral.sh/uv/) for dependency management (or pip, below)
- A few GB of free disk (HuggingFace caches the Wikipedia dump when the corpus is built)
- A GPU is optional: the code runs on Apple MPS, CUDA, or CPU (`device: "auto"` in the
  config picks the best available)

## 1. Install

From the **repo root** (the folder above `GPT/`):

```bash
uv sync        # installs torch, numpy, regex, pydantic, … from pyproject.toml
```

GPT-only subset, if you don't want the whole portfolio's dependencies:

```bash
pip install -r GPT/requirements.txt   # from the repo root, in your own venv
```

GPT imports a few modules from `transformer/`, `BERT/` and `common/` unchanged, so it
needs the whole repo checked out, not just the `GPT/` folder.

## 2. The corpus comes from BERT

GPT trains on the same Bengali Wikipedia articles as the BERT replication and reads
BERT's corpus file. If `BERT/data/bn_wiki_tiny.txt` does not exist yet, build it once:

```bash
python -m BERT.scripts.prepare_corpus BERT/configs/tiny.yaml   # → BERT/data/bn_wiki_tiny.txt (355 articles)
```

## 3. Sanity-check the pipeline

Everything runs from the **repo root** as a module; the leading `GPT.` matters.

```bash
python -m GPT.scripts.prepare_corpus  GPT/configs/tiny.yaml   # adds <|endoftext|> between articles      < 1 s
python -m GPT.scripts.train_tokenizer GPT/configs/tiny.yaml   # byte-level BPE, 1,024 tokens              ~10 s
python -m GPT.scripts.prepare_data    GPT/configs/tiny.yaml   # → train_tiny.bin / val_tiny.bin           ~3 s
python -m GPT.scripts.pretrain --config GPT/configs/tiny.yaml # 200 steps                                 ~12 s
python -m GPT.scripts.evaluate --checkpoint GPT/checkpoints/tiny/best.pt                      #  ~2 s
python -m GPT.scripts.generate --checkpoint GPT/checkpoints/tiny/best.pt --prompt "কলকাতা"   #  ~5 s
```

Note the two styles: the three preparation scripts take the config as a plain
argument, `pretrain` takes `--config`, and `evaluate` / `generate` take only
`--checkpoint` (they read the config saved beside it).

The pipeline is healthy if:

- `pretrain`'s loss falls (from about 7 to about 6) and it ends with a line like
  `Step 200/200 … | Train 5.9669 | Val 5.9217 (ppl 373.0)`. The run is deterministic:
  the same corpus on the same device repeats these numbers exactly. Yours should be
  close.
- `evaluate` prints the same val loss, plus perplexity and bits per byte (here
  `perplexity 373.0 | 8.543 bits/token | 1.4942 bits/byte`).
- `generate` prints Bengali characters. At this size they are not words yet; that is
  expected.

## 4. Where things land

The commands above create everything else; nothing here is committed to git.

| Path | What |
|---|---|
| `GPT/data/` | `bn_wiki_tiny.txt` (the corpus with end markers), `train_tiny.bin`, `val_tiny.bin` (token ids, 2 bytes each) |
| `GPT/tokenizer/tiny/` | the trained BPE: `merges.txt`, `vocab.json` |
| `GPT/checkpoints/tiny/run_<ts>/` | `best.pt`, `last.pt`, and a frozen `config.yaml` snapshot |
| `GPT/checkpoints/tiny/` | `leaderboard.json` and a `best.pt` symlink to the best run |
| `GPT/logs/tiny/run_<ts>/` | `train.log` and TensorBoard event files (`tensorboard --logdir GPT/logs`) |

`tiny` and `base` write to separate folders, so this test never touches the real run.

## Next

- **[Training](training.md)**: the real run on the `base` config.
- **[Architecture](architecture.md)**: model spec, config field reference, checkpoint layout.
- **[Performance](performance.md)**: what speeds GPT training up on an M1, and what doesn't.
