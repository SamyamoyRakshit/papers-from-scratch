# Training

Reproduce the result from scratch: pre-train on Bengali Wikipedia, then fine-tune
on IndicGLUE `sna.bn` to **86.5% test accuracy**. Two stages, ~28 hours + ~10
minutes, on a single 16 GB Apple M1.

New here? Install first — see [Getting Started](getting-started.md). All commands
run from the **repo root** as modules (the leading `BERT.` matters).

## Stage 1 — Pre-train

### 1. Build the corpus

Streams Bengali Wikipedia (`wikimedia/wikipedia`, `20231101.bn`) to a flat text
file, one article per block. Reads `configs/base.yaml` by default.

```bash
python -m BERT.scripts.prepare_corpus        # → BERT/data/bn_wiki.txt (~114 MB)
```

### 2. Pre-train (MLM + NSP)

~28 hours on an M1. `caffeinate -s` keeps the Mac awake for the whole run.

```bash
caffeinate -s uv run python -m BERT.scripts.pretrain
```

Each invocation owns a fresh `checkpoints/base/run_<ts>/` with its own `best.pt`,
`last.pt`, and frozen `config.yaml`. Resume an interrupted run from its `last.pt`:

```bash
python -m BERT.scripts.pretrain --resume BERT/checkpoints/base/run_<ts>/last.pt
```

Watch it train in another terminal:

```bash
tensorboard --logdir BERT/logs
```

## Stage 2 — Fine-tune

### 3. Point the fine-tune config at your pre-trained run

Open [`configs/finetune.yaml`](../configs/finetune.yaml) and set the `pretrained:`
block to the run you just produced — both the checkpoint **and** its sibling
config snapshot (which carries the encoder dimensions):

```yaml
pretrained:
  checkpoint: "BERT/checkpoints/base/run_<ts>/best.pt"
  config:     "BERT/checkpoints/base/run_<ts>/config.yaml"
```

### 4. Fine-tune on `sna.bn`

Transplants the encoder body, drops the MLM/NSP heads, attaches a fresh
`Linear(256, 6)`, and trains ~9.5 minutes.

```bash
python -m BERT.scripts.finetune              # reads configs/finetune.yaml
```

BERT fine-tunes with a learning-rate sweep (§A.3: `{5e-5, 3e-5, 2e-5}`). Edit the
`optimizer.lr` field in `finetune.yaml` and rerun for each value — `leaderboard.json`
and the `best.pt` symlink one level up track the global best across runs. Our
winner was **5e-5** (val 0.8533).

## Evaluate and serve

`evaluate` and `inference` take only `--checkpoint` (default: the leaderboard
`best.pt` symlink) and read the config snapshot sitting beside it — no `--config`
flag.

```bash
# Score the best checkpoint on the held-out test split (+ per-class report)
python -m BERT.scripts.evaluate

# Classify one sentence → predicted topic + full softmax distribution
python -m BERT.scripts.inference --text "কলকাতায় আজ বৃষ্টি হবে"

# Gradio demo in the browser (loads the model once)
uv run python -m BERT.scripts.app            # http://127.0.0.1:7860
```

## What to expect

| Stage | Command | Time (M1, MPS) | Result |
|---|---|---|---|
| Pre-train | `pretrain` | ~28 h | MLM + NSP loss converged |
| Fine-tune | `finetune` (×3 for the lr sweep) | ~9.5 min each | val acc 0.8533 @ lr 5e-5 |
| Evaluate | `evaluate` | seconds | **test acc 0.865** |

Config field reference and the checkpoint/run layout: [Architecture](architecture.md). \
The *why* behind every choice: [BERT, from scratch](https://www.samyamoyrakshit.com/blog/bert-from-scratch/).
