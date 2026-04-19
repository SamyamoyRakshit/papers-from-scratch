## Table of Contents

1. [Overview — The Training Pipeline](#overview--the-training-pipeline)
2. [One Batch — Step by Step](#one-batch--step-by-step)
3. [Code Explanation](#code-explanation)
   - [`train_on_epoch`](#train_on_epoch)
   - [`validate`](#validate)
   - [`train`](#train)
4. [Teacher Forcing — Why `tgt[:-1]` and `tgt[1:]`](#teacher-forcing--why-tgt-1-and-tgt1)
5. [Gradient Clipping — Preventing Exploding Gradients](#gradient-clipping--preventing-exploding-gradients)
   - [The Exploding Gradient Problem](#the-exploding-gradient-problem)
   - [Two Types of Clipping — Norm vs Value](#two-types-of-clipping--norm-vs-value)
   - [Norm Clipping — How It Works (Our Approach)](#norm-clipping--how-it-works-our-approach)
   - [Value Clipping — How It Works (Alternative)](#value-clipping--how-it-works-alternative)
   - [Why Norm Clipping Is Better — Preserving Direction](#why-norm-clipping-is-better--preserving-direction)
   - [What `max_norm = 1.0` Means in Our Config](#what-max_norm--10-means-in-our-config)
   - [PyTorch Source — What Actually Happens](#pytorch-source--what-actually-happens)
6. [Loss Tracking — Why Multiply Back by `n_tokens`](#loss-tracking--why-multiply-back-by-n_tokens)
7. [`model.train()` vs `model.eval()` — What Changes](#modeltrain-vs-modeleval--what-changes)
8. [`@torch.no_grad()` — Why Validation Skips Gradients](#torchno_grad--why-validation-skips-gradients)
9. [Checkpointing — `best.pt` vs `last.pt`](#checkpointing--bestpt-vs-lastpt)
   - [What Gets Saved](#what-gets-saved)
   - [Why Save Optimizer and Scheduler State](#why-save-optimizer-and-scheduler-state)
   - [Overfitting Detection — Train vs Val Loss](#overfitting-detection--train-vs-val-loss)
10. [Progress Bar — tqdm](#progress-bar--tqdm)
11. [Full Training Output — What You'll See](#full-training-output--what-youll-see)
12. [References](#references)

---

# Overview — The Training Pipeline

The training pipeline has three layers:

```
train()                          ← outermost: loops over epochs
  └── train_on_epoch()           ← middle: loops over batches, updates weights
  └── validate()                 ← middle: loops over val batches, no weight updates
```

One full training run:

```
train() called once
  │
  ├── Epoch 1
  │     ├── train_on_epoch()  →  e.g. ~1237 batches  →  returns train_loss
  │     ├── validate()        →  e.g. ~138 batches   →  returns val_loss
  │     └── save checkpoint
  │
  ├── Epoch 2
  │     ├── train_on_epoch()  →  e.g. ~1237 batches  →  returns train_loss
  │     ├── validate()        →  e.g. ~138 batches   →  returns val_loss
  │     └── save checkpoint
  │
  ...
  │
  └── Epoch 30
        ├── train_on_epoch()  →  e.g. ~1237 batches  →  returns train_loss
        ├── validate()        →  e.g. ~138 batches   →  returns val_loss
        └── save checkpoint
```

Where do the example numbers come from?

```
e.g. 500K sentence pairs (from config: max_rows = 500000)
  → 90% train = ~450K pairs  (val_split = 0.1 in data_utils.py)
  → 10% val   = ~50K pairs

max_tokens_per_batch = 8000, avg sentence ≈ 22 tokens
  → train: ~450K × 22 / 8000 ≈ ~1237 batches (varies with padding overhead)
  → val:   ~50K × 22 / 8000  ≈ ~138 batches

These numbers will vary depending on your data split and sentence lengths.
```

---

# One Batch — Step by Step

Each batch goes through 8 steps inside `train_on_epoch`:

```
Step 1: Move to device       src, tgt → GPU/MPS
Step 2: Teacher forcing       tgt_input = tgt[:, :-1], tgt_output = tgt[:, 1:]
Step 3: Create masks          src_mask, tgt_mask, memory_mask
Step 4: Forward pass          logits = model(src, tgt_input, masks)
Step 5: Compute loss          flatten → LabelSmoothedLoss → scalar loss
Step 6: Backward pass         zero_grad → loss.backward → gradients computed
Step 7: Gradient clipping     clip_grad_norm_ → scale down if too large
Step 8: Update weights        optimizer.step → scheduler.step → next LR
```

Visual flow:

```
src ──→ Encoder ──→ encoder_output ──┐
                                     │
tgt_input ──→ Decoder ───────────────┘──→ logits ──→ flatten ──→ loss
                                                                  │
                                                           loss.backward()
                                                                  │
                                                            ∂loss/∂weights
                                                                  │
                                                         clip_grad_norm_()
                                                                  │
                                                          optimizer.step()
                                                                  │
                                                         weights updated
```

---

# Code Explanation

## `train_on_epoch`

```python
def train_on_epoch(
        model: nn.Module,
        train_loader,
        criterion: nn.Module,           # LabelSmoothedLoss
        optimizer,                       # Adam
        scheduler,                       # LambdaLR with TransformerScheduler
        pad_idx: int,                    # 0
        clip_grad_norm: float,           # 1.0
        device: torch.device,            # "mps" or "cuda" or "cpu"
        epoch: int,                      # current epoch (1, 2, ..., 30)
        num_epochs: int                  # total epochs (30)
) -> float:                              # returns average loss for this epoch
```

**What it does:** Processes all training batches in one epoch. For each batch: forward pass → compute loss → backward pass → clip gradients → update weights → update learning rate.

**Key lines:**

```python
model.train()           # Enables dropout (only module affected — no BatchNorm in our model)
```

```python
pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
```

Wraps the dataloader with a progress bar. `tqdm` internally tracks `batch_idx` — no need for `enumerate`. Output:

```
Epoch 1/30:  40%|████████          | 495/1237 [00:55<01:23, loss=2.18, lr=1.85e-04]
```

```python
optimizer.zero_grad()   # Clear previous batch's gradients (without this, gradients accumulate)
loss.backward()         # Compute ∂loss/∂weight for every parameter
```

`zero_grad()` is necessary because PyTorch **accumulates** gradients by default. Without it:

```
Batch 1: gradient = 0.5      → accumulated = 0.5 (correct)
Batch 2: gradient = 0.3      → accumulated = 0.5 + 0.3 = 0.8 (wrong! should be 0.3)
Batch 3: gradient = 0.2      → accumulated = 0.8 + 0.2 = 1.0 (wrong! should be 0.2)

With zero_grad():
Batch 1: zero → gradient = 0.5  → accumulated = 0.5 (correct)
Batch 2: zero → gradient = 0.3  → accumulated = 0.3 (correct)
Batch 3: zero → gradient = 0.2  → accumulated = 0.2 (correct)
```

```python
optimizer.step()        # Update weights: w = w - lr × gradient
scheduler.step()        # Update lr using warmup→decay formula (see optimizer.md)
```

Order matters — `optimizer.step()` uses the **current** LR to update weights, then `scheduler.step()` computes the **next** LR for the next batch.

```python
pbar.set_postfix(loss=..., lr=...)    # Updates the right side of the progress bar
```

## `validate`

```python
@torch.no_grad()                    # Decorator: disables gradient computation
def validate(model, val_loader, criterion, pad_idx, device) -> float:
    model.eval()                    # Disables dropout
```

**Same as `train_on_epoch` but without:**
- No `loss.backward()` — no gradient computation
- No `optimizer.step()` — no weight updates
- No `scheduler.step()` — no LR changes
- No `clip_grad_norm_` — no gradients to clip
- No `tqdm` — validation is fast, just returns final loss

**Why both `@torch.no_grad()` and `model.eval()`?**

They do **different things**:

```
model.eval()        → changes MODEL behavior (disables dropout)
@torch.no_grad()    → changes PYTORCH behavior (doesn't track gradients)
```

Both are needed. Without `@torch.no_grad()`, PyTorch still builds the computation graph and stores intermediate tensors for backprop — wasting memory and time, even though we never call `backward()`.

## `train`

```python
def train(model, train_loader, val_loader, criterion, optimizer, scheduler,
          pad_idx, clip_grad_norm, device, num_epochs, checkpoint_dir) -> None:
```

**What it does:** The outermost loop — runs `num_epochs` (30) epochs, validates after each, saves checkpoints.

```python
os.makedirs(checkpoint_dir, exist_ok=True)   # Create checkpoints/ if it doesn't exist
best_val_loss = float("inf")                  # Initialize to infinity — any real loss is better
```

```python
for epoch in range(1, num_epochs+1):          # 1, 2, 3, ..., 30
    train_loss = train_on_epoch(...)
    val_loss = validate(...)
```

---

# Teacher Forcing — Why `tgt[:-1]` and `tgt[1:]`

During training, the decoder gets the **correct** previous tokens as input instead of its own predictions. This is called **teacher forcing**:

```
Full target: [<sos>, আমি, AI, ভালোবাসি, <eos>]
              idx:  1     3   4    5       2

tgt_input  = tgt[:, :-1] = [<sos>, আমি, AI, ভালোবাসি]    ← remove last
tgt_output = tgt[:, 1:]  = [আমি, AI, ভালোবাসি, <eos>]    ← remove first
```

At each position, the decoder sees everything before and predicts the next token:

```
Decoder sees:           Should predict:
[<sos>]              → আমি
[<sos>, আমি]         → AI
[<sos>, আমি, AI]     → ভালোবাসি
[<sos>, আমি, AI, ভালোবাসি] → <eos>
```

**Why not let the decoder use its own predictions?** Because early in training, the model's predictions are garbage. If it predicts wrong at position 1, every subsequent position gets wrong input — errors compound. Teacher forcing gives it correct context so it can learn the **real** patterns.

**Logits shape:** `(batch_size, tgt_seq_len-1, vocab_size)` — `tgt_seq_len-1` because we removed the last token from `tgt_input`. The model produces one prediction per input position = `len(tgt_input)` predictions.

---

# Gradient Clipping — Preventing Exploding Gradients

## The Exploding Gradient Problem

During backpropagation, gradients flow **backward** through every layer. Because of the **chain rule**, the gradient at each layer = gradient from the layer above **×** the local gradient of that layer:

```
Forward:  input → layer 1 → layer 2 → ... → layer 8 → loss
Backward: input ← layer 1 ← layer 2 ← ... ← layer 8 ← loss
                   gradients flow backward (chain rule)

Chain rule:
  ∂loss/∂layer_7 = ∂loss/∂layer_8 × ∂layer_8/∂layer_7
                                      ↑ local gradient (how much layer 8's output
                                        changes when layer 7's output changes)

What is "local gradient"?
  Each layer computes: output = f(input)
  Local gradient = ∂output/∂input = "how much does this layer's output
                                     change when its input changes?"

  It's "local" because it only looks at ONE layer's own computation,
  not the full chain from loss to input.

  e.g. Layer 7: output_7 = ReLU(W₇ × input_7 + b₇)
       local gradient = ∂output_7/∂input_7
       = if input_7 changes by 0.01, output_7 changes by e.g. 0.012
       = local gradient ≈ 1.2

Each layer's local gradient acts as a MULTIPLIER in the chain.
If several layers have multipliers > 1, the product grows exponentially.

What "close to 1.0" means:
  local gradient = 1.0 → output changes same as input (no amplification)
  local gradient = 1.2 → output changes 20% MORE than input (slight amplification)
  local gradient = 0.9 → output changes 10% LESS than input (slight shrinking)

"Close to 1.0" = the gradient neither grows nor shrinks much at that layer.
When multiplied across layers:
  Close to 1.0:  0.5 × 1.2 × 1.1 × 0.9 = 0.59  ← stays reasonable
  Far from 1.0:  0.5 × 3.0 × 2.5 × 4.0 = 15.0   ← explodes
```

In a deep network (our transformer has 4 encoder + 4 decoder = 8 layers), this multiplication can cause gradients to explode:

```
Normal training (each layer's local gradient is close to 1.0 — no big amplification):
  Layer 8 gradient: 0.5                    ← ∂loss/∂layer_8
  Layer 7 gradient: 0.5 × 1.2 = 0.6       ← ∂loss/∂layer_7 = ∂loss/∂layer_8 × ∂layer_8/∂layer_7
  Layer 6 gradient: 0.6 × 1.1 = 0.66      ← multiplier 1.1 (close to 1)
  Layer 5 gradient: 0.66 × 0.9 = 0.59     ← multiplier 0.9 (close to 1)
  → multipliers stay near 1.0, so gradients don't explode

Exploding gradients (one bad batch with unusual data):
  Layer 8 gradient: 0.5
  Layer 7 gradient: 0.5 × 3.0 = 1.5
  Layer 6 gradient: 1.5 × 2.5 = 3.75
  Layer 5 gradient: 3.75 × 4.0 = 15.0
  Layer 4 gradient: 15.0 × 3.0 = 45.0
  Layer 3 gradient: 45.0 × 2.0 = 90.0
  → gradients EXPLODE
```

What happens with exploding gradients:

```
weight update = w - lr × gradient

Normal:    w = 0.5 - 0.0003 × 0.6  = 0.4998     ← small, controlled step
Exploding: w = 0.5 - 0.0003 × 90.0 = 0.473      ← huge jump!

One huge jump can destroy everything the model learned in previous batches.
The loss spikes:

loss
 │
5│                          ×  ← one bad batch with exploding gradients
 │                        ╱  ╲
4│                       ╱    ╲
 │                      ╱      ╲
3│                     ╱        ╲
 │        ╱─╲        ╱          ╲
2│       ╱   ─╲     ╱            ╲
 │      ╱      ─╲  ╱              ─ ─ ─  slow recovery
1│     ╱         ─╲╱
 │    ╱
0│───╱
 └──────────────────────────────────────── batch
     training was going well... then one bad batch undoes progress
```

**Why does this happen in transformers specifically?**

1. **Self-attention** — attention creates **dense gradient paths** between all tokens.

In a standard feed-forward network, each layer connects to the next in a simple chain. But self-attention connects **every token to every other token** — one token's gradient flows through all other tokens simultaneously:

```
Feed-forward:   token 0 → token 0 → token 0    (simple chain)

Self-attention:  token 0 ──→ token 0
                 token 0 ──→ token 1
                 token 0 ──→ token 2    (connected to ALL tokens)
                 token 0 ──→ token 3
                 ...

More connections = more paths for gradients to flow through
= gradient from one token accumulates contributions from ALL tokens
= larger total gradient
```

**Why does this create more gradient paths?** In self-attention, every token computes an attention score with every other token (the `Q @ K^T` matrix). So token 0's output depends on **all** tokens:

```
output[0] = 0.2 × value[0] + 0.3 × value[1] + 0.15 × value[2] + 0.35 × value[3]
            ↑ attends to self   ↑ attends to 1   ↑ attends to 2   ↑ attends to 3
```

During backpropagation, the gradient of `output[0]` flows back to all 4 tokens' values — because it used all of them. That's **4 gradient paths** from just one token's output.

In our feed-forward layer, `token 0`'s output only depends on `token 0`'s input — just **1 gradient path**. The FFN is **position-wise**: it applies the same `W2 · ReLU(W1 · x + b1) + b2` to each token independently. Token 0's FFN output has no idea what token 1 or token 2 contain, so `∂output[0]/∂input[1] = 0` — no gradient flows between tokens. Self-attention is the **only** place in the transformer where tokens interact with each other.

With `seq_len = 22`, each token has 22 gradient paths instead of 1. Over 8 layers, this creates a much denser gradient flow than a simple chain, making it easier for gradients to accumulate and explode.

These accumulated gradients get multiplied through 8 layers via the chain rule → can cause exploding gradients.

2. **Residual connections** — gradients flow through skip connections **and** the sub-layer, doubling up
3. **Variable sequence lengths** — a batch with unusually long sequences can produce larger-than-normal gradient norms

## Two Types of Clipping — Norm vs Value

PyTorch provides two approaches:

```
Norm clipping:   clip_grad_norm_(parameters, max_norm=1.0)
Value clipping:  clip_grad_value_(parameters, clip_value=1.0)
```

**We use norm clipping.** Here's why both exist and why norm is better:

## Norm Clipping — How It Works (Our Approach)

`clip_grad_norm_` treats ALL gradients as ONE big vector, computes its total length (norm), and scales it down if too long:

**Step 1 — Collect all gradients into one vector:**

```
Our model has millions of parameters across many layers.
Each parameter has a gradient after loss.backward().

Imagine 3 parameters (simplified):
  param_1.grad = [0.3, -0.5, 0.8]
  param_2.grad = [1.2, -0.4]
  param_3.grad = [-0.6, 0.9, 0.1, -0.3]

Flatten into one vector:
  all_grads = [0.3, -0.5, 0.8, 1.2, -0.4, -0.6, 0.9, 0.1, -0.3]
```

**Step 2 — Compute L2 norm (total length):**

```
total_norm = √(0.3² + (-0.5)² + 0.8² + 1.2² + (-0.4)² + (-0.6)² + 0.9² + 0.1² + (-0.3)²)
           = √(0.09 + 0.25 + 0.64 + 1.44 + 0.16 + 0.36 + 0.81 + 0.01 + 0.09)
           = √3.85
           = 1.96
```

**Step 3 — Compute clip coefficient:**

```
clip_coef = max_norm / (total_norm + 1e-6)
          = 1.0 / (1.96 + 0.000001)
          = 0.51

Clamp to max 1.0:
  clip_coef_clamped = min(0.51, 1.0) = 0.51
```

**Step 4 — Scale ALL gradients by the same factor:**

```
Before clipping:                    After clipping (× 0.51):
param_1.grad = [0.3, -0.5, 0.8]  → [0.153, -0.255, 0.408]
param_2.grad = [1.2, -0.4]       → [0.612, -0.204]
param_3.grad = [-0.6, 0.9, 0.1]  → [-0.306, 0.459, 0.051]

New total_norm = 1.96 × 0.51 = 1.0 ✓ (exactly max_norm)
```

**Key insight:** Every gradient gets multiplied by the **same** factor (0.51). The **direction** of the gradient vector is preserved — we just make it shorter.

**When total_norm < max_norm (no clipping needed):**

```
total_norm = 0.7, max_norm = 1.0

clip_coef = 1.0 / 0.7 = 1.43
clip_coef_clamped = min(1.43, 1.0) = 1.0     ← clamped to 1.0!

All gradients × 1.0 = unchanged. No clipping happens.
```

The `clamp(max=1.0)` ensures we **never scale UP** — only scale down or leave unchanged.

## Value Clipping — How It Works (Alternative)

`clip_grad_value_` clips each gradient element **independently** to `[-clip_value, +clip_value]`:

```
clip_value = 1.0

Before:                          After:
param_1.grad = [0.3, -0.5, 0.8]  → [0.3, -0.5, 0.8]       ← all within [-1, 1], unchanged
param_2.grad = [1.2, -0.4]       → [1.0, -0.4]             ← 1.2 clipped to 1.0
param_3.grad = [-0.6, 2.5, 0.1]  → [-0.6, 1.0, 0.1]       ← 2.5 clipped to 1.0
```

Each element is clamped independently: if > 1.0, set to 1.0. If < -1.0, set to -1.0.

**PyTorch source** (`clip_grad_value_`, line 286):

```python
grad.clamp_(min=-clip_value, max=clip_value)
```

## Why Norm Clipping Is Better — Preserving Direction

The gradient vector has both **magnitude** (how far to step) and **direction** (which way to step). Direction is critical — it points toward lower loss.

```
Gradient vector: [3.0, 6.0]
Direction: atan(6.0/3.0) = 63.4° ← points toward the optimal weight update
```

**Norm clipping** preserves direction:

```
Original:        [3.0, 6.0]     direction = 63.4°
After norm clip: [0.45, 0.89]   direction = 63.4°  ← SAME direction, shorter step
                 ↑ both scaled by same factor (0.149)
```

**Value clipping** distorts direction:

```
Original:        [3.0, 6.0]     direction = 63.4°
After value clip: [1.0, 1.0]   direction = 45.0°  ← WRONG direction!
                  ↑ each clipped independently
```

**Side-by-side on the same graph:**

```
      param_2
        │
    6.0 │                              × original (3.0, 6.0)
        │                            ╱
    5.0 │                          ╱
        │                        ╱
    4.0 │                      ╱
        │                    ╱
    3.0 │                  ╱
        │                ╱
    2.0 │              ╱
        │            ╱
    1.0 │      ■   ╱                   ■ = value-clipped (1.0, 1.0) → 45°
        │     ╱  ● ← norm-clipped     ● = norm-clipped (0.45, 0.89) → 63.4°
    0.5 │   ╱  ╱
        │  ╱ ╱
    0.0 └──────────────────────────── param_1
        0    0.5   1.0   2.0   3.0

    ● sits on the SAME line as × (same angle from origin = same direction)
    ■ sits on a DIFFERENT line (45° diagonal, not 63.4°)

    ● = scaled down uniformly (both × 0.149) → direction preserved ✓
    ■ = each dimension clipped independently → direction distorted ✗
```

**Why direction matters:**

```
Gradient = [3.0, 6.0] means:
  "param_1 needs a small push (3.0)"
  "param_2 needs a BIG push (6.0) — it's twice as important"

Norm clip: [0.45, 0.89] → param_2 still gets 2× more push than param_1 ✓
Value clip: [1.0, 1.0]  → param_2 gets EQUAL push to param_1 ✗
                           The relative importance is lost!
```

This is why **all major transformer implementations** (fairseq, HuggingFace, PyTorch examples) use norm clipping.

## What `max_norm = 1.0` Means in Our Config

From `base.yaml`:

```yaml
clip_grad_norm: 1.0     # Max gradient norm — not in the paper, safety net
```

`1.0` is the **threshold** — if the total gradient norm exceeds 1.0, scale all gradients down so the total norm equals exactly 1.0. If already below 1.0, do nothing.

```
total_norm = 0.7  → below 1.0  → no clipping  → gradients unchanged
total_norm = 1.0  → equals 1.0 → no clipping  → gradients unchanged
total_norm = 1.96 → above 1.0  → clip!        → all gradients × 0.51
total_norm = 15.0 → way above  → clip!        → all gradients × 0.067
total_norm = 500  → exploding  → clip!        → all gradients × 0.002
```

**Typical values used in practice:**

| Project | max_norm | Notes |
|---------|----------|-------|
| Our model | 1.0 | Standard choice |
| fairseq | 1.0 | Default for transformers |
| HuggingFace | 1.0 | Default in Trainer |
| GPT-2 | 1.0 | OpenAI's choice |
| Original paper | not used | No gradient clipping mentioned |

The paper doesn't mention gradient clipping, but it's standard practice — a safety net that costs nothing when gradients are healthy and saves training when they're not.

**Different threshold example:**

```
max_norm = 0.5 (more aggressive clipping):
  total_norm = 0.7  → clip! → gradients × 0.71    ← clips even small gradients
  total_norm = 1.96 → clip! → gradients × 0.26    ← very aggressive

max_norm = 5.0 (more lenient):
  total_norm = 0.7  → no clip
  total_norm = 1.96 → no clip                      ← lets larger gradients through
  total_norm = 8.0  → clip! → gradients × 0.63    ← only clips big ones

max_norm = 1.0 is the sweet spot — clips dangerous spikes without
interfering with normal training.
```

## PyTorch Source — What Actually Happens

`nn.utils.clip_grad_norm_` calls two internal functions:

```python
# From torch/nn/utils/clip_grad.py

def clip_grad_norm_(parameters, max_norm, norm_type=2.0, ...):
    grads = [p.grad for p in parameters if p.grad is not None]   # collect all gradients
    total_norm = _get_total_norm(grads, norm_type)                # √(Σ grad²)
    _clip_grads_with_norm_(parameters, max_norm, total_norm)      # scale if needed
    return total_norm
```

Inside `_clip_grads_with_norm_`:

```python
clip_coef = max_norm / (total_norm + 1e-6)       # ratio: desired / actual
clip_coef_clamped = torch.clamp(clip_coef, max=1.0)  # never scale UP

for g in device_grads:
    g.mul_(clip_coef_clamped)                     # scale each gradient in-place
```

**Why `+ 1e-6`?** Prevents division by zero if `total_norm = 0` (all gradients are zero — unlikely but possible).

**Why `clamp(max=1.0)`?** Without it, small gradients would get **amplified**. If `total_norm = 0.3` and `max_norm = 1.0`, then `clip_coef = 1.0 / 0.3 = 3.33` — that would **triple** all gradients, making them larger than they originally were. That's the opposite of what we want. The `clamp(max=1.0)` caps `3.33` down to `1.0`, so all gradients get multiplied by `1.0` — left **unchanged**. We **never scale UP**, only scale down or leave unchanged.

**Why `norm_type=2.0`?** This is L2 norm (Euclidean distance). The default. You could use `float('inf')` for max-norm (largest single gradient element), but L2 is standard.

---

# Loss Tracking — Why Multiply Back by `n_tokens`

```python
n_tokens = (tgt_output != pad_idx).sum().item()
total_loss += loss.item() * n_tokens
total_tokens += n_tokens
```

`loss.py` divides by `n_tokens` for **consistent gradients** — so every batch contributes equally per token regardless of size. But for **logging**, we need the correct weighted epoch average:

```
Why not just average per-batch losses?

Batch A: loss = 0.80, 200 tokens
Batch B: loss = 1.20, 50 tokens

Simple average: (0.80 + 1.20) / 2 = 1.00
  ← 50 tokens count same as 200 tokens (unfair)

Weighted average: (0.80×200 + 1.20×50) / (200 + 50) = 220/250 = 0.88
  ← 200 tokens contribute 4× more (fair)
```

So `loss.item() * n_tokens` reconstructs the total loss before per-token normalization:

```
loss.py:        180.0 / 200 = 0.90     ← divide for consistent gradients
train_utils.py: 0.90 × 200 = 180.0    ← multiply back for epoch total

Epoch average = total_loss / total_tokens
              = Σ(loss × n_tokens) / Σ(n_tokens)
              = true average loss per token
```

---

# `model.train()` vs `model.eval()` — What Changes

In our model, only **dropout** is affected:

```python
model.train()    # dropout active: randomly zeros ~10% of values
model.eval()     # dropout disabled: passes everything through unchanged
```

| Mode | Dropout | LayerNorm | Attention | FFN |
|------|---------|-----------|-----------|-----|
| `model.train()` | Active (random zeroing) | Same | Same | Same |
| `model.eval()` | Disabled (pass-through) | Same | Same | Same |

**No BatchNorm in our model.** If we had BatchNorm, it would also behave differently between train/eval (using batch stats vs running stats). But we only use LayerNorm, which behaves identically in both modes.

---

# `@torch.no_grad()` — Why Validation Skips Gradients

```python
@torch.no_grad()
def validate(...):
```

During the forward pass, PyTorch normally builds a **computation graph** — tracking every operation so it can compute gradients later with `backward()`. This graph consumes memory:

```
With gradient tracking (training):
  Forward pass:  logits = model(src, tgt_input, ...)
  Memory used:   model weights + activations + computation graph
                 ≈ 100MB       + 200MB        + 200MB = 500MB

Without gradient tracking (validation):
  Forward pass:  logits = model(src, tgt_input, ...)
  Memory used:   model weights + activations
                 ≈ 100MB       + 200MB        = 300MB

Saves ~40% memory and runs ~20% faster
```

Since we never call `backward()` during validation, building the graph is wasted work. `@torch.no_grad()` tells PyTorch to skip it.

---

# Checkpointing — `best.pt` vs `last.pt`

## What Gets Saved

```python
checkpoint = {
    'epoch': epoch,                                    # which epoch (e.g., 15)
    'model_state_dict': model.state_dict(),            # all model weights
    'optimizer_state_dict': optimizer.state_dict(),     # Adam's internal state
    'scheduler_state_dict': scheduler.state_dict(),    # LR scheduler state
    'train_loss': train_loss,                          # for logging
    'val_loss': val_loss                               # for logging
}
```

Two files saved to `transformer/checkpoints/`:

```
best.pt  — model with lowest validation loss (for inference)
           Only overwritten when a NEW best val_loss is found.

last.pt  — model from the most recent epoch (for resuming training)
           Overwritten EVERY epoch.
```

Example over 30 epochs:

```
Epoch 1:  val_loss = 2.50 → best! save best.pt ✓    save last.pt (epoch 1)
Epoch 5:  val_loss = 1.20 → best! save best.pt ✓    save last.pt (epoch 5)
Epoch 10: val_loss = 0.80 → best! save best.pt ✓    save last.pt (epoch 10)
Epoch 15: val_loss = 0.48 → best! save best.pt ✓    save last.pt (epoch 15)
Epoch 16: val_loss = 0.52 → worse, skip              save last.pt (epoch 16)
Epoch 20: val_loss = 0.60 → worse, skip              save last.pt (epoch 20)
Epoch 30: val_loss = 0.70 → worse, skip              save last.pt (epoch 30)

After training:
  best.pt = epoch 15 weights (val_loss = 0.48)
  last.pt = epoch 30 weights (val_loss = 0.70)
```

## Why Save Optimizer and Scheduler State

**For resuming training.** If training crashes at epoch 20, you can reload `last.pt` and continue from epoch 20 instead of starting over.

Without saving these:

```
Adam optimizer has internal momentum (running averages of gradients).
If you restart without optimizer state:
  → momentum resets to zero
  → model "forgets" which direction it was heading
  → first few epochs after restart are wasted re-learning momentum

Scheduler tracks the current step count.
If you restart without scheduler state:
  → step resets to 0
  → LR jumps back to warmup phase (very low LR)
  → wastes 4000 steps re-warming up
```

With saved state, training resumes seamlessly — as if it never stopped.

**For inference only**, you just need `model_state_dict`:

```python
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
# Ready to translate
```

## Overfitting Detection — Train vs Val Loss

By comparing `train_loss` and `val_loss` each epoch, you can detect overfitting:

```
Underfitting (both high):
  Train: 3.2  Val: 3.4   ← model not learning enough
  → Need: more epochs, bigger model, or better data

Good fit (both low, close together):
  Train: 0.8  Val: 0.9   ← healthy gap
  → Model generalizes well

Overfitting (train low, val rising):
  Epoch 10: Train: 0.5  Val: 0.9
  Epoch 15: Train: 0.3  Val: 1.2    ← gap growing
  Epoch 20: Train: 0.1  Val: 1.8    ← memorizing training data
  → best.pt saves epoch 10, the last good model

loss
 │
3│
 │         ╱── val loss (rising = overfitting)
2│        ╱
 │       ╱
1│──╲   ╱
 │   ╲─╱─── val loss (good region, close to train)
 │    ╲
0│     ╲──── train loss (keeps decreasing)
 └──────────────────────────────────── epoch
          ↑
     best.pt saved here (lowest val loss)
```

This is why we save `best.pt` based on **validation loss**, not training loss — training loss always decreases, but validation loss shows when the model stops generalizing.

---

# Progress Bar — tqdm

Instead of printing a new line every 50 batches (840 lines for 30 epochs), `tqdm` shows **one updating line** per epoch that overwrites itself:

```python
pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")

for src, tgt in pbar:
    ...
    pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}")
```

`tqdm` gets the total batch count from `len(train_loader)` automatically. Each iteration, it increments its internal counter and updates the bar — no need for `enumerate` or `batch_idx`.

**`set_postfix`** updates the right side of the bar with current metrics:

```
Epoch 5/30:  63%|████████████▌       | 779/1237 [01:27<00:51, loss=1.42, lr=2.85e-04]
              ↑                       ↑    ↑     ↑         ↑      ↑           ↑
           progress                current total elapsed remaining loss        lr
```

---

# Full Training Output — What You'll See

```
Epoch 1/30: 100%|████████████████████| 1237/1237 [02:15<00:00, loss=2.34, lr=1.42e-04]
  Train Loss: 2.34 | Val Loss: 2.50
  Saved best model (val_loss: 2.5000) -> checkpoints/best.pt

Epoch 2/30: 100%|████████████████████| 1237/1237 [02:14<00:00, loss=1.80, lr=2.10e-04]
  Train Loss: 1.80 | Val Loss: 1.90
  Saved best model (val_loss: 1.9000) -> checkpoints/best.pt

...

Epoch 15/30: 100%|███████████████████| 1237/1237 [02:13<00:00, loss=0.45, lr=1.80e-04]
  Train Loss: 0.45 | Val Loss: 0.48
  Saved best model (val_loss: 0.4800) -> checkpoints/best.pt

Epoch 16/30: 100%|███████████████████| 1237/1237 [02:13<00:00, loss=0.40, lr=1.70e-04]
  Train Loss: 0.40 | Val Loss: 0.52

...

Epoch 30/30: 100%|███████████████████| 1237/1237 [02:13<00:00, loss=0.12, lr=1.05e-04]
  Train Loss: 0.12 | Val Loss: 0.70

Training complete. Best val loss: 0.4800
```

30 epochs = ~30 lines of tqdm output + summary lines. Clean, readable, no noise.

---

# References

### Gradient Clipping

1. [On the difficulty of training recurrent neural networks](https://arxiv.org/abs/1211.5063) — Pascanu et al., 2013 (introduced gradient clipping)
2. [PyTorch `clip_grad_norm_` docs](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html)
3. [PyTorch `clip_grad_value_` docs](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_value_.html)

### Training

4. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017 (Section 5 — Training)
5. [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) — Harvard NLP (training loop reference implementation)

### tqdm

6. [tqdm documentation](https://tqdm.github.io/) — progress bar library
