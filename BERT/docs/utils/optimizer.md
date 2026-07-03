# BERT Optimizer & LR Schedule (`optimizer.py`)

> Module: [`BERT/utils/optimizer.py`](../../utils/optimizer.py) — `build_optimizer`, `LinearWarmupScheduler`, `_decay_groups`
> Paper: Devlin et al. 2019, [*BERT*](https://arxiv.org/abs/1810.04805), Appendix A.2 (Training details)

`build_optimizer(model, total_steps, …)` returns a **`(optimizer, scheduler)` pair**:

- **optimizer** — `AdamW`, with the model's parameters split into two weight-decay groups.
- **scheduler** — a `LambdaLR` driving the **linear warmup → linear decay** learning-rate curve.

The training loop calls `optimizer.step()` then `scheduler.step()` **once per batch**, so the
learning rate moves every step, not every epoch. The whole module is ~90 lines, but almost
every line traces back to one sentence in the paper — and, as we'll see, a few of them trace
to the *released code* instead, because the paper's prose is an incomplete summary.

## Contents

- [The paper says it in one sentence](#the-paper-says-it-in-one-sentence)
- [Paper text vs. released code — three quiet differences](#paper-text-vs-released-code--three-quiet-differences)
- [AdamW vs. "Adam + L2": why decoupled](#adamw-vs-adam--l2-why-decoupled)
- [Weight decay — but not on everything (`_decay_groups`)](#weight-decay--but-not-on-everything-_decay_groups)
- [The learning-rate schedule: a triangle](#the-learning-rate-schedule-a-triangle)
  - [The factor is a *multiplier*, not the lr](#the-factor-is-a-multiplier-not-the-lr)
  - [Warmup, then decay — with a full trace](#warmup-then-decay--with-a-full-trace)
  - [Why warmup at all — and why decay](#why-warmup-at-all--and-why-decay)
- [Epoch vs. batch vs. step — where `total_steps` comes from](#epoch-vs-batch-vs-step--where-total_steps-comes-from)
- [How `LambdaLR` applies it, per step](#how-lambdalr-applies-it-per-step)
- [Putting it together (`build_optimizer`)](#putting-it-together-build_optimizer)
- [Config caveat](#config-caveat)
- [References](#references)

---

## The paper says it in one sentence

Appendix A.2, verbatim:

> *"We use Adam with learning rate of 1e-4, β₁ = 0.9, β₂ = 0.999, L2 weight decay of 0.01,
> learning rate warmup over the first 10,000 steps, and linear decay of the learning rate."*

Every hyperparameter in `optimizer.py` is in that sentence — **except one** (ε), and **two** of
them (the optimizer flavour and "L2") are described in pre-2019 vocabulary that the released
code quietly contradicts. So before the math, a map of what's faithful to the *paper* vs. what's
faithful to the *code*.

## Paper text vs. released code — three quiet differences

| hyperparameter | paper text (A.2) | released code ([`optimization.py`](https://github.com/google-research/bert/blob/master/optimization.py)) | what our [`optimizer.py`](../../utils/optimizer.py) uses | why |
|---|---|---|---|---|
| learning rate | `1e-4` | `1e-4` | `1e-4` | agree |
| β₁, β₂ | `0.9, 0.999` | `0.9, 0.999` | `(0.9, 0.999)` | agree |
| warmup | `10,000` steps | `10,000` | `10000` | agree |
| decay | "linear decay" | linear to 0 | linear to 0 | agree |
| weight decay | "**L2** weight decay 0.01" | **decoupled** (AdamW) 0.01 | **AdamW** 0.01 | code is decoupled, not L2 |
| **ε** | *not mentioned* | `1e-6` | `1e-6` | **paper omits it; code sets it** |

> **The rule for replications:** when the paper prose and the released code disagree, **the code is
> ground truth.** The prose is a human summary written in the language of its time; the optimizer
> that actually trained the weights is the one in the repo. Two rows above are exactly this case.

**ε in detail.** PyTorch's `Adam`/`AdamW` default ε is `1e-8`. If you trusted the framework
default you'd silently get the *wrong* value — BERT's code overrides it to `1e-6` (a larger ε =
a bit more numerical damping in the `√v + ε` denominator). That's why the docstring credits the
*code*, not the paper, for ε.

## AdamW vs. "Adam + L2": why decoupled

The paper says "Adam ... L2 weight decay." Those are **two different things** for Adam, and the
distinction only became famous *after* BERT (Loshchilov & Hutter, *Decoupled Weight Decay
Regularization*, ICLR 2019 — brand new when BERT was written, so "AdamW" wasn't standard
vocabulary yet).

| | what it does | PyTorch |
|---|---|---|
| **L2 (coupled)** — paper's literal words | adds `λ·w` to the **gradient**, so the decay then gets divided by Adam's `√v`. High-gradient weights get *less* effective decay. | `Adam(weight_decay=…)` |
| **Decoupled (AdamW)** | decays the weight directly, `w ← w − lr·λ·w`, **outside** the adaptive scaling | `AdamW(weight_decay=…)` |

Google's released BERT uses `AdamWeightDecayOptimizer`, which is **decoupled** — i.e. AdamW. So
the prose ("L2") describes the *intent* in the old terminology, while the code carries the precise
mechanics. We follow the code.

## Weight decay — but not on everything (`_decay_groups`)

Weight decay pulls each parameter toward 0 every step. That's good regularization for some
parameters and **actively harmful** for others — so the model's parameters are split into two
groups:

```python
[
    {"params": decay,    "weight_decay": 0.01},   # 2-D weight matrices
    {"params": no_decay, "weight_decay": 0.0},    # 1-D: biases + LayerNorm γ/β
]
```

The split is by **shape** — `param.ndim < 2` (a vector or scalar) goes to `no_decay`:

| param | shape | ndim | bucket |
|---|---|---|---|
| `token_embedding.weight` | (10000, 256) | 2 | **decay** |
| attention Q/K/V/O `.weight` | (256, 256) | 2 | **decay** |
| FFN dense `.weight` | (256, 1024) | 2 | **decay** |
| every Linear `.bias` | (256,) | 1 | no_decay |
| `LayerNorm.weight` (γ) | (256,) | 1 | no_decay |
| `LayerNorm.bias` (β) | (256,) | 1 | no_decay |

**Why only the matrices.** Weight decay regularizes by *limiting capacity to fit input→output
mappings* — and only the weight matrices carry that capacity, because they **multiply the
inputs**. Take one linear unit `y = w·x + b`, where the data truly follows `y = 0.3·x + 5`:

- **`w` (the matrices) — decay helps.** `w` is the *slope*; it scales the input, controlling how
  sharp/complex a function the model can fit. If a noisy feature tempts the model into `w = 8`
  (overfitting), decay pulls it back toward the gentle true `0.3`. Smaller `w` = simpler function
  = less overfitting. **Real benefit.**
- **`b` (a bias, 1-D) — decay hurts.** `b` only *shifts* the line; it never multiplies the input,
  so it can't overfit. The data needs `b = 5`, but decay keeps tugging it toward 0 → the model
  settles at `b ≈ 4.2` → every prediction is off by ~0.8. **Pure error, zero overfitting
  prevented.**
- **LayerNorm γ (1-D) — decay hurts, worse.** LayerNorm computes `γ · x_norm + β` with γ starting
  at 1 — it's a per-feature *gain knob* restoring signal strength after normalizing. Decaying γ
  toward 0 attenuates the signal, and across the 6 layers it **compounds** (0.8 per layer ⇒
  0.8⁶ ≈ 0.26 of the signal survives). γ isn't capacity; shrinking it just turns the network's
  volume down.

> **One-line rule:** 2-D = "multiplies inputs" = has overfitting capacity = **decay**.
> 1-D = offset (bias) or gain (LayerNorm) = no capacity = **leave alone**.
> This is the canonical BERT/HuggingFace setup (`no_decay = ["bias", "LayerNorm.weight"]`).

`_decay_groups` also skips frozen params (`if not param.requires_grad: continue`) — irrelevant in
pre-training (nothing is frozen) but correct hygiene for fine-tuning, where you might freeze layers.

## The learning-rate schedule: a triangle

`LinearWarmupScheduler` implements the paper's "warmup ... and linear decay" as a **triangle** —
ramp the lr up from 0 to the peak, then ramp it back down to 0:

```
lr
1e-4 |                        /\          ← peak at step = warmup_steps (10k), factor = 1.0
     |                     /     \
     |                  /          \
     |               /               \
     |            /                    \
     |         /                         \
     |      /                              \
   0 |___/___________________________________\____  step
     0                       10k            15k
        ────────── warmup ─────  ── decay ────
```

The apex lands at `10k` — **two-thirds** across, because warmup (10k) is longer than decay
(15k−10k = 5k). The peak's horizontal position is always at `warmup_steps`, so its placement
depends on the `warmup_steps / total_steps` ratio, not the middle.

### The factor is a *multiplier*, not the lr

`__call__` returns a number in **[0, 1]** — a *factor*, not a learning rate. `LambdaLR` multiplies
the **base lr** (`1e-4`, set on the optimizer) by this factor. So `0.5` means
`lr = 0.5 × 1e-4 = 5e-5`. Unlike the [transformer's Noam schedule](../../../transformer/utils/optimizer.py)
(which bakes a `d_model^(-0.5)` scale into the lr itself), here the peak is set directly as the
optimizer's lr, and the schedule only ever scales it between 0 and 1.

```python
def __call__(self, step: int) -> float:
    if step < self.warmup_steps:
        return step / max(1, self.warmup_steps)                                    # warmup: 0 → 1
    return max(0.0, (self.total_steps - step) / max(1, self.total_steps - self.warmup_steps))  # decay: 1 → 0
```

- **Warmup** (`step / warmup_steps`): rises linearly from 0 (step 0) to 1.0 (step = warmup_steps).
- **Decay** (`(total − step) / (total − warmup)`): the denominator is just the *length* of the
  decay phase, so the fraction falls cleanly from 1.0 (at `step = warmup`) to 0 (at `step = total`).
- The two phases **meet at 1.0** at `step = warmup_steps`, so the curve is continuous — no jump at
  the peak.
- `max(0.0, …)` clamps to 0 if training runs past `total_steps`; `max(1, …)` only guards against
  divide-by-zero. They make the function *robust*, not *correct on their own* — see the caveat below.

### Warmup, then decay — with a full trace

Using our config (`lr = 1e-4`, `warmup_steps = 10000`) and an example `total_steps = 15000`:

| `scheduler.step()` call | step | branch | factor | actual lr (`1e-4 × factor`) |
|---|---|---|---|---|
| at construction | 0 | warmup | 0 / 10000 = 0 | 0 |
| after batch 1 | 1 | warmup | 1 / 10000 | 1e-8 |
| … | 5000 | warmup | 0.5 | 5e-5 |
| … | 10000 | decay | (15k−10k)/(15k−10k) = 1.0 | **1e-4 (peak)** |
| … | 11000 | decay | (15k−11k)/(15k−10k) = 0.8 | 8e-5 |
| … | 15000 | decay | (15k−15k)/… = 0 | 0 |

> **A small quirk:** `LambdaLR` sets the initial lr at construction using `factor(0) = 0`, and the
> loop runs `optimizer.step()` *before* `scheduler.step()` — so the very first weight update happens
> at lr ≈ 0 (a null step), then warmup begins. This matches HuggingFace's
> `get_linear_schedule_with_warmup` exactly; it's one wasted step out of thousands where lr is
> near-zero anyway, so it's left as-is.

### Why warmup at all — and why decay

The triangle isn't arbitrary; each side fixes a real failure mode of training a fresh network with
Adam.

**Why warmup (the rising side).** Adam doesn't use the raw gradient — it updates by `m / (√v + ε)`,
where `v` is a *running average of squared gradients* (the per-parameter variance estimate). In the
first handful of steps, `v` has been averaged over almost no samples, so it's both **unreliable and
often tiny** — and a tiny `√v` in the denominator makes `m/√v` **explode**. That's the worst
possible moment for a giant update: the weights are still fresh random noise, so one wild step can
knock the model into a bad region it never recovers from (loss spikes to NaN, or the embeddings get
scrambled). Ramping the lr up from ≈0 keeps those early updates small until `v` has seen enough
gradients to be a trustworthy estimate. *Then* it's safe to let the lr be large. This is why
warmup matters far more for Adam-family optimizers than for plain SGD, and why deep stacks like
BERT (where early instability compounds through 12 layers) lean on it.

**Why decay (the falling side).** Late in training the model is near a good minimum, and a large lr
makes it **bounce around** that minimum instead of settling into it — every step overshoots and the
loss plateaus noisily. Shrinking the lr toward 0 turns the late updates into fine adjustments, so
the model can descend into a sharper, lower point. It's the standard "big steps to get to the right
neighborhood, small steps to park the car" intuition — anneal the step size as you close in.

> **The shape in one line:** warmup protects the model *from Adam* while its variance estimate is
> still garbage; decay lets the model *settle* once it's near a minimum. Peak lr in the middle is
> where both pressures are satisfied — moments are stable, but you're not yet close enough to need
> small steps.

## Epoch vs. batch vs. step — where `total_steps` comes from

The schedule counts in **steps**, and one step = one batch. The hierarchy:

- **batch** — a group of examples processed together in one forward → backward → update.
  `batch_size: 32` means 32 sequences per batch. **One batch = one optimizer/scheduler step.**
- **epoch** — one complete pass over the *entire* training set.

```
batches (= steps) per epoch = num_examples / batch_size
total_steps                 = batches per epoch × num_epochs
```

Concrete — **6,400** examples, `batch_size = 32`, `num_epochs = 10`:

| quantity | value |
|---|---|
| examples (rows) | 6,400 |
| batch size (rows per batch) | 32 |
| **batches = steps per epoch** | 6400 / 32 = **200** |
| epochs | 10 |
| **total_steps** | 200 × 10 = **2,000** |

> **Common mix-up:** a batch is **not** one row — it's a *bundle* of 32 rows. And `batch_size` is
> fixed at 32 across *all* epochs; what grows with more epochs is the **step count** (200 → 2,000),
> never the batch size.

That's why `warmup_steps: 10000` is **steps, not epochs**, and why `total_steps` can only be known
*after* the dataloader is built. `pretrain.py` computes it at runtime and passes it in:

```python
total_steps = len(train_loader) * config.training.num_epochs   # batches/epoch × epochs
```

(`len(train_loader)` = number of batches = `ceil(num_examples / batch_size)`.)

## How `LambdaLR` applies it, per step

At construction, `LambdaLR` records the optimizer's lr (`1e-4`) as `base_lr`. Every
`scheduler.step()` then does:

```
new_lr = base_lr × lr_lambda(current_step)
```

**Crucially, it multiplies the fixed base `1e-4` every time — not the previous step's lr.** It's
not compounding; each call recomputes `1e-4 × factor(step)` from scratch, so the anchor is always
the original peak.

The per-batch flow inside `train_on_epoch`:

```
for each batch:
    loss.backward()
    clip_grad_norm_(...)
    optimizer.step()        # update weights at the CURRENT lr
    scheduler.step()        # advance step → recompute lr = 1e-4 × factor(step), write onto optimizer
```

So `optimizer.step()` and `scheduler.step()` both fire **once per batch** (200×/epoch in the example
above; 2,000× over the run). Validation and checkpointing, by contrast, fire **once per epoch**
(10×). `scheduler.last_epoch` is just `LambdaLR`'s name for that running **step** counter — it's the
value handed to `lr_lambda`, and the x-axis logged to TensorBoard.

## Putting it together (`build_optimizer`)

```python
optimizer = AdamW(
    params=_decay_groups(model, weight_decay),  # two groups: matrices decay, 1-D don't
    lr=lr,                                       # peak lr — scheduler scales it by [0, 1]
    betas=tuple(betas),                          # YAML gives a list; AdamW wants a tuple
    eps=eps,
)
scheduler = LambdaLR(optimizer, lr_lambda=LinearWarmupScheduler(warmup_steps, total_steps))
return optimizer, scheduler
```

Each param-group dict sets its own `weight_decay`, which **overrides** AdamW's optimizer-level
default for that group — so the decay group gets `0.01` and the no-decay group gets `0.0`,
regardless of the top-level default. `total_steps` is a **required** argument (no default), forcing
`pretrain.py` to compute it from the actual dataloader rather than guess.

## Config caveat

The schedule is only correct when **`warmup_steps < total_steps`**. `base.yaml` sets
`warmup_steps: 10000`; if a run produces fewer than 10k total steps, the lr never reaches its peak
and the decay branch clamps straight to 0. The `max(...)` guards keep it from *crashing*, but the
curve would be wrong. For quick runs (`tiny.yaml`), scale `warmup_steps` down to a few hundred so
the triangle fits inside the run.

## References

- **Paper:** Devlin et al. 2019, [*BERT*](https://arxiv.org/abs/1810.04805) — Appendix A.2 (the one-sentence optimizer spec)
- **Released BERT code:** [google-research/bert — `optimization.py`](https://github.com/google-research/bert/blob/master/optimization.py) — `AdamWeightDecayOptimizer` (decoupled decay, ε `1e-6`); the ground-truth optimizer that the paper prose summarizes
- **Decoupled weight decay (AdamW):** Loshchilov & Hutter 2019, [*Decoupled Weight Decay Regularization*](https://arxiv.org/abs/1711.05101) (ICLR) — why "Adam + L2" ≠ AdamW
- **The schedule we adapted from (Noam):** [`transformer/utils/optimizer.py`](../../../transformer/utils/optimizer.py) — same builder shape, different curve
- **Where it's driven (per-batch step + logging):** [`train_utils.py`](../../utils/train_utils.py)
- **Where `total_steps` is computed and passed:** [`pretrain.md`](../scripts/pretrain.md)
- **PyTorch:** [`AdamW`](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html) (decoupled decay, default ε `1e-8`), [`LambdaLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LambdaLR.html) (multiplies base lr by a factor)
