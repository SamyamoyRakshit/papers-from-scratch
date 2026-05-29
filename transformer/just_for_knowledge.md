## 📋 Table of Contents
1. [`nn.Parameter`](#1-nnparameter)
2. [How the `forward` function in PyTorch returns `class` value when we do not call `forward`?](#2-how-the-forward-function-in-pytorch-returns-class-value-when-we-do-not-call-forward)
3. [`hooks`](#3-hooks)
4. [`torch.tensor` vs `torch.Tensor`](#4-torchtensor-vs-torchtensor)
5. [Git hygiene: secrets, artifacts, and LFS](#5-git-hygiene-secrets-artifacts-and-lfs)
6. [Post-Transformer architectures (FlashAttention, Linear Attention, Mamba)](#6-post-transformer-architectures-flashattention-linear-attention-mamba)


## 1. `nn.Parameter`: 

`nn.Parameter` designates a tensor as a **trainable model parameter**, ensuring it is tracked by autograd, updated by optimizers, and persisted in the model’s state for checkpointing and deployment.

**Example:**
In Layer Normalization, the scale (γ) and shift (β) are defined as parameters so the model can learn how much to rescale and shift normalized activations; without marking them as parameters, these values would remain fixed and reduce model expressiveness.

---

## 2. How the `forward` function in pytorch returns `class` value when we do not call `forward`?

This is a **core PyTorch “magic” question**—and it’s not magic once you see it 😄

### The key idea

**You *are* calling `forward`, just not directly.**
PyTorch calls it for you via `__call__`.

---

### What you write

```python
output = model(x)
```

You never wrote:

```python
model.forward(x)
```

Yet it works. Why?

---

### What actually happens under the hood

`nn.Module` (the parent class of all models) defines `__call__`.

So this:

```python
model(x)
```

is really:

```python
model.__call__(x)
```

And **inside `__call__`**, PyTorch does roughly this:

```python
def __call__(self, *args, **kwargs):
    # 1. pre-forward hooks
    # 2. autocast / grad logic
    output = self.forward(*args, **kwargs)
    # 3. post-forward hooks
    return output
```

So:
👉 **`__call__` → `forward` → return output**

---

### Minimal example

```python
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        return x * 2

model = MyModel()
print(model(3))
```

Output:

```
6
```

You didn’t call `forward`, but PyTorch did.

---

### Why PyTorch forces this pattern (important)

You **should NOT** call `forward()` directly in practice.

Why?

Because `__call__` handles:

* autograd (gradient tracking)
* hooks (hooks let you peek inside the model’s “black box” by observing (or intercepting) what flows through its layers during execution.)
* `model.train()` / `model.eval()` behavior
* mixed precision
* distributed logic

If you do this:

```python
model.forward(x)  # ❌ bad practice
```

You **bypass all of that**.

---

### Analogy (simple but accurate)

* `forward()` → engine of the car
* `model(x)` → turning the key

You never start a car by directly spinning the engine 😄

---

### TL;DR

* `model(x)` calls `__call__`
* `__call__` internally calls `forward`
* PyTorch controls the wrapper for safety + features
* You define **what happens**, PyTorch defines **how it’s executed**

---

## 3. `hooks`:
Cool, let’s do this **step-by-step with a real, simple example**—no theory fluff.

---

#### Problem we want to solve

> *“I want to see the output of a hidden layer, but I don’t want to change the model’s `forward()`.”*

This is **exactly** why hooks exist.

---

#### Example model

```python
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 3)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(3, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = Net()
```

---

#### Without hooks (bad / invasive)

You’d have to modify `forward()` just to print or save values ❌.

---

#### With a **forward hook** (clean way)

#### Step 1: Define a hook function

```python
def save_activation(module, input, output):
    print("Layer:", module)
    print("Activation:", output)
```

---

#### Step 2: Register the hook

```python
handle = model.fc1.register_forward_hook(save_activation)
```

Now the hook is **attached** to `fc1`.

---

#### Step 3: Run the model

```python
x = torch.randn(1, 4)
y = model(x)
```

#### What happens internally

```text
model(x)
  ↓
fc1.forward(x)
  ↓
HOOK RUNS  ← 👀 you intercept here
  ↓
relu
  ↓
fc2
```

Output (example):

```text
Layer: Linear(in_features=4, out_features=3)
Activation: tensor([[ 0.21, -0.48, 0.77]], grad_fn=<AddmmBackward>)
```

---

#### What just happened?

* You **did not touch** `forward()`
* PyTorch **called your hook automatically**
* You saw intermediate activations

That’s the power of hooks.

---

#### Real-world use case

#### Feature extraction

```python
features = {}

def hook(module, inp, out):
    features["fc1"] = out.detach()

model.fc1.register_forward_hook(hook)
model(x)

print(features["fc1"].shape)
```

Used in:

* CNN feature maps
* transformer hidden states
* recommendation embeddings

---

#### Cleaning up (important ⚠️)

Hooks stay active unless removed.

```python
handle.remove()
```

Always do this.

---

#### One-line mental model

**A hook is a function that PyTorch calls for you when data passes through a layer.**

---

### **More Information about `hooks`:**
Hooks in PyTorch are **callbacks** that let you **intercept a model while it’s running** — without changing the model code.

Think of them as *listeners* you attach to a layer or model.

---

#### Why hooks exist (intuition)

You may want to:

* inspect activations
* debug gradients
* visualize feature maps
* modify outputs on the fly

Hooks let you do this **from the outside**.

---

#### Types of hooks (the important ones)

#### 1️⃣ Forward hook

Runs **after** a layer’s `forward()`.

```python
def forward_hook(module, inp, out):
    print(module)
    print("output shape:", out.shape)

layer = nn.Linear(4, 2)
layer.register_forward_hook(forward_hook)
```

Used for: activations, debugging shapes.

---

#### 2️⃣ Forward *pre*-hook

Runs **before** `forward()`.

```python
def pre_hook(module, inp):
    print("input:", inp)

layer.register_forward_pre_hook(pre_hook)
```

Used for: inspecting or modifying inputs.

---

#### 3️⃣ Backward hook

Runs during **backpropagation**.

```python
def backward_hook(module, grad_in, grad_out):
    print("grad_out:", grad_out)

layer.register_full_backward_hook(backward_hook)
```

Used for: gradient analysis, debugging vanishing/exploding grads.

---

#### Tiny end-to-end example

```python
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 2)

    def forward(self, x):
        return self.fc(x)

net = Net()

def hook(module, inp, out):
    print("Activation:", out)

net.fc.register_forward_hook(hook)

x = torch.randn(1, 3)
y = net(x)
```

👉 You didn’t change `forward()`, yet you saw the activation.

---

#### Where hooks sit in execution

```text
model(x)
  ↓
__call__()
  ↓
forward_pre_hook
  ↓
forward()
  ↓
forward_hook
  ↓
output
```

Backward hooks trigger during `.backward()`.

---

#### Why hooks are powerful (and dangerous ⚠️)

✅ Powerful:

* Non-invasive debugging
* Feature extraction (CNNs, transformers)
* Grad-CAM, attention visualization

⚠️ Dangerous:

* Can silently change behavior
* Easy to forget to remove
* Can break distributed / JIT code

Always store and remove handles:

```python
handle = layer.register_forward_hook(hook)
handle.remove()
```

---

#### One-line summary

**Hooks are functions that let you watch or modify what flows through a PyTorch model during forward or backward passes.**

___

## 4. `torch.tensor` vs `torch.Tensor`

These look almost identical but serve completely different purposes.

### `torch.tensor` — a **function** (lowercase t)

Creates a new tensor from data:

```python
x = torch.tensor([1, 2, 3])        # creates tensor from a list
y = torch.tensor(5.0)               # creates a scalar tensor
z = torch.tensor([[1, 2], [3, 4]])  # creates a 2D tensor
```

### `torch.Tensor` — a **class** (capital T)

The type of all tensors in PyTorch. Used for type hints and `isinstance` checks:

```python
# Type hint in function signatures
def forward(self, src: torch.Tensor) -> torch.Tensor:
    ...

# Type checking
x = torch.tensor([1, 2, 3])
print(isinstance(x, torch.Tensor))  # True
print(type(x))                       # <class 'torch.Tensor'>
```

### Common mistake

```python
# ❌ WRONG — torch.tensor is a function, not a type
def forward(self, src: torch.tensor) -> torch.tensor:

# ✅ CORRECT — torch.Tensor is the class/type
def forward(self, src: torch.Tensor) -> torch.Tensor:
```

### Quick reference

| | `torch.tensor` | `torch.Tensor` |
|---|---|---|
| What | Function | Class |
| Purpose | Create a tensor from data | Type annotation, isinstance checks |
| Example | `x = torch.tensor([1,2,3])` | `def forward(self, x: torch.Tensor)` |

### One-line summary

**`torch.tensor` makes tensors. `torch.Tensor` describes them.**

___

## 5. Git hygiene: secrets, artifacts, and LFS

Three closely-related ideas that together control **what enters your repo** — secrets (never), runtime artifacts (almost never), and large binaries (carefully, via LFS).

---

### 5.1 Secrets via `.env` + `python-dotenv`

The problem: API tokens (HuggingFace, OpenAI, etc.) need to be in `os.environ` so libraries can pick them up — but they must **never** end up in git history.

The pattern:

1. Add a real values file at the repo root:

   ```
   # .env (gitignored)
   HF_TOKEN=hf_AbCdEfGhIjKlMnOpQrStUvWxYz0123456789
   ```

2. Load it at the very top of your entry script — **before any library imports that might read the env var**:

   ```python
   from dotenv import load_dotenv
   load_dotenv()  # populates os.environ from .env

   from datasets import load_dataset   # now sees HF_TOKEN
   ```

3. Ignore it in `.gitignore`:

   ```
   .env
   .env.*
   !.env.example
   ```

#### Why order matters

`huggingface_hub` and `datasets` read `os.environ["HF_TOKEN"]` at import time / first call. If you call `load_dotenv()` *after* importing them, the token is loaded but the libraries already gave up on finding one — they treat you as anonymous and you hit the rate-limit warning.

Mental model: `load_dotenv()` is a translator that copies `.env` → `os.environ`. Any code that reads `os.environ` *after* the translation sees the values. Anything that read it *before* does not.

#### `.env.example` for open-source

You commit a **template** (no secrets, just structure) so collaborators know what env vars to set:

```
# .env.example (committed)
# HuggingFace read-only token — raises anonymous rate limit when fetching datasets.
# Get one at https://huggingface.co/settings/tokens
HF_TOKEN=
```

The `!.env.example` line in `.gitignore` *un-ignores* this one specific filename. Standard pattern: clone → `cp .env.example .env` → fill in real values.

---

### 5.2 Runtime artifacts: `checkpoints/`, `logs/`, etc.

Anything **produced by running the code** generally shouldn't enter git:

* `.pt` checkpoints (100s of MB each, churn every epoch)
* TensorBoard `events.*` files
* `train.log` per run
* `__pycache__/`, `.ipynb_checkpoints/`

Why: source repos describe **how to produce** outputs; they shouldn't *be* the outputs. Committing artifacts bloats clones, makes diffs noisy, and means every training run wants to be a commit.

Pattern:

```
# .gitignore
checkpoints/
logs/
```

Patterns without a leading `/` match anywhere in the tree, so this covers `transformer/checkpoints/`, `ViT/checkpoints/`, etc.

---

### 5.3 `.gitattributes` (Git LFS) vs `.gitignore`

The trick everyone trips on: **these two files do not conflict.** They answer different questions.

| File | Question it answers | Triggered when |
|---|---|---|
| `.gitignore` | "Should git see this file at all?" | `git add` (silently skips ignored paths) |
| `.gitattributes` | "**If** I commit this file, how should git store it?" | `git add` of a non-ignored file |

So a typical setup looks like:

```
# .gitattributes — route any committed .pt through LFS, not regular git storage
*.pt filter=lfs diff=lfs merge=lfs -text
*.pth filter=lfs diff=lfs merge=lfs -text
```

```
# .gitignore — don't let me commit checkpoint runs at all
checkpoints/
logs/
```

These work **together**:
* `.gitignore` blocks accidental adds of every-epoch garbage in `checkpoints/`.
* `.gitattributes` ensures that a *deliberately* committed `.pt` (e.g. `released/transformer-v1.pt`) goes to LFS, not bloat your regular pack files.

LFS rules sit dormant for ignored paths and only fire when you knowingly add a file outside the ignore.

#### Why LFS exists at all

GitHub blocks single files >100 MB and warns at >50 MB. LFS replaces big files in the regular git history with tiny pointer files; the actual bytes live on a separate LFS server. Clones download pointers immediately and large files lazily — keeps regular git fast.

But **LFS has its own quotas** (free tier: 1 GB storage, 1 GB/month bandwidth). Don't use it as an excuse to commit every checkpoint — be deliberate.

---

### 5.4 Untracking files already committed

Adding a path to `.gitignore` only stops **new** tracking. Files already in the index stay tracked until you explicitly remove them:

```bash
git rm --cached transformer/checkpoints/best.pt transformer/checkpoints/last.pt
git commit -m "Stop tracking checkpoint artifacts; rely on .gitignore"
git push
```

* `--cached` means "remove from git's index, **keep the file on disk**." Local files survive; git just stops tracking them.
* New clones won't pull these files anymore (they're no longer in HEAD).
* History still contains them — old commits can still reference the LFS objects on the LFS server. Usually fine; running LFS garbage collection is rarely worth it for a few stragglers.

Verify after:

```bash
git ls-files | grep checkpoints   # should print nothing
```

---

### One-line summary

**`.env` for secrets, `.gitignore` for artifacts, `.gitattributes` for LFS storage rules — and `git rm --cached` to clean up files you ignored too late.**

___

## 6. Post-Transformer architectures (FlashAttention, Linear Attention, Mamba)

The "Attention Is All You Need" Transformer scales **quadratically** in sequence length: doubling `n` quadruples compute and activation memory ([why?](docs/utils/data_utils.md#why-not-max_seq_len--5000-then)). For our setup (`max_seq_len = 128`) that's a non-issue. But once context grows to 10K, 100K, or 1M tokens, the n² matrix becomes the dominant cost — and a whole line of research has grown up around getting past it.

None of these are implemented in this repo. We're replicating the 2017 paper. Read this once, file it away.

---

### 6.1 FlashAttention (Dao et al., 2022)

**Paper:** [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)

**The idea:** same math as standard attention, but a smarter GPU memory access pattern. The full `(n, n)` score matrix never gets materialized in HBM (the slow GPU DRAM) — it's computed in tiles inside fast on-chip SRAM and the softmax is applied incrementally as those tiles stream through.

**Result:** exact same outputs as standard attention, ~2-4× faster, and activation memory drops from O(n²) to O(n). A pure systems-level trick — no approximation, no quality loss. Now the default attention kernel in PyTorch (`F.scaled_dot_product_attention`), HuggingFace, vLLM, etc.

**Takeaway:** for our 128-token setup, FlashAttention wouldn't change anything. But for any model training on long contexts, it's free performance.

---

### 6.2 Linear Attention

**Paper (one of many):** [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236) (Katharopoulos et al., 2020)

**The idea:** the n² cost comes from `softmax(QK^T)V` — softmax forces you to materialize the full score matrix before normalizing. Replace softmax with a kernel function `φ` that lets you reassociate the multiplication:

```
softmax(QK^T) V        →    O(n²)   (must compute n×n first)
(φ(Q) φ(K)^T) V        →    same shape, still n²
φ(Q) (φ(K)^T V)        →    O(n)    ← reassociated! the inner product is (d, d)
```

By computing `φ(K)^T V` first (a small `d × d` matrix), you skip ever forming the n×n. Cost drops to O(n) in sequence length.

**Trade-off:** approximation, not exact. Quality is typically a notch below softmax attention, though specific variants (Performer, Linformer, etc.) close the gap.

**Takeaway:** scales beautifully to long sequences. But for translation-scale tasks (n < 200), the constant factor often makes full softmax attention faster *and* better-quality.

---

### 6.3 Mamba / State Space Models (2023)

**Paper:** [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) (Gu & Dao, 2023)

**The idea:** drop attention entirely. Instead, use a **recurrent state** — a fixed-size hidden vector `h` that gets updated as each token streams in:

```
h_t = A · h_{t-1} + B · x_t     ← state update (matrix A learned, "selective")
y_t = C · h_t                    ← output
```

This is the structure of an RNN, but with a clever parameterization (the "selective state space") that makes it: (a) trainable in parallel like a Transformer, (b) O(n) in sequence length, and (c) competitive with Transformers at small-to-mid scale.

**Trade-off:** because `h` has a fixed size, the model has to *compress* the whole past into it — losing the perfect-recall property of attention. Empirically Mamba is strong on long-range tasks but mixed on tasks needing precise lookup over long context.

**Takeaway:** an active research frontier. Hybrid architectures (Mamba + a few attention layers) are showing strong results. Worth watching, not replacing your understanding of Transformers.

---

### 6.4 When this matters for us

| Setup | Sequence length | n² cost? | What to use |
|---|---|---|---|
| Our translation replication | n ≤ 128 | Trivial | Vanilla attention (the paper) |
| Modern chat / code LLMs | n = 4K-32K | Painful | FlashAttention (almost universal now) |
| Long-document / long-context | n = 100K+ | Prohibitive | FlashAttention + sparse / linear / SSM hybrids |

For replicating "Attention Is All You Need" on 128-token sentences, none of this is needed.

---

### 6.5 Reading order if you want to go deeper

1. **FlashAttention** — easiest to grok; same math, just systems. ([blog post](https://crfm.stanford.edu/2023/01/13/flashattention.html) is a gentler intro than the paper)
2. **Linear Attention** — read [Transformers are RNNs](https://arxiv.org/abs/2006.16236) for the kernel trick; [Performer](https://arxiv.org/abs/2009.14794) for the random-feature variant
3. **Mamba** — paper is dense; [this annotated walkthrough](https://srush.github.io/annotated-mamba/hard.html) by Sasha Rush is the gentlest start

### One-line summary

**The n² wall is real, but only at long context — FlashAttention is the universal default now, Linear Attention trades quality for speed, and Mamba ditches attention for a recurrent state.**

___
