## 📋 Table of Contents
1. [`nn.Parameter`](#1-nnparameter)
2. [How the `forward` function in PyTorch returns `class` value when we do not call `forward`?](#2-how-the-forward-function-in-pytorch-returns-class-value-when-we-do-not-call-forward)
3. [`hooks`](#3-hooks)
4. [`torch.tensor` vs `torch.Tensor`](#4-torchtensor-vs-torchtensor)
5. [Git hygiene: secrets, artifacts, and LFS](#5-git-hygiene-secrets-artifacts-and-lfs)


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
