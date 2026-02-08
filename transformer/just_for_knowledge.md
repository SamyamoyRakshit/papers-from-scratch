## 📋 Table of Contents
1. [`nn.Parameter`](#1-nnparameter)
2. [How the `forward` function in PyTorch returns `class` value when we do not call `forward`?](#2-how-the-forward-function-in-pytorch-returns-class-value-when-we-do-not-call-forward)
3. [`hooks`](#3-hooks)


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

## Problem we want to solve

> *“I want to see the output of a hidden layer, but I don’t want to change the model’s `forward()`.”*

This is **exactly** why hooks exist.

---

## Example model

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

## Without hooks (bad / invasive)

You’d have to modify `forward()` just to print or save values ❌.

---

## With a **forward hook** (clean way)

### Step 1: Define a hook function

```python
def save_activation(module, input, output):
    print("Layer:", module)
    print("Activation:", output)
```

---

### Step 2: Register the hook

```python
handle = model.fc1.register_forward_hook(save_activation)
```

Now the hook is **attached** to `fc1`.

---

### Step 3: Run the model

```python
x = torch.randn(1, 4)
y = model(x)
```

### What happens internally

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

## What just happened?

* You **did not touch** `forward()`
* PyTorch **called your hook automatically**
* You saw intermediate activations

That’s the power of hooks.

---

## Real-world use case

### Feature extraction

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

## Cleaning up (important ⚠️)

Hooks stay active unless removed.

```python
handle.remove()
```

Always do this.

---

## One-line mental model

**A hook is a function that PyTorch calls for you when data passes through a layer.**

---

### More Information about `hooks`:
Hooks in PyTorch are **callbacks** that let you **intercept a model while it’s running** — without changing the model code.

Think of them as *listeners* you attach to a layer or model.

---

## Why hooks exist (intuition)

You may want to:

* inspect activations
* debug gradients
* visualize feature maps
* modify outputs on the fly

Hooks let you do this **from the outside**.

---

### Types of hooks (the important ones)

### 1️⃣ Forward hook

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

### 2️⃣ Forward *pre*-hook

Runs **before** `forward()`.

```python
def pre_hook(module, inp):
    print("input:", inp)

layer.register_forward_pre_hook(pre_hook)
```

Used for: inspecting or modifying inputs.

---

### 3️⃣ Backward hook

Runs during **backpropagation**.

```python
def backward_hook(module, grad_in, grad_out):
    print("grad_out:", grad_out)

layer.register_full_backward_hook(backward_hook)
```

Used for: gradient analysis, debugging vanishing/exploding grads.

---

## Tiny end-to-end example

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

## Where hooks sit in execution

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

## Why hooks are powerful (and dangerous ⚠️)

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

## One-line summary

**Hooks are functions that let you watch or modify what flows through a PyTorch model during forward or backward passes.**

___
