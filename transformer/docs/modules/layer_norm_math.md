# A Mathematical Walkthrough of `LayerNorm` Using an Example

I'll take **one sentence, a few tokens, small embedding size**, and go through **every mathematical step**.

---

## Setup (very small & clear)

### Sentence

```
"my name is khan"
```

### Assumptions

* `batch_size = 1` (single sentence)
* `seq_len = 4` (4 words)
* `embedding_dim = 3`
* `gamma = [1, 1, 1]`
* `beta  = [0, 0, 0]`
* `eps = 1e-5` (we'll ignore it in manual math since values are clean)

---

## Step 1️⃣ Input embeddings (`x`)

Assume embeddings are:

```text
my    → [1, 2, 3]
name  → [4, 5, 6]
is    → [7, 8, 9]
khan  → [10,11,12]
```

So tensor `x` is:

```text
x.shape = (1, 4, 3)

x = [
  [
    [ 1,  2,  3],   # my
    [ 4,  5,  6],   # name
    [ 7,  8,  9],   # is
    [10, 11, 12]    # khan
  ]
]
```

---

## Step 2️⃣ Compute mean (over last dimension)

### Formula (LayerNorm):

$$
\mu = \frac{1}{D}\sum_{i=1}^{D} x_i
$$

---

### Token-wise mean

#### Token 1: `"my"`

$$
\mu = (1 + 2 + 3) / 3 = 2
$$

#### Token 2: `"name"`

$$
\mu = (4 + 5 + 6) / 3 = 5
$$

#### Token 3: `"is"`

$$
\mu = (7 + 8 + 9) / 3 = 8
$$

#### Token 4: `"khan"`

$$
\mu = (10 + 11 + 12) / 3 = 11
$$

---

### Mean tensor

```text
mean.shape = (1, 4, 1)

mean = [
  [
    [ 2],
    [ 5],
    [ 8],
    [11]
  ]
]
```

✅ **One mean per token**

---

## Step 3️⃣ Compute variance (population variance)

### Formula:

$$
\sigma^2 = \frac{1}{D}\sum (x_i - \mu)^2
$$

---

### Token-wise variance

#### `"my"` → `[1,2,3]`

$$
(1-2)^2 = 1
$$

$$
(2-2)^2 = 0
$$

$$
(3-2)^2 = 1
$$

$$
\sigma^2 = (1+0+1)/3 = 2/3
$$

---

#### `"name"` → `[4,5,6]`

Same pattern → variance = `2/3`

#### `"is"` → `[7,8,9]`

Same → variance = `2/3`

#### `"khan"` → `[10,11,12]`

Same → variance = `2/3`

---

### Variance tensor

```text
var = [
  [
    [0.6667],
    [0.6667],
    [0.6667],
    [0.6667]
  ]
]
```

---

## Step 4️⃣ Compute standard deviation

$$
\sigma = \sqrt{\sigma^2}
$$

$$
\sqrt{2/3} \approx 0.8165
$$

---

```text
std = [
  [
    [0.8165],
    [0.8165],
    [0.8165],
    [0.8165]
  ]
]
```

---

## Step 5️⃣ Normalize (`x̂`)

Formula:

$$
\hat{x} = \frac{x - \mu}{\sigma}
$$

---

### Token-wise normalization

#### `"my"`

$$
[1,2,3] - 2 = [-1, 0, 1]
$$

$$
[-1,0,1] / 0.8165 \approx [-1.225, 0, 1.225]
$$

---

#### `"name"`

$$
[4,5,6] - 5 = [-1,0,1]
$$

$$
[-1,0,1] / 0.8165 \approx [-1.225, 0, 1.225]
$$

(Same pattern for all tokens)

---

### Normalized tensor (`x_hat`)

```text
x_hat = [
  [
    [-1.225,  0.000,  1.225],
    [-1.225,  0.000,  1.225],
    [-1.225,  0.000,  1.225],
    [-1.225,  0.000,  1.225]
  ]
]
```

---

## Step 6️⃣ Scale & shift (γ, β)

Formula:

$$
y = \gamma \cdot \hat{x} + \beta
$$

Here:

```text
gamma = [1, 1, 1]
beta  = [0, 0, 0]
```

So output = `x_hat`.

---

# ✅ Final Output (LayerNorm result)

```text
y = [
  [
    [-1.225,  0.000,  1.225],
    [-1.225,  0.000,  1.225],
    [-1.225,  0.000,  1.225],
    [-1.225,  0.000,  1.225]
  ]
]
```

---

# 🔑 Key Insight (very important)

* **Each word is normalized independently**
* Mean = 0, variance = 1 **within each token**
* No interaction between:
  * different words
  * different sentences
  * batch size

---

## One-line intuition

> **LayerNorm rescales each word's embedding so its components are comparable to each other.**
