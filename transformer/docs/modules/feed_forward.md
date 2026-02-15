This is the **Position-wise Feed-Forward Network** from Section 3.3 of the paper.

> FFN(x) = max(0, xW₁ + b₁)W₂ + b₂

**In simple terms:**
* Two linear transformations with ReLU activation in between

* Applied to each position separately and identically

* Input: `d_model = 512`, Hidden: `d_ff = 2048`, Output: `d_model = 512`