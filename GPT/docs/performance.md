# Performance on Apple Silicon

What makes GPT training fast on 8 × A100 (Karpathy's build-nanoGPT), and what each
technique actually does on **one Apple M1 (16 GB), PyTorch 2.12, MPS backend**. Every
row was measured on this machine; none is taken from CUDA results.

The short version: the base run is plain **fp32 + SDPA + gradient accumulation + a
memmap token stream**. Three CUDA staples (bf16, `torch.compile`, FlashAttention) are
slower or unavailable here.

## Summary

| technique | 8 × A100 | M1, MPS | used in the base run |
|---|---|---|---|
| FlashAttention | ✅ large win | ❌ not available; SDPA runs but memory is still O(T²) | SDPA, yes |
| bf16 autocast | ✅ | ❌ 1.7× **slower** than fp32 | no |
| fp16 autocast | (bf16 preferred) | ⚠️ 1.24× faster matmul; safe only with SDPA | no |
| `torch.compile` | ✅ large win | ❌ ~25% **less** throughput | no |
| gradient accumulation | ✅ | ✅ essential | yes, 8 micro-batches |
| memmap uint16 token stream | ✅ | ✅ | yes |
| weight tying | ✅ | ✅ 2.1M of 8.99M params saved | yes |
| vocab padded to a multiple of 64 | ✅ | ❌ no measured gain | 8192, chosen for other reasons |
| TF32 matmul | ✅ | ❌ CUDA-only, silent no-op | no |
| fused AdamW | ✅ | ⚠️ runs on MPS; speed not measured | no |
| `pin_memory` / `non_blocking` | ✅ | ❌ no-op: unified memory | no |
| DDP | ✅ 8 GPUs | ❌ one device | no |

## Attention: SDPA is not FlashAttention here

`F.scaled_dot_product_attention` runs on MPS, but it does not use a FlashAttention
kernel: memory still grows with T². Measured, B = 8, 8 heads, d = 64 per head:

| T | SDPA | hand-written attention |
|---|---|---|
| 512, 1024 | no memory saved, ~5% slower | |
| 2048 | 4.97 GB | 7.12 GB |
| 4096 | swaps: 14 GB, 10.9 s | |
| 8192 | out of memory | |

Real FlashAttention is O(T) in memory and would not notice these lengths. This is why
`block_size` is 256 here and not GPT-2's 1024.

At `block_size` 256, the whole model's forward + backward for one micro-batch of
32 × 256: SDPA 985–1,037 ms and 4.78 GB, the hand-written path (transformer/'s
attention, the same weights) 1,023–1,177 ms and 4.88 GB. The two paths agree to 1.67e-06
in the logits. SDPA is used for its fp16 behaviour (below), for not building a mask
tensor, and because the same call becomes FlashAttention on CUDA.

## Precision

**bf16 is slower than fp32.** A 4096 × 4096 matmul: fp32 83 ms, bf16 140 ms. The M1's
GPU has no native bf16 (it arrives with M2/M3), so it is emulated. The raw kernel is
slow, not just autocast.

**fp16 is the fastest option**: the same matmul in 67 ms, 1.24× faster than fp32. The
risk is overflow: fp16's largest value is 65,504, and the transformer/ replication's
attention scores reached 115,200 before the softmax and turned into NaN at epoch 4.
**SDPA survives that.** At a magnitude where the raw q·kᵀ is inf and the hand-written
path returns NaN, SDPA returns clean values, because it applies the 1/√d scale before
the matmul instead of after. So fp16 with SDPA is the one configuration that is both
faster and safe here.

It is **not wired in**. The base run is fp32. Using fp16 for training needs a
`GradScaler`, a NaN guard, the hand-written attention path kept in fp32, and a full run
to show it holds.

## `torch.compile`

Measured on the base configuration (6 layers, d 256, T 256, forward + backward + AdamW):
**0.77–0.81× the throughput** of eager mode over three runs, about 25% slower. It is not
flaky: it compiles, and the outputs match eager to 1.9e-6. The Metal kernels it
generates are slower than the ones MPS ships. It does fuse (memory 2.33 GB against
2.57 GB), but at d = 256 that does not pay. The first compile took 270 s. Not used, and
there is no config flag for it.

## What works

**Gradient accumulation** is the only way to a large batch on one device. One
micro-batch of 32 × 256 already holds 4.78 GB; the base run adds 8 of them per optimizer
step, 65,536 tokens, and the summed gradient equals one batch of 256 sequences to
1e-07.

**A memmap uint16 token stream.** `train.bin` is 22 MB of 2-byte ids, opened once with
`np.memmap`; a batch is 32 random slices. No `Dataset`, no `DataLoader`, no worker
processes, no tokenizing during training. A batch costs about 0.5 ms to build and copy
to the GPU.

**Weight tying.** The output layer reuses the 8,192 × 256 token table: 2,097,152
parameters fewer, 23% of the untied model.

**Vocab padded to a multiple of 64: no gain measured.** The output-layer matmul,
forward + backward, at vocab 8191, 8192, 8193, 8256, and at 50257 against 50304: all
within noise, in fp32 and in fp16. 8192 is kept because it fits the corpus, costs
nothing, and is the right shape on CUDA.

## CUDA-only settings that do nothing here

- `torch.backends.cuda.matmul.allow_tf32 = True`: TF32 is an NVIDIA format. On MPS the
  line runs and changes nothing.
- `pin_memory=True`, `non_blocking=True`: the M1's CPU and GPU share memory, so there is
  no host-to-device copy to overlap.
- DDP: one device.
- `AdamW(fused=True)` is no longer CUDA-only: in PyTorch 2.12 it steps on MPS. Its speed
  against the default was not measured, so the code uses the default.

## The result

| | |
|---|---|
| training throughput, whole run | 111.4M tokens in 4 h 15 m: about **7,300 tokens/s** |
| one optimizer step (8 micro-batches) | median 8.50 s |
| one full validation pass (1.2M tokens) | about 45 s |
| generation, one forward at a full 256-token context | 10.8 ms: about 100 tokens/s, no KV cache |

The same model at GPT-2's 124M size would run about 18 × slower per token by FLOPs:
roughly 400 tokens/s, extrapolated, not measured. GPT-3's 260B-token schedule would
take about 20 years at that rate.

---

Measured on Apple M1, 16 GB, macOS, Python 3.12, PyTorch 2.12.0. Timings move a few
percent between runs; the comparisons in each table were run back to back.
