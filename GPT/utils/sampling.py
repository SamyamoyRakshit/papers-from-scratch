"""
Autoregressive sampling for GPT: predict one token, append it, repeat.

    logits[:, -1] ─► / temperature ─► top-k ─► top-p ─► softmax ─► multinomial ─► next id

No KV cache: every step re-runs the whole (cropped) context. At block_size 256 and
6.9M params that costs milliseconds per token; nanoGPT's generate() does the same.

No defaults here: the values come from configs/generate.yaml, range-checked by GenerateConfig.
"""
import torch
import torch.nn as nn


@torch.no_grad()
def generate(model: nn.Module, idx: torch.Tensor, max_new_tokens: int, block_size: int,
             temperature: float, top_k: int | None, top_p: float | None,
             eot_id: int) -> torch.Tensor:
    """
    Args:
        idx: prompt ids, shape (1, T), int64, on the model's device. One sample at a time,
            so the <|endoftext|> stop below is one comparison, not a per-row mask.
        max_new_tokens: tokens to add at most.
        block_size: config.model.block_size. wpe has only this many rows, so the
            context is cropped to the last block_size ids before every forward.
        temperature: logits / temperature. < 1 sharpens, > 1 flattens, 0 = greedy (argmax).
        top_k: keep the k most likely ids. GPT-2's samples use k = 40 (Fan et al. 2018).
        top_p: keep the smallest set of ids whose probability sums to ≥ p
            (nucleus sampling, Holtzman et al. 2019 — after GPT-2, not in the paper).
        eot_id: stop when the model emits it (the article ended). Not appended.
    Returns:
        (1, T + n) ids: the prompt followed by n ≤ max_new_tokens new ids.
    """
    model.eval()                                      # dropout OFF
    for _ in range(max_new_tokens):
        logits = model(idx[:, -block_size:])[:, -1]   # (1, vocab): only the last position predicts the next token

        if temperature == 0.0:                        # dividing by 0 would give inf / nan
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k is not None:
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                probs = sorted_logits.softmax(dim=-1)
                # mass BEFORE each id > p → drop it. The id that crosses p stays, so ≥ 1 id survives.
                sorted_logits[probs.cumsum(dim=-1) - probs > top_p] = float("-inf")
                logits = logits.scatter(-1, sorted_idx, sorted_logits)   # back to vocab order
            next_id = torch.multinomial(logits.softmax(dim=-1), num_samples=1)   # (1, 1)

        if next_id.item() == eot_id:
            break
        idx = torch.cat([idx, next_id], dim=1)
    return idx
