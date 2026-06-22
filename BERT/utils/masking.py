import torch


def mask_tokens(
        token_ids: torch.Tensor,
        vocab_size: int,
        mask_token_id: int,
        special_token_ids,
        mlm_probability: float = 0.15,
        ignore_index: int = -100
):
    """
    Masked-LM masking for ONE example (Devlin et al. 2019, §3.1, "Task #1").

    Takes a single tokenized sequence — already assembled as
    `[CLS] A [SEP] B [SEP]` (+ optional [PAD]) by the NSP builder — and produces
    the (masked input, label) pair the MLM head + loss consume.

    The 15% / 80-10-10 rule (§3.1):
      - 15% of the NON-special tokens are picked to be PREDICTED.
      - Of those picked positions:
          80% → replaced with [MASK]
          10% → replaced with a RANDOM token
          10% → left UNCHANGED (kept as the original word)
      - The split exists so the model can't assume "an input slot is only ever
        wrong where it sees [MASK]" — at fine-tune time there is no [MASK] token,
        so it must build a representation for every token, not just masked ones.

    The label convention (the seam with `loss.py`):
      - `labels` starts as all `ignore_index` (-100).
      - At every PICKED position we write the ORIGINAL token id — so those are the
        only positions `F.cross_entropy(..., ignore_index=-100)` scores. The other
        ~85% contribute nothing. NOTE all 15% picked are scored, including the
        10%-random and 10%-keep ones — not just the [MASK] 80%.

    Special tokens ([CLS], [SEP], [PAD], ...) are never picked and never become a
    random replacement target.

    Args:
        token_ids: 1-D LongTensor (seq_len,) — one example's WordPiece ids.
        vocab_size: size of the WordPiece vocab — range to draw random tokens from.
        mask_token_id: id of the [MASK] token.
        special_token_ids: iterable of ids that must never be masked (e.g.
            [CLS], [SEP], [PAD]).
        mlm_probability: fraction of non-special tokens to predict. Default: 0.15.
        ignore_index: label value for non-predicted positions. Default: -100
            (matches loss.py's `ignore_index`).

    Returns:
        masked_ids: 1-D LongTensor (seq_len,) — input after 80/10/10 corruption.
        labels:     1-D LongTensor (seq_len,) — original id at predicted positions,
                    `ignore_index` everywhere else.

    References:
        - Paper: Devlin et al. 2019, https://arxiv.org/abs/1810.04805 — §3.1.
        - Google BERT (TF) — `create_masked_lm_predictions` in create_pretraining_data.py:
          https://github.com/google-research/bert/blob/master/create_pretraining_data.py
    """
    labels = torch.full_like(token_ids, ignore_index)   # make all the token_ids to ignore_index(-100)

    # Candidate positions = everything that is NOT a special token.
    special_mask = torch.zeros_like(token_ids, dtype=torch.bool)     # all False = "nothing special yet"
    for sid in special_token_ids:   # special_token_ids; eg. [PAD], [CLS], [SEP] etc.
        special_mask |= token_ids == sid
    candidates = (~special_mask).nonzero(as_tuple=True)[0]

    # Pick 15% of the candidates to predict (at least 1, as long as any exist).
    num_to_predict = max(1, int(round(len(candidates) * mlm_probability)))
    perm = torch.randperm(len(candidates))
    selected = candidates[perm[:num_to_predict]]

    # The answer key: original ids at the selected positions, -100 elsewhere.
    labels[selected] = token_ids[selected]

    # Now corrupt the INPUT at the selected positions: 80% [MASK] / 10% random /
    # 10% keep. One uniform draw per selected position decides the bucket.
    masked_ids = token_ids.clone()
    decision = torch.rand(num_to_predict)

    mask_bucket = selected[decision < 0.8]                       # 80% → [MASK]
    random_bucket = selected[(decision >= 0.8) & (decision < 0.9)]  # 10% → random
    # remaining 10% (decision >= 0.9) are left unchanged — nothing to do.

    masked_ids[mask_bucket] = mask_token_id
    masked_ids[random_bucket] = torch.randint(
        low=0, high=vocab_size, size=(len(random_bucket),)
    )

    return masked_ids, labels