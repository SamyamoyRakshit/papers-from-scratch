import torch

def create_padding_mask(seq: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """
    Create a padding mask to prevent attention to <pad> tokens.

    Marks real tokens as 1 and <pad> tokens as 0.

    Args:
        seq: Token IDs, shape (batch_size, seq_len)
        pad_idx: Index of the <pad> token in the vocabulary

    Returns:
        Padding mask, shape (batch_size, 1, 1, seq_len)
        The two middle dimensions (1, 1) allow broadcasting across
        num_heads and query positions.
    """
    # (batch_size, seq_len) → (batch_size, 1, 1, seq_len)
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


def create_causal_mask(size: int) -> torch.Tensor:
    """
    Create a causal (look-ahead) mask to prevent attending to future positions.

    Lower-triangular matrix: position i can only attend to positions <= i.

    Args:
        size: Sequence length (tgt_seq_len)

    Returns:
        Causal mask, shape (1, 1, size, size)
        The two leading dimensions (1, 1) allow broadcasting across
        batch_size and num_heads.
    """
    # torch.tril: lower triangular — 1s on and below diagonal, 0s above.
    # dtype=torch.bool so this combines with the bool padding mask via `&`
    # (a float mask raises "Unsupported type Float" on bitwise AND).
    return torch.tril(torch.ones(size, size, dtype=torch.bool)).unsqueeze(0).unsqueeze(1)


def create_src_mask(src: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """
    Create source mask for encoder self-attention.

    Only needs padding mask — encoder attends to all real tokens.

    Args:
        src: Source token IDs, shape (batch_size, src_seq_len)
        pad_idx: Index of the <pad> token

    Returns:
        Source mask, shape (batch_size, 1, 1, src_seq_len)
    """
    return create_padding_mask(src, pad_idx)


def create_tgt_mask(tgt: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """
    Create target mask for decoder self-attention.

    Combines causal mask (no future peeking) with padding mask (no attending
    to <pad> tokens). Both conditions must be satisfied — use bitwise AND.

    Args:
        tgt: Target token IDs, shape (batch_size, tgt_seq_len)
        pad_idx: Index of the <pad> token

    Returns:
        Target mask, shape (batch_size, 1, tgt_seq_len, tgt_seq_len)
    """
    padding_mask = create_padding_mask(tgt, pad_idx)
    # padding_mask: (batch_size, 1, 1, tgt_seq_len) — broadcasts across rows

    causal_mask = create_causal_mask(tgt.size(1)).to(tgt.device)
    # causal_mask: (1, 1, tgt_seq_len, tgt_seq_len) — broadcasts across batch

    # Both must be 1 to attend: real token AND not future
    return padding_mask & causal_mask


def create_memory_mask(src: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """
    Create memory mask for decoder cross-attention.

    Only needs source padding mask — decoder can attend to all real source
    tokens (no causal restriction since source is already fully encoded).

    Args:
        src: Source token IDs, shape (batch_size, src_seq_len)
        pad_idx: Index of the <pad> token

    Returns:
        Memory mask, shape (batch_size, 1, 1, src_seq_len)
    """
    return create_padding_mask(src, pad_idx)