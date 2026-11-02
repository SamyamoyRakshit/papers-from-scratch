"""
train.bin / val.bin → (x, y) batches for next-token training.

nanoGPT-style: no Dataset, no DataLoader, no padding, no attention mask.
A batch is batch_size random windows of block_size + 1 ids; y is x shifted
left by one, so every position is a label (BERT: only the 15% masked).
"""
import numpy as np
import torch


def load_ids(path: str) -> np.memmap:
    """Open a .bin from prepare_data.py; ids are read from disk on access, not all loaded into RAM."""
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_batch(ids: np.memmap, batch_size: int, block_size: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    # randint's high is exclusive, so the last start is len - block_size - 1,
    # and that window of block_size + 1 ids ends exactly on the last id.
    starts = torch.randint(len(ids) - block_size, (batch_size,))
    # uint16 → int64: nn.Embedding and cross_entropy want int64 ids.
    windows = torch.stack([torch.from_numpy(ids[i:i + block_size + 1].astype(np.int64))
                           for i in starts.tolist()])
    x, y = windows[:, :-1], windows[:, 1:]    # y[:, t] is the token after x[:, t]
    return x.to(device), y.to(device)
