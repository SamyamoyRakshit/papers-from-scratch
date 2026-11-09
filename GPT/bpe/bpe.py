"""
Byte-level BPE, trained from scratch (Sennrich et al., 2016; Radford et al., 2019, §2.2).

Training starts from the 256 byte values and repeatedly merges the most frequent
adjacent pair into a new token. Every merge gets the next id, so merge number r
is token 256 + r. After the merges comes one special token, <|endoftext|>:

    ids 0–255        the raw bytes
    ids 256 … N-2    one per merge, in the order they were learned
    id  N-1          <|endoftext|>        (GPT-2: N = 50257, so 50256)

Byte-level means there is no <unk>: any text is bytes, and every byte is a token.

<|endoftext|> never goes through `pretokenize`. The text is split on the whole
marker first, the pieces are BPE'd, and the marker is mapped straight to its id.
Sent through the regex, it would be cut into '<|', 'endoftext', '|>'.

Files, as in GPT-2:

    vocab.json   {"Ġà¦ķ": 1234, ...}   token (printable form) → id, for reading
    merges.txt   "Ġà¦ķ à¦²"            one merge per line, in rank order

`load` rebuilds everything from merges.txt alone; vocab.json is for humans.
"""

import json
import os
import re
from collections import Counter, defaultdict

from .byte_coder import decode_bytes, encode_bytes
from .pretokenize import pretokenize

EOT = "<|endoftext|>"
# re.escape is required: `|` is regex alternation. The group keeps the marker in the split.
_EOT_SPLIT = re.compile(f"({re.escape(EOT)})")


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every non-overlapping `pair` in `ids`, left to right: [1, 2, 1, 2] → [9, 9]."""
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self, merges: dict[tuple[int, int], int]):
        """
        Args:
            merges: (left id, right id) → new id, in rank order, e.g. {(224, 166): 256, ...}
        """
        self.merges = merges
        self.vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), new_id in merges.items():
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]
        self.eot_id = 256 + len(merges)
        self.vocab[self.eot_id] = EOT.encode("utf-8")
        self._cache: dict[str, list[int]] = {}  # chunk → ids; 8.1M chunks, only 428k distinct

    @classmethod
    def train(cls, text: str, vocab_size: int) -> "BPETokenizer":
        """
        Learn vocab_size - 257 merges from `text` (256 bytes + 1 for <|endoftext|>).

        BPE never merges across a chunk, so only the unique chunks and their counts
        matter: bn_wiki has 8.1M chunks but only 428k distinct ones.

        Pair counts are updated incrementally. Recounting all pairs after every merge
        costs ~2 s on bn_wiki, i.e. ~17 h for 32k merges; a merge only changes the
        chunks that contain the pair, so only those are recounted.
        """
        num_merges = vocab_size - 257
        assert num_merges >= 0, f"vocab_size must be ≥ 257, got {vocab_size}"

        counts = Counter()
        for piece in _EOT_SPLIT.split(text):
            if piece != EOT:
                counts.update(pretokenize(piece))
        words = [list(chunk.encode("utf-8")) for chunk in counts]
        freqs = list(counts.values())

        pair_counts = Counter()  # (a, b) → occurrences in the corpus
        where = defaultdict(set)  # (a, b) → indices of words that contain it (may be stale)
        for i, (w, f) in enumerate(zip(words, freqs)):
            for pair in zip(w, w[1:]):
                pair_counts[pair] += f
                where[pair].add(i)

        merges = {}
        for new_id in range(256, 256 + num_merges):
            if not pair_counts:
                break  # every chunk is already a single token
            # On a tie, the larger pair wins, so the result never depends on dict order.
            best = max(pair_counts, key=lambda p: (pair_counts[p], p))
            merges[best] = new_id
            for i in where.pop(best):
                w, f = words[i], freqs[i]
                for pair in zip(w, w[1:]):
                    pair_counts[pair] -= f
                    if pair_counts[pair] == 0:
                        del pair_counts[pair]
                w = words[i] = _merge(w, best, new_id)
                for pair in zip(w, w[1:]):
                    pair_counts[pair] += f
                    where[pair].add(i)
        return cls(merges)

    def _encode_chunk(self, chunk: str) -> list[int]:
        """Apply the merges to one chunk, lowest rank first, as GPT-2's `bpe()` does."""
        if chunk in self._cache:
            return self._cache[chunk]
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            pair = min(zip(ids, ids[1:]), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = _merge(ids, pair, self.merges[pair])
        self._cache[chunk] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        """
        Args:
            text: e.g. 'শেষ।<|endoftext|>শুরু'

        Returns:
            Token ids, with <|endoftext|> as the single id `self.eot_id`.
        """
        ids = []
        for piece in _EOT_SPLIT.split(text):
            if piece == EOT:
                ids.append(self.eot_id)
            else:
                for chunk in pretokenize(piece):
                    ids.extend(self._encode_chunk(chunk))
        return ids

    def decode(self, ids: list[int]) -> str:
        """
        Join the tokens' bytes and decode as UTF-8. A model can emit ids that cut a
        3-byte Bengali character in half; errors="replace" turns those into '�',
        as GPT-2's decoder does.
        """
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def save(self, out_dir: str) -> None:
        """Write vocab.json and merges.txt to `out_dir`."""
        os.makedirs(out_dir, exist_ok=True)
        vocab = {encode_bytes(self.vocab[i]): i for i in range(self.eot_id)}
        assert len(vocab) == self.eot_id, "two merges produced the same bytes; merges.txt would be ambiguous"
        vocab[EOT] = self.eot_id
        with open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=0)
        with open(os.path.join(out_dir, "merges.txt"), "w", encoding="utf-8") as f:
            f.write("#version: 0.2\n")
            for a, b in self.merges:
                f.write(f"{encode_bytes(self.vocab[a])} {encode_bytes(self.vocab[b])}\n")

    @classmethod
    def load(cls, out_dir: str) -> "BPETokenizer":
        """Rebuild the tokenizer from merges.txt. The byte map has no space, so ' ' splits each line safely."""
        to_id = {bytes([i]): i for i in range(256)}
        merges = {}
        with open(os.path.join(out_dir, "merges.txt"), encoding="utf-8") as f:
            next(f)  # "#version: 0.2"
            for new_id, line in enumerate(f, start=256):
                a, b = (decode_bytes(s) for s in line.rstrip("\n").split(" "))
                merges[(to_id[a], to_id[b])] = new_id
                to_id[a + b] = new_id
        return cls(merges)
