"""
GPT-2's reversible byte ↔ unicode map (Radford et al., 2019, §2.2).

The problem it solves: byte-level BPE operates on the 256 raw byte values, but
many of those (0x00–0x1F, 0x7F, 0x80–0xA0) are control characters or invalid
UTF-8 on their own. Writing them into `vocab.json` produces a file that is
unreadable, un-diffable, and mangled by any editor that touches it.

The fix, straight from OpenAI's `encoder.py`: bijectively map all 256 byte
values onto 256 *printable* unicode codepoints. Bytes that are already printable
(`!`–`~`, `¡`–`¬`, `®`–`ÿ`) map to themselves; the remaining 68 are pushed up
into the unused range starting at U+0100 (Ā, ā, Ă, …).

The famous artifact of this is `Ġ` (U+0120): byte 0x20, the space. GPT-2 vocab
dumps are full of `Ġthe`, `Ġis` — that is a leading space, not a Latin letter.
For Bengali the map is doing more work than it does for English: every Bengali
codepoint is 3 UTF-8 bytes, all of them ≥ 0x80, so *every* byte of Bengali text
lands in the remapped region. `ক` (U+0995 → e0 a6 95) shows up in the vocab file
as `à¦ķ`. Unreadable as Bengali, but stable, ASCII-safe, and reversible — which
is the entire point.

This is a pure encoding convenience. It never changes which merges happen: the
BPE loop runs on integers 0–255, and this map only affects how the trained vocab
is serialised to disk.
"""

from functools import lru_cache


@lru_cache(maxsize=1)
def bytes_to_unicode() -> dict[int, str]:
    """
    Build the 256-entry byte → printable-codepoint map.

    Returns:
        dict mapping each byte value 0–255 to a distinct printable character.
    """
    # The three already-printable runs, kept as themselves (inclusive ranges).
    bs = (
        list(range(ord("!"), ord("~") + 1))      # 33–126   ASCII printables
        + list(range(ord("¡"), ord("¬") + 1))    # 161–172  Latin-1 printables
        + list(range(ord("®"), ord("ÿ") + 1))    # 174–255  Latin-1 printables
    )
    cs = bs[:]

    # Everything else (the 68 control / gap bytes) gets pushed to U+0100 onward.
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1

    return dict(zip(bs, [chr(c) for c in cs]))


@lru_cache(maxsize=1)
def unicode_to_bytes() -> dict[str, int]:
    """The inverse map, used when loading a saved vocab back off disk."""
    return {c: b for b, c in bytes_to_unicode().items()}


def encode_bytes(token: bytes) -> str:
    """
    Render a raw token as its printable vocab-file form.

    Args:
        token: The token's raw bytes, e.g. b' \\xe0\\xa6\\x95'

    Returns:
        Printable string, e.g. 'Ġà¦ķ'
    """
    b2u = bytes_to_unicode()
    return "".join(b2u[b] for b in token)


def decode_bytes(s: str) -> bytes:
    """
    Parse a printable vocab-file token back to its raw bytes.

    Args:
        s: Printable string as written by `encode_bytes`.

    Returns:
        The original `bytes` object.
    """
    u2b = unicode_to_bytes()
    return bytes(u2b[c] for c in s)