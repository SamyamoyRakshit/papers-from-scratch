r"""
GPT-2's pre-tokenizer: the regex that cuts text into chunks before BPE runs
(Radford et al., 2019, §2.2).

BPE never merges across a chunk boundary, so this split decides what a token
*can* be. ` কলকাতায়` can only become one token if it reaches BPE as one chunk.

Two changes from GPT-2's pattern in OpenAI's `encoder.py`:

1. `\p{M}` is added next to `\p{L}`. Bengali vowel signs, hasanta and nukta are
   Unicode *marks*, not letters, so GPT-2's ` ?\p{L}+` stops at every one:

       'আমি কলকাতায় থাকি।'
       GPT-2 → ['আম', 'ি', ' কলক', 'া', 'ত', 'া', 'য', '়', ' থ', 'া', 'ক', 'ি।']
       ours  → ['আমি', ' কলকাতায়', ' থাকি', '।']

2. The English contractions ('s|'t|'re|'ve|'m|'ll|'d) are dropped. They never
   fire on Bengali and only cost regex time.

The three Unicode classes, on the word কলকাতায় (ক ল ক া ত া য ়):

    \p{L}  letter   ['কলক', 'ত', 'য']
    \p{M}  mark     ['া', 'া', '়']
    \p{N}  number   []

The alternatives, tried left to right (the first one that matches wins):

     ?[\p{L}\p{M}]+          space? + word          ' কলকাতায়'  ' reading'
     ?\p{N}+                 space? + digits        ' ২০২৪'  ' 2024'
     ?[^\s\p{L}\p{M}\p{N}]+  space? + punctuation   '।'  ','  ' (('
    \s+(?!\S)                spaces, except the     'a   b' → ['a', '  ', ' b']
                             last one before a word
    \s+                      leftover whitespace    trailing '\n'

Needs the `regex` package: stdlib `re` has no `\p{...}`.
"""

import regex

PATTERN = regex.compile(
    r" ?[\p{L}\p{M}]+"
    r"| ?\p{N}+"
    r"| ?[^\s\p{L}\p{M}\p{N}]+"
    r"|\s+(?!\S)"
    r"|\s+"
)


def pretokenize(text: str) -> list[str]:
    """
    Split text into the chunks BPE trains and encodes on.

    Args:
        text: Raw text, e.g. 'আমি কলকাতায় থাকি।'

    Returns:
        The chunks in order, e.g. ['আমি', ' কলকাতায়', ' থাকি', '।'].
        Joining them gives back `text` exactly.
    """
    return PATTERN.findall(text)
