# Building the pre-training corpus (`prepare_corpus.py`)

> Module: [`BERT/scripts/prepare_corpus.py`](../../scripts/prepare_corpus.py)
> Runs **once**, before any training. Standalone — nothing in the library imports it.

BERT pre-trains on raw text, so before anything else we need a **corpus file**: plain
Bengali text, shaped so the rest of the pipeline can read it. `prepare_corpus.py`
downloads Bengali Wikipedia, cleans each article, and writes one `.txt` that
[`build_documents()`](../utils/data_utils.md) consumes.

It is a **data-prep helper**, not part of the model code. The dependency runs one way
only:

```
prepare_corpus.py  ──writes──▶  corpus .txt  ──read by──▶  build_documents()  ──▶  Dataset
```

So if the script is absent, nothing in the codebase breaks — you just need the `.txt`
to exist (built however you like).

## Contents

- [Where this sits](#where-this-sits)
- [It's config-driven](#its-config-driven)
- [The flow, end to end](#the-flow-end-to-end)
- [The one rule that matters: one article = one line](#the-one-rule-that-matters-one-article--one-line)
- [The optional keep-hook](#the-optional-keep-hook)
- [How to run it](#how-to-run-it)
- [What the output looks like](#what-the-output-looks-like)
- [The contract with `build_documents`](#the-contract-with-build_documents)
- [Gotchas](#gotchas)

---

## Where this sits

```
 Hugging Face:  wikimedia/wikipedia  (config 20231101.bn — Bengali)
   │   one row = one article = { id, url, title, text }
   ▼
 prepare_corpus.py
   ├─ skip stubs (too short)
   ├─ (optional) keep-hook — drop unwanted articles
   ├─ collapse each article to ONE line
   └─ write: article, blank line, article, blank line, ...
   ▼
 data/bn_wiki.txt          ← the corpus file
   │
   ▼
 build_documents()  (data_utils.py)  →  all_documents  →  Dataset → batches
```

Why Wikipedia and not a sentence dump (e.g. a deduplicated sentence corpus)? Because
NSP needs **documents** — contiguous sentences with real "next sentence" relationships.
A Wikipedia **article is a document**; a shuffled bag of sentences has no document
structure, so NSP would be meaningless.

## It's config-driven

Every knob comes from the YAML config (read via `Config.from_yaml`), not hardcoded:

```yaml
# configs/base.yaml
data:
  dataset: wikimedia/wikipedia
  wiki_dump: "20231101.bn"      # Bengali Wikipedia snapshot
  corpus_path: data/bn_wiki.txt
  max_articles: 20000           # null = use the full dump
  min_chars: 200                # skip stubs shorter than this
```

```python
def prepare_corpus(config):
    data = config.data
    ds = load_dataset(data.dataset, data.wiki_dump, split="train")   # both from config
    ...
```

So `base.yaml` and `tiny.yaml` can point at different sizes without touching code.

## The flow, end to end

```
load_dataset(dataset, wiki_dump)         ── all Bengali Wikipedia articles
        │
   max_articles set?  ──yes──▶  ds.select(range(max_articles))   ── cap for speed/memory
        │
        ▼
   for each article:
        text = article["text"].strip()
        │
        ├─ len(text) < min_chars ? ──▶ skip  (dropped_short += 1)
        │
        ├─ keep-hook says drop ?   ──▶ skip  (dropped_filter += 1)
        │
        ▼
        text = " ".join(text.split())     ── collapse to ONE line
        f.write(text + "\n\n")            ── article + blank-line boundary
        kept += 1
        │
        ▼
   print(kept | dropped_filter | dropped_short)
```

The three counters at the end are your sanity check — they tell you how much survived
and why the rest didn't.

## The one rule that matters: one article = one line

This is the heart of the script. Wikipedia's `text` field has **blank lines between
paragraphs and section headers**. If those survived into the corpus, the downstream
splitter (which treats *a blank line as a document boundary*) would shatter one article
into many tiny "documents" — breaking NSP. So we flatten each article to a single line,
and use the blank line **only** between articles.

**Raw `article["text"]`** (newlines shown):

```
মধুসূদন দত্ত একজন কবি।\n\nতিনি ১৮২৪ সালে জন্মান।\n\nজীবন\n\n১৮২৪ সালের ২৫ জানুয়ারি তিনি জন্মগ্রহণ করেন।
```

**Step 1 — `text.strip()`** removes leading/trailing whitespace only (ends, not inside):

```
"\n\n  মধুসূদন দত্ত একজন কবি। ...  \n"   →   "মধুসূদন দত্ত একজন কবি। ..."
```

**Step 2 — `" ".join(text.split())`** collapses every internal newline/space run into a
single space → the whole article becomes one line:

```
মধুসূদন দত্ত একজন কবি। তিনি ১৮২৪ সালে জন্মান। জীবন ১৮২৪ সালের ২৫ জানুয়ারি তিনি জন্মগ্রহণ করেন।
```

`text.split()` with no argument splits on *any* whitespace and drops the empties, so
`['মধুসূদন','দত্ত','একজন','কবি।','তিনি', ...]` — no newlines left; `" ".join(...)` glues
them with single spaces.

**Step 3 — `f.write(text + "\n\n")`** writes the one-line article, then a blank line as
the boundary:

```
মধুসূদন দত্ত একজন কবি। তিনি ১৮২৪ সালে জন্মান। ...      ← article 1 (one line)
                                                       ← blank line = boundary
কলকাতা একটি শহর। এটি বড়।                               ← article 2
                                                       ← blank line
```

Now the **only** blank lines in the file are real article boundaries — exactly what
`build_documents` keys on.

## The optional keep-hook

`prepare_corpus` supports an **optional, pluggable** per-article filter. It tries to
import a `keep_article(text)` function; if none is available it falls back to keeping
**every** article:

```python
try:
    from ._corpus_filter import keep_article
except ImportError:
    def keep_article(text):     # default: keep everything
        return True
```

- **Hook present** → `keep_article` decides per article (`True` = keep, `False` = drop),
  and the drop count shows up in `dropped (filter)`.
- **Hook absent** → the fallback returns `True` for everything → `dropped (filter)` is
  `0`, and you get the full, unfiltered corpus.

This is just an extensibility point — somewhere to plug in corpus curation
(deduplication, topic selection, language checks, …) **without editing the script**.
The fallback guarantees the script always runs, with or without a hook.

In the loop it's a one-liner:

```python
if not keep_article(text):
    dropped_filter += 1
    continue
```

## How to run it

From the repo root, as a module (so `..utils.config` resolves):

```bash
python -m BERT.scripts.prepare_corpus                       # uses configs/base.yaml
python -m BERT.scripts.prepare_corpus BERT/configs/tiny.yaml   # any config you pass
```

The entry point picks the config path from the command line, else a default:

```python
config_path = sys.argv[1] if len(sys.argv) > 1 else "BERT/configs/base.yaml"
prepare_corpus(Config.from_yaml(config_path))
```

`sys.argv[0]` is the script's own path (auto); `sys.argv[1]` is the first thing **you**
type after it — so `len > 1` means "a config was passed."

A run prints:

```
Kept 18742 | dropped (filter) 0 | dropped (short) 1258
→ data/bn_wiki.txt
```

## What the output looks like

```
মধুসূদন দত্ত একজন কবি। তিনি ১৮২৪ সালে জন্মান।

কলকাতা একটি শহর। এটি বড়।

দার্জিলিং একটি পার্বত্য শহর। এটি চা বাগানের জন্য বিখ্যাত।
```

Each article is one line; a blank line separates articles. Plain UTF-8 text — no special
tokens, no ids; tokenization happens later in `build_documents`.

## The contract with `build_documents`

The corpus file is the **handoff** between this script and the library. The promise:

| separator | means | who makes it |
|---|---|---|
| **blank line** (`\n\n`) | document boundary | `prepare_corpus` (`f.write(text + "\n\n")`) |
| **`।` / `.` / `!` / `?`** | sentence boundary | natural punctuation in the text |

`build_documents` reads exactly this: split on blank lines → documents, split each on
`।` → sentences. See [`data_utils.md`](../utils/data_utils.md#2-corpus--all_documents).

## Gotchas

- **Run it before training.** `build_documents` needs the `.txt` to already exist; this
  script is what creates it. Run once (re-run only to rebuild with different settings).
- **It's standalone.** Nothing imports `prepare_corpus.py`. Its absence can't raise an
  ImportError anywhere — the only consequence is you won't have a freshly-built corpus.
- **Watch the counters.** If `dropped (short)` eats most of the corpus, your `min_chars`
  is too high. The counts are there precisely so a bad config is obvious immediately.
- **Needs `datasets`.** Same dependency the transformer already used — no new install.
