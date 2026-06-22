import random


def build_nsp_example(
        a_index: int,
        document: list[list[int]],
        all_documents: list[list[list[int]]],
        cls_id: int,
        sep_id: int
):
    """
    Build ONE NSP example (Devlin et al. 2019, §3.1, "Task #2").

    Given sentence A and the document it came from, pick sentence B:
      - 50% → the REAL next sentence in the same document  → IsNext  (label 0)
      - 50% → a RANDOM sentence from a DIFFERENT document   → NotNext (label 1)

    Then assemble the single packed sequence BERT actually eats:

        [CLS] A [SEP] B [SEP]

    and the parallel `token_type_ids` (segment ids) that tell the model which
    half each position belongs to: 0 for `[CLS] A [SEP]`, 1 for `B [SEP]`.

    The label convention is the seam with `loss.py`: NSP is a 2-class CE where
    0 = IsNext, 1 = NotNext (matches `nsp_labels` in BERTPreTrainingLoss).

    Args:
        a_index: position of sentence A within `document`. Passed in (not
            searched for) so duplicate sentences can't be confused, and the
            TRUE next sentence is unambiguously `document[a_index + 1]`.
        document: the list of sentences (each a list of ids) A came from —
            used to grab the TRUE next sentence.
        all_documents: every document in the corpus — the pool we draw a
            random B from for the NotNext case.
        cls_id, sep_id: ids of [CLS] and [SEP].

    Returns:
        token_ids:      list[int] — [CLS] A [SEP] B [SEP].
        token_type_ids: list[int] — 0 over segment A, 1 over segment B.
        nsp_label:      int — 0 (IsNext) or 1 (NotNext).
    """
    sentence_a = document[a_index]
    has_real_next = a_index + 1 < len(document)

    # 50/50 — but if A is the last sentence (no real next), force NotNext.
    if has_real_next and random.random() < 0.5:
        sentence_b = document[a_index + 1]
        nsp_label = 0                                   # IsNext
    else:
        sentence_b = _random_sentence(all_documents, document)
        nsp_label = 1                                   # NotNext

    # Assemble [CLS] A [SEP] B [SEP] and the matching segment ids.
    token_ids = [cls_id] + sentence_a + [sep_id] + sentence_b + [sep_id]
    token_type_ids = (
        [0] * (len(sentence_a) + 2)                     # [CLS] A [SEP]
        + [1] * (len(sentence_b) + 1)                   # B [SEP]
    )

    return token_ids, token_type_ids, nsp_label


def _random_sentence(all_documents, exclude_document):
    """Pick a random sentence from any document that ISN'T A's document."""
    if len(all_documents) < 2:
        raise ValueError(
            "NSP needs at least 2 documents — can't draw a NotNext sentence "
            "from a different document with only one."
        )
    while True:
        doc = random.choice(all_documents)
        if doc is not exclude_document:     # skip A's own document — B must come from a different one
            return random.choice(doc)