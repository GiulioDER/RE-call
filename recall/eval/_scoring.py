"""Which questions the RETRIEVAL metrics are scored on — ONE definition, two harnesses.

`recall.eval.labelled` and `recall.eval.longmemeval_perq` both publish a key called `hit_at_k`.
If one of them can be widened to every answerable question and the other cannot, that key names
two different populations depending on which module wrote the JSON, with nothing in either
artifact saying which — so the flag, its default, and its MEANING all live here rather than in
either harness.

This module is a leaf on purpose: it imports nothing from `recall`. The alternative (perq
importing the constant from `labelled`) pulls the whole indexing stack — `bm25`, `index`,
`retriever`, `trust` — into every `longmemeval_perq` import, to obtain a two-string tuple, and
`longmemeval_perq` already defers its other borrow from `labelled` into `main()` to avoid that.
"""
from __future__ import annotations

#: Which questions the RETRIEVAL metrics (`hit_at_k`, `mrr`, `latency_ms`, `misses`, and in
#: `longmemeval_perq` also `by_type` and `haystack_chunks`) are scored on.
#:
#: `"held"` is the default and reproduces every RATE this repo has published. `"all"` scores
#: every answerable question instead, which is methodologically free — `hit_at_k` and `mrr` never
#: read the calibration, so the fit/held split buys them nothing and halves their sample — but it
#: is NOT the default, because the default decides what a published figure MEANS. Silently
#: widening it would change the value of `docs/EVIDENCE.md`'s PEPs table under a reproduce command
#: that ships in the same file, and would move `results/gap/*.json` across 17 corpora without
#: changing one byte of their provenance. Opt in per run, and record which mode produced the
#: artifact.
#:
#: Abstention is unaffected by this setting: `false_abstain` and `abstention_accuracy` are always
#: scored on the held half, because the threshold is fitted on the other one.
SCORE_RETRIEVAL_ON = ("held", "all")

#: The default, named once. It was previously spelled `"held"` in four independent places — two
#: function signatures and two `argparse` defaults — and only the signatures were pinned by a
#: test, so a typo in either CLI would have shipped green on the one setting that decides what an
#: already-published `hit_at_k` counts.
DEFAULT_SCORE_RETRIEVAL_ON = SCORE_RETRIEVAL_ON[0]


def check_score_retrieval_on(score_retrieval_on: str) -> None:
    """Refuse an unknown mode loudly, instead of falling through to one of the two."""
    if score_retrieval_on not in SCORE_RETRIEVAL_ON:
        raise ValueError(
            f"score_retrieval_on must be one of {SCORE_RETRIEVAL_ON}, got {score_retrieval_on!r}"
        )


def scores_retrieval(*, is_held: bool, mode: str) -> bool:
    """Is an ANSWERABLE question at this side of the fit/held split in the retrieval sample?

    The caller supplies answerability, because the two harnesses reach it differently: one filters
    a list, the other branches inside its scoring loop. What must NOT differ is this rule, which
    is why it is a function and not a comment repeated in each of them.

    `is_held` is index parity — `questions[1::2]`, the half the abstention threshold is scored on.
    Under `"held"` only that half is retrieved; under `"all"` the fit half is retrieved too. Note
    what this predicate deliberately cannot express: it never widens ABSTENTION, because it is
    never consulted for it.
    """
    return is_held or mode == "all"
