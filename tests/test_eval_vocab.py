"""Corpus vocabulary predictors — the candidate explanations for the cloud-embedder gap.

FINDINGS §8 measured a conditional rule (pay for a cloud embedder only when the corpus
vocabulary is unusual) but never operationalised "unusual". These are the candidate
operationalisations, and they exist to be *falsified* against a null model, not to be shipped
as truth. See `docs/EMBEDDER_GAP_STUDY.md` for what beat what.
"""
from __future__ import annotations

import pytest

from recall.eval.vocab import oov_rate


def _pieces(word: str) -> list[str]:
    """Stand-in subword tokenizer: known words stay whole, anything else shatters per character.

    Injected rather than mocked. The real tokenizer is a model artifact whose segmentation is
    not this module's business; what this module must get right is the *counting*.
    """
    known = {"the", "system", "restarts", "after", "a", "crash"}
    return [word] if word in known else list(word)


def test_oov_rate_counts_word_types_not_token_occurrences():
    # 'zqx' is the only shattered word, but it occurs 5 times out of 11 tokens.
    # Types: {the, system, zqx, restarts, after, a, crash} = 7, of which 1 is OOV -> 1/7.
    # Token-weighted would give 5/11. The distinction is the whole point: a corpus is
    # idiosyncratic because of how many odd words it uses, not how often it repeats one.
    texts = ["the system zqx zqx zqx", "zqx zqx restarts after a crash"]
    assert oov_rate(texts, _pieces) == pytest.approx(1 / 7)
    assert oov_rate(texts, _pieces) != pytest.approx(5 / 11)
