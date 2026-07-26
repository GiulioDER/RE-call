"""Corpus vocabulary predictors — the candidate explanations for the cloud-embedder gap.

FINDINGS §8 measured a conditional rule (pay for a cloud embedder only when the corpus
vocabulary is unusual) but never operationalised "unusual". These are the candidate
operationalisations, and they exist to be *falsified* against a null model, not to be shipped
as truth. See `docs/EMBEDDER_GAP_STUDY.md` for what beat what.
"""
from __future__ import annotations

import math

import pytest

from recall.eval.vocab import (
    bge_encoder,
    crowding,
    oov_rate,
    query_overlap,
    subword_pieces,
)


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


def _bert_style_encode(word: str) -> list[str]:
    """A BERT-family tokenizer's raw output: every input is wrapped in [CLS] ... [SEP]."""
    inner = {"crash": ["crash"], "zqx": ["z", "##q", "##x"]}[word]
    return ["[CLS]", *inner, "[SEP]"]


def test_subword_pieces_drops_the_special_tokens():
    # bge-small is BERT-family, so raw encode() wraps every single word in [CLS]/[SEP]. Counting
    # those makes the shortest possible word three pieces, which makes EVERY type look shattered
    # and oov_rate exactly 1.0 for every corpus ever measured — a number that is plausible,
    # comparable across corpora, and completely wrong. This is the failure the study cannot
    # survive, so it is pinned before the loader that would produce it exists.
    assert subword_pieces("crash", _bert_style_encode) == ["crash"]
    assert subword_pieces("zqx", _bert_style_encode) == ["z", "##q", "##x"]


def test_oov_rate_over_a_bert_style_encoder_separates_the_shattered_word():
    # The composition that matters: with specials stripped, 'crash' is one piece (in-vocabulary)
    # and 'zqx' is three (shattered), so a two-type corpus scores exactly 0.5. Leave the specials
    # in and both are >=3 pieces, so it scores 1.0 and the metric is dead.
    assert oov_rate(["crash zqx"], lambda w: subword_pieces(w, _bert_style_encode)) == pytest.approx(0.5)


def test_bge_encoder_splits_a_codename_but_not_a_common_word():
    """The metric's load-bearing assumption, checked against the real tokenizer.

    `oov_rate` only means anything if the tokenizer it is handed actually distinguishes words the
    model knows from words it does not. Everything upstream of this is a fake encoder agreeing
    with itself. Skips (rather than fails) when the tokenizer cannot be fetched, because that is
    an environment problem, not a defect in the metric.
    """
    try:
        encode = bge_encoder()
    except Exception as exc:  # offline, no HF cache, hub down
        pytest.skip(f"bge tokenizer unavailable: {type(exc).__name__}: {exc}")

    # Ordinary English the model was pretrained on: one piece, therefore not OOV.
    for word in ("crash", "system", "database", "retrieval"):
        assert len(subword_pieces(word, encode)) == 1, word

    # Invented identifiers of the kind the private memory corpus is full of: shattered.
    for word in ("rmk_kill", "riskbook", "zqxvt"):
        assert len(subword_pieces(word, encode)) > 1, word


def _heaps_corpus(n_docs: int) -> list[str]:
    """`n_docs` documents with IDENTICAL per-document composition: 4 words from a fixed pool of
    common words, plus one codename unique to that document.

    Composition is constant by construction, so any corpus-level statistic that claims to measure
    *composition* must return the same value for n_docs=10 and n_docs=200. A type-based one does
    not: the common pool saturates at 20 types while codename types grow with the corpus, which is
    Heaps' law and is exactly the confound that would let corpus SIZE masquerade as vocabulary
    idiosyncrasy.
    """
    pool = [f"word{i:02d}" for i in range(20)]
    return [
        " ".join([pool[(d * 4 + j) % 20] for j in range(4)] + [f"zqxcode{d:04d}"])
        for d in range(n_docs)
    ]


def _pool_aware_pieces(word: str) -> list[str]:
    """Common-pool words are in vocabulary; the unique codenames shatter."""
    return [word] if word.startswith("word") else list(word)


def test_oov_rate_without_a_token_budget_tracks_corpus_size_not_composition():
    # The defect, demonstrated. Same composition, 20x the documents, and the statistic nearly
    # doubles. Any predictor built on this would partly be measuring "how big is your corpus",
    # and the study's corpora differ in size by more than 10x.
    small = oov_rate(_heaps_corpus(10), _pool_aware_pieces)
    large = oov_rate(_heaps_corpus(200), _pool_aware_pieces)
    assert small == pytest.approx(10 / 30)   # 10 codenames + 20 pool words
    assert large == pytest.approx(200 / 220) # 200 codenames + the same 20 pool words
    assert large - small > 0.4               # pure size artefact


def test_oov_rate_with_a_token_budget_is_stable_across_corpus_sizes():
    # The fix: count types over a fixed number of TOKENS rather than over everything present.
    # Heaps' law relates types to tokens, so holding tokens fixed is what makes two corpora of
    # different sizes comparable at all.
    small = oov_rate(_heaps_corpus(10), _pool_aware_pieces, token_budget=50, seed=0)
    large = oov_rate(_heaps_corpus(200), _pool_aware_pieces, token_budget=50, seed=0)
    assert abs(large - small) < 0.15


def test_oov_rate_token_budget_is_deterministic_for_a_given_seed():
    # A statistic that moves between runs cannot be correlated against anything.
    texts = _heaps_corpus(200)
    a = oov_rate(texts, _pool_aware_pieces, token_budget=100, seed=7)
    b = oov_rate(texts, _pool_aware_pieces, token_budget=100, seed=7)
    c = oov_rate(texts, _pool_aware_pieces, token_budget=100, seed=8)
    assert a == b
    assert a != c  # different seed genuinely resamples rather than ignoring the seed


def test_query_overlap_ignores_function_words():
    """The defect this metric has to survive: function words swamp the signal.

    Every English question overlaps every English document on {the, is, of, a}. Counted, the
    statistic sits near a corpus-independent constant and correlates with nothing — the same
    silent-plausible-number failure as [CLS] and Heaps' law. Here the two queries share three
    function words and no content at all: unfiltered that scores 0.5, which is noise wearing the
    costume of evidence.
    """
    assert query_overlap("what is the status of the deployment", "the cost of a widget is fixed") == 0.0


def test_query_overlap_is_the_share_of_query_content_words_present_in_the_document():
    # Content words in the query: {status, deployment, pipeline}. The document contains 'status'
    # and 'pipeline' but not 'deployment' -> 2/3. Query-side coverage, deliberately: the question
    # is how much of what the ASKER said survives into the document, which is what a lexical
    # retriever gets for free and a dense retriever must supply.
    q = "what is the status of the deployment pipeline"
    doc = "the pipeline reports its status once a minute"
    assert query_overlap(q, doc) == pytest.approx(2 / 3)


def test_query_overlap_is_nan_when_the_query_has_no_content_words():
    # A query of pure function words has no denominator. Returning 0.0 would mean "no overlap",
    # which is a measurement; this is the absence of one, and averaging it in would drag a
    # corpus's mean down for a reason that has nothing to do with its vocabulary.
    assert math.isnan(query_overlap("what is it", "anything at all"))


def test_crowding_excludes_each_document_from_its_own_neighbourhood():
    """The defect: every vector's nearest neighbour is itself, at cosine 1.0.

    Left in, `crowding` returns exactly 1.0 for every corpus ever measured — the [CLS] failure
    again, in a different costume. Three mutually orthogonal documents are as UNcrowded as a
    corpus can be, so the honest answer is 0.0.
    """
    assert crowding([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]) == pytest.approx(0.0)


def test_crowding_is_one_when_documents_are_indistinguishable():
    # Two identical documents: each one's nearest OTHER document sits on top of it.
    assert crowding([[1.0, 0.0], [1.0, 0.0]]) == pytest.approx(1.0)


def test_crowding_is_invariant_to_vector_magnitude():
    # Cosine, not dot product. An embedder that returns unnormalised vectors would otherwise make
    # a corpus look crowded for having long vectors, which is a property of the model's output
    # scale and not of whether its documents can be told apart.
    unit = crowding([[1.0, 0.0], [1.0, 1.0]])
    scaled = crowding([[10.0, 0.0], [1.0, 1.0]])
    assert unit == pytest.approx(1 / math.sqrt(2))
    assert scaled == pytest.approx(unit)


def test_crowding_is_nan_for_fewer_than_two_documents():
    # "How close is the nearest other document" is undefined when there is no other document.
    assert math.isnan(crowding([[1.0, 0.0]]))
    assert math.isnan(crowding([]))
