"""Recheck measures whether temperature 0 actually held, rather than assuming it did.

Temperature 0 is not a determinism guarantee from any hosted provider. If it does not hold,
then the CACHE, not the sampler, is what makes runs reproducible, and that is worth knowing
before a cache eviction silently renumbers every proposal id derived from it.

Properties, one test each:

1. A stable engine reports zero mismatches.
2. A drifting engine is reported, not hidden.
3. Recheck does not mutate the cache. A measurement that changes what it measures cannot be
   repeated, and overwriting the entry would erase the very evidence of drift.
4. Only cached files are checked; an uncached file is not silently counted as agreeing.
5. The engine is actually re-called. A recheck that read the cache twice would report perfect
   determinism for an engine that had never been asked again.
6. A file whose recheck refuses at a batch rung counts as a mismatch when the cache holds
   claims, rather than being read as agreement.

Every assertion calls the production function. Nothing here re-implements it.
"""
import pytest

from recall.truth_extraction._cache import (
    InMemoryExtractionCache,
    RecheckReport,
    recheck_cached_extractions,
)
from recall.truth_extraction._engine import DeterministicExtractionEngine
from recall.truth_extraction.extract import extract_corpus_claims

# Self-referential so the supersession targets RESOLVE. A target outside the corpus is refused
# at the `target_not_in_corpus` rung, which would leave every cached entry holding zero claims
# and make a drift test pass while comparing nothing to nothing.
DOCS = {
    "old_2026-01-01.md": "The original call.\n",
    "new_2026-02-01.md": "This replaces old_2026-01-01.md after review.\n",
}
NAMES = tuple(sorted(DOCS))


class _Drifting(DeterministicExtractionEngine):
    """Answers normally once, then returns an empty batch for the same prompt."""

    def __init__(self) -> None:
        self.seen: set[str] = set()
        self.calls = 0

    def run(self, prompt) -> str:
        self.calls += 1
        if prompt.human_body in self.seen:
            return '{"claims": []}'
        self.seen.add(prompt.human_body)
        return super().run(prompt)


class _Counting(DeterministicExtractionEngine):
    def __init__(self) -> None:
        self.calls = 0

    def run(self, prompt) -> str:
        self.calls += 1
        return super().run(prompt)


def _warm(engine, documents=DOCS):
    cache = InMemoryExtractionCache()
    # `corpus_names` is pinned to NAMES rather than left to default to `documents`' own keys.
    # It is part of the cache key on purpose (a corpus that gained a file can resolve a target
    # that previously refused), so warming a subset with its own narrower names would produce
    # keys that never match the recheck, and every test here would measure a cold cache.
    extract_corpus_claims(documents, engine=engine, corpus_names=NAMES, cache=cache)
    return cache


def test_a_stable_engine_reports_zero_mismatches():
    engine = _Counting()
    cache = _warm(engine)
    report = recheck_cached_extractions(
        DOCS, engine=engine, corpus_names=NAMES, cache=cache
    )
    assert isinstance(report, RecheckReport)
    assert report.checked == 2
    assert report.mismatched == 0
    assert report.mismatched_files == ()


def test_a_drifting_engine_is_reported_not_hidden():
    engine = _Drifting()
    cache = _warm(engine)
    report = recheck_cached_extractions(
        DOCS, engine=engine, corpus_names=NAMES, cache=cache
    )
    assert report.mismatched >= 1
    assert "new_2026-02-01.md" in report.mismatched_files


class _WriteRecordingCache(InMemoryExtractionCache):
    """Records every `put`, so a write is detectable even when it stores an identical value."""

    def __init__(self) -> None:
        super().__init__()
        self.writes: list[str] = []

    def put(self, key, value) -> None:
        self.writes.append(key)
        super().put(key, value)


def test_recheck_never_writes_to_the_cache():
    """Comparing contents is not enough: a write that stores the SAME value changes nothing.

    An earlier version of this test compared before and after, and a mutation that wrote the
    cached entry straight back left every test green. The property that can actually fail is
    that recheck performs no write at all.
    """
    engine = _Drifting()
    cache = _WriteRecordingCache()
    extract_corpus_claims(DOCS, engine=engine, corpus_names=NAMES, cache=cache)
    writes_after_warming = len(cache.writes)
    assert writes_after_warming > 0, "the warm run must write, or this test proves nothing"

    recheck_cached_extractions(DOCS, engine=engine, corpus_names=NAMES, cache=cache)
    assert len(cache.writes) == writes_after_warming, (
        "recheck wrote to the cache; a measurement that changes what it measures cannot be "
        "repeated, and overwriting the entry erases the drift it exists to surface"
    )


def test_recheck_leaves_the_cache_contents_alone():
    engine = _Drifting()
    cache = _warm(engine)
    before = {k: cache.get(k) for k in list(cache._entries)}  # noqa: SLF001 - asserting storage
    recheck_cached_extractions(DOCS, engine=engine, corpus_names=NAMES, cache=cache)
    after = {k: cache.get(k) for k in list(cache._entries)}  # noqa: SLF001
    assert after == before, "recheck must measure the cache, not overwrite it"


def test_an_uncached_file_is_not_counted_as_checked():
    """Silently counting a miss as agreement would inflate the determinism it reports."""
    engine = _Counting()
    cache = _warm(engine, {"old_2026-01-01.md": DOCS["old_2026-01-01.md"]})
    report = recheck_cached_extractions(
        DOCS, engine=engine, corpus_names=NAMES, cache=cache
    )
    assert report.checked == 1


def test_the_engine_is_actually_re_called():
    """A recheck that read the cache twice would report perfect determinism, always."""
    engine = _Counting()
    cache = _warm(engine)
    before = engine.calls
    recheck_cached_extractions(DOCS, engine=engine, corpus_names=NAMES, cache=cache)
    assert engine.calls == before + 2, "recheck did not re-ask the engine"


def test_a_refused_recheck_counts_as_a_mismatch_not_agreement():
    """An answer that now fails the ladder is a change, and must not read as agreement."""

    class _GoesBad(DeterministicExtractionEngine):
        def __init__(self) -> None:
            self.seen: set[str] = set()

        def run(self, prompt) -> str:
            if prompt.human_body in self.seen:
                return "not json at all"
            self.seen.add(prompt.human_body)
            return super().run(prompt)

    engine = _GoesBad()
    cache = _warm(engine)
    report = recheck_cached_extractions(
        DOCS, engine=engine, corpus_names=NAMES, cache=cache
    )
    assert "new_2026-02-01.md" in report.mismatched_files


def test_the_warm_cache_actually_holds_claims():
    """Guards every test above: comparing nothing to nothing would agree trivially."""
    cache = _warm(_Counting())
    stored = [cache.get(k) for k in list(cache._entries)]  # noqa: SLF001
    assert any(entry is not None and entry.claims for entry in stored)


def test_a_total_collapse_is_not_reported_as_agreement():
    """The failure that made the claims-only comparison nearly useless.

    Most memos yield no claims at all: on this repo's own docs corpus, 34 of 36 cached entries
    hold `claims == ()`. Comparing claims alone made a file that now fails the ladder entirely
    compare `() != ()` as EQUAL, so replacing every answer with garbage reported a 5.6%
    mismatch rate and called 34 files deterministic. Comparing the batch rung too is what
    distinguishes "passed the ladder and found nothing" from "failed the ladder".
    """

    class _Collapse(DeterministicExtractionEngine):
        def run(self, prompt) -> str:
            return "not json at all"

    cache = _warm(_Counting())
    report = recheck_cached_extractions(
        DOCS, engine=_Collapse(), corpus_names=NAMES, cache=cache
    )
    assert report.checked == 2
    assert report.mismatched == 2, "a file caching zero claims was read as agreeing"
    assert set(report.mismatched_files) == set(DOCS)


def test_the_fixture_has_a_file_that_caches_zero_claims():
    """Guards the test above: it only bites when some cached entry holds no claims."""
    cache = _warm(_Counting())
    stored = [cache.get(k) for k in list(cache._entries)]  # noqa: SLF001
    assert any(e is not None and not e.claims for e in stored)
    assert any(e is not None and e.claims for e in stored)


def test_an_engine_failure_is_counted_apart_from_agreement_and_drift():
    """A dropped connection is not evidence about determinism in either direction."""

    class _Down(DeterministicExtractionEngine):
        def run(self, prompt) -> str:
            raise ConnectionError("connection reset by peer")

    cache = _warm(_Counting())
    report = recheck_cached_extractions(
        DOCS, engine=_Down(), corpus_names=NAMES, cache=cache
    )
    assert report.errored == 2
    assert report.checked == 0
    assert report.mismatched == 0
    assert set(report.errored_files) == set(DOCS)


def test_one_engine_failure_does_not_abort_the_whole_recheck():
    """Measurements already made must survive, as they do in extract_file_claims."""

    class _Flaky(DeterministicExtractionEngine):
        def run(self, prompt) -> str:
            if prompt.file == "new_2026-02-01.md":
                raise ConnectionError("blip")
            return super().run(prompt)

    cache = _warm(_Counting())
    report = recheck_cached_extractions(
        DOCS, engine=_Flaky(), corpus_names=NAMES, cache=cache
    )
    assert report.checked == 1, "the surviving measurement was lost"
    assert report.errored == 1


def test_the_recheck_stops_asking_a_dead_engine():
    class _Down(DeterministicExtractionEngine):
        def __init__(self) -> None:
            self.calls = 0

        def run(self, prompt) -> str:
            self.calls += 1
            raise ConnectionError("refused")

    documents = {f"{i:02d}.md": f"Doc {i}. This replaces {i - 1:02d}.md.\n" if i else "Doc 0.\n"
                 for i in range(20)}
    names = tuple(sorted(documents))
    cache = InMemoryExtractionCache()
    extract_corpus_claims(documents, engine=_Counting(), corpus_names=names, cache=cache)

    engine = _Down()
    report = recheck_cached_extractions(
        documents, engine=engine, corpus_names=names, cache=cache
    )
    assert engine.calls == 3, f"the engine was asked {engine.calls} times against a dead endpoint"
    assert report.errored == 20


def test_corpus_name_order_does_not_change_the_question_asked():
    """The key hashes SORTED names; the prompt renders them in the order GIVEN.

    So passing the same names in a different order than the warm run hits the cached entry
    while asking the model a textually different question, and any difference in the answer is
    then blamed on the engine.

    Asserting on claims cannot catch this: the deterministic engine never reads `corpus_names`
    from the prompt text, so both orders give identical claims and the test passes whether or
    not recheck normalises. The property that can actually fail is that the PROMPT TEXT is the
    same either way, so this records what the engine was asked.
    """

    class _Recording(DeterministicExtractionEngine):
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def run(self, prompt) -> str:
            self.prompts.append(prompt.user)
            return super().run(prompt)

    cache = _warm(_Counting())

    forward = _Recording()
    recheck_cached_extractions(DOCS, engine=forward, corpus_names=NAMES, cache=cache)
    reversed_ = _Recording()
    recheck_cached_extractions(
        DOCS, engine=reversed_, corpus_names=tuple(reversed(NAMES)), cache=cache
    )

    assert forward.prompts, "the engine was never asked, so this proves nothing"
    assert forward.prompts == reversed_.prompts, (
        "corpus name order changed the prompt text, so the same cached entry was compared "
        "against the answer to a different question"
    )


def test_a_cold_cache_reports_no_rate_rather_than_zero_drift():
    """`0.0` would render as "no drift detected" when nothing was measured at all."""
    report = recheck_cached_extractions(
        DOCS, engine=_Counting(), corpus_names=NAMES, cache=InMemoryExtractionCache()
    )
    assert report.checked == 0
    assert report.mismatch_rate is None


def test_a_mismatch_rate_is_reportable():
    engine = _Drifting()
    cache = _warm(engine)
    report = recheck_cached_extractions(
        DOCS, engine=engine, corpus_names=NAMES, cache=cache
    )
    assert 0.0 < report.mismatch_rate <= 1.0
    assert report.mismatch_rate == pytest.approx(report.mismatched / report.checked)
