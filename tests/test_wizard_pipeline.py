"""The per-corpus pipeline: what runs, in what order, and what it refuses before spending time.

Three properties are worth more than the rest and are what this file mostly pins.

**Order is load-bearing and is not a comment.** Promotion gives a generation a fresh corpus
fingerprint, so a calibration measured before it binds to the predecessor and comes back
`CALIBRATION_STALE` after it. Correct order is build, validate, calibrate, publish, promote. A test
that only checks the end state cannot see an inversion, so these assert the SEQUENCE.

**The certification floor is checked before the build, not after.** The design puts query generation
after `generation validate`, and it does not have to be there: `chunks_from_directory` reads the
corpus off disk rather than out of the database, so the query set can be generated and counted first.
That matters because a corpus too small to certify is the one failure that cannot be recovered from
later: it builds and promotes happily and then refuses every query forever. Measured previously at
roughly seven minutes for a 1793-chunk build, which is what checking first saves on a doomed corpus.

**A corpus that fails certification degrades, it does not abort the install.** The other corpora are
independent, and a wizard that dies on the third of three leaves the user worse off than one that
reports which corpus is degraded and why.

No database here. Every collaborator is injected, so the ordering and refusal logic is testable
without a container; the DB-backed end-to-end lives in the `requires_db` test at the bottom.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from recall.wizard.corpora import docs_corpus, memory_corpus


class _Recorder:
    """Records the calls a pipeline makes, in order, so a sequence can be asserted."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def note(self, name: str) -> None:
        self.calls.append(name)


class _FakeManager:
    """Stands in for `GenerationManager`, recording the generation lifecycle calls."""

    def __init__(self, recorder: _Recorder, *, environment: str = "development") -> None:
        self._recorder = recorder
        self.environment = environment
        self.built = False

    def create(self, manifest: Any, pipeline: Any, *, allow_unverified: bool = False) -> Any:
        self._recorder.note("create")
        from types import SimpleNamespace

        return SimpleNamespace(generation_id="gen_test")

    def build(
        self, generation_id: str, reader: Any, embedder: Any, chunker: Any, *, provenance: Any = None
    ) -> Any:
        self._recorder.note("build")
        self.built = True
        from types import SimpleNamespace

        return SimpleNamespace(generation_id=generation_id, objects=1, chunks=40, reused_objects=0)

    def validate(self, generation_id: str) -> Any:
        self._recorder.note("validate")
        from types import SimpleNamespace

        return SimpleNamespace(generation_id=generation_id, sources=1, chunks=40)

    def promote(self, generation_id: str, *, unsafe_development: bool = False) -> None:
        self._recorder.note("promote")


class _FakeCalibrations:
    """Stands in for `CalibrationRepository`."""

    def __init__(self, recorder: _Recorder, *, certified: bool = True) -> None:
        self._recorder = recorder
        self._certified = certified

    def calibrate(self, generation_id: str, queries: Any, embedder: Any) -> Any:
        self._recorder.note("calibrate")
        from types import SimpleNamespace

        return SimpleNamespace(
            calibration_id="cal_test",
            generation_id=generation_id,
            certified=self._certified,
            certification_reason="" if self._certified else "separability below the bar",
            separability=0.97 if self._certified else 0.61,
        )

    def publish(self, calibration_id: str) -> Any:
        self._recorder.note("publish")
        from types import SimpleNamespace

        return SimpleNamespace(calibration_id=calibration_id, certified=self._certified)


class _FakeEmbedder:
    name = "hashing-64"
    dim = 64

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * self.dim for _ in texts]


_SYLLABLES = ("ka", "lo", "mi", "ne", "pu", "ra", "se", "ti", "vo", "zu", "be", "cha", "dro", "fen", "gil")

#: Filler shared by every document, so it has a high document frequency and cannot be mistaken for
#: what any one chunk is about. `_distinctive_terms` ranks by `tf * log(total / df)`.
_FILLER = "the system stores records and returns them when asked for a value " * 6


def _subject_word(n: int) -> str:
    """A distinct, purely ALPHABETIC pseudo-word.

    Alphabetic is the requirement, not decoration. `queryset._is_prose` refuses any token holding a
    digit or an underscore, because rare-token ranking otherwise surfaces identifiers like
    `test_parity_reports_a_missing_shadow` as topics and the resulting question measures string
    matching rather than meaning. A first attempt at this fixture used `topic0word1` bodies under
    `# Heading 0` headings, so every heading reduced to the single word "heading" and the generator
    correctly reported one distinct subject across the whole corpus.
    """
    first = _SYLLABLES[n % len(_SYLLABLES)]
    second = _SYLLABLES[(n // len(_SYLLABLES)) % len(_SYLLABLES)]
    return f"{first}{second}ium"


def _corpus(tmp_path: Path, *, files: int) -> Path:
    """A markdown corpus whose documents have genuinely distinct subjects.

    One chunk per document (bodies stay under `max_chars`), each with its own heading, so `files`
    is also the number of distinct subjects available to the generator.
    """
    root = tmp_path / "corpus"
    root.mkdir(exist_ok=True)
    for n in range(files):
        word = _subject_word(n)
        body = f"{word} " * 10 + f"{word}craft " * 5 + _FILLER
        (root / f"doc{n}.md").write_text(
            f"# {word.capitalize()} handbook\n\n{body}\n", encoding="utf-8"
        )
    return root


# ----------------------------------------------------------------------------------------------
# Order
# ----------------------------------------------------------------------------------------------


def test_the_steps_run_in_the_only_order_that_produces_a_bound_calibration(
    tmp_path: Path,
) -> None:
    """build, validate, calibrate, publish, promote — and promote LAST.

    Promotion gives the generation a fresh corpus fingerprint, so a calibration measured after it
    binds to a fingerprint that no longer matches and the tenant answers `CALIBRATION_STALE`. The
    sequence is asserted rather than the end state, because an inverted run reaches the same end
    state and is broken.
    """
    from recall.wizard.pipeline import run_corpus

    recorder = _Recorder()
    spec = docs_corpus(_corpus(tmp_path, files=45))

    run_corpus(
        spec,
        manager=_FakeManager(recorder),
        calibrations=_FakeCalibrations(recorder),
        embedder=_FakeEmbedder(),
        corpus_version="2026-01-01",
    )

    assert recorder.calls == ["create", "build", "validate", "calibrate", "publish", "promote"]


def test_promote_never_runs_before_publish(tmp_path: Path) -> None:
    """The specific inversion that produces CALIBRATION_STALE, pinned on its own.

    Kept separate from the full-sequence assertion above: a single equality over the whole list
    fails for any reordering, which makes it a poor witness for WHICH ordering broke. This one names
    the pair whose order is the actual hazard.
    """
    from recall.wizard.pipeline import run_corpus

    recorder = _Recorder()
    spec = docs_corpus(_corpus(tmp_path, files=45))

    run_corpus(
        spec,
        manager=_FakeManager(recorder),
        calibrations=_FakeCalibrations(recorder),
        embedder=_FakeEmbedder(),
        corpus_version="2026-01-01",
    )

    assert recorder.calls.index("publish") < recorder.calls.index("promote")
    assert recorder.calls.index("calibrate") < recorder.calls.index("promote")


# ----------------------------------------------------------------------------------------------
# The floor, checked before the expensive step
# ----------------------------------------------------------------------------------------------


def test_a_corpus_too_small_to_certify_is_refused_before_anything_is_built(
    tmp_path: Path,
) -> None:
    """The one failure that cannot be recovered from later, so it is caught first.

    Certification needs at least 20 answerable and 20 unanswerable. A corpus that cannot supply them
    still builds, validates and promotes without complaint, and then refuses every query forever with
    nothing in the output naming the size as the cause. Generating the query set costs no database
    and reads the corpus off disk, so it can run BEFORE the build rather than after `validate` as the
    design sequences it — which is what turns a seven-minute doomed build into an immediate refusal.

    `manager.built` is the witness: the assertion is that nothing was built, not merely that an error
    was raised.
    """
    from recall.wizard.pipeline import PipelineRefusal, run_corpus

    recorder = _Recorder()
    manager = _FakeManager(recorder)
    spec = docs_corpus(_corpus(tmp_path, files=3))

    with pytest.raises(PipelineRefusal, match="too small to certify"):
        run_corpus(
            spec,
            manager=manager,
            calibrations=_FakeCalibrations(recorder),
            embedder=_FakeEmbedder(),
            corpus_version="2026-01-01",
        )

    assert manager.built is False, "the floor must be checked before the build, not after"
    assert recorder.calls == [], "no generation call should have been made at all"


def test_a_corpus_that_cannot_even_generate_a_set_says_how_far_short_it_is(
    tmp_path: Path,
) -> None:
    """The FIRST of two refusal paths, and the one a tiny corpus actually takes.

    Below a certain size the generator cannot produce `per_class` distinct subjects at all, so it
    refuses before any balance can be counted. Its message already names the shortfall and the
    floor, which is what the user acts on, so the pipeline wraps rather than replaces it.
    """
    from recall.wizard.pipeline import PipelineRefusal, run_corpus

    with pytest.raises(PipelineRefusal) as refusal:
        run_corpus(
            docs_corpus(_corpus(tmp_path, files=3)),
            manager=_FakeManager(_Recorder()),
            calibrations=_FakeCalibrations(_Recorder()),
            embedder=_FakeEmbedder(),
            corpus_version="2026-01-01",
        )

    message = str(refusal.value)
    assert "too small to certify" in message
    assert "20" in message, "the floor itself must appear, so the gap is legible"
    assert str(tmp_path) in message, "the corpus that is short must be named"


def test_an_unbalanced_set_names_BOTH_counts_and_the_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The SECOND path: a set generates, then one class falls short after canonicalisation.

    Which class is thin decides what the user changes — a thin answerable side means the corpus is
    too small, a thin unanswerable side means the off-topic pool ran out — so both counts are named.
    "Calibration failed" tells them neither.

    Reached by replacing the generator rather than by hunting for a corpus size that lands in this
    window, because that size depends on the generator's internals and a test tuned to it would
    break for reasons unrelated to what it checks. The real `canonicalize` and the real balance
    check still run on the result.
    """
    import recall.wizard.pipeline as module
    from recall.wizard.pipeline import PipelineRefusal, run_corpus

    def lopsided(chunks: Any, *, per_class: int, seed: int) -> list[dict[str, Any]]:
        return [
            {"query": f"what is subject {i}", "answerable": True, "relevant_ids": []}
            for i in range(30)
        ] + [
            {"query": f"what is offtopic {i}", "answerable": False, "relevant_ids": []}
            for i in range(4)
        ]

    monkeypatch.setattr(module, "generate_offline", lopsided)

    with pytest.raises(PipelineRefusal) as refusal:
        run_corpus(
            docs_corpus(_corpus(tmp_path, files=45)),
            manager=_FakeManager(_Recorder()),
            calibrations=_FakeCalibrations(_Recorder()),
            embedder=_FakeEmbedder(),
            corpus_version="2026-01-01",
        )

    message = str(refusal.value)
    assert "30 answerable" in message
    assert "4 unanswerable" in message
    assert "20" in message


# ----------------------------------------------------------------------------------------------
# Degradation rather than abort
# ----------------------------------------------------------------------------------------------


def test_a_corpus_that_fails_certification_is_reported_degraded_and_still_promoted(
    tmp_path: Path,
) -> None:
    """Certification failing is a measurement, not a crash.

    The artifact is kept as evidence, the corpus is marked degraded with the measured separability,
    and the install continues. A wizard that aborts on the third of three corpora leaves the user
    with less than one that says which corpus is degraded and why.
    """
    from recall.wizard.pipeline import run_corpus

    recorder = _Recorder()
    spec = docs_corpus(_corpus(tmp_path, files=45))

    outcome = run_corpus(
        spec,
        manager=_FakeManager(recorder),
        calibrations=_FakeCalibrations(recorder, certified=False),
        embedder=_FakeEmbedder(),
        corpus_version="2026-01-01",
    )

    assert outcome.certified is False
    assert outcome.degraded_reason
    assert "separability" in outcome.degraded_reason
    assert outcome.calibration_id == "cal_test", "the rejected artifact is kept as evidence"
    assert "promote" in recorder.calls


def test_a_certified_corpus_reports_no_degradation(tmp_path: Path) -> None:
    """The allow path, so `degraded_reason` cannot be satisfied by always being set."""
    from recall.wizard.pipeline import run_corpus

    recorder = _Recorder()
    spec = docs_corpus(_corpus(tmp_path, files=45))

    outcome = run_corpus(
        spec,
        manager=_FakeManager(recorder),
        calibrations=_FakeCalibrations(recorder, certified=True),
        embedder=_FakeEmbedder(),
        corpus_version="2026-01-01",
    )

    assert outcome.certified is True
    assert outcome.degraded_reason is None


# ----------------------------------------------------------------------------------------------
# What the pipeline must refuse to be handed
# ----------------------------------------------------------------------------------------------


def test_the_uncalibrated_tenant_is_not_driven_down_the_generation_path(tmp_path: Path) -> None:
    """`memory` is indexed into the legacy table and has no generation to build.

    Driving it here would produce a promoted generation nothing can calibrate, which is the state
    that answers INDEX_NOT_READY forever.
    """
    from recall.wizard.pipeline import PipelineRefusal, run_corpus

    recorder = _Recorder()
    spec = memory_corpus(_corpus(tmp_path, files=45))

    with pytest.raises(PipelineRefusal, match="not calibrated"):
        run_corpus(
            spec,
            manager=_FakeManager(recorder),
            calibrations=_FakeCalibrations(recorder),
            embedder=_FakeEmbedder(),
            corpus_version="2026-01-01",
        )

    assert recorder.calls == []


def test_the_build_runs_under_development_whatever_the_serving_environment(
    tmp_path: Path,
) -> None:
    """A `file://` manifest is refused in production, and `docs` is SERVED in production.

    So the manager the build runs against must be a development one. This is the constraint that
    makes the two environments coexist on one machine, and getting it wrong fails at build time with
    "production generation builds require a versioned S3 manifest", which names neither the wizard
    nor the corpus.
    """
    from recall.wizard.pipeline import PipelineRefusal, run_corpus

    recorder = _Recorder()
    spec = docs_corpus(_corpus(tmp_path, files=45))

    with pytest.raises(PipelineRefusal, match="development"):
        run_corpus(
            spec,
            manager=_FakeManager(recorder, environment="production"),
            calibrations=_FakeCalibrations(recorder),
            embedder=_FakeEmbedder(),
            corpus_version="2026-01-01",
        )

    assert recorder.calls == []
