"""The per-corpus pipeline: what runs, in what order, and what it refuses before spending time.

Three properties are worth more than the rest and are what this file mostly pins.

**Order is load-bearing and is not a comment.** Correct order is build, validate, calibrate,
publish, promote, and promotion is last because it is irreversible and RETIRES whatever the tenant
was serving. A test that only checks the end state cannot see an inversion, so these assert the
SEQUENCE.

🔁 This docstring used to justify the order by claiming promotion gives a generation a fresh corpus
fingerprint, making an earlier calibration `CALIBRATION_STALE`. That is false: `corpus_fingerprint`
is written only by `create` and by `forget`. The order is right, the reason was not.

**The certification floor is checked before the build, not after.** `chunks_from_directory` reads
the corpus off disk rather than out of the database, so the query set can be generated and counted
first, and a corpus that cannot certify is refused in seconds rather than after a build measured in
minutes (roughly seven for 1793 chunks, per the 2026-08-16 pre-registration).

**A corpus that fails certification is NOT promoted.** It is built, validated, reported degraded and
left unpromoted, so whatever the tenant was serving keeps serving. Promoting anyway was this
module's original behaviour and it is destructive: a calibrated corpus is served strictly, and a
strict tenant with an uncertified calibration refuses every query, having just retired the
generation that worked.

No database here. Every collaborator is injected, so the ordering and refusal logic is testable
without a container. ⚠️ There is NO DB-backed end-to-end yet — an earlier version of this line
claimed one lived "at the bottom" of this file and none existed, so nothing has ever exercised
`run_corpus` against a real `GenerationManager` or `CalibrationRepository`. That gap is exactly how
the fake below came to model a `publish` contract the real class does not have.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from recall.wizard.corpora import docs_corpus, memory_corpus
from recall.wizard.queryset import MIN_PER_CLASS


class _Recorder:
    """Records the calls a pipeline makes, in order, so a sequence can be asserted."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def note(self, name: str) -> None:
        self.calls.append(name)


class _FakeManager:
    """Stands in for `GenerationManager`, recording the generation lifecycle calls."""

    def __init__(
        self,
        recorder: _Recorder,
        *,
        environment: str = "development",
        tenant_id: str = "docs",
    ) -> None:
        self._recorder = recorder
        self.environment = environment
        self.tenant_id = tenant_id
        self.built = False
        self.failed: list[str] = []

    def fail(self, generation_id: str, reason: str) -> None:
        self._recorder.note("fail")
        self.failed.append(reason)

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
        """RAISES for an uncertified artifact, exactly as `CalibrationRepository.publish` does.

        The first version of this fake RETURNED an uncertified namespace, a value the real class
        can never produce because it raises first. That single divergence is what made the module's
        headline behaviour reachable in tests and unreachable in production, and four auditors found
        it independently. The fake was the thing that was wrong.
        """
        self._recorder.note("publish")
        if not self._certified:
            from recall.calibration_v2 import CalibrationUncertified

            raise CalibrationUncertified("separability below the bar")
        from types import SimpleNamespace

        return SimpleNamespace(calibration_id=calibration_id, certified=True)


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

    with pytest.raises(PipelineRefusal, match="cannot produce a certifiable query set"):
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
    # A NEUTRAL prefix. "too small to certify" was applied to every generator refusal, including
    # the one whose own text says the corpus OVERLAPS the off-topic pool — whose fix is a disjoint
    # subject list, not more content. The inner message carries the actual cause.
    assert "cannot produce a certifiable query set" in message
    assert "distinct chunks" in message, "pin the branch this docstring names, not any refusal"
    # NOT `"20" in message`. The message interpolates `str(tmp_path)`, and pytest's basetemp
    # carries an incrementing counter (pytest-11485), so a run numbered ...20xx satisfies a bare
    # "20" with the floor deleted from the text. Asserted against the path-stripped remainder.
    assert f"floor is {MIN_PER_CLASS}" in message.replace(str(tmp_path), "")
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
    assert f"at least {MIN_PER_CLASS} of each" in message.replace(str(tmp_path), "")


# ----------------------------------------------------------------------------------------------
# Degradation rather than abort
# ----------------------------------------------------------------------------------------------


def test_a_corpus_that_fails_certification_is_NOT_promoted(tmp_path: Path) -> None:
    """The correction the audit forced, and the reverse of what this test used to assert.

    Promoting anyway looked like the friendly choice. It is destructive: a calibrated `CorpusSpec`
    is forced to `TrustMode.STRICT`, a strict tenant whose calibration is uncertified refuses every
    query before retrieval, and `promote` has by then RETIRED the generation that was working. So a
    re-run over a healthy tenant would replace something that answers with something that answers
    nothing, while the outcome text promised a development-trust fallback that `CorpusSpec` forbids.

    Degrade therefore means leave it unpromoted and name it, so whatever was serving keeps serving.
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
    assert outcome.promoted is False
    assert "promote" not in recorder.calls, "an uncertified generation must not go live"
    assert outcome.calibration_id == "cal_test", "the rejected artifact is kept as evidence"
    assert outcome.degraded_reason and "NOT promoted" in outcome.degraded_reason
    assert outcome.generation_id in outcome.degraded_reason, "name it so it can be promoted by hand"


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


# ----------------------------------------------------------------------------------------------
# The refusal contract, and the fakes that stand in for real collaborators
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "which",
    ["missing-root", "empty-directory", "glob-matches-nothing"],
)
def test_the_commonest_first_run_mistakes_are_refusals_not_leaks(
    tmp_path: Path, which: str
) -> None:
    """A missing root, an empty directory and a wrong glob all reach a caller as `PipelineRefusal`.

    They used to escape as a bare `QuerySetError`, because the corpus read sat one line ABOVE the
    `try` that converts it. These are the three commonest first-run mistakes, and a caller doing
    `except PipelineRefusal` to learn "nothing was built" caught none of them.
    """
    from recall.wizard.pipeline import PipelineRefusal, run_corpus

    if which == "missing-root":
        root = tmp_path / "absent"
    elif which == "empty-directory":
        root = tmp_path / "empty"
        root.mkdir()
    else:
        root = tmp_path / "wrongext"
        root.mkdir()
        (root / "notes.txt").write_text("not markdown", encoding="utf-8")

    recorder = _Recorder()
    with pytest.raises(PipelineRefusal, match="cannot produce a certifiable query set"):
        run_corpus(
            docs_corpus(root),
            manager=_FakeManager(recorder),
            calibrations=_FakeCalibrations(recorder),
            embedder=_FakeEmbedder(),
            corpus_version="2026-01-01",
        )
    assert recorder.calls == [], "a pre-build refusal must not have touched the generation store"


def test_a_tenant_mismatch_is_refused_before_the_build(tmp_path: Path) -> None:
    """A mis-paired manager must not cost the walk, the query generation and the build.

    `create` refuses a manifest whose tenant differs, but only after all of that has been spent.
    """
    from recall.wizard.pipeline import PipelineRefusal, run_corpus

    recorder = _Recorder()
    with pytest.raises(PipelineRefusal, match="bound to tenant"):
        run_corpus(
            docs_corpus(_corpus(tmp_path, files=45)),
            manager=_FakeManager(recorder, tenant_id="somewhere-else"),
            calibrations=_FakeCalibrations(recorder),
            embedder=_FakeEmbedder(),
            corpus_version="2026-01-01",
        )
    assert recorder.calls == []


def test_a_blank_corpus_version_is_refused_before_the_build(tmp_path: Path) -> None:
    """It reached `IndexManifestV1` as a raw `LineageError`, after the corpus had been chunked."""
    from recall.wizard.pipeline import PipelineRefusal, run_corpus

    recorder = _Recorder()
    with pytest.raises(PipelineRefusal, match="corpus_version must be non-empty"):
        run_corpus(
            docs_corpus(_corpus(tmp_path, files=45)),
            manager=_FakeManager(recorder),
            calibrations=_FakeCalibrations(recorder),
            embedder=_FakeEmbedder(),
            corpus_version="   ",
        )
    assert recorder.calls == []


def test_the_fakes_match_the_signatures_of_the_classes_they_stand_for() -> None:
    """The check that would have caught this module's worst defect.

    `_FakeCalibrations.publish` modelled a contract `CalibrationRepository.publish` does not have,
    returning an uncertified artifact where the real class raises, and that single divergence made
    the module's headline behaviour pass in tests while being unreachable in production. Nothing
    compared the two, so nothing noticed.

    A signature is not the whole contract (the raise is not in it), but a rename or a reordered
    parameter is the commonest way a fake stops standing for anything, and this is the cheapest
    guard against that.
    """
    import inspect

    from recall.calibration_v2 import CalibrationRepository
    from recall.generations import GenerationManager

    for fake, real, methods in (
        (_FakeManager, GenerationManager, ("create", "build", "validate", "promote", "fail")),
        (_FakeCalibrations, CalibrationRepository, ("calibrate", "publish")),
    ):
        for name in methods:
            fake_params = list(inspect.signature(getattr(fake, name)).parameters)
            real_params = list(inspect.signature(getattr(real, name)).parameters)
            missing = [p for p in real_params if p not in fake_params and p != "generation_id"]
            assert not missing, f"{fake.__name__}.{name} is missing {missing} from {real.__name__}"
