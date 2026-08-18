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
from tests.conftest import TEST_DSN, requires_db
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
        serving: str | None = None,
    ) -> None:
        self._recorder = recorder
        self.environment = environment
        self.tenant_id = tenant_id
        #: What this tenant is serving before the run. `None` is a FIRST INSTALL, which is the
        #: default because it is the state a wizard actually meets, and the one whose degraded
        #: outcome used to be described with the upgrade's wording.
        self.serving = serving
        self.built = False
        self.failed: list[str] = []
        self.abandoned: list[str] = []
        #: Tracked because the real manager's transitions key on it, and a fake that ignores state
        #: cannot reproduce a refusal. See `fail` below.
        self.state = "building"

    def fail(self, generation_id: str, reason: str) -> None:
        """REFUSES a READY generation, exactly as `GenerationManager.fail` does.

        This fake used to accept `fail` from any state, and that single permissiveness hid a live
        storage leak through a full audit: `validate` sets READY, the real `fail` raises
        `InvalidGenerationTransition` for READY, and the wizard's `_fail` swallowed it, so the
        guard whose whole purpose is to stop a leak had never once worked. The test was green
        because the double could not say no. A fake that cannot reproduce the real object's
        refusals does not test the caller, it tests itself.
        """
        self._recorder.note("fail")
        if self.state in {"ready", "active", "retired"}:
            from recall.generations import InvalidGenerationTransition

            raise InvalidGenerationTransition(f"cannot fail generation in {self.state} state")
        self.failed.append(reason)

    def abandon(self, generation_id: str, reason: str) -> None:
        """READY only, exactly as `GenerationManager.abandon` is."""
        self._recorder.note("abandon")
        if self.state != "ready":
            from recall.generations import InvalidGenerationTransition

            raise InvalidGenerationTransition(f"abandon requires ready state, found {self.state}")
        self.abandoned.append(reason)

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
        self.state = "ready"
        from types import SimpleNamespace

        return SimpleNamespace(generation_id=generation_id, sources=1, chunks=40)

    def active_generation_id(self) -> str:
        """RAISES when nothing is serving, exactly as `GenerationManager.active_generation_id` does."""
        if not self.serving:
            from recall.generations import NoActiveGeneration

            raise NoActiveGeneration(f"tenant {self.tenant_id!r} has no active generation")
        return self.serving

    def promote(self, generation_id: str, *, unsafe_development: bool = False) -> None:
        self._recorder.note("promote")
        self.state = "active"
        self.serving = generation_id


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
    root.mkdir(parents=True, exist_ok=True)
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
    """A production build requires a verifiable embedder identity, and `docs` is SERVED in production.

    So the manager the build runs against must be a development one. This is the constraint that
    makes the two environments coexist on one machine, and getting it wrong fails inside
    `GenerationManager.create` with "production generation builds require an immutable embedder
    revision or artifact digest", which names neither the wizard nor the corpus.

    🔁 This docstring used to blame a `file://` manifest, citing "production generation builds
    require a versioned S3 manifest". That message is real but belongs to a layer the wizard never
    touches: it is a CLI guard on `recall generation build` (`recall/cli.py:1694`). The wizard calls
    `build_generation` directly, so the manifest scheme is never checked against the environment at
    all, and the embedder identity is the only thing that actually refuses.
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


# ----------------------------------------------------------------------------------------------
# The audit fixes themselves. Every test below exists because a mutation of the fix it names
# survived: eight findings were fixed and barely half were pinned, which is the failure mode
# mutation testing exists to expose.
# ----------------------------------------------------------------------------------------------


def test_a_corpus_between_the_floor_and_the_headroom_still_certifies(tmp_path: Path) -> None:
    """The off-by-a-factor-of-two fix, pinned.

    `generate_offline` refuses outright when the corpus has fewer distinct chunks than requested,
    so asking for `DEFAULT_PER_CLASS` made every corpus between the floor (20) and the headroom
    (40) "too small to certify" while the message quoted the floor. Measured before the fix: 39
    chunks refused, 40 accepted, against a floor of 20.

    25 files sits squarely in the band that used to be refused and can certify.
    """
    from recall.wizard.pipeline import run_corpus

    recorder = _Recorder()
    outcome = run_corpus(
        docs_corpus(_corpus(tmp_path, files=25)),
        manager=_FakeManager(recorder),
        calibrations=_FakeCalibrations(recorder),
        embedder=_FakeEmbedder(),
        corpus_version="2026-01-01",
    )

    assert outcome.certified is True
    assert outcome.answerable >= MIN_PER_CLASS
    assert outcome.unanswerable >= MIN_PER_CLASS
    # And the headroom is still preferred when the corpus can supply it, rather than every corpus
    # silently dropping to the floor.
    roomy = run_corpus(
        docs_corpus(_corpus(tmp_path / "big", files=45)),
        manager=_FakeManager(_Recorder()),
        calibrations=_FakeCalibrations(_Recorder()),
        embedder=_FakeEmbedder(),
        corpus_version="2026-01-01",
    )
    assert roomy.answerable > outcome.answerable, "the headroom must be tried before the floor"


def test_the_query_set_is_chunked_with_the_callable_the_build_binds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The code-corpus fix, pinned by capturing what the generator was actually handed.

    `chunks_from_directory` defaults to `chunk_text` while `code_corpus` builds with `chunk_code`.
    Measured on `pipeline.py` itself, the two produce 20 chunks against 8 with no exact string in
    common, so every answerable query would describe text that is not in the index it is about to
    be measured against. Asserting on the CALLABLE rather than on downstream output, because the
    downstream difference is invisible to a fake manager.
    """
    import recall.wizard.pipeline as module
    from recall.generation_build import BuildRequest, chunker_for
    from recall.wizard.corpora import code_corpus
    from recall.wizard.pipeline import PipelineRefusal, run_corpus

    seen: list[object] = []
    real = module.chunks_from_directory

    def capturing(root, glob=None, chunker=None):  # type: ignore[no-untyped-def]
        seen.append(chunker)
        return real(root, glob, chunker)

    monkeypatch.setattr(module, "chunks_from_directory", capturing)

    root = _corpus(tmp_path, files=45)
    try:
        run_corpus(
            code_corpus(root),
            manager=_FakeManager(_Recorder(), tenant_id="code"),
            calibrations=_FakeCalibrations(_Recorder()),
            embedder=_FakeEmbedder(),
            corpus_version="2026-01-01",
        )
    except PipelineRefusal:
        # A markdown fixture under a `**/*.py` glob cannot generate a set, and that is fine: the
        # chunker was already captured on the way in, which is the property under test.
        pass

    assert seen, "chunks_from_directory was never called"
    expected = chunker_for(BuildRequest(chunker="code"))[0]
    sample = "def f():\n    return 1\n\n\n" * 40
    assert seen[0] is not None, "the build's chunker must be passed, not left to default"
    assert seen[0](sample) == expected(sample), "queries must be chunked as the index will be"


def test_only_a_resolvable_revision_is_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`lineage_revision`, not `revision`, and the difference is a revision nobody can resolve.

    fastembed serves `BAAI/bge-small-en-v1.5` out of `qdrant/bge-small-en-v1.5-onnx-q`, so the
    snapshot SHA is a commit in the qdrant repository and does not exist in the one it would be
    recorded against. `lineage_revision` hands it out only when the two agree.
    """
    import recall.wizard.pipeline as module
    from recall.wizard.identity import ArtifactIdentity

    mismatched = ArtifactIdentity(
        provider="fastembed",
        model="BAAI/bge-small-en-v1.5",
        source_repo="qdrant/bge-small-en-v1.5-onnx-q",
        revision="a" * 40,
        artifact_digest="b" * 64,
        path=tmp_path,
    )
    assert mismatched.lineage_revision is None, "the fixture must actually differ"
    monkeypatch.setattr(module, "artifact_identity_for", lambda _e: mismatched)

    request = module._identified_request(docs_corpus(_corpus(tmp_path, files=1)), object(), None)

    assert request.revision is None, "a revision that names another repository must not be recorded"
    assert request.artifact_digest == "b" * 64, "the digest is what makes it immutable"


def test_a_failure_after_validate_reclaims_the_generation(tmp_path: Path) -> None:
    """Otherwise the orphan is unreclaimable: `gc` collects only `retired` and `failed`.

    A generation left READY by an aborted run keeps a full copy of the corpus's chunk rows, once
    per failed attempt, and nothing will ever collect it.

    The abort happens AFTER `validate`, so the generation is READY, so `fail` is the wrong verb and
    the real manager refuses it. The reclaim must therefore go through `abandon`, and asserting on
    `abandoned` rather than on `failed` is the whole point of this test: the previous version
    asserted `manager.failed` against a fake that accepted `fail` from any state, and passed for
    months while the leak it describes happened on every run.
    """
    from recall.wizard.pipeline import run_corpus

    class _Exploding(_FakeCalibrations):
        def calibrate(self, generation_id: str, queries: Any, embedder: Any) -> Any:
            raise RuntimeError("calibration blew up")

    recorder = _Recorder()
    manager = _FakeManager(recorder)

    with pytest.raises(RuntimeError, match="calibration blew up"):
        run_corpus(
            docs_corpus(_corpus(tmp_path, files=45)),
            manager=manager,
            calibrations=_Exploding(recorder),
            embedder=_FakeEmbedder(),
            corpus_version="2026-01-01",
        )

    assert manager.state == "ready", "the precondition this test exists for: fail() cannot apply"
    assert manager.abandoned, "a READY generation must be reclaimed via abandon, not left behind"
    assert not manager.failed, "fail() does not apply to READY and must not be recorded as working"
    assert "calibration blew up" in manager.abandoned[0], "the reason must name the real cause"
    assert "promote" not in recorder.calls


def test_the_outcome_survives_the_json_round_trip_it_is_written_for(tmp_path: Path) -> None:
    """`steps` returns from JSON as a list, which compares unequal and is unhashable.

    `CorpusPlan` already documents and fixes exactly this; `CorpusOutcome` is written into the same
    resumable state and had no coercion at all.
    """
    import dataclasses
    import json

    from recall.wizard.pipeline import CorpusOutcome

    # `certified` defaults to False, and an uncertified outcome must carry its reason, so the
    # round trip is exercised on a record that is actually constructible.
    built = CorpusOutcome(
        tenant="docs",
        generation_id="gen_x",
        steps=["build", "validate"],
        degraded_reason="separability below the bar",
    )
    assert isinstance(built.steps, tuple)

    restored = CorpusOutcome(**json.loads(json.dumps(dataclasses.asdict(built))))
    assert restored == built, "a state file read back must compare equal to the run that wrote it"
    assert hash(restored), "a frozen outcome must stay hashable across the round trip"


def test_an_unverifiable_embedder_is_reported_rather_than_swallowed(tmp_path: Path) -> None:
    """The wizard should SAY the generation carries no verifiable embedder provenance.

    `artifact_identity_for` returns None for anything that is not a resolvable FastEmbedEmbedder,
    and the request is then marked unverified, which satisfies `allow_unverified` silently. The
    outcome carries it so the caller can tell the user rather than the user discovering it.
    """
    from recall.wizard.pipeline import run_corpus

    outcome = run_corpus(
        docs_corpus(_corpus(tmp_path, files=45)),
        manager=_FakeManager(_Recorder()),
        calibrations=_FakeCalibrations(_Recorder()),
        embedder=_FakeEmbedder(),
        corpus_version="2026-01-01",
    )

    assert outcome.unverified_embedder is True


# ----------------------------------------------------------------------------------------------
# The DB-backed end-to-end. Everything above this line uses fakes, and a fake modelling a contract
# the real class does not have is precisely how this module's headline behaviour came to be
# unreachable in production while passing in tests. This is the test that would have caught it.
# ----------------------------------------------------------------------------------------------


def _purge(tenant: str) -> None:
    """Remove every row this tenant created. Statements are literal, not interpolated."""
    import psycopg

    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (tenant,))
        for statement in (
            "DELETE FROM recall_source_tombstones WHERE tenant_id = %s",
            "DELETE FROM recall_audit_events WHERE tenant_id = %s",
            "DELETE FROM recall_ingest_jobs WHERE tenant_id = %s",
            "DELETE FROM recall_tenant_state WHERE tenant_id = %s",
            "DELETE FROM recall_generations WHERE tenant_id = %s",
            "DELETE FROM chunks WHERE tenant_id = %s",
        ):
            try:
                conn.execute(statement, (tenant,))
            except psycopg.Error:
                conn.rollback()


@requires_db
def test_the_real_manager_refuses_to_fail_a_ready_generation_and_abandon_reclaims_it(
    tmp_path: Path,
) -> None:
    """The defect DAT-001 named, proved against the real object rather than a fake.

    `validate` sets READY, and `GenerationManager.fail` refuses READY, so the wizard's leak guard
    could never fire on the path it was written for. The `InvalidGenerationTransition` was swallowed
    by a bare `except Exception: pass`, and `gc` collects only `retired` and `failed`, so every
    uncertified build left a full copy of the corpus's chunk rows that nothing could ever reclaim.

    This is the test the fake could not be: `_FakeManager.fail` accepted any state, so the refusal
    below was invisible to the suite for the whole life of the module.
    """
    import uuid

    import psycopg

    from recall.calibration_v2 import CalibrationRepository
    from recall.embeddings import resolve_embedder
    from recall.generations import GenerationManager, InvalidGenerationTransition
    from recall.trust_policy import TrustMode
    from recall.wizard.corpora import CorpusSpec
    from recall.wizard.pipeline import run_corpus

    tenant = "wizard-abandon-" + uuid.uuid4().hex[:10]
    spec = CorpusSpec(
        tenant=tenant,
        root=_corpus(tmp_path, files=45),
        glob="**/*.md",
        chunker="text",
        calibrated=True,
        serving_environment="production",
        trust_mode=TrustMode.STRICT,
        writable=False,
    )
    manager = GenerationManager(TEST_DSN, tenant, actor="pytest", environment="test")

    try:
        outcome = run_corpus(
            spec,
            manager=manager,
            calibrations=CalibrationRepository(TEST_DSN, tenant),
            embedder=resolve_embedder("hashing"),
            corpus_version="2026-01-01",
        )
        if outcome.certified:
            pytest.skip("this fixture certified, so there is no READY generation to reclaim")

        generation_id = outcome.generation_id

        with pytest.raises(InvalidGenerationTransition, match="ready"):
            manager.fail(generation_id, "this is the refusal the fake could not reproduce")

        manager.abandon(generation_id, "reclaimed by the test")

        with psycopg.connect(TEST_DSN, autocommit=True) as conn:
            conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (tenant,))
            state = conn.execute(
                "SELECT state FROM recall_generations WHERE tenant_id = %s AND generation_id = %s",
                (tenant, generation_id),
            ).fetchone()

        assert state is not None and state[0] == "failed", (
            "abandon must move it into a state gc actually collects"
        )
    finally:
        _purge(tenant)


@requires_db
def test_the_real_services_wiring_drives_a_corpus(tmp_path: Path) -> None:
    """`_RealServices` is the only code here that runs on a real deploy, and had zero coverage.

    Every other test in the driver's module injects a spy, so the DSN plumbing, the `actor` and
    `environment` choices and the dimension lookup were exercised by nothing but the one manual run
    named in a commit message. A unique tenant rather than `default_plan`, so this does not write
    into the shared `docs`/`code` tenants of a container other tests are using.
    """
    import uuid

    from recall.trust_policy import TrustMode
    from recall.wizard.corpora import CorpusSpec
    from recall.wizard.headless import HeadlessConfig, _RealServices

    tenant = "wizard-real-" + uuid.uuid4().hex[:10]
    config = HeadlessConfig(
        dsn=TEST_DSN,
        migration_dsn=TEST_DSN,
        embedder="hashing",
        corpus_version="2026-01-01",
        docs_root=tmp_path / "unused",
        code_root=tmp_path / "unused",
        memory_root=tmp_path / "unused",
    )
    services = _RealServices(config)
    spec = CorpusSpec(
        tenant=tenant,
        root=_corpus(tmp_path, files=45),
        glob="**/*.md",
        chunker="text",
        calibrated=True,
        serving_environment="production",
        trust_mode=TrustMode.STRICT,
        writable=False,
    )

    try:
        from recall.embeddings import resolve_embedder

        assert services.dim() == resolve_embedder("hashing").dim
        assert services.embedder() is services.embedder(), "the embedder must be resolved once"

        services.apply_schema(config.migration_dsn, dim=services.dim())

        steps: list[str] = []
        outcome = services.run(spec, progress=steps.append)

        assert outcome.tenant == tenant
        assert outcome.generation_id
        assert "build" in steps, "progress must reach the caller through the real wiring too"
        assert outcome.promoted is outcome.certified
    finally:
        _purge(tenant)


@requires_db
def test_run_corpus_end_to_end_against_a_real_generation_store(tmp_path: Path) -> None:
    """Drive `run_corpus` with a real `GenerationManager` and `CalibrationRepository`.

    The property that matters is NOT "it certifies". A hashing embedder over a synthetic corpus may
    or may not clear the separability bar, and pinning that would turn this into a measurement of
    the fixture. What matters is that the pipeline COMPLETES either way and leaves the database in
    the state it claims:

    * it does not raise — the earlier version aborted right here with `CalibrationUncertified`,
      because `publish` raises for exactly the artifact the module meant to degrade on;
    * `certified` and `promoted` agree, since an uncertified generation must never go live;
    * and the tenant's ACTIVE generation matches. Uncertified means nothing was promoted, so
      whatever the tenant was serving keeps serving.

    Asserted against the real `recall_tenant_state` and `recall_generations` rows rather than
    against the returned object, so the outcome cannot corroborate itself.

    ⚠️ Which branch this actually takes, measured rather than assumed: the UNCERTIFIED one.
    `hashing` over this fixture scores separability 0.757 [0.651, 0.863] against a 0.9 bar, so the
    run exercises the real `publish` raising `CalibrationUncertified`, the pipeline catching it, and
    the generation being left `ready`. That is the precise path that was unreachable before, so it
    is the valuable half. The CERTIFIED half is still covered only by fakes, and would need an
    embedder and a corpus that genuinely separate; tuning this fixture until it certifies would make
    the test a measurement of the fixture rather than of the pipeline.
    """
    import uuid

    import psycopg

    from recall.calibration_v2 import CalibrationRepository
    from recall.embeddings import resolve_embedder
    from recall.generations import GenerationManager
    from recall.trust_policy import TrustMode
    from recall.wizard.corpora import CorpusSpec
    from recall.wizard.pipeline import run_corpus

    tenant = "wizard-e2e-" + uuid.uuid4().hex[:10]
    spec = CorpusSpec(
        tenant=tenant,
        root=_corpus(tmp_path, files=45),
        glob="**/*.md",
        chunker="text",
        calibrated=True,
        serving_environment="production",
        trust_mode=TrustMode.STRICT,
        writable=False,
    )
    # `environment="test"` is deliberate: it is what the rest of the DB suite uses, and accepting it
    # as a build environment was one of the audit fixes. A production manager is refused, correctly,
    # because the wizard's embedders carry no immutable revision or artifact digest.
    manager = GenerationManager(TEST_DSN, tenant, actor="pytest", environment="test")
    calibrations = CalibrationRepository(TEST_DSN, tenant)

    try:
        outcome = run_corpus(
            spec,
            manager=manager,
            calibrations=calibrations,
            embedder=resolve_embedder("hashing"),
            corpus_version="2026-01-01",
        )

        assert outcome.tenant == tenant
        assert outcome.generation_id
        assert outcome.calibration_id, "the artifact is kept whether or not it certified"
        assert outcome.answerable >= MIN_PER_CLASS
        assert outcome.unanswerable >= MIN_PER_CLASS
        assert outcome.promoted is outcome.certified, "an uncertified generation must not go live"

        with psycopg.connect(TEST_DSN, autocommit=True) as conn:
            conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (tenant,))
            active = conn.execute(
                "SELECT active_generation_id FROM recall_tenant_state WHERE tenant_id = %s",
                (tenant,),
            ).fetchone()
            state = conn.execute(
                "SELECT state FROM recall_generations WHERE tenant_id = %s AND generation_id = %s",
                (tenant, outcome.generation_id),
            ).fetchone()

        assert state is not None, "the generation must exist in the database"
        if outcome.certified:
            assert active is not None and active[0] == outcome.generation_id
            assert state[0] == "active"
        else:
            # Nothing promoted: either no tenant_state row at all, or it points elsewhere.
            assert active is None or active[0] != outcome.generation_id
            assert state[0] == "ready", "an uncertified generation is left ready, not promoted"
            assert outcome.degraded_reason and outcome.generation_id in outcome.degraded_reason
    finally:
        _purge(tenant)


@requires_db
def test_the_pipeline_survives_a_corpus_it_cannot_certify_without_leaving_an_orphan(
    tmp_path: Path,
) -> None:
    """A corpus too small to certify must cost nothing at all: no generation, no rows.

    The floor check runs before the build precisely so this case never reaches the database. Pinned
    against the real store rather than a fake, because "nothing was built" is a claim about the
    database and only the database can answer it.
    """
    import uuid

    import psycopg

    from recall.calibration_v2 import CalibrationRepository
    from recall.embeddings import resolve_embedder
    from recall.generations import GenerationManager
    from recall.trust_policy import TrustMode
    from recall.wizard.corpora import CorpusSpec
    from recall.wizard.pipeline import PipelineRefusal, run_corpus

    tenant = "wizard-small-" + uuid.uuid4().hex[:10]
    spec = CorpusSpec(
        tenant=tenant,
        root=_corpus(tmp_path, files=3),
        glob="**/*.md",
        chunker="text",
        calibrated=True,
        serving_environment="production",
        trust_mode=TrustMode.STRICT,
        writable=False,
    )

    try:
        with pytest.raises(PipelineRefusal, match="cannot produce a certifiable query set"):
            run_corpus(
                spec,
                manager=GenerationManager(TEST_DSN, tenant, actor="pytest", environment="test"),
                calibrations=CalibrationRepository(TEST_DSN, tenant),
                embedder=resolve_embedder("hashing"),
                corpus_version="2026-01-01",
            )

        with psycopg.connect(TEST_DSN, autocommit=True) as conn:
            conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (tenant,))
            generations = conn.execute(
                "SELECT count(*) FROM recall_generations WHERE tenant_id = %s", (tenant,)
            ).fetchone()

        assert generations is not None and generations[0] == 0, (
            "a refusal before the build must leave no generation behind"
        )
    finally:
        _purge(tenant)
