"""`recall_ingest` must write to the store its server SERVES FROM.

The defect this file exists to prevent produced no error at all. On a server serving the legacy
`chunks` table — which is every uncalibrated corpus, so every project added after install and the
wizard's own memory corpus — `recall_ingest` built and activated a GENERATION instead. Measured
end to end against a real stack before the fix:

    ingest  -> 'Built and activated generation gen_21a9... with 3 chunk(s) from 3 file(s)'
    stats   -> {'chunks': 0, 'stale': True}
    search  -> 0 hits, abstained: false

and after it, on the same tenant with the same routing:

    ingest  -> 'Indexed 3 chunk(s) from 3 file(s) into memory.'
    stats   -> {'chunks': 3, 'stale': False}
    search  -> 3 hits

Both ends were telling the truth about a different table. That is the shape this project keeps
meeting, and the only defence is asserting that ONE decision drives both.
"""

from __future__ import annotations

import textwrap

import pytest

from types import SimpleNamespace

import recall_mcp.server as server


class _Recorder:
    """Stands in for the two ingest paths, recording which one was chosen."""

    def __init__(self) -> None:  # noqa: D107 - see the class docstring
        self.calls: list[str] = []
        #: The chunker each branch was called with. ⚠️ Recorded because the first version of this
        #: fake accepted no `chunker` at all and the tests asserted only WHICH function was called,
        #: never what it was called with — so `index_memory` losing `category` (and therefore
        #: falling back to `chunk_text` for code) passed every test and survived a mutation run.
        self.chunkers: list[object] = []

    def generation(self, store, embedder, staged_root, category):  # noqa: ANN001, ANN201
        self.calls.append("generation")
        self.chunkers.append(category)
        return "generation-result"

    def legacy(self, store, embedder, staged_root, chunker=None):  # noqa: ANN001, ANN201
        self.calls.append("legacy")
        self.chunkers.append(chunker)
        return "legacy-result"


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    rec = _Recorder()
    monkeypatch.setattr(server, "generation_ingest", rec.generation)
    monkeypatch.setattr(server, "index_memory", rec.legacy)
    return rec


def test_a_generation_server_builds_a_generation(recorder: _Recorder) -> None:
    state = {"embedder": object(), "generation_mode": True}

    result = server.ingest_into_serving_store(state, object(), "/staged", "documents")

    assert recorder.calls == ["generation"]
    assert result == "generation-result"


def test_a_legacy_server_indexes_into_the_legacy_table(recorder: _Recorder) -> None:
    """The case that was broken. A legacy-routed server must NOT build a generation."""
    state = {"embedder": object(), "generation_mode": False}

    result = server.ingest_into_serving_store(state, object(), "/staged", "documents")

    assert recorder.calls == ["legacy"], (
        "a server serving the legacy chunks table built a generation instead, so the upload "
        "succeeded and then could not be found"
    )
    assert result == "legacy-result"


def test_both_branches_chunk_code_as_code(recorder: _Recorder) -> None:
    """`category` must reach BOTH branches, not just the generation one.

    ⚠️ The first version of this fix passed `category` to `generation_ingest` and dropped it on the
    legacy call, so `index_memory` fell back to `chunk_text`. Every project created by `add_project`
    is development-mode by construction, so that was the DEFAULT path: source files uploaded into a
    `-code` corpus were chunked as prose. Nothing errored and nothing was missing — only the chunk
    boundaries were wrong, which is the quietest failure available.

    The tests that shipped with that fix asserted only which function was called. That is why this
    one asserts the argument.
    """
    from recall.index import chunk_code, chunk_text

    server.ingest_into_serving_store(
        {"embedder": object(), "generation_mode": False}, object(), "/staged", "code"
    )
    assert recorder.chunkers == [chunk_code], "a code upload must be chunked as code"

    recorder.calls.clear()
    recorder.chunkers.clear()
    server.ingest_into_serving_store(
        {"embedder": object(), "generation_mode": False}, object(), "/staged", "documents"
    )
    assert recorder.chunkers == [chunk_text], "and prose as prose"


def test_an_absent_flag_is_treated_as_legacy(recorder: _Recorder) -> None:
    """Missing must mean legacy, matching `RECALL_ENV`'s own default of development.

    Defaulting the other way would restore the exact defect for any state built without the key,
    and it would do it silently, which is how the original survived.
    """
    server.ingest_into_serving_store({"embedder": object()}, object(), "/staged", "memory")

    assert recorder.calls == ["legacy"]


def test_the_lifespan_publishes_the_flag_the_tool_reads() -> None:
    """The seam is only honest if the server actually sets what the tool consults.

    Asserted against the source rather than a live lifespan, which needs a database: the failure
    mode is a renamed key on one side, and that is visible here.
    """
    from pathlib import Path

    source = Path(server.__file__).read_text(encoding="utf-8")

    assert '"generation_mode": generation_mode and not enterprise' in source, (
        "the lifespan must publish `generation_mode` into state, or the tool silently falls back "
        "to the legacy path on a production server"
    )
    assert 'state.get("generation_mode")' in source


# ----------------------------------------------------------------------------------------------
# A refused promotion must not lose files or leak corpora
# ----------------------------------------------------------------------------------------------


def test_a_new_build_carries_forward_from_the_newest_servable_generation() -> None:
    """⛔ Seeding from the ACTIVE generation silently drops every un-promoted upload's files.

    A desktop upload whose promotion is refused leaves its generation READY, so
    `active_generation_id` never advances. The next upload then rebuilt its manifest from the OLD
    active generation and contained none of the previous upload's files — two READY generations,
    neither holding the whole corpus, while the first upload's message told the user to certify the
    one about to be superseded. Three auditors found this independently; measured end to end against
    a real database, upload #2 reported 1 file where it should have reported 3.

    Asserted on the SOURCE rather than through a database, so it runs offline: the defect was a
    single wrong method name, and naming the right one is the whole fix.
    """
    import inspect

    import recall_mcp.service as service

    source = inspect.getsource(service.generation_ingest)
    assert "servable_manifest()" in source, (
        "the new build must seed from the newest READY-or-ACTIVE generation"
    )
    assert "active_manifest()" not in source, (
        "seeding from the active manifest is the defect: it drops un-promoted uploads"
    )


def test_the_reclaim_runs_on_both_outcomes_and_is_confined() -> None:
    """⛔ A READY generation holds a full copy of the corpus and `gc` collects only retired/failed.

    Two defects, both found by three auditors independently:

    * The reclaim lived only inside `except UnsafePromotion`, so the leak reopened the moment an
      upload finally certified — every earlier refused build stayed READY forever.
    * It selected on STATE ALONE, so it could abandon a generation another path built and
      deliberately left un-promoted. See
      `test_generations.py::test_the_reclaim_never_touches_a_generation_another_path_built`.

    ⚠️ This assertion has now been rewritten twice. It began as a substring match on
    `inspect.getsource`, which could not tell which BRANCH the call sat in — the exact reason the
    first defect was invisible. It is parsed, and it walks the CALL GRAPH rather than one function,
    because the reclaim is now a named helper and a source-scoped check would have missed that too.
    """
    import ast
    import inspect
    import textwrap

    import recall_mcp.service as service

    helper = ast.parse(textwrap.dedent(inspect.getsource(service._release_superseded)))
    calls = [
        node
        for node in ast.walk(helper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "superseded_ready_generations"
    ]
    assert calls, "the helper must ask which builds it supersedes"
    # ⚠️ The confinement, the abandons and the two call sites are asserted BEHAVIOURALLY at the
    # bottom of this file. The assertions that used to be here checked that a keyword was NAMED
    # `corpus_version_prefix` and that there were two call sites, and two auditors defeated both by
    # executed mutation: `corpus_version_prefix=None` and `if False:` each left this file green.


def test_a_failure_after_validate_does_not_strand_a_generation() -> None:
    """⛔ `except Exception: raise` is a no-op, and a docstring reasoned about it as if it were not.

    After `validate()` the generation is READY, and READY is the one state `gc` cannot reclaim. Any
    failure that is not `UnsafePromotion` — a dropped connection during calibration, a binding
    error, an unexpected transition — stranded a full copy of the corpus forever.
    `recall/wizard/pipeline.py::_fail` does exactly this on the same shape of failure.
    """
    import ast
    import inspect
    import textwrap

    import recall_mcp.service as service

    tree = ast.parse(textwrap.dedent(inspect.getsource(service.generation_ingest)))
    handlers = [node for node in ast.walk(tree) if isinstance(node, ast.ExceptHandler)]
    bare_reraise = [
        handler
        for handler in handlers
        if len(handler.body) == 1 and isinstance(handler.body[0], ast.Raise)
        and handler.body[0].exc is None
    ]
    assert not bare_reraise, (
        "a handler whose entire body is `raise` implements nothing; either give it the cleanup its "
        "docstring implies or remove it"
    )
    # ⚠️ This used to assert that `abandon` appeared literally inside `generation_ingest`, which is
    # the trap a source-shape test always sets: the reclaim moved into `_reclaim_failed` — because
    # `abandon` refuses every state but READY and so missed every failure inside `build()` — and the
    # test went red for a fix that made the behaviour strictly better. The property is now asserted
    # where it lives, in `test_a_failure_before_validate_is_also_reclaimed`. What is kept here is
    # only the claim this test's docstring is actually about: no handler is a bare re-raise.
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_reclaim_failed"
        for node in ast.walk(tree)
    ), "the failure path must release the generation it created"


# ----------------------------------------------------------------------------------------------
# A production tenant must have a reachable route to CERTIFIED
# ----------------------------------------------------------------------------------------------


def test_a_production_upload_calibrates_before_it_promotes() -> None:
    """⛔ The certification gate had no reachable way through on the desktop path.

    `promote` requires a published, certified calibration when the tenant is served under
    production, and nothing in `generation_ingest` produced one. Every upload therefore ended
    READY-and-never-live, with the only route to a live corpus being the CLI. A gate with no door
    is a wall.

    Measured end to end after the fix, against a real 384-dimension database with
    `RECALL_ENV=production`: a ten-file corpus reported "Built and activated generation
    gen_84411e27... with 25 chunk(s) from 10 file(s)" and `active_generation_id()` returned it. A
    one-file corpus reported "no certifiable query set could be generated from 1 chunk(s)" and
    stayed READY — the corpus reason, not the gate's generic refusal.

    Asserted on the source so it runs offline; the live proof is quoted above.
    """
    import inspect

    import recall_mcp.service as service

    source = inspect.getsource(service.generation_ingest)
    assert "if manager.certification_required:" in source, (
        "the calibration must run exactly when the gate will demand it, on the same switch"
    )
    assert "_certify_upload(" in source
    assert source.index("_certify_upload(") < source.index("manager.promote("), (
        "calibrating after promoting certifies nothing that was promoted"
    )
    assert "{uncertified or exc}" in source, (
        "the refusal message must carry the CORPUS reason when there is one: `promote` can only say "
        "the calibration is missing, which is true and useless"
    )


def test_the_upload_uses_the_installers_query_set_generator() -> None:
    """One generator, or the desktop's corpora are judged by different evidence than the wizard's.

    `_generated_calibration_queries` is the other candidate and is deliberately NOT used: its
    negatives are a hardcoded list never checked against the corpus, and its positives are
    500-character chunk bodies rather than questions. `recall/wizard/queryset.py` exists because a
    measurement showed a non-disjoint gap class is not separable at all.
    """
    import inspect

    import recall_mcp.service as service

    source = inspect.getsource(service._certify_upload)
    assert "from recall.wizard.queryset import" in source
    assert "generate_offline" in source and "canonicalize" in source
    # The NAME appears, in the docstring explaining why it is not used. The CALL must not.
    assert "_generated_calibration_queries(" not in source


def test_an_uncertified_upload_is_reported_rather_than_raised() -> None:
    """The upload SUCCEEDED; only activation did not. Raising invites a rebuild for the same refusal."""
    import inspect

    import recall_mcp.service as service

    source = inspect.getsource(service._certify_upload)
    assert "return f\"calibration" in source, "a refusal to certify returns a reason"
    assert "return f\"no certifiable query set" in source

    # Parsed, not grepped. The first version of this assertion tested for the SUBSTRING "raise" and
    # went red on the docstring's own word "raises" — a check that cannot tell a statement from
    # prose is a check that gets deleted the first time it is inconvenient.
    import ast

    body = ast.parse(textwrap.dedent(source)).body[0]
    raises = [node for node in ast.walk(body) if isinstance(node, ast.Raise)]
    assert not raises, (
        "no domain failure here may raise past the caller: `generation_ingest`'s outer handler "
        "re-raises, so an escape leaves the generation READY and unreclaimable — the leak the "
        "refusal path was fixed to prevent, one step earlier"
    )
    handled = {
        node.type.id if isinstance(node.type, ast.Name) else None
        for node in ast.walk(body)
        if isinstance(node, ast.ExceptHandler)
    }
    assert "Exception" not in handled, (
        "`CalibrationError`, not `Exception`: a domain failure is a reason, a bug is still a bug"
    )


# ----------------------------------------------------------------------------------------------
# One definition of what a pipeline IS
# ----------------------------------------------------------------------------------------------


def test_the_upload_records_the_same_pipeline_identity_the_cli_would() -> None:
    """⛔ A hand-assembled identity wrote a different fingerprint for the same pipeline.

    `generation_ingest` used to construct its own `EmbedderIdentity` and `ChunkerIdentity` instead
    of calling `pipeline_for`. Both were copies of rules that already existed, and both were wrong:

    * `provider="fastembed"` was hardcoded. `HashingEmbedder` is shipped in this repository and
      identifies itself as provider `recall` at revision `hashing-md5-bow-v1`, so an upload recorded
      it as a fastembed artifact — false provenance in an IMMUTABLE lineage record, the one place a
      wrong value cannot later be corrected.
    * The chunker identity was spelled `ChunkerIdentity("recall.chunk_text", 1, {})`, with empty
      configuration. `chunker_for` records the real parameters.

    The second is the bigger half, because it bites EVERY upload rather than only the hashing one.
    Measured before the fix, on `HashingEmbedder(dim=64)`:

        documents: old chunker config {}   new {'document_blocks': 'table-row-groups-v1',
                                                'max_chars': 800, 'overlap': 80}
                   fingerprint a7ab8ced8b2e754d... vs 895236ceca0112df...
        code:      old chunker config {}   new {'max_chars': 800}
                   fingerprint 327a84f9b078a30c... vs d25473cebd6a10c1...

    A generation built by the wizard and one built by a desktop upload therefore carried different
    pipeline fingerprints for the same pipeline, and a fingerprint mismatch is exactly what makes a
    published calibration resolve STALE.
    """
    import ast
    import inspect

    import recall_mcp.service as service

    # ⚠️ **Parsed, not grepped.** A substring test failed on the COMMENT that explains the removal,
    # which is the second time in this file that a check unable to tell code from prose went red on
    # its own documentation. A check like that gets deleted the first time it is inconvenient.
    tree = ast.parse(textwrap.dedent(inspect.getsource(service.generation_ingest)))
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "pipeline_for" in called, "the identity must come from the shared builder"
    assert "EmbedderIdentity" not in called, (
        "hardcoding the provider records a fastembed artifact for an embedder that is not one"
    )
    assert "ChunkerIdentity" not in called, (
        "spelling the chunker identity by hand disagrees with `chunker_for` about its configuration"
    )


def test_the_shared_builder_disagrees_with_the_hand_written_identity() -> None:
    """The claim above is only worth pinning if the two really differ. Assert that they do.

    Guards against a future change that makes `pipeline_for` produce the old empty configuration,
    which would leave the test above passing while the defect returned.
    """
    from recall.embeddings import HashingEmbedder
    from recall.generation_build import BuildRequest, pipeline_for
    from recall.lineage import ChunkerIdentity, EmbedderIdentity, PipelineIdentity

    embedder = HashingEmbedder(dim=64)
    hand_written = PipelineIdentity(
        EmbedderIdentity(
            provider="fastembed",
            model=embedder.name,
            dimension=embedder.dim,
            unverified_reason="desktop local development build",
        ),
        ChunkerIdentity("recall.chunk_text", 1, {}),
    )
    shared = pipeline_for(embedder, BuildRequest(chunker="text", unverified=True))[1]

    assert shared.embedder.provider == "recall", (
        "a HashingEmbedder is shipped here and identifies itself; recording it as fastembed is a "
        "provenance claim about somebody else's weights"
    )
    assert dict(shared.chunker.configuration), (
        "the shared builder must record the chunker's real parameters, not an empty mapping"
    )
    assert shared.fingerprint != hand_written.fingerprint, (
        "if these agree the hand-written identity was harmless and this whole change is noise; "
        "they did not agree when measured"
    )


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# The reclaim, asked BEHAVIOURALLY.
#
# ⛔ The three source-inspection tests above these were defeated by executed mutation, twice, by two
# independent auditors: `corpus_version_prefix=None` (the full pre-fix destructive behaviour) and
# `if False:` around the success-path call both left the file green. An AST test can see that a
# keyword is NAMED but not what it is BOUND to, and can count call sites but not tell a reachable
# one from dead code. A third auditor watched one of them fail spuriously when `inspect.getsource`
# returned a single line. These drive the real function against a recording stub instead.
# ─────────────────────────────────────────────────────────────────────────────────────────────────


class _RecordingManager:
    """The two methods `_release_superseded` uses, recording how they were called."""

    def __init__(self, stale: tuple[str, ...] = (), *, listing_error: Exception | None = None):
        self._stale = stale
        self._listing_error = listing_error
        self.listed: list[tuple[str, object]] = []
        self.abandoned: list[str] = []

    def superseded_ready_generations(self, keep, *, corpus_version_prefix=None):
        self.listed.append((keep, corpus_version_prefix))
        if self._listing_error is not None:
            raise self._listing_error
        return self._stale

    def abandon(self, generation_id, reason):  # noqa: ARG002 - the reason is not under test
        self.abandoned.append(generation_id)


def test_the_reclaim_is_confined_to_this_paths_own_corpus_prefix() -> None:
    """⛔ The VALUE of the prefix, not just the presence of the keyword.

    `superseded_ready_generations` keeps the unfiltered branch for `corpus_version_prefix=None`, so
    passing None restores exactly the behaviour that abandons a generation a wizard install built
    and deliberately left READY — and gc then deletes a corpus this path never created. The AST test
    this replaces accepted None, which was demonstrated by mutation.
    """
    from recall_mcp.service import _DESKTOP_CORPUS_PREFIX, _release_superseded

    manager = _RecordingManager(stale=("gen-old-1", "gen-old-2"))
    released = _release_superseded(manager, "gen-new")

    assert manager.listed == [("gen-new", _DESKTOP_CORPUS_PREFIX)], (
        "the reclaim must be confined by the desktop corpus prefix, and the prefix must be the "
        "constant rather than None, which selects on state alone"
    )
    assert _DESKTOP_CORPUS_PREFIX, "an empty prefix would match every corpus, confining nothing"
    assert manager.abandoned == ["gen-old-1", "gen-old-2"]
    assert released == 2
    assert "gen-new" not in manager.abandoned, "the generation being kept must never be abandoned"


def test_a_failed_listing_does_not_lose_the_upload() -> None:
    """⛔ The listing opens its own database connection, and it used to sit OUTSIDE the try.

    `_release_superseded`'s docstring says losing a reclaim must not lose the upload report. It was
    only true of the abandons. A raise from the listing was destructive in both directions: on the
    refusal path the call sits inside `generation_ingest`'s outer try, so it reached the cleanup
    handler and abandoned the very generation the design keeps; on the success path it reported an
    already-promoted, already-live upload as failed. Four auditors found this.
    """
    from recall_mcp.service import _release_superseded

    manager = _RecordingManager(listing_error=RuntimeError("connection reset"))

    assert _release_superseded(manager, "gen-new") == 0, "a failed reclaim reclaims nothing"
    assert manager.abandoned == [], "and abandons nothing"


def test_a_failed_abandon_does_not_stop_the_others() -> None:
    """Best effort is per generation, not all-or-nothing."""
    from recall_mcp.service import _release_superseded

    manager = _RecordingManager(stale=("gen-a", "gen-b", "gen-c"))
    refused = {"gen-b"}
    original = manager.abandon

    def abandon(generation_id, reason):
        if generation_id in refused:
            raise RuntimeError("already retired")
        original(generation_id, reason)

    manager.abandon = abandon  # type: ignore[method-assign]

    assert _release_superseded(manager, "gen-new") == 2
    assert manager.abandoned == ["gen-a", "gen-c"]


def test_the_reclaim_call_sites_are_both_reachable() -> None:
    """⛔ Counting call sites cannot tell a reachable call from dead code.

    An auditor wrapped the success-path call in `if False:` and the previous `len(sites) >= 2`
    assertion stayed green — the leak reopened with the suite passing. This asserts PLACEMENT: one
    call inside the `except UnsafePromotion` handler, and one that is a direct statement of the
    function body, not nested in any conditional or handler.
    """
    import ast
    import inspect
    import textwrap

    import recall_mcp.service as service

    tree = ast.parse(textwrap.dedent(inspect.getsource(service.generation_ingest)))
    function = tree.body[0]
    assert isinstance(function, ast.FunctionDef)

    def calls_reclaim(node: ast.AST) -> bool:
        return any(
            isinstance(inner, ast.Call)
            and isinstance(inner.func, ast.Name)
            and inner.func.id == "_release_superseded"
            for inner in ast.walk(node)
        )

    top_level = [statement for statement in function.body if calls_reclaim(statement)]
    assert any(isinstance(statement, ast.Expr) for statement in top_level), (
        "the success-path reclaim must be an unconditional statement of the function body; nesting "
        "it in a conditional makes it dead code that a call-site count cannot see"
    )

    handlers = [
        handler
        for handler in ast.walk(function)
        if isinstance(handler, ast.ExceptHandler)
        and isinstance(handler.type, ast.Name)
        and handler.type.id == "UnsafePromotion"
    ]
    assert handlers, "the refusal path must still be an UnsafePromotion handler"
    assert any(calls_reclaim(handler) for handler in handlers), (
        "the refusal path must reclaim too: one call site puts us back where we started"
    )


def test_a_failure_before_validate_is_also_reclaimed() -> None:
    """⛔ `abandon` refuses every state but READY, and `validate()` is what sets READY.

    So the cleanup handler, which called only `abandon`, was a no-op for any failure inside
    `build()` — the longest and most failure-prone step — and `suppress` made the refusal silent.
    `gc` collects `retired` and `failed`, so those generations stranded a full corpus copy exactly
    as before the fix that claimed to close the leak.
    """
    from recall.generations import InvalidGenerationTransition
    from recall_mcp.service import _reclaim_failed

    class _Building:
        def __init__(self):
            self.failed: list[str] = []
            self.abandoned: list[str] = []

        def fail(self, generation_id, reason):  # noqa: ARG002
            self.failed.append(generation_id)

        def abandon(self, generation_id, reason):  # noqa: ARG002
            self.abandoned.append(generation_id)

    building = _Building()
    _reclaim_failed(building, "gen-1", "build blew up")
    assert building.failed == ["gen-1"], "an in-flight generation is failed, which gc can collect"

    class _Ready(_Building):
        def fail(self, generation_id, reason):  # noqa: ARG002
            raise InvalidGenerationTransition("ready generations are abandoned, not failed")

    ready = _Ready()
    _reclaim_failed(ready, "gen-2", "promote blew up")
    assert ready.abandoned == ["gen-2"], "and a READY one falls back to abandon"

    class _Broken(_Building):
        def fail(self, generation_id, reason):  # noqa: ARG002
            raise RuntimeError("database gone")

        def abandon(self, generation_id, reason):  # noqa: ARG002
            raise RuntimeError("database gone")

    _reclaim_failed(_Broken(), "gen-3", "everything blew up")  # must not raise


def test_a_carried_forward_file_that_still_exists_is_never_dropped(tmp_path) -> None:
    """⛔ The carry-forward filter has now been wrong in BOTH directions, so both are pinned.

    Version one confined the build reader to the upload staging directory and kept every object, so
    a wizard-built corpus under `data_root` failed the whole upload. Version two dropped whatever
    the reader could not reach, which was worse: the upload succeeded with a truncated manifest,
    promotion put it live, the generation still holding those files was retired, and gc deleted it
    after the retention window — silent, permanent loss of the user's corpus.

    The only object that may be dropped is one whose bytes are gone, because nothing can rebuild it.
    """
    from recall_mcp.service import _local_path, _roots_of

    elsewhere = tmp_path / "wizard-data" / "docs"
    elsewhere.mkdir(parents=True)
    present = elsewhere / "kept.md"
    present.write_text("carried forward", encoding="utf-8")
    missing = elsewhere / "gone.md"

    assert _local_path(present.as_uri()) == present.resolve()
    assert _local_path(present.as_uri()).is_file()
    assert not _local_path(missing.as_uri()).is_file(), (
        "a staged file whose container was recreated is gone, and dropping it is the only option "
        "left; keeping it makes every later upload fail forever on a file that cannot be read"
    )

    roots = _roots_of({present.as_uri(): object()})
    assert elsewhere.resolve() in roots, (
        "the build reader must be widened to the directory a carried-forward object lives in, "
        "rather than the object being filtered out for living outside the staging root"
    )

    # A percent-encoded name and a non-local URI: the first must survive, the second names no root.
    spaced = elsewhere / "two words.md"
    spaced.write_text("x", encoding="utf-8")
    assert _local_path(spaced.as_uri()) == spaced.resolve()
    assert _local_path("s3://bucket/key.md") is None
    assert _roots_of({"s3://bucket/key.md": object()}) == (), (
        "a non-local object contributes no root, so the reader refuses it and the build fails "
        "LOUDLY — which is the right outcome for a corpus this path cannot rebuild"
    )

    # ⛔ **The filter and the fetcher must give the SAME answer, so this asks both.** The first
    # version of this helper re-derived URI-to-path with `urlparse` and `unquote`, and disagreed
    # with `LocalObjectReader` in three ways: it decoded twice (so `percent%20literal.md` named a
    # different existing file), it dropped the UNC authority (so a network-share corpus was called
    # unreachable and silently excluded), and it raised `ValueError` on an unbalanced `[` straight
    # out of the carry-forward loop, failing the whole upload from the code meant to make a bad URI
    # harmless. It delegates now, and this compares the two rather than restating one of them.
    from pathlib import Path

    from recall.lineage import LineageError
    from recall.manifest import LocalObjectReader, ManifestObjectV1, ObjectNotAllowed

    def reader_says(uri: str, roots: tuple[Path, ...]) -> Path | None:
        # A file:// object's version_id must BE its content digest; `recall.lineage` enforces
        # that, because a local file has no version other than its contents.
        digest = "0" * 64
        try:
            # The TYPE refuses some URIs outright, before any resolution happens. That is worth
            # knowing: a malformed one can only reach `_local_path` from stored data, never
            # from a freshly constructed manifest, which is exactly why the filter must
            # answer rather than raise.
            entry = ManifestObjectV1(uri, digest, "text/markdown", 0, digest)
        except (ValueError, LineageError):
            return None
        try:
            return LocalObjectReader(roots)._resolve(entry)
        except ObjectNotAllowed:
            return None

    # Wide enough that only the URI-to-path decision can differ, never the containment check.
    wide = (Path("/"), Path("C:/"), Path("//nas1/share"))
    for uri in (
        present.as_uri(),
        spaced.as_uri(),
        "file://nas1/share/docs/a.md",
        "file:///C:/x/percent%2520literal.md",
        "s3://bucket/key.md",
        "file://[unbalanced/x.md",
        "not a uri at all",
    ):
        mine = _local_path(uri)
        theirs = reader_says(uri, wide)
        if mine is not None and theirs is not None:
            assert mine == theirs, f"{uri}: filter says {mine}, reader says {theirs}"

    assert _local_path("file://[unbalanced/x.md") is None, (
        "a malformed URI is an ANSWER, not a crash: raising here failed the entire upload"
    )
    assert _local_path("file://nas1/share/docs/a.md") is not None, (
        "a UNC share is an ordinary thing to have on the platform this targets, and the reader "
        "carries the authority deliberately; calling it unreachable excluded the whole corpus"
    )


def test_the_certification_asks_for_headroom_before_it_asks_for_the_floor() -> None:
    """⛔ Asking for the FLOOR refuses corpora that are genuinely separable.

    Certification tests the LOWER bound of the Hanley-McNeil interval, and that bound tightens as
    the sample grows, so the floor is a certification threshold and not a generation target. The
    numbers below are executed rather than quoted, because the whole argument for the ladder rests
    on them and a quoted number in a comment is one nobody re-checks:

        separability_interval(0.95, 20, 20) -> lower ~0.8786   refused (bar is 0.90)
        separability_interval(0.95, 40, 40) -> lower ~0.9001   certified

    So every corpus whose true separability lies in roughly [0.950, 0.962) certifies through the
    installer and was refused by the desktop upload, leaving it READY and never live. The fix
    shipped with no test at all, so a revert to `per_class=MIN_PER_CLASS` rebuilt that wall
    silently.
    """
    import ast
    import inspect
    import textwrap

    import recall_mcp.service as service
    from recall.calibration import separability_interval
    from recall.wizard.queryset import DEFAULT_PER_CLASS, MIN_PER_CLASS

    # The measured premise. If this stops holding, the ladder is arguing from a fact that changed.
    at_floor = separability_interval(0.95, MIN_PER_CLASS, MIN_PER_CLASS)[0]
    at_headroom = separability_interval(0.95, DEFAULT_PER_CLASS, DEFAULT_PER_CLASS)[0]
    assert at_floor < 0.90 <= at_headroom, (
        f"the ladder exists because the floor's lower bound ({at_floor:.4f}) misses the 0.90 bar "
        f"that the headroom's ({at_headroom:.4f}) clears; if that is no longer true, the ladder is "
        "arguing from a fact that has changed"
    )

    # And the order it actually climbs them in. Structural, because `_certify_upload` opens a
    # database before it reaches the ladder; the behavioural twin is the wizard's, below.
    tree = ast.parse(textwrap.dedent(inspect.getsource(service._certify_upload)))
    ladders = [
        [element.id for element in node.iter.elts if isinstance(element, ast.Name)]
        for node in ast.walk(tree)
        if isinstance(node, ast.For) and isinstance(node.iter, ast.Tuple)
    ]
    assert ["DEFAULT_PER_CLASS", "MIN_PER_CLASS"] in ladders, (
        f"the headroom must be tried FIRST and the floor only as a fallback, found {ladders}"
    )


def test_the_wizard_falls_back_to_the_floor_rather_than_refusing(tmp_path) -> None:
    """The same ladder, driven rather than read, on the path where driving it is cheap.

    ⚠️ The two are NOT identical in shape and this no longer pretends otherwise. An earlier version
    of this test asserted both spelled the ladder as a literal tuple; the wizard builds it as
    `[per_class] if per_class <= MIN_PER_CLASS else [per_class, MIN_PER_CLASS]`, because its
    per-class size is a parameter rather than a constant. The shared property is the BEHAVIOUR:
    ask high, fall back to the floor, refuse only when the floor also fails.
    """
    from recall.wizard import pipeline
    from recall.wizard.queryset import MIN_PER_CLASS, QuerySetError

    asked: list[int] = []

    def generator(chunks, *, per_class, seed):  # noqa: ARG001
        asked.append(per_class)
        if per_class != MIN_PER_CLASS:
            raise QuerySetError(f"corpus cannot produce {per_class} per class")
        return [{"query": "q", "answer": "a"}]

    original_generate = pipeline.generate_offline
    original_chunks = pipeline.chunks_from_directory
    original_canon = pipeline.canonicalize
    try:
        pipeline.generate_offline = generator  # type: ignore[assignment]
        pipeline.chunks_from_directory = lambda *a, **k: ["chunk"]  # type: ignore[assignment]
        pipeline.canonicalize = lambda entries: entries  # type: ignore[assignment]

        spec = SimpleNamespace(root=tmp_path, glob="*.md", tenant="docs")
        entries, used = pipeline._labelled_set(
            spec, lambda text: [text], per_class=MIN_PER_CLASS * 2, seed=1
        )
        assert asked == [MIN_PER_CLASS * 2, MIN_PER_CLASS], (
            f"headroom first, floor as the fallback, got {asked}"
        )
        assert used == MIN_PER_CLASS, "and it reports the size it actually used, not the one it asked for"
        assert entries

        # A request already at or below the floor makes ONE attempt: there is nothing to fall back to.
        asked.clear()
        pipeline._labelled_set(spec, lambda text: [text], per_class=MIN_PER_CLASS, seed=1)
        assert asked == [MIN_PER_CLASS]
    finally:
        pipeline.generate_offline = original_generate  # type: ignore[assignment]
        pipeline.chunks_from_directory = original_chunks  # type: ignore[assignment]
        pipeline.canonicalize = original_canon  # type: ignore[assignment]
