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


def test_a_refused_upload_releases_the_builds_it_supersedes() -> None:
    """⛔ A READY generation holds a full copy of the corpus and `gc` collects only retired/failed.

    `abandon` exists for exactly this state and says so in its own docstring; `wizard/pipeline.py`
    does the same on the same shape of failure. The refusal branch returned success without it, so
    every retry left another unreclaimable copy behind.

    The NEWEST is deliberately kept — it carries the whole corpus forward and is the one the message
    tells the user to certify.
    """
    import inspect

    import recall_mcp.service as service

    source = inspect.getsource(service.generation_ingest)
    assert "superseded_ready_generations(generation.generation_id)" in source, (
        "the refusal path must release the builds this one supersedes"
    )
    assert "manager.abandon(" in source, "released means abandoned, so gc can reclaim the rows"


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
