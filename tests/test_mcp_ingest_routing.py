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

import pytest

import recall_mcp.server as server


class _Recorder:
    """Stands in for the two ingest paths, recording which one was chosen."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def generation(self, store, embedder, staged_root, category):  # noqa: ANN001, ANN201
        self.calls.append("generation")
        return "generation-result"

    def legacy(self, store, embedder, staged_root):  # noqa: ANN001, ANN201
        self.calls.append("legacy")
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
