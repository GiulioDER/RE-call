"""PERF-001: bound accumulated upload calibration before materializing corpus text.

Red proof is run by the primary on the remote test runner against baseline 160a23a3f with this same
file. The budget limits and unavailable-budget case fail the READY-state assertion on that
baseline, which certifies and activates them. No large payload is allocated here:
the database boundary reports its aggregate size and records materialization.
"""

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from recall.generations import NoActiveGeneration, UnsafePromotion
import recall.generation_build as generation_build
import recall_mcp.desktop_ingest as ingest


class _Manager:
    certification_required = True

    def __init__(self):
        self.state = "new"
        self.certified = False
        self.manifest = None
        self.failed = []

    def tenant_ingest_lock(self):
        return nullcontext()

    def servable_manifest(self):
        raise NoActiveGeneration("first upload")

    def create(self, manifest, pipeline, **kwargs):
        self.manifest = manifest
        self.state = "created"
        return SimpleNamespace(generation_id="gen-upload")

    def build(self, generation_id, reader, embedder, chunker, **kwargs):
        self.state = "built"
        return SimpleNamespace(objects=1, chunks=4)

    def validate(self, generation_id):
        self.state = "ready"

    def promote(self, generation_id, **kwargs):
        if not self.certified:
            raise UnsafePromotion("no published certified calibration")
        self.state = "active"

    def superseded_ready_generations(self, keep, **kwargs):
        return ()

    def fail(self, generation_id, reason):
        self.failed.append((generation_id, reason))
        self.state = "failed"


class _Connection:
    def __init__(self, count, size):
        self.count = count
        self.size = size
        self.materialized = False
        self.queries = []
        self.texts = ["First document", "Second document", "Third document", "Fourth document"]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def transaction(self):
        return nullcontext()

    def execute(self, query, params=None):
        self.queries.append((query, params))
        return self

    def fetchone(self):
        if self.count is None:
            return None
        return self.count, self.size

    def fetchall(self):
        self.materialized = True
        return [(text,) for text in self.texts]


def _upload(monkeypatch, tmp_path, *, count, size):
    manager = _Manager()
    connection = _Connection(count, size)
    query_sets = []
    published = []

    class Repository:
        def __init__(self, *args, **kwargs):
            pass

        def calibrate(self, generation_id, entries, embedder):
            manager.certified = True
            return SimpleNamespace(certified=True, calibration_id="cal-upload")

        def publish(self, calibration_id):
            published.append(calibration_id)

    def query_set(chunks):
        query_sets.append(chunks)
        return [{"query": "offline calibration question", "answerable": True}], None

    monkeypatch.setattr(ingest, "GenerationManager", lambda *args, **kwargs: manager)
    monkeypatch.setattr(ingest, "CalibrationRepository", Repository)
    monkeypatch.setattr(ingest.psycopg, "connect", lambda *args, **kwargs: connection)
    monkeypatch.setattr(ingest, "_query_set_for", query_set)
    monkeypatch.setattr(ingest, "embedder_artifact_digest", lambda embedder: "a" * 64)
    monkeypatch.setattr(
        generation_build,
        "pipeline_for",
        lambda *args: (object(), SimpleNamespace(verified=True)),
    )
    staged = tmp_path / "tenant" / "upload"
    staged.mkdir(parents=True)
    (staged / "new.md").write_text("new uploaded document", encoding="utf-8")
    result = ingest.generation_ingest(
        SimpleNamespace(_dsn="unused", tenant="tenant-docs"),
        object(),
        str(staged),
        "documents",
        env={"RECALL_ENV": "production"},
    )
    return result, manager, connection, query_sets, published


@pytest.mark.parametrize(
    ("count", "size"),
    [(4, 64 * 1024 * 1024 + 1), (100_001, 4), (None, 0)],
    ids=["text-byte-limit", "chunk-count-limit", "missing-budget-row"],
)
def test_over_budget_calibration_keeps_upload_ready_without_materializing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, count: int, size: int
) -> None:
    """Pin both cumulative limits at generation_ingest's real certification consumer.

    Baseline 160a23a3f lacks the guard in desktop_ingest._certify_upload and leaves
    manager.state='active', violating the first assertion. The same nodes run
    green after the guard. The remote proof receipt records the observed runs.
    """
    result, manager, connection, query_sets, published = _upload(
        monkeypatch, tmp_path, count=count, size=size
    )

    assert manager.state == "ready", (
        "an upload exceeding the automatic calibration budget must stay READY"
    )
    assert connection.materialized is False, "the budget must be checked before fetching corpus text"
    assert query_sets == []
    assert published == []
    assert manager.failed == [], "a bounded calibration refusal must retain the validated upload"
    assert manager.manifest is not None and len(manager.manifest.objects) == 1
    assert "budget" in result.message.lower()
    assert "not live" in result.message.lower()
    assert "calibrate" in result.message.lower()

    # The unchanged success control belongs to this same baseline-red test node.
    below, normal, bounded, complete, published_below = _upload(
        monkeypatch, tmp_path / "below-budget", count=4, size=64
    )
    assert normal.state == "active", "a below-budget upload must still calibrate and activate"
    assert bounded.materialized is True
    assert complete == [bounded.texts], "the guard must not sample away corpus validation input"
    assert published_below == ["cal-upload"]
    assert normal.failed == []
    assert "activated" in below.message.lower()
