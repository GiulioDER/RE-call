"""The opt in retrieval side of the memory observability surface.

`recall.explanations.memory_audit` reports what one retrieval decision can observe about itself
(verdict mix, source diversity, validity coverage, staleness, evidence selection) and states the
measurements it cannot make (precision, forgetting, action influence) as ``not_measured`` rather
than inventing them. It is reached by clients only through ``explanation.details.memory_audit`` on
``recall_search`` and ``recall_evidence`` with ``explain=true``, so the last two tests drive the
registered MCP tools through ``Tool.run`` (argument validation, the shared retrieval body, the
REAL ``trusted_search`` trust layer and the JSON serializer), not the function underneath. Only the
storage is a stub: an in-process store in the same shape as ``tests/test_advice_injection.py``'s,
whose supersession edge makes the trust layer itself produce the ``superseded`` verdict.

Red proofs (2026-09-25, base ``e9172125``), each run on its own node, each failing in the
assertion named, then restored and seen green:

* ``test_memory_audit_reports_observed_retrieval_and_context_counts``: mutating
  ``stale_verdicts`` in ``memory_audit`` to ``{"superseded"}`` fails with
  ``assert 1 == 2`` on ``audit["stale_count"]``.
* ``test_memory_audit_does_not_leak_corpus_content_and_handles_empty_results``: deleting the
  ``"distinct_source_count": None`` key from the ``not_applicable`` context shape fails the
  ``audit["context"] == {...}`` equality; replacing ``source_diversity``'s ``else None`` with
  ``else 0.0`` fails ``assert 0.0 is None``.
* ``test_recall_search_tool_serves_the_memory_audit_only_when_explained``: deleting the
  ``details={"memory_audit": ...}`` argument from ``recall_mcp.retrieval.search_memory`` fails
  with ``KeyError: 'memory_audit'`` raised by the assertion's own lookup of
  ``explanation["details"]["memory_audit"]`` (details is ``{}``); changing ``if explain:`` there
  to ``if True:`` fails ``assert "explanation" not in plain``.
* ``test_recall_evidence_tool_audits_the_bundle_selection``: passing ``context_chunk_ids=None``
  in ``recall_mcp.retrieval.evidence_memory`` fails ``assert None == 1`` on
  ``context["selected_count"]``; deleting its ``details=`` argument fails with
  ``KeyError: 'memory_audit'`` in the same lookup.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

from recall.calibration import Calibration
from recall.calibration_v2 import CalibrationResolution, CalibrationStatus
from recall.explanations import memory_audit
from recall.types import Chunk, Provenance, ScoredChunk, TrustedHit, Validity
from recall_mcp import server
from recall_mcp.settings import Settings

NOW = datetime(2026, 8, 28, 12, 0, tzinfo=UTC)


def _hit(
    chunk_id: str,
    source: str,
    verdict: str,
    *,
    valid_from: datetime | None = None,
    superseded_by: str | None = None,
) -> TrustedHit:
    return TrustedHit(
        chunk=Chunk(chunk_id, source, f"private text for {chunk_id}", {"file": source}),
        cosine=0.9,
        confidence=0.8,
        verdict=verdict,  # type: ignore[arg-type]
        provenance=Provenance(source, source, 0, NOW),
        validity=Validity(valid_from, None, superseded_by),
    )


def test_memory_audit_reports_observed_retrieval_and_context_counts() -> None:
    hits = [
        _hit("one", "a.md", "ok", valid_from=NOW),
        _hit("two", "a.md", "superseded", superseded_by="new.md"),
        _hit("three", "b.md", "expired"),
    ]

    audit = memory_audit(hits, context_chunk_ids=["one", "three"])

    assert audit["retrieved_count"] == 3
    assert audit["trusted_count"] == 1
    assert audit["untrusted_count"] == 2
    assert audit["verdict_counts"] == {"expired": 1, "ok": 1, "superseded": 1}
    assert audit["distinct_source_count"] == 2
    assert audit["source_diversity"] == 0.6667
    assert audit["validity_declared_count"] == 1
    assert audit["validity_coverage"] == 0.3333
    assert audit["supersession_declared_count"] == 1
    assert audit["stale_count"] == 2
    assert audit["stale_rate"] == 0.6667
    assert audit["context"] == {
        "selected_count": 2,
        "selection_ratio": 0.6667,
        "distinct_source_count": 2,
        "status": "selected",
    }
    assert audit["quality"]["status"] == "not_measured"
    assert audit["quality"]["retrieval_precision"] is None
    assert audit["influence"]["status"] == "not_measured"


def test_memory_audit_does_not_leak_corpus_content_and_handles_empty_results() -> None:
    audit = memory_audit([])

    assert audit["retrieved_count"] == 0
    assert audit["source_diversity"] is None
    assert audit["validity_coverage"] is None
    assert audit["stale_rate"] is None
    # One schema for both tools: the search shape carries the same keys as the evidence shape.
    assert audit["context"] == {
        "selected_count": None,
        "selection_ratio": None,
        "distinct_source_count": None,
        "status": "not_applicable",
    }
    assert "retrieval_precision" in audit["quality"]

    populated = memory_audit([_hit("one", "secret-name.md", "ok")], context_chunk_ids=["one"])
    assert "private text" not in repr(populated)
    assert "secret-name" not in repr(populated)
    assert "'one'" not in repr(populated)


# --------------------------------------------------------------------------------------------
# The registered MCP tools, end to end through the real trust layer
# --------------------------------------------------------------------------------------------


class _Store:
    """In-process store: two chunks of `new.md`, one of `old.md`, and `old.md -> new.md`."""

    tenant = "default"
    generation_id = "legacy"

    def query_dense(self, vector, k, source=None):  # type: ignore[no-untyped-def]
        def scored(chunk_id: str, file: str, ordinal: int, score: float) -> ScoredChunk:
            return ScoredChunk(
                chunk=Chunk(
                    id=chunk_id,
                    source=f"/corpus/{file}",
                    text=f"private text for {chunk_id}",
                    metadata={"file": file, "ord": ordinal},
                ),
                score=score,
                indexed_at=NOW,
            )

        return [
            scored("new#0", "new.md", 0, 0.95),
            scored("old#0", "old.md", 0, 0.94),
            scored("new#1", "new.md", 1, 0.93),
        ]

    def query_sparse(self, query, k, source=None, vec=None):  # type: ignore[no-untyped-def]
        return []

    def newest_indexed_at(self) -> datetime:
        return NOW

    def supersession(self):  # type: ignore[no-untyped-def]
        return {"old.md": "new.md"}, frozenset()

    def resolve_calibration(self) -> CalibrationResolution:
        # A certified threshold, so the trust layer issues `ok` verdicts instead of `unverified`.
        # Only the three attributes `trusted_search` reads from the artifact are provided.
        artifact = SimpleNamespace(
            runtime=Calibration(embedder="audit-fixture", threshold=0.5),
            calibration_id="cal-audit-fixture",
            query_set_digest="qs-audit-fixture",
        )
        return CalibrationResolution(CalibrationStatus.CERTIFIED, artifact)  # type: ignore[arg-type]


class _Embedder:
    dim = 2
    name = "audit-fixture"

    def embed(self, texts):  # type: ignore[no-untyped-def]
        return [[1.0, 0.0] for _ in texts]


def _call(tool_name: str, **arguments: Any) -> dict[str, Any]:
    tool = {item.name: item for item in server.build_server()._tool_manager.list_tools()}[tool_name]
    context = SimpleNamespace(
        request_context=SimpleNamespace(
            lifespan_context={
                "store": _Store(),
                "embedder": _Embedder(),
                "calibration": None,
                "settings": Settings.from_env({"RECALL_TRUST_MODE": "development"}),
            }
        )
    )
    raw = asyncio.run(tool.run({"query": "what is the rate limit?", **arguments}, context))
    return json.loads(raw)


def test_recall_search_tool_serves_the_memory_audit_only_when_explained() -> None:
    plain = _call("recall_search")
    assert "explanation" not in plain

    payload = _call("recall_search", explain=True)
    verdicts = sorted(hit["verdict"] for hit in payload["hits"])
    assert verdicts == ["ok", "ok", "superseded"]  # produced by the trust layer, not the fixture

    audit = payload["explanation"]["details"]["memory_audit"]
    assert audit["retrieved_count"] == 3
    assert audit["trusted_count"] == 2
    assert audit["verdict_counts"] == {"ok": 2, "superseded": 1}
    assert audit["distinct_source_count"] == 2
    assert audit["supersession_declared_count"] == 1
    assert audit["stale_count"] == 1
    assert audit["context"]["status"] == "not_applicable"
    assert audit["quality"] == {
        "retrieval_precision": None,
        "status": "not_measured",
        "reason": "requires labelled queries or downstream judgement",
    }
    assert "private text" not in json.dumps(payload["explanation"])


def test_recall_evidence_tool_audits_the_bundle_selection() -> None:
    payload = _call("recall_evidence", explain=True, max_items=1)
    assert [item["chunk_id"] for item in payload["items"]] == ["new#0"]

    audit = payload["explanation"]["details"]["memory_audit"]
    assert audit["retrieved_count"] == 3
    assert audit["verdict_counts"] == {"ok": 2, "superseded": 1}
    assert audit["context"] == {
        "selected_count": 1,
        "selection_ratio": 0.3333,
        "distinct_source_count": 1,
        "status": "selected",
    }
    assert "private text" not in json.dumps(payload["explanation"])
