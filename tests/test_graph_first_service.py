from __future__ import annotations

from datetime import UTC, datetime, timedelta

from recall.semantic_graph import build_semantic_graph
from recall.trust_policy import TrustPolicy
from recall.types import Chunk, Provenance, StalenessReport, TrustedHit, TrustedResult, Validity
from recall_mcp import service


NOW = datetime(2026, 8, 28, tzinfo=UTC)


def _graph():
    return build_semantic_graph(
        [
            Chunk(
                "c1",
                "release.md",
                "Release decision",
                {
                    "file": "release.md",
                    "recall_graph": {
                        "entities": [
                            {"name": "Release", "kind": "project"},
                            {"name": "Deploy", "kind": "service"},
                        ],
                        "relations": [
                            {
                                "relation": "supports",
                                "subject": "Release",
                                "object": "Deploy",
                            }
                        ],
                    },
                },
            ),
            Chunk("c2", "deploy.md", "Deploy procedure", {"file": "deploy.md", "service": "Deploy"}),
        ],
        tenant_id="tenant-a",
        generation_id="generation-a",
        pipeline_fingerprint="p" * 64,
        corpus_fingerprint="c" * 64,
    )


def _result(query: str, ids: tuple[str, ...]) -> TrustedResult:
    hits = [
        TrustedHit(
            chunk=Chunk(chunk_id, f"{chunk_id}.md", f"trusted {chunk_id}"),
            cosine=0.9 - index * 0.01,
            confidence=0.9,
            verdict="ok",
            provenance=Provenance(f"{chunk_id}.md", f"{chunk_id}.md", index, NOW),
            validity=Validity(NOW, None, None),
        )
        for index, chunk_id in enumerate(ids)
    ]
    return TrustedResult(
        query=query,
        hits=hits,
        abstained=not hits,
        reason="" if hits else "retrieval_gap",
        gap_warning=not hits,
        staleness=StalenessReport(False, NOW, timedelta(days=1), timedelta(days=2)),
        calibration_status="certified",
        tenant_id="tenant-a",
        generation_id="generation-a",
        pipeline_fingerprint="p" * 64,
        corpus_fingerprint="c" * 64,
    )


class _Store:
    tenant = "tenant-a"
    generation_id = "generation-a"

    def generation_binding(self):
        return {
            "generation_id": self.generation_id,
            "pipeline_fingerprint": "p" * 64,
            "corpus_fingerprint": "c" * 64,
        }

    def graph_readiness(self):
        return _graph().readiness()

    def load_semantic_graph(self, generation_id=None):
        assert generation_id == self.generation_id
        return _graph()


def test_graph_first_builds_candidates_before_trusted_retrieval(monkeypatch):
    calls: list[str] = []

    def fake_retrieve(_store, _embedder, query, _source, _k, _calibration, _policy):
        calls.append(query)
        ids = ("c1",) if query == "release procedure" else ("c2",)
        return type("Wrapper", (), {"result": _result(query, ids)})()

    monkeypatch.setattr(service, "_retrieve_trusted", fake_retrieve)
    response = service.graph_first_retrieval(
        _Store(),
        object(),
        "release procedure",
        mode="hybrid",
        expected_generation_id="generation-a",
        policy=TrustPolicy.development(),
    )

    assert response["status"] == "complete"
    assert response["diagnostics"]["graph"]["readiness"] == "ready"
    assert response["diagnostics"]["retrieval_calls"] == len(calls)
    assert calls[0] == "release procedure"
    assert len(response["candidate_queries"]) <= 3
    assert response["new_trusted_chunk_ids"] == ["c2"]


def test_graph_first_fails_closed_on_generation_mismatch(monkeypatch):
    called = False

    def fail(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("retrieval must not run")

    monkeypatch.setattr(service, "_retrieve_trusted", fail)
    response = service.graph_first_retrieval(
        _Store(),
        object(),
        "release procedure",
        expected_generation_id="wrong-generation",
        policy=TrustPolicy.development(),
    )

    assert response["status"] == "refused"
    assert response["refusal_reason"] == "generation_mismatch"
    assert not called
