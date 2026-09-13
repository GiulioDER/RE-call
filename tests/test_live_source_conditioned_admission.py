from __future__ import annotations

from datetime import UTC, datetime, timedelta

from recall.calibration import Calibration
from recall.types import (
    Chunk,
    Provenance,
    RetrievalResult,
    ScoredChunk,
    StalenessReport,
    TrustedHit,
    TrustedResult,
    Validity,
)
from recall.retriever import RetrievalCandidateTrace
from recall_mcp import service
from scripts.run_live_source_conditioned_admission import (
    _leave_one_query_out_support,
    _source_select,
)
from scripts.run_live_tty_graph_precision import _command


def _chunk(chunk_id: str, source: str, ordinal: int, score: float) -> ScoredChunk:
    return ScoredChunk(
        Chunk(chunk_id, f"stored/{source}", chunk_id, {"file": source, "ord": ordinal}),
        score,
    )


def _staleness() -> StalenessReport:
    return StalenessReport(False, datetime.now(UTC), timedelta(0), timedelta(days=2))


def _trusted(hit: ScoredChunk, verdict: str) -> TrustedHit:
    source = str(hit.chunk.metadata["file"])
    ordinal = int(hit.chunk.metadata["ord"])
    return TrustedHit(
        chunk=hit.chunk,
        cosine=hit.score,
        confidence=0.8 if verdict == "ok" else 0.2,
        verdict=verdict,  # type: ignore[arg-type]
        provenance=Provenance(hit.chunk.source, source, ordinal, None),
        validity=Validity(None, None, None),
    )


def test_source_admission_audit_requires_generation_pin(monkeypatch) -> None:
    """The full trust pool is exposed only under both private benchmark gates.

    Red proof node ``source-admission-gate-01`` changes the production conjunction to a
    disjunction. The flag-only assertion then fails at its intended assertion.
    """
    monkeypatch.delenv("RECALL_BENCHMARK_PIN", raising=False)
    monkeypatch.delenv("RECALL_BENCHMARK_SOURCE_ADMISSION_AUDIT", raising=False)
    assert service._source_admission_benchmark_audit_enabled() is False

    monkeypatch.setenv("RECALL_BENCHMARK_SOURCE_ADMISSION_AUDIT", "1")
    assert service._source_admission_benchmark_audit_enabled() is False

    monkeypatch.setenv("RECALL_BENCHMARK_PIN", "1")
    assert service._source_admission_benchmark_audit_enabled() is True


def test_source_conditioning_reuse_audit_requires_generation_pin(monkeypatch) -> None:
    """The duplicate comparison cannot run from an ordinary shadow process.

    Red proof node ``source-conditioning-reuse-gate-01`` changes the production conjunction to a
    disjunction. The flag only assertion then fails at its intended assertion.
    """
    monkeypatch.delenv("RECALL_BENCHMARK_PIN", raising=False)
    monkeypatch.delenv("RECALL_BENCHMARK_SOURCE_CONDITIONING_REUSE_AUDIT", raising=False)
    assert service._source_conditioning_reuse_benchmark_audit_enabled() is False

    monkeypatch.setenv("RECALL_BENCHMARK_SOURCE_CONDITIONING_REUSE_AUDIT", "1")
    assert service._source_conditioning_reuse_benchmark_audit_enabled() is False

    monkeypatch.setenv("RECALL_BENCHMARK_PIN", "1")
    assert service._source_conditioning_reuse_benchmark_audit_enabled() is True


def test_source_admission_payload_preserves_pool_order_and_trust_verdict(monkeypatch) -> None:
    """The collector joins trust verdicts back to the pre-trust fused pool by chunk id.

    Red proof node ``source-admission-payload-01`` enumerates ``TrustedResult.hits`` instead of
    the captured pool. Trust evaluation moves the ok hit first, so the pool rank assertion fails.
    """
    first = _chunk("first", "recall/first.md", 0, 0.48)
    second = _chunk("second", "recall/second.md", 1, 0.62)

    def fake_trusted_search(store, embedder, query, **kwargs):
        assert query == "question"
        assert kwargs["k"] == 40
        assert kwargs["candidate_k"] == 20
        raw = RetrievalResult("question", [first, second], False, _staleness())
        transformed = kwargs["pre_trust_transform"](raw)
        assert transformed is raw
        return TrustedResult(
            "question",
            [_trusted(second, "ok"), _trusted(first, "low_confidence")],
            False,
            "",
            False,
            _staleness(),
        )

    class Embedder:
        dim = 2
        name = "test"

    monkeypatch.setattr(service, "trusted_search", fake_trusted_search)
    monkeypatch.setattr(service, "_build_reranker", lambda profile, env: None)
    payload = service._source_admission_benchmark_audit_payload(
        object(),
        Embedder(),
        "question",
        [0.2, 0.8],
        None,
        Calibration("test", 0.5, 0.05),
        None,
        service.FAST_PROFILE,
    )

    assert [
        (item["chunk_id"], item["pool_rank"], item["verdict"]) for item in payload["items"]
    ] == [
        ("first", 1, "low_confidence"),
        ("second", 2, "ok"),
    ]
    assert payload["threshold"] == 0.5


def test_reused_audits_preserve_fetched_leg_and_pretrust_pool_order() -> None:
    """The reused payload is identical in shape to the duplicate query audits.

    Red proof on 2026-09-13: the helper initially returned deliberately empty audit lists. The
    test failed at the dense identifier assertion, proving it reads the candidate trace rather
    than a separately reconstructed fixture.
    """
    first = _chunk("first", "recall/first.md", 0, 0.48)
    second = _chunk("second", "recall/second.md", 1, 0.62)
    raw = RetrievalResult("question", [first, second], False, _staleness())
    trusted = TrustedResult(
        "question",
        [_trusted(second, "ok"), _trusted(first, "low_confidence")],
        False,
        "",
        False,
        _staleness(),
    )
    trace = RetrievalCandidateTrace(
        raw,
        (second, first),
        (first,),
        tuple(),
    )

    legs, pool = service._source_conditioning_reused_audits(
        (trace, trusted, Calibration("test", 0.5, 0.05)),
        service.FAST_PROFILE,
    )

    assert [item["chunk_id"] for item in legs["dense"]] == ["second", "first"]
    assert [item["chunk_id"] for item in legs["sparse"]] == ["first"]
    assert [item["chunk_id"] for item in pool["items"]] == ["first", "second"]
    assert [item["verdict"] for item in pool["items"]] == ["low_confidence", "ok"]
    assert pool["threshold"] == 0.5


def test_source_admission_payload_reads_generation_calibration_when_not_injected(
    monkeypatch,
) -> None:
    """The audit reports the same resolved generation threshold that serving uses.

    Red proof node ``source-admission-calibration-01`` substitutes a 0.4 threshold after
    resolution. The final threshold assertion then fails at its intended assertion.
    """

    class Artifact:
        runtime = Calibration("test", 0.509, 0.05)

    class Resolution:
        artifact = Artifact()

    class Store:
        def resolve_calibration(self):
            return Resolution()

    class Embedder:
        dim = 2
        name = "test"

    def fake_trusted_search(store, embedder, query, **kwargs):
        assert kwargs["calibration"] is None
        raw = RetrievalResult(query, [], False, _staleness())
        kwargs["pre_trust_transform"](raw)
        return TrustedResult(query, [], True, "empty", False, _staleness())

    monkeypatch.setattr(service, "trusted_search", fake_trusted_search)
    monkeypatch.setattr(service, "_build_reranker", lambda profile, env: None)

    payload = service._source_admission_benchmark_audit_payload(
        Store(), Embedder(), "question", [0.2, 0.8], None, None, None, service.FAST_PROFILE
    )

    assert payload["threshold"] == 0.509


def test_source_selection_cannot_rescue_invalid_or_floor_breaching_chunks() -> None:
    """Source support can promote only a still plausible low-confidence chunk.

    Red proof node ``source-admission-boundary-01`` removes the raw cosine floor. The
    ``too-low`` candidate is then selected and this test fails at its intended assertion.
    """
    pool = [
        {
            "chunk_id": "invalid",
            "source": "strong",
            "pool_rank": 1,
            "cosine": 0.70,
            "verdict": "superseded",
        },
        {
            "chunk_id": "weak",
            "source": "weak",
            "pool_rank": 2,
            "cosine": 0.40,
            "verdict": "ok",
        },
        {
            "chunk_id": "promoted",
            "source": "strong",
            "pool_rank": 3,
            "cosine": 0.32,
            "verdict": "low_confidence",
        },
        {
            "chunk_id": "too-low",
            "source": "strong",
            "pool_rank": 4,
            "cosine": 0.29,
            "verdict": "low_confidence",
        },
    ]

    selected = _source_select(pool, {"strong": 1.0, "weak": 0.0}, 0.35, 0.20)

    assert [item["chunk_id"] for item in selected] == ["promoted"]


def test_leave_one_query_out_source_support_uses_general_features() -> None:
    """A held out source is scored from learned features, not its source or query identity.

    Red proof node ``source-admission-oof-01`` replaces every held out probability with 0.5. The
    positive versus negative ordering assertion then fails at its intended assertion.
    """
    rows = []
    positive = (0.9, 0.02, 0.02, 0.08, 0.8, 1.8, 0.04)
    negative = (0.2, 0.0, 0.0, 0.01, 0.0, 0.7, -0.04)
    for query_index in range(6):
        rows.extend(
            [
                {
                    "query_index": query_index,
                    "source": f"gold-{query_index}",
                    "features": positive,
                    "label": 1,
                },
                {
                    "query_index": query_index,
                    "source": f"noise-{query_index}",
                    "features": negative,
                    "label": 0,
                },
            ]
        )
    rows.extend(
        {
            "query_index": 6,
            "source": f"control-{index}",
            "features": negative,
            "label": 0,
        }
        for index in range(2)
    )

    support = _leave_one_query_out_support(rows)

    assert support[(0, "gold-0")] > support[(0, "noise-0")]
    assert support[(0, "gold-0")] > support[(6, "control-0")]


def test_tty_command_enables_source_admission_audit_only_when_requested(monkeypatch) -> None:
    """The live runner explicitly enables the private source admission trace.

    Red proof node ``source-admission-command-01`` omits the environment assignment. The audited
    command assertion then fails while the ordinary command stays unchanged.
    """
    monkeypatch.setenv(
        "RECALL_BENCHMARK_REMOTE_CODE_ROOT", "/home/sentiment/recall-repos/source-admission"
    )
    ordinary = _command(
        "memory", "voyage:voyage-4", "/srv/memory", "fast", "combined", "none", 1, 32, 0.10
    )[-1]
    audited = _command(
        "memory",
        "voyage:voyage-4",
        "/srv/memory",
        "fast",
        "combined",
        "none",
        1,
        32,
        0.10,
        "generation-one",
        benchmark_source_admission_audit=True,
    )[-1]

    assert "RECALL_BENCHMARK_SOURCE_ADMISSION_AUDIT" not in ordinary
    assert "RECALL_BENCHMARK_SOURCE_ADMISSION_AUDIT=1" in audited


def test_tty_command_enables_trace_reuse_audit_only_when_requested(monkeypatch) -> None:
    """The paired duplicate query arm stays behind both private benchmark gates.

    Red proof on 2026-09-13: the new command argument was accepted but deliberately discarded.
    The audited command assertion failed because the reuse audit environment flag was absent.
    """
    monkeypatch.setenv(
        "RECALL_BENCHMARK_REMOTE_CODE_ROOT", "/home/sentiment/recall-repos/source-reuse"
    )
    ordinary = _command(
        "memory", "voyage:voyage-4", "/srv/memory", "fast", "combined", "none", 1, 32, 0.10
    )[-1]
    audited = _command(
        "memory",
        "voyage:voyage-4",
        "/srv/memory",
        "fast",
        "combined",
        "none",
        1,
        32,
        0.10,
        "generation-one",
        benchmark_source_conditioning_reuse_audit=True,
    )[-1]

    assert "RECALL_BENCHMARK_SOURCE_CONDITIONING_REUSE_AUDIT" not in ordinary
    assert "RECALL_BENCHMARK_SOURCE_CONDITIONING_REUSE_AUDIT=1" in audited
