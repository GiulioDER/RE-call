from __future__ import annotations

import json

import pytest

from recall.source_conditioning import (
    REGISTERED_ALPHA,
    REGISTERED_CANDIDATE_K,
    REGISTERED_ITEM_BUDGET,
    REGISTERED_RAW_COSINE_FLOOR,
    REGISTERED_RRF_K,
    SOURCE_CONDITIONING_MODEL_ID,
    SOURCE_CONDITIONING_SCHEMA_VERSION,
    SOURCE_FEATURE_NAMES,
    SourceConditioningArtifact,
    SourceConditioningArtifactError,
    chunk_identifier_hash,
    select_source_conditioned,
    source_features,
)
from recall.types import (
    Chunk,
    Provenance,
    StalenessReport,
    TrustedHit,
    TrustedResult,
    Validity,
)
from recall_mcp import service
from recall_mcp.settings import ENVIRONMENT_SCHEMA, Settings
from scripts.run_live_source_conditioned_admission import _source_rows
from scripts.run_live_source_conditioning_same_vector import _decision as _same_vector_decision
from scripts.run_live_source_conditioning_shadow import _public_signature, _shadow_internal_ms
from scripts.run_live_tty_graph_precision import _command
from datetime import UTC, datetime, timedelta


def _artifact() -> SourceConditioningArtifact:
    return SourceConditioningArtifact(
        schema_version=SOURCE_CONDITIONING_SCHEMA_VERSION,
        model_id=SOURCE_CONDITIONING_MODEL_ID,
        feature_names=SOURCE_FEATURE_NAMES,
        means=(0.0,) * len(SOURCE_FEATURE_NAMES),
        scales=(1.0,) * len(SOURCE_FEATURE_NAMES),
        intercept=0.0,
        coefficients=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        alpha=REGISTERED_ALPHA,
        raw_cosine_floor=REGISTERED_RAW_COSINE_FLOOR,
        rrf_k=REGISTERED_RRF_K,
        candidate_k=REGISTERED_CANDIDATE_K,
        item_budget=REGISTERED_ITEM_BUDGET,
        embedding_profile="voyage-context-4-v1",
        retrieval_profile="fast",
        training_generation_id="generation",
        training_calibration_id="calibration",
        training_pipeline_fingerprint="pipeline",
        training_corpus_fingerprint="corpus",
        training_query_set_digest="queries",
        training_fact_labels_digest="facts",
        training_artifact_sha256="artifact",
        training_source_rows=10,
        artifact_fingerprint="",
    ).with_fingerprint()


def _item(
    chunk_id: str,
    source: str,
    cosine: float,
    pool_rank: int,
    verdict: str = "ok",
) -> dict[str, object]:
    return {
        "chunk_id": chunk_id,
        "source": source,
        "ordinal": pool_rank,
        "pool_rank": pool_rank,
        "text": chunk_id,
        "cosine": cosine,
        "confidence": 0.8,
        "verdict": verdict,
    }


def test_source_model_round_trip_refuses_tampering() -> None:
    """The artifact fingerprint makes fitted coefficients immutable.

    Red proof node ``source-shadow-artifact-01`` bypasses the fingerprint comparison.
    The tampered coefficient assertion then fails at its intended refusal.
    """
    artifact = _artifact()
    loaded = SourceConditioningArtifact.from_json(artifact.to_json())
    assert loaded == artifact

    tampered = json.loads(artifact.to_json())
    tampered["coefficients"][0] += 0.1
    with pytest.raises(SourceConditioningArtifactError, match="fingerprint mismatch"):
        SourceConditioningArtifact.from_dict(tampered)


def test_production_feature_extraction_matches_registered_audit_features() -> None:
    """Production scoring uses the exact feature definition measured by the audit.

    Red proof node ``source-shadow-features-01`` reverses the source margin sign. The
    feature equality assertion then fails at its intended assertion.
    """
    pool = [
        _item("a1", "a", 0.8, 1),
        _item("b1", "b", 0.7, 2),
        _item("a2", "a", 0.6, 3),
    ]
    dense = [
        {"chunk_id": "a1", "rank": 1},
        {"chunk_id": "b1", "rank": 2},
        {"chunk_id": "a2", "rank": 3},
    ]
    sparse = [
        {"chunk_id": "b1", "rank": 1},
        {"chunk_id": "a2", "rank": 2},
    ]
    row = {
        "query_index": 0,
        "query": {"answerable": True, "relevant_files": ["a"]},
        "source_admission_audit": {"candidate_k": 20, "items": pool},
        "leg_audit": {"dense": dense, "sparse": sparse},
    }
    expected = {str(value["source"]): tuple(value["features"]) for value in _source_rows(row)}

    assert source_features(pool, dense, sparse) == expected


def test_source_model_selection_cannot_override_trust_or_raw_floor() -> None:
    """Source support never overrides the existing trust and plausibility boundaries.

    Red proof node ``source-shadow-boundary-01`` disables the verdict check. The
    selected chunk assertion then fails at its intended assertion.
    """
    artifact = _artifact()
    pool = [
        _item("invalid", "strong", 0.8, 1, "superseded"),
        _item("too-low", "strong", 0.29, 2, "low_confidence"),
        _item("promoted", "strong", 0.48, 3, "low_confidence"),
        _item("ok", "weak", 0.53, 4),
    ]
    dense = [
        {"chunk_id": item["chunk_id"], "rank": index}
        for index, item in enumerate(pool, start=1)
    ]

    selected = select_source_conditioned(
        artifact, pool, dense, [], threshold=0.49
    )

    assert [item["chunk_id"] for item in selected] == ["ok", "promoted"]


def test_source_model_compatibility_ignores_refresh_identity_but_not_pipeline() -> None:
    """Routine corpus refreshes are allowed but a feature pipeline change is refused.

    Red proof node ``source-shadow-compatibility-01`` bypasses the pipeline check. The
    incompatible artifact assertion then fails at its intended refusal.
    """
    artifact = _artifact()
    artifact.assert_compatible(
        pipeline_fingerprint="pipeline",
        embedding_profile="voyage-context-4-v1",
        retrieval_profile="fast",
        candidate_k=20,
    )

    with pytest.raises(SourceConditioningArtifactError, match="pipeline_fingerprint"):
        artifact.assert_compatible(
            pipeline_fingerprint="other",
            embedding_profile="voyage-context-4-v1",
            retrieval_profile="fast",
            candidate_k=20,
        )


def test_shadow_sampling_is_off_by_default_and_has_no_active_mode() -> None:
    """Source conditioning cannot become active through an environment typo.

    Red proof node ``source-shadow-mode-01`` accepts `active` as a shadow alias. The
    unsupported mode assertion then fails at its intended refusal.
    """
    assert service._source_conditioning_shadow_sampled("query", {}) is False
    assert (
        service._source_conditioning_shadow_sampled(
            "query",
            {
                "RECALL_SOURCE_CONDITIONING_MODE": "shadow",
                "RECALL_SOURCE_CONDITIONING_SHADOW_SAMPLE_RATE": "0",
            },
        )
        is False
    )
    assert (
        service._source_conditioning_shadow_sampled(
            "query",
            {
                "RECALL_SOURCE_CONDITIONING_MODE": "shadow",
                "RECALL_SOURCE_CONDITIONING_SHADOW_SAMPLE_RATE": "1",
            },
        )
        is True
    )
    with pytest.raises(SourceConditioningArtifactError, match="off or shadow"):
        service._source_conditioning_shadow_sampled(
            "query", {"RECALL_SOURCE_CONDITIONING_MODE": "active"}
        )


def _trusted_result() -> TrustedResult:
    chunk = Chunk("served", "stored/source", "served text", {"file": "source", "ord": 0})
    hit = TrustedHit(
        chunk=chunk,
        cosine=0.8,
        confidence=0.9,
        verdict="ok",
        provenance=Provenance(chunk.source, "source", 0, None),
        validity=Validity(None, None, None),
    )
    return TrustedResult(
        "query",
        [hit],
        False,
        "",
        False,
        StalenessReport(False, datetime.now(UTC), timedelta(0), timedelta(days=2)),
        calibration_id="calibration",
        calibration_status="certified",
        tenant_id="memory",
        generation_id="generation-new",
        pipeline_fingerprint="pipeline",
        corpus_fingerprint="corpus-new",
    )


def test_shadow_payload_contains_hashes_but_no_candidate_content(tmp_path, monkeypatch) -> None:
    """A diagnostic cannot expose unserved chunk identifiers, paths, or text.

    Red proof node ``source-shadow-redaction-01`` emits raw chunk identifiers. The
    payload content assertion then fails at its intended assertion.

    Red proof receipt ``source-shadow-baseline-link-01``: before the same vector validation
    receipt was added, the baseline hash assertion failed with a missing key. Targeted production
    symbol: ``_source_conditioning_shadow_payload``.
    """
    artifact_path = tmp_path / "source-model.json"
    artifact_path.write_text(_artifact().to_json(), encoding="utf-8")
    monkeypatch.setattr(service, "embedding_profile_id", lambda embedder: "voyage-context-4-v1")
    pool = [_item("private-chunk", "private/source.md", 0.8, 1)]
    leg = {"dense": [{"chunk_id": "private-chunk", "rank": 1}], "sparse": []}
    payload = service._source_conditioning_shadow_payload(
        artifact_path=str(artifact_path),
        leg_audit=leg,
        pool_audit={"candidate_k": 20, "threshold": 0.5, "items": pool},
        baseline=_trusted_result(),
        embedder=object(),
        profile=service.FAST_PROFILE,
    )
    serialized = json.dumps(payload)

    assert payload["status"] == "ok"
    assert payload["selected_count"] == 1
    assert payload["baseline_chunk_hashes"] == [chunk_identifier_hash("served")]
    assert "private-chunk" not in serialized
    assert "private/source.md" not in serialized


def test_tty_command_adds_source_shadow_only_when_requested(monkeypatch) -> None:
    """The validation process explicitly opts into the otherwise absent shadow.

    Red proof node ``source-shadow-command-01`` omits the shadow environment block. The
    audited command assertion then fails while the ordinary command remains unchanged.
    """
    monkeypatch.setenv("RECALL_BENCHMARK_REMOTE_CODE_ROOT", "/srv/recall")
    ordinary = _command(
        "memory", "voyage-context:voyage-context-4", "/srv/memory", "fast",
        "combined", "none", 1, 32, 0.10,
    )[-1]
    shadow = _command(
        "memory", "voyage-context:voyage-context-4", "/srv/memory", "fast",
        "combined", "none", 1, 32, 0.10,
        "generation",
        benchmark_retrieval_leg_audit=True,
        benchmark_source_admission_audit=True,
        source_conditioning_mode="shadow",
        source_conditioning_artifact="docs/model.json",
        source_conditioning_sample_rate=1.0,
    )[-1]

    assert "RECALL_SOURCE_CONDITIONING_MODE" not in ordinary
    assert "RECALL_SOURCE_CONDITIONING_MODE=shadow" in shadow
    assert "RECALL_SOURCE_CONDITIONING_ARTIFACT=docs/model.json" in shadow
    assert "RECALL_SOURCE_CONDITIONING_SHADOW_SAMPLE_RATE=1.000000" in shadow


def test_source_conditioning_settings_are_documented_and_validated() -> None:
    """The MCP configuration contract must expose and reject malformed shadow settings.

    Red proof receipt ``source-shadow-settings-01``: before the settings schema and parser were
    extended, the schema-name assertion below failed because all three production variables were
    absent. Targeted production symbols: ``ENVIRONMENT_SCHEMA`` and
    ``_validate_runtime_options``.
    """
    schema_names = {spec.name for spec in ENVIRONMENT_SCHEMA}
    assert {
        "RECALL_SOURCE_CONDITIONING_MODE",
        "RECALL_SOURCE_CONDITIONING_ARTIFACT",
        "RECALL_SOURCE_CONDITIONING_SHADOW_SAMPLE_RATE",
    } <= schema_names

    Settings.from_env(
        {
            "RECALL_SOURCE_CONDITIONING_MODE": "shadow",
            "RECALL_SOURCE_CONDITIONING_SHADOW_SAMPLE_RATE": "1",
        }
    )
    with pytest.raises(ValueError, match="RECALL_SOURCE_CONDITIONING_MODE"):
        Settings.from_env({"RECALL_SOURCE_CONDITIONING_MODE": "active"})
    with pytest.raises(ValueError, match="RECALL_SOURCE_CONDITIONING_SHADOW_SAMPLE_RATE"):
        Settings.from_env({"RECALL_SOURCE_CONDITIONING_SHADOW_SAMPLE_RATE": "1.01"})


def test_public_parity_projects_only_the_registered_serving_contract() -> None:
    """Score noise is outside parity, while identity, order, verdict, and decision are binding.

    Red proof receipt ``source-shadow-public-projection-01``: the first implementation returned
    the complete ``trusted_evidence`` object. Changing only ``cosine`` below then failed the first
    equality assertion. Targeted production symbol: ``_public_signature``.
    """
    left = {
        "outcome": "answer",
        "refusal_reason": None,
        "trust_state": "trusted",
        "generation_id": "generation",
        "calibration_id": "calibration",
        "pipeline_fingerprint": "pipeline",
        "corpus_fingerprint": "corpus",
        "trusted_evidence": {
            "decision": "answer",
            "reason_code": None,
            "decision_state": "supported",
            "failure_code": None,
            "items": [
                {"chunk_id": "a", "verdict": "ok", "cosine": 0.6},
                {"chunk_id": "b", "verdict": "ok", "cosine": 0.5},
            ],
        },
    }
    score_only_change = json.loads(json.dumps(left))
    score_only_change["trusted_evidence"]["items"][0]["cosine"] = 0.60000001
    reordered = json.loads(json.dumps(left))
    reordered["trusted_evidence"]["items"].reverse()

    assert _public_signature(left) == _public_signature(score_only_change)
    assert _public_signature(left) != _public_signature(reordered)


def test_shadow_timing_uses_performance_span_surface() -> None:
    """The result runner reads the request trace span emitted by ``PerformanceTrace``.

    Red proof receipt ``source-shadow-timing-01``: mutating ``_shadow_internal_ms`` to read
    ``stage_ms`` instead of ``spans_ms`` makes this node fail with ``99.0 != 12.5``. Targeted
    production symbol: ``_shadow_internal_ms``.
    """
    performance = {
        "spans_ms": {"source_conditioning_shadow_ms": 12.5},
        "stage_ms": {"source_conditioning_shadow_ms": 99.0},
    }

    assert _shadow_internal_ms(performance) == 12.5


def test_same_vector_decision_requires_baseline_linkage() -> None:
    """The same vector runner cannot build when its public baseline receipt is incomplete.

    Red proof receipt ``source-shadow-same-vector-gate-01``: mutating the baseline parity
    comparison in ``_decision`` from 50 to 49 makes this node return `BUILD SAMPLED SHADOW`
    instead of `REPAIR`. Targeted production symbol: ``_decision`` in the same vector runner.
    """
    summary = {
        "arms": {
            "baseline": {
                "complete_queries": 17,
                "covered_facts": 19,
                "unanswerable_answers": 1,
                "context_precision": 0.48,
            },
            "candidate": {
                "complete_queries": 18,
                "covered_facts": 20,
                "unanswerable_answers": 1,
                "context_precision": 0.61,
            },
        }
    }

    assert _same_vector_decision(
        summary,
        candidate_hash_parity_count=50,
        baseline_hash_parity_count=49,
        errors=0,
        timing_receipts=50,
        elapsed_ms=100_000.0,
    ) == "REPAIR"
