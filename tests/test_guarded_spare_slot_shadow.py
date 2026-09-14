from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from recall.calibration import Calibration
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
    chunk_identifier_hash,
    fill_source_conditioned_spare_slots,
    fill_strict_source_conditioned_spare_slot,
)
from recall.retriever import RetrievalCandidateTrace
from recall.types import Chunk, Provenance, RetrievalResult, ScoredChunk, TrustedHit
from recall_mcp import service
from recall_mcp.settings import (
    ENVIRONMENT_SCHEMA,
    Settings,
    activate_runtime_settings,
    reset_runtime_settings,
)
from tests.test_source_conditioning import _trusted_result
from scripts.run_live_tty_graph_precision import _command


def _artifact() -> SourceConditioningArtifact:
    return SourceConditioningArtifact(
        schema_version=SOURCE_CONDITIONING_SCHEMA_VERSION,
        model_id=SOURCE_CONDITIONING_MODEL_ID,
        feature_names=SOURCE_FEATURE_NAMES,
        means=(0.0,) * len(SOURCE_FEATURE_NAMES),
        scales=(1.0,) * len(SOURCE_FEATURE_NAMES),
        intercept=0.0,
        coefficients=(0.0,) * len(SOURCE_FEATURE_NAMES),
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
        training_artifact_sha256="training",
        training_source_rows=1,
        artifact_fingerprint="",
    ).with_fingerprint()


def _item(source: str, rank: int, cosine: float) -> dict[str, object]:
    return {
        "chunk_id": f"{source}-{rank}",
        "source": source,
        "ordinal": rank,
        "pool_rank": rank,
        "rank": rank,
        "text": f"private {source} evidence",
        "cosine": cosine,
        "confidence": 0.5,
        "verdict": "ok",
    }


def test_guarded_rescue_preserves_the_complete_base_prefix_and_budget() -> None:
    """A rescue may fill a spare slot but cannot displace or overfill base evidence.

    Red proof receipt ``guarded-spare-slot-budget-01`` targets
    ``fill_source_conditioned_spare_slots``. Mutating its inner budget guard from ``>=`` to ``>``
    adds a sixth item and fails the intended five-item assertion.
    """
    base = [_item("base", rank, 0.5) for rank in range(1, 5)]
    first = _item("rescue", 1, 0.40)
    second = _item("rescue-two", 2, 0.39)

    selected, receipts = fill_source_conditioned_spare_slots(
        _artifact(), base, [first, second], [first, second], [first, second]
    )

    assert selected == [*base, first]
    assert len(selected) == REGISTERED_ITEM_BUDGET
    assert len(receipts) == 1


def test_strict_guarded_spare_slot_applies_registered_first_addition_rule() -> None:
    """The strict policy admits at most the first guarded proposal when every gate passes.

    Red proof: the test initially failed because the strict selector did not exist.
    """
    artifact = _artifact()
    base = [_item("base", 1, 0.8)]
    accepted = _item("accepted", 1, 0.7)
    later = _item("later", 2, 0.65)
    pool = [*base, accepted, later]
    dense = [accepted, later]
    sparse = [accepted, later]

    selected, receipts = fill_strict_source_conditioned_spare_slot(
        artifact, base, pool, dense, sparse
    )

    assert [item["chunk_id"] for item in selected] == [
        base[0]["chunk_id"],
        accepted["chunk_id"],
    ]
    assert len(receipts) == 1
    assert receipts[0]["policy"] == "extractive_strict_v1"


def test_strict_guarded_spare_slot_does_not_substitute_after_first_gate_failure() -> None:
    """No later candidate may substitute when the first guarded proposal fails a gate."""
    artifact = _artifact()
    base = [_item("base", 1, 0.8)]
    first = _item("first", 1, 0.7)
    later = _item("later", 2, 0.65)
    pool = [*base, first, later]
    dense = [
        {**first, "rank": 1},
        {**later, "rank": 5},
    ]
    sparse = [
        {**first, "rank": 3},
        {**later, "rank": 1},
    ]

    selected, receipts = fill_strict_source_conditioned_spare_slot(
        artifact, base, pool, dense, sparse
    )

    assert selected == base
    assert receipts == []


def test_guarded_shadow_payload_is_private_and_links_base_to_additions(
    tmp_path, monkeypatch
) -> None:
    """The production diagnostic exposes hashes and aggregates, never candidate content.

    Red proof receipt ``guarded-spare-slot-redaction-01`` targets
    ``_source_conditioning_shadow_payload``. Returning a raw rescue receipt makes the serialized
    payload fail the source and text assertions below.
    """
    artifact_path = tmp_path / "source-model.json"
    artifact_path.write_text(_artifact().to_json(), encoding="utf-8")
    monkeypatch.setattr(service, "embedding_profile_id", lambda embedder: "voyage-context-4-v1")
    rescue = _item("private/source.md", 1, 0.40)
    payload = service._source_conditioning_shadow_payload(
        artifact_path=str(artifact_path),
        leg_audit={"dense": [rescue], "sparse": [rescue]},
        pool_audit={"candidate_k": 20, "threshold": 0.5, "items": [rescue]},
        baseline=_trusted_result(),
        embedder=object(),
        profile=service.FAST_PROFILE,
        policy="guarded_spare_slot",
    )
    serialized = json.dumps(payload)

    assert payload["policy"] == "guarded_spare_slot"
    assert payload["alpha008_selected_count"] == 0
    assert payload["added_count"] == 1
    assert payload["base_prefix_preserved"] is True
    assert payload["lane_counts"] == {"dual_leg": 1, "lexical_dominant": 0}
    assert payload["added_chunk_hashes"] == [chunk_identifier_hash(rescue["chunk_id"])]
    assert "private/source.md" not in serialized
    assert "private private/source.md evidence" not in serialized


def test_guarded_shadow_policy_is_documented_and_rejects_unknown_values() -> None:
    """Only the two registered shadow policies are accepted.

    Red proof receipt ``guarded-spare-slot-settings-01`` targets
    ``_validate_runtime_options``. Temporarily admitting ``replace_base`` makes the final
    refusal assertion fail because no exception is raised.
    """
    schema_names = {spec.name for spec in ENVIRONMENT_SCHEMA}
    assert "RECALL_SOURCE_CONDITIONING_SHADOW_POLICY" in schema_names
    Settings.from_env({"RECALL_SOURCE_CONDITIONING_SHADOW_POLICY": "guarded_spare_slot"})
    with pytest.raises(ValueError, match="RECALL_SOURCE_CONDITIONING_SHADOW_POLICY"):
        Settings.from_env({"RECALL_SOURCE_CONDITIONING_SHADOW_POLICY": "replace_base"})


def test_guarded_shadow_command_sets_the_registered_policy_only_when_enabled(monkeypatch) -> None:
    """A live screen can select the guarded policy without changing ordinary launches.

    Red proof receipt ``guarded-spare-slot-command-01`` targets ``_command``. Temporarily
    omitting the policy assignment makes the guarded command assertion fail while the ordinary
    command remains unchanged.
    """
    monkeypatch.setenv("RECALL_BENCHMARK_REMOTE_CODE_ROOT", "/srv/recall")
    ordinary = _command(
        "memory", "voyage-context:voyage-context-4", "/srv/memory", "fast",
        "combined", "none", 1, 32, 0.10,
    )[-1]
    guarded = _command(
        "memory", "voyage-context:voyage-context-4", "/srv/memory", "fast",
        "combined", "none", 1, 32, 0.10,
        source_conditioning_mode="shadow",
        source_conditioning_artifact="docs/model.json",
        source_conditioning_sample_rate=1.0,
        source_conditioning_policy="guarded_spare_slot",
    )[-1]

    assert "RECALL_SOURCE_CONDITIONING_SHADOW_POLICY" not in ordinary
    assert "RECALL_SOURCE_CONDITIONING_SHADOW_POLICY=guarded_spare_slot" in guarded


def test_guarded_shadow_reuses_the_main_request_trace_and_preserves_public_evidence(
    tmp_path, monkeypatch
) -> None:
    """The guarded shadow adds no provider call and cannot alter served evidence.

    Red proof receipt ``guarded-spare-slot-reuse-01`` targets the shadow consumer in
    ``_execute_reasoning_query``. A plausible mutation that calls
    ``_retrieval_leg_benchmark_audit_payload`` from the guarded path fails at the injected
    assertion before a response is returned. Restoring trace reuse makes the same node green.
    """
    artifact_path = tmp_path / "source-model.json"
    artifact_path.write_text(_artifact().to_json(), encoding="utf-8")
    baseline = _trusted_result()
    served = baseline.hits[0]
    rescue_chunk = Chunk(
        "rescue", "stored/private", "private rescue text", {"file": "private", "ord": 1}
    )
    rescue_scored = ScoredChunk(rescue_chunk, 0.40)
    rescue_trusted = TrustedHit(
        chunk=rescue_chunk,
        cosine=0.40,
        confidence=0.2,
        verdict="low_confidence",
        provenance=Provenance(rescue_chunk.source, "private", 1, None),
        validity=served.validity,
    )
    raw_hits = [ScoredChunk(served.chunk, served.cosine), rescue_scored]
    raw = RetrievalResult(baseline.query, raw_hits, False, baseline.staleness)
    traced = baseline.__class__(
        baseline.query,
        [served, rescue_trusted],
        baseline.abstained,
        baseline.reason,
        baseline.gap_warning,
        baseline.staleness,
        calibration_id=baseline.calibration_id,
        calibration_status=baseline.calibration_status,
        tenant_id=baseline.tenant_id,
        generation_id=baseline.generation_id,
        pipeline_fingerprint=baseline.pipeline_fingerprint,
        corpus_fingerprint=baseline.corpus_fingerprint,
    )
    candidate_trace = (
        RetrievalCandidateTrace(raw, tuple(raw_hits), tuple(raw_hits), tuple()),
        traced,
        Calibration("test", 0.5, 0.05),
    )
    calls = {name: 0 for name in ("embed", "dense", "sparse", "rerank", "trust")}

    def fake_retrieve(*_args, **_kwargs):
        for name in calls:
            calls[name] += 1
        return SimpleNamespace(
            result=baseline,
            query_vector=[1.0, 0.0],
            profile=service.FAST_PROFILE,
            candidate_trace=candidate_trace,
        )

    class Store:
        tenant = "memory"
        generation_id = "generation-new"

    monkeypatch.setattr(service, "_retrieve_trusted", fake_retrieve)
    monkeypatch.setattr(service, "embedding_profile_id", lambda _embedder: "voyage-context-4-v1")
    monkeypatch.setattr(
        service,
        "_retrieval_leg_benchmark_audit_payload",
        lambda *_args, **_kwargs: pytest.fail("guarded shadow repeated dense or sparse retrieval"),
    )
    monkeypatch.setattr(
        service,
        "_source_admission_benchmark_audit_payload",
        lambda *_args, **_kwargs: pytest.fail("guarded shadow repeated reranking or trust"),
    )

    off_settings = Settings.from_env({"RECALL_SOURCE_CONDITIONING_MODE": "off"})
    token = activate_runtime_settings(off_settings)
    try:
        off = service.reasoning_query(
            Store(), object(), baseline.query, mode="retrieval_only", graph_expansion="off",
            policy=service.TrustPolicy.development(),
        )
    finally:
        reset_runtime_settings(token)
    assert calls == {name: 1 for name in calls}

    guarded_settings = Settings.from_env(
        {
            "RECALL_SOURCE_CONDITIONING_MODE": "shadow",
            "RECALL_SOURCE_CONDITIONING_ARTIFACT": str(artifact_path),
            "RECALL_SOURCE_CONDITIONING_SHADOW_SAMPLE_RATE": "1",
            "RECALL_SOURCE_CONDITIONING_SHADOW_POLICY": "guarded_spare_slot",
        }
    )
    token = activate_runtime_settings(guarded_settings)
    try:
        guarded = service.reasoning_query(
            Store(), object(), baseline.query, mode="retrieval_only", graph_expansion="off",
            policy=service.TrustPolicy.development(),
        )
    finally:
        reset_runtime_settings(token)

    assert calls == {name: 2 for name in calls}
    assert guarded.trusted_evidence == off.trusted_evidence
    values = guarded.diagnostics.performance["values"]
    assert isinstance(values, dict)
    shadow = values["source_conditioning_shadow"]
    assert isinstance(shadow, dict)
    assert shadow["status"] == "ok"
    assert shadow["added_count"] == 1
    spans = guarded.diagnostics.performance["spans_ms"]
    assert isinstance(spans, dict)
    assert "source_conditioning_shadow_ms" in spans
