"""Focused safety and scoring proofs for the frozen live C7 qualification driver."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import aml_c7_qualification as qualification


def _version(variant: str) -> qualification.Call:
    payload = {
        "variant": variant,
        "embedding_profile": "voyage-code-4-v1",
        "exact_dense": True,
        "ordering_profile": "source-session-c-collation-segment-v1",
        "window_renderer_profile": "message-content-only-v1",
        "embedding_call_lock": True,
    }
    if variant == qualification.C7_VARIANT:
        payload.update(
            {
                "context_embedding_profile": "voyage-context-4-v1",
                "multimodal_embedding_profile": "voyage-multimodal-3.5-v1",
                "multimodal_embedding_model": "voyage-multimodal-3.5",
                "specialist_router_profile": "conservative-specialist-router-v1",
                "specialist_fusion_profile": "routed-rank-fusion-v1",
                "rrf_constant": 60,
            }
        )
    return qualification.Call(200, payload, {})


def test_endpoint_preflight_refuses_any_variant_other_than_exact_c6_and_c7() -> None:
    """A mislabeled live endpoint must be refused before any corpus Add.

    Red proof: mutating ``validate_endpoint_versions`` to accept any HTTP 200 makes the final
    assertion fail because no ``QualificationRefusal`` is raised. The protected symbol is
    ``scripts.aml_c7_qualification.validate_endpoint_versions``.
    """
    checks = qualification.validate_endpoint_versions(
        _version(qualification.C6_VARIANT), _version(qualification.C7_VARIANT)
    )
    assert checks == {"c6_frozen_shape": True, "c7_frozen_shape": True}

    with pytest.raises(qualification.QualificationRefusal) as error:
        qualification.validate_endpoint_versions(
            _version("C5_code4_bm25"), _version(qualification.C7_VARIANT)
        )
    assert error.value.code == "c6_endpoint_is_not_exact_variant"

    with pytest.raises(qualification.QualificationRefusal) as error:
        qualification.validate_endpoint_versions(
            _version(qualification.C6_VARIANT), _version("C7_experimental")
        )
    assert error.value.code == "c7_endpoint_is_not_exact_variant"


def test_distinct_endpoint_credentials_are_used_without_entering_the_receipt() -> None:
    """Dedicated C6 and C7 keys must remain distinct and secret safe.

    Red proof: mutating the C7 selection to read ``c6_name`` makes the equality assertion fail.
    The protected symbol is ``scripts.aml_c7_qualification.resolve_credentials``.
    """
    c6, c7, summary = qualification.resolve_credentials(
        {
            "C6_KEY": "c6-secret-value",
            "C7_KEY": "c7-secret-value",
            "SHARED": "shared-secret-value",
        },
        c6_name="C6_KEY",
        c7_name="C7_KEY",
        shared_name="SHARED",
    )

    assert (c6, c7) == ("c6-secret-value", "c7-secret-value")
    rendered = json.dumps(summary, sort_keys=True)
    assert summary["keys_distinct"] is True
    assert "secret-value" not in rendered


def test_dedicated_credentials_with_the_same_value_are_refused() -> None:
    """Two dedicated variables carrying one key do not satisfy endpoint key separation.

    Red proof: deleting the dedicated key equality refusal makes the expected exception absent.
    The protected symbol is ``scripts.aml_c7_qualification.resolve_credentials``.
    """
    with pytest.raises(qualification.QualificationRefusal) as error:
        qualification.resolve_credentials(
            {"C6_KEY": "same", "C7_KEY": "same"},
            c6_name="C6_KEY",
            c7_name="C7_KEY",
            shared_name="SHARED",
        )
    assert error.value.code == "dedicated_keys_not_distinct"


class _RankingClient:
    def __init__(self, *, mutate_last: bool = False) -> None:
        self.mutate_last = mutate_last

    def call(self, path: str, payload: dict | None = None) -> qualification.Call:
        assert path == "/v1/search"
        assert payload is not None
        ids = [f"chunk-{index:03d}" for index in range(100)]
        if self.mutate_last:
            ids[-1] = "changed"
        data = [
            {
                "id": chunk_id,
                "session_id": "sessions/task-a/p01.jsonl" if index == 0 else "noise",
            }
            for index, chunk_id in enumerate(ids)
        ]
        return qualification.Call(
            200,
            {"data": data},
            {"x-recall-variant": (
                qualification.C7_VARIANT
                if str(payload["user_id"]).startswith("c7")
                else qualification.C6_VARIANT
            )},
        )


def test_rank_comparison_requires_exact_ordered_top100_and_scores_gold_source() -> None:
    """Parity must cover all 100 positions while recall and MRR use source session identity.

    Red proof: mutating the equality to compare only ``[:10]`` makes the changed rank 100 fixture
    report one exact query instead of zero. The protected symbol is
    ``scripts.aml_c7_qualification.compare_coding_rankings``.
    """
    corpus = qualification.FrozenCorpus(
        Path("corpus"),
        {},
        {},
        (("task-a", "Fix parser.py with pytest", frozenset({"sessions/task-a/p01.jsonl"})),),
    )
    metrics, responses = qualification.compare_coding_rankings(
        corpus,
        _RankingClient(),
        _RankingClient(mutate_last=True),
        "c6-user",
        "c7-user",
    )

    assert metrics == {
        "queries": 1,
        "exact_top100": 0,
        "mismatch_task_ids": ["task-a"],
        "source_recall_at_10_count": 1,
        "mean_reciprocal_rank": 1.0,
    }
    assert len(responses) == 2


def test_query_receipt_rejects_raw_cross_model_vector_fields() -> None:
    """Gate 9 scans nested response keys while allowing the ordinary fused score field.

    Red proof: mutating the forbidden key expression to exclude ``embedding`` makes the exact
    key list assertion fail. The protected symbol is
    ``scripts.aml_c7_qualification.forbidden_response_keys``.
    """
    assert qualification.forbidden_response_keys(
        {"data": [{"id": "one", "score": 0.2, "detail": {"embedding_vector": [1.0]}}]}
    ) == ["embedding_vector"]
    assert qualification.forbidden_response_keys(
        {"data": [{"id": "one", "score": 0.2}]}
    ) == []


def test_unexpected_failure_receipt_does_not_serialize_secrets(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Unexpected exceptions expose only their class, never their message or credentials.

    Red proof: mutating the terminal receipt to include ``str(exc)`` leaks all three planted
    secret values and fails the final assertions. The protected symbol is
    ``scripts.aml_c7_qualification.main``.
    """
    monkeypatch.setenv("RECALL_AML_C6_API_KEY", "c6-very-secret")
    monkeypatch.setenv("RECALL_AML_C7_API_KEY", "c7-very-secret")
    monkeypatch.setenv("RECALL_AML_DATABASE_URL", "postgresql://db-very-secret")

    def fail(**_kwargs: object) -> dict[str, object]:
        raise RuntimeError(
            "c6-very-secret c7-very-secret postgresql://db-very-secret"
        )

    monkeypatch.setattr(qualification, "run_qualification", fail)
    monkeypatch.setattr(
        "sys.argv",
        [
            "aml_c7_qualification.py",
            "--amb-root",
            str(tmp_path),
            "--c6-url",
            "https://c6.invalid",
            "--c7-url",
            "https://c7.invalid",
            "--execute-live-qualification",
        ],
    )
    with pytest.raises(SystemExit) as error:
        qualification.main()
    assert error.value.code == 1
    output = capsys.readouterr().out
    assert "c6-very-secret" not in output
    assert "c7-very-secret" not in output
    assert "db-very-secret" not in output
    assert json.loads(output)["error_class"] == "RuntimeError"
