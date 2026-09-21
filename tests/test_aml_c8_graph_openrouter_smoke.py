"""Contract proofs for the live C8 OpenRouter graph verifier."""

from __future__ import annotations

from pathlib import Path

from scripts.aml_c8_graph_openrouter_smoke import Call, verify


class _Client:
    def __init__(self) -> None:
        self.paths: list[str] = []

    def call(self, path, payload=None):
        self.paths.append(path)
        responses = {
            "/version": Call(
                200,
                {
                    "variant": "C8_routed_specialists_grounded_graph",
                    "generation_provider": "openrouter",
                    "generation_model": "openai/gpt-4o-mini",
                    "active_components": {"graph_sidecar": True},
                },
                {},
            ),
            "/v1/add": Call(200, {"compiler_fallback": False, "compiled_count": 1}, {}),
            "/v1/corpus/status": Call(
                200, {"authored_relation_count": 1, "eligible_relation_count": 1}, {}
            ),
            "/v1/search": Call(
                200,
                {"data": [{"id": "raw_a"}]},
                {
                    "X-Recall-Graph-Attempted": "1",
                    "X-Recall-Graph-Fallback": "0",
                    "X-Recall-Graph-Relation-Hits": "1",
                },
            ),
            "/v1/delete": Call(200, {"deleted": 2}, {}),
        }
        return responses[path]


def test_c8_graph_verifier_requires_real_compiler_relation_and_search_hit() -> None:
    """The verifier fails if any provider, relation, or graph hit condition is removed."""
    client = _Client()
    result = verify(client, clock=lambda: 1_726_133_200)

    assert result["passed"] is True
    assert result["model"] == "openai/gpt-4o-mini"
    assert result["checks"]["live_compiler_created_records"] is True
    assert result["checks"]["grounded_relations_persisted"] is True
    assert result["checks"]["graph_search_used_relation"] is True
    assert client.paths == [
        "/version",
        "/v1/add",
        "/v1/corpus/status",
        "/v1/search",
        "/v1/delete",
    ]


def test_c8_vps_setup_activates_only_the_generation_bound_atomic_contract() -> None:
    script = (Path(__file__).parents[1] / "scripts" / "aml_experience_vps2_setup.sh").read_text(
        encoding="utf-8"
    )

    assert '"$selected_variant" == "C8_routed_specialists_grounded_graph"' in script
    assert "RECALL_ATOMIC_RESCUE_MODE=active" in script
    assert "RECALL_AML_ATOMIC_RESCUE_CALIBRATION_ID=aml-c8-atomic-rescue-v1" in script
    assert (
        "RECALL_AML_ATOMIC_RESCUE_PIPELINE_FINGERPRINT="
        "aml-c8-routed-specialists-grounded-graph-v1"
    ) in script
