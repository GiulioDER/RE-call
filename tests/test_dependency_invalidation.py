"""Pure dependency projection regression coverage."""

from __future__ import annotations

from recall.dependency_invalidation import build_dependency_projection
from recall.types import Chunk


def _chunk(
    source: str,
    *,
    authority: str = "tool_observation",
    depends_on: list[str] | None = None,
) -> Chunk:
    graph: dict[str, object] = {"authority": authority}
    if depends_on is not None:
        graph["depends_on"] = depends_on
    return Chunk(source, source, source, {"file": source, "recall_graph": graph})


def test_direct_and_transitive_invalidations_are_retained() -> None:
    projection = build_dependency_projection(
        [_chunk("dependent.md", depends_on=["prerequisite.md"]), _chunk("prerequisite.md")],
        tenant_id="tenant",
        generation_id="generation",
        base_states={"dependent.md": "current", "prerequisite.md": "expired"},
    )

    assert projection.reason_for("prerequisite.md") is not None
    reason = projection.reason_for("dependent.md")
    assert reason is not None
    assert reason.dependency == "prerequisite.md"
    assert reason.cause == "expired"
    assert reason.path == ("dependent.md", "prerequisite.md")


def test_model_inference_is_not_invalidated_by_temporal_state() -> None:
    projection = build_dependency_projection(
        [_chunk("inference.md", authority="model_inference")],
        tenant_id="tenant",
        generation_id="generation",
        base_states={"inference.md": "expired"},
    )
