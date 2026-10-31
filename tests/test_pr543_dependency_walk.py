"""Bounded dependency traversal and inferred-cycle regression proof."""
from __future__ import annotations

import pytest

from recall.dependency_invalidation import build_dependency_projection
from recall.types import Chunk


def _chunk(name, authority="tool_observation", depends_on=()):
    return Chunk(name, name, name, {"file": name, "recall_graph": {
        "authority": authority, "depends_on": list(depends_on),
    }})


@pytest.mark.parametrize("missing", [False, True])
def test_inference_cycle_keeps_external_prerequisites(missing):
    """STAKES-001: skip only the inferred backedge, retaining other prerequisites.

    Baseline 76a0b36ea, build_dependency_projection.walk returns None at the
    inferred cycle and fails the 'hid an external prerequisite' assertion.
    The pure inferred-cycle assertion is a compatibility control in this proof.
    """
    cycle = [_chunk("b.md", "model_inference", ["c.md"]),
             _chunk("c.md", "model_inference", ["b.md"])]
    assert not build_dependency_projection(cycle, tenant_id="t", generation_id="g").invalidations
    chunks = [_chunk("a.md", "policy", ["b.md"]), cycle[0],
              _chunk("c.md", "model_inference", ["b.md", "z.md"])]
    if not missing:
        chunks.append(_chunk("z.md"))
    projection = build_dependency_projection(chunks, tenant_id="t", generation_id="g",
                                             base_states={"z.md": "expired"})
    for source in ("a.md", "b.md", "c.md"):
        reason = projection.reason_for(source)
        assert reason is not None, "ignored inference cycle hid an external prerequisite"
        assert reason.dependency == "z.md"
        assert reason.cause == ("unresolved" if missing else "expired")


def test_healthy_dag_revisits_are_bounded():
    """PERF-001: count graph visits without a wall-time benchmark.

    Baseline 76a0b36ea, build_dependency_projection.walk revisits shared healthy
    descendants exponentially and fails the state-read bound below.
    """
    class CountedStates(dict):
        calls = 0

        def get(self, key, default=None):
            self.calls += 1
            return super().get(key, default)

    layers = 12
    chunks = [_chunk(f"n{level:02}-{side}.md", depends_on=(
        [f"n{level+1:02}-0.md", f"n{level+1:02}-1.md"] if level + 1 < layers else []
    )) for level in range(layers) for side in range(2)]
    states = CountedStates({chunk.source: "current" for chunk in chunks})
    projection = build_dependency_projection(chunks, tenant_id="t", generation_id="g", base_states=states)
    assert not projection.invalidations
    assert states.calls <= len(chunks) * layers * 3, "shared healthy prerequisites were traversed exponentially"
