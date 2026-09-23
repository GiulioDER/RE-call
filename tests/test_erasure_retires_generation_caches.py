"""An erasure retires every in-process cache of a generation's content, with no restart.

Invariant: `GenerationManager.forget` deletes a source's rows from every live generation IN PLACE
and rewrites that generation's `corpus_fingerprint` in the same transaction, while the
generation id does not change. Every cache of a generation's content must therefore be keyed on
the fingerprint as well as the id, which is what `GenerationStore._serving_identity` supplies.

Failure mode caught: these caches were keyed on the generation id (plus, for the graph caches,
the graph marker, which an erasure leaves in place). A long-lived server kept applying the erased
source's supersession edges, kept serving its text from the reasoning graph projection, kept the
pre-erasure semantic graph, and for up to 30 seconds kept reporting a calibration CERTIFIED that
the erasure had invalidated.

Red proof, recorded 2026-09-22 against `origin/master` at `3cc57b81` (the pre-fix
`recall/generation_store.py`, `recall_mcp/graph_projection.py` and `recall_mcp/service.py`
restored into the worktree for the run). Each test failed in its final assertion:

* `test_supersession_closure_is_rescanned_after_an_erasure`: `assert 1 == 2`, the closure was
  served from the cache.
* `test_calibration_verdict_is_re_resolved_after_an_erasure`: `assert 1 == 2`, the verdict was
  served from the cache inside its TTL.
* `test_graph_readiness_is_re_read_after_an_erasure`: `assert 1 == 2`.
* `test_the_reasoning_projection_is_rebuilt_after_an_erasure`: `assert 1 == 2`.
* `test_the_semantic_graph_is_reloaded_after_an_erasure`: `assert 1 == 2`.

The end-to-end form against a real database is
`tests/test_calibration_v2.py::test_a_live_store_sees_an_erasure_invalidate_its_calibration`.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

import pytest

import recall_mcp.service as service
from recall.generation_store import GenerationStore
from recall.reasoning_graph import build_reasoning_graph
from recall.semantic_graph import GraphReadiness, SemanticGraphProjection
from recall_mcp.service import reasoning_projection


class _Rows:
    def __init__(self, rows):
        self.rows = rows

    def fetchall(self):
        return self.rows


class _StubGenerationStore(GenerationStore):
    """A real `GenerationStore` whose database is replaced by counters.

    `_serving_identity` is the real method: the fingerprint is pinned the way `snapshot()` pins
    it, so the tests exercise the production key construction rather than a stand-in for it.
    """

    def __init__(self) -> None:  # deliberately does not call super().__init__
        self._tenant = "acme"
        self._dsn = "postgresql://unused/never-connected"
        self._pinned_generation = ContextVar("pinned_generation", default="gen-1")
        self._pinned_corpus = ContextVar("pinned_corpus", default=("gen-1", "corpus-1"))
        self._fixed_generation = None
        self._supersession_cache = None
        self._supersession_scans = 0
        self._calibration_resolution = None
        self._graph_readiness_cache = None
        self.statements = 0

    def erase(self) -> None:
        """What `forget` does to a live generation: same id, new fingerprint."""
        self._pinned_corpus.set(("gen-1", "corpus-2"))

    def _with_retry(self, op):  # type: ignore[no-untyped-def]
        self.statements += 1
        return op(self)

    def execute(self, _sql, _params=None):  # type: ignore[no-untyped-def]
        return _Rows([("old.md", "new.md", None)])


def test_supersession_closure_is_rescanned_after_an_erasure() -> None:
    store = _StubGenerationStore()

    store.supersession_all()
    store.supersession_all()
    assert store._supersession_scans == 1

    store.erase()
    store.supersession_all()

    assert store._supersession_scans == 2


def test_calibration_verdict_is_re_resolved_after_an_erasure(monkeypatch) -> None:
    store = _StubGenerationStore()
    resolutions: list[str] = []

    def resolve(self, op):  # type: ignore[no-untyped-def]
        resolutions.append("resolved")
        return object()

    monkeypatch.setattr(_StubGenerationStore, "_with_retry", resolve)

    store.resolve_calibration()
    store.resolve_calibration()
    assert len(resolutions) == 1

    store.erase()
    store.resolve_calibration()

    assert len(resolutions) == 2


def test_graph_readiness_is_re_read_after_an_erasure(monkeypatch) -> None:
    store = _StubGenerationStore()
    reads: list[str] = []
    ready = GraphReadiness(
        ready=True,
        tenant_id="acme",
        generation_id="gen-1",
        graph_id="graph-1",
        graph_fingerprint="graph-a",
        entity_count=1,
        mention_count=1,
        relation_count=1,
        diagnostic_count=0,
    )

    def read(_connection, _tenant, generation):  # type: ignore[no-untyped-def]
        reads.append(generation)
        return ready

    monkeypatch.setattr("recall.generation_store.read_graph_readiness", read)

    store.graph_readiness()
    store.graph_readiness()
    assert len(reads) == 1

    store.erase()
    store.graph_readiness()

    assert len(reads) == 2


class _Readiness:
    ready = True
    graph_fingerprint = "graph-a"


class _ErasableGenerationStore:
    """The duck-typed store the MCP projection cache reads, with an erasable corpus."""

    tenant = "acme"

    def __init__(self) -> None:
        self.corpus = "corpus-1"
        self.semantic_loads = 0

    def active_generation_id(self) -> str:
        return "gen-1"

    @contextmanager
    def snapshot(self):  # type: ignore[no-untyped-def]
        yield "gen-1"

    def graph_readiness(self) -> _Readiness:
        return _Readiness()

    def _serving_identity(self, generation_id=None):  # type: ignore[no-untyped-def]
        return generation_id or "gen-1", self.corpus

    def load_semantic_graph(self, generation_id: str) -> SemanticGraphProjection:
        self.semantic_loads += 1
        return SemanticGraphProjection(
            schema_version=2,
            graph_id="graph-1",
            tenant_id="acme",
            generation_id=generation_id,
            pipeline_fingerprint=None,
            corpus_fingerprint=None,
            entities=(),
            mentions=(),
            relations=(),
            diagnostics=(),
        )


@pytest.fixture
def fresh_graph_caches():
    service._reset_graph_projection_cache()
    yield
    service._reset_graph_projection_cache()


def test_the_reasoning_projection_is_rebuilt_after_an_erasure(
    fresh_graph_caches, monkeypatch
) -> None:
    projections: list[str] = []

    def project(store, *, include_text=False, **_kwargs):  # type: ignore[no-untyped-def]
        projections.append(store.corpus)
        return build_reasoning_graph(
            [], tenant_id="acme", generation_id="gen-1", include_text=include_text
        )

    monkeypatch.setattr(service, "project_store_graph", project)
    store = _ErasableGenerationStore()

    reasoning_projection(store, include_text=True)
    reasoning_projection(store, include_text=True)
    assert len(projections) == 1

    store.corpus = "corpus-2"
    reasoning_projection(store, include_text=True)

    assert len(projections) == 2


def test_the_semantic_graph_is_reloaded_after_an_erasure(fresh_graph_caches) -> None:
    store = _ErasableGenerationStore()
    readiness = type("Readiness", (), {"graph_fingerprint": None})()

    service._cached_semantic_graph(store, "gen-1", readiness, "policy")
    service._cached_semantic_graph(store, "gen-1", readiness, "policy")
    assert store.semantic_loads == 1

    store.corpus = "corpus-2"
    service._cached_semantic_graph(store, "gen-1", readiness, "policy")

    assert store.semantic_loads == 2
