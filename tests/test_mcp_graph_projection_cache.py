"""The service caches the reasoning graph projection per (tenant, generation, include_text).

`project_store_graph` streams every chunk of the generation and rebuilds the whole graph, and
five tool paths in `recall_mcp.service` ask for it per request. The projection is deterministic
in that key and a generation is immutable once active, so the second identical request must be
answered from the process cache, and a promotion (a new active generation id) must bust it.

The spy replaces `recall_mcp.service.project_store_graph`, which is the module's own reference
and therefore exactly what the cache guards.
"""

from __future__ import annotations

from contextlib import contextmanager

import pytest

import recall_mcp.service as service
from recall.reasoning_graph import build_reasoning_graph
from recall_mcp.service import reasoning_projection


class _Readiness:
    """The readiness row the cache key reads its graph fingerprint from."""

    def __init__(self, graph_fingerprint: str | None) -> None:
        self.ready = graph_fingerprint is not None
        self.graph_fingerprint = graph_fingerprint


class _FakeGenerationStore:
    """Just enough store for the cache: a tenant, an active generation, and a graph row."""

    def __init__(
        self,
        tenant: str = "acme",
        active: str = "gen-1",
        graph_fingerprint: str | None = "graph-a",
    ) -> None:
        self.tenant = tenant
        self.active = active
        self.graph_fingerprint = graph_fingerprint
        self.lookups = 0

    def active_generation_id(self) -> str:
        self.lookups += 1
        return self.active

    def graph_readiness(self) -> _Readiness:
        return _Readiness(self.graph_fingerprint)


class _LegacyStore:
    """No `active_generation_id`: a mutable corpus that must never be cached."""

    tenant = "acme"


@pytest.fixture(autouse=True)
def _fresh_cache():
    # Resolved defensively: the reset helper arrives with the cache, so against the pre-fix
    # module these tests must still reach their assertions and fail on the behaviour (the
    # projector runs twice) rather than erroring in fixture setup.
    reset = getattr(service, "_reset_graph_projection_cache", None)
    if reset is not None:
        reset()
    yield
    if reset is not None:
        reset()


@pytest.fixture
def projector_spy(monkeypatch):
    calls: list[tuple[str, str, bool]] = []

    def _spy(store, *, include_text=False, **kwargs):
        generation_id = (
            store.active_generation_id()
            if callable(getattr(store, "active_generation_id", None))
            else "legacy"
        )
        calls.append((store.tenant, generation_id, include_text))
        return build_reasoning_graph(
            [],
            tenant_id=store.tenant,
            generation_id=generation_id,
            include_text=include_text,
        )

    monkeypatch.setattr(service, "project_store_graph", _spy)
    return calls


def test_a_second_call_with_the_same_generation_does_not_reproject(projector_spy) -> None:
    store = _FakeGenerationStore()

    first = reasoning_projection(store, include_text=True)
    second = reasoning_projection(store, include_text=True)

    assert len(projector_spy) == 1
    assert second.generation_id == first.generation_id == "gen-1"


def test_a_changed_active_generation_busts_the_cache(projector_spy) -> None:
    store = _FakeGenerationStore()

    reasoning_projection(store, include_text=True)
    store.active = "gen-2"
    result = reasoning_projection(store, include_text=True)

    assert len(projector_spy) == 2
    assert result.generation_id == "gen-2"
    # And the new generation is itself served from the cache afterwards.
    reasoning_projection(store, include_text=True)
    assert len(projector_spy) == 2


def test_include_text_variants_are_distinct_entries(projector_spy) -> None:
    # Imported inside the test: the uncached inner helper does not exist before the fix, so a
    # module level import would break collection instead of failing the assertion.
    from recall_mcp.service import _store_graph

    store = _FakeGenerationStore()

    _store_graph(store, include_text=False)
    _store_graph(store, include_text=True)
    _store_graph(store, include_text=False)
    _store_graph(store, include_text=True)

    assert projector_spy == [("acme", "gen-1", False), ("acme", "gen-1", True)]


def test_a_legacy_store_is_projected_fresh_every_call(projector_spy) -> None:
    # Imported inside the test: the uncached inner helper does not exist before the fix, so a
    # module level import would break collection instead of failing the assertion.
    from recall_mcp.service import _store_graph

    store = _LegacyStore()

    _store_graph(store, include_text=True)
    _store_graph(store, include_text=True)

    assert len(projector_spy) == 2


def test_a_projection_of_another_generation_is_served_but_never_cached(monkeypatch) -> None:
    """A pinned snapshot or a promotion mid flight can project a generation other than the one
    the key names; caching it would answer future requests for the WRONG generation."""
    # Imported inside the test: the uncached inner helper does not exist before the fix, so a
    # module level import would break collection instead of failing the assertion.
    from recall_mcp.service import _store_graph

    store = _FakeGenerationStore(active="gen-2")
    calls: list[str] = []

    def _stale_projector(inner_store, *, include_text=False, **kwargs):
        calls.append(inner_store.tenant)
        return build_reasoning_graph(
            [], tenant_id=inner_store.tenant, generation_id="gen-1", include_text=include_text
        )

    monkeypatch.setattr(service, "project_store_graph", _stale_projector)

    graph = _store_graph(store, include_text=True)
    assert graph.generation_id == "gen-1"
    assert service._GRAPH_PROJECTIONS == {}
    _store_graph(store, include_text=True)
    assert len(calls) == 2


def test_the_cache_is_bounded_and_evicts_the_oldest_entry(projector_spy) -> None:
    # Imported inside the test: the uncached inner helper does not exist before the fix, so a
    # module level import would break collection instead of failing the assertion.
    from recall_mcp.service import _store_graph

    for index in range(6):
        _store_graph(_FakeGenerationStore(tenant=f"tenant-{index}"), include_text=True)

    assert len(service._GRAPH_PROJECTIONS) == service._GRAPH_PROJECTION_CACHE_MAX
    # The oldest tenants were evicted, so asking for the first again reprojects.
    _store_graph(_FakeGenerationStore(tenant="tenant-0"), include_text=True)
    assert len(projector_spy) == 7


def test_an_in_place_graph_rebuild_retires_the_cached_projection(projector_spy) -> None:
    """`recall graph rebuild` rewrites a generation's graph rows without moving its id.

    Keyed on the generation alone, a long lived server would serve the pre-rebuild projection
    for the rest of its life while `graph_readiness` reported the new graph. The readiness
    fingerprint moves when those rows are rewritten, so it belongs in the key.
    """
    store = _FakeGenerationStore()

    reasoning_projection(store, include_text=True)
    reasoning_projection(store, include_text=True)
    assert len(projector_spy) == 1

    store.graph_fingerprint = "graph-b"  # what `rebuild_graph` changes, id untouched
    reasoning_projection(store, include_text=True)

    assert len(projector_spy) == 2, "an in place rebuild must not be served from the cache"
    assert store.active == "gen-1"


def test_a_store_without_a_graph_row_is_still_cached(projector_spy) -> None:
    """No semantic graph means nothing mutable under the key, so caching stays correct."""
    store = _FakeGenerationStore(graph_fingerprint=None)

    reasoning_projection(store, include_text=True)
    reasoning_projection(store, include_text=True)

    assert len(projector_spy) == 1


def test_a_request_reads_the_readiness_once(projector_spy) -> None:
    """Reading readiness costs a full semantic-graph load, so a request must pay for it once.

    The cache key needs the graph fingerprint, and `reasoning_projection` needs the readiness
    row itself. Reading it in both places doubled that cost per request, which is the opposite
    of what this cache is for.
    """
    store = _FakeGenerationStore()
    reads = []
    original = store.graph_readiness

    def _counted():  # type: ignore[no-untyped-def]
        reads.append(1)
        return original()

    store.graph_readiness = _counted  # type: ignore[method-assign]

    reasoning_projection(store, include_text=True)

    assert len(reads) == 1


class _PinnedGenerationStore(_FakeGenerationStore):
    """A benchmark server pin: `snapshot` serves a retired generation, `active` moved on.

    `RECALL_PINNED_GENERATION_ID` (recall_mcp/server.py) calls `set_fixed_generation`, after
    which `snapshot()` yields the fixed id while `active_generation_id()` keeps naming the
    active one. `project_store_graph` builds against the snapshot, so the cache must key on it.
    """

    def __init__(self, pinned: str = "gen-retired", active: str = "gen-1") -> None:
        super().__init__(active=active)
        self.pinned = pinned

    @contextmanager
    def snapshot(self):  # type: ignore[no-untyped-def]
        yield self.pinned


def test_a_pinned_generation_is_keyed_on_the_snapshot_not_the_active_pointer(
    projector_spy, monkeypatch
) -> None:
    """Keying on the active pointer would file a pinned projection under another name."""
    import recall_mcp.service as service_module

    store = _PinnedGenerationStore()

    def _pinned_projector(target, *, include_text=False, **kwargs):  # type: ignore[no-untyped-def]
        projector_spy.append((target.tenant, target.pinned, include_text))
        return build_reasoning_graph(
            [],
            tenant_id=target.tenant,
            generation_id=target.pinned,
            include_text=include_text,
        )

    monkeypatch.setattr(service_module, "project_store_graph", _pinned_projector)

    first = reasoning_projection(store, include_text=True)
    second = reasoning_projection(store, include_text=True)

    assert first.generation_id == "gen-retired"
    assert second.generation_id == "gen-retired"
    # Cached under the snapshot's generation, so the second call is served, not reprojected.
    assert len(projector_spy) == 1
    assert all(key[1] == "gen-retired" for key in service._GRAPH_PROJECTIONS)
