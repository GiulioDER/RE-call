"""Tenant isolation at the RETRIEVER and TRUST layers.

`test_tenancy.py` pins the store. Neither `HybridRetriever` nor the trust layer issues SQL of its
own — both reach data only through a tenant-bound `PgVectorStore` — so isolation at these layers
holds *by construction* today.

That is exactly why these tests exist. "By construction" is a property of the current code, not an
enforced boundary: the day someone gives the retriever its own query, resolves a successor by
scanning the table, or caches a supersession map on something wider than one store, every
store-level test in `test_tenancy.py` still passes and the boundary is gone. These fail instead.

The supersession cases are the ones worth having. A leak there does not surface as another
tenant's text appearing in your results — it surfaces as *your own* memory being marked
`superseded` or withheld because of an edge someone else authored, which reads as a retrieval-
quality bug and would be debugged as one.
"""
from __future__ import annotations

import uuid

import psycopg
import pytest

from recall.retriever import HybridRetriever
from recall.store import PgVectorStore
from recall.trust import trusted_search
from recall.types import Chunk

from tests.conftest import TEST_DSN, requires_db


class DictEmbedder:
    dim = 4
    name = "dict"

    def __init__(self, mapping, default):
        self._mapping, self._default = mapping, default

    def embed(self, texts):
        return [self._mapping.get(t, self._default) for t in texts]


def _vec(i: int, dim: int = 4) -> list[float]:
    v = [0.0] * dim
    v[i % dim] = 1.0
    return v


@pytest.fixture
def tenant_table():
    name = "tr_" + uuid.uuid4().hex[:8]
    yield name
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute(f"DROP TABLE IF EXISTS {name}")


@pytest.fixture
def two_tenants(tenant_table):
    a = PgVectorStore(TEST_DSN, dim=4, table=tenant_table, tenant="acme")
    b = PgVectorStore(TEST_DSN, dim=4, table=tenant_table, tenant="globex")
    a.ensure_schema()
    b.ensure_schema()
    try:
        yield a, b
    finally:
        a.close()
        b.close()


@requires_db
def test_the_retriever_never_fuses_in_another_tenants_chunk(two_tenants):
    """Both retrieval legs, through the fusion that combines them.

    `HybridRetriever` issues a dense query and a sparse one and fuses the two rankings. Each leg
    is a separate statement and a separate chance to lose the predicate, and the fused result is
    where a leak would actually reach a caller — so this drives the public `search`, with text
    both tenants match lexically and vectors both match densely.
    """
    a, b = two_tenants
    a.upsert([Chunk("x", "a.md", "quarterly revenue figures")], [_vec(0)])
    b.upsert([Chunk("y", "b.md", "quarterly revenue figures")], [_vec(0)])

    embedder = DictEmbedder({}, _vec(0))

    for store, own in ((a, "a.md"), (b, "b.md")):
        result = HybridRetriever(store, embedder).search("quarterly revenue", k=10)
        assert [h.chunk.source for h in result.hits] == [own]


@requires_db
def test_one_tenants_supersedes_edge_cannot_demote_another_tenants_memory(two_tenants):
    """The contamination case: A says `shared.md` is superseded; B's `shared.md` must stay `ok`.

    A leak here is silent and expensive. B's memory is still returned, so nothing looks missing —
    it just arrives flagged `superseded`, pointing at a successor B does not have and cannot see.
    """
    a, b = two_tenants
    a.upsert(
        [Chunk("a2", "v2.md", "acme current", {"file": "v2.md", "ord": 0,
                                               "supersedes": "shared.md"})],
        [_vec(1)],
    )
    a.upsert([Chunk("a1", "shared.md", "acme old", {"file": "shared.md", "ord": 0})], [_vec(0)])
    b.upsert([Chunk("b1", "shared.md", "globex note", {"file": "shared.md", "ord": 0})], [_vec(0)])

    embedder = DictEmbedder({}, _vec(0))

    # A authored the edge, so A's own copy is correctly superseded — this pins that the test
    # setup really did create a live edge, so B's `ok` below is isolation and not a no-op.
    a_hit = next(h for h in trusted_search(a, embedder, "note", k=10).hits
                 if h.chunk.source == "shared.md")
    assert a_hit.verdict == "superseded"

    b_hit = next(h for h in trusted_search(b, embedder, "note", k=10).hits
                 if h.chunk.source == "shared.md")
    assert b_hit.verdict == "ok"
    assert b_hit.validity.superseded_by is None


@requires_db
def test_another_tenants_same_named_file_cannot_make_an_edge_ambiguous(two_tenants):
    """The subtle direction: a leak here *suppresses* a real edge rather than inventing one.

    `supersedes:` names its target by basename. When two indexed files share that basename the
    reference is ambiguous, and the resolver refuses to guess — the edge is dropped and reported
    `unresolved` so the read path fails closed. That is correct within a tenant.

    Across tenants it becomes a denial: B indexing `dir2/x.md` would add a second candidate for A's
    reference to `x.md`, so A's own, unambiguous, correctly-authored edge stops applying and A's
    superseded memory silently starts being served as current. Nothing in A's corpus changed, and
    no other tenant's text is ever exposed — which is what makes it hard to attribute.
    """
    a, b = two_tenants
    a.upsert([Chunk("a1", "dir1/x.md", "acme old", {"file": "dir1/x.md", "ord": 0})], [_vec(0)])
    a.upsert(
        [Chunk("a2", "new.md", "acme current", {"file": "new.md", "ord": 0, "supersedes": "x.md"})],
        [_vec(1)],
    )
    b.upsert([Chunk("b1", "dir2/x.md", "globex note", {"file": "dir2/x.md", "ord": 0})], [_vec(0)])

    # A sees exactly one `x.md`, so the edge resolves and nothing is unresolved.
    assert a.supersession() == ({"dir1/x.md": "new.md"}, frozenset())

    hit = next(h for h in trusted_search(a, DictEmbedder({}, _vec(0)), "old", k=10).hits
               if h.chunk.source == "dir1/x.md")
    assert hit.verdict == "superseded"


@requires_db
def test_the_ambiguity_rule_this_relies_on_is_real(two_tenants):
    """Guards the test above from passing vacuously.

    `test_another_tenants_same_named_file_cannot_make_an_edge_ambiguous` is only meaningful if two
    same-named files genuinely DO defeat the edge — otherwise it would pass on a build where
    ambiguity detection was broken, and prove nothing about tenancy. Here both copies belong to
    one tenant, which is the case that must fail closed.
    """
    a, _ = two_tenants
    a.upsert([Chunk("a1", "dir1/x.md", "old", {"file": "dir1/x.md", "ord": 0})], [_vec(0)])
    a.upsert([Chunk("a3", "dir2/x.md", "other", {"file": "dir2/x.md", "ord": 0})], [_vec(0)])
    a.upsert(
        [Chunk("a2", "new.md", "current", {"file": "new.md", "ord": 0, "supersedes": "x.md"})],
        [_vec(1)],
    )

    edges, unresolved = a.supersession()
    assert edges == {}
    assert unresolved != frozenset()
