"""The contract the hybrid retrieval path has to keep, pinned.

Three separate concerns, all discovered while investigating why a working sparse leg added
nothing measurable to fused top-k on FinanceBench:

1. `query_dense(k)` must actually return `k` candidates. On the UNFILTERED arm it does not:
   pgvector's `hnsw.ef_search` defaults to 40, and an HNSW scan cannot return more rows than it
   examined, so `LIMIT 100` silently yields 40. No error, no warning. This is the #81 failure
   class exactly — a knob that looks set and is not — and it silently capped `candidate_k` for
   every caller that raised it above 40.

2. The `_rrf` damping constant `k` cannot reorder candidates ACROSS legs. Pinned because it is
   counter-intuitive and because "tune the RRF k constant" is the first thing anyone proposes
   when fusion underperforms. It is a no-op, and a test is cheaper than re-measuring that.

3. Fusion CAN rank a hit below where the better single leg had it. Pinned as the real behaviour
   rather than asserted away, because the intuitive invariant ("fusion never does worse than its
   best input") is false for RRF and any fix claiming otherwise needs to say so explicitly.

Plus the #81 regression floor itself: a sparse leg that stops matching must fail loudly.
"""
from __future__ import annotations

import random

import psycopg
import pytest

from recall.retriever import _rrf
from recall.types import Chunk

from tests.conftest import TEST_DSN, requires_db

DIM = 64


# ---------------------------------------------------------------------------------------
# 1. query_dense must honour k  (the ef_search silent cap)
# ---------------------------------------------------------------------------------------

#: Comfortably above pgvector's default `hnsw.ef_search` of 40, and above any candidate_k a
#: caller is likely to pick, so the assertion fails on the cap rather than on a boundary.
N_ROWS = 5_000


@pytest.fixture
def dense_corpus(make_store):
    """A corpus the planner actually serves from the HNSW index.

    The ANALYZE is load-bearing, and finding out why cost an hour. Without it these tests are
    FLAKY: on a freshly-filled table Postgres has no statistics, estimates ~36 rows, and picks a
    Sort plan, which is exact — so `LIMIT k` returns k, the cap never appears, and the test
    passes for entirely the wrong reason. Measured on this exact fixture:

        N=5000, pre-ANALYZE  -> Sort plan        -> k=50 gives 50, k=100 gives 100  (passes)
        N=5000, post-ANALYZE -> HNSW Index Scan  -> k=50 gives 40, k=100 gives 40   (the bug)

    A real deployment is always in the second state, because autovacuum analyses. ANALYZE here
    puts the fixture in the state production is in, rather than leaving the test's outcome to
    whether autovacuum happened to have run.
    """
    store = make_store(DIM)
    rng = random.Random(11)
    chunks, vecs = [], []
    for i in range(N_ROWS):
        chunks.append(Chunk(id=f"c{i}", source=f"s{i % 50}.md", text=f"chunk {i}", metadata={}))
        vecs.append([rng.random() for _ in range(DIM)])
    for lo in range(0, N_ROWS, 1_000):
        store.upsert(chunks[lo : lo + 1_000], vecs[lo : lo + 1_000])
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute(f"ANALYZE {store.table}")
    return store


def _uses_hnsw_index(store, vector, k) -> bool:
    """Whether the planner serves this query from the HNSW index.

    Guards against the test going vacuously green. If a planner or pgvector change stops
    choosing the index, `query_dense` becomes exact and the k-honouring assertions below pass
    without exercising the path they exist to cover — so that condition is asserted, not assumed.
    """
    lit = "[" + ",".join(f"{x:.6f}" for x in vector) + "]"
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        plan = conn.execute(
            f"EXPLAIN SELECT id FROM {store.table} WHERE tenant_id = 'default' "
            f"ORDER BY embedding <=> '{lit}' LIMIT {k}"
        ).fetchall()
    return any("_emb_idx" in row[0] for row in plan)


@requires_db
@pytest.mark.parametrize("k", [50, 100, 200])
def test_query_dense_returns_k_rows_when_k_exceeds_default_ef_search(dense_corpus, k):
    """An unfiltered `query_dense(k)` must return k rows for k well below the corpus size.

    Fails on master: HNSW examines only `ef_search` (default 40) candidates, so the answer is
    truncated to 40 and the caller is told nothing. `HybridRetriever(candidate_k=...)` is
    exactly such a caller.
    """
    rng = random.Random(99)
    vec = [rng.random() for _ in range(DIM)]
    assert _uses_hnsw_index(dense_corpus, vec, k), (
        "planner is not using the HNSW index for this query, so the test is not covering the "
        "truncation path — grow the fixture or check that ensure_schema built the index"
    )
    hits = dense_corpus.query_dense(vec, k=k)
    assert len(hits) == k, (
        f"asked for {k} dense candidates, got {len(hits)} — hnsw.ef_search is capping the scan "
        f"and query_dense is not reporting it"
    )


@requires_db
def test_query_dense_k_is_not_silently_capped_across_a_range(dense_corpus):
    """The cap is a PLATEAU, which is the tell. Pinned as a shape, not a single point.

    Two different `k` returning the same row count is the smell that found this: the size of the
    answer is being decided by something other than k.
    """
    vec = [0.5] * DIM
    assert _uses_hnsw_index(dense_corpus, vec, 120)
    counts = {k: len(dense_corpus.query_dense(vec, k=k)) for k in (30, 60, 120)}
    assert counts == {30: 30, 60: 60, 120: 120}, f"k is not controlling the result size: {counts}"


# ---------------------------------------------------------------------------------------
# 2. what the RRF damping constant can and cannot do
# ---------------------------------------------------------------------------------------

DENSE = [f"D{i}" for i in range(50)]
SPARSE = [f"S{i}" for i in range(50)]


def _order(rankings, k):
    f = _rrf(rankings, k=k)
    return sorted(f, key=lambda cid: f[cid], reverse=True)


@pytest.mark.parametrize("k", [0, 1, 10, 60, 200, 10_000])
def test_rrf_k_constant_cannot_reorder_disjoint_legs(k):
    """The damping constant is a no-op on the cross-leg ordering.

    `dense[r]` and `sparse[r]` both score `1/(k+r+1)` — identical for EVERY k — so no choice of
    k expresses "the dense leg is the better leg". Only a per-leg WEIGHT can. This is why the
    measured hit@5 was flat at 0.3200 across k in {10 … 10000} on FinanceBench.
    """
    assert _order([DENSE, SPARSE], k=k) == _order([DENSE, SPARSE], k=60)


def test_rrf_interleaves_disjoint_legs_round_robin():
    """The consequence: half the fused top-k belongs to the sparse leg regardless of its quality.

    With legs that share no candidates, the fused order is D0, S0, D1, S1, … so a fused top-5
    spends 2 of its 5 slots on the sparse leg even when that leg is pure noise.
    """
    assert _order([DENSE, SPARSE], k=60)[:6] == ["D0", "S0", "D1", "S1", "D2", "S2"]


def test_rrf_dilates_a_dense_hit_when_the_sparse_leg_contributes_nothing():
    """A hit at dense rank r lands at fused rank 2r-1. At r=3 it just survives a top-5 cut; at
    r=5 it does not. This is the mechanism by which switching the sparse leg ON can LOWER hit@5.
    """
    for r in (1, 2, 3, 5):
        dense = [f"D{i}" for i in range(r - 1)] + ["GOLD"] + [f"D{i}" for i in range(r, 50)]
        fused = _order([dense, SPARSE], k=60)
        assert fused.index("GOLD") + 1 == 2 * r - 1
    assert "GOLD" not in fused[:5]  # r == 5: present in the dense top-5, gone from the fused one


def test_rrf_can_rank_a_hit_below_its_position_in_the_better_leg():
    """The intuitive invariant "fusion is never worse than its best input" is FALSE for RRF.

    Pinned deliberately. On FinanceBench the fused rank of gold was worse than the best single
    leg's on 77 of the 106 questions where any leg found it. A change that makes fusion
    best-input-dominant is a real behaviour change and must update this test on purpose.
    """
    dense = ["GOLD"] + [f"D{i}" for i in range(49)]
    fused = _order([dense, SPARSE], k=60)
    assert fused.index("GOLD") == 0
    # ...but let one candidate appear in BOTH legs and the co-occurrence bonus outranks it:
    shared = "X"
    fused2 = _order([["GOLD", shared] + [f"D{i}" for i in range(48)], [shared] + SPARSE], k=60)
    assert fused2[0] == shared and fused2.index("GOLD") == 1, fused2[:3]


def test_rrf_weights_do_what_the_k_constant_cannot():
    """The contrast that makes the point actionable: a weight breaks the cross-leg tie."""
    from recall.retriever import _rrf as _plain

    assert _order([DENSE, SPARSE], k=60)[:4] == ["D0", "S0", "D1", "S1"]
    scores: dict[str, float] = {}
    for ranking, w in ((DENSE, 2.0), (SPARSE, 1.0)):
        for rank, cid in enumerate(ranking):
            scores[cid] = scores.get(cid, 0.0) + w / (60 + rank + 1)
    weighted = sorted(scores, key=lambda c: scores[c], reverse=True)
    assert weighted[:4] == ["D0", "D1", "D2", "D3"]
    assert _plain is _rrf  # the plain rule is the one shipped; the weighted one is not


# ---------------------------------------------------------------------------------------
# 3. #81 regression floor — a dead sparse leg must fail loudly
# ---------------------------------------------------------------------------------------

#: Each doc pairs ONE rare discriminative term with shared boilerplate, so a question naming that
#: term has exactly one right answer. Questions are full sentences: under the conjunctive
#: `websearch_to_tsquery` form that #81 fixed, a doc had to contain EVERY word including
#: "what"/"was"/"the", which no doc does — recall collapses to 0.
_FIXTURE = {
    "zylographic": "Consolidated statements of operations for the fiscal year. Total zylographic revenue recognised.",
    "quixotry": "Consolidated statements of operations for the fiscal year. Total quixotry revenue recognised.",
    "brachiate": "Consolidated statements of operations for the fiscal year. Total brachiate revenue recognised.",
    "pellucid": "Consolidated statements of operations for the fiscal year. Total pellucid revenue recognised.",
    "tergiversate": "Consolidated statements of operations for the fiscal year. Total tergiversate revenue recognised.",
}
_QUESTION = "What was the total {term} revenue recognised in the consolidated statements for the fiscal year?"

#: 5/5 with the disjunctive form; 0/5 with the conjunctive one. A floor of 4 leaves room for
#: tokeniser drift without leaving room for the leg to die.
SPARSE_RECALL_FLOOR = 4


@requires_db
def test_sparse_leg_recall_floor(make_store):
    """#81 existed because a dead sparse leg is indistinguishable from a working one at the API.

    This pins the leg's recall directly, so any change that stops it matching — reverting to
    `websearch_to_tsquery`, dropping the tsvector, mangling the tsquery build — fails here
    rather than being absorbed silently by the dense leg.
    """
    store = make_store(DIM)
    rng = random.Random(3)
    terms = list(_FIXTURE)
    store.upsert(
        [
            Chunk(id=t, source=f"{t}.md", text=_FIXTURE[t], metadata={"file": f"{t}.md"})
            for t in terms
        ],
        [[rng.random() for _ in range(DIM)] for _ in terms],
    )

    found = 0
    for t in terms:
        hits = store.query_sparse(_QUESTION.format(term=t), k=50)
        assert hits, f"sparse leg returned NOTHING for {t!r} — the leg is dead (#81)"
        if any(h.chunk.id == t for h in hits):
            found += 1
    assert found >= SPARSE_RECALL_FLOOR, (
        f"sparse leg recall@50 fell to {found}/{len(terms)} (floor {SPARSE_RECALL_FLOOR}). "
        f"A conjunctive tsquery scores 0/{len(terms)} here."
    )


@requires_db
def test_sparse_leg_matches_a_partial_question(make_store):
    """The specific #81 shape: a question whose words are NOT all in any single document.

    This is the assertion that a conjunctive tsquery cannot pass, isolated from ranking.
    """
    store = make_store(DIM)
    store.upsert(
        [Chunk(id="d", source="d.md", text="Total quixotry revenue recognised.", metadata={})],
        [[0.5] * DIM],
    )
    hits = store.query_sparse(
        "What was the total quixotry revenue for the fiscal year ended December 2023?", k=10
    )
    assert [h.chunk.id for h in hits] == ["d"]
