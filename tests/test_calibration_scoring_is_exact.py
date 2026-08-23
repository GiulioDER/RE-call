"""Calibration evidence is the true maximum, whatever plan Postgres picks for a dense query.

`measure_top_cosines` is documented as the best dense cosine per labelled query, and three
verdicts rest on it: the fitted threshold (`calibrate`), whether an inherited threshold still
separates (`carry_forward`), and drift. It used to compute that with `query_dense(k=1)` — an
`ORDER BY embedding <=> v LIMIT 1` that Postgres may serve from `recall_chunks_v1`'s HNSW index.
That index is global over every tenant and generation and its walk is filter-blind, so a
generation-scoped query can return a row that was merely reachable rather than the nearest one.

Measured 2026-08-22 on a 60,000-chunk generation, one query, same slice and session: the planner
chose `recall_chunks_v1_embedding_idx` unprompted and `query_dense(k=1)` reported **0.000000**
against a true maximum of **0.707107**. 60k chunks under one tenant is an ordinary corpus, so this
was reachable in production and not only under a forced plan.

The consequence is not a smaller number, it is the wrong verdict in the dangerous direction.
Under-measuring the UNANSWERABLE class drives `false_confirm_rate` down, and that is what makes a
threshold which has stopped separating look like it still separates:
`test_carry_forward_rejects_when_the_threshold_stops_separating` flipped to `assert True is False`
under this plan, which is the exact failure that test exists to prevent.

Two levers make the fixture deterministic, and both are levers rather than causes.

- **`enable_sort = off`** forces the ordering to be satisfied by the index instead of by a sort,
  which is the plan a large corpus gets on its own. `tests/test_hnsw_filtered_recall.py` pins its
  plan for the same reason.
- **6,000 hay chunks against 2 needles.** The miss rate is a property of how rare the true nearest
  neighbours are within the slice, and it is NOT reliable at small sizes: measured 2026-08-22 over
  12 trials per size, the approximate plan missed 0/12 at 200, 500 and 1,000 hay chunks, 2/12 at
  2,000, and 12/12 at 5,000. A 22-row fixture is a coin flip, so a test built on one would be
  green about half the time with the fix reverted. 6,000 is 5,000 plus margin.
"""
from __future__ import annotations

import hashlib
import uuid

import psycopg
import pytest

from recall.calibration_v2 import CalibrationRepository
from recall.generation_store import GenerationStore
from recall.generations import GenerationManager
from recall.lineage import IndexManifestV1, ManifestObjectV1
from recall.manifest import S3Allowlist, S3ObjectReader
from recall.store import (
    DEFAULT_HNSW_EF_SEARCH_FILTERED,
    DEFAULT_HNSW_ITERATIVE_SCAN_FILTERED,
)

from tests.conftest import TEST_DSN, requires_db
from tests.test_calibration_carry_forward import _CarryEmbedder, _labels
from tests.test_generations import _S3, _pipeline

# The module-scoped fixture builds 6,002 chunks through the real `GenerationManager.build`, which
# writes one row per chunk and measured 31 s on an idle machine. That is charged to whichever test
# runs first and would sit close to the global `timeout = 120` on a loaded box. Same escape hatch,
# and same reasoning, as `tests/test_hnsw_filtered_recall.py`: the global timeout exists to fail a
# HUNG chunker fast, and it keeps doing that here.
pytestmark = [requires_db, pytest.mark.timeout(300)]

#: Hay chunks in the generation. See the module docstring for the measured miss rate by size.
N_HAY = 6_000
#: Needle chunks — the only ones an unanswerable query has any cosine with.
N_NEEDLE = 2
#: What a `missing-*` query scores against a needle: cosine of [0,1,...] with [1,1,...].
NEEDLE_COSINE = 0.7071067811865476
#: What it scores against hay, which is every other chunk in the generation.
HAY_COSINE = 0.0


class _HayNeedleEmbedder(_CarryEmbedder):
    """Queries on the axes, hay on axis 0, needles on the diagonal.

    Subclasses the carry fixture's embedder so the pipeline identity, dimension and model name
    stay the ones the calibration path already accepts, and only the document rule changes.
    """

    def embed(self, texts: list[str]) -> list[list[float]]:
        values: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dim
            if text.startswith("answer-"):
                vector[0] = 1.0
            elif text.startswith("missing-"):
                vector[1] = 1.0
            elif text.startswith("needle"):
                vector[0] = 1.0
                vector[1] = 1.0
            else:
                vector[0] = 1.0
            values.append(vector)
        return values


def _chunker(text: str) -> list[str]:
    """One source explodes into the hay, the other into the needles.

    Chunks rather than documents because the build writes a row per chunk inside one transaction
    per source: 6,002 chunks over two sources costs 31 s, where 6,002 separate manifest objects
    would pay a fetch, a lock and a transaction each.
    """
    if text.startswith("needle"):
        return [f"needle {index}" for index in range(N_NEEDLE)]
    return [f"hay {index}" for index in range(N_HAY)]


def _approximate_dsn() -> str:
    """`TEST_DSN` with the planner pushed onto the index for `ORDER BY ... LIMIT`.

    Carried in the connection string rather than set with `ALTER DATABASE`: it travels with every
    store this module opens, needs no privilege on the database, and cannot outlive the test the
    way a persisted database-level setting would.
    """
    separator = "&" if "?" in TEST_DSN else "?"
    return f"{TEST_DSN}{separator}options=-c%20enable_sort%3Doff"


def _build(dsn: str, tenant: str) -> str:
    bodies = {"hay.md": b"hay corpus", "needle.md": b"needle corpus"}
    manifest = IndexManifestV1(
        tenant,
        "v1",
        tuple(
            ManifestObjectV1(
                f"s3://approved/corpora/{tenant}/{name}",
                hashlib.sha256(data).hexdigest(),
                "text/markdown",
                len(data),
                hashlib.sha256(data).hexdigest(),
            )
            for name, data in sorted(bodies.items())
        ),
    )
    objects = {}
    for entry in manifest.objects:
        name = entry.uri.rsplit("/", 1)[1]
        objects[("approved", f"corpora/{tenant}/{name}", entry.version_id)] = bodies[name]
    reader = S3ObjectReader(_S3(objects), S3Allowlist.parse("approved/corpora/"))

    manager = GenerationManager(dsn, tenant, actor="pytest", environment="test")
    embedder = _HayNeedleEmbedder()
    generation = manager.create(manifest, _pipeline(embedder.name))
    manager.build(generation.generation_id, reader, embedder, _chunker)
    manager.validate(generation.generation_id)
    return generation.generation_id


@pytest.fixture(scope="module")
def approximate_generation():
    """One generation big enough that the approximate plan reliably misses its needles."""
    tenant = "exact-scoring-" + uuid.uuid4().hex[:10]
    dsn = _approximate_dsn()
    try:
        yield tenant, _build(dsn, tenant), dsn
    finally:
        with psycopg.connect(TEST_DSN, autocommit=True) as conn:
            conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (tenant,))
            conn.execute("DELETE FROM recall_calibrations WHERE tenant_id = %s", (tenant,))
            conn.execute("DELETE FROM recall_audit_events WHERE tenant_id = %s", (tenant,))
            conn.execute("DELETE FROM recall_tenant_state WHERE tenant_id = %s", (tenant,))
            conn.execute("DELETE FROM recall_generations WHERE tenant_id = %s", (tenant,))


@pytest.fixture(autouse=True)
def _pinned_hnsw_tuning(monkeypatch):
    """Hold the filtered-search tuning at its defaults for every test here.

    The miss rate this module depends on is a function of `hnsw.ef_search` against the number of
    hay chunks, so a host that exports a wider scan would silently turn these tests green by
    removing the condition rather than by fixing anything.
    """
    monkeypatch.setenv("RECALL_HNSW_EF_SEARCH_FILTERED", str(DEFAULT_HNSW_EF_SEARCH_FILTERED))
    monkeypatch.setenv(
        "RECALL_HNSW_ITERATIVE_SCAN_FILTERED", DEFAULT_HNSW_ITERATIVE_SCAN_FILTERED
    )


def _unanswerable_vector(dim: int) -> list[float]:
    vector = [0.0] * dim
    vector[1] = 1.0
    return vector


def _plan_uses_the_index(dsn: str, tenant: str, generation: str, vector: list[float]) -> bool:
    """Did this connection really get the approximate plan? Asserted, never assumed.

    Without this the tests can pass because the planner quietly kept the exact plan, which is the
    green that means nothing: the fixture would no longer be reproducing the hole it guards.
    """
    literal = "[" + ",".join(str(value) for value in vector) + "]"
    with psycopg.connect(dsn, autocommit=True) as conn, conn.transaction():
        # `set_config(..., is_local => true)` rather than `SET LOCAL`: Postgres does not accept a
        # bound parameter for a `SET` value, and interpolating one into the statement is the shape
        # this repository refuses on sight. Same substitution `PgVectorStore` makes for the
        # unfiltered arm's widening, and for the same reason.
        conn.execute(
            "SELECT set_config('hnsw.ef_search', %s, true)",
            (str(DEFAULT_HNSW_EF_SEARCH_FILTERED),),
        )
        conn.execute(
            "SELECT set_config('hnsw.iterative_scan', %s, true)",
            (DEFAULT_HNSW_ITERATIVE_SCAN_FILTERED,),
        )
        rows = conn.execute(
            "EXPLAIN (COSTS OFF) SELECT chunk_id FROM recall_chunks_v1 "
            "WHERE tenant_id = %s AND generation_id = %s "
            "ORDER BY embedding <=> %s::vector LIMIT 1",
            (tenant, generation, literal),
        ).fetchall()
    return any("recall_chunks_v1_embedding_idx" in row[0] for row in rows)


@requires_db
def test_top_cosine_is_the_maximum_where_query_dense_is_only_a_neighbour(
    approximate_generation,
) -> None:
    """`top_cosine` returns the true maximum on the plan where `query_dense(k=1)` does not.

    Both halves are asserted, and the second is not decoration. Showing only that `top_cosine` is
    right would leave open that the plan never changed and the test proved nothing; showing the
    approximate answer differ on the same slice, in the same session, is what establishes that
    the fixture still reproduces the defect.
    """
    tenant, generation, dsn = approximate_generation
    embedder = _HayNeedleEmbedder()
    vector = _unanswerable_vector(embedder.dim)

    assert _plan_uses_the_index(dsn, tenant, generation, vector), (
        "the planner kept the exact plan, so this test is not exercising the approximate one and "
        "cannot be evidence for anything"
    )

    store = GenerationStore(dsn, embedder.dim, tenant=tenant)
    try:
        with store.pin_generation(generation):
            exact = store.top_cosine(vector)
            hits = store.query_dense(vector, k=1)
    finally:
        store.close()

    assert exact == pytest.approx(NEEDLE_COSINE), (
        "top_cosine must be the maximum cosine in the pinned generation. An aggregate has no "
        "ORDER BY and no LIMIT, so the ordering index cannot serve it and no plan may move this "
        "number"
    )
    approximate = hits[0].score if hits else 0.0
    assert approximate == pytest.approx(HAY_COSINE), (
        f"the approximate plan is expected to return hay here and did not ({approximate}); with "
        f"{N_HAY} hay against {N_NEEDLE} needles the miss measured 12/12, so if it now finds the "
        f"needle this fixture has stopped reproducing the defect and the assertion above is "
        f"vacuous"
    )


@requires_db
def test_score_query_set_reports_the_maximum_not_the_neighbour(approximate_generation) -> None:
    """The calibration path is wired to the exact measurement, not merely offered one.

    `CalibrationRepository.score_query_set` is what `calibrate`, `carry_forward` and drift all
    call, so this is the assertion that fails if `measure_top_cosines` ever goes back to
    `query_dense(k=1)` — which is the change that produced the flipped verdict.
    """
    tenant, generation, dsn = approximate_generation
    embedder = _HayNeedleEmbedder()

    assert _plan_uses_the_index(dsn, tenant, generation, _unanswerable_vector(embedder.dim)), (
        "the planner kept the exact plan, so this run does not exercise the condition that used "
        "to under-measure the unanswerable class"
    )

    repository = CalibrationRepository(dsn, tenant, actor="pytest")
    answerable, unanswerable = repository.score_query_set(generation, embedder, _labels())

    assert unanswerable == pytest.approx([NEEDLE_COSINE] * len(unanswerable)), (
        "every unanswerable query must measure the needle it can actually reach; measuring 0.0 "
        "here is the under-measurement itself, and it is what drives false_confirm_rate to zero "
        "and certifies a threshold that has stopped deciding anything"
    )
    assert answerable == pytest.approx([1.0] * len(answerable)), (
        "the answerable class is unaffected by the miss — it matches the hay, which is 6,000 of "
        "the 6,002 chunks — and that asymmetry is why the defect is invisible in separability"
    )
