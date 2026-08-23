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

⛔ **These tests assert the CAUSE, never the consequence, and the first draft got that wrong.**
The obvious test builds a corpus where the approximate walk misses and asserts it under-reports.
That assertion depends on HNSW graph construction, which `recall/calibration.py:145` already
records as nondeterministic, so it is a coin flip dressed as a guard: measured over 12 trials per
size with direct inserts the miss ran 0/12 at 1,000 hay chunks, 2/12 at 2,000 and 12/12 at 5,000,
yet the same shape built through `GenerationManager` at 6,000 still found the needle in 2 of 3
runs, because the build path inserts in a different order and builds a different graph. Writing a
flaky assertion into the change that removes ANN nondeterminism from this path would have been the
defect wearing the fix's clothes.

What is deterministic, and therefore what is asserted here:

- an aggregate CANNOT be served by an ordering index, while `ORDER BY ... LIMIT 1` can, on the
  same slice in the same session (`test_an_aggregate_cannot_be_served_by_the_ordering_index`);
- so the shape of the SQL is the whole property, and it is pinned at the source
  (`test_top_cosine_is_written_as_an_aggregate_in_both_stores`);
- the value it returns is the known maximum of a fixture whose answer is arithmetic, not a
  retrieval outcome (`test_top_cosine_returns_the_true_maximum`);
- a degenerate row must not poison it (`test_a_degenerate_vector_does_not_poison_the_maximum`);
- and `measure_top_cosines` must actually call it (`test_measure_top_cosines_asks_for_the_maximum`,
  which needs no database at all).
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import textwrap
import uuid

import psycopg
import pytest

from recall.eval.calibrate import measure_top_cosines
from recall.generation_store import GenerationStore
from recall.generations import GenerationManager
from recall.lineage import IndexManifestV1, ManifestObjectV1
from recall.manifest import S3Allowlist, S3ObjectReader
from recall.store import PgVectorStore

from tests.conftest import TEST_DSN, requires_db
from tests.test_calibration_carry_forward import _CarryEmbedder
from tests.test_generations import _S3, _pipeline

#: What a `missing-*` query scores against a needle: cosine of [0,1,...] with [1,1,...].
NEEDLE_COSINE = 0.7071067811865476
#: The embedding index the aggregate must never be able to use.
EMBEDDING_INDEX = "recall_chunks_v1_embedding_idx"


class _HayNeedleEmbedder(_CarryEmbedder):
    """Queries on the axes, hay on axis 0, needles on the diagonal."""

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


class _DegenerateEmbedder(_CarryEmbedder):
    """One document embeds to the zero vector, which pgvector scores as NaN.

    Not contrived: an embedder that returns a zero vector for an input it cannot encode, a
    truncated write, or a model that underflows on a very short document all produce this row, and
    the store's only constraint on `embedding` is NOT NULL.
    """

    def embed(self, texts: list[str]) -> list[list[float]]:
        values: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dim
            if not text.startswith("degenerate"):
                vector[0] = 1.0
            values.append(vector)
        return values


def _approximate_dsn() -> str:
    """`TEST_DSN` with the planner pushed onto the index for `ORDER BY ... LIMIT`.

    Carried in the connection string rather than set with `ALTER DATABASE`: it travels with every
    store this module opens, needs no privilege on the database, and cannot outlive the test the
    way a persisted database-level setting would.
    """
    separator = "&" if "?" in TEST_DSN else "?"
    return f"{TEST_DSN}{separator}options=-c%20enable_sort%3Doff"


def _build(dsn: str, tenant: str, embedder, bodies: dict[str, bytes]) -> str:
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
    generation = manager.create(manifest, _pipeline(embedder.name))
    manager.build(generation.generation_id, reader, embedder, lambda text: [text])
    manager.validate(generation.generation_id)
    return generation.generation_id


def _cleanup(tenant: str) -> None:
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (tenant,))
        conn.execute("DELETE FROM recall_calibrations WHERE tenant_id = %s", (tenant,))
        conn.execute("DELETE FROM recall_audit_events WHERE tenant_id = %s", (tenant,))
        conn.execute("DELETE FROM recall_tenant_state WHERE tenant_id = %s", (tenant,))
        conn.execute("DELETE FROM recall_generations WHERE tenant_id = %s", (tenant,))


@pytest.fixture
def hay_needle_generation():
    """A small generation: hay on axis 0, one needle on the diagonal.

    Small on purpose. Nothing here depends on the approximate walk missing, so nothing here needs
    a corpus large enough to make it miss, and a two-source build costs a second where the
    6,000-chunk version cost 31 and was flaky anyway.
    """
    tenant = "exact-scoring-" + uuid.uuid4().hex[:10]
    embedder = _HayNeedleEmbedder()
    dsn = _approximate_dsn()
    try:
        bodies = {"hay.md": b"hay corpus", "needle.md": b"needle corpus"}
        yield tenant, _build(dsn, tenant, embedder, bodies), dsn, embedder
    finally:
        _cleanup(tenant)


def _unanswerable_vector(dim: int) -> list[float]:
    vector = [0.0] * dim
    vector[1] = 1.0
    return vector


@requires_db
def test_an_aggregate_cannot_be_served_by_the_ordering_index(hay_needle_generation) -> None:
    """The property the fix rests on, measured rather than assumed.

    `enable_sort = off` is in force for this session, which is the strongest push a planner can be
    given toward satisfying an ORDER BY from the index. Under exactly that pressure the two shapes
    must still plan differently, because an aggregate has no ordering to satisfy. If this ever
    stops holding, `top_cosine` is no longer exact by construction and the whole approach needs
    revisiting rather than patching.
    """
    tenant, generation, dsn, embedder = hay_needle_generation
    literal = "[" + ",".join(str(v) for v in _unanswerable_vector(embedder.dim)) + "]"

    with psycopg.connect(dsn, autocommit=True) as conn:
        ordered = conn.execute(
            "EXPLAIN (COSTS OFF) SELECT chunk_id FROM recall_chunks_v1 "
            "WHERE tenant_id = %s AND generation_id = %s "
            "ORDER BY embedding <=> %s::vector LIMIT 1",
            (tenant, generation, literal),
        ).fetchall()
        aggregated = conn.execute(
            "EXPLAIN (COSTS OFF) SELECT 1 - min(embedding <=> %s::vector) FROM recall_chunks_v1 "
            "WHERE tenant_id = %s AND generation_id = %s",
            (literal, tenant, generation),
        ).fetchall()

    ordered_plan = "\n".join(row[0] for row in ordered)
    aggregated_plan = "\n".join(row[0] for row in aggregated)
    assert EMBEDDING_INDEX in ordered_plan, (
        f"with enable_sort=off the ordered form is expected to be served by the approximate "
        f"index; it planned as:\n{ordered_plan}"
    )
    assert EMBEDDING_INDEX not in aggregated_plan, (
        f"an aggregate must not be servable by the ordering index, which is the entire reason "
        f"top_cosine is written as one; it planned as:\n{aggregated_plan}"
    )


@requires_db
def test_top_cosine_returns_the_true_maximum(hay_needle_generation) -> None:
    """The value, on a session forced onto the approximate plan for ordered queries."""
    tenant, generation, dsn, embedder = hay_needle_generation
    store = GenerationStore(dsn, embedder.dim, tenant=tenant)
    try:
        with store.pin_generation(generation):
            measured = store.top_cosine(_unanswerable_vector(embedder.dim))
    finally:
        store.close()
    assert measured == pytest.approx(NEEDLE_COSINE), (
        "the needle is the only chunk this query has any cosine with, so its similarity is "
        "arithmetic and no plan may change it"
    )


@requires_db
def test_a_degenerate_vector_does_not_poison_the_maximum() -> None:
    """One zero-norm embedding must not turn the whole generation's score into NaN.

    pgvector's cosine distance is NaN for a zero-norm vector, and Postgres orders NaN as LARGER
    than every number inside an aggregate but LAST in an ascending sort. So the obvious spelling,
    `max(1 - distance)`, answers NaN for a corpus holding a single degenerate row where the
    `ORDER BY distance LIMIT 1` it replaces answers the real nearest neighbour. `top_cosine` takes
    the minimum DISTANCE instead, which reproduces the sort's semantics exactly.

    The direction is what makes this worth a test rather than a comment. A NaN flows into the
    calibration sample lists, and `NaN >= threshold` is false, so it is counted as a correct
    abstention and pushes `false_confirm_rate` DOWN — the same way a broken threshold gets
    certified, which is the failure this module exists to close.
    """
    tenant = "degenerate-" + uuid.uuid4().hex[:10]
    embedder = _DegenerateEmbedder()
    try:
        bodies = {"real.md": b"real corpus", "degenerate.md": b"degenerate corpus"}
        generation = _build(TEST_DSN, tenant, embedder, bodies)
        query = [0.0] * embedder.dim
        query[0] = 1.0
        store = GenerationStore(TEST_DSN, embedder.dim, tenant=tenant)
        try:
            with store.pin_generation(generation):
                measured = store.top_cosine(query)
        finally:
            store.close()
    finally:
        _cleanup(tenant)

    # `approx` is already the NaN check: NaN compares false against any approx target, so a
    # poisoned aggregate fails here rather than needing a second assertion to catch it.
    assert measured == pytest.approx(1.0), (
        f"the real document is an exact match and must be reported as one; {measured!r} means the "
        f"degenerate row's NaN won the aggregate"
    )


def _sql_in(method) -> str:
    """Every string literal in `method`'s body, with the docstring removed.

    The docstring has to go, and finding that out cost a test that failed five times for five
    wrong reasons: these methods EXPLAIN their own reasoning, so the prose contains the exact
    phrases the assertions below look for, and a naive scan of the source matches its own
    explanation rather than its SQL.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
    function = tree.body[0]
    assert isinstance(function, ast.FunctionDef)
    body = function.body[1:] if ast.get_docstring(function) is not None else function.body
    return " ".join(
        node.value
        for statement in body
        for node in ast.walk(statement)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    )


def test_top_cosine_is_written_as_an_aggregate_in_both_stores() -> None:
    """Pin the SQL shape at the source, because the value tests cannot catch a regression.

    A `top_cosine` rewritten as `ORDER BY ... LIMIT 1` returns the RIGHT answer on any corpus small
    enough to test quickly, and the wrong one only when the approximate walk misses, which is not
    reproducible on demand. So the value assertions above would stay green through exactly the
    regression this change exists to prevent. Reading the source is what makes the guard
    deterministic.

    Precedent: `test_timed_public_methods_matches_the_actual_timer_call_sites` parses this same
    module rather than trusting a declaration, and for the same reason.
    """
    for owner in (PgVectorStore, GenerationStore):
        statement = _sql_in(owner.top_cosine)
        assert "min(embedding <=>" in statement, (
            f"{owner.__name__}.top_cosine must take the minimum DISTANCE: an aggregate is what "
            f"the ordering index cannot serve, and min-of-distance is what handles a NaN row the "
            f"way the ORDER BY it replaces did. Found: {statement!r}"
        )
        assert "ORDER BY" not in statement.upper(), (
            f"{owner.__name__}.top_cosine must not order: an ORDER BY makes it servable by the "
            f"approximate index and the measurement stops being exact. Found: {statement!r}"
        )
        assert "LIMIT" not in statement.upper(), (
            f"{owner.__name__}.top_cosine must not LIMIT. Found: {statement!r}"
        )


def test_measure_top_cosines_asks_for_the_maximum() -> None:
    """The wiring, pinned without a database.

    `measure_top_cosines` is the single sampling rule behind `calibrate`, `carry_forward` and
    drift, so what it CALLS is the whole question. A store whose `query_dense` refuses makes the
    regression impossible to miss, where a value comparison would only catch it on a corpus where
    the approximate walk happened to go wrong.
    """
    calls: list[str] = []

    class _RefusingStore:
        def top_cosine(self, vector):
            calls.append("top_cosine")
            return 0.5

        def query_dense(self, vector, k, source=None):
            raise AssertionError(
                "measure_top_cosines called query_dense; that is an approximate nearest-neighbour "
                "lookup, not the maximum it documents, and using it here is the defect this "
                "module exists to keep out"
            )

    class _Embedder:
        dim = 4
        name = "stub"

        def embed(self, texts):
            return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    labels = [
        {"query": "answer-0", "answerable": True},
        {"query": "missing-0", "answerable": False},
        {"query": "trust-0", "answerable": True, "trust": True},
    ]
    answerable, unanswerable = measure_top_cosines(_RefusingStore(), _Embedder(), labels)

    assert calls == ["top_cosine", "top_cosine"], (
        "one call per labelled query, and the `trust` entry carries no answerable label so it is "
        "skipped — the sampling rule this function is the single source of"
    )
    assert answerable == [0.5]
    assert unanswerable == [0.5]
