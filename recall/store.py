from __future__ import annotations

import json
import os
import re
import warnings
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime
from functools import partial
from typing import TYPE_CHECKING, Any, TypeVar
from uuid import uuid4

import psycopg
from pgvector import SparseVector, Vector
from pgvector.psycopg import register_vector

from recall.supersession import (  # noqa: F401  # re-exported: public names of this module
    EdgeCandidates,
    SupersessionGraph,
    resolve_supersession_graph,
    supersession_rows_sql,
    _basename,
    _resolve_rows,
    resolve_supersession,
    SUPERSEDES_TARGET_ROWS_SQL,
    resolve_supersession_candidates,
)
from recall.supersession_records import (
    erasure_subjects,
    pg_records,
    records_matching,
)
from recall.db_constants import (  # noqa: F401  # re-exported: public names of this module
    _PG_MAX_INT,
    DEFAULT_SCHEMA_LOCK_TIMEOUT_MS,
    DEFAULT_TABLE,
    DEFAULT_TENANT,
    TENANT_GUC,
    _schema_lock_timeout_ms,
)
from recall.observability import METRICS, _payload_bytes, current_performance_trace, get_logger
from recall.scope import Scope, coerce_scope, group_expression
from recall.types import Chunk, ScoredChunk
from recall.dsn_security import (  # noqa: F401  # re-exported: public names of this module
    _DEFAULT_CREDS,
    _LOCAL_HOSTS,
    _WARN_ONLY_HOSTS,
    _ENV_TRUE,
    INSECURE_DSN_OPT_OUT,
    _is_local_host,
    _dsn_passwords,
    scrub_dsn_secrets,
    redacted_dsn,
    warn_if_insecure_dsn,
    _env_opt_out,
    require_secure_dsn,
)
from recall.store_mutation import SourceMutationMixin, _chunk_identity  # noqa: F401
from recall.store_receipts import ReceiptsAuditMixin
from recall.store_supersession import SupersessionMixin

if TYPE_CHECKING:  # the pool extra is optional; the annotation must not require it at runtime
    from psycopg_pool import ConnectionPool

    from recall.pool import SharedPool

_NUMERIC_TOKEN_RE = re.compile(r"(?<![\w.])[+-]?\d+(?:[.,]\d+)?%?(?![\w.])")


class _CountingCursor:
    """Small cursor proxy for request-local statement and result byte accounting."""

    def __init__(self, cursor: Any, trace: Any) -> None:
        self._cursor = cursor
        self._trace = trace

    def __enter__(self) -> "_CountingCursor":
        self._cursor.__enter__()
        return self

    def __exit__(self, *args: Any) -> Any:
        return self._cursor.__exit__(*args)

    def _record(self, value: Any) -> Any:
        self._trace.add("db_result_bytes", _payload_bytes(value))
        return value

    def fetchone(self) -> Any:
        return self._record(self._cursor.fetchone())

    def fetchmany(self, size: int | None = None) -> Any:
        result = self._cursor.fetchmany() if size is None else self._cursor.fetchmany(size)
        return self._record(result)

    def fetchall(self) -> Any:
        return self._record(self._cursor.fetchall())

    def __iter__(self) -> Iterator[Any]:
        for row in self._cursor:
            yield self._record(row)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cursor, name)


class _CountingConnection:
    """Duck typed psycopg connection proxy used only while a performance trace is active."""

    def __init__(self, connection: Any, trace: Any) -> None:
        self._connection = connection
        self._trace = trace

    def execute(self, query: Any, params: Any = None, *args: Any, **kwargs: Any) -> _CountingCursor:
        self._trace.add("db_statement_count")
        self._trace.add("db_parameter_bytes", _payload_bytes(params))
        cursor = self._connection.execute(query, params, *args, **kwargs)
        return _CountingCursor(cursor, self._trace)

    def cursor(self, *args: Any, **kwargs: Any) -> _CountingCursor:
        return _CountingCursor(self._connection.cursor(*args, **kwargs), self._trace)

    @contextmanager
    def transaction(self, *args: Any, **kwargs: Any) -> Iterator[Any]:
        """Count transaction-control statements, which `execute` never sees.

        psycopg sends `BEGIN` and `COMMIT` (or `SAVEPOINT` and `RELEASE` when nested) for every
        `conn.transaction()`. Leaving them out made `db_statement_count` report a filtered dense
        query as three statements when it was five round trips.
        A nested rollback sends both `ROLLBACK TO` and `RELEASE`, including when
        requested by `force_rollback` or `psycopg.Rollback`.
        """
        status = getattr(
            getattr(self._connection, "info", None),
            "transaction_status",
            psycopg.pq.TransactionStatus.IDLE,
        )
        nested = status != psycopg.pq.TransactionStatus.IDLE
        self._trace.add("db_statement_count")
        try:
            with self._connection.transaction(*args, **kwargs) as tx:
                failed = False
                try:
                    yield tx
                except BaseException:
                    failed = True
                    raise
                finally:
                    if nested and (failed or getattr(tx, "force_rollback", False)):
                        self._trace.add("db_statement_count")  # RELEASE after ROLLBACK TO
        finally:  # COMMIT, ROLLBACK, or the nested block's RELEASE/ROLLBACK TO
            self._trace.add("db_statement_count")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._connection, name)


def _observed_db_call(connection: Any, operation: Callable[[Any], _T]) -> _T:
    trace = current_performance_trace()
    if trace is None:
        return operation(connection)
    return operation(_CountingConnection(connection, trace))


def _numeric_query_terms(text: str) -> list[str]:
    """Return normalized numeric terms for the table exact-match boost."""
    terms: list[str] = []
    for value in _NUMERIC_TOKEN_RE.findall(text):
        normalized = value.replace(",", ".")
        if normalized not in terms:
            terms.append(normalized)
    return terms


_T = TypeVar("_T")

#: Wall time of one retrieval leg, in ms. ONE series with a `leg` label rather than two names,
#: so the two legs of a hybrid query are compared without correlating separate metrics.
#:
#: Recorded by `PgVectorStore`, and inherited by `GenerationStore` because that subclass overrides
#: the PRIVATE `_query_*` methods — overriding the public pair would silently drop the timing on
#: the `RECALL_ENV=production` path.
#:
#: Scope, stated precisely because an earlier version of this comment overstated it: `METRICS.timer`
#: was ALREADY used for tool-level latency in `recall_mcp/server.py` (`recall_tool_latency_ms`).
#: What had no series was any leg INSIDE `recall/`. Note also that `HybridRetriever` records
#: per-query stage timings in `diagnostics.stage_ms`, so this metric is not the only per-leg
#: measurement — it adds process-wide aggregation (percentiles an operator can read via MCP) and
#: `LEG_META`, which no `stage_ms` bracket covers.
STORE_QUERY_METRIC = "recall_store_query_ms"
LEG_DENSE = "dense"
LEG_SPARSE = "sparse"
#: The LEARNED sparse leg (SPLADE), distinct from `LEG_SPARSE` (Postgres full-text / ts_rank).
#: Two different retrievers that are both "sparse"; collapsing them into one metric label would
#: make a latency or error-rate regression in either one unattributable.
LEG_LEARNED_SPARSE = "learned_sparse"

#: Global sidecar holding learned sparse vectors for every chunk table. See migration 0012 for why
#: it is global rather than per-target. Without a foreign key to the chunk table, erasure must
#: explicitly clean this table before the learned sparse path can be enabled in production.
SPARSE_TABLE = "recall_sparse_v1"

#: Vocabulary width of both supported checkpoints (bert-base-uncased). Fixed in the `sparsevec`
#: column type, so a model with a different vocabulary needs its own table, not a wider column.
SPARSE_DIM = 30522

#: How far past `k` the learned sparse HNSW walk is widened.
#:
#: MEASURED on the real index (183,408 clapnq rows inside a 366,479-row sidecar shared by four
#: corpora), asking for k=100:
#:
#:     ef_search   40 (pgvector default) ->   6 candidates
#:     ef_search  100                    ->  19
#:     ef_search  400                    ->  72
#:     ef_search 1000                    -> 100
#:
#: 10x, not the dense leg's 4x, because this index is shared: the tenant/chunk_table/profile_id
#: predicates and the strictly-positive-overlap filter discard most of what the walk visits, so
#: the scan runs out of budget long before it has k survivors. The dense leg scans a table that
#: is already one corpus, and needs less.
#:
#: ⚠️ Without this the leg silently returns a TWENTIETH of the candidates it was asked for. It
#: does not error; it reads on a benchmark as "learned sparse does not help", which is a wrong
#: conclusion manufactured by an index setting.
SPARSE_EF_SEARCH_MULTIPLIER = 10


def sparse_ef_search(k: int) -> int:
    """`hnsw.ef_search` for a learned sparse query returning `k` rows.

    Raise-only: never below pgvector's own default, and capped at what pgvector accepts rather
    than raising on a `k` that is otherwise legal.
    """
    widened = k * SPARSE_EF_SEARCH_MULTIPLIER
    return max(_PGVECTOR_DEFAULT_EF_SEARCH, min(widened, _HNSW_EF_SEARCH_MAX))


#: pgvector's HNSW ceiling on non-zero elements. Duplicated from `recall.sparse.HNSW_MAX_NONZERO`
#: ON PURPOSE: `recall.store` must not import `recall.sparse`, which would drag torch into the
#: import graph of every store user. The migration's CHECK constraint is the third copy and the
#: authoritative one — it is the only place the database itself enforces.
SPARSE_MAX_NONZERO = 1000
#: The THIRD store round trip on the search path, and the one an attribution silently misses.
#: `HybridRetriever.search` calls `newest_indexed_at()` once per query for its staleness report,
#: and that is an uncached `SELECT max(indexed_at)`. It is store work, and it is inside end-to-end
#: search latency, so leaving it untimed books a Postgres round trip as Python glue and UNDERSTATES
#: the store's share — in the same direction as the "the store is cheap" hypothesis this metric
#: exists to test. Same standard as the HNSW `set_config` widening, which is deliberately inside
#: the dense timer because it is part of what a dense retrieval costs.
#:
#: `count()` is deliberately NOT timed: it is not on the query path, so its samples would not
#: correspond one-to-one with queries and would break the per-query denominator that lets a
#: caller assert "one sample per leg per query".
LEG_META = "meta"

#: Re-scoring specific ids against a query vector. Its own leg rather than folded into
#: `LEG_META`: it is on the query path only for fused searches, so an operator comparing
#: `search` against `search_fused` needs to see it separately to know what fusion costs.
LEG_RESCORE = "rescore"

#: EVERY leg label `STORE_QUERY_METRIC` is emitted under, for callers that must drain all of them.
#:
#: `METRICS` is process-wide, so a caller measuring one configuration after another has to clear
#: the ring between them or the previous configuration's samples are averaged into the next one's.
#: Two callers did that against their own hand-written tuple of legs (`recall/eval/harness.py`
#: between ablation configurations, `benchmarks/store_latency_share.py` between its probes and the
#: measured run), and BOTH drifted when the learned sparse leg was added.
#:
#: Neither could fail loudly, which is why this is a constant rather than a convention: an
#: undrained series does not raise, it silently accumulates and contaminates a published mean. A
#: missing leg is invisible in exactly the direction that looks like a healthy run.
#:
#: Hand-maintained like `TIMED_PUBLIC_METHODS` and checked the same way —
#: `test_store_query_legs_matches_the_actual_timer_labels` parses this module and requires this
#: tuple to EQUAL the set of `leg=` labels the timers actually emit.
STORE_QUERY_LEGS = (LEG_DENSE, LEG_SPARSE, LEG_LEARNED_SPARSE, LEG_META, LEG_RESCORE)

#: Public methods that carry a `METRICS.timer` and delegate to a private twin. A subclass MUST
#: override the `_`-prefixed twin, NEVER the name listed here — overriding the public method
#: silently drops the timing, and an absent series reads exactly like a store that costs nothing.
#:
#: This is a tuple rather than a docstring because `GenerationStore` made that mistake TWICE: once
#: on the query legs, then again on `newest_indexed_at` immediately after the query legs were
#: fixed. The guard could not catch the second one because it enumerated two method names inline
#: instead of reading a list. `tests/test_store_query_latency.py` iterates this tuple.
#:
#: The tuple is HAND-MAINTAINED, so it bounds the guard only while it stays in step with the real
#: timer call sites — a declaration nothing checks is the same failure one level up, which is how
#: this recurred twice already. `test_timed_public_methods_matches_the_actual_timer_call_sites`
#: parses this module and requires this tuple to EQUAL the set of methods that actually open a
#: `METRICS.timer` on `STORE_QUERY_METRIC`.
TIMED_PUBLIC_METHODS = (
    "query_dense",
    "query_dense_exact",
    "query_sparse",
    "query_learned_sparse",
    "newest_indexed_at",
    "cosines_for",
)

_log = get_logger("store")


#: HNSW tuning applied to a filtered `query_dense` call (see issue #11's third checkbox). An HNSW
#: index walk is filter-blind: it finds the globally nearest neighbours, THEN discards the ones
#: that fail `WHERE tenant_id = ...` or `WHERE source = ...`, so a selective filter can silently
#: return fewer than `k` rows, or fewer true neighbours than exist. Measured on 20,000 rows / dim
#: 64 / a filter matching 10% of rows / 40 queries, recall@10 against an exact scan:
#:
#:   default (ef_search=40, iterative_scan=off) ..................... 0.385 recall, 40/40 truncated
#:   ef_search=200 alone ............................................. 0.942 recall,  1/40 truncated
#:   iterative_scan=relaxed_order alone .............................. 0.825 recall,  0/40 truncated
#:   iterative_scan=strict_order alone ................................0.568 recall,  0/40 truncated
#:   iterative_scan=relaxed_order + ef_search=200 (the defaults below) 0.947 recall,  0/40 truncated
#:
#: Neither knob alone is enough: `ef_search` widens the candidate list (fixes recall) but a
#: filtered scan can still exhaust it before reaching k matches (truncation); `iterative_scan`
#: re-widens the scan on exhaustion (fixes truncation) but not recall by itself. Both are needed
#: together. These two knobs apply to FILTERED queries, which includes the normal tenant-scoped
#: arm. The defensive unfiltered path below retains its own widening for `k` past ef_search's
#: default (see `_PGVECTOR_DEFAULT_EF_SEARCH`), but a real `PgVectorStore` never takes that path:
#: its tenant predicate is always present.
#:
#: Both are read at CALL time (not import time) via `os.environ`, matching how
#: `RECALL_INDEX_MAX_FILES` / `RECALL_INDEX_MAX_BYTES` are read in `recall_mcp/service.py`, so a
#: test can `monkeypatch.setenv` per-case and a long-lived process can pick up a changed value
#: without restarting.
DEFAULT_HNSW_EF_SEARCH_FILTERED = 200
#: pgvector's accepted range for `hnsw.ef_search` is 1..1000; a value outside it is rejected at
#: config time rather than reaching `hnsw.ef_search` and erroring on every filtered search.
_HNSW_EF_SEARCH_MAX = 1000
DEFAULT_HNSW_ITERATIVE_SCAN_FILTERED = "relaxed_order"
#: pgvector's own default for `hnsw.ef_search`, and therefore the point at which an UNFILTERED
#: query starts silently returning fewer rows than it asked for.
#:
#: An HNSW scan cannot return more rows than it examined, so `LIMIT k` with `k > ef_search`
#: yields ef_search rows — no error, no warning, no way for the caller to tell. The comment on
#: DEFAULT_HNSW_EF_SEARCH_FILTERED above says the unfiltered arm needs no tuning because it
#: "already measures recall 1.000 at ef_search's default". That measurement was taken at k=10;
#: it holds for k <= ef_search and does not generalise past it, which is the gap this closes.
#:
#: Measured on 72,151 chunks (FinanceBench, dim 1024): `query_dense(k=50)` returned exactly 40
#: rows for 150 of 150 queries, so a `HybridRetriever(candidate_k=50)` was really running at 40
#: and could not say so. Raising ef_search to cover k lifted that leg's page recall from 0.600
#: to 0.727 at k=50 and to 0.853 at k=100 — and the truncation had been manufacturing an
#: apparent advantage for the sparse leg, since pages the dense leg would have returned at
#: ranks 41-100 showed up as "only the sparse leg found this".
_PGVECTOR_DEFAULT_EF_SEARCH = 40

#: How far past `k` to widen an unfiltered HNSW scan.
#:
#: `ef_search = k` is not enough. It returns k rows, but they are not the true top-k: the walk is
#: explored just wide enough to fill the answer and then stops. Measured on the 72,151-chunk
#: FinanceBench index (dim 1024) — overlap with an exact scan's top-k, and the resulting
#: evidence-page hit@5, over 60 questions:
#:
#:        ef_search |  k=50            |  k=100
#:       -----------+------------------+------------------
#:               50 |  0.911 / 0.3000  |       -
#:              100 |  0.960 / 0.3167  |  0.951 / 0.3167
#:              200 |  0.989 / 0.3333  |  0.983 / 0.3333
#:              400 |  1.000 / 0.3667  |  1.000 / 0.3667   <- exact-scan parity, BOTH k
#:              800 |       -          |  1.000 / 0.3667
#:
#: Read the columns, not the multiples: parity arrives at ef_search ~= 400 for k=50 AND k=100, so
#: on this index the requirement is an ABSOLUTE search width — a property of the graph — and not a
#: multiple of k. 4x merely happened to coincide with 400 at k=100, which is exactly the kind of
#: agreement that reads as a law until a second k is measured.
#:
#: A multiplier is nonetheless the right shape for the DEFAULT, for two reasons: `ef_search >= k`
#: is the part that is universal (below it, rows are silently dropped — the bug this closes), and
#: the absolute floor is corpus-dependent, so no single number can be shipped honestly from one
#: index. 4x buys 0.989 overlap at k=50 and 1.000 at k=100 while keeping that guarantee.
#: Deployments wanting exact-scan parity should raise RECALL_HNSW_EF_SEARCH_MULTIPLIER, and the
#: figure to tune toward is an absolute width measured on their own corpus.
#:
#: UNPROVEN: one index, one embedder, two values of k. Whether ~400 generalises across corpus size
#: or across `m` / `ef_construction` is untested.
#:
#: Latency, same index: k=100 at 4x costs 24.3 ms median / 57.6 ms p95 against 6.4 ms at 1x, while
#: the sparse leg it is fused with costs ~496 ms median — the widened dense walk stays an order of
#: magnitude cheaper than its own fusion partner. At the shipped default `candidate_k=20` this
#: moves ef_search 40 -> 80 for no measurable change (11.2 ms -> 9.0 ms, i.e. noise).
DEFAULT_HNSW_EF_SEARCH_MULTIPLIER = 4
#: pgvector's only valid values for this GUC, checked by `_hnsw_filtered_tuning()` below. The
#: value now travels as a bound parameter to `set_config`, so this is no longer an injection
#: guard; it keeps a typo from surfacing as a query-time error on every filtered search.
_HNSW_ITERATIVE_SCAN_VALUES = frozenset({"off", "relaxed_order", "strict_order"})

#: `SET LOCAL` statements that take every ORDERED index-scan path off the planner's table for the
#: rest of the transaction. One property, two consumers: a query whose answer must be EXACT must
#: not be servable by the approximate ordering index (`recall_chunks_v1_embedding_idx` and its
#: legacy twin), whose walk is filter-blind and returns "reachable", not "nearest".
#:
#: `top_cosine` needs this because "an aggregate has no ORDER BY, so the ordering index cannot
#: serve it" turned out to be false. Postgres's min/max optimization (planagg.c) rewrites
#: `min(embedding <=> v)` into `ORDER BY embedding <=> v LIMIT 1` behind an InitPlan whenever
#: that path costs less than the plain aggregate, and pgvector's HNSW index serves exactly that
#: ORDER BY. Whether the rewrite fires is a COST decision, so it tracks table statistics: CI run
#: 32988012712 (2026-08-26, commit aecc9dc1) planned the aggregate through the index that a green
#: run of the same commit had seq-scanned twenty minutes earlier, with identical resolved wheels.
#: `tests/test_calibration_scoring_is_exact.py` demonstrates the rewrite deterministically.
#:
#: `_dense_exact_fallback` needs it for the same reason from the other side: when a filtered HNSW
#: walk has demonstrably failed (zero rows), the retry must be a plan the index cannot serve.
#:
#: `enable_indexscan` / `enable_indexonlyscan` are the only plan types that can satisfy an ORDER
#: BY from an index. Bitmap scans stay enabled on purpose: they cannot produce ordering, and they
#: are what serves the tenant/generation filter cheaply on a large corpus.
#:
#: ONE statement for both, because each is a round trip on the query path: `set_config(name, v,
#: true)` is exactly `SET LOCAL name = v`. Kept a tuple so callers still iterate it.
_EXACT_SCAN_GUARDS: tuple[str, ...] = (
    "SELECT set_config('enable_indexscan', 'off', true), "
    "set_config('enable_indexonlyscan', 'off', true)",
)

#: The filtered HNSW tuning, both GUCs in one round trip. `set_config(name, value, true)` is
#: `SET LOCAL`, and unlike `SET` it takes bound parameters, so the validated values travel as
#: parameters rather than being spliced into the statement text.
_HNSW_FILTERED_TUNING_SQL = (
    "SELECT set_config('hnsw.ef_search', %s, true), "
    "set_config('hnsw.iterative_scan', %s, true)"
)


#: PostgreSQL's OWN default trigger for an automatic analyze — `autovacuum_analyze_threshold` and
#: `autovacuum_analyze_scale_factor`. Mirrored (not invented) so that `analyze_if_stale` can only
#: ever issue an ANALYZE that autovacuum was already going to issue: the total amount of analyze
#: work is unchanged, and all that moves is WHEN it happens — at the end of the run that made the
#: rows stale, rather than up to an `autovacuum_naptime` (60s by default) afterwards.
#:
#: That equivalence is the whole justification for doing this in the foreground. An unconditional
#: ANALYZE would not have it: a server indexing one small file at a time into a large table would
#: pay for a statistics refresh on every call that autovacuum would have declined to make.
#:
#: Not env-tunable, and deliberately not read from the server's own settings. Both would be
#: precision this does not have — the values are per-table overridable, so any single reading can
#: be wrong, and being wrong here only means analyzing slightly more or less eagerly than
#: autovacuum would.
AUTOANALYZE_THRESHOLD = 50
AUTOANALYZE_SCALE_FACTOR = 0.1


def _ef_search_multiplier() -> int:
    """How far past `k` to widen an unfiltered HNSW scan; see DEFAULT_HNSW_EF_SEARCH_MULTIPLIER.

    Read at call time, not import time — same convention as `_hnsw_filtered_tuning`, so a test
    can `monkeypatch.setenv` per case and a long-lived process picks up a change without a
    restart. Values below 1 are refused rather than clamped: a 0 would silently disable the
    widening and reintroduce the truncation this exists to prevent.
    """
    raw = os.environ.get("RECALL_HNSW_EF_SEARCH_MULTIPLIER", str(DEFAULT_HNSW_EF_SEARCH_MULTIPLIER))
    try:
        mult = int(raw)
    except ValueError:
        raise ValueError(f"RECALL_HNSW_EF_SEARCH_MULTIPLIER={raw!r} is not an integer") from None
    if mult < 1:
        raise ValueError(f"RECALL_HNSW_EF_SEARCH_MULTIPLIER={mult} must be >= 1")
    return mult


def _first_write_sql(alias: str = "") -> str:
    """SQL for a row's first write: `first_indexed_at`, or `indexed_at` where it is NULL.

    The one definition both chunk tables read that date through when they need the fallback:
    `recall.generation_store` imports it, and the iterator the two stores share uses it. A
    generation row's `indexed_at` is the build that wrote or copied it, so a read that served it in
    place of `first_indexed_at` dated the whole corpus to the last rebuild (#835); one definition
    keeps a new read path from doing that again.
    """
    return f"COALESCE({alias}first_indexed_at, {alias}indexed_at)"


def _prepare_connection(
    conn: "psycopg.Connection", tenant: str, statement_timeout_ms: int | None
) -> None:
    """Per-connection setup, as a FREE function so a pool's `configure` need not hold the store.

    `ConnectionPool` keeps its `configure` callable for the pool's whole life. Passing a bound
    method here made `store -> _pool -> configure -> store` a reference cycle, so refcounting
    alone could never free a store and its connections; release waited on the cyclic collector.
    That is invisible while stores are long-lived and shows up when they are not: a store built
    and then rejected (a schema mismatch, say) held a live backend and its pool threads until a
    generational collection happened to run.

    Both values a connection needs are fixed at construction, so capturing them costs nothing and
    keeps the pool from pinning the store.
    """
    register_vector(conn)
    # Per-connection tenant for the RLS policy. Safe to set once at connection setup because a
    # store is bound to ONE tenant: the pool belongs to the store, so no connection is ever shared
    # between tenants. A server handling many tenants opens a store per tenant.
    conn.execute(f"SELECT set_config('{TENANT_GUC}', %s, false)", (tenant,))
    if statement_timeout_ms is not None:
        conn.execute(f"SET statement_timeout = {int(statement_timeout_ms)}")


class PgVectorStore(SourceMutationMixin, SupersessionMixin, ReceiptsAuditMixin):
    """The single, production-grade vector store: PostgreSQL + pgvector."""

    #: Errors that mean the connection itself is broken (dropped socket, server restart, idle
    #: timeout) — as opposed to a query/data error. These trigger a reconnect-and-retry.
    _CONN_ERRORS = (psycopg.OperationalError, psycopg.InterfaceError)

    #: Class-level default so `_pool` always exists, including on an instance built without
    #: __init__ (the store tests do this to exercise the retry logic against a fake connection).
    #: Single-connection mode is the default, so None is also the honest value.
    _pool = None
    #: Same reasoning for the shared-pool mode, and the same trap: `_with_retry` now reads
    #: `_shared` before `_pool`, so an instance built without `__init__` would raise
    #: AttributeError on the very first branch and mask whatever the test was actually asserting.
    _shared = None
    _owns_shared = False
    #: The chunk table's identifier and source columns. `GenerationStore`'s table names them
    #: `chunk_id` and `source_uri`; every read the two classes share names its columns through
    #: these and confines its rows through `_row_scope`, so one statement serves both tables.
    _ID_COL = "id"
    _SOURCE_COL = "source"

    def __init__(
        self,
        dsn: str,
        dim: int,
        table: str = DEFAULT_TABLE,
        *,
        tenant: str = DEFAULT_TENANT,
        pool_size: int | None = None,
        statement_timeout_ms: int | None = None,
        connect_timeout_s: int | None = 10,
        generation_id: str = "legacy",
        dependency_mode: str | None = None,
        shared_pool: "SharedPool | None" = None,
        owns_pool: bool = False,
    ) -> None:
        """Open a store against `dsn`.

        `pool_size` selects the CONNECTION MODE, and the default (None) is deliberate:

        - **None — one long-lived connection.** Correct for a CLI or any single-threaded caller,
          and the mode whose reconnect semantics the rest of this class is built around. Sharing
          it across threads serialises them, and a reconnect swaps `self._conn` underneath a
          thread that is using it.
        - **an int — a connection pool.** What a server needs: each operation borrows its own
          connection, so concurrent callers actually proceed concurrently. Opt-in rather than
          default because a pool starts a background maintenance thread, which a one-shot CLI
          invocation should not pay for or have to shut down.

        `statement_timeout_ms` bounds every statement server-side. Without it a single runaway
        query occupies a connection until the process dies, with nothing to cancel it; that is
        the difference between a slow request and an exhausted pool.

        Normal serving code calls the read-only ``check_schema()`` compatibility gate. The
        deprecated ``ensure_schema()`` method is an explicit v0.8 compatibility wrapper around
        the versioned migrator; operators should use ``recall schema apply`` with a separate DSN.
        """
        # `dim` and `table` are interpolated directly into SQL — as a type modifier and an
        # identifier respectively — because Postgres cannot bind those as parameters. They
        # are therefore strictly validated here: this is the SQL-injection guard. Every
        # other value in this class is passed via psycopg bound parameters, never formatted.
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError("dim must be a positive int")
        if not table.isidentifier():
            raise ValueError("table must be a valid SQL identifier")
        if pool_size is not None and (not isinstance(pool_size, int) or pool_size < 1):
            raise ValueError("pool_size must be a positive int or None")
        if not isinstance(tenant, str) or not tenant:
            raise ValueError("tenant must be a non-empty str")
        if not isinstance(generation_id, str) or not generation_id:
            raise ValueError("generation_id must be a non-empty str")
        if dependency_mode is not None and dependency_mode not in {"off", "enforce"}:
            raise ValueError("dependency_mode must be 'off', 'enforce', or None")
        self._dsn = dsn
        self._dim = dim
        self._table = table
        self._tenant = tenant
        self._index_generation_id = generation_id
        self._dependency_mode = dependency_mode
        self._statement_timeout_ms = statement_timeout_ms
        self._connect_timeout_s = connect_timeout_s
        #: (fingerprint, SupersessionGraph): see `supersession_graph()`. The fingerprint is what
        #: makes the cache safe to reuse across processes.
        self._supersession_cache: tuple | None = None
        #: Count of full supersession scans actually performed (cache misses). Surfaced so a
        #: test can prove the cache still works, and so a rescan storm is visible as a metric.
        self._supersession_scans = 0
        self._closed = False
        #: Third connection mode, and the one a multi-tenant server should use. When set, this
        #: store owns no connections at all: every operation borrows from a process-wide pool and
        #: runs in a transaction that carries the tenant as a `SET LOCAL`. See `recall.pool`.
        self._shared = shared_pool
        #: Whether THIS store closes the shared pool. False by default because the common owner
        #: is whoever constructed the pool (the registry), and a view made by `for_tenant` must
        #: never close it out from under its siblings. Without an explicit flag this was
        #: unreachable: nothing set it True, so the ownership branch in `close()` was dead code
        #: and a library caller who built their own pool could not release it at all.
        self._owns_shared = bool(owns_pool) and shared_pool is not None
        if shared_pool is not None:
            if pool_size:
                raise ValueError("pass either pool_size or shared_pool, not both")
            self._pool = None
            self._conn = None
        else:
            self._pool = self._open_pool(pool_size) if pool_size else None
            self._conn = None if self._pool is not None else self._connect()

    def _connect_kwargs(self) -> dict:
        kw: dict = {"autocommit": True}
        if self._connect_timeout_s is not None:
            # Without this a dead host hangs the caller on the TCP handshake indefinitely.
            kw["connect_timeout"] = self._connect_timeout_s
        return kw

    def _open_pool(self, size: int) -> "ConnectionPool":
        try:
            from psycopg_pool import ConnectionPool
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                'pool_size requires the pool extra: pip install "recall-rag[pool]"'
            ) from exc

        # `configure` runs on every connection the pool creates, not just the first — the vector
        # type registration is per-connection state, so a pool that skipped it would work until
        # it opened its second connection and then fail on a Vector parameter.
        pool = ConnectionPool(
            self._dsn,
            min_size=1,
            max_size=size,
            kwargs=self._connect_kwargs(),
            # `partial` over a free function, NOT `self._prepare`: a bound method would make the
            # pool hold the store (`store -> _pool -> configure -> store`), a cycle only the
            # collector can break. See `_prepare_connection`.
            configure=partial(
                _prepare_connection,
                tenant=self._tenant,
                statement_timeout_ms=self._statement_timeout_ms,
            ),
            open=False,
        )
        pool.open(wait=True, timeout=self._connect_timeout_s or 30)
        return pool

    def _prepare(self, conn: "psycopg.Connection") -> None:
        """Per-connection setup: vector codec, tenant context and statement timeout.

        Deliberately contains no DDL. A missing pgvector extension is a pending migration, not
        something a serving credential is allowed to repair during startup.
        """
        _prepare_connection(conn, self._tenant, self._statement_timeout_ms)

    def _connect(self) -> "psycopg.Connection":
        """Open one autocommit serving connection and prepare its session state."""
        conn = psycopg.connect(self._dsn, **self._connect_kwargs())
        try:
            self._prepare(conn)  # same per-connection setup the pool applies via `configure`
        except Exception:  # BROAD-CATCH: fail-closed
            conn.close()
            raise
        return conn

    @property
    def _direct(self) -> "psycopg.Connection":
        """The single long-lived connection, for the non-pooled mode only.

        `__init__` establishes the invariant "`_conn` is not None exactly when `_pool` is None",
        and every caller below is already on the non-pooled branch — but that is a correlation
        between two attributes, which no type checker can follow. This accessor states it once
        and turns a violation into a named error instead of `AttributeError: 'NoneType'`.
        """
        if self._conn is None:  # pragma: no cover - unreachable while the invariant holds
            raise RuntimeError("no direct connection: this store is in pooled mode")
        return self._conn

    def _reconnect(self) -> None:
        """Discard the (broken) connection and open a fresh, prepared one."""
        try:
            self._direct.close()
        except Exception:  # BROAD-CATCH: fail-open
            pass
        self._conn = self._connect()

    def _with_retry(self, op: Callable[["psycopg.Connection"], _T]) -> _T:
        """Run ``op(conn)``; on a broken-connection error, reconnect once and retry.

        A single long-lived autocommit connection can be severed by the server (idle timeout,
        restart, transient network blip). Rather than failing every subsequent call, transparently
        reconnect and retry the operation exactly once — a second failure propagates. Safe because
        every ``op`` here is a single self-contained statement or an atomic transaction that rolls
        back cleanly, so re-running it on a fresh connection cannot double-apply.

        The retry is deliberately narrow. ``OperationalError`` is NOT a synonym for "the
        connection is gone" — ``QueryCanceled`` (statement_timeout), ``DeadlockDetected`` and
        ``SerializationFailure`` are all subclasses raised on a perfectly LIVE connection.
        Retrying those re-runs the statement on a fresh session that no longer carries the
        setting which killed it, i.e. silently escapes the very guard that fired. So the retry
        additionally requires the connection to be observably dead.

        A reconnect is REPORTED to stderr: a silent one hides an outage behind a process that
        still looks healthy, which is how a dead dependency goes unnoticed for days.
        """
        if self._closed:
            raise RuntimeError("store is closed")
        if self._shared is not None:
            return self._with_retry_shared(op)
        if self._pool is not None:
            return self._with_retry_pooled(op)
        try:
            return _observed_db_call(self._direct, op)
        except self._CONN_ERRORS:
            # getattr: `broken` only exists from psycopg 3.2 and the declared floor is 3.1 —
            # without the default this except-block would raise AttributeError and mask the
            # original database error on an older install.
            if not (self._direct.closed or getattr(self._direct, "broken", False)):
                raise
            _log.warning("database connection lost — reconnecting")
            METRICS.increment("recall_db_reconnects_total")
            self._reconnect()
            return _observed_db_call(self._direct, op)

    def _with_retry_pooled(self, op: Callable[["psycopg.Connection"], _T]) -> _T:
        """Pooled variant: borrow a connection per operation, retry once on a dead one.

        The pool already replaces connections it knows are broken, but it can hand out one that
        died while idle and has not been probed yet, so the first use still fails. Retrying
        borrows a DIFFERENT connection — the pooled equivalent of reconnecting.

        The same narrow predicate as the single-connection path applies, and for the same
        reason: `QueryCanceled` from `statement_timeout` is an `OperationalError` raised on a
        perfectly live connection, and retrying it on a fresh session would silently escape the
        timeout that fired. So a retry additionally requires the connection to be observably
        dead. Nothing here is a `nonlocal` on shared state: each borrow is thread-confined,
        which is the whole point of the mode.
        """
        pool = self._pool
        assert pool is not None  # caller checked; restates the mode invariant for the checker
        for attempt in (0, 1):
            with pool.connection() as conn:
                try:
                    return _observed_db_call(conn, op)
                except self._CONN_ERRORS:
                    dead = conn.closed or getattr(conn, "broken", False)
                    if not dead or attempt == 1:
                        raise
                    _log.warning("pooled database connection lost — retrying on another")
                    METRICS.increment("recall_db_reconnects_total", pooled="true")
        raise AssertionError("unreachable: the loop either returns or raises")  # pragma: no cover

    def _with_retry_shared(self, op: Callable[["psycopg.Connection"], _T]) -> _T:
        """Shared-pool variant: one transaction per operation, tenant scoped to that transaction.

        The difference from `_with_retry_pooled` is not the retry, it is what the borrow means. A
        per-tenant pool could set the tenant once per CONNECTION because no other tenant would
        ever use it. A shared pool cannot, so the tenant is set per TRANSACTION with `SET LOCAL`
        and the database discards it at COMMIT or ROLLBACK. `recall.pool` explains why that is the
        property that makes sharing safe at all.

        The retry predicate is deliberately the same narrow one as both other modes: an
        `OperationalError` is only retried when the connection is observably dead. `QueryCanceled`
        from `statement_timeout` is an `OperationalError` on a perfectly live connection, and
        retrying it would silently escape the guard that fired.
        """
        shared = self._shared
        assert shared is not None  # caller checked; restates the mode invariant for the checker
        for attempt in (0, 1):
            # Decided INSIDE the lease, deliberately. Reading `conn.closed` after the context
            # manager exits inspects an object the pool has already taken back — and because a
            # pool configured with a `reset` hook returns connections on a background worker
            # thread, that read is a race in both directions: a genuinely dead connection usually
            # still reports alive, and if the worker wins, a `QueryCanceled` from
            # `statement_timeout` gets judged "dead" and re-run on a fresh transaction, which is
            # exactly the escape from the timeout this predicate exists to prevent.
            retryable = False
            try:
                with shared.tenant_transaction(self._tenant) as conn:
                    trace = current_performance_trace()
                    if trace is not None:
                        # BEGIN, the tenant `set_config` and COMMIT, sent by
                        # `tenant_transaction` outside the counted proxy.
                        trace.add("db_statement_count", 3)
                    try:
                        return _observed_db_call(conn, op)
                    except self._CONN_ERRORS:
                        retryable = conn.closed or getattr(conn, "broken", False)
                        raise
            except self._CONN_ERRORS:
                # The COMMIT happens in `tenant_transaction.__exit__`, i.e. OUTSIDE the inner try,
                # so a failure there leaves `retryable` False and propagates unretried. That is
                # required, not incidental: a connection error at commit time has an indeterminate
                # outcome, and re-running the whole mutation could double-apply it.
                if not retryable or attempt == 1:
                    raise
                _log.warning("shared-pool database connection lost — retrying on another")
                METRICS.increment("recall_db_reconnects_total", pooled="true")
        raise AssertionError("unreachable: the loop either returns or raises")  # pragma: no cover

    def for_tenant(self, tenant: str) -> "PgVectorStore":
        """A view of this store bound to another tenant, sharing the same pool.

        A VIEW, never a new pool — that is the whole point, and it is what makes 1,000 tenants a
        configuration rather than 1,000 pools. The returned store shares `_shared`, so the process
        holds one set of connections no matter how many tenants it serves.

        Only available in shared-pool mode. In the other two modes the tenant is baked into the
        connection's session state, so a "view" would silently read as the wrong tenant; refusing
        is the honest answer rather than returning something that looks like a view and is not.

        The view does not own the pool, so closing it is a no-op on the connections. Closing the
        store that owns the pool closes it for every view, which is the correct coupling: they are
        one process's connections.
        """
        if self._shared is None:
            raise RuntimeError(
                "for_tenant() requires shared-pool mode: in single-connection and per-store "
                "pool modes the tenant is session state on the connection, so a view would "
                "read as the wrong tenant. Construct the store with shared_pool=..."
            )
        if not isinstance(tenant, str) or not tenant.strip():
            raise ValueError("tenant must be a non-empty str")
        view = object.__new__(type(self))
        view.__dict__.update(self.__dict__)
        view._tenant = tenant
        view._owns_shared = False
        view._reset_tenant_state()
        return view

    def _reset_tenant_state(self) -> None:
        """Drop every piece of state derived from the PREVIOUS tenant. Subclasses must extend.

        `for_tenant` copies `__dict__` wholesale, which is what makes a view cheap and is also
        what makes this hook necessary: anything tenant-derived is inherited by reference unless
        it is named here. Enumerating by hand at the call site is how the supersession cache got
        cleared and `GenerationStore._pinned_generation` did not — one tenant's pinned generation
        then governed another tenant's reads. A hook a subclass can override turns that from a
        list someone must remember to extend into one the subclass owns.
        """
        self._supersession_cache = None
        self._supersession_scans = 0

    @property
    def table(self) -> str:
        return self._table

    @property
    def tenant(self) -> str:
        """The tenant this store reads and writes as.

        Exposed so a caller can NAME it in an error. A guard that reports "this tenant already
        holds data" without saying which tenant sends the reader hunting through config for a
        value the store already knows.
        """
        return self._tenant

    @property
    def generation_id(self) -> str:
        return self._index_generation_id

    @property
    def dsn(self) -> str:
        """Connection string used by tenant-bound ledger adapters."""
        return self._dsn

    def chunk_by_id(self, chunk_id: str) -> Chunk | None:
        """Return one tenant and generation bound chunk without exposing its embedding."""
        if not isinstance(chunk_id, str) or not chunk_id:
            raise ValueError("chunk_id must be a non-empty string")
        row = self._with_retry(
            lambda conn: conn.execute(
                f"SELECT id, source, text, metadata FROM {self._table} "
                "WHERE tenant_id = %s AND id = %s",
                (self._tenant, chunk_id),
            ).fetchone()
        )
        if row is None:
            return None
        return Chunk(id=row[0], source=row[1], text=row[2], metadata=row[3] or {})

    def chunks_by_ids(self, chunk_ids: Sequence[str]) -> dict[str, Chunk]:
        """Return a tenant scoped batch of chunks without exposing embeddings.

        Serving paths often have a bounded set of opaque candidate ids after ranking. Fetching
        that set in one statement keeps database round trips constant and keeps text transfer
        proportional to the admitted candidates rather than the corpus.
        """
        if isinstance(chunk_ids, (str, bytes, bytearray)):
            raise ValueError("chunk_ids must be a sequence of strings")
        wanted = list(dict.fromkeys(chunk_ids))
        if any(not isinstance(chunk_id, str) or not chunk_id for chunk_id in wanted):
            raise ValueError("chunk_ids must contain only non-empty strings")
        if not wanted:
            return {}
        scope, scope_args = self._row_scope_args()
        rows = self._with_retry(
            lambda conn: conn.execute(
                f"SELECT {self._ID_COL}, {self._SOURCE_COL}, text, metadata FROM {self._table} "
                f"WHERE tenant_id = %s{scope} AND {self._ID_COL} = ANY(%s)",
                (self._tenant, *scope_args, wanted),
            ).fetchall()
        )
        found = {
            str(row[0]): Chunk(id=row[0], source=row[1], text=row[2], metadata=row[3] or {})
            for row in rows
        }
        return {chunk_id: found[chunk_id] for chunk_id in wanted if chunk_id in found}

    def chunk_metadata_by_ids(self, chunk_ids: Sequence[str]) -> dict[str, Chunk]:
        """Return candidate identity and metadata without transferring passage text."""
        if isinstance(chunk_ids, (str, bytes, bytearray)):
            raise ValueError("chunk_ids must be a sequence of strings")
        wanted = list(dict.fromkeys(chunk_ids))
        if any(not isinstance(chunk_id, str) or not chunk_id for chunk_id in wanted):
            raise ValueError("chunk_ids must contain only non-empty strings")
        if not wanted:
            return {}
        scope, scope_args = self._row_scope_args()
        rows = self._with_retry(
            lambda conn: conn.execute(
                f"SELECT {self._ID_COL}, {self._SOURCE_COL}, metadata FROM {self._table} "
                f"WHERE tenant_id = %s{scope} AND {self._ID_COL} = ANY(%s)",
                (self._tenant, *scope_args, wanted),
            ).fetchall()
        )
        found = {
            str(row[0]): Chunk(id=row[0], source=row[1], text="", metadata=row[2] or {})
            for row in rows
        }
        return {chunk_id: found[chunk_id] for chunk_id in wanted if chunk_id in found}

    def chunks_for_source(self, source: str) -> list[Chunk]:
        """Return all chunks for one exact source in stable ingestion order.

        Hosted ingestion uses one opaque source per session.  This bounded read gives a compiler
        the earlier records for that session without weakening the tenant boundary or searching
        unrelated memories.
        """
        if not isinstance(source, str) or not source:
            raise ValueError("source must be a non-empty string")
        rows = self._with_retry(
            lambda conn: conn.execute(
                f"SELECT id, source, text, metadata FROM {self._table} "
                "WHERE tenant_id = %s AND source = %s ORDER BY indexed_at, id",
                (self._tenant, source),
            ).fetchall()
        )
        return [Chunk(id=row[0], source=row[1], text=row[2], metadata=row[3] or {}) for row in rows]

    def compiled_chunks_for_source_newest_first(
        self,
        source: str,
        *,
        limit: int,
        before: tuple[datetime, str] | None = None,
    ) -> list[tuple[datetime, Chunk]]:
        """One page of a source's compiled chunks, newest first, with each row's ``indexed_at``.

        The order is exactly ``chunks_for_source`` reversed (``indexed_at, id``), restricted to rows
        whose ``record_type`` is ``compiled``. ``before`` is the ``(indexed_at, id)`` of the last
        row of the previous page, so a caller that only needs a session's latest records can page
        backwards instead of reading every row the session ever wrote.
        """
        if not isinstance(source, str) or not source:
            raise ValueError("source must be a non-empty string")
        if limit <= 0:
            raise ValueError("limit must be positive")
        cursor = "" if before is None else " AND (indexed_at, id) < (%s, %s)"
        params: tuple[Any, ...] = (self._tenant, source, *(before or ()), limit)
        rows = self._with_retry(
            lambda conn: conn.execute(
                f"SELECT id, source, text, metadata, indexed_at FROM {self._table} "  # noqa: S608
                "WHERE tenant_id = %s AND source = %s "
                f"AND metadata->>'record_type' = 'compiled'{cursor} "
                "ORDER BY indexed_at DESC, id DESC LIMIT %s",
                params,
            ).fetchall()
        )
        return [
            (row[4], Chunk(id=row[0], source=row[1], text=row[2], metadata=row[3] or {}))
            for row in rows
        ]

    def delete_tenant_data(self) -> int:
        """Delete this tenant's hosted chunks and idempotency receipts atomically.

        The method intentionally leaves other tenants untouched and does not drop shared schema.
        It also scrubs learned sparse sidecars when present, even though the hosted-quality
        profile does not create them.
        """
        self._supersession_cache = None

        def _op(conn: "psycopg.Connection") -> int:
            with conn.transaction():
                ids = [
                    str(row[0])
                    for row in conn.execute(
                        f"SELECT id FROM {self._table} WHERE tenant_id = %s", (self._tenant,)
                    ).fetchall()
                ]
                self._scrub_sparse_rows(conn, self._table, ids)
                deleted = (
                    conn.execute(
                        f"DELETE FROM {self._table} WHERE tenant_id = %s", (self._tenant,)
                    ).rowcount
                    or 0
                )
                conn.execute(
                    "DELETE FROM recall_idempotency_receipts WHERE tenant_id = %s",
                    (self._tenant,),
                )
                return int(deleted)

        return self._with_retry(_op)

    def explicit_superseded_chunk_ids(self) -> frozenset[str]:
        """Return hosted record ids explicitly superseded anywhere in this tenant.

        This is intentionally independent of the retrieval candidate pool. Otherwise an obsolete
        record can be served whenever its successor ranks at candidate 101, even though the
        corpus has an explicit edge saying not to use it.
        """
        rows = self._with_retry(
            lambda conn: conn.execute(
                f"SELECT DISTINCT jsonb_array_elements_text(metadata->'supersedes') "
                f"FROM {self._table} WHERE tenant_id = %s "
                "AND jsonb_typeof(metadata->'supersedes') = 'array'",
                (self._tenant,),
            ).fetchall()
        )
        return frozenset(str(row[0]) for row in rows if row and row[0])

    def dependency_invalidation_mode(self) -> str | None:
        """Return the optional mode bound to this store or generation view.

        ``None`` deliberately means that the trust layer may use its process configuration. An
        explicit constructor value binds the mode to this tenant and generation store, which is
        the safe deployment shape for serving more than one tenant in one process.
        """
        return self._dependency_mode

    def close(self) -> None:
        """Close the connection (or pool) for good.

        Sticky by design: without the flag, any later call would hit `_with_retry`'s reconnect
        and silently resurrect a connection nobody owns — a leak on first accidental reuse.
        """
        self._closed = True
        if self._shared is not None:
            # A view owns nothing; closing it must not take the process's connections with it.
            # The owner closes the pool, and every view sharing it becomes unusable, which is the
            # correct coupling: they are one process's connections, not one store's.
            if self._owns_shared:
                self._shared.close()
        elif self._pool is not None:
            self._pool.close()  # also stops the pool's background maintenance thread
        else:
            self._direct.close()

    def drop_table(self) -> None:
        """Drop this store's table if it exists.

        Exists so callers stop reaching into `store._conn` to do it (the eval harness, the
        calibration runner, the semantic linter and the test fixtures all did). That reach-through
        is not merely untidy: in pooled mode there IS no `_conn`, so every one of those call
        sites would raise `AttributeError` on a store configured for a server.
        """
        self._supersession_cache = None

        def _drop(conn: "psycopg.Connection") -> None:
            # Disposable eval/test tables may be recreated with the same name. Remove their
            # migration target atomically with the table; otherwise the ledger says every phase
            # is applied and the next explicit ensure_schema() correctly skips all SQL, leaving
            # the requested table absent.
            with conn.transaction():
                # The learned sparse sidecar cannot cascade: its parent is a column VALUE
                # (`chunk_table`), not a relation, so there is no foreign key to fire. Without
                # this DELETE every throwaway store leaves a uuid-named row set addressable by a
                # name that no longer resolves, and nothing ever looks for them again.
                #
                # The absent tenant filter is deliberate, and it works: `DROP TABLE` below is
                # DDL and removes the table for every tenant, so a cleanup scoped to one would
                # strand the rest. MEASURED rather than reasoned, because the sidecar does carry
                # FORCE ROW LEVEL SECURITY with a tenant isolation policy (migration 0012) and
                # that looks like it should narrow this statement: two tenants were given a
                # sidecar row on one chunk table, one tenant's store dropped it, and both rows
                # were gone. The roles this code runs under (local dev, CI, the test container)
                # are superuser and BYPASSRLS, and RLS does not apply to them, FORCE or not.
                # ⚠️ That is the CONDITION to watch. Under a role that is neither superuser nor
                # BYPASSRLS the policy would engage, this DELETE would silently narrow to one
                # tenant, and the drop would strand every other tenant's rows. Nothing here
                # detects that; it is a property of the connecting role.
                sidecar = conn.execute(f"SELECT to_regclass('{SPARSE_TABLE}')").fetchone()
                if sidecar and sidecar[0]:
                    conn.execute(
                        f"DELETE FROM {SPARSE_TABLE} WHERE chunk_table = %s", (self._table,)
                    )
                conn.execute(f"DROP TABLE IF EXISTS {self._table}")
                ledger = conn.execute("SELECT to_regclass('recall_schema_migrations')").fetchone()
                if ledger and ledger[0]:
                    conn.execute(
                        "DELETE FROM recall_schema_migrations WHERE target_table = %s",
                        (self._table,),
                    )

        self._with_retry(_drop)

    def __enter__(self) -> "PgVectorStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def ensure_schema(self) -> None:
        """Explicitly apply versioned migrations using this store's DSN.

        Deprecated for production provisioning: use ``recall schema apply`` with a dedicated
        migration DSN. The compatibility wrapper remains for v0.8 callers and disposable
        benchmark/test stores; importantly, it no longer contains an independent runtime DDL
        implementation — every change goes through the checksum-verified migration ledger.
        """
        from recall.schema import apply_migrations

        apply_migrations(self._dsn, table=self._table, dim=self._dim)

    def check_schema(self) -> None:
        """Verify schema compatibility using SELECT statements only."""
        from recall.schema import check_schema

        self._with_retry(lambda conn: check_schema(conn, table=self._table, dim=self._dim))

    def migrate_schema(self, migration_dsn: str | None = None) -> None:
        """Explicitly apply versioned migrations for this store's table.

        Kept as a convenience for disposable benchmark/test tables. Production provisioning
        should use ``recall schema apply`` so the migration and serving credentials stay visibly
        separate.
        """
        from recall.schema import apply_migrations

        apply_migrations(migration_dsn or self._dsn, table=self._table, dim=self._dim)

    def check_rls_effective(self) -> bool:
        """True when row-level security actually constrains THIS connection's role.

        Returns False for a superuser or a `BYPASSRLS` role — for whom the policy created above
        is inert. Exposed rather than merely documented because "we enabled RLS" is the kind of
        claim that gets believed without being true, and the difference is invisible until a
        tenant reads another tenant's memory.
        """
        row = self._with_retry(
            lambda conn: conn.execute(
                "SELECT rolsuper OR rolbypassrls FROM pg_roles WHERE rolname = current_user"
            ).fetchone()
        )
        return not (row and row[0])

    def readiness_facts(self) -> dict[str, object]:
        """Catalog facts used by enterprise readiness without exposing corpus content."""
        expected = {f"{self._table}_tsv_idx", f"{self._table}_emb_idx"}

        def _op(conn: "psycopg.Connection") -> dict[str, object]:
            table_row = conn.execute(
                "SELECT c.relrowsecurity, c.relforcerowsecurity, a.atttypmod "
                "FROM pg_class c JOIN pg_attribute a ON a.attrelid = c.oid "
                "AND a.attname = 'embedding' AND NOT a.attisdropped "
                "WHERE c.oid = %s::regclass",
                (self._table,),
            ).fetchone()
            indexes = conn.execute(
                "SELECT c.relname, i.indisvalid FROM pg_index i "
                "JOIN pg_class c ON c.oid = i.indexrelid "
                "WHERE i.indrelid = %s::regclass",
                (self._table,),
            ).fetchall()
            counts = conn.execute(
                f"SELECT count(*), count(*) FILTER (WHERE NOT metadata ? 'embedding_profile') "
                f"FROM {self._table} WHERE tenant_id = %s",
                (self._tenant,),
            ).fetchone()
            valid = {str(name) for name, is_valid in indexes if is_valid}
            return {
                "rls_enabled": bool(table_row and table_row[0] and table_row[1]),
                "indexes_valid": expected.issubset(valid),
                "dimension": int(table_row[2]) if table_row and table_row[2] > 0 else None,
                "rows": int(counts[0]) if counts else 0,
                "rows_without_profile": int(counts[1]) if counts else 0,
            }

        return self._with_retry(_op)

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        self._supersession_cache = None  # metadata may change; recompute on next read
        self._with_retry(lambda conn: self._upsert_in(conn, chunks, embeddings))
        return len(chunks)

    def _upsert_in(
        self, conn: "psycopg.Connection", chunks: list[Chunk], embeddings: list[list[float]]
    ) -> None:
        # NUL bytes cannot be stored in a PostgreSQL text column, and psycopg's own error names
        # neither the row nor the source — indexing a real 792-file corpus failed on TWO bytes in
        # ONE file with no indication of which. Fail here instead, pointing at the chunk. The
        # Indexer strips them before this (with a warning); this catches the direct-API caller.
        for c in chunks:
            if "\x00" in c.text or "\x00" in c.id or "\x00" in c.source:
                raise ValueError(
                    f"chunk {c.id!r} from source {c.source!r} contains a NUL (0x00) byte, which "
                    f"PostgreSQL text columns cannot store — strip it before upserting"
                )
        # One transaction for the whole batch: a mid-loop failure rolls the batch back
        # instead of leaving earlier rows committed (the connection is autocommit). When called
        # from replace_sources' outer transaction this becomes a savepoint (same commit).
        t = self._table
        # executemany, not a Python loop of execute(): psycopg3 pipelines it into one round
        # trip per batch instead of one per row. A full re-index is thousands of rows, and at
        # ~0.2-0.5ms of round-trip each that loop was seconds of pure latency.
        with conn.transaction(), conn.cursor() as cur:
            cur.executemany(
                f"""
                INSERT INTO {t}
                    (tenant_id, id, source, text, metadata, embedding,
                     indexed_at, first_indexed_at)
                VALUES (%s, %s, %s, %s, %s, %s, now(), now())
                ON CONFLICT (tenant_id, id) DO UPDATE SET
                    source = EXCLUDED.source,
                    text = EXCLUDED.text,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding,
                    indexed_at = now(),
                    -- LEAST over COALESCE, so re-writing a chunk never moves its first-seen
                    -- forward: a stored date IN THE PAST always wins (a stored date in the
                    -- FUTURE does not, and clock skew or a restore can produce one). A row
                    -- predating the column has NULL, and COALESCE gives it its own
                    -- `indexed_at` — the same fallback both read paths use — rather than
                    -- stamping it with this write, which would claim a memo written in January
                    -- was first seen the day someone re-indexed it.
                    first_indexed_at = LEAST(
                        COALESCE({t}.first_indexed_at, {t}.indexed_at),
                        EXCLUDED.first_indexed_at
                    )
                """,
                [
                    (self._tenant, c.id, c.source, c.text, json.dumps(c.metadata), Vector(e))
                    for c, e in zip(chunks, embeddings)
                ],
            )

    def analyze(self) -> bool:
        """Refresh the planner's statistics for this table. Best-effort; never raises.

        Worth doing explicitly because the planner's choice for `query_dense`'s source-filtered
        arm is decided by these statistics, and a freshly built table has none. Postgres reports
        `reltuples = -1` / `relpages = 0` and carries no `pg_stats` row for `source`, so it
        estimates ONE matching row and picks an exact plan (a Bitmap Heap Scan + Sort, cost ~15)
        over `Index Scan using <table>_emb_idx`. The answers stay correct — an exact scan is an
        exact search — but the HNSW index is not consulted at all, which also makes the
        `hnsw.ef_search` / `hnsw.iterative_scan` tuning in `_query_dense` inert. On a 20,000-row
        corpus that costs a millisecond or two per query; on a large one it is a full scan plus a
        sort of every matching row, per query, until autovacuum's analyze lands.

        Best-effort on purpose. This is an optimisation, not a correctness step: an index run
        that wrote every row correctly must not be reported as failed because a statistics
        refresh did not land, and autovacuum will make the same refresh in the background
        regardless — which is exactly today's behaviour, so failing soft is strictly better than
        never trying.

        `statement_timeout` is deliberately NOT lifted, unlike `ensure_schema`'s DDL. That lift
        is justified by an HNSW build being genuinely unbounded and by its failure stranding an
        INVALID index; neither applies here. ANALYZE samples a bounded number of rows
        (30,000 at the default `default_statistics_target`) whatever the table's size, so it fits
        an ordinary statement budget, and lifting a timeout for a pure optimisation is the wrong
        trade.

        ⚠️ Returns whether the statement COMPLETED, which is not the same as "statistics were
        refreshed". A role that does not own the table gets `WARNING: permission denied to
        analyze "<table>", skipping it` and a successful return — verified against the container,
        not assumed. Nothing is broken by that: `reltuples` stays -1, so `analyze_if_stale` keeps
        finding the table never-analyzed and retrying, and a permission-denied ANALYZE returns
        immediately. Autovacuum, which runs as a role that does have the privilege, still does
        the real work.
        """
        try:
            self._with_retry(lambda conn: conn.execute(f"ANALYZE {self._table}"))
            return True
        except psycopg.Error as exc:
            _log.warning("could not refresh planner statistics for %s: %s", self._table, exc)
            return False

    def analyze_if_stale(self, modified: int) -> bool:
        """`analyze()`, but only when autovacuum would have. Returns whether one was issued.

        `modified` is how many rows this run wrote. The rule mirrors autovacuum's own trigger
        (see `AUTOANALYZE_THRESHOLD` / `AUTOANALYZE_SCALE_FACTOR`), plus the case autovacuum
        cannot express: a table that has never been analyzed at all (`reltuples < 0`), which is
        every table a first index run builds and the case this exists for.

        The threshold is what keeps a never-analyzed check from being enough on its own. A table
        analyzed while it held three rows has `reltuples = 3` and is no longer "never analyzed",
        so the next run's bulk load would otherwise land against statistics describing three rows
        and reopen the same window.

        Both statements run inside ONE `_with_retry` op. In pooled mode each `_with_retry` is a
        separate borrow, so splitting them would decide against one connection's catalog view and
        then ANALYZE on another's.
        """

        def _op(conn: "psycopg.Connection") -> bool:
            row = conn.execute(
                "SELECT reltuples FROM pg_class WHERE oid = %s::regclass", (self._table,)
            ).fetchone()
            # A missing row means the table is not there (dropped from under us). Treated as
            # never-analyzed so the ANALYZE below is what reports the problem, rather than this
            # method inventing a diagnosis from a catalog miss.
            reltuples = float(row[0]) if row and row[0] is not None else -1.0
            if reltuples >= 0 and modified < AUTOANALYZE_THRESHOLD + (
                AUTOANALYZE_SCALE_FACTOR * reltuples
            ):
                return False
            conn.execute(f"ANALYZE {self._table}")
            return True

        try:
            return self._with_retry(_op)
        except psycopg.Error as exc:
            _log.warning("could not refresh planner statistics for %s: %s", self._table, exc)
            return False

    def _row_scope(self, alias: str = "") -> tuple[str, dict[str, Any]]:
        """The predicate, beyond the tenant, that confines a read to the rows this store serves.

        Empty here: this table holds one copy of a tenant's corpus. `GenerationStore` returns its
        pinned generation, because its table holds every generation the tenant has built and a
        read without the predicate would mix them. SQL beginning with ` AND `, with NAMED
        parameters; `_row_scope_args` is the positional form, spliced right after the tenant.
        """
        return "", {}

    def _row_scope_args(self, alias: str = "") -> tuple[str, tuple[Any, ...]]:
        """`_row_scope` for a statement with positional parameters."""
        return "", ()

    def _hit_first_write(self, alias: str = "") -> str:
        """The expression a scored hit reads `first_indexed_at` from."""
        return f"{alias}first_indexed_at"

    def _rows_to_hits(self, rows: list[tuple]) -> list[ScoredChunk]:
        hits: list[ScoredChunk] = []
        for cid, source, text, metadata, indexed_at, first_indexed_at, score in rows:
            md = metadata if isinstance(metadata, dict) else json.loads(metadata)
            hits.append(
                ScoredChunk(
                    chunk=Chunk(id=cid, source=source, text=text, metadata=md),
                    score=float(score),
                    indexed_at=indexed_at,
                    first_indexed_at=first_indexed_at,
                )
            )
        return hits

    def _hnsw_filtered_tuning(self, k: int | None = None) -> tuple[int, str]:
        """`(ef_search, iterative_scan)` for a filtered dense query, from env or the defaults.

        Read fresh on every call (not cached at import/construction time) so a test can
        `monkeypatch.setenv` per-case and a long-lived process can pick up a changed value without
        restarting — the same convention `index_memory()` uses for `RECALL_INDEX_MAX_FILES`.

        When ``k`` is supplied, the scan is widened far enough to return the requested page even
        when it exceeds the default filtered width. The configured filtered width remains the
        floor, because the over-fetch multiplier is a correctness margin rather than a reason to
        narrow an operator's explicit setting.
        """
        raw_ef = os.environ.get(
            "RECALL_HNSW_EF_SEARCH_FILTERED", str(DEFAULT_HNSW_EF_SEARCH_FILTERED)
        )
        try:
            ef_search = int(raw_ef)
        except ValueError:
            raise ValueError(
                f"RECALL_HNSW_EF_SEARCH_FILTERED={raw_ef!r} is not an integer"
            ) from None
        if not 1 <= ef_search <= _HNSW_EF_SEARCH_MAX:
            # An out-of-range value would only surface as a query-time error on every filtered
            # search. Catch it here, naming the variable, exactly as iterative_scan and the
            # multiplier do.
            raise ValueError(
                f"RECALL_HNSW_EF_SEARCH_FILTERED={ef_search} is out of range; "
                f"pgvector accepts 1..{_HNSW_EF_SEARCH_MAX}"
            )
        iterative_scan = os.environ.get(
            "RECALL_HNSW_ITERATIVE_SCAN_FILTERED", DEFAULT_HNSW_ITERATIVE_SCAN_FILTERED
        )
        if iterative_scan not in _HNSW_ITERATIVE_SCAN_VALUES:
            raise ValueError(
                f"RECALL_HNSW_ITERATIVE_SCAN_FILTERED={iterative_scan!r} is not one of "
                f"{sorted(_HNSW_ITERATIVE_SCAN_VALUES)}"
            )
        if k is not None:
            if k > _HNSW_EF_SEARCH_MAX:
                raise ValueError(
                    f"k={k} exceeds pgvector's maximum hnsw.ef_search of {_HNSW_EF_SEARCH_MAX}, "
                    "so a filtered HNSW scan cannot be widened far enough to return k rows and "
                    "would silently truncate. Ask for fewer candidates."
                )
            desired_ef = k * _ef_search_multiplier()
            if desired_ef > _HNSW_EF_SEARCH_MAX:
                warnings.warn(
                    f"hnsw.ef_search capped at {_HNSW_EF_SEARCH_MAX}: k={k} x multiplier "
                    f"{_ef_search_multiplier()} = {desired_ef}, above pgvector's maximum. The "
                    f"scan still covers k={k}, so only the over-fetch margin is reduced.",
                    RuntimeWarning,
                    stacklevel=3,
                )
            ef_search = max(
                ef_search,
                min(desired_ef, _HNSW_EF_SEARCH_MAX),
            )
        return ef_search, iterative_scan

    def query_dense(
        self,
        vector: list[float],
        k: int,
        source: str | None = None,
        scope: Scope | None = None,
    ) -> list[ScoredChunk]:
        """Timed wrapper; the search itself is `_query_dense`.

        The `k` check stays OUTSIDE the timer on purpose: a rejected call issues no statement, and
        recording it would mix ~0 ms samples into a distribution meant to describe real queries.
        Everything after it is inside, including the HNSW tuning statements, which are part of what
        a dense retrieval costs and would flatter the measurement if excluded.
        """
        if k <= 0:
            raise ValueError("k must be a positive int")
        with METRICS.timer(STORE_QUERY_METRIC, leg=LEG_DENSE):
            return self._query_dense(vector, k, source, scope)

    def query_dense_exact(
        self,
        vector: list[float],
        k: int,
        source: str | None = None,
        scope: Scope | None = None,
    ) -> list[ScoredChunk]:
        """Return exact dense ranks with stable hosted window tie ordering."""
        if k <= 0:
            raise ValueError("k must be a positive int")
        with METRICS.timer(STORE_QUERY_METRIC, leg=LEG_DENSE):
            return self._query_dense_exact(vector, k, source, scope)

    def _query_dense_exact(
        self,
        vector: list[float],
        k: int,
        source: str | None = None,
        scope: Scope | None = None,
    ) -> list[ScoredChunk]:
        t = self._table
        effective = coerce_scope(scope, source)
        where, scope_params = effective.predicate("c", source_column=self._SOURCE_COL)
        rows_scope, rows_params = self._row_scope("c.")
        sql = f"""
            SELECT c.{self._ID_COL}, c.{self._SOURCE_COL}, c.text, c.metadata, c.indexed_at,
                   {self._hit_first_write("c.")},
                   1 - (c.embedding <=> %(vec)s) AS score
            FROM {t} c
            WHERE c.tenant_id = %(tenant)s{rows_scope} {where}
            ORDER BY c.embedding <=> %(vec)s,
                     COALESCE(c.metadata->>'source_session_id', '') COLLATE "C",
                     CASE WHEN (c.metadata->>'segment') ~ '^[0-9]+$'
                          THEN (c.metadata->>'segment')::bigint ELSE 0 END,
                     c.{self._ID_COL}
            LIMIT %(k)s
        """
        params: dict = {"vec": Vector(vector), "k": k, "tenant": self._tenant}
        params.update(rows_params)
        params.update(scope_params)

        def _op(conn: "psycopg.Connection") -> list[tuple]:
            with conn.transaction():
                for guard in _EXACT_SCAN_GUARDS:
                    conn.execute(guard)
                return conn.execute(sql, params).fetchall()

        return self._rows_to_hits(self._with_retry(_op))

    def _query_dense(
        self,
        vector: list[float],
        k: int,
        source: str | None = None,
        scope: Scope | None = None,
    ) -> list[ScoredChunk]:
        t = self._table
        # `recall.scope` owns what a scope predicate is, for all three legs, and every VALUE in it
        # travels as a bound parameter — only the fixed predicate shape is spliced.
        #
        # The `source` arm matches the caller-facing identifier: recall_search surfaces the
        # root-relative `metadata->>'file'` (never the absolute `source` column), so a `source=`
        # filter passed back from a hit must resolve against `file`, falling back to `source` for
        # legacy rows (no `file` metadata) and absolute-path callers. Same rule as recall_forget.
        effective = coerce_scope(scope, source)
        where, scope_params = effective.predicate("c", source_column=self._SOURCE_COL)
        rows_scope, rows_params = self._row_scope("c.")
        sql = f"""
            SELECT c.{self._ID_COL}, c.{self._SOURCE_COL}, c.text, c.metadata, c.indexed_at,
                   {self._hit_first_write("c.")},
                   1 - (c.embedding <=> %(vec)s) AS score
            FROM {t} c
            WHERE c.tenant_id = %(tenant)s{rows_scope} {where}
            ORDER BY c.embedding <=> %(vec)s
            LIMIT %(k)s
        """
        params: dict = {"vec": Vector(vector), "k": k, "tenant": self._tenant}
        params.update(rows_params)
        params.update(scope_params)

        if self._tenant or not effective.is_empty:
            # Every PgVectorStore query is tenant-filtered, even when `source` is absent. The
            # tenant predicate is a post-filter on the shared HNSW index just like the source
            # predicate, so the filtered tuning is required for the normal tenant-scoped path too.
            # A transaction-local setting only takes effect inside a transaction block; on the
            # autocommit connections this store uses, that means explicitly opening one here,
            # tuning the GUCs, then running the query, all before the transaction closes and the
            # tuning reverts. Both GUCs are set in one statement (`_HNSW_FILTERED_TUNING_SQL`).
            ef_search, iterative_scan = self._hnsw_filtered_tuning(k)

            def _op(conn: "psycopg.Connection") -> list[tuple]:
                with conn.transaction():
                    conn.execute(_HNSW_FILTERED_TUNING_SQL, (str(ef_search), iterative_scan))
                    return conn.execute(sql, params).fetchall()

            rows = self._with_retry(_op)
        elif k * _ef_search_multiplier() > _PGVECTOR_DEFAULT_EF_SEARCH:
            # Unfiltered, and pgvector's default HNSW scan is too narrow for this k. Widen it, so
            # that `LIMIT k` is what decides the size of the answer AND the rows returned are
            # actually the nearest k rather than whatever the walk happened to reach.
            #
            # `set_config(..., is_local => true)` rather than `SET LOCAL <literal>` because it
            # takes a bound parameter and can therefore compute GREATEST(current, k * multiplier):
            # an operator who has already raised ef_search (per-database, per-role, or in
            # postgresql.conf) must not have it LOWERED by this call. Raise-only, never clamp down.
            #
            # Skipped entirely when the widened value would not exceed pgvector's own default —
            # there the scan is already at least this wide, so opening a transaction and issuing
            # the extra statement would buy nothing. That keeps the original rationale for leaving
            # the unfiltered arm alone intact over the range where it was actually correct.
            widen = (
                "SELECT set_config('hnsw.ef_search', GREATEST("
                "COALESCE(NULLIF(current_setting('hnsw.ef_search', true), ''), %(default_ef)s)"
                "::int, %(k)s)::text, true)"
            )
            # Bound the DERIVED value the way the filtered path bounds its configured one.
            #
            # `k * multiplier` is over-fetch margin, not a requirement: correctness needs only
            # `ef_search >= k`, or the walk returns fewer than k rows and the caller never learns
            # — the silent truncation #84 exists to prevent. So past pgvector's 1..1000 range the
            # MARGIN is what gives way, and the request still succeeds at the widest legal scan.
            #
            # Unbounded, this reached Postgres as ef_search=2400 for a 600-candidate pool and
            # failed the query with a message naming `hnsw.ef_search` — a knob the caller never
            # set — instead of the `k` they did. The filtered arm has validated this since it was
            # written; the unfiltered arm computes the same quantity and checked nothing: one
            # derivation with the guard on only one of its two paths.
            desired_ef = k * _ef_search_multiplier()
            if k > _HNSW_EF_SEARCH_MAX:
                raise ValueError(
                    f"k={k} exceeds pgvector's maximum hnsw.ef_search of {_HNSW_EF_SEARCH_MAX}, "
                    f"so an unfiltered HNSW scan cannot be widened far enough to return k rows "
                    f"and would silently truncate. Ask for fewer candidates."
                )
            if desired_ef > _HNSW_EF_SEARCH_MAX:
                warnings.warn(
                    f"hnsw.ef_search capped at {_HNSW_EF_SEARCH_MAX}: k={k} x multiplier "
                    f"{_ef_search_multiplier()} = {desired_ef}, above pgvector's maximum. The "
                    f"scan still covers k={k}, so only the over-fetch margin is reduced.",
                    RuntimeWarning,
                    # 3, not 2: the timed `query_dense` wrapper sits between this frame and the
                    # caller. At 2 the warning names this module's own line, so a `-W` filter
                    # keyed on the caller stops selecting it and every call site collapses onto
                    # one `__warningregistry__` entry under the default once-per-location action.
                    stacklevel=3,
                )
            widen_params = {
                "default_ef": str(_PGVECTOR_DEFAULT_EF_SEARCH),
                "k": min(desired_ef, _HNSW_EF_SEARCH_MAX),
            }

            def _op_unfiltered(conn: "psycopg.Connection") -> list[tuple]:
                # SET LOCAL / set_config(local) only survive inside a transaction block, and this
                # store's connections are autocommit, so the transaction is opened explicitly —
                # same shape as the filtered arm above, same reason.
                with conn.transaction():
                    conn.execute(widen, widen_params)
                    return conn.execute(sql, params).fetchall()

            rows = self._with_retry(_op_unfiltered)
        else:
            rows = self._with_retry(lambda conn: conn.execute(sql, params).fetchall())
        if not rows:
            rows = self._dense_exact_fallback(sql, params)
        return self._rows_to_hits(rows)

    def _dense_exact_fallback(self, sql: str, params: dict) -> list[tuple]:
        """Re-run a dense query that returned NOTHING, with ordered index scans disabled.

        A filtered HNSW walk can spend its entire candidate budget on rows that fail the tenant /
        generation / source filter and return zero rows while matching rows exist: the index is
        global and its walk is filter-blind, so every candidate it surfaces can belong to someone
        else. `ef_search` and `iterative_scan` push that boundary out (see the tuning above) but
        cannot remove it — an iterative scan still ends at `hnsw.max_scan_tuples`, and a graph
        region the walk cannot reach (duplicate-heavy clusters after vacuum) is unreachable at
        any setting. CI run 32988012712 (2026-08-26) hit exactly this: three `test_generations`
        tests got zero rows from one-chunk generations that a green run of the same commit had
        found.

        Zero is the one result that is CHEAP to distrust: if the scope is truly empty the exact
        retry scans nothing and agrees, and if it is not, returning [] would have been wrong. A
        partial result stays as-is — fewer-than-k from a populated scope is the approximate
        contract this store documents, and re-running every short result would put an exact scan
        on the serving path.
        """

        def _op(conn: "psycopg.Connection") -> list[tuple]:
            with conn.transaction():
                for guard in _EXACT_SCAN_GUARDS:
                    conn.execute(guard)
                return conn.execute(sql, params).fetchall()

        rows = self._with_retry(_op)
        if rows:
            _log.warning(
                "dense index scan over %s returned no rows where an exact scan finds %d; the "
                "HNSW walk exhausted its candidates on rows outside the query's filter",
                self._table,
                len(rows),
            )
        return rows

    def top_cosine(self, vector: list[float]) -> float:
        """The BEST cosine any chunk in scope has with `vector`, computed EXACTLY.

        Not `query_dense(k=1)[0].score`, and the difference is the whole point. That is an
        `ORDER BY embedding <=> v LIMIT 1`, which Postgres may serve from the HNSW index — an
        APPROXIMATE structure whose walk is filter-blind, so a tenant- or generation-scoped query
        can return a row that is merely reachable rather than nearest, or no row at all. Whether
        that happens is decided by the planner from table size and statistics, so the SAME corpus
        yields a different "best cosine" depending on how much unrelated data shares the table.

        An aggregate has no ORDER BY and no LIMIT of its own, which keeps the ordering index out
        of the DIRECT plan.

        🔁 Corrected 2026-08-26: the aggregate alone is NOT exact by construction, which is what
        this docstring used to claim. Postgres's min/max optimization (planagg.c) rewrites
        `min(embedding <=> v)` into `ORDER BY embedding <=> v LIMIT 1` behind an InitPlan
        whenever that path costs less than the plain aggregate, and pgvector's HNSW index serves
        exactly that ORDER BY — approximately. Whether the rewrite fires is a cost decision, so
        it tracks table statistics: CI run 32988012712 (commit aecc9dc1) planned this aggregate
        through `recall_chunks_v1_embedding_idx` while a green run of the same commit, same
        wheels, twenty minutes earlier had seq-scanned it. So the statement runs under
        `_EXACT_SCAN_GUARDS`, which disable ordered index scans for the transaction; the
        aggregate FORM is still required, for the NaN semantics below, and the guards are what
        make it exact. Bitmap scans stay available for the tenant filter; they cannot produce
        ordering.

        ⛔ `1 - min(distance)`, NOT `max(1 - distance)`, and the two are not interchangeable. A
        zero-norm embedding makes pgvector's cosine distance NaN, and Postgres orders NaN as
        LARGER than every number in an aggregate but LAST in an ascending sort. So `max(1 - d)`
        returns NaN for a corpus holding one degenerate row, where the `ORDER BY d LIMIT 1` this
        replaces returns the real nearest neighbour. Taking the minimum DISTANCE reproduces the
        sort's semantics exactly: NaN is never the minimum unless every row is NaN, in which case
        both forms agree. Verified 2026-08-23 against pgvector on all four cases (no degenerate
        row, one, all, and an empty scope).

        That matters more than it looks. A NaN would flow into the calibration sample lists, and
        `NaN >= threshold` is false, so it would be counted as a correct abstention and lower
        `false_confirm_rate` — the same direction as the defect this method exists to remove.

        Measured 2026-08-22 on a 60,000-chunk generation, one query, same slice, same session:
        `query_dense(k=1)` reported **0.000000** where the true maximum was **0.707107** — the
        planner picked `recall_chunks_v1_embedding_idx` unprompted, because 60k chunks in one
        tenant is an ordinary corpus rather than an extreme. Cost of exactness there: 29.9 ms
        against 7.75 ms, on a path that runs once per labelled query at calibration time.

        Callers are measurement code (`recall.eval.calibrate.measure_top_cosines`, and through it
        `calibrate`, `carry_forward` and drift monitoring), where the number IS the finding. The
        serving path deliberately keeps the approximate search: a search may be approximate, but a
        measurement of whether a threshold still separates two classes may not be.

        Returns 0.0 for an empty scope. That is a measurement, not a fallback: an exact aggregate
        over no rows proves nothing in scope resembles the query. It is NOT what
        `measure_top_cosines` does when its approximate `query_dense` fallback returns nothing,
        which it refuses (since 62e008eb), because an empty approximate result proves only that
        the index walk found nothing.

        ⛔ Deliberately NOT in `TIMED_PUBLIC_METHODS`, and therefore deliberately safe for a
        subclass to override by this name — the opposite of the rule that tuple states, because
        the reason behind that rule does not reach here. `STORE_QUERY_METRIC` attributes cost on
        the QUERY path, and its legs exist so an operator can see what fusion costs; this runs at
        calibration time over a whole generation and would put 30 ms full-slice scans into a
        distribution that is read as serving latency. If a timer is ever wanted here it needs its
        own leg in `STORE_QUERY_LEGS`, not one of the serving ones — and at that point this method
        must gain a private `_top_cosine` twin and move into `TIMED_PUBLIC_METHODS`, or
        `GenerationStore`'s override silently drops the series for the third time.
        """

        rows_scope, rows_params = self._row_scope()

        def _op(conn: "psycopg.Connection") -> "tuple[Any, ...] | None":
            # SET LOCAL needs a transaction block, and this store's connections are autocommit —
            # same shape, same reason as the tuned arms of `_query_dense`.
            with conn.transaction():
                for guard in _EXACT_SCAN_GUARDS:
                    conn.execute(guard)
                return conn.execute(
                    f"SELECT 1 - min(embedding <=> %(vec)s) FROM {self._table} "
                    f"WHERE tenant_id = %(tenant)s{rows_scope}",
                    {"vec": Vector(vector), "tenant": self._tenant, **rows_params},
                ).fetchone()

        row = self._with_retry(_op)
        return 0.0 if row is None or row[0] is None else float(row[0])

    def scope_inventory(self, dimension: str = "folder") -> list[tuple[str, int, int]]:
        """``[(value, chunk_count, document_count)]`` for one dimension, largest first.

        The cheap counterpart to `scope_centroids`: no vectors are read, so this is what a
        "what can I filter on" listing should call.

        It exists because the alternative to knowing is guessing, and guessing at a scope fails
        SILENTLY. A folder that does not exist and a facet that has not been indexed both return
        an empty result set, which is the same thing a corpus with no answer returns. An operator
        who can list the values gets a wrong guess back as "that is not in this list" instead of
        as "your memory does not contain this", and those are very different conclusions.

        A row with a NULL value is excluded for the same reason it is in `scope_centroids`: no
        facet is not a facet. The count of documents carrying none is therefore not in this
        listing, and `recall scopes` reports it separately so it cannot be mistaken for a value.
        """
        t = self._table
        value_sql = group_expression(dimension, "c")
        rows_scope, rows_params = self._row_scope("c.")
        sql = f"""
            SELECT {value_sql} AS scope_value,
                   count(*) AS chunks,
                   count(DISTINCT COALESCE(c.metadata->>'file', c.{self._SOURCE_COL})) AS documents
            FROM {t} c
            WHERE c.tenant_id = %(tenant)s{rows_scope} AND {value_sql} IS NOT NULL
            GROUP BY {value_sql}
            ORDER BY chunks DESC, scope_value
        """
        params = {"tenant": self._tenant, **rows_params}
        rows = self._with_retry(lambda conn: conn.execute(sql, params).fetchall())
        return [(str(value), int(chunks), int(documents)) for value, chunks, documents in rows]

    def scope_undeclared_count(self, dimension: str = "folder") -> int:
        """How many chunks have NO value on `dimension`.

        Reported beside the inventory rather than folded into it. On the facet dimension this is
        the number that answers the question an empty facet listing actually raises: is this
        corpus un-faceted because nobody declares one, or because it was indexed before facets
        were read? A large count here with an empty inventory means the second, and the fix is a
        rebuild rather than an authoring convention.
        """
        t = self._table
        value_sql = group_expression(dimension, "c")
        rows_scope, rows_params = self._row_scope("c.")
        sql = f"""
            SELECT count(*) FROM {t} c
            WHERE c.tenant_id = %(tenant)s{rows_scope} AND {value_sql} IS NULL
        """
        params = {"tenant": self._tenant, **rows_params}
        row = self._with_retry(lambda conn: conn.execute(sql, params).fetchone())
        return 0 if row is None else int(row[0])

    def scope_centroids(
        self, dimension: str = "folder", min_chunks: int = 1
    ) -> list[tuple[str, int, list[float]]]:
        """``[(value, chunk_count, centroid)]`` for one scope dimension, over this tenant.

        The centroid is the MEAN of the chunk embeddings already stored, so building it costs one
        aggregate scan and **no embedding calls**. That is the whole reason the folder prior is
        cheap enough to consider: the folder vectors are not a new representation of the corpus,
        they are a summary of the one it already has.

        ⚠️ **Centroid quality tracks folder size and the caller must weigh that.** A folder of
        three chunks has a centroid that is nearly one chunk; a folder holding most of the corpus
        has one that is nearly the corpus mean, and a cosine against the corpus mean discriminates
        nothing. `chunk_count` is returned beside the vector rather than hidden so a caller can
        refuse the degenerate ends, and `min_chunks` refuses the small one here.

        Rows whose dimension value is NULL (a facet nobody declared) are excluded: "no facet" is
        not a facet, and letting it aggregate would build a centroid of the corpus's leftovers and
        then rank against it as though it meant something.
        """
        if min_chunks < 1:
            raise ValueError(f"min_chunks must be >= 1, got {min_chunks}")
        t = self._table
        # `group_expression` looks the dimension up in a fixed table and raises on anything else,
        # so no caller string reaches the statement. The tenant travels as a bound parameter.
        value_sql = group_expression(dimension, "c")
        rows_scope, rows_params = self._row_scope("c.")
        sql = f"""
            SELECT {value_sql} AS scope_value, count(*) AS n, AVG(c.embedding) AS centroid
            FROM {t} c
            WHERE c.tenant_id = %(tenant)s{rows_scope} AND {value_sql} IS NOT NULL
            GROUP BY {value_sql}
            HAVING count(*) >= %(min_chunks)s
            ORDER BY n DESC
        """
        params = {"tenant": self._tenant, **rows_params, "min_chunks": min_chunks}
        rows = self._with_retry(lambda conn: conn.execute(sql, params).fetchall())
        # `AVG(vector)` comes back as a pgvector `Vector` here and as a numpy array where
        # `register_vector` has been applied to the connection. Neither is a plain list and only
        # one of them is iterable, so the conversion asks for `to_list` before falling back.
        return [
            (
                str(value),
                int(n),
                [
                    float(x)
                    for x in (centroid.to_list() if hasattr(centroid, "to_list") else centroid)
                ],
            )
            for value, n, centroid in rows
        ]

    def query_sparse(
        self,
        text: str,
        k: int,
        source: str | None = None,
        vec: list[float] | None = None,
        scope: Scope | None = None,
    ) -> list[ScoredChunk]:
        """Full-text search. Lexical ranking is ts_rank plus a bounded numeric match boost when
        the query contains numeric terms. When `vec` is given, each hit's `score` is its true dense
        cosine against `vec` instead of the lexical value, so lexical-only hits are comparable
        with dense hits downstream.

        The query is a DISJUNCTION of the question's lexemes, not a conjunction. This used to
        build its tsquery with ``websearch_to_tsquery``, which implements web-search-box
        semantics and ANDs every term — so a chunk had to contain *every* word of the question
        to match at all. On natural-language questions that is essentially never true: measured
        over 150 real questions (mean 15.9 content terms each), the conjunctive form returned
        rows for **0** of them, `_rrf` had a single non-empty list to fuse, and `HybridRetriever`
        silently degraded to dense-only. The disjunctive form matched all 150.

        Requiring every term is also the wrong contract for top-k retrieval: deciding *how much*
        a partial match is worth is `ts_rank`'s job, and it already scores a chunk matching four
        query terms above one matching a single term. The AND was doing that job badly by
        answering "nothing" instead of "less".

        The tsquery is built by normalising the question with the same text-search
        configuration used to build `tsv` (so query lexemes and indexed lexemes agree), then
        OR-ing the lexemes. Each is passed through ``quote_literal``, which is what makes a
        question containing a quote or a tsquery operator a search term rather than syntax.
        A question that normalises to no lexemes (empty, punctuation, stopwords only) yields a
        NULL tsquery, which matches nothing — the same answer as before, without an error.

        Timed under `STORE_QUERY_METRIC{leg=sparse}`; see `query_dense` for why the `k` check sits
        outside the timer.
        """
        if k <= 0:
            raise ValueError("k must be a positive int")
        with METRICS.timer(STORE_QUERY_METRIC, leg=LEG_SPARSE):
            return self._query_sparse(text, k, source, vec, scope)

    def _query_sparse(
        self,
        text: str,
        k: int,
        source: str | None = None,
        vec: list[float] | None = None,
        scope: Scope | None = None,
    ) -> list[ScoredChunk]:
        t = self._table
        numeric_terms = _numeric_query_terms(text)
        numeric_boost = (
            "CASE WHEN c.metadata->'numeric_values' ?| %(numeric_terms)s::text[] "
            "THEN 0.25 ELSE 0 END"
            if numeric_terms
            else "0"
        )
        # See `_query_dense`: one predicate, from `recall.scope`, every value bound. This leg
        # already aliases the chunk table `c`, which is the alias the predicate is rendered for.
        effective = coerce_scope(scope, source)
        where, scope_params = effective.predicate("c")
        # One CTE so the tsquery is built once and both the filter and the ranking see the
        # identical value; repeating the expression risked them drifting apart under edits.
        tsquery_cte = """
            WITH q AS (
                SELECT (
                    SELECT string_agg(quote_literal(lexeme), ' | ')
                    FROM unnest(to_tsvector('english', %(q)s))
                )::tsquery AS tsq
            )
        """
        if vec is not None:
            # cosine only for the k ts_rank winners — computed in the SELECT list of the flat
            # query it would run for EVERY tsquery-matching row before the sort discards them
            sql = f"""
                {tsquery_cte}
                SELECT id, source, text, metadata, indexed_at, first_indexed_at,
                       1 - (embedding <=> %(vec)s) AS score
                FROM (
                    SELECT c.id, c.source, c.text, c.metadata, c.indexed_at, c.first_indexed_at,
                           c.embedding,
                           ts_rank(c.tsv, q.tsq) + {numeric_boost} AS rank
                    FROM {t} c, q
                    WHERE c.tenant_id = %(tenant)s
                      AND c.tsv @@ q.tsq
                    {where}
                    ORDER BY rank DESC
                    LIMIT %(k)s
                ) top_k
                ORDER BY rank DESC
            """
        else:
            sql = f"""
                {tsquery_cte}
                SELECT c.id, c.source, c.text, c.metadata, c.indexed_at, c.first_indexed_at,
                       ts_rank(c.tsv, q.tsq) + {numeric_boost} AS score
                FROM {t} c, q
                WHERE c.tenant_id = %(tenant)s
                  AND c.tsv @@ q.tsq
                {where}
                ORDER BY score DESC
                LIMIT %(k)s
            """
        params: dict = {
            "q": text,
            "k": k,
            "tenant": self._tenant,
            "numeric_terms": numeric_terms,
        }
        if vec is not None:
            params["vec"] = Vector(vec)
        params.update(scope_params)
        rows = self._with_retry(lambda conn: conn.execute(sql, params).fetchall())
        return self._rows_to_hits(rows)

    # ── Learned sparse (SPLADE) sidecar ──────────────────────────────────────────────────────

    def _scrub_sparse_rows(
        self, conn: "psycopg.Connection", chunk_table: str, chunk_ids: list[str]
    ) -> int:
        """Delete every profile's sidecar rows for chunks that were just removed.

        Same transaction as the chunk delete: the sidecar holds partially reconstructable
        content (term weights over a 30,522-term vocabulary), so a commit that removes the
        chunk and strands its weights is an erasure that did not happen. `profile_id` is
        deliberately absent from the WHERE — a dead chunk's rows die under every profile.
        Tenant-filtered, so unlike `drop_table`'s DELETE this is RLS-safe under a
        non-BYPASSRLS role. `chunk_table` travels as a bound VALUE: it is a column value in
        the sidecar, never an identifier here.
        """
        if not chunk_ids:
            return 0
        sidecar = conn.execute("SELECT to_regclass(%s)", (SPARSE_TABLE,)).fetchone()
        if not (sidecar and sidecar[0]):
            return 0  # pre-0012 install: no sidecar exists to scrub
        statement = psycopg.sql.SQL(
            "DELETE FROM {} WHERE tenant_id = %s AND chunk_table = %s AND id = ANY(%s)"
        ).format(psycopg.sql.Identifier(SPARSE_TABLE))
        return conn.execute(statement, (self._tenant, chunk_table, chunk_ids)).rowcount or 0

    def upsert_sparse(self, profile_id: str, vectors: dict[str, dict[int, float]]) -> int:
        """Write learned sparse vectors for `vectors`' chunk ids under `profile_id`.

        Keys are chunk ids; values are ``{term_id: weight}`` with term ids as the model emits
        them (0-based). `pgvector.SparseVector` performs the 0-based to 1-based conversion that
        pgvector's wire format needs — verified against the database, not assumed.

        Over-budget vectors are refused HERE rather than left to the INSERT. pgvector raises past
        the HNSW ceiling, so relying on the database means a 366k-passage load dies partway
        through with an arbitrary prefix already committed.
        """
        if not vectors:
            return 0
        for chunk_id, weights in vectors.items():
            if len(weights) > SPARSE_MAX_NONZERO:
                raise ValueError(
                    f"chunk {chunk_id!r} has {len(weights)} non-zero terms, above pgvector's "
                    f"HNSW limit of {SPARSE_MAX_NONZERO}; prune before storing"
                )
            if not weights:
                raise ValueError(
                    f"chunk {chunk_id!r} encoded to an EMPTY sparse vector. That is not a valid "
                    f"row (the CHECK requires nnz > 0) and it is far more likely to be a broken "
                    f"encoder than a genuinely term-free passage, so it fails loudly."
                )

        rows = [
            (
                self._tenant,
                self._table,
                profile_id,
                chunk_id,
                SparseVector(weights, SPARSE_DIM),
                len(weights),
            )
            for chunk_id, weights in vectors.items()
        ]

        def _op(conn: "psycopg.Connection") -> int:
            with conn.cursor() as cur:
                cur.executemany(
                    f"""
                    INSERT INTO {SPARSE_TABLE}
                        (tenant_id, chunk_table, profile_id, id, vec, nnz)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (tenant_id, chunk_table, profile_id, id)
                    DO UPDATE SET vec = EXCLUDED.vec,
                                  nnz = EXCLUDED.nnz,
                                  indexed_at = clock_timestamp()
                    """,
                    rows,
                )
            return len(rows)

        return self._with_retry(_op)

    def sparse_row_count(self, profile_id: str) -> int:
        """How many chunks of THIS table are encoded under `profile_id`.

        A number, not a boolean, because "the corpus is half encoded" is a state that exists and
        that a caller comparing it against the chunk count can detect. Under RLS this counts only
        the current tenant's rows, which is the intended scope.
        """

        def _op(conn: "psycopg.Connection") -> int:
            row = conn.execute(
                f"SELECT count(*) FROM {SPARSE_TABLE} "
                f"WHERE tenant_id = %s AND chunk_table = %s AND profile_id = %s",
                (self._tenant, self._table, profile_id),
            ).fetchone()
            # `count(*)` always returns a row, so the None branch is unreachable in practice. It
            # is still written, because the ONE caller uses this number to decide whether to
            # REFUSE, and an unchecked `[0]` would turn an impossible state into a TypeError
            # raised from inside a retry wrapper. 0 is the fail-CLOSED answer: it makes the query
            # refuse rather than proceed on a count nobody established.
            return int(row[0]) if row is not None else 0

        return self._with_retry(_op)

    def sparse_covered_sources(self, profile_id: str) -> set[str]:
        """Sources whose EVERY chunk has a sidecar row under `profile_id`.

        `index_path`'s skip predicate needs a per-SOURCE answer, and the sidecar is keyed by
        chunk id, so the join has to be made here rather than inferred from a count. Partial
        coverage of a source is deliberately reported as NOT covered: re-encoding a source that
        is already half done is cheap, and skipping one that is half done leaves a hole no later
        run would fill.

        ⚠️ A source containing a chunk that encodes to an EMPTY vector can never reach full
        coverage, because `store_sparse_vectors` skips those rows by design. Such a source is
        therefore treated as not-yet-covered on every run, and `Indexer.index_path`'s skip
        predicate re-runs the WHOLE index path for it: re-read, re-parse, re-chunk, re-embed
        through `embed_with_cache`, and a full `replace_sources` delete and insert. The SPLADE
        encode this recurs is cheap; the recurring cost that actually matters is the DENSE
        embed, which on a metered embedder is a bill, not the sparse encode. That is wasted
        work, not wrong work, and it is rare (a passage with no surviving terms at all); the
        alternative would be a record of attempts, which is a bigger structure than the saving
        justifies.

        ⚠️ Coverage is keyed on `profile_id` ALONE, not on `SparseProfile.fingerprint()`. Two
        encodings of the SAME model under different `top_k` or a different pinned `revision`
        (folded into `artifact_digest`) share one `profile_id`, so changing either without
        re-encoding reads here as fully covered: the corpus encoded at the OLD budget or
        revision is reported covered under the NEW one, and `Indexer`'s skip predicate leaves
        the stale vectors in place rather than re-encoding them. Re-keying the sidecar on the
        full fingerprint would fix this; nothing does that today.
        """

        def _op(conn: "psycopg.Connection") -> set[str]:
            rows = conn.execute(
                f"""
                SELECT c.source
                FROM {self._table} c
                LEFT JOIN {SPARSE_TABLE} s
                  ON s.tenant_id = %(tenant)s
                 AND s.chunk_table = %(chunk_table)s
                 AND s.profile_id = %(profile)s
                 AND s.id = c.id
                WHERE c.tenant_id = %(tenant)s
                GROUP BY c.source
                HAVING count(*) FILTER (WHERE s.id IS NULL) = 0
                """,
                {"tenant": self._tenant, "chunk_table": self._table, "profile": profile_id},
            ).fetchall()
            return {row[0] for row in rows}

        return self._with_retry(_op)

    def query_learned_sparse(
        self,
        weights: dict[int, float],
        k: int,
        profile_id: str,
        source: str | None = None,
        vec: list[float] | None = None,
        scope: Scope | None = None,
    ) -> list[ScoredChunk]:
        """Learned sparse search, ranked by INNER PRODUCT against the query's term weights.

        `<#>` returns the NEGATIVE inner product, so ascending order is best-first and the score
        is negated back to a real dot product. As with `query_sparse`, passing `vec` makes each
        hit carry its true dense cosine instead, so hits from this leg stay comparable with dense
        hits after fusion.

        ⛔ Raises `LookupError` when this corpus has no rows under `profile_id`. It does NOT
        return an empty list. An unencoded corpus and a corpus with no matches are different
        facts, and this project has already shipped the bug where they were the same one: the
        conjunctive tsquery matched nothing on every real question, fusion silently saw a single
        list, and the hybrid retriever degraded to dense-only with the suite green throughout.

        Timed under `STORE_QUERY_METRIC{leg=learned_sparse}`.
        """
        if k <= 0:
            raise ValueError("k must be a positive int")
        if not weights:
            raise ValueError(
                "query encoded to an empty sparse vector; it would match nothing and the caller "
                "should know that the QUERY is degenerate, not the corpus"
            )
        with METRICS.timer(STORE_QUERY_METRIC, leg=LEG_LEARNED_SPARSE):
            return self._query_learned_sparse(weights, k, profile_id, source, vec, scope)

    def _query_learned_sparse(
        self,
        weights: dict[int, float],
        k: int,
        profile_id: str,
        source: str | None = None,
        vec: list[float] | None = None,
        scope: Scope | None = None,
    ) -> list[ScoredChunk]:
        if self.sparse_row_count(profile_id) == 0:
            raise LookupError(
                f"table {self._table!r} is not indexed for learned sparse profile "
                f"{profile_id!r} (0 rows in {SPARSE_TABLE}). Encode the corpus first; refusing "
                f"to answer, because an empty result here is indistinguishable from a corpus "
                f"that simply had no match."
            )
        t = self._table
        # See `_query_dense`: one predicate, from `recall.scope`, every value bound.
        effective = coerce_scope(scope, source)
        where, scope_params = effective.predicate("c")
        score_expr = (
            "1 - (c.embedding <=> %(dense)s)" if vec is not None else "-(s.vec <#> %(qvec)s)"
        )
        sql = f"""
            SELECT c.id, c.source, c.text, c.metadata, c.indexed_at, c.first_indexed_at,
                   {score_expr} AS score
            FROM {SPARSE_TABLE} s
            JOIN {t} c ON c.tenant_id = s.tenant_id AND c.id = s.id
            WHERE s.tenant_id = %(tenant)s
              AND s.chunk_table = %(chunk_table)s
              AND s.profile_id = %(profile)s
              -- Zero-overlap documents are NOT results. `<#>` is the negative inner product, so
              -- this keeps only a strictly positive dot product. Without it the JOIN returns
              -- every encoded chunk and `LIMIT k` pads the tail with documents sharing no term
              -- with the query — which then collect reciprocal-rank credit in RRF purely for
              -- having been returned. That is the fusion noise the MTRAGEval rank-1 team
              -- reported from ensembling weak retrievers, except manufactured by our own leg.
              AND (s.vec <#> %(qvec)s) < 0
            {where}
            ORDER BY s.vec <#> %(qvec)s
            LIMIT %(k)s
        """
        params: dict = {
            "qvec": SparseVector(weights, SPARSE_DIM),
            "k": k,
            "tenant": self._tenant,
            "chunk_table": self._table,
            "profile": profile_id,
        }
        if vec is not None:
            params["dense"] = Vector(vec)
        params.update(scope_params)

        # Widen the HNSW walk, exactly as `_query_dense` does and for the same reason. See
        # SPARSE_EF_SEARCH_MULTIPLIER: at pgvector's default this leg returned 6 of 100.
        # `SET LOCAL` only survives inside a transaction, and this store's connections are
        # autocommit, so the transaction is opened explicitly. `set_config(..., is_local => true)`
        # rather than a literal SET so the value can be GREATEST(current, wanted): an operator who
        # already raised ef_search must not have it LOWERED here.
        ef_search = sparse_ef_search(k)
        widen = (
            "SELECT set_config('hnsw.ef_search', GREATEST("
            "COALESCE(NULLIF(current_setting('hnsw.ef_search', true), ''), %(default_ef)s)"
            "::int, %(want_ef)s)::text, true)"
        )
        widen_params = {"default_ef": str(_PGVECTOR_DEFAULT_EF_SEARCH), "want_ef": ef_search}

        def _op(conn: "psycopg.Connection") -> list[tuple]:
            with conn.transaction():
                conn.execute(widen, widen_params)
                return conn.execute(sql, params).fetchall()

        rows = self._with_retry(_op)
        return self._rows_to_hits(rows)


    # `supersession_dates()` deliberately does NOT exist. It was a second accessor onto the same
    # cache, and its own docstring had to warn that pairing it with `supersession()` reintroduced
    # the two-validation race `supersession_all()` was written to close. An API whose
    # documentation is a warning against using it is better deleted; it had no callers.

    def sources_for_identifiers(self, identifiers: list[str]) -> dict[str, list[str]]:
        """Resolve caller-facing identifiers to the DB `source` value(s) to delete, this tenant only.

        A ``recall_search`` hit's ``source`` field is the root-relative ``metadata['file']`` for an
        indexed chunk (or the raw ``source`` for a legacy row that predates ``file`` metadata),
        while deletion keys on the absolute ``source`` column. So a caller-supplied identifier
        matches a row by EITHER its ``file`` metadata or its ``source``. Returns
        ``{identifier: [source, ...]}`` for the identifiers that resolved to at least one row;
        identifiers that matched nothing are simply absent (the caller reports them not-found).
        """
        if not identifiers:
            return {}
        rows = self._with_retry(
            lambda conn: conn.execute(
                f"""
                SELECT DISTINCT source, metadata->>'file' AS file
                FROM {self._table}
                WHERE tenant_id = %s AND (metadata->>'file' = ANY(%s) OR source = ANY(%s))
                """,
                (self._tenant, identifiers, identifiers),
            ).fetchall()
        )
        requested = set(identifiers)
        resolved: dict[str, list[str]] = {}
        for source, file in rows:
            for ident in {file, source} & requested:
                bucket = resolved.setdefault(ident, [])
                if source not in bucket:
                    bucket.append(source)
        return resolved

    #: The chunk table a `SupersessionRecord` of this store is scoped to.
    @property
    def record_scope(self) -> str:
        return self._table

    def recorded_sources_for_identifiers(self, identifiers: list[str]) -> dict[str, list[str]]:
        """`sources_for_identifiers` for memos that are no longer indexed but that a
        `SupersessionRecord` still stands for: what a withdrawal of a memo forgotten or pruned
        earlier names. Matches every record's `file` or `source`; a memo indexed again under
        the same name is found by `sources_for_identifiers` as well."""
        if not identifiers:
            return {}
        records = self._with_retry(lambda conn: pg_records(conn, self._tenant, self.record_scope))
        return records_matching(records, identifiers)

    def erasure_subjects(self, sources: list[str], *, recorded: bool = False) -> dict[str, set[str]]:
        """`{source: {name, ...}}`: every name a source security policy decides `sources` by
        (`recall.supersession_records.erasure_subjects` states the rule).

        What an erasure must authorize after `sources_for_identifiers` resolved the names it was
        asked for: a name matches a row by its `file` OR its `source`, so the rows it reaches can
        be authorized by names the caller never gave. With `recorded`, the names of the
        `SupersessionRecord`s standing for `sources` count too, as a withdrawal reaches them.
        """
        if not sources:
            return {}
        rows = self._erasure_subject_rows(sources)
        records = (
            self._with_retry(lambda conn: pg_records(conn, self._tenant, self.record_scope, sources))
            if recorded
            else []
        )
        return erasure_subjects(rows, records)

    def _erasure_subject_rows(self, sources: list[str]) -> list[tuple[Any, ...]]:
        """`(source, file)` for every row of `sources` in this tenant."""
        return list(
            self._with_retry(
                lambda conn: conn.execute(
                    f"SELECT DISTINCT source, metadata->>'file' FROM {self._table} "
                    "WHERE tenant_id = %s AND source = ANY(%s)",
                    (self._tenant, sources),
                ).fetchall()
            )
        )

    def served_sources_for_identifiers(self, identifiers: list[str]) -> dict[str, list[str]]:
        """`sources_for_identifiers`: every row of this table is served, so the erasure resolver
        is also the read-time one (`recall.fact_retraction.served_sources`); `GenerationStore`
        narrows it to its pinned generation."""
        return self.sources_for_identifiers(list(identifiers))

    def served_identifiers(self, identifiers: list[str]) -> set[str]:
        """The identifiers (a row's `file` or its `source`) this store serves now.

        What `recall_current_facts` asks to hide a fact whose every memo is gone
        (`recall.provenance_controller.absent_fact_ids`). Here every row is served, so it is the
        keys of `sources_for_identifiers`; `GenerationStore` narrows it to its pinned generation.
        """
        return set(self.sources_for_identifiers(list(identifiers)))

    def project_file_hashes(self, project: str) -> dict[str, str]:
        """`{root-relative file: content hash}` for one declared project.

        ⛔ **Machine-independent, which `source` is not.** `source_content_hashes` keys on the
        absolute path, so the same memo indexed from `/home/.../memory/acme/note.md` and from
        `C:\\Users\\...\\memstores\\acme\\note.md` looks like two different files: the skip test
        misses, every file is re-embedded, and `replace_sources` deletes nothing because it too
        keys on `source`. Measured on a live corpus: 452 duplicate rows in one project, 615 in
        another, and a search returning the same chunk twice.

        Scoped to a DECLARED project. Without one, two roots indexed into a single tenant may be
        different corpora that happen to share relative paths, and merging them is not a guess this
        layer may make. `metadata->>'file'` is already the root-relative path and already documents
        itself as the thing that identifies a file; it simply was not what the decisions used.
        """
        rows = self._with_retry(
            lambda conn: conn.execute(
                f"SELECT DISTINCT metadata->>'file', "
                f"coalesce(metadata->>'index_fingerprint', metadata->>'content_hash', '') "
                f"FROM {self._table} "
                f"WHERE tenant_id = %s AND metadata->>'project' = %s "
                f"AND metadata->>'file' IS NOT NULL",
                (self._tenant, project),
            ).fetchall()
        )
        return {name: digest for name, digest in rows}

    def source_content_hashes(self) -> dict[str, str]:
        """`{source: content_hash}` for this tenant — what the indexer compares against.

        One row per source: the hash is a property of the file, so every chunk of it carries the
        same value and `DISTINCT` collapses them. A source indexed before content hashing existed
        has no hash and is reported as `""`, which can never equal a real sha256 — so it is
        re-indexed once and then skipped like everything else.
        """
        rows = self._with_retry(
            lambda conn: conn.execute(
                f"SELECT DISTINCT source, coalesce(metadata->>'index_fingerprint', "
                f"metadata->>'content_hash', '') "
                f"FROM {self._table} WHERE tenant_id = %s",
                (self._tenant,),
            ).fetchall()
        )
        return {source: content_hash for source, content_hash in rows}

    def source_raw_hashes(self) -> dict[str, str]:
        """Raw content hashes by source, independent of embedding profile or context mode."""
        rows = self._with_retry(
            lambda conn: conn.execute(
                f"SELECT DISTINCT source, coalesce(metadata->>'content_hash', '') "
                f"FROM {self._table} WHERE tenant_id = %s",
                (self._tenant,),
            ).fetchall()
        )
        return {source: content_hash for source, content_hash in rows}

    def newest_indexed_at(self) -> datetime | None:
        """Newest `indexed_at` for this tenant. Timed: `HybridRetriever` calls it EVERY search.

        Uncached, so this is a real round trip on the query path — see `LEG_META`. Timing it is
        what stops a latency attribution from booking it as Python glue.

        Subclasses override `_newest_indexed_at`, not this; see `TIMED_PUBLIC_METHODS`.
        """
        with METRICS.timer(STORE_QUERY_METRIC, leg=LEG_META):
            return self._newest_indexed_at()

    def _newest_indexed_at(self) -> datetime | None:
        row = self._with_retry(
            lambda conn: conn.execute(
                f"SELECT max(indexed_at) FROM {self._table} WHERE tenant_id = %s",
                (self._tenant,),
            ).fetchone()
        )
        return row[0] if row else None

    def cosines_for(self, ids: Sequence[str], vec: list[float]) -> dict[str, float]:
        """Cosine similarity between `vec` and each of `ids`, for ids that exist.

        `search_fused` retrieves with two query embeddings, so a hit surfaced only by the history
        variant carries a cosine against the HISTORY, not the query. `hit.score` is not
        decorative: `trust.py` thresholds on it and feeds it to `cal.confidence()`, a calibration
        fitted on cosines against the query. This puts every returned hit back on that one basis.

        Ids that do not exist are OMITTED rather than returned as 0.0. Zero is a real cosine and
        would read as a genuine poor match; absence is a different fact and the caller can tell.

        Subclasses override `_cosines_for`, not this; see `TIMED_PUBLIC_METHODS`.
        """
        if not ids:
            return {}
        with METRICS.timer(STORE_QUERY_METRIC, leg=LEG_RESCORE):
            return self._cosines_for(ids, vec)

    def _cosines_for(self, ids: Sequence[str], vec: list[float]) -> dict[str, float]:
        # De-duplicated but order-preserving: a repeated id would not change the answer, only
        # the round trip's payload size, and `dict.fromkeys` is the standard way to do that
        # without reaching for a set (which would make the query non-deterministic to read).
        wanted = list(dict.fromkeys(str(i) for i in ids))

        scope, scope_args = self._row_scope_args()

        def _op(conn: "psycopg.Connection") -> list[tuple]:
            return conn.execute(
                f"SELECT {self._ID_COL}, 1 - (embedding <=> %s) FROM {self._table} "
                f"WHERE tenant_id = %s{scope} AND {self._ID_COL} = ANY(%s)",
                (Vector(vec), self._tenant, *scope_args, wanted),
            ).fetchall()

        rows = self._with_retry(_op)
        return {str(row[0]): float(row[1]) for row in rows}

    def count(self) -> int:
        scope, scope_args = self._row_scope_args()
        row = self._with_retry(
            lambda conn: conn.execute(
                f"SELECT count(*) FROM {self._table} WHERE tenant_id = %s{scope}",
                (self._tenant, *scope_args),
            ).fetchone()
        )
        return int(row[0]) if row else 0

    def authored_graph_relation_count(self) -> int:
        """Count persisted authored semantic relations for this tenant and generation."""
        row = self._with_retry(
            lambda conn: conn.execute(
                "SELECT count(*) FROM recall_graph_relations_v1 "
                "WHERE tenant_id = %s AND generation_id = %s AND status = 'authored'",
                (self._tenant, self.generation_id),
            ).fetchone()
        )
        return int(row[0]) if row else 0

    @contextmanager
    def _borrowed(self) -> "Iterator[psycopg.Connection]":
        """Hold ONE connection for the whole of a streaming read.

        `_with_retry` borrows per operation, which is right for a single statement and wrong for
        a server-side cursor: the cursor lives in the session that declared it, so returning the
        connection to the pool between `FETCH`es would invalidate it.
        """
        if self._closed:
            raise RuntimeError("store is closed")
        if self._shared is not None:
            # One transaction spanning the whole iteration, deliberately. A server-side cursor
            # lives in the transaction that declared it, so the tenant `SET LOCAL` and the cursor
            # have exactly the same lifetime here — which is what a streaming read needs, and is
            # why this cannot go through `_with_retry_shared`'s per-operation transaction.
            with self._shared.tenant_transaction(self._tenant) as conn:
                yield conn
        elif self._pool is not None:
            with self._pool.connection() as conn:
                yield conn
        else:
            yield self._direct

    def iter_chunks(self, batch_size: int = 1000) -> "Iterator[Chunk]":
        """Yield every chunk for this tenant, text and metadata but NOT the embedding.

        Streams through a server-side cursor so a corpus larger than memory does not have to be
        materialised client-side — the same reason `Indexer` writes in batches.

        Embeddings are deliberately excluded: the callers are text-side (the lexical baseline in
        `recall.eval.bm25`, corpus export, audit), the vector column is the overwhelming majority
        of each row's bytes, and returning it would make the bounded-memory cursor pointless.

        Snapshot-consistent for the life of the iterator: rows written after it opened are not
        seen. Nothing here mutates, so a concurrent index run is safe — merely not fully visible.

        `batch_size` must be >= 1; it is the FETCH size, so 0 would ask the server for nothing
        and hang the iteration rather than raising.
        """
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive int")
        # Deliberately NOT wrapped in `_with_retry`. A reconnect halfway through a cursor scan
        # would restart it from the top on the new session and yield the earlier rows a second
        # time; a caller building an index over the result would silently double-count. A lost
        # connection raises here instead, which is the honest outcome.
        scope, scope_args = self._row_scope_args()
        with self._borrowed() as conn:
            # An explicit transaction even though the connection is autocommit: a server-side
            # cursor is transaction-scoped, and under autocommit each FETCH would otherwise land
            # in its own transaction, where the cursor no longer exists.
            with conn.transaction():
                with conn.cursor(name=f"recall_iter_{uuid4().hex[:12]}") as cur:
                    cur.itersize = batch_size
                    cur.execute(
                        f"SELECT {self._ID_COL}, {self._SOURCE_COL}, text, metadata "
                        f"FROM {self._table} "
                        f"WHERE tenant_id = %s{scope} ORDER BY {self._ID_COL}",
                        (self._tenant, *scope_args),
                    )
                    for cid, source, text, metadata in cur:
                        yield Chunk(id=cid, source=source, text=text, metadata=metadata or {})

    def iter_chunks_with_times(
        self, batch_size: int = 1000
    ) -> "Iterator[tuple[Chunk, datetime | None]]":
        """Yield chunks with their first transaction-time visibility instant.

        Dependency replay needs the same first-write timestamp used by supersession and trust.
        The timestamp is returned beside the chunk rather than injected into corpus metadata, so
        the authored document remains the only source of metadata values.
        """
        yield from self._iter_chunks_with_times(batch_size, include_text=True)

    def _iter_chunks_with_times(
        self, batch_size: int, *, include_text: bool
    ) -> "Iterator[tuple[Chunk, datetime | None]]":
        """`iter_chunks_with_times`, optionally without transferring chunk text (yielded as "").

        The current state projection reads every chunk of the corpus and never its text, which is
        most of the bytes a row carries.
        """
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive int")
        text_column = "text" if include_text else "''"
        scope, scope_args = self._row_scope_args()
        with self._borrowed() as conn:
            with conn.transaction():
                with conn.cursor(name=f"recall_iter_times_{uuid4().hex[:12]}") as cur:
                    cur.itersize = batch_size
                    cur.execute(
                        f"SELECT {self._ID_COL}, {self._SOURCE_COL}, {text_column}, metadata, "
                        f"{_first_write_sql()} FROM {self._table} "
                        f"WHERE tenant_id = %s{scope} ORDER BY {self._ID_COL}",
                        (self._tenant, *scope_args),
                    )
                    for cid, source, text, metadata, first_indexed_at in cur:
                        yield (
                            Chunk(id=cid, source=source, text=text, metadata=metadata or {}),
                            first_indexed_at,
                        )

    def related_chunks(
        self, seed_chunk_id: str, relation: str, max_items: int
    ) -> tuple[Chunk, list[Chunk]] | None:
        """Fetch bounded source or ordinal neighbors without materializing the corpus.

        Supersession relations still use the shared lineage resolver because that relation must
        inspect every authored edge. Returning ``None`` for that relation keeps the generic
        fallback explicit while making the common source and ordinal paths bounded at the store.
        """
        if relation not in {"source", "ordinal"}:
            return None
        if isinstance(max_items, bool) or not isinstance(max_items, int) or max_items < 1:
            raise ValueError("max_items must be a positive int")

        seed_row = self._with_retry(
            lambda conn: conn.execute(
                f"SELECT id, source, text, metadata FROM {self._table} "
                "WHERE tenant_id = %s AND id = %s",
                (self._tenant, seed_chunk_id),
            ).fetchone()
        )
        if seed_row is None:
            raise ValueError(f"seed chunk not found: {seed_chunk_id!r}")
        seed_id, seed_source, seed_text, seed_metadata = seed_row
        seed_metadata = seed_metadata or {}
        seed_ord = seed_metadata.get("ord")

        source_match = "source = %s"
        if relation == "source":
            where = source_match
            params: tuple[object, ...] = (self._tenant, seed_source, seed_id, max_items)
            order = (
                "CASE WHEN metadata->>'ord' ~ '^[0-9]+$' "
                "THEN (metadata->>'ord')::int END NULLS LAST, id"
            )
        else:
            if not isinstance(seed_ord, int) or isinstance(seed_ord, bool):
                return (
                    Chunk(seed_id, seed_source, seed_text, seed_metadata),
                    [],
                )
            where = (
                f"{source_match} AND (metadata->>'ord') ~ '^[0-9]+$' "
                "AND abs((metadata->>'ord')::int - %s) <= 2"
            )
            order = "abs((metadata->>'ord')::int - %s), id"
            params = (
                self._tenant,
                seed_source,
                seed_ord,
                seed_id,
                seed_ord,
                max_items,
            )

        rows = self._with_retry(
            lambda conn: conn.execute(
                f"SELECT id, source, text, metadata FROM {self._table} "
                f"WHERE tenant_id = %s AND {where} AND id <> %s "
                f"ORDER BY {order} LIMIT %s",
                params,
            ).fetchall()
        )
        seed = Chunk(seed_id, seed_source, seed_text, seed_metadata)
        return seed, [
            Chunk(cid, source, text, metadata or {}) for cid, source, text, metadata in rows
        ]
