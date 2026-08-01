from __future__ import annotations

import json
import os
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import TYPE_CHECKING, TypeVar
from uuid import uuid4
from ipaddress import ip_address
from urllib.parse import unquote, urlsplit

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector

from recall.frontmatter import supersedes_key
from recall.observability import METRICS, get_logger
from recall.types import Chunk, ScoredChunk

if TYPE_CHECKING:  # the pool extra is optional; the annotation must not require it at runtime
    from psycopg_pool import ConnectionPool

#: The built-in dev credentials shipped in the default DSN — safe only against a local database.
_DEFAULT_CREDS = ("recall", "recall")
#: "" covers a hostless/unix-socket DSN. Bracketed IPv6 is absent on purpose: urlsplit strips
#: the brackets. All of 127.0.0.0/8 is handled numerically by `_is_local_host`.
_LOCAL_HOSTS = ("", "localhost", "::1", "0.0.0.0", "host.docker.internal")


def _is_local_host(host: str) -> bool:
    """True when `host` cannot reach a shared database (loopback, unix socket, or unset)."""
    if host in _LOCAL_HOSTS or host.startswith(("/", "%2f")):  # %2f: percent-encoded socket dir
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


def redacted_dsn(dsn: str) -> str:
    """`dsn` with any password removed — safe to print to a log or a systemd journal.

    A connection failure is exactly when an operator wants the DSN in the logs, and exactly when
    printing it verbatim would write the password to disk.
    """
    try:
        parts = urlsplit(dsn)
        if not parts.hostname:
            return "<dsn>"
        userinfo = f"{parts.username}:***@" if parts.password else (
            f"{parts.username}@" if parts.username else ""
        )
        port = f":{parts.port}" if parts.port else ""
        return f"{parts.scheme}://{userinfo}{parts.hostname}{port}{parts.path}"
    except ValueError:  # pragma: no cover - malformed URL
        return "<dsn>"

_T = TypeVar("_T")

#: How long schema DDL may WAIT FOR A LOCK before giving up (ms). Not a bound on the work — the
#: HNSW build is deliberately unbounded, see `ensure_schema` — only on queueing. Short on purpose:
#: waiting on a lock is never progress, the DDL is idempotent and retried on the next open, and a
#: fast failure is diagnosable where an indefinite stall is not. Overridable because it can refuse
#: where the previous code waited: `0` restores the old unbounded wait.
DEFAULT_SCHEMA_LOCK_TIMEOUT_MS = 5000
#: PostgreSQL stores `lock_timeout` as a signed 32-bit int; anything larger is a parameter error.
_PG_MAX_INT = 2147483647


def _schema_lock_timeout_ms() -> int:
    """`RECALL_SCHEMA_LOCK_TIMEOUT_MS`, non-negative; anything malformed falls back to default.

    Read per call rather than at import so a long-lived process can be retuned without a reload —
    the same convention `RECALL_MAX_PRUNE_FRACTION` and `RECALL_INDEX_MAX_FILES` follow.
    """
    raw = os.environ.get("RECALL_SCHEMA_LOCK_TIMEOUT_MS")
    if raw is None:
        return DEFAULT_SCHEMA_LOCK_TIMEOUT_MS
    try:
        value = int(raw)
    except ValueError:
        _log.warning("ignoring malformed RECALL_SCHEMA_LOCK_TIMEOUT_MS=%r", raw)
        return DEFAULT_SCHEMA_LOCK_TIMEOUT_MS
    if value < 0:
        _log.warning("ignoring negative RECALL_SCHEMA_LOCK_TIMEOUT_MS=%r", raw)
        return DEFAULT_SCHEMA_LOCK_TIMEOUT_MS
    if value > _PG_MAX_INT:
        # Above PostgreSQL's integer range `SET lock_timeout` raises InvalidParameterValue, which
        # would make ensure_schema fail outright — a knob for loosening a bound must not be able
        # to break the thing it loosens. Clamped, not rejected: the intent ("effectively never")
        # is unambiguous, and 24.8 days of lock wait is indistinguishable from it.
        _log.warning(
            "clamping RECALL_SCHEMA_LOCK_TIMEOUT_MS=%r to %d (PostgreSQL integer range)",
            raw, _PG_MAX_INT,
        )
        return _PG_MAX_INT
    return value

_log = get_logger("store")


def warn_if_insecure_dsn(dsn: str) -> str | None:
    """Warn (to stderr) when the built-in ``recall:recall`` credentials target a NON-local host.

    A shared, well-known password is fine against localhost but a footgun the moment the DSN
    points at a real remote database. This warns loudly and returns the message; it never blocks
    execution (returns None when there is nothing to warn about).
    """
    try:
        parts = urlsplit(dsn)
    except ValueError:
        return None
    # unquote: urlsplit returns the RAW percent-encoded form, so "recal%6C" is the password
    # "recall" and must not slip past the comparison
    if (unquote(parts.username or ""), unquote(parts.password or "")) != _DEFAULT_CREDS:
        return None
    if _is_local_host((parts.hostname or "").lower()):
        return None
    msg = (
        f"recall: WARNING — using the default 'recall:recall' credentials against non-local host "
        f"{parts.hostname!r}. Set a strong password via RECALL_DSN before using a remote database."
    )
    _log.warning(msg)
    return msg


#: Tenant assigned to rows written before tenancy existed, and the default for a
#: single-tenant deployment — so an upgrade changes nothing for an existing install.
DEFAULT_TENANT = "default"
#: Default table name. Named rather than repeated as a literal so a caller that needs to pass it
#: explicitly (see `recall_mcp.stores`) cannot drift from the constructor's default.
DEFAULT_TABLE = "chunks"
#: Postgres session variable the row-level-security policy reads. A custom GUC (it must contain
#: a dot) set per connection, so the policy compares against the connection's own tenant.
TENANT_GUC = "recall.tenant_id"

#: Opt-out for `require_secure_dsn`. Named so it cannot be set by accident, and so its presence
#: in a deploy is a visible, greppable decision rather than an oversight.
INSECURE_DSN_OPT_OUT = "RECALL_ALLOW_INSECURE_DSN"

#: HNSW tuning applied ONLY to a `source`-filtered `query_dense` call (see issue #11's third
#: checkbox). An HNSW index walk is filter-blind: it finds the globally nearest neighbours, THEN
#: discards the ones that fail `WHERE source = ...`, so a selective filter can silently return
#: fewer than `k` rows, or fewer true neighbours than exist. Measured on 20,000 rows / dim 64 /
#: a filter matching 10% of rows / 40 queries, recall@10 against an exact scan:
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
#: together. These two knobs apply to FILTERED queries; the unfiltered arm has its own, separate
#: widening for `k` past ef_search's default (see `_PGVECTOR_DEFAULT_EF_SEARCH` below). The earlier
#: claim that the unfiltered arm needs no tuning held only at `k <= ef_search` — at k=10, where it
#: was measured; it does not generalise past ef_search's default, which is the gap that widening
#: closes.
#:
#: Both are read at CALL time (not import time) via `os.environ`, matching how
#: `RECALL_INDEX_MAX_FILES` / `RECALL_INDEX_MAX_BYTES` are read in `recall_mcp/service.py`, so a
#: test can `monkeypatch.setenv` per-case and a long-lived process can pick up a changed value
#: without restarting.
DEFAULT_HNSW_EF_SEARCH_FILTERED = 200
#: pgvector's accepted range for `hnsw.ef_search` is 1..1000; a value outside it is rejected at
#: config time rather than being interpolated into `SET LOCAL hnsw.ef_search` and erroring on
#: every filtered search.
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
#: pgvector's only valid values for this GUC, checked by `_hnsw_filtered_tuning()` below — the
#: configured value is interpolated into `SET LOCAL` (Postgres does not accept a bound parameter
#: there), so it is validated against this allowlist rather than trusted as-is.
_HNSW_ITERATIVE_SCAN_VALUES = frozenset({"off", "relaxed_order", "strict_order"})
#: Values that count as "yes" for an opt-OUT of a safety guard. Deliberately an allowlist rather
#: than a truthiness test: `0`/`false`/`no` must read as "keep the guard", and anything
#: unrecognised must too, because a typo in a security switch may not grant permission.
_ENV_TRUE = frozenset({"1", "true", "yes", "on"})


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


def _env_opt_out(name: str) -> bool:
    """True only when `name` is set to an explicit affirmative (see `_ENV_TRUE`)."""
    return os.environ.get(name, "").strip().lower() in _ENV_TRUE


def require_secure_dsn(dsn: str) -> None:
    """Raise unless `dsn` is safe to use unattended; the fail-closed form of the warning above.

    `warn_if_insecure_dsn` detects the built-in `recall:recall` credentials against a remote host
    and then RETURNS, so the process carries on talking to a shared database with a password
    published in this repository's README. A warning on stderr is not a control: under systemd it
    lands in a journal nobody reads, and the server comes up looking healthy.

    A server should therefore call this instead. The escape hatch is an explicit environment
    variable, because the legitimate case (a private network where the operator has genuinely
    accepted the risk) must be expressible — just not by default and not silently.
    """
    # Parsed, not merely truthy. A bare `os.environ.get(...)` reads ANY non-empty string as
    # "opt out", so `RECALL_ALLOW_INSECURE_DSN=0` — which an operator writes meaning "keep the
    # guard on" — silently switched the guard OFF, the exact inverse of the intent. Anything
    # unrecognised keeps the guard: this is a security control, so an ambiguous value must not
    # be read as permission.
    if _env_opt_out(INSECURE_DSN_OPT_OUT):
        return
    if warn_if_insecure_dsn(dsn) is None:
        return
    raise PermissionError(
        f"refusing to start against {redacted_dsn(dsn)}: the default 'recall:recall' credentials "
        f"are published in this project's README and this DSN points at a non-local host. Set a "
        f"real password, or set {INSECURE_DSN_OPT_OUT}=1 to accept the risk deliberately."
    )


def _basename(file: str) -> str:
    """Stem of a root-relative (posix) file identifier — the key a `supersedes:` target resolves
    to. A stem rather than a full basename so `name`, `name.md`, `[name]` and `[[name]]` all
    designate the same document; see `supersedes_key`."""
    return supersedes_key(file)


def resolve_edge_dates(
    rows: list[tuple[str | None, str | None, datetime | None]],
    edges: dict[str, str],
) -> dict[str, datetime]:
    """When each edge in `edges` became assertable, from ``(file, supersedes, first_indexed)`` rows.

    An edge ``A -> B`` becomes assertable when B is written, because the claim lives in B's
    `supersedes:` frontmatter. So it is dated by the EARLIEST `indexed_at` among B's chunks: any
    chunk of B existing implies the frontmatter existed, and the earliest is the conservative
    reading.

    Dated **per claim**, keyed on `(superseding file, superseded basename)`, not per file. One
    file can carry several different `supersedes` values, and it need not be an authoring
    mistake: `Indexer._prune_vanished` notes that a corpus may be indexed under several roots,
    `metadata['file']` is root-relative while `replace_sources` deletes by absolute `source`, so
    two roots with divergent frontmatter coexist under one `file` key. A per-file minimum then
    dates a LATER claim from an EARLIER one, and the edge applies at instants before it was
    written. Grouping by the pair is already what the SQL does; throwing the pair away here was
    the bug.

    The "carries a claim" predicate must match `resolve_supersession`'s exactly, because both
    consume the same rows. It uses falsiness, so a bare `supersedes:` in frontmatter (which
    parses to the empty string) builds no edge; treating it as claim-carrying here would date an
    edge that another row asserts, which is the same failure through a second door.

    An edge whose superseding file has no recorded date is simply absent from the result, and
    `recall.trust.resolve_successor` applies an undated edge. That is the fail-closed direction:
    an edge of unknown age keeps demoting, rather than silently reviving a memory the corpus
    marks as stale. Note this is the INVERSE of the rule for hits, where an unknown `indexed_at`
    leaves the hit visible; both choices refuse to serve something as healthier than it is.

    ⚠️ The date is only as good as `indexed_at`, which records the LAST write, not the first: see
    `supersession_all` for the open defect that makes this unsafe to merge as it stands.

    Pure and DB-free, so the dating rule is unit-testable without a database — same reason
    `resolve_supersession` is.
    """
    # (superseding file, superseded basename) -> when that CLAIM was first written.
    by_claim: dict[tuple[str, str], datetime] = {}
    for file, supersedes, first_indexed in rows:
        if not file or not supersedes or first_indexed is None:
            continue
        # BOTH key forms, because `resolve_supersession` uses two. A RESOLVED edge is keyed on the
        # target's root-relative path, matched through `_basename`. A DANGLING one is keyed on the
        # raw `supersedes.rsplit("/", 1)[-1]`, deliberately keeping the author's spelling. Those
        # differ whenever the claim carries both a path and brackets: `[[dir/gone]]` normalises to
        # `gone` but dangles as `gone]]`. Indexing only the normalised form left every bracketed
        # dangling edge undated, which the per-file keying this replaced happened to get right.
        for form in {_basename(supersedes), supersedes.rsplit("/", 1)[-1]}:
            key = (file, form)
            known = by_claim.get(key)
            if known is None or first_indexed < known:
                by_claim[key] = first_indexed
    dated: dict[str, datetime] = {}
    for superseded, superseding in edges.items():
        when = by_claim.get((superseding, _basename(superseded)))
        if when is not None:
            dated[superseded] = when
    return dated


def resolve_supersession(
    rows: list[tuple[str | None, str | None]],
) -> tuple[dict[str, str], frozenset[str]]:
    """Build the superseded -> superseding map from ``(file, supersedes)`` rows.

    ``file`` is a root-relative path; ``supersedes`` references its target by basename (the
    authoring convention). Three cases:

    - **Unambiguous** (exactly one indexed file bears that basename): resolve to its
      root-relative path. This is the fix for the original bug — a naive basename key would
      have collided with an unrelated same-named file in another directory.
    - **Dangling** (no indexed file bears that basename — the predecessor was never indexed,
      or was deleted): fall back to the raw basename as the key. There is nothing to
      disambiguate, so this cannot mis-map; dropping it would just as silently discard a valid
      supersession claim (e.g. a memo intentionally superseding a doc that was since removed).
    - **Ambiguous** (two or more indexed files share that basename): do not guess — a silent
      mis-map to the wrong file is worse than a broken chain, since we cannot tell which one the
      author meant. The candidates are returned in ``unresolved`` so the read path can fail
      closed on them; dropping the edge and saying nothing would leave a possibly-superseded
      memory looking perfectly healthy.

    Both keys and values in the mapping are root-relative paths (or a bare basename for the
    dangling case). ``unresolved`` holds root-relative paths.

    Pure and DB-free so the resolution rule can be unit-tested without a database.
    """
    # DEDUPED. `rows` carries one entry per (file, supersedes) pair, so a file asserting two
    # different claims appeared TWICE and made ITSELF read as an ambiguous basename: its own
    # incoming edge was dropped and it was named in `unresolved`, telling the operator to
    # disambiguate a basename that exactly one document carries. Ambiguity is a property of two
    # FILES sharing a stem, never of one file carrying two claims.
    files = list(dict.fromkeys(f for f, _ in rows if f))
    by_base: dict[str, list[str]] = {}
    for f in files:
        by_base.setdefault(_basename(f), []).append(f)
    mapping: dict[str, str] = {}
    unresolved: set[str] = set()
    for file, supersedes in rows:
        if not file or not supersedes:
            continue
        target_basename = _basename(supersedes)
        candidates = by_base.get(target_basename, [])
        if len(candidates) == 1:
            mapping[candidates[0]] = file
        elif len(candidates) == 0:
            # Dangling: key on the raw basename as written. Normalisation exists to make MATCHING
            # tolerant of how humans spell a reference; this key matches no real file either way,
            # so it keeps the author's form rather than inventing a normalised one.
            mapping[supersedes.rsplit("/", 1)[-1]] = file
        else:
            # Ambiguous: don't guess — but don't stay silent either. Dropping the edge alone
            # would leave the (possibly superseded) memories looking perfectly `ok`, which is
            # the same wrong answer the trust layer exists to prevent. Naming them lets the
            # read path fail closed and tell the operator what to fix.
            unresolved.update(candidates)
    return mapping, frozenset(unresolved)


def _ef_search_multiplier() -> int:
    """How far past `k` to widen an unfiltered HNSW scan; see DEFAULT_HNSW_EF_SEARCH_MULTIPLIER.

    Read at call time, not import time — same convention as `_hnsw_filtered_tuning`, so a test
    can `monkeypatch.setenv` per case and a long-lived process picks up a change without a
    restart. Values below 1 are refused rather than clamped: a 0 would silently disable the
    widening and reintroduce the truncation this exists to prevent.
    """
    raw = os.environ.get(
        "RECALL_HNSW_EF_SEARCH_MULTIPLIER", str(DEFAULT_HNSW_EF_SEARCH_MULTIPLIER)
    )
    try:
        mult = int(raw)
    except ValueError:
        raise ValueError(f"RECALL_HNSW_EF_SEARCH_MULTIPLIER={raw!r} is not an integer") from None
    if mult < 1:
        raise ValueError(f"RECALL_HNSW_EF_SEARCH_MULTIPLIER={mult} must be >= 1")
    return mult


class PgVectorStore:
    """The single, production-grade vector store: PostgreSQL + pgvector."""

    #: Errors that mean the connection itself is broken (dropped socket, server restart, idle
    #: timeout) — as opposed to a query/data error. These trigger a reconnect-and-retry.
    _CONN_ERRORS = (psycopg.OperationalError, psycopg.InterfaceError)

    #: Class-level default so `_pool` always exists, including on an instance built without
    #: __init__ (the store tests do this to exercise the retry logic against a fake connection).
    #: Single-connection mode is the default, so None is also the honest value.
    _pool = None

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

        One documented exception: `ensure_schema()` lifts it for the duration of its DDL, because
        an HNSW build legitimately outlasts any sane query bound, and restores it before the
        connection is reused. A `lock_timeout` bounds that window instead — see `ensure_schema`.
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
        self._dsn = dsn
        self._dim = dim
        self._table = table
        self._tenant = tenant
        self._statement_timeout_ms = statement_timeout_ms
        self._connect_timeout_s = connect_timeout_s
        #: (fingerprint, edges, unresolved, edge_dates) — see `supersession_all()`. The fingerprint is what
        #: makes the cache safe to reuse across processes.
        self._supersession_cache: tuple | None = None
        #: Count of full supersession scans actually performed (cache misses). Surfaced so a
        #: test can prove the cache still works, and so a rescan storm is visible as a metric.
        self._supersession_scans = 0
        self._closed = False
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
                "pool_size requires the pool extra: pip install recall[pool]"
            ) from exc

        # `configure` runs on every connection the pool creates, not just the first — the vector
        # type registration is per-connection state, so a pool that skipped it would work until
        # it opened its second connection and then fail on a Vector parameter.
        pool = ConnectionPool(
            self._dsn,
            min_size=1,
            max_size=size,
            kwargs=self._connect_kwargs(),
            configure=self._prepare,
            open=False,
        )
        pool.open(wait=True, timeout=self._connect_timeout_s or 30)
        return pool

    def _prepare(self, conn: "psycopg.Connection") -> None:
        """Per-connection setup: extension, vector type registration, statement timeout."""
        # register_vector needs the `vector` type to already exist, so ensure the extension
        # is installed first — this makes a brand-new database work out of the box (the
        # README quickstart path). If this role lacks privilege to create it, fall through:
        # register_vector still succeeds when an admin has pre-installed the extension, and
        # fails with a clear "vector type not found" if it genuinely isn't there.
        try:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        except psycopg.Error:
            pass
        register_vector(conn)
        # Per-connection tenant for the RLS policy. Safe to set once at connection setup because
        # a store is bound to ONE tenant: the pool belongs to the store, so no connection is ever
        # shared between tenants. A server handling many tenants opens a store per tenant.
        conn.execute(f"SELECT set_config('{TENANT_GUC}', %s, false)", (self._tenant,))
        if self._statement_timeout_ms is not None:
            conn.execute(f"SET statement_timeout = {int(self._statement_timeout_ms)}")

    def _connect(self) -> "psycopg.Connection":
        """Open one autocommit connection and prepare it (extension + vector type registration)."""
        conn = psycopg.connect(self._dsn, **self._connect_kwargs())
        try:
            self._prepare(conn)  # same per-connection setup the pool applies via `configure`
        except Exception:
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
        except Exception:
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
        if self._pool is not None:
            return self._with_retry_pooled(op)
        try:
            return op(self._direct)
        except self._CONN_ERRORS:
            # getattr: `broken` only exists from psycopg 3.2 and the declared floor is 3.1 —
            # without the default this except-block would raise AttributeError and mask the
            # original database error on an older install.
            if not (self._direct.closed or getattr(self._direct, "broken", False)):
                raise
            _log.warning("database connection lost — reconnecting")
            METRICS.increment("recall_db_reconnects_total")
            self._reconnect()
            return op(self._direct)

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
                    return op(conn)
                except self._CONN_ERRORS:
                    dead = conn.closed or getattr(conn, "broken", False)
                    if not dead or attempt == 1:
                        raise
                    _log.warning("pooled database connection lost — retrying on another")
                    METRICS.increment("recall_db_reconnects_total", pooled="true")
        raise AssertionError("unreachable: the loop either returns or raises")  # pragma: no cover

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

    def close(self) -> None:
        """Close the connection (or pool) for good.

        Sticky by design: without the flag, any later call would hit `_with_retry`'s reconnect
        and silently resurrect a connection nobody owns — a leak on first accidental reuse.
        """
        self._closed = True
        if self._pool is not None:
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
        self._with_retry(lambda conn: conn.execute(f"DROP TABLE IF EXISTS {self._table}"))

    def __enter__(self) -> "PgVectorStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def ensure_schema(self) -> None:
        def _op(conn: "psycopg.Connection") -> None:
            # Schema DDL runs WITHOUT the per-connection statement_timeout. `_prepare` sets that
            # timeout on every connection (it is the pool's `configure` hook, and `_connect` calls
            # it too) to bound runaway QUERIES — but the HNSW build below is not a runaway query,
            # and on a real corpus it takes minutes, far past the MCP server's 15s default. Left
            # in place the timeout cancels the build, which fails startup against any populated
            # database and leaves an INVALID index that the `IF NOT EXISTS` below then treats as
            # present forever (see the CONCURRENTLY comment further down). Restored in `finally`
            # so a pooled connection is never handed back with the guard still lifted.
            #
            # Lifting statement_timeout removes the bound on how long the DDL may WORK, which is
            # intended. It also removes the only bound on how long it may QUEUE, which is not:
            # statement_timeout counts lock-wait time, so it was previously what broke a wait on
            # a lock that never came. `CREATE INDEX CONCURRENTLY` waits for every concurrent
            # transaction on the table, and the ALTERs below take ACCESS EXCLUSIVE — so one
            # `idle in transaction` session elsewhere is enough to park schema setup forever,
            # with every query arriving afterwards queued behind it and no error to say why.
            # `lock_timeout` restores that bound where it belongs: unbounded work, bounded
            # waiting. A build that cannot get its lock fails fast and is retried on next open,
            # which is idempotent.
            statement_timeout_ms = self._statement_timeout_ms
            timeout_lifted = statement_timeout_ms is not None
            # `SHOW` always returns exactly one row, so `row` is None only if the connection died
            # between the two statements. Default to Postgres' own default rather than indexing
            # None: the restore below is best-effort and must not raise over the real failure.
            row = conn.execute("SHOW lock_timeout").fetchone()
            prior_lock_timeout = row[0] if row else "0"
            # BOTH `SET`s go inside the `try`. Outside it, a failure of the second would skip the
            # `finally` entirely and hand a connection back to the pool with statement_timeout
            # still lifted — the one thing the paragraph above promises cannot happen.
            try:
                if timeout_lifted:
                    conn.execute("SET statement_timeout = 0")
                conn.execute(f"SET lock_timeout = {_schema_lock_timeout_ms()}")
                self._ensure_schema_ddl(conn)
            finally:
                # Never let a restore displace the exception already in flight: if the DDL failed
                # because the connection died, these SETs raise too, and the operator would be
                # shown the restore failure instead of the cause. The connection is discarded
                # either way, so a failed restore costs nothing. Each restore is guarded
                # independently — sharing one `try` would let the first failure skip the second.
                if statement_timeout_ms is not None:
                    # Plain `SET`: this value is an int this object owns, so there is nothing to
                    # bind and nothing to quote.
                    self._safe_execute(
                        conn, f"SET statement_timeout = {int(statement_timeout_ms)}"
                    )
                # `set_config` here, because `SHOW` hands the prior value back as a FORMATTED
                # string ('0', '5s', '1min') and `SET` does not take a bind parameter.
                self._safe_execute(
                    conn, "SELECT set_config('lock_timeout', %s, false)", (prior_lock_timeout,)
                )

        self._with_retry(_op)

    @staticmethod
    def _safe_execute(
        conn: "psycopg.Connection", sql: str, params: tuple | None = None
    ) -> None:
        """Run a session-restore statement that must never mask an exception already in flight.

        Called only from `finally` blocks. If the DDL failed because the connection died, the
        restore raises too and would REPLACE the real error with itself; the connection is being
        discarded either way, so swallowing the restore failure loses nothing and keeps the
        diagnosis.
        """
        try:
            conn.execute(sql, params) if params else conn.execute(sql)
        except psycopg.Error as exc:  # pragma: no cover - requires a broken connection
            _log.warning("could not restore session setting after schema DDL: %s", exc)

    def _ensure_schema_ddl(self, conn: "psycopg.Connection") -> None:
        t = self._table
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {t} (
                tenant_id TEXT NOT NULL DEFAULT '{DEFAULT_TENANT}',
                id TEXT NOT NULL,
                source TEXT NOT NULL,
                text TEXT NOT NULL,
                metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                embedding vector({self._dim}),
                indexed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
                PRIMARY KEY (tenant_id, id)
            )
            """
        )
        self._migrate_to_tenanted(conn)
        dim_row = conn.execute(
            "SELECT atttypmod FROM pg_attribute "
            "WHERE attrelid = %s::regclass AND attname = 'embedding'",
            (t,),
        ).fetchone()
        if dim_row is None:
            # `CREATE TABLE IF NOT EXISTS` above is a no-op against a table that already exists
            # under this name with a different shape, so an unrelated `chunks` table reaches here
            # with no `embedding` column at all. Indexing `None` gave a bare TypeError that named
            # neither the table nor the cause.
            raise ValueError(
                f"table {t!r} exists but has no 'embedding' column — it is not a recall table. "
                f"Point this store at a different table name."
            )
        actual_dim = dim_row[0]
        if actual_dim > 0 and actual_dim != self._dim:
            raise ValueError(
                f"table {t!r} has a vector({actual_dim}) embedding column but this store is "
                f"configured for dim {self._dim} — use a matching embedder or a different "
                f"table (drop and re-index for a clean slate)."
            )
        # CONCURRENTLY: `ensure_schema` is not just a bootstrap step — it runs every time a
        # store is opened, including against a table that already exists and is taking live
        # writes (e.g. a server restart). A plain `CREATE INDEX` takes a lock that blocks
        # writers for as long as the build takes, which for the HNSW index on a real corpus is
        # minutes, not milliseconds. `CONCURRENTLY` builds the index without that lock (a small
        # number of exclusive locks at the very start/end only). This is safe here because
        # `_op` runs on an autocommit connection (see `_connect_kwargs`) and, unlike
        # `replace_sources`/`upsert`, is never wrapped in `conn.transaction()` — every
        # statement in this method is already its own implicit transaction, which is the one
        # precondition `CREATE INDEX CONCURRENTLY` has (Postgres refuses it inside an explicit
        # transaction block). Verified directly against the container: back-to-back
        # `CREATE INDEX CONCURRENTLY IF NOT EXISTS` calls on this same autocommit-connection
        # shape all complete and land `indisvalid = true`.
        #
        # Trade-off accepted, not hidden: if the process is killed mid-build, Postgres can
        # leave an INVALID index behind, and `IF NOT EXISTS` treats that name as "already
        # there" on the next `ensure_schema()` — it will not retry the build. A plain
        # `CREATE INDEX` cannot fail this way (it is one transaction: abort undoes it), so this
        # trades a low-probability manual-cleanup case (`DROP INDEX <name>` then re-run) for
        # not blocking writers on every restart. `check_rls_effective`-style tooling could
        # detect an invalid index the same way; not added here to keep this change scoped to
        # issue #11's checkbox.
        conn.execute(f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {t}_tsv_idx ON {t} USING GIN (tsv)")
        conn.execute(
            f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {t}_emb_idx ON {t} "
            f"USING hnsw (embedding vector_cosine_ops)"
        )
        # The vector and full-text indexes cover the two retrieval legs; these cover the
        # rest of the hot path, each of which was a sequential scan growing with the corpus:
        #   indexed_at — `newest_indexed_at()` runs a max() on EVERY search; a DESC index
        #                turns it into a one-row backward scan.
        #   source     — a source-filtered search cannot use HNSW without it, and
        #                replace_sources/delete_sources match `source = ANY(...)` per re-index.
        #   file       — the supersession map groups on metadata->>'file'.
        conn.execute(
            f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {t}_indexed_at_idx ON {t} (indexed_at DESC)"
        )
        conn.execute(f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {t}_source_idx ON {t} (source)")
        conn.execute(
            f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {t}_file_idx ON {t} ((metadata->>'file'))"
        )
        # Every hot-path predicate leads with tenant_id, so it leads the index too.
        conn.execute(f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {t}_tenant_idx ON {t} (tenant_id)")
        self._enable_rls(conn)

    def _migrate_to_tenanted(self, conn: "psycopg.Connection") -> None:
        """Add `tenant_id` to a table created before tenancy existed, idempotently.

        An existing deployment has `id` as the sole primary key. Chunk ids are derived from the
        file path, so two tenants indexing the same path produce the SAME id — with a single-column
        key, one tenant's re-index would overwrite the other's row. The key therefore has to become
        `(tenant_id, id)`, which means dropping the old constraint.

        Existing rows are assigned to `DEFAULT_TENANT`, so an upgrade is invisible to a
        single-tenant deployment: the store's default tenant is the same value.

        Runs as ONE transaction, and its "already done?" check is the KEY, not the column. Both
        matter for the same reason: the migration is three ALTERs, and on an autocommit connection
        each commits separately. Interrupted after the DROP CONSTRAINT — a crash, a restart, a
        cancelled statement — the table is left with `tenant_id` but no primary key, and a check
        that only looked for the column would report the migration complete forever after. Every
        later `upsert` then fails on `ON CONFLICT (tenant_id, id)` with no matching constraint.
        DDL is transactional in Postgres (unlike the CONCURRENTLY builds in the caller), so the
        whole migration can simply be atomic.
        """
        t = self._table
        # Resolve through `search_path` (`%s::regclass`) rather than matching a bare table_name
        # across every visible schema: with a same-named table in another schema, an
        # information_schema lookup answers about the WRONG relation and skips the migration on
        # the real one. Every ALTER below resolves via search_path, so the check must too.
        has_tenant = conn.execute(
            "SELECT 1 FROM pg_attribute "
            "WHERE attrelid = %s::regclass AND attname = 'tenant_id' AND NOT attisdropped",
            (t,),
        ).fetchone()
        pkey = conn.execute(
            "SELECT conname, array_length(conkey, 1) FROM pg_constraint "
            "WHERE conrelid = %s::regclass AND contype = 'p'",
            (t,),
        ).fetchone()
        # Done only when BOTH halves landed: the column exists AND the key is the composite one.
        if has_tenant and pkey and pkey[1] == 2:
            return

        with conn.transaction():
            if not has_tenant:
                conn.execute(
                    f"ALTER TABLE {t} ADD COLUMN tenant_id TEXT NOT NULL "
                    f"DEFAULT '{DEFAULT_TENANT}'"
                )
            if pkey:
                conn.execute(f'ALTER TABLE {t} DROP CONSTRAINT "{pkey[0]}"')
            conn.execute(f"ALTER TABLE {t} ADD PRIMARY KEY (tenant_id, id)")

    def _enable_rls(self, conn: "psycopg.Connection") -> None:
        """Enforce tenant isolation in the DATABASE, not only in this class's WHERE clauses.

        Every query here already filters on `tenant_id`. That is the correctness mechanism and it
        works for any role. This is the CONTROL: a policy means a forgotten predicate — in future
        code, in a migration script, in someone's psql session — returns nothing instead of
        another tenant's memories.

        `FORCE` matters: without it the policy does not apply to the table's OWNER, which is
        usually the very role the application connects as.

        ⚠️ **A superuser bypasses RLS entirely, and so does a role with BYPASSRLS.** If the
        application connects as one (the default `docker-compose.yml` role IS a superuser), this
        policy is decoration and only the WHERE clauses are protecting you. See
        `check_rls_effective()`.
        """
        t = self._table
        policy = f"{t}_tenant_isolation"
        # Check before altering. `ensure_schema` runs on EVERY store open — including once per
        # tenant in the MCP server's registry — and these are ALTER-TABLE-class statements that
        # take ACCESS EXCLUSIVE whether or not the state is already correct, so an unconditional
        # version serialises the whole table against every concurrent reader on a routine open.
        # The DROP/CREATE pair is worse than slow: `CREATE POLICY` still has no `IF NOT EXISTS`,
        # so the old code dropped first — and on an autocommit connection those are two separate
        # commits, leaving a window in which the table has RLS FORCEd with NO policy. A query
        # from another tenant landing in that window matches nothing and returns zero rows, which
        # the trust layer reports as an ordinary "no memories found" rather than an error.
        # `polroles = {0}` is PUBLIC and `polcmd = '*'` is ALL — the two the CREATE below implies.
        # They are compared for the same reason the expressions are: `ALTER POLICY ... TO <role>`
        # narrows who the policy applies to, and a policy that applies to nobody relevant is an
        # isolation boundary switched off while still reporting as present.
        state = conn.execute(
            "SELECT c.relrowsecurity, c.relforcerowsecurity, "
            "       p.polname IS NOT NULL, "
            "       pg_get_expr(p.polqual, p.polrelid), "
            "       pg_get_expr(p.polwithcheck, p.polrelid), "
            "       p.polroles = '{0}'::oid[], p.polcmd "
            "FROM pg_class c "
            "LEFT JOIN pg_policy p ON p.polrelid = c.oid AND p.polname = %s "
            "WHERE c.oid = %s::regclass",
            (policy, t),
        ).fetchone()
        if state:
            rls_on, rls_forced, has_policy, using_expr, check_expr, all_roles, cmd = state
        else:  # pragma: no cover - the table was just created above, so it exists
            rls_on = rls_forced = has_policy = all_roles = False
            using_expr = check_expr = cmd = None

        if not rls_on:
            conn.execute(f"ALTER TABLE {t} ENABLE ROW LEVEL SECURITY")
        if not rls_forced:
            conn.execute(f"ALTER TABLE {t} FORCE ROW LEVEL SECURITY")

        # Compare the policy's DEFINITION, not merely its name. Checking only for existence
        # would make this stop being self-healing: a policy whose predicate had been altered —
        # by hand, by a migration, by an older version of this code — would be left in place
        # because something with the right name was there, and a changed predicate here is a
        # changed isolation boundary. `pg_get_expr` renders both sides in Postgres' own
        # normalised form, so the comparison is against what the server actually stores.
        want = f"(tenant_id = current_setting('{TENANT_GUC}'::text, true))"
        correct = (
            has_policy
            and using_expr == want
            and check_expr == want
            and all_roles          # applies to PUBLIC, not a narrowed role set
            and cmd == "*"         # applies to ALL commands, not just SELECT
        )
        if not correct:
            # DROP + CREATE in ONE transaction. Separately committed (this connection is
            # autocommit) they leave a window in which the table has RLS FORCEd and no policy,
            # during which every concurrent query returns zero rows — which the trust layer
            # reports as an ordinary "no memories found" rather than an error.
            with conn.transaction():
                conn.execute(f"DROP POLICY IF EXISTS {policy} ON {t}")
                conn.execute(
                    f"CREATE POLICY {policy} ON {t} "
                    f"USING (tenant_id = current_setting('{TENANT_GUC}', true)) "
                    f"WITH CHECK (tenant_id = current_setting('{TENANT_GUC}', true))"
                )

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
                    (tenant_id, id, source, text, metadata, embedding, indexed_at)
                VALUES (%s, %s, %s, %s, %s, %s, now())
                ON CONFLICT (tenant_id, id) DO UPDATE SET
                    source = EXCLUDED.source,
                    text = EXCLUDED.text,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding,
                    indexed_at = now()
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
        `hnsw.ef_search` / `hnsw.iterative_scan` tuning in `query_dense` inert. On a 20,000-row
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

    def _rows_to_hits(self, rows: list[tuple]) -> list[ScoredChunk]:
        hits: list[ScoredChunk] = []
        for cid, source, text, metadata, indexed_at, score in rows:
            md = metadata if isinstance(metadata, dict) else json.loads(metadata)
            hits.append(
                ScoredChunk(
                    chunk=Chunk(id=cid, source=source, text=text, metadata=md),
                    score=float(score),
                    indexed_at=indexed_at,
                )
            )
        return hits

    def _hnsw_filtered_tuning(self) -> tuple[int, str]:
        """`(ef_search, iterative_scan)` for a filtered dense query, from env or the defaults.

        Read fresh on every call (not cached at import/construction time) so a test can
        `monkeypatch.setenv` per-case and a long-lived process can pick up a changed value without
        restarting — the same convention `index_memory()` uses for `RECALL_INDEX_MAX_FILES`.
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
            # Interpolated into `SET LOCAL hnsw.ef_search` below, never bound — an out-of-range
            # value would only surface as a query-time error on every filtered search. Catch it
            # here, naming the variable, exactly as iterative_scan and the multiplier do.
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
        return ef_search, iterative_scan

    def query_dense(
        self, vector: list[float], k: int, source: str | None = None
    ) -> list[ScoredChunk]:
        if k <= 0:
            raise ValueError("k must be a positive int")
        t = self._table
        # Match the caller-facing identifier: recall_search surfaces the root-relative
        # `metadata->>'file'` (never the absolute `source` column), so a `source=` filter passed
        # back from a hit must resolve against `file` — falling back to `source` keeps legacy rows
        # (no `file` metadata) and any absolute-path caller working. Same rule as recall_forget.
        where = "AND (metadata->>'file' = %(source)s OR source = %(source)s)" if source else ""
        sql = f"""
            SELECT id, source, text, metadata, indexed_at, 1 - (embedding <=> %(vec)s) AS score
            FROM {t}
            WHERE tenant_id = %(tenant)s {where}
            ORDER BY embedding <=> %(vec)s
            LIMIT %(k)s
        """
        params: dict = {"vec": Vector(vector), "k": k, "tenant": self._tenant}
        if source:
            params["source"] = source

        if source:
            # Filtered arm only — see `DEFAULT_HNSW_EF_SEARCH_FILTERED` above for why the
            # unfiltered arm skips this. `SET LOCAL` only takes effect inside a transaction block;
            # on the autocommit connections this store uses, that means explicitly opening one
            # here, tuning the GUCs, then running the query, all before the transaction closes and
            # the tuning reverts. Values are validated/int-cast above, never taken as a bound
            # parameter — Postgres' `SET` does not accept one for the value.
            ef_search, iterative_scan = self._hnsw_filtered_tuning()

            def _op(conn: "psycopg.Connection") -> list[tuple]:
                with conn.transaction():
                    conn.execute(f"SET LOCAL hnsw.ef_search = {ef_search}")
                    conn.execute(f"SET LOCAL hnsw.iterative_scan = {iterative_scan}")
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
                    stacklevel=2,
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
        return self._rows_to_hits(rows)

    def query_sparse(
        self, text: str, k: int, source: str | None = None, vec: list[float] | None = None
    ) -> list[ScoredChunk]:
        """Full-text search. Ranking is always ts_rank; when `vec` is given, each hit's `score`
        is its true dense cosine against `vec` instead of the ts_rank value, so lexical-only
        hits are comparable with dense hits downstream.

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
        """
        if k <= 0:
            raise ValueError("k must be a positive int")
        t = self._table
        # See query_dense: the caller-facing identifier is the relative `file`, so match it (with
        # a `source` fall-back for legacy rows). Aliased `c.` here.
        where = (
            "AND (c.metadata->>'file' = %(source)s OR c.source = %(source)s)" if source else ""
        )
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
                SELECT id, source, text, metadata, indexed_at,
                       1 - (embedding <=> %(vec)s) AS score
                FROM (
                    SELECT c.id, c.source, c.text, c.metadata, c.indexed_at, c.embedding,
                           ts_rank(c.tsv, q.tsq) AS rank
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
                SELECT c.id, c.source, c.text, c.metadata, c.indexed_at,
                       ts_rank(c.tsv, q.tsq) AS score
                FROM {t} c, q
                WHERE c.tenant_id = %(tenant)s
                  AND c.tsv @@ q.tsq
                {where}
                ORDER BY score DESC
                LIMIT %(k)s
            """
        params: dict = {"q": text, "k": k, "tenant": self._tenant}
        if vec is not None:
            params["vec"] = Vector(vec)
        if source:
            params["source"] = source
        rows = self._with_retry(lambda conn: conn.execute(sql, params).fetchall())
        return self._rows_to_hits(rows)


    def replace_sources(
        self, sources: list[str], chunks: list[Chunk], embeddings: list[list[float]]
    ) -> int:
        """Atomically replace every row of `sources` with the given chunks.

        Delete + insert run in ONE transaction: a failure (or a concurrent reader) never
        observes the sources deleted without their replacement rows. Callers must compute
        `embeddings` BEFORE calling — an embedding failure then leaves the old rows intact.
        """
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        self._supersession_cache = None

        def _op(conn: "psycopg.Connection") -> int:
            with conn.transaction():
                if sources:
                    conn.execute(
                        f"DELETE FROM {self._table} "
                        f"WHERE tenant_id = %s AND source = ANY(%s)",
                        (self._tenant, sources),
                    )
                if chunks:
                    self._upsert_in(conn, chunks, embeddings)  # savepoint, same commit
            return len(chunks)

        self._with_retry(_op)
        return len(chunks)

    def delete_sources(self, sources: list[str]) -> int:
        """Delete every chunk belonging to the given `source` values; returns rows removed.

        Standalone removal API (the Indexer uses the atomic `replace_sources` instead).
        """
        if not sources:
            return 0
        self._supersession_cache = None
        # Read `rowcount` INSIDE the borrow, not from an escaped cursor: in pooled mode
        # `_with_retry` returns from within `with self._pool.connection()`, so a returned cursor
        # outlives its lease and belongs to a connection another thread may already hold.
        return self._with_retry(
            lambda conn: conn.execute(
                f"DELETE FROM {self._table} WHERE tenant_id = %s AND source = ANY(%s)",
                (self._tenant, sources),
            ).rowcount
            or 0
        )

    def touch_files(self, files: list[str]) -> int:
        """Reset ``indexed_at`` to now() for every chunk whose metadata file name matches.

        A timestamp-only touch — text and embeddings are untouched by construction, so it is
        the honest way to simulate a re-sync (the eval's recency arm uses it; re-indexing
        identical text would also re-embed, which is only a no-op for deterministic
        embedders). Matches on the ``file`` metadata key (basename), so it works for nested
        corpora too. Returns rows updated.
        """
        if not files:
            return 0
        # rowcount read inside the borrow — see `delete_sources` for why.
        return self._with_retry(
            lambda conn: conn.execute(
                f"UPDATE {self._table} SET indexed_at = now() "
                f"WHERE tenant_id = %s AND metadata->>'file' = ANY(%s)",
                (self._tenant, files),
            ).rowcount
            or 0
        )

    def supersession(self) -> tuple[dict[str, str], frozenset[str]]:
        """The supersession relation: ``(edges, unresolved)``.

        `edges` maps superseded file -> superseding file (both root-relative). `unresolved`
        names the files an edge pointed at but could not identify, so the read path can fail
        closed on them rather than serve them as healthy.


        The ``supersedes:`` frontmatter references its target by basename (the authoring
        convention), but files are identified by their root-relative path so same-named files in
        different directories cannot collide. This resolves each basename reference to the unique
        file that bears it: an AMBIGUOUS target (a basename shared by two files) is skipped
        rather than mis-mapped — the same refusal `recall lint` makes when it flags
        ``ambiguous-supersedes-target`` — so a stray sibling can never be silently marked
        superseded. Only the ambiguous case is skipped. A DANGLING target (no such basename in
        the corpus) is instead kept, as an edge keyed on the raw basename it was written as:
        harmless, because that key matches no indexed file, and dropping it would silently
        discard a valid supersession claim (e.g. a memo superseding a doc since removed).

        The result is cached, but the cache is VALIDATED against the table on every call rather
        than trusted. It previously was not: it was invalidated only by this instance's own
        writes, so a long-lived reader (an MCP server holds one store for its lifetime) never saw
        an edge written by a separate `recall index` run. It kept serving the superseded memory as
        `ok` until someone restarted the process — the trust layer returning exactly the wrong
        answer, silently, which is the failure it exists to prevent.

        Freshness is established by a cheap fingerprint — `(max(indexed_at), count(*))` for this
        tenant — and the expensive grouped scan runs only when that fingerprint moves. Both
        halves are needed: `max(indexed_at)` alone cannot see a DELETE, and deleting a superseding
        document must stop its edge from applying, or the reader keeps demoting a memory that is
        current again.

        Measured on a 50k-row table: fingerprint ~12 ms, full scan ~80 ms. So this is cheaper than
        rescanning on every call and, unlike caching indefinitely, it is correct.

        Fingerprint and scan share ONE connection, so they cannot straddle a concurrent write and
        cache a result under a fingerprint that never described it.
        """

        edges, unresolved, _dates = self.supersession_all()  # already copies
        return edges, unresolved

    def supersession_all(
        self,
    ) -> tuple[dict[str, str], frozenset[str], dict[str, datetime]]:
        """``(edges, unresolved, edge_dates)`` behind one validated cache and one scan.

        `edge_dates` maps a superseded file to the moment its edge became assertable, which is
        when the SUPERSEDING document was first written. That is `min(indexed_at)` over the
        superseding file's claim-carrying chunks: a chunk of it existing implies its
        `supersedes:` frontmatter existed, and the earliest is the conservative choice.

        Prefer this over calling `supersession()` and `supersession_dates()` in sequence. Each
        enters its own cache validation with its own fingerprint query, so a concurrent index
        between the two can hand back edges from one scan dated by the next.

        🔴 **SECOND OPEN DEFECT, independent of `first_indexed_at`: fan-in resolves
        LEXICOGRAPHICALLY, and lexicographic is orthogonal to time.** Where `b1.md` (written
        Monday) and `b2.md` (Wednesday) both supersede `a.md`, only `b2.md` survives, so a replay
        as of Tuesday finds its single edge dated Wednesday, drops it, and answers `ok`. On
        Tuesday `a.md` WAS superseded, by `b1.md`. Whether a corpus gets the wrong answer is
        decided by the alphabet: rename the two and the same data answers correctly. The older
        edge's date is computed and then discarded, because only the surviving winner is dated.
        Fixing it means keeping every candidate edge per target and letting `resolve_successor`
        choose the latest one asserted at or before the instant, rather than picking one winner
        here. Pinned by `test_fan_in_replay_must_use_the_edge_live_at_the_instant`.

        `ORDER BY` is load-bearing, not tidiness. `resolve_supersession` resolves fan-in by
        last-row-wins, so an unordered scan lets the
        winner change between runs. The predecessor query was `SELECT DISTINCT`, and swapping it
        for `GROUP BY` preserves the row SET but not the row ORDER, which would have re-rolled
        that winner for existing `supersession()` callers who never asked for any of this.
        `COLLATE "C"` pins it to byte order, so the winner is the same on a `C`-collation CI
        database and an `en_US.UTF-8` developer one. It is stable, not necessarily IDENTICAL to
        what `SELECT DISTINCT` produced: under a Sort+Unique plan it is the same winner, under
        HashAggregate the old one was bucket order.

        🔴 **OPEN DEFECT, do not merge the point-in-time rewind on this alone.** `indexed_at`
        records the LAST write, not the first: `replace_sources` re-inserts with `indexed_at =
        now()` while `Indexer.index_path` skips files whose content hash is unchanged. So editing
        a superseding memo moves ITS date forward while its predecessor keeps the old one, and a
        past replay then drops a long-standing edge and serves the superseded memory as `ok` —
        the exact wrong answer this layer exists to prevent, where the pre-change code correctly
        said `superseded`. The same flaw already affects `known_as_of` on HITS (a re-indexed
        memory reads `not_yet_known` for a past instant), so the root cause predates this change;
        what is new is that it flips a verdict from safe to unsafe. Fixing it properly needs a
        `first_indexed_at` column preserved across upserts, which is a migration and needs a
        database to verify.
        """

        def _op(
            conn: "psycopg.Connection",
        ) -> tuple[dict[str, str], frozenset[str], dict[str, datetime]]:
            fingerprint = conn.execute(
                f"SELECT max(indexed_at), count(*) FROM {self._table} WHERE tenant_id = %s",
                (self._tenant,),
            ).fetchone()
            cached = self._supersession_cache
            if cached is not None and cached[0] == fingerprint:
                return cached[1], cached[2], cached[3]
            rows = conn.execute(
                f"""
                SELECT metadata->>'file' AS file,
                       metadata->>'supersedes' AS supersedes,
                       min(indexed_at) AS first_indexed
                FROM {self._table}
                WHERE tenant_id = %s AND metadata ? 'file'
                GROUP BY 1, 2
                ORDER BY 1 COLLATE "C", 2 COLLATE "C"
                """,
                (self._tenant,),
            ).fetchall()
            self._supersession_scans += 1
            METRICS.increment("recall_supersession_scans_total")
            edges, unresolved = resolve_supersession([(r[0], r[1]) for r in rows])
            edge_dates = resolve_edge_dates(rows, edges)
            self._supersession_cache = (fingerprint, edges, unresolved, edge_dates)
            return edges, unresolved, edge_dates

        edges, unresolved, edge_dates = self._with_retry(_op)
        # COPIES. This is public, and the cache is process-wide and validated by fingerprint
        # rather than rebuilt, so handing out the live dicts would let one caller's mutation
        # survive every later cache hit and redirect other callers' supersession verdicts.
        # `supersession()` and `supersession_dates()` have always copied; this now matches them.
        return dict(edges), unresolved, dict(edge_dates)

    def supersession_dates(self) -> dict[str, datetime]:
        """When each supersession edge became assertable, for point-in-time replay.

        Feeds `resolve_successor(..., edge_dates=, known_as_of=)` so a past-instant query resolves
        to the successor that was current *then*. Shares `supersession()`'s validated cache, so
        this needs no second scan when the edges have already been read, though it does re-run
        the fingerprint query. ⚠️ Pairing this with `supersession()` reintroduces the race that
        `supersession_all()` exists to close: two independent validations can return edges from
        one scan dated by the next. Prefer `supersession_all()` whenever you want both.
        """
        return self.supersession_all()[2]  # already a copy

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

    def source_content_hashes(self) -> dict[str, str]:
        """`{source: content_hash}` for this tenant — what the indexer compares against.

        One row per source: the hash is a property of the file, so every chunk of it carries the
        same value and `DISTINCT` collapses them. A source indexed before content hashing existed
        has no hash and is reported as `""`, which can never equal a real sha256 — so it is
        re-indexed once and then skipped like everything else.
        """
        rows = self._with_retry(
            lambda conn: conn.execute(
                f"SELECT DISTINCT source, coalesce(metadata->>'content_hash', '') "
                f"FROM {self._table} WHERE tenant_id = %s",
                (self._tenant,),
            ).fetchall()
        )
        return {source: content_hash for source, content_hash in rows}

    def supersession_map(self) -> dict[str, str]:
        """Resolvable supersession edges only — convenience view of `supersession`."""
        return self.supersession()[0]

    def newest_indexed_at(self) -> datetime | None:
        row = self._with_retry(
            lambda conn: conn.execute(
                f"SELECT max(indexed_at) FROM {self._table} WHERE tenant_id = %s",
                (self._tenant,),
            ).fetchone()
        )
        return row[0] if row else None

    def count(self) -> int:
        row = self._with_retry(
            lambda conn: conn.execute(
                f"SELECT count(*) FROM {self._table} WHERE tenant_id = %s",
                (self._tenant,),
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
        if self._pool is not None:
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
        with self._borrowed() as conn:
            # An explicit transaction even though the connection is autocommit: a server-side
            # cursor is transaction-scoped, and under autocommit each FETCH would otherwise land
            # in its own transaction, where the cursor no longer exists.
            with conn.transaction():
                with conn.cursor(name=f"recall_iter_{uuid4().hex[:12]}") as cur:
                    cur.itersize = batch_size
                    cur.execute(
                        f"SELECT id, source, text, metadata FROM {self._table} "
                        "WHERE tenant_id = %s ORDER BY id",
                        (self._tenant,),
                    )
                    for cid, source, text, metadata in cur:
                        yield Chunk(id=cid, source=source, text=text, metadata=metadata or {})
