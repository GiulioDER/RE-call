"""The decision ledger: an append-only witness of what the trust layer decided, and why.

`trusted_search` already assembles everything an auditor would later ask for — the query, every
hit's verdict and score, the calibration and generation that governed the decision, the losing
evidence, and the abstention reason — and then returns it in a `TrustedResult` that dies with
the caller's stack frame. The aggregate counters in `recall.observability` answer HOW OFTEN it
abstains; nothing answers WHY it abstained on Tuesday. This module closes that gap: one row per
decision, appended to ``recall_audit_events``, the same append-only, tenant-isolated table the
write side (generation lifecycle, calibration lifecycle) has recorded into since migration 0008.

**Witness, not enforcer.** The trust layer is the enforcer at the boundary — strict mode
refuses, degraded mode demotes — and this module only records what that boundary returned. It
follows that a ledger failure must never fail the search: every write is best-effort, counted
(``recall_ledger_write_failures_total``) and logged once per failure kind, never raised. The one
thing a witness must not do is become load-bearing.

**A record holds references, never corpus bytes.** Source, file, ord, chunk id, scores and
verdicts go in; chunk text does not, so enabling the ledger cannot create a second, unguarded
copy of the corpus. The query IS recorded — it is the trigger of the decision and the record is
unreadable without it. That is deliberately a different rule than `TrustRefusal` follows (no
query in exceptions): an exception leaks into every log line and traceback that touches it,
while a ledger row lands only in a table that is RLS-isolated per tenant and reachable only
through serving credentials.

Off by default. Enable per process with ``RECALL_DECISION_LEDGER=1`` (read by the CLI ``search``
command and the MCP service), or per call by passing a `DecisionLedger` to
``trusted_search(ledger=...)``.

The record's shape follows the reasoning-ledger pattern (trigger, evidence with authority
identity, losing evidence, outcome, two time axes): ``valid_time`` is the instant the verdicts
were judged against, ``known_as_of`` the transaction-time replay instant, and ``created_at`` is
stamped by the database on the row itself.
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Protocol

from recall.observability import METRICS, get_logger
from recall.trust_policy import TrustRefusal
from recall.types import TrustedResult

_log = get_logger("decision_ledger")

#: Event types this module appends. Namespaced under the table's existing `event_type` column so
#: a reader can separate read-side decisions from the write-side lifecycle events that share it.
SEARCH_DECISION_EVENT = "search_decision"
SEARCH_REFUSAL_EVENT = "search_refusal"

#: Payload schema version, bumped when a field changes meaning. A ledger is read years after it
#: is written, by code that no longer remembers this release; the version is what lets that
#: reader tell a missing field from a field that did not exist yet.
RECORD_VERSION = 1

#: Longest query recorded verbatim. The query is the trigger and belongs in the record, but the
#: record must stay bounded: an unbounded caller string is an unbounded row. The MCP service
#: REFUSES queries over its own 4096-char limit (it never truncates), while the CLI accepts any
#: length, so the ledger bounds the recorded copy itself. Truncation is declared in the payload
#: rather than silent — `query_chars` keeps the true length.
MAX_QUERY_CHARS_RECORDED = 2000

#: Longest corpus-controlled identifier (source, file, successor) recorded per hit. These are
#: paths chosen by whoever can write a file into the corpus, so without a bound one degenerate
#: file name controls the row size — the same reasoning `MAX_QUERY_CHARS_RECORDED` states for
#: the caller's string, applied to the corpus author's.
MAX_REF_CHARS_RECORDED = 512

#: The env flag `from_env` reads, and the values it accepts. Same vocabulary as
#: `RECALL_ENTAILMENT` so operators learn one boolean dialect.
LEDGER_ENV = "RECALL_DECISION_LEDGER"
_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"", "0", "false", "no", "off"})

#: Warnings already emitted, keyed by kind, so a per-query failure (dead table, revoked grant)
#: is reported once per process instead of once per search. The metric still counts every one.
#: Not thread-guarded for the same reason `trust._WARNED_UNCALIBRATED` is not: a duplicate
#: warning under a race is harmless, a lock on the search path is not.
_WARNED: set[str] = set()


def _warn_once(kind: str, message: str, *args: object) -> None:
    if kind in _WARNED:
        return
    _WARNED.add(kind)
    _log.warning(message, *args)


def _terse(exc: Exception) -> str:
    """The exception's first line, bounded — the shape safe to put in a log.

    Driver error text can quote payload fragments: a jsonb parse failure's CONTEXT line carries
    the JSON data, and a unique-violation DETAIL quotes key values. The payload holds the
    query, which is the one string this module's own rules keep out of log lines, so the log
    gets the error's headline and the metric gets the count — nothing an operator needs is in
    the tail this drops.
    """
    lines = str(exc).splitlines() or [""]
    return lines[0][:200]


class _AuditStore(Protocol):
    """The one store capability this module needs. Kept structural on purpose: trust-layer
    stores are duck-typed in tests and adapters, and demanding `PgVectorStore` here would force
    every fake to inherit a database."""

    def append_audit_event(
        self,
        event_type: str,
        payload: dict[str, Any],
        *,
        generation_id: str | None = ...,
        actor: str = ...,
    ) -> str: ...


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _iso_instant(value: datetime | None) -> str | None:
    """A CALLER-supplied instant, serialized under the same rule enforcement used.

    `evaluate` interprets a naive ``now`` (and `_as_utc` a naive ``known_as_of``) as UTC before
    judging any verdict, so a naive value recorded verbatim would be an offset-free string a
    reader could parse as local time — reconstructing an instant off by the local UTC offset
    from the one the verdicts were actually judged against. The record states the offset
    enforcement applied. `_iso` stays verbatim for store-derived timestamps, which arrive
    timezone-aware from TIMESTAMPTZ columns.
    """
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.isoformat()


#: Characters json.dumps encodes as escapes PostgreSQL's jsonb PARSER rejects: U+0000 becomes
#: backslash-u0000 escape ("unsupported Unicode escape sequence") and a lone surrogate an unpairable
#: ``\ud800``-range escape. Left in the payload they make the INSERT fail deterministically on
#: every attempt, which turns the best-effort catch into something worse than lost-row-on-outage:
#: a caller who embeds one in the query chooses to be unrecorded, on the refusal path where the
#: search itself never touches the database. Replaced, not passed through.
_JSONB_UNSTORABLE = re.compile("[\x00\ud800-\udfff]")


def _score(value: Any) -> float | None:
    """A score as a plain, storable float — or None where no storable number exists.

    Raw, not rounded: `round(x, 6)` could move the recorded score to the other side of the
    calibration threshold the verdict was decided on, so the row contradicted its own decision
    at exactly the boundary a drift study inspects. JSON carries the full double; precision
    costs nothing and ambiguity bought nothing. `float()` also collapses numpy scalars from
    duck-typed stores into JSON-native floats, and a non-finite value (a NaN cosine from a
    degenerate vector) becomes None rather than a token PostgreSQL's jsonb parser rejects —
    left in, it would silently drop EVERY decision containing that chunk, not one row once.
    """
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _bounded_ref(value: str | None) -> str | None:
    """A corpus-controlled identifier, bounded the way the query is. Truncation appends an
    ellipsis so a cut name cannot silently impersonate a complete one."""
    if value is None or len(value) <= MAX_REF_CHARS_RECORDED:
        return value
    return value[:MAX_REF_CHARS_RECORDED] + "…"


def _bounded_query(query: str) -> dict[str, Any]:
    truncated = len(query) > MAX_QUERY_CHARS_RECORDED
    recorded = query[:MAX_QUERY_CHARS_RECORDED]
    sanitized = _JSONB_UNSTORABLE.sub("�", recorded)
    fields: dict[str, Any] = {
        "query": sanitized,
        "query_chars": len(query),
        "query_truncated": truncated,
    }
    if sanitized != recorded:
        # Declared, like truncation: a record that was altered on the way in must say so, or a
        # reader will treat the replacement characters as what the caller actually sent.
        fields["query_sanitized"] = True
    return fields


def decision_payload(
    result: TrustedResult,
    *,
    k: int | None = None,
    valid_time: datetime | None = None,
    known_as_of: datetime | None = None,
) -> dict[str, Any]:
    """The JSON body of one decision record, built purely from the result and the call's inputs.

    Pure and total: no clock reads (the row's ``created_at`` is the database's), no store access,
    nothing that can fail on a well-formed `TrustedResult`. Every hit is recorded — the demoted
    ones are the losing evidence, and a record showing only the winners would be justification,
    not audit. Hit entries carry identity and judgment (source, file, ord, chunk id, cosine,
    confidence, verdict, window, successor) and never the chunk's text.

    ``valid_time`` is the caller's ``now`` — None means the verdicts were judged against the
    wall clock, which ``created_at`` then approximates. ``known_as_of`` non-null marks the record
    as a replay, whose conclusions are about the corpus as it stood then, not now.
    """
    staleness = result.staleness
    diagnostics = result.diagnostics
    payload: dict[str, Any] = {
        "record_version": RECORD_VERSION,
        **_bounded_query(result.query),
        "k": k,
        "outcome": "abstained" if result.abstained else "answered",
        "abstained": result.abstained,
        "reason": result.reason,
        "trust_state": result.trust_state,
        "failure_code": result.failure_code,
        "calibration_id": result.calibration_id,
        "calibration_status": result.calibration_status,
        "calibrated": result.calibrated,
        "tenant_id": result.tenant_id,
        "generation_id": result.generation_id,
        "pipeline_fingerprint": result.pipeline_fingerprint,
        "corpus_fingerprint": result.corpus_fingerprint,
        "query_set_digest": result.query_set_digest,
        "gap_warning": result.gap_warning,
        "staleness": {
            "stale": staleness.stale,
            "age_seconds": (
                staleness.age.total_seconds() if staleness.age is not None else None
            ),
            # The bound beside the measurement: `stale` is `age > max_age`, and a row keeping
            # the numerator while discarding the bound records a verdict nobody can re-derive —
            # identical ages carry opposite flags across tenants with different configurations.
            "max_age_seconds": staleness.max_age.total_seconds(),
        },
        "valid_time": _iso_instant(valid_time),
        "known_as_of": _iso_instant(known_as_of),
        "diagnostics": {
            "embedding_profile": diagnostics.embedding_profile,
            "retrieval_profile": diagnostics.retrieval_profile,
            "index_generation": diagnostics.index_generation,
            "candidate_pool_size": diagnostics.candidate_pool_size,
            "reranking_ran": diagnostics.reranking_ran,
            "stage_ms": dict(diagnostics.stage_ms),
        },
        "verdict_counts": dict(Counter(hit.verdict for hit in result.hits)),
        "hits": [
            {
                "chunk_id": hit.chunk.id,
                "source": _bounded_ref(hit.provenance.source),
                "file": _bounded_ref(hit.provenance.file),
                "ord": hit.provenance.ord,
                "cosine": _score(hit.cosine),
                "confidence": _score(hit.confidence),
                "verdict": hit.verdict,
                "superseded_by": _bounded_ref(hit.validity.superseded_by),
                "valid_from": _iso(hit.validity.valid_from),
                "valid_until": _iso(hit.validity.valid_until),
                "indexed_at": _iso(hit.provenance.indexed_at),
            }
            for hit in result.hits
        ],
    }
    return payload


def refusal_payload(
    refusal: TrustRefusal,
    *,
    query: str,
    k: int | None = None,
    known_as_of: datetime | None = None,
) -> dict[str, Any]:
    """The JSON body of one refusal record.

    A refusal is a decision too — the enforcer answered "no answer" — and a ledger that only
    witnessed the successes would go silent exactly when the gate is doing its job. Not only
    strict mode lands here: a ``DEPENDENCY_UNAVAILABLE`` refusal is raised in any mode, before
    the strict gate is consulted, and it is recorded the same way; ``failure_code`` says which. The fields
    are the ones `TrustRefusal` itself carries (all system-controlled identity, no corpus
    bytes), plus the query, which the exception deliberately omits and the record deliberately
    keeps: see the module docstring for why those rules differ.
    """
    return {
        "record_version": RECORD_VERSION,
        **_bounded_query(query),
        "k": k,
        "outcome": "refused",
        "failure_code": refusal.code.value,
        "calibration_status": refusal.calibration_status,
        "tenant_id": refusal.tenant_id,
        "generation_id": refusal.generation_id,
        "calibration_id": refusal.calibration_id,
        "pipeline_fingerprint": refusal.pipeline_fingerprint,
        "corpus_fingerprint": refusal.corpus_fingerprint,
        "query_set_digest": refusal.query_set_digest,
        "known_as_of": _iso_instant(known_as_of),
    }


class DecisionLedger:
    """Appends decision records through a store's audit surface, and never raises.

    Holds no connection of its own: the write goes through
    ``store.append_audit_event``, so it runs on the same tenant-scoped, RLS-guarded
    connections every other statement uses, and a store without that surface simply fails the
    best-effort write. Construction is free — one per call site is fine.
    """

    def __init__(self, store: _AuditStore, *, actor: str = "trust-layer") -> None:
        self._store = store
        self._actor = actor

    @classmethod
    def from_env(
        cls,
        store: _AuditStore,
        *,
        env: dict[str, str] | None = None,
        actor: str = "trust-layer",
    ) -> "DecisionLedger | None":
        """A ledger when ``RECALL_DECISION_LEDGER`` enables one, else None.

        A malformed value disables the ledger with a once-per-process warning rather than
        raising. This runs on serving paths, per request: raising would turn a typo in an env
        var into a refusal of every search, which is enforcement — exactly what a witness must
        not do. The warning (and only the warning) is what stands between the operator and
        silently missing audit rows, so it names the variable and the accepted values.
        """
        source = env if env is not None else os.environ
        raw = source.get(LEDGER_ENV, "").strip().lower()
        if raw in _FALSE:
            return None
        if raw not in _TRUE:
            # Keyed on a CONSTANT, not the raw value: the docstring promises once per process,
            # and a key domain the caller controls would grow _WARNED without bound.
            _warn_once(
                "env",
                "%s=%r is not a boolean, so the decision ledger stays OFF. Use one of %s to "
                "enable it.",
                LEDGER_ENV,
                raw,
                sorted(_TRUE),
            )
            return None
        return cls(store, actor=actor)

    def _append(
        self,
        event_type: str,
        build: "Callable[[], tuple[dict[str, Any], str | None]]",
    ) -> str | None:
        """Build the payload and write it, under ONE protective boundary.

        The payload builders are pure and total over well-formed inputs, but the inputs are
        duck-typed in adapters and tests, and an ill-formed result must fail the WRITE, not the
        search. Building outside this try was the bug: a broken `staleness` object would have
        raised out of the witness — the exact failure mode this class exists to rule out.
        """
        try:
            payload, generation_id = build()
            event_id = self._store.append_audit_event(
                event_type,
                payload,
                generation_id=generation_id,
                actor=self._actor,
            )
            # INSIDE the boundary, so the contract has no three-line gap at the end: a metrics
            # registry that failed here (however unlikely) would otherwise turn an
            # already-computed answer into an error.
            METRICS.increment("recall_ledger_records_total", event=event_type)
            return event_id
        except Exception as exc:
            # Any failure, including AttributeError from a store without the audit surface.
            # The search result is already computed and correct; losing one audit row must not
            # turn it into an error. Counted every time, said once per failure kind — and the
            # bookkeeping itself is guarded, because a raise from inside this clause would
            # escape the witness on exactly the path that promises it cannot.
            try:
                METRICS.increment("recall_ledger_write_failures_total", event=event_type)
                _warn_once(
                    f"write:{type(exc).__name__}",
                    "decision ledger write failed (%s: %s) — the search still answered; "
                    "further failures of this kind are counted in "
                    "recall_ledger_write_failures_total without this warning",
                    type(exc).__name__,
                    _terse(exc),
                )
            except Exception:  # pragma: no cover - bookkeeping must not out-enforce the witness
                pass
            return None

    def record_decision(
        self,
        result: TrustedResult,
        *,
        k: int | None = None,
        valid_time: datetime | None = None,
        known_as_of: datetime | None = None,
    ) -> str | None:
        """Witness one completed decision. Returns the event id, or None if the write failed."""
        return self._append(
            SEARCH_DECISION_EVENT,
            lambda: (
                decision_payload(result, k=k, valid_time=valid_time, known_as_of=known_as_of),
                result.generation_id,
            ),
        )

    def record_refusal(
        self,
        refusal: TrustRefusal,
        *,
        query: str,
        k: int | None = None,
        known_as_of: datetime | None = None,
    ) -> str | None:
        """Witness one refusal (strict gate, or a dependency fault in any mode)."""
        return self._append(
            SEARCH_REFUSAL_EVENT,
            lambda: (
                refusal_payload(refusal, query=query, k=k, known_as_of=known_as_of),
                refusal.generation_id,
            ),
        )
