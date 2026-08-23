"""The decision ledger: search decisions outlive the stack frame, and never break the search.

Three properties are under test, in rising order of integration:

1. the payload builders are pure, bounded, and record losing evidence but never corpus text;
2. `DecisionLedger` is a witness — a failed write costs a counter, never an exception, and a
   malformed env flag disables it rather than refusing anything;
3. through `trusted_search`, one call appends exactly one row to `recall_audit_events` — the
   answered, abstained, and strict-refusal outcomes all leave a record, and the refusal still
   raises.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from recall.decision_ledger import (
    MAX_QUERY_CHARS_RECORDED,
    SEARCH_DECISION_EVENT,
    SEARCH_REFUSAL_EVENT,
    DecisionLedger,
    decision_payload,
    refusal_payload,
)
from recall.observability import METRICS
from recall.trust_policy import TrustFailureCode, TrustRefusal
from recall.types import (
    Chunk,
    Provenance,
    RetrievalDiagnostics,
    StalenessReport,
    TrustedHit,
    TrustedResult,
    Validity,
)
from tests.conftest import requires_db

SECRET_TEXT = "the corpus text that must never reach the ledger"


def _hit(verdict: str, file: str, cosine: float, superseded_by: str | None = None) -> TrustedHit:
    return TrustedHit(
        chunk=Chunk(id=f"{file}#0", source=file, text=SECRET_TEXT),
        cosine=cosine,
        confidence=cosine,
        verdict=verdict,
        provenance=Provenance(
            source=file, file=file, ord=0, indexed_at=datetime(2026, 1, 2, tzinfo=UTC)
        ),
        validity=Validity(valid_from=None, valid_until=None, superseded_by=superseded_by),
    )


def _result(hits: list[TrustedHit], abstained: bool = False, reason: str = "") -> TrustedResult:
    return TrustedResult(
        query="what did we decide about caching?",
        hits=hits,
        abstained=abstained,
        reason=reason,
        gap_warning=False,
        staleness=StalenessReport(
            stale=True,
            newest_indexed_at=datetime(2026, 1, 2, tzinfo=UTC),
            age=timedelta(days=3),
            max_age=timedelta(days=1),
        ),
        diagnostics=RetrievalDiagnostics(stage_ms={"dense": 1.5}),
        calibration_id="cal_abc",
        calibration_status="certified",
        generation_id="gen_1",
        trust_state="trusted",
    )


def test_decision_payload_records_losing_evidence_and_no_corpus_text():
    result = _result(
        [_hit("ok", "rate_v2.md", 0.71), _hit("superseded", "rate_v1.md", 0.80, "rate_v2.md")]
    )
    payload = decision_payload(result, k=5, known_as_of=None, valid_time=None)
    assert payload["outcome"] == "answered"
    assert payload["verdict_counts"] == {"ok": 1, "superseded": 1}
    # The demoted hit is the losing evidence and must be in the record, successor named.
    loser = next(h for h in payload["hits"] if h["verdict"] == "superseded")
    assert loser["superseded_by"] == "rate_v2.md"
    assert loser["cosine"] == 0.8
    # References only: the serialized record must not contain a byte of chunk text.
    assert SECRET_TEXT not in json.dumps(payload)
    assert payload["staleness"] == {"stale": True, "age_seconds": 3 * 86400.0}
    assert payload["calibration_id"] == "cal_abc"
    assert payload["generation_id"] == "gen_1"


def test_decision_payload_abstention_keeps_the_reason():
    result = _result(
        [_hit("expired", "freeze.md", 0.9)], abstained=True, reason="best candidate expired"
    )
    payload = decision_payload(result)
    assert payload["outcome"] == "abstained"
    assert payload["abstained"] is True
    assert payload["reason"] == "best candidate expired"
    assert payload["verdict_counts"] == {"expired": 1}


def test_decision_payload_bounds_the_query_and_says_so():
    result = _result([_hit("ok", "a.md", 0.7)])
    long_query = "q" * (MAX_QUERY_CHARS_RECORDED + 500)
    payload = decision_payload(
        TrustedResult(
            query=long_query,
            hits=result.hits,
            abstained=False,
            reason="",
            gap_warning=False,
            staleness=result.staleness,
        )
    )
    assert len(payload["query"]) == MAX_QUERY_CHARS_RECORDED
    assert payload["query_chars"] == MAX_QUERY_CHARS_RECORDED + 500
    assert payload["query_truncated"] is True


def test_jsonb_unstorable_characters_are_replaced_and_declared():
    """A NUL or lone surrogate would make the jsonb INSERT fail on EVERY attempt, letting a
    caller suppress their own refusal record deterministically (SEC-001). Replaced with U+FFFD
    and declared, so the record lands and says it was altered."""
    result = _result([_hit("ok", "a.md", 0.7)])
    hostile = TrustedResult(
        query="who\x00did\ud800this",
        hits=result.hits,
        abstained=False,
        reason="",
        gap_warning=False,
        staleness=result.staleness,
    )
    payload = decision_payload(hostile)
    assert "\x00" not in payload["query"] and "\ud800" not in payload["query"]
    assert payload["query_sanitized"] is True
    json.dumps(payload)  # must survive the exact serialization Jsonb applies
    # And an untouched query does not carry the flag at all.
    assert "query_sanitized" not in decision_payload(result)


def test_caller_instants_are_recorded_with_the_offset_enforcement_used():
    """evaluate() interprets a naive `now` as UTC before judging verdicts; the record must state
    that offset rather than storing an ambiguous local-looking string (NUM-001)."""
    naive = datetime(2026, 1, 1, 12, 0)
    payload = decision_payload(
        _result([_hit("ok", "a.md", 0.7)]), valid_time=naive, known_as_of=naive
    )
    assert payload["valid_time"] == "2026-01-01T12:00:00+00:00"
    assert payload["known_as_of"] == "2026-01-01T12:00:00+00:00"
    refusal = TrustRefusal(
        code=TrustFailureCode.INDEX_NOT_READY,
        calibration_status="missing",
        tenant_id="default",
        generation_id=None,
    )
    ref = refusal_payload(refusal, query="q", known_as_of=naive)
    assert ref["known_as_of"] == "2026-01-01T12:00:00+00:00"


def test_refusal_payload_carries_code_and_query():
    refusal = TrustRefusal(
        code=TrustFailureCode.INDEX_NOT_READY,
        calibration_status="missing",
        tenant_id="default",
        generation_id=None,
    )
    payload = refusal_payload(refusal, query="who set the rate limit?", k=3)
    assert payload["outcome"] == "refused"
    assert payload["failure_code"] == "INDEX_NOT_READY"
    assert payload["query"] == "who set the rate limit?"
    assert payload["k"] == 3


class _FailingStore:
    def append_audit_event(self, *args, **kwargs):
        raise RuntimeError("relation recall_audit_events does not exist")


class _RecordingStore:
    def __init__(self):
        self.events: list[tuple[str, dict, dict]] = []

    def append_audit_event(self, event_type, payload, **kwargs):
        self.events.append((event_type, payload, kwargs))
        return "evt_test"


def test_write_failure_never_raises_and_is_counted():
    before = METRICS.snapshot()["counters"]
    ledger = DecisionLedger(_FailingStore())
    out = ledger.record_decision(_result([_hit("ok", "a.md", 0.7)]))
    assert out is None  # the search result was already correct; the witness just missed
    after = METRICS.snapshot()["counters"]
    failures = {k: v for k, v in after.items() if k.startswith("recall_ledger_write_failures")}
    assert sum(failures.values()) > sum(
        v for k, v in before.items() if k.startswith("recall_ledger_write_failures")
    )


def test_missing_audit_surface_is_a_write_failure_not_an_error():
    ledger = DecisionLedger(object())  # no append_audit_event at all
    assert ledger.record_decision(_result([_hit("ok", "a.md", 0.7)])) is None


def test_an_ill_formed_result_fails_the_write_not_the_caller():
    """Payload building happens under the same protective boundary as the insert.

    Adapters and tests duck-type results, so the builder can meet an object whose fields are
    broken. That must lose the audit row, never raise out of the witness: building the payload
    OUTSIDE the try was a real bug caught in review.
    """
    from dataclasses import replace

    ledger = DecisionLedger(_RecordingStore())
    broken = replace(_result([_hit("ok", "a.md", 0.7)]), staleness=object())
    assert ledger.record_decision(broken) is None


def test_from_env_defaults_off_and_rejects_gibberish():
    store = _RecordingStore()
    assert DecisionLedger.from_env(store, env={}) is None
    assert DecisionLedger.from_env(store, env={"RECALL_DECISION_LEDGER": "0"}) is None
    assert DecisionLedger.from_env(store, env={"RECALL_DECISION_LEDGER": "1"}) is not None
    assert DecisionLedger.from_env(store, env={"RECALL_DECISION_LEDGER": "yes"}) is not None
    # A typo disables with a warning; raising here would refuse every search over an env var,
    # which is enforcement, and the one thing the witness must never do.
    assert DecisionLedger.from_env(store, env={"RECALL_DECISION_LEDGER": "ture"}) is None


def test_wrapper_records_the_final_result_once(monkeypatch):
    import recall.trust as trust

    canned = _result([_hit("ok", "a.md", 0.7)])
    monkeypatch.setattr(trust, "_trusted_search", lambda *a, **k: canned)
    store = _RecordingStore()
    out = trust.trusted_search(
        store, object(), "what did we decide about caching?", k=5, ledger=DecisionLedger(store)
    )
    assert out is canned
    assert [e[0] for e in store.events] == [SEARCH_DECISION_EVENT]
    assert store.events[0][1]["outcome"] == "answered"
    assert store.events[0][2]["generation_id"] == "gen_1"


def test_wrapper_witnesses_a_refusal_and_reraises_it(monkeypatch):
    import recall.trust as trust

    refusal = TrustRefusal(
        code=TrustFailureCode.CALIBRATION_MISSING,
        calibration_status="missing",
        tenant_id="default",
        generation_id="gen_1",
    )

    def _refuse(*a, **k):
        raise refusal

    monkeypatch.setattr(trust, "_trusted_search", _refuse)
    store = _RecordingStore()
    with pytest.raises(TrustRefusal):
        trust.trusted_search(
            store, object(), "who set the rate limit?", k=2, ledger=DecisionLedger(store)
        )
    assert [e[0] for e in store.events] == [SEARCH_REFUSAL_EVENT]
    assert store.events[0][1]["failure_code"] == "CALIBRATION_MISSING"
    assert store.events[0][1]["query"] == "who set the rate limit?"


def test_no_ledger_means_no_record(monkeypatch):
    import recall.trust as trust

    canned = _result([_hit("ok", "a.md", 0.7)])
    monkeypatch.setattr(trust, "_trusted_search", lambda *a, **k: canned)
    # The default path must not construct, consult, or require anything ledger-shaped.
    assert trust.trusted_search(object(), object(), "q", k=1) is canned


# --------------------------------------------------------------------------------------------
# Against the real database: the row lands, is tenant-scoped, and holds no corpus bytes.
# --------------------------------------------------------------------------------------------

V1 = "The API rate limit is one hundred requests per second per client key.\n"
V2 = (
    "---\nsupersedes: rate_v1.md\n---\n"
    "Rate limiting update: twenty requests per second per client key.\n"
)


def _audit_rows(store, event_type: str, query_marker: str) -> list[tuple[str, str, dict]]:
    """(event_id, actor, payload) rows for this test's own records, via the store's tenant."""
    import psycopg

    from tests.conftest import TEST_DSN

    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute(
            "SELECT set_config('recall.tenant_id', %s, false)", (store.tenant,)
        )
        rows = conn.execute(
            "SELECT event_id, actor, payload FROM recall_audit_events "
            "WHERE tenant_id = %s AND event_type = %s AND payload->>'query' LIKE %s",
            (store.tenant, event_type, f"%{query_marker}%"),
        ).fetchall()
    return rows


def _delete_events(store, event_ids: list[str]) -> None:
    import psycopg

    from tests.conftest import TEST_DSN

    if not event_ids:
        return
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute(
            "SELECT set_config('recall.tenant_id', %s, false)", (store.tenant,)
        )
        conn.execute(
            "DELETE FROM recall_audit_events WHERE tenant_id = %s AND event_id = ANY(%s)",
            (store.tenant, event_ids),
        )


@requires_db
def test_one_search_appends_one_decision_row(tmp_path, make_store):
    import uuid

    from recall.embeddings import HashingEmbedder
    from recall.index import Indexer
    from tests.conftest import dev_search

    store = make_store(64)
    for name, text in {"rate_v1.md": V1, "rate_v2.md": V2}.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    Indexer(store, HashingEmbedder(dim=64)).index_path(tmp_path)

    marker = uuid.uuid4().hex
    event_ids: list[str] = []
    try:
        res = dev_search(
            store,
            HashingEmbedder(dim=64),
            f"API rate limit requests per second {marker}",
            k=5,
            ledger=DecisionLedger(store),
        )
        rows = _audit_rows(store, SEARCH_DECISION_EVENT, marker)
        event_ids = [r[0] for r in rows]
        assert len(rows) == 1, "one call, one record"
        _, actor, payload = rows[0]
        assert actor == "trust-layer"
        assert payload["outcome"] == ("abstained" if res.abstained else "answered")
        recorded = {h["file"]: h["verdict"] for h in payload["hits"]}
        assert recorded.get("rate_v1.md") == "superseded"  # the losing evidence is in the row
        assert "requests per second per client key" not in json.dumps(payload)
    finally:
        _delete_events(store, event_ids)


@requires_db
def test_strict_refusal_is_witnessed_and_still_raises(make_store):
    import uuid

    from recall.embeddings import HashingEmbedder
    from recall.trust import trusted_search

    store = make_store(64)
    marker = uuid.uuid4().hex
    event_ids: list[str] = []
    try:
        with pytest.raises(TrustRefusal):
            # Default policy is strict, and a throwaway table has no generation and no
            # calibration, so this refuses — and the refusal must leave a record behind.
            trusted_search(
                store,
                HashingEmbedder(dim=64),
                f"who set the rate limit? {marker}",
                k=3,
                ledger=DecisionLedger(store),
            )
        rows = _audit_rows(store, SEARCH_REFUSAL_EVENT, marker)
        event_ids = [r[0] for r in rows]
        assert len(rows) == 1
        payload = rows[0][2]
        assert payload["outcome"] == "refused"
        assert payload["failure_code"] in {"INDEX_NOT_READY", "CALIBRATION_MISSING"}
    finally:
        _delete_events(store, event_ids)


@requires_db
def test_a_nul_query_still_lands_a_refusal_row(make_store):
    """Red-to-green for SEC-001: before sanitisation this exact write failed on every attempt
    (Postgres jsonb rejects the escaped NUL), so the one caller most worth auditing was the one
    who could choose not to be."""
    store = make_store(64)
    refusal = TrustRefusal(
        code=TrustFailureCode.INDEX_NOT_READY,
        calibration_status="missing",
        tenant_id=store.tenant,
        generation_id=None,
    )
    event_id = DecisionLedger(store).record_refusal(refusal, query="who\x00did this?", k=3)
    try:
        assert event_id is not None, "the sanitized refusal record must land"
        rows = _audit_rows(store, SEARCH_REFUSAL_EVENT, "did this?")
        assert len(rows) == 1
        assert rows[0][2]["query_sanitized"] is True
    finally:
        _delete_events(store, [event_id] if event_id else [])


@requires_db
def test_mcp_service_honours_the_env_flag(tmp_path, make_store, monkeypatch):
    """RECALL_DECISION_LEDGER=1 makes the MCP service leave a record, with its own actor."""
    import uuid

    from recall.embeddings import HashingEmbedder
    from recall.index import Indexer
    from tests.conftest import dev_search_memory

    store = make_store(64)
    (tmp_path / "rate_v1.md").write_text(V1, encoding="utf-8")
    Indexer(store, HashingEmbedder(dim=64)).index_path(tmp_path)

    marker = uuid.uuid4().hex
    monkeypatch.setenv("RECALL_DECISION_LEDGER", "1")
    event_ids: list[str] = []
    try:
        dev_search_memory(store, HashingEmbedder(dim=64), f"API rate limit {marker}")
        rows = _audit_rows(store, SEARCH_DECISION_EVENT, marker)
        event_ids = [r[0] for r in rows]
        assert len(rows) == 1
        assert rows[0][1] == "mcp-service"
    finally:
        _delete_events(store, event_ids)


@requires_db
def test_append_audit_event_validates_and_returns_id(make_store):
    store = make_store(64)
    with pytest.raises(ValueError):
        store.append_audit_event("", {})
    event_id = store.append_audit_event(
        "test_event", {"probe": True}, actor="test", generation_id=None
    )
    try:
        assert event_id.startswith("evt_")
    finally:
        import psycopg

        from tests.conftest import TEST_DSN

        with psycopg.connect(TEST_DSN, autocommit=True) as conn:
            conn.execute(
                "SELECT set_config('recall.tenant_id', %s, false)", (store.tenant,)
            )
            conn.execute(
                "DELETE FROM recall_audit_events WHERE tenant_id = %s AND event_id = %s",
                (store.tenant, event_id),
            )
