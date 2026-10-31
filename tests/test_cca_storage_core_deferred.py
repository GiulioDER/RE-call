"""Regression tests for the deferred P3 pass over the storage core CCA hunt (run 20261031T220410Z).

Each test names its finding (`DEF-<n>`, with the hunt's original id in brackets) and records its
red proof: the pre-fix behaviour at 8f67f5c70, which this branch starts from, and the assertion
that fails against it. Documentation-only findings carry no test here; see the PR for those.
"""

from __future__ import annotations

import io
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import psycopg
import pytest

from recall.errors import RecallError
from recall.security_policy import AccessContext, SourceSecurityPolicy, load_source_policy


def _load(tmp_path: Path, payload: object) -> SourceSecurityPolicy | None:
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return load_source_policy({"RECALL_SOURCE_POLICY_FILE": str(path)})


_OPEN_RULE = {"prefix": "notes", "principals": [], "purposes": ["retrieval"], "redactions": []}


# --- DEF-20 (ENV-003): a misspelled redaction name is refused when the policy loads ----------


def test_policy_file_refuses_an_unknown_redaction_name(tmp_path: Path) -> None:
    """DEF-20: `load_source_policy` refuses a redaction name `_REDACTIONS` does not define.

    Red proof: at 8f67f5c70 the policy loaded with `redactions=('emial',)` and the typo surfaced
    only at index time, in `redact_with_decision`; the `pytest.raises` fails with DID NOT RAISE.
    Production line: the `_REDACTIONS` membership check in `load_source_policy`.
    """
    rule = {**_OPEN_RULE, "redactions": ["emial"]}
    with pytest.raises(ValueError, match="unknown redactions: emial"):
        _load(tmp_path, {"default_deny": True, "rules": [rule]})


def test_policy_file_with_a_known_redaction_still_loads(tmp_path: Path) -> None:
    """DEF-20 control: a correctly spelled redaction loads and is kept in order."""
    policy = _load(tmp_path, {"default_deny": True, "rules": [{**_OPEN_RULE, "redactions": ["email"]}]})
    assert policy is not None
    decision = policy.decide("notes/a.md", AccessContext("alice", "t"))
    assert decision.allowed and decision.redactions == ("email",)


# --- DEF-22 (STAKES-002): an omitted list is named as missing, never defaulted --------------


@pytest.mark.parametrize("key", ["principals", "purposes", "redactions"])
def test_policy_file_names_a_missing_rule_list(tmp_path: Path, key: str) -> None:
    """DEF-22: an omitted `principals`, `purposes` or `redactions` is refused as missing.

    The old tuple defaults could never apply (they failed the list check), and the refusal said
    the key "must be an array of strings" about a key that was not there. It must stay a refusal:
    an omitted `principals` read as empty would admit every principal.

    Red proof: at 8f67f5c70 the message was "... {key} must be an array of strings", so the
    `match` fails. Production line: the `missing` check in `load_source_policy`.
    """
    rule = {name: value for name, value in _OPEN_RULE.items() if name != key}
    with pytest.raises(ValueError, match=f"missing required keys: {key}"):
        _load(tmp_path, {"default_deny": True, "rules": [rule]})


# --- DEF-14 (DOC-013): the recall.eval exceptions share the package's base -----------------


@pytest.mark.parametrize(
    ("module", "name", "builtin"),
    [
        ("recall.eval.arm_check", "InertArmError", RuntimeError),
        ("recall.eval.arm_check", "EmptySampleError", RuntimeError),
        ("recall.eval.hostload", "HostTooBusyError", RuntimeError),
        ("recall.eval.longmemeval", "ConversionError", ValueError),
        ("recall.eval.promotion.adapters", "CorpusUnavailable", RuntimeError),
        ("recall.eval.promotion.aggregate", "UnpairedArms", ValueError),
        ("recall.eval.promotion.aggregate", "VacuousArm", ValueError),
        ("recall.eval.resume", "LedgerCorrupted", ValueError),
    ],
)
def test_eval_exceptions_are_recall_errors(module: str, name: str, builtin: type) -> None:
    """DEF-14: `recall.errors` promises RecallError is the base of every deliberate exception.

    `tests/test_error_hierarchy.py` skips the whole `recall.eval` prefix, so these eight were
    never checked. Each keeps its built-in as the first base, as `recall.errors` prescribes.

    Red proof: at 8f67f5c70 each class derived only from its built-in, so `issubclass(cls,
    RecallError)` is False and the assertion fails for all eight.
    """
    import importlib

    cls = getattr(importlib.import_module(module), name)
    assert issubclass(cls, RecallError)
    assert cls.__mro__[1] is builtin


# --- DEF-21 (ENV-006): padded RECALL_LOG_FORMAT / RECALL_LOG_LEVEL are honoured -------------


def test_padded_log_format_and_level_are_honoured(monkeypatch: pytest.MonkeyPatch) -> None:
    """DEF-21: `configure_logging` strips the two variables like every other RECALL_* read.

    Red proof: at 8f67f5c70 `RECALL_LOG_FORMAT="json "` was not in `_LOG_FORMATS`, so the
    handler got the text formatter and the JSON assertion fails. Production lines: the two
    `.strip()` reads in `recall.observability.configure_logging`.
    """
    from recall.observability import _JsonFormatter, configure_logging, get_logger

    logger = get_logger()
    saved = (list(logger.handlers), logger.level, logger.propagate)
    monkeypatch.setenv("RECALL_LOG_FORMAT", "json ")
    monkeypatch.setenv("RECALL_LOG_LEVEL", " debug ")
    try:
        configure_logging(stream=io.StringIO())
        assert isinstance(logger.handlers[0].formatter, _JsonFormatter)
        assert logger.level == logging.DEBUG
    finally:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
        for handler in saved[0]:
            logger.addHandler(handler)
        logger.setLevel(saved[1])
        logger.propagate = saved[2]


# --- DEF-24 (NUM-002): vector parameters count toward the byte counters ---------------------


def test_payload_bytes_counts_vector_parameters() -> None:
    """DEF-24: `_payload_bytes` counted a pgvector `Vector`, `SparseVector` and datetime as 0.

    Every dense and sparse query passes its query vector as one of these, so the diagnostic
    `db_parameter_bytes` left out the largest parameter. Sizes are pgvector's binary forms.

    Red proof: at 8f67f5c70 `_payload_bytes({"vec": Vector([0.1] * 1024)})` was 3 (the key
    only), so the first assertion fails. Production lines: the shape checks at the end of
    `recall.observability._payload_bytes`.
    """
    from pgvector import SparseVector, Vector

    from recall.observability import _payload_bytes

    assert _payload_bytes({"vec": Vector([0.1] * 1024)}) == 3 + 4 * 1024
    assert _payload_bytes(SparseVector({1: 0.5, 7: 0.25}, 30522)) == 16
    assert _payload_bytes(datetime(2026, 10, 31, tzinfo=timezone.utc)) == 8
    # Controls: the cases that already counted are unchanged.
    assert _payload_bytes(["ab", 1, None, True]) == 2 + 8


# --- DEF-25 (NUM-003): a hashing dimension it cannot build is refused ----------------------


@pytest.mark.parametrize("name", ["hashing-128", "hashing:384"])
def test_hashing_embedder_refuses_a_dimension_other_than_64(name: str) -> None:
    """DEF-25: `resolve_embedder` built a 64 dimension embedder for any `hashing-<n>`.

    Red proof: at 8f67f5c70 `resolve_embedder("hashing-128").dim == 64` and nothing raised, so
    the `pytest.raises` fails with DID NOT RAISE. Production line: the suffix check in the
    hashing branch of `recall.embeddings.resolve_embedder`.
    """
    from recall.embeddings import resolve_embedder

    with pytest.raises(ValueError, match="always 64"):
        resolve_embedder(name, env={})


@pytest.mark.parametrize("name", ["hashing", "hashing-64", "hashing:64"])
def test_hashing_embedder_spellings_that_mean_64_still_resolve(name: str) -> None:
    """DEF-25 control: the spellings that already meant 64 dimensions are unchanged."""
    from recall.embeddings import resolve_embedder

    assert resolve_embedder(name, env={}).dim == 64


# --- DEF-27 (NUM-005): the learned sparse leg refuses a k it cannot return ------------------


def test_sparse_ef_search_refuses_a_k_beyond_the_hnsw_ceiling() -> None:
    """DEF-27: `sparse_ef_search` clamped a k above 1000 silently; the dense leg refuses it.

    No legal `hnsw.ef_search` can then return k rows, so the caller would never learn that the
    leg truncated. Reachable when the dense leg is off (a SPLADE-only configuration).

    Red proof: at 8f67f5c70 `sparse_ef_search(1001)` returned 1000, so the `pytest.raises` fails
    with DID NOT RAISE. Production line: the `k > _HNSW_EF_SEARCH_MAX` check in
    `recall.store.sparse_ef_search`.
    """
    from recall.store import _HNSW_EF_SEARCH_MAX, sparse_ef_search

    with pytest.raises(ValueError, match="silently truncate"):
        sparse_ef_search(_HNSW_EF_SEARCH_MAX + 1)
    # Control: the largest legal k still gets the capped value.
    assert sparse_ef_search(_HNSW_EF_SEARCH_MAX) == _HNSW_EF_SEARCH_MAX


# --- DEF-30 (DAT-005): the lexical leg breaks rank ties deterministically -------------------


def _lexical_sql(vec: list[float] | None) -> str:
    from recall.store import PgVectorStore

    store = PgVectorStore.__new__(PgVectorStore)
    store._table = "chunks"
    store._tenant = "t"
    captured: list[str] = []

    class _Conn:
        def execute(self, sql: str, params: object = None) -> "_Conn":
            captured.append(sql)
            return self

        def fetchall(self) -> list[object]:
            return []

    store._with_retry = lambda op: op(_Conn())  # type: ignore[method-assign]
    assert store._query_sparse("alpha beta", 5, vec=vec) == []
    return " ".join(captured[0].split())


@pytest.mark.parametrize("vec", [None, [0.1, 0.2, 0.3]], ids=["lexical", "lexical-with-vector"])
def test_lexical_leg_orders_ties_by_chunk_id(vec: list[float] | None) -> None:
    """DEF-30: both lexical SQL shapes ordered by rank alone before LIMIT.

    ts_rank plus the constant numeric boost ties readily, so which tied rows survived LIMIT k,
    and their order into rank-based fusion, depended on the plan. Asserted on the statement the
    store sends, because the tie only shows on a database whose plan happens to differ.

    Red proof: at 8f67f5c70 every ORDER BY in `PgVectorStore._query_sparse` ended at
    `rank DESC` / `score DESC`, so the regex finds no `, c.id` / `, id` tie-breaker and the
    assertion fails for both shapes.
    """
    sql = _lexical_sql(vec)
    orders = re.findall(r"ORDER BY (\w+) DESC(, (?:c\.)?id)?", sql)
    assert orders, sql
    assert all(tie for _, tie in orders), sql


# --- DEF-32 (DEPLOY-002): an unmigrated database says so, with the command that fixes it ----


def test_missing_vector_type_raises_schema_too_old(monkeypatch: pytest.MonkeyPatch) -> None:
    """DEF-32: `_prepare_connection` let pgvector's ProgrammingError escape.

    On a database where migration 0001 (the only CREATE EXTENSION) has not run, store
    construction failed before `check_schema` could raise SchemaTooOld with its guidance.

    Red proof: at 8f67f5c70 the ProgrammingError escaped unchanged, so `pytest.raises(
    SchemaTooOld)` fails with that error. Production line: the `except psycopg.ProgrammingError`
    around `register_vector` in `recall.store._prepare_connection`.
    """
    import recall.store as store_module
    from recall.schema import SchemaTooOld

    def _absent(conn: object) -> None:
        raise psycopg.ProgrammingError("vector type not found in the database")

    monkeypatch.setattr(store_module, "register_vector", _absent)
    with pytest.raises(SchemaTooOld, match="recall schema apply"):
        store_module._prepare_connection(object(), "t", None)  # type: ignore[arg-type]


def test_other_programming_errors_from_register_vector_are_not_relabelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DEF-32 control: only pgvector's missing-type error becomes SchemaTooOld."""
    import recall.store as store_module

    def _other(conn: object) -> None:
        raise psycopg.ProgrammingError("something else")

    monkeypatch.setattr(store_module, "register_vector", _other)
    with pytest.raises(psycopg.ProgrammingError, match="something else"):
        store_module._prepare_connection(object(), "t", None)  # type: ignore[arg-type]
