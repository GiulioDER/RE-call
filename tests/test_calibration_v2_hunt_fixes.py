"""Regression tests for the 2026-10-31 hunt audit of `recall/calibration_v2.py`.

None of these needs a database. Each replaces the connection with a fake that either answers
the one query under test or fails the test if it is reached, so the property pinned is where a
refusal happens, not what PostgreSQL would have done with the input.

Red proof (the project rule): every test below was run against the pre-fix module (the CCA
snapshot of `recall/calibration_v2.py` at master 7b1ef2aa8) or, where the pre-fix module lacks
the symbol, against a mutation of the new line, and failed in its own assertion. The baseline
or mutation, the target and the failure are in each docstring.
"""

from __future__ import annotations

import json
from contextlib import contextmanager, nullcontext
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import psycopg
import pytest

import recall.calibration_v2 as calibration_v2
from recall.calibration_v2 import (
    CalibrationArtifactV2,
    CalibrationBindingError,
    CalibrationRepository,
    canonical_query_set,
    canonical_sha256,
)

DIGEST = "a" * 64


def _repository() -> CalibrationRepository:
    return CalibrationRepository("postgresql://unused.invalid/none", "hunt-tenant", actor="pytest")


def _forbid_database(repository: CalibrationRepository, why: str) -> None:
    """Fail the test if the repository opens a connection: the refusal belonged before it."""

    @contextmanager
    def _connect():
        raise AssertionError(why)
        yield  # pragma: no cover

    repository._connect = _connect  # type: ignore[method-assign]


class _Result:
    def __init__(self, one: Any = None, many: list[Any] | None = None) -> None:
        self._one, self._many = one, many or []

    def fetchone(self) -> Any:
        return self._one

    def fetchall(self) -> list[Any]:
        return self._many


class _RecordingConnection:
    """Records every statement; `answer(sql)` decides what each returns."""

    def __init__(self, answer) -> None:
        self.statements: list[tuple[str, Any]] = []
        self._answer = answer

    def execute(self, sql: str, params: Any = None) -> _Result:
        self.statements.append((sql, params))
        return self._answer(sql)

    def transaction(self):
        return nullcontext()


def _use_connection(repository: CalibrationRepository, conn: _RecordingConnection) -> None:
    @contextmanager
    def _connect():
        yield conn

    repository._connect = _connect  # type: ignore[method-assign]


def _artifact(**overrides: Any) -> CalibrationArtifactV2:
    values: dict[str, Any] = {
        "calibration_id": "cal_hunt",
        "tenant_id": "hunt-tenant",
        "generation_id": "gen_hunt",
        "embedder_identity": {"model": "m"},
        "pipeline_fingerprint": DIGEST,
        "corpus_fingerprint": "b" * 64,
        "query_set_digest": "c" * 64,
        "threshold": 0.5,
        "scale": 0.05,
        "separability": 1.0,
        "separability_ci": (0.9, 1.0),
        "n_answerable": 20,
        "n_unanswerable": 20,
        "certified": True,
        "certification_reason": "certified",
        "lifecycle_state": "draft",
        "created_at": datetime.now(UTC).isoformat(),
        "created_by": "pytest",
        "scores": {"answerable": [], "unanswerable": []},
        "checksum": "0" * 64,
    }
    values.update(overrides)
    return CalibrationArtifactV2(**values)


def _sealed(path: Path, artifact: dict[str, Any], query_set: list[Any]) -> Path:
    payload: dict[str, Any] = {"artifact": artifact, "query_set": query_set}
    payload["bundle_checksum"] = canonical_sha256(dict(payload))
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _bundle_artifact(**overrides: Any) -> dict[str, Any]:
    """An artifact dict that reaches the field checks: version, tenant and digest all match."""
    queries = [{"query": "q", "answerable": True}]
    _labels, digest = canonical_query_set(queries)
    values: dict[str, Any] = {
        "artifact_version": 2,
        "tenant_id": "hunt-tenant",
        "generation_id": "gen_hunt",
        "query_set_digest": digest,
        "threshold": 0.5,
        "scale": 0.05,
        "separability": 1.0,
        "separability_ci": [0.9, 1.0],
        "n_answerable": 20,
        "n_unanswerable": 20,
        "certified": True,
        "embedder_identity": {"model": "m"},
        "scores": {"answerable": [], "unanswerable": []},
    }
    values.update(overrides)
    return values


QUERIES = [{"query": "q", "answerable": True}]


# BUG-101 ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "queries",
    [
        [
            {"query": "same", "answerable": True, "relevant_ids": ["a"]},
            {"query": "same", "answerable": True, "relevant_ids": ["b"]},
        ],
        [
            {"query": "same", "answerable": True},
            {"query": "same", "answerable": False},
        ],
    ],
    ids=["different-relevant-ids", "contradictory-labels"],
)
def test_a_query_repeated_with_other_metadata_is_refused_before_scoring(queries) -> None:
    """BUG-101: the scorer reads only `query` and `answerable`, so a repeat is one cosine twice.

    Red: pre-fix `calibrate` (snapshot) accepted both sets and opened a connection to score
    them, failing this test's `AssertionError("scored a repeated query")` inside `pytest.raises`.
    Target: `calibrate`'s call to `require_distinct_scored_queries`.
    """
    repository = _repository()
    _forbid_database(repository, "scored a repeated query")
    with pytest.raises(CalibrationBindingError, match="repeats the query 'same'"):
        repository.calibrate("gen_hunt", queries, embedder=SimpleNamespace())  # type: ignore[arg-type]


def test_distinct_queries_still_reach_scoring() -> None:
    """Control for BUG-101: distinct queries are not refused, they reach the database."""
    repository = _repository()
    _forbid_database(repository, "reached the database")
    with pytest.raises(AssertionError, match="reached the database"):
        repository.calibrate(
            "gen_hunt",
            [{"query": "one", "answerable": True}, {"query": "two", "answerable": False}],
            embedder=SimpleNamespace(),  # type: ignore[arg-type]
        )


# NUM-101 ------------------------------------------------------------------------------------


def test_a_chain_summing_to_exactly_the_bound_is_not_refused() -> None:
    """NUM-101: 25 steps of 0.01 sum to 0.25000000000000006 in binary floating point.

    Red: mutation of `_exceeds_cumulative_bound` to the pre-fix `cumulative > max_corpus_delta`
    raised "cumulative corpus delta 0.250 ... exceeds the bound 0.250". (The pre-fix module has
    no `_carried_cumulative_delta`, so the mutation is the proof.)
    """
    from recall.calibration_v2 import _carried_cumulative_delta

    chain = 0.0
    for _ in range(24):
        chain += 0.01
    parent = SimpleNamespace(carry_forward={"corpus_delta": 0.01, "cumulative_corpus_delta": chain})
    step = {"corpus_delta": 0.01}
    assert _carried_cumulative_delta(parent, step, 0.25) == pytest.approx(0.25)  # type: ignore[arg-type]


def test_a_chain_past_the_bound_is_still_refused() -> None:
    """Control for NUM-101: the tolerance absorbs ulps, not a real step past the bound."""
    from recall.calibration_v2 import _carried_cumulative_delta

    parent = SimpleNamespace(carry_forward={"corpus_delta": 0.01, "cumulative_corpus_delta": 0.245})
    with pytest.raises(CalibrationBindingError, match="cumulative corpus delta"):
        _carried_cumulative_delta(parent, {"corpus_delta": 0.01}, 0.25)  # type: ignore[arg-type]


# BUG-002 (with DAT-003 and the codex run's BUG-002) -----------------------------------------


def test_a_missing_bundle_file_is_a_named_refusal(tmp_path: Path) -> None:
    """BUG-002: red on the snapshot, which raised a raw FileNotFoundError from `read_bytes`."""
    with pytest.raises(CalibrationBindingError, match="cannot read calibration bundle"):
        _repository().import_bundle(tmp_path / "absent.json")


def test_a_bundle_that_is_not_utf8_is_a_named_refusal(tmp_path: Path) -> None:
    """BUG-002: red on the snapshot, which raised UnicodeDecodeError (not a JSONDecodeError)."""
    path = tmp_path / "latin1.json"
    path.write_bytes(b'{"artifact": "\xff"}')
    with pytest.raises(CalibrationBindingError, match="invalid calibration JSON"):
        _repository().import_bundle(path)


@pytest.mark.parametrize("version", [None, "v2", [2], 2.0, True])
def test_a_malformed_artifact_version_is_a_named_refusal(tmp_path: Path, version) -> None:
    """BUG-002: red on the snapshot for None, "v2" and [2] (raw TypeError or ValueError from
    `int()`), and for 2.0, which `int()` accepted: the bundle then went on to the database.
    True is a control: `int(True)` is 1, so the snapshot refused it as well."""
    repository = _repository()
    _forbid_database(repository, "accepted a malformed artifact_version")
    path = _sealed(tmp_path / "b.json", _bundle_artifact(artifact_version=version), QUERIES)
    with pytest.raises(CalibrationBindingError, match="unsupported calibration artifact version"):
        repository.import_bundle(path)


def test_a_query_set_entry_that_is_not_an_object_is_a_named_refusal(tmp_path: Path) -> None:
    """BUG-002: red on the snapshot, where `canonical_query_set` raised a raw AttributeError
    from `None.get`."""
    path = _sealed(tmp_path / "b.json", _bundle_artifact(), [None])
    with pytest.raises(CalibrationBindingError, match="query_set must hold objects"):
        _repository().import_bundle(path)


# BUG-001 and BUG-103 ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("scores", [0.5, 0.4], "scores must be an object"),
        ("embedder_identity", [1], "embedder_identity must be an object"),
        ("carry_forward", ["p"], "carry_forward must be an object"),
        ("separability", 1, "separability must be a JSON number with a fraction part"),
        ("threshold", 0, "threshold must be a JSON number with a fraction part"),
        ("separability_ci", [1, 1], "separability_ci must be a pair of bounds"),
        ("n_answerable", 20.0, "n_answerable must be a JSON integer"),
        ("certified", 1, "certified must be true or false"),
    ],
)
def test_a_field_that_would_not_survive_storage_is_refused_before_the_database(
    tmp_path: Path, field: str, value: Any, message: str
) -> None:
    """BUG-001 (shapes) and BUG-103 (numbers spelled as the column will not return them).

    Red: the snapshot accepted every one of these up to the generation lookup, failing this
    test's `AssertionError("reached the database ...")`. Downstream, the shapes crashed
    `verify_checksum` with a bare TypeError, and the numbers stored a row whose checksum fails on
    every read once DOUBLE PRECISION or INTEGER hands back the other spelling (1 -> 1.0).
    Target: `_require_bundle_field_types`.
    """
    repository = _repository()
    _forbid_database(repository, f"reached the database with a malformed {field}")
    path = _sealed(tmp_path / "b.json", _bundle_artifact(**{field: value}), QUERIES)
    with pytest.raises(CalibrationBindingError, match=message):
        repository.import_bundle(path)


def test_a_canonically_typed_bundle_still_reaches_the_database(tmp_path: Path) -> None:
    """Control for BUG-001 and BUG-103: an export's own types pass the field check."""
    repository = _repository()
    _forbid_database(repository, "reached the database")
    path = _sealed(tmp_path / "b.json", _bundle_artifact(), QUERIES)
    with pytest.raises(AssertionError, match="reached the database"):
        repository.import_bundle(path)


# DEPLOY-001 ---------------------------------------------------------------------------------


def test_a_missing_column_is_reported_as_schema_too_old(monkeypatch) -> None:
    """DEPLOY-001: an unmigrated database must name the remedy, not a column.

    Red: the snapshot let psycopg's UndefinedColumn escape `list_records`, failing
    `pytest.raises(SchemaTooOld)`. Target: the `except` in `CalibrationRepository._connect`.
    """
    from recall.schema import SchemaTooOld

    class _Old:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def execute(self, sql: str, params: Any = None) -> _Result:
            if "set_config" in sql:
                return _Result()
            raise psycopg.errors.UndefinedColumn('column "carry_forward" does not exist')

    monkeypatch.setattr(calibration_v2.psycopg, "connect", lambda *a, **k: _Old())
    with pytest.raises(SchemaTooOld, match="recall schema apply"):
        _repository().list_records()


# DAT-001 ------------------------------------------------------------------------------------


def test_publish_locks_the_generation_row_it_checks() -> None:
    """DAT-001: a plain read does not wait for `forget()`'s FOR UPDATE, so a concurrent erasure
    could commit between publish's lineage check and its write, leaving a stale row published.

    This pins the mechanism (the read takes a row lock), not the race itself, which needs two
    live connections. Red: the snapshot's generation read had no FOR SHARE, failing the assert.
    Target: `_generation(..., lock=True)` in `publish`.
    """
    artifact = _artifact()
    repository = _repository()

    def answer(sql: str) -> _Result:
        if "FROM recall_generations" in sql:
            return _Result(one=("ready", {}, artifact.pipeline_fingerprint, artifact.corpus_fingerprint))
        return _Result()

    conn = _RecordingConnection(answer)
    _use_connection(repository, conn)
    repository.get = lambda _id: artifact  # type: ignore[method-assign]

    repository.publish(artifact.calibration_id)

    reads = [sql for sql, _params in conn.statements if "FROM recall_generations" in sql]
    assert reads and all(sql.rstrip().endswith("FOR SHARE") for sql in reads), reads


# CODE-002 -----------------------------------------------------------------------------------


def test_export_refuses_a_stored_query_set_that_is_not_a_list(tmp_path: Path) -> None:
    """CODE-002: the other three readers of this table refused a non-list; export did not.

    Red: the snapshot wrote the bundle (no exception), failing `pytest.raises`.
    Target: `export_bundle`'s use of `_query_set_rows`.
    """
    artifact = _artifact()
    repository = _repository()
    _use_connection(repository, _RecordingConnection(lambda _sql: _Result(one=({"q": 1},))))
    repository.get = lambda _id: artifact  # type: ignore[method-assign]
    with pytest.raises(CalibrationBindingError, match="query set is missing"):
        repository.export_bundle(artifact.calibration_id, tmp_path / "out.json")


# PERF-002 -----------------------------------------------------------------------------------


def test_list_records_reads_only_the_named_generation() -> None:
    """PERF-002: a caller that wants one generation must not read the tenant's whole history.

    Red: mutation of `list_records` back to `WHERE tenant_id = %s` with params `(tenant,)`
    (the pre-fix query) failed the parameter assertion. (The snapshot's `list_records` takes no
    argument, so the mutation is the proof.)
    """
    repository = _repository()
    conn = _RecordingConnection(lambda _sql: _Result(many=[]))
    _use_connection(repository, conn)

    assert repository.list_records("gen_hunt") == []

    sql, params = conn.statements[-1]
    assert "generation_id = %s" in sql
    assert params == ("hunt-tenant", "gen_hunt", "gen_hunt")
