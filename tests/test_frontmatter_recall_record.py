"""`recall_record` is parsed, admitted and stored exactly as authored (Task 1.1, audit memory plan).

Invariants and the failure each one catches:
- R1 every golden memo (contract C4) yields, on EVERY chunk the indexer writes, a
  `metadata["recall_record"]` equal to its `expected_records.json` entry. Catches a parser that
  renames, adds, defaults or drops a field.
- R2 an invalid record is dropped with a warning naming the file and the reason, and the memo is
  still indexed; a truncated JSON line is one such case. Catches a record stored unvalidated, and
  a record that vanishes in silence.
- R3 a record of a version this code does not read is refused with a warning saying so, never
  partly read.
- R4 a chunk set stored under the previous `_METADATA_RULE_VERSION` is not reused, so memos that
  carried a record before the parser existed gain it on the next generation build.
- R5 the legacy index path re-derives a memo that declares a record once, then skips it again.
- R6 the schema interpreter in `recall.records` agrees with `jsonschema` on every golden record
  and on mutations of them.
- R7 `name` and `description` are kept only beside a record.
- R8 a record line nested too deep for `json.loads` (RecursionError, not JSONDecodeError) is
  warned and dropped; it does not stop the index run.
- R9 a record holding U+0000 (written by `json.dumps` as the escape `\\u0000`) is refused with a
  reason on both stores: PostgreSQL jsonb cannot store it, so admitting it failed the whole
  generation build there while a lite store accepted it (R2's `nul` case is the lite half).

Red proof, recorded 2026-10-21 on the deployment host inside the test bounds, each test failing
in its own assertion, then restored and green:
- R1 `tests/test_frontmatter_recall_record.py::test_golden_memos_parse_to_expected_records`:
  `parse_frontmatter`'s `recall_record` branch mutated to drop `ordinal`: the chunk's record is
  absent (admission refuses the now invalid record), so the equality assertion fails.
- R2 `...::test_invalid_record_warns_and_indexes_without_it`: `admit_record` mutated to return
  `meta` unchanged: the stored chunk keeps the invalid record and the warning assertion fails.
- R3 `...::test_unknown_version_refused_with_warning`: the version check in `record_problem`
  removed: the warning names the schema error `v: must be 1` instead of the version, so the
  message assertion fails.
- R4 `...::test_metadata_rule_version_bumped`: `_METADATA_RULE_VERSION` reverted to
  `relative-file-paths-2026-08-29`: the old chunk set is reused (`reused_objects == 1`).
- R5 `...::test_legacy_index_rederives_a_record_memo_once`: the `_declares_record` branch of
  `recall.index._body_derivation_hash` removed: the memo is skipped as unchanged and keeps no
  record.
- R6 `...::test_validator_agrees_with_jsonschema`: `additionalProperties: false` ignored by
  `recall.records._errors`: the unknown-field mutation validates here and not in `jsonschema`.
- R7 `...::test_labels_are_kept_only_beside_a_record`: the `if RECORD_KEY in meta` guard removed:
  a memo with no record gains `name`.
- R8 (2026-10-22) `...::test_a_deeply_nested_record_warns_and_does_not_stop_the_run`: the parser
  catching only `json.JSONDecodeError`: "a deeply nested recall_record stopped the index run".
- R2[nul] and R9 (2026-10-22) `...::test_invalid_record_warns_and_indexes_without_it[nul]` and
  `...::test_a_nul_record_does_not_fail_a_postgres_generation_build`: the `_holds_nul` check in
  `record_problem` disabled: lite stores the record ("an invalid record was stored"), and the
  PostgreSQL build raises ("a NUL in a recall_record failed the generation build").
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

import pytest

import recall.index as index_module
from recall.document import parse_document
from recall.embeddings import resolve_embedder
from recall.frontmatter import RECORD_KEY
from recall.index import Indexer
from recall.lite import LiteStore
from recall.records import record_problem, record_schema
from tests.audit_memory_support import EXPECTED, MEMOS, copy_memos, indexed_lite_store, lite_dsn
from tests.conftest import requires_db
from tests.test_generations import (  # noqa: F401  # `manager` is a fixture, used by name
    _Embedder,
    _manifest,
    _pipeline,
    _reader,
    _ready,
    manager,
)

PREVIOUS_RULE = "relative-file-paths-2026-08-29"


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RECALL_EMBED_CACHE", "off")


def _chunks_by_memo(store: LiteStore) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for chunk in store.iter_chunks():
        out.setdefault(Path(chunk.source).name, []).append(chunk.metadata)
    return out


def test_golden_memos_parse_to_expected_records(tmp_path: Path) -> None:
    """R1, through the real indexer into a lite store, every chunk of every memo."""
    store = indexed_lite_store(tmp_path)
    chunks = _chunks_by_memo(store)
    assert set(chunks) == set(EXPECTED), sorted(chunks)
    for name, expected in EXPECTED.items():
        for metadata in chunks[name]:
            assert metadata.get(RECORD_KEY) == expected, f"{name}: the stored record differs"


def _memo_with_record(record_line: str) -> str:
    return (
        "---\nname: broken-memo\ndescription: \"a memo\"\nvalid_from: 2026-10-21\n"
        f"recall_record: {record_line}\n---\n\n# A memo\n\nIts body is still evidence.\n"
    )


def _index_one(tmp_path: Path, text: str) -> tuple[LiteStore, Path]:
    folder = tmp_path / "memos"
    folder.mkdir()
    memo = folder / "broken.md"
    memo.write_text(text, encoding="utf-8")
    embedder = resolve_embedder("hashing")
    store = LiteStore.from_dsn(lite_dsn(tmp_path / "memory.db"), dim=embedder.dim)
    Indexer(store, embedder, allow_prune=True, env={}).index_path(folder)
    return store, memo


@pytest.mark.parametrize(
    ("record_line", "reason"),
    [
        pytest.param(
            json.dumps({k: v for k, v in EXPECTED["RE-call__0123456789ab__BUG-001.md"].items() if k != "ordinal"}),
            "missing required field 'ordinal'",
            id="missing-field",
        ),
        pytest.param(
            json.dumps(EXPECTED["RE-call__0123456789ab__BUG-001.md"])[:120],
            "recall_record must be one line of JSON",
            id="truncated-json",
        ),
        pytest.param(
            # `json.dumps` writes U+0000 as the escape `\u0000`, which `json.loads` decodes.
            json.dumps({**EXPECTED["RE-call__0123456789ab__BUG-001.md"], "summary": "before\x00after"}),
            "U+0000",
            id="nul",
        ),
    ],
)
def test_invalid_record_warns_and_indexes_without_it(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, record_line: str, reason: str
) -> None:
    """R2."""
    with caplog.at_level(logging.WARNING, logger="recall.records"):
        store, memo = _index_one(tmp_path, _memo_with_record(record_line))
    chunks = list(store.iter_chunks())
    assert chunks, "the memo itself must still be indexed"
    assert all(RECORD_KEY not in c.metadata for c in chunks), "an invalid record was stored"
    assert all("name" not in c.metadata for c in chunks), "labels outlived the refused record"
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any(str(memo) in w and reason in w for w in warnings), warnings


def test_unknown_version_refused_with_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """R3."""
    record = {**EXPECTED["RE-call__0123456789ab__BUG-001.md"], "v": 2}
    with caplog.at_level(logging.WARNING, logger="recall.records"):
        store, memo = _index_one(tmp_path, _memo_with_record(json.dumps(record)))
    assert all(RECORD_KEY not in c.metadata for c in store.iter_chunks())
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any(str(memo) in w and "version 2 is not one this RE-call reads" in w for w in warnings), warnings


def test_a_deeply_nested_record_warns_and_does_not_stop_the_run(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """R8: `json.loads` raises RecursionError, not JSONDecodeError, on a line nested this deep."""
    depth = 100_000
    with caplog.at_level(logging.WARNING, logger="recall.records"):
        try:
            store, memo = _index_one(tmp_path, _memo_with_record("[" * depth + "]" * depth))
        except RecursionError as exc:
            pytest.fail(f"a deeply nested recall_record stopped the index run: {exc!r}")
    assert all(RECORD_KEY not in c.metadata for c in store.iter_chunks())
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any(str(memo) in w and "one line of JSON" in w for w in warnings), warnings


@requires_db
def test_a_nul_record_does_not_fail_a_postgres_generation_build(
    manager, caplog: pytest.LogCaptureFixture  # noqa: F811  # the imported fixture
) -> None:
    """R9: jsonb refuses U+0000, so a record holding one must be refused before PostgreSQL sees it."""
    record = {**EXPECTED["RE-call__0123456789ab__BUG-001.md"], "summary": "before\x00after"}
    data = _memo_with_record(json.dumps(record)).encode("utf-8")
    manifest = _manifest(manager.tenant_id, data)
    reader = _reader(manifest, data)
    with caplog.at_level(logging.WARNING, logger="recall.records"):
        try:
            generation = _ready(manager, manifest, _pipeline("model-a"), reader, _Embedder(1))
        except Exception as exc:  # noqa: BLE001  # any failure of the build is the defect
            pytest.fail(f"a NUL in a recall_record failed the generation build: {exc!r}")
    with manager._connect() as conn:
        rows = conn.execute(
            "SELECT metadata ? 'recall_record' FROM recall_chunks_v1 "
            "WHERE tenant_id = %s AND generation_id = %s",
            (manager.tenant_id, generation),
        ).fetchall()
    assert rows and not any(row[0] for row in rows), "the memo must be indexed without its record"
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("U+0000" in w for w in warnings), warnings


@requires_db
def test_metadata_rule_version_bumped(manager) -> None:  # noqa: F811  # the imported fixture
    """R4, on the generation build path, which is the one production serves."""
    name = "RE-call__0123456789ab__BUG-001.md"
    data = (MEMOS / name).read_bytes()
    manifest = _manifest(manager.tenant_id, data)
    reader = _reader(manifest, data)
    pipeline = _pipeline("model-a")
    first = _ready(manager, manifest, pipeline, reader, _Embedder(1))
    manager.promote(first, unsafe_development=True)
    # What the parser before this change stored for the same bytes: no record, no labels, and
    # the previous rule version.
    with manager._connect() as conn, conn.transaction():
        conn.execute(
            "UPDATE recall_chunks_v1 SET metadata = "
            "(metadata - 'recall_record' - 'name' - 'description') "
            "|| jsonb_build_object('metadata_rule_version', %s::text) "
            "WHERE tenant_id = %s AND generation_id = %s",
            (PREVIOUS_RULE, manager.tenant_id, first),
        )

    must_run = _Embedder(9)
    second = manager.create(manifest, pipeline)
    stats = manager.build(second.generation_id, reader, must_run, lambda text: [text])

    assert stats.reused_objects == 0, "a chunk set from the previous metadata rule was reused"
    with manager._connect() as conn:
        rows = conn.execute(
            "SELECT metadata -> 'recall_record' FROM recall_chunks_v1 "
            "WHERE tenant_id = %s AND generation_id = %s",
            (manager.tenant_id, second.generation_id),
        ).fetchall()
    assert rows and all(row[0] == EXPECTED[name] for row in rows), rows


def test_legacy_index_rederives_a_record_memo_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """R5: a lite store indexed by a parser that did not know the key gains the record."""
    folder = copy_memos(tmp_path / "memos")
    embedder = resolve_embedder("hashing")
    store = LiteStore.from_dsn(lite_dsn(tmp_path / "memory.db"), dim=embedder.dim)
    indexer = Indexer(store, embedder, allow_prune=True, env={})
    with monkeypatch.context() as old:
        # The code before this change: no trigger, and nothing admitted.
        old.setattr(index_module, "_declares_record", lambda raw: False)
        old.setattr(index_module, "admit_record", lambda meta, source: {
            k: v for k, v in meta.items() if k not in (RECORD_KEY, "name", "description")
        })
        indexer.index_path(folder)
    assert not any(RECORD_KEY in c.metadata for c in store.iter_chunks())

    again = indexer.index_path(folder)
    assert again.files == len(EXPECTED), "memos declaring a record were skipped as unchanged"
    assert all(RECORD_KEY in c.metadata for c in store.iter_chunks())
    third = indexer.index_path(folder)
    assert third.files == 0 and third.skipped == len(EXPECTED), "re-derived more than once"


def _mutations() -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []
    for name, record in sorted(EXPECTED.items()):
        out.append((f"{name}:as-is", record))
        extra = copy.deepcopy(record)
        extra["status"] = "open"
        out.append((f"{name}:unknown-field", extra))
        missing = copy.deepcopy(record)
        missing.pop("date")
        out.append((f"{name}:missing-date", missing))
        bad_date = {**copy.deepcopy(record), "date": "2026-10-21\n"}
        out.append((f"{name}:date-with-newline", bad_date))
        if record.get("anchors"):
            windows = copy.deepcopy(record)
            windows["anchors"][0]["file"] = windows["anchors"][0]["file"].replace("/", "\\")
            out.append((f"{name}:windows-path", windows))
            nulled = copy.deepcopy(record)
            nulled["anchors"][0]["code_sha"] = None
            out.append((f"{name}:null-sha", nulled))
        if record["kind"] == "bug":
            out.append((f"{name}:bool-ordinal", {**copy.deepcopy(record), "ordinal": True}))
            out.append((f"{name}:dup-caused-by", {**copy.deepcopy(record), "caused_by": ["a", "a"]}))
        if record["kind"] == "lesson":
            out.append((f"{name}:two-members", {**copy.deepcopy(record), "members": record["members"][:2]}))
        out.append((f"{name}:other-kind", {**copy.deepcopy(record), "kind": "rumour"}))
    return out


@pytest.mark.parametrize(("label", "record"), _mutations(), ids=[label for label, _ in _mutations()])
def test_validator_agrees_with_jsonschema(label: str, record: dict[str, Any]) -> None:
    """R6. `jsonschema` arrives with the `mcp` extra the test environment installs."""
    jsonschema = pytest.importorskip("jsonschema")
    reference = jsonschema.Draft202012Validator(record_schema())
    # `jsonschema` uses Python's `re.search`, whose `$` also matches before a final newline, so
    # for that one case ECMA-262's reading (the schema's dialect) is the reference.
    expected_valid = not list(reference.iter_errors(record)) and not label.endswith("date-with-newline")
    assert (record_problem(record) is None) == expected_valid, (label, record_problem(record))


def test_labels_are_kept_only_beside_a_record() -> None:
    """R7: learning `name` and `description` changes no other document's metadata."""
    plain = parse_document("---\nname: plain\ndescription: \"x\"\ntype: project\n---\n\nBody.\n").meta
    assert plain == {"type": "project"}, plain
    golden = parse_document((MEMOS / "RE-call__111111111111__SEC-003.md").read_text(encoding="utf-8")).meta
    assert golden["name"] == "re-call-111111111111-sec-003"
    assert golden["description"] == (
        "RE-call SEC-003 (proved): Ünïcødé “quotes” and 'single' and \"double\": a key: value pair "
        "--- not a fence; 日本語 and emoji ✅ survive."
    )
