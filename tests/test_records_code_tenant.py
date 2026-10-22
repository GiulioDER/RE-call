"""`recall_records` over a code tenant built by the anchored code chunker (#976), and the escaped
line separators of the canonical record line.

Invariants and the failure each one catches:
- T1 chunks made by `CodeChunker.pieces` (a split symbol, module-level lines, two files) come back
  as ONE record per distinct `(repo, file, symbol, code_sha)`, `kind == "code"`, with `chunk_id`
  the lowest-ordinal chunk and `chunk_ids` every chunk carrying that anchor in ordinal order.
  Catches one record per chunk, and a `chunk_ids` list that drops the split pieces.
- T2 a record's anchors match a code tenant as `exact` (same symbol and hash), `symbol_changed`
  (same symbol, other hash) and `file_only` (same file, other symbol).
- T3 a code anchor written with no `--project` (`repo` null) never matches a `repo` query, and is
  still returned by an `ids` or `all_records` query that names no repository.
- T4 the same, built through the real generation build into PostgreSQL (DB suite).
- T5 a record whose text holds U+2028, U+2029 and U+0085, written in the canonical form that
  escapes them (`docs/AUDIT_MEMORY_PLAN.md`, Global Constraints), has no raw line separator in
  its memo and round-trips through a lite store unchanged.

Red proof, recorded 2026-10-22 on the deployment host inside the test bounds, each failing in its
own assertion, then restored and green:
- T1 `tests/test_records_code_tenant.py::test_one_record_per_anchor_with_every_chunk`: code
  records keyed per chunk id instead of per anchor (`key = ("code", chunk.id)` in
  `recall.records.lookup_records`): "one record per distinct anchor".
- T1b same test: `chunk_ids` reduced to the first chunk (`[members[0][0].id]`): "chunk_ids misses
  a chunk carrying the anchor".
- T3 `...::test_a_null_repo_never_matches_a_repo_query`: the repository filter mutated to let a
  null repo through (`_repo_of(record) not in (repo, None)`): "a null-repo anchor matched".
- T4 `...::test_a_generation_built_code_tenant_returns_one_record_per_anchor`: the per-anchor key
  mutated as in T1: the record count assertion fails.
- T5 `...::test_escaped_line_separators_round_trip`: the parser's `recall_record` branch mutated to
  double the backslash of every unicode escape before decoding, the classic escape bug: the
  summary reads back as six literal characters per separator and the equality assertion fails.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from recall.index import CODE_ANCHOR_KEY, CodeChunker
from recall.lite import LiteStore
from recall.records import lookup_records
from recall.types import Chunk
from tests.audit_memory_support import EXPECTED, REPO, indexed_lite_store, lite_dsn
from tests.conftest import TEST_DSN, requires_db
from tests.test_code_anchor_chunks import (  # noqa: F401  # `manager` is a fixture, used by name
    _S3,
    _Embedder,
    manager,
)

DIM = 8
MAX_CHARS = 200

#: `lease` is longer than MAX_CHARS, so it splits into several chunks sharing one anchor.
POOL = (
    '"""Pool module."""\n'
    "import threading\n\n"
    "LIMIT = 3\n\n\n"
    "class Pool:\n"
    '    """A pool."""\n\n'
    "    size = LIMIT\n\n"
    "    def lease(self):\n"
    + "".join(f"        step_{i} = self.check_connection_liveness_inside_the_lease({i})\n" for i in range(8))
    + "        return step_0\n\n"
    "    def release(self, conn):\n"
    "        return conn\n\n\n"
    "def helper():\n"
    "    return LIMIT\n"
)
OTHER = "def standalone():\n    return 1\n"


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RECALL_EMBED_CACHE", "off")


def _pieces(text: str, file: str, repo: str | None) -> list[tuple[str, dict[str, Any]]]:
    return CodeChunker(max_chars=MAX_CHARS).pieces(text, file=file, repo=repo)


def _code_store(tmp_path: Path) -> tuple[LiteStore, dict[str, list[str]]]:
    """A lite store holding the chunks the anchored chunker makes, keyed as a build keys them.

    Returns the store and, per anchor (canonical JSON), the chunk ids carrying it in ordinal order.
    """
    store = LiteStore.from_dsn(lite_dsn(tmp_path / "code.db"), dim=DIM)
    chunks: list[Chunk] = []
    by_anchor: dict[str, list[str]] = {}
    for file, text, repo in (("pkg/pool.py", POOL, REPO), ("pkg/other.py", OTHER, None)):
        for ordinal, (piece, extra) in enumerate(_pieces(text, file, repo)):
            chunk_id = hashlib.md5(f"{file}:{ordinal}".encode(), usedforsecurity=False).hexdigest()
            chunks.append(Chunk(chunk_id, file, piece, {"file": file, "ord": ordinal, **extra}))
            key = json.dumps(extra[CODE_ANCHOR_KEY], sort_keys=True)
            by_anchor.setdefault(key, []).append(chunk_id)
    store.upsert(chunks, [[1.0] + [0.0] * (DIM - 1) for _ in chunks])
    return store, by_anchor


def _anchor_of(record: dict[str, Any]) -> str:
    return json.dumps(record["recall_record"], sort_keys=True)


def test_one_record_per_anchor_with_every_chunk(tmp_path: Path) -> None:
    """T1."""
    store, by_anchor = _code_store(tmp_path)
    split = [ids for ids in by_anchor.values() if len(ids) > 1]
    assert split, "the fixture must split a symbol, or this test proves nothing about chunk_ids"
    records = lookup_records(store, all_records=True)["records"]
    assert len(records) == len(by_anchor), "one record per distinct anchor"
    for record in records:
        assert record["kind"] == "code"
        assert record["chunk_ids"] == by_anchor[_anchor_of(record)], (
            "chunk_ids misses a chunk carrying the anchor"
        )
        assert record["chunk_id"] == record["chunk_ids"][0] == record["id"]


def test_record_anchors_match_a_code_tenant(tmp_path: Path) -> None:
    """T2, with the three anchors a record would carry for `Pool.lease`."""
    store, _by_anchor = _code_store(tmp_path)
    lease = next(
        extra[CODE_ANCHOR_KEY]
        for _piece, extra in _pieces(POOL, "pkg/pool.py", REPO)
        if extra[CODE_ANCHOR_KEY]["symbol"] == "Pool.lease"
    )
    asked = {"file": lease["file"], "symbol": lease["symbol"], "code_sha": lease["code_sha"]}
    cases = {
        "exact": asked,
        "symbol_changed": {**asked, "code_sha": "0" * 64},
        "file_only": {**asked, "symbol": "Pool.vanished"},
    }
    for want, anchor in cases.items():
        records = lookup_records(store, repo=REPO, anchors=[anchor])["records"]
        lease_records = [r for r in records if r["recall_record"]["symbol"] == "Pool.lease"]
        assert [r["anchor_match"] for r in lease_records] == [want], (want, records)
        others = {r["anchor_match"] for r in records if r["recall_record"]["symbol"] != "Pool.lease"}
        assert others <= {"file_only"}, others


def test_a_null_repo_never_matches_a_repo_query(tmp_path: Path) -> None:
    """T3."""
    store, _by_anchor = _code_store(tmp_path)
    standalone = {"file": "pkg/other.py", "symbol": "standalone", "code_sha": None}
    for repo in (REPO, "someone/else"):
        found = lookup_records(store, repo=repo, anchors=[standalone])["records"]
        assert found == [], f"a null-repo anchor matched repo {repo!r}"
    everything = lookup_records(store, all_records=True)["records"]
    assert {r["recall_record"]["repo"] for r in everything} == {REPO, None}
    null_ids = [r["id"] for r in everything if r["recall_record"]["repo"] is None]
    assert [r["id"] for r in lookup_records(store, ids=null_ids)["records"]] == sorted(null_ids)


@requires_db
def test_a_generation_built_code_tenant_returns_one_record_per_anchor(manager) -> None:  # type: ignore[no-untyped-def]  # noqa: F811
    """T4: the production build path, `chunker_for("code")` bound to the anchored chunker."""
    from recall.generation_build import BuildRequest, chunker_for
    from recall.generation_store import GenerationStore
    from recall.lineage import EmbedderIdentity, IndexManifestV1, ManifestObjectV1, PipelineIdentity
    from recall.manifest import S3Allowlist, S3ObjectReader

    tenant = manager.tenant_id
    data = POOL.encode("utf-8")
    sha = hashlib.sha256(data).hexdigest()
    readme = b"# Readme\n\nThe pool package.\n"
    readme_sha = hashlib.sha256(readme).hexdigest()
    objects = {
        ("approved", f"corpora/{tenant}/pkg/pool.py", sha): data,
        ("approved", f"corpora/{tenant}/README.md", readme_sha): readme,
    }
    manifest = IndexManifestV1(
        tenant,
        "corpus-v1",
        (
            ManifestObjectV1(f"s3://approved/corpora/{tenant}/pkg/pool.py", sha, "text/x-python", len(data), sha),
            ManifestObjectV1(
                f"s3://approved/corpora/{tenant}/README.md", readme_sha, "text/markdown", len(readme), readme_sha
            ),
        ),
    )
    reader = S3ObjectReader(_S3(objects), S3Allowlist.parse("approved/corpora/"))
    chunker, chunker_identity = chunker_for(BuildRequest(chunker="code", max_chars=MAX_CHARS))
    pipeline = PipelineIdentity(
        EmbedderIdentity("fixture", "model-a", 64, revision="model-a-commit"),
        chunker_identity,
        fts_configuration={"language": "english", "schema_version": 1},
    )
    generation = manager.create(manifest, pipeline)
    manager.build(generation.generation_id, reader, _Embedder(), chunker, provenance={"project": REPO})
    manager.validate(generation.generation_id)
    manager.promote(generation.generation_id, unsafe_development=True)

    expected: dict[str, int] = {}
    for _piece, extra in _pieces(POOL, "pkg/pool.py", REPO):
        key = json.dumps(extra[CODE_ANCHOR_KEY], sort_keys=True)
        expected[key] = expected.get(key, 0) + 1
    with GenerationStore(TEST_DSN, 64, tenant=tenant) as store:
        records = lookup_records(store, repo=REPO, all_records=True)["records"]
    assert len(records) == len(expected), "one record per distinct anchor"
    assert {_anchor_of(r): len(r["chunk_ids"]) for r in records} == expected
    assert all(r["kind"] == "code" and r["chunk_id"] == r["chunk_ids"][0] for r in records)


#: The three characters `str.splitlines()` breaks on that JSON does not escape by default.
SEPARATORS = "\u2028\u2029\u0085"


def _json_line(value: object, *, escaped: bool) -> str:
    """`value` as one line of JSON; `escaped` writes the three separators as `\\uXXXX` escapes,
    as cca-audit #87 writes them, and otherwise leaves them raw, as `json.dumps` does."""
    line = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if escaped:
        for char in SEPARATORS:
            line = line.replace(char, f"\\u{ord(char):04x}")
    return line


@pytest.mark.parametrize("escaped", [True, False], ids=["escaped", "raw"])
def test_escaped_line_separators_round_trip(tmp_path: Path, escaped: bool) -> None:
    """T5: the summary and the description both hold all three separators, in either form."""
    name = "RE-call__0123456789ab__BUG-001.md"
    text = f"one{SEPARATORS[0]}two{SEPARATORS[1]}three{SEPARATORS[2]}four"
    record = {**EXPECTED[name], "summary": text}
    description = f"RE-call BUG-001 (proved): {text}"
    golden = (Path(__file__).parent / "fixtures" / "audit_memory" / "memos" / name).read_text(encoding="utf-8")
    lines = golden.split("\n")
    for index, line in enumerate(lines):
        if line.startswith("recall_record: "):
            lines[index] = "recall_record: " + _json_line(record, escaped=escaped)
        elif line.startswith("description: "):
            lines[index] = "description: " + _json_line(description, escaped=escaped)
    memo = "\n".join(lines)
    if escaped:
        assert not any(char in memo for char in SEPARATORS), "a raw separator reached the memo"
    folder = tmp_path / "memos"
    folder.mkdir()
    (folder / name).write_text(memo, encoding="utf-8")
    store = indexed_lite_store(tmp_path, memos=folder)
    found = lookup_records(store, ids=[record["id"]])["records"]
    assert [r["recall_record"] for r in found] == [record]
    assert [r["description"] for r in found] == [description]
