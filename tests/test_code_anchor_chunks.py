"""The `code` chunker anchors every Python chunk to the code it holds.

Each chunk of a ``.py`` file carries ``metadata["code_anchor"] = {"repo", "file", "symbol",
"code_sha"}``, computed to the shared anchor vectors, so a `recall_record` anchor written by CCA
joins to the current code by equality (``docs/AUDIT_MEMORY_DESIGN.md``, "Code join"). Every other
file is chunked exactly as before and carries no anchor.

Red proof (2026-10-21, run on the deployment host inside the test bounds). Each mutation was made
to the production line named, the named test failed on the assertion quoted, and the line was then
restored and the test run green:

- `test_each_symbol_gets_its_own_chunk_and_one_anchor`: in `recall.index.code_pieces`, the owner
  assignment with ``span.end_line`` replaced by ``span.end_line - 1`` (the span end line); failed
  on ``assert got == EXPECTED``, the last line of every symbol having moved to its parent.
- `test_an_oversized_symbol_splits_and_every_piece_keeps_its_anchor`: in
  `recall.index._pack_lines`, the size flush ``if buf and size + 1 + len(line) > max_chars``
  deleted; failed on ``assert len(pieces) > 1``.
- `test_anchor_sha_equals_the_reference_symbol_sha`: in `code_pieces`, the anchor hashed over the
  symbol's own lines (``code_sha("\\n".join(line for _, line in owner_lines))``) instead of its whole
  span; failed on ``assert anchor["code_sha"] == expected[anchor["symbol"]]`` for ``Pool``.
- `test_a_non_python_file_is_chunked_as_before_with_no_anchor`: in `code_pieces`, the ``.py``
  suffix check removed; failed on ``assert pieces == [(piece, {}) ...]`` for ``notes.md``.
- `test_unparseable_python_is_anchored_to_the_whole_file`: in `code_pieces`, the ``except`` branch
  returning ``{}`` instead of the whole-file anchor; failed on ``assert metas == [...]``.
- `test_the_code_chunker_version_moves_the_pipeline_fingerprint`: in
  `recall.generation_build.chunker_for`, ``CODE_CHUNKER_VERSION`` replaced by ``1``; failed on
  ``assert anchored.fingerprint != previous.fingerprint``.
- `test_the_build_binds_the_anchored_chunker`: in `chunker_for`, ``CodeChunker(...)`` replaced by
  the previous ``functools.partial(chunk_code, ...)``; failed on ``assert callable(getattr(chunker,
  "pieces", None))``.
- `test_the_wizard_query_set_chunks_python_as_the_build_does`: in
  `recall.wizard.queryset.chunks_from_directory`, the ``pieces`` branch removed; failed on
  ``assert chunks == expected``.
- `test_a_generation_build_stores_the_anchor_on_every_python_chunk` (database):
  in `recall.generations.GenerationManager._chunks_for_source`, the ``pieces`` branch removed
  (every chunker called with the text alone); failed on ``assert anchors == expected``. And with
  ``unquote(...)`` removed from the ``file`` argument, failed on the same assertion for
  ``pkg/my mod.py``.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from io import BytesIO
from pathlib import Path

import psycopg
import pytest

from recall.anchors import code_sha, resolve_symbols, symbol_sha
from recall.generation_build import BuildRequest, chunker_for
from recall.index import (
    CODE_ANCHOR_KEY,
    CODE_CHUNKER_VERSION,
    DEFAULT_MAX_CHARS,
    CodeChunker,
    chunk_code,
    code_pieces,
)
from recall.lineage import (
    ChunkerIdentity,
    EmbedderIdentity,
    IndexManifestV1,
    ManifestObjectV1,
    PipelineIdentity,
)
from tests.conftest import TEST_DSN, _db_available, db_unreachable_reason, requires_db

VECTORS = json.loads(
    (Path(__file__).parent / "fixtures" / "audit_memory" / "anchor_vectors.json").read_text(
        encoding="utf-8"
    )
)

SOURCE = '''"""Pool module."""
import threading

LIMIT = 3


class Pool:
    """A pool."""

    size = LIMIT

    def lease(self):
        def check(conn):
            return conn.alive
        return check(self._conn)

    @property
    def free(self):
        return self.size

    @free.setter
    def free(self, value):
        self.size = value


def main():
    return Pool()


if __name__ == "__main__":
    main()
'''

#: Written by hand from the rule (each line belongs to the innermost symbol containing it), not
#: taken from the implementation.
EXPECTED = [
    (None, '"""Pool module."""\nimport threading\n\nLIMIT = 3\n\nif __name__ == "__main__":\n    main()'),
    ("Pool", 'class Pool:\n    """A pool."""\n\n    size = LIMIT'),
    ("Pool.lease", "    def lease(self):\n        return check(self._conn)"),
    ("Pool.lease.check", "        def check(conn):\n            return conn.alive"),
    ("Pool.free", "    @property\n    def free(self):\n        return self.size"),
    ("Pool.free#2", "    @free.setter\n    def free(self, value):\n        self.size = value"),
    ("main", "def main():\n    return Pool()"),
]


def test_each_symbol_gets_its_own_chunk_and_one_anchor() -> None:
    pieces = code_pieces(SOURCE, file="pkg/pool.py", repo="owner/name")

    got = [(meta[CODE_ANCHOR_KEY]["symbol"], text) for text, meta in pieces]
    assert got == EXPECTED
    assert {symbol for symbol, _ in got} == {s.symbol for s in resolve_symbols(SOURCE)} | {None}
    for _, meta in pieces:
        assert set(meta) == {CODE_ANCHOR_KEY}
        anchor = meta[CODE_ANCHOR_KEY]
        assert list(anchor) == ["repo", "file", "symbol", "code_sha"]
        assert (anchor["repo"], anchor["file"]) == ("owner/name", "pkg/pool.py")


def test_an_oversized_symbol_splits_and_every_piece_keeps_its_anchor() -> None:
    body = "".join(f"    value_{i} = compute({i})  # padding to grow the function\n" for i in range(40))
    source = f"def big():\n{body}    return value_0\n"
    max_chars = 300

    pieces = code_pieces(source, file="big.py", repo=None, max_chars=max_chars)

    assert len(pieces) > 1
    assert all(len(text) <= max_chars for text, _ in pieces)
    expected_anchor = {
        "repo": None,
        "file": "big.py",
        "symbol": "big",
        "code_sha": symbol_sha(source, "big"),
    }
    assert [meta[CODE_ANCHOR_KEY] for _, meta in pieces] == [expected_anchor] * len(pieces)
    assert "\n".join(text for text, _ in pieces) == source.rstrip("\n")


@pytest.mark.parametrize("case", VECTORS, ids=[c["name"] for c in VECTORS])
def test_anchor_sha_equals_the_reference_symbol_sha(case: dict) -> None:
    source = case["source"]
    expected = {e["symbol"]: e["sha"] for e in case["expect"]}
    expected[None] = case["file_sha"]

    anchors = [meta[CODE_ANCHOR_KEY] for _, meta in code_pieces(source, file="m.py", repo="r")]

    assert {a["symbol"] for a in anchors} == set(expected) - _symbols_without_lines(case)
    for anchor in anchors:
        assert anchor["code_sha"] == expected[anchor["symbol"]]
        assert anchor["code_sha"] == symbol_sha(source, anchor["symbol"])


def _symbols_without_lines(case: dict) -> set[str | None]:
    """`None` when every line of the file belongs to a symbol, so no module-level chunk exists."""
    covered: set[int] = set()
    for e in case["expect"]:
        covered.update(range(e["start_line"], e["end_line"] + 1))
    lines = case["source"].replace("\r\n", "\n").replace("\r", "\n").split("\n")
    module_text = [line for n, line in enumerate(lines, 1) if n not in covered and line.strip()]
    return set() if module_text else {None}


@pytest.mark.parametrize("file", ["notes.md", "lib.js", "setup.cfg"])
def test_a_non_python_file_is_chunked_as_before_with_no_anchor(file: str) -> None:
    # Parseable Python on purpose: only the file name may decide that it is not Python.
    pieces = code_pieces(SOURCE, file=file, repo="owner/name")

    assert pieces == [(piece, {}) for piece in chunk_code(SOURCE)]


def test_unparseable_python_is_anchored_to_the_whole_file() -> None:
    source = "def broken(:\n    pass\n\n\nclass Fine:\n    x = 1\n"

    pieces = code_pieces(source, file="pkg/broken.py", repo="owner/name")
    metas = [meta for _, meta in pieces]

    assert [text for text, _ in pieces] == chunk_code(source)
    whole_file = {
        "repo": "owner/name",
        "file": "pkg/broken.py",
        "symbol": None,
        "code_sha": code_sha(source),
    }
    assert metas == [{CODE_ANCHOR_KEY: whole_file}] * len(pieces)


def test_the_code_chunker_version_moves_the_pipeline_fingerprint() -> None:
    """Generation reuse is keyed on the pipeline fingerprint, so this is what makes code tenants
    re-chunk on their next build. The text chunker's identity must not move with it."""
    embedder = EmbedderIdentity("fixture", "model-a", 64, revision="model-a-commit")
    _, identity = chunker_for(BuildRequest(chunker="code"))
    previous = ChunkerIdentity("recall.chunk_code", 1, {"max_chars": DEFAULT_MAX_CHARS})

    anchored = PipelineIdentity(embedder, identity)
    assert anchored.fingerprint != PipelineIdentity(embedder, previous).fingerprint
    assert identity.schema_version == CODE_CHUNKER_VERSION

    _, text_identity = chunker_for(BuildRequest(chunker="text"))
    assert (text_identity.algorithm, text_identity.schema_version) == ("recall.chunk_text", 1)


def test_the_build_binds_the_anchored_chunker() -> None:
    chunker, _ = chunker_for(BuildRequest(chunker="code", max_chars=300))

    assert callable(getattr(chunker, "pieces", None))
    assert chunker.pieces(SOURCE, file="pkg/pool.py", repo="o/n") == code_pieces(  # type: ignore[attr-defined]
        SOURCE, file="pkg/pool.py", repo="o/n", max_chars=300
    )


def test_the_wizard_query_set_chunks_python_as_the_build_does(tmp_path: Path) -> None:
    """The query set must hold the texts the index will hold, or its questions describe chunks
    that do not exist (`recall.wizard.queryset.chunks_from_directory`)."""
    from recall.wizard.queryset import chunks_from_directory

    (tmp_path / "pool.py").write_text(SOURCE, encoding="utf-8")
    chunker = CodeChunker()

    chunks = chunks_from_directory(tmp_path, "**/*.py", chunker)

    expected = [text for text, _ in code_pieces(SOURCE, file="pool.py", repo=None)]
    assert chunks == expected
    assert chunks != chunk_code(SOURCE)


# --- through a real generation build -------------------------------------------------------------


class _S3:
    def __init__(self, objects: dict[tuple[str, str, str], bytes]) -> None:
        self.objects = objects

    def get_object(self, **kwargs):  # type: ignore[no-untyped-def]
        data = self.objects[(kwargs["Bucket"], kwargs["Key"], kwargs["VersionId"])]
        return {"Body": BytesIO(data), "ContentLength": len(data), "VersionId": kwargs["VersionId"]}


class _Embedder:
    dim = 64
    name = "model-a"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(index % 7) for index in range(self.dim)] for _ in texts]


@pytest.fixture
def manager():  # type: ignore[no-untyped-def]
    from recall.generations import GenerationManager

    if not _db_available():
        pytest.skip(db_unreachable_reason())
    tenant = "code-anchor-" + uuid.uuid4().hex[:10]
    value = GenerationManager(TEST_DSN, tenant, actor="pytest", environment="test")
    yield value
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (tenant,))
        conn.execute("DELETE FROM recall_source_tombstones WHERE tenant_id = %s", (tenant,))
        conn.execute("DELETE FROM recall_audit_events WHERE tenant_id = %s", (tenant,))
        conn.execute("DELETE FROM recall_ingest_jobs WHERE tenant_id = %s", (tenant,))
        conn.execute("DELETE FROM recall_tenant_state WHERE tenant_id = %s", (tenant,))
        conn.execute("DELETE FROM recall_generations WHERE tenant_id = %s", (tenant,))


@requires_db
def test_a_generation_build_stores_the_anchor_on_every_python_chunk(manager) -> None:  # type: ignore[no-untyped-def]
    from recall.manifest import S3Allowlist, S3ObjectReader

    tenant = manager.tenant_id
    files = {
        "pkg/pool.py": ("text/x-python", SOURCE.encode("utf-8")),
        "pkg/my mod.py": ("text/x-python", b"def spaced():\n    return 1\n"),
        "README.md": ("text/markdown", b"# Readme\n\ndef not_code():\n    pass\n"),
    }
    objects: dict[tuple[str, str, str], bytes] = {}
    entries = []
    for path, (media_type, data) in files.items():
        sha = hashlib.sha256(data).hexdigest()
        key = f"corpora/{tenant}/{path}"
        objects[("approved", key, sha)] = data
        uri = f"s3://approved/corpora/{tenant}/{path.replace(' ', '%20')}"
        entries.append(ManifestObjectV1(uri, sha, media_type, len(data), sha))
    manifest = IndexManifestV1(tenant, "corpus-v1", tuple(entries))
    reader = S3ObjectReader(_S3(objects), S3Allowlist.parse("approved/corpora/"))
    chunker, chunker_identity = chunker_for(BuildRequest(chunker="code"))
    pipeline = PipelineIdentity(
        EmbedderIdentity("fixture", "model-a", 64, revision="model-a-commit"),
        chunker_identity,
        fts_configuration={"language": "english", "schema_version": 1},
    )

    generation = manager.create(manifest, pipeline)
    manager.build(
        generation.generation_id, reader, _Embedder(), chunker, provenance={"project": "owner/name"}
    )

    with psycopg.connect(TEST_DSN) as conn:
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (tenant,))
        rows = conn.execute(
            "SELECT metadata FROM recall_chunks_v1 WHERE tenant_id = %s AND generation_id = %s",
            (tenant, generation.generation_id),
        ).fetchall()
    anchors = sorted(
        (
            (row[0]["file"], json.dumps(row[0].get(CODE_ANCHOR_KEY), sort_keys=True))
            for row in rows
        ),
    )
    expected = sorted(
        [
            ("pkg/pool.py", json.dumps(meta[CODE_ANCHOR_KEY], sort_keys=True))
            for _, meta in code_pieces(SOURCE, file="pkg/pool.py", repo="owner/name")
        ]
        + [
            (
                "pkg/my%20mod.py",
                json.dumps(
                    {
                        "repo": "owner/name",
                        "file": "pkg/my mod.py",
                        "symbol": "spaced",
                        "code_sha": symbol_sha("def spaced():\n    return 1\n", "spaced"),
                    },
                    sort_keys=True,
                ),
            ),
            ("README.md", json.dumps(None)),
        ]
    )
    assert anchors == expected


def _source_build(  # type: ignore[no-untyped-def]
    manager, tmp_path, source, project, *, extracting=True, encoding="utf-8", **build_options
):
    from recall.generation_build import build_generation
    from recall.manifest import LocalObjectReader, reader_for_manifest

    path = tmp_path / "source.py"
    data = source.encode(encoding)
    path.write_bytes(data)
    sha = hashlib.sha256(data).hexdigest()
    entry = ManifestObjectV1(path.as_uri(), sha, "text/x-python", len(data), sha)
    manifest = IndexManifestV1(manager.tenant_id, "source-v1", (entry,))
    reader = (
        reader_for_manifest(manifest, local_roots=[tmp_path])
        if extracting else LocalObjectReader([tmp_path])
    )
    stats = build_generation(
        manager, manifest, reader, _Embedder(),
        BuildRequest(chunker="code", project=project, revision="fixture", commit_root=None),
        **build_options,
    )
    with psycopg.connect(TEST_DSN) as conn:
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (manager.tenant_id,))
        rows = conn.execute(
            "SELECT text, metadata FROM recall_chunks_v1 "
            "WHERE tenant_id = %s AND generation_id = %s ORDER BY chunk_ordinal",
            (manager.tenant_id, stats.generation_id),
        ).fetchall()
    return stats, rows


@requires_db
@pytest.mark.parametrize("extracting", [True, False], ids=["cli-reader", "raw-reader"])
@pytest.mark.parametrize("source", [
    "X = 1\n", "\n\nX = 1\n\n", "\ufeffX = 1\n",
    "\ufeffdef first():\n    return 1\n", "\n\ndef first():\n    return 1\n",
])
def test_generation_anchors_hash_original_python_source(manager, tmp_path, source, extracting):  # type: ignore[no-untyped-def]
    """BUG-001: build readers must preserve the reference anchor hashing input.

    Red proof (2026-10-26, bounded Linux host): PR976 baseline 1695f8417 failed six cases at stored-anchor
    equality: extraction loses outer whitespace and both readers lose a BOM. All ten cases
    also failed that assertion when _chunks_for_source stamped an all-zero code_sha. The
    remaining baseline cases are preservation controls, not claims of an existing defect.
    """
    _, rows = _source_build(manager, tmp_path, source, "owner/name", extracting=extracting)
    expected = code_pieces(source, file="source.py", repo="owner/name")
    assert [(text, metadata[CODE_ANCHOR_KEY]) for text, metadata in rows] == [
        (text, metadata[CODE_ANCHOR_KEY]) for text, metadata in expected
    ]


@requires_db
@pytest.mark.parametrize("before,after", [
    (None, "owner/b"), ("owner/a", "owner/b"), ("owner/a", None),
    ("owner/a", "owner/a"), (None, None),
])
def test_reused_code_anchors_obey_current_repository(manager, tmp_path, before, after):  # type: ignore[no-untyped-def]
    """DAT-002: repository provenance is part of the code-anchor reuse contract.

    Red proof (2026-10-26, bounded Linux host): PR976 baseline 1695f8417, GenerationManager._reuse_source;
    all three changed-project cases failed at repo equality. All five cases also failed
    there when _chunks_for_source stamped repo='mutation'. The two same-project cases
    must retain reuse, so a fix cannot simply disable source reuse.
    """
    first, _ = _source_build(manager, tmp_path, "def value():\n    return 1\n", before)
    manager.validate(first.generation_id)
    second, rows = _source_build(manager, tmp_path, "def value():\n    return 1\n", after)
    assert rows
    assert {metadata[CODE_ANCHOR_KEY]["repo"] for _, metadata in rows} == {after}
    assert second.reused_objects == int(before == after)


@requires_db
@pytest.mark.parametrize("allowed", [True, False], ids=["redacted", "denied"])
def test_original_python_still_passes_source_security(manager, tmp_path, allowed):  # type: ignore[no-untyped-def]
    """BUG-001 control: raw source selection must preserve authorization and redaction.

    Mutation proof: set security_policy=None before GenerationManager.build's source loop.
    The redacted case fails its content assertion; the denied case fails DID NOT RAISE.
    """
    from recall.generations import GenerationError
    from recall.security_policy import AccessContext, SourceRule, SourceSecurityPolicy

    policy = SourceSecurityPolicy((SourceRule(
        "source.py", principals=frozenset({"allowed"}),
        purposes=frozenset({"indexing"}), redactions=("email",),
    ),))
    options = {
        "security_policy": policy,
        "security_context": AccessContext(
            "allowed" if allowed else "denied", manager.tenant_id, purpose="indexing"
        ),
    }
    source = '\ufeffEMAIL = "person@example.com"\n'
    if not allowed:
        with pytest.raises(GenerationError, match="denied by source security policy"):
            _source_build(manager, tmp_path, source, "owner/name", **options)
        return
    _, rows = _source_build(manager, tmp_path, source, "owner/name", **options)
    assert rows
    assert all("person@example.com" not in text for text, _ in rows)
    assert any("[REDACTED:email]" in text for text, _ in rows)


@requires_db
def test_original_python_keeps_extraction_bounds(manager, tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    """BUG-001 control: preserving bytes must retain the extracting reader's refusal.

    Mutation proof: replace extract(entry, original) with original for anchored Python;
    this test fails DID NOT RAISE instead of silently discarding the extraction cap.
    """
    from recall.extraction import DocumentExtractionError

    monkeypatch.setattr("recall.extraction.MAX_EXTRACTED_CHARACTERS", 10)
    with pytest.raises(DocumentExtractionError, match="extracted more than"):
        _source_build(manager, tmp_path, "VALUE = 'long enough'\n", "owner/name")


@requires_db
def test_original_python_keeps_extracting_reader_encoding(manager, tmp_path):  # type: ignore[no-untyped-def]
    """BUG-001 regression: preserving source must retain the reader's supported encodings.

    Red proof: the first repair replaces the extracted UTF-8 view with cp1252 bytes;
    _decoded_secure_text raises GenerationError, failing the explicit consumer assertion.
    The corrected build must hash the original decoded text including its final newline.
    """
    from recall.generations import GenerationError

    source = '# coding: cp1252\nNAME = "caf\u00e9"\n'
    try:
        _, rows = _source_build(manager, tmp_path, source, "owner/name", encoding="cp1252")
    except GenerationError as exc:
        raise AssertionError("the extracting reader must retain supported source decoding") from exc
    assert [(text, meta[CODE_ANCHOR_KEY]) for text, meta in rows] == [
        (text, meta[CODE_ANCHOR_KEY])
        for text, meta in code_pieces(source, file="source.py", repo="owner/name")
    ]
