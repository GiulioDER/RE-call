"""The encode -> artifact -> load -> query round trip, without a model download.

Splitting encoding from loading is what makes this testable at all: the artifact is a plain file,
so the half that needs a GPU and the half that needs a database can be exercised separately.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from encode_sparse import passage_id, passage_text, prepare_output  # noqa: E402
from load_sparse import read_artifact  # noqa: E402
from recall.types import Chunk  # noqa: E402
from tests.conftest import requires_db  # noqa: E402

HEADER = {
    "_header": True,
    "model_name": "prithivida/Splade_PP_en_v1",
    "profile_id": "prithivida__Splade_PP_en_v1",
    "artifact_digest": "sha256:pinned",
    "top_k": 1000,
    "dimension": 30522,
    "max_length": 512,
    "fingerprint": "f" * 64,
}


def _write_artifact(path: Path, header: dict | None, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        if header is not None:
            handle.write(json.dumps(header) + "\n")
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_an_empty_artifact_is_refused_rather_than_loaded_as_a_no_op(tmp_path: Path) -> None:
    """An encode that produced nothing must not load as a silent success.

    This is the branch a header-less file reaches only when it contains no vectors either — a
    file with vectors but no header trips the ordering check below first. Loading it quietly
    would report "done, 0 rows" and leave the retriever refusing later, far from the cause.
    """
    artifact = tmp_path / "vectors.jsonl"
    artifact.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="no header"):
        read_artifact(artifact)


def test_a_vector_before_the_header_is_refused(tmp_path: Path) -> None:
    """Order matters, and the check is not merely "a header exists somewhere in the file".

    Without this, a concatenation of two artifacts passes: the second file's header is present,
    so a whole-file existence check is satisfied while the first file's vectors are attributed to
    the second file's model.
    """
    artifact = tmp_path / "vectors.jsonl"
    with artifact.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"id": "a", "weights": {"7": 1.0}, "nnz": 1}) + "\n")
        handle.write(json.dumps(HEADER) + "\n")

    with pytest.raises(ValueError, match="before any header"):
        read_artifact(artifact)


def test_term_ids_survive_the_json_round_trip_as_integers(tmp_path: Path) -> None:
    """JSON object keys are STRINGS; pgvector needs integer term ids.

    An unconverted "7" would reach SparseVector as a string key. That is a silent corruption
    risk at exactly the boundary where the vocabulary index lives, so it is pinned.
    """
    artifact = tmp_path / "vectors.jsonl"
    _write_artifact(artifact, HEADER, [{"id": "a", "weights": {"7": 2.5, "30521": 1.0}, "nnz": 2}])

    _header, rows = read_artifact(artifact)

    assert rows == [("a", {7: 2.5, 30521: 1.0})]


@requires_db
def test_a_loaded_artifact_is_retrievable(make_store, tmp_path: Path) -> None:
    """The whole point, executed: an artifact produced offline becomes a searchable index."""
    artifact = tmp_path / "vectors.jsonl"
    _write_artifact(
        artifact, HEADER,
        [
            {"id": "alpha", "weights": {"7": 3.0}, "nnz": 1},
            {"id": "beta", "weights": {"99": 3.0}, "nnz": 1},
        ],
    )
    store = make_store(64)
    store.upsert(
        [
            Chunk(id="alpha", source="/c/alpha.md", text="alpha", metadata={}),
            Chunk(id="beta", source="/c/beta.md", text="beta", metadata={}),
        ],
        [[0.1] * 64, [0.1] * 64],
    )
    header, rows = read_artifact(artifact)

    store.upsert_sparse(header["profile_id"], dict(rows))
    hits = store.query_learned_sparse({7: 1.0}, k=5, profile_id=header["profile_id"])

    assert [hit.chunk.id for hit in hits] == ["alpha"]
    assert store.sparse_row_count(header["profile_id"]) == 2


def test_passage_id_matches_the_indexer_precedence() -> None:
    """`_id` wins over `id`, because that is what the chunk rows are keyed on.

    benchmarks/mtrag/run.py writes chunks with `str(item.get("_id") or item.get("id"))`. In the
    MTRAG corpora the two fields happen to be equal, so keying on the wrong one works by luck --
    and if a corpus ever disagreed, the sparse rows would key on one id while the chunks key on
    the other, the JOIN would match nothing, and the leg would return an empty result that looks
    exactly like a query with no matches.
    """
    assert passage_id({"_id": "canonical", "id": "other"}) == "canonical"
    assert passage_id({"id": "fallback"}) == "fallback"


def test_passage_text_is_normalised_the_same_way_the_indexer_normalises_it() -> None:
    """Both legs must describe the SAME text, or they are indexing two different corpora.

    The indexer strips NUL bytes before storing. An encoder that does not strip them is encoding
    a string the dense leg never saw.
    """
    assert passage_text({"text": "a" + chr(0) + "b"}) == "ab"


def test_output_directory_is_created_rather_than_assumed(tmp_path: Path) -> None:
    """The encoder owns its output path, including the directory.

    It failed on the GPU box for exactly this: the caller had only created `vectors/` on one of
    its two code paths, so the measurement run died on FileNotFoundError AFTER loading the model
    and reading 1000 passages. Cheap to prevent, and the failure lands at the worst moment -- on
    rented hardware, with the meter running.
    """
    target = tmp_path / "deep" / "nested" / "vectors.jsonl"

    prepare_output(target)

    assert target.parent.is_dir()
    assert not target.exists()  # the FILE is the caller's to write, only the directory is made


@pytest.fixture
def stub_sparse_encoder(monkeypatch):
    from types import SimpleNamespace

    from recall.sparse import SpladeEncoder

    profile = SimpleNamespace(
        **{key: HEADER[key] for key in (
            "model_name", "profile_id", "artifact_digest", "top_k", "dimension",
        )},
        fingerprint=lambda: HEADER["fingerprint"],
    )
    encoder = SimpleNamespace(
        profile=profile, encode=lambda texts: [{7: 1.0} for _ in texts],
    )
    monkeypatch.setattr(SpladeEncoder, "from_pretrained", lambda *args, **kwargs: encoder)
    return encoder


@pytest.mark.parametrize("tail", ['{"id": "b"', '{"id": "b", "weights": {"7": 1.0}}'])
def test_resume_repairs_final_record_boundary(tmp_path, stub_sparse_encoder, tail):
    """BUG-001: baseline leaves partial JSON or concatenates a complete newline-less row."""
    from encode_sparse import main

    artifact = tmp_path / "vectors.jsonl"
    _write_artifact(artifact, HEADER, [{"id": "a", "weights": {"7": 1.0}}])
    with artifact.open("a", encoding="utf-8") as handle:
        handle.write(tail)
    passages = tmp_path / "passages.jsonl"
    passages.write_text("".join(json.dumps({"id": key, "text": key}) + "\n"
                                for key in ("a", "b", "c")), encoding="utf-8")

    assert main(["--input", str(passages), "--output", str(artifact), "--resume"]) == 0
    try:
        _, rows = read_artifact(artifact)
    except ValueError as exc:
        pytest.fail(f"resumed artifact must be loadable: {exc}")
    assert [key for key, _ in rows] == ["a", "b", "c"]


@pytest.mark.parametrize("field,value", [
    ("fingerprint", "other"), ("model_name", "other"), ("profile_id", "other"),
    ("artifact_digest", "other"), ("top_k", 2), ("dimension", 4),
])
def test_resume_refuses_changed_identity_without_writing(
    tmp_path, stub_sparse_encoder, field, value,
):
    """BUG-002: every case appends under the original identity on baseline a6051cac5."""
    from encode_sparse import main

    artifact = tmp_path / "vectors.jsonl"
    _write_artifact(artifact, HEADER | {field: value}, [])
    original = artifact.read_bytes()
    passages = tmp_path / "passages.jsonl"
    passages.write_text('{"id": "a", "text": "a"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="identity|header"):
        main(["--input", str(passages), "--output", str(artifact), "--resume"])
    assert artifact.read_bytes() == original


@pytest.mark.parametrize("stored_length", [None, 128], ids=["missing", "mismatch"])
def test_resume_refuses_changed_or_unknown_max_length_without_writing(
    tmp_path, stub_sparse_encoder, stored_length,
):
    """BUG-002 follow-up: the first repair ignored changed or unrecorded truncation."""
    from encode_sparse import main

    header = dict(HEADER)
    if stored_length is None:
        del header["max_length"]
    else:
        header["max_length"] = stored_length
    artifact = tmp_path / "vectors.jsonl"
    _write_artifact(artifact, header, [])
    original = artifact.read_bytes()
    passages = tmp_path / "passages.jsonl"
    passages.write_text('{"id": "a", "text": "a"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="identity|header"):
        main(["--input", str(passages), "--output", str(artifact), "--resume"])
    assert artifact.read_bytes() == original


def test_encode_records_max_length_in_artifact_header(tmp_path, stub_sparse_encoder):
    """BUG-002 follow-up: the emitted identity must state the actual truncation setting."""
    from encode_sparse import main

    artifact = tmp_path / "vectors.jsonl"
    passages = tmp_path / "passages.jsonl"
    passages.write_text('{"id": "a", "text": "a"}\n', encoding="utf-8")
    assert main([
        "--input", str(passages), "--output", str(artifact), "--max-length", "128",
    ]) == 0
    header, _ = read_artifact(artifact)
    assert header.get("max_length") == 128, "artifact identity must record truncation length"


@pytest.mark.parametrize("contents", [
    json.dumps(HEADER) + '\nnot-json\n{"id": "a", "weights": {"7": 1.0}}\n',
    '{"id": "a", "weights": {"7": 1.0}}\n',
    json.dumps(HEADER) + "\n" + json.dumps(HEADER) + "\n",
])
def test_resume_refuses_interior_corruption_or_missing_single_header(
    tmp_path, stub_sparse_encoder, contents,
):
    """BUG-001/002: baseline ignores corrupt interiors and accepts missing/repeated headers."""
    from encode_sparse import main

    artifact = tmp_path / "vectors.jsonl"
    artifact.write_text(contents, encoding="utf-8")
    original = artifact.read_bytes()
    passages = tmp_path / "passages.jsonl"
    passages.write_text('{"id": "b", "text": "b"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="interior|header"):
        main(["--input", str(passages), "--output", str(artifact), "--resume"])
    assert artifact.read_bytes() == original


@pytest.mark.parametrize("rows", [
    [{"id": "a", "weights": {"7": 1.0}}, HEADER],
    [{"id": "a", "weights": {"7": 1.0}}, HEADER | {"profile_id": "other"}],
])
def test_loader_refuses_second_header(tmp_path, rows):
    """BUG-003: baseline retains vectors while replacing their header identity."""
    artifact = tmp_path / "vectors.jsonl"
    _write_artifact(artifact, HEADER, rows)

    with pytest.raises(ValueError, match="header"):
        read_artifact(artifact)


@pytest.mark.parametrize("cli", ["encode", "load"])
@pytest.mark.parametrize("batch_size", [0, -1])
def test_sparse_clis_reject_nonpositive_batches_before_side_effects(
    monkeypatch, tmp_path, stub_sparse_encoder, cli, batch_size,
):
    """BUG-005: baseline reaches model/database work before rejecting, or returns success."""
    from encode_sparse import main as encode_main
    from load_sparse import main as load_main
    from recall.sparse import SpladeEncoder
    import recall.store as store_module

    touched = []

    def unexpected_work(*args, **kwargs):
        touched.append(True)
        raise ValueError("side effect before argument validation")

    monkeypatch.setattr(SpladeEncoder, "from_pretrained", unexpected_work)
    monkeypatch.setattr(store_module, "PgVectorStore", unexpected_work)
    monkeypatch.setenv("RECALL_MIGRATION_DSN", "unused")
    artifact = tmp_path / "vectors.jsonl"
    _write_artifact(artifact, HEADER, [{"id": "a", "weights": {"7": 1.0}}])
    args = ["--input", str(artifact), "--batch-size", str(batch_size)]
    args += ["--output", str(tmp_path / "new.jsonl")] if cli == "encode" else [
        "--table", "chunks", "--dim", "4",
    ]
    result = None
    try:
        (encode_main if cli == "encode" else load_main)(args)
    except SystemExit as exc:
        result = exc.code
    except ValueError:
        pass
    assert result == 2, "nonpositive batches must fail at the argument boundary"
    assert not touched
