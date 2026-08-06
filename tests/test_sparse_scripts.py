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

from encode_sparse import passage_id, passage_text  # noqa: E402
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
