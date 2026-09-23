"""The content-addressed micro view store, the incremental builder, and memory-mapped loading.

Red proofs (2026-09-23, each mutation applied and reverted):

* ``test_a_second_assembly_embeds_nothing_and_matches_the_first`` failed when ``assemble``
  treated every chunk as missing (``store.get(...) is None`` replaced by ``True``).
* ``test_changing_one_chunk_re_embeds_only_that_chunk`` failed when the digest ignored the chunk
  text (``chunk_view_digest`` hashed only the header).
* ``test_view_parameters_are_part_of_the_key`` failed when the header dropped the stride.
* ``test_a_damaged_row_is_a_miss_not_a_wrong_vector`` failed when ``get`` skipped the byte-length
  check: the store raised ``ValueError`` from ``np.frombuffer`` on the damaged row instead of
  reporting a miss (a behavioural failure of the store, not of the test's setup).
* ``test_micro_views_are_word_ranges_of_the_chunk`` failed when ``chunk_micro_views`` stopped
  deduplicating.
* ``test_artifacts_load_memory_mapped`` failed when the loader used ``np.load`` without
  ``mmap_mode``.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from recall.atomic_rescue import (
    clear_atomic_rescue_artifact_cache,
    load_atomic_rescue_artifact,
    write_atomic_rescue_artifact,
)
from recall.atomic_view_store import AtomicViewStore, chunk_view_digest
from recall.atomizer import chunk_micro_views
from scripts.build_atomic_micro_artifact import assemble

DIM = 4
PROFILE = "profile-a"


def _parents(texts: list[str]) -> list[tuple[str, str, int, str]]:
    return [(f"chunk-{i}", f"memo-{i // 2}.md", i % 2, text) for i, text in enumerate(texts)]


def _text(seed: int) -> str:
    return " ".join(f"w{seed}x{i}" for i in range(60))


class _Embedder:
    def __init__(self) -> None:
        self.groups = 0
        self.views = 0

    def __call__(self, groups: list[list[str]]) -> list[list[list[float]]]:
        self.groups += len(groups)
        out = []
        for group in groups:
            self.views += len(group)
            out.append([[float(len(view)), float(hash(view) % 97) + 1.0, 1.0, 2.0] for view in group])
        return out


def test_a_second_assembly_embeds_nothing_and_matches_the_first(tmp_path: Path) -> None:
    parents = _parents([_text(i) for i in range(6)])
    with AtomicViewStore(tmp_path / "views.sqlite") as store:
        first_embedder = _Embedder()
        first, meta_a, counts_a = assemble(parents, store, PROFILE, DIM, first_embedder)
        second_embedder = _Embedder()
        second, meta_b, counts_b = assemble(parents, store, PROFILE, DIM, second_embedder)
    assert first_embedder.groups == 6
    assert second_embedder.groups == 0
    assert counts_b["embedded_chunks"] == 0 and counts_b["reused_chunks"] == 6
    assert np.array_equal(first, second)
    assert meta_a == meta_b


def test_changing_one_chunk_re_embeds_only_that_chunk(tmp_path: Path) -> None:
    texts = [_text(i) for i in range(6)]
    with AtomicViewStore(tmp_path / "views.sqlite") as store:
        assemble(_parents(texts), store, PROFILE, DIM, _Embedder())
        texts[3] = _text(99)
        embedder = _Embedder()
        _, _, counts = assemble(_parents(texts), store, PROFILE, DIM, embedder)
    assert embedder.groups == 1
    assert counts["embedded_chunks"] == 1 and counts["reused_chunks"] == 5


def test_a_new_embedder_profile_never_reuses_another_profiles_vectors(tmp_path: Path) -> None:
    parents = _parents([_text(i) for i in range(3)])
    with AtomicViewStore(tmp_path / "views.sqlite") as store:
        assemble(parents, store, PROFILE, DIM, _Embedder())
        embedder = _Embedder()
        assemble(parents, store, "profile-b", DIM, embedder)
    assert embedder.groups == 3


def test_view_parameters_are_part_of_the_key() -> None:
    text = _text(1)
    assert chunk_view_digest(text) != chunk_view_digest(text, stride=24)
    assert chunk_view_digest(text) != chunk_view_digest(text, size=20)
    assert chunk_view_digest(text) != chunk_view_digest(_text(2))


def test_a_damaged_row_is_a_miss_not_a_wrong_vector(tmp_path: Path) -> None:
    path = tmp_path / "views.sqlite"
    with AtomicViewStore(path) as store:
        store.put(PROFILE, "d1", np.ones((3, DIM), dtype=np.float32))
        store.commit()
        assert store.get(PROFILE, "d1", view_count=3, dimension=DIM) is not None
        assert store.get(PROFILE, "d1", view_count=2, dimension=DIM) is None
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE micro_views SET vectors = ? WHERE digest = 'd1'", (b"\x00" * 7,))
    with AtomicViewStore(path) as store:
        assert store.get(PROFILE, "d1", view_count=3, dimension=DIM) is None


def test_micro_views_are_word_ranges_of_the_chunk() -> None:
    repeated = " ".join(["alpha beta gamma delta epsilon zeta"] * 12)
    views = chunk_micro_views(repeated)
    assert len(views) == len(set(views))
    words = repeated.split()
    for view in views:
        assert len(view.split()) <= 24
        assert view in " ".join(words)
    assert chunk_micro_views("a b c") == []


def test_artifacts_load_memory_mapped(tmp_path: Path) -> None:
    clear_atomic_rescue_artifact_cache()
    matrix = np.eye(3, DIM, dtype=np.float32)
    manifest = write_atomic_rescue_artifact(
        tmp_path / "artifact",
        matrix=matrix,
        views=[
            {"chunk_id": f"c{i}", "source": "s.md", "parent_ordinal": i, "view_ordinal": 0}
            for i in range(3)
        ],
        generation_id="g",
        calibration_id="c",
        pipeline_fingerprint="p",
        corpus_fingerprint="0" * 64,
        embedding_profile="t",
        embedding_fingerprint="f",
        ordinary_chunk_count=3,
        source_commit="test",
    )
    artifact = load_atomic_rescue_artifact(manifest)
    base = artifact.matrix
    while getattr(base, "base", None) is not None and not isinstance(base, np.memmap):
        base = base.base
    assert isinstance(base, np.memmap)
    assert not artifact.matrix.flags.writeable
    clear_atomic_rescue_artifact_cache()
