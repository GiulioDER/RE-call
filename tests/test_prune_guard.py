"""The re-index prune guard.

Re-indexing deletes rows for files that are gone from disk. That is correct — without it the
trust layer serves a deleted memory as `ok` forever — but it makes `recall index` a destructive
command that nobody treats as one. `recall forget`, which *is* obviously destructive, is dry-run
by default and demands `--yes`. `index` had no such brake: point it at a corpus directory whose
contents are missing (unmounted volume, interrupted sync, a path that still resolves) and the
whole corpus is deleted silently, exit code 0.

The guard refuses a mass disappearance rather than trying to distinguish one from a real mass
deletion — at the filesystem level the two are identical, so the only safe reading is that
something is wrong. These tests pin both directions: it must fire on the disaster, and it must
stay out of the way of ordinary deletions, or it will be turned off.
"""
from __future__ import annotations

import pathlib
import uuid

import psycopg
import pytest

from recall.index import (
    DEFAULT_MAX_PRUNE_FRACTION,
    PRUNE_GUARD_MIN_SOURCES,
    Indexer,
    PruneGuardTripped,
    _prune_fraction_from_env,
)
from recall.store import PgVectorStore

from tests.conftest import TEST_DSN, requires_db


class _E:
    dim = 4
    name = "e"

    def embed(self, texts):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


@pytest.fixture
def corpus(tmp_path) -> pathlib.Path:
    """A corpus comfortably above PRUNE_GUARD_MIN_SOURCES, so the guard is in scope."""
    d = tmp_path / "memory"
    d.mkdir()
    for i in range(PRUNE_GUARD_MIN_SOURCES + 5):
        (d / f"memo{i}.md").write_text(f"memory number {i}\n", encoding="utf-8")
    return d


@pytest.fixture
def store():
    name = "pg_" + uuid.uuid4().hex[:8]
    s = PgVectorStore(TEST_DSN, dim=4, table=name)
    s.ensure_schema()
    try:
        yield s
    finally:
        s.close()
        with psycopg.connect(TEST_DSN, autocommit=True) as conn:
            conn.execute(f"DROP TABLE IF EXISTS {name}")


@requires_db
def test_a_vanished_corpus_is_refused_and_nothing_is_deleted(corpus, store):
    """The disaster case. `deleted` is not the assertion — surviving row count is."""
    Indexer(store, _E()).index_path(corpus)
    before = store.count()
    assert before > 0

    for f in corpus.glob("*.md"):
        f.unlink()

    with pytest.raises(PruneGuardTripped) as exc:
        Indexer(store, _E()).index_path(corpus)

    assert store.count() == before, "guard raised but rows were deleted anyway"
    # The message has to be actionable: what it refused, and how to proceed deliberately.
    assert "--allow-prune" in str(exc.value)


@requires_db
def test_deleting_a_few_files_still_prunes_normally(corpus, store):
    """The guard must not break the ordinary case, or it gets disabled and protects nothing."""
    Indexer(store, _E()).index_path(corpus)
    before = store.count()

    (corpus / "memo0.md").unlink()
    stats = Indexer(store, _E()).index_path(corpus)

    assert stats.deleted == 1
    assert store.count() == before - 1


@requires_db
def test_allow_prune_lets_a_confirmed_deletion_through(corpus, store):
    """The escape hatch has to actually work, or the guard is a wall."""
    Indexer(store, _E()).index_path(corpus)
    for f in corpus.glob("*.md"):
        f.unlink()

    stats = Indexer(store, _E(), allow_prune=True).index_path(corpus)

    assert stats.deleted == PRUNE_GUARD_MIN_SOURCES + 5
    assert store.count() == 0


@requires_db
def test_a_small_corpus_is_not_guarded(tmp_path, store):
    """Below the floor a fraction is meaningless — deleting one of two memos is 50% and routine."""
    d = tmp_path / "small"
    d.mkdir()
    for i in range(PRUNE_GUARD_MIN_SOURCES - 1):
        (d / f"m{i}.md").write_text(f"note {i}\n", encoding="utf-8")
    Indexer(store, _E()).index_path(d)

    for f in d.glob("*.md"):
        f.unlink()
    stats = Indexer(store, _E()).index_path(d)  # must not raise

    assert stats.deleted == PRUNE_GUARD_MIN_SOURCES - 1
    assert store.count() == 0


@requires_db
def test_another_corpus_under_a_different_root_cannot_dilute_the_fraction(tmp_path, store):
    """The denominator is scoped to the root being indexed.

    Measured against the whole table instead, a second healthy corpus would keep the ratio below
    the threshold and a total wipe of THIS one would sail through — the guard silently weakening
    as more corpora are added, which is the opposite of what you want.
    """
    a = tmp_path / "a"
    b = tmp_path / "b"
    for d in (a, b):
        d.mkdir()
        for i in range(PRUNE_GUARD_MIN_SOURCES + 5):
            (d / f"m{i}.md").write_text(f"{d.name} note {i}\n", encoding="utf-8")
    Indexer(store, _E()).index_path(a)
    Indexer(store, _E()).index_path(b)
    before = store.count()

    for f in a.glob("*.md"):
        f.unlink()

    with pytest.raises(PruneGuardTripped):
        Indexer(store, _E()).index_path(a)
    assert store.count() == before


# --------------------------------------------------------------------------------------------
# "Gone from disk" must mean gone from disk
# --------------------------------------------------------------------------------------------


@requires_db
def test_a_file_excluded_by_this_run_s_glob_is_not_treated_as_deleted(tmp_path, store):
    """A narrower glob must not delete what a wider one indexed.

    The prune asked "is this source in the set the current glob matched?" and read "no" as
    "deleted from disk". Those are different questions the moment `--glob` varies between runs on
    one root — an advertised workflow (`--glob`'s own help offers `'**/*.py'` for code), and the
    two globs share a table because `--table` defaults to the same name for every invocation.

    The guard does not catch this: the .py sources are a minority of the corpus, so the fraction
    stays under the threshold and the deletion is silent, exit code 0.
    """
    d = tmp_path / "mixed"
    d.mkdir()
    for i in range(PRUNE_GUARD_MIN_SOURCES + 15):
        (d / f"memo{i}.md").write_text(f"memory number {i}\n", encoding="utf-8")
    for i in range(PRUNE_GUARD_MIN_SOURCES + 1):
        (d / f"mod{i}.py").write_text(f"# module {i}\n", encoding="utf-8")

    Indexer(store, _E()).index_path(d, glob="**/*")
    before = store.count()
    py_sources = {s for s in store.source_content_hashes() if s.endswith(".py")}
    assert py_sources, "fixture failed to index the .py files"

    # Re-index the same root with the default markdown glob. Every .py file is still on disk.
    stats = Indexer(store, _E()).index_path(d)

    assert stats.deleted == 0, "pruned files that are still on disk"
    assert store.count() == before
    assert {s for s in store.source_content_hashes() if s.endswith(".py")} == py_sources


@requires_db
def test_a_genuinely_deleted_file_is_still_pruned_under_a_narrower_glob(tmp_path, store):
    """The control for the test above: don't fix the false positive by pruning nothing.

    Same shape — a wide glob then a narrow one — except one .py file is really unlinked. It must
    still go, or the fix has simply disabled the feature for mixed corpora.
    """
    d = tmp_path / "mixed"
    d.mkdir()
    for i in range(PRUNE_GUARD_MIN_SOURCES + 15):
        (d / f"memo{i}.md").write_text(f"memory number {i}\n", encoding="utf-8")
    for i in range(PRUNE_GUARD_MIN_SOURCES + 1):
        (d / f"mod{i}.py").write_text(f"# module {i}\n", encoding="utf-8")

    Indexer(store, _E()).index_path(d, glob="**/*")
    before = store.count()

    (d / "mod0.py").unlink()
    stats = Indexer(store, _E()).index_path(d)

    assert stats.deleted == 1
    assert store.count() < before
    assert not any(s.endswith("mod0.py") for s in store.source_content_hashes())


# --------------------------------------------------------------------------------------------
# The env override — no database needed
# --------------------------------------------------------------------------------------------


def test_prune_fraction_defaults_when_unset(monkeypatch):
    monkeypatch.delenv("RECALL_MAX_PRUNE_FRACTION", raising=False)
    assert _prune_fraction_from_env() == DEFAULT_MAX_PRUNE_FRACTION


def test_prune_fraction_reads_a_valid_override(monkeypatch):
    monkeypatch.setenv("RECALL_MAX_PRUNE_FRACTION", "0.9")
    assert _prune_fraction_from_env() == 0.9


@pytest.mark.parametrize("raw", ["", "half", "0", "-0.5", "1.5", "50", "nan", "inf"])
def test_a_malformed_or_out_of_range_override_falls_back_to_the_default(monkeypatch, raw):
    """Falls back rather than clamping.

    `50` is the interesting one: someone meaning 50 percent. Clamping it to 1.0 would read as
    "only guard a total wipe" — weakening the protection at the exact moment it was being
    configured. `nan` matters because every comparison against it is False, so a NaN threshold
    would make the guard silently unreachable.
    """
    monkeypatch.setenv("RECALL_MAX_PRUNE_FRACTION", raw)
    assert _prune_fraction_from_env() == DEFAULT_MAX_PRUNE_FRACTION
