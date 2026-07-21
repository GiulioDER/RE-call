"""Fail-closed behaviour for the three places this library previously failed open.

Each of these is the same shape of defect: the unsafe condition is DETECTED and then execution
continues anyway. A warning nobody reads is not a control.
"""
from __future__ import annotations

import os
import sys

import pytest

from recall.index import Indexer
from recall.store import require_secure_dsn, warn_if_insecure_dsn


class _NullEmbedder:
    dim = 2
    name = "null"

    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


class _RecordingStore:
    """Captures what an Indexer would write, without a database."""

    def __init__(self):
        self.chunks = []

    def replace_sources(self, sources, chunks, embeddings):
        self.chunks = list(chunks)
        return len(chunks)


# --------------------------------------------------------------------------------------------
# Default credentials against a remote host
# --------------------------------------------------------------------------------------------

REMOTE_DEFAULT = "postgresql://recall:recall@db.example.com:5432/recall"


def test_require_secure_dsn_refuses_default_credentials_on_a_remote_host():
    """`warn_if_insecure_dsn` prints and RETURNS — the process carries on talking to a shared
    database with a published password. A server must refuse instead."""
    assert warn_if_insecure_dsn(REMOTE_DEFAULT) is not None  # detected before, and ignored
    with pytest.raises(PermissionError, match="default 'recall:recall' credentials"):
        require_secure_dsn(REMOTE_DEFAULT)


def test_require_secure_dsn_allows_an_explicit_opt_out(monkeypatch):
    monkeypatch.setenv("RECALL_ALLOW_INSECURE_DSN", "1")
    require_secure_dsn(REMOTE_DEFAULT)  # must not raise


def test_require_secure_dsn_allows_local_and_non_default_credentials():
    require_secure_dsn("postgresql://recall:recall@localhost:5432/recall")
    require_secure_dsn("postgresql://recall:a-real-password@db.example.com:5432/recall")


def test_require_secure_dsn_never_leaks_the_password_in_its_message():
    """The exception is going to a log; the DSN in it must be redacted."""
    with pytest.raises(PermissionError) as exc:
        require_secure_dsn("postgresql://recall:recall@db.example.com:5432/recall")
    assert "recall:recall@" not in str(exc.value)
    assert "***" in str(exc.value)


# --------------------------------------------------------------------------------------------
# Index-root confinement vs symlinks
# --------------------------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32" and not os.environ.get("CI"),
                    reason="creating a directory symlink on Windows needs elevation")
def test_indexing_does_not_follow_a_symlink_out_of_the_index_root(tmp_path):
    """`RECALL_INDEX_ROOT` confines the PATH ARGUMENT, but the walk is a glob.

    `pathlib` only gained `recurse_symlinks` (defaulting False) in 3.13, and this package
    supports 3.11+. On 3.11/3.12 `**` follows directory symlinks, so a symlink planted inside an
    otherwise-confined root reads files from outside it — the confinement check passes and the
    read happens anyway.
    """
    root = tmp_path / "memory"
    root.mkdir()
    (root / "inside.md").write_text("a memory that belongs here", encoding="utf-8")

    secret_dir = tmp_path / "elsewhere"
    secret_dir.mkdir()
    (secret_dir / "secret.md").write_text("private notes from outside the root", encoding="utf-8")

    try:
        (root / "escape").symlink_to(secret_dir, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:  # pragma: no cover - platform dependent
        pytest.skip(f"cannot create a directory symlink here: {exc}")

    store = _RecordingStore()
    Indexer(store, _NullEmbedder()).index_path(root)

    texts = " ".join(c.text for c in store.chunks)
    assert "belongs here" in texts
    assert "private notes" not in texts, "indexed a file from outside the index root"


def test_a_symlinked_file_inside_the_root_pointing_out_is_also_refused(tmp_path):
    """The file-level case: no directory symlink needed, so it reproduces on every platform."""
    root = tmp_path / "memory"
    root.mkdir()
    (root / "inside.md").write_text("a memory that belongs here", encoding="utf-8")

    outside = tmp_path / "secret.md"
    outside.write_text("private notes from outside the root", encoding="utf-8")
    try:
        (root / "linked.md").symlink_to(outside)
    except (OSError, NotImplementedError) as exc:  # pragma: no cover - platform dependent
        pytest.skip(f"cannot create a symlink here: {exc}")

    store = _RecordingStore()
    Indexer(store, _NullEmbedder()).index_path(root)

    texts = " ".join(c.text for c in store.chunks)
    assert "belongs here" in texts
    assert "private notes" not in texts, "indexed a file from outside the index root"


def test_indexing_a_single_file_directly_still_works(tmp_path):
    """The confinement filter must not break the explicit single-file case."""
    f = tmp_path / "note.md"
    f.write_text("just one note", encoding="utf-8")
    store = _RecordingStore()
    stats = Indexer(store, _NullEmbedder()).index_path(f)
    assert stats.chunks == 1


def test_confined_to_filters_paths_outside_the_root_without_needing_a_symlink():
    """Direct unit test of the filter, so the guard is covered on platforms where a test cannot
    create a symlink (Windows without developer mode) and the cases above skip."""
    from pathlib import Path

    from recall.index import _confined_to

    root = Path(__file__).resolve().parent
    inside = Path(__file__).resolve()
    outside = root.parent / "pyproject.toml"

    kept = _confined_to(root, [inside, outside])
    assert kept == [inside]


def test_confined_to_drops_a_nonexistent_or_directory_path():
    """Only real files may be indexed: a resolved path that is a directory (or gone) is dropped
    rather than handed to `read_text`."""
    from pathlib import Path

    from recall.index import _confined_to

    root = Path(__file__).resolve().parent
    assert _confined_to(root, [root]) == []                    # a directory, not a file
    assert _confined_to(root, [root / "nope.md"]) == []        # does not exist
