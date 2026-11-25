"""A memo a person forgot stays forgotten on a lite store while its file is still in the folder.

Owner decision, 2026-11-25: `recall lite refresh` used to index a forgotten memo again on its next
run, because the file was still there, which made `recall forget` undo itself. Now a forget or a
withdraw leaves a `memo_forgotten` mark in the store's audit ledger, in the delete's own
transaction, and the indexer skips a marked file (a refresh, or `recall index <folder>`) until the
file itself is indexed by name (`recall index <file>`), which records `memo_restored`. The
indexer's prune of a file deleted from disk marks nothing.

Invariants and the failure each one catches:
- F1 forget then refresh: the memo is not indexed again, and the refresh reports it as forgotten.
- F2 forget then `recall index <folder>`: the same.
- F3 `recall index <file>` brings it back, and a later refresh keeps it.
- F4 withdraw marks it too.
- F5 a file deleted from disk and pruned by a refresh comes back when it returns.

Red proof: in each docstring.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from tests.lite_cli_support import _offline, run_cli  # noqa: F401  # _offline: autouse fixture
from tests.test_lite_cli import _dsn, _memos

MEMOS = {
    "keep.md": "# Keep\n\nThe keep memo stays indexed throughout.\n",
    "secret.md": "# Secret\n\nThe zephyr launch code lives in this memo.\n",
}


def _setup(tmp_path: Path) -> tuple[list[str], Path, str]:
    folder = _memos(tmp_path / "mem", MEMOS)
    base = ["--embedder", "hashing", "--dsn", _dsn(tmp_path)]
    run_cli([*base, "index", str(folder)])
    return base, folder, str((folder / "secret.md").resolve())


def _refresh(tmp_path: Path, base: list[str], folder: Path) -> dict[str, object]:
    out = run_cli([*base, "lite", "refresh", "--store", str(tmp_path / "store" / "memory.db"),
                   "--memory", str(folder), "--json"])
    report: dict[str, object] = json.loads(out)
    return report


def _indexed(tmp_path: Path) -> set[str]:
    with sqlite3.connect(f"file:{(tmp_path / 'store' / 'memory.db').as_posix()}?mode=ro", uri=True) as conn:
        return {Path(source).name for (source,) in conn.execute("SELECT DISTINCT source FROM chunks")}


def test_f1_a_forgotten_memo_is_not_refreshed_back(tmp_path: Path) -> None:
    """Node tests/test_lite_forget_stays_forgotten.py::test_f1_a_forgotten_memo_is_not_refreshed_back.

    Red on master (`c6104f688`): fails on "refresh brought the forgotten memo back"
    (`{'keep.md', 'secret.md'}`). Red, mutation of `_refresh_locked` not adding `stats.forgotten`:
    fails on `report["forgotten"] == 1`.
    """
    base, folder, secret = _setup(tmp_path)
    run_cli([*base, "forget", secret, "--yes"])
    report = _refresh(tmp_path, base, folder)
    assert _indexed(tmp_path) == {"keep.md"}, "refresh brought the forgotten memo back"
    assert report["forgotten"] == 1, report


def test_f2_a_folder_index_skips_a_forgotten_memo(tmp_path: Path) -> None:
    """Node tests/test_lite_forget_stays_forgotten.py::test_f2_a_folder_index_skips_a_forgotten_memo.

    Red on master (`c6104f688`): fails on "a folder index brought the forgotten memo back". Red,
    mutation of `_index_summary` dropping the forgotten clause: fails on "left out 1 file(s) you
    forgot" (the output said only "1 unchanged").
    """
    base, folder, secret = _setup(tmp_path)
    run_cli([*base, "forget", secret, "--yes"])
    out = run_cli([*base, "index", str(folder)])
    assert _indexed(tmp_path) == {"keep.md"}, "a folder index brought the forgotten memo back"
    assert "left out 1 file(s) you forgot" in out, out


def test_f3_indexing_the_file_by_name_brings_it_back_for_good(tmp_path: Path) -> None:
    """Node tests/test_lite_forget_stays_forgotten.py::test_f3_indexing_the_file_by_name_brings_it_back_for_good.

    Red, mutation of `Indexer._index_path` not recording `memo_restored`: fails on
    `report["forgotten"] == 0` (1: the mark outlived the explicit index, so the next walk left the
    edited file out; the prune removes only files gone from disk, so its old rows stayed).
    """
    base, folder, secret = _setup(tmp_path)
    run_cli([*base, "forget", secret, "--yes"])
    run_cli([*base, "index", secret])
    assert _indexed(tmp_path) == {"keep.md", "secret.md"}, "indexing the file by name did not bring it back"
    (folder / "secret.md").write_text(MEMOS["secret.md"] + "\nEdited after it came back.\n", encoding="utf-8")
    report = _refresh(tmp_path, base, folder)
    assert _indexed(tmp_path) == {"keep.md", "secret.md"}, "a refresh after the explicit index dropped it again"
    assert report["forgotten"] == 0, report


def test_f4_a_withdrawn_memo_is_not_refreshed_back(tmp_path: Path) -> None:
    """Node tests/test_lite_forget_stays_forgotten.py::test_f4_a_withdrawn_memo_is_not_refreshed_back.

    Red on master (`c6104f688`), and red by mutation of `_forget_memory_unlocked`
    (recall_mcp/lifecycle.py) marking forgets only: fails on "refresh brought the withdrawn memo
    back".
    """
    base, folder, secret = _setup(tmp_path)
    run_cli([*base, "withdraw", secret, "--yes"])
    _refresh(tmp_path, base, folder)
    assert _indexed(tmp_path) == {"keep.md"}, "refresh brought the withdrawn memo back"


def test_f5_a_pruned_file_comes_back_when_it_returns(tmp_path: Path) -> None:
    """Node tests/test_lite_forget_stays_forgotten.py::test_f5_a_pruned_file_comes_back_when_it_returns.

    Red, mutation of `LiteStore.delete_sources` marking every delete (the prune's too): fails on
    "a returned file was treated as forgotten".
    """
    base, folder, _ = _setup(tmp_path)
    (folder / "secret.md").unlink()
    _refresh(tmp_path, base, folder)
    assert _indexed(tmp_path) == {"keep.md"}
    (folder / "secret.md").write_text(MEMOS["secret.md"], encoding="utf-8")
    _refresh(tmp_path, base, folder)
    assert _indexed(tmp_path) == {"keep.md", "secret.md"}, "a returned file was treated as forgotten"
