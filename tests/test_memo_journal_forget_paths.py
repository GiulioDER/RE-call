"""A memo's request journal hashes go however the memo leaves the index, not only by `recall_forget_memo`.

The assistant's request journal (`.recall-authors.json`) holds unsalted SHA-256 hashes over each memo's
bytes, and a correction's identity hashes the corrected memo's revision. Once the memo is forgotten, a
guessable template and a short value make them an enumeration oracle (0000 to 9999 recovers a door
code). PR 950 erased them on `recall_forget_memo`; a memo removed any other way kept them.

Invariant, for every path: after the forget, the journal holds no token derived from the forgotten memo's
value (a token is derived from it when the same requests, written in a twin folder with that one value
changed, journal a different token), a retry of the forgotten request is refused whatever its text, and
the correction that named it still replays.

Paths, one test each:
- P1 the MCP `recall_forget` tool on a lite server whose index root holds the assistant memo folder;
- P2 the `recall forget --yes` CLI on a lite DSN;
- P3 a deletion by hand pruned by `recall index`;
- P4 a deletion by hand pruned by the lite refresh the server runs (`MemoFolder.refresh`).

P5 is the second note of PR 950's push review: rows holding the memo's bytes under its file name from
another path, with a stored file name that is not the one refresh gives it.

Red proof, all against 3f2de9156 (PR 950's head) with this change's production files reverted and these
tests kept, run 2026-10-19: P1 to P4 fail `a hash of the forgotten memo is still in the
journal`; P5 fails `forget deleted the file of a memo still served under a third name`. Mutations of the
fix, each red at the named assertion: the `forget_memo_requests` call removed from `forget_memory`
(P1, P2, journal assertion); removed from `Indexer` prune (P3, P4, journal assertion); the byte and name
match in `AssistantMemos._indexed_under_another_name` replaced by False (P5); the name test dropped from
it (the P5 control); and
`erase_memo_requests=False` dropped from `AssistantMemos.forget`, which turns
`tests/test_memo_corrections.py::test_a_failed_unlink_leaves_the_request_finishable` red.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from recall.lite.assistant_memos import MemoFolder
from tests.lite_cli_support import _offline, run_cli  # noqa: F401  # _offline: autouse fixture
from tests.test_assistant_surface import _serve_both, envs  # noqa: F401  # envs: fixture
from tests.test_lite_calibration import ContentWordEmbedder
from tests.test_lite_cli import _dsn
from tests.test_memo_corrections import remember, serving  # noqa: F401  # serving: fixture

GONE, KEPT, TWIN = "4512", "9031", "4513"


def _door(code: str) -> dict[str, str]:
    return {"subject": "user", "predicate": "door_code", "object": code}


def _text(code: str) -> str:
    return f"The user door code is {code}."


def _hashes(folder: Path) -> set[str]:
    """Every SHA-256-shaped token the folder's request journal holds."""
    return set(re.findall(r"[0-9a-f]{64}", (folder / ".recall-authors.json").read_text(encoding="utf-8")))


def _write_pair(folder: MemoFolder, first_code: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """The forgotten memo (request `first`) and its correction (request `second`), unindexed."""
    first = folder.write(_text(first_code), fact=_door(first_code), request_id="first")
    second = folder.write(_text(KEPT), fact=_door(KEPT), supersedes=first["memo_id"],
                          expected_revision=first["receipt"]["revision"], request_id="second")
    return first, second


def _derived(folder: Path, tmp_path: Path) -> set[str]:
    """The journal tokens derived from the forgotten value: the twin writes the same requests with it
    changed."""
    _write_pair(MemoFolder(tmp_path / "twin"), TWIN)
    derived = _hashes(folder) - _hashes(tmp_path / "twin")
    assert derived, "the scenario must journal hashes derived from the value it forgets"
    return derived


def _assert_erased(folder: MemoFolder, derived: set[str], first: dict[str, Any]) -> None:
    assert not derived & _hashes(folder.folder), "a hash of the forgotten memo is still in the journal"
    for text in (_text(GONE), _text("0000")):
        retry = folder.write(text, fact=_door(GONE), request_id="first")
        assert retry.get("conflict") == "memo_missing", f"a retry of the forgotten request was answered: {retry}"
    replay = folder.write(_text(KEPT), fact=_door(KEPT), supersedes=first["memo_id"],
                          expected_revision=first["receipt"]["revision"], request_id="second")
    assert replay.get("receipt", {}).get("replayed") is True, f"the correction no longer replays: {replay}"


def test_p1_the_mcp_forget_tool_erases_the_journal(envs, tmp_path: Path) -> None:  # noqa: F811
    folder = MemoFolder(tmp_path / "memos", index_root=tmp_path)
    first, _ = _write_pair(folder, GONE)
    derived = _derived(folder.folder, tmp_path)
    path = str(folder.path_of(first["memo_id"]))

    async def body(a: Any, p: Any, state: dict[str, Any]) -> None:
        await p("recall_index", path=str(folder.folder))
        forgot = json.loads(await p("recall_forget", sources=[path]))
        assert forgot["sources_removed"] == [path], forgot

    _serve_both(envs, body)
    _assert_erased(folder, derived, first)


def test_p2_the_cli_forget_erases_the_journal(tmp_path: Path) -> None:
    folder = MemoFolder(tmp_path / "memos")
    first, _ = _write_pair(folder, GONE)
    derived = _derived(folder.folder, tmp_path)
    base = ["--embedder", "hashing", "--dsn", _dsn(tmp_path)]
    run_cli([*base, "index", str(folder.folder)])
    path = str(folder.path_of(first["memo_id"]))
    done = run_cli([*base, "forget", path, "--yes"])
    assert "forgot" in done and "WARNING" not in done, done
    _assert_erased(folder, derived, first)


def test_p3_a_hand_deletion_pruned_by_recall_index_erases_the_journal(tmp_path: Path) -> None:
    folder = MemoFolder(tmp_path / "memos")
    first, _ = _write_pair(folder, GONE)
    derived = _derived(folder.folder, tmp_path)
    base = ["--embedder", "hashing", "--dsn", _dsn(tmp_path)]
    run_cli([*base, "index", str(folder.folder)])
    folder.path_of(first["memo_id"]).unlink()
    pruned = run_cli([*base, "index", str(folder.folder)])
    assert "pruned 1 source(s)" in pruned, pruned
    _assert_erased(folder, derived, first)


def test_p4_a_hand_deletion_pruned_by_the_lite_refresh_erases_the_journal(serving, tmp_path: Path) -> None:  # noqa: F811
    store, memos = serving
    first = remember(store, memos, _text(GONE), fact=_door(GONE), request_id="first")
    remember(store, memos, _text(KEPT), fact=_door(KEPT), supersedes=first["memo_id"],
             expected_revision=first["receipt"]["revision"], request_id="second")
    derived = _derived(memos.folder, tmp_path)
    path = memos.path_of(first["memo_id"])
    path.unlink()
    report = memos.refresh(store, ContentWordEmbedder())
    assert report.deleted == 1 and str(path) not in store.source_raw_hashes(), report
    _assert_erased(memos, derived, first)


def test_p5_forget_refuses_a_memo_served_under_a_third_name(serving) -> None:  # noqa: F811
    """The rows hold the memo's bytes under another absolute path AND another stored file name, so
    neither name `AssistantMemos.forget` knew finds them; the file must stay, since deleting it would
    leave its text served while the caller reads the memo as gone."""
    store, memos = serving
    written = remember(store, memos, _text(GONE))
    path = memos.path_of(written["memo_id"])
    elsewhere = path.parent.parent / "moved" / path.name
    with store._write():
        store._conn.execute(
            "UPDATE chunks SET source = ?, metadata = json_set(metadata, '$.file', ?) WHERE source = ?",
            (str(elsewhere), f"moved/{path.name}", str(path)),
        )
    assert not store.sources_for_identifiers([str(path), f"memos/{path.name}"])
    refused = ""
    try:
        memos.forget(store, ContentWordEmbedder(), written["memo_id"])
    except RuntimeError as exc:
        refused = str(exc)
    assert path.exists() and "incomplete" in refused, (
        f"forget deleted the file of a memo still served under a third name (refusal: {refused!r})")


def test_p5_control_a_same_text_memo_under_another_name_does_not_block(serving) -> None:  # noqa: F811
    """Control for P5's check: two memos with the same text are two memos. Forgetting the unindexed
    one, while the other's rows hold the same bytes, deletes it (the name differs). Passes on both
    the base and the fix. Red with the name test dropped from the byte match in
    `AssistantMemos._indexed_under_another_name`: `a same-text memo under another name blocked the
    forget`."""
    store, memos = serving
    remember(store, memos, _text(GONE))
    second = memos.write(_text(GONE))  # published, not indexed
    path = memos.path_of(second["memo_id"])
    assert str(path) not in store.source_raw_hashes()
    try:
        forgot = memos.forget(store, ContentWordEmbedder(), second["memo_id"])
    except RuntimeError as exc:
        pytest.fail(f"a same-text memo under another name blocked the forget: {exc}")
    assert forgot["deleted"] == 1 and not path.exists(), forgot
