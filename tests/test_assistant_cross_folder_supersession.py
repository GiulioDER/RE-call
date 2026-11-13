"""`recall_remember(supersedes=...)` when agent memo folders share one index root.

Several agents on one store: RECALL_INDEX_ROOT is shared and each agent's
RECALL_ASSISTANT_MEMOS is its own subfolder (`<root>/agent-a`, `<root>/agent-b`). Each
folder's ledger numbers from `note-0001`, so the same memo id exists in both. `recall_search`
names a memo of another folder by its path under the index root (`agent-b/note-0001`), and that
qualified id is what `supersedes` accepts for another agent's memo.

Invariants and the failure each one catches:
- X1 a correction one agent makes of its own memo reads `superseded` on that memo and leaves the
  other agent's same-named memo `ok`: `supersedes` written as a bare file name, which the resolver
  matches by stem anywhere under the index root, so both memos read ambiguous.
- X2 one agent can retire another agent's memo by its qualified id, written as that memo's path
  under the index root, and the memo reads `superseded`; `recall_forget_memo` still refuses a
  qualified id: the qualified form refused as `no memo`, or forget widened along with it. A
  successor of the target already indexed from a third folder makes the correction `stale_target`:
  the store check still resolving only this folder's ids.
- X3 a qualified target is refused when it is unknown, leaves the index root, lies in a folder
  without an authorship ledger, or was not written by the assistant there: the other folder's
  memos taken on trust.
- X4 the scan for a successor not yet indexed compares paths, not stems: a claim on
  `agent-b/note-0001` makes a second correction of it `stale_target`, and does not block the
  writer's own `note-0001`; a successor in the target's own folder counts too.

Red proof, 2026-11-12, on this workstation (single tests, light checks). At the baseline
77e81a0bf all four fail in their intended assertion: X1 "own memo 'ambiguous_supersession'
(superseded_by None), other agent's 'ambiguous_supersession'" (the memo nobody corrected is demoted
too); X2, X3 and X4 at the qualified id, refused as "no memo 'agent-b/note-0001'" (X3 at its
seeded case). Then each mutation alone on the implementation, its test run alone, the file
restored byte for byte (sha256 checked), and the file green:
- M1 (X1) `MemoFolder.write` writing `supersedes: <file name>`: the same ambiguous verdicts.
- M3 (X3) `MemoFolder.target_of` resolving without `within`: the escape is no longer refused
  (`RAISED ValueError ... is not in the subpath of ...` in place of `no memo`).
- M4 (X3) the author check reading the writer's own ledger: the seeded memo reads `no memo`.
- M5 (X4) `_target_conflict`'s scan comparing stems: the writer's own correction is `stale_target`.
- M6 (X4) the scan reading the writer's folder only: the successor in `agent-b/` is missed
  (`{'memo_id': 'note-0004'}`).
- M7 (X2) `AssistantMemos._indexed_successor` resolving the target as this folder's id: the
  successor indexed from `agent-c/` is missed (`{'memo_id': 'note-0003'}`).
- M8 (X2) `AssistantMemos.forget` resolving its id through `target_of`: the other agent's memo is
  deleted (`{'deleted': 1, 'memo_id': 'agent-b/note-0001', ...}`).
A first form of `target_of` also refused a folder without a ledger and a path through a link; a
mutation dropping the ledger check stayed green (no ledger means no authors, so the author check
refuses it), and the link check guarded nothing (a link inside the root resolves to its real
folder, and that folder's ledger decides), so both were removed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from recall.lite.assistant_memos import MemoFolder
from tests.test_assistant_surface import Call, _serve_both
from tests.test_assistant_surface import envs as envs  # noqa: F401  # the fixture, reused


@pytest.fixture
def agent_envs(envs: tuple[dict[str, str], dict[str, str]], tmp_path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """The assistant server writes as agent A; another agent's folder sits beside it."""
    assistant, plain = envs
    return {**assistant, "RECALL_ASSISTANT_MEMOS": str(tmp_path / "agent-a")}, plain


def test_a_correction_in_one_agent_folder_supersedes_only_that_agents_memo(
    agent_envs: tuple[dict[str, str], dict[str, str]], tmp_path: Path
) -> None:
    """X1."""
    other = MemoFolder(tmp_path / "agent-b", index_root=tmp_path)

    async def body(a: Call, p: Call, state: dict[str, Any]) -> None:
        assert other.write("The user drink tea at the office.") == {"memo_id": "note-0001"}
        await p("recall_index", path=str(tmp_path / "agent-b"))
        first = json.loads(await a("recall_remember", text="The user drink tea, every morning."))
        assert first == {"memo_id": "note-0001"}, first
        second = json.loads(await a("recall_remember", text="The user drink coffee now.", supersedes="note-0001"))
        assert second == {"memo_id": "note-0002"}, second
        hits = {h["memo_id"]: h for h in json.loads(await a("recall_search", query="user drink tea", k=10))["hits"]}
        assert {"note-0001", "agent-b/note-0001"} <= set(hits), sorted(hits)
        own, others = hits["note-0001"], hits["agent-b/note-0001"]
        assert (own["verdict"], others["verdict"]) == ("superseded", "ok"), (
            f"own memo {own['verdict']!r} (superseded_by {own['superseded_by']!r}), "
            f"other agent's {others['verdict']!r}"
        )

    _serve_both(agent_envs, body)


def test_one_agent_retires_another_agents_memo_by_its_qualified_id(
    agent_envs: tuple[dict[str, str], dict[str, str]], tmp_path: Path
) -> None:
    """X2."""
    other = MemoFolder(tmp_path / "agent-b", index_root=tmp_path)

    async def body(a: Call, p: Call, state: dict[str, Any]) -> None:
        assert other.write("The user drink tea at the office.") == {"memo_id": "note-0001"}
        await p("recall_index", path=str(tmp_path / "agent-b"))
        await a("recall_remember", text="The user drink tea, every morning.")
        named = {h["memo_id"] for h in json.loads(await a("recall_search", query="user drink tea", k=10))["hits"]}
        assert "agent-b/note-0001" in named, sorted(named)
        replaced = json.loads(await a(
            "recall_remember", text="The user drink coffee at the office now.", supersedes="agent-b/note-0001.md",
        ))
        assert replaced == {"memo_id": "note-0002"}, replaced
        written = (tmp_path / "agent-a" / "note-0002.md").read_text(encoding="utf-8")
        assert written == "---\nsupersedes: agent-b/note-0001.md\n---\nThe user drink coffee at the office now.\n", written
        hits = {h["memo_id"]: h for h in json.loads(await a("recall_search", query="user drink tea", k=10))["hits"]}
        retired, own = hits["agent-b/note-0001"], hits["note-0001"]
        assert (retired["verdict"], own["verdict"]) == ("superseded", "ok"), (retired, own)
        assert retired["superseded_by"], retired
        kept = json.loads(await a("recall_forget_memo", memo_id="agent-b/note-0001"))
        assert kept == {"deleted": 0}, kept
        assert (tmp_path / "agent-b" / "note-0001.md").is_file(), "a forget by qualified id reached another folder"
        # A successor in a third folder, which no folder scan of this write reads: the store says so.
        assert other.write("The user lives in Lyon.") == {"memo_id": "note-0002"}
        third = MemoFolder(tmp_path / "agent-c", index_root=tmp_path)
        assert third.write("The user lives in Turin.", supersedes="agent-b/note-0002") == {"memo_id": "note-0001"}
        await p("recall_index", path=str(tmp_path / "agent-b"))
        await p("recall_index", path=str(tmp_path / "agent-c"))
        late = json.loads(await a("recall_remember", text="The user lives in Milan.", supersedes="agent-b/note-0002"))
        assert late.get("conflict") == "stale_target", late

    _serve_both(agent_envs, body)


def test_a_qualified_target_must_be_another_assistant_folders_own_memo(tmp_path: Path) -> None:
    """X3."""
    writer = MemoFolder(tmp_path / "agent-a", index_root=tmp_path)
    other = MemoFolder(tmp_path / "agent-b", index_root=tmp_path)
    assert other.write("The user lives in Lyon.") == {"memo_id": "note-0001"}
    assert other.write("The user lives in Paris.", author="seed") == {"memo_id": "note-0002"}
    loose = tmp_path / "plain"
    loose.mkdir()
    (loose / "note-0001.md").write_text("The user lives in Rome.\n", encoding="utf-8")
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    MemoFolder(outside, index_root=outside).write("The user lives in Oslo.")
    escape = f"../{outside.name}/note-0001"
    for target, reason in (
        ("agent-b/note-0099", "no memo 'agent-b/note-0099'"),
        ("plain/note-0001", "no memo 'plain/note-0001'"),
        (escape, f"no memo {escape!r}"),
        ("agent-b/note-0002", "agent-b/note-0002 was not written by the assistant; it cannot be superseded"),
    ):
        try:
            refused: object = writer.write("The user lives in Turin.", supersedes=target)
        except Exception as exc:  # noqa: BLE001  # a raise in place of the refusal is what is asserted on
            refused = f"RAISED {type(exc).__name__}: {exc}"
        assert refused == {"refused": reason}, (target, refused)
    assert not list((tmp_path / "agent-a").glob("*.md")), "a refused correction wrote a memo"


def test_the_unindexed_successor_scan_compares_paths_not_stems(tmp_path: Path) -> None:
    """X4."""
    writer = MemoFolder(tmp_path / "agent-a", index_root=tmp_path)
    other = MemoFolder(tmp_path / "agent-b", index_root=tmp_path)
    assert other.write("The user lives in Lyon.") == {"memo_id": "note-0001"}
    assert writer.write("The user lives in Lyon, near the station.") == {"memo_id": "note-0001"}
    assert writer.write("The user lives in Turin.", supersedes="agent-b/note-0001") == {"memo_id": "note-0002"}
    again = writer.write("The user lives in Milan.", supersedes="agent-b/note-0001")
    assert again.get("conflict") == "stale_target", again
    own = writer.write("The user lives in Turin, near the station.", supersedes="note-0001")
    assert own == {"memo_id": "note-0003"}, own
    # A successor the target's own agent wrote, in the target's folder.
    assert other.write("The user works in Lyon.") == {"memo_id": "note-0002"}
    assert other.write("The user works in Turin.", supersedes="note-0002") == {"memo_id": "note-0003"}
    theirs = writer.write("The user works in Milan.", supersedes="agent-b/note-0002")
    assert theirs.get("conflict") == "stale_target", theirs
