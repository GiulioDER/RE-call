"""`recall_report_use` on the assistant surface accepts the ids `recall_search` named the memos by.

With RECALL_ASSISTANT_MEMOS set, `recall_search` answers in the compact view: a hit is named by
its `memo_id` (`note-0001` in the agent's own folder, `agent-b/note-0001` in another folder under
the index root) and never by its source. A deployment that also serves `recall_report_use`
(`RECALL_MCP_TOOLS=assistant,recall_report_use`) must accept those ids, and its description must
say so; otherwise the model is told to "name each by its source as search returned it" while no
source was returned.

Invariant and the failure it catches:
- R1 each `memo_id` a compact search returned, own folder and qualified, is accepted in `used` and
  `wrong` and recorded as that memo's stored source; an id naming no memo is still refused; the
  description names `memo_id`: report_use resolving only a stored source or a bare file name, so
  every id search showed is refused as "not a memory in this corpus".
- R2 under a source policy an id is authorized as the source it names, and a refused id is named
  as the caller wrote it: the policy asked about the id itself (so a readable memo is refused), or
  the refusal printing the translated source (so a memo the caller may not read reads differently
  from one that does not exist).
- R3 an id whose stored file name two sources share is left unresolved rather than credited to
  either: `files_named` taking the first source.

Red proof, 2026-11-13, on this workstation (single file, light check). At the baseline b724962f0
R1 fails in its intended assertion: "search's own ids were refused: ... 'note-0001' in used is not
a memory in this corpus; name it as search returned it", the refusal an agent met in use.
R2 and R3 call symbols the baseline lacks (`aliases`, `files_named`), so they are proved by
mutating the fix instead. Each mutation alone, its test run alone, the file restored byte for byte
(sha256 checked), then the file green:
- M1 (R1) the server passing no aliases: the same refusal of `note-0001`.
- M2 (R1) the server keeping the plain description: the description assertion (no `memo_id`).
- M3 (R1) `AssistantMemos.files_named` resolving every id in the own folder: `agent-b/note-0001`
  in wrong refused.
- M4 (R2) `report_use` asking the policy about the ids, not their sources: `note-0001` refused.
- M5 (R2) `resolve_sources` refusing by the translated source: the reason names
  `/root/agent-b/note-0001.md`.
- M6 (R2) `resolve_sources` ignoring aliases: `note-0001` refused.
- M7 (R3) `files_named` taking the first of two sources: `agent-b/note-0001` resolved.
M3 and M7 were first proved against this method as `sources_named`; it was renamed `files_named`
and made to return the file beside the source for `recall_report_stale`, and both were run again
on it the same day, red in the same assertions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from recall.lite.assistant_memos import MemoFolder
from recall_mcp import use_reports_api
from recall_mcp.assistant import AssistantMemos
from recall_mcp.use_reports_api import UseReportRefused, report_use
from tests.test_assistant_surface import Call, _published, _serve_both
from tests.test_assistant_surface import envs as envs  # noqa: F401  # the fixture, reused


@pytest.fixture
def agent_envs(envs: tuple[dict[str, str], dict[str, str]], tmp_path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Agent A's assistant server, serving `recall_report_use` beside the assistant tools."""
    assistant, plain = envs
    return {
        **assistant,
        "RECALL_ASSISTANT_MEMOS": str(tmp_path / "agent-a"),
        "RECALL_MCP_TOOLS": "assistant,recall_report_use",
    }, plain


def test_report_use_accepts_the_memo_ids_search_returned(
    agent_envs: tuple[dict[str, str], dict[str, str]], tmp_path: Path
) -> None:
    """R1."""
    other = MemoFolder(tmp_path / "agent-b", index_root=tmp_path)

    async def body(a: Call, p: Call, state: dict[str, Any]) -> dict[str, Any]:
        assert other.write("The user drink tea at the office.") == {"memo_id": "note-0001"}
        await p("recall_index", path=str(tmp_path / "agent-b"))
        assert json.loads(await a("recall_remember", text="The user drink tea, every morning.")) == {"memo_id": "note-0001"}
        named = [h["memo_id"] for h in json.loads(await a("recall_search", query="user drink tea", k=10))["hits"]]
        assert {"note-0001", "agent-b/note-0001"} <= set(named), named
        reported = await a(
            "recall_report_use", task="Answer what the user drinks.", effect="misled",
            used=["note-0001"], wrong=["agent-b/note-0001"],
        )
        assert not reported.startswith("RAISED"), f"search's own ids were refused: {reported}"
        unknown = await a("recall_report_use", task="Answer what the user drinks.", effect="helped", used=["note-0099"])
        assert unknown.startswith("RAISED") and "note-0099" in unknown, unknown
        return dict(json.loads(reported))

    result = _serve_both(agent_envs, body)
    assert [Path(s).resolve() for s in result["used"]] == [(tmp_path / "agent-a" / "note-0001.md").resolve()], result
    assert [Path(s).resolve() for s in result["wrong"]] == [(tmp_path / "agent-b" / "note-0001.md").resolve()], result
    description = _published(agent_envs[0])["recall_report_use"]["description"]
    assert "memo_id" in description, description


class _Ledger:
    tenant = "memory"

    def append_audit_event(self, event_type: str, payload: dict[str, Any], *, actor: str = "serving", **_: Any) -> str:
        return "evt-1"


def test_an_alias_is_authorized_as_its_source_and_refused_by_its_own_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """R2."""
    readable = {"/root/agent-a/note-0001.md"}
    aliases = {"note-0001": "/root/agent-a/note-0001.md", "agent-b/note-0001": "/root/agent-b/note-0001.md"}

    def authorized(store: Any, names: Any, policy: Any, context: Any, **_: Any) -> dict[str, list[Any]]:
        return {name: [] for name in names if name in readable}

    monkeypatch.setattr(use_reports_api, "authorized_report_rows", authorized)
    policy: Any = object()
    result = report_use(
        _Ledger(), task="Answer what the user drinks.", effect="helped", used=["note-0001"],
        security_policy=policy, access_context=policy, aliases=aliases,
    )
    assert result["used"] == ["/root/agent-a/note-0001.md"], result
    with pytest.raises(UseReportRefused) as refused:
        report_use(
            _Ledger(), task="Answer what the user drinks.", effect="misled", wrong=["agent-b/note-0001"],
            security_policy=policy, access_context=policy, aliases=aliases,
        )
    reason = str(refused.value)
    assert "'agent-b/note-0001'" in reason and "/root/agent-b" not in reason, reason


class _Rows:
    """Stored file names: one kept under two absolute sources (the folder indexed from two paths)."""

    rows = {
        "agent-a/note-0001.md": ["/one/agent-a/note-0001.md"],
        "agent-b/note-0001.md": ["/one/agent-b/note-0001.md", "/two/agent-b/note-0001.md"],
    }

    def sources_for_identifiers(self, identifiers: list[str]) -> dict[str, list[str]]:
        return {key: self.rows[key] for key in identifiers if key in self.rows}


def test_an_id_naming_two_sources_is_left_unresolved(tmp_path: Path) -> None:
    """R3."""
    memos = AssistantMemos(tmp_path / "agent-a", index_root=tmp_path)
    named = memos.files_named(cast(Any, _Rows()), ["note-0001.md", "agent-b/note-0001", "note-0099"])
    assert named == {"note-0001.md": ("agent-a/note-0001.md", "/one/agent-a/note-0001.md")}, named
