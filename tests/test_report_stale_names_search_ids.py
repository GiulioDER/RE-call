"""`recall_report_stale` on the assistant surface accepts the ids `recall_search` named the memos by.

With RECALL_ASSISTANT_MEMOS set, `recall_search` names a hit by its `memo_id` (`note-0001` in the
agent's own folder, `agent-b/note-0001` in another folder under the index root) and never by its
source. A stale report goes to two places that read a name differently: the review queue under
the index root (`recall.stale_reports.resolve_memo`, a path relative to the root) and the tenant's
audit ledger (`recall_mcp.use_reports_api.resolve_sources`, a stored source). A deployment serving
`recall_report_stale` beside the assistant tools must take search's ids into both, and its
description must say which id to pass.

Invariant and the failure it catches:
- S1 each `memo_id` a compact search returned, own folder and qualified, is recorded in both the
  queue and the ledger under the memo's path under the index root; an id naming no memo is still
  refused by its own name; the description names `memo_id`: the ids refused outright (neither
  place reads a memo id), or recorded in one place only (a root-relative name the ledger refuses,
  a file name the queue cannot find).
- S2 under a source policy, a memo the caller may not read is refused with the same reason as an
  id naming no memo, each named as the caller wrote it: ids translated to file names whenever
  they name a stored memo, so the reason shows which ids exist.
- S3 under a source policy an id is authorized as the source it names and recorded under it:
  the policy asked about the id itself, so a readable memo is refused.

Red proof, 2026-11-13, on this workstation (single file, light check). At 94898b19d, which carries
the `recall_report_use` fix and not this one, S1 fails in its intended assertion: "search's own id
'note-0001' was refused: ... 'note-0001' in stale_source is not a memory in this corpus". S2
passes there (nothing was translated yet) and S3 calls an argument it lacks (`aliases`), so both
are proved by mutating the fix. Each mutation alone, its test run alone, the file restored byte
for byte (sha256 checked), then the file green:
- N1 (S2) the server translating ids for the queue under a policy too: the hidden memo's reason
  names `'agent-b/note-0001.md'`, the missing one's `'agent-b/note-0099'`.
- N2 (S1) the server passing no aliases: `recorded_in` is `['sidecar']`.
- N3 (S1) the server passing the ids untranslated: `recorded_in` is `['audit_ledger']`.
- N4 (S1) the plain description kept: the description assertion (no `memo_id`).
- N5 (S3) `_record_served` asking the policy about the ids: `'note-0001' in stale_source is not a
  memory in this corpus`.
- N6 (S1) `_record_served` resolving the stale source without aliases: `recorded_in` is
  `['sidecar']`.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from recall.lite.assistant_memos import MemoFolder
from recall.security_policy import SourceRule, SourceSecurityPolicy
from recall_mcp import stale_reports_api
from recall_mcp.stale_reports_api import report_stale
from tests.test_assistant_surface import Call, _published, _serve_both
from tests.test_assistant_surface import envs as envs  # noqa: F401  # the fixture, reused

OLD = "The user drinks green tea every single morning."
NEW = "The user switched to black coffee every single morning."
OTHER = "The user drinks green tea at the office desk."


@pytest.fixture
def agent_envs(envs: tuple[dict[str, str], dict[str, str]], tmp_path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Agent A's assistant server, serving `recall_report_stale` beside the assistant tools."""
    assistant, plain = envs
    return {
        **assistant,
        "RECALL_ASSISTANT_MEMOS": str(tmp_path / "agent-a"),
        "RECALL_MCP_TOOLS": "assistant,recall_report_stale",
    }, plain


def test_report_stale_records_the_memo_ids_search_returned_in_both_places(
    agent_envs: tuple[dict[str, str], dict[str, str]], tmp_path: Path
) -> None:
    """S1."""
    other = MemoFolder(tmp_path / "agent-b", index_root=tmp_path)

    async def body(a: Call, p: Call, state: dict[str, Any]) -> list[str]:
        assert other.write(OTHER) == {"memo_id": "note-0001"}
        await p("recall_index", path=str(tmp_path / "agent-b"))
        assert json.loads(await a("recall_remember", text=OLD)) == {"memo_id": "note-0001"}
        assert json.loads(await a("recall_remember", text=NEW)) == {"memo_id": "note-0002"}
        named = [h["memo_id"] for h in json.loads(await a("recall_search", query="user drinks green tea", k=10))["hits"]]
        assert {"note-0001", "note-0002", "agent-b/note-0001"} <= set(named), named
        results = []
        for stale, quote in (("note-0001", OLD), ("agent-b/note-0001", OTHER)):
            out = await a(
                "recall_report_stale", stale_source=stale, replacing_source="note-0002",
                stale_quote=quote, current_quote=NEW, task="Answer what the user drinks.",
            )
            assert not out.startswith("RAISED"), f"search's own id {stale!r} was refused: {out}"
            result = json.loads(out)
            assert result["recorded_in"] == ["sidecar", "audit_ledger"], result
            assert "audit_ledger" not in result, result
            results.append(result["stale_source"])
            assert result["replacing_source"] == "agent-a/note-0002.md", result
        unknown = await a(
            "recall_report_stale", stale_source="note-0099", replacing_source="note-0002",
            stale_quote=OLD, current_quote=NEW,
        )
        assert unknown.startswith("RAISED") and "note-0099" in unknown, unknown
        return results

    assert _serve_both(agent_envs, body) == ["agent-a/note-0001.md", "agent-b/note-0001.md"]
    description = _published(agent_envs[0])["recall_report_stale"]["description"]
    assert "memo_id" in description, description


def test_under_a_source_policy_a_hidden_memo_is_refused_as_a_missing_one(
    agent_envs: tuple[dict[str, str], dict[str, str]], tmp_path: Path
) -> None:
    """S2."""
    other = MemoFolder(tmp_path / "agent-b", index_root=tmp_path)

    async def body(a: Call, p: Call, state: dict[str, Any]) -> dict[str, str]:
        assert other.write(OTHER) == {"memo_id": "note-0001"}
        await p("recall_index", path=str(tmp_path / "agent-b"))
        assert json.loads(await a("recall_remember", text=NEW)) == {"memo_id": "note-0001"}
        # Rows indexed before the policy carry no digest of it, so every memo is hidden.
        state["source_security_policy"] = SourceSecurityPolicy((SourceRule("agent-a"),))
        reasons = {}
        for stale in ("agent-b/note-0001", "agent-b/note-0099"):
            out = await a(
                "recall_report_stale", stale_source=stale, replacing_source="note-0001",
                stale_quote=OTHER, current_quote=NEW,
            )
            assert out.startswith("RAISED ToolError: "), out
            reasons[stale] = json.loads(out.removeprefix("RAISED ToolError: "))["reason"]
        return reasons

    reasons = _serve_both(agent_envs, body)
    hidden, missing = reasons["agent-b/note-0001"], reasons["agent-b/note-0099"]
    assert hidden.replace("note-0001", "note-0099") == missing, reasons
    assert "'agent-b/note-0001' " in hidden, reasons


class _Ledger:
    tenant = "memory"

    def source_content_hashes(self) -> dict[str, str]:
        return {}

    def append_audit_event(self, event_type: str, payload: dict[str, Any], **_: Any) -> str:
        return "evt-1"


def test_under_a_source_policy_an_id_is_authorized_as_its_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """S3."""
    texts = {"/root/agent-a/note-0001.md": OLD, "/root/agent-a/note-0002.md": NEW}

    def authorized(store: Any, names: Any, policy: Any, context: Any, *, include_text: bool, **_: Any) -> Any:
        return {n: [SimpleNamespace(text=texts[n], metadata={}, id=n)] if include_text else [] for n in names if n in texts}

    monkeypatch.setattr(stale_reports_api, "authorized_report_rows", authorized)
    policy: Any = object()
    result = report_stale(
        _Ledger(), stale_source="note-0001", replacing_source="note-0002", stale_quote=OLD, current_quote=NEW,
        task=None, env={}, sidecar=False, security_policy=policy, access_context=policy,
        aliases={"note-0001": "/root/agent-a/note-0001.md", "note-0002": "/root/agent-a/note-0002.md"},
    )
    assert (result["recorded_in"], result["stale_source"]) == (["audit_ledger"], "/root/agent-a/note-0001.md"), result
