"""`recall_report_use` on the plain lite server accepts the names search showed.

Without RECALL_ASSISTANT_MEMOS, `recall_search` on a lite server shows each hit's `source` as its
file under the index root (`notes/old.md`), while the store keeps the absolute path, and the
audit ledger (`recall_mcp.use_reports_api.resolve_sources`) accepts only the stored source or an
unambiguous bare file name. So a report naming a memo exactly as search showed it was refused,
although the tool's description says to name each memory "by its source as search returned it".

Invariant and the failure it catches:
- U1 with no source policy, a report naming the memos as plain search showed them is recorded,
  under their stored sources; a path naming no memo is still refused by its own name: the
  report refused outright ("'notes/old.md' in used is not a memory in this corpus").
- U2 under a source policy, a memo the caller may not read is refused with the same reason as a
  path naming no memo, each named as the caller wrote it: names replaced by their stored sources,
  so the reason shows which paths exist.
- U3 under a source policy, the names a policy-scoped search showed are recorded, and a bare file
  name still resolves among the memos the caller may read, even when a hidden memo carries that
  exact file name: the bare name resolved unscoped to the hidden memo, so its refusal shows that
  the hidden memo exists.

Red proof, 2026-11-14, on this workstation (single file, light check). At cb20ec39f, which carries
the `recall_report_stale` fix and not this one, U1 and U3 fail in their intended assertion: "the
names search showed were refused: ... 'notes/new.md' in used is not a memory in this corpus", and
for U3 the same reason under "the name 'notes/old.md' was refused". U2 passes there (nothing was
resolved yet), so it is proved by mutating the fix. Each mutation alone, its test run alone,
`recall_mcp/server.py` restored byte for byte (sha256 checked), then the file green:
- M1 (U3) the bare-name exclusion in `recall_mcp.server._lite_paths_named` dropped: "'old.md' in
  wrong is not a memory in this corpus".
- M2 (U2) the server passing each name's stored source in `used` and `wrong` instead of as an
  alias: the hidden memo's reason names its absolute path, the missing one's `'notes/gone.md'`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from recall.security_policy import SourceRule, SourceSecurityPolicy
from tests.test_assistant_surface import Call, _serve_both
from tests.test_assistant_surface import envs as envs  # noqa: F401  # the fixture, reused
from tests.test_report_stale_plain_lite import OLD, _notes, _shown

TASK = "Answer what the user drinks."


def test_names_copied_from_plain_search_are_recorded(
    envs: tuple[dict[str, str], dict[str, str]], tmp_path: Path
) -> None:
    """U1."""
    plain = envs[1]
    notes = _notes(tmp_path)

    async def body(_: Call, p: Call, state: dict[str, Any]) -> dict[str, Any]:
        await p("recall_index", path=str(notes))
        shown = await _shown(p)
        assert shown == {"old.md": "notes/old.md", "new.md": "notes/new.md"}, shown
        out = await p("recall_report_use", task=TASK, effect="misled", used=[shown["new.md"]], wrong=[shown["old.md"]])
        assert not out.startswith("RAISED"), f"the names search showed were refused: {out}"
        missing = await p("recall_report_use", task=TASK, effect="helped", used=["notes/gone.md"])
        assert missing.startswith("RAISED") and "'notes/gone.md'" in missing, missing
        result: dict[str, Any] = json.loads(out)
        return result

    result = _serve_both((plain, plain), body)
    assert (result["used"], result["wrong"]) == ([str(notes / "new.md")], [str(notes / "old.md")]), result


def test_under_a_source_policy_a_hidden_memo_is_refused_as_a_missing_one(
    envs: tuple[dict[str, str], dict[str, str]], tmp_path: Path
) -> None:
    """U2."""
    plain = envs[1]
    notes = _notes(tmp_path)

    async def body(a: Call, p: Call, state: dict[str, Any]) -> dict[str, str]:
        await p("recall_index", path=str(notes))
        # Rows indexed before the policy carry no digest of it, so every memo is hidden.
        state["source_security_policy"] = SourceSecurityPolicy((SourceRule("notes"),))
        reasons = {}
        for name in ("notes/old.md", "notes/gone.md"):
            out = await a("recall_report_use", task=TASK, effect="helped", used=[name])
            assert out.startswith("RAISED ToolError: "), out
            reasons[name] = json.loads(out.removeprefix("RAISED ToolError: "))["reason"]
        return reasons

    reasons = _serve_both((plain, plain), body)
    hidden, missing = reasons["notes/old.md"], reasons["notes/gone.md"]
    assert hidden.replace("notes/old.md", "notes/gone.md") == missing, reasons
    assert "'notes/old.md' " in hidden, reasons


def test_under_a_source_policy_search_names_are_recorded_and_a_bare_name_stays_scoped(
    envs: tuple[dict[str, str], dict[str, str]], tmp_path: Path
) -> None:
    """U3."""
    plain = envs[1]
    notes = _notes(tmp_path)
    (tmp_path / "old.md").write_text("A root copy. " + OLD + "\n", encoding="utf-8")

    async def body(a: Call, p: Call, state: dict[str, Any]) -> list[dict[str, Any]]:
        # Indexed with no policy, so it carries no digest and stays hidden from the caller.
        await p("recall_index", path=str(tmp_path / "old.md"))
        state["source_security_policy"] = SourceSecurityPolicy(
            (SourceRule("notes", purposes=frozenset({"retrieval", "indexing"})),)
        )
        await a("recall_index", path=str(notes))
        shown = await _shown(a)
        assert shown == {"old.md": "notes/old.md", "new.md": "notes/new.md"}, shown
        results = []
        for wrong in (shown["old.md"], "old.md"):
            out = await a("recall_report_use", task=TASK, effect="misled", used=[shown["new.md"]], wrong=[wrong])
            assert not out.startswith("RAISED"), f"the name {wrong!r} was refused: {out}"
            results.append(json.loads(out))
        return results

    for result in _serve_both((plain, plain), body):
        assert (result["used"], result["wrong"]) == ([str(notes / "new.md")], [str(notes / "old.md")]), result
