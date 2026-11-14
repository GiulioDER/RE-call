"""`recall_report_stale` on the plain lite server records a name copied from search in both places.

Without RECALL_ASSISTANT_MEMOS, `recall_search` on a lite server shows each hit's `source` as its
file under the index root (`notes/old.md`), while the store keeps the absolute path. A report goes
to two places that read a name differently: the review queue (`recall.stale_reports.resolve_memo`,
a path relative to the root, absolute paths refused) and the tenant's audit ledger
(`recall_mcp.use_reports_api.resolve_sources`, the stored source or an unambiguous bare file
name). So the name search showed reached the queue only, the ledger refusing it, and a bare file
name reached the ledger only, under a different `claim_key`.

Invariant and the failure it catches:
- P1 with no source policy, the two names copied from a plain search hit are recorded in both the
  queue and the ledger, and the ledger row names the stored sources; a path naming no memo is
  still refused by its own name: `recorded_in` is `['sidecar']` with `audit_ledger: not recorded`.
- P2 under a source policy, a memo the caller may not read is refused with the same reason as a
  path naming no memo, each named as the caller wrote it: names replaced by their stored sources,
  so the reason shows which paths exist.
- P3 under a source policy, the names a policy-scoped search showed are recorded in the ledger,
  and a bare file name still resolves among the memos the caller may read, even when a hidden memo
  carries that exact file name: the paths refused outright, or the bare name resolved unscoped to
  the hidden memo, so its refusal shows that the hidden memo exists.

Red proof, 2026-11-14, on this workstation (single file, light check). At aa4997333, the commit
before this fix, P1 fails in its intended assertion: `recorded_in` is `['sidecar']`, with
"audit_ledger: not recorded: 'notes/old.md' in stale_source is not a memory in this corpus".
P3 fails there too, in its intended assertion: "the name 'notes/old.md' was refused: ...
'notes/old.md' in stale_source is not a memory in this corpus". P2 passes there (nothing was
resolved yet), so it is proved by mutating the fix. Each mutation alone, its test run alone, the
file restored byte for byte (sha256 checked), then the file green:
- N1 (P2, P1) the server passing each name's stored source as the name instead of as an alias: the
  hidden memo's reason names its absolute path, the missing one's `'notes/gone.md'`; P1's queue
  refuses the absolute path.
- N2 (P3) the bare-name exclusion dropped (every name resolved): the bare `old.md` resolves to the
  hidden root memo and is refused, "'old.md' in stale_source is not a memory in this corpus".
Both were run again the same day after the resolution moved into
`recall_mcp.server._lite_paths_named`, shared with `recall_report_use`, with the same failures.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from recall.dashboard.lite_db import stale_reports
from recall.security_policy import SourceRule, SourceSecurityPolicy
from tests.test_assistant_surface import Call, _serve_both
from tests.test_assistant_surface import envs as envs  # noqa: F401  # the fixture, reused

OLD = "The user drinks green tea every single morning."
NEW = "The user switched to black coffee every single morning."


def _notes(root: Path) -> Path:
    notes = root / "notes"
    notes.mkdir()
    (notes / "old.md").write_text(OLD + "\n", encoding="utf-8")
    (notes / "new.md").write_text(NEW + "\n", encoding="utf-8")
    return notes


async def _shown(call: Call) -> dict[str, str]:
    """`{file name: source}` for the notes hits, exactly as `recall_search` showed them."""
    hits = json.loads(await call("recall_search", query="user drinks green tea morning", k=10))["hits"]
    return {Path(h["source"]).name: h["source"] for h in hits if "notes" in h["source"]}


def test_a_name_copied_from_plain_search_is_recorded_in_both_places(
    envs: tuple[dict[str, str], dict[str, str]], tmp_path: Path
) -> None:
    """P1."""
    plain = envs[1]
    notes = _notes(tmp_path)

    async def body(_: Call, p: Call, state: dict[str, Any]) -> dict[str, Any]:
        await p("recall_index", path=str(notes))
        shown = await _shown(p)
        assert shown == {"old.md": "notes/old.md", "new.md": "notes/new.md"}, shown
        out = await p(
            "recall_report_stale", stale_source=shown["old.md"], replacing_source=shown["new.md"],
            stale_quote=OLD, current_quote=NEW, task="Answer what the user drinks.",
        )
        assert not out.startswith("RAISED"), out
        missing = await p(
            "recall_report_stale", stale_source="notes/gone.md", replacing_source=shown["new.md"],
            stale_quote=OLD, current_quote=NEW,
        )
        assert missing.startswith("RAISED") and "'notes/gone.md'" in missing, missing
        result: dict[str, Any] = json.loads(out)
        return result

    result = _serve_both((plain, plain), body)
    assert result["recorded_in"] == ["sidecar", "audit_ledger"], result
    assert "audit_ledger" not in result, result
    assert (result["stale_source"], result["replacing_source"]) == ("notes/old.md", "notes/new.md"), result
    rows = stale_reports(plain["RECALL_DSN"], ("default",), 10)
    assert [(r["stale_source"], r["replacing_source"]) for r in rows] == [
        (str(notes / "old.md"), str(notes / "new.md"))
    ], rows


def test_under_a_source_policy_a_hidden_memo_is_refused_as_a_missing_one(
    envs: tuple[dict[str, str], dict[str, str]], tmp_path: Path
) -> None:
    """P2."""
    plain = envs[1]
    notes = _notes(tmp_path)

    async def body(a: Call, p: Call, state: dict[str, Any]) -> dict[str, str]:
        await p("recall_index", path=str(notes))
        # Rows indexed before the policy carry no digest of it, so every memo is hidden.
        state["source_security_policy"] = SourceSecurityPolicy((SourceRule("notes"),))
        reasons = {}
        for stale in ("notes/old.md", "notes/gone.md"):
            out = await a(
                "recall_report_stale", stale_source=stale, replacing_source="notes/new.md",
                stale_quote=OLD, current_quote=NEW,
            )
            assert out.startswith("RAISED ToolError: "), out
            reasons[stale] = json.loads(out.removeprefix("RAISED ToolError: "))["reason"]
        return reasons

    reasons = _serve_both((plain, plain), body)
    hidden, missing = reasons["notes/old.md"], reasons["notes/gone.md"]
    assert hidden.replace("notes/old.md", "notes/gone.md") == missing, reasons
    assert "'notes/old.md' " in hidden, reasons


def test_under_a_source_policy_search_names_reach_the_ledger_and_a_bare_name_stays_scoped(
    envs: tuple[dict[str, str], dict[str, str]], tmp_path: Path
) -> None:
    """P3."""
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
        for stale in (shown["old.md"], "old.md"):
            out = await a(
                "recall_report_stale", stale_source=stale, replacing_source=shown["new.md"],
                stale_quote=OLD, current_quote=NEW,
            )
            assert not out.startswith("RAISED"), f"the name {stale!r} was refused: {out}"
            results.append(json.loads(out))
        return results

    for result in _serve_both((plain, plain), body):
        assert result["recorded_in"] == ["audit_ledger"], result
        assert (result["stale_source"], result["replacing_source"]) == (
            str(notes / "old.md"), str(notes / "new.md")
        ), result
