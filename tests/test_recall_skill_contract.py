"""The distributed RE-call skill stays focused while its full tool map remains complete."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLAUDE_SKILL = ROOT / "plugin" / "skills" / "re-call"
CODEX_SKILL = ROOT / "codex-plugin" / "skills" / "re-call"


def _tool_names() -> set[str]:
    source = (ROOT / "recall_mcp" / "server.py").read_text(encoding="utf-8")
    return set(re.findall(r'name="(recall_[a-z_]+)"', source))


def test_recall_skill_bundles_are_byte_identical() -> None:
    """Red proof receipt ``focused-skill-parity-01``.

    Before implementation the two main files happened to match, but no supporting reference
    existed. The assertion deliberately includes every file in the skill directory so adding a
    reference to only one bundle fails at the consumer boundary.
    """

    claude_files = {
        path.relative_to(CLAUDE_SKILL): path.read_bytes()
        for path in CLAUDE_SKILL.rglob("*")
        if path.is_file()
    }
    codex_files = {
        path.relative_to(CODEX_SKILL): path.read_bytes()
        for path in CODEX_SKILL.rglob("*")
        if path.is_file()
    }
    assert claude_files == codex_files


def test_default_read_path_is_focused_and_non_blocking() -> None:
    """Red proof receipt ``focused-skill-default-path-01``.

    The pre-change skill named search and evidence independently but did not require a selected
    source, did not cap the focused bundle, and used the false rule that current code always wins.
    """

    text = (CLAUDE_SKILL / "SKILL.md").read_text(encoding="utf-8")
    assert "call `recall_search` once" in text
    assert "that exact `source`, and `max_items=2`" in text
    assert "Code that has not implemented it yet is the work to do, not a conflict." in text
    assert "references/tool-routing.md" in text


def test_reference_routes_every_mcp_tool_exactly_once() -> None:
    """Red proof receipt ``focused-skill-tool-map-01``.

    No tool-routing reference existed before this test, so the first run fails on the missing
    file rather than passing against a partial hand-written list.
    """

    reference = (CLAUDE_SKILL / "references" / "tool-routing.md").read_text(encoding="utf-8")
    routed = re.findall(r"^\| `(recall_[a-z_]+)` \|", reference, flags=re.MULTILINE)
    assert len(routed) == len(set(routed)), "a tool appears in more than one routing row"
    assert set(routed) == _tool_names()
    assert len(routed) == 24


def test_the_skill_asks_for_one_honest_report_after_a_task() -> None:
    """Red proof receipt ``skill-end-of-task-report-01``, 2026-10-07.

    Invariant: the skill tells an agent that searched memory to report once, after the task,
    including when memory made no difference, and keeps the report out of the read path budget.
    Red: run against the skill as it stood at ``d21e2812`` (no report section), the first
    assertion failed; restored to this text, green.
    """

    text = (CLAUDE_SKILL / "SKILL.md").read_text(encoding="utf-8")
    assert "When a task that searched memory is finished, and `recall_report_use` is available, call it once." in text
    assert "Report `no_difference` as readily as `helped`" in text
    assert "Do not report a task that did not search memory, and do not report a task\ntwice." in text
    assert "The end-of-task report below is the one standing exception." in text
