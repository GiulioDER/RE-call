"""The MCP side of agent use reports: what memory did for one task, in the agent's own words.

`recall_report_use` appends one `use_report` row to the tenant's audit ledger
(`recall_audit_events`), the same append-only table the decision ledger writes each search to, so
a reader sees what was retrieved and what the agent said about it side by side. It changes no
memo, no verdict and no ranking: a report is testimony, counted on the dashboard's Control page
and labelled there as agent-reported.

Every memory named must exist in the tenant's served generation. A report about a source the
corpus does not hold is refused rather than stored, so a mistyped or invented name cannot inflate
a memory's counts. A name may be the stored source exactly or, when that is unambiguous, its file
name, which is how search results usually show it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

#: The `event_type` of a report in `recall_audit_events`; the dashboard reads it by this name.
USE_REPORT_EVENT = "use_report"
#: What the memory did for the task, as the agent judges it.
EFFECTS = ("helped", "no_difference", "misled")
MAX_SOURCES = 20
#: Context for a reader, not evidence; bounded so a report cannot store arbitrary text.
MAX_TASK_CHARS = 2000
MAX_NOTE_CHARS = 1000
MAX_QUERY_CHARS = 1000


class UseReportRefused(ValueError):
    """The report was not recorded; the message says what to change."""


def _basename(source: str) -> str:
    return source.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def resolve_sources(names: Sequence[str], known: set[str], field: str) -> list[str]:
    if isinstance(names, str) or not isinstance(names, Sequence):
        raise UseReportRefused(f"{field} must be a list of source names")
    if len(names) > MAX_SOURCES:
        raise UseReportRefused(f"{field} names {len(names)} memories; report at most {MAX_SOURCES}")
    by_name: dict[str, list[str]] = {}
    for source in known:
        by_name.setdefault(_basename(source), []).append(source)
    resolved: list[str] = []
    for raw in names:
        name = str(raw).strip()
        if not name:
            raise UseReportRefused(f"{field} contains an empty name")
        if name in known:
            source = name
        else:
            matches = by_name.get(_basename(name), [])
            if not matches:
                raise UseReportRefused(f"{name!r} in {field} is not a memory in this corpus; name it as search returned it")
            if len(matches) > 1:
                raise UseReportRefused(f"{name!r} in {field} matches {len(matches)} memories; give the full source")
            source = matches[0]
        if source not in resolved:
            resolved.append(source)
    return resolved


def report_use(
    store: Any,
    *,
    task: str,
    effect: str,
    used: Sequence[str] = (),
    wrong: Sequence[str] = (),
    task_succeeded: bool | None = None,
    query: str | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    """Record one report, or raise `UseReportRefused` with the reason."""
    task = (task or "").strip()
    if not task:
        raise UseReportRefused("task is required: say in a sentence what you were doing")
    if effect not in EFFECTS:
        raise UseReportRefused(f"effect must be one of {', '.join(EFFECTS)}")
    known = set(store.source_content_hashes())
    used_sources = resolve_sources(used, known, "used")
    wrong_sources = resolve_sources(wrong, known, "wrong")
    both = sorted(set(used_sources) & set(wrong_sources))
    if both:
        raise UseReportRefused(f"{both[0]!r} is listed as both used and wrong; pick one")
    if effect == "helped" and not used_sources:
        raise UseReportRefused("effect 'helped' needs at least one memory in used: which one helped?")
    if effect == "misled" and not wrong_sources:
        raise UseReportRefused("effect 'misled' needs at least one memory in wrong: which one misled?")
    payload = {
        "task": task[:MAX_TASK_CHARS],
        "effect": effect,
        "used": used_sources,
        "wrong": wrong_sources,
        "task_succeeded": task_succeeded if isinstance(task_succeeded, bool) else None,
        "query": (query or "").strip()[:MAX_QUERY_CHARS] or None,
        "note": (note or "").strip()[:MAX_NOTE_CHARS] or None,
    }
    event_id = store.append_audit_event(USE_REPORT_EVENT, payload, actor="agent-report")
    return {
        "recorded": True,
        "event_id": event_id,
        "effect": effect,
        "used": used_sources,
        "wrong": wrong_sources,
        "message": "Recorded. It changes no memory and no search result.",
    }
