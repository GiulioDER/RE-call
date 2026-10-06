"""The MCP side of agent stale reports: record one report, change nothing else.

`recall_report_stale` records a report in up to two places, and never writes a memo, never
declares `supersedes:`, and never changes what search returns; a person accepts or rejects the
report later (`recall rewrite`, or the dashboard). That keeps the rule
`tests/test_mcp_rewrite_plan.py` states for this package, that the MCP client, being the model,
cannot be the one who edits a document.

* **The sidecar queue** (`recall.stale_reports`, `<RECALL_INDEX_ROOT>/.recall/reports.sqlite3`),
  when both memos are files under `RECALL_INDEX_ROOT` (default: the server's working directory).
  The quotes are checked against those files. This is the local-only path, and it works with no
  database.
* **The tenant's audit ledger**, as one `stale_report` row in `recall_audit_events`, when the
  store is database-backed and both memos are in its served generation. The quotes are checked
  against the text search served. This is the row a dashboard on another machine reads over its
  read-only connection, which the sidecar cannot reach when the server runs on a remote host.

A report is refused only when neither place takes it. The ledger row is testimony, like
`recall_report_use`'s: the dashboard maps its names to the memo files it has and checks both
quotes against those files again before a person sees the claim.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from recall.rewrite import claim_key
from recall.stale_reports import (
    MIN_QUOTE_CHARS,
    StaleReportQueue,
    StaleReportRefused,
    default_queue_path,
    grounded,
    resolve_memo,
)
from recall_mcp.use_reports_api import UseReportRefused, resolve_sources

#: The `event_type` of a report in `recall_audit_events`; the dashboard reads it by this name.
STALE_REPORT_EVENT = "stale_report"
#: Task text is context for the reviewer, not evidence; it is bounded so a report cannot become a
#: channel for storing arbitrary amounts of text in the sidecar or the ledger.
MAX_TASK_CHARS = 2000
#: A quote is a sentence or two of evidence, not a copy of the memo.
MAX_QUOTE_CHARS = 1000


def _database_backed(store: Any) -> bool:
    return callable(getattr(store, "append_audit_event", None)) and callable(getattr(store, "source_content_hashes", None))


def served_text(store: Any, source: str) -> str | None:
    """The text search serves for one source, or None when this store cannot say."""
    reader = getattr(store, "source_chunk_texts", None)
    try:
        if callable(reader):
            texts = list(reader(source))
        else:
            texts = [chunk.text for chunk in store.chunks_for_source(source)]
    except (NotImplementedError, AttributeError):
        return None
    return "\n".join(texts)


def _record_served(
    store: Any,
    *,
    stale_source: str,
    replacing_source: str,
    stale_quote: str,
    current_quote: str,
    client: str,
    task: str | None,
    checked_locally: bool,
) -> tuple[str, str, str]:
    """Append one `stale_report` row, or refuse; returns the event id and the two served names."""
    known = set(store.source_content_hashes())
    try:
        stale = resolve_sources([stale_source], known, "stale_source")[0]
        replacing = resolve_sources([replacing_source], known, "replacing_source")[0]
    except UseReportRefused as exc:
        raise StaleReportRefused(str(exc)) from exc
    if stale == replacing:
        raise StaleReportRefused("a memo cannot supersede itself")
    for field, quote, source in (("stale_quote", stale_quote, stale), ("current_quote", current_quote, replacing)):
        text = served_text(store, source)
        if text is None:
            if checked_locally:
                continue
            raise StaleReportRefused(f"{field} cannot be checked against the served text of {source} on this server")
        if not grounded(quote, text):
            raise StaleReportRefused(
                f"{field} is not verbatim in {source} as search served it "
                f"(at least {MIN_QUOTE_CHARS} characters, copied exactly)"
            )
    payload = {
        "stale_source": stale,
        "replacing_source": replacing,
        "stale_quote": stale_quote,
        "current_quote": current_quote,
        "claim_key": claim_key("supersedes", stale, replacing),
        "client": client,
        "task": task,
    }
    event_id = store.append_audit_event(STALE_REPORT_EVENT, payload, actor="agent-report")
    return str(event_id), stale, replacing


def report_stale(
    store: Any,
    *,
    stale_source: str,
    replacing_source: str,
    stale_quote: str,
    current_quote: str,
    task: str | None,
    env: Mapping[str, str] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Record one report, or raise `StaleReportRefused` with the reason.

    `store` is the one `_require` authorised. It names the reporting tenant, and when it is
    database-backed it is also where the ledger row goes.
    """
    for field, quote in (("stale_quote", stale_quote), ("current_quote", current_quote)):
        if isinstance(quote, str) and len(quote) > MAX_QUOTE_CHARS:
            raise StaleReportRefused(f"{field} is {len(quote)} characters; quote at most {MAX_QUOTE_CHARS}")
    values = os.environ if env is None else env
    root = Path(values.get("RECALL_INDEX_ROOT", ".")).resolve()
    client = "mcp:" + str(getattr(store, "tenant", None) or "local")
    task = task[:MAX_TASK_CHARS] if task else None

    # Resolve before opening the queue: opening it creates `.recall/` under the root, and on a
    # server whose root holds no memos (a serving checkout) that would only leave litter behind.
    local_refusal: StaleReportRefused | None = None
    try:
        resolve_memo(root, stale_source)
        resolve_memo(root, replacing_source)
    except StaleReportRefused as exc:
        local_refusal = exc
    report = None
    if local_refusal is None:
        with StaleReportQueue(default_queue_path(root)) as queue:
            report = queue.submit(
                root,
                stale_source=stale_source,
                replacing_source=replacing_source,
                stale_quote=stale_quote,
                current_quote=current_quote,
                client=client,
                task=task,
                reported_at=now or datetime.now(UTC),
            )

    event_id: str | None = None
    served: tuple[str, str] | None = None
    ledger_refusal: str | None = None
    if _database_backed(store):
        try:
            event_id, served_stale, served_replacing = _record_served(
                store,
                stale_source=stale_source,
                replacing_source=replacing_source,
                stale_quote=stale_quote,
                current_quote=current_quote,
                client=client,
                task=task,
                checked_locally=report is not None,
            )
            served = (served_stale, served_replacing)
        except StaleReportRefused as exc:
            if report is None:
                raise
            ledger_refusal = str(exc)
    if report is None and served is None:
        assert local_refusal is not None
        raise local_refusal

    recorded = (["sidecar"] if report is not None else []) + (["audit_ledger"] if served is not None else [])
    if report is not None:
        stale_name, replacing_name, key, count = report.stale_source, report.replacing_source, report.claim_key, report.report_count
    else:
        assert served is not None
        stale_name, replacing_name = served
        key, count = claim_key("supersedes", stale_name, replacing_name), None
    result: dict[str, Any] = {
        "queued": True,
        "recorded_in": recorded,
        "claim_key": key,
        "stale_source": stale_name,
        "replacing_source": replacing_name,
        "report_count": count,
        "status": "pending",
        "message": "Queued for a person to review. Nothing in memory changed.",
    }
    if event_id is not None:
        result["event_id"] = event_id
    if ledger_refusal is not None:
        result["audit_ledger"] = f"not recorded: {ledger_refusal}"
    return result
