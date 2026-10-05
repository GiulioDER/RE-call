"""The MCP side of agent stale reports: queue one report, change nothing else.

`recall_report_stale` writes one row to the corpus's report queue (`recall.stale_reports`), in the
`.recall` sidecar beside the rejection ledger. It never writes a memo, never declares
`supersedes:`, and never changes what search returns; a person accepts or rejects the report later
(`recall rewrite`, or the dashboard). That keeps the rule `tests/test_mcp_rewrite_plan.py` states
for this package, that the MCP client, being the model, cannot be the one who edits a document.

Like `recall_index`, it works on the files under `RECALL_INDEX_ROOT` (default: the server's
working directory), because the quotes are checked against the memo files a reviewer will read.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from recall.stale_reports import StaleReportQueue, default_queue_path

#: Task text is context for the reviewer, not evidence; it is bounded so a report cannot become a
#: channel for storing arbitrary amounts of text in the sidecar.
MAX_TASK_CHARS = 2000


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
    """Queue one report, or raise `StaleReportRefused` with the reason.

    `store` is the one `_require` authorised; it is used only to name the reporting tenant, and the
    report itself never reaches the database.
    """
    values = os.environ if env is None else env
    root = Path(values.get("RECALL_INDEX_ROOT", ".")).resolve()
    with StaleReportQueue(default_queue_path(root)) as queue:
        report = queue.submit(
            root,
            stale_source=stale_source,
            replacing_source=replacing_source,
            stale_quote=stale_quote,
            current_quote=current_quote,
            client="mcp:" + str(getattr(store, "tenant", None) or "local"),
            task=task[:MAX_TASK_CHARS] if task else None,
            reported_at=now or datetime.now(UTC),
        )
    return {
        "queued": True,
        "claim_key": report.claim_key,
        "stale_source": report.stale_source,
        "replacing_source": report.replacing_source,
        "report_count": report.report_count,
        "status": report.status,
        "message": "Queued for a person to review. Nothing in memory changed.",
    }
