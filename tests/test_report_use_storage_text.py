"""DAT-002: report context must survive PostgreSQL JSONB storage.

Baseline red: NUL and lone surrogates cause a storage exception instead of a
recorded report. The assertion inspects the durable row after sanitization.
"""
from __future__ import annotations

import psycopg
import pytest

from recall_mcp.use_reports_api import report_use
from tests.conftest import requires_db


@requires_db
@pytest.mark.parametrize("field", ["task", "query", "note"])
@pytest.mark.parametrize("bad", ["\x00", "\ud800", "\udfff"], ids=["nul", "high-surrogate", "low-surrogate"])
def test_report_context_is_storable_and_preserves_valid_unicode(make_store, field: str, bad: str) -> None:
    store = make_store(8)
    text = "lead" + bad + "middle\U0001f9e0end"
    args = {"task": "check a decision", "query": "query", "note": "note", field: text}
    error = None
    result = None
    try:
        result = report_use(store, effect="no_difference", **args)
    except (psycopg.Error, UnicodeError) as exc:
        error = exc
    assert error is None, "report context failed PostgreSQL storage"
    assert result is not None and result["recorded"]
    with store._connect() as conn:
        row = conn.execute("select payload from recall_audit_events where tenant_id=%s and event_id=%s",
                           (store.tenant, result["event_id"])).fetchone()
    assert row is not None
    assert row[0][field] == "lead\ufffdmiddle\U0001f9e0end"
