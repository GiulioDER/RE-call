"""Reports must preserve source identity instead of accepting a matching path suffix.

FIX-012 proof plan, not an execution receipt:
The unknown-qualified-source tests must fail against the pre-fix resolver with
``DID NOT RAISE`` at the consumer's refusal assertion. Exact-source and bare-name
controls pass that baseline and need separate production mutations: bypass exact
matching for the former, and disable filename fallback for the latter. The parent
records the executed node IDs and outcomes in its proof receipt.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from recall.stale_reports import StaleReportRefused
from recall_mcp.stale_reports_api import report_stale
from recall_mcp.use_reports_api import UseReportRefused, report_use

SOURCE = "file:///memory/current/note.md"
REPLACEMENT = "file:///memory/current/replacement.md"
OLD_QUOTE = "The service still uses the original port."
NEW_QUOTE = "The service now uses the replacement port."


class ReportStore:
    tenant = "memory"

    def __init__(self, sources: dict[str, str]) -> None:
        self.sources = sources
        self.rows: list[tuple[str, dict[str, Any], str]] = []

    def source_content_hashes(self) -> dict[str, str]:
        return {source: "hash" for source in self.sources}

    def source_chunk_texts(self, source: str) -> list[str]:
        return [self.sources[source]]

    def append_audit_event(self, event_type: str, payload: dict[str, Any], *, actor: str, **_: Any) -> str:
        self.rows.append((event_type, payload, actor))
        return f"evt-{len(self.rows)}"


@pytest.mark.parametrize("field", ["used", "wrong"])
@pytest.mark.parametrize(
    "name",
    [
        pytest.param("wrong-root/note.md", id="relative-posix"),
        pytest.param(r"wrong-root\note.md", id="relative-windows"),
        pytest.param("/other/note.md", id="absolute-posix"),
        pytest.param(r"C:\other\note.md", id="absolute-windows"),
        pytest.param("file:///other/note.md", id="file-uri"),
        pytest.param("s3://other/note.md", id="remote-uri"),
        pytest.param("note.md/", id="trailing-posix-separator"),
        pytest.param("note.md\\", id="trailing-windows-separator"),
    ],
)
def test_use_report_refuses_unknown_qualified_source(name: str, field: str) -> None:
    """A qualified typo must not credit the known source sharing its filename."""
    store = ReportStore({SOURCE: OLD_QUOTE})
    with pytest.raises(UseReportRefused, match="not a memory in this corpus"):
        report_use(store, task="check the port", effect="no_difference", **{field: [name]})
    assert store.rows == [], "a refused qualified source added an audit event"


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(SOURCE, id="file-uri"),
        pytest.param("/memory/current/note.md", id="absolute-posix"),
        pytest.param(r"C:\memory\current\note.md", id="absolute-windows"),
        pytest.param("current/note.md", id="relative-posix"),
        pytest.param(r"current\note.md", id="relative-windows"),
    ],
)
def test_use_report_preserves_exact_qualified_identity(source: str) -> None:
    """Exact identity wins even when a second source has the same filename."""
    store = ReportStore({source: OLD_QUOTE, "file:///elsewhere/note.md": NEW_QUOTE})
    result = report_use(store, task="check the port", effect="helped", used=[source])
    assert result["used"] == [source]
    ((event_type, payload, actor),) = store.rows
    assert (event_type, actor) == ("use_report", "agent-report")
    assert payload["used"] == [source], "the audit event lost the exact qualified identity"


def test_use_report_preserves_unique_bare_filename() -> None:
    """A unique bare filename still records its canonical stored identity."""
    store = ReportStore({SOURCE: OLD_QUOTE})
    result = report_use(store, task="check the port", effect="helped", used=["note.md"])
    assert result["used"] == [SOURCE]
    ((_, payload, _),) = store.rows
    assert payload["used"] == [SOURCE], "the audit event kept an alias instead of its source"


@pytest.mark.parametrize("field", ["stale_source", "replacing_source"])
def test_served_stale_report_refuses_unknown_qualified_source(tmp_path: Path, field: str) -> None:
    """The other public consumer must also refuse a qualified typo before append."""
    store = ReportStore({SOURCE: OLD_QUOTE, REPLACEMENT: NEW_QUOTE})
    names = {"stale_source": SOURCE, "replacing_source": REPLACEMENT}
    filename = "note.md" if field == "stale_source" else "replacement.md"
    names[field] = f"wrong-root/{filename}"
    with pytest.raises(StaleReportRefused, match="not a memory in this corpus"):
        report_stale(
            store,
            **names,
            stale_quote=OLD_QUOTE,
            current_quote=NEW_QUOTE,
            task="check the port",
            env={"RECALL_INDEX_ROOT": str(tmp_path)},
        )
    assert store.rows == [], "a stale report credited a different qualified source"
    assert not (tmp_path / ".recall").exists(), "the served-only refusal created a sidecar"
