"""The report tools on a database-backed server accept the names search showed.

`recall_search` shows a hit's `source` as its file under the index root (`provenance.file`,
`recall_mcp/retrieval.py`), while a Postgres store keeps another name: the legacy indexer stores the
absolute path, a generation stores the manifest's `file://` URI. Measured 2026-11-14 on the served
`re-call-code-gen` tenant, read-only: search showed `recall_mcp/stale_reports_api.py`, and
`recall_inventory` listed that file as a `file:///.../recall_mcp/stale_reports_api.py` URI. The audit
ledger (`recall_mcp.use_reports_api.resolve_sources`) accepts only the stored name or an unambiguous
bare file name, so both report tools refused every name with a folder; an authenticated server has
no review queue, so the stale report was refused outright.

The store here is a stand-in with exactly that shape, reached through the registered tools: two
files served under `file://` URIs of the current checkout, and an older generation that still holds
the same files under a previous checkout's URIs, which `sources_for_identifiers` (erasure's lookup,
across every generation) returns and `sources_present` (the served generation, as
`source_content_hashes`) does not.

Invariant and the failure it catches:
- D1 `recall_report_stale` with the names search showed records one ledger row under the served
  URIs: "'recall/old.md' in stale_source is not a memory in this corpus".
- D2 `recall_report_use` with the names search showed records them under the served URIs: the same
  refusal for `used`.
- D3 a file an older generation also held under another URI still resolves to the one served URI:
  the lookup's two matches read as ambiguous, and the name refused.
- D4 a name the served generation does not hold is refused by its own name.

Red proof, 2026-11-14, on this workstation (single file, light check). At abc0ff8f0, where the
names were resolved on a lite store only, D1 and D2 fail in their intended assertion: "the names
search showed were refused: ... 'recall/old.md' in stale_source is not a memory in this corpus",
and the same for `'recall/new.md' in used`. Then, `recall_mcp/server.py` restored byte for byte
(sha256 checked) and the file green:
- M1 (D3, through D1 and D2) `recall_mcp.server._paths_named` keeping every match of the lookup
  instead of only the served ones: the same two refusals, the retired URI making each name
  ambiguous.
D4 is a control, not a behaviour this change adds: it passes at abc0ff8f0 and after, and guards
that resolving names never makes an unknown one resolvable.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from recall.stale_reports import STALE_REPORT_EVENT
from recall_mcp.settings import Settings
from recall_mcp.use_reports_api import USE_REPORT_EVENT

OLD_QUOTE = "it is still being prepared for PyPI"
NEW_QUOTE = "was published to PyPI on 2 September"
SERVED = "file:///srv/checkout-b/"
RETIRED = "file:///srv/checkout-a/"
FILES = {"recall/old.md": f"The release {OLD_QUOTE}.", "recall/new.md": f"Release 0.12.0 {NEW_QUOTE}."}


class GenerationLikeStore:
    """A database-backed store whose rows name each memo by URI and record its file under the root."""

    tenant = "re-call-code-gen"

    def __init__(self) -> None:
        self.served = {SERVED + f: text for f, text in FILES.items()}
        self.rows: list[tuple[str, dict[str, Any]]] = []

    def source_content_hashes(self) -> dict[str, str]:
        return {uri: "h" for uri in self.served}

    def source_chunk_texts(self, source: str) -> list[str]:
        return [self.served[source]] if source in self.served else []

    def sources_for_identifiers(self, identifiers: list[str]) -> dict[str, list[str]]:
        """As `GenerationStore.sources_for_identifiers`: a row's file or URI, in every generation."""
        rows = [(prefix + f, f) for f in FILES for prefix in (RETIRED, SERVED)]
        found: dict[str, list[str]] = {}
        for uri, file in rows:
            for ident in {uri, file} & set(identifiers):
                found.setdefault(ident, []).append(uri)
        return found

    def sources_present(self, sources: list[str]) -> set[str]:
        """As `GenerationStore.sources_present`: the served generation only."""
        return set(sources) & set(self.served)

    def append_audit_event(self, event_type: str, payload: dict[str, Any], **_: Any) -> str:
        self.rows.append((event_type, payload))
        return f"evt-{len(self.rows)}"


@pytest.fixture
def call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    from recall_mcp.server import build_server

    monkeypatch.setenv("RECALL_INDEX_ROOT", str(tmp_path))  # a serving checkout: no memo files
    store = GenerationLikeStore()
    tools = {t.name: t for t in build_server()._tool_manager.list_tools()}
    ctx = SimpleNamespace(request_context=SimpleNamespace(lifespan_context={
        "store": store, "embedder": None, "calibration": None, "settings": Settings.from_env(dict(os.environ)),
    }))

    def run(name: str, **kwargs: Any) -> str:
        try:
            out = asyncio.run(tools[name].fn(ctx=ctx, **kwargs))
        except Exception as exc:  # noqa: BLE001  # the text of a refusal is what is asserted on
            return f"RAISED {type(exc).__name__}: {exc}"
        return str(out)

    run.store = store  # type: ignore[attr-defined]
    return run


def test_report_stale_records_the_names_search_showed(call: Any) -> None:
    """D1, D3."""
    out = call(
        "recall_report_stale", stale_source="recall/old.md", replacing_source="recall/new.md",
        stale_quote=OLD_QUOTE, current_quote=NEW_QUOTE,
    )
    assert not out.startswith("RAISED"), f"the names search showed were refused: {out}"
    assert json.loads(out)["recorded_in"] == ["audit_ledger"], out
    ((event, payload),) = call.store.rows
    assert event == STALE_REPORT_EVENT
    assert (payload["stale_source"], payload["replacing_source"]) == (SERVED + "recall/old.md", SERVED + "recall/new.md")


def test_report_use_records_the_names_search_showed(call: Any) -> None:
    """D2, D3."""
    out = call("recall_report_use", task="Which release is current?", effect="misled",
               used=["recall/new.md"], wrong=["recall/old.md"])
    assert not out.startswith("RAISED"), f"the names search showed were refused: {out}"
    ((event, payload),) = call.store.rows
    assert event == USE_REPORT_EVENT
    assert (payload["used"], payload["wrong"]) == ([SERVED + "recall/new.md"], [SERVED + "recall/old.md"])


def test_a_name_the_served_generation_lacks_is_refused_by_its_own_name(call: Any) -> None:
    """D4."""
    out = call("recall_report_use", task="Which release is current?", effect="helped", used=["recall/gone.md"])
    assert out.startswith("RAISED") and "'recall/gone.md' in used is not a memory" in out, out
    assert call.store.rows == []
