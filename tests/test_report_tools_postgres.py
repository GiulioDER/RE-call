"""The report tools record the names search showed, on a real Postgres store of each kind.

`recall_search` shows a hit's `source` as the memo's `file` under the root (`provenance.file`,
`recall_mcp/retrieval.py`). A generation stores each memo under its manifest URI and the legacy
table under the absolute path, and the audit ledger (`recall_mcp.use_reports_api.resolve_sources`)
accepts only that stored name or an unambiguous bare file name. `tests/test_report_tools_database_names.py`
covers the wiring with a stand-in store; this file builds the real stores the served tenants use
and reaches them through the registered tools.

Invariant and the failure it catches:
- G1 on a promoted generation, with an older generation still holding the same files under other
  URIs, `recall_report_stale` and `recall_report_use` given the `file` names the served rows carry
  record one ledger row each under the served URIs: the names refused ("'notes/old.md' in
  stale_source is not a memory in this corpus"), or read as ambiguous between the two URIs.
- L1 on the legacy table indexed by `Indexer`, the same two reports record under the absolute
  paths: the same refusal.

Red proof, 2026-11-14, on the testbench against a throwaway `pgvector/pgvector:pg18` container (one CPU,
2 GB, labelled, removed after each run), single files with `-n 0`, server.py swapped per run and
checked by sha256:
- the fix (c967b3bb): 5 passed with `tests/test_report_tools_database_names.py`.
- baseline, server.py at abc0ff8f0 (names resolved on a lite store only): G1 and L1 fail in their
  intended assertion, "the names search showed were refused: ... 'notes/old.md' in stale_source is
  not a memory in this corpus".
- M1, `recall_mcp.server._paths_named` keeping every match of the lookup instead of only the
  served ones: G1 fails with the same refusal, the older generation's URI making the name
  ambiguous; L1 passes, having no second URI.
The first run of this file failed L1 in its own setup: `Indexer.index_path` records `file` relative
to the directory indexed, and indexing `notes/` gave bare names, which the ledger already accepts.
It now indexes the corpus root, and the baseline above was run after that change.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import uuid
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import psycopg
import pytest

from recall.embeddings import HashingEmbedder
from recall.index import Indexer
from recall.stale_reports import STALE_REPORT_EVENT
from recall.store import PgVectorStore
from recall_mcp.settings import Settings
from recall_mcp.use_reports_api import USE_REPORT_EVENT
from tests.conftest import TEST_DSN, requires_db
from tests.test_code_anchor_chunks import _S3, _Embedder, manager  # noqa: F401  # `manager` is a fixture

pytestmark = requires_db

OLD_QUOTE = "it is still being prepared for PyPI"
NEW_QUOTE = "was published to PyPI on 2 September"
FILES = {
    "notes/old.md": f"# Release\n\nThe release {OLD_QUOTE}.\n",
    "notes/new.md": f"# Release\n\nRelease 0.12.0 {NEW_QUOTE}.\n",
    "README.md": "# Readme\n\nNotes about releases.\n",
}


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RECALL_EMBED_CACHE", "off")


def _tools(store: Any) -> Any:
    """Call a registered tool with `store` as the served store, as `test_mcp_report_stale_ledger` L7 does."""
    from recall_mcp.server import build_server

    tools = {t.name: t for t in build_server()._tool_manager.list_tools()}
    ctx = SimpleNamespace(request_context=SimpleNamespace(lifespan_context={
        "store": store, "embedder": None, "calibration": None, "settings": Settings.from_env(dict(os.environ)),
    }))

    def call(name: str, **kwargs: Any) -> str:
        try:
            return str(asyncio.run(tools[name].fn(ctx=ctx, **kwargs)))
        except Exception as exc:  # noqa: BLE001  # the text of a refusal is what is asserted on
            return f"RAISED {type(exc).__name__}: {exc}"

    return call


def _report_both(call: Any, old: str, new: str) -> tuple[dict[str, Any], dict[str, Any]]:
    stale = call("recall_report_stale", stale_source=old, replacing_source=new,
                 stale_quote=OLD_QUOTE, current_quote=NEW_QUOTE)
    assert not stale.startswith("RAISED"), f"the names search showed were refused: {stale}"
    use = call("recall_report_use", task="Which release is current?", effect="misled", used=[new], wrong=[old])
    assert not use.startswith("RAISED"), f"the names search showed were refused: {use}"
    return json.loads(stale), json.loads(use)


def _ledger(tenant: str) -> dict[str, dict[str, Any]]:
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (tenant,))
        rows = conn.execute(
            "SELECT event_type, payload FROM recall_audit_events WHERE tenant_id = %s AND event_type = ANY(%s)",
            (tenant, [STALE_REPORT_EVENT, USE_REPORT_EVENT]),
        ).fetchall()
    assert len(rows) == 2, rows
    return {str(event): payload for event, payload in rows}


def _build(manager: Any, prefix: str, corpus: str) -> None:  # noqa: F811
    from recall.generation_build import BuildRequest, chunker_for
    from recall.lineage import EmbedderIdentity, IndexManifestV1, ManifestObjectV1, PipelineIdentity
    from recall.manifest import S3Allowlist, S3ObjectReader

    objects, entries = {}, []
    for file, text in FILES.items():
        data = text.encode("utf-8")
        sha = hashlib.sha256(data).hexdigest()
        objects[("approved", f"{prefix}{file}", sha)] = data
        entries.append(ManifestObjectV1(f"s3://approved/{prefix}{file}", sha, "text/markdown", len(data), sha))
    chunker, chunker_identity = chunker_for(BuildRequest())
    pipeline = PipelineIdentity(
        EmbedderIdentity("fixture", "model-a", 64, revision="model-a-commit"),
        chunker_identity,
        fts_configuration={"language": "english", "schema_version": 1},
    )
    generation = manager.create(IndexManifestV1(manager.tenant_id, corpus, tuple(entries)), pipeline)
    reader = S3ObjectReader(_S3(objects), S3Allowlist.parse("approved/corpora/"))
    manager.build(generation.generation_id, reader, _Embedder(), chunker)
    manager.validate(generation.generation_id)
    manager.promote(generation.generation_id, unsafe_development=True)


def _served_files(store: Any) -> dict[str, str]:
    """`{file: stored source}` for the served rows: the name search shows, and the one stored."""
    return {str(c.metadata["file"]): c.source for c in store.iter_chunks()}


def test_a_generation_records_the_names_search_showed(manager) -> None:  # type: ignore[no-untyped-def]  # noqa: F811
    """G1."""
    from recall.generation_store import GenerationStore

    tenant = manager.tenant_id
    _build(manager, f"corpora/{tenant}/checkout-a/", "corpus-a")
    _build(manager, f"corpora/{tenant}/checkout-b/", "corpus-b")
    served = f"s3://approved/corpora/{tenant}/checkout-b/"
    with GenerationStore(TEST_DSN, 64, tenant=tenant) as store:
        files = _served_files(store)
        assert files["notes/old.md"] == served + "notes/old.md", files
        # The older generation's rows are what erasure's lookup also returns.
        assert len(store.sources_for_identifiers(["notes/old.md"])["notes/old.md"]) == 2
        stale, use = _report_both(_tools(store), "notes/old.md", "notes/new.md")
    assert stale["recorded_in"] == ["audit_ledger"], stale
    assert (use["used"], use["wrong"]) == ([served + "notes/new.md"], [served + "notes/old.md"]), use
    rows = _ledger(tenant)
    stale_row = rows[STALE_REPORT_EVENT]
    assert (stale_row["stale_source"], stale_row["replacing_source"]) == (served + "notes/old.md", served + "notes/new.md")


@pytest.fixture
def legacy_store(make_store: Any) -> Iterator[PgVectorStore]:
    table = make_store(64).table
    tenant = "report-legacy-" + uuid.uuid4().hex[:10]
    store = PgVectorStore(TEST_DSN, dim=64, table=table, tenant=tenant)
    yield store
    store.close()
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (tenant,))
        conn.execute("DELETE FROM recall_audit_events WHERE tenant_id = %s", (tenant,))


def test_the_legacy_table_records_the_names_search_showed(
    legacy_store: PgVectorStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """L1."""
    for file, text in FILES.items():
        (tmp_path / file).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / file).write_text(text, encoding="utf-8")
    monkeypatch.setenv("RECALL_INDEX_ROOT", str(tmp_path))
    # `file` is relative to the directory indexed, so the corpus root gives names with a folder.
    Indexer(legacy_store, HashingEmbedder(dim=64)).index_path(tmp_path)
    files = _served_files(legacy_store)
    assert {k: files.get(k) for k in ("notes/old.md", "notes/new.md")} == {
        "notes/old.md": str(tmp_path / "notes" / "old.md"), "notes/new.md": str(tmp_path / "notes" / "new.md"),
    }, files
    # A serving checkout: the root the queue reads holds no memo files.
    monkeypatch.setenv("RECALL_INDEX_ROOT", str(tmp_path / "serving"))
    (tmp_path / "serving").mkdir()
    stale, use = _report_both(_tools(legacy_store), "notes/old.md", "notes/new.md")
    assert stale["recorded_in"] == ["audit_ledger"], stale
    assert (use["used"], use["wrong"]) == ([files["notes/new.md"]], [files["notes/old.md"]]), use
    stale_row = _ledger(legacy_store.tenant)[STALE_REPORT_EVENT]
    assert (stale_row["stale_source"], stale_row["replacing_source"]) == (files["notes/old.md"], files["notes/new.md"])
