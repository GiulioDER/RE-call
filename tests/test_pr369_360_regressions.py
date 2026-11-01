"""PR369 through360 regressions. Red baseline: 94f72b887 on the test host.

Each defect case must fail its behavior assertion before the production repair.
The exact nodes, assertion failures and source hashes are retained in the CCA receipts.
"""
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from recall.desktop.models import RuntimeMode, RuntimeProfile, SourceCategory, SourceSelection
from recall.desktop.runtime import RuntimeErrorBase, SdkMcpGateway, VpsMcpRuntime
from recall.desktop.sources import collect_files, scan_files
from recall.desktop.uploads import UploadError, stage_uploads
from recall.extraction import chunk_extracted_document, extract_document
from recall.reasoning import GenerationSelection, ReasoningPolicy, reason
from recall.reasoning_expansion import ExpansionProposal, ExpansionReport
from recall.reasoning_planner import ReasoningBudget
from tests.test_desktop import FakeGateway
from tests.test_reasoning_api import _chunk, _hit, _request, _result
from tests.conftest import requires_db


@pytest.mark.parametrize("selector", [collect_files, scan_files])
def test_directory_discovery_confines_links(tmp_path, selector):
    """SEC-001: implicit external targets must never enter the upload selection."""
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "private.txt"
    outside.write_text("synthetic-private-marker", encoding="utf8")
    external = root / "notes.md"
    try:
        external.symlink_to(outside)
        inside = root / "inside.md"
        inside.write_text("public", encoding="utf8")
        (root / "alias.md").symlink_to(inside)
    except OSError:
        pytest.skip("file symlink creation is unavailable; proof runs on the test host")
    found = selector([root])
    assert external not in found, "external target escaped the selected directory"
    assert inside in found or root / "alias.md" in found
    assert collect_files([outside]) == (outside,), "explicit file selection remains supported"


class StagingGateway(FakeGateway):
    def __init__(self):
        super().__init__()
        self.contents = []
        self.names = []

    def call(self, name, arguments):
        if name == "recall_ingest":
            job, root, _ = stage_uploads(arguments["tenant"], arguments["files"])
            self.contents.extend(p.read_bytes() for p in root.rglob("*") if p.is_file())
            self.names.append(tuple(sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())))
            self.calls.append((name, arguments))
            return {"job_id": job, "state": "completed", "files": len(arguments["files"])}
        return super().call(name, arguments)


def test_desktop_preserves_same_basename_source_identity(tmp_path, monkeypatch):
    """BUG-001: actual runtime entries must survive the real staging consumer."""
    monkeypatch.setenv("RECALL_INDEX_ROOT", str(tmp_path / "uploads"))
    paths = []
    for name in ("first", "second"):
        path = tmp_path / name / "README.md"
        path.parent.mkdir()
        path.write_text(name, encoding="utf8")
        paths.append(path)
    gateway = StagingGateway()
    runtime = VpsMcpRuntime(RuntimeProfile(RuntimeMode.VPS_MCP, endpoint="https://example.test/mcp"), gateway)
    error = None
    try:
        runtime.start_ingest(SourceSelection(SourceCategory.DOCUMENTS, tuple(paths), "default"))
    except UploadError as exc:
        error = str(exc)
    assert error is None, f"distinct sources collided: {error}"
    assert set(gateway.contents) == {b"first", b"second"}
    runtime.start_ingest(SourceSelection(SourceCategory.DOCUMENTS, (paths[0],), "default"))
    assert gateway.names[1][0] in gateway.names[0], "source identity changed with batch membership"


@pytest.fixture
def desktop_window(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from recall.desktop.ui import MainWindow
    monkeypatch.setattr(MainWindow, "_run", lambda *_args: None)
    app = QApplication.instance() or QApplication([])
    gateway = StagingGateway()
    profile = RuntimeProfile(RuntimeMode.VPS_MCP, endpoint="https://example.test/mcp")
    window = MainWindow(profile, runtime=VpsMcpRuntime(profile, gateway))
    yield window, gateway
    window.close()
    app.processEvents()


def test_drop_collection_uses_worker(desktop_window, tmp_path, monkeypatch):
    """PERF-001: collection must not execute in the event handler."""
    window, _ = desktop_window
    source = tmp_path / "note.md"
    source.write_text("note", encoding="utf8")
    collected, dispatched = [], []
    def collect(paths, _category):
        collected.append(paths)
        return collect_files(paths)
    monkeypatch.setattr("recall.desktop.ui.collect_files", collect)
    monkeypatch.setattr(window, "_run", lambda *args: dispatched.append(args))
    window._accept_drop([str(source)])
    assert not collected, "directory collection ran on the GUI event thread"
    assert len(dispatched) == 1
    fn, done = dispatched[0][:2]
    done(fn())
    assert window.pending_files == [(source, SourceCategory.DOCUMENTS)]
    assert collect_files([source, source.parent, source]) == (source,)


@pytest.mark.parametrize("mode", ["count", "bytes", "partial"])
def test_desktop_batches_at_upload_boundary(desktop_window, tmp_path, monkeypatch, mode):
    """BUG-002: drive the UI, runtime and real staging, including partial failure."""
    window, gateway = desktop_window
    monkeypatch.setenv("RECALL_INDEX_ROOT", str(tmp_path / "uploads"))
    paths = []
    count = 2 if mode == "bytes" else 501
    if mode == "bytes":
        monkeypatch.setattr("recall.desktop.uploads._MAX_TOTAL_BYTES", 10)
    for i in range(count):
        path = tmp_path / f"note{i}.md"
        path.write_text(f"{i:06}", encoding="utf8")
        paths.append(path)
    window.pending_files = [(p, SourceCategory.DOCUMENTS) for p in paths]
    monkeypatch.setattr(window, "_pending_scope", lambda _i: {"tenant": "default", "shared": False})
    original = gateway.call
    if mode == "partial":
        def fail_second(name, arguments):
            if name == "recall_ingest" and gateway.calls:
                raise RuntimeErrorBase("synthetic second-batch refusal")
            result = original(name, arguments)
            if name == "recall_ingest":
                result["files"] += 100  # Generation results include the existing corpus.
            return result
        monkeypatch.setattr(gateway, "call", fail_second)
    errors = []
    def run(fn, done, *_args):
        try:
            result = fn()
        except (RuntimeErrorBase, UploadError, ValueError) as exc:
            errors.append(str(exc))
        else:
            done(result)
    monkeypatch.setattr(window, "_run", run)
    window._start_embedding()
    if mode == "partial":
        assert len(gateway.contents) == 500, "the first valid batch did not complete"
        assert errors and "500" in errors[0] and "completed" in errors[0]
    else:
        assert not errors, f"supported selection violated upload limits: {errors}"
        assert len(gateway.calls) == 2
        assert len(gateway.contents) == count
        assert set(gateway.contents) == {p.read_bytes() for p in paths}
        assert all(len(args["files"]) <= 500 for _, args in gateway.calls)


@pytest.mark.parametrize("failure", [None, "initialize", "call_tool", "transport"])
def test_remote_gateway_closes_owned_client(monkeypatch, failure):
    """BUG-003: an externally supplied SDK client is caller-owned on every exit."""
    import httpx2
    import mcp.client.session
    import mcp.client.streamable_http
    clients = []
    class Client:
        def __init__(self, **kwargs):
            self.headers, self.closed = kwargs["headers"], False
            clients.append(self)
        async def aclose(self):
            self.closed = True
    @asynccontextmanager
    async def transport(_endpoint, *, http_client):
        assert http_client.headers == {"Authorization": "Bearer synthetic-token"}
        if failure == "transport":
            raise RuntimeErrorBase("synthetic failure")
        yield (None, None)
    class Session:
        def __init__(self, *_args): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *_args): pass
        async def initialize(self):
            if failure == "initialize":
                raise RuntimeErrorBase("synthetic failure")
        async def call_tool(self, *_args):
            if failure == "call_tool":
                raise RuntimeErrorBase("synthetic failure")
            return SimpleNamespace(is_error=False, content=[SimpleNamespace(text='{"chunks":3}')])
    monkeypatch.setattr(httpx2, "AsyncClient", Client)
    monkeypatch.setattr(mcp.client.session, "ClientSession", Session)
    monkeypatch.setattr(mcp.client.streamable_http, "streamable_http_client", transport)
    monkeypatch.setattr("recall.desktop.runtime.read_token", lambda _p: "synthetic-token")
    gateway = SdkMcpGateway(RuntimeProfile(RuntimeMode.VPS_MCP, endpoint="https://example.test/mcp"))
    if failure:
        with pytest.raises(RuntimeErrorBase, match="synthetic failure"):
            gateway.call("recall_stats", {})
    else:
        assert gateway.call("recall_stats", {}) == {"chunks": 3}
    assert clients and all(client.closed for client in clients), "HTTP client leaked"


@pytest.mark.parametrize("field", ["generation_id", "pipeline_fingerprint", "corpus_fingerprint"])
@pytest.mark.parametrize("mode", ["depth", "model"])
def test_unpinned_expansion_retains_initial_lineage(field, mode):
    """STAKES-001: public unpinned requests cannot relabel foreign expansion evidence."""
    initial = _result(_hit(_chunk("first", "first.md", "Project context.")), gap_warning=mode == "depth")
    foreign = replace(_result(_hit(_chunk("foreign", "other.md", "Foreign owner."))), **{field: "other"})
    proposal = ExpansionProposal(id="rewrite", mode="rewrite", query="project owner")
    request = _request(initial, policy=ReasoningPolicy(allow_retrieval_expansion=True),
        budget=ReasoningBudget(max_model_calls=1),
        expansion_provider=lambda _r: ExpansionReport(proposals=(proposal,)),
        expansion_retriever=lambda *_a: foreign)
    request = replace(request, generation=GenerationSelection())
    response = reason(request)
    assert {item.chunk_id for item in response.trusted_evidence.items} == {"first"}, "foreign evidence inherited initial lineage"
    assert response.provider_failures
    matching = replace(foreign, **{field: getattr(initial, field)})
    request = replace(request, providers=replace(request.providers, expansion_retriever=lambda *_a: matching))
    response = reason(request)
    assert {item.chunk_id for item in response.trusted_evidence.items} == {"first", "foreign"}


@pytest.mark.parametrize("text,cap", [("H" * 100 + "\n", 40), ("H\nvalue\n", 10)], ids=["header-only", "newline-boundary"])
def test_table_header_boundaries_respect_chunk_cap(text, cap):
    """NUM-001: header-only and header-plus-newline boundary stay within the cap."""
    document = extract_document(Path("table.csv"), text.encode())
    if "value" in text:
        base = "\n".join(document.blocks[0].text.splitlines()[:2])
        cap = len(base) + 1
    chunks = chunk_extracted_document(document, max_chars=cap, overlap=0)
    assert chunks and all(len(piece) <= cap for piece, _ in chunks)


@requires_db
@pytest.mark.parametrize("text,marker", [("| " + "padding " * 160 + "TAILFACT", "TAILFACT"), ("# HEADINGFACT\n\nPress the switch.", "HEADINGFACT")], ids=["single-pipe", "heading"])
def test_non_markdown_indexing_preserves_source_text(tmp_path, make_store, text, marker):
    """DAT-001/002: omitted suffix/headings must remain stored and embedded."""
    from recall.embeddings import HashingEmbedder
    from recall.index import Indexer
    class RecordingEmbedder(HashingEmbedder):
        def __init__(self):
            super().__init__(dim=64)
            self.seen = []
        def embed(self, texts):
            self.seen.extend(texts)
            return super().embed(texts)
    path = tmp_path / "source.txt"
    path.write_text(text, encoding="utf8")
    store, embedder = make_store(64), RecordingEmbedder()
    Indexer(store, embedder).index_path(path)
    assert any(marker in value for value in embedder.seen), "source text was lost before embedding"
    assert any(marker in hit.chunk.text for hit in store.query_sparse(marker, k=5)), "source text was not stored"


def test_drop_completion_keeps_submission_scope(desktop_window, tmp_path, monkeypatch):
    """PERF-001 review repair: scope is captured before the collection worker starts.

    Red mutation: the initial asynchronous fix called _scope_data in its callback.
    """
    window, _ = desktop_window
    source = tmp_path / "note.md"
    source.write_text("note", encoding="utf8")
    scope = {"tenant": "first", "shared": False}
    dispatched = []
    monkeypatch.setattr(window, "_scope_data", lambda: dict(scope))
    monkeypatch.setattr(window, "_run", lambda *args: dispatched.append(args))
    window._accept_drop([str(source)])
    if dispatched:
        fn, done = dispatched[0][:2]
        scope.update(tenant="second", shared=True)
        done(fn())
    assert window.pending_scopes == [{"tenant": "first", "shared": False}], "drop inherited a later tenant selection"
