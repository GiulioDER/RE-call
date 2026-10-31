"""Regression boundaries for the surviving changes in PRs 499 through 490.

Each parametrized case was run unchanged against addaa974d remotely before the
repair. The intended assertions, baseline hashes and exact node IDs are recorded
in the audit's remote red-state and regression-proof.json receipts.
"""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from benchmarks.evidence_graph_pilot import run_deepseek_answers, run_pilot
from recall.calibration import Calibration
from recall.desktop.sources import scan_files
from recall.related import trusted_related
from recall.semantic_graph import build_semantic_graph
from recall.types import Chunk


class RelatedStore:
    tenant = "tenant-a"
    generation_id = "generation-a"

    def __init__(self, chunks, *, bounded=False, superseded=False):
        self.chunks = chunks
        self.bounded = bounded
        self.superseded = superseded

    def iter_chunks(self, batch_size=1000):
        return iter(self.chunks)

    def supersession(self):
        return ({"notes.md": "replacement.md"} if self.superseded else {}), frozenset()

    def related_chunks(self, seed_chunk_id, relation, max_items):
        if not self.bounded:
            return None
        return self.chunks[0], self.chunks[1:]


@pytest.mark.parametrize("relation", ["source", "ordinal"])
@pytest.mark.parametrize("bounded", [False, True])
def test_related_neighbors_reject_superseded_documents(relation, bounded):
    """BUG-001: the trust gate must see supersession for every related mode."""
    store = RelatedStore(
        [
            Chunk("seed", "notes.md", "seed", {"file": "notes.md", "ord": 0}),
            Chunk("neighbor", "notes.md", "old evidence", {"file": "notes.md", "ord": 1}),
        ],
        bounded=bounded,
        superseded=True,
    )
    result = trusted_related(
        store, "seed", relation=relation,
        calibration=Calibration(embedder="fixture", threshold=0.5),
        now=datetime(2026, 10, 31, tzinfo=UTC),
    )
    assert result.items == (), "superseded neighbors must never be trusted current evidence"
    assert result.rejected_count == 1


@pytest.mark.parametrize("relation", ["source", "ordinal"])
def test_related_neighbors_keep_complete_source_identity(relation):
    """BUG-005: two roots may contain the same relative filename."""
    store = RelatedStore([
        Chunk("seed", "file:///root-a/notes.md", "seed", {"file": "notes.md", "ord": 0}),
        Chunk("foreign", "file:///root-b/notes.md", "unrelated", {"file": "notes.md", "ord": 1}),
        Chunk("neighbor", "file:///root-a/notes.md", "related", {"file": "notes.md", "ord": 1}),
    ])
    result = trusted_related(
        store, "seed", relation=relation,
        calibration=Calibration(embedder="fixture", threshold=0.5),
        now=datetime(2026, 10, 31, tzinfo=UTC),
    )
    assert [hit.chunk.id for hit in result.items] == ["neighbor"]


@pytest.mark.parametrize("uncertainty", [None, 42, "unclear", {"bad": "flag"}, [1]])
def test_invalid_relation_uncertainty_is_diagnostic(uncertainty):
    """BUG-002: invalid JSON metadata must not abort the corpus graph build."""
    failure = None
    graph = None
    try:
        graph = build_semantic_graph(
            [Chunk("c1", "memo.md", "", {
                "project": "Alpha", "service": "Beta",
                "relations": [{"relation": "supports", "subject": "Alpha", "object": "Beta", "uncertainty": uncertainty}],
            })], tenant_id="tenant-a", generation_id="generation-a",
        )
    except (TypeError, ValueError) as exc:
        failure = exc
    assert failure is None, f"malformed uncertainty must produce a diagnostic: {failure}"
    assert graph is not None
    assert not graph.relations, "a malformed relation must not be admitted"
    assert any(item.kind == "invalid_relation" for item in graph.diagnostics)


@pytest.mark.parametrize("session_name", ["feature", "Claude"])
def test_project_worktree_remains_a_normal_scan_root(tmp_path, session_name):
    """BUG-004: a .claude ancestor does not make a project configuration."""
    project = tmp_path / "repo" / ".claude" / "worktrees" / session_name
    (project / "docs").mkdir(parents=True)
    code = project / "app.py"
    document = project / "docs" / "design.md"
    code.write_text("print('fixture')", encoding="utf-8")
    document.write_text("# Design", encoding="utf-8")
    assert set(scan_files([project])) == {code, document}


def _private_config_fixture(tmp_path, monkeypatch):
    home = tmp_path / "user"
    config = home / ".claude"
    config.mkdir(parents=True)
    credential = config / ".credentials.json"
    credential.write_text('{"fixture": true}', encoding="utf-8")
    (config / "settings.json").write_text("{}", encoding="utf-8")
    memory = config / "MEMORY.md"
    memory.write_text("# Durable fixture", encoding="utf-8")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    return home, config, credential, memory


def test_home_scan_excludes_private_configuration(tmp_path, monkeypatch):
    """SEC-001: ancestor scans retain the actual configuration exclusion."""
    home, _config, credential, memory = _private_config_fixture(tmp_path, monkeypatch)
    found = scan_files([home])
    assert credential not in found, "an ancestor scan must not admit credential JSON"
    assert credential.parent / "settings.json" not in found
    assert memory in found


def test_scan_excludes_links_to_private_configuration(tmp_path, monkeypatch):
    """SEC-001: admission follows the resolved target, before deduplication.

    Linux exercises the default Documents scan with a file symlink. A Windows
    host without symlink privileges uses a directory junction as its scan root.
    """
    import sys

    home, config, credential, memory = _private_config_fixture(tmp_path, monkeypatch)
    documents = home / "Documents"
    documents.mkdir()
    link = documents / "selected.json"
    try:
        link.symlink_to(credential)
    except OSError:
        if sys.platform != "win32":
            raise
        import _winapi

        junction = documents / "linked-config"
        _winapi.CreateJunction(str(config), str(junction))
        found = scan_files([junction])
        assert not any(path.name == credential.name for path in found)
        assert any(path.name == memory.name for path in found)
    else:
        found = scan_files([documents])
        assert link not in found, "a file link must not bypass configuration admission"


def test_pilot_default_model_uses_normalized_provider_presence(monkeypatch):
    """ENV-001: model resolution and transport use the same credential presence."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", " \t")
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture")
    calls = []

    def completion(prompt, model):
        calls.append(model)
        return '{"answer":"", "abstained":true, "citation_chunk_ids":[]}'

    answers = run_deepseek_answers(run_pilot(), completion=completion)
    assert answers
    assert calls and set(calls) == {"deepseek/deepseek-chat"}
