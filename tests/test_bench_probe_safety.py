"""Probe safety regressions, proved red on df339c87a through CLI consumers."""
from __future__ import annotations

import json
import sys
import tempfile
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.beam import regime_sweep, selfquery_probe


def _stub_regime(monkeypatch):
    class Store:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def ensure_schema(self):
            pass

        def iter_chunks(self):
            return iter(())

    monkeypatch.setattr(regime_sweep, "PgVectorStore", Store)
    monkeypatch.setattr(regime_sweep, "Indexer", lambda *a, **kw: SimpleNamespace(index_path=lambda path: None))
    monkeypatch.setattr(regime_sweep, "_beam_queries", lambda *a: [])
    monkeypatch.setattr(regime_sweep, "_memory_queries", lambda *a: ["synthetic fixture query"])
    monkeypatch.setattr(regime_sweep, "iter_conversations", lambda *a: iter(()))
    monkeypatch.setattr(regime_sweep, "_top1_scores", lambda *a: [0.7])


@pytest.mark.parametrize("probe", [regime_sweep, selfquery_probe])
def test_cloud_conditions_require_explicit_opt_in(probe, monkeypatch, tmp_path):
    """SEC-003: baseline resolves router without consent; fail at the cloud assertion.

    The same test also checks that explicit opt-in still reaches the cloud condition.
    Only model/storage boundaries are replaced; the actual CLI and condition loop run.
    """
    _stub_regime(monkeypatch)
    sweep = tmp_path / "sweep.json"
    sweep.write_text(json.dumps({"conditions": {
        label: {"median": 0.7, "quantile_floors": {"0.05": 0.6}}
        for label, *_ in selfquery_probe.CONDITIONS
    }}))
    monkeypatch.setattr(selfquery_probe, "_self_query_scores", lambda *a: ([0.8], 0))
    specs = []
    monkeypatch.setattr(probe, "resolve_embedder", lambda spec: specs.append(spec) or SimpleNamespace(dim=1))
    args = [probe.__name__, "--out", str(tmp_path / "report.json")]
    if probe is regime_sweep:
        args += ["--memory", str(tmp_path)]
    else:
        args += ["--sweep", str(sweep)]
    monkeypatch.setattr(sys, "argv", args)
    probe.main()
    assert specs, "the CLI must exercise real local conditions"
    assert not any(spec.startswith("router:") for spec in specs), "cloud embedder reached without explicit opt-in"
    specs.clear()
    monkeypatch.setattr(sys, "argv", args + ["--allow-cloud"])
    probe.main()
    assert any(spec.startswith("router:") for spec in specs), "explicit cloud opt-in must remain usable"


def test_regime_requires_an_explicit_memory_corpus(monkeypatch, tmp_path):
    """SEC-003: baseline completes instead of refusing an omitted private corpus."""
    _stub_regime(monkeypatch)
    monkeypatch.setattr(regime_sweep, "resolve_embedder", lambda spec: SimpleNamespace(dim=1))
    monkeypatch.setattr(sys, "argv", ["regime_sweep", "--out", str(tmp_path / "report.json")])
    with pytest.raises(SystemExit) as exc:
        regime_sweep.main()
    assert exc.value.code == 2


def test_regime_uses_owned_workspace_and_ignores_hostile_legacy_path(monkeypatch, tmp_path):
    """SEC-002: df339c87a overwrites a symlink target in its predictable workspace."""
    _stub_regime(monkeypatch)
    index = int(uuid.uuid4().hex[:12], 16)
    legacy = Path(tempfile.gettempdir()) / f"regime-bge-small-{index}"
    legacy.mkdir()
    victim = tmp_path / "victim.md"
    victim.write_text("untouched")
    link = legacy / "turn_000000.md"
    try:
        link.symlink_to(victim)
    except OSError:
        legacy.rmdir()
        pytest.skip("symlinks unavailable on this platform")
    workspaces = []
    try:
        monkeypatch.setattr(regime_sweep, "EMBEDDERS", (("bge-small", "fastembed"),))
        monkeypatch.setattr(regime_sweep, "resolve_embedder", lambda spec: SimpleNamespace(dim=1))
        monkeypatch.setattr(regime_sweep, "iter_conversations", lambda *a: iter([
            SimpleNamespace(index=index, turns=[{"role": "user", "content": "synthetic turn"}]),
        ]))

        def index_path(path):
            if path != tmp_path:
                assert (path / "turn_000000.md").read_text().endswith("synthetic turn\n")
                workspaces.append(path)

        monkeypatch.setattr(regime_sweep, "Indexer", lambda *a, **kw: SimpleNamespace(index_path=index_path))
        monkeypatch.setattr(sys, "argv", ["regime_sweep", "--memory", str(tmp_path), "--out", str(tmp_path / "report.json")])
        regime_sweep.main()
        assert victim.read_text() == "untouched", "the predictable path redirected a write"
        assert len(workspaces) == 1
        assert not workspaces[0].exists(), "owned workspace was not cleaned"
        assert legacy.is_dir(), "the probe removed a foreign workspace"
    finally:
        link.unlink()
        legacy.rmdir()


@pytest.mark.parametrize("family", ["memory", "beam"])
def test_regime_refuses_unverified_cached_corpora(family, monkeypatch, tmp_path):
    """DAT-001: a second run must not measure changed queries against old indexed rows."""
    _stub_regime(monkeypatch)
    indexed = set()
    queries = []

    class Store:
        def __init__(self, *a, table, tenant, **kw):
            self.key = (table, tenant)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def ensure_schema(self):
            pass

        def iter_chunks(self):
            cached_family = self.key[0].startswith("regime_mem_") if family == "memory" else self.key[0].startswith("regime_beam_")
            return iter([object()] if cached_family and self.key in indexed else [])

    monkeypatch.setattr(regime_sweep, "PgVectorStore", Store)
    monkeypatch.setattr(regime_sweep, "Indexer", lambda store, *a, **kw: SimpleNamespace(index_path=lambda path: indexed.add(store.key)))
    monkeypatch.setattr(regime_sweep, "EMBEDDERS", (("bge-small", "fastembed"),))
    monkeypatch.setattr(regime_sweep, "resolve_embedder", lambda spec: SimpleNamespace(dim=1))
    monkeypatch.setattr(regime_sweep, "iter_conversations", lambda *a: iter([SimpleNamespace(index=0, turns=[])]))
    monkeypatch.setattr(regime_sweep, "_memory_queries", lambda path: queries.append(path) or [str(path)])
    for name in ("first", "changed"):
        monkeypatch.setattr(sys, "argv", ["regime_sweep", "--memory", str(tmp_path / name), "--out", str(tmp_path / f"{name}.json")])
        if name == "first":
            regime_sweep.main()
        else:
            with pytest.raises(RuntimeError, match="existing|already|source|corpus"):
                regime_sweep.main()
    assert queries == [tmp_path / "first", tmp_path / "changed"]
