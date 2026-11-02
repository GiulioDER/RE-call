"""Builder boundary regressions, red on pre-fix 33759a686 (see audit proof receipt)."""
from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from benchmarks import h2h_artifact as h2h


@pytest.fixture
def artifact():
    return json.loads(Path("results/head_to_head/paired_accuracy.json").read_text("utf-8"))


def _old_output(out):
    (out / "outcomes").mkdir()
    (out / "paired_accuracy.json").write_bytes(b'{"version":"old"}\n')
    (out / "outcomes/current.jsonl").write_bytes(b"old-vector\n")
    (out / "outcomes/stale.jsonl").write_bytes(b"old-stale\n")
    (out / "notes.txt").write_bytes(b"unrelated\n")


def _snapshot(out):
    return {str(p.relative_to(out)): p.read_bytes() for p in out.rglob("*") if p.is_file()}


@pytest.mark.parametrize("field", [
    "mem0", "replicate_low_accuracy", "replicate_high_accuracy", "margin_min", "margin_max",
    "replicate_spread_points",
])
def test_as_shipped_mismatch_refuses_before_writing(tmp_path, monkeypatch, artifact, field):
    """BUG-002: baseline returns success and overwrites verified output."""
    shipped = artifact["as_shipped_embedder"]
    if field == "mem0":
        shipped["mem0"]["accuracy"] = 0.9
    else:
        shipped[field] = 0.9
    monkeypatch.setattr(h2h, "build", lambda runs: (artifact, {"current": ["new-vector"]}))
    _old_output(tmp_path)
    before = _snapshot(tmp_path)
    result = h2h.main(["--runs", str(tmp_path), "--out", str(tmp_path)])
    assert result == 1
    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize("replicate", [0, 1])
def test_as_shipped_significance_mismatch_refuses_before_writing(tmp_path, monkeypatch, artifact,
                                                               replicate):
    """BUG-002 architect correction: matching margins do not imply the published p bound."""
    artifact["as_shipped_embedder"]["recall_replicates"][replicate]["mcnemar"]["p_value"] = 0.1
    monkeypatch.setattr(h2h, "build", lambda runs: (artifact, {"current": ["new-vector"]}))
    _old_output(tmp_path)
    before = _snapshot(tmp_path)
    assert h2h.main(["--runs", str(tmp_path), "--out", str(tmp_path)]) == 1
    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize("arm", ["recall", "mem0"])
@pytest.mark.parametrize("field", ["arm", "generator", "judge"])
def test_builder_refuses_mislabeled_run_config(tmp_path, monkeypatch, arm, field):
    """DAT-001: baseline accepts unchanged outcomes under a false configuration label."""
    row = h2h.PUBLISHED_ROWS[0]
    monkeypatch.setattr(h2h, "PUBLISHED_ROWS", (row,))
    monkeypatch.setattr(h2h, "_digest", lambda path: "fixture-digest")
    monkeypatch.setattr(h2h, "_as_shipped", lambda runs, vectors: {})
    outcomes = [{"question_id": "q1", "correct": True, "is_adversarial": False}]

    def load(path):
        actual_arm = "recall" if path.name == row.recall_run else "mem0"
        config = {"arm": actual_arm, "generator": "openai/" + row.generator,
                  "judge": "openai/" + row.judge}
        if actual_arm == arm:
            config[field] = "different"
        return outcomes, config

    monkeypatch.setattr(h2h, "load_run", load)
    refused = False
    try:
        h2h.build(tmp_path)
    except ValueError:
        refused = True
    assert refused, "a changed configuration must not retain the original comparison label"


def test_matching_artifact_and_configs_are_accepted(tmp_path, monkeypatch, artifact):
    """Positive control: existing qualified names remain valid."""
    assert h2h.verify_against_published(artifact) == []
    row = h2h.PUBLISHED_ROWS[0]
    monkeypatch.setattr(h2h, "PUBLISHED_ROWS", (row,))
    monkeypatch.setattr(h2h, "_digest", lambda path: "fixture-digest")
    monkeypatch.setattr(h2h, "_as_shipped", lambda runs, vectors: {})
    monkeypatch.setattr(h2h, "load_run", lambda path: (
        [{"question_id": "q1", "correct": True, "is_adversarial": False}],
        {"arm": "recall" if path.name == row.recall_run else "mem0",
         "generator": "openai/" + row.generator, "judge": "openai/" + row.judge},
    ))
    built, _ = h2h.build(tmp_path)
    assert built["rows"][row.row_id]["generator"] == row.generator


@pytest.mark.parametrize("failure", ["serialization", "vector-write", "summary-write", "publish"])
def test_failed_publication_preserves_complete_old_output(tmp_path, monkeypatch, failure):
    """STAKES-001/DAT-002: baseline truncates or mixes files before a later failure."""
    _old_output(tmp_path)
    before = _snapshot(tmp_path)
    replacement = {"version": "new-summary"}
    if failure == "serialization":
        replacement["unserializable"] = object()
    else:
        original_open, original_fdopen, original_replace = Path.open, os.fdopen, os.replace
        fired = False
        marker = "new-vector" if failure == "vector-write" else "new-summary"

        class FailingWriter:
            def __init__(self, handle):
                self.handle = handle

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return self.handle.__exit__(*args)

            def __getattr__(self, name):
                return getattr(self.handle, name)

            def write(self, data):
                nonlocal fired
                text = data.decode() if isinstance(data, bytes) else data
                if not fired and marker in text and failure != "publish":
                    fired = True
                    self.handle.write(data[:1])
                    raise OSError("injected staging write failure")
                return self.handle.write(data)

        def open_path(path, mode="r", *args, **kwargs):
            nonlocal fired
            if failure == "publish" and not fired and path == tmp_path / "paired_accuracy.json" and "w" in mode:
                fired = True
                raise OSError("injected publication failure")
            handle = original_open(path, mode, *args, **kwargs)
            return FailingWriter(handle) if "w" in mode else handle

        def replace(source, destination):
            nonlocal fired
            if failure == "publish" and not fired and Path(destination) == tmp_path / "paired_accuracy.json":
                fired = True
                raise OSError("injected publication failure")
            return original_replace(source, destination)

        monkeypatch.setattr(Path, "open", open_path)
        monkeypatch.setattr(os, "fdopen", lambda *a, **kw: FailingWriter(original_fdopen(*a, **kw)))
        monkeypatch.setattr(os, "replace", replace)
    with pytest.raises((TypeError, OSError)):
        h2h.write_artifact(tmp_path, replacement, {"current": ["new-vector"]})
    assert _snapshot(tmp_path) == before


def test_successful_publication_prunes_only_vectors_and_uses_lf(tmp_path):
    _old_output(tmp_path)
    h2h.write_artifact(tmp_path, {"version": "new-summary"}, {"current": ["new-vector"]})
    assert json.loads((tmp_path / "paired_accuracy.json").read_bytes()) == {"version": "new-summary"}
    assert (tmp_path / "outcomes/current.jsonl").read_bytes() == b"new-vector\n"
    assert not (tmp_path / "outcomes/stale.jsonl").exists()
    assert (tmp_path / "notes.txt").read_bytes() == b"unrelated\n"


def test_publication_preserves_existing_file_modes(tmp_path):
    """DAT-002: the first staged repair made existing 0644 files private (0600)."""
    _old_output(tmp_path)
    targets = [tmp_path / "paired_accuracy.json", tmp_path / "outcomes/current.jsonl"]
    for target in targets:
        target.chmod(0o644)
    before = [stat.S_IMODE(target.stat().st_mode) for target in targets]
    h2h.write_artifact(tmp_path, {"version": "new-summary"}, {"current": ["new-vector"]})
    assert [stat.S_IMODE(target.stat().st_mode) for target in targets] == before


def test_publication_uses_standard_modes_for_new_files(tmp_path):
    """DAT-002: staging must honor the ordinary file-creation umask for fresh output."""
    reference = tmp_path / "ordinary-file"
    reference.write_bytes(b"ordinary creation")
    expected = stat.S_IMODE(reference.stat().st_mode)
    h2h.write_artifact(tmp_path, {"version": "new-summary"}, {"current": ["new-vector"]})
    targets = [tmp_path / "paired_accuracy.json", tmp_path / "outcomes/current.jsonl"]
    assert [stat.S_IMODE(target.stat().st_mode) for target in targets] == [expected, expected]
