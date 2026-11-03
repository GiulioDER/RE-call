"""Offline consumer evidence for BUG-001 in the PR59 through PR50 audit.

The selected-profile and mismatch cases fail on baseline 76b05b712 in
labelled._make_embedder: the CLI evaluates the legacy model or fails to refuse
the incompatible profile. The no-profile controls are red-proved by changing
the factory's resolver argument to "fastembed", which selects the wrong model.
The exact parametrized nodes and assertion outcomes are retained in the audit
proof receipt. Provider constructors are stubbed; no API or model runs.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from recall import embeddings
from recall.eval import labelled, locomo


@pytest.mark.parametrize(
    "name,profile,expected",
    [
        ("voyage", "voyage-4-v1", "registered-voyage-4"),
        ("voyage", "bge-small-context-section-v1", None),
        ("st:fixture-model", "voyage-4-v1", None),
        ("hashing", "voyage-4-v1", None),
        ("voyage", None, "legacy-voyage-3"),
        ("st:fixture-model", None, "st:fixture-model"),
        ("hashing", None, "hashing-64"),
    ],
    ids=["selected-voyage", "voyage-mismatch", "st-mismatch", "hashing-mismatch",
         "legacy-voyage", "legacy-st", "legacy-hashing"],
)
def test_labelled_cli_respects_embedding_profile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    capsys: pytest.CaptureFixture[str], name: str, profile: str | None,
    expected: str | None,
) -> None:
    monkeypatch.delenv("RECALL_EMBED_PROFILE", raising=False)
    if profile is not None:
        monkeypatch.setenv("RECALL_EMBED_PROFILE", profile)
    monkeypatch.setattr(embeddings, "VoyageEmbedder", lambda **kw: SimpleNamespace(
        name="legacy-voyage-3", dim=1024,
    ))
    monkeypatch.setattr(embeddings, "SentenceTransformerEmbedder", lambda model: SimpleNamespace(
        name=f"st:{model}", dim=64,
    ))
    monkeypatch.setattr(embeddings, "FastEmbedEmbedder", lambda **kw: SimpleNamespace(
        name="unexpected-fastembed", dim=384,
    ))
    monkeypatch.setattr(embeddings, "resolve_registered_embedder", lambda *a: SimpleNamespace(
        name="registered-voyage-4", dim=1024,
    ))
    observed = []

    def evaluate(dsn, corpus, questions, embedder, **kwargs):
        observed.append(embedder.name)
        return {"embedder": embedder.name}

    monkeypatch.setattr(labelled, "evaluate", evaluate)
    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps([{"id": "u1", "query": "fixture", "answerable": False}]))
    monkeypatch.setattr(sys, "argv", [
        "labelled", "--corpus", str(tmp_path), "--questions", str(questions),
        "--embedder", name,
    ])
    if expected is None:
        with pytest.raises(ValueError, match="RECALL_EMBED_PROFILE"):
            labelled.main()
        assert observed == []
    else:
        labelled.main()
        assert observed == [expected]
        assert json.loads(capsys.readouterr().out) == {"embedder": expected}
    assert locomo._make_embedder is labelled._make_embedder
