"""The validation command must actually compare at least one query before claiming MATCH."""

from argparse import Namespace
from types import SimpleNamespace

import pytest

from benchmarks.mtrag import multiquery


def test_validate_refuses_empty_actual_sample(monkeypatch, tmp_path, capsys):
    """BUG-004: baseline a6051cac5 returns MATCH after zero comparisons."""
    import recall.embeddings as embeddings
    import recall.sparse as sparse

    (tmp_path / "legs").mkdir()
    (tmp_path / "legs" / "last.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setattr(multiquery, "load_dsn", lambda path: "unused")
    monkeypatch.setattr(embeddings, "FastEmbedEmbedder", lambda *args: SimpleNamespace(dim=4))
    monkeypatch.setattr(sparse.SpladeEncoder, "from_pretrained", lambda *args: object())
    args = Namespace(
        output_dir=tmp_path, dsn_env_file=tmp_path / "unused", sparse_model=None,
        sample=1, table_prefix="unused",
    )

    with pytest.raises(ValueError, match="empty|no quer"):
        multiquery.cmd_validate(args)
    assert '"verdict": "MATCH"' not in capsys.readouterr().out
