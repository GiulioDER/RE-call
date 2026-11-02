from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

from examples import basic_trusted_search as example


def test_example_uses_dotenv(tmp_path: Path, monkeypatch) -> None:
    """ENV-001: baseline a6051cac5 fails at the DSN assertion.

    Only the database and search transports are replaced; main reads a real local
    configuration file.
    """
    monkeypatch.chdir(tmp_path)
    values = {
        "RECALL_SERVING_DSN": "postgresql://example.invalid/local_demo",
        "RECALL_TABLE": "local_demo", "RECALL_EMBEDDER": "hashing",
    }
    for name in values:
        monkeypatch.setenv(name, "")  # Record absence too, since the loader writes directly.
        monkeypatch.delenv(name, raising=False)
    (tmp_path / ".env").write_text(
        "\n".join(f"{name}={value}" for name, value in values.items()), encoding="utf-8",
    )
    seen = {}

    def store(dsn, *, dim, table):
        seen.update(dsn=dsn, dim=dim, table=table)
        return nullcontext(object())

    def embedder(name):
        seen["embedder"] = name
        return SimpleNamespace(dim=64)

    monkeypatch.setattr(example, "PgVectorStore", store)
    monkeypatch.setattr(example, "make_embedder", embedder)
    monkeypatch.setattr(example, "trusted_search", lambda *args, **kwargs: SimpleNamespace(
        abstained=True, reason="test transport", hits=(),
    ))
    example.main()

    assert seen["dsn"] == values["RECALL_SERVING_DSN"]
    assert seen["table"] == values["RECALL_TABLE"]
    assert seen["embedder"] == values["RECALL_EMBEDDER"]
