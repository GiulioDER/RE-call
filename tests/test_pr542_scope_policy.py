"""Source policy must protect scope labels and counts before inventory reads."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from recall.cli import main
from recall.cli_commands import index_search
from tests.conftest import TEST_DSN


@pytest.mark.parametrize("backend", ["legacy", "generation", "lite"])
@pytest.mark.parametrize("policy_kind", ["deny", "malformed", "unreadable"])
def test_scopes_refuses_source_policy_before_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    backend: str,
    policy_kind: str,
) -> None:
    """SEC-001: every backend must refuse policy-protected aggregate inventory.

    Node: tests/test_pr542_scope_policy.py::test_scopes_refuses_source_policy_before_inventory.
    Red baseline: 76a0b36ea, recall.cli_commands.index_search._cmd_scopes ignored the policy.
    Intended failure: the final assertion observes store/metadata reads and an unrestricted label
    instead of a source-policy refusal. The initial no-policy call is a compatibility control.
    The parent audit runs these unchanged nodes red and green on the remote test host.
    """
    for key in ("RECALL_INDEX_MODE", "RECALL_SOURCE_POLICY_FILE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("RECALL_ENV", "production" if backend == "generation" else "development")
    # Every backend is replaced below. This only lets the CLI reach those doubles with TEST_DSN.
    monkeypatch.setenv("RECALL_ALLOW_INSECURE_DSN", "1")
    reads: list[str] = []

    class MetadataStore:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            reads.append("store")

        def __enter__(self) -> MetadataStore:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def scope_inventory(self, _dimension: str) -> list[tuple[str, int, int]]:
            reads.append("inventory")
            return [("restricted-area", 7, 3)]

        def scope_undeclared_count(self, _dimension: str) -> int:
            reads.append("undeclared")
            return 2

        def scope_summary(self, dimension: str) -> tuple[list[tuple[str, int, int]], int]:
            return self.scope_inventory(dimension), self.scope_undeclared_count(dimension)

    def lite_metadata(*_args: object) -> dict[str, object]:
        reads.append("lite-metadata")
        return {"tenant": "default", "dim": 64}

    def forbidden_embedder(_name: str) -> None:
        pytest.fail("metadata inventory must not resolve an embedder")

    monkeypatch.setattr(index_search, "PgVectorStore", MetadataStore)
    monkeypatch.setattr(index_search, "open_legacy_store", MetadataStore)
    monkeypatch.setattr(index_search, "_make_embedder", forbidden_embedder)
    monkeypatch.setattr("recall.generation_store.GenerationStore", MetadataStore)
    monkeypatch.setattr("recall.cli_commands._lite.lite_read_meta", lite_metadata)
    dsn = "sqlite:///" + (tmp_path / "memory.db").as_posix() if backend == "lite" else TEST_DSN
    command = ["--dsn", dsn, "--tenant", "default", "scopes"]

    main(command)
    control_output = capsys.readouterr().out
    assert "restricted-area" in control_output and "7" in control_output and "3" in control_output
    assert "inventory" in reads
    reads.clear()

    policy = tmp_path / "source-policy.json"
    if policy_kind == "deny":
        policy.write_text(json.dumps({"rules": [], "default_deny": True}), encoding="utf-8")
    elif policy_kind == "malformed":
        policy.write_text("[]", encoding="utf-8")
    # The unreadable case deliberately names a missing file.
    monkeypatch.setenv("RECALL_SOURCE_POLICY_FILE", str(policy))
    refusal: SystemExit | None = None
    try:
        main(command)
    except SystemExit as exc:
        refusal = exc
    output = capsys.readouterr().out
    assert refusal is not None and "source policy" in str(refusal) and not reads and not output, (
        "source policy must refuse scope inventory before metadata reads or disclosure",
        refusal,
        reads,
        output,
    )
