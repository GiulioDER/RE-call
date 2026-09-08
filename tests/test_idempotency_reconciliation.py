"""Direct coverage for the operator reconciliation workflow."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import pytest

from recall_mcp.server import _durable_replay, _mutation_fingerprint, build_server
from recall_mcp.settings import bootstrap_settings


class _ReceiptStore:
    def __init__(self) -> None:
        self.receipts: dict[tuple[str, str], tuple[str, str]] = {}

    def __enter__(self) -> "_ReceiptStore":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def get_operation_receipt(
        self,
        key: str,
        *,
        operation: str,
        request_fingerprint: str,
    ) -> str | None:
        row = self.receipts.get((key, operation))
        if row is None:
            return None
        if row[0] != request_fingerprint:
            from recall.errors import IdempotencyConflict

            raise IdempotencyConflict()
        return row[1]

    def record_operation_receipt(
        self,
        key: str,
        result: str,
        *,
        operation: str,
        request_fingerprint: str,
    ) -> None:
        self.receipts.setdefault((key, operation), (request_fingerprint, result))


def _args(result_file: Path, *, confirm: str | None = None) -> argparse.Namespace:
    return argparse.Namespace(
        idempotency_cmd="reconcile",
        key="mutation-1",
        operation="recall_index",
        fingerprint=_mutation_fingerprint("recall_index", {"path": "memory"}),
        result_file=str(result_file),
        confirm=confirm,
        dim=64,
        dsn="postgresql://unused",
        table="chunks",
        tenant="tenant-a",
    )


def test_reconciliation_command_records_a_verified_result_and_never_runs_mutation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from recall.cli_commands import idempotency_cmd

    result = '{"files":1,"chunks":2}'
    result_file = tmp_path / "verified-result.json"
    result_file.write_text(result, encoding="utf-8")
    store = _ReceiptStore()
    monkeypatch.setattr(idempotency_cmd, "PgVectorStore", lambda *_args, **_kwargs: store)

    idempotency_cmd._cmd_idempotency(_args(result_file))
    assert json.loads(capsys.readouterr().out)["status"] == "dry_run"
    assert store.receipts == {}

    idempotency_cmd._cmd_idempotency(_args(result_file, confirm="RECONCILE_IDEMPOTENCY"))
    assert json.loads(capsys.readouterr().out)["status"] == "reconciled"
    assert store.receipts[("mutation-1", "recall_index")] == (
        _args(result_file).fingerprint,
        result,
    )
    assert asyncio.run(
        _durable_replay(
            store,
            "mutation-1",
            "recall_index",
            _args(result_file).fingerprint,
        )
    ) == result


def test_reconciliation_command_is_idempotent_and_rejects_invalid_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from recall.cli_commands import idempotency_cmd

    result_file = tmp_path / "verified-result.json"
    result_file.write_text('{"ok":true}', encoding="utf-8")
    store = _ReceiptStore()
    monkeypatch.setattr(idempotency_cmd, "PgVectorStore", lambda *_args, **_kwargs: store)
    args = _args(result_file, confirm="RECONCILE_IDEMPOTENCY")
    idempotency_cmd._cmd_idempotency(args)
    capsys.readouterr()
    idempotency_cmd._cmd_idempotency(args)
    assert json.loads(capsys.readouterr().out)["status"] == "already_reconciled"

    invalid = tmp_path / "invalid.json"
    invalid.write_text("not-json", encoding="utf-8")
    invalid_args = _args(invalid, confirm="RECONCILE_IDEMPOTENCY")
    invalid_args.key = "mutation-invalid"
    with pytest.raises(SystemExit, match="valid JSON"):
        idempotency_cmd._cmd_idempotency(invalid_args)


def test_server_bootstrap_resolves_secrets_before_auth_and_lifespan_configuration() -> None:
    class Provider:
        def resolve_env(self, mapping):
            return {
                destination: type("Secret", (), {"value": "resolved", "version_id": "v1"})()
                for destination in mapping
            }

    settings = bootstrap_settings(
        {
            "RECALL_TRANSPORT": "stdio",
            "RECALL_AWS_SECRET_MAPPING": '{"RECALL_SERVING_DSN":"database-secret"}',
        },
        secret_provider=Provider(),
    )

    server = build_server(settings)

    assert settings.serving_dsn == "resolved"
    assert settings.secret_versions == {"RECALL_SERVING_DSN": "v1"}
    assert server.name == "recall_mcp"
