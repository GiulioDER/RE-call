from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from recall.ops.secrets import AwsSecretsManagerProvider
from recall_mcp.settings import (
    ENVIRONMENT_SCHEMA,
    Settings,
    activate_runtime_settings,
    bootstrap_settings,
    render_env_example,
    reset_runtime_settings,
    runtime_environment,
)


def test_settings_is_immutable_and_parses_the_canonical_region() -> None:
    settings = Settings.from_env({"RECALL_AWS_REGION": "eu-west-1"})

    assert settings.aws_region == "eu-west-1"
    with pytest.raises((AttributeError, TypeError)):
        settings.port = 9000  # type: ignore[misc]


@pytest.mark.parametrize("destination", ["RECALL_ENV", "RECALL_TRUST_MODE", "RECALL_PROVIDER_SECRET"])
def test_secret_mapping_rejects_security_and_legacy_destinations(destination: str) -> None:
    raw = json.dumps({destination: "secret/name"})

    with pytest.raises(ValueError, match="forbidden destination"):
        Settings.from_env({"RECALL_AWS_SECRET_MAPPING": raw})


def test_bootstrap_resolves_secrets_once_without_mutating_process_environment(monkeypatch) -> None:
    class Secret:
        def __init__(self, value: str, version_id: str) -> None:
            self.value = value
            self.version_id = version_id

    class Provider:
        calls = 0

        def resolve_env(self, mapping):
            self.calls += 1
            return {destination: Secret("resolved", "v1") for destination in mapping}

    provider = Provider()
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    settings = bootstrap_settings(
        {
            "RECALL_AWS_SECRET_MAPPING": json.dumps(
                {"OPENROUTER_API_KEY": "recall/provider"}
            ),
            "RECALL_AWS_REGION": "eu-west-1",
        },
        secret_provider=provider,
    )

    assert provider.calls == 1
    assert settings.env["OPENROUTER_API_KEY"] == "resolved"
    assert settings.secret_versions == {"OPENROUTER_API_KEY": "v1"}


def test_bootstrap_does_not_mutate_the_real_process_environment(monkeypatch) -> None:
    """Secret resolution must stay inside the bootstrap snapshot.

    Invariant: calling ``bootstrap_settings`` without an explicit environment mapping leaves the
    real process environment byte-for-byte equivalent as a key/value mapping. Targeted production
    symbol: ``bootstrap_settings``. Red proof receipt: mutating ``source`` to ``os.environ`` at
    the secret assignment loop makes this node fail because ``OPENROUTER_API_KEY`` is added to the
    process environment; the production implementation is restored afterward.
    """

    class Secret:
        value = "resolved"
        version_id = "v1"

    class Provider:
        def resolve_env(self, mapping):
            return {destination: Secret() for destination in mapping}

    monkeypatch.setenv(
        "RECALL_AWS_SECRET_MAPPING",
        json.dumps({"OPENROUTER_API_KEY": "recall/provider"}),
    )
    monkeypatch.setenv("RECALL_AWS_REGION", "eu-west-1")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    before = dict(os.environ)

    settings = bootstrap_settings(secret_provider=Provider())

    assert settings.env["OPENROUTER_API_KEY"] == "resolved"
    assert dict(os.environ) == before


def test_runtime_environment_uses_the_active_immutable_snapshot() -> None:
    """Regression proof: mutating ``runtime_environment`` to return ``os.environ`` fails here.

    Invariant: request code must see the immutable bootstrap snapshot, not a later process
    environment mutation. Targeted production symbol: ``runtime_environment``.
    """
    settings = Settings.from_env({"RECALL_ROUTING_MODE": "active"})
    token = activate_runtime_settings(settings)
    try:
        assert runtime_environment()["RECALL_ROUTING_MODE"] == "active"
    finally:
        reset_runtime_settings(token)


def test_settings_snapshot_does_not_follow_later_process_environment_changes(monkeypatch) -> None:
    """Settings remain per-server snapshots after unrelated process environment changes."""
    settings = Settings.from_env({"RECALL_MCP_TOOLS": "search", "RECALL_TENANT": "tenant-a"})
    monkeypatch.setenv("RECALL_MCP_TOOLS", "recall_forget")
    monkeypatch.setenv("RECALL_TENANT", "tenant-b")

    assert settings.env["RECALL_MCP_TOOLS"] == "search"
    assert settings.tenant == "tenant-a"


def test_aws_provider_uses_the_canonical_region_name(monkeypatch) -> None:
    monkeypatch.setenv("RECALL_AWS_REGION", "eu-west-1")
    monkeypatch.setenv("AWS_REGION", "us-east-1")

    provider = AwsSecretsManagerProvider(client=object())
    assert provider._region_name == "eu-west-1"


def test_generated_environment_reference_comes_from_the_schema() -> None:
    assert "RECALL_AWS_REGION" in render_env_example()
    assert "OPENROUTER_API_KEY" in render_env_example()
    assert "RECALL_PROVIDER_SECRET" not in render_env_example()
    document = (Path(__file__).resolve().parents[1] / "docs" / "ENVIRONMENT_GENERATED.md").read_text(
        encoding="utf-8"
    )
    assert document.endswith(render_env_example() + "\n")


def _generated_environment_defaults() -> list[tuple[str, str]]:
    defaults: list[tuple[str, str]] = []
    for line in render_env_example().splitlines():
        match = re.fullmatch(r"# ([A-Z][A-Z0-9_]*)=(.*?)\s+# .+", line)
        if match and match.group(2):
            defaults.append((match.group(1), match.group(2)))
    assert defaults == [
        (spec.name, spec.default) for spec in ENVIRONMENT_SCHEMA if spec.default
    ]
    return defaults


@pytest.mark.parametrize("name, default", _generated_environment_defaults())
def test_every_generated_default_is_accepted_by_settings_parser(name: str, default: str) -> None:
    """Regression proof for generated configuration reaching the real bootstrap parser.

    Invariant: every nonempty default emitted by ``render_env_example`` is accepted by
    ``Settings.from_env``. Targeted production symbols: ``ENVIRONMENT_SCHEMA``,
    ``render_env_example``, and ``Settings.from_env``. Red proof receipt: changing the
    ``RECALL_MCP_STATELESS`` schema default from ``1`` to ``auto`` must fail this node with
    the parser's ``is not a boolean`` error; the production default is restored afterward.
    """
    try:
        Settings.from_env({name: default})
    except ValueError as exc:
        pytest.fail(f"generated default {name}={default!r} is rejected: {exc}")
