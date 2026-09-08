from __future__ import annotations

import json
from pathlib import Path

import pytest

from recall.ops.secrets import AwsSecretsManagerProvider
from recall_mcp.settings import Settings, bootstrap_settings, render_env_example


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
