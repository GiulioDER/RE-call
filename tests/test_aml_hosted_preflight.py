from __future__ import annotations

import json

from scripts.aml_hosted_preflight import inspect_environment


def prepared_environment() -> dict[str, str]:
    return {
        "RECALL_AML_GIT_COMMIT": "87fbaa94d1462ffd50b33919317cb0cef2423698",
        "OPENROUTER_API_KEY": "openrouter-secret-sentinel",
        "VOYAGE_API_KEY": "voyage-secret-sentinel",
    }


def launch_environment() -> dict[str, str]:
    return {
        **prepared_environment(),
        "RECALL_AML_DATABASE_URL": "postgresql://secret-sentinel",
        "RECALL_AML_API_KEY": "evaluation-secret-sentinel",
    }


def test_prepare_allows_only_admission_values_to_be_pending():
    result = inspect_environment(prepared_environment(), "prepare")

    assert result["ready"] is True
    checks = result["checks"]
    assert checks["OPENROUTER_API_KEY"] == {"required": True, "state": "present"}
    assert checks["RECALL_AML_DATABASE_URL"] == {
        "required": False,
        "state": "pending_admission",
    }
    assert checks["RECALL_AML_API_KEY"] == {
        "required": False,
        "state": "pending_admission",
    }


def test_launch_requires_every_runtime_value_and_does_not_accept_openai_substitution():
    complete = launch_environment()
    assert inspect_environment(complete, "launch")["ready"] is True

    for name in complete:
        incomplete = {**complete, name: ""}
        if name == "OPENROUTER_API_KEY":
            incomplete["OPENAI_API_KEY"] = "legacy-key-must-not-substitute"
        result = inspect_environment(incomplete, "launch")
        assert result["ready"] is False, name
        assert result["checks"][name]["state"] == "missing", name


def test_preflight_output_never_contains_configuration_values():
    env = launch_environment()
    rendered = json.dumps(inspect_environment(env, "launch"), sort_keys=True)

    assert all(value not in rendered for value in env.values())
