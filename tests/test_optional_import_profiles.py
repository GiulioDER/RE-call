from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def checker():
    path = Path(__file__).parents[1] / "scripts" / "check_optional_imports.py"
    spec = importlib.util.spec_from_file_location("check_optional_imports", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_required_profiles_have_independent_contracts(checker) -> None:
    assert {
        "core",
        "fastembed",
        "mcp",
        "agent",
        "desktop",
        "langchain",
        "llamaindex",
    } <= set(checker.PROFILE_IMPORTS)
    assert len(checker.PROFILE_IMPORTS) == len(set(checker.PROFILE_IMPORTS))


def test_core_profile_imports_in_the_current_environment(checker) -> None:
    assert checker.verify_profile("core") == []


def test_llamaindex_profile_checks_the_guarded_external_host(checker) -> None:
    failures = checker.verify_profile("llamaindex")
    if failures:
        assert all("llama-index-core" in failure for failure in failures), failures
