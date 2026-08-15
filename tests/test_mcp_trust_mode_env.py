"""`RECALL_TRUST_MODE` must reach the MCP server, because the docs tell users to set it.

`docs/USING_WITH_CLAUDE.md` names the variable three times, and line 224 says to set it "for local
work against a corpus with no published calibration". That is the documented first-run path: install,
register the server, point it at a corpus you just indexed, and search.

It did not work. `RECALL_TRUST_MODE` appeared **zero times** in the whole `recall_mcp` package and
`search_memory` was called with no `policy=`, so the service applied its strict default and every
`recall_search` against an uncalibrated corpus returned `INDEX_NOT_READY`. The variable the
documentation told users to reach for was read by nobody. The CLI honours it, which is what made the
gap survive: the same env var works in one entry point and is inert in the other.

The strict default is right and is not what these tests change. What they pin is that the documented
opt-in exists, that it is opt-in rather than a default, and that a typo cannot silently enable it.
"""

from __future__ import annotations

import importlib

import pytest

from recall.trust_policy import TrustPolicy


def _takes_policy(fn: object) -> bool:
    """Does this callable accept a `policy` parameter? Used to derive the gated set."""
    import inspect

    try:
        return "policy" in inspect.signature(fn).parameters  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


@pytest.fixture
def reload_server(monkeypatch: pytest.MonkeyPatch):
    """Reimport the server module under a chosen environment, then put it back.

    The module resolves its configuration at import time, so the environment has to be set before
    the import rather than patched afterwards.

    The teardown is the load-bearing half. `monkeypatch` restores the environment, but it cannot
    restore a module constant that a reload has already baked in, so without this the last
    parametrised case left `recall_mcp.server.TRUST_POLICY` RELAXED, with the variable unset, for
    every later test in the process. A fixture that leaks a relaxed trust gate into the rest of the
    suite is worse than the bug it was written to catch.
    """
    import recall_mcp.server as server

    def _load(**env: str):
        monkeypatch.delenv("RECALL_TRUST_MODE", raising=False)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        return importlib.reload(server)

    yield _load

    monkeypatch.delenv("RECALL_TRUST_MODE", raising=False)
    importlib.reload(server)
    assert server.TRUST_POLICY.strict, "teardown failed to restore a strict server module"


def test_the_server_exposes_a_trust_policy(reload_server) -> None:
    """The module must resolve a policy at all. Without one there is nothing to pass."""
    server = reload_server()
    assert hasattr(server, "TRUST_POLICY"), (
        "recall_mcp.server has no TRUST_POLICY. docs/USING_WITH_CLAUDE.md tells users to set "
        "RECALL_TRUST_MODE; something has to read it."
    )
    assert isinstance(server.TRUST_POLICY, TrustPolicy)


def test_unset_is_strict(reload_server) -> None:
    """Strict stays the default. A server that degraded by omission would degrade in production."""
    server = reload_server()
    assert server.TRUST_POLICY.strict is True


def test_the_documented_value_is_honoured(reload_server) -> None:
    """`development` must actually relax it, because that is what the docs promise."""
    server = reload_server(RECALL_TRUST_MODE="development")
    assert server.TRUST_POLICY.strict is False, (
        "RECALL_TRUST_MODE=development did not relax the policy, so the documented first-run "
        "path still returns INDEX_NOT_READY against an uncalibrated corpus."
    )


@pytest.mark.parametrize("value", ["dev", "developmnet", "1", "true", "yes", "development-mode"])
def test_a_near_miss_stays_strict(reload_server, value: str) -> None:
    """A misspelling or an abbreviation must not open the gate.

    This is the safety half. A near-miss that silently relaxed trust would be worse than one that
    refused: the operator believes they are strict, and the server is not.
    """
    server = reload_server(RECALL_TRUST_MODE=value)
    assert server.TRUST_POLICY.strict is True, (
        f"RECALL_TRUST_MODE={value!r} relaxed the policy. Only the token 'development' may."
    )


@pytest.mark.parametrize("value", ["development", "Development", "DEVELOPMENT", "  development  "])
def test_case_and_whitespace_are_tolerated(reload_server, value: str) -> None:
    """The token is matched after `strip().lower()`, and that is deliberate.

    A capital letter is not a typo. Refusing it would produce the worst outcome available here: a
    strict server that its operator believes is relaxed, debugged against an INDEX_NOT_READY that
    looks like a corpus problem. `TrustPolicy.from_env`'s docstring claimed "the exact string",
    which the code has never done; the docstring was corrected rather than the behaviour.
    """
    server = reload_server(RECALL_TRUST_MODE=value)
    assert server.TRUST_POLICY.strict is False


def test_the_search_tool_passes_the_policy_rather_than_defaulting() -> None:
    """Resolving a policy is useless if the call site still omits it.

    Asserted against the source, because reaching the tool body needs a live database and this
    property is about the call, not the result. The previous defect was exactly this shape: the
    service accepted a `policy` argument and the server never supplied one.
    """
    import ast
    import pathlib

    import recall_mcp.server as server

    source = pathlib.Path(server.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

    # The gated set is DERIVED from the service's own signatures, never listed here.
    #
    # Two earlier drafts of this guard were wrong in the same direction. The first named
    # `build_evidence`, which does not exist, so the loop passed vacuously. The second named
    # `search_memory` and `evidence_memory` as literals and was green while `reasoning_query` and
    # `reasoning_audit`, which also take `policy`, were still called without it: the guard covered
    # two of four call sites and reported success, which is precisely the defect it exists to catch.
    #
    # Deriving it means a fifth gated function cannot be added without this failing.
    import recall_mcp.service as service

    gated = {
        name
        for name in dir(service)
        if not name.startswith("_")
        and callable(getattr(service, name))
        and _takes_policy(getattr(service, name))
    }
    assert gated, "no policy-taking function found in recall_mcp.service; the guard lost its target"

    called = {
        node.func.id: node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    for name in sorted(gated & called.keys()):
        node = called[name]
        assert any(kw.arg == "policy" for kw in node.keywords), (
            f"{name} is called at recall_mcp/server.py:{node.lineno} without policy=, so it "
            "silently falls back to the strict default and RECALL_TRUST_MODE has no effect there."
        )
