"""Integer RECALL_* knobs read at server import time must be validated (ENV-002).

`recall_mcp.server` reads RECALL_PORT / RECALL_POOL_SIZE / RECALL_STATEMENT_TIMEOUT_MS with a
bare ``int()`` at import. A typo then crashes with ``invalid literal for int()`` that names no
variable (unlike the deliberately-validated RECALL_TRANSPORT right beside them), and nothing
bounds-checks the value: a negative RECALL_STATEMENT_TIMEOUT_MS reaches ``SET statement_timeout``
and 0 silently disables the pool-exhaustion cap the knob exists to enforce.

These are import-time reads, so they are exercised by importing the module in a subprocess with
the env under test — the honest path, not a re-implementation of the parsing.
"""
from __future__ import annotations

import inspect
import os
import subprocess
import sys

import pytest

from recall_mcp import server


def _import_server_with(**env_overrides: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.update({k: str(v) for k, v in env_overrides.items()})
    return subprocess.run(
        [sys.executable, "-c", "import recall_mcp.server"],
        capture_output=True,
        text=True,
        env=env,
    )


def test_import_succeeds_with_valid_env():
    r = _import_server_with()
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize("var", ["RECALL_PORT", "RECALL_POOL_SIZE", "RECALL_STATEMENT_TIMEOUT_MS"])
def test_non_int_knob_is_rejected_with_a_named_message(var):
    r = _import_server_with(**{var: "not-an-int"})
    assert r.returncode != 0, f"{var}=not-an-int should fail import"
    # Not just any ValueError: the message must name the variable (the pre-fix int() error says
    # only "invalid literal for int() with base 10: 'not-an-int'").
    assert f"{var}=" in r.stderr and "is not an integer" in r.stderr, r.stderr[-600:]


@pytest.mark.parametrize(
    "var,bad",
    [
        ("RECALL_PORT", "70000"),               # above the 65535 TCP maximum
        ("RECALL_POOL_SIZE", "0"),              # a zero-connection pool is nonsensical
        ("RECALL_STATEMENT_TIMEOUT_MS", "-5"),  # negative reaches SET statement_timeout
        ("RECALL_STATEMENT_TIMEOUT_MS", "0"),   # 0 disables the pool-exhaustion cap (fail-open)
    ],
)
def test_out_of_range_knob_is_rejected_at_import(var, bad):
    r = _import_server_with(**{var: bad})
    assert r.returncode != 0, f"{var}={bad} should be rejected at import, not accepted"
    assert f"{var}=" in r.stderr and "out of range" in r.stderr, r.stderr[-600:]


def test_transport_security_settings_follow_resource_url():
    settings = server._transport_security_settings("https://recall.example.com:8443")

    assert settings.enable_dns_rebinding_protection is True
    assert settings.allowed_hosts == ["recall.example.com:8443"]
    assert settings.allowed_origins == ["https://recall.example.com:8443"]


def test_streamable_http_run_passes_transport_security(monkeypatch):
    calls = {}

    class FakeServer:
        def run(self, **kwargs):
            calls.update(kwargs)

    monkeypatch.setattr(server, "mcp", FakeServer())
    monkeypatch.setattr(server, "TRANSPORT", "streamable-http")
    monkeypatch.setattr(server, "HTTP_HOST", "0.0.0.0")
    monkeypatch.setattr(server, "HTTP_PORT", 9000)
    monkeypatch.setenv("RECALL_AUTH_RESOURCE_URL", "https://recall.example.com")

    server.main()

    assert calls["transport"] == "streamable-http"
    assert calls["host"] == "0.0.0.0"
    assert calls["port"] == 9000
    assert calls["transport_security"].enable_dns_rebinding_protection is True
    assert calls["transport_security"].allowed_hosts == ["recall.example.com"]
    assert calls["transport_security"].allowed_origins == ["https://recall.example.com"]


# ---------------------------------------------------------------------------------------------
# require_effective_rls: an RLS-bypassing role is fatal for multi-tenant serving (SEC audit)
#
# The decision lives in a free function precisely so BOTH verdicts are reachable without an
# embedder, a database and a token registry. Inline in `_lifespan` the refusing half could only
# be reached by a full startup, which is why it was a warning for as long as it was.
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("multi_tenant", [True, False])
def test_an_effective_rls_role_produces_neither_warning_nor_refusal(multi_tenant):
    assert server.require_effective_rls(rls_effective=True, multi_tenant=multi_tenant) is None


def test_a_bypassing_role_only_warns_when_one_tenant_is_served():
    """stdio and single-tenant keep the warning on purpose.

    There is no second tenant to leak to, and `docker-compose.desktop.yml` ships the cluster
    superuser deliberately, so refusing here would break every local install to defend a boundary
    those deployments do not have.
    """
    warning = server.require_effective_rls(rls_effective=False, multi_tenant=False)
    assert warning is not None
    assert "bypasses row-level security" in warning


def test_a_bypassing_role_refuses_to_serve_multiple_tenants():
    with pytest.raises(RuntimeError) as excinfo:
        server.require_effective_rls(rls_effective=False, multi_tenant=True)
    message = str(excinfo.value)
    # Names the condition, the consequence, and the fix. An operator hitting this at boot has to
    # be able to act on it without reading the source; a bare "refusing to serve" cannot be acted
    # on and is the failure mode that gets worked around with the wrong knob.
    assert "bypasses row-level security" in message
    assert "unprivileged role" in message


def test_the_refusal_is_wired_to_the_registry_and_not_to_something_weaker():
    """The guard must be asked the MULTI-TENANT question, not a proxy for it.

    Asserted against the source because this is where a refactor silently defeats the gate: pass
    `multi_tenant=False`, or a condition that is merely correlated with multi-tenancy, and every
    test above still passes while the server goes back to warning. `registry is not None` holds
    exactly when a token registry was built, which `build_auth` permits only for HTTP.
    """
    source = inspect.getsource(server)
    assert "multi_tenant=registry is not None" in source, (
        "the RLS gate is no longer driven by whether a store registry exists; if the call site "
        "moved, re-derive the multi-tenant condition rather than relaxing this assertion"
    )


def test_the_refusal_releases_the_pool_it_was_holding():
    """A raise out of the lifespan after the registry is open must not strand its connections.

    Read from the module source rather than from `server._lifespan`: the lifespan is nested
    inside `build_server`, so it is not a module attribute and `inspect.getsource` cannot reach
    it. Slicing the source is the honest way to assert on a closure without building a server.
    """
    source = inspect.getsource(server)
    # The CALL site, not the definition: the definition's docstring also names the function.
    after_call = source.split("rls_effective=probe.check_rls_effective()", 1)[1]
    handler = after_call.split("if rls_warning is not None", 1)[0]
    # Comments stripped before the ordering check. The first version of this test matched the
    # word "raises" inside the explanatory comment and reported the close as coming too late,
    # which is a test failing on its own prose rather than on the code it guards.
    statements = "\n".join(
        ln for ln in handler.splitlines() if not ln.strip().startswith("#")
    )
    assert "registry.close()" in statements
    assert statements.index("registry.close()") < statements.index("raise")
