"""DEF-45 and DEF-46 of the deferred pass over the MCP hunt (#1164), owner decisions 2026-11-04.

DEF-45: stdio authenticates nobody, so a leftover RECALL_AUTH_TOKENS_FILE there is warned about
and never loaded; before, `build_auth` loaded it first, and a production environment (or a file
mounted only for the HTTP deployment) stopped a stdio server that would never read it.

DEF-46: the HTTP Host/Origin allowlist was one raw netloc from RECALL_AUTH_RESOURCE_URL. It is
now stripped and lowercased, accepts the scheme's default port written or omitted, and takes
extra names from RECALL_ALLOWED_HOSTS.

Red proofs are recorded in the pull request that adds this file.
"""

from __future__ import annotations

import json
import logging

import pytest

import recall_mcp.server as server
from recall_mcp.auth import AuthConfigError
from recall_mcp.server_auth import build_auth


def test_a_production_stdio_server_starts_with_a_leftover_token_file(tmp_path, caplog) -> None:
    """Invariant: on stdio, a token file is warned about, never loaded, under any RECALL_ENV.

    Failure mode caught: the file was loaded before the transport check, so RECALL_ENV=production
    refused it and the stdio server, which uses no tokens, did not start.
    """
    tokens = tmp_path / "tokens.json"
    tokens.write_text(json.dumps({"principals": [{"name": "a", "token": "t" * 40}]}), encoding="utf-8")
    env = {"RECALL_ENV": "production", "RECALL_AUTH_TOKENS_FILE": str(tokens)}
    with caplog.at_level(logging.WARNING):
        assert build_auth("stdio", env=env) == (None, None, None)
    assert any("tokens are unused" in r.message for r in caplog.records)


def test_a_stdio_server_starts_when_the_token_file_is_not_mounted(tmp_path, caplog) -> None:
    """Invariant: a token file mounted only for the HTTP deployment does not stop stdio."""
    env = {"RECALL_AUTH_TOKENS_FILE": str(tmp_path / "absent.json")}
    with caplog.at_level(logging.WARNING):
        assert build_auth("stdio", env=env) == (None, None, None)
    assert any("tokens are unused" in r.message for r in caplog.records)


def test_http_still_refuses_the_token_file_in_production(tmp_path) -> None:
    """Control: the production refusal still holds wherever the tokens would be used."""
    tokens = tmp_path / "tokens.json"
    tokens.write_text(json.dumps({"principals": [{"name": "a", "token": "t" * 40}]}), encoding="utf-8")
    env = {"RECALL_ENV": "production", "RECALL_AUTH_TOKENS_FILE": str(tokens),
           "RECALL_AUTH_ISSUER_URL": "https://recall.example.com",
           "RECALL_AUTH_RESOURCE_URL": "https://recall.example.com"}
    with pytest.raises(AuthConfigError):
        build_auth("streamable-http", env=env)


def test_the_host_allowlist_is_normalised_and_accepts_the_default_port() -> None:
    """Invariant: a padded or mixed-case resource URL still yields the Host a client sends.

    Failure mode caught: the raw netloc was the only entry, so `Host: recall.example.com` was
    refused when the URL said `:443`, had different case, or carried env-file whitespace.
    """
    settings = server._transport_security_settings(" https://Recall.Example.com:443 \n")
    assert settings.allowed_hosts == ["recall.example.com", "recall.example.com:443"]
    assert "https://recall.example.com" in settings.allowed_origins


def test_extra_allowed_hosts_are_added_and_validated() -> None:
    """Invariant: RECALL_ALLOWED_HOSTS adds names; a value that is not host[:port] refuses."""
    settings = server._transport_security_settings(
        "https://recall.example.com", "recall.internal:8000, Recall-Svc ,"
    )
    assert settings.allowed_hosts == [
        "recall.example.com", "recall.example.com:443", "recall.internal:8000", "recall-svc",
    ]
    assert "https://recall.internal:8000" in settings.allowed_origins
    for bad in ("https://x.example", "user@host", "host:"):
        with pytest.raises(AuthConfigError, match="RECALL_ALLOWED_HOSTS"):
            server._transport_security_settings("https://recall.example.com", bad)
