"""Three small hook defects the 2026-11-04 CCA verifiers noted beside their findings.

Fixed at the owner's request after PR #1187: the code half of ENV-010 (an empty
`RECALL_HOOK_CLIENT`), the credentials fallback readers raising on bytes that are not UTF-8, and
an existing but unreadable nearest `.mcp.json` being walked past.

Red proof: each test was run on the test host against the pre-fix modules, failed in its own
assertion, and passed on the fix; the pull request records the run. No test imports a symbol the
fix introduced. Exceptions are caught and asserted as booleans, so no exception object or type
name reaches the output of a red run.
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import pytest

from recall_hooks import credentials as cred
from recall_hooks import general_knowledge


def _raises(call: Any) -> tuple[Any, bool]:
    try:
        return call(), False
    except Exception:  # noqa: BLE001 - asserted by the caller
        return None, True


def test_env010_an_empty_client_name_is_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """ENV-010, code half. Invariant: an empty or blank `RECALL_HOOK_CLIENT` logs the default
    client, `claude`, as every other variable this package reads treats blank as unset.

    Defect caught: `os.environ.get(name, "claude")` returned "" for a set-but-empty variable,
    so events were logged under an empty client name.
    Production line: `general_knowledge.client`.
    """
    for value in ("", "  "):
        monkeypatch.setenv("RECALL_HOOK_CLIENT", value)
        assert general_knowledge.client() == "claude", repr(value)
    monkeypatch.setenv("RECALL_HOOK_CLIENT", "codex")
    assert general_knowledge.client() == "codex"


def test_a_fallback_credential_file_that_is_not_utf8_does_not_raise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Invariant: the credential readers treat a fallback file that is not UTF-8 like a corrupt
    one: reading finds no token, a new login replaces the file, and logout finishes.

    Defect caught: they caught JSONDecodeError but not UnicodeDecodeError, so such a file made
    `read_refresh_token`, `write_refresh_token` (and so `login`) and `logout` raise.
    Production lines: the `except` clauses of the fallback readers in `recall_hooks.credentials`.
    """
    monkeypatch.setenv("RECALL_HOOK_CONFIG_HOME", str(tmp_path))
    monkeypatch.setattr(cred, "_keyring", lambda: None)
    config = {"account": "a@example.test",
              "auth": {"token_url": "https://id.example.test/token", "client_id": "c"}}
    cred.fallback_path().write_bytes(b"\xff\xfe not utf-8")

    token, raised = _raises(lambda: cred.read_refresh_token(config))
    assert not raised and token is None, "reading an undecodable fallback file failed"
    where, raised = _raises(lambda: cred.write_refresh_token(config, "NEW"))
    assert not raised and where == "file", "writing over an undecodable fallback file failed"
    assert cred.read_refresh_token(config) == "NEW"
    cred.fallback_path().write_bytes(b"\xff\xfe not utf-8")
    _result, raised = _raises(lambda: cred.logout(config, out=open(tmp_path / "out.txt", "w")))
    assert not raised, "logout failed on an undecodable fallback file"


def test_an_unreadable_nearest_mcp_config_is_a_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Invariant: a nearest `.mcp.json` that exists but cannot be read stops the identity walk,
    exactly as an unparsable one does; only an absent file lets the walk go on.

    Defect caught: any OSError (no permission, or a directory named `.mcp.json`) walked on to an
    enclosing checkout and adopted ITS session ID as this session's cleanup identity.
    Production line: the `except` clauses in `mcp_cleanup._identity_from_config`.
    A directory stands in for an unreadable file: reading one raises OSError on every platform,
    and permission bits cannot make a file unreadable to a privileged test runner.
    """
    for name in ("CLAUDE_PID", "RECALL_MCP_CLIENT", "RECALL_MCP_SESSION_ID"):
        monkeypatch.delenv(name, raising=False)
    import recall_hooks.mcp_cleanup as cleanup

    cleanup = importlib.reload(cleanup)
    outer = tmp_path / "main"
    inner = outer / "worktrees" / "w"
    inner.mkdir(parents=True)
    (outer / ".mcp.json").write_text(json.dumps({"cmd": "RECALL_MCP_SESSION_ID=outer"}), encoding="utf-8")
    (inner / ".mcp.json").mkdir()
    identity, raised = _raises(lambda: cleanup._identity_from_config(str(inner)))
    assert not raised and identity == ("", ""), "an unreadable nearest config was walked past"
    # Control: with no nearest config at all, the walk still reaches the enclosing one.
    (inner / ".mcp.json").rmdir()
    assert cleanup._identity_from_config(str(inner))[0] == "outer"
