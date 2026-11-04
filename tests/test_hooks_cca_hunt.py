"""Regression tests for the 2026-11-04 DEEP CCA hunt of `recall_hooks` (run 20261104T131835Z).

Each test names the finding it pins, the invariant, and the production line it guards. Red proof:
every test here was run against the pre-fix snapshot of the hook modules on the test host
(`python -m cca_checks tautology verify`), failed in its own assertion, and passed on the fixed
tree. The run's L5.6 record holds each verdict; the pull request lists them.

None of these tests imports a symbol the fixes introduced, so each one reaches the pre-fix code.
"""
from __future__ import annotations

import importlib
import io
import json
import os
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any

import pytest

import recall_hooks
from recall_hooks import credentials as cred
from recall_hooks import hosted, prompt_time, write_time


@pytest.fixture(autouse=True)
def _isolate_config_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Never read or write the developer's real ~/.claude."""
    monkeypatch.setenv("RECALL_HOOK_CONFIG_HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()


def _cleanup_module(monkeypatch: pytest.MonkeyPatch, **env: str) -> Any:
    for name in (
        "CLAUDE_PID", "RECALL_MCP_CLIENT", "RECALL_MCP_SESSION_ID", "RECALL_MCP_PATTERN",
        "RECALL_MCP_PS_FILE", "RECALL_MCP_KILL_FILE", "RECALL_MCP_KILL_RC",
    ):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    import recall_hooks.mcp_cleanup as cleanup

    return importlib.reload(cleanup)


# --------------------------------------------------------------------------- P1


def test_stakes003_hosted_scan_does_not_upload_through_a_linked_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """STAKES-003. Invariant: `hosted.scan` returns only files whose real path is inside the root.

    Defect caught: the only guard was `path.is_symlink()` on the leaf file, but on Windows `**`
    walks into a directory junction (which is not a symlink to pathlib), so a memo outside the
    chosen memory directory joined the irreversible upload. Probed on Windows 11 with
    `mklink /J`: scan returned `project/linked/other.md`.

    A junction cannot be made on the Linux test host, and a POSIX symlinked directory is not
    walked by `**` there, so the walk is replaced by the listing a junction-following walk
    produces: a regular file under a directory link that leaves the root.
    Production line: the containment check in `recall_hooks.hosted.scan`.
    """
    root = tmp_path / "memory"
    root.mkdir()
    (root / "a.md").write_text("inside", encoding="utf-8")
    outside = tmp_path / "other-project"
    outside.mkdir()
    (outside / "secret.md").write_text("another project's memo", encoding="utf-8")
    try:
        os.symlink(outside, root / "linked", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("this platform cannot create a directory link without privileges")
    walked = [root / "a.md", root / "linked" / "secret.md"]
    monkeypatch.setattr(type(root), "glob", lambda self, _pattern: iter(walked))
    plan_names = sorted(hosted.scan([("project", root)]))
    assert "project/linked/secret.md" not in plan_names, plan_names
    assert "project/a.md" in plan_names


def test_stakes003_a_memory_root_that_is_itself_a_link_is_not_synced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """STAKES-003, root half. Invariant: `memory_dirs` refuses a memory ROOT that redirects
    elsewhere even when pathlib does not call it a symlink, and still returns an ordinary root
    whose project is reached through a linked ancestor.

    Defect caught: the root check was `not local.is_symlink()`, which is False for a Windows
    junction, and `hosted.scan` resolves the root through the link, so everything under the
    junction target was uploaded. A junction cannot be made on the Linux test host, so a
    directory symlink stands in with `is_symlink` reporting False, as pathlib does for a junction.
    Production line: `linked()` in `recall_hooks.memory_dirs`.
    """
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    try:
        os.symlink(outside, project / "memory", target_is_directory=True)
        os.symlink(tmp_path / "real", tmp_path / "via-link", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("this platform cannot create a directory link without privileges")
    (tmp_path / "real" / "memory").mkdir(parents=True)
    monkeypatch.setattr(type(project), "is_symlink", lambda self: False)
    roots = recall_hooks.memory_dirs(str(project))
    assert [r for r in roots if r[0] == "worktree"] == [], roots
    # Control: an ordinary memory directory, reached through a linked ancestor, is still synced.
    control = recall_hooks.memory_dirs(str(tmp_path / "via-link"))
    assert ("worktree", tmp_path / "via-link" / "memory") in control, control


def test_sec001_sec002_hook_children_never_resolve_a_bare_name_from_the_cwd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SEC-001 and SEC-002. Invariant: before any handler runs, `main` sets
    `NoDefaultCurrentDirectoryInExePath=1`, so Windows CreateProcess does not search the session
    directory for `git`, `powershell` or `taskkill`.

    Defect caught: the auditor brief starts `git`, and SessionEnd cleanup starts `powershell` and
    `taskkill`, by bare name, and a `git.exe` committed to a repository ran in their place
    (probed on Windows 11 with a renamed copy of hostname.exe). Nothing in the hook set it.
    Production line: the `os.environ[...]` assignment at the top of `recall_hooks.main`.
    """
    monkeypatch.delenv("NoDefaultCurrentDirectoryInExePath", raising=False)
    seen: dict[str, Any] = {}

    def record(event: str):
        def handler(_payload: dict[str, Any]) -> int:
            seen[event] = os.environ.get("NoDefaultCurrentDirectoryInExePath")
            return 0
        return handler

    import recall_hooks.auditor_brief as brief

    monkeypatch.setattr(recall_hooks, "session_end", record("session-end"))
    monkeypatch.setattr(brief, "pre_agent", record("pre-agent"))
    for event in ("session-end", "pre-agent"):
        monkeypatch.setattr(sys, "stdin", io.StringIO("{}"))
        assert recall_hooks.main([event]) == 0
    assert seen == {"session-end": "1", "pre-agent": "1"}


# --------------------------------------------------------------------------- MCP cleanup


def test_stakes001_clear_does_not_close_the_live_clients_mcp_servers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """STAKES-001. Invariant: a SessionEnd whose reason is `clear` or `resume` closes no transport.

    Defect caught: `session_end` never read the reason, and the installer registers SessionEnd for
    `clear|resume`, so `/clear` killed the MCP servers of a client that keeps running.
    Production line: the reason check before `close_own_mcp_transports` in `session_end`.
    """
    table = tmp_path / "ps.txt"
    table.write_text("100 1 claude\n101 100 python -m recall_mcp.server\n", encoding="utf-8")
    killed = tmp_path / "killed.txt"
    _cleanup_module(monkeypatch, RECALL_MCP_PS_FILE=str(table), RECALL_MCP_KILL_FILE=str(killed))
    monkeypatch.setattr(recall_hooks, "_index_and_refresh", lambda _payload: 0)
    for reason in ("clear", "resume"):
        recall_hooks.session_end({"reason": reason, "cwd": str(tmp_path), "_client_pid": "100"})
    assert not killed.exists() or killed.read_text(encoding="utf-8") == ""
    # Control: a real exit still closes it.
    recall_hooks.session_end({"reason": "logout", "cwd": str(tmp_path), "_client_pid": "100"})
    assert killed.read_text(encoding="utf-8").split() == ["101"]


def test_stakes002_pid_one_is_no_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """STAKES-002. Invariant: a client pid of 1 (an orphaned hook's parent) selects nothing by
    descent; the session ID decides instead.

    Defect caught: every process descends from pid 1, so a reparented Codex SessionEnd hook
    selected and killed every session's MCP transports.
    Production line: the pid validation in `close_own_mcp_transports`.
    """
    table = tmp_path / "ps.txt"
    table.write_text(
        "1 0 init\n"
        "100 1 codex\n101 100 python -m recall_mcp.server RECALL_MCP_SESSION_ID=aaa\n"
        "200 1 claude\n201 200 python -m recall_mcp.server RECALL_MCP_SESSION_ID=bbb\n",
        encoding="utf-8",
    )
    killed = tmp_path / "killed.txt"
    cleanup = _cleanup_module(
        monkeypatch, RECALL_MCP_PS_FILE=str(table), RECALL_MCP_KILL_FILE=str(killed)
    )
    cleanup.close_own_mcp_transports("1", session_id="aaa", cwd=str(tmp_path))
    assert killed.read_text(encoding="utf-8").split() == ["101"]


def test_stakes002_a_reparented_codex_hook_passes_no_client_pid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """STAKES-002, Codex half. Invariant: when the hook's parent changed after launch (the client
    exited while the hook read stdin), the Codex adapter passes no client pid.

    Production line: the launch-parent comparison in `recall_hooks.codex.main`.
    """
    import recall_hooks.codex as codex

    parents = iter([500, 1, 1, 1])
    monkeypatch.setattr(codex.os, "getppid", lambda: next(parents))
    seen: dict[str, Any] = {}

    def fake_main() -> int:
        seen.update(json.loads(sys.stdin.read()))
        return 0

    monkeypatch.setattr(recall_hooks, "main", fake_main)
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({"session_id": "aaa"})))
    assert codex.main(["session-end"]) == 0
    assert seen.get("_client_pid") == ""
    assert seen.get("_client_session_id") == "aaa"


def test_bug003_an_undecodable_process_table_is_unreadable_not_a_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BUG-003. Invariant: `_process_table` decodes with replacement and treats a missing stdout
    as an unreadable table (None), never raising.

    Defect caught: Windows PowerShell pipes output in the OEM code page; one non-ASCII command
    line made the strict UTF-8 decode fail, `stdout` came back None, and `None.splitlines()`
    raised AttributeError, so cleanup closed nothing on that machine.
    Production line: `errors="replace"` and the `output is None` check in `_process_table`.
    """
    cleanup = _cleanup_module(monkeypatch)
    calls: list[dict[str, Any]] = []

    def fake_run(*_args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return types.SimpleNamespace(stdout=None)

    monkeypatch.setattr(cleanup.subprocess, "run", fake_run)
    raised = False
    rows: Any = "not called"
    try:
        rows = cleanup._process_table()
    except Exception:  # noqa: BLE001 - asserted below, after the decode mode
        raised = True
    assert calls and calls[0].get("errors") == "replace", calls
    assert not raised, "an undecodable table must read as unknown, not raise"
    assert rows is None


def test_deploy001_the_windows_process_table_has_time_to_finish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DEPLOY-001. Invariant: the Windows process listing is given longer than it takes.

    Defect caught: Get-CimInstance Win32_Process took 2.2 to 9 s on a Windows 11 workstation
    (measured 2026-11-04) against a 2 s timeout, so cleanup reported `unknown` every time.
    Production line: the `timeout=` of the PowerShell `subprocess.run` in `_process_table`.
    """
    cleanup = _cleanup_module(monkeypatch)
    calls: list[dict[str, Any]] = []

    def fake_run(*_args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return types.SimpleNamespace(stdout="")

    monkeypatch.setattr(cleanup.subprocess, "run", fake_run)
    monkeypatch.setattr(cleanup.os, "name", "nt")
    try:
        cleanup._process_table()
    finally:
        monkeypatch.undo()
    assert calls and calls[0]["timeout"] >= 5


def test_bug005_the_relay_does_not_inherit_the_hooks_pipes_on_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """BUG-005. Invariant: the detached relay is spawned with `close_fds=True` on every platform.

    Defect caught: on Windows `close_fds=False` let the relay inherit the hook's stdout and stderr
    pipe handles despite DEVNULL, holding the client's pipe open for the relay's whole life
    (measured: EOF 4.4 s after the hook exited).
    Production line: the `close_fds=` argument of `subprocess.Popen` in `relay._spawn`.
    """
    from recall_hooks import relay

    seen: dict[str, Any] = {}

    class Stop(Exception):
        pass

    def fake_popen(*_args: Any, **kwargs: Any) -> Any:
        seen.update(kwargs)
        raise Stop

    monkeypatch.setattr(relay.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(relay.os, "name", "nt")
    try:
        with pytest.raises(Stop):
            relay._spawn(tmp_path / "state.json", "token", "fingerprint", initialize=False)
    finally:
        monkeypatch.undo()
    assert seen.get("close_fds") is True


# --------------------------------------------------------------------------- config


def test_bug001_a_bom_or_non_utf8_config_neither_disables_nor_crashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """BUG-001. Invariant: a UTF-8 BOM config loads; a config in another encoding loads as `{}`
    instead of raising out of a fail-open hook.

    Defect caught: a BOM loaded as `{}` (the integration silently off) and UTF-16 raised
    UnicodeDecodeError out of SessionStart, which runs without a guard.
    Production line: `load_config` (encoding and the caught exceptions).
    """
    path = tmp_path / "recall-hook.json"
    monkeypatch.setattr(recall_hooks, "config_path", lambda: path)
    path.write_bytes(b"\xef\xbb\xbf" + json.dumps({"dsn": "postgresql://h/db"}).encode())
    assert recall_hooks.load_config().get("dsn") == "postgresql://h/db"
    path.write_bytes(json.dumps({"dsn": "x"}).encode("utf-16"))
    assert recall_hooks.load_config() == {}


def test_dat003_a_stats_refresh_does_not_revert_a_concurrent_config_edit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DAT-003. Invariant: `refresh_stats` writes the count into the config as it is NOW, and
    drops it when the corpus identity changed meanwhile.

    Defect caught: it wrote back the snapshot loaded before indexing, so an operator's
    `general_knowledge` switch or a reinstall made in that window was silently reverted.
    Production line: the re-read before `_save_config` in `refresh_stats`.
    """
    path = tmp_path / "recall-hook.json"
    monkeypatch.setattr(recall_hooks, "config_path", lambda: path)
    snapshot = {"dsn": "postgresql://h/db", "tenant": "t", "chunks": 1}
    path.write_text(json.dumps(snapshot), encoding="utf-8")

    class Connection:
        def __enter__(self) -> "Connection":
            return self

        def __exit__(self, *_exc: Any) -> None:
            return None

        def execute(self, *_args: Any) -> Any:
            return types.SimpleNamespace(fetchone=lambda: (7,))

    def connect(*_args: Any, **_kwargs: Any) -> Connection:
        # The edit lands while the count is being taken.
        path.write_text(json.dumps({**snapshot, "general_knowledge": {"enabled": False}}),
                        encoding="utf-8")
        return Connection()

    class SQL:
        def __init__(self, text: str) -> None:
            self.text = text

        def format(self, *_args: Any) -> "SQL":
            return self

    sql = types.ModuleType("psycopg.sql")
    sql.SQL = SQL  # type: ignore[attr-defined]
    sql.Identifier = str  # type: ignore[attr-defined]
    psycopg = types.ModuleType("psycopg")
    psycopg.connect = connect  # type: ignore[attr-defined]
    psycopg.sql = sql  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "psycopg", psycopg)
    monkeypatch.setitem(sys.modules, "psycopg.sql", sql)

    assert recall_hooks.refresh_stats(dict(snapshot)) == 7
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved.get("general_knowledge") == {"enabled": False}
    assert saved["chunks"] == 7


def test_bug008_a_string_false_turns_the_retrieval_hooks_off() -> None:
    """BUG-008. Invariant: `enabled` is honoured only as a real bool; any other present value
    disables the feature, as `general_knowledge.settings` already does.

    Defect caught: `bool("false")` is True, so a hand-written `"false"` kept both hooks on.
    Production lines: the `enabled` entries of `write_time.settings` and `prompt_time.settings`.
    """
    assert write_time.settings({"write_time": {"enabled": "false"}})["enabled"] is False
    assert prompt_time.settings({"prompt_time": {"enabled": "false"}})["enabled"] is False
    assert write_time.settings({})["enabled"] is True
    assert prompt_time.settings({"prompt_time": {"enabled": True}})["enabled"] is True


def test_env002_connect_timeout_never_becomes_psycopgs_130_second_default() -> None:
    """ENV-002. Invariant: a connect timeout below 1 s, non-numeric or non-finite becomes the 2 s
    default; any other value is kept as configured.

    Defect caught: psycopg truncates to whole seconds and turns 0 or less into 130 s, so a
    configured 0.5 hung every tool call until the 5 s client timeout, with no cooldown written.
    Production line: the `connect_timeout` parsing in `write_time.settings`.
    """
    for raw in (0.5, 0, -3, "soon", float("inf")):
        value = write_time.settings({"write_time": {"connect_timeout": raw}})["connect_timeout"]
        assert value == 2.0, (raw, value)
    for raw in (1.5, 4):
        assert write_time.settings({"write_time": {"connect_timeout": raw}})["connect_timeout"] == raw


def test_perf003_a_cold_query_killed_mid_flight_still_leaves_a_cooldown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PERF-003. Invariant: a cold-mode query that never returns (the client kills the process at
    its 5 s timeout) still leaves the cooldown stamp, so the next tool call does not pay again.

    Defect caught: the stamp was written only in the `except Exception` branch, which a killed
    process never reaches, and the cold worst case (start, import, 2 s connect, 5 s statement)
    is past the 5 s budget. KeyboardInterrupt stands in for the kill: it is not an Exception.
    Production line: the `_mark_inflight` before the cold `search` in `write_time.context_for`,
    and the in-flight branch of `write_time._in_cooldown`.
    """
    path = tmp_path / "recall-hook.json"
    path.write_text(json.dumps({
        "dsn": "postgresql://h/db", "project_root": str(tmp_path),
        "write_time": {"connection_mode": "cold"},
    }), encoding="utf-8")
    monkeypatch.setattr(write_time, "load_config", lambda: json.loads(path.read_text("utf-8")))

    def killed(*_args: Any) -> Any:
        raise KeyboardInterrupt

    monkeypatch.setattr(write_time, "search", killed)
    payload = {"tool_name": "Write", "cwd": str(tmp_path),
               "tool_input": {"content": "a draft long enough to query the corpus"}}
    with pytest.raises(KeyboardInterrupt):
        write_time.context_for(payload)
    # A query that may still be running does not silence a concurrent tool call ...
    assert not write_time._in_cooldown()
    # ... but once no live hook can still be holding it, the abandoned query is a failure.
    later = time.time() + 7.0
    monkeypatch.setattr(write_time.time, "time", lambda: later)
    assert write_time._in_cooldown()


# --------------------------------------------------------------------------- credentials

CONFIG = {
    "account": "someone@example.test",
    "auth": {"token_url": "https://id.example.test/token", "client_id": "recall-mcp"},
}


class _Ring:
    def __init__(self) -> None:
        self.store: dict[tuple[str, str], str] = {}
        self.set_fails = False

    def get_password(self, service: str, account: str) -> str | None:
        return self.store.get((service, account))

    def set_password(self, service: str, account: str, value: str) -> None:
        if self.set_fails:
            raise RuntimeError("keychain refused the write")
        self.store[(service, account)] = value

    def delete_password(self, service: str, account: str) -> None:
        del self.store[(service, account)]


def test_stakes005_a_failed_keychain_write_does_not_leave_a_stale_token_in_front(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """STAKES-005. Invariant: after `write_refresh_token` falls back to the file, reading returns
    the token just written, not an older keychain entry.

    Defect caught: the fallback left the old keychain entry in place, and `read_refresh_token`
    prefers the keychain, so a rotated token was shadowed by the dead one.
    Production line: the `delete_password` on the failed-write path of `write_refresh_token`.
    """
    ring = _Ring()
    monkeypatch.setattr(cred, "_keyring", lambda: ring)
    assert cred.write_refresh_token(CONFIG, "R1") == "keyring"
    ring.set_fails = True
    assert cred.write_refresh_token(CONFIG, "R2") == "file"
    assert cred.read_refresh_token(CONFIG) == "R2"


def test_dat002_concurrent_refreshes_spend_the_refresh_token_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DAT-002. Invariant: two callers that both find the access token stale redeem the refresh
    token once; the second reuses the first one's result.

    Defect caught: nothing serialized the refresh, so both posted the same refresh token, which a
    rotating provider refuses the second time (and with reuse detection revokes the family).
    Production line: the lock and cache re-check around `refresh` in `access_token`.
    """
    monkeypatch.setattr(cred, "_keyring", lambda: None)
    cred.write_refresh_token(CONFIG, "R1")
    calls: list[str] = []
    barrier = threading.Barrier(2)

    def fake_refresh(config: dict[str, Any], token: str, *, now: float | None = None) -> Any:
        calls.append(token)
        time.sleep(0.3)
        return cred.Credential(access_token="A", expires_at=time.time() + 3600)

    monkeypatch.setattr(cred, "refresh", fake_refresh)
    errors: list[BaseException] = []

    def worker() -> None:
        barrier.wait()
        try:
            cred.access_token(CONFIG)
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    assert not errors, errors
    assert calls == ["R1"], calls


def test_doc005_logout_removes_a_token_stored_under_the_legacy_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DOC-005. Invariant: logout removes a refresh token stored under the bare account key that
    versions before the context-v1 key used.

    Defect caught: logout looked up only the new key, left the legacy plaintext token on disk and
    printed that there was nothing to remove.
    Production line: the key list in `credentials.logout`.
    """
    monkeypatch.setattr(cred, "_keyring", lambda: None)
    path = cred.fallback_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"someone@example.test": "LEGACY"}), encoding="utf-8")
    out = io.StringIO()
    assert cred.logout(CONFIG, out=out) == 0
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert "someone@example.test" not in stored
    assert "no stored credential" not in out.getvalue()
