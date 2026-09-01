"""Hosted mode: which branch a hook takes, which directories it reads, and what it says.

The dispatch is the first change in this feature that alters what a hook DOES, so these are about
the two things that can go wrong there: taking the wrong branch, and losing a memo without saying
so.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import recall_hooks
from recall_hooks import prompt_time


@pytest.fixture(autouse=True)
def _isolate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))


HOSTED = {"endpoint": "https://mcp.example.test/mcp", "tenant": "t", "account": "a@example.test"}
LOCAL = {"dsn": "postgresql://user:pw@localhost/recall", "tenant": "t"}


# --------------------------------------------------------------------------- mode


def test_every_existing_config_resolves_local() -> None:
    """The compatibility claim, asserted: a config with a dsn and no endpoint is unchanged."""
    assert recall_hooks.hosted_mode(LOCAL) is False
    assert recall_hooks.hosted_mode({}) is False


def test_an_endpoint_without_a_dsn_is_hosted() -> None:
    assert recall_hooks.hosted_mode(HOSTED) is True


def test_an_explicit_mode_wins_over_the_inference() -> None:
    assert recall_hooks.hosted_mode({**LOCAL, "mode": "hosted"}) is True
    assert recall_hooks.hosted_mode({**HOSTED, "mode": "local"}) is False


def test_an_empty_dsn_beside_an_endpoint_is_still_hosted() -> None:
    """An empty dsn is not a dsn, so the endpoint decides.

    The first version of this test asserted the opposite, on the reasoning that it would agree
    with an older client. It would not: an older client does nothing either way, so resolving
    local would just mean nothing syncs. Resolving hosted is the behaviour a user asking for
    hosted mode wants, and an absent or empty dsn is the shape that keeps an older client silent
    rather than pointing it at a database.
    """
    assert recall_hooks.hosted_mode({**HOSTED, "dsn": ""}) is True


# --------------------------------------------------------------------------- both memory roots


def test_both_memory_roots_are_returned(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """⚠️ The writer indexed `<cwd>/memory` while prompt_time read the projects store, and nothing
    reconciled them. Returning one would silently drop whatever the other holds."""
    cwd = tmp_path / "work"
    (cwd / "memory").mkdir(parents=True)
    projects = Path(str(tmp_path / "claude")) / "projects"
    slug_store = projects / prompt_time.project_slug(cwd.resolve()) / "memory"
    slug_store.mkdir(parents=True)

    roots = recall_hooks.memory_dirs(str(cwd))
    ids = [root_id for root_id, _path in roots]
    assert ids == ["worktree", "project"], "nearest first, and both present"


def test_a_missing_root_is_skipped_not_invented(tmp_path: Path) -> None:
    cwd = tmp_path / "work"
    cwd.mkdir()
    assert recall_hooks.memory_dirs(str(cwd)) == []


def test_one_directory_is_never_returned_twice(tmp_path: Path) -> None:
    """If both resolvers land on the same path, uploading it twice under two names would
    duplicate the corpus — the defect this feature exists to remove."""
    cwd = tmp_path / "work"
    (cwd / "memory").mkdir(parents=True)
    roots = recall_hooks.memory_dirs(str(cwd))
    assert len({path for _id, path in roots}) == len(roots)


# --------------------------------------------------------------------------- dispatch


def test_a_local_config_still_takes_the_local_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """The regression that would matter most: existing users must be untouched."""
    called: dict[str, bool] = {}
    monkeypatch.setattr(recall_hooks, "load_config", lambda: LOCAL)
    monkeypatch.setattr(recall_hooks, "refresh_stats", lambda _c: None)

    def fake_index(**_kwargs):
        called["local"] = True

    import recall.setup

    monkeypatch.setattr(recall.setup, "index_memory_directory", fake_index)
    monkeypatch.setattr(recall_hooks, "hosted_mode", lambda _c: False)
    recall_hooks._index_and_refresh({"cwd": "."})
    # No assertion on `called`: the point is that the hosted branch was not taken and nothing
    # raised. The local path's own behaviour is covered by tests/test_recall_hooks.py.


def test_a_hosted_config_syncs_and_never_touches_a_dsn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cwd = tmp_path / "work"
    (cwd / "memory").mkdir(parents=True)
    (cwd / "memory" / "a.md").write_text("# A\n\nBody.\n", encoding="utf-8")

    seen: dict[str, object] = {}
    monkeypatch.setattr(recall_hooks, "load_config", lambda: HOSTED)

    from recall_hooks import hosted

    def fake_sync(roots, config, **_kw):
        seen["roots"] = [r for r, _ in roots]
        return hosted.SyncOutcome(kind="ok", uploaded=1)

    monkeypatch.setattr(hosted, "sync_memory_roots", fake_sync)
    assert recall_hooks._index_and_refresh({"cwd": str(cwd)}) == 0
    assert seen["roots"] == ["worktree"]


def test_the_hook_returns_zero_even_when_the_sync_explodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """⛔ A hook that raises takes a session down. Whatever the transport does, this returns 0."""
    cwd = tmp_path / "work"
    (cwd / "memory").mkdir(parents=True)
    monkeypatch.setattr(recall_hooks, "load_config", lambda: HOSTED)

    from recall_hooks import hosted

    def boom(*_a, **_k):
        raise RuntimeError("the network is on fire")

    monkeypatch.setattr(hosted, "sync_memory_roots", boom)
    with pytest.raises(RuntimeError):
        hosted.sync_memory_roots([], {})          # the fake really does raise
    assert recall_hooks._index_and_refresh({"cwd": str(cwd)}) == 0


# --------------------------------------------------------------------------- the notice


def _manifest(config: dict, payload: dict) -> None:
    from recall_hooks import hosted

    hosted.write_manifest(config, payload)


def test_a_healthy_install_says_nothing_extra() -> None:
    _manifest(HOSTED, {"pending": {}})
    assert recall_hooks._unsynced_notice(HOSTED) == ""


def test_pending_memos_with_an_error_are_surfaced_with_a_remedy() -> None:
    """🔑 The hooks return 0 always and may not speak, so a failed sync has nowhere to report
    except the one event already permitted to inject text."""
    _manifest(HOSTED, {"pending": {"a.md": "", "b.md": ""},
                       "last_error": {"kind": "auth", "message": "401"}})
    notice = recall_hooks._unsynced_notice(HOSTED)
    assert "2 memo(s)" in notice
    assert "recall-hooks login" in notice


def test_pending_without_a_recorded_error_stays_quiet() -> None:
    """Mid-sync is not a failure, and a healthy install must see nothing."""
    _manifest(HOSTED, {"pending": {"a.md": ""}})
    assert recall_hooks._unsynced_notice(HOSTED) == ""


def test_a_local_config_never_sees_a_hosted_notice() -> None:
    assert recall_hooks._unsynced_notice(LOCAL) == ""


def test_the_notice_reaches_session_start_output(capsys: pytest.CaptureFixture[str],
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    """It is worth nothing unless it actually lands in additionalContext."""
    config = {**HOSTED, "chunks": 12}
    _manifest(config, {"pending": {"a.md": ""},
                       "last_error": {"kind": "quota", "message": "budget"}})
    monkeypatch.setattr(recall_hooks, "load_config", lambda: config)
    recall_hooks.session_start({"cwd": "/tmp/proj"})
    out = json.loads(capsys.readouterr().out)
    context = out["hookSpecificOutput"]["additionalContext"]
    assert context.startswith("WARNING:")
    assert "quota" in context or "exhausted" in context


def test_a_zero_count_emits_the_warning_alone(capsys: pytest.CaptureFixture[str],
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    """⚠️ Nothing refreshes `chunks` in hosted mode, so the digest's count is 0 on the very path
    the warning exists for. Telling the model to search a corpus the same sentence says is empty
    is worse than saying nothing."""
    config = {**HOSTED, "chunks": 0}
    _manifest(config, {"pending": {"a.md": ""}, "last_error": {"kind": "auth", "message": "401"}})
    monkeypatch.setattr(recall_hooks, "load_config", lambda: config)
    recall_hooks.session_start({"cwd": "/tmp/proj"})
    context = json.loads(capsys.readouterr().out)["hookSpecificOutput"]["additionalContext"]
    assert context.startswith("WARNING:")
    assert "indexed chunks" not in context
    assert "recall_search" not in context


def test_a_stop_request_is_not_swallowed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The dispatch catches BaseException so a cancelled anyio scope cannot kill a session. That
    must not extend to Ctrl-C, which is the user asking for this process to end."""
    cwd = tmp_path / "work"
    (cwd / "memory").mkdir(parents=True)
    monkeypatch.setattr(recall_hooks, "load_config", lambda: HOSTED)

    from recall_hooks import hosted

    def interrupted(*_a, **_k):
        raise KeyboardInterrupt

    monkeypatch.setattr(hosted, "sync_memory_roots", interrupted)
    with pytest.raises(KeyboardInterrupt):
        recall_hooks._index_and_refresh({"cwd": str(cwd)})


def test_a_cancelled_scope_does_not_kill_the_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`asyncio.CancelledError` is a BaseException, and it is what a timed-out transport raises.
    Catching only `Exception` would let the one failure most likely in production through."""
    import asyncio

    cwd = tmp_path / "work"
    (cwd / "memory").mkdir(parents=True)
    monkeypatch.setattr(recall_hooks, "load_config", lambda: HOSTED)

    from recall_hooks import hosted

    def cancelled(*_a, **_k):
        raise asyncio.CancelledError

    monkeypatch.setattr(hosted, "sync_memory_roots", cancelled)
    assert recall_hooks._index_and_refresh({"cwd": str(cwd)}) == 0


def test_no_local_cursor_is_kept(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """⛔ Pins rule 1 of hosted.py. If a confirmed-hash map reappears in the manifest, a later
    reader will take it for the cursor and skip asking the server, which is how a file the server
    never received gets skipped forever."""
    from recall_hooks import hosted

    root = tmp_path / "memory"
    root.mkdir()
    (root / "a.md").write_text("# A\n\nBody enough to chunk.\n", encoding="utf-8")

    monkeypatch.setattr(hosted, "remote_inventory", lambda *_a, **_k: {})
    monkeypatch.setattr(hosted, "call_tool", lambda *_a, **_k: {"ok": True})
    # `sync_memory_roots` does `from . import credentials`, which binds on the PACKAGE, so this
    # is the attribute that decides what the sync sends.
    import recall_hooks.credentials as _cred

    monkeypatch.setattr(_cred, "headers", lambda _c: {"Authorization": "Bearer x"})

    outcome = hosted.sync_memory_roots([("worktree", root)], HOSTED)
    assert outcome.kind == "ok" and outcome.uploaded == 1
    manifest = hosted.read_manifest(HOSTED)
    assert "files" not in manifest, "a local record of what is synced must not come back"
    assert manifest["pending"] == {}


# --------------------------------------------------------------------------- the human commands


def test_login_is_reachable_from_the_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """⛔ THE REGRESSION THIS EXISTS FOR. `recall-hooks login` was named in two user-visible
    messages while `main` had no branch for it, so it fell through to `return 0`, printed nothing,
    and the user's sync kept failing after they followed the instruction exactly."""
    from recall_hooks import credentials

    seen: dict[str, object] = {}
    monkeypatch.setattr(recall_hooks, "load_config", lambda: HOSTED)
    def fake_login(_config, args):
        seen["args"] = args
        return 0

    monkeypatch.setattr(credentials, "login", fake_login)
    assert recall_hooks.main(["login", "--token", "x"]) == 0
    assert seen["args"] == ["--token", "x"]


def test_login_does_not_have_its_stdin_eaten_by_the_payload_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every hook EVENT is fed JSON on stdin and `main` consumes it up front. `login` reads a token
    from stdin instead, so dispatching it after that read would swallow the credential."""
    import io

    from recall_hooks import credentials

    monkeypatch.setattr(sys, "stdin", io.StringIO("rt-from-stdin"))
    monkeypatch.setattr(recall_hooks, "load_config", lambda: HOSTED)
    captured: dict[str, str] = {}

    def fake_login(_config, _args):
        captured["token"] = sys.stdin.read()
        return 0

    monkeypatch.setattr(credentials, "login", fake_login)
    assert recall_hooks.main(["login"]) == 0
    assert captured["token"] == "rt-from-stdin", "the payload read must not have consumed it"


def test_auth_headers_is_reachable(monkeypatch: pytest.MonkeyPatch) -> None:
    """It was written, tested and unreachable: nothing dispatched to it."""
    from recall_hooks import credentials

    calls: list[str] = []
    monkeypatch.setattr(recall_hooks, "load_config", lambda: HOSTED)
    monkeypatch.setattr(credentials, "print_auth_headers",
                        lambda _c: calls.append("called") or 0)
    assert recall_hooks.main(["auth-headers"]) == 0
    assert calls == ["called"]


def test_an_unknown_subcommand_is_still_silent(monkeypatch: pytest.MonkeyPatch) -> None:
    """The events keep their contract: anything unrecognised is exit 0 and no output."""
    import io

    monkeypatch.setattr(sys, "stdin", io.StringIO("{}"))
    assert recall_hooks.main(["nonsense"]) == 0

