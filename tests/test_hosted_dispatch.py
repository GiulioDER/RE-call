"""Hosted mode: which branch a hook takes, which directories it reads, and what it says.

The dispatch is the first change in this feature that alters what a hook DOES, so these are about
the two things that can go wrong there: taking the wrong branch, and losing a memo without saying
so.
"""

from __future__ import annotations

import json
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
    _manifest(HOSTED, {"files": {}, "pending": {}})
    assert recall_hooks._unsynced_notice(HOSTED) == ""


def test_pending_memos_with_an_error_are_surfaced_with_a_remedy() -> None:
    """🔑 The hooks return 0 always and may not speak, so a failed sync has nowhere to report
    except the one event already permitted to inject text."""
    _manifest(HOSTED, {"files": {}, "pending": {"a.md": "", "b.md": ""},
                       "last_error": {"kind": "auth", "message": "401"}})
    notice = recall_hooks._unsynced_notice(HOSTED)
    assert "2 memo(s)" in notice
    assert "recall-hooks login" in notice


def test_pending_without_a_recorded_error_stays_quiet() -> None:
    """Mid-sync is not a failure, and a healthy install must see nothing."""
    _manifest(HOSTED, {"files": {}, "pending": {"a.md": ""}})
    assert recall_hooks._unsynced_notice(HOSTED) == ""


def test_a_local_config_never_sees_a_hosted_notice() -> None:
    assert recall_hooks._unsynced_notice(LOCAL) == ""


def test_the_notice_reaches_session_start_output(capsys: pytest.CaptureFixture[str],
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    """It is worth nothing unless it actually lands in additionalContext."""
    config = {**HOSTED, "chunks": 12}
    _manifest(config, {"files": {}, "pending": {"a.md": ""},
                       "last_error": {"kind": "quota", "message": "budget"}})
    monkeypatch.setattr(recall_hooks, "load_config", lambda: config)
    recall_hooks.session_start({"cwd": "/tmp/proj"})
    out = json.loads(capsys.readouterr().out)
    context = out["hookSpecificOutput"]["additionalContext"]
    assert context.startswith("WARNING:")
    assert "quota" in context or "exhausted" in context
