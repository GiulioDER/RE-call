"""Tests for the hook processes themselves.

Two properties matter more than any output these produce. They must **fail open**, because a hook
that runs before every session must never be the reason Claude does not start, and they must not
import the `recall` package, because that costs about a second of every session launch.
"""
from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path, PureWindowsPath
from typing import Any

import pytest

import recall_hooks


def _configure(tmp_path: Path, monkeypatch: Any, **values: Any) -> Path:
    config = tmp_path / "recall-hook.json"
    config.write_text(json.dumps(values), encoding="utf-8")
    monkeypatch.setattr(recall_hooks, "config_path", lambda: config)
    return config


def test_session_start_says_nothing_without_a_config(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(recall_hooks, "config_path", lambda: tmp_path / "absent.json")
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    assert recall_hooks.session_start({"cwd": str(tmp_path)}) == 0
    assert sys.stdout.getvalue() == ""


def test_session_start_says_nothing_when_the_corpus_is_empty(
    tmp_path: Path, monkeypatch: Any
) -> None:
    _configure(tmp_path, monkeypatch, dsn="postgresql://h/db", chunks=0)
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    assert recall_hooks.session_start({"cwd": str(tmp_path)}) == 0
    assert sys.stdout.getvalue() == ""


def test_session_start_emits_the_documented_envelope(tmp_path: Path, monkeypatch: Any) -> None:
    _configure(tmp_path, monkeypatch, dsn="postgresql://h/db", chunks=1847)
    monkeypatch.setattr(sys, "stdout", io.StringIO())

    assert recall_hooks.session_start({"cwd": "C:/proj/demo"}) == 0

    emitted = json.loads(sys.stdout.getvalue())["hookSpecificOutput"]
    assert emitted["hookEventName"] == "SessionStart"
    assert "1847" in emitted["additionalContext"]
    assert "demo" in emitted["additionalContext"]


def test_session_start_does_not_touch_the_database(tmp_path: Path, monkeypatch: Any) -> None:
    """The count comes from the cache. A DSN pointing at a closed port must cost nothing."""
    _configure(tmp_path, monkeypatch, dsn="postgresql://u:p@127.0.0.1:1/db", chunks=12)

    def explode(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("session_start must not open a connection")

    monkeypatch.setattr(recall_hooks, "refresh_stats", explode)
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    assert recall_hooks.session_start({"cwd": "C:/proj/demo"}) == 0


def test_a_corrupt_config_is_survived_rather_than_raised(tmp_path: Path, monkeypatch: Any) -> None:
    config = tmp_path / "recall-hook.json"
    config.write_text("{not json at all", encoding="utf-8")
    monkeypatch.setattr(recall_hooks, "config_path", lambda: config)
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    assert recall_hooks.load_config() == {}
    assert recall_hooks.session_start({"cwd": str(tmp_path)}) == 0


def test_refresh_stats_keeps_the_last_known_count_when_the_database_is_down(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """A stale number beats zero: reporting no memory is worse than reporting yesterday's."""
    _configure(tmp_path, monkeypatch, dsn="postgresql://u:p@127.0.0.1:1/db", chunks=99)
    assert recall_hooks.refresh_stats() == 99


def test_session_end_is_a_noop_outside_a_project(tmp_path: Path, monkeypatch: Any) -> None:
    _configure(tmp_path, monkeypatch, dsn="postgresql://h/db")
    monkeypatch.setattr(recall_hooks, "refresh_stats", lambda config=None: 0)
    assert recall_hooks.session_end({"cwd": str(tmp_path)}) == 0


def test_pre_compact_indexes_and_refreshes_like_session_end(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Both want the same thing at different moments: make what was written searchable."""
    _configure(tmp_path, monkeypatch, dsn="postgresql://h/db", embedder="hashing")
    (tmp_path / "memory").mkdir()
    (tmp_path / "memory" / "a.md").write_text("a fact\n", encoding="utf-8")
    indexed: list[Path] = []
    refreshed: list[int] = []
    monkeypatch.setattr(
        "recall.setup.index_memory_directory",
        lambda **kw: indexed.append(kw["memory_dir"]),
    )
    monkeypatch.setattr(recall_hooks, "refresh_stats", lambda config=None: refreshed.append(1))

    assert recall_hooks.pre_compact({"cwd": str(tmp_path)}) == 0

    assert indexed == [tmp_path / "memory"]
    assert refreshed == [1]


def test_pre_compact_never_returns_a_blocking_exit_code(tmp_path: Path, monkeypatch: Any) -> None:
    """Exit code 2 blocks compaction. No failure here may reach that.

    A session compacting because its context window is full is the worst possible moment to be
    told it may not proceed, and the cause would be a memory tool it never asked about.
    """
    _configure(tmp_path, monkeypatch, dsn="postgresql://u:p@127.0.0.1:1/db", embedder="hashing")
    (tmp_path / "memory").mkdir()
    (tmp_path / "memory" / "a.md").write_text("a fact\n", encoding="utf-8")

    def explode(**kwargs: Any) -> None:
        raise RuntimeError("embedder could not be resolved")

    monkeypatch.setattr("recall.setup.index_memory_directory", explode)

    assert recall_hooks.pre_compact({"cwd": str(tmp_path)}) == 0
    # And with nothing configured at all.
    monkeypatch.setattr(recall_hooks, "config_path", lambda: tmp_path / "absent.json")
    assert recall_hooks.pre_compact({"cwd": str(tmp_path)}) == 0
    assert recall_hooks.pre_compact({}) == 0


def test_main_dispatches_pre_compact(monkeypatch: Any) -> None:
    seen: list[dict[str, Any]] = []
    monkeypatch.setattr(sys, "stdin", io.StringIO('{"cwd": "C:/proj"}'))
    monkeypatch.setattr(recall_hooks, "pre_compact", lambda payload: seen.append(payload) or 0)
    assert recall_hooks.main(["pre-compact"]) == 0
    assert seen == [{"cwd": "C:/proj"}]


def test_main_survives_stdin_that_is_not_json(monkeypatch: Any) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO("not json"))
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    assert recall_hooks.main(["session-start"]) == 0


def test_main_ignores_an_event_it_does_not_handle(monkeypatch: Any) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO("{}"))
    assert recall_hooks.main(["PreToolUse"]) == 0
    assert recall_hooks.main([]) == 0


def test_the_hook_path_does_not_import_the_recall_package() -> None:
    """The reason `recall_hooks` is a separate top-level package.

    `recall/__init__.py` eagerly imports the calibration, evidence and lineage modules, so an
    accidental `from recall.x import y` at module scope here would put roughly a second onto every
    session launch. Nothing in the module's own text would reveal that, so it is asserted.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import recall_hooks, sys; print('recall' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "False"


# --------------------------------------------------------------------------------------------
# Where the index actually lands
#
# The defect these cover shipped because the test above them captured only `memory_dir`. An
# index that reports success and writes to the wrong tenant looks identical from there.
# --------------------------------------------------------------------------------------------


def _isolate_client_home(tmp_path: Path, monkeypatch: Any) -> Path:
    """Point `claude_config_home` at a temporary directory.

    Not optional. `_memory_directories` consults `~/.claude/projects`, so without this a test
    would read the developer's real client state and pass or fail on what happens to be there.
    """
    home = tmp_path / "client-home"
    (home / "projects").mkdir(parents=True)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(home))
    return home


def test_the_index_lands_in_the_configured_tenant_not_the_default(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """The whole point of the tenant field, and it was ignored.

    `refresh_stats` counts the CONFIGURED tenant while the index wrote to the default one, so a
    store set up with `tenant: memory` had its memos written where nothing looks for them and its
    count read 0 forever. Both halves have to be asserted together: either alone looks correct.
    """
    _isolate_client_home(tmp_path, monkeypatch)
    _configure(
        tmp_path,
        monkeypatch,
        dsn="postgresql://h/db",
        embedder="hashing",
        tenant="memory",
        table="notes",
    )
    (tmp_path / "memory").mkdir()
    (tmp_path / "memory" / "a.md").write_text("a fact\n", encoding="utf-8")
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr("recall.setup.index_memory_directory", lambda **kw: calls.append(kw))
    monkeypatch.setattr(recall_hooks, "refresh_stats", lambda config=None: 0)

    assert recall_hooks.session_end({"cwd": str(tmp_path)}) == 0

    assert len(calls) == 1
    assert calls[0]["tenant"] == "memory"
    assert calls[0]["table"] == "notes"


def test_an_absent_tenant_still_means_the_documented_defaults(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """A config written before the tenant key existed must keep behaving as it did."""
    _isolate_client_home(tmp_path, monkeypatch)
    _configure(tmp_path, monkeypatch, dsn="postgresql://h/db", embedder="hashing")
    (tmp_path / "memory").mkdir()
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr("recall.setup.index_memory_directory", lambda **kw: calls.append(kw))
    monkeypatch.setattr(recall_hooks, "refresh_stats", lambda config=None: 0)

    recall_hooks.session_end({"cwd": str(tmp_path)})

    assert calls[0]["tenant"] == "default"
    assert calls[0]["table"] == "chunks"


def test_an_indexing_failure_is_explained_rather_than_swallowed(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    """`index_memory_directory` reports its own failures through `print_fn`.

    The hook passed a lambda that discarded them, so the message naming the real cause, most
    often a schema applied for a different embedder's dimension, went nowhere.
    """
    _isolate_client_home(tmp_path, monkeypatch)
    _configure(tmp_path, monkeypatch, dsn="postgresql://h/db", embedder="hashing")
    (tmp_path / "memory").mkdir()

    def report(**kwargs: Any) -> None:
        kwargs["print_fn"]("Could not auto-index: dimension 384 does not match 1024")

    monkeypatch.setattr("recall.setup.index_memory_directory", report)
    monkeypatch.setattr(recall_hooks, "refresh_stats", lambda config=None: 0)

    recall_hooks.session_end({"cwd": str(tmp_path)})

    assert "dimension 384 does not match 1024" in capsys.readouterr().err


def test_one_failing_store_does_not_abandon_the_others(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """An index that raises used to return early, skipping the refresh as well as the next store."""
    home = _isolate_client_home(tmp_path, monkeypatch)
    _configure(tmp_path, monkeypatch, dsn="postgresql://h/db", embedder="hashing")
    (tmp_path / "memory").mkdir()
    client_store = home / "projects" / recall_hooks._claude_project_slug(tmp_path) / "memory"
    client_store.mkdir(parents=True)
    seen: list[Path] = []
    refreshed: list[int] = []

    def sometimes_explode(**kwargs: Any) -> None:
        seen.append(kwargs["memory_dir"])
        if len(seen) == 1:
            raise RuntimeError("first store is broken")

    monkeypatch.setattr("recall.setup.index_memory_directory", sometimes_explode)
    monkeypatch.setattr(recall_hooks, "refresh_stats", lambda config=None: refreshed.append(1))

    assert recall_hooks.session_end({"cwd": str(tmp_path)}) == 0

    assert len(seen) == 2
    assert refreshed == [1]


# --------------------------------------------------------------------------------------------
# Finding the store the agent actually wrote to
# --------------------------------------------------------------------------------------------


def test_the_slug_matches_the_encoding_the_client_actually_writes() -> None:
    """Pinned against two directory names read off a real machine on 2026-08-25.

    If this ever fails, the client changed its encoding: the correct response is to re-read
    `~/.claude/projects` and update it, not to loosen the assertion.
    """
    assert (
        recall_hooks._claude_project_slug(PureWindowsPath(r"C:\Users\gde00\Documents\recall"))
        == "C--Users-gde00-Documents-recall"
    )
    assert recall_hooks._claude_project_slug(PureWindowsPath(r"C:\cca-demos")) == "C--cca-demos"


def test_the_claude_code_memory_store_is_indexed_too(tmp_path: Path, monkeypatch: Any) -> None:
    """The store the client writes, which recall never creates and used to never read.

    recall's own repository has no in-repo `memory/` at all, so before this the hook that claims
    to "index the session so the next one can find it" indexed nothing on the flagship project.
    """
    home = _isolate_client_home(tmp_path, monkeypatch)
    project = tmp_path / "proj"
    project.mkdir()
    store = home / "projects" / recall_hooks._claude_project_slug(project) / "memory"
    store.mkdir(parents=True)
    _configure(tmp_path, monkeypatch, dsn="postgresql://h/db", embedder="hashing")

    assert recall_hooks._memory_directories(project) == [store]


def test_a_worktree_resolves_to_the_main_checkouts_store(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """An agent working in a worktree is the normal arrangement, not an exotic one.

    The client keys its store on the directory the session opened in, so a worktree gets a slug
    of its own with nothing behind it while the real store sits under the main checkout's.
    """
    home = _isolate_client_home(tmp_path, monkeypatch)
    main = tmp_path / "repo"
    worktree = main / ".claude" / "worktrees" / "feature-x"
    worktree.mkdir(parents=True)
    (worktree / ".git").write_text(
        f"gitdir: {main}/.git/worktrees/feature-x\n", encoding="utf-8"
    )
    store = home / "projects" / recall_hooks._claude_project_slug(main) / "memory"
    store.mkdir(parents=True)

    assert recall_hooks._memory_directories(worktree) == [store]


def test_a_git_file_that_is_not_a_worktree_pointer_is_survived(tmp_path: Path) -> None:
    """Degrade to "not a worktree" rather than raising, for anything unexpected in that file."""
    for content in ("", "gitdir: /somewhere/else\n", "not a gitdir line\n", "gitdir:\n"):
        (tmp_path / ".git").write_text(content, encoding="utf-8")
        assert recall_hooks._worktree_parent(tmp_path) is None
    (tmp_path / ".git").unlink()
    (tmp_path / ".git").mkdir()
    assert recall_hooks._worktree_parent(tmp_path) is None


def test_a_missing_projects_root_is_not_an_error(tmp_path: Path, monkeypatch: Any) -> None:
    """A machine that has never run the client at all still has to end its sessions."""
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "never-created"))
    assert recall_hooks._memory_directories(tmp_path / "proj") == []


# --------------------------------------------------------------------------------------------
# Whether the hook may write at all
# --------------------------------------------------------------------------------------------


def _project_with_memory(tmp_path: Path, monkeypatch: Any, **config: Any) -> Path:
    _isolate_client_home(tmp_path, monkeypatch)
    monkeypatch.delenv("RECALL_EMBEDDER", raising=False)
    project = tmp_path / "proj"
    (project / "memory").mkdir(parents=True)
    _configure(tmp_path, monkeypatch, dsn="postgresql://h/db", embedder="hashing", **config)
    return project


def test_auto_index_off_counts_but_does_not_write(tmp_path: Path, monkeypatch: Any) -> None:
    """The switch a corpus indexed deliberately on another host needs.

    `Indexer.index_path` serialises writers correctly, so this is not about corruption. It is
    about a corpus whose indexing is a scheduled, locked, memory-capped operation on one machine
    not wanting an extra writer arriving from whichever workstation closed a session. Reading is
    still allowed: the count costs the corpus nothing and the digest is worth keeping accurate.
    """
    project = _project_with_memory(tmp_path, monkeypatch, auto_index=False)
    calls: list[dict[str, Any]] = []
    refreshed: list[int] = []
    monkeypatch.setattr("recall.setup.index_memory_directory", lambda **kw: calls.append(kw))
    monkeypatch.setattr(recall_hooks, "refresh_stats", lambda config=None: refreshed.append(1))

    assert recall_hooks.session_end({"cwd": str(project)}) == 0
    assert recall_hooks.pre_compact({"cwd": str(project)}) == 0

    assert calls == []
    assert refreshed == [1, 1]


def test_auto_index_defaults_to_on_for_a_config_that_predates_it(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Every config written before this key existed must keep indexing."""
    project = _project_with_memory(tmp_path, monkeypatch)
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr("recall.setup.index_memory_directory", lambda **kw: calls.append(kw))
    monkeypatch.setattr(recall_hooks, "refresh_stats", lambda config=None: 0)

    recall_hooks.session_end({"cwd": str(project)})

    assert len(calls) == 1


@pytest.mark.parametrize(
    "value, expected",
    [
        (False, False), (True, True),
        ("false", False), ("FALSE", False), ("no", False), ("off", False), ("0", False),
        ("true", True), ("yes", True), ("on", True), ("1", True),
    ],
)
def test_the_switch_reads_both_spellings_of_each_answer(
    value: Any, expected: bool, tmp_path: Path, monkeypatch: Any
) -> None:
    """This file is edited by hand, where `false` and `"false"` are the same intention."""
    _configure(tmp_path, monkeypatch, dsn="postgresql://h/db", auto_index=value)
    assert recall_hooks._auto_index_enabled(recall_hooks.load_config()) is expected


def test_an_unreadable_switch_warns_instead_of_picking_a_side(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    """⚠️ Both defaults are wrong in a different direction, so a typo must be loud.

    Failing on runs an indexer somebody tried to disable; failing off silently stops indexing
    somebody expects, which is the class of silent no-op this whole module was rewritten to
    eliminate. The fallback is stated in the message rather than left to be discovered.
    """
    _configure(tmp_path, monkeypatch, dsn="postgresql://h/db", auto_index="flase")

    assert recall_hooks._auto_index_enabled(recall_hooks.load_config()) is True

    err = capsys.readouterr().err
    assert "flase" in err
    assert "indexing anyway" in err


# --------------------------------------------------------------------------------------------
# Which model does the indexing
# --------------------------------------------------------------------------------------------


def test_the_project_beats_the_recorded_embedder_when_they_disagree(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    """⛔ A dimension match is not a model match, and nothing raises when they differ.

    The hook config records an embedder once, at install, and never reconciles it. `bge-large`,
    `voyage-3` and `voyage-4` all emit 1024 dimensions, so a tenant written by two of them returns
    a confidently ranked list that means nothing, and no layer complains. The CLI reads the
    project's declaration, so the hook has to agree with it or the two write different vectors
    into one corpus.
    """
    _isolate_client_home(tmp_path, monkeypatch)
    monkeypatch.delenv("RECALL_EMBEDDER", raising=False)
    project = tmp_path / "proj"
    (project / "memory").mkdir(parents=True)
    (project / ".env").write_text('RECALL_EMBEDDER="voyage:voyage-4"\n', encoding="utf-8")
    _configure(tmp_path, monkeypatch, dsn="postgresql://h/db", embedder="voyage:voyage-3")
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr("recall.setup.index_memory_directory", lambda **kw: calls.append(kw))
    monkeypatch.setattr(recall_hooks, "refresh_stats", lambda config=None: 0)

    recall_hooks.session_end({"cwd": str(project)})

    assert calls[0]["embedder_name"] == "voyage:voyage-4"
    assert "voyage:voyage-3" in capsys.readouterr().err


def test_the_recorded_embedder_stands_when_the_project_declares_nothing(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    """No declaration is not a disagreement, and must not produce a warning."""
    _isolate_client_home(tmp_path, monkeypatch)
    monkeypatch.delenv("RECALL_EMBEDDER", raising=False)
    project = tmp_path / "proj"
    (project / "memory").mkdir(parents=True)
    _configure(tmp_path, monkeypatch, dsn="postgresql://h/db", embedder="fastembed")
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr("recall.setup.index_memory_directory", lambda **kw: calls.append(kw))
    monkeypatch.setattr(recall_hooks, "refresh_stats", lambda config=None: 0)

    recall_hooks.session_end({"cwd": str(project)})

    assert calls[0]["embedder_name"] == "fastembed"
    assert capsys.readouterr().err == ""


def test_a_project_that_agrees_is_not_reported_as_drift(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    """The common case, and the one a mutation showed was untested.

    Changing `if declared and declared != recorded` to `if declared` survived the whole suite,
    because every case here either declared nothing or declared something different. A hook that
    warned on every agreeing session would train the reader to ignore the one warning that matters.
    """
    _isolate_client_home(tmp_path, monkeypatch)
    monkeypatch.delenv("RECALL_EMBEDDER", raising=False)
    project = tmp_path / "proj"
    (project / "memory").mkdir(parents=True)
    (project / ".env").write_text("RECALL_EMBEDDER=fastembed\n", encoding="utf-8")
    _configure(tmp_path, monkeypatch, dsn="postgresql://h/db", embedder="fastembed")
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr("recall.setup.index_memory_directory", lambda **kw: calls.append(kw))
    monkeypatch.setattr(recall_hooks, "refresh_stats", lambda config=None: 0)

    recall_hooks.session_end({"cwd": str(project)})

    assert calls[0]["embedder_name"] == "fastembed"
    assert capsys.readouterr().err == ""


def test_a_commented_out_declaration_is_not_a_declaration(tmp_path: Path, monkeypatch: Any) -> None:
    """The `.env` parse is small enough to get wrong in exactly these two ways.

    Both are rejected by the same exact key comparison rather than by two separate guards: a
    commented line yields the key `# RECALL_EMBEDDER` and a neighbouring variable yields
    `RECALL_EMBEDDER_EXTRA`, and neither equals `RECALL_EMBEDDER`. A dedicated comment skip used
    to sit beside it and was removed when no mutation of it could turn this red.
    """
    monkeypatch.delenv("RECALL_EMBEDDER", raising=False)
    (tmp_path / ".env").write_text(
        "# RECALL_EMBEDDER=commented-out\nRECALL_EMBEDDER_EXTRA=not-the-key\n", encoding="utf-8"
    )
    assert recall_hooks._declared_embedder(tmp_path) is None

    (tmp_path / ".env").write_text("RECALL_EMBEDDER='quoted'\n", encoding="utf-8")
    assert recall_hooks._declared_embedder(tmp_path) == "quoted"


def test_the_process_environment_wins_over_the_projects_file(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """An explicitly exported variable is a deliberate override of a checked-in default."""
    monkeypatch.setenv("RECALL_EMBEDDER", "hashing")
    (tmp_path / ".env").write_text("RECALL_EMBEDDER=fastembed\n", encoding="utf-8")
    assert recall_hooks._declared_embedder(tmp_path) == "hashing"


# --------------------------------------------------------------------------------------------
# Telling "installed and broken" apart from "not installed"
# --------------------------------------------------------------------------------------------


def test_an_unreachable_database_says_so_instead_of_serving_a_silent_zero(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    """The silent path, and the one that is permanent rather than transient.

    A DSN that never resolves sticks the count at whatever the installer wrote, which is 0. The
    digest then emits nothing and the integration is indistinguishable from an uninstalled one.
    Measured on the author's own machine on 2026-08-25, where exactly that had happened.
    """
    config = _configure(tmp_path, monkeypatch, dsn="postgresql://example/recall", chunks=0)

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise OSError("failed to resolve host 'example'")

    monkeypatch.setattr("psycopg.connect", refuse)

    assert recall_hooks.refresh_stats() == 0

    assert "cannot reach the database" in capsys.readouterr().err
    assert json.loads(config.read_text(encoding="utf-8"))["status"] == "unreachable"


def test_a_stuck_count_does_not_rewrite_the_config_every_session(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """The warning repeats; the write must not. A hook that runs forever writes forever."""
    config = _configure(tmp_path, monkeypatch, dsn="postgresql://example/recall", chunks=0)

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise OSError("unreachable")

    monkeypatch.setattr("psycopg.connect", refuse)
    recall_hooks.refresh_stats()
    first = config.stat().st_mtime_ns
    writes: list[int] = []
    monkeypatch.setattr(
        recall_hooks, "_save_config", lambda cfg: writes.append(1)
    )
    recall_hooks.refresh_stats()

    assert writes == []
    assert config.stat().st_mtime_ns == first


def test_a_successful_count_clears_the_stale_status(tmp_path: Path, monkeypatch: Any) -> None:
    """Otherwise a store that recovers keeps reporting itself broken."""
    config = _configure(
        tmp_path, monkeypatch, dsn="postgresql://h/db", chunks=0, status="unreachable"
    )

    class _Cursor:
        def fetchone(self) -> tuple[int]:
            return (91,)

    class _Connection:
        def __enter__(self) -> "_Connection":
            return self

        def __exit__(self, *exc: Any) -> bool:
            return False

        def execute(self, *args: Any, **kwargs: Any) -> "_Cursor":
            return _Cursor()

    monkeypatch.setattr("psycopg.connect", lambda *a, **k: _Connection())

    assert recall_hooks.refresh_stats() == 91

    document = json.loads(config.read_text(encoding="utf-8"))
    assert document["status"] == "ok"
    assert document["chunks"] == 91


def test_session_start_explains_a_broken_store_on_stderr_not_in_context(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    """A broken memory tool must not spend the model's context complaining about itself."""
    _configure(
        tmp_path, monkeypatch, dsn="postgresql://example/recall", chunks=0, status="unreachable"
    )
    monkeypatch.setattr(sys, "stdout", io.StringIO())

    assert recall_hooks.session_start({"cwd": str(tmp_path)}) == 0

    assert sys.stdout.getvalue() == ""
    assert "configured but the last count could not be taken" in capsys.readouterr().err


def test_an_empty_corpus_stays_quiet_on_both_channels(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    """An indexed-but-empty store is ordinary. Only a store that cannot be reached is news."""
    _configure(tmp_path, monkeypatch, dsn="postgresql://h/db", chunks=0, status="ok")
    monkeypatch.setattr(sys, "stdout", io.StringIO())

    assert recall_hooks.session_start({"cwd": str(tmp_path)}) == 0

    assert sys.stdout.getvalue() == ""
    assert capsys.readouterr().err == ""


# --------------------------------------------------------------------------------------------
# What the digest says
# --------------------------------------------------------------------------------------------


def test_the_digest_does_not_assert_a_calibration_state_it_cannot_know(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """It used to end "This corpus is uncalibrated" unconditionally, which is a claim this hook
    has no way to check: it deliberately never opens the database. After `recall setup` fits a
    threshold the sentence was simply false, and it taught the agent to discount the verdicts the
    product exists to make trustworthy. The trust layer already marks its own results
    `DEGRADED:INDEX_NOT_READY` when it applies, which is where the claim is true.
    """
    _configure(tmp_path, monkeypatch, dsn="postgresql://h/db", chunks=1847)
    monkeypatch.setattr(sys, "stdout", io.StringIO())

    recall_hooks.session_start({"cwd": "C:/proj/demo"})

    context = json.loads(sys.stdout.getvalue())["hookSpecificOutput"]["additionalContext"]
    assert "uncalibrated" not in context


def test_the_digest_teaches_operation_vocabulary_rather_than_goal_vocabulary(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Run `agent-ab-skill-001` measured the two framings against each other.

    Searching for the goal ("proposing an idea", "forming a hypothesis") is the framing whose
    misses were recorded; searching for the operation is the one that moved the governing-memo
    rate to 0.674 from 0.319. This channel carried the losing framing until 2026-08-25.
    Pre-registration: `docs/preregistrations/2026-08-25-instruction-channel-unification.md`.
    """
    _configure(tmp_path, monkeypatch, dsn="postgresql://h/db", chunks=1847)
    monkeypatch.setattr(sys, "stdout", io.StringIO())

    recall_hooks.session_start({"cwd": "C:/proj/demo"})

    context = json.loads(sys.stdout.getvalue())["hookSpecificOutput"]["additionalContext"]
    assert "OPERATIONS" in context
    assert "forming a hypothesis" not in context
    assert "superseded" in context
