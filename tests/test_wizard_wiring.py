"""MCP registration: the artifact that decides whether any of the install is reachable.

Each property below is a way this could produce a configuration that looks complete and answers
nothing, which is the worst outcome available here: the operator has no reason to doubt a file the
installer wrote.

The servers are registered at **local scope**, in Claude Code's own `~/.claude.json` under this
project's entry, and the wizard deliberately writes no project-scoped `.mcp.json`. Project scope is
gated behind an approval prompt; user scope skips the prompt but loads in every project on the
machine, and these servers each carry a `RECALL_TENANT` for THIS project's corpus. Local scope is
the one that skips the prompt without answering about the wrong repository.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from recall.wizard.corpora import default_plan
from recall.wizard.wiring import (
    mcp_config,
    register_local_scope,
    server_blocks,
    write_project_files,
)


def _plan(tmp_path: Path):
    return default_plan(
        embedder="hashing",
        docs_root=tmp_path / "docs",
        code_root=tmp_path / "repo",
        memory_root=tmp_path / "memory",
    )


def _registered(client: Path, project_root: Path) -> dict:
    """The servers Claude Code would load for `project_root`, read back from the client's config."""
    document = json.loads(client.read_text(encoding="utf-8"))
    entries = [
        entry.get("mcpServers", {})
        for key, entry in document.get("projects", {}).items()
        if Path(key).resolve() == project_root.resolve()
    ]
    assert len(entries) == 1, f"expected exactly one project entry, got {len(entries)}"
    return entries[0]


def _client(tmp_path: Path) -> Path:
    """A `~/.claude.json` that already exists, which is the ordinary case: Claude Code has run.

    The wizard refuses to CREATE another application's config, so a test that wants a successful
    registration has to stand one up. Written here rather than inline so the tests below assert
    about smoke and trust rather than about JSON.
    """
    client = tmp_path / ".claude.json"
    client.write_text(json.dumps({"projects": {}}), encoding="utf-8")
    return client


def _by_name(blocks):
    return {b.name: b for b in blocks}


# ----------------------------------------------------------------------------------------------
# Which tenants get a server, and why
# ----------------------------------------------------------------------------------------------


def test_a_promoted_tenant_is_served_from_its_generation_under_strict_trust(tmp_path: Path) -> None:
    """`RECALL_ENV=production` is what selects `GenerationStore` (`recall_mcp/server.py:629`).

    Without it the server reads the legacy `chunks` table, which a generation build never wrote to,
    so the tenant answers nothing while looking configured. That is the failure this asserts against,
    not a style preference.
    """
    blocks, unservable = server_blocks(
        _plan(tmp_path),
        dsn="postgresql://recall:pw@127.0.0.1:5432/recall",
        promoted=frozenset({"default-docs", "default-code"}),
        serving=frozenset({"default-docs", "default-code", "default-memory"}),
    )

    assert unservable == ()
    docs = _by_name(blocks)["default-docs"]
    assert docs.env["RECALL_ENV"] == "production"
    assert docs.env["RECALL_TRUST_MODE"] == "strict"
    assert docs.env["RECALL_TENANT"] == "default-docs"


def test_a_degraded_tenant_with_a_predecessor_is_served_with_relaxed_trust(tmp_path: Path) -> None:
    """An earlier generation still answers, so a server is correct; the calibration is stale."""
    blocks, unservable = server_blocks(
        _plan(tmp_path),
        dsn="postgresql://recall:pw@127.0.0.1:5432/recall",
        promoted=frozenset(),
        serving=frozenset({"default-docs", "default-code", "default-memory"}),
    )

    assert unservable == ()
    docs = _by_name(blocks)["default-docs"]
    assert docs.env["RECALL_ENV"] == "production"
    assert docs.env["RECALL_TRUST_MODE"] == "development"
    assert "EARLIER generation" in docs.rationale, "the operator must be told what is serving"


def test_a_degraded_tenant_with_nothing_serving_gets_no_server_at_all(tmp_path: Path) -> None:
    """The case a wizard gets wrong, and the reason this module exists.

    Nothing was promoted and there is no predecessor, so under `RECALL_ENV=production` the tenant has
    no active generation and `GenerationStore.snapshot` raises `NoActiveGeneration` from OUTSIDE
    `trusted_search`'s try block: a raw exception, no failure code, no advice. Relaxing the trust
    mode does NOT help, because the failure is upstream of the trust gate — which is exactly why it
    is tempting to write the block anyway.
    """
    blocks, unservable = server_blocks(
        _plan(tmp_path),
        dsn="postgresql://recall:pw@127.0.0.1:5432/recall",
        promoted=frozenset(),
        serving=frozenset({"default-memory"}),
    )

    served = set(_by_name(blocks))
    assert served == {"default-memory"}, "no server may be written for a tenant that cannot answer"
    assert {u.tenant for u in unservable} == {"default-docs", "default-code"}
    assert all("NoActiveGeneration" in u.reason for u in unservable), (
        "the reason must name what would actually happen, not just say it failed"
    )


def test_the_uncalibrated_tenant_is_never_put_into_production_mode(tmp_path: Path) -> None:
    """Its rows are in the legacy `chunks` table, which production mode routes past entirely."""
    blocks, _ = server_blocks(
        _plan(tmp_path),
        dsn="postgresql://recall:pw@127.0.0.1:5432/recall",
        promoted=frozenset({"default-docs", "default-code"}),
        serving=frozenset({"default-docs", "default-code", "default-memory"}),
    )

    memory = _by_name(blocks)["default-memory"]
    assert "RECALL_ENV" not in memory.env, "production mode would route past the legacy table"
    assert memory.env["RECALL_TRUST_MODE"] == "development"


def test_every_block_sets_trust_in_its_own_env_rather_than_relying_on_the_shell(
    tmp_path: Path,
) -> None:
    """A stdio server launched with an explicit `env` inherits NOTHING.

    A corpus that searched correctly from the terminal answered INDEX_NOT_READY through the client
    for exactly this reason, which is recorded in this repository's own CLAUDE.md.
    """
    blocks, _ = server_blocks(
        _plan(tmp_path),
        dsn="postgresql://recall:pw@127.0.0.1:5432/recall",
        promoted=frozenset({"default-docs", "default-code"}),
        serving=frozenset({"default-docs", "default-code", "default-memory"}),
    )

    for block in blocks:
        assert "RECALL_TRUST_MODE" in block.env, f"{block.name} would inherit nothing"
        assert "RECALL_DSN" in block.env
        assert "RECALL_TENANT" in block.env


# ----------------------------------------------------------------------------------------------
# Registering at local scope
# ----------------------------------------------------------------------------------------------


def _blocks(tmp_path: Path, *, promoted=("default-docs", "default-code")):
    blocks, _ = server_blocks(
        _plan(tmp_path),
        dsn="postgresql://recall:pw@127.0.0.1:5432/recall",
        promoted=frozenset(promoted),
        serving=frozenset({"default-docs", "default-code", "default-memory"}),
    )
    return blocks


def test_registering_preserves_servers_this_wizard_knows_nothing_about(tmp_path: Path) -> None:
    """`~/.claude.json` is CLAUDE CODE's file and holds every server the user has.

    Replacing it, or its `mcpServers` key, would delete a working configuration and the operator
    would discover it the next time they reached for a tool that had vanished.
    """
    client = tmp_path / ".claude.json"
    client.write_text(
        json.dumps(
            {
                "someOtherSetting": {"kept": True},
                "projects": {
                    str(tmp_path): {
                        "someProjectSetting": 7,
                        "mcpServers": {
                            "someone-elses": {"type": "http", "url": "https://example.invalid/mcp"},
                            # A previous run of THIS install: updated, not refused.
                            "default-docs": {
                                "type": "stdio",
                                "command": "old",
                                "cwd": str(tmp_path),
                            },
                        },
                    },
                    "C:\\elsewhere": {"mcpServers": {"theirs": {"type": "stdio"}}},
                },
            }
        ),
        encoding="utf-8",
    )

    result = register_local_scope(_blocks(tmp_path), project_root=tmp_path, config_path=client)

    assert result.recorded
    assert result.conflicts == (), "our own previous install is not a stranger"
    document = json.loads(client.read_text(encoding="utf-8"))
    written = _registered(client, tmp_path)
    assert document["projects"][str(tmp_path)]["someProjectSetting"] == 7, (
        "the rest of this project's entry must survive"
    )
    assert document["projects"]["C:\\elsewhere"] == {"mcpServers": {"theirs": {"type": "stdio"}}}, (
        "another project's servers must survive"
    )
    assert written["someone-elses"]["url"] == "https://example.invalid/mcp", "must survive"
    assert document["someOtherSetting"] == {"kept": True}, "unrelated settings must survive"
    # 🔁 Was `== "python"`. A bare `python` resolves against the CLIENT's PATH, not the environment
    # recall is installed into; on Windows that is routinely the Microsoft Store stub, which opens
    # the Store rather than running anything, and the user sees a server that will not start with
    # no cause named. The absolute interpreter is the fix, so the assertion moves with it.
    assert written["default-docs"]["command"] == sys.executable, (
        "ours must be replaced, not merged into, and must name an absolute interpreter"
    )
    assert Path(written["default-docs"]["command"]).is_absolute(), (
        "a bare name would be resolved by whatever PATH the client happens to have"
    )
    assert set(written) == {"someone-elses", "default-docs", "default-code", "default-memory"}
    assert (client.with_name(client.name + ".recall-backup")).exists(), (
        "a write into another application's config must leave a way back"
    )



def test_an_entry_with_no_cwd_is_left_alone_because_this_wizard_did_not_write_it(
    tmp_path: Path,
) -> None:
    """Every block this module writes carries a `cwd`, so one without it belongs to somebody else.

    Reading a missing `cwd` as "probably ours" would silently replace a server the operator
    configured by hand, under a name they picked first. That is the harm the conflict path exists
    to prevent, and it is invisible: the entry is gone and the tool they had simply behaves
    differently.
    """
    client = tmp_path / ".claude.json"
    theirs = {"type": "stdio", "command": "their-own-thing"}
    client.write_text(
        json.dumps({"projects": {str(tmp_path): {"mcpServers": {"default-docs": theirs}}}}),
        encoding="utf-8",
    )

    result = register_local_scope(_blocks(tmp_path), project_root=tmp_path, config_path=client)

    assert result.conflicts == (("default-docs", "no cwd, so it was not written by this wizard"),)
    assert _registered(client, tmp_path)["default-docs"] == theirs


def test_the_same_project_named_relatively_is_not_reported_as_a_stranger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A user's own previous install must not come back as a name they have to rename around.

    `cwd` is compared RESOLVED for exactly this: a config naming the project relatively on one run
    and absolutely on the next is the same directory, and telling the user otherwise would send
    them to rename a project that was already theirs.
    """
    client = tmp_path / ".claude.json"
    client.write_text("{}", encoding="utf-8")
    project = tmp_path / "project"
    project.mkdir()

    register_local_scope(_blocks(tmp_path), project_root=project, config_path=client)
    monkeypatch.chdir(tmp_path)
    again = register_local_scope(_blocks(tmp_path), project_root=Path("project"), config_path=client)

    assert again.conflicts == (), f"our own install came back as a stranger: {again.conflicts}"
    assert again.recorded



def test_every_spelling_the_client_already_uses_for_this_project_is_registered(
    tmp_path: Path,
) -> None:
    """⚠️ **The client does not normalise its project keys, and one project can hold several.**

    Measured 2026-08-19 on a real config with 313 project keys: the same directory appeared as both
    a backslash path and a forward-slash path, which is what a native launch and a Git Bash launch
    produce. Registering under one spelling would leave the other launching with no recall tools,
    with no error, which is the exact failure this whole change removes. So every existing key that
    resolves to this directory is written, rather than one invented one.
    """
    client = tmp_path / ".claude.json"
    project = tmp_path / "proj"
    project.mkdir()

    # ⚠️ THREE spellings, not two, and the third is what makes this test run at all on Linux.
    # `str(p)` and `p.as_posix()` are the SAME string on POSIX, because the separator already is a
    # slash, so the original two collapsed into one key and the test had nothing left to assert. It
    # passed on Windows, where they genuinely differ, and failed in CI. A `.` component survives as
    # a distinct string on every platform and resolves away everywhere.
    spellings = {
        str(project),
        project.as_posix(),
        os.path.join(str(tmp_path), ".", "proj"),
    }
    # The fixture guard comes FIRST and is not optional. The count below is derived from this set,
    # and a derived count alone would let a degenerate fixture pass in silence: one spelling,
    # one key, one match, green. Measured: 3 distinct spellings on Windows, 2 on POSIX.
    assert len(spellings) >= 2, (
        f"the fixture must offer more than one spelling on every platform, got {spellings}"
    )
    client.write_text(
        json.dumps({"projects": {spelling: {} for spelling in spellings}}), encoding="utf-8"
    )

    result = register_local_scope(_blocks(tmp_path), project_root=project, config_path=client)

    # Derived from the fixture rather than hardcoded, so the number cannot drift away from the set
    # it is meant to describe.
    assert set(result.project_keys) == spellings
    projects = json.loads(client.read_text(encoding="utf-8"))["projects"]
    for key in spellings:
        assert set(projects[key]["mcpServers"]) == {
            "default-docs",
            "default-code",
            "default-memory",
        }, f"the client launching as {key} would find no recall tools"


def test_a_project_the_client_has_never_opened_gets_one_invented_key(tmp_path: Path) -> None:
    """No existing key means Claude Code has not been run here, so there is nothing to match."""
    client = tmp_path / ".claude.json"
    client.write_text(json.dumps({"projects": {}}), encoding="utf-8")

    result = register_local_scope(_blocks(tmp_path), project_root=tmp_path, config_path=client)

    assert result.project_keys == (str(tmp_path.resolve()),)
    assert set(_registered(client, tmp_path)) == {
        "default-docs",
        "default-code",
        "default-memory",
    }


def test_a_name_belonging_to_another_install_is_refused_not_repointed(tmp_path: Path) -> None:
    """Server names are `{project}-{kind}`, so two installs sharing a project name collide.

    Overwriting would repoint the FIRST install's servers at a different corpus: its user would
    keep asking the same questions and quietly get another project's answers. The name is left
    alone, reported, and the remedy (a distinct `project`) is named.
    """
    client = tmp_path / ".claude.json"
    elsewhere = str(tmp_path / "some-other-install")
    client.write_text(
        json.dumps(
            {
                "projects": {
                    str(tmp_path): {
                        "mcpServers": {"default-docs": {"type": "stdio", "cwd": elsewhere}}
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    result = register_local_scope(_blocks(tmp_path), project_root=tmp_path, config_path=client)

    assert result.conflicts == (("default-docs", f"cwd {elsewhere}"),)
    assert "default-docs" not in result.registered
    assert set(result.registered) == {"default-code", "default-memory"}, (
        "the names that do NOT collide must still be registered"
    )
    assert _registered(client, tmp_path)["default-docs"]["cwd"] == elsewhere, (
        "the other install must be untouched"
    )


def test_the_written_file_has_no_temporary_left_behind_and_lf_endings(tmp_path: Path) -> None:
    client = tmp_path / ".claude.json"
    client.write_text("{}", encoding="utf-8")

    register_local_scope(_blocks(tmp_path), project_root=tmp_path, config_path=client)

    assert not list(tmp_path.glob("*.recall-tmp"))
    assert b"\r\n" not in client.read_bytes(), "CRLF would rewrite every line on every platform"


def test_the_server_command_launches_the_real_module_from_the_project_root(tmp_path: Path) -> None:
    """A `cwd` or module name that is wrong produces a server that never starts, silently."""
    blocks, _ = server_blocks(
        _plan(tmp_path),
        dsn="postgresql://recall:pw@127.0.0.1:5432/recall",
        promoted=frozenset({"default-docs"}),
        serving=frozenset({"default-docs", "default-memory"}),
    )
    document = mcp_config(blocks, project_root=tmp_path / "project")

    entry = document["mcpServers"]["default-docs"]  # type: ignore[index]
    assert entry["type"] == "stdio"
    assert entry["args"] == ["-m", "recall_mcp.server"]
    assert entry["cwd"] == str(tmp_path / "project")


# ----------------------------------------------------------------------------------------------
# Through the driver
# ----------------------------------------------------------------------------------------------


def test_the_profile_the_wizard_writes_is_the_one_the_real_ui_reads(tmp_path: Path) -> None:
    """The handoff, checked against the ACTUAL desktop window rather than against its source.

    `save_profile` had zero callers, and `main.py` fell back to a RELATIVE `compose_file` resolved
    against the process working directory, so Docker mode worked only when the app happened to be
    launched from the repository root.

    This constructs the real `MainWindow` offscreen from the written profile, because the UI is
    where an assumption about the shape would bite: verified by running it, the window makes NO
    runtime calls on startup and builds its scope selector from the PROFILE, so what is written
    here is exactly what the user sees. It also appends the corpus kind ITSELF, which is why
    `default_tenant` must be the project scope and not a complete tenant.
    """
    pytest.importorskip("PySide6")
    import os

    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PySide6.QtWidgets import QApplication

    from recall.desktop.profiles import load_profile
    from recall.desktop.ui import MainWindow
    from recall.wizard.wiring import write_runtime_profile

    compose = tmp_path / "store" / "docker-compose.recall.yml"
    written = write_runtime_profile(
        compose_path=compose,
        project="myapp",
        compose_project="recall-desktop",
        path=tmp_path / "runtime.json",
    )
    profile = load_profile(written)
    assert profile is not None

    assert Path(profile.compose_file or "").is_absolute(), (
        "a relative compose path resolves against the process working directory"
    )
    assert profile.default_tenant == "myapp", "the SCOPE, not a tenant: the UI appends the kind"

    class _Runtime:
        def start(self) -> None: ...
        def stop(self) -> None: ...
        def health(self) -> dict[str, str]:
            return {"status": "ready"}
        def list_tenants(self) -> list[str]:
            return ["myapp"]

    QApplication.instance() or QApplication([])
    window = MainWindow(profile, runtime=_Runtime())
    try:
        offered = [window.scope.itemText(i) for i in range(window.scope.count())]
        assert "myapp" in offered, f"the UI must offer the wizard's project, got {offered}"

        # And the tenant the UI would build from that scope is the wizard's, not `myapp-docs-docs`.
        from recall.desktop.models import SourceCategory, SourceSelection
        from recall.wizard.corpora import tenant_for

        selection = SourceSelection(
            category=SourceCategory.DOCUMENTS, paths=(tmp_path,), tenant="myapp"
        )
        assert selection.physical_tenant == tenant_for("myapp", "docs")
    finally:
        window.close()


def test_no_project_root_writes_nothing_and_says_so(tmp_path: Path) -> None:
    """Guessing a location for an MCP configuration is a side effect nobody can predict."""
    from recall.wizard.headless import run_headless
    from tests.test_wizard_state import _CountingSpy, _config

    report = run_headless(_config(tmp_path), services=_CountingSpy())

    assert report.registration is None
    assert report.servers == ()
    assert not list(tmp_path.glob(".mcp.json"))


def test_a_project_root_registers_the_servers_at_local_scope(tmp_path: Path) -> None:
    """⚠️ **And writes NO `.mcp.json`.** Both would be worse than either.

    Precedence is local, then project, then user, and entries are not merged, so a project-scoped
    file would be dead weight that still has to be trusted and kept in step. The prompt behind it is
    what stands between "the installer said it worked" and an assistant with no tools, and the
    wizard's audience cannot be expected to know that.
    """
    from recall.wizard.headless import run_headless
    from tests.test_wizard_state import _CountingSpy, _config

    root = tmp_path / "project"
    root.mkdir()
    client = tmp_path / ".claude.json"
    client.write_text(json.dumps({"projects": {}}), encoding="utf-8")
    config = _config(tmp_path, project_root=str(root))

    report = run_headless(config, services=_CountingSpy(), claude_config_path=client)

    assert report.registration is not None
    assert report.registration.recorded
    assert {b.name for b in report.servers} == {"default-docs", "default-code", "default-memory"}
    written = _registered(client, root)
    assert set(written) == {"default-docs", "default-code", "default-memory"}
    assert written["default-docs"]["cwd"] == str(root), "the server must run from the project"
    assert "mcpServers" not in json.loads(client.read_text(encoding="utf-8")), (
        "a top-level entry is USER scope, which would load these tenants in every unrelated "
        "checkout the user opens"
    )
    assert not (root / ".mcp.json").exists(), (
        "a project-scoped file is the one that carries the approval prompt"
    )
    rendered = report.render()
    assert "local scope" in rendered
    assert str(root) in rendered, (
        "the entry is keyed by path, so moving the project orphans it silently; the report has to "
        "say which path was registered"
    )
    assert "restart" in rendered, "a registration nobody restarts into is not yet loaded"

    # `.env` and `CLAUDE.md` too, and every file touched must be NAMED: these are block-scoped
    # edits to files the operator owns, and a block-scoped edit is invisible in a listing.
    assert (root / ".env").exists()
    assert (root / "CLAUDE.md").exists()
    assert "RECALL_DSN" in (root / ".env").read_text(encoding="utf-8")
    for path in report.files_written:
        assert str(path) in rendered, f"{path} was written and not reported"
    assert {p.name for p in report.files_written} >= {".env", "CLAUDE.md"}


def test_an_install_with_both_roots_reports_every_file_it_wrote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`data_root` AND `project_root`, which is the ORDINARY install and the one CI drives.

    Two steps write files, and they used to disagree about how: the stack and desktop step appended
    to `files_written`, the wiring step ASSIGNED, so on every install that reached both the compose
    file and `runtime.json` were dropped and the report never named them.

    `runtime.json` is the one that matters. It is written to the user's config directory, outside
    `data_root` and outside `project_root` both, so nothing in the install points at it and a
    listing of either root cannot find it. The report line is the only thing that tells an operator
    where their desktop UI was aimed.

    Asserted on the RENDERED report as well as the tuple, because the report is the artifact the
    operator actually reads, and a path in a field nobody prints is not a report.
    """
    import recall.wizard.headless as H
    from tests.test_wizard_state import _CountingSpy, _config

    # `dsn` and `data_root` are refused together by `load_config`, so an install that provisions is
    # the only shape in which both branches run. The database is faked; this test is about which
    # paths are reported, not about Docker.
    monkeypatch.setattr(
        H, "bring_up", lambda p, *, project_name, services=(), timeout=300.0: None
    )
    monkeypatch.setattr(H, "wait_for_database", lambda dsn, **kw: None)

    location = tmp_path / "store"
    root = tmp_path / "project"
    root.mkdir()
    profile = tmp_path / "runtime.json"
    config = _config(
        tmp_path,
        dsn=None,
        migration_dsn=None,
        data_root=str(location),
        project_root=str(root),
    )

    report = H.run_headless(
        config,
        services=_CountingSpy(),
        profile_path=profile,
        claude_config_path=_client(tmp_path),
    )

    compose = location / H.COMPOSE_NAME
    assert compose.exists() and profile.exists() and (root / ".env").exists(), (
        "the premise of this test is that all three steps wrote something"
    )

    # Every file on disk is in the tuple. Named individually rather than compared as a set, so a
    # failure says WHICH artifact stopped being reported instead of printing two path lists.
    written = set(report.files_written)
    assert compose in written, "the compose file is written by the stack step and must survive"
    assert profile in written, (
        "the desktop handoff lives outside both roots; losing it from the report loses it entirely"
    )
    assert root / ".env" in written
    assert root / "CLAUDE.md" in written

    # No path is reported twice: appending is the fix, and appending is also how a step gets counted
    # twice if it is ever run twice.
    assert len(report.files_written) == len(written), (
        f"duplicate paths in the report: {sorted(map(str, report.files_written))}"
    )

    rendered = report.render()
    for path in report.files_written:
        assert str(path) in rendered, f"{path} was written and not reported"


def test_an_operators_existing_claude_md_and_env_survive(tmp_path: Path) -> None:
    """The writers are BLOCK-scoped, and that is the whole reason they are reused rather than rewritten.

    An installer that replaced a project's `CLAUDE.md` would destroy work it has no business
    touching, and the operator would not find out until they noticed their own instructions gone.
    """
    from recall.wizard.headless import run_headless
    from tests.test_wizard_state import _CountingSpy, _config

    root = tmp_path / "project"
    root.mkdir()
    (root / "CLAUDE.md").write_text(
        "# My project\n\nMy own standing instructions, which must survive.\n", encoding="utf-8"
    )
    (root / ".env").write_text("MY_OWN_VAR=keepme\n", encoding="utf-8")

    run_headless(_config(tmp_path, project_root=str(root)), services=_CountingSpy())

    claude = (root / "CLAUDE.md").read_text(encoding="utf-8")
    assert "My own standing instructions, which must survive." in claude
    assert "Using recall" in claude, "and the wizard's own block must have been added"

    env = (root / ".env").read_text(encoding="utf-8")
    assert "MY_OWN_VAR=keepme" in env
    assert "RECALL_DSN" in env


def test_every_written_server_is_smoke_tested(tmp_path: Path) -> None:
    """Writing a config is not the same as the config working, and only one of those was checked.

    Found the honest way: adding the smoke step left this module green with the spies having no
    `smoke` method at all, because the driver caught the AttributeError and reported it. A feature
    nothing exercises is a feature nothing protects.
    """
    from recall.wizard.headless import run_headless
    from recall.wizard.headless import load_config
    from tests.test_wizard_headless import _Spy, _config, _write

    root = tmp_path / "project"
    root.mkdir()
    spy = _Spy()
    report = run_headless(
        load_config(_write(tmp_path, _config(tmp_path, project_root=str(root)))),
        services=spy,
        claude_config_path=_client(tmp_path),
    )

    assert spy.smoked == ["default-docs", "default-code", "default-memory"], "every written server must be queried"
    assert [s.tenant for s in report.smoke] == ["default-docs", "default-code", "default-memory"]
    assert all(s.answered for s in report.smoke)
    assert report.ok is True
    assert "smoke ok" in report.render()


def test_a_server_whose_query_raises_makes_the_install_not_ok(tmp_path: Path) -> None:
    """The corpora can all be built and promoted correctly and the CONFIG still not reach them.

    That is the only reason to run a smoke query at all, so a raise has to change the exit code.
    Without this the report would say "install complete" over a server that answers nothing.
    """
    from recall.wizard.headless import run_headless
    from recall.wizard.headless import load_config
    from tests.test_wizard_headless import _Spy, _config, _write

    root = tmp_path / "project"
    root.mkdir()
    spy = _Spy(smoke_raises={"default-code"})
    report = run_headless(
        load_config(_write(tmp_path, _config(tmp_path, project_root=str(root)))), services=spy
    )

    assert [o.tenant for o in report.outcomes] == ["default-docs", "default-code"], "the builds all succeeded"
    assert report.failures == (), "and nothing failed while building"
    assert report.ok is False, "yet the install is not ok, because one server cannot answer"
    rendered = report.render()
    assert "SMOKE FAILED" in rendered
    assert "NoActiveGeneration" in rendered, "the reason must reach the operator"


def test_an_abstention_is_not_a_smoke_failure(tmp_path: Path) -> None:
    """Abstaining is a trust decision the gate is entitled to make.

    A smoke test that treated it as broken would fail on a server behaving exactly as designed, and
    would push whoever hit it toward relaxing trust to make the installer happy.
    """
    from recall.wizard.headless import run_headless
    from recall.wizard.wiring import SmokeResult
    from recall.wizard.headless import load_config
    from tests.test_wizard_headless import _Spy, _config, _write

    class _Abstaining(_Spy):
        def smoke(self, block: object) -> SmokeResult:
            return SmokeResult(
                tenant=getattr(block, "tenant"),
                query="q",
                hits=0,
                abstained=True,
                trust_state="trusted",
                failure_code="CALIBRATION_MISSING",
            )

    root = tmp_path / "project"
    root.mkdir()
    report = run_headless(
        load_config(_write(tmp_path, _config(tmp_path, project_root=str(root)))),
        services=_Abstaining(),
        claude_config_path=_client(tmp_path),
    )

    assert all(s.answered for s in report.smoke), "reaching the gate IS answering"
    assert report.ok is True
    assert "abstained" in report.render()
    assert "SMOKE FAILED" not in report.render()


def test_an_unwritable_project_root_reports_rather_than_discarding_the_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every corpus is already built and promoted by the time the wiring is written.

    Throwing that away because a directory is read-only would be the expensive half of the trade,
    so the failure is reported against a `wiring` pseudo-tenant and the corpora keep their outcomes.

    🔁 This asserted `files_written == ()`, and ran with no `data_root`, so it was a claim about
    the WHOLE report that happened to be true because only one step could write anything. What the
    test means is narrower: the step that failed contributed nothing. Stated the old way it also
    quietly asserted that a wiring failure leaves an install with no files at all, which is wrong
    for the ordinary install and would have gone green on a bug that discarded the stack's paths.
    So `data_root` is set here now, and both halves are asserted separately: nothing under the
    project root, everything the earlier steps already earned.
    """
    import recall.wizard.headless as H
    from tests.test_wizard_state import _CountingSpy, _config

    def _boom(**kwargs: object) -> tuple[Path, ...]:
        raise OSError(13, "Permission denied")

    monkeypatch.setattr(H, "write_project_files", _boom)
    monkeypatch.setattr(
        H, "bring_up", lambda p, *, project_name, services=(), timeout=300.0: None
    )
    monkeypatch.setattr(H, "wait_for_database", lambda dsn, **kw: None)

    location = tmp_path / "store"
    root = tmp_path / "project"
    root.mkdir()
    profile = tmp_path / "runtime.json"
    report = H.run_headless(
        _config(
            tmp_path,
            dsn=None,
            migration_dsn=None,
            data_root=str(location),
            project_root=str(root),
        ),
        services=_CountingSpy(),
        profile_path=profile,
    )

    assert [o.tenant for o in report.outcomes] == ["default-docs", "default-code"], "the builds must survive"
    assert [f.tenant for f in report.failures] == ["wiring"]
    assert [p for p in report.files_written if root in p.parents] == [], (
        "the step that failed wrote nothing, so it must claim nothing"
    )
    assert set(report.files_written) == {location / H.COMPOSE_NAME, profile}, (
        "and it must not take the earlier steps' files down with it: the desktop handoff is "
        "written outside both roots, so an unreported one is an unfindable one"
    )
    assert report.ok is False, "an install nobody can reach is not complete"


def test_registration_refuses_rather_than_inventing_a_client_config(tmp_path: Path) -> None:
    """No client config means Claude Code has not run here; writing one would invent its file.

    An unreadable one is left strictly alone. In both cases the report has to SAY the servers are
    not registered, because that is the difference between "recall is broken" and "your client has
    never been started".
    """
    blocks = _blocks(tmp_path)

    absent = register_local_scope(
        blocks, project_root=tmp_path, config_path=tmp_path / "none.json"
    )
    assert not absent.recorded
    assert "no Claude Code config" in absent.skipped_reason
    assert not (tmp_path / "none.json").exists(), "the wizard must not create another app's config"

    corrupt = tmp_path / "broken.json"
    corrupt.write_text("{not json", encoding="utf-8")
    refused = register_local_scope(blocks, project_root=tmp_path, config_path=corrupt)
    assert not refused.recorded
    assert "left untouched" in refused.skipped_reason
    assert corrupt.read_text(encoding="utf-8") == "{not json", "an unreadable client config is kept"


def test_a_registration_that_was_skipped_is_reported_not_swallowed(tmp_path: Path) -> None:
    """The silent-nothing failure, stated as a test.

    An install whose corpora built and whose servers reached no client looks identical, from the
    user's side, to one that is simply broken. The report is the only thing that can tell them
    apart, so a skip must never render as a success.

    ⚠️ **`_Spy` rather than `_CountingSpy`, and that is the whole test.** `_CountingSpy` has no
    `smoke`, so every smoke query raises and `ok` is already False for a reason that has nothing to
    do with the registration. Written that way first, this passed with the `unregistered` term
    deleted from `ok` — it was blessing a failure it did not cause. Caught by mutation, not by
    reading.
    """
    from recall.wizard.headless import load_config, run_headless
    from tests.test_wizard_headless import _Spy, _config, _write

    root = tmp_path / "project"
    root.mkdir()
    report = run_headless(
        load_config(_write(tmp_path, _config(tmp_path, project_root=str(root)))),
        services=_Spy(),
        claude_config_path=tmp_path / "absent.json",
    )

    assert report.registration is not None
    assert not report.registration.recorded
    rendered = report.render()
    assert "NOT registered" in rendered
    assert "no Claude Code config" in rendered

    # Nothing else is wrong: every corpus built, and every smoke query answered. The registration
    # is the ONLY reason this install is incomplete, which is what makes it a test of the term.
    assert not any(s.error for s in report.smoke), "no smoke query failed"
    assert [s.tenant for s in report.smoke] == ["default-docs", "default-code", "default-memory"], (
        "smoke must run even when no client was registered: which half failed is the "
        "information the operator needs, and gating it hides the working half"
    )
    assert report.ok is False, "an install no client can reach is not complete"
    assert "no client was registered" in rendered, (
        "the head line must name the registration, not blame a corpus that built correctly"
    )


def test_the_suite_never_touches_the_real_client_config() -> None:
    """The guard for the fixture that keeps the rest of this honest.

    ⚠️ Written after a run of these very tests appended FIVE entries to the developer's own
    `~/.claude.json`, each pointing at a `pytest-of-.../` directory. Nothing was corrupted, which is
    why it passed unnoticed: the writer is atomic, it backs up first, and no existing project was
    modified. `tests/conftest.py::_confine_claude_client_config` now redirects `Path.home` for every
    test; an autouse fixture that stops working breaks nothing and fails nothing, so this asserts it.
    """
    from recall.wizard.wiring import claude_config_path

    resolved = claude_config_path()
    assert "fake-home" in str(resolved) or "pytest" in str(resolved).lower(), (
        f"the client config resolves to {resolved}, which looks like the real one; "
        "tests/conftest.py::_confine_claude_client_config is meant to redirect Path.home"
    )


def test_project_keys_follow_each_platform_s_own_case_rule(tmp_path: Path) -> None:
    """⛔ Do NOT casefold this comparison, on either platform.

    `_same_directory` compares `Path(key).resolve() == project_root.resolve()`, and `PurePath.__eq__`
    is case-insensitive on Windows and case-sensitive on POSIX. That is each platform's own rule
    about what a directory IS, and it is the rule this has to follow.

    Forcing case-insensitivity everywhere — the obvious way to write this, and the way a peer
    session first wrote it — would match `/home/me/Proj` against `/home/me/proj` on Linux. Those are
    two different directories, so a recall server would be registered under an unrelated project and
    answer about the wrong corpus: the exact corpus-boundary failure local scope exists to prevent,
    reintroduced by a fix for a different bug.

    ⚠️ **This is invisible from a Windows machine**, where both implementations agree on every
    input, which is why it needs a test that asserts the two behaviours SEPARATELY rather than
    asserting whichever one the machine running the suite happens to have. CI runs on Linux and
    development happens on Windows, so both branches are exercised somewhere.
    """
    from recall.wizard.wiring import _same_directory

    # ⚠️ Deliberately NOT created. On Windows `resolve()` asks the filesystem for the canonical
    # casing of a path that EXISTS, so two spellings of one real directory collapse to the same
    # string and a plain `==` on the strings would pass too. That hides the rule being tested here.
    # An absent path keeps the casing it was given, so the comparison itself is what answers.
    lowered = tmp_path / "proj"
    shouted = str(tmp_path / "PROJ")

    if os.name == "nt":
        assert _same_directory(shouted, lowered), (
            "on Windows these name one directory, and refusing to match would report the user's "
            "own project as a stranger"
        )
    else:
        assert not _same_directory(shouted, lowered), (
            "on POSIX these are two directories, and matching them would register this project's "
            "servers under an unrelated one"
        )


def test_the_project_root_is_created_when_only_the_last_directory_is_missing(
    tmp_path: Path,
) -> None:
    """The regression `03456359` left behind, and the reason it went unnoticed for a release.

    Nothing creates `project_root`. It used to appear as a side effect of `write_mcp_config`, whose
    `path.parent.mkdir(parents=True, exist_ok=True)` ran on the way to writing
    `project_root/.mcp.json`; when registration moved to local scope in `~/.claude.json` that
    function was deleted and the mkdir went with it. Every existing test passed, because every one
    of them creates the directory first — which is exactly what a real first install does not do.

    ⚠️ **The assertion that matters is `.env` existing, not `.is_dir()`.** A guard that created the
    directory and then failed to write into it would satisfy a directory check and still leave the
    user with the CI failure this fixes. The files are the deliverable.
    """
    project_root = tmp_path / "project"
    assert not project_root.exists(), "the point of this test is that nothing has created it"

    written = write_project_files(
        project_root=project_root,
        dsn="postgresql://recall:recall@127.0.0.1:1/recall",
        embedder="hashing",
        memory_dir=tmp_path / "memory",
    )

    assert (project_root / ".env").is_file(), "the .env this install exists to write"
    assert (project_root / "CLAUDE.md").is_file()
    assert project_root / ".env" in written, "what was written must be what is reported"


def test_a_project_root_that_is_a_file_is_not_written_into(tmp_path: Path) -> None:
    """`exist_ok=True` must not read a FILE at that path as "already there".

    `Path.mkdir` suppresses `FileExistsError` only for a directory, so this raises rather than
    proceeding to open `afile/.env`. Asserted because the suppression is the sort of thing a later
    edit widens to `parents=True, exist_ok=True` without noticing it changes this case.

    It raises rather than refusing because `run_headless` catches `OSError` around this call and
    reports a `wiring` failure with the offending filename. `load_config` is what refuses this by
    name, before any corpus is built.
    """
    project_root = tmp_path / "afile"
    project_root.write_text("not a directory", encoding="utf-8")

    with pytest.raises(OSError):
        write_project_files(
            project_root=project_root,
            dsn="postgresql://recall:recall@127.0.0.1:1/recall",
            embedder="hashing",
            memory_dir=tmp_path / "memory",
        )

    assert project_root.read_text(encoding="utf-8") == "not a directory", "left alone"


def test_the_tree_above_a_project_root_is_never_manufactured(tmp_path: Path) -> None:
    """`parents=False`, deliberately: a missing PARENT is a typo, and this is not where it is caught.

    Creating the whole tree here would silently produce directories the user never named, at the end
    of a thirty-minute install, on the basis of a path they mistyped. `load_config` refuses that
    case by name before anything is built; see
    `tests/test_wizard_headless.py::test_a_project_root_whose_parent_is_absent_is_refused`.
    """
    project_root = tmp_path / "nowhere" / "project"

    with pytest.raises(OSError):
        write_project_files(
            project_root=project_root,
            dsn="postgresql://recall:recall@127.0.0.1:1/recall",
            embedder="hashing",
            memory_dir=tmp_path / "memory",
        )

    assert not (tmp_path / "nowhere").exists(), "no directory the user did not name"
