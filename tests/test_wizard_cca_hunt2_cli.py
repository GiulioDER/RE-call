"""Regression tests for the second DEEP CCA hunt of the first-run code, the CLI half:
`recall setup --lite`, `recall wizard`, `recall quickstart`, `recall search`, `recall scopes`,
`recall forget`, `recall demo`, `recall code` and `recall lite refresh`.

Each test names the finding it pins. Red proof for every test is taken on the remote test host
against the pre-fix snapshot of the production files, and the verdicts are recorded in the pull
request. A test that needs a symbol the fix added imports it after its first assertion, so the
pre-fix run fails on behaviour rather than on the import. Nothing here touches a database, Docker,
the network or the real home directory: stores, migrators and embedders are fakes or sentinels.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


def _reached(label: str) -> Any:
    """A stand-in that ends the command with a recognisable message when it is reached."""

    def stop(*args: object, **kwargs: object) -> Any:
        raise SystemExit(f"reached {label}")

    return stop


def _quiet_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "RECALL_ENTAILMENT", "RECALL_SOURCE_POLICY_FILE", "RECALL_INDEX_MODE",
        "RECALL_ALLOW_INSECURE_DSN", "RECALL_INDEX_ROOT", "RECALL_EMBEDDER",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RECALL_ENV", "development")
    monkeypatch.setenv("RECALL_EMBED_CACHE", "off")


# ----------------------------------------------------------------------------------------------
# BUG-008, STAKES-010: saving the interactive wizard's answers
# ----------------------------------------------------------------------------------------------


def _interview(monkeypatch: pytest.MonkeyPatch, data_root: Path) -> None:
    from recall.wizard import interactive

    class _Prompter:
        def ask(self, prompt: str, default: str | None = None) -> str:
            return str(default)

    monkeypatch.setattr(interactive, "stdin_prompter", lambda: _Prompter())
    monkeypatch.setattr(
        interactive, "ask_config", lambda prompter, default_root: {"data_root": str(data_root)}
    )


def test_an_unwritable_save_path_ends_with_the_path_not_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """BUG-008. Invariant: an OSError while saving the interview's answers ends `recall wizard`
    with a SystemExit naming the path it could not write. Failure caught: the OSError escaped as a
    traceback after the whole interview. Target: `setup_wizard._cmd_wizard`."""
    from recall.cli_commands.setup_wizard import _cmd_wizard
    from recall.wizard import interactive

    data_root = tmp_path / "D"
    _interview(monkeypatch, data_root)

    def refuse(document: dict[str, Any], target: Path) -> Path:
        raise PermissionError(13, "Permission denied", str(target))

    monkeypatch.setattr(interactive, "write_config", refuse)
    try:
        _cmd_wizard(argparse.Namespace(gui=False, config=None, headless=False))
    except SystemExit as exc:
        message = str(exc.code)
    except OSError as exc:
        message = f"escaped as {type(exc).__name__}"
    else:
        message = "returned"
    assert "cannot save the configuration to" in message, message
    assert str(data_root / "wizard.json") in message


def test_a_rerun_keeps_the_previous_wizard_json_as_a_private_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """STAKES-010. Invariant: when the save target already exists, its previous content is kept
    as `wizard.json.<timestamp>`, owner-only (it can hold a DSN password), never overwriting an
    earlier backup, and the output says so. Failure caught: a re-run replaced the only record
    `recall uninstall` uses to find the first install's registrations, before anything was
    validated. Target: `setup_wizard._cmd_wizard`."""
    from recall.cli_commands.setup_wizard import _cmd_wizard
    from recall.store import DEFAULT_TABLE, DEFAULT_TENANT
    from recall.wizard import headless
    from recall.wizard.pipeline import PipelineRefusal

    data_root = tmp_path / "D"
    data_root.mkdir()
    target = data_root / "wizard.json"
    target.write_text('{"project": "first"}\n', encoding="utf-8")
    _interview(monkeypatch, data_root)

    def stop(path: object) -> Any:
        raise PipelineRefusal("stopped by the test after the save")

    monkeypatch.setattr(headless, "load_config", stop)
    for _run in range(2):
        with pytest.raises(SystemExit, match="stopped by the test"):
            _cmd_wizard(
                argparse.Namespace(
                    gui=False, config=None, headless=False, tenant=DEFAULT_TENANT,
                    table=DEFAULT_TABLE, no_state=True, state=None, fresh=False,
                )
            )

    backups = sorted(p for p in data_root.iterdir() if p.name.startswith("wizard.json."))
    contents = sorted(p.read_text(encoding="utf-8") for p in backups)
    assert len(backups) == 2, [p.name for p in backups]
    assert '{"project": "first"}\n' in contents, contents
    assert "kept the previous configuration as" in capsys.readouterr().out
    if sys.platform != "win32":
        for backup in backups:
            assert backup.stat().st_mode & 0o777 == 0o600, oct(backup.stat().st_mode)


# ----------------------------------------------------------------------------------------------
# SEC-005: the wizard refuses published credentials in fact_write_dsn too
# ----------------------------------------------------------------------------------------------


def test_the_wizard_refuses_published_credentials_in_the_fact_write_dsn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SEC-005. Invariant: `recall wizard` runs the secure-DSN guard on fact_write_dsn as on dsn
    and migration_dsn. Failure caught: a fact_write_dsn with the published recall:recall
    credentials against a remote host reached `run_headless`, which creates a LOGIN role with that
    password. Target: `setup_wizard._cmd_wizard`."""
    from recall.cli_commands.setup_wizard import _cmd_wizard
    from recall.store import DEFAULT_TABLE, DEFAULT_TENANT
    from recall.wizard import headless

    monkeypatch.delenv("RECALL_ALLOW_INSECURE_DSN", raising=False)
    config = SimpleNamespace(
        dsn=None,
        migration_dsn=None,
        fact_write_dsn="postgresql://recall:recall@db.example.com:5432/recall",
    )
    monkeypatch.setattr(headless, "load_config", lambda path: config)
    monkeypatch.setattr(headless, "run_headless", _reached("run_headless"))
    with pytest.raises(SystemExit) as caught:
        _cmd_wizard(
            argparse.Namespace(
                gui=False, config="wizard.json", headless=True, tenant=DEFAULT_TENANT,
                table=DEFAULT_TABLE, no_state=True, state=None, fresh=False,
            )
        )
    assert "fact_write_dsn is refused" in str(caught.value.code), caught.value.code


# ----------------------------------------------------------------------------------------------
# DEPLOY-007: `recall setup --lite` names both extras when the embedder is unusable
# ----------------------------------------------------------------------------------------------


def test_setup_lite_prints_its_install_hint_for_an_unusable_embedder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DEPLOY-007. Invariant: when the embedder cannot be built, `recall setup --lite` ends with
    the lite install line naming both `fastembed` and `mcp`. Failure caught: `_make_embedder`
    raises SystemExit, which the `except Exception` around it never caught, so the hint never
    printed. Target: `setup_wizard._setup_lite`.

    The stub raises RuntimeError, not the ImportError a missing extra really raises: the red-state
    proof reads an ImportError in the pre-fix output as a test that never reached the code."""
    from recall.cli_commands import _shared
    from recall.cli_commands.setup_wizard import _setup_lite

    def missing(name: str) -> Any:
        raise RuntimeError("the fastembed extra is not installed")

    monkeypatch.setattr(_shared, "resolve_embedder", missing)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as caught:
        _setup_lite(
            argparse.Namespace(embedder="fastembed", memory=str(tmp_path), store=None, no_register=True)
        )
    assert 'recall-rag[fastembed,mcp]' in str(caught.value.code), caught.value.code


# ----------------------------------------------------------------------------------------------
# DAT-009: the quickstart never migrates an older recall install's `chunks`
# ----------------------------------------------------------------------------------------------


def test_quickstart_migrates_chunks_only_on_a_database_with_no_recall_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DAT-009. Invariant: when the quickstart table raises SchemaTooOld, the default `chunks`
    target is migrated only if the ledger records nothing; an older install is refused, naming
    `recall schema apply`. Failure caught: every SchemaTooOld was read as "no install to damage",
    so `--existing-dsn` against an older install migrated its real `chunks` and globals forward.
    Target: `setup_wizard._quickstart`."""
    import recall.schema as schema
    from recall.cli_commands import setup_wizard
    from recall.quickstart import QUICKSTART_TABLE
    from recall.store import DEFAULT_TABLE

    _quiet_environment(monkeypatch)
    monkeypatch.setattr(setup_wizard, "PgVectorStore", _reached("the store"))

    import recall.quickstart as quickstart

    # Control for the provisioned path: the quickstart's OWN stack, reused across upgrades.
    monkeypatch.setattr(quickstart, "docker_unavailable_reason", lambda: None)
    monkeypatch.setattr(
        quickstart, "provision", lambda: ("postgresql://u:p@127.0.0.1:1/recall", 1, None, True)
    )

    def run(
        ledger_states: tuple[str, ...], existing_dsn: str | None = "postgresql://u:p@127.0.0.1:1/recall"
    ) -> tuple[list[str], str]:
        applied: list[str] = []

        def apply(dsn: str, *, table: str, dim: int) -> tuple[()]:
            applied.append(table)
            if table == QUICKSTART_TABLE and DEFAULT_TABLE not in applied:
                raise schema.SchemaTooOld("globals first")
            return ()

        monkeypatch.setattr(schema, "apply_migrations", apply)
        monkeypatch.setattr(
            schema,
            "schema_status",
            lambda dsn, *, table, dim: SimpleNamespace(
                migrations=tuple(SimpleNamespace(state=s) for s in ledger_states)
            ),
        )
        with pytest.raises(SystemExit) as caught:
            setup_wizard._quickstart(
                argparse.Namespace(
                    remove=False, existing_dsn=existing_dsn, embedder="hashing",
                )
            )
        return applied, str(caught.value.code)

    applied, message = run(("applied", "pending"))
    assert DEFAULT_TABLE not in applied, applied
    assert "recall schema apply" in message, message

    applied, message = run(("pending", "pending"))
    assert DEFAULT_TABLE in applied, applied
    assert message == "reached the store", message

    # Its own provisioned stack after an upgrade is partly applied and is still migrated forward.
    applied, message = run(("applied", "pending"), existing_dsn=None)
    assert DEFAULT_TABLE in applied, applied
    assert message == "reached the store", message


# ----------------------------------------------------------------------------------------------
# BUG-010: an empty --folder or --facet is a message, before the embedder loads
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("flag", ["folder", "facet"])
def test_an_empty_search_scope_is_refused_before_the_embedder_loads(
    flag: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """BUG-010. Invariant: `recall search q --folder ""` (or --facet) ends with
    "recall search: ..." before the embedder or the store is opened. Failure caught: the Scope
    ValueError was raised inside the search call, after the model loaded, and escaped as a
    traceback. Target: `index_search._cmd_search`."""
    from recall.cli_commands import index_search

    _quiet_environment(monkeypatch)
    monkeypatch.setattr(index_search, "_make_embedder", _reached("the embedder"))
    args = argparse.Namespace(
        query="q", folder=None, facet=None, embedder="hashing", dsn="postgresql://u:p@127.0.0.1:1/recall",
        table="chunks", tenant="default", _runtime_route=None,
    )
    setattr(args, flag, "  ")
    with pytest.raises(SystemExit) as caught:
        index_search._cmd_search(args)
    assert str(caught.value.code).startswith("recall search:"), caught.value.code


# ----------------------------------------------------------------------------------------------
# NUM-004: the scope listing's header lines up with its rows
# ----------------------------------------------------------------------------------------------


def test_the_scope_header_lines_up_with_values_shorter_than_its_label(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """NUM-004. Invariant: every line of the scope listing has the same width, header included.
    Failure caught: the column width ignored the label, so facets `a` and `b` put the header's
    columns four characters right of the rows'. Target: `index_search._print_scopes`."""
    from recall.cli_commands.index_search import _print_scopes

    _print_scopes("facet", [("a", 12, 1), ("b", 3, 1)], 0)
    lines = capsys.readouterr().out.splitlines()
    assert len({len(line) for line in lines}) == 1, lines


# ----------------------------------------------------------------------------------------------
# DOC-017: the strict refusal links to a calibration doc a pip user can open
# ----------------------------------------------------------------------------------------------


def test_the_strict_refusal_points_at_the_published_calibration_doc() -> None:
    """DOC-017. Invariant: the strict-refusal message names the published URL of the calibration
    doc. Failure caught: it named the relative `docs/CALIBRATION.md`, which the wheel does not
    ship. Target: `index_search.refusal_message`."""
    from recall.cli_commands.index_search import refusal_message
    from recall.trust_policy import TrustFailureCode, TrustRefusal

    message = refusal_message(
        TrustRefusal(
            code=TrustFailureCode.CALIBRATION_MISSING, calibration_status="missing",
            tenant_id="default", generation_id=None,
        )
    )
    assert "https://github.com/GiulioDER/RE-call/blob/master/docs/CALIBRATION.md" in message


# ----------------------------------------------------------------------------------------------
# DOC-021: the forget preview names the tombstone on the generation route
# ----------------------------------------------------------------------------------------------


def test_the_forget_preview_warns_of_the_tombstone_on_the_generation_route(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """DOC-021. Invariant: a `recall forget` dry run on the generation route (here
    RECALL_INDEX_MODE=generation under development) says the sources will also be tombstoned.
    Failure caught: the preview said nothing, and the help tied the tombstone to production only,
    so a developer on the generation route barred a URI from every later build unwarned.
    Target: `index_search._cmd_remove`."""
    import recall.generation_store as generation_store
    from recall.cli_commands import index_search

    _quiet_environment(monkeypatch)
    monkeypatch.setenv("RECALL_INDEX_MODE", "generation")

    class _GenerationStore:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def __enter__(self) -> _GenerationStore:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def check_schema(self) -> None:
            return None

        def sources_in_any_generation(self, names: list[str]) -> set[str]:
            return {"memo.md"}

        def manifest_uris_matching(self, names: list[str]) -> set[str]:
            return set()

        def sources_in_legacy_table(self, names: list[str]) -> set[str]:
            return set()

    monkeypatch.setattr(generation_store, "GenerationStore", _GenerationStore)
    monkeypatch.setattr(index_search, "_make_embedder", lambda name: SimpleNamespace(dim=8))
    index_search._cmd_forget(
        argparse.Namespace(
            dsn="postgresql://u:p@127.0.0.1:1/recall", tenant="t", table="chunks",
            sources=["memo.md"], yes=False, vacuum=False, embedder="hashing", _runtime_route=None,
        )
    )
    out = capsys.readouterr().out
    assert "DRY RUN" in out, out
    assert "TOMBSTONED" in out, out


# ----------------------------------------------------------------------------------------------
# SEC-008, STAKES-008: `recall demo` and `recall code`
# ----------------------------------------------------------------------------------------------


def _demo_args(table: str) -> argparse.Namespace:
    return argparse.Namespace(
        dsn="postgresql://u:p@127.0.0.1:1/recall", table=table, tenant="default",
        embedder="hashing", _runtime_route=None,
    )


@pytest.mark.parametrize("command", ["demo", "code"])
def test_demo_and_code_refuse_when_a_source_policy_is_configured(
    command: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SEC-008. Invariant: `recall demo` and `recall code` refuse, as `recall scopes` does, when a
    source policy is configured, before any embedder or store work. Failure caught: both indexed
    and printed names and previews of hits the policy withholds. Target:
    `index_search._cmd_demo`, `index_search._cmd_code`."""
    from recall.cli_commands import index_search

    _quiet_environment(monkeypatch)
    monkeypatch.setattr(index_search, "load_source_policy", lambda: object())
    monkeypatch.setattr(index_search, "_make_embedder", _reached("the embedder"))
    monkeypatch.setattr(index_search, "PgVectorStore", _reached("the store"))
    handler = index_search._cmd_demo if command == "demo" else index_search._cmd_code
    with pytest.raises(SystemExit) as caught:
        handler(_demo_args("scratch_demo"))
    assert "source policy is configured" in str(caught.value.code), caught.value.code


def test_demo_refuses_to_index_into_the_default_table(monkeypatch: pytest.MonkeyPatch) -> None:
    """STAKES-008. Invariant: `recall demo` refuses the default table `chunks`, where real memory
    lives, before loading an embedder. Failure caught: it indexed the fictional demo corpus (or any
    `corpus/` in the cwd) into the configured table and tenant, beside real memory.
    Target: `index_search._cmd_demo`."""
    from recall.cli_commands import index_search

    _quiet_environment(monkeypatch)
    monkeypatch.setattr(index_search, "load_source_policy", lambda: None)
    monkeypatch.setattr(index_search, "_make_embedder", _reached("the embedder"))
    monkeypatch.setattr(index_search, "PgVectorStore", _reached("the store"))
    with pytest.raises(SystemExit) as caught:
        index_search._cmd_demo(_demo_args("chunks"))
    assert "refusing to index the demo corpus into 'chunks'" in str(caught.value.code), caught.value.code


# ----------------------------------------------------------------------------------------------
# ENV-006, NUM-007, PERF-004: `recall lite refresh`
# ----------------------------------------------------------------------------------------------


def _memo_folder(root: Path) -> Path:
    memory = root / "memory"
    memory.mkdir(parents=True)
    # An old mtime with a non-zero nanosecond part, so no snapshot reads the files as racy and a
    # second refresh is a genuine no-op.
    stamp = 1_600_000_000_123_456_789
    for i in range(3):
        path = memory / f"m{i}.md"
        path.write_text(f"# memo {i}\n\nbody {i}\n", encoding="utf-8")
        os.utime(path, ns=(stamp, stamp))
    return memory


def _cli_code(argv: list[str]) -> object:
    from recall import cli

    try:
        cli.main(argv)
    except SystemExit as exc:
        return exc.code
    return 0


def test_lite_refresh_follows_a_lite_dsn_when_no_store_is_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """ENV-006. Invariant: with no --store, `recall lite refresh` uses the file a sqlite:/// DSN
    names before falling back to ./.recall/memory.db. Failure caught: it ignored the DSN that
    `recall setup --lite` prints and `recall search` follows, and reported no store.
    Target: `lite_cmd._cmd_refresh`."""
    from recall.lite import LiteStore

    _quiet_environment(monkeypatch)
    memory = _memo_folder(tmp_path / "notes")
    db = tmp_path / "elsewhere" / "memory.db"
    LiteStore(db, dim=64).close()
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    code = _cli_code(
        ["--dsn", f"sqlite:///{db.as_posix()}", "--embedder", "hashing", "lite", "refresh",
         "--memory", str(memory)]
    )
    assert "no lite store" not in str(code), code
    assert code in (0, None), (code, capsys.readouterr())


@pytest.mark.parametrize("value", ["nan", "inf"])
def test_lite_refresh_refuses_a_non_finite_wait(
    value: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """NUM-007. Invariant: `--wait` refuses nan and inf at parse time (exit 2), as the
    RECALL_LITE_WRITE_WAIT_S path refuses them. Failure caught: nan silently meant "try once" and
    inf "wait forever". Target: `lite_cmd.register` (the `--wait` type)."""
    _quiet_environment(monkeypatch)
    monkeypatch.chdir(tmp_path)
    code = _cli_code(["lite", "refresh", "--wait", value, "--check"])
    assert code == 2, code
    assert "finite" in capsys.readouterr().err


def test_an_unchanged_lite_refresh_walks_the_memo_folder_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """PERF-004. Invariant: one `recall lite refresh` of an unchanged folder takes exactly one
    snapshot, shared by the refresh and the budget warning. Failure caught: the warning walked and
    stat'ed the whole tree again after every refresh, so a scheduled no-op walked it twice.
    Target: `lite_cmd._cmd_refresh`."""
    import recall.lite.refresh as refresh_mod
    from recall.lite import LiteStore

    _quiet_environment(monkeypatch)
    memory = _memo_folder(tmp_path)
    db = tmp_path / ".recall" / "memory.db"
    LiteStore(db, dim=64).close()
    argv = ["lite", "refresh", "--store", str(db), "--memory", str(memory), "--embedder", "hashing"]
    assert _cli_code(argv) in (0, None), capsys.readouterr()
    capsys.readouterr()

    real = refresh_mod.snapshot
    walks: list[object] = []

    def counting(roots: Any) -> Any:
        walks.append(roots)
        return real(roots)

    monkeypatch.setattr(refresh_mod, "snapshot", counting)
    assert _cli_code(argv) in (0, None)
    assert len(walks) == 1, f"{len(walks)} walks of the memo folder for one unchanged refresh"
    assert "unchanged" in capsys.readouterr().out
