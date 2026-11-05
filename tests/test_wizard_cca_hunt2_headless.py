"""Regression tests for the second DEEP CCA hunt of the first-run code: the headless wizard driver,
its resumable state and `add_project`.

Each test names the finding it pins. Red proof for every test is taken on the remote test host
against the pre-fix snapshot of the production files, and the verdicts are recorded in the pull
request. A test that needs a symbol the fix added imports it after its first assertion, or reaches
it through an entry point that already existed, so the pre-fix run fails on behaviour rather than
on the import. An exception the fix turns into a report is caught by the test and asserted on,
so the pre-fix run fails in that assertion rather than as an error.
"""

from __future__ import annotations

import importlib.util
import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from recall.wizard.headless import ConfigRefusal, HeadlessConfig, load_config
from tests.test_wizard_headless import _config, _Spy, _write


def _mcp_installed(monkeypatch: pytest.MonkeyPatch, present: bool) -> None:
    """Pin whether `mcp` is importable, so a test does not depend on the host's extras."""
    real_find_spec = importlib.util.find_spec

    def find_spec(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "mcp":
            return object() if present else None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)


def _client_config(tmp_path: Path) -> Path:
    client = tmp_path / ".claude.json"
    client.write_text(json.dumps({"projects": {}}), encoding="utf-8")
    return client


def _data_root_config(tmp_path: Path) -> HeadlessConfig:
    payload = _config(tmp_path, data_root=str(tmp_path / "store"))
    del payload["dsn"]
    del payload["migration_dsn"]
    return load_config(_write(tmp_path, payload))


class _FakeCursor:
    def __init__(self, row: Any) -> None:
        self._row = row

    def fetchone(self) -> Any:
        return self._row


class _FakeConnection:
    """Records statements; answers every query with no row, as for a role that does not exist."""

    def __init__(self, log: list[str], hashed: list[tuple[bytes, bytes, bytes]]) -> None:
        self._log = log
        self._hashed = hashed
        self.pgconn = SimpleNamespace(encrypt_password=self._encrypt)

    def _encrypt(self, password: bytes, user: bytes, algorithm: bytes) -> bytes:
        self._hashed.append((password, user, algorithm))
        return b"SCRAM-SHA-256$4096:c2FsdA==$c3RvcmVk:c2VydmVy"

    def __enter__(self) -> _FakeConnection:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def execute(self, statement: Any, params: Any = None) -> _FakeCursor:
        self._log.append(statement if isinstance(statement, str) else repr(statement))
        return _FakeCursor(None)

    def commit(self) -> None:
        return None

    def transaction(self) -> _FakeConnection:
        return self


# ----------------------------------------------------------------------------------------------
# BUG-006, BUG-007: a wiring failure is reported, and reported as what it was
# ----------------------------------------------------------------------------------------------


def test_an_undecodable_project_file_is_reported_and_the_smoke_still_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """BUG-006. Invariant: a ValueError from writing the project files (a CLAUDE.md saved in a
    legacy codepage raises UnicodeDecodeError) becomes a `wiring` Failure naming the project, and
    the smoke step still runs. Failure caught: only OSError was caught, so the error escaped
    `run_headless` after every corpus was promoted and the report was lost.
    Target: the wiring block of `headless.run_headless`."""
    from recall.wizard import headless as H

    _mcp_installed(monkeypatch, True)
    root = tmp_path / "project"
    root.mkdir()

    def undecodable(**_kwargs: Any) -> Any:
        raise UnicodeDecodeError("utf-8", b"\x92", 0, 1, "invalid start byte")

    monkeypatch.setattr(H, "write_project_files", undecodable)
    spy = _Spy()
    report: Any = None
    raised: Exception | None = None
    try:
        report = H.run_headless(
            load_config(_write(tmp_path, _config(tmp_path, project_root=str(root)))),
            services=spy,
            claude_config_path=_client_config(tmp_path),
        )
    except Exception as exc:  # noqa: BLE001 - the pre-fix escape is what this test observes
        raised = exc
    assert raised is None, f"the wiring error escaped run_headless: {raised!r}"
    wiring = [f.error for f in report.failures if f.tenant == "wiring"]
    assert any(str(root) in error for error in wiring), report.failures
    assert spy.smoked, "the smoke step must still run after a wiring failure"


def test_a_failed_registration_write_is_not_blamed_on_a_missing_project_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """BUG-007. Invariant: when `register_local_scope` raises OSError, the report says the servers
    were NOT registered and why, and never that the config has no `project_root`. Failure caught:
    `registration` stayed None, so the report told the user to set a key they had already set.
    Target: the wiring block of `headless.run_headless` and `HeadlessReport.render`."""
    from recall.wizard import headless as H

    _mcp_installed(monkeypatch, True)
    root = tmp_path / "project"
    root.mkdir()
    client = _client_config(tmp_path)

    def locked(*_args: Any, **_kwargs: Any) -> Any:
        raise PermissionError(13, "Permission denied", str(client))

    monkeypatch.setattr(H, "register_local_scope", locked)
    report = H.run_headless(
        load_config(_write(tmp_path, _config(tmp_path, project_root=str(root)))),
        services=_Spy(),
        claude_config_path=client,
    )

    # Whitespace flattened, because the report wraps its detail lines at 96 columns and a phrase can
    # straddle a line break, which would let the negative assertion pass on the pre-fix text.
    text = " ".join(report.render().split())
    assert "has no `project_root`" not in text, text
    assert "NOT registered" in text and "Permission denied" in text, text


# ----------------------------------------------------------------------------------------------
# BUG-011: an unusable data_root is refused by name
# ----------------------------------------------------------------------------------------------


def test_a_data_root_the_stack_cannot_be_written_into_is_refused_by_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """BUG-011. Invariant: an OSError writing the stack into data_root is a ConfigRefusal naming
    `data_root`. Failure caught: a read-only or unelevated data_root (or, on Windows, a path that
    is a file) escaped `recall wizard` as a raw PermissionError traceback.
    Target: `headless._provision`."""
    from recall.wizard import headless as H

    def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise PermissionError(13, "Permission denied", str(tmp_path / "store"))

    monkeypatch.setattr(H, "choose_port", lambda: 55432)
    monkeypatch.setattr(H, "write_compose", refuse)
    raised: Exception | None = None
    try:
        H.run_headless(_data_root_config(tmp_path), services=_Spy())
    except Exception as exc:  # noqa: BLE001 - the pre-fix escape is what this test observes
        raised = exc
    assert isinstance(raised, ConfigRefusal), repr(raised)
    assert raised.keys == ("data_root",)


def test_no_free_database_port_is_refused_by_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """BUG-011 (choose_port). Invariant: a RuntimeError from choosing the port is a ConfigRefusal
    naming `data_root`. Failure caught: it escaped as a traceback. Target: `headless._provision`."""
    from recall.wizard import headless as H

    def no_port() -> int:
        raise RuntimeError("no free port in range")

    monkeypatch.setattr(H, "choose_port", no_port)
    raised: Exception | None = None
    try:
        H.run_headless(_data_root_config(tmp_path), services=_Spy())
    except Exception as exc:  # noqa: BLE001 - the pre-fix escape is what this test observes
        raised = exc
    assert isinstance(raised, ConfigRefusal), repr(raised)
    assert raised.keys == ("data_root",)


# ----------------------------------------------------------------------------------------------
# BUG-012, DAT-008: a resumed run asks the database, and survives the asking failing
# ----------------------------------------------------------------------------------------------


def test_a_failed_generation_lookup_on_resume_is_a_reported_failure(tmp_path: Path) -> None:
    """BUG-012. Invariant: when the resume check `active_generation` raises for one tenant, that
    tenant is a Failure and the others are still reported. Failure caught: the lookup ran outside
    the per-corpus try, so a dropped connection escaped `run_headless` after an earlier corpus of
    the run was handled, losing the report. Target: the calibrated loop of `run_headless`."""
    from recall.wizard.headless import run_headless

    config = load_config(_write(tmp_path, _config(tmp_path)))
    state = tmp_path / "state.json"
    run_headless(config, services=_Spy(), state_path=state)

    class _LookupFails(_Spy):
        def active_generation(self, tenant: str) -> str | None:
            if tenant == "default-code":
                raise RuntimeError("psycopg: server closed the connection unexpectedly")
            return super().active_generation(tenant)

    report: Any = None
    raised: Exception | None = None
    try:
        report = run_headless(config, services=_LookupFails(), state_path=state)
    except Exception as exc:  # noqa: BLE001 - the pre-fix escape is what this test observes
        raised = exc
    assert raised is None, f"the lookup error escaped run_headless: {raised!r}"
    assert [f.tenant for f in report.failures] == ["default-code"], report.failures
    assert "default-docs" in report.reused


def test_a_recorded_memory_index_whose_rows_are_gone_is_indexed_again(tmp_path: Path) -> None:
    """DAT-008. Invariant: a recorded legacy index with chunks is reused only while the database
    still holds rows for its tenant. Failure caught: after a removed volume or a recreated database
    (same digest), memory was reported indexed with its old counts while empty, and never
    re-indexed. Target: the legacy loop of `run_headless` (through `has_legacy_rows`)."""
    from recall.wizard.headless import run_headless

    config = load_config(_write(tmp_path, _config(tmp_path)))
    state = tmp_path / "state.json"
    first = _Spy()
    run_headless(config, services=first, state_path=state)
    assert first.legacy == ["default-memory"]

    class _VolumeWasRemoved(_Spy):
        def has_legacy_rows(self, tenant: str) -> bool:
            return False

    second = _VolumeWasRemoved()
    report = run_headless(config, services=second, state_path=state)
    assert second.legacy == ["default-memory"], "an index the database lost must be rebuilt"
    assert "default-memory" not in report.reused

    class _RowsStillThere(_Spy):
        def has_legacy_rows(self, tenant: str) -> bool:
            return True

    third = _RowsStillThere()
    run_headless(config, services=third, state_path=state)
    assert third.legacy == [], "a tenant that still holds rows is reused, not re-indexed"


# ----------------------------------------------------------------------------------------------
# BUG-013, SEC-004 / CODE-035: the state file
# ----------------------------------------------------------------------------------------------


def test_a_state_file_that_is_not_utf8_is_ignored(tmp_path: Path) -> None:
    """BUG-013. Invariant: an undecodable state file is treated as no state. Failure caught:
    UnicodeDecodeError (a ValueError, not a JSONDecodeError) escaped `load_state` and aborted the
    install after the database was started. Target: `recall.wizard.state.load_state`."""
    from recall.wizard.state import WizardState, load_state

    path = tmp_path / "wizard.state.json"
    path.write_bytes(b"\xff\xfe{\x00")
    result: Any = None
    raised: Exception | None = None
    try:
        result = load_state(path, digest="d")
    except Exception as exc:  # noqa: BLE001 - the pre-fix escape is what this test observes
        raised = exc
    assert raised is None, f"a garbled state file stopped the install: {raised!r}"
    assert result == WizardState(digest="d")


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits")
def test_a_new_state_file_is_private(tmp_path: Path) -> None:
    """SEC-004 and CODE-035. Invariant: `save_state` writes through the shared atomic writer, so a
    new state file is created 0600 like the config beside it. Failure caught: it was written with
    `write_text` at the umask default (0644), exposing an unsalted digest over the DSN password to
    other local users. Target: `recall.wizard.state.save_state`."""
    from recall.wizard.state import WizardState, save_state

    path = tmp_path / "wizard.state.json"
    previous = os.umask(0o022)
    try:
        save_state(path, WizardState(digest="d"))
    finally:
        os.umask(previous)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600, oct(path.stat().st_mode)


# ----------------------------------------------------------------------------------------------
# CODE-012: a failed smoke result scrubs every DSN the run holds
# ----------------------------------------------------------------------------------------------


def test_a_failed_smoke_does_not_print_the_migration_password(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CODE-012. Invariant: every failed SmokeResult is scrubbed of the serving AND the migration
    DSN's password. Failure caught: `_RealServices.smoke` scrubbed only the serving DSN while
    `run_headless` scrubbed both, so a psycopg error quoting the owner's password reached the
    report. Target: `headless._RealServices.smoke`."""
    import psycopg

    from recall.wizard.headless import _RealServices
    from recall.wizard.wiring import ServerBlock

    config = load_config(
        _write(
            tmp_path,
            _config(
                tmp_path,
                dsn="postgresql://reader:readsecret@127.0.0.1:1/recall",
                migration_dsn="postgresql://owner:ownersecret@127.0.0.1:1/recall",
            ),
        )
    )

    def connect(*_args: Any, **_kwargs: Any) -> Any:
        raise psycopg.OperationalError("authentication failed, password ownersecret rejected")

    monkeypatch.setattr(psycopg, "connect", connect)
    result = _RealServices(config).smoke(
        ServerBlock(name="default-docs", tenant="default-docs", env={}, rationale="r")
    )
    assert result.error is not None
    assert "ownersecret" not in result.error, result.error


# ----------------------------------------------------------------------------------------------
# DAT-011: the project is normalised once
# ----------------------------------------------------------------------------------------------


def test_a_padded_project_is_stripped_and_a_blank_one_is_refused(tmp_path: Path) -> None:
    """DAT-011. Invariant: `load_config` strips `project` once, and refuses one that is blank after
    stripping. Failure caught: the tenants were built from the stripped value while the desktop
    profile and every chunk took the padded one, which names no tenant.
    Target: `headless.load_config`."""
    config = load_config(_write(tmp_path, _config(tmp_path, project="myapp ")))
    assert config.project == "myapp"

    raised: Exception | None = None
    try:
        load_config(_write(tmp_path, _config(tmp_path, project="   ")))
    except Exception as exc:  # noqa: BLE001 - the pre-fix acceptance is what this test observes
        raised = exc
    assert isinstance(raised, ConfigRefusal), repr(raised)
    assert raised.keys == ("project",)


# ----------------------------------------------------------------------------------------------
# DEPLOY-006, DEPLOY-010: a missing extra is refused before anything is written or built
# ----------------------------------------------------------------------------------------------


def test_a_missing_embedder_extra_is_refused_before_the_stack_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DEPLOY-006. Invariant: the embedder is resolved before `_provision`, so a host without its
    extra is refused naming `embedder` with no compose file written and no container started.
    Failure caught: the wizard wrote the stack, pulled the image and started the database first.
    Target: `headless.run_headless`."""
    from recall.wizard import headless as H

    started: list[object] = []
    monkeypatch.setattr(H, "choose_port", lambda: 55432)
    monkeypatch.setattr(H, "bring_up", lambda *args, **kwargs: started.append(args))
    monkeypatch.setattr(H, "wait_for_database", lambda *args, **kwargs: None)

    class _NoExtra(_Spy):
        def dim(self) -> int:
            raise ImportError("fastembed is not installed")

    config = _data_root_config(tmp_path)
    raised: Exception | None = None
    try:
        H.run_headless(config, services=_NoExtra())
    except Exception as exc:  # noqa: BLE001 - asserted on below
        raised = exc
    assert isinstance(raised, ConfigRefusal), repr(raised)
    assert raised.keys == ("embedder",)
    assert config.data_root is not None
    assert not (config.data_root / H.COMPOSE_NAME).exists(), "nothing may be written first"
    assert not started, "no container may start before the embedder resolves"


def test_a_missing_mcp_extra_is_refused_before_anything_is_built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DEPLOY-010 (and BUG-014, whose report path this makes unreachable). Invariant: with
    `project_root` set and `mcp` not importable, `run_headless` refuses naming `project_root` and
    the extra before any corpus is built. Failure caught: the whole build ran, the servers were
    registered, and only the report said none of them could start.
    Target: `headless.run_headless`."""
    from recall.wizard.headless import run_headless

    _mcp_installed(monkeypatch, False)
    root = tmp_path / "project"
    root.mkdir()
    spy = _Spy()
    raised: Exception | None = None
    try:
        run_headless(
            load_config(_write(tmp_path, _config(tmp_path, project_root=str(root)))),
            services=spy,
            claude_config_path=_client_config(tmp_path),
        )
    except Exception as exc:  # noqa: BLE001 - asserted on below
        raised = exc
    assert isinstance(raised, ConfigRefusal), repr(raised)
    assert raised.keys == ("project_root",)
    assert "recall-rag[mcp]" in str(raised)
    assert spy.corpora == [] and spy.legacy == [], "nothing may be built before the refusal"


# ----------------------------------------------------------------------------------------------
# ENV-010: the report names a CLI command that reaches each tenant
# ----------------------------------------------------------------------------------------------


def test_the_report_prints_a_cli_command_for_each_server() -> None:
    """ENV-010. Invariant: for every server the report prints a ready-to-run `recall --tenant <t>
    search` line carrying the route settings that differ from the CLI's defaults. Failure caught:
    the project `.env` holds only the DSN and embedder, so `recall search` there read tenant
    `default` on the legacy route and found nothing, and nothing named the working form.
    Target: `headless.HeadlessReport.render`."""
    from recall.wizard.headless import HeadlessReport
    from recall.wizard.wiring import ServerBlock

    text = HeadlessReport(
        servers=(
            ServerBlock(
                name="default-docs",
                tenant="default-docs",
                env={
                    "RECALL_ENV": "production",
                    "RECALL_INDEX_MODE": "generation",
                    "RECALL_TRUST_MODE": "strict",
                },
                rationale="certified and promoted",
            ),
            ServerBlock(
                name="default-memory",
                tenant="default-memory",
                env={
                    "RECALL_ENV": "development",
                    "RECALL_INDEX_MODE": "legacy",
                    "RECALL_TRUST_MODE": "development",
                },
                rationale="uncalibrated",
            ),
        )
    ).render()
    assert (
        "RECALL_ENV=production RECALL_INDEX_MODE=generation recall --tenant default-docs search"
        in text
    ), text
    assert "RECALL_TRUST_MODE=development recall --tenant default-memory search" in text, text


# ----------------------------------------------------------------------------------------------
# PERF-008, SEC-007, SEC-005: the wizard's own connections
# ----------------------------------------------------------------------------------------------


def test_every_wizard_connection_has_a_connect_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PERF-008. Invariant: `grant`, `configure_fact_boundary` and `smoke` connect with
    connect_timeout=10, like their siblings. Failure caught: an unreachable host left them waiting
    for the OS TCP limit with no output. Target: `headless._RealServices`."""
    import psycopg

    from recall.wizard.headless import _RealServices
    from recall.wizard.wiring import ServerBlock

    seen: list[object] = []

    def connect(*_args: Any, **kwargs: Any) -> Any:
        seen.append(kwargs.get("connect_timeout"))
        raise psycopg.OperationalError("unreachable")

    monkeypatch.setattr(psycopg, "connect", connect)
    services = _RealServices(load_config(_write(tmp_path, _config(tmp_path))))
    owner = "postgresql://owner:ownerpw@127.0.0.1:1/recall"
    with pytest.raises(psycopg.OperationalError):
        services.grant(owner, role="reader")
    with pytest.raises(psycopg.OperationalError):
        services.configure_fact_boundary(
            owner,
            serving_role="reader",
            controller_dsn="postgresql://fact_writer:factpw-long@127.0.0.1:1/recall",
        )
    services.smoke(ServerBlock(name="default-docs", tenant="default-docs", env={}, rationale="r"))
    assert seen == [10, 10, 10], seen


def test_the_controller_role_is_created_with_a_scram_verifier_not_the_password(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SEC-007. Invariant: CREATE ROLE carries a client-side SCRAM-SHA-256 verifier, never the
    plaintext password. Failure caught: the password was a literal in the statement, which a server
    logging DDL (or pg_stat_activity) records. Target: `headless._RealServices.configure_fact_boundary`."""
    import psycopg

    from recall.wizard.headless import _RealServices

    executed: list[str] = []
    hashed: list[tuple[bytes, bytes, bytes]] = []
    monkeypatch.setattr(psycopg, "connect", lambda *a, **k: _FakeConnection(executed, hashed))
    _RealServices(load_config(_write(tmp_path, _config(tmp_path)))).configure_fact_boundary(
        "postgresql://owner:ownerpw@127.0.0.1:1/recall",
        serving_role="reader",
        controller_dsn="postgresql://fact_writer:factpw-long@127.0.0.1:1/recall",
    )

    created = [s for s in executed if "CREATE ROLE" in s]
    assert created, executed
    assert not any("factpw-long" in s for s in executed), created
    assert "SCRAM-SHA-256$4096" in created[0], created
    assert hashed == [(b"factpw-long", b"fact_writer", b"scram-sha-256")]


def test_a_controller_dsn_with_the_published_credentials_is_refused_before_connecting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SEC-005 (the configure_fact_boundary half). Invariant: a fact_write_dsn carrying the
    published recall:recall credentials against a remote host is refused before any connection, on
    every path that reaches `configure_fact_boundary`, the desktop installer's included. Failure
    caught: a LOGIN role was created with the published password and given the controller grants.
    Target: `headless._RealServices.configure_fact_boundary`."""
    import psycopg

    from recall.wizard.headless import _RealServices

    monkeypatch.delenv("RECALL_ALLOW_INSECURE_DSN", raising=False)
    connects: list[str] = []
    executed: list[str] = []

    def connect(dsn: str, **_kwargs: Any) -> Any:
        connects.append(dsn)
        return _FakeConnection(executed, [])

    monkeypatch.setattr(psycopg, "connect", connect)
    raised: Exception | None = None
    try:
        _RealServices(load_config(_write(tmp_path, _config(tmp_path)))).configure_fact_boundary(
            "postgresql://owner:ownerpw@db.example.com:5432/recall",
            serving_role="reader",
            controller_dsn="postgresql://recall:recall@db.example.com:5432/recall",
        )
    except Exception as exc:  # noqa: BLE001 - asserted on below
        raised = exc
    assert isinstance(raised, PermissionError), repr(raised)
    assert connects == [], "nothing may connect before the credential guard"
