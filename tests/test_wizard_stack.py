"""The generated stack: one database, at the user's location, reachable from both worlds.

Every property here is one that, if wrong, produces two disconnected corpora for one user while
both surfaces report themselves healthy. That silence is the reason these are asserted rather than
eyeballed in a compose file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from recall.wizard.stack import (
    DB_VOLUME,
    DOCKERFILE_NAME,
    DB_INTERNAL_PORT,
    DB_MOUNT,
    StackSpec,
    choose_port,
    compose_document,
    container_dsn,
    host_dsn,
    write_compose,
)


def _spec(tmp_path: Path, **overrides: object) -> StackSpec:
    payload: dict[str, object] = {
        "data_root": tmp_path / "chosen by the user",
        "port": 5487,
        "tenants": ("myapp-docs", "myapp-code", "myapp-memory"),
        "env": {
            "myapp-docs": {
                "RECALL_DSN": host_dsn(5487),
                "RECALL_TENANT": "myapp-docs",
                "RECALL_ENV": "production",
                "RECALL_TRUST_MODE": "strict",
            },
            "myapp-code": {
                "RECALL_DSN": host_dsn(5487),
                "RECALL_TENANT": "myapp-code",
                "RECALL_ENV": "production",
                "RECALL_TRUST_MODE": "strict",
            },
            "myapp-memory": {
                "RECALL_DSN": host_dsn(5487),
                "RECALL_TENANT": "myapp-memory",
                "RECALL_TRUST_MODE": "development",
            },
        },
    }
    payload.update(overrides)
    return StackSpec(**payload)  # type: ignore[arg-type]


def test_the_database_publishes_a_host_port(tmp_path: Path) -> None:
    """Without this the wizard and the agent cannot reach the store the UI is filling.

    `docker-compose.desktop.yml` published NO ports, so its database was reachable only inside the
    compose network. That is the whole defect this module removes, and it failed silently: each
    surface was healthy and telling the truth about a different database.
    """
    document = compose_document(_spec(tmp_path))
    db = document["services"]["db"]  # type: ignore[index]

    # 🔁 Was `f"5487:{DB_INTERNAL_PORT}"`, and that omission was an exposure rather than a style
    # choice: Compose's short form with no host IP binds 0.0.0.0, so Docker Desktop published this
    # database — with the `recall:recall` credentials this project's README publishes — on every
    # interface of the Windows host. The comment justifying `RECALL_ALLOW_INSECURE_DSN` asserted
    # "the port that IS published is bound to loopback"; this is what makes that true.
    assert db["ports"] == [f"127.0.0.1:5487:{DB_INTERNAL_PORT}"]
    assert db["ports"][0].startswith("127.0.0.1:"), "the database must never bind every interface"


def test_the_database_lives_on_a_named_volume_not_the_users_directory(tmp_path: Path) -> None:
    """🔁 **This test asserted the opposite until 2026-08-19, and the measurement overturned it.**

    It read: "A bind mount, not a Docker-managed volume, so the user can find and back up their
    index." That is what the wizard promises everywhere else, and it is the requirement this
    module was built around. It does not survive contact with Docker Desktop on Windows, whose
    filesystem passthrough returns EINTR on writes that PostgreSQL treats as fatal:

        FATAL:  could not write to file "pg_wal/xlogtemp.1218": Interrupted system call
        LOG:  shutting down due to startup process failure

    The failure is INTERMITTENT. An earlier full install on the bind mount built, calibrated and
    certified both corpora; the next run died mid-flight. Intermittent WAL write failure is a
    corruption risk, not merely an availability one, so the index cannot live there — a promise
    about where files sit is not worth an index that is occasionally wrong.

    The user's directory keeps everything they need to see; only the database moved.
    """
    root = tmp_path / "chosen by the user"
    document = compose_document(_spec(tmp_path, data_root=root))
    db = document["services"]["db"]  # type: ignore[index]

    assert db["volumes"] == [f"{DB_VOLUME}:{DB_MOUNT}"]
    assert document["volumes"] == {DB_VOLUME: None}, (
        "a named volume must be declared at the top level or compose treats it as a bind path"
    )
    assert str(root) not in json.dumps(db), (
        "the database must not reference the user's directory at all; a stray path here is how a "
        "bind mount would creep back in"
    )


def test_the_first_install_is_given_time_to_initdb(tmp_path: Path) -> None:
    """Without a start_period the FIRST `up --wait` of every new install fails.

    `initdb` runs before the first successful health check can pass, and it can outlast
    interval x retries. Measured on a fresh install: `dependency failed to start: container
    recall-...-db-1 is unhealthy`, with all three MCP services left at `created`, and a second
    `up` then working — which a user reads as an installer that broke and then fixed itself.
    """
    db = compose_document(_spec(tmp_path))["services"]["db"]  # type: ignore[index]
    health = db["healthcheck"]

    assert "start_period" in health, "initdb needs a grace period before failures count"
    assert health["start_period"].endswith("s")
    assert int(health["start_period"].rstrip("s")) >= 60


def test_the_mount_point_is_one_level_above_data(tmp_path: Path) -> None:
    """pg18 REFUSES `/var/lib/postgresql/data` and exits 1, measured.

    It wants a single mount one level up so `pg_upgrade --link` works across a version bump without
    crossing a mount boundary. Asserted because the wrong path looks equally plausible and fails
    only at runtime, on the user's machine, on their first install.
    """
    assert DB_MOUNT == "/var/lib/postgresql"
    assert not DB_MOUNT.endswith("/data")


def test_each_service_talks_to_the_database_over_the_compose_network(tmp_path: Path) -> None:
    """Same store, shorter path. The host DSN would not resolve inside the network at all."""
    document = compose_document(_spec(tmp_path))
    services = document["services"]  # type: ignore[index]

    for tenant in ("myapp-docs", "myapp-code", "myapp-memory"):
        env = services[f"recall-{tenant}"]["environment"]
        assert env["RECALL_DSN"] == container_dsn(), f"{tenant} must use the in-network address"
        assert "127.0.0.1" not in env["RECALL_DSN"], "a host address does not resolve in-network"
        assert env["RECALL_TENANT"] == tenant


def test_the_trust_posture_comes_from_the_caller_not_from_here(tmp_path: Path) -> None:
    """One decision per tenant, applied to the agent's server and the UI's service alike.

    Recomputing it here would let the two drift, so that the same corpus is strict for the agent and
    relaxed for the UI, which is the kind of difference nobody notices until it matters.
    """
    document = compose_document(_spec(tmp_path))
    services = document["services"]  # type: ignore[index]

    assert services["recall-myapp-docs"]["environment"]["RECALL_TRUST_MODE"] == "strict"
    assert services["recall-myapp-docs"]["environment"]["RECALL_ENV"] == "production"
    # The uncalibrated one keeps development trust and no production routing.
    memory = services["recall-myapp-memory"]["environment"]
    assert memory["RECALL_TRUST_MODE"] == "development"
    assert "RECALL_ENV" not in memory


def test_one_service_per_tenant(tmp_path: Path) -> None:
    """Not a style choice. On unauthenticated stdio `_require` returns the server's own single
    store and IGNORES the requested tenant, so one server genuinely cannot serve two tenants."""
    document = compose_document(_spec(tmp_path))
    services = document["services"]  # type: ignore[index]

    assert set(services) == {"db", "recall-myapp-docs", "recall-myapp-code", "recall-myapp-memory"}


def test_a_relative_location_is_refused(tmp_path: Path) -> None:
    """It would resolve against whatever directory the installer ran from."""
    with pytest.raises(ValueError, match="data_root must be absolute"):
        _spec(tmp_path, data_root=Path("somewhere/relative"))


def test_a_tenant_without_an_environment_is_refused(tmp_path: Path) -> None:
    """Inventing one here would put the agent and the UI on different rules for one corpus."""
    with pytest.raises(ValueError, match="no environment for"):
        _spec(tmp_path, tenants=("myapp-docs", "myapp-unknown"))


def test_the_compose_file_is_json_and_survives_a_path_with_spaces(tmp_path: Path) -> None:
    """JSON is valid YAML and `docker compose` parses it, verified against the real binary.

    It is used here so `json.dumps` handles quoting: a Windows path containing a space is the
    normal case (`C:\\Users\\Given Name\\...`) and hand-built YAML gets it wrong.
    """
    path = tmp_path / "nested" / "docker-compose.recall.yml"
    root = tmp_path / "My Documents" / "RE-call data"
    write_compose(path, compose_document(_spec(tmp_path, data_root=root)))

    assert not list(path.parent.glob("*.tmp")), "no temporary may survive the write"
    assert b"\r\n" not in path.read_bytes(), "CRLF would rewrite every line on every platform"

    reloaded = json.loads(path.read_text(encoding="utf-8"))
    assert reloaded["services"]["db"]["volumes"] == [f"{DB_VOLUME}:{DB_MOUNT}"]
    assert (path.parent / DOCKERFILE_NAME).exists(), (
        "the build stanza names a Dockerfile beside the compose file; writing one without the "
        "other produces a stack that cannot come up"
    )

    # 🔁 This used to assert the user's path appeared INSIDE the db volume line, which was the
    # point of the JSON quoting. The database moved to a named volume, so the document no longer
    # embeds that path anywhere — but the file is still WRITTEN to a directory with spaces in it,
    # which is the part that has to keep working, and JSON is still what keeps the write
    # deterministic and LF-only.
    assert "My Documents" in str(root) and str(root) not in path.read_text(encoding="utf-8")


def test_the_chosen_port_is_free_and_not_the_usual_postgres_one() -> None:
    """5432 collides with a user's own Postgres and with this repo's root compose.

    A collision does not fail at install time: compose reports a bind error, the stack half starts,
    and the first symptom is a query that cannot connect.
    """
    import socket

    port = choose_port()
    assert port != 5432
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        # No SO_REUSEADDR, for the same reason the implementation does not set it: with the flag,
        # this bind succeeds against a listening port and the assertion proves nothing.
        probe.bind(("127.0.0.1", port))  # free right now, which is what was claimed


def test_the_port_is_read_back_rather_than_rechosen(tmp_path: Path) -> None:
    """A re-install must not repoint the database out from under the UI and the agent.

    `runtime.json` names a compose file and the desktop UI connects through whatever that file
    publishes; `.mcp.json` carries the host address directly. Re-choosing a free port on every run
    would silently break both, and the symptom is a UI showing an empty corpus rather than an error.
    """
    from recall.wizard.stack import existing_port

    path = tmp_path / "docker-compose.recall.yml"
    write_compose(path, compose_document(_spec(tmp_path, port=5501)))

    assert existing_port(path) == 5501


def test_an_absent_or_unreadable_compose_yields_no_port(tmp_path: Path) -> None:
    """Not an error: the caller chooses a fresh port and writes a correct file over it."""
    from recall.wizard.stack import existing_port

    assert existing_port(tmp_path / "nothing-here.yml") is None

    broken = tmp_path / "broken.yml"
    broken.write_text("{ not json", encoding="utf-8")
    assert existing_port(broken) is None


def test_run_headless_provisions_from_data_root_and_reuses_the_port(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The install path: no DSN in the config, a location instead, and a database at the end.

    Docker is stubbed so this runs everywhere, but the SEQUENCE is the real one and the assertions
    are on what a user would care about: the compose lands under their location, the DSN handed to
    everything downstream is the published port, and a SECOND run reuses that port instead of
    repointing the store out from under the UI.
    """
    import recall.wizard.headless as H
    from recall.desktop.profiles import profile_path as _default_profile_path
    from recall.wizard.stack import existing_port
    from tests.test_wizard_state import _CountingSpy, _config

    # Snapshot BEFORE anything runs, so the guard below compares against reality rather than
    # assuming the machine is pristine.
    _real = _default_profile_path()
    _real_profile_before = (_real.exists(), _real.stat().st_mtime_ns if _real.exists() else 0)

    started: list[tuple[Path, tuple[str, ...]]] = []
    waited: list[str] = []
    monkeypatch.setattr(
        H, "bring_up", lambda p, *, project_name, services=(), timeout=300.0: started.append((p, services))
    )
    monkeypatch.setattr(H, "wait_for_database", lambda dsn, **kw: waited.append(dsn))

    location = tmp_path / "My RE-call data"
    config = _config(tmp_path, dsn=None, migration_dsn=None, data_root=str(location))

    class _Recording(_CountingSpy):
        """Records the DSN the schema was applied with, which is the provisioned one or nothing."""

        schema_dsns: list[str] = []

        def apply_schema(self, dsn: str, *, dim: int) -> None:
            self.schema_dsns.append(dsn)

    spy = _Recording()
    spy.schema_dsns = []
    profile = tmp_path / "runtime.json"
    report = H.run_headless(config, services=spy, profile_path=profile)

    compose = location / H.COMPOSE_NAME
    assert compose.exists(), "the stack must be written under the user's own location"
    assert started and started[0][1] == ("db",), "only the database starts before the build"
    assert waited, "the published port must be polled; --wait does not prove it is usable"
    assert waited[0] == spy.schema_dsns[0], "everything downstream uses the provisioned address"
    assert report.ok is True
    assert profile.exists(), "the desktop profile is written where the CALLER said"

    from recall.desktop.profiles import profile_path as default_profile_path

    # 🔁 This asserted the real profile does not EXIST. That is a claim about the machine, not
    # about the test: anyone who has run the installer once has one, and the test then failed for a
    # reason unrelated to the code — which is exactly how it failed here, on a profile written by
    # this project's own clean-install run. Asserting it is UNCHANGED is strictly stronger: it
    # still catches this test writing the file, and it no longer fails when something else did.
    # The same defect was found in `tests/test_desktop.py` by four auditors and fixed the same way.
    real = default_profile_path()
    assert (real.exists(), real.stat().st_mtime_ns if real.exists() else 0) == _real_profile_before, (
        "a test must never write the real user profile; this one did, pointing it at a "
        "pytest temp directory"
    )

    first_port = existing_port(compose)
    assert first_port is not None and f":{first_port}" in waited[0]

    # A re-install must NOT repoint the database: runtime.json names this compose file.
    H.run_headless(config, services=_CountingSpy(), profile_path=profile)
    assert existing_port(compose) == first_port, "a re-run must reuse the port it already published"


def test_choose_port_steps_past_a_busy_one() -> None:
    """The allow path for the search, so it cannot be satisfied by always returning the preferred."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as taken:
        taken.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        taken.bind(("127.0.0.1", 0))
        busy = taken.getsockname()[1]
        taken.listen(1)

        assert choose_port(busy) != busy, "a listening port must not be handed out"


def test_every_generated_service_can_pass_the_insecure_dsn_guard(tmp_path: Path) -> None:
    """Without this key the whole generated stack is inert, and nothing said so.

    `require_secure_dsn` refuses the built-in `recall:recall` credentials against any host it does
    not consider local, and the compose hostname `db` is not local by that test. So every service
    here exits 1 before connecting, for both commands the desktop runs inside them:
    `recall schema apply` and `python -m recall_mcp.server`. Demonstrated by running the CLI with
    exactly this DSN and no opt-out:

        PermissionError: refusing to start against postgresql://recall:***@db:5432/recall

    and after the fix the same command reaches `failed to resolve host 'db'`, which is the correct
    answer from OUTSIDE the compose network.

    The hand-written `docker-compose.desktop.yml` sets this on all four of its services, which is
    why the pre-wizard stack worked and the generated one did not. This test exists so the two
    cannot drift apart silently again.
    """
    document = compose_document(_spec(tmp_path))
    services = document["services"]
    assert isinstance(services, dict)

    for name, service in services.items():
        if name == "db":
            continue
        assert isinstance(service, dict)
        environment = service["environment"]
        assert isinstance(environment, dict)
        assert environment["RECALL_ALLOW_INSECURE_DSN"] == "1", (
            f"service {name} would refuse to start against {environment['RECALL_DSN']}"
        )
