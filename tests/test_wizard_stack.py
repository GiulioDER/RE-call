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

    assert db["ports"] == [f"5487:{DB_INTERNAL_PORT}"]


def test_the_data_lands_at_the_location_the_user_chose(tmp_path: Path) -> None:
    """A bind mount, not a Docker-managed volume, so the user can find and back up their index."""
    root = tmp_path / "chosen by the user"
    document = compose_document(_spec(tmp_path, data_root=root))
    db = document["services"]["db"]  # type: ignore[index]

    assert db["volumes"] == [f"{(root / 'database').as_posix()}:{DB_MOUNT}"]


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
    volume = reloaded["services"]["db"]["volumes"][0]
    assert "My Documents" in volume and "RE-call data" in volume


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


def test_choose_port_steps_past_a_busy_one() -> None:
    """The allow path for the search, so it cannot be satisfied by always returning the preferred."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as taken:
        taken.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        taken.bind(("127.0.0.1", 0))
        busy = taken.getsockname()[1]
        taken.listen(1)

        assert choose_port(busy) != busy, "a listening port must not be handed out"
