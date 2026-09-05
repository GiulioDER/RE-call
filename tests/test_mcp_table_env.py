"""`RECALL_TABLE` must reach the MCP server, because the plugin asks the user for it.

⛔ **The failure this exists for was silent, which is what makes it the worst one in this area.**

`recall quickstart` indexes its 22 sample documents into `quickstart_chunks`, deliberately: they are
fiction about a fictional service, and in `chunks` they would be retrieved beside a reader's real
memory from the same database. `plugin/README.md` then tells a first-time reader to point the Claude
Code plugin at exactly that corpus. The plugin passed a DSN, a tenant and a trust mode, and this
server had no table knob at all, so `PgVectorStore` opened its default `chunks` — which the
quickstart creates (global migrations are recorded against it and no other table may be migrated
first) and leaves EMPTY.

Measured 2026-08-25 against a live quickstart database, driving the stdio server with exactly the
three variables the plugin shipped: `recall_search` returned ``0 relevant memory hit(s)``. No
exception, no warning, nothing naming the table. The reader's conclusion is "this product finds
nothing", not "I am pointed at the wrong table", and there is no line anywhere for them to read that
would distinguish the two.

Compare the sibling defect in `test_mcp_trust_mode_env.py`: a strict server against an uncalibrated
corpus at least answers `INDEX_NOT_READY`, which names its own cause. An empty answer names nothing.

These tests pin three things: the variable is read, a value that could not be a table is refused at
import rather than interpolated into SQL, and the two stores that have no table to choose refuse it
loudly instead of ignoring it.
"""

from __future__ import annotations

import importlib
import os

import pytest

from recall.store import DEFAULT_TABLE
from tests.conftest import TEST_DSN, requires_db


@pytest.fixture
def reload_server(monkeypatch: pytest.MonkeyPatch):
    """Reimport the server module under a chosen environment, then put it back.

    The module resolves its configuration at import time, so the environment has to be set before
    the import rather than patched afterwards. The teardown is the load-bearing half, for the
    reason spelled out in `test_mcp_trust_mode_env.py`: `monkeypatch` restores the environment but
    cannot restore a module constant a reload has already baked in, so without it the last
    parametrised case would leave `server.TABLE` pointing at a throwaway table for every later test
    in the process.
    """
    import recall_mcp.server as server

    def _load(**env: str):
        monkeypatch.delenv("RECALL_TABLE", raising=False)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        return importlib.reload(server)

    yield _load

    monkeypatch.delenv("RECALL_TABLE", raising=False)
    importlib.reload(server)


def test_unset_leaves_the_default_table(reload_server) -> None:
    """The variable is opt-in. Nobody who has not heard of it changes behaviour by upgrading."""
    server = reload_server()
    assert server.TABLE == DEFAULT_TABLE


def test_the_quickstart_table_reaches_the_module(reload_server) -> None:
    """The whole point: the value the plugin collects is the value the store opens."""
    from recall.quickstart import QUICKSTART_TABLE

    server = reload_server(RECALL_TABLE=QUICKSTART_TABLE)
    assert server.TABLE == QUICKSTART_TABLE
    assert server.TABLE != DEFAULT_TABLE, "the test is vacuous if the quickstart uses the default"


@pytest.mark.parametrize(
    "value",
    [
        "drop table; --",
        "chunks; SELECT 1",
        "chunks chunks",
        "public.chunks",
        "chunks-v2",
    ],
)
def test_a_value_that_is_not_an_identifier_is_refused_at_import(reload_server, value: str) -> None:
    """Refused before a connection exists, not sanitised on the way into a query.

    `dim` and `table` are interpolated into SQL directly in `PgVectorStore`, which is why that
    class validates `table.isidentifier()` too. This is the same check moved to the earliest point
    it can be made: a server that refuses to start is a server that cannot be induced to run this
    string against anything.
    """
    with pytest.raises(ValueError, match="RECALL_TABLE"):
        reload_server(RECALL_TABLE=value)


@pytest.mark.parametrize("value", ["", "   ", "\t"])
def test_an_empty_or_padded_value_means_UNSET_rather_than_an_import_crash(
    reload_server, value: str
) -> None:
    """⚠️ **Contract change, from audit finding F24.** This case used to assert a raise.

    `os.environ.get("RECALL_TABLE", DEFAULT_TABLE)` returns `""` for a variable that is PRESENT and
    empty, not the default — so `RECALL_TABLE=` in a `.env` or a compose file, which is the normal
    way to clear a value, raised `ValueError` at module scope. An MCP client renders that as a
    server with no tools: the exact silent symptom this variable was added to eliminate.

    Padding is the same defect wearing a different hat, and this project has paid for it before:
    `recall/_env.py` records that a trailing space from a systemd EnvironmentFile "read as
    production at some gates and development at others". Measured here before the fix,
    `"chunks ".isidentifier()` is False, so a trailing space typed into the plugin's free-text
    Table field killed the server at import.

    The parametrised list above is the control: every genuinely invalid value still raises, so this
    is a normalisation, not a weakening of the gate.
    """
    server = reload_server(RECALL_TABLE=value)
    assert server.TABLE == DEFAULT_TABLE


def test_the_default_is_accepted_everywhere(reload_server) -> None:
    """Not setting it must never be the thing that stops a production server booting."""
    import recall_mcp.server as server

    for generation_mode in (True, False):
        for authenticated in (True, False):
            assert (
                server.table_override_refusal(
                    DEFAULT_TABLE,
                    generation_mode=generation_mode,
                    authenticated=authenticated,
                )
                is None
            )


def test_the_legacy_single_tenant_path_honours_an_override() -> None:
    """The one configuration that has a table to choose is the one the plugin uses."""
    import recall_mcp.server as server

    assert (
        server.table_override_refusal(
            "quickstart_chunks", generation_mode=False, authenticated=False
        )
        is None
    )


@pytest.mark.parametrize(
    ("generation_mode", "authenticated", "expected"),
    [
        (True, False, "generation mode"),
        (False, True, "authenticated tenant routing"),
        (True, True, "generation mode"),
    ],
)
def test_a_store_with_no_table_to_choose_refuses_rather_than_ignores(
    generation_mode: bool, authenticated: bool, expected: str
) -> None:
    """⚠️ Ignoring it would rebuild the original bug one layer up.

    `GenerationStore` is welded to `recall_chunks_v1` and the authenticated registry is
    generation-aware, so neither can serve a named table. Dropping the variable there would leave
    an operator who set it reading a corpus other than the one they asked for, with nothing said
    about it — which is precisely the silent wrong answer this variable was added to end. The
    message names which of the two configurations refused, because they are set by different
    variables and an operator who does not know which one is on cannot act on "not supported".
    """
    import recall_mcp.server as server

    refusal = server.table_override_refusal(
        "quickstart_chunks", generation_mode=generation_mode, authenticated=authenticated
    )
    assert refusal is not None
    assert expected in refusal
    assert "quickstart_chunks" in refusal
    assert "RECALL_ENV" in refusal, "a refusal must name the way out, not only the problem"


# ------------------------------------------------------------------------------------------------
# Does the value actually reach the STORE? Everything above this line proves it reaches the MODULE.
# ------------------------------------------------------------------------------------------------


@requires_db
def test_the_configured_table_reaches_the_store_the_server_actually_opens() -> None:
    """⛔ **The headline fix of this branch, and nothing bound it until now.**

    Everything else in this file tests the module constant and the pure `table_override_refusal`.
    The file's own title says "`RECALL_TABLE` must reach the MCP server", and it reached the module.
    An architect gate deleted `table=TABLE` from the `PgVectorStore` construction in
    `_make_lifespan` and ran everything that could plausibly see it: **363 tests passed** with the
    defect fully restored.

    That is worse than an untested fix. `tests/test_claude_code_plugin.py` asserted the variable
    "REACHES something" by grepping the server's SOURCE, which cannot show reach and stayed green
    under that same mutation — a false comment in the file written to stop false comments.

    Six published surfaces now promise this behaviour: `site/troubleshooting.html`
    ("add RECALL_TABLE ... and restart Claude. Re-indexing is not needed"), `site/claude-code.html`,
    `docs/USING_WITH_CLAUDE.md`, `docs/ENVIRONMENT.md`, `.env.example`, and the plugin manifest,
    which now asks a user for the value.

    ⚠️ **Deliberately NOT a monkeypatched `PgVectorStore` capturing kwargs.** A kwargs-sink fake is
    exactly what let `server_env` ship with a parameter no caller passed: it would stay green even
    if the real constructor stopped accepting `table`. This drives the real lifespan against a real
    database and asks the resulting store what table it opened.
    """
    import asyncio
    import importlib

    table = "mcp_wiring_chunks"
    _rebuild_for_wiring(table, 64)

    previous = {k: os.environ.get(k) for k in ("RECALL_TABLE", "RECALL_SERVING_DSN", "RECALL_EMBEDDER")}
    os.environ["RECALL_TABLE"] = table
    os.environ["RECALL_SERVING_DSN"] = TEST_DSN
    os.environ["RECALL_EMBEDDER"] = "hashing"
    try:
        import recall_mcp.server as server

        server = importlib.reload(server)
        assert server.TABLE == table, "the module did not even read it"

        opened: list[str] = []

        async def _drive() -> None:
            lifespan = server._make_lifespan(None)
            async with lifespan(None) as state:  # type: ignore[arg-type]
                opened.append(state["store"].table)

        asyncio.run(_drive())
        assert opened == [table], (
            f"the server opened {opened!r} while RECALL_TABLE named {table!r}; the value reaches "
            "the module and stops there, which is the silent-zero-hits bug this variable exists "
            "to end"
        )
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        import recall_mcp.server as server

        importlib.reload(server)


def _rebuild_for_wiring(table: str, dim: int) -> None:
    """Drop `table`, forget it in the migration ledger, then migrate it fresh.

    Dropping alone is not enough: `apply_migrations` records what it applied keyed by target table,
    so a dropped table leaves the ledger claiming those migrations are still applied and the next
    run finds `relation ... does not exist`. Same pattern as `tests/test_doctor_db.py::_rebuild`.
    """
    import psycopg
    from psycopg import sql

    from recall.schema import LEDGER_TABLE, apply_migrations

    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute(sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(sql.Identifier(table)))
        conn.execute(
            sql.SQL("DELETE FROM {} WHERE target_table = %s").format(sql.Identifier(LEDGER_TABLE)),
            (table,),
        )
    apply_migrations(TEST_DSN, table=table, dim=dim)
