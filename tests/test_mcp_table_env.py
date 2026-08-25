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

import pytest

from recall.store import DEFAULT_TABLE


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
        "",
        "public.chunks",
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
