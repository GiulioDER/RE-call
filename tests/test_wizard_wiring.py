"""`.mcp.json`: the artifact that decides whether any of the install is reachable.

Each property below is a way this could produce a configuration that looks complete and answers
nothing, which is the worst outcome available here: the operator has no reason to doubt a file the
installer wrote.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from recall.wizard.corpora import default_plan
from recall.wizard.wiring import mcp_config, server_blocks, write_mcp_config


def _plan(tmp_path: Path):
    return default_plan(
        embedder="hashing",
        docs_root=tmp_path / "docs",
        code_root=tmp_path / "repo",
        memory_root=tmp_path / "memory",
    )


def _by_name(blocks):
    return {b.name: b for b in blocks}


# ----------------------------------------------------------------------------------------------
# Which tenants get a server, and why
# ----------------------------------------------------------------------------------------------


def test_a_promoted_tenant_is_served_from_its_generation_under_strict_trust(tmp_path: Path) -> None:
    """`RECALL_ENV=production` is what selects `GenerationStore` (`recall_mcp/server.py:627`).

    Without it the server reads the legacy `chunks` table, which a generation build never wrote to,
    so the tenant answers nothing while looking configured. That is the failure this asserts against,
    not a style preference.
    """
    blocks, unservable = server_blocks(
        _plan(tmp_path),
        dsn="postgresql://recall:pw@127.0.0.1:5432/recall",
        promoted=frozenset({"docs", "code"}),
        serving=frozenset({"docs", "code", "memory"}),
    )

    assert unservable == ()
    docs = _by_name(blocks)["docs"]
    assert docs.env["RECALL_ENV"] == "production"
    assert docs.env["RECALL_TRUST_MODE"] == "strict"
    assert docs.env["RECALL_TENANT"] == "docs"


def test_a_degraded_tenant_with_a_predecessor_is_served_with_relaxed_trust(tmp_path: Path) -> None:
    """An earlier generation still answers, so a server is correct; the calibration is stale."""
    blocks, unservable = server_blocks(
        _plan(tmp_path),
        dsn="postgresql://recall:pw@127.0.0.1:5432/recall",
        promoted=frozenset(),
        serving=frozenset({"docs", "code", "memory"}),
    )

    assert unservable == ()
    docs = _by_name(blocks)["docs"]
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
        serving=frozenset({"memory"}),
    )

    served = set(_by_name(blocks))
    assert served == {"memory"}, "no server may be written for a tenant that cannot answer"
    assert {u.tenant for u in unservable} == {"docs", "code"}
    assert all("NoActiveGeneration" in u.reason for u in unservable), (
        "the reason must name what would actually happen, not just say it failed"
    )


def test_the_uncalibrated_tenant_is_never_put_into_production_mode(tmp_path: Path) -> None:
    """Its rows are in the legacy `chunks` table, which production mode routes past entirely."""
    blocks, _ = server_blocks(
        _plan(tmp_path),
        dsn="postgresql://recall:pw@127.0.0.1:5432/recall",
        promoted=frozenset({"docs", "code"}),
        serving=frozenset({"docs", "code", "memory"}),
    )

    memory = _by_name(blocks)["memory"]
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
        promoted=frozenset({"docs", "code"}),
        serving=frozenset({"docs", "code", "memory"}),
    )

    for block in blocks:
        assert "RECALL_TRUST_MODE" in block.env, f"{block.name} would inherit nothing"
        assert "RECALL_DSN" in block.env
        assert "RECALL_TENANT" in block.env


# ----------------------------------------------------------------------------------------------
# Writing the file
# ----------------------------------------------------------------------------------------------


def test_writing_preserves_servers_this_wizard_knows_nothing_about(tmp_path: Path) -> None:
    """An operator's `.mcp.json` almost certainly holds other servers, and losing them is silent.

    Replacing the file would delete a working configuration and the operator would discover it the
    next time they reached for a tool that had vanished.
    """
    path = tmp_path / ".mcp.json"
    path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "someone-elses": {"type": "http", "url": "https://example.invalid/mcp"},
                    "docs": {"type": "stdio", "command": "old"},
                }
            }
        ),
        encoding="utf-8",
    )

    blocks, _ = server_blocks(
        _plan(tmp_path),
        dsn="postgresql://recall:pw@127.0.0.1:5432/recall",
        promoted=frozenset({"docs", "code"}),
        serving=frozenset({"docs", "code", "memory"}),
    )
    write_mcp_config(path, mcp_config(blocks, project_root=tmp_path))

    written = json.loads(path.read_text(encoding="utf-8"))["mcpServers"]
    assert written["someone-elses"]["url"] == "https://example.invalid/mcp", "must survive"
    assert written["docs"]["command"] == "python", "and ours must be replaced, not merged into"
    assert set(written) == {"someone-elses", "docs", "code", "memory"}


def test_writing_over_a_corrupt_file_does_not_lose_the_install(tmp_path: Path) -> None:
    """A file that cannot be parsed is replaced. There is nothing in it to preserve."""
    path = tmp_path / ".mcp.json"
    path.write_text("{ this is not json", encoding="utf-8")

    blocks, _ = server_blocks(
        _plan(tmp_path),
        dsn="postgresql://recall:pw@127.0.0.1:5432/recall",
        promoted=frozenset({"docs", "code"}),
        serving=frozenset({"docs", "code", "memory"}),
    )
    write_mcp_config(path, mcp_config(blocks, project_root=tmp_path))

    assert set(json.loads(path.read_text(encoding="utf-8"))["mcpServers"]) == {
        "docs",
        "code",
        "memory",
    }


def test_the_written_file_has_no_temporary_left_behind_and_lf_endings(tmp_path: Path) -> None:
    path = tmp_path / "nested" / ".mcp.json"
    blocks, _ = server_blocks(
        _plan(tmp_path),
        dsn="postgresql://recall:pw@127.0.0.1:5432/recall",
        promoted=frozenset({"docs"}),
        serving=frozenset({"docs", "memory"}),
    )
    write_mcp_config(path, mcp_config(blocks, project_root=tmp_path))

    assert not list(path.parent.glob("*.tmp"))
    assert b"\r\n" not in path.read_bytes(), "CRLF would rewrite every line on every platform"


def test_the_server_command_launches_the_real_module_from_the_project_root(tmp_path: Path) -> None:
    """A `cwd` or module name that is wrong produces a server that never starts, silently."""
    blocks, _ = server_blocks(
        _plan(tmp_path),
        dsn="postgresql://recall:pw@127.0.0.1:5432/recall",
        promoted=frozenset({"docs"}),
        serving=frozenset({"docs", "memory"}),
    )
    document = mcp_config(blocks, project_root=tmp_path / "project")

    entry = document["mcpServers"]["docs"]  # type: ignore[index]
    assert entry["type"] == "stdio"
    assert entry["args"] == ["-m", "recall_mcp.server"]
    assert entry["cwd"] == str(tmp_path / "project")


# ----------------------------------------------------------------------------------------------
# Through the driver
# ----------------------------------------------------------------------------------------------


def test_no_project_root_writes_nothing_and_says_so(tmp_path: Path) -> None:
    """Guessing a location for an MCP configuration is a side effect nobody can predict."""
    from recall.wizard.headless import run_headless
    from tests.test_wizard_state import _CountingSpy, _config

    report = run_headless(_config(tmp_path), services=_CountingSpy())

    assert report.mcp_path is None
    assert report.servers == ()
    assert not list(tmp_path.glob(".mcp.json"))


def test_a_project_root_writes_the_configuration(tmp_path: Path) -> None:
    from recall.wizard.headless import run_headless
    from tests.test_wizard_state import _CountingSpy, _config

    root = tmp_path / "project"
    root.mkdir()
    config = _config(tmp_path, project_root=str(root))

    report = run_headless(config, services=_CountingSpy())

    assert report.mcp_path == root / ".mcp.json"
    assert {b.name for b in report.servers} == {"docs", "code", "memory"}
    written = json.loads((root / ".mcp.json").read_text(encoding="utf-8"))
    assert set(written["mcpServers"]) == {"docs", "code", "memory"}
    assert "wrote" in report.render()

    # `.env` and `CLAUDE.md` too, and every file touched must be NAMED: these are block-scoped
    # edits to files the operator owns, and a block-scoped edit is invisible in a listing.
    assert (root / ".env").exists()
    assert (root / "CLAUDE.md").exists()
    assert "RECALL_DSN" in (root / ".env").read_text(encoding="utf-8")
    named = report.render()
    for path in report.files_written:
        assert str(path) in named, f"{path} was written and not reported"
    assert {p.name for p in report.files_written} >= {".mcp.json", ".env", "CLAUDE.md"}


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


def test_an_unwritable_project_root_reports_rather_than_discarding_the_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every corpus is already built and promoted by the time the wiring is written.

    Throwing that away because a directory is read-only would be the expensive half of the trade,
    so the failure is reported against a `wiring` pseudo-tenant and the corpora keep their outcomes.
    """
    import recall.wizard.headless as H
    from tests.test_wizard_state import _CountingSpy, _config

    def _boom(path: Path, document: dict) -> None:
        raise OSError(13, "Permission denied")

    monkeypatch.setattr(H, "write_mcp_config", _boom)

    root = tmp_path / "project"
    root.mkdir()
    report = H.run_headless(_config(tmp_path, project_root=str(root)), services=_CountingSpy())

    assert [o.tenant for o in report.outcomes] == ["docs", "code"], "the builds must survive"
    assert [f.tenant for f in report.failures] == ["wiring"]
    assert report.mcp_path is None
    assert report.ok is False, "an install nobody can reach is not complete"
