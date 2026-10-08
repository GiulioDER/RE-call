"""The MCP server on a `sqlite:///` DSN: its real startup path, then its tools, with no database server.

The store is opened by the server's own lifespan, not handed in, so the Postgres-only startup
probes (migrations, row-level security) are exercised being skipped. The embedder is the
content-word embedder from `test_lite_calibration`, patched in where the lifespan builds its own,
so calibration certifies offline.

Invariants and the failure each one catches:
- N1 a lite server does not offer the tools that need PostgreSQL (the fact ledger, calibration
  publish) and offers the rest.
- N2 `recall_index` calibrates a lite store, after which strict `recall_search` answers, trusted,
  from the memo that holds the answer.
- N3 `recall_calibration_status` reports the lite calibration instead of reaching for a DSN.
- N4 `recall_report_use` and `recall_report_stale` write their rows to the lite audit ledger.
- N5 `recall_forget` resolves a root-relative file name, as a search hit shows it, and deletes it.
- N6 the generation route is refused at startup, by name.
- N7 the reasoning tools work on a lite store before any calibration exists.

Red proof, 2026-10-07, each mutation alone, failing in its intended assertion (JUnit XML), then
restored byte for byte and green:
- S1 (N1) `LITE_UNSERVED_TOOLS` not subtracted in `build_server`: "a Postgres-only tool was offered".
- S2 (N2) `ensure_calibrated` removed from `recall_mcp.indexing.index_memory`: "recall_index did not
  calibrate the lite store".
- S3 (N3) the lite branch removed from `recall_mcp.status.calibration_status`: "the status was not
  the lite calibration" (the call reached for `store._dsn`).
- S4 (N4) `LiteStore.append_audit_event` writing nothing: "the use report left no ledger row".
- S5 (N5) `LiteStore.sources_for_identifiers` matching the absolute source only: "forget by file
  name deleted nothing".
- S6 (N6) the lite route refusal removed from the lifespan: the startup error no longer names the
  missing generations.
- S7 (N7) `_reasoning_generation` indexing `payload["pipeline_fingerprint"]` again: "a reasoning
  tool failed on an uncalibrated lite store" (KeyError).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from recall_mcp import server as srv
from recall_mcp.settings import bootstrap_settings
from recall_mcp.tool_surface import LITE_UNSERVED_TOOLS
from tests.test_lite_calibration import _NOUNS, ContentWordEmbedder, _word, _write_memos

QUESTION = f"who services the {_word(7, 1)} {_NOUNS[7]}"


def _env(tmp_path: Path) -> dict[str, str]:
    return {
        "RECALL_DSN": "sqlite:///" + (tmp_path / "store" / "memory.db").as_posix(),
        "RECALL_EMBEDDER": "hashing",
        "RECALL_INDEX_ROOT": str(tmp_path),
        "RECALL_EMBED_CACHE": "off",
        "RECALL_TRANSPORT": "stdio",
    }


@pytest.fixture
def lite_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    for name in ("RECALL_SERVING_DSN", "RECALL_ENV", "RECALL_INDEX_MODE", "RECALL_TRUST_MODE",
                 "RECALL_PINNED_GENERATION_ID", "RECALL_MCP_TOOLS", "RECALL_DECISION_LEDGER"):
        monkeypatch.delenv(name, raising=False)
    env = _env(tmp_path)
    for name, value in env.items():
        monkeypatch.setenv(name, value)  # the route is resolved from the process environment
    monkeypatch.setattr(srv, "make_embedder", lambda *a, **k: ContentWordEmbedder())
    return env


def _serve(env: dict[str, str], body: Callable[[dict[str, Any], Callable[..., Any]], Any]) -> Any:
    """Build the server, run its real lifespan, and hand `body` the state and a tool caller."""
    settings = bootstrap_settings(env)
    mcp = srv.build_server(settings)
    tools = {t.name: t for t in mcp._tool_manager.list_tools()}

    async def run() -> Any:
        async with srv._make_lifespan(None, None, {}, settings)(mcp) as state:
            ctx = SimpleNamespace(request_context=SimpleNamespace(lifespan_context=state))

            async def call(name: str, **kwargs: Any) -> str:
                try:
                    out = await tools[name].fn(ctx=ctx, **kwargs)
                except Exception as exc:  # noqa: BLE001  # the text of a failure is what is asserted on
                    return f"RAISED {type(exc).__name__}: {exc}"
                return out if isinstance(out, str) else json.dumps(out, default=str)

            return await body(state, call)

    return asyncio.run(run())


def test_a_lite_server_does_not_offer_postgres_only_tools(lite_env: dict[str, str]) -> None:
    """N1."""
    tools = {t.name for t in srv.build_server(bootstrap_settings(lite_env))._tool_manager.list_tools()}
    assert not tools & LITE_UNSERVED_TOOLS, f"a Postgres-only tool was offered: {sorted(tools & LITE_UNSERVED_TOOLS)}"
    assert {"recall_search", "recall_index", "recall_forget", "recall_calibration_status"} <= tools


def test_index_calibrates_then_strict_search_answers(lite_env: dict[str, str], tmp_path: Path) -> None:
    """N2 and N3."""
    _write_memos(tmp_path / "mem", 48)

    async def body(state: dict[str, Any], call: Callable[..., Any]) -> None:
        raw = await call("recall_calibration_status")
        assert '"store": "lite"' in raw and '"status": "missing"' in raw, f"the status was not the lite calibration: {raw}"
        indexed = json.loads(await call("recall_index", path=str(tmp_path / "mem")))
        assert indexed["chunks"] == 48
        assert "Calibration: certified" in indexed["message"], f"recall_index did not calibrate the lite store: {indexed}"
        status = json.loads(await call("recall_calibration_status"))
        assert status["status"] == "certified" and status["generation_id"].startswith("lite-")
        found = await call("recall_search", query=QUESTION)
        result = json.loads(found)
        assert result["calibrated"] is True and result["abstained"] is False, found
        top = result["hits"][0]
        assert top["verdict"] == "ok" and "memo-007.md" in json.dumps(top), f"strict search did not answer from memo-007: {top}"

    _serve(lite_env, body)


def test_the_report_tools_write_the_lite_ledger(lite_env: dict[str, str], tmp_path: Path) -> None:
    """N4."""
    _write_memos(tmp_path / "mem", 48)

    async def body(state: dict[str, Any], call: Callable[..., Any]) -> None:
        await call("recall_index", path=str(tmp_path / "mem"))
        used = await call("recall_report_use", task="find the crew", effect="helped", used=["memo-007.md"], query=QUESTION)
        assert json.loads(used)["recorded"] is True, used
        stale = await call(
            "recall_report_stale",
            stale_source="memo-001.md",
            replacing_source="memo-002.md",
            stale_quote=f"is serviced by the {_word(1, 2)} team",
            current_quote=f"is serviced by the {_word(2, 2)} team",
        )
        assert "audit_ledger" in json.loads(stale)["recorded_in"], stale
        rows = state["store"]._conn.execute("SELECT event_type FROM audit_events ORDER BY event_type").fetchall()
        kinds = [row[0] for row in rows]
        assert "use_report" in kinds, "the use report left no ledger row"
        assert "stale_report" in kinds, "the stale report left no ledger row"

    _serve(lite_env, body)


def test_forget_by_file_name(lite_env: dict[str, str], tmp_path: Path) -> None:
    """N5."""
    _write_memos(tmp_path / "mem", 4)

    async def body(state: dict[str, Any], call: Callable[..., Any]) -> None:
        await call("recall_index", path=str(tmp_path / "mem"))
        forgot = json.loads(await call("recall_forget", sources=["memo-003.md"]))
        assert forgot["chunks_removed"] == 1, f"forget by file name deleted nothing: {forgot}"
        assert state["store"].count() == 3

    _serve(lite_env, body)


def test_the_generation_route_is_refused_at_startup(lite_env: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> None:
    """N6. A pinned generation needs this route too, so the same refusal covers it."""
    monkeypatch.setenv("RECALL_INDEX_MODE", "generation")
    env = {**lite_env, "RECALL_INDEX_MODE": "generation"}

    async def body(state: dict[str, Any], call: Callable[..., Any]) -> None:
        return None

    try:
        _serve(env, body)
    except Exception as exc:  # noqa: BLE001  # which error it is, is what is asserted
        failure = f"{type(exc).__name__}: {exc}"
    else:
        failure = "the server started"
    assert failure.startswith("RuntimeError") and "has no generations" in failure, (
        f"the startup error did not name the missing generations: {failure}"
    )


def test_reasoning_tools_work_before_any_calibration(lite_env: dict[str, str], tmp_path: Path) -> None:
    """N7."""
    _write_memos(tmp_path / "mem", 4)  # under the certification floor: nothing is calibrated

    async def body(state: dict[str, Any], call: Callable[..., Any]) -> None:
        await call("recall_index", path=str(tmp_path / "mem"))
        for name, kwargs in (("recall_reasoning_query", {"query": QUESTION}), ("recall_reasoning_audit", {"query": QUESTION})):
            out = await call(name, **kwargs)
            assert not out.startswith("RAISED"), f"a reasoning tool failed on an uncalibrated lite store: {out}"

    _serve(lite_env, body)
