"""The assistant surface on PostgreSQL: memos written now, indexed by the next build of the corpus.

Production indexes nothing at write time (`recall.runtime_route`): a memo reaches a tenant only in a
generation the operator builds and promotes. So on PostgreSQL `recall_remember` writes the memo and
leaves it to that build, and `recall_remember(settle=true)` settles against what the served
generation holds. These tests drive the registered tools against a real promoted generation, whose
rows name each memo by its manifest URI and its file under the corpus root (`metadata.file`).

Invariants and the failure each one catches:
- Q0 a server whose DSN is PostgreSQL and which names RECALL_ASSISTANT_MEMOS starts and registers
  `recall_remember`: the surface refused as lite only.
- Q1 a settled write of a later value names the built memo it replaces (`supersedes`), its receipt
  says `pending`, and once the next generation is promoted the key reads current with the new value
  alone: the write failing on the lite-only refresh, a receipt claiming the memo indexed, or an edge
  the generation does not resolve.
- Q2 `recall_forget_memo` of a built memo erases its rows and its fact, and deletes the file, so the
  next build leaves it out: the forget naming the memo by a path the generation does not store, and
  refusing as incomplete.
- Q3 `recall_withdraw` given a memo id deletes the file too: the held tool falling back to the plain
  erase on a store that is not lite, which leaves the file for the next build to index again.

Not covered: two servers writing one folder at once (the folder lane, `AssistantMemos._lane`).

Red proof: in the pull request that added this file.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from recall_mcp.assistant import SETTLE_ENV
from recall_mcp.settings import Settings
from tests.conftest import requires_db
from tests.test_code_anchor_chunks import manager  # noqa: F401  # `manager` is a fixture
from tests.test_stated_facts_postgres import listed, promote, served, uri

TEA = {"subject": "user", "predicate": "drink", "object": "tea"}
#: A file outside the memo folder, so the manifest's common directory is the corpus root and the
#: memos keep their folder in `metadata.file` (`recall.generations.manifest_relative_paths`).
OTHER = {"mem/other.md": b"# Other\n\nA plain note about the garden.\n"}


@pytest.fixture
def corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> tuple[Any, Path]:
    """(the generation manager, the corpus root), with the surface and settle on and the memo
    folder stating facts."""
    generations = request.getfixturevalue("manager")
    root = tmp_path / "root"
    (root / "mem").mkdir(parents=True)
    monkeypatch.setenv("RECALL_EMBED_CACHE", "off")
    monkeypatch.setenv("RECALL_INDEX_ROOT", str(root))
    monkeypatch.setenv("RECALL_ASSISTANT_MEMOS", str(root / "memos"))
    monkeypatch.setenv(SETTLE_ENV, "1")
    monkeypatch.setenv("RECALL_STATED_FACT_SOURCES", uri(generations, "memos"))
    promote(generations, OTHER, "corpus-v0")
    return generations, root


def _corpus_files(root: Path) -> dict[str, bytes]:
    """What a build of `root` reads: its markdown files, hidden ones (the journal, the lane) left out."""
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*.md"))
        if not any(part.startswith(".") for part in path.relative_to(root).parts)
    }


def _tools(store: Any) -> Any:
    """Call a registered tool with `store` as the served store (`tests/test_report_tools_postgres.py`)."""
    from recall_mcp.server import build_server

    tools = {t.name: t for t in build_server()._tool_manager.list_tools()}
    ctx = SimpleNamespace(request_context=SimpleNamespace(lifespan_context={
        "store": store, "embedder": None, "calibration": None, "settings": Settings.from_env(dict(os.environ)),
    }))

    def call(name: str, **kwargs: Any) -> str:
        try:
            return str(asyncio.run(tools[name].fn(ctx=ctx, **kwargs)))
        except Exception as exc:  # noqa: BLE001  # the text of a failure is what is asserted on
            return f"RAISED {type(exc).__name__}: {exc}"

    return call


def _remember(call: Any, obj: str, day: str, **kwargs: Any) -> Any:
    answer = call("recall_remember", text=f"The user drink {obj} now.", kind="pref", fact={**TEA, "object": obj},
                  valid_from=day, settle=True, **kwargs)
    return answer if answer.startswith("RAISED") else json.loads(answer)


def _build_tea(corpus: tuple[Any, Path]) -> None:
    """pref-0001 (tea) written by the surface and promoted in a generation."""
    generations, root = corpus
    with served(generations) as store:
        first = _remember(_tools(store), "tea", "2026-01-01")
    assert first == {"memo_id": "pref-0001", "settle": {"settled": None}}, first
    promote(generations, _corpus_files(root), "corpus-v1")


def test_q0_a_postgres_server_serves_the_surface(tmp_path: Path) -> None:
    """Node tests/test_assistant_postgres.py::test_q0_a_postgres_server_serves_the_surface. Needs no database."""
    from recall_mcp.assistant import AssistantSurfaceError
    from recall_mcp.server import build_server
    from recall_mcp.settings import bootstrap_settings

    env = {"RECALL_DSN": "postgresql://recall@127.0.0.1:1/recall", "RECALL_INDEX_ROOT": str(tmp_path),
           "RECALL_ASSISTANT_MEMOS": str(tmp_path / "memos"), SETTLE_ENV: "1"}
    try:
        names: object = {t.name for t in asyncio.run(build_server(bootstrap_settings(env)).list_tools())}
    except AssistantSurfaceError as exc:
        names = f"refused: {exc}"
    assert isinstance(names, set) and "recall_remember" in names, f"the surface is not served on PostgreSQL: {names}"


@requires_db
def test_q1_a_settled_write_supersedes_the_built_memo(corpus: tuple[Any, Path]) -> None:
    """Node tests/test_assistant_postgres.py::test_q1_a_settled_write_supersedes_the_built_memo."""
    _build_tea(corpus)
    generations, root = corpus
    with served(generations) as store:
        second = _remember(_tools(store), "coffee", "2026-02-01", include_receipt=True)
    assert not isinstance(second, str), second
    assert (second["memo_id"], second["settle"]) == ("pref-0002", {"settled": "pref-0001"}), second
    receipt = second["receipt"]
    assert (receipt["index_state"], receipt["retrieval_ready"]) == ("pending", False), receipt
    assert receipt["fact_key"]["state"] == "unchecked", receipt
    memo = (root / "memos" / "pref-0002.md").read_text(encoding="utf-8")
    assert "supersedes: memos/pref-0001.md" in memo, memo
    promote(generations, _corpus_files(root), "corpus-v2")
    with served(generations) as store:
        got = listed(store)
    assert got == [("coffee", ("memos/pref-0002.md",))], f"the settled edge did not retire tea: {got}"


@requires_db
def test_q2_forget_memo_erases_a_built_memo_and_its_file(corpus: tuple[Any, Path]) -> None:
    """Node tests/test_assistant_postgres.py::test_q2_forget_memo_erases_a_built_memo_and_its_file."""
    _build_tea(corpus)
    generations, root = corpus
    with served(generations) as store:
        answer = _tools(store)("recall_forget_memo", memo_id="pref-0001")
        got = listed(store)
    assert not answer.startswith("RAISED"), answer
    assert json.loads(answer)["deleted"] == 1, answer
    assert got == [], f"the forgotten memo's fact is still listed: {got}"
    assert not (root / "memos" / "pref-0001.md").exists(), "the next build would index the forgotten memo again"


@requires_db
def test_q3_withdraw_by_memo_id_deletes_the_file(corpus: tuple[Any, Path]) -> None:
    """Node tests/test_assistant_postgres.py::test_q3_withdraw_by_memo_id_deletes_the_file."""
    _build_tea(corpus)
    generations, root = corpus
    with served(generations) as store:
        answer = _tools(store)("recall_withdraw", sources=["pref-0001"])
        got = listed(store)
    assert not answer.startswith("RAISED"), answer
    assert got == [], f"the withdrawn memo's fact is still listed: {got}"
    assert not (root / "memos" / "pref-0001.md").exists(), (
        f"the withdrawn memo's file is left for the next build to index again: {answer}"
    )
