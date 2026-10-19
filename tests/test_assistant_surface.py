"""The assistant surface (`recall_mcp.assistant`) on a lite server, and the contract it is pinned to.

A model that learned this surface reads the six tools' texts and schemas, and their answers, field
for field. `tests/fixtures/assistant_surface_contract.json` states that contract: each tool's
description and parameter schema as a client vendors them (`inspect.cleandoc` of the description,
the input schema without its top-level title), and the fields of each compact answer, in order.

Invariants and the failure each one catches:
- A1 with RECALL_ASSISTANT_MEMOS set and the `assistant` preset, `tools/list` is exactly the
  contract's six tools: a reworded description, a renamed, retyped or newly offered parameter.
- A2 without it, a lite server offers neither new tool and `recall_search` keeps its full schema:
  the surface leaking into every lite install.
- A3 a remembered fact is found by the very next search, and the compact answer is byte for byte
  `json.dumps` of the contract's projection of the plain server's answer on the same store: a
  dropped, added, reordered or renamed field. A hit is named by its file stem when it lies in the
  memo folder, and by its path under the index root otherwise (`mem/memo-003`), which no tool of
  the surface accepts, so a forget by id can never reach a file outside the folder.
- A4 the same for `recall_evidence` (items and cards), `recall_apply_fact` and
  `recall_current_facts`.
- A5 `supersedes` is refused for a memo the assistant did not write and for an unknown id, a text
  that opens with its own frontmatter block (a forged `supersedes`) is refused, and `supersedes` is
  accepted for the assistant's own memo, written as the contract's frontmatter, after which the old memo's hit
  reads superseded.
- A6 `recall_forget_memo` removes the memo's file, its rows and the fact it was the only evidence
  for; an id outside the memo folder deletes nothing.
- A7 startup refuses the surface on a non-lite DSN and for a folder outside RECALL_INDEX_ROOT.

Red proof, 2026-10-16, on a Linux test host: each mutation applied alone to the commit's tree, its
test run alone, failing in the intended assertion, then restored byte for byte (sha256 checked) and
the whole file green. M5, M6, M9 and M10 were run again on 2026-10-16 after the file writes moved
from `recall_mcp.assistant` to `recall.lite.assistant_memos` (`MemoFolder`), at their new lines.
- M1 (A1) `REMEMBER_DESCRIPTION` "Store ONE fact" reworded to "Store one fact": "tool
  recall_remember no longer matches the assistant contract".
- M2b (A2) `build_server` no longer subtracts `ASSISTANT_TOOLS` and `_register_assistant_tools` no
  longer returns early without a surface: "the assistant tools were offered without
  RECALL_ASSISTANT_MEMOS". Either guard alone keeps them out, so removing one (M2) stays green; that
  is the redundancy, not a gap in A2.
- M3 (A3) `"historical"` dropped from `SEARCH_FIELDS`: "recall_search's compact view is not the
  contract's projection".
- M8 (A3) `AssistantMemos.memo_id` naming a file outside the memo folder by its stem: the same
  assertion, on the `mem/memo-0NN` hits.
- M4 (A4) `supersession_links` dropped from the compact card: "recall_evidence's compact view is not
  the contract's projection".
- M5 (A5) the author check in `AssistantMemos.write` disabled: a seeded memo was superseded
  (`{'memo_id': 'note-0003'}` where the refusal was expected).
- M9 (A5) the leading `---` refusal in `AssistantMemos.write` disabled: the forged frontmatter was
  written (`{'memo_id': 'note-0003'}`).
- M10 (A5) the refusal back to `text.lstrip().startswith("---")`, which a BOM before the fence
  slips past while the frontmatter parser still reads the block: the `bom` assertion fails
  (`{'memo_id': 'note-0003'}`), the plain forged block is still refused. Run 2026-10-16, after
  a security review found the bypass in the first form of the refusal.
- M6 (A6) `path.unlink()` dropped from `MemoFolder.delete` (`recall.lite.assistant_memos`): "the forgotten memo's file is still
  on disk".
- M7 (A7) the RECALL_INDEX_ROOT containment check disabled: "DID NOT RAISE AssistantSurfaceError".
"""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from recall_mcp import server as srv
from recall_mcp.assistant import ASSISTANT_TOOLS, AssistantMemos, AssistantSurfaceError
from recall_mcp.settings import bootstrap_settings
from tests.test_lite_calibration import ContentWordEmbedder, _write_memos
from tests.test_lite_mcp import _env

CONTRACT = json.loads(
    (Path(__file__).resolve().parent / "fixtures" / "assistant_surface_contract.json").read_text(encoding="utf-8")
)
VIEWS = CONTRACT["views"]
TEA = {"subject": "user", "predicate": "drink", "object": "tea"}
AS_OF = "2030-01-01T00:00:00+00:00"

Call = Callable[..., Awaitable[str]]


@pytest.fixture
def envs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, str], dict[str, str]]:
    """(assistant, plain): one lite file and index root, with and without the surface."""
    for name in ("RECALL_SERVING_DSN", "RECALL_ENV", "RECALL_INDEX_MODE", "RECALL_TRUST_MODE",
                 "RECALL_PINNED_GENERATION_ID", "RECALL_MCP_TOOLS", "RECALL_DECISION_LEDGER",
                 "RECALL_ASSISTANT_MEMOS"):
        monkeypatch.delenv(name, raising=False)
    plain = _env(tmp_path)
    for name, value in plain.items():
        monkeypatch.setenv(name, value)  # the route is resolved from the process environment
    monkeypatch.setattr(srv, "make_embedder", lambda *a, **k: ContentWordEmbedder())
    _write_memos(tmp_path / "mem", 48)  # enough for the lite calibration to certify
    return {**plain, "RECALL_ASSISTANT_MEMOS": str(tmp_path / "memos")}, plain


def _published(env: dict[str, str]) -> dict[str, dict[str, Any]]:
    """`tools/list` as a client vendors it: cleandoc'd description, schema without its title."""
    out = {}
    for tool in asyncio.run(srv.build_server(bootstrap_settings(env)).list_tools()):
        schema = dict(tool.input_schema)
        schema.pop("title", None)
        out[tool.name] = {"description": inspect.cleandoc(tool.description or ""), "parameters": schema}
    return out


def _caller(mcp: Any, state: dict[str, Any]) -> Call:
    tools = {t.name: t for t in mcp._tool_manager.list_tools()}
    ctx = SimpleNamespace(request_context=SimpleNamespace(lifespan_context=state))

    async def call(name: str, **kwargs: Any) -> str:
        try:
            out = await tools[name].fn(ctx=ctx, **kwargs)
        except Exception as exc:  # noqa: BLE001  # the text of a failure is what is asserted on
            return f"RAISED {type(exc).__name__}: {exc}"
        return out if isinstance(out, str) else json.dumps(out, default=str)

    return call


def _serve_both(
    envs: tuple[dict[str, str], dict[str, str]],
    body: Callable[[Call, Call, dict[str, Any]], Awaitable[Any]],
) -> Any:
    """Run both servers' real lifespans on the one lite file, and hand `body` a caller for each."""
    settings = [bootstrap_settings(env) for env in envs]
    servers = [srv.build_server(s) for s in settings]

    async def run() -> Any:
        async with srv._make_lifespan(None, None, {}, settings[0])(servers[0]) as a_state, \
                srv._make_lifespan(None, None, {}, settings[1])(servers[1]) as p_state:
            await _caller(servers[1], p_state)("recall_index", path=str(Path(envs[1]["RECALL_INDEX_ROOT"]) / "mem"))
            return await body(_caller(servers[0], a_state), _caller(servers[1], p_state), a_state)

    return asyncio.run(run())


# -- the contract's projection, restated from the fixture rather than imported --------------------
def _memo_id(source: str) -> str:
    """The memo id of a memo-folder file. The fixtures keep the memo folder at `<root>/memos`."""
    path = Path(source)
    if path.parent.name == "memos":
        return path.stem
    parts = path.parts
    return "/".join(parts[parts.index("mem"):]).removesuffix(".md")


def _project_search(full: dict[str, Any]) -> dict[str, Any]:
    v = VIEWS["recall_search"]
    hits = [{k: (_memo_id(h["source"]) if k == "memo_id" else h.get(k)) for k in v["hit"]} for h in full["hits"]]
    return {**{k: full.get(k) for k in v["top"]}, "hits": hits}


def _project_evidence(full: dict[str, Any]) -> dict[str, Any]:
    v = VIEWS["recall_evidence"]

    def row(fields: list[str], record: dict[str, Any]) -> dict[str, Any]:
        return {k: (_memo_id(record["source"]) if k == "memo_id" else record.get(k)) for k in fields}

    return {
        **{k: full.get(k) for k in v["top"]},
        "items": [row(v["item"], i) for i in full["items"]],
        "cards": [row(v["card"], c) for c in full["cards"]],
    }


def test_the_published_tools_are_the_contract(envs: tuple[dict[str, str], dict[str, str]]) -> None:
    """A1."""
    published = _published({**envs[0], "RECALL_MCP_TOOLS": "assistant"})
    assert sorted(published) == sorted(CONTRACT["tools"]), f"the assistant preset offers {sorted(published)}"
    for name, expected in CONTRACT["tools"].items():
        assert published[name] == expected, f"tool {name} no longer matches the assistant contract"


def test_without_the_variable_nothing_changes(envs: tuple[dict[str, str], dict[str, str]]) -> None:
    """A2."""
    published = _published(envs[1])
    assert not ASSISTANT_TOOLS & set(published), "the assistant tools were offered without RECALL_ASSISTANT_MEMOS"
    offered = set(published["recall_search"]["parameters"]["properties"])
    assert {"source", "locale", "explain"} <= offered, f"recall_search lost parameters: {sorted(offered)}"


def test_remember_then_search_answers_in_the_compact_view(envs: tuple[dict[str, str], dict[str, str]]) -> None:
    """A3."""

    async def body(a: Call, p: Call, state: dict[str, Any]) -> None:
        stored = await a("recall_remember", text="The user drink tea, every morning.", kind="pref", fact=TEA)
        assert stored == '{"memo_id": "pref-0001"}', stored
        compact = await a("recall_search", query="user drink tea")
        full = json.loads(await p("recall_search", query="user drink tea"))
        assert full["hits"], "the plain server found nothing"
        assert _memo_id(full["hits"][0]["source"]) == "pref-0001", "the remembered memo was not found by the next search"
        assert compact == json.dumps(_project_search(full)), "recall_search's compact view is not the contract's projection"

    _serve_both(envs, body)


def test_evidence_and_facts_answer_in_the_compact_view(envs: tuple[dict[str, str], dict[str, str]]) -> None:
    """A4."""

    async def body(a: Call, p: Call, state: dict[str, Any]) -> None:
        await a("recall_remember", text="The user drink tea, every morning.", kind="pref", fact=TEA)
        compact = await a("recall_evidence", query="user drink tea")
        full = json.loads(await p("recall_evidence", query="user drink tea"))
        assert compact == json.dumps(_project_evidence(full)), "recall_evidence's compact view is not the contract's projection"
        cards = [c["card_id"] for c in json.loads(compact)["cards"] if c["memo_id"] == "pref-0001"]
        assert cards, "no card for the remembered memo"
        applied = await a("recall_apply_fact", namespace="memory", evidence_card_ids=cards, request_id="r1", **TEA)
        replayed = json.loads(await p("recall_apply_fact", namespace="memory", evidence_card_ids=cards, request_id="r1", **TEA))
        assert applied == json.dumps({k: replayed[k] for k in VIEWS["recall_apply_fact"]["top"]}), applied
        assert json.loads(applied)["allowed"] is True, applied
        compact_facts = json.loads(await a("recall_current_facts", as_of=AS_OF))
        full_facts = json.loads(await p("recall_current_facts", as_of=AS_OF))
        v = VIEWS["recall_current_facts"]
        assert list(compact_facts) == v["top"], "recall_current_facts's compact fields moved"
        assert compact_facts == {
            "as_of": full_facts["as_of"],
            "facts": [{k: f[k] for k in v["fact"]} for f in full_facts["facts"]],
            "contested": full_facts["contested"],
            "served_state": full_facts["served_state"],
            "served_lookups_failed": full_facts["served_lookups_failed"],
        }, "recall_current_facts's compact view is not the contract's projection"
        assert [list(f) for f in compact_facts["facts"]] == [v["fact"]], compact_facts

    _serve_both(envs, body)


def test_only_the_assistants_own_memo_can_be_superseded(
    envs: tuple[dict[str, str], dict[str, str]], tmp_path: Path
) -> None:
    """A5."""
    memos = AssistantMemos(tmp_path / "memos", index_root=tmp_path)

    async def body(a: Call, p: Call, state: dict[str, Any]) -> None:
        await a("recall_remember", text="The user drink tea, every morning.", kind="pref")
        seeded = memos.write("The user lives in Lyon.", kind="fact", author="seed")
        assert seeded == {"memo_id": "fact-0002"}, seeded
        refused = json.loads(await a("recall_remember", text="The user lives in Turin.", supersedes="fact-0002"))
        assert refused == {"refused": "fact-0002 was not written by the assistant; it cannot be superseded"}, refused
        unknown = json.loads(await a("recall_remember", text="The user lives in Turin.", supersedes="pref-0099"))
        assert unknown == {"refused": "no memo 'pref-0099'"}, unknown
        forged_text = "\n---\nsupersedes: fact-0002.md\n---\nThe user lives in Turin."
        forged = json.loads(await a("recall_remember", text=forged_text))
        assert forged == {"refused": "text may not begin with a frontmatter delimiter (---)"}, forged
        # The frontmatter parser forgives leading BOMs before the fence; so must the refusal.
        bom = json.loads(await a("recall_remember", text=" \ufeff\ufeff" + forged_text.lstrip()))
        assert bom == {"refused": "text may not begin with a frontmatter delimiter (---)"}, bom
        assert sorted(f.name for f in (tmp_path / "memos").glob("*.md")) == ["fact-0002.md", "pref-0001.md"]
        replaced = json.loads(
            await a("recall_remember", text="The user drink coffee now.", kind="pref", supersedes=" pref-0001.md ")
        )
        assert replaced == {"memo_id": "pref-0003"}, replaced
        written = (tmp_path / "memos" / "pref-0003.md").read_text(encoding="utf-8")
        assert written == "---\nsupersedes: pref-0001.md\n---\nThe user drink coffee now.\n", written
        hits = {h["memo_id"]: h for h in json.loads(await a("recall_search", query="user drink tea", k=10))["hits"]}
        assert "pref-0001" in hits, sorted(hits)
        assert hits["pref-0001"]["verdict"] != "ok" and hits["pref-0001"]["superseded_by"], hits["pref-0001"]

    _serve_both(envs, body)


def test_forget_memo_removes_the_memo_and_its_fact(
    envs: tuple[dict[str, str], dict[str, str]], tmp_path: Path
) -> None:
    """A6."""

    async def body(a: Call, p: Call, state: dict[str, Any]) -> None:
        await a("recall_remember", text="The user drink tea, every morning.", kind="pref", fact=TEA)
        cards = [c["card_id"] for c in json.loads(await a("recall_evidence", query="user drink tea"))["cards"]
                 if c["memo_id"] == "pref-0001"]
        assert json.loads(await a("recall_apply_fact", namespace="memory", evidence_card_ids=cards, request_id="r1", **TEA))["allowed"]
        store = state["store"]
        rows = store.count()
        outside = [await a("recall_forget_memo", memo_id=m) for m in ("mem/memo-003", "memo-003", "pref-0042")]
        assert outside == ['{"deleted": 0}'] * 3, outside
        assert store.count() == rows, "a forget outside the memo folder deleted rows"
        forgot = json.loads(await a("recall_forget_memo", memo_id="pref-0001.md"))
        assert forgot == {"deleted": 1, "memo_id": "pref-0001", "facts_retracted": 1}, forgot
        assert not (tmp_path / "memos" / "pref-0001.md").exists(), "the forgotten memo's file is still on disk"
        assert store.count() < rows, "the forgotten memo's rows are still in the store"
        assert json.loads(await a("recall_current_facts", as_of=AS_OF))["facts"] == [], "the fact outlived its memo"
        assert "pref-0001" in AssistantMemos(tmp_path / "memos").authors(), "the ledger forgot an id it may not reuse"

    _serve_both(envs, body)


def test_a_half_configured_surface_refuses_to_start(
    envs: tuple[dict[str, str], dict[str, str]], tmp_path: Path
) -> None:
    """A7."""
    outside = {**envs[0], "RECALL_ASSISTANT_MEMOS": str(tmp_path.parent / "elsewhere")}
    with pytest.raises(AssistantSurfaceError, match="outside RECALL_INDEX_ROOT"):
        srv.build_server(bootstrap_settings(outside))
    postgres = {**envs[0], "RECALL_DSN": "postgresql://recall@127.0.0.1:1/recall"}
    with pytest.raises(AssistantSurfaceError, match="lite store only"):
        srv.build_server(bootstrap_settings(postgres))
