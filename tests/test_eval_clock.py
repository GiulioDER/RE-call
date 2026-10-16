"""The evaluation clock (`recall.eval_clock`): a lite server that reads "now" from a file.

An evaluation driver that replays weeks in minutes keeps time in a file; the server must age its
memory with it exactly as an in-process driver does when it binds `now` and `known_as_of` on every
search, patches the lite row clock, and hands the fact controller its own clock.

Invariants and the failure each one catches:
- K1 a fact's verdict follows the file: `ok` before its `valid_until`, `expired` after, though the
  wall clock is years earlier than both: a search that ignores the clock.
- K2 a memo written under the clock carries the clock's time as its row time: the lite row clock
  left on the wall.
- K3 the server's compact search under the clock is byte for byte the in-process search with
  `now` and `known_as_of` bound to the same time, rendered the same way: any divergence between
  driving the server and driving the library.
- K3b the clock is the known-as-of time as well: a memo superseded by a memo written on a later
  simulated day reads `ok` when the clock is moved back before that write, as the library bound
  to the same time answers: a clock bound as valid time only.
- K4 a fact applied under the clock is asserted at the clock's time, and the current facts are
  read as of it: the controller or the projection left on the wall.
- K5 it fails closed: a clock file that vanishes fails the next search instead of answering from
  the wall clock, a time with no offset refuses startup, and a PostgreSQL DSN refuses the clock.
- K6 the clock is gone once the server stops: a clock leaking into the rest of the process.

Red proof, 2026-10-16, on a Linux test host: each mutation applied alone, its test run alone,
failing in the intended assertion, then restored byte for byte (sha256 checked) and the file green.
- C1 (K1) `trusted_search` ignoring the clock: "the verdict did not follow the clock file" (`ok`).
- C2 (K2) `recall.lite.store._now` back on the wall clock: "row time was not the clock's".
- C3 (K3b) `trusted_search` binding the clock as `now` but not `known_as_of`: "a supersession
  written on 01-20 was already known on 01-10". K3 alone stayed green under C3, because nothing in
  its store is superseded; K3b exists for that reason.
- C4 (K4) `ProvenanceController` back on the wall clock: "the fact was not asserted at the clock's
  time".
- C5 (K4) `current_facts_memory` back on the wall clock: "current facts were not read at the
  clock's time".
- C6 (K5) an unreadable clock file answering with the wall clock: "a vanished clock file did not
  fail the search".
- C7 (K5) the offset check removed: "DID NOT RAISE EvalClockError".
- C8 (K6) the lifespan no longer uninstalling the clock: "the clock outlived the server".
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any

import pytest

from recall import eval_clock
from recall_mcp import retrieval
from recall_mcp.assistant import AssistantMemos
from tests import test_assistant_surface as surface
from tests.test_assistant_surface import TEA, Call, _serve_both

T1 = "2030-01-05T09:00:00+00:00"
T2 = "2030-01-20T09:00:00+00:00"
QUERY = "user drink tea"

#: The assistant and plain lite servers on one file, from the surface's own tests.
envs = surface.envs


@pytest.fixture
def clocked(
    envs: tuple[dict[str, str], dict[str, str]], tmp_path_factory: pytest.TempPathFactory
) -> tuple[tuple[dict[str, str], dict[str, str]], Path]:
    """The assistant server under a clock file kept outside the index root; the plain one beside it."""
    clock = tmp_path_factory.mktemp("clock") / "now.txt"
    clock.write_text(T1, encoding="utf-8")
    assistant, plain = envs
    return ({**assistant, "RECALL_EVAL_CLOCK": str(clock)}, plain), clock


def _hit(raw: str, memo_id: str) -> dict[str, Any]:
    hits = {h["memo_id"]: h for h in json.loads(raw)["hits"]}
    assert memo_id in hits, f"{memo_id} not among {sorted(hits)}"
    return hits[memo_id]


def test_a_verdict_follows_the_clock_file(clocked: Any) -> None:
    """K1."""
    envs, clock = clocked

    async def body(a: Call, p: Call, state: dict[str, Any]) -> None:
        stored = await a("recall_remember", text="The user drink tea, every morning.", kind="pref",
                         valid_until="2030-01-10")
        assert stored == '{"memo_id": "pref-0001"}', stored
        assert _hit(await a("recall_search", query=QUERY, k=10), "pref-0001")["verdict"] == "ok"
        clock.write_text(T2, encoding="utf-8")
        after = _hit(await a("recall_search", query=QUERY, k=10), "pref-0001")
        assert after["verdict"] == "expired", f"the verdict did not follow the clock file: {after}"

    _serve_both(envs, body)


def test_a_row_written_under_the_clock_carries_its_time(clocked: Any) -> None:
    """K2."""
    envs, _ = clocked

    async def body(a: Call, p: Call, state: dict[str, Any]) -> None:
        await a("recall_remember", text="The user drink tea, every morning.", kind="pref")
        full = json.loads(await p("recall_search", query=QUERY))
        mine = [h for h in full["hits"] if Path(h["source"]).stem == "pref-0001"]
        assert mine, "the plain server did not find the remembered memo"
        assert str(mine[0]["indexed_at"]).startswith("2030-01-05"), f"row time was not the clock's: {mine[0]['indexed_at']}"

    _serve_both(envs, body)


def test_the_server_under_the_clock_is_the_library_bound_to_it(clocked: Any, tmp_path: Path) -> None:
    """K3."""
    envs, clock = clocked
    memos = AssistantMemos(tmp_path / "memos", index_root=tmp_path)

    async def body(a: Call, p: Call, state: dict[str, Any]) -> None:
        await a("recall_remember", text="The user drink tea, every morning.", kind="pref", valid_until="2030-01-10")
        clock.write_text(T2, encoding="utf-8")
        served = await a("recall_search", query=QUERY)
        when = eval_clock.file_clock(clock)()
        bound = functools.partial(retrieval.trusted_search, now=when, known_as_of=when)
        result = retrieval.search_memory(
            state["store"], state["embedder"], QUERY, k=5, env={},
            _retrieve_trusted_fn=functools.partial(retrieval._retrieve_trusted, trusted_search_fn=bound),
        )
        library = json.dumps(memos.render_search(json.loads(result.model_dump_json())))
        assert served == library, "the server under the clock answered unlike the library bound to it"

    _serve_both(envs, body)


def test_a_supersession_written_later_is_not_yet_known_earlier(clocked: Any, tmp_path: Path) -> None:
    """K3b: the clock is the known-as-of time too, as the in-process driver binds it."""
    envs, clock = clocked
    memos = AssistantMemos(tmp_path / "memos", index_root=tmp_path)

    async def body(a: Call, p: Call, state: dict[str, Any]) -> None:
        await a("recall_remember", text="The user drink tea, every morning.", kind="pref")
        clock.write_text(T2, encoding="utf-8")
        await a("recall_remember", text="The user drink coffee now.", kind="pref", supersedes="pref-0001")
        assert _hit(await a("recall_search", query=QUERY, k=10), "pref-0001")["verdict"] == "superseded"
        clock.write_text("2030-01-10T09:00:00+00:00", encoding="utf-8")
        served = await a("recall_search", query=QUERY, k=10)
        rewound = _hit(served, "pref-0001")
        assert rewound["verdict"] == "ok", f"a supersession written on 01-20 was already known on 01-10: {rewound}"
        when = eval_clock.file_clock(clock)()
        bound = functools.partial(retrieval.trusted_search, now=when, known_as_of=when)
        result = retrieval.search_memory(
            state["store"], state["embedder"], QUERY, k=10, env={},
            _retrieve_trusted_fn=functools.partial(retrieval._retrieve_trusted, trusted_search_fn=bound),
        )
        assert served == json.dumps(memos.render_search(json.loads(result.model_dump_json()))), (
            "the rewound server answered unlike the library bound to the same time"
        )

    _serve_both(envs, body)


def test_facts_are_applied_and_read_at_the_clocks_time(clocked: Any) -> None:
    """K4."""
    envs, _ = clocked

    async def body(a: Call, p: Call, state: dict[str, Any]) -> None:
        await a("recall_remember", text="The user drink tea, every morning.", kind="pref", fact=TEA)
        cards = [c["card_id"] for c in json.loads(await a("recall_evidence", query=QUERY))["cards"]
                 if c["memo_id"] == "pref-0001"]
        applied = json.loads(await a("recall_apply_fact", namespace="memory", evidence_card_ids=cards,
                                     request_id="r1", **TEA))
        assert applied["allowed"] is True, applied
        current = json.loads(await a("recall_current_facts"))
        assert current["as_of"].startswith("2030-01-05"), f"current facts were not read at the clock's time: {current}"
        assert [f["asserted_at"][:10] for f in current["facts"]] == ["2030-01-05"], (
            f"the fact was not asserted at the clock's time: {current}"
        )

    _serve_both(envs, body)


def test_the_clock_fails_closed(clocked: Any) -> None:
    """K5."""
    envs, clock = clocked

    async def body(a: Call, p: Call, state: dict[str, Any]) -> None:
        clock.unlink()
        answer = await a("recall_search", query=QUERY)
        assert answer.startswith("RAISED") and "cannot be read" in answer, (
            f"a vanished clock file did not fail the search: {answer[:200]}"
        )

    _serve_both(envs, body)
    clock.write_text("2030-01-05T09:00:00", encoding="utf-8")
    with pytest.raises(eval_clock.EvalClockError, match="no offset"):
        _serve_both(envs, body)
    assert not eval_clock.installed()
    with pytest.raises(eval_clock.EvalClockError, match="lite store only"):
        eval_clock.from_env({"RECALL_EVAL_CLOCK": str(clock)}, lite=False)


def test_the_clock_is_gone_once_the_server_stops(clocked: Any) -> None:
    """K6."""
    envs, _ = clocked
    seen: list[bool] = []

    async def body(a: Call, p: Call, state: dict[str, Any]) -> None:
        seen.append(eval_clock.installed())

    _serve_both(envs, body)
    assert seen == [True], "the clock was not installed while the server ran"
    assert not eval_clock.installed(), "the clock outlived the server"
