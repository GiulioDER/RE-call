"""`recall_remember(settle=true)`: the writer settles rule R(a')'s pair in the write that makes it.

A receipt has long named the `supersedes` rule R(a') proposes for a new memo (`fact_key`), and left it
to a second write that no measured writer makes: with the proposals unreviewed, a store of declared
keys answered 23 of 100 FactConsolidation questions, and 95 when the writer settled each pair as it
wrote (the maintainer's private study DK-1, 2026-11-25). `settle` makes that one write.

Invariants and the failure each one catches:
- S1 with the server's RECALL_ASSISTANT_SETTLE_SUCCESSORS on and `settle`, a later value of a key
  is written with `supersedes` naming the earlier memo, the key reads current with the new value,
  the authorship ledger's `settled` map names the rule, and `recall_withdraw` of the new memo gives
  the old value back: the edge dropped, or settled without a record.
- S2 `settle` on a server without the setting is refused and writes nothing: the operator's half
  ignored.
- S3 an `email` fact the writer declares `single` is declined, `not_single_in_table`, and both
  values stay in force: G1 reading the declaration instead of RE-call's table.
- S4 the other guards decline by name and write the memo without an edge: an undated memo, two
  older memos of one start, a writer's own `supersedes` (which wins), and an older memo the
  assistant did not write (which `write` would otherwise refuse, failing the whole write).
- S5 a retry with the same `request_id` replays the settled write: the superseded memo is no
  longer in force on the retry, so computing the pair again would find none and the journal, which
  hashes `supersedes` into the request, would answer `request_mismatch`.

Not covered: that the listing read and the write share one write lane. `remember` takes the lane
once around both, and a test would need a second writer timed between them.

Red proof, 2026-11-25, on the remote test host, in a copy of the branch tree: each mutation applied
alone, its test run alone, failing with an AssertionError, then the file restored byte for byte
(sha256 checked). `settle_target`'s pure test is K6 in `tests/test_declared_keys.py`.
- M1 (S1) `remember` discarding `_settle`'s target: the memo carries no `supersedes:`.
- M2 (S1) `write_memo` not recording `settled`: the ledger holds `authors` only.
- M3 (S2) the server-setting refusal disabled: the write settled `pref-0001`.
- M4 (S3, K6) G1 reading `predicate_cardinality(fact)`: the email fact settled `fact-0001`, and
  `settle_target` returned `('m/1.md', None)`.
- M5 (S4) the assistant-author check in `_settle` dropped: the whole write refused, "memos/fact-0006
  was not written by the assistant; it cannot be superseded".
- M6 (S5) the journal replay in `_settle` disabled: the retry answered `request_mismatch`.
- M7 (S4) `settle_target` without the two-older-memos guard: `pref-0004` settled `pref-0001`.
- M8 (S4) `settle_target` without the undated guard: `{"settled": None}` in place of `undated`.
- M9 (S4) `_settle` without the writer's-own-supersedes check: `pref-0005` settled `pref-0004`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from recall.declared_keys import STATED_KEY_SUCCESSOR_RULE
from recall.lite.assistant_memos import LEDGER_NAME, MemoFolder
import recall_mcp.provenance
from recall.types import AtomicFact
from recall_mcp.assistant import SETTLE_ENV, AssistantMemos
from tests.test_assistant_surface import TEA, Call, _serve_both
from tests.test_lite_calibration import ContentWordEmbedder
from tests.test_assistant_surface import envs as envs  # noqa: F401  # the fixture, reused


@pytest.fixture
def settle_envs(envs: tuple[dict[str, str], dict[str, str]]) -> tuple[dict[str, str], dict[str, str]]:
    """The assistant server with the operator's half of `settle` on."""
    assistant, plain = envs
    return {**assistant, SETTLE_ENV: "1"}, plain


def _remember(a: Call, obj: str, day: str | None, **kwargs: Any) -> Any:
    return a("recall_remember", text=f"The user drink {obj} now.", kind="pref", fact={**TEA, "object": obj},
             **({"valid_from": day} if day else {}), **kwargs)


def _values(answer: str) -> list[str]:
    facts = json.loads(answer)
    held = [f["fact"]["object"] for f in facts["facts"]]
    held += [v["fact"]["object"] for g in facts["contested"] for v in g["values"]]
    return sorted(held)


def test_a_settled_write_supersedes_the_earlier_memo_and_records_it(
    settle_envs: tuple[dict[str, str], dict[str, str]], tmp_path: Path
) -> None:
    """S1."""

    async def body(a: Call, p: Call, state: dict[str, Any]) -> None:
        first = json.loads(await _remember(a, "tea", "2026-01-01", settle=True))
        assert first == {"memo_id": "pref-0001", "settle": {"settled": None}}, first
        second = json.loads(await _remember(a, "coffee", "2026-02-01", settle=True))
        assert second == {"memo_id": "pref-0002", "settle": {"settled": "pref-0001"}}, second
        memo = (tmp_path / "memos" / "pref-0002.md").read_text(encoding="utf-8")
        assert "supersedes: memos/pref-0001.md" in memo, memo
        current = json.loads(await a("recall_current_facts", subject="user"))
        assert [f["fact"]["object"] for f in current["facts"]] == ["coffee"] and not current["contested"], current
        ledger = json.loads((tmp_path / "memos" / LEDGER_NAME).read_text(encoding="utf-8"))
        assert ledger.get("settled") == {"pref-0002": STATED_KEY_SUCCESSOR_RULE}, ledger
        await a("recall_withdraw", sources=["pref-0002"])
        assert _values(await a("recall_current_facts", subject="user")) == ["tea"], "withdraw did not give tea back"

    _serve_both(settle_envs, body)


def test_settle_is_refused_where_the_operator_has_not_turned_it_on(
    envs: tuple[dict[str, str], dict[str, str]], tmp_path: Path
) -> None:
    """S2."""

    async def body(a: Call, p: Call, state: dict[str, Any]) -> None:
        await _remember(a, "tea", "2026-01-01")
        refused = json.loads(await _remember(a, "coffee", "2026-02-01", settle=True))
        assert refused == {"refused": f"settle is off on this server; the operator turns it on with {SETTLE_ENV}"}, refused
        assert not (tmp_path / "memos" / "pref-0002.md").exists(), "a refused settle wrote a memo"

    _serve_both(envs, body)


def test_a_set_valued_predicate_is_never_settled_whatever_the_writer_declares(
    settle_envs: tuple[dict[str, str], dict[str, str]], tmp_path: Path
) -> None:
    """S3."""

    async def body(a: Call, p: Call, state: dict[str, Any]) -> None:
        for n, (address, day) in enumerate((("a@example.org", "2026-01-01"), ("b@example.org", "2026-02-01")), 1):
            answer = json.loads(await a(
                "recall_remember", text=f"The user's email address is {address}.", kind="fact", valid_from=day,
                fact={"subject": "user", "predicate": "email", "object": address, "cardinality": "single"},
                settle=True))
            assert answer["memo_id"] == f"fact-000{n}", answer
        assert answer.get("settle") == {"declined": "not_single_in_table"}, answer
        assert "supersedes" not in (tmp_path / "memos" / "fact-0002.md").read_text(encoding="utf-8")
        assert _values(await a("recall_current_facts", subject="user")) == ["a@example.org", "b@example.org"]

    _serve_both(settle_envs, body)


def test_the_other_guards_decline_by_name_and_still_write_the_memo(
    settle_envs: tuple[dict[str, str], dict[str, str]], tmp_path: Path
) -> None:
    """S4."""
    memos = AssistantMemos(tmp_path / "memos", index_root=tmp_path)

    async def body(a: Call, p: Call, state: dict[str, Any]) -> None:
        await _remember(a, "tea", "2026-01-01")
        undated = json.loads(await _remember(a, "coffee", None, settle=True))
        assert undated.get("settle") == {"declined": "undated"}, undated
        await _remember(a, "milk", "2026-01-01")  # a second memo of the first one's start
        several = json.loads(await _remember(a, "water", "2026-02-01", settle=True))
        assert several.get("settle") == {"declined": "several_older_memos"}, several
        own = json.loads(await _remember(a, "juice", "2026-03-01", settle=True, supersedes="pref-0004"))
        assert own.get("settle") == {"declined": "writer_named_supersedes"}, own
        assert "supersedes: memos/pref-0004.md" in (tmp_path / "memos" / f"{own['memo_id']}.md").read_text(encoding="utf-8")
        seeded = memos.write("The user's sister lives in Lyon.", kind="fact", valid_from="2026-01-01",
                             fact={"subject": "sister", "predicate": "city", "object": "Lyon"}, author="seed")
        await _remember(a, "soda", "2026-04-01")  # its refresh indexes the seeded memo
        moved = json.loads(await a(
            "recall_remember", text="The user's sister lives in Turin.", kind="fact", valid_from="2026-02-01",
            fact={"subject": "sister", "predicate": "city", "object": "Turin"}, settle=True))
        assert moved.get("settle") == {"declined": "not_assistant_memo"}, moved
        assert "supersedes" not in (tmp_path / "memos" / f"{moved['memo_id']}.md").read_text(encoding="utf-8")
        assert seeded["memo_id"] != moved["memo_id"], (seeded, moved)

    _serve_both(settle_envs, body)


def test_a_retry_replays_the_settled_write(settle_envs: tuple[dict[str, str], dict[str, str]]) -> None:
    """S5."""

    async def body(a: Call, p: Call, state: dict[str, Any]) -> None:
        await _remember(a, "tea", "2026-01-01")
        first = json.loads(await _remember(a, "coffee", "2026-02-01", settle=True, request_id="r-1"))
        again = json.loads(await _remember(a, "coffee", "2026-02-01", settle=True, request_id="r-1"))
        assert first.get("settle") == {"settled": "pref-0001"}, first
        assert "refused" not in again and again["memo_id"] == first["memo_id"], again
        assert again.get("receipt", {}).get("replayed") is True and again.get("settle") == {"settled": "pref-0001"}, again

    _serve_both(settle_envs, body)


SISTER = {"subject": "sister", "predicate": "city", "object": "Lyon"}


def test_settle_declines_a_source_that_is_not_the_memo_file_of_its_stem(
    settle_envs: tuple[dict[str, str], dict[str, str]], tmp_path: Path
) -> None:
    """S6. The older statement of the key is `memos/pref-0001.markdown`, beside the assistant's own
    `pref-0001.md` about another key: settle declines `not_assistant_memo` and retires nothing. Its
    failure: the source's suffix stripped into `pref-0001`, so the edge retired the `.md` memo,
    which never stated the key.

    Red proof (CCA run 20261125T120125Z, SEC-001), on the remote test host: against the pre-fix
    `recall_mcp.assistant.AssistantMemos._settle` (`PurePath(source).with_suffix("")`), the answer's
    `settle` was `{"settled": "pref-0001"}` and the assertion failed.
    """

    async def body(a: Call, p: Call, state: dict[str, Any]) -> None:
        own = json.loads(await a("recall_remember", text="The user's sister lives in Lyon.", kind="pref",
                                 valid_from="2026-01-01", fact=SISTER))
        assert own["memo_id"] == "pref-0001", own
        graph = {"facts": [AtomicFact.from_payload(TEA).to_payload()]}
        (tmp_path / "memos" / "pref-0001.markdown").write_text(
            "---\nvalid_from: 2026-01-01\nrecall_graph: " + json.dumps(graph) + "\n---\nThe user drink tea now.\n",
            encoding="utf-8")
        await a("recall_remember", text="The user went to Rome.", kind="note")  # its refresh indexes the .markdown file
        assert "tea" in _values(await a("recall_current_facts", subject="user")), "the .markdown file was not indexed"
        moved = json.loads(await _remember(a, "coffee", "2026-02-01", settle=True))
        assert moved.get("settle") == {"declined": "not_assistant_memo"}, moved
        assert "supersedes" not in (tmp_path / "memos" / f"{moved['memo_id']}.md").read_text(encoding="utf-8")

    _serve_both(settle_envs, body)


def test_settle_declines_a_memo_at_the_index_root_from_a_folder_below_it(
    settle_envs: tuple[dict[str, str], dict[str, str]], tmp_path: Path
) -> None:
    """S7. An assistant memo at the index root, `pref-0001.md`, holds the key, and the assistant
    folder `memos/` has its own `pref-0001` about another key: settle declines `not_assistant_memo`,
    since no reference from `memos/` names the root memo. Its failure: the bare id read as this
    folder's memo, so the edge retired `memos/pref-0001.md` and the root memo stayed in force.

    Red proof (CCA run 20261125T120125Z, BUG-001), on the remote test host: against the pre-fix
    `recall_mcp.assistant.AssistantMemos._settle` (`self.target_of` of the bare stem), the answer's
    `settle` was `{"settled": "pref-0001"}` and the assertion failed.
    """
    root = MemoFolder(tmp_path, index_root=tmp_path)
    assert root.write("The user drink tea now.", kind="pref", valid_from="2026-01-01", fact=TEA)["memo_id"] == "pref-0001"

    async def body(a: Call, p: Call, state: dict[str, Any]) -> None:
        own = json.loads(await a("recall_remember", text="The user's sister lives in Lyon.", kind="pref",
                                 valid_from="2026-01-01", fact=SISTER))
        assert own["memo_id"] == "pref-0001", own
        root.refresh(state["store"], ContentWordEmbedder())  # the root folder's own refresh; this server's indexes memos/ only
        assert "tea" in _values(await a("recall_current_facts", subject="user")), "the root memo was not indexed"
        moved = json.loads(await _remember(a, "coffee", "2026-02-01", settle=True))
        assert moved.get("settle") == {"declined": "not_assistant_memo"}, moved
        assert "supersedes" not in (tmp_path / "memos" / f"{moved['memo_id']}.md").read_text(encoding="utf-8")

    _serve_both(settle_envs, body)


def test_a_settle_the_listing_cannot_change_does_not_read_it(
    settle_envs: tuple[dict[str, str], dict[str, str]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S8. A settle of a predicate the table calls set-valued is declined without reading the
    current facts, which `_settle` reads under the store's write lane. Its failure: the whole
    ledger projected, and other writers held, for a write declined whatever it says.

    Red proof (CCA run 20261125T120125Z, PERF-001), on the remote test host: against the pre-fix
    `recall_mcp.assistant.AssistantMemos._settle`, which read the listing before `settle_target`'s
    first check, the counted reader was called once and the assertion on `reads` failed.
    """

    reads: list[object] = []
    real = recall_mcp.provenance.current_facts_memory

    def counted(*args: Any, **kwargs: Any) -> Any:
        reads.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(recall_mcp.provenance, "current_facts_memory", counted)

    async def body(a: Call, p: Call, state: dict[str, Any]) -> None:
        answer = json.loads(await a(
            "recall_remember", text="The user's email address is a@example.org.", kind="fact", valid_from="2026-01-01",
            fact={"subject": "user", "predicate": "email", "object": "a@example.org"}, settle=True))
        assert answer.get("settle") == {"declined": "not_single_in_table"}, answer
        assert reads == [], f"the listing was read {len(reads)} time(s) for a settle it cannot change"

    _serve_both(settle_envs, body)


def test_settle_is_declined_where_a_source_policy_filters_the_listing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S9. With a source security policy configured, a settle is declined `unchecked` before any
    listing is read: the listing the caller may read can miss a memo on the key that the policy
    hides, so the count of earlier holders would be wrong, and counting over the whole store would
    let the decline name a memo the caller may not read (owner's decision on CCA STAKES-001,
    2026-11-25).

    Red proof (CCA run 20261125T120125Z, STAKES-001), on the remote test host: against
    `recall_mcp.assistant.AssistantMemos._settle` before this change, the listing was read once and
    the answer was `{"settled": None}`, failing the first assertion.
    """
    reads: list[object] = []

    def empty(*args: Any, **kwargs: Any) -> Any:
        reads.append(kwargs)
        return {"facts": [], "contested": [], "served_state": "ok", "served_lookups_failed": []}

    monkeypatch.setattr(recall_mcp.provenance, "current_facts_memory", empty)
    memos = AssistantMemos(tmp_path / "memos", index_root=tmp_path)
    (tmp_path / "memos").mkdir()
    answer = memos._settle(
        None,  # type: ignore[arg-type]  # the store is never reached under a policy
        fact={**TEA, "object": "coffee"}, valid_from="2026-02-01", supersedes=None, request_id=None,
        access_context=None, read_facts=True, security_policy=object(),
    )
    assert answer == (None, {"declined": "unchecked"}), answer
    assert reads == [], "the listing was read under a source policy"


def test_settle_declines_a_target_whose_successor_is_written_but_not_indexed(
    settle_envs: tuple[dict[str, str], dict[str, str]], tmp_path: Path
) -> None:
    """S10. The rule's older memo already has a successor on disk that no refresh has indexed yet:
    settle declines `target_has_successor` and the memo is written without an edge. Its failure:
    the settle passed the target to `write`, whose folder scan refused the whole call
    `stale_target`, for a target the caller never named (CCA BUG-002).

    Red proof (CCA run 20261125T132233Z, BUG-002), on the remote test host: against
    `recall_mcp.assistant.AssistantMemos._settle` before this change, the answer was the refusal
    `stale_target` with no `settle` note, and the assertion on `moved["settle"]` failed.
    """
    memos = AssistantMemos(tmp_path / "memos", index_root=tmp_path)

    async def body(a: Call, p: Call, state: dict[str, Any]) -> None:
        await _remember(a, "tea", "2026-01-01")
        # Written, not indexed: the server's next refresh has not run yet.
        written = memos.write("The user's tea note is corrected.", kind="note", supersedes="pref-0001")
        assert "memo_id" in written, written
        moved = json.loads(await _remember(a, "coffee", "2026-02-01", settle=True))
        assert moved.get("settle") == {"declined": "target_has_successor"}, moved
        assert "supersedes" not in (tmp_path / "memos" / f"{moved['memo_id']}.md").read_text(encoding="utf-8")

    _serve_both(settle_envs, body)


def test_a_retry_answers_what_the_first_settle_did(settle_envs: tuple[dict[str, str], dict[str, str]]) -> None:
    """S11. A retry of a settling write whose first call declined answers the same decline: the
    note is journalled with the request. Its failure: the retry inferred the note from the
    journalled `supersedes`, and a declined first call replayed as `{"settled": None}`, "nothing to
    settle" (CCA STAKES-003).

    Red proof (CCA run 20261125T132233Z, STAKES-003), on the remote test host: against
    `recall_mcp.assistant.AssistantMemos._settle` and `recall.lite.memo_requests.write_memo` before
    this change, the retry's `settle` was `{"settled": None}` and the second assertion failed.
    """

    async def body(a: Call, p: Call, state: dict[str, Any]) -> None:
        await _remember(a, "tea", "2026-01-01")
        await _remember(a, "milk", "2026-01-01")
        first = json.loads(await _remember(a, "water", "2026-02-01", settle=True, request_id="r-2"))
        assert first.get("settle") == {"declined": "several_older_memos"}, first
        again = json.loads(await _remember(a, "water", "2026-02-01", settle=True, request_id="r-2"))
        assert again.get("settle") == {"declined": "several_older_memos"}, again
        assert again["memo_id"] == first["memo_id"] and again.get("receipt", {}).get("replayed") is True, again

    _serve_both(settle_envs, body)


def test_a_settled_write_checks_its_target_once(
    settle_envs: tuple[dict[str, str], dict[str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """S12. A write that settles walks the index root for successor folders once, in `write`: the
    write lane is held for the walk and for the scan of every memo folder it feeds. Its failure: a
    pre-check in `_settle` repeated the walk and the scan that `write_memo` then made again for the
    same target, in the same lane (CCA fix audit FA-1-7, run 20261125T132233Z).

    Red proof (CCA run 20261125T132233Z, FA-1-7), on the remote test host: against the pass-1
    snapshot of `recall_mcp.assistant.AssistantMemos._settle`, which called
    `MemoFolder.successor_conflict`, the walk ran twice and the assertion on `walks` failed.
    """
    import recall.lite.assistant_memos as memo_folder

    walks: list[object] = []
    real = memo_folder._journal_ledgers

    def counted(root: Path) -> list[Path]:
        walks.append(root)
        return real(root)

    monkeypatch.setattr(memo_folder, "_journal_ledgers", counted)

    async def body(a: Call, p: Call, state: dict[str, Any]) -> None:
        await _remember(a, "tea", "2026-01-01")
        walks.clear()
        moved = json.loads(await _remember(a, "coffee", "2026-02-01", settle=True))
        assert moved.get("settle") == {"settled": "pref-0001"}, moved
        assert len(walks) == 1, f"the index root was walked {len(walks)} times for one settled write"

    _serve_both(settle_envs, body)
