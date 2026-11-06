"""Phase 4 of resolving `supersedes:` references at write time: the agent asks before it writes.

`recall_resolve_reference` answers which memo a reference names, by search's own rule
(`ReferenceIndex.match`), from the index the store's supersession graph was resolved with; the
`recall_index` receipt lists every stored reference that names more than one memo.

Red proof, 2026-11-06, each failing in its own assertion (recorded in the pull request), since
the pre-change code has no resolver to run against:
- the graph not keeping its index (`_resolve_rows` building the graph without `index`): every
  lookup "dangling" ("the lookup found no memo it should have");
- `_readable_keys` not narrowing to the caller's sources: the unreadable namesake listed
  ("a memo the caller cannot read was named");
- `resolve_served_reference` choosing the form under a policy by counting every memo:
  "the advice under a policy depends on memos the caller cannot see";
- `ambiguous_references` returning nothing: "an ambiguous stored reference went unreported";
- `ambiguous_references` not requiring every named memo to be readable: "an unreadable memo was
  revealed";
- `index_memory` letting the receipt raise: "a committed index was reported as failed".
"""

from __future__ import annotations

from dataclasses import dataclass

from recall.supersession import resolve_supersession_graph
from recall_mcp.lifecycle import ambiguous_references, resolve_served_reference


def _store(rows: list[tuple[object, ...]]) -> object:
    graph = resolve_supersession_graph(rows)

    class Store:
        tenant = "t"

        def supersession_graph(self) -> object:
            return graph

    return Store()


ROWS = [
    ("a/notes.md", None, None, "file:///m/a/notes.md"),
    ("b/notes.md", None, None, "file:///m/b/notes.md"),
    ("old.md", None, None, "file:///m/old.md"),
    ("new.md", "notes", None, "file:///m/new.md"),
    ("c/solo.md", None, None, "file:///m/c/solo.md"),
]


def test_the_lookup_answers_as_search_resolves() -> None:
    store = _store(ROWS)
    shared = resolve_served_reference(store, "notes", written_in="new.md")
    assert shared.status == "ambiguous", f"the lookup found no memo it should have: {shared.status}"
    assert [c.file for c in shared.candidates] == ["a/notes.md", "b/notes.md"]
    by_path = resolve_served_reference(store, "a/notes.md", written_in="new.md")
    assert (by_path.status, by_path.write) == ("resolved", "a/notes.md")
    unique = resolve_served_reference(store, "old", written_in="new.md")
    assert (unique.status, unique.write) == ("resolved", "old.md"), "a unique name was not offered bare"
    assert resolve_served_reference(store, "ghost").status == "dangling"
    assert resolve_served_reference(store, "new", written_in="new.md").status == "self"


@dataclass
class _Decision:
    allowed: bool


class _Policy:
    def decide(self, source: str, context: object) -> _Decision:
        return _Decision(allowed="/b/" not in source)


@dataclass
class _Context:
    tenant: str = "t"


def test_the_lookup_names_only_memos_the_caller_may_read() -> None:
    found = resolve_served_reference(
        _store(ROWS), "notes", written_in="new.md", security_policy=_Policy(), access_context=_Context()  # type: ignore[arg-type]
    )
    assert [c.file for c in found.candidates] == ["a/notes.md"], "a memo the caller cannot read was named"
    assert found.status == "resolved"
    # Search sees `b/notes.md` too, so a bare `notes` would resolve to no memo there.
    assert found.write == "a/notes.md", f"a bare name was advised that search resolves to no memo: {found.write}"
    # A memo no other shares is advised in the same form: a bare name here, and the folder only
    # where an unreadable namesake exists, would say that one does.
    solo = resolve_served_reference(
        _store(ROWS), "solo", written_in="new.md", security_policy=_Policy(), access_context=_Context()  # type: ignore[arg-type]
    )
    assert solo.write == "c/solo.md", f"the advice under a policy depends on memos the caller cannot see: {solo.write}"


def test_the_index_receipt_does_not_say_an_unreadable_namesake_exists() -> None:
    """`new.md`'s `notes` is ambiguous only because of `b/notes.md`, which the caller cannot read.
    Red proof: listing it without asking whether every memo it names is readable ("an unreadable
    memo was revealed")."""
    found = ambiguous_references(_store(ROWS), security_policy=_Policy(), access_context=_Context())  # type: ignore[arg-type]
    assert found == [], f"an unreadable memo was revealed: {found}"


def test_the_index_receipt_lists_a_reference_that_names_two_memos() -> None:
    found = ambiguous_references(_store(ROWS))
    assert [(a.memo, a.reference) for a in found] == [("new.md", "notes")], "an ambiguous stored reference went unreported"
    assert ambiguous_references(_store(ROWS[:3])) == []


def test_the_index_receipt_cannot_fail_a_committed_index(tmp_path, monkeypatch) -> None:
    """The receipt runs after the index has committed; a failure to check is said, not raised."""
    import recall_mcp.lifecycle as lifecycle
    from recall.lite import LiteStore
    from recall_mcp.indexing import index_memory
    from tests.test_lite_calibration import DIM, ContentWordEmbedder, _write_memos

    def _broken(*args: object, **kwargs: object) -> object:
        raise RuntimeError("the graph scan failed")

    monkeypatch.setattr(lifecycle, "ambiguous_references", _broken)
    monkeypatch.setenv("RECALL_EMBED_CACHE", "off")
    _write_memos(tmp_path / "memory", 4)
    with LiteStore(tmp_path / ".recall" / "memory.db", dim=DIM) as store:
        try:
            result = index_memory(store, ContentWordEmbedder(), str(tmp_path / "memory"), env={"RECALL_INDEX_ROOT": str(tmp_path)})  # type: ignore[arg-type]
        except RuntimeError as exc:
            raise AssertionError(f"a committed index was reported as failed: {exc}") from exc
        assert result.chunks == 4
        assert "were not checked (RuntimeError)" in result.message, result.message
