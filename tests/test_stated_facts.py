"""A fact a memo states in its `recall_graph` frontmatter reaches a lite store's fact ledger.

Before this, the only way such a fact became current was `recall_apply_fact` citing evidence
cards from a trusted search that carried exactly that fact, a chain agents were measured not to
complete. The store now records the fact itself after the write that indexed the memo
(`recall.stated_facts.reconcile_stated_facts`), with the memo's own chunk as its evidence; what is
current is decided at read time from what the store serves.

Invariants and the failure each one catches: in each test's docstring, with its red proof.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from recall.lite import LiteStore
from recall.lite.facts import LiteFactLedger
from recall.lite.refresh import Outcome, RefreshReport, refresh
from recall_mcp.provenance import current_facts_memory
from tests.test_lite_calibration import DIM, ContentWordEmbedder, _write_memos

DOOR = {"subject": "user", "predicate": "door_code", "object": "4512"}
FUTURE = datetime(2099, 1, 2, tzinfo=UTC)


@pytest.fixture
def folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("RECALL_EMBED_CACHE", "off")
    monkeypatch.delenv("RECALL_STATED_FACT_SOURCES", raising=False)
    _write_memos(tmp_path / "memory", 4)  # background, so the refresh can calibrate
    return tmp_path / "memory"


@pytest.fixture
def store(tmp_path: Path) -> Iterator[LiteStore]:
    with LiteStore(tmp_path / ".recall" / "memory.db", dim=DIM) as opened:
        yield opened


def memo(folder: Path, name: str, *, facts: list[dict[str, object]] | None = None,
         graph: str | None = None, body: str = "A note about the front door.", **front: str) -> Path:
    """Write memo `name` into `folder`, with `facts` (or a raw `graph` line) and frontmatter `front`."""
    lines = [f"{key}: {value}" for key, value in front.items()]
    if graph is not None:
        lines.append("recall_graph: " + graph)
    elif facts is not None:
        lines.append("recall_graph: " + json.dumps({"facts": facts}))
    head = "---\n" + "\n".join(lines) + "\n---\n" if lines else ""
    path = folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"{head}# {name}\n\n{body}\n".encode())
    return path


def refreshed(store: LiteStore, *roots: Path, key_root: Path | None = None) -> RefreshReport:
    report = refresh(store, list(roots), ContentWordEmbedder(), force=True, key_root=key_root)  # type: ignore[arg-type]
    assert report.outcome is Outcome.REFRESHED, report.to_dict()
    return report


def listed(store: LiteStore, as_of: datetime | None = None) -> list[str]:
    """The objects `recall_current_facts` lists under `facts`, sorted."""
    return sorted(str(item["fact"]["object"]) for item in current_facts_memory(store, as_of=as_of)["facts"])  # type: ignore[index, union-attr]


def reconcile(store: LiteStore, *roots: Path) -> Any:
    """`reconcile_stated_facts` with `roots`, or with the refresh state's roots when none is given."""
    from recall.stated_facts import reconcile_stated_facts

    return reconcile_stated_facts(store, roots=list(roots) or None)


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A RECALL_INDEX_ROOT with background memos in `memory/`, as a lite server lays it out."""
    path = tmp_path / "root"
    monkeypatch.setenv("RECALL_INDEX_ROOT", str(path))
    monkeypatch.setenv("RECALL_EMBED_CACHE", "off")
    monkeypatch.delenv("RECALL_STATED_FACT_SOURCES", raising=False)
    path.mkdir()
    _write_memos(path / "memory", 4)
    return path


def test_s3a_an_upload_never_states(root: Path, store: LiteStore) -> None:
    """Node tests/test_stated_facts.py::test_s3a_an_upload_never_states.

    Spec test 3 and review focus 5: an upload staged under RECALL_INDEX_ROOT/uploads is inside the
    refreshed folder and still does not state. Red, mutation of `StatingRule.admits`
    (recall/stated_facts.py): delete its `if self.uploads is not None ...: return False` lines.
    Fails on "an upload stated a fact".
    """
    memo(root / "memory", "door.md", facts=[DOOR])
    memo(root / "uploads" / store.tenant / "job1", "upload.md", facts=[{**DOOR, "object": "6666"}])
    refreshed(store, root, key_root=root)
    reconcile(store, root)
    assert listed(store) == ["4512"], f"an upload stated a fact: {listed(store)}"


def test_s3b_a_memo_outside_the_refreshed_folders_does_not_state(root: Path, store: LiteStore) -> None:
    """Node tests/test_stated_facts.py::test_s3b_a_memo_outside_the_refreshed_folders_does_not_state.

    Spec test 3 and review focus 5: `docs/` is under the key root but not refreshed, and is indexed
    by an indexer that records nothing; the reconcile reads the refresh state's roots. Red,
    mutation of `StatingRule.admits`: `return True` as its first line. Fails on "a memo outside
    the refreshed folders stated a fact".
    """
    from typing import cast

    from recall.context import context_policy_for_profile
    from recall.embeddings import embedding_profile_id
    from recall.index import Indexer, chunk_text
    from recall.store import PgVectorStore

    memo(root / "memory", "door.md", facts=[DOOR])
    memo(root / "docs", "doc.md", facts=[{**DOOR, "predicate": "gate_code", "object": "7777"}])
    refreshed(store, root / "memory", key_root=root)
    embedder = ContentWordEmbedder()
    Indexer(  # built as the refresh builds it, minus the reconcile the entry points add
        cast(PgVectorStore, store), embedder, chunker=chunk_text,  # type: ignore[arg-type]
        context_policy=context_policy_for_profile(embedding_profile_id(embedder)),  # type: ignore[arg-type]
    ).index_path(root / "docs", key_root=root)
    reconcile(store)
    assert listed(store) == ["4512"], f"a memo outside the refreshed folders stated a fact: {listed(store)}"


def test_s3c_the_operators_prefixes_decide(
    store: LiteStore, folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Node tests/test_stated_facts.py::test_s3c_the_operators_prefixes_decide.

    Spec test 3: with RECALL_STATED_FACT_SOURCES set, only memos under those prefixes state, on a
    path boundary. Red, mutation of `StatingRule.admits`: drop the `+ "/"` so `notes` also admits
    `notes2/`. Fails on "a memo outside the named prefix stated a fact".
    """
    monkeypatch.setenv("RECALL_STATED_FACT_SOURCES", "notes")
    memo(folder / "notes", "gate.md", facts=[{**DOOR, "predicate": "gate_code", "object": "8888"}])
    memo(folder / "notes2", "door.md", facts=[DOOR])
    refreshed(store, folder)
    reconcile(store, folder)
    assert listed(store) == ["8888"], f"a memo outside the named prefix stated a fact: {listed(store)}"


def test_s9_reconciling_an_unchanged_store_appends_nothing(store: LiteStore, folder: Path) -> None:
    """Node tests/test_stated_facts.py::test_s9_reconciling_an_unchanged_store_appends_nothing.

    Spec test 9. Red, mutation of `stated_request_id` (recall/stated_facts.py): add
    `uuid.uuid4().hex` to `key` (import uuid). Fails on "a second reconcile appended events".
    (Skipping the `known` filter is NOT a valid mutation here: the ledger's own request check
    returns the prior event and appends nothing, so that mutation stays green.)
    """
    memo(folder, "door.md", facts=[DOOR])
    refreshed(store, folder)
    first = reconcile(store, folder)
    before = len(LiteFactLedger(store).events)
    second = reconcile(store, folder)
    after = len(LiteFactLedger(store).events)
    assert after == before, f"a second reconcile appended events: {before} -> {after}"
    assert second.facts_stated == 0 and first.enabled


def test_s12_none_and_a_store_that_is_not_lite_state_nothing(
    store: LiteStore, folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Node tests/test_stated_facts.py::test_s12_none_and_a_store_that_is_not_lite_state_nothing.

    Spec test 12. The memo sits under a folder named `none`, so `none` read as a prefix would
    admit it: that is what tells the keyword from a prefix. Red, mutation of `stating_rule`
    (recall/stated_facts.py): delete its `if raw.lower() == "none":` branch. Fails on
    "RECALL_STATED_FACT_SOURCES=none still stated a fact".
    """
    from recall.stated_facts import stating_rule

    monkeypatch.setenv("RECALL_STATED_FACT_SOURCES", "none")
    memo(folder / "none", "door.md", facts=[DOOR])
    refreshed(store, folder)
    reconcile(store, folder)
    assert listed(store) == [], f"RECALL_STATED_FACT_SOURCES=none still stated a fact: {listed(store)}"
    assert isinstance(stating_rule(object(), roots=[folder]), str)


def test_s8_a_memo_valid_from_the_future_is_not_current_yet(store: LiteStore, folder: Path) -> None:
    """Node tests/test_stated_facts.py::test_s8_a_memo_valid_from_the_future_is_not_current_yet.

    Spec test 8. Red, mutation of `memo_facts` (recall/stated_facts.py): drop the `replace(...)`
    that gives a fact its memo's window. Fails on "a fact whose memo starts in 2099 is current
    now".
    """
    memo(folder, "door.md", facts=[DOOR], valid_from="2099-01-01")
    refreshed(store, folder)
    reconcile(store, folder)
    assert listed(store) == [], f"a fact whose memo starts in 2099 is current now: {listed(store)}"
    assert listed(store, as_of=FUTURE) == ["4512"], "the fact did not become current when its memo did"


def test_rf1_malformed_graph_json_is_counted_and_spares_the_rest(store: LiteStore, folder: Path) -> None:
    """Node tests/test_stated_facts.py::test_rf1_malformed_graph_json_is_counted_and_spares_the_rest.

    Review focus 1. Red, mutation of `memo_facts`: return `((), 0)` for a `__parse_error__` graph.
    Fails on "a malformed recall_graph was not counted: 0".
    """
    memo(folder, "door.md", facts=[DOOR])
    memo(folder, "broken.md", graph='{"facts": [')
    refreshed(store, folder)
    report = reconcile(store, folder)
    assert report.facts_malformed == 1, f"a malformed recall_graph was not counted: {report.facts_malformed}"
    assert listed(store) == ["4512"]


def test_rf2_a_nan_object_is_counted_not_raised(store: LiteStore, folder: Path) -> None:
    """Node tests/test_stated_facts.py::test_rf2_a_nan_object_is_counted_not_raised.

    Review focus 2: `json.loads` accepts NaN, canonical JSON refuses it. The memo also states a
    good fact, which must survive its neighbour. Red (2026-10-15), mutation of `memo_facts`:
    replace `identity = fact_identity(fact)` with `identity = repr(fact)`, so the probe no longer
    runs. The NaN then reaches `stated_request_id`, whose canonical JSON raises, the whole
    reconcile reports `facts_stated=-1` with nothing counted, and the test fails on "a NaN fact
    was not counted malformed alone" (0 == 1).
    """
    memo(folder, "door.md", facts=[DOOR])
    memo(folder, "lucky.md", graph='{"facts": [{"subject": "user", "predicate": "lucky", "object": NaN}, '
                                   '{"subject": "user", "predicate": "lucky_number", "object": 7}]}')
    refreshed(store, folder)
    report = reconcile(store, folder)
    assert report.facts_malformed == 1, f"a NaN fact was not counted malformed alone: {report}"
    assert listed(store) == ["4512", "7"], f"a NaN fact took its memo's good fact with it: {listed(store)}"


def test_rf3_a_fact_window_against_the_memo_window_is_malformed(store: LiteStore, folder: Path) -> None:
    """Node tests/test_stated_facts.py::test_rf3_a_fact_window_against_the_memo_window_is_malformed.

    Review focus 3. Red, mutation of `memo_facts`: catch only `KeyError` (not `ValueError`) around
    the per-fact parse. Fails on "a fact ending before its memo starts was not counted malformed".
    """
    early = {**DOOR, "object": "1111", "valid_until": "2025-01-01T00:00:00+00:00"}
    memo(folder, "door.md", facts=[early], valid_from="2026-01-01")
    refreshed(store, folder)
    report = reconcile(store, folder)
    assert report.facts_malformed == 1, "a fact ending before its memo starts was not counted malformed"


def test_s4_an_edit_hides_what_the_old_text_stated_and_a_revert_restores_it(store: LiteStore, folder: Path) -> None:
    """Node tests/test_stated_facts.py::test_s4_an_edit_hides_what_the_old_text_stated_and_a_revert_restores_it.

    Spec test 4. Red against `ad9554273` plus Tasks 1 to 3 without this task: fails on "the edited
    memo still lists its old value". Red, mutation of `served_sources`
    (recall/fact_retraction.py): return the plain `served` callable for a lite store too: the same
    failure. The listed values in the message differ between the two runs (both values before
    Task 5, none once Task 5 contests them); the assertion line is what is recorded.
    """
    original = memo(folder, "door.md", facts=[DOOR])
    text = original.read_bytes()
    refreshed(store, folder)
    reconcile(store, folder)
    memo(folder, "door.md", facts=[{**DOOR, "object": "9001"}])
    refreshed(store, folder)
    reconcile(store, folder)
    assert listed(store) == ["9001"], f"the edited memo still lists its old value: {listed(store)}"
    before = len(LiteFactLedger(store).events)
    original.write_bytes(text)
    refreshed(store, folder)  # once Task 6 lands, this refresh reconciles too
    reconcile(store, folder)
    assert len(LiteFactLedger(store).events) == before, "the revert appended an event its request id already held"
    assert listed(store) == ["4512"], f"the revert did not restore the old value: {listed(store)}"


def test_s5_a_superseded_memo_states_nothing_until_its_successor_goes(store: LiteStore, folder: Path) -> None:
    """Node tests/test_stated_facts.py::test_s5_a_superseded_memo_states_nothing_until_its_successor_goes.

    Spec test 5: memo-level, so a successor stating no fact retires the old memo's fact too, and
    forgetting the successor brings it back. Red, mutation of `served_sources`: pass
    `superseded=None` for a lite store. Fails on "a superseded memo's fact is still listed".
    """
    from recall_mcp.lifecycle import forget_memory

    memo(folder, "door-a.md", facts=[DOOR])
    refreshed(store, folder)
    reconcile(store, folder)
    successor = memo(folder, "note-b.md", body="The front door was rekeyed.", supersedes="door-a.md")
    refreshed(store, folder)
    reconcile(store, folder)
    assert listed(store) == [], f"a superseded memo's fact is still listed: {listed(store)}"
    forget_memory(store, ["note-b.md"])  # type: ignore[arg-type]
    successor.unlink()
    assert listed(store) == ["4512"], f"forgetting the successor did not restore the fact: {listed(store)}"
