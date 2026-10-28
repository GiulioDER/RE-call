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
from recall_mcp.lifecycle import forget_memory
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
    (recall/stated_facts.py): delete its `_runs_through_uploads` check. Fails on "an upload
    stated a fact".
    """
    memo(root / "memory", "door.md", facts=[DOOR])
    memo(root / "uploads" / store.tenant / "job1", "upload.md", facts=[{**DOOR, "predicate": "wifi_password", "object": "6666"}])
    refreshed(store, root, key_root=root)
    reconcile(store, root)
    assert listed(store) == ["4512"], f"an upload stated a fact: {listed(store)}"


def test_s3f_an_upload_cannot_contest_a_real_fact(root: Path, store: LiteStore) -> None:
    """Node tests/test_stated_facts.py::test_s3f_an_upload_cannot_contest_a_real_fact.

    Security review follow-up, 2026-10-15: an upload stating ANOTHER value for a real fact's key
    would not show up as a new fact at all, it would move the real one to `contested` and so
    erase the answer. Red, mutation of `StatingRule.admits` (recall/stated_facts.py): delete its
    `_runs_through_uploads` check. Fails on "an upload contested a real fact".
    """
    memo(root / "memory", "door.md", facts=[DOOR])
    memo(root / "uploads" / store.tenant / "job1", "upload.md", facts=[{**DOOR, "object": "6666"}])
    refreshed(store, root, key_root=root)
    reconcile(store, root)
    out = current_facts_memory(store)
    assert listed(store) == ["4512"] and out["contested"] == [], f"an upload contested a real fact: {out['contested']}"


def test_s3d_an_upload_stays_out_when_the_refresh_runs_without_the_server_env(
    root: Path, store: LiteStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Node tests/test_stated_facts.py::test_s3d_an_upload_stays_out_when_the_refresh_runs_without_the_server_env.

    Security review finding (authorization bypass), 2026-10-15: the uploads root came from the
    PROCESS's RECALL_INDEX_ROOT, so a `recall lite refresh` run by a person without the server's
    environment did not recognise `uploads/` and stated an upload's facts. The store's own names
    are key-rooted, so `uploads/<tenant>/` is an upload whatever the environment says. Red against
    89af30d8b: fails on "an upload stated a fact without the server env: ['4512', '6666']". Red,
    mutation of `StatingRule.admits`: delete its `_runs_through_uploads` check: the same failure.
    """
    memo(root / "memory", "door.md", facts=[DOOR])
    memo(root / "uploads" / store.tenant / "job1", "upload.md", facts=[{**DOOR, "predicate": "wifi_password", "object": "6666"}])
    monkeypatch.delenv("RECALL_INDEX_ROOT")
    refreshed(store, root, key_root=root)
    reconcile(store, root)
    assert listed(store) == ["4512"], f"an upload stated a fact without the server env: {listed(store)}"


def test_s3e_the_env_a_caller_passes_also_names_the_uploads(
    root: Path, store: LiteStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Node tests/test_stated_facts.py::test_s3e_the_env_a_caller_passes_also_names_the_uploads.

    Security review finding (parser differential), 2026-10-15: `stating_rule` read the prefixes
    from the `env` a caller passes (`recall_index` passes its runtime env) but the uploads root
    from the process environment. Here the uploads folder is refreshed as a root of its own, so
    its names carry no `uploads/` prefix and only the absolute path can tell, and since the lite
    refresh records facts itself (Task 6) that refresh reconciles with no RECALL_INDEX_ROOT at all:
    an env-derived uploads folder cannot exist there, which is why the check is structural. Red
    against 89af30d8b: fails on "an upload stated a fact under the caller's env". Red, mutation of
    `StatingRule.admits`: test only the stored name with `_runs_through_uploads`, not the source:
    the same failure.
    """
    from recall.stated_facts import reconcile_stated_facts

    upload = root / "uploads" / store.tenant / "job1"
    memo(upload, "upload.md", facts=[{**DOOR, "object": "6666"}])
    monkeypatch.delenv("RECALL_INDEX_ROOT")
    refreshed(store, upload)
    reconcile_stated_facts(store, roots=[upload], env={"RECALL_INDEX_ROOT": str(root)})
    assert listed(store) == [], f"an upload stated a fact under the caller's env: {listed(store)}"


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


def test_s5w_a_successor_memo_retires_stated_facts_only_inside_its_window(store: LiteStore, folder: Path) -> None:
    """Node tests/test_stated_facts.py::test_s5w_a_successor_memo_retires_stated_facts_only_inside_its_window.

    The window rule on the memo-level path (owner decision, 2026-10-21): a successor memo dated
    from 2099 retires the old memo's stated fact from then, not now. Red on master ``67ffcce59``
    (`served_sources` asked the supersession graph with no instant): fails on "a successor memo
    not yet valid retired a stated fact".
    """
    memo(folder, "door-a.md", facts=[DOOR])
    memo(folder, "note-b.md", body="The front door will be rekeyed.", supersedes="door-a.md", valid_from="2099-01-01")
    refreshed(store, folder)
    reconcile(store, folder)
    assert listed(store) == ["4512"], f"a successor memo not yet valid retired a stated fact: {listed(store)}"
    assert listed(store, as_of=FUTURE) == [], "the successor memo retires the fact inside its window"


def test_s1_a_stated_fact_is_listed_with_its_memo(store: LiteStore, folder: Path) -> None:
    """Node tests/test_stated_facts.py::test_s1_a_stated_fact_is_listed_with_its_memo.

    Spec test 1 (sources half). Red, mutation of `current_facts_memory` (recall_mcp/provenance.py):
    emit `"sources": []`. Fails on "the fact does not name its memo: []".
    """
    memo(folder, "door.md", facts=[DOOR])
    refreshed(store, folder)
    reconcile(store, folder)
    (item,) = current_facts_memory(store)["facts"]  # type: ignore[misc]
    assert item["sources"] == ["door.md"], f"the fact does not name its memo: {item['sources']}"


def test_s6_two_memos_that_disagree_are_contested(store: LiteStore, folder: Path) -> None:
    """Node tests/test_stated_facts.py::test_s6_two_memos_that_disagree_are_contested.

    Spec test 6. Red, mutation of `split_contested` (recall/provenance_controller.py): return
    `(tuple(current), ())`. Fails on "two disagreeing memos were listed as current".
    """
    memo(folder, "door-a.md", facts=[DOOR])
    memo(folder, "door-b.md", facts=[{**DOOR, "object": "9001"}])
    refreshed(store, folder)
    reconcile(store, folder)
    out = current_facts_memory(store)
    assert listed(store) == [], f"two disagreeing memos were listed as current: {listed(store)}"
    (group,) = out["contested"]  # type: ignore[misc]
    assert (group["subject"], group["predicate"]) == ("user", "door_code")
    assert sorted((v["fact"]["object"], tuple(v["sources"])) for v in group["values"]) == [
        ("4512", ("door-a.md",)), ("9001", ("door-b.md",)),
    ]

    # Spec test 6, second half: a model's application of a third value still has to supersede
    # every contested one. It cites a plain memo the store serves, so its card stays live.
    from recall.fact_retraction import served_sources
    from recall.lite.facts import LiteEvidenceCardStore
    from recall.provenance_controller import DecisionCode, FactApplicationRequest, ProvenanceController
    from recall.types import AtomicFact
    from tests.fact_fixtures import evidence_card

    memo(folder, "door-c.md", body="The front door code is now 7777.")
    refreshed(store, folder)
    third = AtomicFact(namespace="memory", subject="user", predicate="door_code", object="7777")
    cards = LiteEvidenceCardStore(store)

    def apply(request_id: str, links: tuple[str, ...]) -> DecisionCode:
        card = evidence_card(third, tenant=store.tenant, generation="g1", chunk=f"c-{request_id}",
                             source="door-c.md", links=links)
        cards.put((card,))
        return ProvenanceController(
            tenant_id=store.tenant, generation_id="g1", cards=cards, ledger=LiteFactLedger(store),
            source_digest_for=lambda _card: "digest", writer="model",
            served_sources=served_sources(store),
        ).apply_fact(FactApplicationRequest(third, (card.card_id,), request_id)).code

    assert apply("m1", ()) == DecisionCode.CONTRADICTION_WITHOUT_SUPERSESSION
    assert apply("m2", ("door-a.md", "door-b.md")) == DecisionCode.APPLIED
    assert listed(store) == ["7777"], f"superseding both contested values did not settle it: {listed(store)}"


def test_s1b_a_refresh_alone_lists_the_stated_fact(store: LiteStore, folder: Path) -> None:
    """Node tests/test_stated_facts.py::test_s1b_a_refresh_alone_lists_the_stated_fact.

    Spec test 1: no model call and no explicit reconcile, only the refresh a lite server runs. Red
    against `ad9554273`: fails on "a refreshed memo's fact is not listed: []". Red, mutation of
    `_refresh_locked` (recall/lite/refresh.py): delete the `reconcile_stated_facts(...)` call: the
    same failure.
    """
    memo(folder, "door.md", facts=[DOOR])
    report = refreshed(store, folder)
    assert listed(store) == ["4512"], f"a refreshed memo's fact is not listed: {listed(store)}"
    assert getattr(report, "facts_stated", None) == 1


def test_s7_forget_retracts_and_a_prune_hides_a_stated_fact(store: LiteStore, folder: Path, tmp_path: Path) -> None:
    """Node tests/test_stated_facts.py::test_s7_forget_retracts_and_a_prune_hides_a_stated_fact.

    Spec test 7 (the lite half; gc is PostgreSQL's, PR B). Red, mutation of `card_liveness`
    (recall/provenance_controller.py): make `live` return True first. Fails on "a pruned memo's
    stated fact is still listed".
    """
    memo(folder, "door.md", facts=[DOOR])
    gate = memo(folder, "gate.md", facts=[{**DOOR, "predicate": "gate_code", "object": "5150"}])
    refreshed(store, folder)
    forget_memory(store, ["door.md"])  # type: ignore[arg-type]
    (folder / "door.md").unlink()
    kept = tmp_path / "gate.md.kept"
    gate.rename(kept)
    refreshed(store, folder)
    assert listed(store) == [], f"a pruned memo's stated fact is still listed: {listed(store)}"
    kept.rename(gate)
    refreshed(store, folder)
    assert listed(store) == ["5150"], "restoring the pruned memo did not bring its fact back"


def test_rf4_a_fact_two_memos_state_survives_forgetting_one(store: LiteStore, folder: Path) -> None:
    """Node tests/test_stated_facts.py::test_rf4_a_fact_two_memos_state_survives_forgetting_one.

    Review focus 4. Red, mutation of `live_sources` (recall/provenance_controller.py): read only
    APPLIED events (`asserts_fact(event)` for `event.event_type == "asserted"`). Fails on "the
    surviving memo is not named: []".
    """
    memo(folder, "door-a.md", facts=[DOOR])
    memo(folder, "door-b.md", facts=[DOOR])
    refreshed(store, folder)
    forget_memory(store, ["door-a.md"])  # type: ignore[arg-type]
    (folder / "door-a.md").unlink()
    (item,) = current_facts_memory(store)["facts"]  # type: ignore[misc]
    assert item["sources"] == ["door-b.md"], f"the surviving memo is not named: {item['sources']}"
    forget_memory(store, ["door-b.md"])  # type: ignore[arg-type]
    (folder / "door-b.md").unlink()
    assert listed(store) == []


def test_s1c_recall_index_and_recall_index_cli_record_stated_facts(
    store: LiteStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Node tests/test_stated_facts.py::test_s1c_recall_index_and_recall_index_cli_record_stated_facts.

    Red, mutation of `_index_memory_unlocked` (recall_mcp/indexing.py): delete the
    `reconcile_stated_facts` lines. Fails on "recall_index did not record the stated fact".
    """
    from recall_mcp.indexing import index_memory

    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setenv("RECALL_EMBED_CACHE", "off")
    _write_memos(root / "memory", 4)
    memo(root / "memory", "door.md", facts=[DOOR])
    # The rule is the process's, as at read time; `env` carries only what recall_index itself needs.
    monkeypatch.setenv("RECALL_STATED_FACT_SOURCES", "memory")
    env = {"RECALL_INDEX_ROOT": str(root)}
    result = index_memory(store, ContentWordEmbedder(), str(root / "memory"), env=env)  # type: ignore[arg-type]
    assert listed(store) == ["4512"], "recall_index did not record the stated fact"
    assert "Stated facts: 1 new" in result.message, result.message


def test_s1d_recall_index_on_a_lite_dsn_prints_the_stated_facts_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Node tests/test_stated_facts.py::test_s1d_recall_index_on_a_lite_dsn_prints_the_stated_facts_line.

    Red, mutation of the `recall index` lite branch (recall/cli_commands/index_search.py): delete
    the `print(reconcile_stated_facts(...).render())` line. Fails on "recall index did not report
    stated facts".
    """
    from tests.test_lite_cli import _dsn, _run

    folder = tmp_path / "mem"
    _write_memos(folder, 4)
    memo(folder, "door.md", facts=[DOOR])
    monkeypatch.setenv("RECALL_EMBED_CACHE", "off")
    monkeypatch.setenv("RECALL_STATED_FACT_SOURCES", "door.md")
    out = _run(["--embedder", "hashing", "--dsn", _dsn(tmp_path), "index", str(folder)])
    assert "stated facts 1 new" in out, f"recall index did not report stated facts: {out}"


def test_s11a_a_failed_append_is_reported_and_the_refresh_still_succeeds(
    store: LiteStore, folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Node tests/test_stated_facts.py::test_s11a_a_failed_append_is_reported_and_the_refresh_still_succeeds.

    Spec 4.6, the lite half of spec test 11. Red, mutation of `_reconcile`
    (recall/stated_facts.py): replace the body of its final `else:` with `continue`. Fails on "a
    failed append was not reported: 0".
    """
    from recall.lite import facts as lite_facts

    def down(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("ledger down")

    monkeypatch.setattr(lite_facts.LiteFactLedger, "apply_assertion", down)
    memo(folder, "door.md", facts=[DOOR])
    report = refreshed(store, folder)
    assert report.facts_stated == -1, f"a failed append was not reported: {report.facts_stated}"
    assert "LEDGER_UNAVAILABLE" in report.stated, report.stated


def test_s11b_an_unexpected_error_never_escapes_the_refresh(
    store: LiteStore, folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Node tests/test_stated_facts.py::test_s11b_an_unexpected_error_never_escapes_the_refresh.

    Spec 4.6: the reconcile never raises into the write that called it. Red, mutation of
    `reconcile_stated_facts` (recall/stated_facts.py): delete its `try:` and `except` lines,
    keeping the body. Fails on "the reconcile's error escaped the refresh".
    """
    from recall import stated_facts

    def broken(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("cannot read the store")

    monkeypatch.setattr(stated_facts, "_admitted_memos", broken)
    memo(folder, "door.md", facts=[DOOR])
    try:
        report = refresh(store, [folder], ContentWordEmbedder(), force=True)  # type: ignore[arg-type]
    except RuntimeError as exc:
        pytest.fail(f"the reconcile's error escaped the refresh: {exc}")
    assert report.outcome is Outcome.REFRESHED and report.facts_stated == -1, report.to_dict()


# Final review fix pass (2026-10-16). Each test was run red against ec98d5fce, the branch before
# the fix, in the assertion named in its docstring.


def test_f1_an_upload_folder_in_another_case_still_never_states(root: Path, store: LiteStore) -> None:
    """Node tests/test_stated_facts.py::test_f1_an_upload_folder_in_another_case_still_never_states.

    Review Important 1: on a case-insensitive filesystem staging writes into an `Uploads/` that
    already exists, and the indexed path carries that case, so a case-sensitive check admitted the
    upload. Red against ec98d5fce: fails on "an upload in another case stated a fact".
    """
    memo(root / "memory", "door.md", facts=[DOOR])
    memo(root / "Uploads" / store.tenant.upper() / "job1", "upload.md",
         facts=[{**DOOR, "predicate": "wifi_password", "object": "6666"}])
    refreshed(store, root, key_root=root)
    reconcile(store, root)
    assert listed(store) == ["4512"], f"an upload in another case stated a fact: {listed(store)}"


def test_f2_turning_stating_off_retires_what_it_recorded(
    store: LiteStore, folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Node tests/test_stated_facts.py::test_f2_turning_stating_off_retires_what_it_recorded.

    Review Important 2: the operator decides which memos state facts, and currency is decided at
    read time, so a rule that no longer admits a memo retires its stated facts, and admitting it
    again brings them back. Red against ec98d5fce: fails on "a fact the operator no longer admits
    is still current".
    """
    memo(folder, "door.md", facts=[DOOR])
    refreshed(store, folder)
    assert listed(store) == ["4512"]
    monkeypatch.setenv("RECALL_STATED_FACT_SOURCES", "none")
    assert listed(store) == [], f"a fact the operator no longer admits is still current: {listed(store)}"
    monkeypatch.delenv("RECALL_STATED_FACT_SOURCES")
    assert listed(store) == ["4512"], "admitting the memo again did not bring its fact back"


def test_f3_an_unchanged_folder_is_backfilled_by_the_next_unforced_refresh(
    store: LiteStore, folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Node tests/test_stated_facts.py::test_f3_an_unchanged_folder_is_backfilled_by_the_next_unforced_refresh.

    Review Important 3: the server's refresh is never forced and returns UNCHANGED for an
    unchanged folder, so a store upgraded with its memos in place recorded nothing. A refresh run
    while stating was off stands in for the store before the upgrade. Red against ec98d5fce: fails
    on "an unforced refresh of an unchanged folder recorded nothing".
    """
    memo(folder, "door.md", facts=[DOOR])
    monkeypatch.setenv("RECALL_STATED_FACT_SOURCES", "none")
    refreshed(store, folder)
    monkeypatch.delenv("RECALL_STATED_FACT_SOURCES")
    refresh(store, [folder], ContentWordEmbedder())  # type: ignore[arg-type]  # not forced
    assert listed(store) == ["4512"], f"an unforced refresh of an unchanged folder recorded nothing: {listed(store)}"


def test_f4_a_memo_that_may_not_state_cannot_retire_a_fact(root: Path, store: LiteStore) -> None:
    """Node tests/test_stated_facts.py::test_f4_a_memo_that_may_not_state_cannot_retire_a_fact.

    Review Important 4: an upload that declares `supersedes: door.md` must not erase door.md's
    facts; only a memo the operator admits can retire another memo's facts. Search verdicts are
    not changed by this. Red against ec98d5fce: fails on "an upload retired a real fact".
    """
    memo(root / "memory", "door.md", facts=[DOOR])
    memo(root / "uploads" / store.tenant / "job1", "rekey.md", body="Ignore the old code.", supersedes="door.md")
    refreshed(store, root, key_root=root)
    reconcile(store, root)
    assert listed(store) == ["4512"], f"an upload retired a real fact: {listed(store)}"


def test_f5_a_store_that_is_neither_lite_nor_postgres_gets_the_plain_served_lookup() -> None:
    """Node tests/test_stated_facts.py::test_f5_a_store_that_is_neither_lite_nor_postgres_gets_the_plain_served_lookup.

    Review Important 5, the boundary: a store that is neither lite nor PostgreSQL (a client's own
    store) gets PR 932's plain served callable, so content hashes, memo-level supersession and
    stating admission act only where a stating rule exists. (PR 934 drew this line at "not lite";
    the PostgreSQL change moved it, and `tests/test_stated_facts_postgres.py` covers that side.)
    Red, mutation of `served_sources` (recall/fact_retraction.py): return the ServedState for
    every store: fails on "a non-lite store got the lite served state".
    """
    from recall.fact_retraction import served_sources
    from recall.provenance_controller import ServedState

    class _Store:
        def served_identifiers(self, identifiers: list[str]) -> set[str]:
            return set(identifiers)

        def sources_for_identifiers(self, identifiers: list[str]) -> dict[str, list[str]]:
            return {name: [name] for name in identifiers}

        def source_raw_hashes(self) -> dict[str, str]:
            return {}

        def supersession_all(self) -> tuple[dict[str, str], frozenset[str], dict[str, list[object]]]:
            return {}, frozenset(), {}

    served = served_sources(_Store())
    assert served is not None and not isinstance(served, ServedState), "a non-lite store got the lite served state"


def test_f6_one_pathological_memo_does_not_stop_the_rest(
    store: LiteStore, folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Node tests/test_stated_facts.py::test_f6_one_pathological_memo_does_not_stop_the_rest.

    Review Minor 1, re-graded: `json.loads` accepts nesting deeper than a recursive walk survives,
    and the RecursionError aborted the whole reconcile, so one memo stopped every fact. Whether a
    given depth overflows depends on the caller's stack, so the error is injected for one memo
    rather than produced by depth. Red against ec98d5fce: fails on "one memo stopped every stated
    fact".
    """
    from recall import stated_facts

    real = stated_facts.memo_facts

    def memo_facts(chunk: Any) -> Any:
        if chunk.source.endswith("deep.md"):
            raise RecursionError("maximum recursion depth exceeded")
        return real(chunk)

    monkeypatch.setattr(stated_facts, "memo_facts", memo_facts)
    memo(folder, "door.md", facts=[DOOR])
    memo(folder, "deep.md", facts=[{**DOOR, "predicate": "deep", "object": "x"}])
    refreshed(store, folder)
    report = reconcile(store, folder)
    assert listed(store) == ["4512"] and report.facts_stated >= 0, f"one memo stopped every stated fact: {report}"
    assert report.facts_malformed >= 1


def test_f7_a_failed_append_logs_a_warning(
    store: LiteStore, folder: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Node tests/test_stated_facts.py::test_f7_a_failed_append_logs_a_warning.

    Review Minor 7, re-graded: spec 4.6 asks for a WARNING, and on a lite server nothing else
    surfaces a failed append. Red against ec98d5fce: fails on "a failed append logged no warning".
    """
    import logging

    from recall.lite import facts as lite_facts

    def down(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("ledger down")

    monkeypatch.setattr(lite_facts.LiteFactLedger, "apply_assertion", down)
    memo(folder, "door.md", facts=[DOOR])
    with caplog.at_level(logging.WARNING):
        refreshed(store, folder)
    assert any("stated facts" in r.getMessage() and r.levelno >= logging.WARNING for r in caplog.records), (
        "a failed append logged no warning"
    )


def test_f8_an_upload_cannot_cancel_an_admitted_supersession(root: Path, store: LiteStore) -> None:
    """Node tests/test_stated_facts.py::test_f8_an_upload_cannot_cancel_an_admitted_supersession.

    Security review of the fix pass, 2026-10-16: `supersession_all()`'s edge map keeps the LAST
    claimant of a target, so an upload that also declares `supersedes: door.md`, and sorts after
    the admitted memo, displaced the admitted claim and kept door.md's old value current. Every
    claimant counts now, and any admitted one retires the target. Red against a545989d3: fails on
    "an upload cancelled an admitted supersession".
    """
    memo(root / "memory", "door.md", facts=[DOOR])
    memo(root / "memory", "rekey.md", body="The front door was rekeyed.", supersedes="door.md")
    memo(root / "uploads" / store.tenant / "job1", "zz.md", body="Keep the old code.", supersedes="door.md")
    refreshed(store, root, key_root=root)
    reconcile(store, root)
    assert listed(store) == [], f"an upload cancelled an admitted supersession: {listed(store)}"


def test_f9_with_stating_off_reading_facts_asks_no_supersession(
    store: LiteStore, folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Node tests/test_stated_facts.py::test_f9_with_stating_off_reading_facts_asks_no_supersession.

    Review of PR B (cost with stating off): memo-level supersession counts only memos the rule
    admits, so with the rule off it can retire nothing, and asking the store's supersession graph
    first was a lookup on every `recall_current_facts` of every default PostgreSQL deployment. Red
    against 0d753ce78: fails on "a supersession lookup ran with stating off: 1".
    """
    memo(folder, "door.md", facts=[DOOR])
    refreshed(store, folder)
    monkeypatch.setenv("RECALL_STATED_FACT_SOURCES", "none")
    calls: list[int] = []
    real = store.supersession_all

    def counted() -> Any:
        calls.append(1)
        return real()

    monkeypatch.setattr(store, "supersession_all", counted)
    current_facts_memory(store)
    assert not calls, f"a supersession lookup ran with stating off: {len(calls)}"


def test_cca946_a_failed_content_lookup_refuses_a_contradicting_apply(
    store: LiteStore, folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Node tests/test_stated_facts.py::test_cca946_a_failed_content_lookup_refuses_a_contradicting_apply.

    CCA DEEP run 20261027T140338Z, BUG-001 (with STAKES-001). A stated fact whose content cannot
    be checked must refuse a contradicting write, not drop out of the conflict set and let it into
    the append-only ledger. Red, mutation of `served_sources` (recall/fact_retraction.py): the
    `strict` branch of `content_hashes` returns {} instead of raising, so the apply is APPLIED and
    `a contradicting apply was admitted during the outage` fails.
    """
    from recall.fact_retraction import served_sources
    from recall.lite.facts import LiteEvidenceCardStore
    from recall.provenance_controller import DecisionCode, FactApplicationRequest, ProvenanceController
    from recall.types import AtomicFact
    from tests.fact_fixtures import evidence_card

    memo(folder, "door-a.md", facts=[DOOR])
    memo(folder, "door-c.md", body="The front door code is now 7777.")
    refreshed(store, folder)
    reconcile(store, folder)
    assert listed(store) == ["4512"]

    def timed_out() -> dict[str, str]:
        raise TimeoutError("injected: the content lookup timed out")

    monkeypatch.setattr(store, "source_raw_hashes", timed_out)
    third = AtomicFact(namespace="memory", subject="user", predicate="door_code", object="7777")
    cards = LiteEvidenceCardStore(store)
    card = evidence_card(third, tenant=store.tenant, generation="g1", chunk="c-outage", source="door-c.md")
    cards.put((card,))
    decision = ProvenanceController(
        tenant_id=store.tenant, generation_id="g1", cards=cards, ledger=LiteFactLedger(store),
        source_digest_for=lambda _card: "digest", writer="model",
        served_sources=served_sources(store, strict=True),
    ).apply_fact(FactApplicationRequest(third, (card.card_id,), "outage-1"))
    assert decision.code == DecisionCode.LEDGER_UNAVAILABLE, (
        f"a contradicting apply was admitted during the outage: {decision.code}")


def test_cca946_recall_apply_fact_checks_served_memos_strictly(
    store: LiteStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Node tests/test_stated_facts.py::test_cca946_recall_apply_fact_checks_served_memos_strictly.

    CCA DEEP run 20261027T140338Z, BUG-001, the wiring. Red against a068a95cb: `apply_fact_memory`
    asks `served_sources(store)` without `strict`, so `the write path checked served memos
    leniently` fails. Target: the `served_sources=` argument in recall_mcp/provenance.py.
    """
    from recall_mcp import provenance

    asked: list[dict[str, object]] = []

    def spy(_store: object, **kwargs: object) -> None:
        asked.append(kwargs)

    class _Stop(Exception):
        pass

    class _Controller:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def apply_fact(self, _request: object) -> object:
            raise _Stop

    monkeypatch.setattr(provenance, "served_sources", spy)
    monkeypatch.setattr(provenance, "ProvenanceController", _Controller)
    with pytest.raises(_Stop):
        provenance.apply_fact_memory(
            store,  # type: ignore[arg-type]
            embedder=ContentWordEmbedder(),  # type: ignore[arg-type]
            claim=DOOR | {"namespace": "memory"},
            evidence_card_ids=["card-1"],
            request_id="wiring-1",
            writer="model",
        )
    assert asked and asked[0].get("strict") is True, f"the write path checked served memos leniently: {asked}"
