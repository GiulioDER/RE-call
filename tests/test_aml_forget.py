"""R2-1: in-conversation forget requests, detected at Add, suppressed at Search, default off.

The examples below follow the templated shapes of PersonaMem-v2's forget turns ("Please forget
that I ...", "Forget that I ...", "... that you remember I ...", "... from your memory") with
invented details; no benchmark text is copied.

Red proofs, run 2026-09-28 with ``PYTHONDONTWRITEBYTECODE=1``, each against one deliberate mutation
of the named production line, each failing at the assertion named, then restored and run green:

- ``test_templated_requests_are_detected_with_their_target``: the ``that\\s+you\\s+(?:remember...)``
  alternative removed from ``forget._PREFIX`` keeps "you remember I" in the target, failing the
  target list equality.
- ``test_resets_and_idioms_are_refused``: the ``if _BLANKET.match(cut): continue`` line in
  ``detect_forget_requests`` deleted lets "Forget all my sports preferences." through as the target
  "all my sports preferences", failing ``== []`` for that phrase.
- ``test_a_negated_request_is_refused``: the ``if _NEGATED.search(sentence): continue`` line
  deleted detects the self-cancelling request, failing ``== []``.
- ``test_task_deletes_need_an_explicit_memory_object``: ``_STRONG_MEMORY_OBJECT`` replaced by
  ``_MEMORY_OBJECT`` in ``detect_forget_requests`` accepts "delete my earlier draft ...", failing
  ``== []``.
- ``test_only_user_turns_of_a_conversation_are_read``: the ``if not _is_user(message): continue``
  line in ``find_forget_requests`` deleted also reads the assistant's turn, failing the ordinal
  list ``== [2]``.
- ``test_a_failing_confirmer_leaves_the_pattern_verdict``: ``confirmed = None`` in the
  ``except`` branch of ``find_forget_requests`` drops the request, failing ``len(...) == 1``.
- ``test_a_retried_add_rewrites_the_same_ledger_row``: ``ledger_id`` returning
  ``"forget_" + uuid.uuid4().hex`` writes a second row on the retry, failing ``len(rows) == 1``.
- ``test_the_ledger_lives_in_its_own_namespace_per_tenant``: ``forget_ledger_store`` returning
  ``self.tenant_store(forget_ledger_tenant("shared"))`` lets tenant B read tenant A's row,
  failing ``forget_requests(tenant_b) == []``.
- ``test_drop_removes_a_window_that_states_the_target_anywhere``: ``TARGET_MATCH_FRACTION``
  lowered from 0.6 to 0.4 also drops the partial mention in another session, failing
  ``"partial" in kept``.
- ``test_drop_removes_the_request_window_even_when_it_is_cut``: the request-shingle clause in
  ``_Text.states`` deleted keeps the cut request window, failing ``"cut-request" not in kept``.
- ``test_drop_removes_the_preceding_exchange_only_in_its_own_session``: the final
  ``return bool(entry.preceding_shingles ...)`` of ``_Text.states`` replaced by ``return False``
  keeps the personalised reply, failing ``"preceding" not in kept``.
- ``test_stub_replaces_only_the_forgotten_sentences``: ``_stub_text`` always returning
  ``STUB_SENTENCE`` for a matched item loses the neutral sentences, failing the exact content
  equality.
- ``test_annotate_keeps_every_item_and_names_the_target_first``: the note built but
  ``content = first.content`` kept in ``annotate_items`` fails ``startswith(ANNOTATION_PREFIX)``.
- ``test_c9_and_every_variant_default_to_off``: ``forget_suppression="drop"`` on C9 fails the
  ``== "off"`` assertion.
- ``test_the_off_path_serves_exactly_what_master_serves``: ``HostedVariant.forget_suppression``
  defaulting to ``"drop"`` (the variant default, which the unset environment falls back to) makes
  the unset-environment body differ from the explicit ``off`` body, failing the byte equality.
- ``test_drop_backfills_to_top_k_on_the_context_route``: skipping ``run.hits[:] = kept`` in
  ``HostedService.search`` serves the forgotten windows, failing the ``not any(DETAIL ...)``
  assertion; applying ``drop_hits`` to the rendered top_k instead (no backfill) fails
  ``len(on.data) == TOP_K``.
- ``test_the_code_route_is_never_filtered``: removing ``and specialist_route != "code"`` from the
  guard in ``HostedService.search`` drops the windows on the code route, failing
  ``forget_items_dropped == 0``.
- ``test_stub_and_annotate_modes_through_search``: skipping the ``stub_items`` call in
  ``HostedService.search`` fails ``STUB_SENTENCE in ...``.
- ``test_a_bad_forget_value_stops_service_startup``: ``self.forget_mode`` removed from the startup
  validation tuple lets ``_service()`` construct, failing with "DID NOT RAISE".
- ``test_a_mode_without_a_ledger_repository_stops_startup``: the repository capability check in
  ``HostedService.__init__`` removed, failing with "DID NOT RAISE".
- ``test_headers_version_and_log_report_the_stage``: the ``X-Recall-Forget-Items-Changed`` header
  removed from ``create_app`` fails its ``.get(...) == "2"`` assertion; ``forget_items_dropped``
  removed from the ``hosted_search_complete`` extra fails the log assertion.
- ``test_postgres_ledger_round_trip_isolation_and_delete``: ``forget_ledger_tenant(tenant)`` removed
  from ``PgHostedRepository.delete_tenant`` fails the final ``count() == 0``.
"""

from __future__ import annotations

import asyncio
import logging
import re

import pytest

from recall.types import Chunk, ScoredChunk
from recall_aml.forget import (
    ANNOTATION_PREFIX,
    FORGET_RECORD_TYPE,
    STUB_SENTENCE,
    DetectedRequest,
    annotate_items,
    detect_forget_requests,
    drop_hits,
    find_forget_requests,
    ledger_chunks,
    ledger_entries,
    stub_items,
)
from recall_aml.identity import forget_ledger_tenant, specialist_tenant, tenant_for
from recall_aml.models import AddRequest, Message, SearchItem, SearchRequest
from recall_aml.retrieval import HostedRetriever
from recall_aml.service import HostedService
from recall_aml.storage import PgHostedRepository
from recall_aml.variants import VARIANTS, variant
from tests.test_aml_atomic_views import (
    _Compiler,
    _Embedder,
    _Repository as _C9Repository,
    _Reranker,
    _Store as _C9Store,
)
from tests.conftest import TEST_DSN, requires_db

C9 = variant("C9_routed_specialists_grounded_graph_atomic")
DETAIL = "fountain"
TOP_K = 5
CONTEXT_QUERY = (
    "Which fountain pens should I buy next? Please recall my related preferences from our "
    "conversation history to give personalized responses."
)
CODE_QUERY = "Fix the fountain pen inventory function in pens.py and run pytest"


# ---------------------------------------------------------------- detection


def test_templated_requests_are_detected_with_their_target() -> None:
    """Every templated shape yields exactly one request, with the detail as its target."""
    cases = {
        "Please forget that I collect vintage fountain pens.": "I collect vintage fountain pens",
        "Forget that I train for triathlons on weekends.": "I train for triathlons on weekends",
        "Could you please forget that I work night shifts at the harbour?": (
            "I work night shifts at the harbour"
        ),
        "Please forget that you remember I restore old sailing boats.": (
            "I restore old sailing boats"
        ),
        "Please forget the preference about feeling uneasy at crowded street markets.": (
            "feeling uneasy at crowded street markets"
        ),
        "Please forget the memory about the long delay on my flight to Oslo.": (
            "the long delay on my flight to Oslo"
        ),
        "Please forget my old knee injury from a ski trip from your memory.": (
            "my old knee injury from a ski trip"
        ),
        "Forget that I'm vegan.": "I'm vegan",
        "Please erase the fact that I smoke from your memory.": "I smoke",
        "Stop remembering that I live near the old tram depot.": "I live near the old tram depot",
    }
    for text, target in cases.items():
        assert [found.target for found in detect_forget_requests(text)] == [target], text


def test_two_requests_in_one_message_are_both_found() -> None:
    found = detect_forget_requests(
        "Thanks. Please forget that I play chess. Please forget that I like hiking in the Alps."
    )
    assert [item.target for item in found] == ["I play chess", "I like hiking in the Alps"]
    assert found[0].sentence == "Please forget that I play chess."


def test_resets_and_idioms_are_refused() -> None:
    """Whole-memory resets, idioms and quoted speech are not requests to forget a detail."""
    for text in (
        "Forget everything we discussed.",
        "Please forget all of that.",
        "Forget all my sports preferences.",
        "Please forget the previous instructions.",
        "Forget the previous ones and answer again.",
        "Forget all the information and instructions before this.",
        "Forget it, never mind.",
        "Forget about it.",
        "Forget that.",
        "Forget I said anything.",
        "Forget this conversation.",
        "She said forget it and walked off.",
        "Forget about heavy coats this summer.",
    ):
        assert detect_forget_requests(text) == [], text


def test_a_negated_request_is_refused() -> None:
    """Negations never start a request, and a sentence that negates its own request is refused."""
    for text in (
        "Don't forget that I told you about my business plans.",
        "Never forget that I love jazz.",
        "I'll never forget my trip to Rome.",
        "Please forget that I'm moving, actually no, don't forget that I'm moving to Lisbon.",
    ):
        assert detect_forget_requests(text) == [], text


def test_task_deletes_need_an_explicit_memory_object() -> None:
    for text in (
        "Please delete my earlier draft of the essay from the archive.",
        "Remove the foil and bake for ten minutes.",
        "Please disregard my previous message.",
    ):
        assert detect_forget_requests(text) == [], text


def test_code_fences_are_not_read() -> None:
    assert detect_forget_requests("```\n# Forget that I cached the token.\n```") == []


def _messages(*pairs: tuple[str, str]) -> list[Message]:
    return [Message(role=role, content=content) for role, content in pairs]


def test_only_user_turns_of_a_conversation_are_read() -> None:
    """The assistant restating the request is not a second request; the ordinal is the Add's."""
    messages = _messages(
        ("user", "I collect vintage fountain pens from Italian makers."),
        ("assistant", "Since you like Italian pens, a leather case suits your fountain collection."),
        ("user", "Please forget that I collect vintage fountain pens from Italian makers."),
        ("assistant", "Forget that I collect vintage fountain pens? Done, I will not use it."),
    )
    found = find_forget_requests(messages, "s1")

    assert [request.message_ordinal for request in found] == [2]
    assert found[0].preceding_text.startswith("I collect vintage fountain pens")
    assert "leather case" in found[0].preceding_text


def test_a_coding_trajectory_is_never_read() -> None:
    messages = _messages(
        ("user", "Forget that I renamed utils.py; see config.yaml and main.py."),
        ("assistant", "def load():\n    return {}  # fixed in loader.py"),
    )
    assert find_forget_requests(messages, "s1") == []


def test_a_failing_confirmer_leaves_the_pattern_verdict() -> None:
    """The optional model check can reject a request, but its failure never means no forgetting."""
    messages = _messages(("user", "Please forget that I collect vintage fountain pens."))

    def broken(_message: str, _request: DetectedRequest) -> DetectedRequest | None:
        raise TimeoutError("model unavailable")

    def rejecting(_message: str, _request: DetectedRequest) -> DetectedRequest | None:
        return None

    assert len(find_forget_requests(messages, "s1", confirm=broken)) == 1
    assert find_forget_requests(messages, "s1", confirm=rejecting) == []


# ---------------------------------------------------------------- the rules, on hits and items


def _entries(session: str = "s1"):
    messages = _messages(
        ("user", "I collect vintage fountain pens from Italian makers."),
        ("assistant", "Since you like Italian pens, a leather case suits your fountain collection."),
        ("user", "Please forget that I collect vintage fountain pens from Italian makers."),
    )
    requests = find_forget_requests(messages, session)
    return ledger_entries(ledger_chunks(tenant_for("u"), requests, created_at="2026-09-28T00:00:00+00:00"))


def _hit(chunk_id: str, text: str, session: str) -> ScoredChunk:
    return ScoredChunk(
        Chunk(
            id=chunk_id,
            source=f"aml://session/{session}",
            text=text,
            metadata={"record_type": "raw", "kind": "raw", "source_session_id": session},
        ),
        0.5,
    )


def _kept(hits: list[ScoredChunk]) -> list[str]:
    kept, _ = drop_hits(hits, _entries())
    return [hit.chunk.id for hit in kept]


def test_drop_removes_a_window_that_states_the_target_anywhere() -> None:
    """Rule R+text reaches every session; three of six stems elsewhere is not the target."""
    kept = _kept(
        [
            _hit("states", "Old note: user collects vintage fountain pens from Italy.", "other"),
            _hit("partial", "Italian fountain pens make a thoughtful gift.", "other"),
            _hit("neutral", "The weather in Lisbon is mild in October.", "other"),
        ]
    )
    assert "states" not in kept
    assert "partial" in kept
    assert "neutral" in kept


def test_drop_removes_the_request_window_even_when_it_is_cut() -> None:
    """A window boundary can cut the request; its shared words still tie it to the request."""
    kept = _kept(
        [_hit("cut-request", "some earlier words here. Please forget that I collect vintage", "s1")]
    )
    assert "cut-request" not in kept


def test_drop_removes_the_preceding_exchange_only_in_its_own_session() -> None:
    """Rule R: the personalised reply goes when it also mentions the target at the lower share."""
    reply = "Since you like Italian pens, a leather case suits your fountain collection."
    kept = _kept(
        [
            _hit("preceding", reply, "s1"),
            _hit("same-words-elsewhere", reply, "s2"),
            _hit("same-session-unrelated", "A leather case is a sensible gift for anyone.", "s1"),
        ]
    )
    assert "preceding" not in kept
    assert "same-words-elsewhere" in kept
    assert "same-session-unrelated" in kept


def _item(item_id: str, content: str, session: str = "s1") -> SearchItem:
    return SearchItem(id=item_id, content=content, source="aml://session/x", session_id=session, kind="raw", score=0.5)


def test_stub_replaces_only_the_forgotten_sentences() -> None:
    items = [
        _item("a", "Good morning. I collect vintage fountain pens from Italian makers. Lunch was fine."),
        _item("b", "The weather in Lisbon is mild in October."),
    ]
    stubbed, applied, changed = stub_items(items, _entries())

    assert [item.id for item in stubbed] == ["a", "b"]
    assert stubbed[0].content == f"Good morning. {STUB_SENTENCE} Lunch was fine."
    assert stubbed[1] is items[1]
    assert changed == 1 and len(applied) == 1


def test_annotate_keeps_every_item_and_names_the_target_first() -> None:
    items = [
        _item("a", "The weather in Lisbon is mild in October."),
        _item("b", "I collect vintage fountain pens from Italian makers."),
    ]
    noted, applied, changed = annotate_items(items, _entries())

    assert [item.id for item in noted] == ["a", "b"]
    assert isinstance(noted[0].content, str)
    assert noted[0].content.startswith(ANNOTATION_PREFIX)
    assert '"I collect vintage fountain pens from Italian makers"' in noted[0].content
    assert noted[0].content.endswith("The weather in Lisbon is mild in October.")
    assert noted[1] is items[1]
    assert changed == 1 and len(applied) == 1


# ---------------------------------------------------------------- ledger


class _MemoryTenantStore:
    def __init__(self, root: "_MemoryBase", tenant: str) -> None:
        self.root = root
        self.tenant = tenant

    def upsert(self, chunks, vectors):
        assert len(chunks) == len(vectors)
        for chunk in chunks:
            self.root.rows.setdefault(self.tenant, {})[chunk.id] = chunk
        return len(chunks)

    def iter_chunks(self, batch_size=256):
        yield from sorted(self.root.rows.get(self.tenant, {}).values(), key=lambda chunk: chunk.id)


class _MemoryBase:
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Chunk]] = {}

    def for_tenant(self, tenant):
        return _MemoryTenantStore(self, tenant)


def test_the_ledger_lives_in_its_own_namespace_per_tenant() -> None:
    """Rows go to the tenant's own ledger namespace, never to its corpus or another tenant."""
    base = _MemoryBase()
    repository = PgHostedRepository(base, _Embedder("code"))  # type: ignore[arg-type]
    tenant_a, tenant_b = tenant_for("forget-a"), tenant_for("forget-b")
    rows = ledger_chunks(
        tenant_a,
        find_forget_requests(_messages(("user", "Please forget that I collect vintage fountain pens.")), "s1"),
        created_at="2026-09-28T00:00:00+00:00",
    )

    repository.persist_forget_requests(tenant_a, rows)

    assert set(base.rows) == {forget_ledger_tenant(tenant_a)}
    assert [row.id for row in repository.forget_requests(tenant_a)] == [rows[0].id]
    assert repository.forget_requests(tenant_b) == []
    assert rows[0].metadata["record_type"] == FORGET_RECORD_TYPE


# ---------------------------------------------------------------- the service, C9 as served


def _rank(row: Chunk) -> tuple[int, str]:
    return (-row.text.count(DETAIL), row.id)


class _Store(_C9Store):
    """The detail's windows rank first on the dense leg; everything else by id."""

    def query_dense_exact(self, vector, k):
        rows = sorted(self._rows(), key=_rank)
        return [ScoredChunk(row, 0.9 - index / 1000) for index, row in enumerate(rows[:k])]

    query_dense = query_dense_exact


class _Repository(_C9Repository):
    def __init__(self) -> None:
        super().__init__()
        self.ledger_writes = 0
        self.fail_next_receipt = False

    def tenant_store(self, tenant):
        return _Store(self, tenant)

    def persist_forget_requests(self, tenant, chunks):
        self.ledger_writes += 1
        return self.persist(forget_ledger_tenant(tenant), chunks)

    def forget_requests(self, tenant):
        return list(self.chunks[forget_ledger_tenant(tenant)].values())

    def record_receipt(self, tenant, request_id, fingerprint, result):
        if self.fail_next_receipt:
            self.fail_next_receipt = False
            raise RuntimeError("database went away after the rows were written")
        super().record_receipt(tenant, request_id, fingerprint, result)


class _NoLedgerRepository(_C9Repository):
    pass


def _service(repository=None) -> tuple[HostedService, _Repository]:
    repository = repository if repository is not None else _Repository()
    service = HostedService(
        repository,  # type: ignore[arg-type]
        _Compiler(),  # type: ignore[arg-type]
        HostedRetriever(_Embedder("code"), _Reranker()),  # type: ignore[arg-type]
        behavior=C9,
        multimodal_embedder=object(),  # type: ignore[arg-type]
        specialist_retrievers={
            C9.context_embedding_profile: HostedRetriever(_Embedder("context"), _Reranker())  # type: ignore[arg-type]
        },
    )
    return service, repository


FILLER = (
    "gardening notes about tomatoes basil and compost",
    "a weekend trip to the coast with friends and a picnic",
    "planning a budget for the new kitchen shelves",
    "a recipe for lentil soup with smoked paprika",
    "morning running routine along the river path",
    "choosing a book club novel for next month",
    "fixing the squeaky bicycle chain before work",
    "learning a few phrases of Portuguese for travel",
)


def _rounds(user: str) -> list[AddRequest]:
    """One Add per round, as the stage-0 replay stored them: filler, the detail, the request."""
    rounds = [
        [
            ("user", f"Tell me more about {topic}. " + " ".join(["please"] * 3)),
            ("assistant", f"Here are some thoughts on {topic}, with a few practical suggestions."),
        ]
        for topic in FILLER
    ]
    rounds.append(
        [
            ("user", "I collect vintage fountain pens from Italian makers; any gift ideas?"),
            ("assistant", "Since you collect vintage fountain pens, a leather fountain pen case is a good gift."),
        ]
    )
    rounds.append(
        [
            ("user", "Please forget that I collect vintage fountain pens from Italian makers."),
            ("assistant", "Understood, I will not use that you collect vintage fountain pens."),
        ]
    )
    return [
        AddRequest(
            request_id=f"{user}-r{index}",
            user_id=user,
            session_id=f"{user}:r{index}",
            messages=_messages(*pairs),
        )
        for index, pairs in enumerate(rounds)
    ]


def _ingest(service: HostedService, user: str) -> None:
    for request in _rounds(user):
        asyncio.run(service.add(request))


def _search(service: HostedService, user: str, query: str = CONTEXT_QUERY, top_k: int = TOP_K):
    return asyncio.run(service.search(SearchRequest(query=query, user_id=user, top_k=top_k)))


def _text(item: SearchItem) -> str:
    return item.content if isinstance(item.content, str) else ""


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch) -> None:
    for name in (
        "RECALL_AML_FORGET",
        "RECALL_ATOMIC_RESCUE_MODE",
        "RECALL_ATOMIC_RESCUE_PLACEMENT",
        "RECALL_ATOMIC_RESCUE_ARTIFACT_ROOT",
        "RECALL_AML_LAST_WINDOW",
    ):
        monkeypatch.delenv(name, raising=False)


def test_c9_and_every_variant_default_to_off() -> None:
    assert C9.forget_suppression == "off"
    assert {item.forget_suppression for item in VARIANTS} == {"off"}


def test_the_off_path_serves_exactly_what_master_serves(monkeypatch) -> None:
    """Unset and ``off`` serve the same bytes, write no ledger row, and never call the stage.

    "What master serves" is established by construction: with every R2-1 function replaced by a
    recorder, nothing is called, so the response is what the code without them computes. The PR
    records the same bytes computed by master's own checkout.
    """
    import recall_aml.service as service_module

    calls: list[str] = []
    for name in ("find_forget_requests", "ledger_entries", "drop_hits", "stub_items", "annotate_items"):
        monkeypatch.setattr(service_module, name, lambda *args, _name=name, **kwargs: calls.append(_name))

    unset_service, unset_repository = _service()
    _ingest(unset_service, "off-user")
    unset = _search(unset_service, "off-user")

    monkeypatch.setenv("RECALL_AML_FORGET", "off")
    off_service, off_repository = _service()
    _ingest(off_service, "off-user")
    off = _search(off_service, "off-user")

    assert unset.model_dump_json() == off.model_dump_json()
    assert any(DETAIL in _text(item) for item in unset.data)
    assert calls == []
    assert unset_repository.ledger_writes == off_repository.ledger_writes == 0
    assert unset.forget_mode == off.forget_mode == "off"


def test_drop_backfills_to_top_k_on_the_context_route(monkeypatch) -> None:
    """Every window stating the detail goes, including the request's, and lower ranks fill in."""
    monkeypatch.setenv("RECALL_AML_FORGET", "off")
    baseline_service, _ = _service()
    _ingest(baseline_service, "drop-user")
    before = _search(baseline_service, "drop-user")
    assert before.specialist_route == "context"
    assert sum(DETAIL in _text(item) for item in before.data) == 2

    monkeypatch.setenv("RECALL_AML_FORGET", "drop")
    service, repository = _service()
    _ingest(service, "drop-user")
    on = _search(service, "drop-user")

    tenant = tenant_for("drop-user")
    assert len(repository.chunks[forget_ledger_tenant(tenant)]) == 1
    for scope in (tenant, specialist_tenant(tenant, C9.context_embedding_profile)):
        assert all(
            row.metadata.get("record_type") != FORGET_RECORD_TYPE
            for row in repository.chunks[scope].values()
        )
    assert not any(DETAIL in _text(item) for item in on.data)
    assert len(on.data) == TOP_K
    assert on.forget_mode == "drop"
    assert on.forget_items_dropped >= 2
    assert on.forget_requests_applied == 1
    kept_before = [item.id for item in before.data if DETAIL not in _text(item)]
    assert [item.id for item in on.data][: len(kept_before)] == kept_before


def test_the_code_route_is_never_filtered(monkeypatch) -> None:
    monkeypatch.setenv("RECALL_AML_FORGET", "drop")
    service, _ = _service()
    _ingest(service, "code-user")
    response = _search(service, "code-user", query=CODE_QUERY)

    assert response.specialist_route == "code"
    assert response.forget_items_dropped == 0
    assert any(DETAIL in _text(item) for item in response.data)


def test_a_forget_in_one_tenant_never_applies_to_another(monkeypatch) -> None:
    monkeypatch.setenv("RECALL_AML_FORGET", "drop")
    service, repository = _service()
    _ingest(service, "tenant-a")
    other = _rounds("tenant-b")[:-1]  # the same detail, never forgotten
    for request in other:
        asyncio.run(service.add(request))

    response = _search(service, "tenant-b")

    assert repository.chunks[forget_ledger_tenant(tenant_for("tenant-b"))] == {}
    assert response.forget_items_dropped == 0
    assert any(DETAIL in _text(item) for item in response.data)


def test_a_retried_add_rewrites_the_same_ledger_row(monkeypatch) -> None:
    """An Add that fails after its rows were written is retried; the ledger still holds one row."""
    monkeypatch.setenv("RECALL_AML_FORGET", "drop")
    service, repository = _service()
    request = _rounds("retry-user")[-1]
    repository.fail_next_receipt = True
    with pytest.raises(RuntimeError, match="went away"):
        asyncio.run(service.add(request))
    asyncio.run(service.add(request))

    rows = list(repository.chunks[forget_ledger_tenant(tenant_for("retry-user"))].values())
    assert repository.ledger_writes == 2
    assert len(rows) == 1
    assert rows[0].metadata["message_ordinal"] == 0
    assert rows[0].metadata["source_session_id"] == request.session_id


def test_stub_and_annotate_modes_through_search(monkeypatch) -> None:
    monkeypatch.setenv("RECALL_AML_FORGET", "stub")
    service, _ = _service()
    _ingest(service, "stub-user")
    stubbed = _search(service, "stub-user")

    assert len(stubbed.data) == TOP_K
    assert not any(DETAIL in _text(item) for item in stubbed.data)
    assert sum(STUB_SENTENCE in _text(item) for item in stubbed.data) == 2
    assert stubbed.forget_items_stubbed == 2

    monkeypatch.setenv("RECALL_AML_FORGET", "annotate")
    annotated = _search(service, "stub-user")
    assert _text(annotated.data[0]).startswith(ANNOTATION_PREFIX)
    assert sum(DETAIL in _text(item) for item in annotated.data) >= 2
    assert annotated.forget_items_annotated == 1


def test_a_bad_forget_value_stops_service_startup(monkeypatch) -> None:
    monkeypatch.setenv("RECALL_AML_FORGET", "sometimes")
    with pytest.raises(ValueError, match="RECALL_AML_FORGET must be one of off, drop, stub, annotate"):
        _service()


def test_a_mode_without_a_ledger_repository_stops_startup(monkeypatch) -> None:
    monkeypatch.setenv("RECALL_AML_FORGET", "drop")
    with pytest.raises(ValueError, match="needs a repository with a forget ledger"):
        _service(_NoLedgerRepository())
    monkeypatch.setenv("RECALL_AML_FORGET", "off")
    _service(_NoLedgerRepository())


def test_headers_version_and_log_report_the_stage(monkeypatch, caplog) -> None:
    from starlette.testclient import TestClient

    from recall_aml.app import create_app
    from recall_aml.config import HostedSettings

    monkeypatch.setenv("RECALL_AML_FORGET", "drop")
    service, _ = _service()
    _ingest(service, "header-user")
    client = TestClient(
        create_app(
            HostedSettings("postgresql://unused", "secret", "forget-commit", variant_name=C9.name),
            service,
        )
    )
    with caplog.at_level(logging.INFO, logger="recall_aml"):
        response = client.post(
            "/v1/search",
            headers={"X-Api-Key": "secret"},
            json={"query": CONTEXT_QUERY, "user_id": "header-user", "top_k": TOP_K},
        )
    version = client.get("/version", headers={"X-Api-Key": "secret"})

    assert response.status_code == 200
    assert list(response.json()) == ["data"]
    assert response.headers.get("X-Recall-Forget-Mode") == "drop"
    assert response.headers.get("X-Recall-Forget-Requests-Applied") == "1"
    assert response.headers.get("X-Recall-Forget-Items-Changed") == "2"
    assert version.json()["forget_suppression"]["mode"] == "drop"
    done = [record for record in caplog.records if record.getMessage().startswith("hosted_search_complete")]
    assert done
    assert getattr(done[-1], "forget_items_dropped", None) == 2
    assert getattr(done[-1], "forget_mode", None) == "drop"
    assert not any(re.search(DETAIL, str(value)) for value in vars(done[-1]).values())


# ---------------------------------------------------------------- real pgvector


@requires_db
def test_postgres_ledger_round_trip_isolation_and_delete(make_store) -> None:
    """Ledger rows round-trip through JSONB, stay out of the corpus and other tenants, die with the user."""
    from recall.pool import SharedPool
    from recall.store import PgVectorStore

    fixture_store = make_store(3)
    pool = SharedPool(TEST_DSN, min_size=1, max_size=4)
    serving_store = PgVectorStore(
        TEST_DSN, 3, table=fixture_store.table, tenant="aml_service_readiness", shared_pool=pool, owns_pool=True
    )
    try:
        repository = PgHostedRepository(serving_store, _Embedder("code"))  # type: ignore[arg-type]
        tenant_a = tenant_for(f"pg-forget-a-{fixture_store.table}")
        tenant_b = tenant_for(f"pg-forget-b-{fixture_store.table}")
        rows = ledger_chunks(
            tenant_a,
            find_forget_requests(
                _messages(
                    ("user", "I collect vintage fountain pens from Italian makers."),
                    ("assistant", "Since you like Italian pens, a leather case suits your collection."),
                    ("user", "Please forget that I collect vintage fountain pens from Italian makers."),
                ),
                "pg-session",
            ),
            created_at="2026-09-28T00:00:00+00:00",
        )
        repository.persist_forget_requests(tenant_a, rows)
        repository.persist_forget_requests(tenant_a, rows)

        stored = repository.forget_requests(tenant_a)
        assert [row.id for row in stored] == [rows[0].id]
        assert stored[0].metadata["target_text"] == "I collect vintage fountain pens from Italian makers"
        assert stored[0].metadata["message_ordinal"] == 2
        assert [entry.entry_id for entry in ledger_entries(stored)] == [rows[0].id]
        assert repository.forget_requests(tenant_b) == []
        assert repository.tenant_store(tenant_a).count() == 0

        repository.delete_tenant(tenant_a)
        assert repository.forget_ledger_store(tenant_a).count() == 0
    finally:
        serving_store.close()
