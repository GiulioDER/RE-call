"""R2-1: in-conversation forget requests, detected at Add, suppressed at Search, default off.

The examples below follow the templated shapes of PersonaMem-v2's forget turns ("Please forget
that I ...", "Forget that I ...", "... that you remember I ...", "... from your memory") with
invented details; no benchmark text is copied.

Red proofs, run 2026-09-28 on the branch after merging master 1f8666df, with
``PYTHONDONTWRITEBYTECODE=1``: each row is one plausible mutation of the named production line
with this file unchanged, the named test run alone, the ``E`` line it failed with (an assertion,
never an import, collection or fixture error), then the file restored and the same test run green.
The DB row ran against this checkout's own ``scripts/session-db.sh`` container. Node ids are
``tests/test_aml_forget.py::<name>``.

- ``test_templated_requests_are_detected_with_their_target``: the ``that\\s+you\\s+(?:remember
  ...)`` alternative removed from ``forget._PREFIX``. Failed: ``'you remember I restore old sailing
  boats' != 'I restore old sailing boats'``.
- ``test_two_requests_in_one_message_are_both_found``: ``rest`` in ``forget._REQUEST`` made a
  consuming group instead of a lookahead. Failed: ``['I play chess'] == [...]``, right contains
  ``'I like hiking in the Alps'``.
- ``test_resets_and_idioms_are_refused``: ``if _BLANKET.match(cut): continue`` deleted from
  ``detect_forget_requests``. Failed on "Forget all my sports preferences." (target ``'all my
  sports preferences'``).
- ``test_a_negated_request_is_refused``: ``if _NEGATED.search(sentence): continue`` deleted.
  Failed on the self-cancelling "... don't forget that I'm moving to Lisbon." sentence.
- ``test_task_deletes_need_an_explicit_memory_object``: ``_STRONG_MEMORY_OBJECT`` replaced by
  ``_MEMORY_OBJECT`` in ``detect_forget_requests``. Failed on "Please delete my earlier draft of the
  essay from the archive."
- ``test_code_fences_are_not_read``: ``text = _FENCED_CODE.sub(" ", text)`` deleted. Failed:
  ``[DetectedRequest(target='I cached the API token', ...)] == []``. The first version of this test
  (a fenced line starting ``# Forget``) stayed GREEN under this mutation, because ``#`` already
  stops the sentence-start pattern, so it guarded nothing; it was rewritten to this form.
- ``test_only_user_turns_of_a_conversation_are_read``: ``if not _is_user(message): continue``
  deleted from ``find_forget_requests``. Failed: ``[2, 3] == [2]``.
- ``test_a_coding_trajectory_is_never_read``: ``if looks_like_coding(messages): return []``
  deleted. Failed: ``[ForgetRequest(..., target='I renamed utils.py; ...')] == []``.
- ``test_a_failing_confirmer_leaves_the_pattern_verdict``: ``confirmed = detected`` in the
  ``except`` branch of ``find_forget_requests`` changed to ``confirmed = None``. Failed:
  ``assert 0 == 1``.
- ``test_drop_removes_a_window_that_states_the_target_anywhere``: ``TARGET_MATCH_FRACTION``
  0.6 to 0.4 failed ``assert 'partial' in ['neutral']``; 0.6 to 1.0 failed ``assert 'states' not
  in ['states', 'partial', 'neutral']``.
- ``test_drop_removes_the_request_window_even_when_it_is_cut``: the request-shingle clause of
  ``_Text.states`` deleted. Failed: ``'cut-request' not in ['cut-request']``.
- ``test_drop_removes_the_preceding_exchange_only_in_its_own_session``: the rule R return of
  ``_Text.states`` changed to ``return False and covers(...)`` failed ``'preceding' not in [...]``;
  the same-session check deleted failed ``'same-words-elsewhere' in ['same-session-unrelated']``.
- ``test_stub_replaces_only_the_forgotten_sentences``: ``stubbed = STUB_SENTENCE`` for every
  matched item in ``_stub_text``. Failed the exact content equality.
- ``test_annotate_keeps_every_item_and_names_the_target_first``: ``content = first.content`` in
  ``annotate_items`` (note built, never used). Failed the ``startswith(ANNOTATION_PREFIX)``.
- ``test_the_ledger_lives_in_its_own_namespace_per_tenant``: ``forget_ledger_store`` returning
  ``self.tenant_store(tenant)`` (the ledger inside the corpus). Failed ``set(base.rows) ==
  {forget_ledger_tenant(tenant_a)}``.
- ``test_c9_and_every_variant_default_to_off``: ``HostedVariant.forget_suppression`` defaulting to
  ``"drop"``. Failed ``'drop' == 'off'``.
- ``test_the_off_path_serves_exactly_what_master_serves``, three mutations: the ``if
  self.forget_mode == "off": return`` guard deleted from ``_record_forget_requests`` failed
  ``calls == []`` (``find_forget_requests`` called); ``forget.mode != "off" and`` deleted from the
  Search guard failed ``calls == []`` (``ledger_entries`` called); the variant default ``"drop"``
  failed the byte equality of the unset and ``off`` bodies. The recorders wrap and still run the
  real functions, so a mutation that calls them is caught by ``calls`` rather than by a crash.
- ``test_drop_backfills_to_top_k_on_the_context_route``: ``run.hits[:] = kept`` deleted from
  ``HostedService.search`` failed ``assert not any(DETAIL ...)``; ``drop_hits`` applied to
  ``run.hits[: request.top_k]`` (filter without backfill) failed ``assert 3 == 5``.
- ``test_the_code_route_is_never_filtered``: ``and specialist_route != "code"`` removed from the
  Search guard. Failed ``assert 2 == 0`` on ``forget_items_dropped``.
- ``test_a_forget_in_one_tenant_never_applies_to_another``: ``forget_ledger_tenant`` digesting
  ``tenant[:4]`` (every tenant shares one ledger). Failed ``assert 1 == 0`` on
  ``forget_items_dropped``.
- ``test_a_retried_add_rewrites_the_same_ledger_row``: ``ledger_id`` with a random uuid in the
  id. Failed ``assert 2 == 1`` on ``len(rows)``.
- ``test_an_add_refused_for_credit_writes_no_ledger_row``: the ledger write moved back to the
  start of ``_add_once`` (where the first draft had it, before the compile). Failed ``{<ledger
  row>} == {}``.
- ``test_stub_and_annotate_modes_through_search``: the stub branch of ``HostedService.search``
  keyed on ``"stubbed"`` failed ``assert not any(DETAIL ...)``; the annotate branch keyed on
  ``"annotated"`` failed ``startswith(ANNOTATION_PREFIX)``.
- ``test_a_bad_forget_value_stops_service_startup``: ``parse_mode`` returning ``"off"`` for an
  unknown value. Failed with DID NOT RAISE. (Removing ``self.forget_mode`` from the startup tuple
  cannot fail this test, since the ledger capability check reads the same property.)
- ``test_a_mode_without_a_ledger_repository_stops_startup``: the capability check in
  ``HostedService.__init__`` made ``if False and ...``. Failed with DID NOT RAISE.
- ``test_headers_version_and_log_report_the_stage``: ``X-Recall-Forget-Items-Changed`` summing
  only stubbed and annotated items failed ``'0' == '2'``; the ``forget_suppression`` key removed
  from ``/version`` failed ``None == {...}``; ``forget_items_dropped`` removed from the
  ``hosted_search_complete`` extra failed ``None == 2``.
- ``test_postgres_ledger_round_trip_isolation_and_delete`` (real pgvector):
  ``forget_ledger_tenant(tenant)`` removed from ``PgHostedRepository.delete_tenant`` failed
  ``assert 1 == 0`` on ``forget_ledger_store(tenant_a).count()``; ``forget_ledger_store``
  returning ``self.tenant_store(tenant)`` failed ``assert 1 == 0`` on
  ``tenant_store(tenant_a).count()``.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time

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
        "Forget the earlier ones and start over.",
        "Forget all the context and rules above this line.",
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
        "Don't forget that I told you about my garden plans.",
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
    """A request-shaped line inside fenced code is content, not a request; prose around it is read."""
    fenced = "```\nForget that I cached the API token.\n```"
    assert detect_forget_requests(fenced) == []
    found = detect_forget_requests("Please forget that I play chess.\n" + fenced)
    assert [item.target for item in found] == ["I play chess"]


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

    "What master serves" is established by construction here: every R2-1 function is wrapped by a
    recorder that still runs it, and none is called, so the response is what the code without them
    computes. The PR records the same bytes computed by master's own checkout.
    """
    import recall_aml.service as service_module

    calls: list[str] = []

    def recorded(name: str):
        real = getattr(service_module, name)

        def wrapper(*args, **kwargs):
            calls.append(name)
            return real(*args, **kwargs)

        return wrapper

    for name in ("find_forget_requests", "ledger_entries", "drop_hits", "stub_items", "annotate_items"):
        monkeypatch.setattr(service_module, name, recorded(name))

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

    assert response.forget_items_dropped == 0
    assert any(DETAIL in _text(item) for item in response.data)
    assert repository.chunks[forget_ledger_tenant(tenant_for("tenant-b"))] == {}


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


class _OutOfCredit(Exception):
    status_code = 402


class _OutOfCreditCompiler:
    def compile_anchored_v3(self, messages, session_id, prior):
        raise _OutOfCredit("insufficient credits")


def test_an_add_refused_for_credit_writes_no_ledger_row(monkeypatch) -> None:
    """C9 stops an Add on a compile 402 before storing anything (#804); the ledger obeys that too.

    Otherwise an abandoned Add would leave a forget request whose windows were never stored.
    """
    from recall_aml.service import CompilerCreditExhausted

    monkeypatch.setenv("RECALL_AML_FORGET", "drop")
    service, repository = _service()
    service._compiler = _OutOfCreditCompiler()  # type: ignore[assignment]
    request = _rounds("credit-user")[-1]
    with pytest.raises(CompilerCreditExhausted):
        asyncio.run(service.add(request))

    assert repository.chunks[forget_ledger_tenant(tenant_for("credit-user"))] == {}
    assert repository.ledger_writes == 0


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
    # FIX-6 (2026-09-30) changed this contract: the count is the served items that state a target
    # (two here, as for stub above), where it was always 1.
    assert annotated.forget_items_annotated == 2


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
    assert version.json().get("forget_suppression") == {
        "mode": "drop",
        "detector": "pattern-v1",
        "confirmation": "none",
        "code_route": "exempt",
    }
    done = [record for record in caplog.records if record.getMessage().startswith("hosted_search_complete")]
    assert done
    assert getattr(done[-1], "forget_items_dropped", None) == 2
    assert getattr(done[-1], "forget_mode", None) == "drop"
    assert not any(re.search(DETAIL, str(value)) for value in vars(done[-1]).values())


# ---------------------------------------------------------------- audit fixes (2026-09-30)


@pytest.mark.parametrize(
    "text",
    [
        ("ok" + "   ") * 13 + "x",
        "." + " " * 4000 + "x",
        "a" + "\n" * 4000 + "b",
        "Please forget that I like chess" + " " * 280 + "y\n",
        " " * 100_000 + "x",
        "\t" * 100_000 + "x",
        "." * 100_000,
        "!" * 100_000 + ")",
        "\nok" * 8_000,
    ],
    ids=[
        "polite-words",
        "spaces-after-stop",
        "blank-lines",
        "trailing-target-spaces",
        "long-spaces",
        "long-tabs",
        "long-dots",
        "long-bangs",
        "polite-words-on-new-lines",
    ],
)
def test_the_detector_is_not_superlinear_on_whitespace(text: str) -> None:
    """FIX-1: one message of whitespace must not hold the interpreter for seconds.

    Every short case took 0.7 s to 10 s before the fix (``_POLITE`` exponential,
    ``_SENTENCE_START`` quadratic, ``_SUFFIX`` cubic), all under the GIL, so the whole service
    stalled. The 100,000-character cases took 13 s to 21 s under the first fix, which left the
    newline collapse and punctuation runs quadratic: 4,000 characters was too few to show it.
    """
    started = time.perf_counter()
    detect_forget_requests(text)
    assert time.perf_counter() - started < 0.5


def test_polite_and_spaced_requests_are_still_detected() -> None:
    """FIX-1 control: collapsing whitespace and possessive quantifiers change no verdict."""
    for text in (
        "Please, forget that I play chess.",
        "Ok  so   please forget that I play chess.",
        "Thanks.   \n\n   Please forget that I play chess   .",
    ):
        assert [request.target for request in detect_forget_requests(text)] == ["I play chess"]


def test_a_request_stores_a_bounded_copy_of_its_context() -> None:
    """FIX-2: each request stored a whole copy of up to four messages, unbounded.

    A 209 KB Add with 50 requests wrote 10.6 MB of ledger text before the fix. The kept copy is
    the tail, the text nearest the request, which is what rule R compares.
    """
    reply = "Since you collect vintage fountain pens, a leather case suits you."
    messages = _messages(
        ("user", "hello"),
        ("assistant", "The weather report says sunny skies over the valley. " * 4000 + reply),
        ("user", "Please forget that I collect vintage fountain pens and " + "more words " * 600),
    )
    [request] = find_forget_requests(messages, "s1")

    assert len(request.preceding_text) <= 4_000
    assert request.preceding_text.endswith(reply)
    assert len(request.request_text) <= 1_000


def test_one_add_stores_a_bounded_number_of_requests() -> None:
    """FIX-2: each row was capped but their number was not; 4,000 requests in one 177 KB Add
    stored 17.6 MB, reparsed by every later Search. At most 64 per Add, a bounded total."""
    lines = " ".join(f"Please forget that I like hobby{index} number." for index in range(300))
    messages = _messages(("assistant", "A long reply about hobbies. " * 400), ("user", lines))
    requests = find_forget_requests(messages, "s1")

    assert 0 < len(requests) <= 64
    assert sum(len(r.request_text) + len(r.preceding_text) for r in requests) <= 64 * 5_000


def test_sentence_ends_are_found_without_copying_the_rest_of_the_message() -> None:
    """NEW-1: every verb at a sentence start copied the rest of the message to find its end, so
    request-dense text was quadratic under the GIL (1.6M characters took 17.6 s).

    Deterministic rather than timed: a timing ratio was reviewed as able to flake on a loaded
    runner and to pass a reintroduced copy on a fast host. The text refuses to be sliced.
    """
    from recall_aml.forget import _sentence_end

    class NoSlice(str):
        def __getitem__(self, key):
            if isinstance(key, slice):
                raise AssertionError("_sentence_end copied the rest of the message")
            return super().__getitem__(key)

    text = NoSlice("x" * 20 + " Remove the foil. " + "word " * 1_000)
    assert _sentence_end(text, 21) == 37


def test_the_preceding_exchange_is_built_once_per_message(monkeypatch) -> None:
    """NEW-1: the exchange before a message was rebuilt for every request in it, before the cap."""
    import recall_aml.forget as forget_module

    calls: list[int] = []
    real = forget_module._preceding_text

    def counted(messages, ordinal):
        calls.append(ordinal)
        return real(messages, ordinal)

    monkeypatch.setattr(forget_module, "_preceding_text", counted)
    lines = " ".join(f"Please forget that I like hobby{index} number." for index in range(10))
    requests = find_forget_requests(_messages(("assistant", "Hello there."), ("user", lines)), "s1")

    assert len(requests) == 10
    assert calls == [1]


def test_a_repeated_request_does_not_use_up_the_cap() -> None:
    """FIX-2: 70 repeats of one request filled the 64 slots and dropped the next, distinct one;
    the repeats are one ledger row anyway."""
    lines = " ".join(["Please forget that I smoke."] * 70 + ["Please forget that I own a boat."])
    requests = find_forget_requests(_messages(("user", lines)), "s1")

    assert [request.target for request in requests] == ["I smoke", "I own a boat"]


def test_a_one_word_target_applies_only_in_its_own_session() -> None:
    """FIX-3: "forget my old messages" yields the single stem ``old``; across the tenant it
    dropped every window with that word ("My daughter turned 7 years old")."""
    requests = find_forget_requests(_messages(("user", "Please forget my old messages.")), "s9")
    entries = ledger_entries(ledger_chunks(tenant_for("u"), requests, created_at="2026-09-30T00:00:00+00:00"))
    assert [entry.stems for entry in entries] == [frozenset({"old"})]

    kept, _ = drop_hits(
        [
            _hit("other-session", "My daughter turned 7 years old in March.", "s1"),
            _hit("same-session", "The old messages were about the move to Lyon.", "s9"),
        ],
        entries,
    )
    assert [hit.chunk.id for hit in kept] == ["other-session"]


def test_stub_never_erases_an_item_whose_sentences_do_not_state_the_target() -> None:
    """FIX-5: an item matched as a whole with no single matching sentence was replaced by the stub,
    erasing unrelated facts (here the sister's visits)."""
    requests = find_forget_requests(
        _messages(("user", "Please forget that I keep bees on the roof of my flat in town.")), "s1"
    )
    entries = ledger_entries(ledger_chunks(tenant_for("u"), requests, created_at="2026-09-30T00:00:00+00:00"))
    item = _item(
        "spread",
        "We keep a spare key under the mat. The roof leaks when it rains in March. "
        "The flat is small but my sister visits on Sundays for lunch.",
        session="s2",
    )
    stubbed, applied, changed = stub_items([item], entries)

    assert stubbed[0] is item
    assert changed == 0 and applied == set()


def test_annotate_counts_every_served_item_that_states_a_target() -> None:
    """FIX-6: annotate reported 1 however many served items stated the target."""
    items = [
        _item("a", "The weather in Lisbon is mild in October."),
        _item("b", "I collect vintage fountain pens from Italian makers."),
        _item("c", "Old note: user collects vintage fountain pens from Italy.", session="other"),
    ]
    _, applied, changed = annotate_items(items, _entries())
    assert changed == 2 and len(applied) == 1


def test_drop_counts_only_what_would_have_been_served(monkeypatch) -> None:
    """FIX-6: the drop count covered the whole candidate pool, so with ``top_k`` 1 it reported two
    items changed when one served item had changed."""
    monkeypatch.setenv("RECALL_AML_FORGET", "drop")
    service, _ = _service()
    _ingest(service, "count-user")
    response = _search(service, "count-user", top_k=1)

    assert response.forget_items_dropped == 1
    assert response.forget_requests_applied == 1


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
