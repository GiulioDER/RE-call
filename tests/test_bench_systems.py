"""`MemorySystem` protocol + `RecallSystem` adapter.

The unit test below needs no DB: it only checks that a plain object satisfying the three
`MemorySystem` members is accepted where the protocol is expected. The integration test at the
bottom is the real proof — it exercises `RecallSystem` end-to-end against Postgres and asserts a
distinctive fact actually round-trips through ingest -> retrieve, not just that a string comes
back.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import requires_fastembed

from recall.eval.locomo import parse_session_date
from recall.frontmatter import parse_frontmatter

from benchmarks.systems import (
    BENCH_TABLE,
    MemorySystem,
    resolve_embedder,
    sample_id_of,
    tenant_for,
)


def _mem0_installed() -> bool:
    return importlib.util.find_spec("mem0") is not None


class _FakeSystem:
    name = "fake"

    def __init__(self) -> None:
        self.ingested: list[dict[str, Any]] = []

    def ingest(self, conversation: dict[str, Any]) -> None:
        self.ingested.append(conversation)

    def retrieve(self, question: str) -> str:
        return f"ctx for {question}"


def test_fake_satisfies_protocol() -> None:
    system: MemorySystem = _FakeSystem()
    # the runtime check, not just the annotation: a `MemorySystem = ...` binding is erased at
    # runtime, so without this the test asserted nothing about the protocol at all
    assert isinstance(_FakeSystem(), MemorySystem)
    system.ingest({"sample_id": "c1"})
    assert system.retrieve("q") == "ctx for q"
    assert system.name == "fake"


def test_sample_id_is_required_and_shared_by_both_adapters() -> None:
    """One identity rule, and it fails loudly. Three divergent fallbacks used to exist.

    A LOCOMO item with no `sample_id` produced the tenant/user `bench-None` in BOTH adapters, so
    every such conversation shared one memory: questions answered out of a neighbour's turns, and
    accuracy inflated with no error anywhere. There is no safe fallback, so there is none.
    """
    assert sample_id_of({"sample_id": "conv-26"}) == "conv-26"
    assert tenant_for({"sample_id": "conv-26"}) == "bench-conv-26"
    # numeric sample_ids exist in the wild; they identify fine, they just are not strings
    assert sample_id_of({"sample_id": 7}) == "7"
    for broken in ({}, {"sample_id": None}, {"sample_id": ""}, {"sample_id": "   "}):
        with pytest.raises(ValueError, match="sample_id"):
            sample_id_of(broken)


def test_recall_system_describe_reports_its_configuration() -> None:
    """The results artifact must be able to name the embedder and budget that produced it."""
    from benchmarks.systems import RecallSystem

    system = RecallSystem("postgresql://x/y", embedder_name="hashing", k=9)
    described = system.describe()
    assert described["system"] == "recall"
    assert described["k"] == 9
    assert described["embedder"] == {"name": "hashing", "model": "hashing-64"}
    assert described["table"] == BENCH_TABLE
    # no DSN anywhere in the published block — it can carry a password
    assert "postgresql://" not in json.dumps(described)


def test_fake_records_ingested_conversations() -> None:
    system = _FakeSystem()
    system.ingest({"sample_id": "c1"})
    system.ingest({"sample_id": "c2"})
    assert [c["sample_id"] for c in system.ingested] == ["c1", "c2"]


class _FakeStore:
    """Stands in for `PgVectorStore` so the trust wiring can be read without a database.

    `retrieve` only uses the store as a context manager and hands it straight to the search call,
    so a bare object with `__enter__`/`__exit__` is the whole surface needed here.
    """

    def __enter__(self) -> _FakeStore:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def test_retrieve_runs_the_research_policy_with_an_explicit_calibration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The two halves of this arm's trust configuration, pinned where CI can actually see them.

    Both integration tests below cover this behaviourally and BOTH are `@requires_fastembed`,
    which the `[dev]` extra does not install — so CI skips them and skipped them on the day the
    strict trust gate landed. `RecallSystem` kept calling `trusted_search` with no policy while
    every other research harness moved to `research_search`, and the whole RE-call arm refused
    every question with `INDEX_NOT_READY` for four days with a green pipeline. This test needs
    neither Postgres nor fastembed, so the next such omission fails on the pull request.

    Patches `trusted_search` where `_research_trust` bound it (module-level `from` import), not
    `research_search` itself: the policy is applied INSIDE `research_search`, so patching that
    would observe only the argument this adapter passes and would go green again the moment the
    call reverted to plain `trusted_search`. Watching the layer underneath both is what makes the
    policy assertion real rather than a restatement of the call.
    """
    from recall.eval import _research_trust
    from recall.trust_policy import TrustMode

    from benchmarks.systems import RecallSystem

    seen: dict[str, Any] = {}

    def _capture(store: Any, embedder: Any, query: str, **kwargs: Any) -> Any:
        seen.update(kwargs)
        raise AssertionError("stop here: the wiring is the thing under test")

    monkeypatch.setattr(_research_trust, "trusted_search", _capture)
    monkeypatch.setattr("recall.store.PgVectorStore", lambda *a, **kw: _FakeStore())

    system = RecallSystem("postgresql://x/y", embedder_name="hashing")
    system._tenant = "bench-itest"  # normally set by ingest(); this test does not index
    with pytest.raises(AssertionError, match="stop here"):
        system.retrieve("q")

    # Half one: development mode, or every query refuses before retrieval ever runs.
    assert seen["policy"].mode is TrustMode.DEVELOPMENT
    # Half two: an explicit calibration, or `recall.trust` forces `abstained=False` on every
    # query and the abstention rate this benchmark publishes becomes a structural zero.
    calibration = seen.get("calibration")
    assert calibration is not None, "retrieve() passed no calibration, so the gate cannot fire"
    assert calibration.threshold == system.describe()["trust"]["threshold"]


def test_describe_reports_the_abstention_gate_that_produced_the_numbers() -> None:
    """The published artifact must name the threshold behind its headline abstention rate.

    Every field is read off `_search_kwargs()`, the same object `retrieve` sends, so this asserts
    what the arm RAN under rather than a hand-written restatement of it. `enforced` in particular:
    reported from an attribute `__init__` always sets, it would have read `true` unconditionally,
    including after a `retrieve` that stopped passing the calibration.
    """
    from recall.guards import DEFAULT_GAP_THRESHOLD
    from recall.trust_policy import TrustPolicy

    from benchmarks.systems import RecallSystem

    system = RecallSystem("postgresql://x/y", embedder_name="hashing")
    assert system.describe()["trust"] == {
        "policy": "development",
        "threshold": DEFAULT_GAP_THRESHOLD,
        "source": "library-default",
        "certified": False,
        "enforced": True,
    }
    # `enforced` follows the ARGUMENTS rather than the attribute. The distinction is the whole
    # point, so the calibration is dropped from `_search_kwargs` — the single source both readers
    # share — and NOT from `self._calibration`. Nulling the attribute would have proved nothing:
    # the regressed `describe()` this is guarding against read that same attribute, so it passes
    # such a test unchanged. Patching the source is what discriminates, because only a `describe()`
    # reading through `_search_kwargs` can see the difference.
    system._search_kwargs = lambda: {"k": system._k, "reranker": system._reranker}  # type: ignore[method-assign]
    assert system.describe()["trust"] == {
        "policy": "development",
        "threshold": None,
        "source": "library-default",
        "certified": False,
        "enforced": False,
    }
    # `policy` obeys the same rule, and it is the field most likely to be varied on purpose: a
    # sweep measuring strict-mode refusals would add one to `_search_kwargs`, and `research_search`
    # lets a caller's policy win over its own. A hardcoded `development` here would publish the
    # opposite of what such a run actually did.
    strict = TrustPolicy.strict_policy()
    system._search_kwargs = lambda: {"k": system._k, "policy": strict}  # type: ignore[method-assign]
    assert system.describe()["trust"]["policy"] == "strict"
    # A PRESENT-but-None policy degrades to the harness default rather than raising. `describe()`
    # writes the results artifact, so an `AttributeError` here would cost a finished run its
    # output; and None is the spelling most likely to be written, since it is `trusted_search`'s
    # own declared default for that parameter.
    system._search_kwargs = lambda: {"k": system._k, "policy": None}  # type: ignore[method-assign]
    assert system.describe()["trust"]["policy"] == "development"


@requires_fastembed
@pytest.mark.skipif(not os.environ.get("RECALL_TEST_DSN"), reason="needs Postgres")
def test_recall_system_indexes_and_retrieves() -> None:
    """A distinctive fact ingested in one turn must come back out of retrieve().

    `quokka-telemetry-4417` cannot appear by chance — it only shows up in the context if
    `RecallSystem` actually indexed the turn that mentions it and `trusted_search` actually
    surfaced that turn for a question about it. Asserting `isinstance(ctx, str)` would pass even
    if ingest/retrieve were both no-ops; this asserts the real behaviour under test.
    """
    from benchmarks.systems import RecallSystem

    conv = {
        "sample_id": "itest-quokka",
        "conversation": {
            "speaker_a": "Alice",
            "speaker_b": "Bob",
            "session_1_date_time": "1 January 2024",
            "session_1": [
                {
                    "speaker": "Alice",
                    "dia_id": "D1:1",
                    "text": "Our new monitoring code name is quokka-telemetry-4417.",
                },
                {
                    "speaker": "Bob",
                    "dia_id": "D1:2",
                    "text": "Got it, I'll wire up the dashboards for that.",
                },
            ],
        },
    }
    system = RecallSystem(os.environ["RECALL_TEST_DSN"])
    system.ingest(conv)
    ctx = system.retrieve("What is the name of the new monitoring code?")
    assert "quokka-telemetry-4417" in ctx


@requires_fastembed
@pytest.mark.skipif(not os.environ.get("RECALL_TEST_DSN"), reason="needs Postgres")
def test_recall_system_returns_empty_string_on_abstention() -> None:
    """Retrieving a question with no relevant indexed content must abstain to an empty string.

    This is the exact behaviour the whole benchmark exists to measure: RecallSystem.retrieve
    returns "" — not an apology, not a fallback sentence — when the trust layer finds nothing it
    is confident enough to serve.
    """
    from benchmarks.systems import RecallSystem

    conv = {
        "sample_id": "itest-abstain",
        "conversation": {
            "speaker_a": "Alice",
            "speaker_b": "Bob",
            "session_1_date_time": "1 January 2024",
            "session_1": [
                {"speaker": "Alice", "dia_id": "D1:1", "text": "I had cereal for breakfast."},
            ],
        },
    }
    system = RecallSystem(os.environ["RECALL_TEST_DSN"])
    system.ingest(conv)
    ctx = system.retrieve("What is the boiling point of neptunium on Mars?")
    assert ctx == ""


@requires_fastembed
@pytest.mark.skipif(not os.environ.get("RECALL_TEST_DSN"), reason="needs Postgres")
def test_recall_system_reingest_does_not_duplicate_the_corpus() -> None:
    """Ingesting the same conversation twice must REPLACE its rows, never accumulate them.

    This is the defect that made results a function of how many times the harness was run.
    `index_conversation` writes each turn into a fresh `mkdtemp` directory, and `Indexer` derives
    the chunk id from the absolute path (``md5(f"{abs_path}:{i}")``) and the row's `source` from
    it too — so a second run presents ids that collide with nothing (`ON CONFLICT` never fires)
    and sources that match nothing (`replace_sources` deletes nothing). Every turn gets inserted
    beside its twin, top-k fills with duplicates, and RE-call's effective context shrinks.

    Asserts real CONTENT counts, not just that ingest returned: the row count after the second
    ingest must equal the count after the first (and the number of turns), and the distinctive
    fact must appear exactly once in the retrieved context.
    """
    from recall.store import PgVectorStore

    from benchmarks.systems import RecallSystem

    marker = "quokka-telemetry-4417"
    conv = {
        "sample_id": "itest-reingest",
        "conversation": {
            "speaker_a": "Alice",
            "speaker_b": "Bob",
            "session_1_date_time": "1 January 2024",
            "session_1": [
                {
                    "speaker": "Alice",
                    "dia_id": "D1:1",
                    "text": f"Our new monitoring code name is {marker}.",
                },
                {"speaker": "Bob", "dia_id": "D1:2", "text": "Got it, I'll wire the dashboards."},
            ],
        },
    }
    dsn = os.environ["RECALL_TEST_DSN"]
    system = RecallSystem(dsn)
    tenant = tenant_for(conv)

    def _rows() -> tuple[int, int]:
        with PgVectorStore(
            dsn, dim=system.embedder.dim, tenant=tenant, table=BENCH_TABLE
        ) as store:
            chunks = list(store.iter_chunks())
        return len(chunks), sum(1 for c in chunks if marker in c.text)

    system.ingest(conv)
    first_total, first_marker = _rows()
    system.ingest(conv)
    second_total, second_marker = _rows()

    assert first_total == 2  # the two turns, indexed once each
    assert first_marker == 1
    assert (second_total, second_marker) == (first_total, first_marker)
    # and the duplication is absent where it would actually cost accuracy: the served context
    assert system.retrieve("What is the name of the new monitoring code?").count(marker) == 1


def test_mem0_config_points_llm_at_openrouter_and_local_embedder() -> None:
    from benchmarks.systems import mem0_config

    cfg = mem0_config(openrouter_key="sk-x", model="openai/gpt-4o-mini")
    assert cfg["llm"]["provider"] == "openai"
    assert cfg["llm"]["config"]["openai_base_url"] == "https://openrouter.ai/api/v1"
    assert cfg["llm"]["config"]["model"] == "openai/gpt-4o-mini"
    assert cfg["llm"]["config"]["api_key"] == "sk-x"
    assert cfg["embedder"]["provider"] == "huggingface"  # free local, not OpenAI
    assert "bge-small" in cfg["embedder"]["config"]["model"]


def test_mem0_config_default_arm_uses_openai_embeddings() -> None:
    from benchmarks.systems import mem0_config

    cfg = mem0_config(
        openrouter_key="sk-x", model="openai/gpt-4o-mini", embedder="openai", openai_key="sk-emb"
    )
    assert cfg["embedder"]["provider"] == "openai"
    assert cfg["embedder"]["config"]["model"] == "text-embedding-3-small"
    assert cfg["embedder"]["config"]["api_key"] == "sk-emb"
    # the store must be sized for THIS embedder; Mem0's Qdrant default (1536) is right here and
    # silently wrong for the 384-wide bge-small the controlled arm uses
    assert cfg["vector_store"]["config"]["embedding_model_dims"] == 1536


def test_mem0_config_isolates_storage_per_run(tmp_path: Path) -> None:
    """Two runs must not share a vector store, or run two measures run one's memories as well.

    Left unconfigured, Mem0 opens ONE on-disk Qdrant database at a fixed path under a fixed
    collection name, so a second benchmark run reopens the first run's collection and adds to it —
    the mirror of the RE-call re-ingest defect, and it has to be fixed on both arms or the
    comparison is between one system that was reset and one that was not.
    """
    from benchmarks.systems import mem0_config

    first = mem0_config("sk-x", "openai/gpt-4o-mini", run_id="run-A", storage_dir=tmp_path)
    second = mem0_config("sk-x", "openai/gpt-4o-mini", run_id="run-B", storage_dir=tmp_path)

    assert first["vector_store"]["provider"] == "qdrant"
    assert first["vector_store"]["config"]["embedding_model_dims"] == 384  # bge-small, not 1536
    # path AND collection differ, so neither Qdrant's storage nor its namespace is shared
    assert (
        first["vector_store"]["config"]["path"] != second["vector_store"]["config"]["path"]
    )
    assert (
        first["vector_store"]["config"]["collection_name"]
        != second["vector_store"]["config"]["collection_name"]
    )
    assert str(tmp_path) in first["vector_store"]["config"]["path"]
    # the SQLite history db defaults to one shared ~/.mem0/history.db; stamped for the same reason
    assert first["history_db_path"] != second["history_db_path"]


def test_mem0_describe_redacts_api_keys() -> None:
    """`describe()` is copied verbatim into a PUBLISHED artifact. It must never carry a key."""
    from benchmarks.systems import Mem0System

    system = Mem0System("sk-openrouter-secret", model="openai/gpt-4o-mini", k=7)
    described = system.describe()
    dumped = json.dumps(described)
    assert "sk-openrouter-secret" not in dumped
    assert described["llm"]["config"]["api_key"] == "***redacted***"
    # but everything a reader needs to reproduce the arm is still there
    assert described["k"] == 7
    assert described["llm"]["config"]["model"] == "openai/gpt-4o-mini"
    assert described["embedder"]["provider"] == "huggingface"
    assert described["vector_store"]["provider"] == "qdrant"
    assert "mem0ai_version" in described  # None when the bench extra is absent, and that is fine


_IMAGE_MARKER = "\n\n[shared an image: "

PARITY_CONVERSATION: dict[str, Any] = {
    "speaker_a": "Alice",
    "speaker_b": "Bob",
    "session_2_date_time": "2 January 2024",
    "session_2": [
        {"speaker": "Bob", "dia_id": "D2:1", "text": "Second session first turn."},
    ],
    "session_1_date_time": "1 January 2024",
    "session_1": [
        {"speaker": "Alice", "dia_id": "D1:1", "text": "First session first turn."},
        {"speaker": "Bob", "text": "No dia_id, must be skipped by both systems."},
        {
            "speaker": "Alice",
            "dia_id": "D1:2",
            "text": "First session second turn.",
            "blip_caption": "a whiteboard diagram",
        },
    ],
}


def _split_caption(body: str) -> tuple[str, str]:
    """Split a turn body into (spoken part, image caption). Caption is "" when there is none.

    Shared by both extractors below so the caption is compared as a FIELD on both sides rather
    than as a substring that happens to appear somewhere in one of them.
    """
    spoken, marker, rest = body.partition(_IMAGE_MARKER)
    if not marker:
        return body.strip(), ""
    caption = rest.rstrip()
    assert caption.endswith("]"), f"malformed image line: {rest!r}"
    return spoken.strip(), caption[:-1]


def _recall_payload(document: str) -> dict[str, str]:
    """Informational fields of one on-disk RE-call corpus document.

    Parses the real file `write_conversation_corpus` wrote (an optional validity frontmatter
    block, then a ``# speaker — date`` header, then a ``speaker: text`` body and an optional image
    line) back into named fields, so the comparison against Mem0 is field-by-field and independent
    of either side's formatting.

    The frontmatter is stripped BEFORE the field comparison, and its ``valid_from`` is separately
    asserted to equal the date already printed in the body. That check is the point rather than
    housekeeping: this test is the head-to-head FAIRNESS invariant, so a frontmatter key carrying
    anything Mem0 does not also receive would hand RE-call information its competitor never sees.
    ``valid_from`` is derived from the session stamp that is already in the body, and in Mem0's own
    payload, so it re-encodes an existing field rather than adding one. This asserts that equality
    rather than trusting it.
    """
    meta, document = parse_frontmatter(document)
    header, sep, body = document.partition("\n\n")
    assert sep and header.startswith("# "), f"unexpected document shape: {document!r}"
    if "valid_from" in meta:
        body_date = header[2:].partition(" — ")[2].strip()
        assert parse_session_date(body_date) == meta["valid_from"], (
            "valid_from disagrees with the session date in the body, so the frontmatter would be "
            "carrying information Mem0 never receives"
        )
    header_speaker, dash, date = header[2:].partition(" — ")
    assert dash, f"document header carries no session date: {header!r}"
    spoken, caption = _split_caption(body)
    speaker, colon, text = spoken.partition(": ")
    assert colon, f"document body carries no speaker: {spoken!r}"
    assert speaker == header_speaker.strip()
    return {"speaker": speaker, "date": date.strip(), "text": text, "caption": caption}


def _mem0_payload(content: str) -> dict[str, str]:
    """Informational fields of one Mem0 message content (``[date] speaker: text`` + image line)."""
    assert content.startswith("["), f"message carries no session date: {content!r}"
    date, sep, rest = content[1:].partition("] ")
    assert sep, f"message carries no session date: {content!r}"
    spoken, caption = _split_caption(rest)
    speaker, colon, text = spoken.partition(": ")
    assert colon, f"message carries no speaker: {spoken!r}"
    return {"speaker": speaker, "date": date.strip(), "text": text, "caption": caption}


def test_conversation_to_messages_mirrors_recall_turn_walk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Differential test: what Mem0 is fed must equal what RE-call actually indexed.

    `write_conversation_corpus` is the function `RecallSystem.ingest` -> `index_conversation`
    indexes turns through, so this reads the corpus IT WROTE back off disk — the real documents,
    not a restatement of them — and compares the information in each against
    `_conversation_to_messages`'s output, turn for turn:

    - **order** comes from the walk itself. `_turn_document` is wrapped to record the dia_id of
      every turn RE-call emits, in emission order; the documents are then replayed in that order.
      Reading the directory alone could not do this (a filesystem listing has no walk order), and
      re-deriving the order from the session/turn numbers would just reimplement the walk under
      test. `session_2` is declared before `session_1` in the fixture so a naive walk is caught.
    - **content** is compared as parsed fields (speaker, session date, text, image caption), so a
      change to `_turn_document`'s body — the session date above all, which LOCOMO's temporal
      questions turn on — fails here instead of silently handing RE-call an advantage Mem0 never
      got.
    - **skips** are compared by count: the dia_id-less turn must vanish from both sides.
    """
    from recall.eval import locomo

    from benchmarks.systems import _conversation_to_messages

    walk_order: list[str] = []
    original_turn_document = locomo._turn_document

    def _recording_turn_document(turn: dict[str, Any], session_date: str) -> str:
        walk_order.append(str(turn.get("dia_id")))
        return original_turn_document(turn, session_date)

    monkeypatch.setattr(locomo, "_turn_document", _recording_turn_document)

    written = locomo.write_conversation_corpus(PARITY_CONVERSATION, tmp_path)
    messages = _conversation_to_messages(PARITY_CONVERSATION)

    documents = {
        locomo._filename_to_dia_id(p.name): p.read_text(encoding="utf-8")
        for p in sorted(tmp_path.iterdir())
    }
    # Every turn the walk emitted landed on disk under its own dia_id, and nothing else did.
    assert sorted(documents) == sorted(walk_order)
    # Same COUNT: the dia_id-less turn is dropped by write_conversation_corpus (RecallSystem's
    # indexing path); _conversation_to_messages (Mem0's) must drop it too or corpora diverge.
    assert len(messages) == written == len(documents) == 3
    # Same ORDER and same INFORMATION, turn by turn, against the documents RE-call really indexed.
    recall_payloads = [_recall_payload(documents[dia_id]) for dia_id in walk_order]
    mem0_payloads = [_mem0_payload(m["content"]) for m in messages]
    assert mem0_payloads == recall_payloads
    # And the walk really did produce the turns the fixture describes, in numeric session order —
    # pinning this catches a change applied to BOTH sides at once, which the equality above cannot.
    assert recall_payloads == [
        {
            "speaker": "Alice",
            "date": "1 January 2024",
            "text": "First session first turn.",
            "caption": "",
        },
        {
            "speaker": "Alice",
            "date": "1 January 2024",
            "text": "First session second turn.",
            "caption": "a whiteboard diagram",
        },
        {
            "speaker": "Bob",
            "date": "2 January 2024",
            "text": "Second session first turn.",
            "caption": "",
        },
    ]
    # Role tracks speaker (Alice == speaker_a -> "user", Bob -> "assistant"), not turn position.
    assert [m["role"] for m in messages] == ["user", "user", "assistant"]


def test_conversation_to_session_messages_groups_by_session_without_losing_turns() -> None:
    """Session grouping is the ingest chunk unit; it must partition the flat walk, not alter it."""
    from benchmarks.systems import _conversation_to_messages, _conversation_to_session_messages

    grouped = _conversation_to_session_messages(PARITY_CONVERSATION)

    assert [len(session) for session in grouped] == [2, 1]
    assert [m for session in grouped for m in session] == _conversation_to_messages(
        PARITY_CONVERSATION
    )


class _FakeMemory:
    """Stand-in for `mem0.Memory`, recording exactly how ingest chunked its `add()` calls.

    `search` mirrors the mem0ai 2.x signature — keyword-only `filters`/`top_k`, and a hard
    rejection of the 1.x `user_id=`/`limit=` spelling, which is what the real client does via
    `_reject_top_level_entity_params`. A permissive fake would let the adapter keep calling an API
    that no longer exists and the failure would only appear during a paid run.
    """

    def __init__(self, memories: list[str] | None = None) -> None:
        self.calls: list[tuple[list[dict[str, str]], str]] = []
        self.searches: list[tuple[str, dict[str, Any], int]] = []
        self._memories = memories or []

    def add(self, messages: list[dict[str, str]], user_id: str) -> None:
        self.calls.append((list(messages), user_id))

    def search(
        self, query: str, *, filters: dict[str, Any], top_k: int, **kwargs: Any
    ) -> dict[str, Any]:
        if kwargs:
            raise TypeError(f"mem0ai 2.x rejects top-level entity params: {sorted(kwargs)}")
        self.searches.append((query, filters, top_k))
        return {"results": [{"memory": m} for m in self._memories]}


def test_mem0_ingest_chunks_one_add_call_per_session() -> None:
    """Ingestion must not funnel a whole conversation through a single `add()`.

    Mem0 runs an LLM fact-extraction pass per `add()`, so one call carrying every turn invites
    truncation and under-extraction — a self-inflicted handicap on the competitor, and the easiest
    possible rebuttal to the published result. Sessions are the chunk unit; this pins the call
    count, the `user_id` on every call, and that chunking neither reorders nor loses a turn.

    Runs without `mem0ai` installed: `Mem0System._memory` returns the already-set `_mem`, so no
    `from mem0 import Memory` ever executes.
    """
    from benchmarks.systems import Mem0System, _conversation_to_messages

    system = Mem0System("sk-x", model="openai/gpt-4o-mini")
    fake = _FakeMemory()
    system._mem = fake
    system.ingest({"sample_id": "c1", "conversation": PARITY_CONVERSATION})

    assert len(fake.calls) == 2  # one per session, NOT one for the whole conversation
    assert [user_id for _, user_id in fake.calls] == ["bench-c1", "bench-c1"]
    assert [len(messages) for messages, _ in fake.calls] == [2, 1]
    # Chunking is a split, not a rewrite: same turns, same order, nothing dropped.
    assert [m for messages, _ in fake.calls for m in messages] == _conversation_to_messages(
        PARITY_CONVERSATION
    )


def test_mem0_retrieve_uses_the_2x_search_api_and_the_configured_budget() -> None:
    """The entity id goes in `filters` and the budget is `top_k` — the 1.x spelling now RAISES."""
    from benchmarks.systems import Mem0System

    system = Mem0System("sk-x", model="openai/gpt-4o-mini", k=7)
    fake = _FakeMemory(["fact one", "fact two"])
    system._mem = fake
    system.ingest({"sample_id": "c1", "conversation": PARITY_CONVERSATION})

    assert system.retrieve("what happened?") == "fact one\nfact two"
    assert fake.searches == [("what happened?", {"user_id": "bench-c1"}, 7)]


def test_mem0_retrieve_before_ingest_is_an_error() -> None:
    from benchmarks.systems import Mem0System

    system = Mem0System("sk-x", model="openai/gpt-4o-mini")
    system._mem = _FakeMemory()
    with pytest.raises(RuntimeError):
        system.retrieve("q")


class _FakeVectorStore:
    def __init__(self, client: Any) -> None:
        self.client = client


class _RecordingClient:
    """Stands in for the `QdrantClient` that `mem0.Memory` holds under `vector_store.client`."""

    def __init__(self, explode: bool = False) -> None:
        self.closed = 0
        self._explode = explode

    def close(self) -> None:
        self.closed += 1
        if self._explode:
            raise RuntimeError("client refused to close")


class _RecordingMemory:
    """A `mem0.Memory` stand-in that records which handles were released.

    Carries all three stores a real `Memory` can own, because the leak is per-CLIENT and the three
    do not map one-to-one onto clients: `_telemetry_vector_store` is a genuinely separate Qdrant
    client on a machine-global path, while `_entity_store` deliberately SHARES the primary one.
    """

    def __init__(
        self,
        client: Any | None = None,
        explode: bool = False,
        telemetry_client: Any | None = None,
        share_entity_client: bool = False,
    ) -> None:
        primary = client if client is not None else _RecordingClient()
        self.vector_store = _FakeVectorStore(primary)
        self._telemetry_vector_store = (
            _FakeVectorStore(telemetry_client) if telemetry_client is not None else None
        )
        self._entity_store = _FakeVectorStore(primary) if share_entity_client else None
        self.closed = 0
        self._explode = explode
        self.entity_store_touched = False

    @property
    def entity_store(self) -> Any:
        """The PUBLIC name, which teardown must never read.

        On a real `mem0.Memory` this is a lazy property that CONSTRUCTS the entity store on first
        access — so reading it during `close()` would build a vector store, and take a lock, in the
        middle of releasing them. Raising here would be swallowed by `close()`'s total `_read`, so
        the flag is what the test actually asserts on.
        """
        self.entity_store_touched = True
        raise AssertionError("close() must not read the lazily-constructing `entity_store`")

    def close(self) -> None:
        self.closed += 1
        if self._explode:
            raise RuntimeError("memory refused to close")


def test_close_releases_the_telemetry_store_client_too() -> None:
    """`mem0.Memory` opens a SECOND Qdrant client, and missing it reproduces the whole defect.

    With `MEM0_TELEMETRY` on (its default, and this repo never turns it off), `Memory.__init__`
    builds `_telemetry_vector_store` at `~/.mem0/migrations_qdrant`. That path is derived from the
    mem0 home directory, NOT from `run_id`, so unlike the bench store every run and every
    concurrent process shares it — which makes it a worse collision than the one that started this,
    not a lesser one. Verified against a real `Memory`: closing only `vector_store.client` left
    `Storage folder C:\\Users\\...\\.mem0\\migrations_qdrant is already accessed` on the next build.
    """
    from benchmarks.systems import Mem0System

    primary, telemetry = _RecordingClient(), _RecordingClient()
    system = Mem0System("sk-x", model="openai/gpt-4o-mini")
    system._mem = _RecordingMemory(client=primary, telemetry_client=telemetry)

    system.close()

    assert primary.closed == 1
    assert telemetry.closed == 1


def test_close_does_not_double_close_a_shared_client() -> None:
    """For Qdrant, mem0 hands the entity store the SAME client "to avoid RocksDB lock contention".

    So the stores must be deduped by identity: closing per store rather than per client would call
    `close()` twice on one handle. Harmless with qdrant-client today, which guards on an
    already-closed lock file, but it is not a property this code should be relying on.
    """
    from benchmarks.systems import Mem0System

    primary = _RecordingClient()
    system = Mem0System("sk-x", model="openai/gpt-4o-mini")
    system._mem = _RecordingMemory(client=primary, share_entity_client=True)

    system.close()

    assert primary.closed == 1


@pytest.mark.parametrize("call", ["ingest", "retrieve"])
def test_using_a_closed_system_is_an_error_rather_than_a_silent_relock(call: str) -> None:
    """After `close()`, `_memory()` must refuse rather than lazily rebuild.

    `_memory()` is lazy and `close()` only nulls `_mem`, so without this a call after `close()`
    would silently construct a NEW `Memory`, re-acquire the exclusive lock, and leave it held with
    no `__exit__` remaining to release it — the original defect, restored through a silent path.
    The context-manager shape is what makes that reachable, so the guard ships with it.
    """
    from benchmarks.systems import Mem0System

    system = Mem0System("sk-x", model="openai/gpt-4o-mini")
    system._mem = _RecordingMemory()
    system._user = "bench-c1"  # so `retrieve` gets past its own before-ingest guard
    system.close()

    with pytest.raises(RuntimeError, match="after close"):
        if call == "ingest":
            system.ingest({"sample_id": "c1", "conversation": PARITY_CONVERSATION})
        else:
            system.retrieve("q")


def test_close_reads_the_private_entity_store_not_the_lazy_property() -> None:
    """`_entity_store`, deliberately, and this pins the choice against a tidy-up.

    `Memory.entity_store` is a lazy property that constructs the store on first access, so
    "use the public API" would make teardown ACQUIRE a vector store while it is releasing them —
    the same hazard `_closed` guards at the other end. The private attribute is the correct read.
    """
    from benchmarks.systems import Mem0System

    memory = _RecordingMemory()
    system = Mem0System("sk-x", model="openai/gpt-4o-mini")
    system._mem = memory

    system.close()

    assert not memory.entity_store_touched


def test_a_handle_whose_repr_also_raises_still_warns_and_does_not_propagate() -> None:
    """The diagnostic must not become the failure.

    `repr()` of a bound method embeds `repr(__self__)`, so warning with `{release!r}` handed an
    arbitrary object's `__repr__` a path straight out of `close()` — from inside a `finally`,
    where it would replace the exception that explains the run. And the two conditions correlate:
    a handle that refuses to close is a handle in a broken state, which is exactly when a
    `__repr__` that reads that state raises too.
    """
    from benchmarks.systems import Mem0System

    class _Unprintable:
        def close(self) -> None:
            raise RuntimeError("handle refused to close")

        def __repr__(self) -> str:
            raise ValueError("repr exploded")

    memory = _RecordingMemory(client=_Unprintable())
    system = Mem0System("sk-x", model="openai/gpt-4o-mini")
    system._mem = memory

    with pytest.warns(UserWarning, match="vector_store.client"):
        system.close()  # must not propagate

    assert memory.closed == 1  # and the other handle was still released


def test_close_stays_total_under_warnings_as_errors() -> None:
    """`-W error` must not let the diagnostic abort the thing it was added to protect.

    `warnings.warn` raises under warnings-as-errors. Sitting unguarded inside the release loop, it
    turned the FIRST failed handle into an abort of every later one, so a wedged primary client
    left the machine-global telemetry lock and the SQLite handle both open — precisely inverting
    "letting go of one must never be conditional on another closing cleanly". It also escaped
    `latency.py`'s `finally` and `__exit__`, replacing the error that explains the run.

    Not hypothetical: `-W error` and `PYTHONWARNINGS=error` are ordinary CI settings, and this
    suite already emits live warnings from mem0.
    """
    from benchmarks.systems import Mem0System

    class _Wedged:
        def close(self) -> None:
            raise RuntimeError("handle wedged")

    telemetry = _RecordingClient()
    memory = _RecordingMemory(client=_Wedged(), telemetry_client=telemetry)
    system = Mem0System("sk-x", model="openai/gpt-4o-mini")
    system._mem = memory

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        system.close()  # must not raise, even though the report cannot be delivered

    # The handles after the wedged one were still released. That is the whole invariant.
    assert telemetry.closed == 1
    assert memory.closed == 1


def test_an_exception_whose_own_repr_raises_still_warns() -> None:
    """The other half of the same hazard: `BaseException.__repr__` formats `args`.

    So an exception carrying an object with a hostile `__repr__` re-opens the escape path even
    though the handle itself is printable. Naming the handle is not sufficient on its own —
    building the *detail* has to be total too.
    """
    from benchmarks.systems import Mem0System

    class _HostileArg:
        def __repr__(self) -> str:
            raise ValueError("arg repr exploded")

    class _RaisesUnprintable:
        def close(self) -> None:
            raise RuntimeError(_HostileArg())

    memory = _RecordingMemory(client=_RaisesUnprintable())
    system = Mem0System("sk-x", model="openai/gpt-4o-mini")
    system._mem = memory

    with pytest.warns(UserWarning, match="unprintable RuntimeError"):
        system.close()  # must not propagate

    assert memory.closed == 1


def test_a_store_whose_attributes_raise_does_not_break_close() -> None:
    """A probe that raises must mean "nothing there", not a new exception.

    `close()` runs inside `finally` blocks and `__exit__`, so an escaping `getattr` would replace
    the exception that actually explains the run — the same failure this method exists to prevent,
    arriving by a different route. And the other handles must still be released: one unreadable
    store cannot be allowed to strand the locks belonging to the others.
    """
    from benchmarks.systems import Mem0System

    class _HostileStore:
        @property
        def client(self) -> Any:
            raise ValueError("store exploded")

    good = _RecordingClient()
    memory = _RecordingMemory(client=good)
    memory._telemetry_vector_store = _HostileStore()  # type: ignore[assignment]
    system = Mem0System("sk-x", model="openai/gpt-4o-mini")
    system._mem = memory

    system.close()  # must not propagate

    assert good.closed == 1
    assert memory.closed == 1


def test_the_attribute_path_close_walks_still_exists_in_mem0() -> None:
    """Pin `close()`'s attribute chain against the real package, not against our own stand-in.

    Every other close test injects `_RecordingMemory`, whose shape is asserted by nothing but
    itself. If mem0 renamed `vector_store`, `_telemetry_vector_store` or `.client`, the `getattr`
    chain would quietly yield `None`, `close()` would return cleanly, and all of them would still
    pass while the lock leaked exactly as before.

    Reads mem0's SOURCE rather than importing it. `import mem0` costs ~100s here (it drags in
    torch), which is not a price the default suite should pay, and the attribute NAMES are the
    whole contract under test — text pins them just as well as a code object would.
    """
    spec = importlib.util.find_spec("mem0")
    if spec is None or not spec.origin:
        pytest.skip('needs the bench extra (pip install "recall-rag[bench]")')

    root = Path(spec.origin).parent
    main = (root / "memory" / "main.py").read_text(encoding="utf-8", errors="ignore")
    qdrant = (root / "vector_stores" / "qdrant.py").read_text(encoding="utf-8", errors="ignore")

    for attribute in ("vector_store", "_telemetry_vector_store", "_entity_store"):
        assert re.search(rf"self\.{attribute}\s*=", main), f"mem0 no longer sets Memory.{attribute}"
    assert re.search(r"self\.client\s*=", qdrant), "mem0's Qdrant store no longer exposes `client`"


@pytest.mark.filterwarnings(r"ignore:unclosed file.*\.lock:ResourceWarning")
def test_close_releases_the_real_qdrant_storage_lock(tmp_path: Path) -> None:
    """The end-to-end proof, against the real lock rather than a mock of it.

    This is the defect that produced `RuntimeError: Storage folder ... is already accessed by
    another instance of Qdrant client` on 2026-08-05 and was misread as a billing failure: the
    embedded Qdrant client holds an exclusive `portalocker` lock for as long as it lives, and
    `Mem0System` had no way to let go of it.

    Uses a real `QdrantClient` but NOT a real `mem0.Memory`, deliberately: building one loads the
    HuggingFace embedder (~56s here). The lock is the thing under test, and it is real.

    The `ResourceWarning` filter is for an UPSTREAM leak, not ours. It is matched on the `.lock`
    message rather than on the category, and scoped to this one test rather than the module, so a
    leak of any OTHER handle here still fails a `-W error` run: a blanket `ignore::ResourceWarning`
    would have made this test the one place a real leak could hide. qdrant-client opens the `.lock`
    file and only then attempts to lock it, so the constructor that loses the race below raises
    without closing its own handle. Verified in isolation: a plain open/close cycle is clean under
    `-W error`, and the warning appears exactly at the contention assertion.
    """
    qdrant_client = pytest.importorskip("qdrant_client")
    from benchmarks.systems import Mem0System

    path = tmp_path / "qdrant"
    path.mkdir()
    held = _RecordingMemory(client=qdrant_client.QdrantClient(path=str(path)))
    system = Mem0System("sk-x", model="openai/gpt-4o-mini")
    system._mem = held

    # Precondition: the lock really is held, so the assertion after `close()` means something.
    with pytest.raises(RuntimeError, match="already accessed"):
        qdrant_client.QdrantClient(path=str(path))

    system.close()

    reopened = qdrant_client.QdrantClient(path=str(path))  # must not raise
    reopened.close()
    assert held.closed == 1  # and the SQLite history handle was released too


def test_close_releases_both_handles_not_just_the_memory() -> None:
    """`mem0.Memory.close()` closes its SQLite history db and NOTHING else — it does not touch the
    vector store. A `close()` that only forwarded to it would look right and leak the lock."""
    from benchmarks.systems import Mem0System

    client = _RecordingClient()
    system = Mem0System("sk-x", model="openai/gpt-4o-mini")
    system._mem = _RecordingMemory(client=client)

    system.close()

    assert client.closed == 1
    assert system._mem is None


def test_close_is_idempotent_and_harmless_before_any_memory_exists() -> None:
    """Teardown runs in `finally` blocks, after failures, and sometimes twice. None of that may
    raise: a teardown error would replace the exception that actually explains the run."""
    from benchmarks.systems import Mem0System

    system = Mem0System("sk-x", model="openai/gpt-4o-mini")
    system.close()  # never ingested, so `_mem` was never built

    memory = _RecordingMemory()
    system._mem = memory
    system.close()
    system.close()
    assert memory.closed == 1  # the second call is a no-op, not a double close


@pytest.mark.parametrize("broken", ["client", "memory"])
def test_one_handle_refusing_to_close_does_not_strand_the_other(broken: str) -> None:
    """The lock is the handle that causes cross-test damage, so releasing it must not be
    conditional on the SQLite handle closing cleanly, or the other way round."""
    from benchmarks.systems import Mem0System

    client = _RecordingClient(explode=broken == "client")
    memory = _RecordingMemory(client=client, explode=broken == "memory")
    system = Mem0System("sk-x", model="openai/gpt-4o-mini")
    system._mem = memory

    # Must not propagate — but must not be silent either. A release that fails leaves the next run
    # facing the same baffling "already accessed" error, so the warning is the breadcrumb that
    # keeps a second incident from being investigated from scratch, as this one was.
    with pytest.warns(UserWarning, match="Mem0System.close"):
        system.close()

    assert client.closed == 1
    assert memory.closed == 1
    assert system._mem is None


def test_the_context_manager_closes_even_when_the_body_raises() -> None:
    """`with Mem0System(...) as system:` is the shape that makes the leak hard to reintroduce.

    The 2026-08-05 leak happened on the failure path specifically — `ingest()` raised, and the
    handle stayed open — so the exception case is the one worth pinning.
    """
    from benchmarks.systems import Mem0System

    memory = _RecordingMemory()
    system = Mem0System("sk-x", model="openai/gpt-4o-mini")
    with pytest.raises(ValueError, match="boom"):
        with system as entered:
            assert entered is system
            entered._mem = memory
            raise ValueError("boom")

    assert memory.closed == 1
    assert system._mem is None


#: A 402 says the ACCOUNT is out of credit. It says nothing about the code under test, so failing
#: on it reports a billing state as a regression — the same shape as the DeepSeek PR-review job
#: that was disabled for returning a red check when its credits ran out: infrastructure dressed as
#: a verdict.
#:
#: Deliberately narrow. Skipping on any exception would turn a real `Mem0System` regression into
#: silence, which is worse than the noise it removes. Only "the account cannot pay" qualifies —
#: a rate limit, a bad response, or a changed API still fail.
#:
#: The evidence below is a WIRE FACT, not a sentence: the HTTP status and OpenRouter's own
#: `limit_source` taxonomy. Provider prose gets rewritten without notice, so a matcher keyed on
#: the sentence is a matcher with an expiry date. `"insufficient credits"` survives only as a
#: fallback for a layer that stringifies the human message and drops the status with it.
#:
#: Narrowness is now enforced in THREE places, and all three must be read before judging this
#: guard's blast radius: this tuple (message text), `_402_IN_TEXT` (a 402 spelled out in a
#: message), and `_carries_a_402_status` (attribute names, walked over the `__cause__` chain by
#: `_the_billing_evidence_in`).
_CANNOT_PAY = (
    "openrouter_credits",    # `metadata.limit_source` — outlives any rewording of `message`
    "insufficient credits",  # today's prose; the fallback, not the primary evidence
)

#: A 402 written into a message rather than left on an attribute. One anchored pattern instead of
#: a hand-maintained list of spellings, for two reasons. It is the difference between a marker
#: list that grows by one entry per provider punctuation change, and a rule; and the substring
#: form was actively WRONG — `"'code': 402"` matches `{'code': 4029}`, so an unrelated upstream
#: failure carrying a four-digit vendor code was skipped as a billing refusal. `(?!\d)` is the
#: whole fix, and the `{'code': 4029}` / `Error code: 4020` cases in `test_everything_else_still_fails`
#: pin it.
_402_IN_TEXT = re.compile(
    r"""(?:
            error[ ]code:[ ]*          # openai builds this in `_base_client._make_status_error`,
                                       # and mem0 folds it into its `LLMError` message
          | status[ _]code[=: ][ ]*    # requests-/urllib-flavoured stringifications, and the
                                       # plain-English "status code 402" a wrapper may write
          | ['"]code['"]:[ ]*          # OpenRouter's JSON error body, repr'd into a message
        )402(?!\d)""",
    re.VERBOSE,
)


def _is_402(value: object) -> bool:
    # Bools are ints in Python. `True == 402` is already False, so this guard changes no outcome
    # today; it is here so that a future change from `== 402` to anything truthiness-shaped cannot
    # silently reinterpret a flag named `code` as a payment refusal.
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return value == 402
    return isinstance(value, str) and value.strip() == "402"


def _read(obj: object, name: str) -> object:
    """`getattr` that cannot itself raise.

    `getattr(x, name, None)` swallows only `AttributeError`, but the thing being probed is an
    arbitrary exception from an arbitrary library, and a property is free to raise anything
    (`aiohttp.ClientResponseError.code` is deprecated, so it raises under `-W error`). If that
    escaped, it would do so from inside the smoke test's `except` block, replacing the real
    diagnostic with a red herring. Unreadable must mean "no evidence", never "new exception".
    """
    try:
        return getattr(obj, name, None)
    except Exception:  # noqa: BLE001 - a probe that raises has told us nothing, not something
        return None


def _carries_a_402_status(exc: BaseException) -> bool:
    """True if `exc` exposes an HTTP 402 as a STATUS, on itself or on its `response`.

    `openai.APIStatusError` sets both `status_code` and `response.status_code`; `urllib`'s
    `HTTPError` spells it `code`. Reading the attribute rather than the message is the whole point:
    a wrapper is free to re-word, truncate, or discard the text, but it cannot re-word a 402.
    """
    candidates = [_read(exc, "status_code"), _read(exc, "http_status")]
    response = _read(exc, "response")
    if response is not None:
        candidates.append(_read(response, "status_code"))
    # `code` is read ONLY off a `headers`-bearing carrier, which is the `urllib.error.HTTPError`
    # shape — the sole reason `code` is probed at all. Unqualified it is the loosest name of the
    # three and means different things elsewhere: `SystemExit(402).code` is an exit status. And
    # qualifying on `response` instead would readmit exactly the family this exists to exclude,
    # since openai puts the response BODY's error code in `code`, not the transport status — a
    # 500 whose body carries `code: 402` would then skip as a billing refusal.
    if _read(exc, "headers") is not None:
        candidates.append(_read(exc, "code"))
    return any(_is_402(candidate) for candidate in candidates)


def _the_billing_evidence_in(exc: BaseException) -> str | None:
    """The first proof that `exc` — or something it was explicitly raised FROM — is a 402.

    Walks `__cause__` only, never `__context__`. `raise X from Y` is a claim that Y caused X;
    `__context__` is merely "Y happened to be in flight when X was raised", which is a coincidence
    of timing. Following it would mean any failure occurring inside an `except` block handling a
    402 gets reported as a billing skip — turning this guard into the catch-all it exists not to
    be.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))  # a hand-built `__cause__` chain can be cyclic; do not loop forever
        if _carries_a_402_status(current):
            return "HTTP 402"
        try:
            text = str(current).lower()
        except Exception:  # noqa: BLE001 - same rule as `_read`: unreadable means no evidence
            text = ""
        if _402_IN_TEXT.search(text):
            return "402 in the message"
        for marker in _CANNOT_PAY:
            if marker in text:
                return marker
        current = current.__cause__
    return None


def _skip_if_the_account_cannot_pay(exc: BaseException) -> None:
    """`pytest.skip` iff `exc` is a billing refusal. Returns normally otherwise, so callers re-raise.

    The skip is loud on purpose: `pytest -rs` names it, and the reason says to top up rather than
    to debug. A silent pass would let this smoke test rot unnoticed the day the balance runs out.
    The matched evidence goes in the reason too, so a future widening can be judged from the skip
    line alone rather than by re-reading this function.
    """
    evidence = _the_billing_evidence_in(exc)
    if evidence is not None:
        pytest.skip(
            f"OpenRouter account is out of credit, not a code failure "
            f"(matched {evidence}): {exc}"
        )


#: This test calls OpenRouter for real and spends credit on every run. Gating it on the API key
#: alone meant that anyone who had configured a key — which is everyone who uses the cloud
#: embedder or the extraction engine — paid for it on every `pytest tests/`, including runs that
#: had nothing to do with benchmarks. Opt in explicitly instead:
#:
#:     RECALL_RUN_PAID_TESTS=1 python -m pytest tests/test_bench_systems.py
#:
#: The key check stays, because the opt-in says "I am willing to spend", not "I have an account".
_PAID_TESTS_OPT_IN = "RECALL_RUN_PAID_TESTS"


@pytest.mark.skipif(
    not (
        os.environ.get(_PAID_TESTS_OPT_IN)
        and os.environ.get("OPENROUTER_API_KEY")
        and _mem0_installed()
    ),
    # NB: the key check verifies a key EXISTS, not that it can pay — and it cannot check that
    # without spending money. The billing case is handled inside the test instead.
    reason=f"spends OpenRouter credit; set {_PAID_TESTS_OPT_IN}=1 with mem0ai + OPENROUTER_API_KEY",
)
def test_mem0_system_smoke() -> None:
    """Round-trip proof, mirroring `test_recall_system_indexes_and_retrieves`.

    A distinctive fact must survive ingest -> retrieve. `isinstance(ctx, str)` alone would pass
    even if `add`/`search` were both no-ops; asserting the fact is actually present asserts the
    real behaviour under test.
    """
    from benchmarks.systems import Mem0System

    conv = {
        "sample_id": "itest-mem0-quokka",
        "conversation": {
            "speaker_a": "Alice",
            "speaker_b": "Bob",
            "session_1_date_time": "1 January 2024",
            "session_1": [
                {
                    "speaker": "Alice",
                    "dia_id": "D1:1",
                    "text": "Our new monitoring code name is quokka-telemetry-4417.",
                },
                {
                    "speaker": "Bob",
                    "dia_id": "D1:2",
                    "text": "Got it, I'll wire up the dashboards for that.",
                },
            ],
        },
    }
    # `with`, not a bare constructor: this test is the site that leaked the Qdrant storage lock on
    # 2026-08-05. It builds with the DEFAULT `run_id="adhoc"`, so the lock it holds is on the one
    # path every ad-hoc `Mem0System` shares — and it held it on BOTH failure paths, the raise and
    # the billing skip. `pytest.skip` raises a `BaseException`, so only `__exit__` (not `except
    # Exception`) is guaranteed to run on the skip path.
    with Mem0System(os.environ["OPENROUTER_API_KEY"], model="openai/gpt-4o-mini") as system:
        try:
            system.ingest(conv)
            ctx = system.retrieve("What is the name of the new monitoring code?")
        except Exception as exc:  # noqa: BLE001 - re-raised below unless it is a billing refusal
            _skip_if_the_account_cannot_pay(exc)
            raise
    assert "quokka-telemetry-4417" in ctx


@pytest.mark.parametrize(
    "message",
    [
        "LLM extraction failed: Error code: 402 - {'error': {'message': 'Insufficient credits.'}}",
        "insufficient credits, add more at openrouter.ai",
        "{'code': 402, 'metadata': {'limit_source': 'openrouter_credits'}}",
    ],
)
def test_a_billing_refusal_skips_rather_than_fails(message: str) -> None:
    with pytest.raises(pytest.skip.Exception):
        _skip_if_the_account_cannot_pay(RuntimeError(message))


@pytest.mark.parametrize(
    "message",
    [
        "Error code: 429 - rate limited",          # environmental, but NOT a billing refusal
        "Error code: 500 - upstream exploded",
        "KeyError: 'results'",                     # a real Mem0System regression
        "assertion failed: quokka-telemetry-4417 not in context",
        # The RuntimeError that actually surfaced on 2026-08-05 and was misread as a second
        # billing shape. It is qdrant-client's storage lock: `Mem0System` never releases the
        # embedded Qdrant client, and every instance built WITHOUT an explicit `run_id` shares the
        # one `adhoc` path — which is what this smoke test does, and what ad-hoc/REPL use does.
        # (The benchmark arms are fine: `benchmarks/run.py` stamps a unique `run_id` per run.)
        # So it is a REAL defect and must stay red: skipping here would trade a resource leak for
        # silence. Reproduced verbatim from `qdrant_client/local/qdrant_local.py`.
        "Storage folder /tmp/recall-bench-mem0/adhoc/qdrant is already accessed by another "
        "instance of Qdrant client. If you require concurrent access, use Qdrant server instead.",
        # Anchoring. A four-digit vendor code beginning 402 is not a 402: OpenRouter forwards
        # upstream provider bodies, and `{'code': 4029}` used to match the substring `'code': 402`
        # and skip. This case is the difference between a rule and a coincidence.
        "upstream error {'code': 4029, 'msg': 'model overloaded'}",
        "Error code: 4020 - vendor bad request",
    ],
)
def test_everything_else_still_fails(message: str) -> None:
    """The guard must not become a catch-all: that trades noise for silence."""
    _the_guard_must_not_skip(RuntimeError(message))


def _the_guard_must_not_skip(exc: BaseException) -> None:
    """Assert the guard leaves `exc` alone, so a real failure keeps propagating.

    Calling `_skip_if_the_account_cannot_pay(exc)` bare does NOT assert this, which is the trap
    every negative test here originally fell into: a WRONG skip raises `pytest.skip.Exception`,
    and pytest reports that as SKIPPED, not FAILED. The whole anti-catch-all net was therefore
    unarmed — a widening that swallowed the Qdrant lock error left the suite at exit code 0. A
    test whose failure mode is "gets skipped" cannot fail, so assert on the matcher instead, and
    convert a stray skip into an explicit failure.
    """
    # `{exc!r}`, not `{exc}`: `BaseException.__repr__` reads `args`, so it still works on a carrier
    # whose `__str__` raises — the very shape `_read`'s totality exists for. `{exc}` would replace
    # this assertion's message with a `TypeError` from formatting it.
    evidence = _the_billing_evidence_in(exc)
    assert evidence is None, f"guard matched {evidence!r} on a NON-billing failure: {exc!r}"
    try:
        _skip_if_the_account_cannot_pay(exc)
    except pytest.skip.Exception as skipped:  # pragma: no cover - the assert above gets there first
        pytest.fail(f"guard skipped a non-billing failure: {skipped}")


class _StubResponse:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


class _StubAPIStatusError(Exception):
    """Duck-types the attributes the matcher reads off `openai.APIStatusError`.

    A stub rather than the real class so this file keeps working without the `bench` extra
    (`openai` is not a core dependency). `test_a_real_openai_402_is_recognised` pins it against
    the genuine article, but only where the bench extra is installed — CI installs `.[dev]`, so
    that pin catches drift locally and NOT in CI. Do not read it as an unconditional guarantee.
    """

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response = _StubResponse(status_code)


class _StubWrappedError(Exception):
    """A carrier that exposes the status ONLY nested under `response`, as httpx-shaped errors do.

    Separate from `_StubAPIStatusError` because that one sets `status_code` too, and `any()`
    short-circuits: with only that stub, the `response.status_code` branch could be deleted and
    no test would notice.
    """

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.response = _StubResponse(status_code)


class _StubStatusOnlyError(Exception):
    """A carrier with ONLY a top-level `status_code` and no `response`."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class _StubOldSdkError(Exception):
    """A carrier that spells the status `http_status`, as openai 0.x and its contemporaries did."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.http_status = status_code


def _the_shape_that_skipped() -> BaseException:
    """Shape A: mem0 2.x's `LLMError` text, as seen skipping correctly on 2026-08-05.

    `mem0/memory/main.py` does `raise LLMError(f"LLM extraction failed: {e}") from e`, so the
    openai message ("Error code: 402 - {...}", composed in `openai/_base_client.py` rather than by
    any `__str__` on the exception class) lands inside the text.
    """
    return RuntimeError(
        "LLM extraction failed: Error code: 402 - {'error': {'message': 'Insufficient credits. "
        "Add more using https://openrouter.ai/settings/credits', 'code': 402, 'metadata': "
        "{'limit_source': 'openrouter_credits'}}}"
    )


def _a_status_only_shape() -> BaseException:
    """Shape B: the SAME billing condition behind a wrapper that kept none of the prose.

    This is the shape the prose-only matcher could not see. Nothing in the wrapper's message says
    "credits", "402", or anything else quotable — but the 402 is still on the chained cause,
    because `raise ... from ...` preserves it. Matching the STATUS rather than the sentence is
    what makes the guard survive a provider (or a wrapper) rewording its text.

    Constructed, not observed. The unrecognised error of 2026-08-05 was NOT this: it was the
    Qdrant storage lock, which is not billing at all and is pinned as a must-stay-red case above.
    This shape is the insurance the incident argued for, not a transcript of it.
    """
    wrapper = RuntimeError("Stopped: the memory backend refused the request")
    wrapper.__cause__ = _StubAPIStatusError("Payment Required", status_code=402)
    return wrapper


def _a_nested_status_only_shape() -> BaseException:
    """Shape B again, with the status reachable only via `response.status_code`."""
    wrapper = RuntimeError("Stopped: the memory backend refused the request")
    wrapper.__cause__ = _StubWrappedError("Payment Required", status_code=402)
    return wrapper


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(_the_shape_that_skipped, id="prose"),
        pytest.param(_a_status_only_shape, id="status-only"),
        pytest.param(_a_nested_status_only_shape, id="status-only-nested"),
    ],
)
def test_both_shapes_of_one_billing_condition_skip(build: Callable[[], BaseException]) -> None:
    """One condition, two shapes. The guard's whole promise is that BOTH skip, not just the one
    whose wording happened to be in the marker list on the day it was written."""
    with pytest.raises(pytest.skip.Exception):
        _skip_if_the_account_cannot_pay(build())


def test_a_real_openai_402_is_recognised() -> None:
    """Pin `_StubAPIStatusError` against the genuine `openai.APIStatusError`.

    Without this, the stub could keep passing long after openai renamed the attribute the matcher
    reads — a test that proves only that the stub matches itself. Both duck-typed attributes are
    asserted, since the matcher short-circuits on the first and would otherwise hide a rename of
    the second.
    """
    openai = pytest.importorskip(
        "openai", reason='needs the bench extra (pip install "recall-rag[bench]")'
    )
    httpx = pytest.importorskip("httpx", reason="openai's transport dep; only missing if it drops it")

    body = {"error": {"message": "Insufficient credits.", "code": 402}}
    response = httpx.Response(
        402,
        request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
        json=body,
    )
    real = openai.APIStatusError("Error code: 402", response=response, body=body)
    wrapper = RuntimeError("Stopped: the memory backend refused the request")
    wrapper.__cause__ = real

    assert real.status_code == 402                 # what `_StubAPIStatusError` duck-types
    assert real.response.status_code == 402        # what `_StubWrappedError` duck-types
    with pytest.raises(pytest.skip.Exception):
        _skip_if_the_account_cannot_pay(wrapper)


@pytest.mark.parametrize(
    "message",
    [
        # One case per marker, each carrying ONLY that marker, so none of them can be deleted
        # without a test going red. The status walk does NOT cover these: a wrapper that folds
        # `str(e)` into its own message keeps no attribute for the walk to read.
        "backend returned status code 402 from upstream",
        "HTTPError(status_code=402, detail='no funds')",
        'upstream said {"code": 402}',
        "upstream refused: {'metadata': {'limit_source': 'openrouter_credits'}}",
        "the account has insufficient credits",
    ],
)
def test_each_message_only_marker_earns_its_place(message: str) -> None:
    """`metadata.limit_source` is OpenRouter's own taxonomy, so it outlives any rewording of
    `message` — the reason it is a marker even though no human ever reads it. The same argument
    applies to each 402 spelling: untested markers get deleted by the next reader as redundant."""
    with pytest.raises(pytest.skip.Exception):
        _skip_if_the_account_cannot_pay(RuntimeError(message))


@pytest.mark.parametrize(
    "carrier",
    [
        pytest.param(_StubStatusOnlyError, id="status_code"),
        pytest.param(_StubOldSdkError, id="http_status"),
        pytest.param(_StubWrappedError, id="response.status_code"),
    ],
)
def test_each_probed_attribute_name_earns_its_place(
    carrier: Callable[..., BaseException],
) -> None:
    """One minimal carrier per probed name, each exposing ONLY that name.

    `_StubAPIStatusError` sets `status_code` AND `response.status_code`, and `any()`
    short-circuits, so on its own it lets either probe be deleted unnoticed. The argument that
    gives every message marker its own case applies just as much to every attribute name.
    """
    with pytest.raises(pytest.skip.Exception):
        _skip_if_the_account_cannot_pay(carrier("Payment Required", status_code=402))


def test_a_body_error_code_does_not_override_the_transport_status() -> None:
    """A 500 whose BODY carries `code: 402` is an upstream failure, not a billing refusal.

    This is the case that decides the qualifier in `_carries_a_402_status`: reading `code` off
    anything with a `response` would readmit the openai family, where `code` is the body's error
    code. Reading it only off a `headers`-bearing carrier keeps the urllib case and drops this one.
    """

    class _BodyCode(Exception):
        def __init__(self) -> None:
            super().__init__("upstream exploded")
            self.response = _StubResponse(500)
            self.code = 402

    _the_guard_must_not_skip(_BodyCode())


def test_a_bare_code_attribute_is_not_an_http_status() -> None:
    """`code` means too many things to read unqualified. `SystemExit(402).code` is an exit status.

    Not reachable from the smoke test today (`except Exception` does not catch `SystemExit`), but
    the matcher is the thing under test here, not its current caller.
    """
    _the_guard_must_not_skip(SystemExit(402))


def test_a_urllib_style_http_error_still_matches_on_code() -> None:
    """The qualifier must not cost the case `code` was probed for: `urllib.error.HTTPError` spells
    the status `code` and carries `headers`, which is what marks it as an HTTP carrier."""
    import urllib.error

    error = urllib.error.HTTPError(
        "https://openrouter.ai/api/v1/chat/completions", 402, "Payment Required", {}, None
    )
    # `HTTPError` is file-like (it subclasses `addinfourl`), so an un-closed one emits a
    # `ResourceWarning` on collection — a real leak in this test, not a third-party wart, and it
    # fails the suite under `-W error`.
    try:
        assert error.code == 402
        with pytest.raises(pytest.skip.Exception):
            _skip_if_the_account_cannot_pay(error)
    finally:
        error.close()


def test_a_probe_that_raises_is_treated_as_no_evidence() -> None:
    """A raising property or `__str__` must not escape the guard.

    It would do so from inside the smoke test's `except` block, so the `raise` on the next line
    would never run and the real diagnostic would survive only as `__context__`.
    """

    class _Hostile(Exception):
        # `status_code`, not `code`: `code` is only read off an HTTP-shaped carrier, so a probe
        # that explodes there is never reached and would pin nothing. This is the name the walk
        # reads first and unconditionally.
        @property
        def status_code(self) -> int:
            raise ValueError("property exploded")

        def __str__(self) -> str:
            raise TypeError("str exploded")

    _the_guard_must_not_skip(_Hostile())


def test_a_cyclic_cause_chain_terminates() -> None:
    """The `seen` set is the only thing standing between a hand-built cycle and a hung suite."""
    first = RuntimeError("first")
    second = RuntimeError("second")
    first.__cause__ = second
    second.__cause__ = first
    _the_guard_must_not_skip(first)


def test_a_non_402_status_on_the_chain_still_fails() -> None:
    """The walk reads the status, not merely the presence of one. 429 is environmental too, but it
    is not a billing refusal and must stay red."""
    wrapper = RuntimeError("Stopped: the memory backend refused the request")
    wrapper.__cause__ = _StubAPIStatusError("Too Many Requests", status_code=429)
    _the_guard_must_not_skip(wrapper)


def test_an_unrelated_failure_raised_while_a_402_was_in_flight_still_fails() -> None:
    """`__cause__` is a claim ("this caused that"); `__context__` is a coincidence of timing.

    Only the claim counts. Following `__context__` would mean that ANY failure occurring inside an
    `except` block that is handling a 402 gets reported as a billing skip — which is exactly how
    the leaked Qdrant lock would have been buried had the guard been widened to swallow it.
    """
    unrelated = RuntimeError(
        "Storage folder /tmp/recall-bench-mem0/adhoc/qdrant is already accessed by another "
        "instance of Qdrant client."
    )
    unrelated.__context__ = _StubAPIStatusError("Insufficient credits", status_code=402)
    _the_guard_must_not_skip(unrelated)


def test_resolve_embedder_routes_fastembed_prefix_to_a_named_model(monkeypatch: pytest.MonkeyPatch) -> None:
    # The retrieval-quality ablation depends on this route reaching FastEmbedEmbedder with the
    # model name after the prefix. Monkeypatch the class so the test never downloads a model.
    captured: dict[str, Any] = {}

    class _FakeFastEmbed:
        def __init__(self, model_name: str) -> None:
            captured["model_name"] = model_name

    import recall.embeddings

    monkeypatch.setattr(recall.embeddings, "FastEmbedEmbedder", _FakeFastEmbed)
    resolve_embedder("fastembed:BAAI/bge-large-en-v1.5")
    assert captured["model_name"] == "BAAI/bge-large-en-v1.5"


def test_resolve_embedder_defers_other_names_to_the_shared_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, str] = {}

    import recall.eval.locomo

    monkeypatch.setattr(
        recall.eval.locomo, "_make_embedder", lambda name: seen.setdefault("name", name)
    )
    resolve_embedder("hashing")
    assert seen["name"] == "hashing"


def test_resolve_embedder_routes_voyage_prefix_to_a_named_current_model(monkeypatch: pytest.MonkeyPatch) -> None:
    # The bare "voyage" route resolves to VoyageEmbedder's legacy voyage-3 default; "voyage:MODEL"
    # pins a current model. Monkeypatch so the test never needs VOYAGE_API_KEY or the network.
    captured: dict[str, Any] = {}

    class _FakeVoyage:
        def __init__(self, model: str) -> None:
            captured["model"] = model

    import recall.embeddings

    monkeypatch.setattr(recall.embeddings, "VoyageEmbedder", _FakeVoyage)
    resolve_embedder("voyage:voyage-4-large")
    assert captured["model"] == "voyage-4-large"


def test_mem0_config_uses_the_given_huggingface_model_and_its_dim() -> None:
    from benchmarks.systems import mem0_config

    cfg = mem0_config("sk-or-x", "openai/gpt-4o-mini", hf_model="BAAI/bge-large-en-v1.5")
    assert cfg["embedder"]["provider"] == "huggingface"
    assert cfg["embedder"]["config"]["model"] == "BAAI/bge-large-en-v1.5"
    # the Qdrant collection must be created at the model's real width, not the bge-small default
    assert cfg["vector_store"]["config"]["embedding_model_dims"] == 1024


def test_mem0_config_defaults_to_bge_small_384() -> None:
    from benchmarks.systems import mem0_config

    cfg = mem0_config("sk-or-x", "openai/gpt-4o-mini")
    assert cfg["embedder"]["config"]["model"] == "BAAI/bge-small-en-v1.5"
    assert cfg["vector_store"]["config"]["embedding_model_dims"] == 384
