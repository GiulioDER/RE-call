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
from pathlib import Path
from typing import Any

import pytest

from benchmarks.systems import BENCH_TABLE, MemorySystem, sample_id_of, tenant_for


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

    Parses the real file `write_conversation_corpus` wrote (``# speaker — date`` header, then a
    ``speaker: text`` body and an optional image line) back into named fields, so the comparison
    against Mem0 is field-by-field and independent of either side's formatting.
    """
    header, sep, body = document.partition("\n\n")
    assert sep and header.startswith("# "), f"unexpected document shape: {document!r}"
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


@pytest.mark.skipif(
    not (os.environ.get("OPENROUTER_API_KEY") and _mem0_installed()),
    reason="needs mem0ai + OPENROUTER_API_KEY",
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
    system = Mem0System(os.environ["OPENROUTER_API_KEY"], model="openai/gpt-4o-mini")
    system.ingest(conv)
    ctx = system.retrieve("What is the name of the new monitoring code?")
    assert "quokka-telemetry-4417" in ctx
