"""`MemorySystem` protocol + `RecallSystem` adapter.

The unit test below needs no DB: it only checks that a plain object satisfying the three
`MemorySystem` members is accepted where the protocol is expected. The integration test at the
bottom is the real proof — it exercises `RecallSystem` end-to-end against Postgres and asserts a
distinctive fact actually round-trips through ingest -> retrieve, not just that a string comes
back.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

import pytest

from benchmarks.systems import MemorySystem


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
    system.ingest({"sample_id": "c1"})
    assert system.retrieve("q") == "ctx for q"
    assert system.name == "fake"


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


def test_conversation_to_messages_mirrors_recall_turn_walk(tmp_path: Path) -> None:
    """Mem0's turn walk must match RecallSystem's exactly, or the benchmark is invalid.

    `write_conversation_corpus` is the function `RecallSystem.ingest` -> `index_conversation`
    actually indexes turns through. This test drives BOTH it and `_conversation_to_messages` off
    the same fixture and checks they agree on which turns survive (a turn missing `dia_id` is
    silently dropped by `write_conversation_corpus`, so `_conversation_to_messages` must drop it
    too) and in what order (numeric session order, not dict insertion order — `session_2` is
    inserted before `session_1` below specifically to catch a naive walk).
    """
    from recall.eval.locomo import write_conversation_corpus

    from benchmarks.systems import _conversation_to_messages

    conversation = {
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

    written = write_conversation_corpus(conversation, tmp_path)
    messages = _conversation_to_messages(conversation)

    # Same COUNT: the dia_id-less turn is dropped by write_conversation_corpus (RecallSystem's
    # indexing path); _conversation_to_messages (Mem0's) must drop it too or corpora diverge.
    assert len(messages) == written == 3
    # Same ORDER: numeric session order (session_1 before session_2), turn order within a session.
    assert [m["content"].splitlines()[0] for m in messages] == [
        "Alice: First session first turn.",
        "Alice: First session second turn.",
        "Bob: Second session first turn.",
    ]
    # Same CONTENT: the image caption RecallSystem embeds in the chunk body must survive too.
    assert "a whiteboard diagram" in messages[1]["content"]
    # Role tracks speaker (Alice == speaker_a -> "user", Bob -> "assistant"), not turn position.
    assert [m["role"] for m in messages] == ["user", "user", "assistant"]


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
