"""`MemorySystem` protocol + `RecallSystem` adapter.

The unit test below needs no DB: it only checks that a plain object satisfying the three
`MemorySystem` members is accepted where the protocol is expected. The integration test at the
bottom is the real proof — it exercises `RecallSystem` end-to-end against Postgres and asserts a
distinctive fact actually round-trips through ingest -> retrieve, not just that a string comes
back.
"""
from __future__ import annotations

import os
from typing import Any

import pytest

from benchmarks.systems import MemorySystem


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
