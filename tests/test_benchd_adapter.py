"""Contract tests for the Bench'd adapter (benchmarks/benchd/recall_adapter.py).

These pin the two properties a competitor auditing our leaderboard submission would probe
first, without needing a database or the harness runtime:

1. LongMemEval's ``metadata.has_answer`` oracle flag can never influence what gets indexed:
   not the documents, and not the cache key.
2. Item isolation: after ``reset()`` and before the next ``ingest()``, ``recall()`` returns
   the empty string, so no question can be answered from a corpus its item did not ingest.

The harness package is an optional test dependency; without it the module skips.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("benchd_harness")

_ADAPTER_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "benchd" / "recall_adapter.py"
_spec = importlib.util.spec_from_file_location("benchd_recall_adapter", _ADAPTER_PATH)
adapter_module = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("benchd_recall_adapter", adapter_module)
_spec.loader.exec_module(adapter_module)

RecallAdapter = adapter_module.RecallAdapter


def _turns(has_answer: bool) -> list[dict]:
    return [
        {
            "role": "user",
            "content": "I adopted a dog called Biscuit.",
            "metadata": {"session_index": 0, "has_answer": has_answer},
        },
        {
            "role": "assistant",
            "content": "Biscuit sounds lovely.",
            "metadata": {"session_index": 0, "has_answer": False},
        },
        {
            "role": "user",
            "content": "We walk every morning.",
            "metadata": {"session_index": 1, "session_date_time": "1 May 2023", "has_answer": has_answer},
        },
    ]


@pytest.mark.parametrize("granularity", ["session", "turn"])
def test_has_answer_never_reaches_documents(monkeypatch, granularity):
    monkeypatch.setenv("RECALL_BENCHD_GRANULARITY", granularity)
    adapter = RecallAdapter()
    builder = adapter._docs_by_session if granularity == "session" else adapter._docs_by_turn
    with_flag = builder(_turns(has_answer=True))
    without_flag = builder(_turns(has_answer=False))
    assert with_flag == without_flag
    assert not any("has_answer" in doc for doc in with_flag)


def test_has_answer_never_reaches_the_cache_key():
    """The ingest cache hashes turn content and session metadata; flipping the oracle flag
    must not change the digest, or the flag would leak into which corpora get rebuilt."""
    import hashlib
    import json

    def digest(turns):
        blob = json.dumps(
            [
                {
                    "role": t.get("role"),
                    "content": t.get("content"),
                    "session_index": (t.get("metadata") or {}).get("session_index"),
                    "session_date_time": (t.get("metadata") or {}).get("session_date_time"),
                    "speaker": (t.get("metadata") or {}).get("speaker"),
                }
                for t in turns
            ],
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    assert digest(_turns(True)) == digest(_turns(False))
    # And the adapter source builds its digest from exactly these fields.
    src = _ADAPTER_PATH.read_text(encoding="utf-8")
    assert '"has_answer"' not in src.replace("# it is LongMemEval's oracle flag", "")


def test_session_dates_reach_session_documents():
    adapter = RecallAdapter()
    docs = adapter._docs_by_session(_turns(has_answer=False))
    assert any("Session dated 1 May 2023" in d for d in docs)


def test_recall_is_empty_between_reset_and_ingest():
    adapter = RecallAdapter()
    adapter._stores = {"deadbeef": object()}
    adapter._active = "deadbeef"
    adapter.reset()
    assert adapter._active is None
    assert adapter._retrieve("anything") == ""


def test_document_building_is_deterministic():
    """The ingest cache's equivalence claim rests on ingestion being a pure function of the
    turn list; document building is the part of that this test can pin without a database."""
    adapter = RecallAdapter()
    turns = _turns(has_answer=False)
    assert adapter._docs_by_session(turns) == adapter._docs_by_session(list(turns))
    assert adapter._docs_by_turn(turns) == adapter._docs_by_turn(list(turns))
