"""Contract tests for the Bench'd adapter (benchmarks/benchd/recall_adapter.py).

These pin the two properties a competitor auditing our leaderboard submission would probe
first, without needing a database or the harness runtime:

1. LongMemEval's ``metadata.has_answer`` oracle flag can never influence what gets indexed:
   not the documents, and not the cache key.
2. Item isolation: after ``reset()`` and before the next ``ingest()``, ``recall()`` returns
   the empty string, so no question can be answered from a corpus its item did not ingest.

The harness base class is stubbed when absent; these unit tests exercise the adapter without its runtime.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from types import SimpleNamespace
from pathlib import Path

import pytest


_ADAPTER_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "benchd" / "recall_adapter.py"
_spec = importlib.util.spec_from_file_location("benchd_recall_adapter", _ADAPTER_PATH)
adapter_module = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("benchd_recall_adapter", adapter_module)
# Only BaseAdapter is needed to define the adapter; no harness behavior is exercised here.
with pytest.MonkeyPatch.context() as _imports:
    try:
        import benchd_harness.adapters.base  # noqa: F401
    except ImportError:
        _base = types.ModuleType("benchd_harness.adapters.base")
        _base.BaseAdapter = object
        _imports.setitem(sys.modules, "benchd_harness", types.ModuleType("benchd_harness"))
        _imports.setitem(sys.modules, "benchd_harness.adapters", types.ModuleType("benchd_harness.adapters"))
        _imports.setitem(sys.modules, "benchd_harness.adapters.base", _base)
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


@pytest.fixture
def ingest_adapter(monkeypatch, tmp_path):
    """Use real ingest/cache logic with only storage and indexing replaced."""
    import recall.index

    calls = []
    deleted = []

    class Store:
        rows = ()

        def close(self):
            pass

        def iter_chunks(self):
            return iter(self.rows)

        def delete_sources(self, sources):
            deleted.extend(sources)
            self.rows = ()

    class Indexer:
        def __init__(self, store, embedder):
            self.store = store

        def index_path(self, workspace, *, files):
            calls.append(tuple(files))
            self.store.rows = tuple(SimpleNamespace(source=str(path)) for path in files)

    monkeypatch.setattr(recall.index, "Indexer", Indexer)
    adapter = RecallAdapter()
    adapter._workspace = tmp_path
    monkeypatch.setattr(adapter, "_open_store", lambda digest: Store())
    return adapter, calls, deleted


def test_has_answer_never_reaches_the_cache_key(ingest_adapter):
    """CODE-001: real ingest must reuse a corpus when only its oracle flag changes.

    Red proof: serialize the complete metadata mapping in RecallAdapter.ingest.
    This exact test must fail its active-digest assertion on that production mutation.
    """
    adapter, calls, _ = ingest_adapter
    adapter.ingest(_turns(True))
    original = adapter._active
    adapter.reset()
    adapter.ingest(_turns(False))
    assert adapter._active == original
    assert len(calls) == 1


def test_disabled_ingest_cache_rebuilds_after_reset(ingest_adapter):
    """BUG-001: baseline 882bd50c2 reuses A after reset with cache disabled.

    Red proof targets RecallAdapter.ingest: the exact test fails len(calls) == 2.
    """
    adapter, calls, deleted = ingest_adapter
    adapter._cache = False
    adapter.reset()
    adapter.ingest(_turns(True))
    adapter.reset()
    adapter.ingest(_turns(True))
    assert len(calls) == 2
    adapter.reset()
    other = _turns(True)
    other[0]["content"] = "A different conversation."
    adapter.ingest(other)
    assert len(calls) == 3
    assert len(adapter._tenants) == len(adapter._stores) == 1
    assert deleted


def test_changed_content_builds_a_distinct_corpus(ingest_adapter):
    """Control: cache reuse must still depend on conversation content."""
    adapter, calls, _ = ingest_adapter
    adapter.ingest(_turns(True))
    original = adapter._active
    other = _turns(True)
    other[0]["content"] = "A different conversation."
    adapter.reset()
    adapter.ingest(other)
    assert adapter._active != original
    assert len(calls) == 2


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
