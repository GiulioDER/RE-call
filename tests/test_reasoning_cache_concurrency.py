"""Regression proof for BUG-001 from the PR 729 through 720 CCA audit.

Invariant: concurrent public reasoning requests must survive eviction of a full
validation cache. The scheduling dictionary pauses after each eviction key is
selected, exposing the real select/remove interleaving without replacing cache
semantics. Baseline d49eb8dbd raises KeyError in recall.reasoning._checked; the
test captures it and fails the errors assertion. The same test must pass after
cache synchronization, without weakening the graph identity checks.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import threading

import recall.reasoning as reasoning
from recall.reasoning_graph import build_reasoning_graph
from recall.types import (
    Chunk, Provenance, RetrievalDiagnostics, StalenessReport, TrustedHit,
    TrustedResult, Validity,
)


def _request(graph):
    now = datetime(2026, 10, 29, tzinfo=timezone.utc)
    chunk = Chunk("c1", "owner.md", "Ada owns rollout.", {"file": "owner.md", "ord": 0})
    retrieval = TrustedResult(
        query="Who owns rollout?",
        hits=[TrustedHit(
            chunk, 0.91, 0.97, "ok",
            Provenance(chunk.source, "owner.md", 0, now), Validity(now, None, None),
        )],
        abstained=False, reason="", gap_warning=False,
        staleness=StalenessReport(False, now, timedelta(0), timedelta(days=1)),
        diagnostics=RetrievalDiagnostics(index_generation="gen-1"),
        calibration_id="cal-1", calibration_status="certified", tenant_id="acme",
        generation_id="gen-1", pipeline_fingerprint="pipe-a",
    )
    return reasoning.ReasoningRequest(
        query=retrieval.query, tenant_id="acme",
        generation=reasoning.GenerationSelection("gen-1", "pipe-a"),
        policy=reasoning.ReasoningPolicy(name="proposal_assisted"),
        providers=reasoning.ReasoningProviderPorts(
            retriever=lambda request: retrieval,
            graph_provider=lambda request, result: graph,
            answer_provider=lambda system, user: {
                "answer": "Ada owns rollout.", "citations": ["c1"],
                "insufficient_evidence": False,
            },
        ),
        as_of=now,
    )


def test_concurrent_public_reasoning_survives_cache_eviction(monkeypatch):
    class ScheduledCache(dict):
        def __init__(self):
            super().__init__()
            self.arrivals = 0
            self.arrivals_lock = threading.Lock()
            self.both_arrived = threading.Event()

        def pop(self, key, *default):
            with self.arrivals_lock:
                self.arrivals += 1
                if self.arrivals == 2:
                    self.both_arrived.set()
            # A serialized eviction proceeds when the other caller cannot enter.
            # The wait itself is not asserted and never raises a timeout error.
            self.both_arrived.wait(timeout=0.5)
            return super().pop(key, *default)

    cache = ScheduledCache()
    monkeypatch.setattr(reasoning, "_CHECKED_GRAPHS", cache)
    chunk = Chunk("c1", "owner.md", "Ada owns rollout.", {"file": "owner.md", "ord": 0})
    graphs = [build_reasoning_graph(
        [chunk], tenant_id="acme", generation_id="gen-1",
        pipeline_fingerprint="pipe-a", include_text=True,
    ) for _ in range(reasoning._CHECKED_GRAPHS_MAX + 2)]
    for graph in graphs[:-2]:
        assert reasoning.reason(_request(graph)).outcome == "answered"
    assert len(cache) == reasoning._CHECKED_GRAPHS_MAX

    def run(graph):
        try:
            return reasoning.reason(_request(graph))
        except Exception as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, graphs[-2:]))
    errors = [result for result in results if isinstance(result, Exception)]
    assert not errors, [repr(error) for error in errors]
    assert all(result.outcome == "answered" for result in results)
    assert cache.arrivals >= 2
    assert len(cache) <= reasoning._CHECKED_GRAPHS_MAX
