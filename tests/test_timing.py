import time

from recall.timing import TimedEmbedder, TimedReranker, TimingStats, timed_call
from recall.types import Chunk, ScoredChunk


def test_timed_call_records_elapsed_and_call_count():
    stats = TimingStats()
    calls = {"n": 0}

    def work():
        calls["n"] += 1
        time.sleep(0.01)
        return calls["n"]

    assert timed_call(stats, work) == 1
    assert timed_call(stats, work) == 2
    assert stats.calls == 2
    assert stats.total_ms >= 20.0  # two ~10ms sleeps
    assert stats.last_ms >= 10.0
    assert stats.mean_ms == stats.total_ms / 2


def test_timed_call_records_even_on_exception():
    stats = TimingStats()

    def boom():
        raise RuntimeError("nope")

    try:
        timed_call(stats, boom)
    except RuntimeError:
        pass
    assert stats.calls == 1  # the failed call still cost time and is counted


def test_timing_stats_mean_is_zero_before_any_call():
    assert TimingStats().mean_ms == 0.0


class _Emb:
    dim = 2
    name = "fake"

    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


def test_timed_embedder_counts_calls_and_preserves_output():
    emb = TimedEmbedder(_Emb())
    assert emb.dim == 2 and emb.name == "fake"  # interface preserved
    out = emb.embed(["a", "b"])
    assert out == [[1.0, 0.0], [1.0, 0.0]]
    assert emb.stats.calls == 1  # one embed() batch call recorded


class _RR:
    def rerank(self, query, hits):
        return list(reversed(hits))


def test_timed_reranker_counts_calls_and_preserves_output():
    rr = TimedReranker(_RR())
    hits = [ScoredChunk(Chunk("a", "f", "x"), 0.1), ScoredChunk(Chunk("b", "f", "y"), 0.2)]
    out = rr.rerank("q", hits)
    assert [h.chunk.id for h in out] == ["b", "a"]
    assert rr.stats.calls == 1
