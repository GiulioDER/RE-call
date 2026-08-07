"""The service-level cost contract: what is spent, in what order, and what is reported.

Five properties, each of which a naive implementation satisfies vacuously and is therefore
asserted against a discriminating fixture rather than a final count:

* an overloaded process refuses BEFORE it embeds — the ordering is the whole point of admission
  control, and an admission gate placed after the embedder would still produce a rejection;
* one reranker per worker, built once even under a concurrent first-request storm;
* every stage of a retrieval is timed, including evidence assembly, which was the one bracket the
  response surface did not carry;
* reranking reorders and never rewrites the dense cosine the trust layer calibrates against;
* no query text and no corpus text reaches any log record or exception message.
"""
from __future__ import annotations

import json
import logging
import threading
import time

import pytest

from recall.observability import METRICS
from recall.profiles import RetrievalOverloaded, resolve_retrieval_profile
from recall.types import Chunk

from tests.conftest import dev_search_memory, requires_db


class CountingEmbedder:
    """Records every call so a test can assert that nothing was spent."""

    dim = 3
    name = "counting"

    def __init__(self, mapping: dict[str, list[float]], default: list[float]) -> None:
        self._mapping, self._default = mapping, default
        self.calls = 0
        self.texts: list[str] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        self.texts.extend(texts)
        return [self._mapping.get(t, self._default) for t in texts]


class ReversingReranker:
    """A reranker with a visible, order-only effect and no scoring of its own."""

    def __init__(self) -> None:
        self.calls = 0

    def rerank(self, query: str, hits):
        self.calls += 1
        return list(reversed(hits))


@pytest.fixture
def bounded_fast_profile(monkeypatch):
    """A fast profile with room for exactly one running request and one queued one."""
    monkeypatch.setenv("RECALL_RETRIEVAL_PROFILE", "fast")
    monkeypatch.delenv("RECALL_RERANK", raising=False)
    monkeypatch.setenv("RECALL_SEARCH_CONCURRENCY", "1")
    monkeypatch.setenv("RECALL_SEARCH_QUEUE", "1")
    return resolve_retrieval_profile()


# --------------------------------------------------------------------------------------------
# Overload rejection happens BEFORE embedding
# --------------------------------------------------------------------------------------------


@requires_db
def test_an_overloaded_process_refuses_before_it_embeds(make_store, bounded_fast_profile):
    """The ordering is the point: a rejection that arrives after the embedding is not a budget.

    The discriminator is `embedder.calls`, not the exception. Moving the admission gate below the
    embedder would still raise `RetrievalOverloaded` and would still look like working overload
    control; it would simply have paid for the query first.
    """
    from recall_mcp.service import _admission

    store = make_store(3)
    store.upsert([Chunk("a", "notes.md", "cats")], [[1.0, 0.0, 0.0]])
    embedder = CountingEmbedder({"cats": [1.0, 0.0, 0.0]}, default=[0.0, 0.0, 1.0])

    admission = _admission(bounded_fast_profile)
    held = threading.Event()
    release = threading.Event()

    def occupy() -> None:
        with admission:
            held.set()
            release.wait(10)

    holder = threading.Thread(target=occupy, daemon=True)
    holder.start()
    assert held.wait(5)
    try:
        with pytest.raises(RetrievalOverloaded) as excinfo:
            dev_search_memory(store, embedder, "cats")
        assert excinfo.value.reason == "budget_exhausted"
        assert embedder.calls == 0, "the query was embedded before the process refused it"
    finally:
        release.set()
        holder.join(10)


@requires_db
def test_a_client_cannot_ask_for_more_hits_than_the_profile_returns(make_store, monkeypatch):
    """`k` is clamped DOWN and never raised. This is the whole of "selection is process level".

    Newly advertised in three places (the service comment, `docs/ENTERPRISE_RETRIEVAL.md`, the
    `recall_search` tool docstring) and, until this test, guarded by nothing: deleting the clamp
    left the suite green. The legacy arm is what makes it discriminating — without it, a store
    that simply held five chunks would satisfy the fast assertion for the wrong reason.
    """
    monkeypatch.delenv("RECALL_RERANK", raising=False)
    store = make_store(3)
    store.upsert(
        [Chunk(f"c{i}", f"n{i}.md", "cats") for i in range(8)],
        [[1.0, 0.0, 0.0]] * 8,
    )
    embedder = CountingEmbedder({"cats": [1.0, 0.0, 0.0]}, default=[0.0, 0.0, 1.0])

    monkeypatch.setenv("RECALL_RETRIEVAL_PROFILE", "fast")
    assert len(dev_search_memory(store, embedder, "cats", k=10).hits) == 5

    monkeypatch.delenv("RECALL_RETRIEVAL_PROFILE")
    assert len(dev_search_memory(store, embedder, "cats", k=10).hits) > 5


def test_the_thread_limit_reaches_the_reranker_not_just_the_profile(monkeypatch):
    """Resolving `inference_threads` is not the same as handing it to the model.

    The line that carries it into `CrossEncoderReranker` sits in a branch marked
    `# pragma: no cover`. `torch.set_num_threads` itself needs the `rerank` extra and stays
    deferred, but the plumbing does not, and an unbounded reranker is exactly the resource bound
    this profile exists to impose.
    """
    import recall.rerank as rerank_module
    import recall_mcp.service as service
    from recall.rerank import PINNED_RERANKER_SHA256

    seen = {}

    class RecordingReranker:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(rerank_module, "CrossEncoderReranker", RecordingReranker)
    service._new_reranker(
        {
            "RECALL_RETRIEVAL_PROFILE": "quality",
            "RECALL_RERANK_PATH": "/opt/recall-enterprise/models/ms-marco-MiniLM-L-6-v2",
            "RECALL_RERANK_SHA256": PINNED_RERANKER_SHA256,
            "RECALL_RERANK_THREADS": "3",
        }
    )
    assert seen["inference_threads"] == 3
    assert seen["local_files_only"] is True
    assert seen["artifact_sha256"] == PINNED_RERANKER_SHA256


@requires_db
def test_a_shed_request_is_not_recorded_as_a_failed_one(make_store, bounded_fast_profile):
    """Shedding is the design working. It must not page as an error, or poison the latency.

    A shed request did no work by construction, so counting it in
    `recall_retrieval_failed_total` makes healthy load shedding indistinguishable from an
    outage, and injecting its budget-length wait into `recall_retrieval_total_ms` contaminates
    the served-latency population with rejections — in exactly the overload regime where the p95
    this program is blocked on would matter most.
    """
    from recall_mcp.service import _admission

    METRICS.reset()
    store = make_store(3)
    store.upsert([Chunk("a", "notes.md", "cats")], [[1.0, 0.0, 0.0]])
    embedder = CountingEmbedder({"cats": [1.0, 0.0, 0.0]}, default=[0.0, 0.0, 1.0])

    admission = _admission(bounded_fast_profile)
    held = threading.Event()
    release = threading.Event()

    def occupy() -> None:
        with admission:
            held.set()
            release.wait(10)

    holder = threading.Thread(target=occupy, daemon=True)
    holder.start()
    assert held.wait(5)
    try:
        with pytest.raises(RetrievalOverloaded):
            dev_search_memory(store, embedder, "cats")
    finally:
        release.set()
        holder.join(10)

    snapshot = METRICS.snapshot()
    assert snapshot["counters"].get(
        "recall_retrieval_rejected_total{profile=fast,reason=budget_exhausted}"
    ) == 1
    assert "recall_retrieval_failed_total{profile=fast}" not in snapshot["counters"]
    assert "recall_retrieval_total_ms{profile=fast}" not in snapshot["histograms"]


@requires_db
def test_a_rejection_is_counted_and_carries_a_retry_hint(make_store, bounded_fast_profile):
    from recall_mcp.service import _admission

    METRICS.reset()
    store = make_store(3)
    store.upsert([Chunk("a", "notes.md", "cats")], [[1.0, 0.0, 0.0]])
    embedder = CountingEmbedder({"cats": [1.0, 0.0, 0.0]}, default=[0.0, 0.0, 1.0])

    admission = _admission(bounded_fast_profile)
    held = threading.Event()
    release = threading.Event()

    def occupy() -> None:
        with admission:
            held.set()
            release.wait(10)

    holder = threading.Thread(target=occupy, daemon=True)
    holder.start()
    assert held.wait(5)
    try:
        with pytest.raises(RetrievalOverloaded) as excinfo:
            dev_search_memory(store, embedder, "cats")
        assert 0 < excinfo.value.retry_after_seconds <= 5.0
    finally:
        release.set()
        holder.join(10)
    counters = METRICS.snapshot()["counters"]
    key = "recall_retrieval_rejected_total{profile=fast,reason=budget_exhausted}"
    assert counters.get(key) == 1


# --------------------------------------------------------------------------------------------
# One reranker per worker
# --------------------------------------------------------------------------------------------


def test_a_concurrent_first_request_storm_builds_exactly_one_reranker(monkeypatch):
    """`lru_cache` is not a construction lock: concurrent misses each build their own model.

    A cross-encoder is hundreds of megabytes. Building N of them because N requests arrived
    together is an out-of-memory event at exactly the moment the process is busiest.
    """
    import recall_mcp.service as service

    built = []
    lock = threading.Lock()

    def slow_factory(env=None):
        time.sleep(0.05)  # widen the window so the race is deterministic, not incidental
        instance = ReversingReranker()
        with lock:
            built.append(instance)
        return instance

    monkeypatch.setattr(service, "_new_reranker", slow_factory)
    service._reset_reranker_cache()
    monkeypatch.setenv("RECALL_RETRIEVAL_PROFILE", "quality")
    monkeypatch.delenv("RECALL_RERANK", raising=False)

    got: list[object] = []
    got_lock = threading.Lock()

    def call() -> None:
        instance = service._build_reranker()
        with got_lock:
            got.append(instance)

    threads = [threading.Thread(target=call) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)

    service._reset_reranker_cache()
    assert len(built) == 1, f"built {len(built)} rerankers for one process"
    assert len(got) == 8
    assert all(instance is built[0] for instance in got)


# --------------------------------------------------------------------------------------------
# Complete stage timing
# --------------------------------------------------------------------------------------------

#: Every stage of a retrieval, in execution order. A response missing one of these is a response
#: whose latency cannot be attributed, which is the state this program exists to leave behind.
EXPECTED_STAGES = {
    "admission_wait",
    "query_embedding",
    "dense_retrieval",
    "sparse_retrieval",
    # Added with the learned sparse leg. It is recorded on EVERY retrieval, including the
    # lexical-backend default where the leg does not run and the timing is ~0 -- a stage
    # that vanishes when idle would make "no learned sparse leg" and "the timer is broken"
    # look identical in the cost surface.
    "learned_sparse_retrieval",
    "fusion",
    "reranking",
    "trust_evaluation",
    "evidence_assembly",
}


@requires_db
def test_every_stage_of_a_retrieval_is_timed(make_store, monkeypatch):
    monkeypatch.setenv("RECALL_RETRIEVAL_PROFILE", "fast")
    monkeypatch.delenv("RECALL_RERANK", raising=False)
    store = make_store(3)
    store.upsert([Chunk("a", "notes.md", "cats")], [[1.0, 0.0, 0.0]])
    embedder = CountingEmbedder({"cats": [1.0, 0.0, 0.0]}, default=[0.0, 0.0, 1.0])

    result = dev_search_memory(store, embedder, "cats")
    assert set(result.stage_ms) == EXPECTED_STAGES
    assert all(value >= 0.0 for value in result.stage_ms.values())


@requires_db
def test_stage_timings_reach_the_process_metrics(make_store, monkeypatch):
    """Per-response timings answer "why was THAT query slow". Percentiles need the population."""
    monkeypatch.setenv("RECALL_RETRIEVAL_PROFILE", "fast")
    monkeypatch.delenv("RECALL_RERANK", raising=False)
    METRICS.reset()
    store = make_store(3)
    store.upsert([Chunk("a", "notes.md", "cats")], [[1.0, 0.0, 0.0]])
    embedder = CountingEmbedder({"cats": [1.0, 0.0, 0.0]}, default=[0.0, 0.0, 1.0])

    dev_search_memory(store, embedder, "cats")
    histograms = METRICS.snapshot()["histograms"]
    for stage in EXPECTED_STAGES:
        assert f"recall_retrieval_stage_ms{{profile=fast,stage={stage}}}" in histograms
    assert "recall_retrieval_total_ms{profile=fast}" in histograms


@requires_db
def test_the_result_reports_the_whole_identity_of_the_retrieval(make_store, monkeypatch):
    monkeypatch.setenv("RECALL_RETRIEVAL_PROFILE", "fast")
    monkeypatch.delenv("RECALL_RERANK", raising=False)
    store = make_store(3)
    store.upsert([Chunk("a", "notes.md", "cats")], [[1.0, 0.0, 0.0]])
    embedder = CountingEmbedder({"cats": [1.0, 0.0, 0.0]}, default=[0.0, 0.0, 1.0])

    result = dev_search_memory(store, embedder, "cats")
    assert result.retrieval_profile == "fast"
    assert result.candidate_pool_size == 20
    assert result.reranking_ran is False
    assert result.embedding_profile  # non-empty; the embedder's declared identity
    assert result.index_generation
    assert result.latency_budget_ms == 250
    assert result.total_ms > 0.0
    assert isinstance(result.budget_exceeded, bool)


@requires_db
def test_an_over_budget_request_is_labelled_and_counted(make_store, monkeypatch):
    """The budget is observable on the response, not only in a log nobody reads."""
    monkeypatch.setenv("RECALL_RETRIEVAL_PROFILE", "fast")
    monkeypatch.delenv("RECALL_RERANK", raising=False)
    METRICS.reset()
    store = make_store(3)
    store.upsert([Chunk("a", "notes.md", "cats")], [[1.0, 0.0, 0.0]])

    class SlowEmbedder(CountingEmbedder):
        def embed(self, texts):
            time.sleep(0.4)  # over the fast profile's 250 ms budget, on its own
            return super().embed(texts)

    embedder = SlowEmbedder({"cats": [1.0, 0.0, 0.0]}, default=[0.0, 0.0, 1.0])
    result = dev_search_memory(store, embedder, "cats")
    assert result.budget_exceeded is True
    assert result.total_ms > result.latency_budget_ms
    counters = METRICS.snapshot()["counters"]
    assert counters.get("recall_retrieval_budget_exceeded_total{profile=fast}") == 1


@requires_db
def test_the_budget_verdict_is_charged_once_not_twice(make_store, monkeypatch):
    """The budget bounds the WAIT. It must not also be spent on the end-to-end total.

    `latency_budget_ms` is the admission timeout, so a request may legitimately wait almost the
    whole budget before it starts. Comparing the budget against a total that INCLUDES that wait
    charges the same allowance twice: a request whose own retrieval was fast gets labelled over
    budget because someone else was ahead of it, and the counter saturates under any queueing.
    That destroys exactly the attribution the seven-stage surface exists to provide.
    """
    from recall_mcp.service import _admission

    monkeypatch.setenv("RECALL_RETRIEVAL_PROFILE", "fast")
    monkeypatch.delenv("RECALL_RERANK", raising=False)
    monkeypatch.setenv("RECALL_SEARCH_CONCURRENCY", "1")
    monkeypatch.setenv("RECALL_SEARCH_QUEUE", "1")
    METRICS.reset()
    store = make_store(3)
    store.upsert([Chunk("a", "notes.md", "cats")], [[1.0, 0.0, 0.0]])

    admission = _admission(resolve_retrieval_profile())
    held = threading.Event()
    release = threading.Event()

    def occupy() -> None:
        with admission:
            held.set()
            release.wait(10)

    holder = threading.Thread(target=occupy, daemon=True)
    holder.start()
    assert held.wait(5)
    # The fixture has to make wait + work cross the 250 ms budget while the WORK alone stays under
    # it, or the assertion is vacuous: with an instant search, charging the wait as well still
    # lands inside the budget and a double-charging implementation passes. ~150 ms queued plus
    # ~150 ms of work gives ~300 ms total against 250 ms of work allowance. (Found by mutation:
    # an earlier fixture used a 200 ms wait and a microsecond search, and survived.)
    threading.Timer(0.15, release.set).start()

    class SlowEmbedder(CountingEmbedder):
        def embed(self, texts):
            time.sleep(0.15)
            return super().embed(texts)

    slow = SlowEmbedder({"cats": [1.0, 0.0, 0.0]}, default=[0.0, 0.0, 1.0])
    result = dev_search_memory(store, slow, "cats")
    holder.join(10)

    assert result.stage_ms["admission_wait"] >= 100, "the wait must be measured, not assumed"
    assert result.total_ms > 250, "wait + work must cross the budget, or this proves nothing"
    served = result.total_ms - result.stage_ms["admission_wait"]
    assert served < 250, "the work alone must stay inside the budget for this to discriminate"
    assert result.budget_exceeded is False, "a fast retrieval was labelled slow because it queued"
    counters = METRICS.snapshot()["counters"]
    assert "recall_retrieval_budget_exceeded_total{profile=fast}" not in counters


@requires_db
def test_a_profile_with_no_budget_reports_none_rather_than_a_sentinel(make_store, monkeypatch):
    """The legacy budget is a 24-day sentinel. Shipping it as a number invites arithmetic on it.

    `RetrievalOverloaded` already refuses to hand that value to a client as a retry hint. The
    same reasoning applies to the result surface, and this field is new, so nothing depends on
    the sentinel spelling yet.
    """
    monkeypatch.delenv("RECALL_RETRIEVAL_PROFILE", raising=False)
    monkeypatch.delenv("RECALL_RERANK", raising=False)
    store = make_store(3)
    store.upsert([Chunk("a", "notes.md", "cats")], [[1.0, 0.0, 0.0]])
    embedder = CountingEmbedder({"cats": [1.0, 0.0, 0.0]}, default=[0.0, 0.0, 1.0])

    result = dev_search_memory(store, embedder, "cats")
    assert result.retrieval_profile == "legacy"
    assert result.latency_budget_ms is None
    assert result.budget_exceeded is False


@requires_db
def test_total_latency_is_recorded_even_when_the_search_raises(make_store, monkeypatch):
    """A timer that only records on success hides the slow path worth finding.

    `recall/observability.py` states this rule for `METRICS.timer` in as many words. The new
    per-request total was emitted after the `with` block, so a store fault or a strict trust
    refusal, the two slowest ways a request can end, contributed nothing to the histogram.
    """
    monkeypatch.setenv("RECALL_RETRIEVAL_PROFILE", "fast")
    monkeypatch.delenv("RECALL_RERANK", raising=False)
    METRICS.reset()
    store = make_store(3)
    store.upsert([Chunk("a", "notes.md", "cats")], [[1.0, 0.0, 0.0]])

    class ExplodingEmbedder(CountingEmbedder):
        def embed(self, texts):
            raise RuntimeError("embedder is down")

    embedder = ExplodingEmbedder({}, default=[0.0, 0.0, 1.0])
    with pytest.raises(RuntimeError, match="embedder is down"):
        dev_search_memory(store, embedder, "cats")

    snapshot = METRICS.snapshot()
    assert "recall_retrieval_total_ms{profile=fast}" in snapshot["histograms"]
    assert snapshot["counters"].get("recall_retrieval_failed_total{profile=fast}") == 1


# --------------------------------------------------------------------------------------------
# Reranking preserves the dense cosine
# --------------------------------------------------------------------------------------------


@requires_db
def test_reranking_reorders_without_rewriting_the_dense_cosine(make_store, monkeypatch):
    """Calibration thresholds are stated in cosine units. A reranker score is not one.

    The fixture is deliberately discriminating: the two chunks have DIFFERENT cosines (1.0 and
    0.6), so a `score` that had been overwritten with a fused rank or a cross-encoder logit
    could not coincidentally equal the right value for both.
    """
    import recall_mcp.service as service

    monkeypatch.setenv("RECALL_RETRIEVAL_PROFILE", "quality")
    monkeypatch.delenv("RECALL_RERANK", raising=False)
    reranker = ReversingReranker()
    monkeypatch.setattr(service, "_build_reranker", lambda env=None: reranker)

    store = make_store(3)
    store.upsert(
        [Chunk("near", "near.md", "cats"), Chunk("far", "far.md", "cats and dogs")],
        [[1.0, 0.0, 0.0], [0.6, 0.8, 0.0]],
    )
    embedder = CountingEmbedder({"cats": [1.0, 0.0, 0.0]}, default=[0.0, 0.0, 1.0])

    result = dev_search_memory(store, embedder, "cats")
    assert reranker.calls == 1
    assert result.reranking_ran is True
    by_source = {hit.source: hit.score for hit in result.hits}
    assert by_source["near.md"] == pytest.approx(1.0, abs=1e-3)
    assert by_source["far.md"] == pytest.approx(0.6, abs=1e-3)


# --------------------------------------------------------------------------------------------
# No query text and no corpus text in logs
# --------------------------------------------------------------------------------------------

#: A string that cannot occur by accident anywhere in this codebase or its dependencies.
QUERY_SENTINEL = "zqxjvwlm-querysentinel-4417"
CORPUS_SENTINEL = "zqxjvwlm-corpussentinel-9930"


class _RecordCollector(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _rendered(record: logging.LogRecord) -> str:
    """Everything a formatter could possibly emit from this record, as one string.

    Not `record.getMessage()` alone: the JSON formatter in `recall.observability` also serialises
    every non-reserved attribute (the `extra=` channel) and the formatted exception, so a leak
    could ride out through a field the plain text formatter never renders.
    """
    parts = [str(record.msg), record.getMessage(), repr(record.args)]
    if record.exc_info and record.exc_info[1] is not None:
        parts.append(repr(record.exc_info[1]))
    if record.exc_text:
        parts.append(record.exc_text)
    parts.append(json.dumps(record.__dict__, default=str))
    return "\n".join(parts)


def _capture_recall_logs(collector: _RecordCollector):
    """Attach `collector` to every logger the library and the service can emit through."""
    from contextlib import contextmanager

    @contextmanager
    def _ctx():
        root = logging.getLogger()
        # ONE handler, on the root, and propagation forced on below. Attaching to `recall` as well
        # would capture each record twice — `recall_mcp.service` logs under `recall.mcp.service`,
        # so every record already passes both — and a doubled capture makes a count assertion
        # lie about how many records were emitted.
        loggers = [root, logging.getLogger("recall"), logging.getLogger("recall_mcp")]
        previous = [(lg, lg.level, lg.propagate) for lg in loggers]
        root.addHandler(collector)
        for lg in loggers:
            lg.setLevel(logging.DEBUG)
            lg.propagate = True
        try:
            yield
        finally:
            root.removeHandler(collector)
            for lg, level, propagate in previous:
                lg.setLevel(level)
                lg.propagate = propagate

    return _ctx()


def test_the_log_leak_detector_can_fire() -> None:
    """The positive control. Without it, "no leaks" is indistinguishable from "no detector".

    Three plants, one per channel the detector inspects, because a detector that only reads
    `getMessage()` would pass the first and miss the other two.
    """
    collector = _RecordCollector()
    log = logging.getLogger("recall.leak_detector_control")
    with _capture_recall_logs(collector):
        log.warning("interpolated %s", QUERY_SENTINEL)
        log.warning("structured", extra={"payload": CORPUS_SENTINEL})
        try:
            raise ValueError(QUERY_SENTINEL)
        except ValueError:
            log.warning("with an exception", exc_info=True)
    rendered = [_rendered(record) for record in collector.records]
    assert len(rendered) == 3
    assert all(QUERY_SENTINEL in text or CORPUS_SENTINEL in text for text in rendered)


@requires_db
def test_neither_query_text_nor_corpus_text_reaches_a_log_record(make_store, monkeypatch):
    """Driven down a path that DOES log, so the assertion ranges over real records.

    An uncalibrated development search emits the untuned-threshold warning, which is the noisiest
    thing an ordinary search can produce. Asserting over an empty capture would prove nothing.
    """
    import recall.trust as trust
    from recall.trust_policy import TrustPolicy
    from recall_mcp.service import search_memory

    monkeypatch.setenv("RECALL_RETRIEVAL_PROFILE", "fast")
    monkeypatch.delenv("RECALL_RERANK", raising=False)
    store = make_store(3)
    store.upsert(
        [Chunk("a", "notes.md", f"a memo about {CORPUS_SENTINEL}")],
        [[1.0, 0.0, 0.0]],
    )
    embedder = CountingEmbedder(
        {f"what about {QUERY_SENTINEL}": [1.0, 0.0, 0.0]}, default=[0.0, 0.0, 1.0]
    )
    # The warning is emitted once per embedder name per process; clear it so this run gets it.
    trust._WARNED_UNCALIBRATED.discard(embedder.name)

    collector = _RecordCollector()
    with _capture_recall_logs(collector):
        result = search_memory(
            store,
            embedder,
            f"what about {QUERY_SENTINEL}",
            calibration=None,
            policy=TrustPolicy.development(),
        )

    assert result.hits, "the search must actually have run for this to prove anything"
    assert collector.records, "no log records captured — the assertion below would be vacuous"
    leaked = [
        _rendered(record)
        for record in collector.records
        if QUERY_SENTINEL in _rendered(record) or CORPUS_SENTINEL in _rendered(record)
    ]
    assert leaked == [], f"{len(leaked)} log record(s) carried corpus or query text"


@requires_db
def test_no_query_text_survives_into_an_exception_message(make_store, monkeypatch):
    """An exception is a log record waiting to happen: callers print `str(exc)` by reflex."""
    monkeypatch.setenv("RECALL_RETRIEVAL_PROFILE", "fast")
    monkeypatch.delenv("RECALL_RERANK", raising=False)
    from recall_mcp.service import MAX_QUERY_CHARS

    store = make_store(3)
    embedder = CountingEmbedder({}, default=[0.0, 0.0, 1.0])
    oversized = QUERY_SENTINEL + "x" * (MAX_QUERY_CHARS + 1)
    with pytest.raises(ValueError) as excinfo:
        dev_search_memory(store, embedder, oversized)
    assert QUERY_SENTINEL not in str(excinfo.value)
    assert QUERY_SENTINEL not in repr(excinfo.value)
