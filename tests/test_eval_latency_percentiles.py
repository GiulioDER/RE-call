"""The harness must PUBLISH the nearest-rank percentile, not merely have one available.

`tests/test_percentiles.py` pins the definition. This file pins the wiring, because the two are
different claims: `recall.eval.metrics.latency_report` can be correct while `labelled.evaluate`
goes on computing its own index expression three lines away, and the resulting JSON is a number
of the right magnitude either way — nothing about a latency one rank too high looks wrong.

The clock is faked so the assertion is on an exact value rather than on an ordering. Real timings
would only support `p50 <= p95`, which the broken expression satisfies too.

n=2 is chosen deliberately, and it separates the two implementations **on p50 only**: `n // 2` is
one rank high for every even n, so the smallest possible sample already discriminates there (10.0
against 30.0), while `int(0.95 * n)` is one rank high only when n is a multiple of 20 and both
implementations agree on p95 at this size. The p95 slot is therefore pinned in
tests/test_percentiles.py, at n=20 and n=44, where it can actually fail. Saying "n=2 separates
the two implementations" without that qualifier would name a guard wider than the one that exists.
"""
from __future__ import annotations

import uuid

from recall.eval import labelled as labelled_mod
from recall.eval.labelled import evaluate
from recall.embeddings import HashingEmbedder
from recall.store import PgVectorStore

from .conftest import TEST_DSN, requires_db

#: Two answerable per half so the DEFAULT (`held`) scores exactly n=2 per arm.
_QUESTIONS = [
    {"id": "q1", "query": "deploy target", "relevant_files": ["a.md"], "answerable": True},
    {"id": "q2", "query": "database backup", "relevant_files": ["b.md"], "answerable": True},
    {"id": "q3", "query": "oncall rotation", "relevant_files": ["c.md"], "answerable": True},
    {"id": "q4", "query": "incident review", "relevant_files": ["d.md"], "answerable": True},
    {"id": "q5", "query": "airspeed of a swallow", "relevant_files": [], "answerable": False},
    {"id": "q6", "query": "unrelated nonsense xyzzy", "relevant_files": [], "answerable": False},
]

#: Seconds. Every `perf_counter` call in `evaluate` is one half of a start/end pair, so the fake
#: hands out pair 0 -> _DELTAS[0], pair 1 -> _DELTAS[1], pair 2 -> _DELTAS[0], and so on. Pair 0
#: is the index timing; each arm then takes two pairs, one per scored question, giving every arm
#: the sample [10 ms, 30 ms] regardless of how many arms run.
_DELTAS = (0.030, 0.010)


class _PairedClock:
    """A `perf_counter` that returns START, END, START, END, ... with fixed deltas."""

    def __init__(self, deltas: tuple[float, ...]) -> None:
        self._deltas = deltas
        self._calls = 0
        self._start = 0.0

    def perf_counter(self) -> float:
        i, self._calls = self._calls, self._calls + 1
        if i % 2 == 0:
            self._start = float(i)
            return self._start
        return self._start + self._deltas[(i // 2) % len(self._deltas)]


def test_the_paired_clock_produces_the_sample_this_file_claims() -> None:
    """The apparatus first: a clock that did not actually alternate would fake the whole test.

    Asserted rather than assumed, because every value below is read through it — a fake that
    silently returned a constant would make the report `{"p50": 0.0, "p95": 0.0}` under BOTH
    implementations and the wiring test would pass while measuring nothing.
    """
    clock = _PairedClock(_DELTAS)
    deltas_ms = []
    for _ in range(3):  # three pairs: the index timing, then two scored questions
        t = clock.perf_counter()
        deltas_ms.append(round((clock.perf_counter() - t) * 1000, 6))

    assert deltas_ms == [30.0, 10.0, 30.0]
    # The two the arms see are unequal, which is what makes rank 1 and rank 2 distinguishable.
    assert deltas_ms[1] != deltas_ms[2]


@requires_db
def test_every_arm_publishes_the_nearest_rank_percentile(tmp_path, monkeypatch) -> None:
    """End to end: the JSON `labelled` prints carries p50 = the LOWER of the two samples.

    Under the expression this replaced (`lat[len(lat) // 2]`) a two-sample latency reports index
    1 — the slower query — as the median of both. That is the assertion that separates them, and
    it is checked on every arm rather than on `hybrid` alone, because each arm builds its own
    `latency_ms` from its own list.
    """
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for name, text in (
        ("a.md", "the deploy target is staging"),
        ("b.md", "the database backup runs nightly"),
        ("c.md", "the oncall rotation is weekly"),
        ("d.md", "the incident review is on friday"),
    ):
        (corpus / name).write_text(text, encoding="utf-8")

    emb = HashingEmbedder(dim=64)
    table = "lat_" + uuid.uuid4().hex[:8]
    store = PgVectorStore(TEST_DSN, dim=emb.dim, table=table)
    monkeypatch.setattr(labelled_mod, "time", _PairedClock(_DELTAS))
    try:
        rep = evaluate(TEST_DSN, corpus, _QUESTIONS, emb, k=3, table=table)

        assert rep["questions"]["retrieval_scored_on"] == 2, "fixture must give each arm n=2"
        for name, arm in rep["arms"].items():
            assert arm["latency_ms"] == {"p50": 10.0, "p95": 30.0}, f"{name} reports a rank high"
    finally:
        store.drop_table()
        store.close()


@requires_db
def test_a_run_with_no_timed_query_reports_no_latency(tmp_path) -> None:
    """`{}`, not `{"p50": 0.0}`. An unmeasured tail must not read as an instantaneous one.

    Reachable in practice: `score_retrieval_on="held"` over a question set whose held half holds
    no answerable question scores every arm on nothing.
    """
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text("the deploy target is staging", encoding="utf-8")

    #: fit = idx 0, 2 -> q1 answerable + q3 unanswerable; held = idx 1 -> q2 unanswerable only.
    questions = [
        {"id": "q1", "query": "deploy target", "relevant_files": ["a.md"], "answerable": True},
        {"id": "q2", "query": "airspeed of a swallow", "relevant_files": [], "answerable": False},
        {"id": "q3", "query": "unrelated nonsense xyzzy", "relevant_files": [],
         "answerable": False},
    ]

    emb = HashingEmbedder(dim=64)
    table = "lat0_" + uuid.uuid4().hex[:8]
    store = PgVectorStore(TEST_DSN, dim=emb.dim, table=table)
    try:
        rep = evaluate(TEST_DSN, corpus, questions, emb, k=3, table=table)

        assert rep["questions"]["retrieval_scored_on"] == 0
        for name, arm in rep["arms"].items():
            assert arm["latency_ms"] == {}, f"{name} invented a latency from an empty sample"
            assert arm["hit_at_3"]["n"] == 0
    finally:
        store.drop_table()
        store.close()


def test_the_perq_harness_reports_through_the_same_helper() -> None:
    """Both harnesses import the shared helper, so neither can be fixed while the other is not.

    A weak check on its own — it is the DB-backed test above that proves the value — but it is
    what catches the copy being reintroduced in the arm this file cannot cheaply run.
    """
    from recall.eval import longmemeval_perq
    from recall.eval.metrics import latency_report

    assert longmemeval_perq.latency_report is latency_report
    assert labelled_mod.latency_report is latency_report
