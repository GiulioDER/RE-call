"""The third-party boundary. An interface with one implementation is a class, not a protocol.

Prior work: [[project-recall-abstention-bounded-domain-2026-07-24]] — six abstention signals
measured, all failed; this measures the axis they were measured on, not a seventh signal.
"""
from __future__ import annotations

from collections.abc import Iterable

from benchmarks.ladder.adapter import Document, MemorySystem, Response


class _Fake:
    name = "fake"

    def __init__(self) -> None:
        self._docs: dict[str, str] = {}

    def ingest(self, docs: Iterable[Document]) -> None:
        self._docs = {d.doc_id: d.text for d in docs}

    def indexed_doc_ids(self) -> frozenset[str]:
        return frozenset(self._docs)

    def query(self, question: str) -> Response:
        term = question.split()[0]
        hit = next((i for i, t in sorted(self._docs.items()) if term in t), None)
        if hit is None:
            return Response(answer=None)
        return Response(answer=self._docs[hit], cited_ids=(hit,))


def test_a_minimal_implementation_satisfies_the_protocol():
    system: MemorySystem = _Fake()
    system.ingest([Document("d1", "alpha text")])
    assert system.indexed_doc_ids() == {"d1"}


def test_none_answer_is_the_abstention():
    assert Response(answer=None).abstained is True
    assert Response(answer="something").abstained is False


def test_an_empty_string_is_an_answer_not_an_abstention():
    """A system returning '' has answered badly, not declined. Conflating them would score a
    broken generator as a well-calibrated one."""
    assert Response(answer="").abstained is False


def test_query_returns_an_abstention_when_nothing_matches():
    system = _Fake()
    system.ingest([Document("d1", "alpha text")])
    assert system.query("zulu something").abstained is True
