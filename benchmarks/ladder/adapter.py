"""The boundary a third party implements to be scored by this benchmark.

Prior work: [[project-recall-abstention-bounded-domain-2026-07-24]] — six abstention signals
measured, all failed; this measures the axis they were measured on, not a seventh signal.

Three methods and two dataclasses, deliberately. Anything richer would encode assumptions about
how a memory system is built, and the benchmark's whole claim is that it scores the OUTCOME rather
than the mechanism.

`answer=None` **is** the abstention. An empty string is an answer, badly given — conflating the two
would score a broken generator as a well-calibrated one.

`indexed_doc_ids()` is not optional plumbing: it is how invariant 1 proves a system really dropped
the excised turns. A system that cached across rings would otherwise pass every rung and look like
a strong result rather than a broken harness.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str


@dataclass(frozen=True)
class Response:
    answer: str | None
    cited_ids: tuple[str, ...] = ()
    tokens: int = 0

    @property
    def abstained(self) -> bool:
        return self.answer is None


@runtime_checkable
class MemorySystem(Protocol):
    name: str

    def ingest(self, docs: Iterable[Document]) -> None:
        """Replace this system's corpus with `docs`. Must not retain anything from a prior call."""

    def indexed_doc_ids(self) -> frozenset[str]:
        """Every doc id currently retrievable. Read by invariant 1."""

    def query(self, question: str) -> Response:
        """Answer, or abstain with `answer=None`."""
