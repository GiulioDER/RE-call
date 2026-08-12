"""Extraction caching, keyed on everything that can change an answer.

Extraction runs on the ingest path over a whole corpus, and re-ingesting an unchanged memo
must not re-pay for it. The risk a cache introduces is the opposite of the one it solves:
serving an answer produced by a different prompt or a different engine, which would make the
audit record wrong about how a claim was produced. So the key covers engine identity, engine
revision, prompt revision, the file, the body, and the corpus names the ladder resolves
targets against — a corpus that gained a file can resolve a target that previously refused.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from recall.lineage import canonical_sha256
from recall.truth_extraction._prompt import ExtractionPrompt
from recall.truth_extraction.types import FileExtraction


class _EngineIdentity(Protocol):
    engine_id: str
    model_id: str
    revision: str


def extraction_cache_key(*, engine: _EngineIdentity, prompt: ExtractionPrompt) -> str:
    """Content hash of every input that can change the answer."""
    return "tx_" + canonical_sha256(
        {
            "engine_id": engine.engine_id,
            "model_id": engine.model_id,
            "engine_revision": engine.revision,
            "prompt_revision": prompt.revision,
            "file": prompt.file,
            "human_body": prompt.human_body,
            "corpus_names": tuple(sorted(prompt.corpus_names)),
        }
    )[:32]


class ExtractionCache(Protocol):
    """Port for extraction result storage. Implementations must never mutate a stored result."""

    def get(self, key: str) -> FileExtraction | None:
        ...

    def put(self, key: str, value: FileExtraction) -> None:
        ...


class InMemoryExtractionCache:
    """Process local cache. `hits` and `misses` are for tests and ingest reporting."""

    def __init__(self, initial: Mapping[str, FileExtraction] | None = None) -> None:
        self._entries: dict[str, FileExtraction] = dict(initial or {})
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> FileExtraction | None:
        entry = self._entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        self.hits += 1
        return entry

    def put(self, key: str, value: FileExtraction) -> None:
        self._entries[key] = value

    def __len__(self) -> int:
        return len(self._entries)


__all__ = [
    "ExtractionCache",
    "InMemoryExtractionCache",
    "extraction_cache_key",
]
