"""Generation lifecycle errors and records.

A leaf, so code that only needs to catch `NoActiveGeneration` or read a `GenerationRecord`
does not load the build pipeline (extraction, embedding, manifests). `recall.generations`
re-exports every name.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from recall.errors import RecallError
from recall.lineage import GenerationState


class GenerationError(RuntimeError, RecallError):
    """A generation lifecycle invariant was violated."""


class GenerationNotFound(GenerationError):
    pass


class InvalidGenerationTransition(GenerationError):
    pass


class UnsafePromotion(GenerationError):
    pass


class ConcurrentIngest(GenerationError):
    """Another upload holds this tenant's bounded ingest lock."""


class NoActiveGeneration(GenerationError):
    pass


@dataclass(frozen=True)
class GenerationRecord:
    tenant_id: str
    generation_id: str
    state: GenerationState
    pipeline_fingerprint: str
    corpus_fingerprint: str
    manifest_digest: str
    corpus_version: str
    parent_generation_id: str | None
    failure_reason: str | None
    created_at: datetime
    activated_at: datetime | None
    retired_at: datetime | None


@dataclass(frozen=True)
class BuildStats:
    generation_id: str
    objects: int
    chunks: int
    reused_objects: int
    reused_chunks: int
    tombstoned_objects: int
    empty_objects: int


@dataclass(frozen=True)
class ValidationResult:
    generation_id: str
    sources: int
    chunks: int
    state: GenerationState


@dataclass(frozen=True)
class ErasureResult:
    source_uri: str
    generations: tuple[str, ...]
    chunks_removed: int
    event_id: str
