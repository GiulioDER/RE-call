"""Generation parity validation performed before an enterprise cutover."""
from __future__ import annotations

from dataclasses import dataclass

from recall.store import PgVectorStore


@dataclass(frozen=True)
class GenerationParity:
    valid: bool
    active_chunks: int
    shadow_chunks: int
    missing_sources: tuple[str, ...]
    extra_sources: tuple[str, ...]
    hash_mismatches: tuple[str, ...]
    failures: tuple[str, ...]


def validate_generation_parity(
    active: PgVectorStore, shadow: PgVectorStore
) -> GenerationParity:
    """Compare source sets and raw hashes while allowing embeddings and metadata to differ."""
    if active.tenant != shadow.tenant:
        raise ValueError("generation parity requires stores for the same tenant")
    active_hashes = active.source_raw_hashes()
    shadow_hashes = shadow.source_raw_hashes()
    active_sources = set(active_hashes)
    shadow_sources = set(shadow_hashes)
    missing = tuple(sorted(active_sources - shadow_sources))
    extra = tuple(sorted(shadow_sources - active_sources))
    mismatches = tuple(
        sorted(
            source for source in active_sources & shadow_sources
            if active_hashes[source] != shadow_hashes[source]
        )
    )
    failures: list[str] = []
    active_facts = active.readiness_facts()
    shadow_facts = shadow.readiness_facts()
    if not active_facts["indexes_valid"] or not shadow_facts["indexes_valid"]:
        failures.append("one or more generation indexes are invalid")
    if not active_facts["rls_enabled"] or not shadow_facts["rls_enabled"]:
        failures.append("row level security is not forced on both generations")
    if missing:
        failures.append("shadow generation is missing sources")
    if extra:
        failures.append("shadow generation contains extra sources")
    if mismatches:
        failures.append("raw content hashes differ between generations")
    active_chunks = active.count()
    shadow_chunks = shadow.count()
    if active_chunks != shadow_chunks:
        failures.append("chunk counts differ between generations")
    return GenerationParity(
        valid=not failures,
        active_chunks=active_chunks,
        shadow_chunks=shadow_chunks,
        missing_sources=missing,
        extra_sources=extra,
        hash_mismatches=mismatches,
        failures=tuple(failures),
    )
