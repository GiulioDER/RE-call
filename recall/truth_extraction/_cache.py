"""Extraction caching, keyed on everything that can change an answer.

Extraction runs on the ingest path over a whole corpus, and re-ingesting an unchanged memo
must not re-pay for it. The risk a cache introduces is the opposite of the one it solves:
serving an answer produced by a different prompt or a different engine, which would make the
audit record wrong about how a claim was produced. So the key covers engine identity, engine
revision, prompt revision, the file, the body, and the corpus names the ladder resolves
targets against — a corpus that gained a file can resolve a target that previously refused.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from recall.lineage import canonical_sha256
from recall.truth_extraction._prompt import ExtractionPrompt
from recall.truth_extraction._sqlite_cache import (
    CACHE_SCHEMA_VERSION,
    ExtractionCacheRefused,
    SqliteExtractionCache,
)
from recall.truth_extraction.types import FileExtraction

if TYPE_CHECKING:  # `_engine` imports `_prompt`, and typing-only keeps the cycle off runtime
    from recall.truth_extraction._engine import ExtractionEngine


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


@dataclass(frozen=True)
class RecheckReport:
    """How often re-asking the engine disagreed with what the cache already held.

    A non zero `mismatched` means the CACHE, not the sampler, is what makes runs reproducible.
    Temperature 0 is not a determinism guarantee from any hosted provider, so this measures
    whether it held rather than assuming it did, which is worth knowing before a cache eviction
    silently renumbers every proposal id derived from these claims.
    """

    checked: int
    mismatched: int
    mismatched_files: tuple[str, ...]
    #: Files the engine never answered for, because it raised. Counted apart from `checked`: a
    #: network failure is not evidence about determinism in either direction, and folding it
    #: into agreement or into drift would put a number in this report that means neither.
    errored: int = 0
    errored_files: tuple[str, ...] = ()

    @property
    def mismatch_rate(self) -> float | None:
        """Mismatches per checked file, or None when nothing was checked.

        `None` rather than `0.0`. A cold cache checks nothing, and `0.0` would render as
        "no drift detected" when the truth is "nothing was measured": the two call for opposite
        responses from whoever asked.
        """
        return self.mismatched / self.checked if self.checked else None


def recheck_cached_extractions(
    documents: Mapping[str, str],
    *,
    engine: ExtractionEngine,
    corpus_names: Sequence[str],
    cache: ExtractionCache,
) -> RecheckReport:
    """Re-run the engine on already cached keys and report disagreement.

    Deliberately does NOT write back. A measurement that changes what it measures cannot be
    repeated, and overwriting the entry would erase the very drift this exists to surface.

    One honest exception, stated rather than discovered: reading through `cache.get` bumps an
    implementation's own hit and miss counters, so a recheck sharing a process with an ingest
    inflates that ingest's reported hit rate by one per checked file. The STORED RESULTS are
    untouched, which is the property that matters here. If those counters are ever reported
    next to a recheck, this needs a counter free read on the port rather than a note.

    Only cached files are checked. Counting a cache miss as agreement would inflate the
    determinism being reported, which is the one number this function exists to produce.

    **Comparison is on claims AND the batch rung**, not on claims alone. Comparing claims alone
    looks right and is nearly useless: on this repo's own docs corpus 34 of 36 cached entries
    hold zero claims, so replacing every engine answer with unparseable text (the largest drift
    possible) compared `() != ()` on all but two files and reported a 5.6% mismatch rate. The
    rung distinguishes "passed the ladder and found nothing" from "failed the ladder entirely",
    which is exactly the distinction that was being lost. Rejection REASONS are still excluded:
    they carry positional detail that can differ without the extracted truth differing.

    Engine failures are their own bucket. A dropped connection says nothing about determinism,
    so it is neither agreement nor drift, and it does not count toward `checked`.
    """
    from recall.truth_extraction._normalize import human_body_of, normalize_extraction
    from recall.truth_extraction._prompt import build_extraction_prompt
    from recall.truth_extraction.types import (
        CONSECUTIVE_ENGINE_FAILURE_LIMIT,
        ExtractionBatchRejected,
    )

    # Sorted once, here. `extraction_cache_key` hashes `sorted(corpus_names)` while
    # `build_extraction_prompt` renders them in the order given, so passing the same names in a
    # different order than the warm run hits the cached entry while asking the model a
    # textually different question, and any difference is then reported as engine drift.
    names = tuple(sorted(corpus_names))
    checked = 0
    mismatched: list[str] = []
    errored: list[str] = []
    consecutive_failures = 0
    for file, text in sorted(documents.items()):
        body = human_body_of(text)
        prompt = build_extraction_prompt(file=file, human_body=body, corpus_names=names)
        cached = cache.get(extraction_cache_key(engine=engine, prompt=prompt))
        if cached is None:
            continue
        if consecutive_failures >= CONSECUTIVE_ENGINE_FAILURE_LIMIT:
            # The engine is down, not drifting. Without this, every remaining cached file pays
            # the full retry and timeout budget to learn the same thing.
            errored.append(file)
            continue
        try:
            answer = engine.run(prompt)
        except Exception:  # noqa: BLE001 - the engine is third party code and reaches the network
            # Guarded like `extract_file_claims` guards the same call, and for the same reason:
            # one failure must not abort the run and discard every measurement already made.
            errored.append(file)
            consecutive_failures += 1
            continue
        consecutive_failures = 0
        try:
            claims, _ = normalize_extraction(
                answer, file=file, human_body=body, corpus_names=names
            )
            rung: str | None = None
        except ExtractionBatchRejected as refused:
            claims, rung = (), refused.rung
        cached_rung = cached.batch_rejection.rung if cached.batch_rejection else None
        if (tuple(claims), rung) != (tuple(cached.claims), cached_rung):
            mismatched.append(file)
        checked += 1
    return RecheckReport(
        checked=checked,
        mismatched=len(mismatched),
        mismatched_files=tuple(mismatched),
        errored=len(errored),
        errored_files=tuple(errored),
    )


__all__ = [
    # `SqliteExtractionCache` lives in `_sqlite_cache` and is re-exported here so that the two
    # implementations of the port are found together, from the module that defines the port.
    "CACHE_SCHEMA_VERSION",
    "ExtractionCache",
    "ExtractionCacheRefused",
    "InMemoryExtractionCache",
    "RecheckReport",
    "SqliteExtractionCache",
    "extraction_cache_key",
    "recheck_cached_extractions",
]
