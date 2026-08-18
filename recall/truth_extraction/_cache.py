"""Extraction caching, keyed on everything that can change an answer.

Extraction runs on the ingest path over a whole corpus, and re-ingesting an unchanged memo
must not re-pay for it. The risk a cache introduces is the opposite of the one it solves:
serving an answer produced by a different prompt or a different engine, which would make the
audit record wrong about how a claim was produced. So the key covers engine identity, engine
revision, prompt revision, the file, the body, and the corpus names the ladder resolves
targets against — a corpus that gained a file can resolve a target that previously refused.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

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


#: Prefix for the stand-in. NUL cannot appear in a path on either platform, and a body that
#: does contain it is handled by diverting it too, which is what keeps `_hashable` injective.
_SURROGATE_MARK = "\x00surrogate:"


def _hashable(text: str) -> str:
    """`text` unchanged when it is valid UTF-8, and a stable stand-in when it is not.

    A POSIX filename is bytes, not text. One that is not valid UTF-8 reaches here as a lone
    surrogate through `Path.glob`'s surrogateescape, and `canonical_sha256` encodes as UTF-8, so
    hashing it raised `UnicodeEncodeError`. That is computed unconditionally by
    `extract_file_claims`, BEFORE and independently of any cache, and outside the guard that
    keeps one bad memo from killing a run: a single such filename aborted the whole ingest and
    discarded every file already extracted. Hardening the cache's own writes against the same
    byte, as `put` does, was pointless while the key computation one frame earlier still threw.

    The valid-UTF-8 path returns the string itself, so no existing cache entry changes key.

    INJECTIVE, and the first version was not. It reasoned that the NUL prefix cannot occur in a
    real filename, which is true of `file` and `corpus_names` and false of `human_body`: that is
    file CONTENT, a NUL survives `read_text` intact, and a body literally beginning with the
    marker collided with the surrogate string that marker encodes. Two different documents then
    shared a cache key and one was served for the other, which is the single failure this whole
    module exists to prevent, introduced by the guard meant to protect it. So a string that
    already starts with the marker is diverted too, and the map is one to one: a diverted output
    always starts with the marker, a passed-through one never can.
    """
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        pass
    else:
        if not text.startswith(_SURROGATE_MARK):
            return text
    return _SURROGATE_MARK + text.encode("utf-8", "surrogatepass").hex()


#: Which `ExtractionPrompt` fields the key hashes. ITERATED below rather than merely documented:
#: a constant the hash does not read is a comment, and a field present here and absent from the
#: hash would have passed the contract test that pins this against the dataclass. `system` and
#: `user` are excluded because they are RENDERED from the others.
#:
#: `status_vocabulary` was added to the prompt and to the key in the same commit only because a
#: human noticed, and a field added to the dataclass alone gives one key to two prompts, which
#: is the single failure this function exists to prevent.
#:
#: The stronger design, considered and not taken: hash `prompt.system` and `prompt.user`
#: directly. It would cover the fields below automatically and also `MAX_CLAIMS_PER_FILE` and
#: `VALIDITY_CLAIM_KEYS`, which are rendered in from `types.py` while `PROMPT_REVISION`'s "bump
#: on ANY wording change" only governs `_prompt.py`. It is not taken here because it silently
#: reverses a documented decision: `corpus_names` is SORTED for the key and rendered in the order
#: GIVEN, so hashing the text would make a reordered corpus list a full cache miss on every file.
#: That is a cost worth choosing deliberately, not as a side effect of tightening a guard.
_KEY_PROMPT_FIELDS = ("revision", "file", "human_body", "corpus_names", "status_vocabulary")

#: How each prompt field is folded in. `corpus_names` is SORTED and the others are not: the
#: sorting is a deliberate decision documented in `recheck_cached_extractions`, and the default
#: applies `_hashable` element-wise to a sequence or whole to a string.
_KEY_FIELD_FORM: Mapping[str, Callable[[Any], Any]] = {
    "corpus_names": lambda value: tuple(sorted(_hashable(n) for n in value)),
    "revision": _hashable,
    "file": _hashable,
    "human_body": _hashable,
    "status_vocabulary": lambda value: tuple(_hashable(v) for v in value),
}


def extraction_cache_key(*, engine: _EngineIdentity, prompt: ExtractionPrompt) -> str:
    """Content hash of every input that can change the answer."""
    # EVERY string, not the three that happened to be filenames. The identity fields come from
    # the environment (`RECALL_EXTRACTION_MODEL`, `RECALL_EXTRACTION_REVISION`, and the host in
    # `RECALL_EXTRACTION_BASE_URL`), and `os.environ` decodes with surrogateescape on POSIX by
    # exactly the mechanism that puts a lone surrogate in a filename. Guarding three of seven
    # fields is the asymmetry this module criticises elsewhere, and here it is worse: an engine
    # identity is fixed for the run, so it would raise on file 1 of 792 rather than on the one
    # awkward memo. The property is "this never raises", not "these fields are safe".
    #
    # The prompt half is ITERATED from `_KEY_PROMPT_FIELDS`, so a field added to
    # `ExtractionPrompt` and to that tuple is hashed without anyone remembering to edit a
    # literal here. `revision` is written out as `prompt_revision` to keep the hashed name
    # distinct from the engine's own.
    hashed: dict[str, Any] = {
        "engine_id": _hashable(engine.engine_id),
        "model_id": _hashable(engine.model_id),
        "engine_revision": _hashable(engine.revision),
    }
    for field in _KEY_PROMPT_FIELDS:
        name = "prompt_revision" if field == "revision" else field
        hashed[name] = _KEY_FIELD_FORM[field](getattr(prompt, field))
    return "tx_" + canonical_sha256(hashed)[:32]


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
    status_vocabulary: Sequence[str] | None = None,
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

    `status_vocabulary` must match the warm run's, for the same reason `corpus_names` is
    sorted above: it is part of the cache key AND of the rendered prompt. Omitted, a cache
    warmed under a corpus's own status words computes a different key, every file misses,
    and the report reads `checked=0` — a determinism measurement that silently became a
    non-measurement. It is also passed to the ladder, so a vocabulary difference is not
    reported as engine drift.

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
        prompt = (
            build_extraction_prompt(file=file, human_body=body, corpus_names=names)
            if status_vocabulary is None
            else build_extraction_prompt(
                file=file,
                human_body=body,
                corpus_names=names,
                status_vocabulary=status_vocabulary,
            )
        )
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
                answer,
                file=file,
                human_body=body,
                corpus_names=names,
                status_vocabulary=prompt.status_vocabulary,
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
