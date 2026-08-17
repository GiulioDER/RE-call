"""Turn a resolved embedder and a chunker choice into a generation, once, for every caller.

This is the hundred lines that used to sit inline in `cli.py`'s `main()`. The installation wizard
builds generations too, and a second implementation of this assembly is not a refactoring
preference: `EmbedderIdentity` and `ChunkerIdentity` are written verbatim into a record the rest of
the system treats as immutable evidence, so two implementations means two provenance vocabularies
that drift without anything failing. `tests/test_generation_build_assembly.py` pins the strings.

Deliberately a leaf. It imports `generations`, `lineage`, `index` and `embeddings` and nothing
imports it except its callers, so pulling the chunkers in cannot make a cycle. That is also why
this is not a method on `GenerationManager`, which would give `generations.py` a dependency on
`recall.index`.

**Nothing here reads the environment.** `GenerationManager` takes its environment as a constructor
argument and `manager.build` takes its reader as a parameter, so a caller that wants a `file://`
corpus passes `LocalObjectReader([root])` rather than setting `RECALL_LOCAL_ALLOWLIST` and hoping
the right code reads it back. A wizard driving several corpora in one process cannot afford
process-global switches, and it turns out it does not need any.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, get_args

from recall.embeddings import Embedder, HashingEmbedder
from recall.generations import BuildStats, GenerationManager
from recall.index import (
    DEFAULT_MAX_CHARS,
    DEFAULT_OVERLAP_CHARS,
    ChunkerKind,
    chunk_code,
    chunk_text,
    head_commit,
)
from recall.lineage import ChunkerIdentity, EmbedderIdentity, IndexManifestV1, PipelineIdentity
from recall.manifest import ObjectReader

#: `hashing` is shipped in this repository and deterministic, so it identifies itself. Every other
#: embedder's weights come from somewhere else, and fastembed is the only local provider offered.
_HASHING_PROVIDER = "recall"
_HASHING_REVISION = "hashing-md5-bow-v1"
_DEFAULT_PROVIDER = "fastembed"

#: Stamped when the operator asked for a development build AND supplied no other identity. The
#: string is written into the record, so it is a constant rather than a message.
_UNVERIFIED_REASON = "explicit development build"

#: `ChunkerKind` is re-exported so a caller building a `BuildRequest` need not know which module
#: owns the vocabulary. It is defined in `recall.index`, beside the two chunkers it names, so that
#: argparse can read it without making this module's import eager in `recall.cli`.
__all__ = [
    "BuildRequest",
    "ChunkerKind",
    "build_generation",
    "build_provenance",
    "chunker_for",
    "embedder_identity",
    "pipeline_for",
    "pipeline_identity",
]


def _validate_chunk_sizes(max_chars: int, overlap: int) -> None:
    """Refuse sizes that produce a record disagreeing with the chunks it describes.

    `max_chars < 1` is the one that matters, and it fails differently on the two paths, which is
    why neither path caught it. Measured: `chunk_code(text, max_chars=0)` returns the whole text as
    a SINGLE chunk, because it reaches `_pack` with `hard_split=False` and so never meets the
    `max_chars < 1` guard in `recall.index` whose whole purpose is to stop silent embedder
    truncation. The record then says `{"max_chars": 0}` for a 10,000-character chunk. On the text
    path it does raise, but only inside `manager.build`, which is after `manager.create` has
    written the generation row, so `recall generation build --max-chars 0` left an orphan behind.

    `overlap >= max_chars` is deliberately NOT refused. It looks degenerate and is not: measured at
    `max_chars=100, overlap=500`, the chunker clamps and returns exactly what `overlap=100` returns,
    133 chunks with the longest at 99. Inventing a bound I have not measured a harm for is how a
    constant that merely looks principled gets shipped.
    """
    if max_chars < 1:
        raise ValueError(f"max_chars must be at least 1, not {max_chars!r}")
    if overlap < 0:
        raise ValueError(f"overlap cannot be negative, not {overlap!r}")


@dataclass(frozen=True)
class BuildRequest:
    """Everything about a build that is a choice rather than a resource.

    Frozen and free of connections, readers and embedders on purpose: the wizard has to write this
    into resumable state and hand the identical request back on the next run, and a request that
    carried a live handle could not be serialised or compared.
    """

    chunker: ChunkerKind = "text"
    max_chars: int = DEFAULT_MAX_CHARS
    overlap: int = DEFAULT_OVERLAP_CHARS
    #: Overrides for the embedder identity. `None` means "use the default for this embedder".
    provider: str | None = None
    revision: str | None = None
    artifact_digest: str | None = None
    #: Build without an immutable identity, and say so in the record.
    unverified: bool = False
    #: Provenance, matching what `recall index` stamps on every chunk.
    project: str | None = None
    #: Where to read the HEAD commit from, or `None` to stamp no commit. The CLI passes `"."`,
    #: which is the directory the command was run in and not necessarily the corpus. That is
    #: preserved rather than corrected here; a caller that knows its corpus root should pass it.
    commit_root: str | None = "."

    def __post_init__(self) -> None:
        # The only thing validating this field was argparse's `choices=`, and that did not travel
        # with the extraction. `ChunkerKind` is a `Literal` and is erased at runtime, so without
        # this a typo or a case difference selects the TEXT chunker and records it as
        # `recall.chunk_text`: the callable and the record agree with each other and disagree with
        # what was asked, which is the one shape nothing downstream can detect.
        #
        # It matters more here than it did behind argparse. This is a serialisable request the
        # wizard writes into resumable state and hands back on a later run, so the value can
        # arrive from a file somebody edited rather than from a parser.
        #
        # The accepted set is read off the type rather than written out again. It was already
        # spelled three times (the annotation, `chunker_for`'s branches, argparse's `choices`), and
        # a fourth copy is a fourth thing to forget when a chunker is added. This guard covers the
        # constructor only; `chunker_for` refuses as well, because a request restored without
        # `__init__` never reaches this line.
        if self.chunker not in get_args(ChunkerKind):
            raise ValueError(
                f"chunker must be one of {get_args(ChunkerKind)}, not {self.chunker!r}"
            )
        # The sizes go into the same immutable record and arrive from the same edited file, so
        # validating only the chunker covered a third of the request the threat model describes.
        _validate_chunk_sizes(self.max_chars, self.overlap)


def embedder_identity(embedder: Embedder | Any, request: BuildRequest) -> EmbedderIdentity:
    """The identity recorded for `embedder`, with the request's overrides applied."""
    revision = request.revision
    provider = request.provider
    if isinstance(embedder, HashingEmbedder):
        provider = provider or _HASHING_PROVIDER
        revision = revision or _HASHING_REVISION
    else:
        provider = provider or _DEFAULT_PROVIDER
    return EmbedderIdentity(
        provider=provider,
        model=embedder.name,
        dimension=embedder.dim,
        revision=revision,
        artifact_digest=request.artifact_digest,
        # Only when nothing else identifies the embedder. A record carrying both a real revision
        # and "explicit development build" is self-contradicting provenance, and `EmbedderIdentity`
        # refuses to be constructed that way.
        unverified_reason=(
            _UNVERIFIED_REASON
            if request.unverified and not revision and not request.artifact_digest
            else None
        ),
    )


def chunker_for(request: BuildRequest) -> tuple[Callable[[str], list[str]], ChunkerIdentity]:
    """The chunking callable and the identity that describes it.

    Returned together because they must agree. A partial bound to the defaults while the record
    carries the operator's parameters is a well-formed generation whose provenance is false, and
    nothing downstream can detect it.

    Exhaustive rather than falling through to text, and that is not merely belt-and-braces on top
    of `BuildRequest.__post_init__`. A frozen dataclass rebuilt by `copy.deepcopy`, or by any
    deserialiser that restores `__dict__` directly, never runs `__post_init__`, so the constructor
    guard does not cover the round-trip the wizard's resumable state depends on. This is the
    decision point, so this is where an unrecognised value has to stop.
    """
    _validate_chunk_sizes(request.max_chars, request.overlap)
    if request.chunker == "code":
        # `chunk_code` takes no overlap, so recording one would describe a pipeline that never ran.
        return (
            functools.partial(chunk_code, max_chars=request.max_chars),
            ChunkerIdentity("recall.chunk_code", 1, {"max_chars": request.max_chars}),
        )
    if request.chunker == "text":
        return (
            functools.partial(chunk_text, max_chars=request.max_chars, overlap=request.overlap),
            ChunkerIdentity(
                "recall.chunk_text",
                1,
                {"max_chars": request.max_chars, "overlap": request.overlap},
            ),
        )
    raise ValueError(f"unknown chunker {request.chunker!r}")


def pipeline_for(
    embedder: Embedder | Any, request: BuildRequest
) -> tuple[Callable[[str], list[str]], PipelineIdentity]:
    """The chunking callable and the pipeline record that describes it, from one `chunker_for` call.

    This shape exists because the two previous ones were both wrong, in opposite directions.
    Building `PipelineIdentity` inline in `build_generation` kept `chunker_for` to one call and
    left this module with two assembly sites for the record. Passing an optional
    `chunker_identity` in here kept one assembly site and moved the "callable and record must
    agree" invariant out of the code and into the caller, where a text request could be recorded
    as a code pipeline with nothing objecting — the exact failure documented in `chunker_for` as
    undetectable downstream, reintroduced by the fix for the previous round's finding.

    Returning the pair from one function removes the choice. There is nowhere to pass a
    contradictory identity, because there is no parameter for one.
    """
    # `embedder_identity` FIRST, which is the order the code this replaced failed in. On a request
    # that is invalid in both ways, evaluating the chunker first surfaces `unknown chunker` where
    # the extracted code surfaced the embedder's `LineageError`. That reads as a no-op and is not.
    embedder_record = embedder_identity(embedder, request)
    chunker, chunker_identity = chunker_for(request)
    return chunker, PipelineIdentity(embedder_record, chunker_identity)


def pipeline_identity(embedder: Embedder | Any, request: BuildRequest) -> PipelineIdentity:
    """The record alone, for a caller that does not need the callable."""
    return pipeline_for(embedder, request)[1]


def build_provenance(request: BuildRequest) -> dict[str, str]:
    """The per-chunk stamps, with absent values omitted rather than stored as `None`.

    A stored `None` is indistinguishable later from a value that was genuinely recorded as empty,
    and it makes an absent record look like a taken one.
    """
    return {
        key: value
        for key, value in (
            ("project", request.project),
            (
                "indexed_commit",
                head_commit(request.commit_root) if request.commit_root is not None else None,
            ),
        )
        if value is not None
    }


def build_generation(
    manager: GenerationManager,
    manifest: IndexManifestV1,
    reader: ObjectReader,
    embedder: Embedder | Any,
    request: BuildRequest,
) -> BuildStats:
    """Create the generation and build it, leaving it BUILT and awaiting `validate`.

    Deliberately stops there. Validation, calibration and promotion are separate steps because
    their order is load-bearing: promotion gives a generation a fresh corpus fingerprint, so a
    calibration measured before it becomes `CALIBRATION_STALE` after it.
    """
    # One `chunker_for` call, and the callable and the record it describes come out of the same
    # one. Calling it twice — once through `pipeline_identity` for the record, once for the
    # callable — gave up the guarantee that they agree while looking like it kept it. They would
    # agree today regardless, because the function is pure over a frozen request; anything later
    # added to it that is not (a cached tokenizer, a seed, a registry lookup) would split them.
    chunker, pipeline = pipeline_for(embedder, request)
    generation = manager.create(manifest, pipeline, allow_unverified=request.unverified)
    return manager.build(
        generation.generation_id,
        reader,
        embedder,
        chunker,
        provenance=build_provenance(request),
    )
