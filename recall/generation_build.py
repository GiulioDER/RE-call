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
from typing import Any, Literal

from recall.embeddings import Embedder, HashingEmbedder
from recall.generations import BuildStats, GenerationManager
from recall.index import (
    DEFAULT_MAX_CHARS,
    DEFAULT_OVERLAP_CHARS,
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

ChunkerKind = Literal["text", "code"]


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
    """
    if request.chunker == "code":
        # `chunk_code` takes no overlap, so recording one would describe a pipeline that never ran.
        return (
            functools.partial(chunk_code, max_chars=request.max_chars),
            ChunkerIdentity("recall.chunk_code", 1, {"max_chars": request.max_chars}),
        )
    return (
        functools.partial(chunk_text, max_chars=request.max_chars, overlap=request.overlap),
        ChunkerIdentity(
            "recall.chunk_text",
            1,
            {"max_chars": request.max_chars, "overlap": request.overlap},
        ),
    )


def pipeline_identity(embedder: Embedder | Any, request: BuildRequest) -> PipelineIdentity:
    return PipelineIdentity(embedder_identity(embedder, request), chunker_for(request)[1])


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
    generation = manager.create(
        manifest,
        pipeline_identity(embedder, request),
        allow_unverified=request.unverified,
    )
    return manager.build(
        generation.generation_id,
        reader,
        embedder,
        chunker_for(request)[0],
        provenance=build_provenance(request),
    )
