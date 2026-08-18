"""The single registry of embedding profile identity.

Every input that can change a stored vector is declared here, once. Before this module the same
profile vocabulary lived in two independent dict literals: a profile-ID -> context-version map
inside `recall_mcp.service.make_embedder`, and a profile-ID -> context-mode map inside
`recall.context.context_policy_for_profile`. They already differed in extent, and nothing asserted
they agreed. Adding a profile to one and not the other does not raise: it indexes under the wrong
context mode and produces vectors that silently disagree with the ones already stored.

Two design choices carry that guarantee, and both are load-bearing rather than stylistic:

* **`context_version` is derived, never declared.** `Indexer.__init__` refuses an embedder whose
  profile does not spell its context exactly `raw-v1` or `context-<mode>-<policy version>`, so the
  derivation here IS that contract, written at the only place a profile is defined. A second copy
  of the string cannot drift from the mode because there is no second copy.
* **`query_mode` / `passage_mode` name the encoder that will actually be called.** They are
  dispatch keys handed to the backend, not documentation of one. A profile that claims
  `query_embed` and gets `embed` would produce a query vector from the passage encoder, the exact
  aliasing this program exists to prevent, and the only way to make the claim checkable is to let
  it choose.

`artifact_digest` is `None` for every profile whose weights the operator provisions, because the
digest is a property of a deployment's artifact tree and not of the profile. Declaring a value
there would be inventing one. Where a digest IS a fact about the profile (the rejected Qwen
experiment, whose verdict belongs to exactly one artifact), it is pinned, and an operator-supplied
digest that differs is refused rather than silently inheriting the recorded verdict.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Mapping

from recall.embeddings import (
    Embedder,
    EmbeddingProfile,
    FastEmbedEmbedder,
    Qwen3EmbeddingEmbedder,
    _package_version,
)

#: How much of a document's surroundings is folded into a chunk's embedding text. Defined here
#: rather than in `recall.context` so the registry stays the root of the identity dependency
#: chain: `recall.context` imports this module, never the reverse.
ContextMode = Literal["none", "document", "section", "neighbor"]

#: Version of the contextual-passage RENDERING rule, distinct from the mode. Bumping it changes
#: the text handed to the embedder for every context profile, so it is part of `context_version`
#: and therefore of the cache key.
CONTEXT_POLICY_VERSION = "v1"

#: Backends a registered profile can be built on. The registry decides which class constructs a
#: profile so that adding one is a data change rather than another branch in `make_embedder`.
Backend = Literal["fastembed", "qwen3"]


def context_version_for(mode: ContextMode, policy_version: str = CONTEXT_POLICY_VERSION) -> str:
    """The one derivation of a context version, matching what `Indexer` enforces."""
    return "raw-v1" if mode == "none" else f"context-{mode}-{policy_version}"


@dataclass(frozen=True)
class RejectionRecord:
    """A profile that was measured, rejected, and kept.

    Deleting a rejected profile deletes the reason it was rejected, and the next session
    re-measures it. The record is what makes "do not reopen this" checkable instead of oral.
    """

    verdict: Literal["rejected"]
    reason: str
    decided_on: str
    revision: str
    artifact_digest: str
    reference_cpu_threads: int
    measurements: tuple[tuple[str, float], ...]
    note: str = ""

    def __post_init__(self) -> None:
        if not self.measurements:
            raise ValueError(
                "a rejection record must carry the measurement that decided it; a verdict with "
                "no number is an opinion the next session will re-litigate"
            )
        if len(self.artifact_digest) != 64:
            raise ValueError("a rejection record's artifact digest must be a SHA256")


@dataclass(frozen=True)
class RegisteredProfile:
    """One embedding profile's complete, immutable identity."""

    profile_id: str
    model_name: str
    dimension: int
    query_mode: str
    passage_mode: str
    context_mode: ContextMode
    backend: Backend
    normalization: str = "l2"
    instruction_version: str = "none"
    chunker_version: str = "chunk-text-v1"
    #: None when the operator provisions the weights; pinned when the digest is a fact about the
    #: profile itself (see the module docstring).
    artifact_digest: str | None = None
    rejection: RejectionRecord | None = None
    #: Environment variable naming the provisioned artifact tree for this backend.
    artifact_path_env: str = "RECALL_MODEL_CACHE"

    def __post_init__(self) -> None:
        if self.dimension < 1:
            raise ValueError("a registered profile needs a positive dimension")
        if self.artifact_digest is not None and len(self.artifact_digest) != 64:
            raise ValueError("a pinned artifact digest must be a SHA256")

    @property
    def context_version(self) -> str:
        return context_version_for(self.context_mode)

    @property
    def rejected(self) -> bool:
        return self.rejection is not None

    def identity(
        self,
        *,
        artifact_digest: str | None = None,
        dependencies: tuple[tuple[str, str], ...] = (),
    ) -> EmbeddingProfile:
        """Build the runtime identity, refusing anything but this profile's own artifact.

        The only constructor of an `EmbeddingProfile` for a registered profile. A second one
        would be a second identity, which is the defect the registry removes.
        """
        digest = artifact_digest or self.artifact_digest
        if not digest:
            raise ValueError(
                f"profile {self.profile_id!r} needs an operator-supplied artifact digest: its "
                "weights are provisioned per deployment and the identity cannot be completed "
                "without one"
            )
        if self.artifact_digest is not None and digest.lower() != self.artifact_digest.lower():
            raise ValueError(
                f"profile {self.profile_id!r} has a pinned artifact digest and the supplied one "
                f"differs; a different artifact tree is a different experiment and does not "
                f"inherit this profile's recorded verdict"
            )
        return EmbeddingProfile(
            profile_id=self.profile_id,
            model_name=self.model_name,
            artifact_digest=digest,
            dimension=self.dimension,
            query_mode=self.query_mode,
            passage_mode=self.passage_mode,
            normalization=self.normalization,
            instruction_version=self.instruction_version,
            chunker_version=self.chunker_version,
            context_version=self.context_version,
            dependencies=dependencies,
        )

    def build(self, *, artifact_path: str | Path, artifact_digest: str) -> Embedder:
        """Construct this profile's embedder from a provisioned local artifact.

        The single construction site for a registered profile. It exists so that the identity
        the runtime carries and the identity the registry declares are the same object rather
        than two builds of the same fields: `make_embedder` parses environment variables and
        delegates here, and nothing else assembles an `EmbeddingProfile` for a registered ID.

        Offline is not optional on this path. The digest is verified against the artifact tree
        before any model loads, and the backend is told to refuse a network fetch.
        """
        identity = self.identity(
            artifact_digest=artifact_digest,
            dependencies=((self._dependency, _package_version(self._dependency)),),
        )
        if self.backend == "fastembed":
            return FastEmbedEmbedder(
                cache_dir=artifact_path,
                artifact_sha256=identity.artifact_digest,
                require_local=True,
                identity=identity,
            )
        return Qwen3EmbeddingEmbedder(
            artifact_path, identity.artifact_digest, identity=identity
        )

    @property
    def _dependency(self) -> str:
        return "fastembed" if self.backend == "fastembed" else "sentence-transformers"


_BGE_SMALL = "BAAI/bge-small-en-v1.5"
_BGE_BASE = "BAAI/bge-base-en-v1.5"
_BGE_LARGE = "BAAI/bge-large-en-v1.5"
_MINILM_L6 = "sentence-transformers/all-MiniLM-L6-v2"
_MINILM_MULTILINGUAL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_ARCTIC_XS = "snowflake/snowflake-arctic-embed-xs"

#: Measured on VPS2 on 2026-08-03 against the provisioned artifact at a four-thread budget, then
#: rejected on CPU latency. Retained verbatim: `/opt/recall-enterprise/qwen-benchmark-result.json`
#: and the deployment manifest are the source, and both live outside any git ref.
_QWEN_REJECTION = RejectionRecord(
    verdict="rejected",
    reason="cpu-latency",
    decided_on="2026-08-03",
    revision="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
    artifact_digest="0e9f06588b7e661b8d8e6d393b5936750e428ec422f9971c7f02838dbe70fc9f",
    reference_cpu_threads=4,
    measurements=(
        ("query_p50_ms", 4638.83),
        ("query_p95_ms", 5816.34),
        ("passage_batch_20_p50_ms", 41016.64),
        ("load_ms", 24558.4),
        ("max_rss_mb", 1739.47),
    ),
    note=(
        "Rejected on serving latency alone; retrieval quality was never measured against "
        "bge-small-asymmetric-v1, so nothing here says the model is worse at retrieval. "
        "A query p95 of 5.8 s and a 20-passage batch p50 of 41 s are both orders of magnitude "
        "outside the fast profile's 250 ms budget."
    ),
)

_PROFILES: tuple[RegisteredProfile, ...] = (
    RegisteredProfile(
        profile_id="bge-small-symmetric-v1",
        model_name=_BGE_SMALL,
        dimension=384,
        query_mode="embed",
        passage_mode="embed",
        context_mode="none",
        backend="fastembed",
    ),
    RegisteredProfile(
        profile_id="bge-small-asymmetric-v1",
        model_name=_BGE_SMALL,
        dimension=384,
        query_mode="query_embed",
        passage_mode="passage_embed",
        context_mode="none",
        backend="fastembed",
    ),
    RegisteredProfile(
        profile_id="bge-small-context-document-v1",
        model_name=_BGE_SMALL,
        dimension=384,
        query_mode="query_embed",
        passage_mode="passage_embed",
        context_mode="document",
        backend="fastembed",
    ),
    RegisteredProfile(
        profile_id="bge-small-context-section-v1",
        model_name=_BGE_SMALL,
        dimension=384,
        query_mode="query_embed",
        passage_mode="passage_embed",
        context_mode="section",
        backend="fastembed",
    ),
    RegisteredProfile(
        profile_id="bge-small-context-neighbor-v1",
        model_name=_BGE_SMALL,
        dimension=384,
        query_mode="query_embed",
        passage_mode="passage_embed",
        context_mode="neighbor",
        backend="fastembed",
    ),
    #: bge-base and bge-large exist so a corpus built at 768 or 1024 dimensions can be SERVED at
    #: all. `make_embedder` builds a registered profile or nothing, and plain `fastembed` is
    #: bge-small, so before these a 1024-dim corpus was unreachable from the MCP server — measured
    #: on a live 8,671-chunk bge-large corpus, where every one of the ten tools failed with
    #: `unknown embedder` before any search ran.
    #:
    #: Registering is deliberately preferred over letting `make_embedder` fall through to
    #: `resolve_embedder`. That fallback was written, audited and rejected: a resolver-built
    #: fastembed embedder carries no identity, so `embeddings.py:794-801` stamps EVERY fastembed
    #: model with `bge-small-symmetric-v1`, and two models of the same width become
    #: indistinguishable to readiness, lineage and calibration alike. `build()` passes a real
    #: identity, so each of these carries its own id and its own fingerprint.
    #:
    #: `context_mode="none"` and symmetric `embed` match how these corpora are indexed by
    #: default; the asymmetric and context-carrying variants are separate profiles for bge-small
    #: and would be separate profiles here too, not flags on these.
    RegisteredProfile(
        profile_id="bge-base-symmetric-v1",
        model_name=_BGE_BASE,
        dimension=768,
        query_mode="embed",
        passage_mode="embed",
        context_mode="none",
        backend="fastembed",
    ),
    RegisteredProfile(
        profile_id="bge-large-symmetric-v1",
        model_name=_BGE_LARGE,
        dimension=1024,
        query_mode="embed",
        passage_mode="embed",
        context_mode="none",
        backend="fastembed",
    ),
    RegisteredProfile(
        profile_id="bge-large-asymmetric-v1",
        model_name=_BGE_LARGE,
        dimension=1024,
        query_mode="query_embed",
        passage_mode="passage_embed",
        context_mode="none",
        backend="fastembed",
    ),
    #: The cheap local end. All three are fastembed-supported and CPU-friendly, and exist so a
    #: small or offline deployment has a registered option rather than reaching for the resolver.
    RegisteredProfile(
        profile_id="minilm-l6-symmetric-v1",
        model_name=_MINILM_L6,
        dimension=384,
        query_mode="embed",
        passage_mode="embed",
        context_mode="none",
        backend="fastembed",
    ),
    RegisteredProfile(
        profile_id="minilm-multilingual-symmetric-v1",
        model_name=_MINILM_MULTILINGUAL,
        dimension=384,
        query_mode="embed",
        passage_mode="embed",
        context_mode="none",
        backend="fastembed",
    ),
    RegisteredProfile(
        profile_id="arctic-embed-xs-symmetric-v1",
        model_name=_ARCTIC_XS,
        dimension=384,
        query_mode="embed",
        passage_mode="embed",
        context_mode="none",
        backend="fastembed",
    ),
    RegisteredProfile(
        profile_id="qwen3-embedding-0.6b-384-v1",
        model_name="Qwen/Qwen3-Embedding-0.6B",
        dimension=384,
        query_mode="instruction-v1",
        passage_mode="document",
        context_mode="none",
        backend="qwen3",
        instruction_version="retrieval-v1",
        artifact_digest=_QWEN_REJECTION.artifact_digest,
        artifact_path_env="RECALL_QWEN_MODEL_PATH",
        rejection=_QWEN_REJECTION,
    ),
)

REGISTERED_PROFILES: Mapping[str, RegisteredProfile] = MappingProxyType(
    {entry.profile_id: entry for entry in _PROFILES}
)


def registered_profile_ids() -> tuple[str, ...]:
    return tuple(REGISTERED_PROFILES)


def registered_profile(profile_id: str) -> RegisteredProfile:
    """Look one up, refusing an unknown ID rather than defaulting it."""
    try:
        return REGISTERED_PROFILES[profile_id]
    except KeyError:
        raise ValueError(
            f"unknown embedding profile: {profile_id!r} "
            f"(registered: {', '.join(REGISTERED_PROFILES)})"
        ) from None


def find_registered_profile(profile_id: str) -> RegisteredProfile | None:
    """Look one up without refusing. For paths that must tolerate a legacy embedder."""
    return REGISTERED_PROFILES.get(profile_id)
