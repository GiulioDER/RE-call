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
    HOSTED_UNVERIFIED_DIGEST,
    Embedder,
    EmbeddingProfile,
    FastEmbedEmbedder,
    OpenAICompatEmbedder,
    Qwen3EmbeddingEmbedder,
    VoyageEmbedder,
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
Backend = Literal["fastembed", "qwen3", "voyage", "openai-compat"]

#: Backends served by a provider's API rather than by weights the operator provisions. They differ
#: from local backends in exactly two ways, and both are consequences of the same fact: nobody
#: can hash what they did not receive:
#:
#: 1. There is no artifact to verify, so `identity()` completes with `HOSTED_UNVERIFIED_DIGEST`
#:    instead of demanding an operator-supplied SHA, and `recall.readiness` refuses to certify a
#:    process serving one. A hosted profile is servable; it is not ATTESTABLE.
#: 2. The declared dimension is the only check available, so it is enforced at construction
#:    (`_check_declared_width`) against the width the endpoint actually returns.
HOSTED_BACKENDS: frozenset[str] = frozenset({"voyage", "openai-compat"})


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
    #: Hosted only. The endpoint the model is served from. Part of the profile because the same
    #: model name resolves to different weights at different providers, and because `build()`
    #: could not otherwise reach anything but its class default.
    base_url: str | None = None
    #: Hosted only. Namespace for the embedder's legacy `name`, kept so a hosted embedder built
    #: WITHOUT an identity still spells itself the way `resolve_embedder` always has.
    name_prefix: str = "openai"
    #: Hosted only. Explicit output width sent as the API's `dimensions` field. `None` asks for
    #: the model's native width. Declared next to `dimension` so the request and the claim cannot
    #: drift apart; `_check_declared_width` holds the provider to the pair.
    output_dimensions: int | None = None
    #: Environment variable naming this profile's API credential.
    api_key_env: str = "OPENROUTER_API_KEY"

    def __post_init__(self) -> None:
        if self.dimension < 1:
            raise ValueError("a registered profile needs a positive dimension")
        if self.artifact_digest is not None and len(self.artifact_digest) != 64:
            raise ValueError("a pinned artifact digest must be a SHA256")
        if self.hosted and self.artifact_digest is not None:
            raise ValueError(
                f"profile {self.profile_id!r} is hosted and cannot pin an artifact digest: "
                "the provider serves weights it may replace behind this model name, so a pinned "
                "digest would be a claim nothing can check"
            )
        if not self.hosted and self.base_url is not None:
            raise ValueError(
                f"profile {self.profile_id!r} is local and has no endpoint to declare"
            )
        if self.output_dimensions is not None:
            if not self.hosted:
                raise ValueError(
                    f"profile {self.profile_id!r} is local; its width comes from the artifact"
                )
            if self.output_dimensions != self.dimension:
                raise ValueError(
                    f"profile {self.profile_id!r} asks the provider for "
                    f"{self.output_dimensions} dimensions but declares {self.dimension}"
                )
            if self.backend == "voyage":
                # `VoyageEmbedder` has no `dimensions` parameter, so this field would be accepted
                # here, dropped on the way to the provider, and the profile would quietly get the
                # model's default width instead of the one it asked for. Refusing beats a request
                # parameter that silently does nothing. `_check_declared_width` would catch the
                # resulting mismatch, but only for a value that differs from the default, and a
                # declaration that is ignored should not depend on a downstream check to notice.
                raise ValueError(
                    f"profile {self.profile_id!r} sets output_dimensions, which the Voyage "
                    "client does not send; register the model's default width instead"
                )

    @property
    def context_version(self) -> str:
        return context_version_for(self.context_mode)

    @property
    def rejected(self) -> bool:
        return self.rejection is not None

    @property
    def hosted(self) -> bool:
        """Whether a provider's API serves this profile rather than a local artifact tree."""
        return self.backend in HOSTED_BACKENDS

    def identity(
        self,
        *,
        artifact_digest: str | None = None,
        dependencies: tuple[tuple[str, str], ...] = (),
    ) -> EmbeddingProfile:
        """Build the runtime identity for this profile.

        The only constructor of an `EmbeddingProfile` for a registered profile. A second one
        would be a second identity, which is the defect the registry removes.

        For a LOCAL profile it refuses anything but this profile's own artifact. For a HOSTED one
        it refuses an artifact digest ENTIRELY and completes with `HOSTED_UNVERIFIED_DIGEST`,
        because a provider serving weights it can replace behind a stable model name leaves
        nothing to hash; accepting a digest there would record a verification that never happened.
        """
        if self.hosted:
            if artifact_digest:
                raise ValueError(
                    f"profile {self.profile_id!r} is hosted and cannot accept an artifact "
                    "digest; there is no artifact tree to have hashed"
                )
            digest: str | None = HOSTED_UNVERIFIED_DIGEST
        else:
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

    def build(
        self,
        *,
        artifact_path: str | Path | None = None,
        artifact_digest: str | None = None,
        api_key: str | None = None,
    ) -> Embedder:
        """Construct this profile's embedder, local or hosted.

        The single construction site for a registered profile. It exists so that the identity
        the runtime carries and the identity the registry declares are the same object rather
        than two builds of the same fields: `make_embedder` parses environment variables and
        delegates here, and nothing else assembles an `EmbeddingProfile` for a registered ID.

        For a LOCAL backend, offline is not optional: the digest is verified against the artifact
        tree before any model loads, and the backend is told to refuse a network fetch.

        For a HOSTED backend there is no artifact and no offline mode, so the guarantee is a
        narrower one and worth naming exactly. What this path still gives is that the identity the
        runtime carries is the identity the registry declared (the profile id, the width, the
        context version and the encoder modes), so a generation and a calibration registered under
        that id can be matched against the running process. What it cannot give is any evidence
        about WHICH weights answered, which is why `recall.readiness` refuses to certify one.

        The returned embedder always carries the identity. An earlier attempt at this feature
        built the identity here, validated it, and then dropped it on the floor because the hosted
        classes took no ``identity`` parameter; every consumer silently fell back to
        `legacy_embedding_profile`, the profile id reverted to ``voyage:voyage-code-3``, and the
        readiness check comparing it against the active generation could never match. The
        assertion below is what makes that failure loud instead of decorative.
        """
        if self.hosted:
            if artifact_path is not None or artifact_digest is not None:
                raise ValueError(
                    f"profile {self.profile_id!r} is hosted; it takes an api_key, not an "
                    "artifact path or digest"
                )
        elif not artifact_path or not artifact_digest:
            raise ValueError(
                f"profile {self.profile_id!r} is local and needs both an artifact path and an "
                "artifact digest"
            )
        identity = self.identity(
            artifact_digest=artifact_digest,
            dependencies=((self._dependency, _package_version(self._dependency)),),
        )
        embedder = self._construct(identity, artifact_path, api_key)
        # Not defensive programming: this is the single defect that made the previous attempt
        # decorative, and a class that silently drops the identity fails here rather than three
        # subsystems away when a generation refuses to match.
        #
        # Compared by `profile_id`, NOT by object identity. `FastEmbedEmbedder` legitimately
        # replaces the identity with an enriched copy (`_with_provider_dependency` folds the
        # resolved ONNX providers into `dependencies`), so an `is` check would refuse the one
        # backend that has always done this correctly. The id is what the defect destroyed and
        # what every downstream consumer matches on.
        built = getattr(embedder, "profile", None)
        if not isinstance(built, EmbeddingProfile) or built.profile_id != self.profile_id:
            raise AssertionError(
                f"{type(embedder).__name__} did not carry the identity {self.profile_id!r} "
                "built for it; every downstream consumer would fall back to a legacy profile"
            )
        return embedder

    def _construct(
        self,
        identity: EmbeddingProfile,
        artifact_path: str | Path | None,
        api_key: str | None,
    ) -> Embedder:
        if self.backend == "fastembed":
            return FastEmbedEmbedder(
                cache_dir=artifact_path,
                artifact_sha256=identity.artifact_digest,
                require_local=True,
                identity=identity,
            )
        if self.backend == "qwen3":
            # `build` refuses a local profile with no artifact path before reaching here; the
            # assert states that for the type checker rather than widening the constructor.
            assert artifact_path is not None
            return Qwen3EmbeddingEmbedder(
                artifact_path, identity.artifact_digest, identity=identity
            )
        if self.backend == "voyage":
            return VoyageEmbedder(api_key=api_key, identity=identity)
        assert self.base_url is not None  # enforced for every hosted profile in __post_init__
        return OpenAICompatEmbedder(
            api_key=api_key,
            base_url=self.base_url,
            dimensions=self.output_dimensions,
            name_prefix=self.name_prefix,
            identity=identity,
        )

    @property
    def _dependency(self) -> str:
        """The package whose version is key material for this profile's vectors.

        Folded into the identity's `dependencies`, so an upgrade re-partitions the embedding
        cache. For a hosted backend this names the CLIENT library, which is weaker than it looks:
        the client does not determine the vector, the provider does, and the provider is free to
        change it without changing anything recorded here. Stated rather than papered over,
        because it is the same gap `HOSTED_UNVERIFIED_DIGEST` exists to declare.
        """
        return {
            "fastembed": "fastembed",
            "qwen3": "sentence-transformers",
            "voyage": "voyageai",
            "openai-compat": "openai",
        }[self.backend]


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


#: Endpoint for every OpenAI-compatible profile below. OpenRouter rather than api.openai.com so
#: one key serves the generator, the judge and the embedder; `base_url` is per-profile precisely
#: so a deployment that wants api.openai.com can register its own entry instead of patching a
#: class default.
_OPENROUTER = "https://openrouter.ai/api/v1"

#: ⚠️ WIDTHS BELOW ARE MEASURED, EXCEPT VOYAGE'S. Measured 2026-08-18 against OpenRouter, one
#: request per model, reporting the returned vector length and its L2 norm:
#:
#:   openai/text-embedding-3-small   1536   norm 1.000430
#:   openai/text-embedding-3-large   3072   norm 1.000066
#:   google/gemini-embedding-001     3072   norm 1.000000
#:
#: Re-measure every hosted profile, including its normalization claim, and exit non-zero on
#: any disagreement (needs the provider keys; one embedding call per profile):
#:
#:   python scripts/measure_hosted_embedding_widths.py
#:
#: Two findings from that measurement are load-bearing here and would not survive being guessed:
#:
#: * **`normalization='l2'` is only true at gemini's NATIVE width.** `gemini-embedding-001`
#:   returns a unit vector at 3072 but NOT at its Matryoshka widths: measured norm 0.694 at 1536
#:   and 0.582 at 768. Truncated widths are therefore deliberately not registered, because the
#:   profile would be declaring a normalization the provider does not apply, and cosine scores
#:   against an abstention threshold calibrated on unit vectors would be quietly wrong.
#: * **OpenRouter's `/v1/models` lists no embedding model at all** (412 models, zero matching
#:   "embed" on 2026-08-18) while `/v1/embeddings` serves them regardless. So the catalogue
#:   endpoint cannot be used to validate these ids, and an id absent from it is not evidence of
#:   anything.
_HOSTED_PROFILES: tuple[RegisteredProfile, ...] = (
    # --- Voyage -------------------------------------------------------------------------------
    # ⚠️ These two widths are the provider's DOCUMENTED defaults and are NOT measured here: this
    # machine has no VOYAGE_API_KEY. That is a real gap, and it is guarded rather than hidden:
    # `_check_declared_width` refuses at construction if the endpoint disagrees, so a wrong
    # declaration costs a loud startup failure and never a corpus of mislabelled vectors.
    # Re-measure with the snippet above against https://api.voyageai.com/v1/embeddings.
    #
    # `voyage-code-3` is the profile the 21 existing code corpora (37,160 chunks) were built
    # under, which is the reason this feature exists. It also serves 2048/512/256 on request, so
    # the 1024 default is load-bearing rather than incidental: a corpus built at another width is
    # a DIFFERENT profile and needs its own entry, not a reused id.
    RegisteredProfile(
        profile_id="voyage-code-3-v1",
        model_name="voyage-code-3",
        dimension=1024,
        query_mode="embed",
        passage_mode="embed",
        context_mode="none",
        backend="voyage",
        api_key_env="VOYAGE_API_KEY",
    ),
    RegisteredProfile(
        profile_id="voyage-3-v1",
        model_name="voyage-3",
        dimension=1024,
        query_mode="embed",
        passage_mode="embed",
        context_mode="none",
        backend="voyage",
        api_key_env="VOYAGE_API_KEY",
    ),
    # --- OpenAI, via OpenRouter ---------------------------------------------------------------
    # The id carries its provider prefix. Measured 2026-08-18, the BARE `text-embedding-3-small`
    # also answers on OpenRouter at the same 1536 width, so the prefix is not required by the
    # endpoint; it is required by this registry, because an unprefixed name does not say which
    # provider's weights produced a stored vector and that is the one thing a profile id exists
    # to say.
    RegisteredProfile(
        profile_id="openai-text-embedding-3-small-v1",
        model_name="openai/text-embedding-3-small",
        dimension=1536,
        query_mode="embed",
        passage_mode="embed",
        context_mode="none",
        backend="openai-compat",
        base_url=_OPENROUTER,
        name_prefix="openai",
    ),
    RegisteredProfile(
        profile_id="openai-text-embedding-3-large-v1",
        model_name="openai/text-embedding-3-large",
        dimension=3072,
        query_mode="embed",
        passage_mode="embed",
        context_mode="none",
        backend="openai-compat",
        base_url=_OPENROUTER,
        name_prefix="openai",
    ),
    # --- Google, via OpenRouter ---------------------------------------------------------------
    # Registered at 3072 ONLY; see the normalization finding above for why the narrower
    # Matryoshka widths are absent rather than merely unlisted.
    RegisteredProfile(
        profile_id="gemini-embedding-001-v1",
        model_name="google/gemini-embedding-001",
        dimension=3072,
        query_mode="embed",
        passage_mode="embed",
        context_mode="none",
        backend="openai-compat",
        base_url=_OPENROUTER,
        name_prefix="openrouter",
    ),
)

REGISTERED_PROFILES: Mapping[str, RegisteredProfile] = MappingProxyType(
    {entry.profile_id: entry for entry in _PROFILES + _HOSTED_PROFILES}
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
