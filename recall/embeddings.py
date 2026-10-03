from __future__ import annotations

import os

from recall._env import truthy
from recall.embedding_core import (  # re-exported: this module is their public home
    _R as _R,
    _TRANSIENT_MARKERS as _TRANSIENT_MARKERS,
    NonTransientError as NonTransientError,
    _probe as _probe,
    _is_transient as _is_transient,
    _MAX_RETRY_AFTER_S as _MAX_RETRY_AFTER_S,
    _retry_after_seconds as _retry_after_seconds,
    _read_retry_after as _read_retry_after,
    _capped as _capped,
    retry_with_backoff as retry_with_backoff,
    batched_embed as batched_embed,
    _batched_embed_concurrently as _batched_embed_concurrently,
    _checked as _checked,
    Embedder as Embedder,
    EmbeddingPurpose as EmbeddingPurpose,
    AsymmetricEmbedder as AsymmetricEmbedder,
    GroupedDocumentEmbedder as GroupedDocumentEmbedder,
    EmbeddingProfile as EmbeddingProfile,
    _package_version as _package_version,
    LEGACY_UNVERIFIED_DIGEST as LEGACY_UNVERIFIED_DIGEST,
    HOSTED_UNVERIFIED_DIGEST as HOSTED_UNVERIFIED_DIGEST,
    UNVERIFIED_ARTIFACT_DIGESTS as UNVERIFIED_ARTIFACT_DIGESTS,
    artifact_is_pinned as artifact_is_pinned,
    _check_declared_width as _check_declared_width,
    legacy_embedding_profile as legacy_embedding_profile,
    embedding_profile as embedding_profile,
    embedding_profile_id as embedding_profile_id,
    resolve_registered_embedder as resolve_registered_embedder,
    embed_query as embed_query,
    embed_passages as embed_passages,
    embed_document_groups as embed_document_groups,
    artifact_tree_sha256 as artifact_tree_sha256,
    _ARTIFACT_DIGESTS as _ARTIFACT_DIGESTS,
    _ARTIFACT_DIGEST_LIMIT as _ARTIFACT_DIGEST_LIMIT,
    _artifact_signature as _artifact_signature,
    embedder_artifact_path as embedder_artifact_path,
    embedder_artifact_digest as embedder_artifact_digest,
    verify_artifact as verify_artifact,
    HashingEmbedder as HashingEmbedder,
)
from recall.embedding_providers.local import (  # re-exported: this module is their public home
    resolve_thread_budget as resolve_thread_budget,
    PROVIDERS_UNKNOWN as PROVIDERS_UNKNOWN,
    _provider_dependencies as _provider_dependencies,
    _with_provider_dependency as _with_provider_dependency,
    _session_providers as _session_providers,
    _LEGACY_FALLBACK_PROFILE_IDS as _LEGACY_FALLBACK_PROFILE_IDS,
    _fallback_profile_id as _fallback_profile_id,
    _WARNED_BATCH_VALUES as _WARNED_BATCH_VALUES,
    _warn_once as _warn_once,
    _batch_size_from_env as _batch_size_from_env,
    FastEmbedEmbedder as FastEmbedEmbedder,
    _accepts_batch_size as _accepts_batch_size,
    SFR_CODE_EMBEDDER_MODEL as SFR_CODE_EMBEDDER_MODEL,
    SFR_CODE_EMBEDDER_REVISION as SFR_CODE_EMBEDDER_REVISION,
    REMOTE_MODEL_CODE_OPT_IN as REMOTE_MODEL_CODE_OPT_IN,
    _require_research_model_opt_in as _require_research_model_opt_in,
    _require_remote_model_code_opt_in as _require_remote_model_code_opt_in,
    SentenceTransformerEmbedder as SentenceTransformerEmbedder,
    QWEN3_RETRIEVAL_INSTRUCTION_V1 as QWEN3_RETRIEVAL_INSTRUCTION_V1,
    Qwen3EmbeddingEmbedder as Qwen3EmbeddingEmbedder,
)
from recall.embedding_providers.voyage import (  # re-exported: this module is their public home
    _voyage_client_class as _voyage_client_class,
    VOYAGE_CONTEXT_WINDOW_TOKENS as VOYAGE_CONTEXT_WINDOW_TOKENS,
    VOYAGE_CONTEXT_TOKEN_HEADROOM as VOYAGE_CONTEXT_TOKEN_HEADROOM,
    _FALLBACK_SPECIAL_TOKENS_PER_CHUNK as _FALLBACK_SPECIAL_TOKENS_PER_CHUNK,
    TokenCounter as TokenCounter,
    _voyage_token_counter as _voyage_token_counter,
    _utf8_token_bound as _utf8_token_bound,
    VoyageContextualizedEmbedder as VoyageContextualizedEmbedder,
    VoyageEmbedder as VoyageEmbedder,
)
from recall.embedding_providers.openai_compat import (  # re-exported: this module is their public home
    _OPENAI_COMPAT_REMOTE_HOSTS as _OPENAI_COMPAT_REMOTE_HOSTS,
    _OPENAI_COMPAT_LOCAL_HOSTS as _OPENAI_COMPAT_LOCAL_HOSTS,
    _OPENAI_COMPAT_REMOTE_PATHS as _OPENAI_COMPAT_REMOTE_PATHS,
    _OPENAI_COMPAT_MAX_BASE_URL_LENGTH as _OPENAI_COMPAT_MAX_BASE_URL_LENGTH,
    _OPENAI_COMPAT_MAX_PATH_LENGTH as _OPENAI_COMPAT_MAX_PATH_LENGTH,
    _OPENAI_COMPAT_MAX_PATH_SEGMENTS as _OPENAI_COMPAT_MAX_PATH_SEGMENTS,
    OpenAICompatEmbedder as OpenAICompatEmbedder,
    _optional_dimensions as _optional_dimensions,
)


def resolve_embedder(name: str, env: dict[str, str] | None = None) -> Embedder:
    """Build an embedder from a short config string.

    Supported spellings:
    ``hashing``, ``fastembed``, ``fastembed:<model>``, ``st:<model>``,
    ``sfr-code``, ``voyage``, ``voyage:<model>``, ``voyage-context``,
    ``voyage-context:<model>``, ``voyage-multimodal``, ``openai``, ``openai:<model>``,
    ``openrouter`` and ``openrouter:<model>``.
    """
    source = os.environ if env is None else env
    profile = source.get("RECALL_EMBED_PROFILE", "").strip()
    if profile:
        from recall.embedding_registry import find_registered_profile

        entry = find_registered_profile(profile)
        if entry is None:
            return resolve_registered_embedder(profile, source)
        if entry.backend == "voyage-multimodal" and not truthy(
            source.get("RECALL_MULTIMODAL_ENABLED", "0")
        ):
            raise ValueError(
                "voyage-multimodal is disabled; set RECALL_MULTIMODAL_ENABLED=1 to opt in"
            )
        accepted = {
            "fastembed": frozenset({"fastembed"}),
            "qwen3": frozenset({"fastembed"}),
            "voyage": frozenset({"voyage"}),
            "voyage-context": frozenset({"voyage-context"}),
            "voyage-multimodal": frozenset({"voyage-multimodal"}),
            "openai-compat": frozenset({"openai", "openrouter"}),
        }[entry.backend]
        if name not in accepted:
            choices = " or ".join(f"RECALL_EMBEDDER={value}" for value in sorted(accepted))
            raise ValueError(
                f"RECALL_EMBED_PROFILE={profile!r} is a {entry.backend} profile and needs {choices}"
            )
        return resolve_registered_embedder(profile, source)
    if name == "hashing" or name.startswith("hashing-") or name.startswith("hashing:"):
        return HashingEmbedder(dim=64)
    if name == "fastembed":
        return FastEmbedEmbedder(env=source)
    if name.startswith("fastembed:"):
        return FastEmbedEmbedder(model_name=name[len("fastembed:"):], env=source)
    if name.startswith("st:"):
        return SentenceTransformerEmbedder(name[3:])
    if name == "sfr-code":
        _require_research_model_opt_in(source, SFR_CODE_EMBEDDER_MODEL)
        _require_remote_model_code_opt_in(source, SFR_CODE_EMBEDDER_MODEL)
        return SentenceTransformerEmbedder(
            SFR_CODE_EMBEDDER_MODEL,
            trust_remote_code=True,
            revision=SFR_CODE_EMBEDDER_REVISION,
            name=f"sfr-code:{SFR_CODE_EMBEDDER_MODEL}",
        )
    if name == "voyage":
        return VoyageEmbedder(api_key=source.get("VOYAGE_API_KEY"))
    if name.startswith("voyage:"):
        return VoyageEmbedder(
            model=name[len("voyage:"):], api_key=source.get("VOYAGE_API_KEY")
        )
    if name == "voyage-context" or name.startswith("voyage-context:"):
        model = name[len("voyage-context:"):] if name.startswith("voyage-context:") else "voyage-context-4"
        if model != "voyage-context-4":
            raise ValueError(
                f"unsupported Voyage Context model: {model!r} "
                "(the registered production profile is voyage-context-4-v1)"
            )
        # Direct CLI selection and RECALL_EMBED_PROFILE selection must converge on the same
        # registered identity. Otherwise a direct `voyage-context:voyage-context-4` build would
        # carry only the hosted legacy profile while the profile based path would carry the
        # grouping policy and request limits that make the vectors reproducible.
        return resolve_registered_embedder(
            "voyage-context-4-v1",
            {**source, "RECALL_EMBED_PROFILE": "voyage-context-4-v1"},
        )
    if name == "voyage-multimodal" or name.startswith("voyage-multimodal:"):
        if not truthy(source.get("RECALL_MULTIMODAL_ENABLED", "0")):
            raise ValueError(
                "voyage-multimodal is disabled; set RECALL_MULTIMODAL_ENABLED=1 to opt in"
            )
        model = (
            name[len("voyage-multimodal:") :]
            if name.startswith("voyage-multimodal:")
            else "voyage-multimodal-3.5"
        )
        if model != "voyage-multimodal-3.5":
            raise ValueError(
                f"unsupported Voyage multimodal model: {model!r} "
                "(the registered production profile is voyage-multimodal-3.5-v1)"
            )
        return resolve_registered_embedder(
            "voyage-multimodal-3.5-v1",
            {**source, "RECALL_EMBED_PROFILE": "voyage-multimodal-3.5-v1"},
        )
    if name == "openai":
        return OpenAICompatEmbedder(
            api_key=source.get("OPENROUTER_API_KEY"),
            dimensions=_optional_dimensions(source),
        )
    if name.startswith("openai:"):
        return OpenAICompatEmbedder(
            model=name[len("openai:"):],
            api_key=source.get("OPENROUTER_API_KEY"),
            dimensions=_optional_dimensions(source),
        )
    if name == "openrouter":
        return OpenAICompatEmbedder(
            model="google/gemini-embedding-2",
            api_key=source.get("OPENROUTER_API_KEY"),
            dimensions=_optional_dimensions(source),
            name_prefix="openrouter",
        )
    if name == "gemini-embedding-2":
        return OpenAICompatEmbedder(
            model="google/gemini-embedding-2",
            api_key=source.get("OPENROUTER_API_KEY"),
            dimensions=_optional_dimensions(source),
            name_prefix="openrouter",
        )
    if name.startswith("openrouter:"):
        return OpenAICompatEmbedder(
            model=name[len("openrouter:"):],
            api_key=source.get("OPENROUTER_API_KEY"),
            dimensions=_optional_dimensions(source),
            name_prefix="openrouter",
        )
    raise ValueError(
        f"unknown embedder: {name!r} (use hashing, fastembed, fastembed:<model>, "
        "st:<model>, sfr-code, voyage, voyage:<model>, voyage-context, "
        "voyage-context:<model>, voyage-multimodal, openai, openai:<model>, "
        "openrouter, or openrouter:<model>)"
    )


def embedder_is_hosted(embedder: object) -> bool:
    """Whether a provider's API produced this embedder's vectors, rather than local weights.

    🔑 **Asked of the LIVE embedder, never of a provider string, and that is not a stylistic
    preference.** `BuildRequest.provider` defaults to `_DEFAULT_PROVIDER`, which is the literal
    ``"fastembed"``, and the ingest path passes no override — so a `voyage:voyage-4` upload records
    ``provider="fastembed"`` and a string check would answer "local" for the exact case this
    function exists to admit. The generation that this repository's own memory corpus is served
    from stores ``profile_id: null`` as well, so keying on the registered profile alone would miss
    it too. Verified against that record on 2026-08-26 before this was written.

    Two signals, because either alone has a measured blind spot:

    * a registered profile whose digest is the hosted marker, which is exact when it is present;
    * the two hosted embedder classes, which covers an embedder built without a registered profile.

    Adding a third hosted backend means adding it here. That is a real maintenance edge, and it is
    deliberately preferred to a `hasattr` duck-type, which would silently admit any object that
    happened to grow a matching attribute.
    """
    profile = getattr(embedder, "profile", None)
    if isinstance(profile, EmbeddingProfile) and profile.artifact_digest == HOSTED_UNVERIFIED_DIGEST:
        return True
    from recall.multimodal import VoyageMultimodalEmbedder

    return isinstance(
        embedder,
        (VoyageEmbedder, VoyageContextualizedEmbedder, VoyageMultimodalEmbedder, OpenAICompatEmbedder),
    )
