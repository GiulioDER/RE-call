"""Embedders that run a model on this machine: fastembed, sentence-transformers, Qwen3.
"""

from __future__ import annotations

import inspect
import math
import os
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

from recall._env import truthy
from recall.embedding_core import (
    LEGACY_UNVERIFIED_DIGEST,
    EmbeddingProfile,
    _package_version,
    verify_artifact,
)
from recall.observability import get_logger


def resolve_thread_budget(
    env: dict[str, str] | None = None, cpu_count: int | None = None
) -> int | None:
    """Threads for the local embedder, or None to leave fastembed's default alone.

    Exists for one situation: several embedding processes sharing a CPU budget. In an unprivileged
    container `os.cpu_count()` reports the HOST's cores while a cgroup quota caps real runtime, and
    fastembed sizes its pool from the former — so N workers each request N x (host cores). Measured
    on a box showing 256 CPUs against a ~61-CPU quota, seven workers spawned ~945 threads and
    aggregate throughput fell to roughly what a single process managed alone.

    Returns None when unset, deliberately. One process is FASTER with the default: measured
    2.2 / 5.7 / 9.0 / 10.3 docs/s at 1 / 8 / 32 / default threads, monotonically increasing. A cap
    must therefore never be the default — it fixes a multi-process regime and pessimises every
    other one.

    Junk is ignored rather than raised: a typo'd variable must not kill hour eight of a long run.
    """
    import os as _os

    raw = (env if env is not None else _os.environ).get("RECALL_EMBED_THREADS")
    if not raw:
        return None
    try:
        want = int(raw)
    except ValueError:
        return None
    if want < 1:
        return None
    ceiling = cpu_count if cpu_count is not None else (_os.cpu_count() or want)
    return min(want, ceiling)


#: What `_session_providers` returns when fastembed's internals do not expose a session. It is a
#: THIRD state, distinct from both a known CPU run and a known GPU run, and it must never be
#: allowed to read as either.
PROVIDERS_UNKNOWN = "<session not reachable>"


def _provider_dependencies(providers: Iterable[str]) -> tuple[tuple[str, str], ...]:
    """The execution provider as fingerprint key material, plus whether it is actually KNOWN.

    Two entries, not one. The provider list alone cannot distinguish "this ran on CPU" from "I
    could not tell what this ran on", and collapsing those two into one key is the same
    negative-guard mistake as reporting an unrecorded session as a match: `_session_providers` is
    documented as observational and must never fail a run, so its failure sentinel reaches here
    on perfectly healthy deployments. Recording the SOURCE alongside the value keeps an
    introspection failure legible in the profile instead of silently minting a new identity that
    looks like a real provider change.
    """
    known = [p for p in providers]
    unknown = not known or known == [PROVIDERS_UNKNOWN]
    return (
        ("onnx-providers-source", "unavailable" if unknown else "session"),
        ("onnx-providers", "unavailable" if unknown else ",".join(known)),
    )


def _with_provider_dependency(
    identity: EmbeddingProfile, providers: Iterable[str]
) -> EmbeddingProfile:
    """Attach the provider entries to a REGISTERED profile without mutating the registry's copy.

    `EmbeddingProfile` is frozen, so this replaces rather than edits. Idempotent: a profile that
    already carries the entries is returned unchanged, so re-wrapping cannot double them and
    change the fingerprint a second time.
    """
    from dataclasses import replace

    if any(k == "onnx-providers" for k, _ in identity.dependencies):
        return identity
    return replace(
        identity, dependencies=identity.dependencies + _provider_dependencies(providers)
    )


def _session_providers(model: object) -> list[str]:
    """The execution providers of fastembed's live ONNX session, or a marker if unreachable.

    fastembed does not expose the session on a stable public attribute, so this walks the couple
    of places it has lived. It returns a marker rather than raising: failing to introspect is not
    a reason to fail an embedding run, but it must not be reported as a known-CPU result either.
    """
    # Wrapped whole. This runs on the CONSTRUCTION path of every FastEmbedEmbedder, including
    # registered enterprise profiles, and it is purely observational — so no change in fastembed's
    # or onnxruntime's internals may ever be the reason a deployment cannot build its embedder.
    # The docstring promised that; nothing enforced it.
    try:
        for outer in ("model", "_model"):
            inner = getattr(model, outer, None)
            if inner is None:
                continue
            for attr in ("model", "session", "_session"):
                getter = getattr(getattr(inner, attr, None), "get_providers", None)
                if callable(getter):
                    return [str(p) for p in getter()]
            inner_getter = getattr(inner, "get_providers", None)
            if callable(inner_getter):
                return [str(p) for p in inner_getter()]
    except Exception:  # pragma: no cover - defensive; fastembed internals are not a contract  # BROAD-CATCH: fail-open
        return [PROVIDERS_UNKNOWN]
    # The COMMON path — fastembed simply exposing no session — must return the constant too. It
    # was left as a bare literal, equal by value today, so `_provider_dependencies` still matched
    # it. Change the constant's text and this path would have started recording the sentinel as a
    # provider NAME: exactly the collapse the constant exists to prevent.
    return [PROVIDERS_UNKNOWN]


#: The identifiers the no-identity path has always minted, keyed by ``asymmetric``.
#:
#: They name ONE model at ONE width. Kept as literals rather than derived so that the default
#: embedder's id is stable by construction: it is the key every shipped calibration file, every
#: recorded promotion decision under `results/`, and every corpus indexed by `FastEmbedEmbedder()`
#: is already written under, and re-deriving it would re-partition all of them for no defect.
_LEGACY_FALLBACK_PROFILE_IDS = {
    False: "bge-small-symmetric-v1",
    True: "bge-small-asymmetric-v1",
}


def _fallback_profile_id(model_name: str, dimension: int, asymmetric: bool) -> str:
    """The profile id for an embedder built with no registered identity and no explicit id.

    The legacy literal is returned ONLY when this embedder is the model that literal names, at
    the width the registry declares for it. Anything else gets an id derived from what actually
    varies, because the literal was previously unconditional and a `profile_id` is a CLAIM about
    which model wrote a vector, not a label. Measured 2026-08-18 before this guard existed: a
    `fastembed:BAAI/bge-large-en-v1.5` embedder reported ``dim=1024`` under
    ``profile_id='bge-small-symmetric-v1'``, an id whose registry entry is 384-dimensional, and a
    production corpus of 8,716 chunks had stored that pairing in its chunk metadata.

    Why the id and not just the fingerprint. `EmbeddingProfile.fingerprint` already covers
    `model_name` and `dimension`, so the embedding cache (`recall/cache.py`) never confused the
    two models. `recall.index._index_fingerprint` does not: it hashes `embedding_profile_id`
    alone, so under the unconditional literal a bge-small corpus and a bge-large corpus produced
    the SAME index fingerprint for the same file, and the incremental skip guard treated a model
    swap as a no-op. Verified by execution: the 384-dimension and 1024-dimension fingerprints were
    equal.

    The comparison covers `model_name` and `dimension` only. The encoder modes cannot disagree
    here: the legacy id is looked up BY ``asymmetric``, and both the registry pair and this
    fallback derive their modes from that same flag, so a mode check could never fail and would
    be a guard that cannot fire.
    """
    # Function-local: `recall.embedding_registry` imports this module at module level, so a
    # top-level import here would be a cycle. Safe at call time because the registry only builds
    # a `FastEmbedEmbedder` inside `RegisteredProfile.build`, never while it is being imported.
    from recall.embedding_registry import find_registered_profile

    # `bool(...)` because the expression this replaced was a conditional on TRUTHINESS, and a dict
    # lookup is not. A library caller passing `asymmetric=2` (or a string out of a config file)
    # used to get the asymmetric branch and would now get a KeyError from a public constructor.
    # Narrowing a public signature is not part of this fix.
    asymmetric = bool(asymmetric)
    legacy = _LEGACY_FALLBACK_PROFILE_IDS[asymmetric]
    entry = find_registered_profile(legacy)
    if entry is not None and entry.model_name == model_name and entry.dimension == dimension:
        return legacy
    # Namespaced so the id is self-describing in a chunk's metadata and cannot be mistaken for a
    # registered profile. Injective in exactly the three inputs that reach it, which is the
    # property `_index_fingerprint` needs and the regression test pins.
    #
    # ⚠️ FILENAME-SAFE, which is a constraint and not a style choice. A profile id is not only
    # compared: `recall.eval.promotion.run.ArmConfig.key` interpolates it into a result FILENAME,
    # and its own docstring says so ("a generation id can be long and a filename cannot"). A raw
    # HuggingFace name would put a `/` in that path and a `:` that Windows refuses outright, so
    # the separator is `__` and the org separator goes the same way. `SparseProfile` already
    # resolved this identically (`model_name.replace("/", "__")`); this follows that precedent
    # rather than inventing a second convention. Injectivity survives it for every real model
    # name, none of which contain `__`.
    kind = "asymmetric" if asymmetric else "symmetric"
    return f"unregistered__{model_name.replace('/', '__')}__{dimension}__{kind}"


_log = get_logger("embeddings")


#: Bad `RECALL_FASTEMBED_BATCH` values already reported. Deduplicated by VALUE, not by a single
#: flag: a variable corrected mid-process should warn again for the new bad value.
_WARNED_BATCH_VALUES: set[str] = set()


def _warn_once(raw: str, problem: str) -> None:
    """Report a bad batch setting the FIRST time it is seen, and not on every batch after.

    `_batch_size_from_env` runs once per batch, so warning unconditionally produced hundreds of
    identical lines during a single index and buried anything real. The docstring below claimed
    "warns once" before the code did.
    """
    if raw in _WARNED_BATCH_VALUES:
        return
    _WARNED_BATCH_VALUES.add(raw)
    _log.warning("RECALL_FASTEMBED_BATCH=%r %s; using the backend default", raw, problem)


def _batch_size_from_env(env: Mapping[str, str] | None = None) -> int | None:
    """`RECALL_FASTEMBED_BATCH` as a positive int, or `None` for the backend's own default.

    ⚠️ **An unreadable value degrades rather than raising.** `int()` on a mistyped variable used to
    raise `ValueError` out of the middle of an index run, after several projects had already been
    written. A guard against running out of memory that instead kills the job on a typo is worse
    than no guard. It warns once so the setting is not silently ignored either.
    """
    source = os.environ if env is None else env
    raw = source.get("RECALL_FASTEMBED_BATCH")
    if not raw:
        return None
    try:
        size = int(raw)
    except ValueError:
        _warn_once(raw, "is not an integer")
        return None
    if size < 1:
        _warn_once(raw, "is not positive")
        return None
    return size


class FastEmbedEmbedder:
    """Real local embeddings (no API key). Requires `pip install "recall-rag[fastembed]"`."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        *,
        asymmetric: bool = False,
        profile_id: str | None = None,
        cache_dir: str | Path | None = None,
        artifact_sha256: str | None = None,
        require_local: bool = False,
        context_version: str = "raw-v1",
        identity: EmbeddingProfile | None = None,
        providers: list[str] | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        """Load a local fastembed model, optionally under a supplied immutable identity.

        ``identity`` is how a registered profile is built: `RegisteredProfile.build` passes the
        one `EmbeddingProfile` the registry constructed, and the encoder methods named in it
        (``query_mode`` / ``passage_mode``) are the ones actually called. Without it the class
        keeps its previous behaviour and derives a profile from ``asymmetric``, the legacy
        default path, where no artifact is pinned and nothing enterprise depends on the result.

        ``providers`` is an ONNX Runtime execution-provider REQUEST forwarded to fastembed. It is
        not a guarantee: asking for ``CUDAExecutionProvider`` against a wheel built for a
        different CUDA major falls back to CPU with only a ``RuntimeWarning``. Read
        ``self.session_providers`` for what the session actually resolved — never
        ``onnxruntime.get_available_providers()``, which reports what the wheel was compiled with
        and stays true while the session sits on CPU.
        """
        self._env = dict(os.environ if env is None else env)
        # Artifact first, backend second. A deployment whose weights are missing or tampered
        # with gets that error whether or not the optional extra happens to be installed, and
        # nothing loads before the tree has been checksummed.
        if require_local and cache_dir is None:
            raise ValueError("offline embedding profiles require a provisioned cache_dir")
        if require_local and artifact_sha256 is None:
            raise ValueError("offline embedding profiles require an artifact_sha256")
        local_cache = None
        if cache_dir is not None:
            local_cache = str(
                verify_artifact(cache_dir, artifact_sha256)
                if artifact_sha256 is not None
                else Path(cache_dir).resolve(strict=True)
            )
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "FastEmbedEmbedder requires the fastembed extra: "
                'pip install "recall-rag[fastembed]"'
            ) from exc
        threads = resolve_thread_budget(self._env)
        kwargs: dict[str, object] = {"model_name": identity.model_name if identity else model_name}
        if threads is not None:
            kwargs["threads"] = threads
        if local_cache is not None:
            kwargs["cache_dir"] = local_cache
        if require_local:
            kwargs["local_files_only"] = True
        if providers is not None:
            kwargs["providers"] = list(providers)
        self._model = (
            TextEmbedding(**kwargs)
        )
        #: The providers the ONNX session ACTUALLY resolved, read back from the live session.
        #:
        #: NOT `onnxruntime.get_available_providers()`, which reports what the wheel was compiled
        #: with and stays true even when the session ran on CPU. Measured on an RTX 5090 host:
        #: requesting `CUDAExecutionProvider` against a CUDA-13 wheel on a CUDA-12.8 box falls
        #: back to CPU with only a `RuntimeWarning`, so anything trusting availability would
        #: record a GPU run that never happened. Exposed because an index built on one provider
        #: and served from another is a provenance fact a benchmark has to be able to state.
        self.session_providers = _session_providers(self._model)
        self._name = identity.model_name if identity else model_name
        self._query_mode = identity.query_mode if identity else (
            "query_embed" if asymmetric else "embed"
        )
        self._passage_mode = identity.passage_mode if identity else (
            "passage_embed" if asymmetric else "embed"
        )
        # Both encoders are resolved HERE, at construction. Resolving the query encoder lazily
        # would let a deployment index an entire corpus and only discover a missing encoder on
        # its first query, which is the worst possible moment to find out.
        self._encoder(self._query_mode)
        # Dimension discovery goes through the PASSAGE encoder: the stored vectors are passages,
        # so the width the store is built at must be the width the passage encoder produces.
        probe = next(iter(self._encoder(self._passage_mode)(["probe"])))
        self._dim = len(list(probe))
        if identity is not None and self._dim != identity.dimension:
            raise ValueError(
                f"profile {identity.profile_id!r} declares dimension {identity.dimension} but "
                f"the provisioned artifact embeds at {self._dim}; this artifact is not that "
                f"profile"
            )
        # Applied to BOTH branches. `identity or EmbeddingProfile(...)` short-circuits, so building
        # the provider pair only inside the right-hand side left every REGISTERED profile carrying
        # fastembed's version and nothing else. A fingerprint fix that skips the registered path
        # is not a fix.
        #
        # ⚠️ SCOPE, stated precisely because an earlier version of this comment overstated it:
        # what this reaches is the v1 profile-fingerprint binding (`calibration.load_for_profile`)
        # and the embedding cache key (`recall/cache.py`). It does NOT reach the CERTIFIED v2
        # binding, which stores `recall.lineage.EmbedderIdentity` — provider, model, dimension,
        # revision, artifact_digest — and has no `dependencies` field at all. A CPU-fit certified
        # calibration therefore still binds cleanly to a CUDA-served pipeline. Closing that needs
        # the providers added to `EmbedderIdentity`, which is a separate change.
        self._profile = (
            _with_provider_dependency(identity, self.session_providers)
            if identity
            else EmbeddingProfile(
                profile_id=profile_id or _fallback_profile_id(
                    model_name, self._dim, asymmetric
                ),
                model_name=model_name,
                artifact_digest=artifact_sha256 or LEGACY_UNVERIFIED_DIGEST,
                dimension=self._dim,
                query_mode=self._query_mode,
                passage_mode=self._passage_mode,
                context_version=context_version,
                dependencies=(
                    ("fastembed", _package_version("fastembed")),
                    # The ONNX execution provider is KEY MATERIAL, not metadata. This class's own
                    # fingerprint docstring gives the reason — "ONNX runtime changes are free to
                    # move the last bits of a vector and a cache cannot tell" — and a provider
                    # swap is exactly such a change. Measured on an RTX 5090: CPU and CUDA
                    # sessions over the same weights moved top-45 SET membership on 2 of 64
                    # queries. Without this the two provenances share one cache key
                    # (recall/cache.py) and one calibration binding (recall/calibration.py), so
                    # CPU vectors would be served for a GPU-configured embedder. fastembed also
                    # reaches CUDA on its own via `cuda=Device.AUTO` whenever onnxruntime-gpu is
                    # importable, so this fires without anyone passing `providers=`.
                    *_provider_dependencies(self.session_providers),
                ),
            )
        )

    def _encoder(self, mode: str) -> Callable[[list[str]], Iterable[Iterable[float]]]:
        """Resolve the encoder a profile NAMES, refusing a mode this backend does not have.

        Refusing matters: a missing `query_embed` that silently fell back to `embed` would give
        an asymmetric profile passage vectors for its queries, which is invisible downstream.
        """
        method = getattr(self._model, mode, None)
        if not callable(method):
            raise ValueError(f"fastembed model has no encoder named {mode!r}")
        encoder: Callable[[list[str]], Iterable[Iterable[float]]] = method
        return encoder

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return self._name

    @property
    def profile(self) -> EmbeddingProfile:
        return self._profile

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(x) for x in vec] for vec in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return [float(x) for x in next(iter(self._encoder(self._query_mode)([text])))]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        # ⚠️ **`RECALL_FASTEMBED_BATCH` bounds the batch fastembed builds internally.** Unset, this
        # behaves exactly as before. The default of 256 is too large for a 1024-dim model on long
        # passages: bge-large asked onnxruntime for a single 1.24 GB buffer and the arena refused,
        # which fails an index part-way through rather than merely running slowly.
        #
        # Passed as fastembed's OWN parameter rather than by slicing the list here. An earlier
        # version sliced and called the encoder once per slice, and at size 16 that wedged: 152 of
        # 208 files in, eight cores pinned, no database writes for three minutes, and every server
        # connection idle in `ClientRead`, so the stall was in this process rather than on I/O.
        encoder = self._encoder(self._passage_mode)
        size = _batch_size_from_env(self._env)
        if size is not None:
            # Decided from the signature, before any work, because a TypeError raised WHILE
            # embedding (a tokenizer refusing an input) is a real failure: catching it around the
            # whole iteration re-ran the full list at the default batch, the bound gone.
            if _accepts_batch_size(encoder):
                return [
                    [float(x) for x in vec]
                    for vec in encoder(texts, batch_size=size)  # type: ignore[call-arg]
                ]
            # The variable is a memory guard, not a contract, so embed anyway, but say so: the
            # operator who set it believes it is in force.
            _log.warning(
                "RECALL_FASTEMBED_BATCH=%s is ignored: this fastembed encoder takes no batch_size",
                size,
            )
        return [[float(x) for x in vec] for vec in encoder(texts)]


def _accepts_batch_size(encoder: Callable[..., object]) -> bool:
    """Whether ``encoder(texts, batch_size=n)`` binds, without calling it."""
    try:
        inspect.signature(encoder).bind(["x"], batch_size=1)
    except TypeError:
        return False
    except ValueError:
        # No introspectable signature (a builtin): try the argument, as before this check.
        return True
    return True


SFR_CODE_EMBEDDER_MODEL = "Salesforce/SFR-Embedding-Code-2B_R"


SFR_CODE_EMBEDDER_REVISION = "c73d8631a005876ed5abde34db514b1fb6566973"


REMOTE_MODEL_CODE_OPT_IN = "RECALL_ACCEPT_REMOTE_MODEL_CODE"


def _require_research_model_opt_in(source: Mapping[str, str], model: str) -> None:
    if not truthy(source.get("RECALL_ACCEPT_RESEARCH_MODEL_LICENSE")):
        raise ValueError(
            f"{model} is a research/Gemma-terms model, not a default RE-call shipping model. "
            "Set RECALL_ACCEPT_RESEARCH_MODEL_LICENSE=1 to use the named research alias, or pass "
            "the full st:<model> spelling if you are deliberately managing the licence outside "
            "RE-call."
        )


def _require_remote_model_code_opt_in(source: Mapping[str, str], model: str) -> None:
    if not truthy(source.get(REMOTE_MODEL_CODE_OPT_IN)):
        raise ValueError(
            f"{model} requires Hugging Face remote model code. Set {REMOTE_MODEL_CODE_OPT_IN}=1 "
            "only after reviewing the pinned model revision and accepting that the model repository "
            "can execute Python code during load."
        )


class SentenceTransformerEmbedder:
    """Any `sentence-transformers` model by name or local path — including one fine-tuned here.

    `finetune/train.py` writes a model to disk; without a way to LOAD it back into the retrieval
    stack, a fine-tuning result can only ever be measured by the trainer's own evaluator, on its
    own split. Pointing the real harness at the saved directory is what makes the lift comparable
    to every other embedder measured in this repo:

        python -m recall.eval.labelled --embedder st:finetune/model ...

    Requires `pip install "recall-rag[rerank]"`, which pulls sentence-transformers.
    """

    def __init__(
        self,
        model: str,
        batch_size: int = 64,
        *,
        trust_remote_code: bool = False,
        revision: str | None = None,
        name: str | None = None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                'SentenceTransformerEmbedder requires: pip install "recall-rag[rerank]"'
            ) from exc
        kwargs: dict[str, object] = {"trust_remote_code": trust_remote_code}
        if revision is not None:
            kwargs["revision"] = revision
        self._model = SentenceTransformer(model, **kwargs)
        self._name = name or f"st:{model}"
        self._batch_size = batch_size
        self._dim = int(self._model.get_sentence_embedding_dimension())

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return self._name

    def embed(self, texts: list[str]) -> list[list[float]]:
        # normalize_embeddings: the store scores with cosine distance, and an unnormalised
        # vector would make those scores incomparable with every other embedder here.
        vecs = self._model.encode(
            texts, batch_size=self._batch_size, normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [[float(x) for x in v] for v in vecs]


class PromptedSentenceTransformerEmbedder:
    """A pinned local model encoded with its published query and document prompts.

    Selected with ``RECALL_EMBEDDER=st-prompted:<hf-id>`` for the models in
    `recall.embedding_prompts.PUBLISHED_PROMPTS`. It is a separate identity from ``st:<hf-id>`` on
    purpose, in its NAME as well as its profile: the serving gate, calibration v2, the generation
    build check and the lite store compare the embedder's name and dimension only, so a prompted
    encoder that kept the ``st:`` name would answer from a store built without prompts, silently.
    The name carries the prompt set's version for the same reason.

    The prompt is always passed explicitly, an empty string included, so a model config's
    ``default_prompt_name`` can never add one nobody recorded. ``embed`` encodes passages, as the
    Qwen3 embedder does, because indexing is the caller that reaches it.
    """

    def __init__(
        self, model: str, *, env: Mapping[str, str] | None = None, batch_size: int = 32
    ) -> None:
        from recall.embedding_prompts import prompts_for

        source = os.environ if env is None else env
        prompts = prompts_for(model)
        if Path(model).exists():
            # sentence-transformers loads an existing local path in preference to the hub id, which
            # would skip the pinned revision and, for a remote-code model, run that directory's code.
            raise ValueError(
                f"a local path named {model!r} exists here and would shadow the pinned hub "
                "revision; st-prompted loads only the hub model, so move or rename that path"
            )
        if prompts.remote_code:
            _require_remote_model_code_opt_in(source, model)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                'PromptedSentenceTransformerEmbedder requires: pip install "recall-rag[rerank]"'
            ) from exc
        self._model = SentenceTransformer(
            model, revision=prompts.revision, trust_remote_code=prompts.remote_code,
            truncate_dim=prompts.dimension,
        )
        modality = getattr(self._model[0], "modality_config", None)
        if modality and "message" in modality:
            # sentence-transformers 5.4 and later route the text of a model with a chat template
            # through that template and send the prompt as a system turn, which is not the
            # published prompt the model was trained with.
            raise ValueError(
                f"st-prompted:{model} would route its text through the model's chat template under "
                "this sentence-transformers version, so the published prompt would not be sent as "
                "trained; install sentence-transformers<5.4"
            )
        self._prompts = prompts
        self._batch_size = batch_size
        self._name = f"st-prompted:{model}@{prompts.version}"
        self._dim = int(
            prompts.dimension or self._model.get_sentence_embedding_dimension()
            or len(self._encode(["width probe"], "")[0])
        )
        self._profile = EmbeddingProfile(
            profile_id=self._name,
            model_name=model,
            artifact_digest=f"hf-revision:{prompts.revision}",
            dimension=self._dim,
            query_mode="prompt:query",
            passage_mode="prompt:document" if prompts.document else "raw",
            instruction_version=prompts.version,
            dependencies=(
                ("sentence-transformers", _package_version("sentence-transformers")),
                ("transformers", _package_version("transformers")),
            ),
        )

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return self._name

    @property
    def profile(self) -> EmbeddingProfile:
        return self._profile

    def _encode(self, texts: list[str], prompt: str) -> list[list[float]]:
        vectors = self._model.encode(
            texts, prompt=prompt, batch_size=self._batch_size, normalize_embeddings=False,
            show_progress_bar=False,
        )
        out: list[list[float]] = []
        for vector in vectors:
            values = [float(x) for x in vector]
            # Normalised here rather than by the library, so a Matryoshka-truncated vector is unit
            # length too: the store scores with cosine, and truncation happens after the model.
            norm = math.sqrt(sum(value * value for value in values))
            if norm == 0.0:
                raise RuntimeError(f"{self._name} produced a zero vector")
            out.append([value / norm for value in values])
        return out

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.embed_passages(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text], self._prompts.query)[0]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts, self._prompts.document)


QWEN3_RETRIEVAL_INSTRUCTION_V1 = (
    "Given a user question, retrieve passages that answer the question"
)


class Qwen3EmbeddingEmbedder:
    """Offline, instruction-aware Qwen3 0.6B experiment truncated to 384 dimensions."""

    def __init__(
        self,
        model_path: str | Path,
        artifact_sha256: str,
        *,
        dimension: int = 384,
        context_version: str = "raw-v1",
        batch_size: int = 32,
        identity: EmbeddingProfile | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        """Load the offline Qwen3 artifact under the identity the registry built for it.

        See `recall.embedding_registry` for the recorded rejection: this profile was measured on
        CPU and refused on latency. The class is retained so the negative result stays
        reproducible, not because the profile is a candidate.
        """
        self._env = dict(os.environ if env is None else env)
        if identity is not None:
            dimension = identity.dimension
        if dimension != 384:
            raise ValueError("the registered Qwen3 experiment is fixed at 384 dimensions")
        local = verify_artifact(model_path, artifact_sha256)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                'Qwen3EmbeddingEmbedder requires: pip install "recall-rag[rerank]"'
            ) from exc
        threads = resolve_thread_budget(self._env)
        if threads is not None:
            import torch

            torch.set_num_threads(threads)
        self._model = SentenceTransformer(
            str(local), local_files_only=True, truncate_dim=dimension
        )
        self._dim = dimension
        self._batch_size = batch_size
        self._profile = identity or EmbeddingProfile(
            profile_id="qwen3-embedding-0.6b-384-v1",
            model_name="Qwen/Qwen3-Embedding-0.6B",
            artifact_digest=artifact_sha256,
            dimension=dimension,
            query_mode="instruction-v1",
            passage_mode="document",
            instruction_version="retrieval-v1",
            context_version=context_version,
            dependencies=(("sentence-transformers", _package_version("sentence-transformers")),),
        )

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return self._profile.model_name

    @property
    def profile(self) -> EmbeddingProfile:
        return self._profile

    def _encode(self, texts: list[str], prompt: str | None = None) -> list[list[float]]:
        kwargs: dict[str, object] = {
            "batch_size": self._batch_size,
            "normalize_embeddings": True,
            "show_progress_bar": False,
        }
        if prompt is not None:
            kwargs["prompt"] = prompt
        vectors = self._model.encode(texts, **kwargs)
        normalized: list[list[float]] = []
        for vector in vectors:
            values = [float(x) for x in vector]
            # Sentence Transformers normalizes before truncate_dim is applied for this
            # model, so the returned 384-wide prefix is no longer unit length.  Normalize
            # the final representation explicitly, which is the vector stored and scored.
            norm = math.sqrt(sum(value * value for value in values))
            if norm == 0.0:
                raise RuntimeError("Qwen3 embedding produced a zero vector")
            normalized.append([value / norm for value in values])
        return normalized

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.embed_passages(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text], prompt=QWEN3_RETRIEVAL_INSTRUCTION_V1)[0]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)
