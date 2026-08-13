from __future__ import annotations

import hashlib
import math
import os
import random
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Literal, Protocol, TypeVar, runtime_checkable

#: Return type of the callable `retry_with_backoff` wraps — it hands back whatever `fn` returns,
#: so the retry is transparent to the caller's type rather than widening it to `object`.
_R = TypeVar("_R")


#: Markers matched in the exception TEXT when no numeric status is available, which for the
#: callers that reach it is the entire retry decision rather than a fallback to one. Module
#: level, not a local, so `tests/test_embeddings_retry_classifier.py` can assert one case per
#: marker: a marker with no case is a marker that can be deleted while the suite stays green.
_TRANSIENT_MARKERS = (
    "429", " 500", " 502", " 503", " 504", "rate limit", "too many requests",
    "timeout", "timed out", "temporarily", "connection", "reset by peer", "unavailable",
)


def _is_transient(exc: Exception) -> bool:
    """Heuristic: is this exception worth retrying?

    Covers rate-limit (429), server (5xx) and network/timeout errors WITHOUT importing any
    provider-specific exception type (voyageai is an optional dependency). A non-transient error
    (e.g. 401 auth) returns False so it fails fast.

    A numeric ``status_code``/``status``/``http_status`` is DECISIVE: when the transport has
    stated the status, that answer is returned and the text markers below are never consulted.
    They used to be, and they could overturn a correct verdict — the marker ``"429"`` is a
    substring of any number containing it, so ``"…your messages resulted in 10429 tokens"`` made
    a permanent HTTP 400 context-length overflow look like a rate limit. That is the worst case
    to be wrong on: ``retry_with_backoff`` resends the entire payload, so a caller whose payload
    is a prompt with a whole document body inside it pays for the same refused request on every
    attempt (three by default, four from ``benchmarks/llm.py``), and no retry can make an
    over-long prompt fit. ``benchmarks/llm.py`` is that shape here; the case this was actually
    found on is an extraction engine that lives on an unlanded branch, so do not go looking for
    it in this tree.

    THREE spellings are read, and the third is not exotic. Every ``voyageai`` error raised by
    ``api_requestor`` FROM A NON-2xx RESPONSE carries the code in ``http_status``:
    ``VoyageError.__init__`` takes it as its third positional argument and the requestor passes
    the real code into it. The qualifier is the whole point, and the paragraph below depends on
    it: errors raised anywhere else in the SDK carry a message and nothing more. An earlier reading of this
    concluded the Voyage path had no status and left the markers to decide it, but that reading
    came from a hand-constructed ``RateLimitError`` whose ``http_status`` was never filled in,
    not from one the SDK raised. The markers cannot decide it: those messages are fixed strings
    with no marker in them, so a real 500 ("The server failed to process the request.") was
    never retried, and a 502/503/504 was retried only by the accident that "unavailable" appears
    inside the class name ``ServiceUnavailableError``.

    The markers remain as a fallback for errors that carry no status at all, which is the only
    evidence available there: ``openai.APIConnectionError``/``APITimeoutError`` carry none, and
    neither do voyageai's CLIENT-side ``Timeout``/``APIConnectionError``, which are raised with a
    message alone and so leave ``http_status`` at its constructor default. A slot that exists but
    was never filled in has said nothing, which is why the read tests the VALUE and not the
    attribute's presence.

    So "network/timeout" above means a CLIENT-side timeout, which arrives with no status and
    keeps the fallback. A server that RETURNS 408 (or 409, which openai's own client retries as a
    lock timeout) is not retried here, because 429 and 5xx is the numeric contract this docstring
    has always claimed. That exclusion is deliberate; widening it is a change to what "transient"
    means, and it belongs in the numeric branch rather than in a text marker that would re-open
    the hole above.

    It is also THIS FUNCTION'S exclusion and not yet the system's. ``OpenAICompatEmbedder`` below
    and ``benchmarks/llm.py`` both build their ``OpenAI`` client without ``max_retries=0``, and
    the SDK retries 408, 409, 429 and 5xx twice on its own before this classifier is consulted at
    all — so end to end a 408 is currently retried anyway, and for THOSE statuses every attempt
    counted here is three requests. A 400 or 402 is not in the SDK's retry set, so those attempts
    stay one request each, and the 10429 overflow this docstring opens on is one of those.
    Fixing the missing ``max_retries=0`` is tracked separately; until it lands, do not read this
    function's numeric contract as describing what reaches the provider.
    """
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(exc, "status", None)
    if status is None:
        status = getattr(exc, "http_status", None)
    if isinstance(status, int):
        return status == 429 or 500 <= status < 600
    text = f"{type(exc).__name__} {exc}".lower()
    return any(m in text for m in _TRANSIENT_MARKERS)


def retry_with_backoff(
    fn: Callable[[], _R],
    *,
    attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    is_transient: Callable[[Exception], bool] = _is_transient,
    sleep: Callable[[float], None] = time.sleep,
) -> _R:
    """Call ``fn()`` with exponential backoff, retrying only transient failures.

    Re-raises immediately for a non-transient error, and re-raises the last error after
    ``attempts`` tries. ``sleep`` is injectable so tests can exercise the retry path without
    real delays.

    Delay for retry i is FULL JITTER over ``min(max_delay, base_delay * 2**i)`` — a uniform
    draw in [0, cap], not the cap itself. A rate-limit or 5xx typically hits every client at
    once, so a deterministic schedule marches the whole fleet back onto the provider in
    lockstep at each step; jitter spreads the retries out instead of reconverging them.
    """
    last: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:
            last = exc
            if i == attempts - 1 or not is_transient(exc):
                raise
            sleep(random.uniform(0.0, min(max_delay, base_delay * (2 ** i))))
    assert last is not None  # unreachable: loop either returns or raises
    raise last


def _batches(seq: list[str], size: int) -> Iterator[list[str]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def batched_embed(
    texts: list[str],
    embed_batch: Callable[[list[str]], list[list[float]]],
    *,
    batch_size: int = 128,
    max_batch_chars: int | None = None,
) -> list[list[float]]:
    """Embed ``texts`` in provider-safe batches, concatenating results in input order.

    ``embed_batch`` embeds a single batch. Batches are cut on ``batch_size`` (count) and, when
    ``max_batch_chars`` is set, also on a cumulative character budget — a guard against a batch
    that is few in count but huge in tokens. A single text over the char budget still goes out
    alone (never dropped). Order is preserved: batch results are appended in sequence.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive int")
    out: list[list[float]] = []
    batch: list[str] = []
    chars = 0
    for t in texts:
        if batch and (
            len(batch) >= batch_size
            or (max_batch_chars is not None and chars + len(t) > max_batch_chars)
        ):
            out.extend(_checked(embed_batch, batch))
            batch, chars = [], 0
        batch.append(t)
        chars += len(t)
    if batch:
        out.extend(_checked(embed_batch, batch))
    return out


def _checked(
    embed_batch: Callable[[list[str]], list[list[float]]], batch: list[str]
) -> list[list[float]]:
    """Embed one batch, refusing a response that does not line up with its input.

    Positional pairing is the whole contract (chunk i <-> vector i). A short batch would shift
    every later chunk onto its neighbour's vector — silently, because the only downstream check
    is the TOTAL count, which a compensating batch satisfies.
    """
    vecs = embed_batch(batch)
    if len(vecs) != len(batch):
        raise RuntimeError(
            f"embedder returned {len(vecs)} embeddings for {len(batch)} texts — refusing to "
            f"index misaligned vectors"
        )
    return vecs


@runtime_checkable
class Embedder(Protocol):
    """Turns text into dense vectors. Implementations must be deterministic and
    order-preserving: `embed(texts)` returns one vector per input text, in input order,
    each of length `dim`. `name` identifies the backend (used in logging / evals).
    """

    @property
    def dim(self) -> int: ...

    @property
    def name(self) -> str: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


EmbeddingPurpose = Literal["query", "passage", "legacy"]


@runtime_checkable
class AsymmetricEmbedder(Embedder, Protocol):
    """Optional extension for models with distinct retrieval encoders."""

    def embed_query(self, text: str) -> list[float]: ...

    def embed_passages(self, texts: list[str]) -> list[list[float]]: ...


@dataclass(frozen=True)
class EmbeddingProfile:
    """Immutable identity for every input that can change stored vectors."""

    profile_id: str
    model_name: str
    artifact_digest: str
    dimension: int
    query_mode: str
    passage_mode: str
    normalization: str = "l2"
    instruction_version: str = "none"
    chunker_version: str = "chunk-text-v1"
    context_version: str = "raw-v1"
    dependencies: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.profile_id or not self.model_name or not self.artifact_digest:
            raise ValueError("embedding profile identity fields must be non-empty")
        if self.dimension < 1:
            raise ValueError("embedding profile dimension must be positive")

    def fingerprint(self) -> str:
        """SHA256 over the COMPLETE identity, as durable cache and provenance key material.

        The encoding, which the pinned test transcribes independently rather than reading back
        off this method: a domain tag, then every field in declaration order, then each
        dependency as name followed by version, each item UTF-8 encoded and terminated by a NUL.
        The terminators are what make the concatenation unambiguous; without them
        ``("ab", "c")`` and ``("a", "bc")`` hash alike.

        Every field is included, including the four that nothing else reads
        (``normalization``, ``instruction_version``, ``chunker_version``, ``dependencies``).
        That is the answer to what those fields are for: they are not documentation, they are
        key material, and a change in any of them re-partitions the cache rather than silently
        serving vectors produced under the old value. ``dependencies`` carries the inference
        library version, so a fastembed upgrade costs a re-embed, deliberately, because ONNX
        runtime changes are free to move the last bits of a vector and a cache cannot tell.

        Stability is the contract. Cached vectors outlive the process that wrote them, so
        changing this encoding invalidates every cache in existence at once; if that is ever
        wanted, bump the domain tag so the change is legible instead of mysterious.
        """
        digest = hashlib.sha256()
        parts = [
            "embedding-profile-fingerprint-v1",
            self.profile_id,
            self.model_name,
            self.artifact_digest,
            str(self.dimension),
            self.query_mode,
            self.passage_mode,
            self.normalization,
            self.instruction_version,
            self.chunker_version,
            self.context_version,
        ]
        for name, pinned_version in self.dependencies:
            parts.extend((name, pinned_version))
        for part in parts:
            digest.update(part.encode("utf-8"))
            digest.update(b"\x00")
        return digest.hexdigest()


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not-installed"


def legacy_embedding_profile(embedder: Embedder) -> EmbeddingProfile:
    """Describe a legacy embedder without changing its public protocol."""
    name = getattr(embedder, "name", type(embedder).__name__)
    dim = int(getattr(embedder, "dim"))
    return EmbeddingProfile(
        profile_id=str(name),
        model_name=str(name),
        artifact_digest="legacy-unverified",
        dimension=dim,
        query_mode="legacy",
        passage_mode="legacy",
        normalization="embedder-defined",
    )


def embedding_profile(embedder: Embedder) -> EmbeddingProfile:
    profile = getattr(embedder, "profile", None)
    return profile if isinstance(profile, EmbeddingProfile) else legacy_embedding_profile(embedder)


def embedding_profile_id(embedder: Embedder) -> str:
    profile = getattr(embedder, "profile", None)
    if isinstance(profile, EmbeddingProfile):
        return profile.profile_id
    name = getattr(embedder, "name", None)
    return name if isinstance(name, str) else type(embedder).__name__


def embed_query(embedder: Embedder, text: str) -> list[float]:
    """Encode one query, falling back to the legacy symmetric interface."""
    method = getattr(embedder, "embed_query", None)
    if callable(method):
        return [float(x) for x in method(text)]
    return [float(x) for x in embedder.embed([text])[0]]


def embed_passages(embedder: Embedder, texts: list[str]) -> list[list[float]]:
    """Encode passages, falling back to the legacy symmetric interface."""
    method = getattr(embedder, "embed_passages", None)
    raw = method(texts) if callable(method) else embedder.embed(texts)
    return [[float(x) for x in vector] for vector in raw]


def artifact_tree_sha256(path: str | Path) -> str:
    """Hash a provisioned file or directory without following directory symlinks."""
    root = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
    if not files:
        raise ValueError(f"model artifact has no files: {root}")
    for file in files:
        resolved = file.resolve(strict=True)
        if root.is_dir() and not resolved.is_relative_to(root):
            raise ValueError(f"model artifact symlink escapes its root: {file}")
        relative = resolved.name if root.is_file() else resolved.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\x00")
        with resolved.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def verify_artifact(path: str | Path, expected_sha256: str) -> Path:
    """Resolve and checksum a local model artifact before a runtime loads it."""
    if len(expected_sha256) != 64 or any(c not in "0123456789abcdefABCDEF" for c in expected_sha256):
        raise ValueError("model artifact SHA256 must be 64 hexadecimal characters")
    resolved = Path(path).resolve(strict=True)
    actual = artifact_tree_sha256(resolved)
    if actual.lower() != expected_sha256.lower():
        raise RuntimeError(
            f"model artifact checksum mismatch: expected {expected_sha256.lower()}, got {actual}"
        )
    return resolved


class HashingEmbedder:
    """Deterministic, dependency-free embedder for tests and offline demos.

    Hashes whitespace tokens into a fixed-width bag-of-words vector, then
    L2-normalizes. Not semantic, but stable and fast — good enough to exercise
    plumbing and to keep the test suite offline.
    """

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return f"hashing-{self._dim}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        for tok in text.lower().split():
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            vec[h % self._dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


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
    except Exception:  # pragma: no cover - defensive; fastembed internals are not a contract
        return [PROVIDERS_UNKNOWN]
    # The COMMON path — fastembed simply exposing no session — must return the constant too. It
    # was left as a bare literal, equal by value today, so `_provider_dependencies` still matched
    # it. Change the constant's text and this path would have started recording the sentinel as a
    # provider NAME: exactly the collapse the constant exists to prevent.
    return [PROVIDERS_UNKNOWN]


class FastEmbedEmbedder:
    """Real local embeddings (no API key). Requires `pip install recall-rag[fastembed]`."""

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
                "FastEmbedEmbedder requires the fastembed extra: pip install recall-rag[fastembed]"
            ) from exc
        threads = resolve_thread_budget()
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
                profile_id=profile_id or (
                    "bge-small-asymmetric-v1" if asymmetric else "bge-small-symmetric-v1"
                ),
                model_name=model_name,
                artifact_digest=artifact_sha256 or "legacy-unverified",
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
        return [[float(x) for x in vec] for vec in self._encoder(self._passage_mode)(texts)]


class SentenceTransformerEmbedder:
    """Any `sentence-transformers` model by name or local path — including one fine-tuned here.

    `finetune/train.py` writes a model to disk; without a way to LOAD it back into the retrieval
    stack, a fine-tuning result can only ever be measured by the trainer's own evaluator, on its
    own split. Pointing the real harness at the saved directory is what makes the lift comparable
    to every other embedder measured in this repo:

        python -m recall.eval.labelled --embedder st:finetune/model ...

    Requires `pip install recall-rag[rerank]` (or `[entail]`) — both pull sentence-transformers.
    """

    def __init__(self, model: str, batch_size: int = 64) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "SentenceTransformerEmbedder requires: pip install recall-rag[rerank]"
            ) from exc
        self._model = SentenceTransformer(model)
        self._name = f"st:{model}"
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
    ) -> None:
        """Load the offline Qwen3 artifact under the identity the registry built for it.

        See `recall.embedding_registry` for the recorded rejection: this profile was measured on
        CPU and refused on latency. The class is retained so the negative result stays
        reproducible, not because the profile is a candidate.
        """
        if identity is not None:
            dimension = identity.dimension
        if dimension != 384:
            raise ValueError("the registered Qwen3 experiment is fixed at 384 dimensions")
        local = verify_artifact(model_path, artifact_sha256)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Qwen3EmbeddingEmbedder requires: pip install recall-rag[rerank]"
            ) from exc
        threads = resolve_thread_budget()
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


class VoyageEmbedder:
    """Voyage cloud embeddings. Requires `pip install recall-rag[voyage]` and VOYAGE_API_KEY."""

    def __init__(
        self,
        model: str = "voyage-3",
        api_key: str | None = None,
        batch_size: int = 128,
        max_retries: int = 3,
    ) -> None:
        key = api_key or os.environ.get("VOYAGE_API_KEY")
        if not key:
            raise RuntimeError("VoyageEmbedder needs VOYAGE_API_KEY (env) or an explicit api_key")
        try:
            import voyageai
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError("VoyageEmbedder requires: pip install recall-rag[voyage]") from exc
        self._client = voyageai.Client(api_key=key)
        self._model = model
        self._name = f"voyage:{model}"
        self._batch_size = batch_size
        self._max_retries = max_retries
        self._dim = len(self._client.embed(["probe"], model=model).embeddings[0])

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return self._name

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed in provider-safe batches with exponential-backoff retry per batch.

        A single request sending every chunk at once will exceed the API's per-request limit on
        a real corpus and has no tolerance for a transient 429/5xx; batching + retry make bulk
        indexing survivable. Results are concatenated in input order (see ``batched_embed``).
        """
        def _embed_batch(batch: list[str]) -> list[list[float]]:
            result = retry_with_backoff(
                lambda: self._client.embed(batch, model=self._model),
                attempts=self._max_retries,
            )
            return [[float(x) for x in v] for v in result.embeddings]

        return batched_embed(texts, _embed_batch, batch_size=self._batch_size)


class OpenAICompatEmbedder:
    """OpenAI-compatible cloud embeddings via any ``base_url`` (OpenRouter, OpenAI, Azure, vLLM).

    Defaults to OpenRouter so the exact ``openai/text-embedding-3-small`` model is reachable on the
    same key the benchmark already uses for its generator and judge — no separate OpenAI billing,
    which is the whole reason this backend exists. The request/response shape is OpenAI's
    ``/v1/embeddings``, so the stock ``openai`` SDK works unchanged against OpenRouter's endpoint.

    ``dimensions`` is deliberately never sent: ``text-embedding-3-small`` returns its native
    1536-wide vector, and some OpenAI-compatible proxies reject the parameter. Both arms of the
    benchmark therefore compare at the model's native width.
    """

    def __init__(
        self,
        model: str = "openai/text-embedding-3-small",
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        batch_size: int = 128,
        max_retries: int = 3,
    ) -> None:
        key = api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "OpenAICompatEmbedder needs an API key (OPENROUTER_API_KEY or OPENAI_API_KEY in "
                "the environment, or an explicit api_key)"
            )
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - exercised only without the SDK
            raise ImportError("OpenAICompatEmbedder requires: pip install openai") from exc
        self._client = OpenAI(api_key=key, base_url=base_url)
        self._model = model
        self._name = f"openai:{model}"
        self._batch_size = batch_size
        self._max_retries = max_retries
        # Probe the width once, the same way the other cloud embedder does, so a store can be built
        # at the matching ``dim`` before the first real batch is embedded.
        self._dim = len(self._embed_one_batch(["probe"])[0])

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return self._name

    def _embed_one_batch(self, batch: list[str]) -> list[list[float]]:
        result = retry_with_backoff(
            lambda: self._client.embeddings.create(
                model=self._model, input=batch, encoding_format="float"
            ),
            attempts=self._max_retries,
        )
        return [[float(x) for x in item.embedding] for item in result.data]

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed in provider-safe batches with exponential-backoff retry per batch — the same
        contract as ``VoyageEmbedder.embed``, so this is a drop-in cloud embedder on the RE-call
        arm."""
        return batched_embed(texts, self._embed_one_batch, batch_size=self._batch_size)


def resolve_embedder(name: str, env: dict[str, str] | None = None) -> Embedder:
    """Build an embedder from a short config string.

    Supported spellings:
    ``hashing``, ``fastembed``, ``fastembed:<model>``, ``st:<model>``,
    ``voyage``, ``voyage:<model>``, ``openai`` and ``openai:<model>``.
    """
    if name == "hashing" or name.startswith("hashing-") or name.startswith("hashing:"):
        return HashingEmbedder(dim=64)
    if name == "fastembed":
        return FastEmbedEmbedder()
    if name.startswith("fastembed:"):
        return FastEmbedEmbedder(model_name=name[len("fastembed:"):])
    if name.startswith("st:"):
        return SentenceTransformerEmbedder(name[3:])
    source = os.environ if env is None else env
    if name == "voyage":
        return VoyageEmbedder(api_key=source.get("VOYAGE_API_KEY"))
    if name.startswith("voyage:"):
        return VoyageEmbedder(
            model=name[len("voyage:"):], api_key=source.get("VOYAGE_API_KEY")
        )
    if name == "openai":
        return OpenAICompatEmbedder(
            api_key=source.get("OPENROUTER_API_KEY") or source.get("OPENAI_API_KEY")
        )
    if name.startswith("openai:"):
        return OpenAICompatEmbedder(
            model=name[len("openai:"):],
            api_key=source.get("OPENROUTER_API_KEY") or source.get("OPENAI_API_KEY"),
        )
    raise ValueError(
        f"unknown embedder: {name!r} (use hashing, fastembed, fastembed:<model>, "
        "st:<model>, voyage, voyage:<model>, openai, or openai:<model>)"
    )
