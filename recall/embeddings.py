from __future__ import annotations

import hashlib
import math
import os
import random
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Literal, Protocol, TypeVar, runtime_checkable

#: Return type of the callable `retry_with_backoff` wraps — it hands back whatever `fn` returns,
#: so the retry is transparent to the caller's type rather than widening it to `object`.
_R = TypeVar("_R")


def _is_transient(exc: Exception) -> bool:
    """Heuristic: is this exception worth retrying?

    Covers rate-limit (429), server (5xx) and network/timeout errors WITHOUT importing any
    provider-specific exception type (voyageai is an optional dependency). Checks a numeric
    ``status_code``/``status`` attribute first, then falls back to matching well-known markers
    in the exception text. A non-transient error (e.g. 401 auth) returns False so it fails fast.
    """
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(exc, "status", None)
    if isinstance(status, int) and (status == 429 or 500 <= status < 600):
        return True
    text = f"{type(exc).__name__} {exc}".lower()
    markers = (
        "429", " 500", " 502", " 503", " 504", "rate limit", "too many requests",
        "timeout", "timed out", "temporarily", "connection", "reset by peer", "unavailable",
    )
    return any(m in text for m in markers)


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


class FastEmbedEmbedder:
    """Real local embeddings (no API key). Requires `pip install recall[fastembed]`."""

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
    ) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "FastEmbedEmbedder requires the fastembed extra: pip install recall[fastembed]"
            ) from exc
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
        threads = resolve_thread_budget()
        kwargs: dict[str, object] = {"model_name": model_name}
        if threads is not None:
            kwargs["threads"] = threads
        if local_cache is not None:
            kwargs["cache_dir"] = local_cache
        if require_local:
            kwargs["local_files_only"] = True
        self._model = (
            TextEmbedding(**kwargs)
        )
        self._name = model_name
        self._asymmetric = asymmetric
        passage_method = self._model.passage_embed if asymmetric else self._model.embed
        self._dim = len(next(iter(passage_method(["probe"]))))
        digest = artifact_sha256 or "legacy-unverified"
        self._profile = EmbeddingProfile(
            profile_id=profile_id or (
                "bge-small-asymmetric-v1" if asymmetric else "bge-small-symmetric-v1"
            ),
            model_name=model_name,
            artifact_digest=digest,
            dimension=self._dim,
            query_mode="query_embed" if asymmetric else "embed",
            passage_mode="passage_embed" if asymmetric else "embed",
            context_version=context_version,
            dependencies=(("fastembed", _package_version("fastembed")),),
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

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(x) for x in vec] for vec in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        method = self._model.query_embed if self._asymmetric else self._model.embed
        return [float(x) for x in next(iter(method([text])))]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        method = self._model.passage_embed if self._asymmetric else self._model.embed
        return [[float(x) for x in vec] for vec in method(texts)]


class SentenceTransformerEmbedder:
    """Any `sentence-transformers` model by name or local path — including one fine-tuned here.

    `finetune/train.py` writes a model to disk; without a way to LOAD it back into the retrieval
    stack, a fine-tuning result can only ever be measured by the trainer's own evaluator, on its
    own split. Pointing the real harness at the saved directory is what makes the lift comparable
    to every other embedder measured in this repo:

        python -m recall.eval.labelled --embedder st:finetune/model ...

    Requires `pip install recall[rerank]` (or `[entail]`) — both pull sentence-transformers.
    """

    def __init__(self, model: str, batch_size: int = 64) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "SentenceTransformerEmbedder requires: pip install recall[rerank]"
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
    ) -> None:
        if dimension != 384:
            raise ValueError("the registered Qwen3 experiment is fixed at 384 dimensions")
        local = verify_artifact(model_path, artifact_sha256)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Qwen3EmbeddingEmbedder requires: pip install recall[rerank]"
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
        self._profile = EmbeddingProfile(
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
        return [[float(x) for x in vector] for vector in vectors]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.embed_passages(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text], prompt=QWEN3_RETRIEVAL_INSTRUCTION_V1)[0]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)


class VoyageEmbedder:
    """Voyage cloud embeddings. Requires `pip install recall[voyage]` and VOYAGE_API_KEY."""

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
            raise ImportError("VoyageEmbedder requires: pip install recall[voyage]") from exc
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
