"""Cross process serialization for hosted embedding provider calls."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any, cast

from recall.cache import EmbeddingCache, embed_with_cache
from recall.embeddings import Embedder, embedding_profile
from recall_aml.models import ContentValue
from recall_aml.multimodal import MultimodalEmbedder, to_voyage_input


LockFactory = Callable[[Path], AbstractContextManager[None]]


def _structured_cache_key(
    *, profile: str, dim: int, purpose: str, value: object
) -> str:
    """Content address a structured embedding without persisting its source payload."""
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    digest = hashlib.sha256()
    for part in ("recall-aml-structured-embedding-v1", profile, str(dim), purpose, encoded):
        digest.update(part.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


class CachedEmbedder:
    """Reuse text embeddings across hosted tenants, variants, corpora, and processes.

    A fresh SQLite connection is opened per method call. Hosted Add operations execute in worker
    threads, and sharing one sqlite3 connection across them would make the cache degrade on its
    first cross-thread access. In production this wrapper sits inside :class:`LockedEmbedder`, so
    the VPS2 provider lock covers the cache lookup and the possible provider miss atomically.
    """

    def __init__(self, inner: Embedder, path: Path) -> None:
        self._inner = inner
        self._path = path

    @property
    def dim(self) -> int:
        return self._inner.dim

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def profile(self) -> Any:
        return getattr(self._inner, "profile", None)

    def embed(self, texts: list[str]) -> list[list[float]]:
        with EmbeddingCache(self._path) as cache:
            return embed_with_cache(self._inner, texts, cache, purpose="legacy")

    def embed_query(self, text: str) -> list[float]:
        with EmbeddingCache(self._path) as cache:
            return embed_with_cache(self._inner, [text], cache, purpose="query")[0]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        if callable(getattr(self._inner, "embed_document_groups", None)):
            return self.embed_document_groups([texts])[0]
        with EmbeddingCache(self._path) as cache:
            return embed_with_cache(self._inner, texts, cache, purpose="passage")

    def embed_document_groups(self, groups: list[list[str]]) -> list[list[list[float]]]:
        method = getattr(self._inner, "embed_document_groups", None)
        if not callable(method):
            return [self.embed_passages(group) for group in groups]

        identity = embedding_profile(self._inner).fingerprint()
        group_keys = [
            [
                _structured_cache_key(
                    profile=identity,
                    dim=self.dim,
                    purpose="context-document-group",
                    value={"group": group, "ordinal": ordinal},
                )
                for ordinal, _text in enumerate(group)
            ]
            for group in groups
        ]
        with EmbeddingCache(self._path) as cache:
            cached = cache.get_many([key for keys in group_keys for key in keys])
            output: list[list[list[float]] | None] = [None] * len(groups)
            missing_indices: list[int] = []
            for index, keys in enumerate(group_keys):
                if all(key in cached for key in keys):
                    output[index] = [cached[key] for key in keys]
                else:
                    missing_indices.append(index)

            if missing_indices:
                fresh_groups = cast(
                    list[list[list[float]]],
                    method([groups[index] for index in missing_indices]),
                )
                if len(fresh_groups) != len(missing_indices):
                    raise RuntimeError("context embedder returned the wrong number of groups")
                cache_rows: list[tuple[str, list[float]]] = []
                for index, vectors in zip(missing_indices, fresh_groups, strict=True):
                    if len(vectors) != len(groups[index]):
                        raise RuntimeError(
                            "context embedder returned the wrong number of document vectors"
                        )
                    output[index] = vectors
                    cache_rows.extend(zip(group_keys[index], vectors, strict=True))
                cache.put_many(cache_rows)

        if any(group is None for group in output):
            raise RuntimeError("context embedding cache left an unfilled document group")
        return cast(list[list[list[float]]], output)


class CachedMultimodalEmbedder:
    """Reuse independent Voyage multimodal document and query vectors by exact input."""

    def __init__(self, inner: MultimodalEmbedder, path: Path) -> None:
        self._inner = inner
        self._path = path

    @property
    def dim(self) -> int:
        return self._inner.dim

    @property
    def profile(self) -> str:
        return self._inner.profile

    def embed_documents(self, inputs: Sequence[dict[str, Any]]) -> list[list[float]]:
        values = list(inputs)
        keys = [
            _structured_cache_key(
                profile=self.profile,
                dim=self.dim,
                purpose="multimodal-document",
                value=value,
            )
            for value in values
        ]
        with EmbeddingCache(self._path) as cache:
            cached = cache.get_many(keys)
            missing: dict[str, dict[str, Any]] = {}
            for key, value in zip(keys, values, strict=True):
                if key not in cached and key not in missing:
                    missing[key] = value
            if missing:
                missing_keys = list(missing)
                fresh = self._inner.embed_documents([missing[key] for key in missing_keys])
                if len(fresh) != len(missing_keys):
                    raise RuntimeError(
                        "multimodal embedder returned the wrong number of document vectors"
                    )
                cache.put_many(list(zip(missing_keys, fresh, strict=True)))
                cached.update(zip(missing_keys, fresh, strict=True))
            return [cached[key] for key in keys]

    def embed_query(self, value: ContentValue) -> list[float]:
        key = _structured_cache_key(
            profile=self.profile,
            dim=self.dim,
            purpose="multimodal-query",
            value=to_voyage_input(value),
        )
        with EmbeddingCache(self._path) as cache:
            cached = cache.get(key)
            if cached is not None:
                return cached
            vector = self._inner.embed_query(value)
            cache.put(key, vector)
            return vector


@contextmanager
def embedding_call_lock(path: Path | None) -> Iterator[None]:
    """Take the one VPS2 advisory lock shared by every embedding experiment."""
    if path is None:
        with nullcontext():
            yield
        return
    try:
        fcntl = importlib.import_module("fcntl")
    except ImportError as exc:  # pragma: no cover, the configured runtime is Linux
        raise RuntimeError("the configured embedding lock requires POSIX flock") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class LockedEmbedder:
    """Preserve the full embedder surface while locking every provider invocation."""

    def __init__(
        self,
        inner: Embedder,
        path: Path,
        *,
        lock_factory: LockFactory = embedding_call_lock,
    ) -> None:
        self._inner = inner
        self._path = path
        self._lock_factory = lock_factory

    @property
    def dim(self) -> int:
        return self._inner.dim

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def profile(self) -> Any:
        return getattr(self._inner, "profile", None)

    def embed(self, texts: list[str]) -> list[list[float]]:
        with self._lock_factory(self._path):
            return self._inner.embed(texts)

    def embed_query(self, text: str) -> list[float]:
        method = getattr(self._inner, "embed_query", None)
        with self._lock_factory(self._path):
            if callable(method):
                return cast(list[float], method(text))
            return self._inner.embed([text])[0]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        method = getattr(self._inner, "embed_passages", None)
        with self._lock_factory(self._path):
            if callable(method):
                return cast(list[list[float]], method(texts))
            return self._inner.embed(texts)

    def embed_document_groups(self, groups: list[list[str]]) -> list[list[list[float]]]:
        grouped_method = getattr(self._inner, "embed_document_groups", None)
        passage_method = getattr(self._inner, "embed_passages", None)
        with self._lock_factory(self._path):
            if callable(grouped_method):
                return cast(list[list[list[float]]], grouped_method(groups))
            if callable(passage_method):
                return [
                    cast(list[list[float]], passage_method(group)) for group in groups
                ]
            return [self._inner.embed(group) for group in groups]


class LockedMultimodalEmbedder:
    """Serialize Voyage Multimodal document and query calls with text embeddings."""

    def __init__(
        self,
        inner: MultimodalEmbedder,
        path: Path,
        *,
        lock_factory: LockFactory = embedding_call_lock,
    ) -> None:
        self._inner = inner
        self._path = path
        self._lock_factory = lock_factory

    @property
    def dim(self) -> int:
        return self._inner.dim

    @property
    def profile(self) -> str:
        return self._inner.profile

    def embed_documents(self, inputs: Sequence[dict[str, Any]]) -> list[list[float]]:
        with self._lock_factory(self._path):
            return self._inner.embed_documents(inputs)

    def embed_query(self, value: ContentValue) -> list[float]:
        with self._lock_factory(self._path):
            return self._inner.embed_query(value)
