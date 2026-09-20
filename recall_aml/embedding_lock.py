"""Cross process serialization for hosted embedding provider calls."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
import importlib
from pathlib import Path
from typing import Any, cast

from recall.embeddings import Embedder
from recall_aml.models import ContentValue
from recall_aml.multimodal import MultimodalEmbedder


LockFactory = Callable[[Path], AbstractContextManager[None]]


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
