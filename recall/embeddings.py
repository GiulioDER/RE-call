from __future__ import annotations

import hashlib
import math
import os
from typing import Protocol, runtime_checkable


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


class FastEmbedEmbedder:
    """Real local embeddings (no API key). Requires `pip install recall[fastembed]`."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "FastEmbedEmbedder requires the fastembed extra: pip install recall[fastembed]"
            ) from exc
        self._model = TextEmbedding(model_name=model_name)
        self._name = model_name
        self._dim = len(next(iter(self._model.embed(["probe"]))))

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return self._name

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(x) for x in vec] for vec in self._model.embed(texts)]


class SentenceTransformerEmbedder:
    """Local sentence-transformers embeddings (base or fine-tuned). Requires recall[rerank]."""

    def __init__(self, model_path: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError("SentenceTransformerEmbedder requires: pip install recall[rerank]") from exc
        self._model = SentenceTransformer(model_path)
        self._name = f"st:{model_path.rsplit('/', 1)[-1]}"
        self._dim = self._model.get_sentence_embedding_dimension()

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return self._name

    def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(texts, normalize_embeddings=False)
        return [[float(x) for x in v] for v in vecs]


class VoyageEmbedder:
    """Voyage cloud embeddings. Requires `pip install recall[voyage]` and VOYAGE_API_KEY."""

    def __init__(self, model: str = "voyage-3", api_key: str | None = None) -> None:
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
        self._dim = len(self._client.embed(["probe"], model=model).embeddings[0])

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return self._name

    def embed(self, texts: list[str]) -> list[list[float]]:
        result = self._client.embed(texts, model=self._model)
        return [[float(x) for x in v] for v in result.embeddings]


class OpenAIEmbedder:
    """OpenAI cloud embeddings. Requires `pip install recall[openai]` and OPENAI_API_KEY."""

    def __init__(self, model: str = "text-embedding-3-small", api_key: str | None = None) -> None:
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OpenAIEmbedder needs OPENAI_API_KEY (env) or an explicit api_key")
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError("OpenAIEmbedder requires: pip install recall[openai]") from exc
        self._client = OpenAI(api_key=key)
        self._model = model
        self._name = f"openai:{model}"
        probe = self._client.embeddings.create(input=["probe"], model=model)
        self._dim = len(probe.data[0].embedding)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return self._name

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(input=texts, model=self._model)
        return [[float(x) for x in d.embedding] for d in resp.data]
