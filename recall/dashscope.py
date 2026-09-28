"""Alibaba Model Studio (DashScope) `text-embedding-v4`, through its native HTTP API.

Why this exists: the AML organisers recommend `text-embedding-v4` for every second Full run
(their written reply of 2026-09-26), so RE-call has to be measured on it before that run. It is
reached through the NATIVE DashScope endpoint rather than the OpenAI-compatible one because only
the native API accepts `text_type` (separate query and document encoders), `instruct` (the query
instruction Qwen3-Embedding models are trained with) and `output_type`. The OpenAI-compatible mode
accepts `dimensions` alone and drops the rest without an error, so a query would silently be
embedded as a document with no instruction.

The request contract, from Alibaba's "Text embedding synchronous API" page (read 2026-09-28):

* `POST {base}/services/embeddings/text-embedding/text-embedding`, `Authorization: Bearer <key>`;
* body `{"model", "input": {"texts": [...]}, "parameters": {"dimension", "text_type",
  "output_type", "instruct"}}`, at most 10 texts per request and 8,192 tokens per text;
* response `output.embeddings[]` carrying `embedding` and `text_index`.

⚠️ Two things that page does not say, and how each is handled rather than assumed:

* **Where `instruct` goes.** The page lists the field without placing it. It is sent inside
  `parameters`, which is where the DashScope SDK puts its keyword options. A misplaced field would
  be ignored silently, so `instruction_changes_query_vector` exists to check it against the live
  endpoint before any measurement leans on the instruction.
* **What happens to a text over 8,192 tokens.** Truncation and refusal are both possible. A batch
  the endpoint refuses with HTTP 400 is resent ONCE with every text cut to `FALLBACK_TEXT_BYTES`
  UTF-8 bytes. A byte-level tokenizer spends at least one byte per token, so that cut bounds the
  tokens whatever the script. A batch it accepts is sent exactly as given, and only the embedding
  input is ever cut, never stored text.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
from typing import Any, Mapping
from urllib.parse import urlsplit

import requests

from recall.embeddings import (
    EmbeddingProfile,
    _check_declared_width,
    batched_embed,
    retry_with_backoff,
)
from recall.errors import RecallError

_log = logging.getLogger(__name__)

MODEL = "text-embedding-v4"
#: The documented Singapore (international) endpoint. It is the endpoint a registered profile
#: declares; `DASHSCOPE_BASE_URL` may name a workspace host in the SAME region instead.
DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com/api/v1"
EMBEDDING_PATH = "/services/embeddings/text-embedding/text-embedding"
MAX_TEXTS_PER_REQUEST = 10
MAX_TOKENS_PER_TEXT = 8_192
#: Under 8,192 with room for special tokens; a byte-level tokenizer spends >= 1 byte per token.
FALLBACK_TEXT_BYTES = 8_000
ALLOWED_DIMENSIONS = frozenset({64, 128, 256, 512, 768, 1024, 1536, 2048})
#: The query instruction RE-call's W0 measures. Qwen3-Embedding models are trained with an
#: instruction on the query side only; documents take none. Worded for both AML tracks, since one
#: endpoint serves Textual and Coding. Changing this text is a new `instruction_version`.
MEMORY_RETRIEVAL_INSTRUCTION = (
    "Given a question or task, retrieve earlier conversation messages that contain the "
    "information needed to answer it"
)
MEMORY_RETRIEVAL_INSTRUCTION_VERSION = "memory-retrieval-v1"
#: A workspace host in the Singapore region, as the Model Studio console issues them.
_WORKSPACE_HOST = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}\.ap-southeast-1\.maas\.aliyuncs\.com$")
_CANONICAL_HOST = "dashscope-intl.aliyuncs.com"
_NORM_TOLERANCE = 1e-3


class DashScopeError(RecallError):
    """A refused or unreadable DashScope request.

    Carries ``http_status`` and ``headers`` under the names `recall.embeddings._is_transient`
    and `_retry_after_seconds` read, so a 429 or 5xx is retried and paced exactly as a Voyage one
    is, and a 400 or 401 fails at once.
    """

    def __init__(
        self,
        message: str,
        *,
        http_status: int | None = None,
        headers: Mapping[str, str] | None = None,
        code: str = "",
    ) -> None:
        super().__init__(message)
        self.http_status = http_status
        self.headers = headers if headers is not None else {}
        self.code = code


def resolve_base_url(value: str | None) -> str:
    """The endpoint to send to: the documented Singapore host, or a Singapore workspace host.

    Refuses any other host. The registered profile declares the Singapore region, and a vector
    from another region's deployment is not evidence about the profile it would be stored under.
    """
    if value is None or not value.strip():
        return DEFAULT_BASE_URL
    parsed = urlsplit(value.strip())
    host = (parsed.hostname or "").lower()
    path = parsed.path.rstrip("/")
    if (
        parsed.scheme != "https"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.port not in {None, 443}
        or path != "/api/v1"
        or not (host == _CANONICAL_HOST or _WORKSPACE_HOST.match(host))
    ):
        raise ValueError(
            "DASHSCOPE_BASE_URL must be https://dashscope-intl.aliyuncs.com/api/v1 or a "
            "Singapore workspace host https://<workspace>.ap-southeast-1.maas.aliyuncs.com/api/v1"
        )
    return f"https://{host}{path}"


def _utf8_prefix(text: str, limit: int) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= limit:
        return text
    return encoded[:limit].decode("utf-8", errors="ignore")


class DashScopeEmbedder:
    """`text-embedding-v4` with separate query and document encoders, under a registered identity.

    ``query_mode`` is ``query`` (no instruction) or ``query-instruct`` (the profile's instruction
    sent with every query); ``passage_mode`` is ``document``. Both come from the identity, so the
    registry decides the request rather than describing it, as for `VoyageEmbedder`.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        identity: EmbeddingProfile | None = None,
        dimension: int = 1024,
        base_url: str | None = None,
        instruction: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        max_parallel_requests: int = 1,
        session_factory: Any = None,
    ) -> None:
        key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not key:
            raise RuntimeError("DashScopeEmbedder needs DASHSCOPE_API_KEY (env) or an explicit api_key")
        if dimension not in ALLOWED_DIMENSIONS:
            raise ValueError(f"text-embedding-v4 has no {dimension}-dimension output")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("the DashScope timeout must be positive")
        if max_parallel_requests < 1:
            raise ValueError("DashScope parallel requests must be positive")
        self._key = key
        self._model = identity.model_name if identity is not None else MODEL
        self._dimension = dimension
        self._url = resolve_base_url(base_url) + EMBEDDING_PATH
        self._timeout = timeout
        self._max_retries = max_retries
        self._max_parallel_requests = max_parallel_requests
        self._session_factory = session_factory or requests.Session
        self._local = threading.local()
        self._profile = identity
        query_mode = identity.query_mode if identity is not None else "query"
        if query_mode not in {"query", "query-instruct"}:
            raise ValueError(f"DashScope profile has unsupported query mode {query_mode!r}")
        passage_mode = identity.passage_mode if identity is not None else "document"
        if passage_mode != "document":
            raise ValueError(f"DashScope profile has unsupported passage mode {passage_mode!r}")
        if query_mode == "query-instruct" and not instruction:
            raise ValueError("a query-instruct profile needs its instruction text")
        self._instruction = instruction if query_mode == "query-instruct" else None
        #: Texts cut to `FALLBACK_TEXT_BYTES` after the endpoint refused their batch.
        self.truncated_inputs = 0
        probe = self._embed_batch(["probe"], text_type="document", instruct=None)[0]
        self._dim = len(probe)
        _check_declared_width(identity, self._dim, f"DashScope {self._model}")
        if identity is not None and identity.normalization == "l2":
            norm = math.sqrt(sum(value * value for value in probe))
            if abs(norm - 1.0) > _NORM_TOLERANCE:
                raise RuntimeError(
                    f"profile {identity.profile_id!r} declares l2 normalization but "
                    f"{self._model} returned a vector of norm {norm:.4f}"
                )

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return f"dashscope:{self._model}"

    @property
    def profile(self) -> EmbeddingProfile | None:
        return self._profile

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.embed_passages(texts)

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, text_type="document", instruct=None)

    def embed_document_groups(self, groups: list[list[str]]) -> list[list[list[float]]]:
        """Embed grouped documents; v4 embeds each text alone, so a group is only bookkeeping.

        C9 wraps its context embedder in `ContextOverflowGuard`, which calls this method
        directly. The groups are flattened into one batched call and split back in order.
        """
        flat = [text for group in groups for text in group]
        vectors = self.embed_passages(flat)
        output: list[list[list[float]]] = []
        start = 0
        for group in groups:
            output.append(vectors[start : start + len(group)])
            start += len(group)
        return output

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], text_type="query", instruct=self._instruction)[0]

    def embed_query_without_instruction(self, text: str) -> list[float]:
        """The query encoder with no instruction, for the W0 instruction ablation only."""
        return self._embed([text], text_type="query", instruct=None)[0]

    def instruction_changes_query_vector(self, text: str = "When did we last talk about it?") -> bool:
        """Whether the endpoint honours `instruct`: the one check the documentation cannot give."""
        if self._instruction is None:
            raise RuntimeError("this profile sends no instruction to check")
        return self.embed_query(text) != self.embed_query_without_instruction(text)

    def _embed(
        self, texts: list[str], *, text_type: str, instruct: str | None
    ) -> list[list[float]]:
        return batched_embed(
            texts,
            lambda batch: self._embed_batch(batch, text_type=text_type, instruct=instruct),
            batch_size=MAX_TEXTS_PER_REQUEST,
            max_workers=self._max_parallel_requests,
        )

    def _embed_batch(
        self, batch: list[str], *, text_type: str, instruct: str | None
    ) -> list[list[float]]:
        try:
            return retry_with_backoff(
                lambda: self._post(batch, text_type=text_type, instruct=instruct),
                attempts=self._max_retries,
            )
        except DashScopeError as exc:
            oversized = [
                text for text in batch if len(text.encode("utf-8")) > FALLBACK_TEXT_BYTES
            ]
            if exc.http_status != 400 or not oversized:
                raise
            fitted = [_utf8_prefix(text, FALLBACK_TEXT_BYTES) for text in batch]
            self.truncated_inputs += len(oversized)
            _log.warning(
                "dashscope_batch_refused_refitted",
                extra={"texts": len(batch), "cut": len(oversized), "code": exc.code},
            )
            return retry_with_backoff(
                lambda: self._post(fitted, text_type=text_type, instruct=instruct),
                attempts=self._max_retries,
            )

    def _session(self) -> Any:
        session = getattr(self._local, "session", None)
        if session is None:
            session = self._session_factory()
            self._local.session = session
        return session

    def _post(
        self, batch: list[str], *, text_type: str, instruct: str | None
    ) -> list[list[float]]:
        parameters: dict[str, object] = {
            "dimension": self._dimension,
            "text_type": text_type,
            "output_type": "dense",
        }
        if instruct is not None:
            parameters["instruct"] = instruct
        body = {"model": self._model, "input": {"texts": batch}, "parameters": parameters}
        response = self._session().post(
            self._url,
            headers={
                "Authorization": f"Bearer {self._key}",
                "Content-Type": "application/json",
            },
            data=json.dumps(body),
            timeout=self._timeout,
        )
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        if response.status_code != 200:
            raise DashScopeError(
                f"DashScope HTTP {response.status_code}: "
                f"{payload.get('code', '')} {payload.get('message', '')}".strip(),
                http_status=response.status_code,
                headers=response.headers,
                code=str(payload.get("code", "")),
            )
        items = (payload.get("output") or {}).get("embeddings")
        if not isinstance(items, list) or len(items) != len(batch):
            raise DashScopeError(
                f"DashScope returned {len(items) if isinstance(items, list) else 'no'} "
                f"embeddings for {len(batch)} texts",
                code=str(payload.get("code", "")),
            )
        ordered: list[list[float] | None] = [None] * len(batch)
        for item in items:
            index = item.get("text_index")
            vector = item.get("embedding")
            if not isinstance(index, int) or not 0 <= index < len(batch) or not isinstance(vector, list):
                raise DashScopeError("DashScope returned an embedding without a valid text_index")
            ordered[index] = [float(value) for value in vector]
        if any(vector is None for vector in ordered):
            raise DashScopeError("DashScope returned a duplicate text_index")
        return [vector for vector in ordered if vector is not None]
