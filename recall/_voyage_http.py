"""The Voyage REST calls RE-call's serving path makes, without importing the `voyageai` SDK.

Why this exists: `import voyageai` runs `voyageai/chunking.py`, which imports
`langchain_text_splitters`, which imports `transformers` and `torch`. On a host where those are
installed (VPS2, where every stdio MCP server runs, and this workstation) that single import was
measured on 2026-09-22 at 11.43 s and 9.97 s with a 778 MB peak, in two samples on VPS2, and it
happened in every server process that built a Voyage embedder. Nothing RE-call calls needs any of
it: the calls below are three HTTPS POSTs with JSON bodies.

The contract is the SDK's wire behaviour at voyageai 0.5.0, reproduced rather than redesigned,
so a vector is byte-for-byte the vector the SDK would have returned:

* the same endpoints (`/embeddings`, `/contextualizedembeddings`, `/rerank`) under the same base
  URL, including the MongoDB host the SDK selects for an `al-` key;
* the same JSON body, key for key, including the `null`s the SDK sends for unset options and the
  `"encoding_format": "base64"` it adds to every embedding request;
* the same decoding: base64 to float32 in the host's byte order (what the SDK's `np.frombuffer`
  does), each value widened to a Python float, which is exact;
* only `Authorization` and `Content-Type` from the SDK's own headers (it strips the rest before
  sending), on a `requests.Session` kept per thread and renewed after 180 s, with connection-level
  retries of 2, exactly as the SDK's requestor does;
* the same error mapping by status, and errors carrying the same attributes (`http_status`,
  `headers` as the case-insensitive `requests` mapping) and class names that
  `recall.embeddings._is_transient` and `_retry_after_seconds` read. Retries stay the caller's:
  `retry_with_backoff` is the single owner, as it was with `max_retries=0` on the SDK.

What is deliberately NOT reproduced: the SDK's local models, client-side chunking, multimodal
inputs (`VoyageMultimodalEmbedder` stays on the SDK; it is opt-in and off by default), and the
chunk-text validation of contextualized responses, which RE-call does not read.

⛔ The `voyageai` package must stay installed with the `voyage` extra even though nothing here
imports it: `RegisteredProfile._dependency` records its VERSION in every Voyage profile's
identity, which keys the embedding cache and generation lineage. Removing it would re-partition
every corpus.
"""

from __future__ import annotations

import base64
import json
import threading
import time
from array import array
from collections import namedtuple
from dataclasses import dataclass, field
from typing import Any

import requests
from requests.adapters import HTTPAdapter

from recall.errors import RecallError

#: The SDK's constants at voyageai 0.5.0 (`voyageai/api_resources/api_requestor.py`).
MAX_SESSION_LIFETIME_SECS = 180
MAX_CONNECTION_RETRIES = 2
DEFAULT_BASE_URL = "https://api.voyageai.com/v1"
MONGODB_BASE_URL = "https://ai.mongodb.com/v1"


class VoyageError(RecallError):
    """Same attributes and rendering as `voyageai.error.VoyageError`.

    Rooted in `RecallError` like every exception this package raises, where the SDK's derives from
    `Exception` directly; `except Exception` and the retry classifier see no difference.
    """

    def __init__(
        self,
        message: str | None = None,
        http_body: str | None = None,
        http_status: int | None = None,
        json_body: Any = None,
        headers: Any = None,
    ) -> None:
        super().__init__(message)
        self._message = message
        self.http_body = http_body
        self.http_status = http_status
        self.json_body = json_body
        self.headers = headers or {}
        self.request_id = self.headers.get("request-id", None)

    def __str__(self) -> str:
        msg = self._message or "<empty message>"
        if self.request_id is not None:
            return f"Request {self.request_id}: {msg}"
        return msg

    @property
    def user_message(self) -> str | None:
        return self._message


class APIError(VoyageError):
    pass


class Timeout(VoyageError):
    pass


class APIConnectionError(VoyageError):
    pass


class InvalidRequestError(VoyageError):
    pass


class MalformedRequestError(VoyageError):
    pass


class AuthenticationError(VoyageError):
    pass


class RateLimitError(VoyageError):
    pass


class ServerError(VoyageError):
    pass


class ServiceUnavailableError(VoyageError):
    pass


@dataclass(frozen=True)
class EmbeddingsObject:
    embeddings: list[list[float]]
    total_tokens: int


@dataclass(frozen=True)
class ContextualizedEmbeddingsResult:
    index: int
    embeddings: list[list[float]]


@dataclass(frozen=True)
class ContextualizedEmbeddingsObject:
    results: list[ContextualizedEmbeddingsResult] = field(default_factory=list)
    total_tokens: int = 0


RerankingResult = namedtuple("RerankingResult", ["index", "document", "relevance_score"])


@dataclass(frozen=True)
class RerankingObject:
    results: list[RerankingResult]
    total_tokens: int


def default_base_url(api_key: str) -> str:
    """The SDK's rule: a MongoDB Atlas key (`al-`) is served from a different host."""
    return MONGODB_BASE_URL if api_key.startswith("al-") else DEFAULT_BASE_URL


def decode_embedding(value: Any, output_dtype: str | None) -> Any:
    """Decode one base64 embedding exactly as the SDK does; a non-string passes through."""
    if type(value) is not str:
        return value
    if output_dtype not in (None, "float"):
        raise ValueError(f"output_dtype {output_dtype!r} is not supported by this client")
    # `array("f")` reads float32 in the host's byte order, exactly as the SDK's
    # `np.frombuffer(..., np.float32)` does, and `tolist()` widens each value to a Python float.
    values = array("f")
    values.frombytes(base64.b64decode(value))
    return values.tolist()


def _make_session() -> requests.Session:
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=MAX_CONNECTION_RETRIES))
    return session


class Client:
    """A drop-in for the subset of `voyageai.Client` RE-call calls.

    The constructor keeps the SDK's keywords so callers and test doubles are interchangeable.
    `max_retries` is accepted for that reason and must be 0: retrying belongs to the caller.
    """

    def __init__(
        self,
        api_key: str | None = None,
        max_retries: int = 0,
        timeout: float | None = None,
        base_url: str | None = None,
    ) -> None:
        if not api_key:
            raise AuthenticationError(
                "An API key is required for API-based models. Set your API key via the "
                "VOYAGE_API_KEY environment variable or pass it to the client (api_key=...)."
            )
        if max_retries != 0:
            raise ValueError("retries belong to recall.embeddings.retry_with_backoff; pass 0")
        self._headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        self._base_url = (base_url or default_base_url(api_key)).rstrip("/")
        self._timeout = timeout if timeout else 600
        self._local = threading.local()

    def _session(self) -> requests.Session:
        local = self._local
        session = getattr(local, "session", None)
        if session is None or time.time() - local.created >= MAX_SESSION_LIFETIME_SECS:
            if session is not None:
                session.close()
            session = local.session = _make_session()
            local.created = time.time()
        return session

    def _post(self, path: str, params: dict[str, Any]) -> Any:
        data = json.dumps(params).encode()
        try:
            result = self._session().request(
                "post",
                self._base_url + path,
                headers=self._headers,
                data=data,
                timeout=self._timeout,
            )
        except requests.exceptions.Timeout as exc:
            raise Timeout(f"Request timed out: {exc}") from exc
        except requests.exceptions.RequestException as exc:
            raise APIConnectionError(f"Error communicating with VoyageAI: {exc}") from exc
        return _interpret(result)

    def embed(
        self,
        texts: list[str],
        model: str,
        input_type: str | None = None,
        truncation: bool = True,
        output_dtype: str | None = None,
        output_dimension: int | None = None,
    ) -> EmbeddingsObject:
        body = self._post(
            "/embeddings",
            {
                "input": texts,
                "model": model,
                "input_type": input_type,
                "truncation": truncation,
                "output_dtype": output_dtype,
                "output_dimension": output_dimension,
                "encoding_format": "base64",
            },
        )
        return EmbeddingsObject(
            embeddings=[decode_embedding(d["embedding"], output_dtype) for d in body["data"]],
            total_tokens=body["usage"]["total_tokens"],
        )

    def contextualized_embed(
        self,
        inputs: list[list[str]],
        model: str,
        input_type: str | None = None,
        output_dtype: str | None = None,
        output_dimension: int | None = None,
    ) -> ContextualizedEmbeddingsObject:
        if not inputs:
            raise ValueError("inputs must not be empty")
        if all(isinstance(item, str) for item in inputs):
            # The SDK accepts flat strings only for queries (without auto-chunking) and wraps
            # each in its own group.
            if input_type != "query":
                raise ValueError(
                    "List[str] inputs requires enable_auto_chunking=True or input_type='query'"
                )
            inputs = [[item] for item in inputs]  # type: ignore[list-item]
        body = self._post(
            "/contextualizedembeddings",
            {
                "inputs": inputs,
                "model": model,
                "input_type": input_type,
                "output_dtype": output_dtype,
                "output_dimension": output_dimension,
                "encoding_format": "base64",
            },
        )
        return ContextualizedEmbeddingsObject(
            results=[
                ContextualizedEmbeddingsResult(
                    index=index,
                    embeddings=[
                        decode_embedding(chunk["embedding"], output_dtype) for chunk in group["data"]
                    ],
                )
                for index, group in enumerate(body["data"])
            ],
            total_tokens=body["usage"]["total_tokens"],
        )

    def rerank(
        self,
        query: str,
        documents: list[str],
        model: str,
        top_k: int | None = None,
        truncation: bool = True,
    ) -> RerankingObject:
        body = self._post(
            "/rerank",
            {
                "query": query,
                "documents": documents,
                "model": model,
                "top_k": top_k,
                "truncation": truncation,
            },
        )
        return RerankingObject(
            results=[
                RerankingResult(
                    index=d["index"],
                    document=documents[d["index"]],
                    relevance_score=d["relevance_score"],
                )
                for d in body["data"]
            ],
            total_tokens=body["usage"]["total_tokens"],
        )


def _interpret(result: requests.Response) -> Any:
    """The SDK's `_interpret_response_line`, status for status."""
    rcode = result.status_code
    rheaders = result.headers
    # Decoded unguarded, as the SDK does: an undecodable body raises `UnicodeDecodeError` there
    # too, and wrapping it here would hand `_is_transient` a status it never saw before.
    rbody = result.content.decode("utf-8")
    if rcode == 204:
        return None
    if rcode == 500:
        raise ServerError("The server failed to process the request.", rbody, rcode, headers=rheaders)
    if rcode in (502, 503, 504):
        raise ServiceUnavailableError(
            "The server is overloaded or not ready yet.", rbody, rcode, headers=rheaders
        )
    try:
        data = rbody if "text/plain" in rheaders.get("Content-Type", "") else json.loads(rbody)
    except json.JSONDecodeError as exc:
        raise APIError(f"HTTP code {rcode} from API ({rbody})", rbody, rcode, headers=rheaders) from exc
    if 400 <= rcode < 500:
        raise _error_for(rbody, rcode, data, rheaders)
    return data


def _error_for(rbody: str, rcode: int, data: Any, rheaders: Any) -> VoyageError:
    try:
        message = data["detail"]
    except (KeyError, TypeError):
        return APIError(
            f"Invalid response object from API: {rbody!r} (HTTP response code was {rcode})",
            rbody,
            rcode,
            data,
        )
    if rcode == 400:
        return InvalidRequestError(message, rbody, rcode, data, rheaders)
    if rcode == 401:
        return AuthenticationError(message, rbody, rcode, data, rheaders)
    if rcode == 422:
        return MalformedRequestError(message, rbody, rcode, data, rheaders)
    if rcode == 429:
        return RateLimitError(message, rbody, rcode, data, rheaders)
    return APIError(f"{message} {rbody} {rcode} {data} {rheaders}", rbody, rcode, data, rheaders)
