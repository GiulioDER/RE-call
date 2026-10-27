"""The Voyage AI embedders, plain and contextualized.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from recall.embedding_core import (
    EmbeddingProfile,
    _check_declared_width,
    _log,
    batched_embed,
    retry_with_backoff,
)


def _voyage_client_class(owner: str) -> type:
    """RE-call's own Voyage HTTP client class, after checking the `voyage` extra is installed.

    `recall._voyage_http` replaces the `voyageai` SDK on this path because importing the SDK drags
    in `transformers` and `torch` (its module docstring has the measurement). The SDK must still be
    INSTALLED: its version is part of every Voyage profile's identity
    (`RegisteredProfile._dependency`), so the check stays, made with `find_spec`, which locates
    the package without importing it.
    """
    from importlib.util import find_spec

    message = f'{owner} requires: pip install "recall-rag[voyage]"'
    try:
        installed = find_spec("voyageai") is not None
    except ValueError:  # present in `sys.modules` without a spec, as a test double is
        installed = True
    if not installed:
        raise ImportError(message)
    try:
        from recall import _voyage_http
    except ImportError as exc:  # pragma: no cover - `requests` arrives with the same extra
        raise ImportError(message) from exc
    return _voyage_http.Client


#: Voyage Context 4's context window. RE-call always sends pre-chunked documents, and for
#: pre-chunked input Voyage caps one example AND a whole request at this many tokens (120K applies
#: only with provider auto-chunking, which RE-call never enables). Contextualized embeddings do
#: not truncate: an example over the window is refused with HTTP 400.
VOYAGE_CONTEXT_WINDOW_TOKENS = 32_000


#: The share of the window one part or one request may fill, counted locally. The local count
#: uses the tokenizer Voyage publishes, but the provider is what decides, and nothing documents
#: whether it adds tokens around each chunk; the remaining tenth absorbs that disagreement.
VOYAGE_CONTEXT_TOKEN_HEADROOM = 0.9


#: Tokens allowed per chunk for special tokens when the tokenizer is unavailable (see below).
_FALLBACK_SPECIAL_TOKENS_PER_CHUNK = 2


TokenCounter = Callable[[list[str]], list[int]]


def _voyage_token_counter(model: str) -> TokenCounter | None:
    """Count tokens with Voyage's published tokenizer for ``model``, or None if it cannot load.

    This is exactly what `voyageai.Client.count_tokens` does (`voyageai/_base.py` at 0.5.0:
    `tokenizers.Tokenizer.from_pretrained(f"voyageai/{model}")` with truncation off, one
    `encode_batch`, the length of each encoding), done without importing the SDK, whose import
    costs about 10 s and 780 MB (see `recall._voyage_http`). It needs no API key. The first load
    downloads the tokenizer from Hugging Face; later loads read the local cache.
    """
    try:
        from tokenizers import Tokenizer
    except ImportError:
        return None
    try:
        tokenizer = Tokenizer.from_pretrained(f"voyageai/{model}")
    except Exception as exc:  # BROAD-CATCH: fail-open; hub errors vary, the byte bound is sound
        _log.warning(
            "voyage_context_tokenizer_unavailable",
            extra={"model": model, "error_class": type(exc).__name__},
        )
        return None
    tokenizer.no_truncation()

    def count(texts: list[str]) -> list[int]:
        return [len(encoding.ids) for encoding in tokenizer.encode_batch(texts)]

    return count


def _utf8_token_bound(texts: list[str]) -> list[int]:
    """An upper bound on each text's token count that needs no tokenizer.

    Every token of a byte-level or byte-fallback tokenizer covers at least one byte, and every
    token of a character-level one at least one character, so UTF-8 bytes bound the count from
    above; the allowance covers the special tokens `count_tokens` includes per text.
    """
    return [len(text.encode("utf-8")) + _FALLBACK_SPECIAL_TOKENS_PER_CHUNK for text in texts]


class VoyageContextualizedEmbedder:
    """Voyage Context 4 with explicit ordered document groups."""

    def __init__(
        self,
        model: str = "voyage-context-4",
        api_key: str | None = None,
        *,
        output_dimension: int = 1024,
        max_request_inputs: int = 1000,
        max_request_chunks: int = 16_000,
        max_request_chars: int = 60_000,
        max_request_tokens: int = VOYAGE_CONTEXT_WINDOW_TOKENS,
        token_counter: TokenCounter | None = None,
        max_retries: int = 3,
        timeout: float = 60.0,
        identity: EmbeddingProfile | None = None,
        max_parallel_requests: int = 1,
    ) -> None:
        """Build the client and probe its width.

        ``max_request_tokens`` is the provider's window; parts and requests are planned to
        `VOYAGE_CONTEXT_TOKEN_HEADROOM` of it. ``token_counter`` returns one count per text and
        defaults to Voyage's own tokenizer, loaded on first use, falling back to a UTF-8 byte
        bound when that cannot be loaded.
        """
        key = api_key or os.environ.get("VOYAGE_API_KEY")
        if not key:
            raise RuntimeError(
                "VoyageContextualizedEmbedder needs VOYAGE_API_KEY (env) or an explicit api_key"
            )
        if output_dimension < 1 or max_request_inputs < 1 or max_request_chunks < 1:
            raise ValueError("Voyage Context request limits must be positive")
        if max_request_chars < 1 or max_retries < 1 or timeout <= 0:
            raise ValueError("Voyage Context request settings are invalid")
        if max_request_tokens < 1:
            raise ValueError("Voyage Context token limit must be positive")
        if max_parallel_requests < 1:
            raise ValueError("Voyage Context parallel requests must be positive")
        self._max_parallel_requests = max_parallel_requests
        client_class = _voyage_client_class("VoyageContextualizedEmbedder")
        self._client = client_class(api_key=key, max_retries=0, timeout=timeout)
        self._model = identity.model_name if identity is not None else model
        self._name = f"voyage-context:{self._model}"
        self._output_dimension = output_dimension
        self._max_request_inputs = max_request_inputs
        self._max_request_chunks = max_request_chunks
        self._max_request_chars = max_request_chars
        self._max_request_tokens = max_request_tokens
        self._token_counter = token_counter
        self._max_retries = max_retries
        probe = self._embed_query("probe")
        self._dim = len(probe)
        _check_declared_width(identity, self._dim, "the Voyage contextualized endpoint")
        self._profile = identity

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return self._name

    @property
    def profile(self) -> EmbeddingProfile | None:
        return self._profile

    def _embed_query(self, text: str) -> list[float]:
        result = retry_with_backoff(
            lambda: self._client.contextualized_embed(
                inputs=[text],
                model=self._model,
                input_type="query",
                output_dimension=self._output_dimension,
                output_dtype="float",
            ),
            attempts=self._max_retries,
        )
        results = getattr(result, "results", None)
        if not isinstance(results, list) or len(results) != 1:
            raise RuntimeError("Voyage Context query response did not contain exactly one result")
        embeddings = getattr(results[0], "embeddings", None)
        if not isinstance(embeddings, list) or len(embeddings) != 1:
            raise RuntimeError("Voyage Context query response did not contain exactly one vector")
        return [float(value) for value in embeddings[0]]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_query(text)

    def _token_budget(self) -> int:
        """Tokens one part, and one request, may hold by the local count."""
        return max(1, int(self._max_request_tokens * VOYAGE_CONTEXT_TOKEN_HEADROOM))

    def _count_tokens(self, texts: list[str]) -> list[int]:
        if self._token_counter is None:
            counter = _voyage_token_counter(self._model)
            if counter is None:
                # Sound but coarse: text denser than one token per byte cannot exist, so no
                # part overflows, but parts are smaller than the tokenizer would allow, and a
                # group longer than the byte budget splits where a tokenizer run would not.
                _log.warning(
                    "voyage_context_token_bound_fallback",
                    extra={"model": self._model, "bound": "utf8-bytes"},
                )
                counter = _utf8_token_bound
            self._token_counter = counter
        counts = self._token_counter(texts)
        if len(counts) != len(texts):
            raise RuntimeError(
                f"token counter returned {len(counts)} counts for {len(texts)} texts"
            )
        return counts

    def _split_group(self, group: list[str]) -> list[tuple[list[str], int]]:
        """Split one document between chunks, never truncating a chunk or merging two documents.

        Returns each part with its local token count. A part closes before it would pass any
        bound: inputs, chunks, characters, or tokens. Tokens are the bound the provider enforces;
        characters are kept because they are part of the profile identity, and keeping them
        means a group every part of which already fitted the token budget splits exactly as it
        did before tokens were counted, so its stored vectors stay reproducible.

        A single chunk over the token budget is sent ALONE rather than refused here. The local
        count estimates the provider's, and only the provider's answer is authoritative: a chunk
        between the budget and the window may well be accepted, and one over the window is
        refused with HTTP 400, which fails the build loudly (nothing is truncated) and is what
        `recall_aml.context_overflow.ContextOverflowGuard` re-plans on.
        """
        for text in group:
            if not isinstance(text, str):
                raise TypeError("Voyage Context document chunks must be strings")
        budget = self._token_budget()
        counts = self._count_tokens(group)
        parts: list[tuple[list[str], int]] = []
        current: list[str] = []
        chars = 0
        tokens = 0
        for position, (text, count) in enumerate(zip(group, counts, strict=True)):
            if current and (
                len(current) >= self._max_request_inputs
                or len(current) >= self._max_request_chunks
                or chars + len(text) > self._max_request_chars
                or tokens + count > budget
            ):
                parts.append((current, tokens))
                current = []
                chars = 0
                tokens = 0
            if count > budget:
                _log.warning(
                    "voyage_context_chunk_over_token_budget",
                    extra={
                        "chunk_position": position,
                        "chunk_tokens": count,
                        "token_budget": budget,
                        "window_tokens": self._max_request_tokens,
                    },
                )
            current.append(text)
            chars += len(text)
            tokens += count
        if current:
            parts.append((current, tokens))
        return parts

    def _embed_group_parts(self, groups: list[list[str]]) -> list[list[list[float]]]:
        result = retry_with_backoff(
            lambda: self._client.contextualized_embed(
                inputs=groups,
                model=self._model,
                input_type="document",
                output_dimension=self._output_dimension,
                output_dtype="float",
            ),
            attempts=self._max_retries,
        )
        results = getattr(result, "results", None)
        if not isinstance(results, list) or len(results) != len(groups):
            raise RuntimeError("Voyage Context document response changed group alignment")
        output: list[list[list[float]]] = []
        for group, response_group in zip(groups, results, strict=True):
            embeddings = getattr(response_group, "embeddings", None)
            if not isinstance(embeddings, list) or len(embeddings) != len(group):
                raise RuntimeError(
                    "Voyage Context document response changed chunk alignment: "
                    f"received {len(embeddings) if isinstance(embeddings, list) else 'invalid'} "
                    f"vectors for {len(group)} chunks"
                )
            output.append([[float(value) for value in vector] for vector in embeddings])
        return output

    def embed_document_groups(self, groups: list[list[str]]) -> list[list[list[float]]]:
        output: list[list[list[float]]] = [[] for _ in groups]
        parts: list[tuple[int, list[str], int]] = [
            (index, part, part_tokens)
            for index, group in enumerate(groups)
            for part, part_tokens in self._split_group(group)
        ]
        budget = self._token_budget()
        # Plan every request first, then send them. Each request is independent (a document
        # part is never split across two), so sending them concurrently returns exactly the
        # vectors a sequential run returns.
        requests: list[tuple[list[list[str]], list[int]]] = []
        cursor = 0
        while cursor < len(parts):
            request: list[list[str]] = []
            request_indices: list[int] = []
            chars = 0
            chunks = 0
            tokens = 0
            while cursor < len(parts):
                index, part, part_tokens = parts[cursor]
                part_chars = sum(len(text) for text in part)
                # Pre-chunked input caps the WHOLE request at the window, not only each part.
                if request and (
                    len(request) >= self._max_request_inputs
                    or chunks + len(part) > self._max_request_chunks
                    or chars + part_chars > self._max_request_chars
                    or tokens + part_tokens > budget
                ):
                    break
                request.append(part)
                request_indices.append(index)
                chars += part_chars
                chunks += len(part)
                tokens += part_tokens
                cursor += 1
            requests.append((request, request_indices))
        if self._max_parallel_requests > 1 and len(requests) > 1:
            workers = min(self._max_parallel_requests, len(requests))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                responses = list(
                    pool.map(lambda planned: self._embed_group_parts(planned[0]), requests)
                )
        else:
            responses = [self._embed_group_parts(request) for request, _ in requests]
        for (_, request_indices), vectors_by_part in zip(requests, responses, strict=True):
            for index, vectors in zip(request_indices, vectors_by_part, strict=True):
                output[index].extend(vectors)
        for group, vectors in zip(groups, output, strict=True):
            if len(vectors) != len(group):
                raise RuntimeError("Voyage Context document response lost chunk alignment")
        return output

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return self.embed_document_groups([texts])[0]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.embed_passages(texts)


class VoyageEmbedder:
    """Voyage cloud embeddings. Requires `pip install "recall-rag[voyage]"` and VOYAGE_API_KEY."""

    def __init__(
        self,
        model: str = "voyage-3",
        api_key: str | None = None,
        batch_size: int = 128,
        max_retries: int = 3,
        identity: EmbeddingProfile | None = None,
        max_parallel_requests: int = 1,
    ) -> None:
        """Build a Voyage client, optionally under a registered profile's immutable identity.

        ``identity`` is how a registered hosted profile is built: `RegisteredProfile.build` passes
        the identity it declared, and this class then carries THAT object rather than minting a
        second one. Without it every consumer falls back to `legacy_embedding_profile`, so the
        profile id a generation and a calibration were registered under could never match the
        runtime and the corpus could not be served. When it is supplied, the identity's model name
        is what gets sent to the provider, so the registry decides the request rather than
        describing it.
        """
        key = api_key or os.environ.get("VOYAGE_API_KEY")
        if not key:
            raise RuntimeError("VoyageEmbedder needs VOYAGE_API_KEY (env) or an explicit api_key")
        client_class = _voyage_client_class("VoyageEmbedder")
        # `max_retries=0` pins the single-owner retry policy `OpenAICompatEmbedder` needs too:
        # `retry_with_backoff` in `embed` below is the only thing that resends. The timeout is
        # stated because the SDK this client replaced defaulted to none.
        from recall.embedding_registry import _voyage_timeout

        self._client = client_class(
            api_key=key, max_retries=0, timeout=_voyage_timeout(os.environ)
        )
        self._model = identity.model_name if identity is not None else model
        self._name = f"voyage:{self._model}"
        self._batch_size = batch_size
        self._max_retries = max_retries
        if max_parallel_requests < 1:
            raise ValueError("Voyage parallel requests must be positive")
        self._max_parallel_requests = max_parallel_requests
        self._dim = len(self._client.embed(["probe"], model=self._model).embeddings[0])
        _check_declared_width(identity, self._dim, "the Voyage endpoint")
        self._profile = identity

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return self._name

    @property
    def profile(self) -> EmbeddingProfile | None:
        """The registered identity, or None for a legacy construction.

        `embedding_profile` reads this attribute and falls back to `legacy_embedding_profile`
        when it is not an `EmbeddingProfile`, so returning None preserves every existing caller.
        """
        return self._profile

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed in provider-safe batches with exponential-backoff retry per batch.

        A single request sending every chunk at once will exceed the API's per-request limit on
        a real corpus and has no tolerance for a transient 429/5xx; batching + retry make bulk
        indexing survivable. Results are concatenated in input order (see ``batched_embed``).
        """
        return self._embed_typed(texts, input_type=None)

    def embed_query(self, text: str) -> list[float]:
        """Use Voyage's retrieval query encoder when the registered profile declares it."""
        mode = self._profile.query_mode if self._profile is not None else "embed"
        if mode not in {"embed", "query"}:
            raise RuntimeError(f"Voyage profile has unsupported query mode {mode!r}")
        return self._embed_typed([text], input_type="query" if mode == "query" else None)[0]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        """Use Voyage's retrieval document encoder when the profile declares it."""
        mode = self._profile.passage_mode if self._profile is not None else "embed"
        if mode not in {"embed", "document"}:
            raise RuntimeError(f"Voyage profile has unsupported passage mode {mode!r}")
        return self._embed_typed(
            texts,
            input_type="document" if mode == "document" else None,
        )

    def _embed_typed(
        self, texts: list[str], *, input_type: str | None
    ) -> list[list[float]]:
        """Embed batches while keeping input type inside the retried provider call.

        A batch is cut by count only, and Voyage also caps the TOKENS in one request: 120K for
        voyage-code-3, 320K for voyage-4, and undocumented for voyage-code-4. Token-dense text,
        such as pasted CSV at about one character per token, fills 128 texts past 240K tokens
        (measured 2026-09-26 on CLBench with the voyage-code-4 tokenizer). voyage-code-4 accepted
        that 240,869-token request the same day, so its cap lies above it, but a model with a lower
        cap, or denser data, gets a 400. A 400 is not transient, so such a request used to fail
        every retry identically. A refused batch of two or more texts is now halved and each half
        sent on its own, recursively; a batch Voyage accepts is sent exactly as before, and a single
        refused text still raises. The model embeds each text independently, so a split changes
        which request carries a text, not its vector.

        Any 400 is split, not only a size refusal, because Voyage's wording is not a contract. A
        400 that is not about size (one malformed text) costs about two requests per halving on
        the path to the offending text, then raises as before; a bad model or key never gets
        here, because construction probes the endpoint.
        """

        def _send(batch: list[str]) -> list[list[float]]:
            kwargs: dict[str, object] = {"model": self._model}
            if input_type is not None:
                kwargs["input_type"] = input_type
            result = retry_with_backoff(
                lambda: self._client.embed(batch, **kwargs),
                attempts=self._max_retries,
            )
            return [[float(x) for x in v] for v in result.embeddings]

        def _embed_batch(batch: list[str]) -> list[list[float]]:
            try:
                return _send(batch)
            # Anything but a provider 400 on two or more texts is re-raised unchanged.
            except Exception as exc:  # BROAD-CATCH: fail-closed
                if len(batch) < 2 or getattr(exc, "http_status", None) != 400:
                    raise
                _log.warning(
                    "voyage_request_split",
                    extra={"model": self._model, "text_count": len(batch),
                           "error_class": type(exc).__name__},
                )
            middle = len(batch) // 2
            return _embed_batch(batch[:middle]) + _embed_batch(batch[middle:])

        return batched_embed(
            texts,
            _embed_batch,
            batch_size=self._batch_size,
            max_workers=self._max_parallel_requests,
        )
