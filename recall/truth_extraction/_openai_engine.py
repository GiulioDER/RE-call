"""An OpenAI compatible chat model as an `ExtractionEngine`.

This is one entry in `_ENGINES`, not a new architecture. Whatever the model returns still goes
through the full validation ladder in `_normalize.py`, so a model cannot skip a rung the rules
engine has to clear: the quote must be a verbatim substring of the body, the target must resolve
to exactly one corpus file, a date must appear in the body, and a batch over
`MAX_CLAIMS_PER_FILE` refuses the whole file rather than being truncated.

That is the entire argument for putting a model here. `recall/fix.py:10` records what happened
without one: on a real 792 memo corpus the mechanical rules proposed zero edges, and the four
candidates that survived them were all wrong on review. Their failures were reported speech, a
claim scoped INSIDE the target twice, and a hedge. Each is a semantic distinction no pattern over
the text can see, and each is restated in the prompt rather than assumed.

**Less of a departure than it looks.** The library already ships an OpenAI compatible HTTP client
defaulting to the OpenRouter base URL (`embeddings.py`), and `openai>=1.0` is already a `bench`
dependency. This is the first *generative* call, not the first network model call.

**What the model is trusted with.** Semantics only: which relation, to which document, on what
evidence. It supplies no identity, because proposal ids are content hashes the library computes
and `reasoning_proposals/_providers.py` recomputes and raises on mismatch. It supplies no
authority, because nothing here reaches corpus metadata; that needs a named human through
`recall/promotion.py`, and `recall/rewrite.py` accepts nothing else.

**Determinism, and the honest caveat.** Temperature 0 is not a guarantee from any hosted
provider. `recheck` exists to MEASURE whether it holds rather than to assume it.

Requires ``pip install recall[extract]``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Protocol
from urllib.parse import urlsplit

from recall.truth_extraction._prompt import ExtractionPrompt

OPENAI_EXTRACTION_ENGINE_ID = "recall.truth_extraction.openai"
DEFAULT_EXTRACTION_MODEL = "anthropic/claude-sonnet-4.5"
DEFAULT_EXTRACTION_BASE_URL = "https://openrouter.ai/api/v1"


class ChatClient(Protocol):
    """The narrow slice of a chat API this engine uses."""

    def complete(self, messages: list[dict[str, str]], **kwargs: object) -> str:
        ...


class OpenAIExtractionEngine:
    """Answers an extraction prompt with a chat model. Supplies semantics, never identity."""

    def __init__(
        self,
        *,
        client: ChatClient,
        model_id: str,
        revision: str,
        base_url: str | None = None,
    ) -> None:
        self._client = client
        self.model_id = model_id
        #: Part of the audit identity. An unpinned model is a moving target, and a claim
        #: attributed to a revision that has since changed is attributed to nothing.
        self.revision = revision
        #: The ENDPOINT is part of the identity, not just the model name. `extraction_cache_key`
        #: hashes engine_id, model_id and revision; without the host in one of them, the default
        #: hosted provider and a local server advertising the same model name produce identical
        #: keys, so the cache serves one endpoint's answers for the other and
        #: `ExtractedClaimProposalProvider` sees a single identity where there are two.
        #: Folded in here rather than into `model_id`, which must keep naming the model alone.
        self.engine_id = (
            f"{OPENAI_EXTRACTION_ENGINE_ID}@{_host_of(base_url)}"
            if base_url
            else OPENAI_EXTRACTION_ENGINE_ID
        )

    def run(self, prompt: ExtractionPrompt) -> str:
        # `ExtractionPrompt` carries `system` and `user` as separately rendered strings rather
        # than a message list, which keeps the prompt module free of any one API's message
        # shape. Assembling them is this adapter's job.
        return self._client.complete(
            [
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ],
            temperature=0,
        )


def _host_of(base_url: str) -> str:
    """The host of `base_url`, for the audit identity. Never the path, never credentials."""
    return urlsplit(base_url).hostname or base_url


def _setting(source: Mapping[str, str], name: str, default: str) -> str:
    """A setting, treating set-to-empty exactly like unset.

    `export RECALL_EXTRACTION_MODEL=` and a compose file's `RECALL_EXTRACTION_MODEL: ""` both
    produce an empty string, and a bare `.get(name, DEFAULT)` returns that empty string rather
    than the default. The engine then records `model_id=""` into the cache key and the audit
    record, and fails at request time with an error naming nothing.
    """
    return source.get(name, "").strip() or default


def _client_from_env(source: Mapping[str, str]) -> ChatClient:
    """Build the HTTP client, refusing clearly when the extra is not installed.

    Mirrors `entailment.py`: the ImportError names the exact install command, because an
    optional extra whose absence surfaces as a bare ModuleNotFoundError reads as a bug in the
    library rather than a choice the user has not yet made.
    """
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "the openai extraction engine requires: pip install recall[extract]"
        ) from exc

    # Imported here, not at module scope: `recall.embeddings` pulls the embedder stack, and this
    # module must stay light enough that naming the deterministic engine costs nothing. Reused
    # rather than reimplemented so one `_is_transient` classifier covers every network model
    # call in the library.
    from recall.embeddings import retry_with_backoff

    key = source.get("RECALL_EXTRACTION_API_KEY", "").strip()
    if not key:
        raise ValueError(
            "RECALL_EXTRACTION_API_KEY is required for the openai extraction engine"
        )
    base_url = _setting(source, "RECALL_EXTRACTION_BASE_URL", DEFAULT_EXTRACTION_BASE_URL)
    model = _setting(source, "RECALL_EXTRACTION_MODEL", DEFAULT_EXTRACTION_MODEL)
    timeout = float(_setting(source, "RECALL_EXTRACTION_TIMEOUT", "60"))
    # `max_retries=0` because `retry_with_backoff` below owns the retry policy. The SDK default
    # is 2 retries against a 600 second read timeout, and layering our own on top of that
    # multiplies the two: a wedged provider would block one memo for close to half an hour.
    inner = OpenAI(api_key=key, base_url=base_url, timeout=timeout, max_retries=0)

    class _Client:
        def complete(self, messages: list[dict[str, str]], **kwargs: object) -> str:
            reply = retry_with_backoff(
                lambda: inner.chat.completions.create(
                    model=model, messages=messages, **kwargs
                ),
                attempts=3,
            )
            # Every shape below returns "" rather than raising, because the ladder's `json`
            # rung already refuses "" and records it as a batch rejection a reviewer can see.
            # Raising here would make this engine's failure mode differ from the rules engine's
            # for the same malformed answer. The guards are not paranoia: OpenRouter returns
            # HTTP 200 with an `{"error": ...}` body and NO `choices` key when an upstream
            # provider fails, which makes `reply.choices[0]` an unguarded crash on a live path.
            choices = getattr(reply, "choices", None) or ()
            if not choices:
                return ""
            message = getattr(choices[0], "message", None)
            return getattr(message, "content", None) or ""

    return _Client()


def openai_engine_from_env(env: Mapping[str, str] | None = None) -> OpenAIExtractionEngine:
    """Construct the engine from `env`, defaulting to the process environment.

    `env` must be honoured rather than ignored. `resolve_extraction_engine` takes an explicit
    mapping so a caller can configure extraction without touching `os.environ`, and an engine
    that read the ambient environment anyway would answer for a DIFFERENT model than the one it
    was asked for, then record that other model's name in the cache key and the audit record.
    """
    source = env if env is not None else os.environ
    return OpenAIExtractionEngine(
        client=_client_from_env(source),
        model_id=_setting(source, "RECALL_EXTRACTION_MODEL", DEFAULT_EXTRACTION_MODEL),
        revision=_setting(source, "RECALL_EXTRACTION_REVISION", "unpinned"),
        base_url=_setting(source, "RECALL_EXTRACTION_BASE_URL", DEFAULT_EXTRACTION_BASE_URL),
    )


__all__ = [
    "DEFAULT_EXTRACTION_BASE_URL",
    "DEFAULT_EXTRACTION_MODEL",
    "OPENAI_EXTRACTION_ENGINE_ID",
    "ChatClient",
    "OpenAIExtractionEngine",
    "openai_engine_from_env",
]
