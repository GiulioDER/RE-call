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

Requires ``pip install "recall-rag[extract]"``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Protocol
from urllib.parse import urlsplit

from recall._chat_content import assistant_text
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


#: Recorded in place of an endpoint this module cannot parse. A literal, because the raw value
#: must never be recorded: a scheme-less base_url gives `urlsplit` no hostname, and falling back
#: to the raw string would write `user:sk-secret@host/v1?token=...` into the audit record and the
#: cache key. An endpoint that cannot be identified safely is better recorded as unidentified.
UNPARSED_ENDPOINT = "unparsed-endpoint"


def _host_of(base_url: str) -> str:
    """The host and port of `base_url`, for the audit identity.

    Never the path, never the query, never credentials. The PORT is included because dropping it
    is what leaves the collision this identity exists to close half open: vLLM on `localhost:8000`
    and LM Studio on `localhost:1234` serving the same model name would otherwise be one identity
    and one cache key, so the cache would serve one endpoint's answers for the other.

    A scheme-less value ("myhost.local/v1") yields no hostname at all. `urlsplit` is retried once
    with a `//` prefix, which is how a network-relative reference is spelled, and anything still
    unparseable becomes `UNPARSED_ENDPOINT` rather than the raw string.
    """
    # Both splits are guarded. `urlsplit` raises ValueError("Invalid IPv6 URL") on an unbalanced
    # bracket, and the `//` retry can raise for input the FIRST split accepted: with no netloc
    # there is no bracket to validate, and prefixing one creates it. A missing closing bracket on
    # a local IPv6 endpoint is an ordinary typo, and this function promises to be total.
    host = port = None
    # The `//` retry is for SCHEME-LESS input only. Applying it to a value that already has a
    # scheme re-reads the scheme as the host, so a malformed "http://[" would be recorded as the
    # endpoint "http" rather than as unidentified.
    candidates = (base_url,) if "://" in base_url else (base_url, f"//{base_url}")
    for candidate in candidates:
        try:
            parts = urlsplit(candidate)
        except ValueError:
            continue
        if parts.hostname is None:
            continue
        host = parts.hostname
        try:
            port = parts.port
        except ValueError:  # a non-numeric port; the host alone still identifies it safely
            port = None
        break
    if host is None:
        return UNPARSED_ENDPOINT
    # An IPv6 literal is re-bracketed before the port is joined. Without it `[::1]:8000` and
    # `[::1:8000]` both render as `::1:8000`, which is one identity for two endpoints: exactly
    # the collision this function exists to close.
    if ":" in host:
        host = f"[{host}]"
    return f"{host}:{port}" if port else host


def _text_of(reply: object) -> str:
    """The assistant text in a chat completion, or `""` when there is none.

    Deliberately total: every degenerate shape returns `""` rather than raising, because the
    ladder's `json` rung already refuses `""` and records it as a batch rejection a reviewer can
    see. Raising here would make this engine's failure mode differ from the rules engine's for
    the same unusable answer.

    The guards are not paranoia. OpenRouter returns HTTP 200 with an `{"error": ...}` body and no
    `choices` key when an upstream provider fails, which makes `reply.choices[0]` an unguarded
    IndexError on a live path, and that exception escapes `extract_file_claims`.

    Module level, and NOT inlined into the client closure, so a test can reach it. A guard only
    exercised through a test's own copy of it is a guard that cannot fail.
    """
    choices = getattr(reply, "choices", None) or ()
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    # Delegated, not inlined. Three OpenAI-compatible clients in this repo read this field and
    # each had its own rule; the copy that lived here raised `TypeError` on a dict block carrying
    # a non-str `text`, which made the "deliberately total" claim above false. Every remaining
    # shape becomes "" and meets the `json` rung, where every unusable answer is already handled.
    return assistant_text(content)


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
            'the openai extraction engine requires: pip install "recall-rag[extract]"'
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
    # Named refusal, matching every other setting on this path. `60s` and `1m` are the natural
    # things to write, and a bare "could not convert string to float" names neither the variable
    # that is wrong nor the form that is right.
    raw_timeout = _setting(source, "RECALL_EXTRACTION_TIMEOUT", "60")
    try:
        timeout = float(raw_timeout)
    except ValueError:
        raise ValueError(
            f"RECALL_EXTRACTION_TIMEOUT={raw_timeout!r} is not a number of seconds"
        ) from None
    if timeout <= 0:
        raise ValueError(
            f"RECALL_EXTRACTION_TIMEOUT={raw_timeout!r} must be greater than zero"
        )
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
            return _text_of(reply)

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
