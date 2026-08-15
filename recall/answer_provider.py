"""A shipped implementation of the answer port, and the reason there was not one.

`recall.reasoning` has declared `ReasoningAnswerProvider` since the reasoning layer landed, and
`reason()` short-circuits to `outcome="abstained", refusal_reason="no_answer_provider"` when the
port is empty. It has been empty in every shipped configuration: the only implementations were
test lambdas, so the generation leg of this library has never run outside a test. `recall setup`
compounds that by writing `RECALL_REASONING`, `RECALL_REASONING_MODEL`,
`RECALL_REASONING_BASE_URL` and `RECALL_REASONING_API_KEY`, four settings nothing reads. This
module is what reads them.

**It ships OFF**, matching `resolve_extraction_engine` and `resolve_entailment_judge`:
`resolve_answer_provider` returns `None` unless `RECALL_REASONING` is explicitly truthy, and a
`None` port leaves `reason()` abstaining exactly as it does today. Nothing about installing this
module puts a model on anyone's query path.

**What this module is not.** It holds no prompt. The system message is `recall.evidence`'s
`SYSTEM_PROMPT`, rendered by `render_evidence_prompt`, and this provider never sees a corpus byte
except inside the user message it is handed. Answer POLICY, when it exists, belongs in its own
module; putting a second prompt here would give the instruction channel two owners.

⚠️ **`max_retries=0` and `retry_with_backoff`**, exactly as `recall/truth_extraction/_openai_engine.py`
argues at length: the SDK default is 2 retries against a long read timeout, and layering a second
policy on top multiplies the two.
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Mapping
from typing import Protocol

from recall.embeddings import NonTransientError
from recall.provider_metadata import ProviderMetadata

#: The identity recorded in artifacts and audit records. A score whose provider is not named is
#: attributable to nothing, which the truth-extraction arms had to learn twice.
ANSWER_PROVIDER_ID = "recall.answer.openai"

DEFAULT_ANSWER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_ANSWER_MODEL = "openai/gpt-4o"

#: Completion ceiling. Generous on purpose: the envelope wraps the prose in
#: `{answer, citations, insufficient_evidence}`, so this path needs MORE completion budget than
#: plain prose, not less. An earlier value of 1024 would have truncated long answers into
#: unterminated JSON, which `parse_answer_envelope` rejects, which a benchmark would then have
#: recorded as an empty answer and a judge would have scored as wrong. Set rather than omitted,
#: because omitting reserves the model's maximum and providers bill the RESERVATION.
DEFAULT_ANSWER_MAX_TOKENS = 16384

#: Values `RECALL_REASONING` may take. Anything else is a REFUSAL rather than a silent off,
#: matching `resolve_extraction_engine` and `resolve_entailment_judge`. A typo that reads as
#: "off" leaves `reason()` abstaining with `no_answer_provider` forever and never says why.
_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off", ""})


class AnswerTruncated(NonTransientError, ValueError):
    """The model hit its completion ceiling, so the reply is not a whole answer.

    Distinguished from a malformed reply on purpose. `benchmarks/llm.py` learned this the hard
    way: a truncated completion was written into a submission and judged as if the system had
    produced it. A cut-off envelope is not an answer the model gave, and must not be scored.

    `NonTransientError` is load bearing, not decoration. `_is_transient` matches substrings of
    the rendered text as a last resort, and a ceiling spelled `4290` or `1429` contains "429";
    without the marker this would be retried three times at full prompt price for a failure that
    the same ceiling guarantees will repeat. `benchmarks/llm.py:CompletionTruncated` is the same
    class carrying the same marker for the same measured reason.
    """

#: A fenced reply, which is the single most common way an otherwise correct envelope fails to
#: parse. Stripped HERE rather than in `recall.evidence`, for two reasons: `parse_answer_envelope`
#: is a strict boundary whose exactness is the point, and `render_evidence_prompt`'s body is
#: pinned by `tests/test_evidence_contract.py` as source text, so that module does not move.
#:
#: Only a WELL FORMED wrapper is removed, and only when it encloses the entire reply. Anything
#: else passes through untouched and fails loudly at the envelope parser, because a de-framer
#: that starts guessing is a lenient parser wearing a transport costume.
_FENCE = re.compile(r"\A\s*```(?:json|JSON)?[ \t]*\r?\n(?P<body>.*?)\r?\n?```\s*\Z", re.DOTALL)

#: A line that is nothing but a fence delimiter. See `strip_json_fence`.
_BARE_FENCE_LINE = re.compile(r"(?:`{3,}|~{3,})\s*[A-Za-z]*")


class ChatClient(Protocol):
    """The one call this module makes. Narrow on purpose, so a fake is three lines."""

    def complete(self, messages: list[dict[str, str]], **kwargs: object) -> str: ...


def strip_json_fence(text: str) -> str:
    """Return `text` with one enclosing markdown fence removed, or `text` unchanged.

    ⚠️ A body containing a bare fence LINE is declined. `.*?` anchored at `\\Z` spans from the
    first opening delimiter to the LAST closing one, so two concatenated blocks would otherwise
    be merged into one body with the inner delimiters still in it. That still fails at
    `parse_answer_envelope`, so nothing was ever mis-scored, but this function's contract says it
    removes a WELL FORMED wrapper and only a well formed wrapper, and merging two blocks is not
    that. Checked on whole lines, so a fence appearing INSIDE a JSON string value survives, which
    is the case the `\\Z` anchor exists to protect.
    """
    match = _FENCE.match(text)
    if match is None:
        return text
    body = match.group("body")
    if any(_BARE_FENCE_LINE.fullmatch(line.strip()) for line in body.splitlines()):
        return text
    return body


class OpenAIAnswerProvider:
    """An OpenAI-compatible chat model behind the `ReasoningAnswerProvider` callable.

    Callable as `(system, user) -> str`. The string is handed to `parse_answer_envelope`
    unmodified apart from fence removal, so every structural rule the evidence boundary enforces
    still applies: exact envelope keys, a citation for every answer, and every citation resolving
    to a chunk id in the bundle. A model gains no ability to skip a rung by being a model, which
    is the same posture the extraction ladder takes.
    """

    def __init__(
        self,
        client: ChatClient,
        *,
        model_id: str = DEFAULT_ANSWER_MODEL,
        revision: str = "unpinned",
        base_url: str = DEFAULT_ANSWER_BASE_URL,
        temperature: float = 0.0,
        max_tokens: int | None = DEFAULT_ANSWER_MAX_TOKENS,
        request_json_object: bool = False,
    ) -> None:
        self.client = client
        self.model_id = model_id
        self.revision = revision
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.request_json_object = request_json_object
        self._last_latency_ms: int | None = None

    def __call__(self, system: str, user: str) -> str:
        kwargs: dict[str, object] = {"temperature": self.temperature}
        if self.max_tokens is not None:
            # Set, never omitted. Omitting reserves the model's maximum and providers bill
            # against the RESERVATION: `benchmarks/llm.py` records a run killed by a 402 over a
            # 65,536 ceiling while its answers measured ~850 completion tokens.
            kwargs["max_tokens"] = self.max_tokens
        if self.request_json_object:
            # ⚠️ OFF by default, and it cannot simply be turned on. OpenAI refuses
            # `response_format={"type": "json_object"}` with a 400 unless the word "json" appears
            # somewhere in the messages, and `SYSTEM_PROMPT` says "Return only an object matching
            # the requested answer envelope" without ever using it. That prompt is frozen: its
            # body is pinned as SOURCE TEXT by `tests/test_evidence_contract.py`, and editing it
            # to satisfy a provider flag would move every arm's score for a transport reason.
            #
            # Nothing is lost by leaving this off. `parse_answer_envelope` is the actual
            # guarantee, and it is stricter than JSON mode: exact key set, no extra fields, no
            # coercion. JSON mode would only have saved the occasional fenced reply, which
            # `strip_json_fence` already handles. Kept as a flag for a caller whose own prompt
            # does say "json".
            kwargs["response_format"] = {"type": "json_object"}
        started = time.monotonic()
        reply = self.client.complete(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            **kwargs,
        )
        self._last_latency_ms = int((time.monotonic() - started) * 1000)
        return strip_json_fence(reply)

    def provider_metadata(self) -> ProviderMetadata:
        """Best effort identity for the audit record. Token counts are not available here.

        The model and its revision are the load-bearing half: every comparison in this repository
        has to pin the model and say so in the artifact, after two committed summaries left their
        configuration recoverable only from a filename.
        """
        return ProviderMetadata(
            provider_id=ANSWER_PROVIDER_ID,
            model_id=self.model_id,
            model_revision=self.revision,
            latency_ms=self._last_latency_ms,
        )


def _setting(source: Mapping[str, str], name: str, default: str) -> str:
    return source.get(name, "").strip() or default


def _client_from_env(source: Mapping[str, str]) -> ChatClient:
    """Build the HTTP client, naming the install command when the extra is absent."""
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            'the openai answer provider requires: pip install "recall-rag[extract]"'
        ) from exc

    # Reused rather than reimplemented, so one `_is_transient` classifier covers every network
    # model call in the library. Imported here rather than at module scope only to match
    # `_openai_engine.py`; the marker class above is imported at module scope because a base
    # class is needed at class-definition time, and that costs nothing measurable: at module
    # scope `recall.embeddings` pulls no heavy dependency and is already on the `recall` import
    # chain, so its marginal import time is 0 ms.
    from recall.embeddings import retry_with_backoff

    key = source.get("RECALL_REASONING_API_KEY", "").strip()
    if not key:
        raise ValueError("RECALL_REASONING_API_KEY is required for the openai answer provider")
    base_url = _setting(source, "RECALL_REASONING_BASE_URL", DEFAULT_ANSWER_BASE_URL)
    model = _setting(source, "RECALL_REASONING_MODEL", DEFAULT_ANSWER_MODEL)
    raw_timeout = _setting(source, "RECALL_REASONING_TIMEOUT", "60")
    try:
        timeout = float(raw_timeout)
    except ValueError:
        raise ValueError(
            f"RECALL_REASONING_TIMEOUT={raw_timeout!r} is not a number of seconds"
        ) from None
    if timeout <= 0:
        raise ValueError(f"RECALL_REASONING_TIMEOUT={raw_timeout!r} must be greater than zero")

    inner = OpenAI(api_key=key, base_url=base_url, timeout=timeout, max_retries=0)

    class _Client:
        def complete(self, messages: list[dict[str, str]], **kwargs: object) -> str:
            def _once() -> str:
                # ⚠️ INSIDE the retried callable, deliberately. An empty `choices` list on a 200
                # is an upstream routing fault that `benchmarks/llm.py` documents and retries;
                # validating after `retry_with_backoff` returned would make the one shape most
                # worth retrying the one shape never retried, and the bare error would then
                # escape a 500 question run and lose the manifest with it.
                reply = inner.chat.completions.create(model=model, messages=messages, **kwargs)
                choices = getattr(reply, "choices", None) or []
                if not choices:
                    raise ConnectionError(f"{model} returned no choices")
                finish = getattr(choices[0], "finish_reason", None)
                content = getattr(getattr(choices[0], "message", None), "content", None)
                if finish == "length":
                    # NOT retried: the same ceiling cuts every further attempt. Raised so the
                    # caller can count it apart from a malformed reply, because a cut-off
                    # envelope is not an answer the model gave.
                    raise AnswerTruncated(
                        f"{model} hit its completion ceiling; the reply is not a whole answer"
                    )
                if not isinstance(content, str) or not content.strip():
                    raise ConnectionError(f"{model} returned an empty message")
                return content

            return retry_with_backoff(_once, attempts=3)

    return _Client()


def answer_provider_from_env(env: Mapping[str, str] | None = None) -> OpenAIAnswerProvider:
    """Construct the provider from `env`, defaulting to the process environment.

    `env` is honoured rather than ignored, for the reason `openai_engine_from_env` states: a
    provider that read the ambient environment anyway would answer for a DIFFERENT model than the
    one it was asked for, and then record that other model's name in the artifact.
    """
    source = env if env is not None else os.environ
    return OpenAIAnswerProvider(
        client=_client_from_env(source),
        model_id=_setting(source, "RECALL_REASONING_MODEL", DEFAULT_ANSWER_MODEL),
        revision=_setting(source, "RECALL_REASONING_REVISION", "unpinned"),
        base_url=_setting(source, "RECALL_REASONING_BASE_URL", DEFAULT_ANSWER_BASE_URL),
    )


def resolve_answer_provider(
    env: Mapping[str, str] | None = None,
) -> OpenAIAnswerProvider | None:
    """The provider this configuration asks for, or `None`, which is the default.

    OFF unless `RECALL_REASONING` is explicitly truthy. `None` is not a degraded provider: it is
    the port staying empty, which leaves `reason()` abstaining with `no_answer_provider` exactly
    as it does today. Enabling generation is a decision someone makes, never a side effect of
    having an API key in the environment.
    """
    source = env if env is not None else os.environ
    raw = _setting(source, "RECALL_REASONING", "0").lower()
    if raw in _FALSE:
        return None
    if raw not in _TRUE:
        # Refused, not read as off. A typo that silently means "off" leaves `reason()` abstaining
        # with `no_answer_provider` on every query and nothing anywhere says why.
        raise ValueError(
            f"RECALL_REASONING={raw!r} is neither true {sorted(_TRUE)} nor false {sorted(_FALSE)}"
        )
    return answer_provider_from_env(source)


__all__ = [
    "ANSWER_PROVIDER_ID",
    "DEFAULT_ANSWER_BASE_URL",
    "DEFAULT_ANSWER_MAX_TOKENS",
    "DEFAULT_ANSWER_MODEL",
    "AnswerTruncated",
    "ChatClient",
    "OpenAIAnswerProvider",
    "answer_provider_from_env",
    "resolve_answer_provider",
    "strip_json_fence",
]
