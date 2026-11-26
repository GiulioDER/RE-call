"""The compile call: an OpenAI-compatible chat request per source, and its failure table.

Its own small client rather than the extraction client, because a budget needs the token usage of
each reply and that client returns text only. Every attempt is reserved against the budget before
it is sent and charged after, because each one may be billed. A 200 reply is never sent again: it
was generated, so it was paid for. Errors never carry the response body, which a provider may echo
the request's key into.
"""

from __future__ import annotations

import http.client
import json
import math
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib import error, request
from urllib.parse import urlsplit, urlunsplit

from recall.compiled_graph.anchors import encode_stored_data
from recall.compiled_graph.records import COMPILE_SYSTEM_PROMPT

if TYPE_CHECKING:
    from recall.compiled_graph.budget import Budget

ENV_PREFIX = "RECALL_COMPILED_GRAPH"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_TIMEOUT_SECONDS = 60.0
#: The AML compiler's output cap (``DEFAULT_COMPILER_MAX_TOKENS``); eight records fit inside it.
MAX_OUTPUT_TOKENS = 2_400
#: Tokens a chat template adds around the two messages, which the request bytes do not hold.
TEMPLATE_SLACK_TOKENS = 64
ATTEMPTS = 3
#: The AML compiler's clamp on a provider's ``Retry-After``, so one source cannot stall a build.
RETRY_AFTER_CAP_SECONDS = 10.0
_RETRIED_STATUSES = frozenset({408, 409, 429})
_CREDIT_STATUSES = frozenset({401, 402})
#: Bodies that name a spend or key limit. Not applied to 429, whose bodies say "rate limit".
_CREDIT_BODY = re.compile(
    r"insufficient[_ ]quota|credit|key limit|spending limit|payment required", re.IGNORECASE
)
_SIZE_BODY = re.compile(
    r"context length|context window|too large|too many tokens|maximum context|reduce the length",
    re.IGNORECASE,
)
_FENCED = re.compile(r"^\s*```[a-zA-Z]*\s*\n(.*)\n\s*```\s*$", re.DOTALL)

#: The request fields besides model, cap and messages; part of the compile cache key.
REQUEST_OPTIONS: Mapping[str, Any] = {"temperature": 0, "response_format": {"type": "json_object"}}
#: The user message around the encoded source; part of the compile cache key.
USER_TEMPLATE = "<stored_data>{}</stored_data>"

#: ``(url, body, headers, timeout) -> (status, headers, body)``. Every HTTP status comes back as a
#: value, an error status included; ``OSError`` or ``HTTPException`` means no status arrived (the
#: attempt is retried), and ``ReplyLost`` means a success status arrived without its body.
Transport = Callable[[str, bytes, Mapping[str, str], float], tuple[int, Mapping[str, str], bytes]]


class CompiledGraphConfigError(ValueError):
    """A required ``RECALL_COMPILED_GRAPH_*`` variable is missing or invalid."""


class CreditStop(RuntimeError):
    """The provider refused for credit or authorization: the whole build stops."""


class BudgetExhausted(RuntimeError):
    """The next attempt's worst case does not fit what is left: the whole build stops."""


class SourceRejected(RuntimeError):
    """This source gets no records; the build goes on with the next one."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class PayloadTooLarge(RuntimeError):
    """The provider refused the request's size; the caller may refit it once."""


class ReplyLost(RuntimeError):
    """A transport received a success status but lost the body: the reply was billed."""

    def __init__(self, status: int) -> None:
        super().__init__(f"HTTP {status} body was lost after the status arrived")
        self.status = status


def required_env(env: Mapping[str, str], name: str) -> str:
    variable = f"{ENV_PREFIX}_{name}"
    value = env.get(variable, "").strip()
    if not value:
        raise CompiledGraphConfigError(f"{variable} is required for a compile")
    return value


def url_without_credentials(url: str) -> str:
    """Scheme, host, port and path of ``url``: userinfo, query and fragment are dropped."""
    parts = urlsplit(url)
    host = parts.hostname or ""
    netloc = f"{host}:{parts.port}" if parts.port else host
    return urlunsplit((parts.scheme, netloc, parts.path, "", ""))


@dataclass(frozen=True)
class ProviderConfig:
    api_key: str
    model: str
    base_url: str = DEFAULT_BASE_URL
    timeout: float = DEFAULT_TIMEOUT_SECONDS

    def __repr__(self) -> str:
        return f"ProviderConfig(model={self.model!r}, base_url={url_without_credentials(self.base_url)!r})"

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> ProviderConfig:
        api_key, model = required_env(env, "API_KEY"), required_env(env, "MODEL")
        timeout_text = env.get(f"{ENV_PREFIX}_TIMEOUT", "").strip()
        try:
            timeout = float(timeout_text) if timeout_text else DEFAULT_TIMEOUT_SECONDS
        except ValueError as exc:
            raise CompiledGraphConfigError(f"{ENV_PREFIX}_TIMEOUT must be a number") from exc
        # A socket refuses an infinite timeout with OverflowError, after the attempt is reserved.
        if not (math.isfinite(timeout) and timeout > 0):
            raise CompiledGraphConfigError(f"{ENV_PREFIX}_TIMEOUT must be a finite number above zero")
        base_url = env.get(f"{ENV_PREFIX}_BASE_URL", "").strip() or DEFAULT_BASE_URL
        return cls(api_key=api_key, model=model, base_url=base_url, timeout=timeout)


@dataclass(frozen=True)
class CompileReply:
    content: dict[str, Any]
    input_tokens: int | None
    output_tokens: int | None


class _RefuseRedirect(request.HTTPRedirectHandler):
    # urllib re-sends every header but the content ones to a redirect's target, the bearer key
    # included, to any host and over plain http; a refused redirect reaches the caller as its 3xx.
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


_OPENER = request.build_opener(_RefuseRedirect)


def urllib_transport(
    url: str, body: bytes, headers: Mapping[str, str], timeout: float
) -> tuple[int, Mapping[str, str], bytes]:
    req = request.Request(url, data=body, headers=dict(headers), method="POST")
    try:
        with _OPENER.open(req, timeout=timeout) as response:
            status, response_headers = response.status, dict(response.headers)
            try:
                return status, response_headers, response.read()
            except (OSError, http.client.HTTPException) as exc:
                raise ReplyLost(status) from exc
    except error.HTTPError as exc:
        return exc.code, dict(exc.headers or {}), exc.read()


def _retry_delay(headers: Mapping[str, str], attempt: int) -> float:
    for name, value in headers.items():
        if name.lower() == "retry-after":
            try:
                seconds = float(value)
            except ValueError:
                break
            # NaN passes min and max unchanged, and time.sleep refuses it.
            if math.isnan(seconds):
                break
            return min(max(seconds, 0.0), RETRY_AFTER_CAP_SECONDS)
    return min(float(2 ** (attempt - 1)), RETRY_AFTER_CAP_SECONDS)


def _usage(reply: Mapping[str, Any], key: str) -> int | None:
    usage = reply.get("usage")
    value = usage.get(key) if isinstance(usage, Mapping) else None
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _error_status(error_object: Mapping[str, Any]) -> int | None:
    code = error_object.get("code")
    return code if isinstance(code, int) and not isinstance(code, bool) else None


def _parse_ok(text: str) -> CompileReply:
    """Read a 200 reply; raises CreditStop or SourceRejected, never asks for a resend."""
    try:
        reply = json.loads(text)
    # json raises RecursionError, not ValueError, for nesting deeper than it can read.
    except (ValueError, RecursionError) as exc:
        raise SourceRejected("reply body was not JSON", 200) from exc
    if not isinstance(reply, dict):
        raise SourceRejected("reply body was not a JSON object", 200)
    error_object = reply.get("error")
    if isinstance(error_object, Mapping):
        status = _error_status(error_object)
        message = error_object.get("message")
        if status in _CREDIT_STATUSES or (
            isinstance(message, str) and _CREDIT_BODY.search(message)
        ):
            raise CreditStop(f"compile provider reported a credit or authorization error ({status})")
        raise SourceRejected("compile provider reported an error inside a 200 reply", status)
    try:
        choice = reply["choices"][0]
        if choice.get("finish_reason") == "length":
            raise SourceRejected("compile output was cut off at the output token cap", 200)
        content_text = choice["message"]["content"]
        if not isinstance(content_text, str):
            raise SourceRejected("reply carried no text content", 200)
        fenced = _FENCED.match(content_text)
        content = json.loads(fenced.group(1) if fenced else content_text)
    except SourceRejected:
        raise
    except (ValueError, RecursionError, KeyError, IndexError, TypeError, AttributeError) as exc:
        raise SourceRejected("reply content was not a JSON object", 200) from exc
    if not isinstance(content, dict):
        raise SourceRejected("reply content was not a JSON object", 200)
    return CompileReply(
        content=content,
        input_tokens=_usage(reply, "prompt_tokens"),
        output_tokens=_usage(reply, "completion_tokens"),
    )


def compile_source(
    config: ProviderConfig,
    transport: Transport,
    payload: Mapping[str, Any],
    budget: Budget,
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> CompileReply:
    """Send one compile within ``budget``; return the parsed JSON object and its usage.

    Every attempt is reserved before it is sent and settled once, at its whole reservation unless
    a 200 reports usage, on every way out, an unexpected exception included.

    Raises:
        BudgetExhausted: the next attempt's worst case does not fit; the build stops.
        CreditStop: 401, 402, or a credit or key-limit refusal (also inside a 200); the build stops.
        PayloadTooLarge: 413, or a 400 naming the request size; the caller may refit once.
        SourceRejected: any other refusal, a 200 that cannot be read (never resent), or
            ``ATTEMPTS`` failed attempts of 408, 409, 429, 5xx or a network error.
    """
    body = json.dumps(
        {
            "model": config.model,
            **REQUEST_OPTIONS,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "messages": [
                {"role": "system", "content": COMPILE_SYSTEM_PROMPT},
                {"role": "user", "content": USER_TEMPLATE.format(encode_stored_data(payload))},
            ],
        }
    ).encode("utf-8")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {config.api_key}"}
    url = f"{config.base_url.rstrip('/')}/chat/completions"
    reason = "no attempt made"
    for attempt in range(1, ATTEMPTS + 1):
        if not budget.reserve(len(body) + TEMPLATE_SLACK_TOKENS):
            raise BudgetExhausted("the next compile attempt does not fit the remaining budget")
        try:
            status, response_headers, raw = transport(url, body, headers, config.timeout)
        except ReplyLost as exc:
            budget.settle(None)
            raise SourceRejected("the reply body was lost; a billed reply is not sent again",
                                 exc.status) from exc
        except (OSError, http.client.HTTPException) as exc:
            budget.settle(None)
            reason = f"network error {type(exc).__name__}"
            if attempt < ATTEMPTS:
                sleep(_retry_delay({}, attempt))
            continue
        except BaseException:
            budget.settle(None)
            raise
        if status == 200:
            try:
                reply = _parse_ok(raw.decode("utf-8", errors="replace"))
            except BaseException:
                budget.settle(None)
                raise
            budget.settle(reply)
            return reply
        budget.settle(None)
        text = raw.decode("utf-8", errors="replace")
        if status in _CREDIT_STATUSES or (
            400 <= status < 500 and status != 429 and _CREDIT_BODY.search(text)
        ):
            raise CreditStop(f"compile provider refused with HTTP {status} (credit or authorization)")
        if status == 413 or (status == 400 and _SIZE_BODY.search(text)):
            raise PayloadTooLarge(f"compile provider refused the request size with HTTP {status}")
        if status in _RETRIED_STATUSES or status >= 500:
            reason = f"HTTP {status}"
            if attempt < ATTEMPTS:
                sleep(_retry_delay(response_headers, attempt))
            continue
        raise SourceRejected(f"compile provider refused with HTTP {status}", status)
    raise SourceRejected(f"compile failed after {ATTEMPTS} attempts: {reason}")
