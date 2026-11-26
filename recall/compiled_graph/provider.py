"""The compile call: an OpenAI-compatible chat request per source, and its failure table.

Its own small client rather than the extraction client, because a budget needs the token usage of
each reply and that client returns text only. Errors never carry the response body, which a
provider may echo the request's key into.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any
from urllib import error, request

from recall.compiled_graph.anchors import encode_stored_data
from recall.compiled_graph.records import COMPILE_SYSTEM_PROMPT

ENV_PREFIX = "RECALL_COMPILED_GRAPH"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_TIMEOUT_SECONDS = 60.0
#: The AML compiler's output cap (``DEFAULT_COMPILER_MAX_TOKENS``); eight records fit inside it.
MAX_OUTPUT_TOKENS = 2_400
ATTEMPTS = 3
#: The AML compiler's clamp on a provider's ``Retry-After``, so one source cannot stall a build.
RETRY_AFTER_CAP_SECONDS = 10.0
_RETRIED_STATUSES = frozenset({408, 409, 429})
_CREDIT_STATUSES = frozenset({401, 402})
_CREDIT_BODY = re.compile(r"insufficient[_ ]quota|credit", re.IGNORECASE)
_SIZE_BODY = re.compile(
    r"context length|context window|too large|too many tokens|maximum context|reduce the length",
    re.IGNORECASE,
)

Transport = Callable[[str, bytes, Mapping[str, str], float], tuple[int, Mapping[str, str], bytes]]


class CompiledGraphConfigError(ValueError):
    """A required ``RECALL_COMPILED_GRAPH_*`` variable is missing or invalid."""


class CreditStop(RuntimeError):
    """The provider refused for credit or authorization: the whole build stops."""


class SourceRejected(RuntimeError):
    """This source gets no records; the build goes on with the next one."""


class PayloadTooLarge(RuntimeError):
    """The provider refused the request's size; the caller may refit it once."""


@dataclass(frozen=True)
class ProviderConfig:
    api_key: str
    model: str
    base_url: str = DEFAULT_BASE_URL
    timeout: float = DEFAULT_TIMEOUT_SECONDS

    def __repr__(self) -> str:
        return f"ProviderConfig(model={self.model!r}, base_url={self.base_url!r})"

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> ProviderConfig:
        def required(name: str) -> str:
            value = env.get(f"{ENV_PREFIX}_{name}", "").strip()
            if not value:
                raise CompiledGraphConfigError(f"{ENV_PREFIX}_{name} is required for a compile")
            return value

        api_key, model = required("API_KEY"), required("MODEL")
        timeout_text = env.get(f"{ENV_PREFIX}_TIMEOUT", "").strip()
        try:
            timeout = float(timeout_text) if timeout_text else DEFAULT_TIMEOUT_SECONDS
        except ValueError as exc:
            raise CompiledGraphConfigError(f"{ENV_PREFIX}_TIMEOUT must be a number") from exc
        if not timeout > 0:
            raise CompiledGraphConfigError(f"{ENV_PREFIX}_TIMEOUT must be positive")
        base_url = env.get(f"{ENV_PREFIX}_BASE_URL", "").strip() or DEFAULT_BASE_URL
        return cls(api_key=api_key, model=model, base_url=base_url, timeout=timeout)


@dataclass(frozen=True)
class CompileReply:
    content: dict[str, Any]
    input_tokens: int | None
    output_tokens: int | None


def urllib_transport(
    url: str, body: bytes, headers: Mapping[str, str], timeout: float
) -> tuple[int, Mapping[str, str], bytes]:
    req = request.Request(url, data=body, headers=dict(headers), method="POST")
    try:
        with request.urlopen(req, timeout=timeout) as response:
            return response.status, dict(response.headers), response.read()
    except error.HTTPError as exc:
        return exc.code, dict(exc.headers or {}), exc.read()


def _retry_delay(headers: Mapping[str, str], attempt: int) -> float:
    for name, value in headers.items():
        if name.lower() == "retry-after":
            try:
                return min(max(float(value), 0.0), RETRY_AFTER_CAP_SECONDS)
            except ValueError:
                break
    return min(float(2 ** (attempt - 1)), RETRY_AFTER_CAP_SECONDS)


def _usage(reply: Mapping[str, Any], key: str) -> int | None:
    usage = reply.get("usage")
    value = usage.get(key) if isinstance(usage, Mapping) else None
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def compile_source(
    config: ProviderConfig,
    transport: Transport,
    payload: Mapping[str, Any],
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> CompileReply:
    """Send one compile; return the parsed JSON object and the reported usage."""
    body = json.dumps(
        {
            "model": config.model,
            "temperature": 0,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": COMPILE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"<stored_data>{encode_stored_data(payload)}</stored_data>",
                },
            ],
        }
    ).encode("utf-8")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {config.api_key}"}
    url = f"{config.base_url.rstrip('/')}/chat/completions"
    reason = "no attempt made"
    for attempt in range(1, ATTEMPTS + 1):
        try:
            status, response_headers, raw = transport(url, body, headers, config.timeout)
        except OSError as exc:
            reason = f"network error {type(exc).__name__}"
            if attempt < ATTEMPTS:
                sleep(_retry_delay({}, attempt))
            continue
        text = raw.decode("utf-8", errors="replace")
        if status in _CREDIT_STATUSES or (400 <= status < 500 and _CREDIT_BODY.search(text)):
            raise CreditStop(f"compile provider refused with HTTP {status} (credit or authorization)")
        if status == 413 or (status == 400 and _SIZE_BODY.search(text)):
            raise PayloadTooLarge(f"compile provider refused the request size with HTTP {status}")
        if status in _RETRIED_STATUSES or status >= 500:
            reason = f"HTTP {status}"
            if attempt < ATTEMPTS:
                sleep(_retry_delay(response_headers, attempt))
            continue
        if status != 200:
            raise SourceRejected(f"compile provider refused with HTTP {status}")
        try:
            reply = json.loads(text)
            choice = reply["choices"][0]
            if choice.get("finish_reason") == "length":
                raise SourceRejected("compile output was cut off at the output token cap")
            content = json.loads(choice["message"]["content"])
        except SourceRejected:
            raise
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            reason = "reply was not a JSON object"
            continue
        if not isinstance(content, dict):
            reason = "reply was not a JSON object"
            continue
        return CompileReply(
            content=content,
            input_tokens=_usage(reply, "prompt_tokens"),
            output_tokens=_usage(reply, "completion_tokens"),
        )
    raise SourceRejected(f"compile failed after {ATTEMPTS} attempts: {reason}")
