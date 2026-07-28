from __future__ import annotations

import time
from collections.abc import Callable
from typing import Protocol

from recall.embeddings import retry_with_backoff

#: The injected-LLM seam: (system_prompt, user_prompt) -> completion text. Everything downstream
#: depends on this, not on any SDK, so the pipeline is testable with a plain function.
Completer = Callable[[str, str], str]

#: Output ceiling per request. Sized from measurement, not taste: the longest answerer completion
#: observed across the BEAM and pilot runs is 4,214 tokens (a multi-session plan, gpt-5 reasoning
#: tokens included), against a mean near 850. 16,384 is ~3.9x that observed maximum and still 4x
#: below the model maximum that omitting this reserves.
#:
#: Erring high is deliberate. The failure this constant prevents costs a refused request; the
#: failure it could CAUSE — truncating an answer that is then scored — corrupts a result, so the
#: headroom is bought on the side where being wrong is cheap.
#:
#: It is a CEILING, not a target — nothing is truncated silently, because `CompletionTruncated`
#: fires when a response stops for `length`. Raise it if that ever happens; do not remove it.
DEFAULT_MAX_TOKENS = 16384


class CompletionTruncated(RuntimeError):
    """A completion stopped because it hit `max_tokens` rather than finishing.

    Raised rather than returned, because the alternative is worse than the error: a truncated
    answer is a plausible-looking string that a judge — human or model — scores as if the system
    had produced it. That is a measurement error introduced by our own configuration, and it is
    indistinguishable from a genuine failure once it lands in a results artifact.

    Deliberately NOT transient: `retry_with_backoff` re-raises immediately instead of paying for
    the same over-long request three more times. The fix is a bigger ceiling, not another attempt.
    """


class LLM(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class OpenRouterLLM:
    """OpenAI-compatible chat client pointed at OpenRouter. Lazily imports the `openai` SDK so the
    module imports without the `bench` extra installed (construction is what tests exercise).

    Every call costs money, and a benchmark run makes thousands of them back to back — which is
    exactly the traffic shape a provider rate-limits. Without a retry, ONE 429 on question 900 of
    1,986 propagates out of `main` and the final `.json` is never written: the incremental sidecar
    survives, but the run has to be finished by hand. `complete` therefore retries transient
    failures (429/5xx/network) with exponential backoff and full jitter.

    The backoff is `recall.embeddings.retry_with_backoff`, the one this repo already ships and
    tests — reused rather than reimplemented, so there is one retry policy in the codebase and one
    definition of "transient" (`_is_transient`, which fails fast on a non-transient error such as
    a 401). `sleep` is injectable so the retry path is exercised offline, at no wall-clock cost.

    **`max_tokens` is set, and that is not a micro-optimisation.** Omitting it reserves the model's
    maximum — 65,536 on gpt-5 — and providers bill availability against that RESERVATION, not
    against what the call returns. A BEAM run died mid-arm on
    ``402 … You requested up to 65536 tokens, but can only afford 64714`` while its answers were
    measuring ~850 completion tokens: the request was refused over a ceiling it would never have
    approached, with roughly 75x the headroom it needed.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        temperature: float = 0.0,
        max_tokens: int | None = DEFAULT_MAX_TOKENS,
        max_attempts: int = 4,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_attempts = max_attempts
        self._api_key = api_key
        self._sleep = sleep
        self._client: object | None = None
        #: The benchmark's OWN generator+judge usage (this instance drives both). Recorded as the
        #: `harness` baseline so the memory layer's cost can be isolated as total - harness.
        self._usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def usage(self) -> dict[str, int]:
        return dict(self._usage)

    def complete(self, system: str, user: str) -> str:
        def _once() -> str:
            return self._complete_once(system, user)

        return retry_with_backoff(_once, attempts=self.max_attempts, sleep=self._sleep)

    def _complete_once(self, system: str, user: str) -> str:
        from openai import OpenAI  # lazy: only needed at real run time

        if self._client is None:
            self._client = OpenAI(api_key=self._api_key, base_url=self.base_url)
        extra = {} if self.max_tokens is None else {"max_tokens": self.max_tokens}
        resp = self._client.chat.completions.create(  # type: ignore[union-attr]
            model=self.model,
            temperature=self.temperature,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            **extra,
        )
        resp_usage = getattr(resp, "usage", None)
        if resp_usage is not None:
            self._usage["calls"] += 1
            self._usage["prompt_tokens"] += int(getattr(resp_usage, "prompt_tokens", 0) or 0)
            self._usage["completion_tokens"] += int(getattr(resp_usage, "completion_tokens", 0) or 0)
        # A response that stopped for `length` is truncated. Fail loudly: scoring a half-written
        # answer would charge our own ceiling to the system under test.
        if getattr(resp.choices[0], "finish_reason", None) == "length":
            raise CompletionTruncated(
                f"completion hit max_tokens={self.max_tokens} and was cut off. Raise max_tokens "
                f"(benchmarks.llm.DEFAULT_MAX_TOKENS) — do NOT score this answer."
            )
        content = resp.choices[0].message.content
        return content or ""
