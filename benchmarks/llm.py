from __future__ import annotations

import time
from collections.abc import Callable
from typing import Protocol

from recall.embeddings import retry_with_backoff

#: The injected-LLM seam: (system_prompt, user_prompt) -> completion text. Everything downstream
#: depends on this, not on any SDK, so the pipeline is testable with a plain function.
Completer = Callable[[str, str], str]


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
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        temperature: float = 0.0,
        max_attempts: int = 4,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_attempts = max_attempts
        self._api_key = api_key
        self._sleep = sleep
        self._client: object | None = None

    def complete(self, system: str, user: str) -> str:
        def _once() -> str:
            return self._complete_once(system, user)

        return retry_with_backoff(_once, attempts=self.max_attempts, sleep=self._sleep)

    def _complete_once(self, system: str, user: str) -> str:
        from openai import OpenAI  # lazy: only needed at real run time

        if self._client is None:
            self._client = OpenAI(api_key=self._api_key, base_url=self.base_url)
        resp = self._client.chat.completions.create(  # type: ignore[union-attr]
            model=self.model,
            temperature=self.temperature,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        content = resp.choices[0].message.content
        return content or ""
