from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

#: The injected-LLM seam: (system_prompt, user_prompt) -> completion text. Everything downstream
#: depends on this, not on any SDK, so the pipeline is testable with a plain function.
Completer = Callable[[str, str], str]


class LLM(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class OpenRouterLLM:
    """OpenAI-compatible chat client pointed at OpenRouter. Lazily imports the `openai` SDK so the
    module imports without the `bench` extra installed (construction is what tests exercise)."""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        temperature: float = 0.0,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self._api_key = api_key
        self._client: object | None = None

    def complete(self, system: str, user: str) -> str:
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
