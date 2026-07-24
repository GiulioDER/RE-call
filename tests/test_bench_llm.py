from benchmarks.llm import Completer, OpenRouterLLM


def test_completer_is_callable_alias() -> None:
    # a plain function satisfies the injected-LLM seam used everywhere downstream
    fn: Completer = lambda system, user: "ok"  # noqa: E731
    assert fn("s", "u") == "ok"


def test_openrouter_llm_builds_with_defaults() -> None:
    llm = OpenRouterLLM(model="openai/gpt-4o-mini", api_key="sk-test")
    assert llm.model == "openai/gpt-4o-mini"
    assert llm.base_url == "https://openrouter.ai/api/v1"
    assert llm.temperature == 0.0
