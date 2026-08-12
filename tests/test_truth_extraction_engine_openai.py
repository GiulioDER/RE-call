"""The model engine is one implementation of the existing port, held to the same ladder.

`_engine.py` defined extraction as a PORT: the whole contract is `run(prompt) -> str`, and
whatever comes back goes through `_normalize.normalize_extraction`. So a model engine is a new
entry in `_ENGINES`, not a new architecture, and it cannot skip a rung the rules engine clears.

Properties, one test each:

1. The engine returns the model's text unchanged; interpreting it is the ladder's job.
2. It calls the model at temperature 0.
3. It sends the prompt's rendered `system` and `user` strings, which is what the prompt object
   actually carries (it has no `messages` field).
4. Naming an unknown engine is refused, never downgraded to rules, because silently running a
   different engine makes the audit record wrong about how a claim was produced.
5. The openai engine is selectable by name.
6. Importing the package does NOT import `openai`: the extra is pulled only when chosen.
7. A model answer still faces the ladder, so a non-verbatim quote is refused.
"""
import subprocess
import sys
from types import SimpleNamespace

import pytest

from recall.truth_extraction._engine import (
    _ENGINES,
    DeterministicExtractionEngine,
    resolve_extraction_engine,
)
from recall.truth_extraction._normalize import normalize_extraction
from recall.truth_extraction._openai_engine import (
    DEFAULT_EXTRACTION_MODEL,
    UNPARSED_ENDPOINT,
    _host_of,
    _text_of,
)
from recall.truth_extraction._prompt import build_extraction_prompt


class _FakeChat:
    """Records the call so the test can assert on what the model was actually asked."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[dict] = []

    def complete(self, messages: list[dict[str, str]], **kwargs: object) -> str:
        self.calls.append({"messages": messages, **kwargs})
        return self.reply


def _engine(reply: str = '{"claims": []}'):
    from recall.truth_extraction._openai_engine import OpenAIExtractionEngine

    chat = _FakeChat(reply)
    return OpenAIExtractionEngine(client=chat, model_id="m", revision="r"), chat


def _prompt(body: str = "body"):
    return build_extraction_prompt(file="a.md", human_body=body, corpus_names=("a.md",))


def test_the_engine_returns_the_model_text_unchanged():
    engine, _ = _engine('{"claims": []}')
    assert engine.run(_prompt()) == '{"claims": []}'


def test_the_engine_calls_the_model_at_temperature_zero():
    engine, chat = _engine()
    engine.run(_prompt())
    assert chat.calls[0]["temperature"] == 0


def test_the_engine_sends_the_rendered_system_and_user_prompts():
    engine, chat = _engine()
    prompt = _prompt()
    engine.run(prompt)
    messages = chat.calls[0]["messages"]
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == prompt.system
    assert messages[1]["content"] == prompt.user


def test_an_unknown_engine_name_is_refused_rather_than_downgraded():
    with pytest.raises(ValueError, match="not a known engine"):
        resolve_extraction_engine(
            {"RECALL_TRUTH_EXTRACTION": "1", "RECALL_TRUTH_EXTRACTION_ENGINE": "gpt9"}
        )


def test_the_openai_engine_is_selectable_by_name():
    assert "openai" in _ENGINES


def test_importing_the_package_does_not_import_openai():
    """The extra is optional. Naming the engine pulls it; importing the package must not.

    Run in a subprocess because `openai` may already be in this process's modules: it is a
    dependency of the `bench` extra, so an in-process check would pass for the wrong reason.
    """
    code = (
        "import sys, recall.truth_extraction as t;"
        "t.resolve_extraction_engine({'RECALL_TRUTH_EXTRACTION': '1'});"
        "print('openai' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False", "importing recall.truth_extraction pulled in openai"


def test_the_resolved_env_reaches_the_engine_rather_than_os_environ(monkeypatch):
    """`resolve_extraction_engine` takes an explicit mapping; the engine must honour IT.

    A nullary factory could only read `os.environ`, so a caller passing a complete env was
    refused for a key it had supplied, and when the ambient environment did name a model the
    engine answered for THAT model and recorded its name in the cache key and audit record.
    """
    monkeypatch.setenv("RECALL_EXTRACTION_API_KEY", "sk-ambient")
    monkeypatch.setenv("RECALL_EXTRACTION_MODEL", "ambient/model")
    engine = resolve_extraction_engine(
        {
            "RECALL_TRUTH_EXTRACTION": "1",
            "RECALL_TRUTH_EXTRACTION_ENGINE": "openai",
            "RECALL_EXTRACTION_API_KEY": "sk-caller",
            "RECALL_EXTRACTION_MODEL": "caller/model",
        }
    )
    assert engine is not None
    assert engine.model_id == "caller/model", "the ambient environment overrode the caller"


def test_the_endpoint_is_part_of_the_engine_identity():
    """Two endpoints advertising one model name must not share a cache key.

    `extraction_cache_key` hashes engine_id, model_id and revision. Without the host in one of
    them, a local server named the same model as the hosted default collides, and the cache
    serves one endpoint's answers for the other.
    """
    from recall.truth_extraction._cache import extraction_cache_key

    common = {
        "RECALL_TRUTH_EXTRACTION": "1",
        "RECALL_TRUTH_EXTRACTION_ENGINE": "openai",
        "RECALL_EXTRACTION_API_KEY": "k",
        "RECALL_EXTRACTION_MODEL": "same/model",
    }
    hosted = resolve_extraction_engine(common | {"RECALL_EXTRACTION_BASE_URL": "https://a.example/v1"})
    local = resolve_extraction_engine(common | {"RECALL_EXTRACTION_BASE_URL": "http://localhost:8000/v1"})
    assert hosted is not None and local is not None
    assert hosted.engine_id != local.engine_id
    prompt = _prompt()
    assert extraction_cache_key(engine=hosted, prompt=prompt) != extraction_cache_key(
        engine=local, prompt=prompt
    )


def test_a_variable_set_to_empty_falls_back_to_the_default():
    """`export RECALL_EXTRACTION_MODEL=` must behave like unset, not record model_id=''."""
    engine = resolve_extraction_engine(
        {
            "RECALL_TRUTH_EXTRACTION": "1",
            "RECALL_TRUTH_EXTRACTION_ENGINE": "openai",
            "RECALL_EXTRACTION_API_KEY": "k",
            "RECALL_EXTRACTION_MODEL": "",
            "RECALL_EXTRACTION_BASE_URL": "",
        }
    )
    assert engine is not None
    assert engine.model_id == DEFAULT_EXTRACTION_MODEL


@pytest.mark.parametrize(
    "reply",
    [
        SimpleNamespace(choices=[]),
        SimpleNamespace(choices=None),
        SimpleNamespace(),
        SimpleNamespace(choices=[SimpleNamespace(message=None)]),
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None))]),
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=[{"a": 1}]))]),
    ],
    ids=["no-choices", "null-choices", "no-choices-attr", "no-message", "null-content", "blocks"],
)
def test_a_degenerate_response_becomes_an_empty_answer_not_a_crash(reply):
    """OpenRouter returns HTTP 200 with an error body and NO choices when an upstream fails.

    `reply.choices[0]` would be an unguarded IndexError on a live path, and that exception
    escapes `extract_file_claims` and discards the whole corpus run.

    Calls the PRODUCTION `_text_of`. An earlier version of this test re-implemented the guard
    in its own fake client, which meant deleting the real guard left the suite green: a guard
    that cannot fail. Import the thing under test, never a copy of it.
    """
    assert _text_of(reply) == ""


def test_the_happy_path_still_returns_the_model_text():
    """The degenerate cases above must not be passing because `_text_of` always returns ""."""
    reply = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="hello"))])
    assert _text_of(reply) == "hello"


def test_the_client_is_built_with_an_explicit_timeout_and_no_sdk_retries(monkeypatch):
    """The SDK default is a 600s read timeout with 2 retries; ours must replace both.

    Nothing pinned this before, so the timeout and `max_retries=0` could be deleted silently.
    """
    import openai

    seen: dict = {}

    class _FakeOpenAI:
        def __init__(self, **kwargs):
            seen.update(kwargs)
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **k: None))

    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)
    resolve_extraction_engine(
        {
            "RECALL_TRUTH_EXTRACTION": "1",
            "RECALL_TRUTH_EXTRACTION_ENGINE": "openai",
            "RECALL_EXTRACTION_API_KEY": "k",
            "RECALL_EXTRACTION_TIMEOUT": "12.5",
        }
    )
    assert seen["timeout"] == 12.5
    assert seen["max_retries"] == 0, "the SDK's own retries would multiply with ours"


def test_a_transient_failure_is_retried_and_a_permanent_one_is_not(monkeypatch):
    import openai

    calls = {"n": 0}

    def _make(fail_with):
        def _create(**kwargs):
            calls["n"] += 1
            raise fail_with

        class _FakeOpenAI:
            def __init__(self, **kwargs):
                self.chat = SimpleNamespace(completions=SimpleNamespace(create=_create))

        return _FakeOpenAI

    env = {
        "RECALL_TRUTH_EXTRACTION": "1",
        "RECALL_TRUTH_EXTRACTION_ENGINE": "openai",
        "RECALL_EXTRACTION_API_KEY": "k",
    }
    monkeypatch.setattr("time.sleep", lambda _s: None)

    monkeypatch.setattr(openai, "OpenAI", _make(TimeoutError("timed out")))
    engine = resolve_extraction_engine(env)
    with pytest.raises(TimeoutError):
        engine.run(_prompt())
    assert calls["n"] == 3, "a transient failure must be retried"

    calls["n"] = 0
    monkeypatch.setattr(openai, "OpenAI", _make(PermissionError("401 unauthorized")))
    engine = resolve_extraction_engine(env)
    with pytest.raises(PermissionError):
        engine.run(_prompt())
    assert calls["n"] == 1, "a permanent failure must fail fast"


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        ("https://openrouter.ai/api/v1", "openrouter.ai"),
        ("http://localhost:8000/v1", "localhost:8000"),
        ("http://localhost:1234/v1", "localhost:1234"),
        ("myhost.local/v1", "myhost.local"),
        ("myproxy.local:8443/v1?token=sk-live-SECRET", "myproxy.local:8443"),
        ("https://user:sk-secret@host.example/v1", "host.example"),
    ],
)
def test_the_endpoint_identity_carries_host_and_port_and_nothing_else(base_url, expected):
    """Credentials, query and path must never reach the audit record or the cache key."""
    assert _host_of(base_url) == expected


def test_two_local_ports_are_two_identities():
    """Dropping the port leaves the collision this identity exists to close half open."""
    assert _host_of("http://localhost:8000/v1") != _host_of("http://localhost:1234/v1")


def test_an_unparseable_endpoint_is_recorded_as_unidentified_not_verbatim():
    assert _host_of("") == UNPARSED_ENDPOINT
    assert "SECRET" not in _host_of("myproxy.local:8443/v1?token=sk-live-SECRET")


def test_a_non_numeric_timeout_names_the_variable():
    with pytest.raises(ValueError, match="RECALL_EXTRACTION_TIMEOUT"):
        resolve_extraction_engine(
            {
                "RECALL_TRUTH_EXTRACTION": "1",
                "RECALL_TRUTH_EXTRACTION_ENGINE": "openai",
                "RECALL_EXTRACTION_API_KEY": "k",
                "RECALL_EXTRACTION_TIMEOUT": "60s",
            }
        )


def test_a_systemic_outage_stops_asking_once_per_file():
    """One failure is this memo's problem; every failure is the engine's.

    Without a break, a wedged endpoint makes all 792 memos pay the full retry and timeout
    budget to learn the same thing, which is over a day of ingest producing nothing.
    """
    from recall.truth_extraction.extract import (
        CONSECUTIVE_ENGINE_FAILURE_LIMIT,
        extract_corpus_claims,
    )

    class _Down(DeterministicExtractionEngine):
        def __init__(self) -> None:
            self.calls = 0

        def run(self, prompt) -> str:
            self.calls += 1
            raise ConnectionError("connection refused")

    engine = _Down()
    documents = {f"{i:02d}.md": "Body.\n" for i in range(20)}
    results = extract_corpus_claims(documents, engine=engine)

    assert len(results) == 20, "every file must still get a result"
    assert all(r.batch_rejection is not None for r in results)
    assert engine.calls == CONSECUTIVE_ENGINE_FAILURE_LIMIT, (
        f"the engine was asked {engine.calls} times; it should stop after "
        f"{CONSECUTIVE_ENGINE_FAILURE_LIMIT}"
    )
    assert "treated as unavailable" in results[-1].batch_rejection.reason


def test_an_isolated_failure_does_not_trip_the_break():
    """A corpus with a few awkward memos must not stop; only a run of failures should."""
    from recall.truth_extraction.extract import extract_corpus_claims

    class _Flaky(DeterministicExtractionEngine):
        def run(self, prompt) -> str:
            if prompt.file in {"02.md", "07.md", "13.md"}:
                raise ConnectionError("blip")
            return super().run(prompt)

    documents = {f"{i:02d}.md": "Body. This replaces other.md.\n" for i in range(20)}
    results = extract_corpus_claims(documents, engine=_Flaky())
    refused = [r for r in results if r.batch_rejection is not None]
    assert len(refused) == 3, "isolated failures tripped the systemic break"


def test_an_engine_failure_refuses_one_file_and_spares_the_rest():
    """One bad memo must not abort ingesting the other 791.

    `extract_corpus_claims` collects into a tuple over a generator, so an exception escaping
    one file discards every extraction already built in that run.
    """
    from recall.truth_extraction.extract import extract_corpus_claims

    class _Flaky(DeterministicExtractionEngine):
        def __init__(self) -> None:
            self.seen = 0

        def run(self, prompt) -> str:
            self.seen += 1
            if prompt.file == "2.md":
                raise ConnectionError("connection reset by peer")
            return super().run(prompt)

    documents = {f"{i}.md": f"Doc {i}. This replaces other_{i}.md.\n" for i in range(5)}
    results = extract_corpus_claims(documents, engine=_Flaky())

    assert len(results) == 5, "a single engine failure discarded the whole run"
    failed = [r for r in results if r.file == "2.md"][0]
    assert failed.batch_rejection is not None
    assert failed.batch_rejection.rung == "engine_error"
    # The class name, never the exception text: provider errors echo the request, and the
    # request carries the API key and the memo body.
    assert "ConnectionError" in failed.batch_rejection.reason
    assert "reset by peer" not in failed.batch_rejection.reason
    assert all(r.batch_rejection is None for r in results if r.file != "2.md")


def test_an_engine_failure_is_not_cached():
    """A rate limit is not an answer. Caching it makes a transient failure permanent."""
    from recall.truth_extraction.extract import extract_corpus_claims

    class _Cache:
        def __init__(self) -> None:
            self.entries: dict = {}

        def get(self, key):
            return self.entries.get(key)

        def put(self, key, value) -> None:
            self.entries[key] = value

    class _Down(DeterministicExtractionEngine):
        def run(self, prompt) -> str:
            raise TimeoutError("upstream timed out")

    cache = _Cache()
    extract_corpus_claims({"a.md": "Body.\n"}, engine=_Down(), cache=cache)
    assert cache.entries == {}, "a transient engine failure was written to the cache"


def test_a_model_answer_still_faces_the_ladder():
    """The point of the port: the model gains no ability to skip a rung.

    A quote that is not a verbatim substring of the body is what the ladder exists to catch,
    and it must be caught regardless of which engine produced it.
    """
    body = "The real body text."
    reply = (
        '{"claims": [{"kind": "status", "value": "deprecated",'
        ' "quote": "a line the body never contained"}]}'
    )
    engine, _ = _engine(reply)
    claims, rejections = normalize_extraction(
        engine.run(_prompt(body)), file="a.md", human_body=body, corpus_names=("a.md",)
    )
    assert claims == ()
    assert [r.rung for r in rejections] == ["quote_not_verbatim"]
