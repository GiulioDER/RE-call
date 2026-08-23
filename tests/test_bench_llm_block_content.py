"""One wire shape, one reader, in all THREE OpenAI-compatible clients this repo ships.

`message.content` can arrive as a LIST of text blocks rather than a string. A gateway that does
that is returning a well formed answer in a shape the OpenAI schema permits, and the repo handled
it three different ways:

    recall/truth_extraction/_openai_engine.py:_text_of   joined the block text
    benchmarks/llm.py:_complete_once                     refused it, naming the type
    benchmarks/mtrag/generation.py:generate_one          crashed with AttributeError

`_text_of`'s reasoning was the right one and its comment already said why: discarding the answer
"would refuse a WELL FORMED reply and blame the model for it".

⚠️ **THE FIRST ATTEMPT AT THIS RECONCILIATION MIRRORED THE RULE INTO A SECOND MODULE AND PINNED
THE PAIR WITH A CROSS-CHECK TEST. THAT FAILED TWICE OVER, WHICH IS WHY THERE IS NOW ONE FUNCTION.**

  1. The cross-check table held only LISTS, so the `str` branch and the `""` fallback of both
     readers were unpinned. Deleting `if isinstance(content, str)` from `_text_of` left it green.
  2. It copied a latent crash. `... else getattr(block, "text", "") or ""` guards only the OBJECT
     branch, and only against FALSY values, so a dict block carrying `{"text": None}` or
     `{"text": 123}` reached `"".join` unconverted and raised `TypeError`. The table's "text is
     None" case used an OBJECT block, which the `or ""` rescues, so the dict branch was never
     exercised. Both readers agreed, on raising, and the test that existed to prove they agreed
     said nothing.
  3. And a THIRD reader existed that the reconciliation had not counted at all.

So the rule now lives once, in `recall/_chat_content.py`, and the tests below check that each site
CALLS it rather than that three copies happen to match today.

🔑 What an empty reading MEANS still differs per client, deliberately, and that is pinned too:
`_text_of` returns `""` because `_batch_rungs("")` refuses it as a batch rejection a reviewer sees;
the benchmark clients raise, because nothing downstream re-reads the answer and `""` was SCORED.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from benchmarks.llm import EmptyCompletion, OpenRouterLLM
from recall._chat_content import assistant_text
from recall.truth_extraction._openai_engine import _text_of

#: Benchmark-harness coverage, not product coverage; product CI can deselect with
#: `-m 'not benchharness'`.
pytestmark = pytest.mark.benchharness

#: Every shape the reader must handle, and the reading all three clients must agree on.
#:
#: ⛔ The last four are the ones the first version of this file missed. `dict text=None` and
#: `dict text=123` are the shapes that raised `TypeError`; the plain `str` entries pin the branch
#: a lists-only table left unguarded.
_SHAPES: list[tuple[str, Any, str]] = [
    ("a plain string", "an answer", "an answer"),
    ("a padded string is untouched", "  an answer  ", "  an answer  "),
    ("one text block", [{"type": "text", "text": "an answer"}], "an answer"),
    ("two blocks join in order",
     [{"type": "text", "text": "an "}, {"type": "text", "text": "answer"}], "an answer"),
    ("object blocks, not dicts", [types.SimpleNamespace(text="an answer")], "an answer"),
    ("a non-text block contributes nothing",
     [{"type": "image", "url": "x"}, {"type": "text", "text": "hi"}], "hi"),
    ("an object block whose text is None",
     [types.SimpleNamespace(text=None), {"type": "text", "text": "hi"}], "hi"),
    ("an empty list", [], ""),
    ("a list of nothing useful", [{"type": "image", "url": "x"}], ""),
    # The four that used to crash or were unpinned.
    ("a DICT block whose text is None", [{"type": "text", "text": None}], ""),
    ("a DICT block whose text is a number", [{"type": "text", "text": 123}], ""),
    ("an object block whose text is a number", [types.SimpleNamespace(text=123)], ""),
    ("content that is neither str nor list", {"text": "hi"}, ""),
    ("content is None", None, ""),
    ("blocks carrying only whitespace", [{"type": "text", "text": "   "}], "   "),
]


def _install_fake_openai(monkeypatch: pytest.MonkeyPatch, content: Any) -> None:
    """The SDK faked into `sys.modules`, following `test_bench_llm_max_tokens.py`.

    `openai` lives in the `bench` extra, which pyproject excludes from `dev`, so an importorskip
    here would make these guards silently absent in CI.
    """

    class _FakeCompletions:
        def create(self, **_: Any) -> types.SimpleNamespace:
            choice = types.SimpleNamespace(
                message=types.SimpleNamespace(content=content), finish_reason="stop"
            )
            return types.SimpleNamespace(choices=[choice], usage=None)

    class _FakeOpenAI:
        def __init__(self, **_: Any) -> None:
            self.chat = types.SimpleNamespace(completions=_FakeCompletions())

    module = types.ModuleType("openai")
    module.OpenAI = _FakeOpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", module)


def _reply(content: Any) -> object:
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content))]
    )


# --------------------------------------------------------------------------------------------
# 1. The shared reader.
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("label,content,expected", [pytest.param(*c, id=c[0]) for c in _SHAPES])
def test_the_shared_reader_reads_every_shape(label: str, content: Any, expected: str) -> None:
    assert assistant_text(content) == expected


@pytest.mark.parametrize("label,content,expected", [pytest.param(*c, id=c[0]) for c in _SHAPES])
def test_the_shared_reader_never_raises(label: str, content: Any, expected: str) -> None:
    """⚠️ The property the whole module exists for. A reader that raises does not merely misread:
    in `benchmarks/llm.py` a `TypeError` escapes past the `EmptyCompletion` guard as a bare Python
    error, and in mtrag an `AttributeError` is not in `PERMANENT_ERROR_NAMES`, so the task pays
    four BILLED attempts before failing."""
    assistant_text(content)  # must not raise


def test_the_shared_reader_survives_a_hostile_block() -> None:
    """⛔ This arm used to assert `pytest.raises(RuntimeError)` under a name promising the
    opposite, while the module docstring claimed the function MUST NEVER RAISE. The claim was
    false and the test pinned the falsity.

    Blocks come off the wire, so `get`, `text` and even iteration can be anything at all. A reader
    that raises escapes `benchmarks/llm.py` past the `EmptyCompletion` guard as a bare Python
    error, and in mtrag lands on a type outside `PERMANENT_ERROR_NAMES`, costing four billed
    attempts."""

    class _HostileDict(dict):
        def get(self, *a: Any, **k: Any) -> Any:
            raise RuntimeError("hostile get")

    class _HostileBlock:
        @property
        def text(self) -> str:
            raise RuntimeError("hostile text")

    class _HostileList(list):
        def __iter__(self) -> Any:
            raise RuntimeError("hostile iter")

    class _HostileClass:
        """⛔ The hole the FIRST attempt at this invariant left open, and the reason the guards
        now wrap the `isinstance` calls too. `isinstance` is not inert: when the real type check
        misses, CPython looks up `inst.__class__`, and that lookup suppresses only
        `AttributeError`. So the three `isinstance` calls sitting outside the guard made the
        function raise on an object that is neither `str` nor `list` — while the docstring said
        MUST NEVER RAISE and this very test claimed to prove it."""

        @property
        def __class__(self) -> type:  # type: ignore[override]
            raise RuntimeError("hostile __class__")

    assert assistant_text(_HostileClass()) == ""
    assert assistant_text([{"text": _HostileClass()}]) == ""
    assert assistant_text([_HostileDict()]) == ""
    assert assistant_text([_HostileBlock()]) == ""
    assert assistant_text(_HostileList([1])) == ""
    assert assistant_text([_HostileBlock(), {"type": "text", "text": "hi"}]) == "hi", (
        "one hostile block must not cost the readable ones"
    )
    # ⛔ The same property for a hostile VALUE rather than a hostile BLOCK. `isinstance(text, str)`
    # sat OUTSIDE the per-block guard, so this escaped to the outer one and collapsed the whole
    # reading to "" — while the assertion just above claimed exactly this could not happen. A
    # mostly-readable answer then became an `EmptyCompletion`, and `_shape_note` reported "no
    # readable text" when readable text was right there.
    assert assistant_text(
        [{"text": "AAA"}, {"text": _HostileClass()}, {"text": "BBB"}]
    ) == "AAABBB", "a hostile block VALUE must cost its own block, not the whole answer"


def test_safe_reason_survives_a_hostile_finish_reason() -> None:
    """`_safe_reason` is evaluated inside mtrag's `try` while an `EmptyCompletion` is built, so a
    raise there turns the intended single-billed-call typed refusal into four attempts and an
    untyped wrapper. Its sibling `_shape_note` got this guard; it did not. `isinstance` consults
    `__class__`, and `reason in KNOWN_FINISH_REASONS` calls `__hash__`."""
    from benchmarks.llm import _safe_reason

    class _HostileClass:
        @property
        def __class__(self) -> type:  # type: ignore[override]
            raise RuntimeError("hostile __class__")

    class _HashBoom(str):
        def __hash__(self) -> int:
            raise RuntimeError("hostile hash")

    assert _safe_reason(_HostileClass()) == "unrecognised"
    assert _safe_reason(_HashBoom("stop")) == "unrecognised"
    assert _safe_reason("stop") == "stop", "guards the guard: a known reason still comes through"


# --------------------------------------------------------------------------------------------
# 2. Every client uses it, rather than three copies that match today.
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label,content,expected", [pytest.param(*c, id=c[0]) for c in _SHAPES if c[2].strip()]
)
def test_the_benchmark_client_reads_what_the_shared_reader_reads(
    monkeypatch: pytest.MonkeyPatch, label: str, content: Any, expected: str
) -> None:
    _install_fake_openai(monkeypatch, content)

    assert OpenRouterLLM(model="m", api_key="k").complete("s", "u") == expected


@pytest.mark.parametrize("label,content,expected", [pytest.param(*c, id=c[0]) for c in _SHAPES])
def test_the_extraction_client_reads_what_the_shared_reader_reads(
    label: str, content: Any, expected: str
) -> None:
    assert _text_of(_reply(content)) == expected


@pytest.mark.parametrize(
    "label,content,expected", [pytest.param(*c, id=c[0]) for c in _SHAPES if c[2].strip()]
)
def test_the_mtrag_client_reads_what_the_shared_reader_reads(
    label: str, content: Any, expected: str
) -> None:
    """The third reader, which `(content or "").strip()` crashed on with `AttributeError` for
    every block list, at four billed attempts a task."""
    from benchmarks.mtrag.generation import generate_one

    class _Client:
        chat = property(lambda self: self)  # type: ignore[assignment]

        @property
        def completions(self) -> "_Client":
            return self

        def create(self, **_: Any) -> object:
            return _reply(content)

    assert generate_one(_Client(), "m", [{"role": "user", "content": "x"}], 128) == expected.strip()


# --------------------------------------------------------------------------------------------
# 3. The empty-reading policy, which is where they SHOULD differ.
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label,content", [pytest.param(c[0], c[1], id=c[0]) for c in _SHAPES if not c[2].strip()]
)
def test_a_reading_of_nothing_is_refused_by_the_benchmark_client(
    monkeypatch: pytest.MonkeyPatch, label: str, content: Any
) -> None:
    """Guards the guard. Reading blocks must not become a new way to return `""` after all."""
    _install_fake_openai(monkeypatch, content)

    with pytest.raises(EmptyCompletion):
        OpenRouterLLM(model="m", api_key="k").complete("s", "u")


def test_an_unreadable_block_list_still_names_the_shape_that_arrived() -> None:
    """⛔ Whitelisting `list` in the shape fragment made an unreadable list produce a message
    byte-identical to `content=None`, telling the operator the model said nothing when the cause
    was a block encoding this reader does not understand. That is the same measurement error the
    reconciliation exists to prevent, one shape further out and with the evidence removed."""
    from benchmarks.llm import _shape_note

    assert _shape_note(None, "") == ""
    assert _shape_note("", "") == ""
    assert _shape_note([], "") == ""
    assert "no readable text" in _shape_note([{"type": "image", "url": "x"}], "")
    assert "dict" in _shape_note({"text": "hi"}, "")
    # Read, but only whitespace: a MODEL fault, not a gateway encoding this reader cannot read.
    assert "only whitespace" in _shape_note([{"type": "text", "text": "   "}], "   ")
    # ⛔ No digits. This message is deliberately kept free of them: `is_terminal` and
    # `_is_transient` substring-match rendered exceptions on the bare markers "401" and "402",
    # and a count would render one for a 402-block list.
    assert not any(ch.isdigit() for ch in _shape_note([{"x": 1}] * 402, ""))

    class _HostileLen(list):
        def __len__(self) -> int:
            raise RuntimeError("hostile len")

    _shape_note(_HostileLen([1]), "")  # must not raise while an exception is being built


def test_an_empty_reading_is_tolerated_by_the_extraction_client() -> None:
    """Pins the DELIBERATE divergence, so nobody 'reconciles' it away: returning `""` is safe
    there precisely because something downstream refuses it."""
    from recall.truth_extraction._normalize import _batch_rungs
    from recall.truth_extraction.types import STATUS_VOCABULARY

    assert _text_of(_reply([])) == ""
    with pytest.raises(Exception) as caught:
        # Passed explicitly: `_batch_rungs` no longer defaults its vocabulary, because a default
        # there is a call site silently reverting to the memo-shaped words.
        _batch_rungs("", STATUS_VOCABULARY)
    assert "json" in str(caught.value)


@pytest.mark.parametrize(
    "label,content", [pytest.param(c[0], c[1], id=c[0]) for c in _SHAPES if not c[2].strip()]
)
def test_mtrag_refuses_an_empty_reading_instead_of_submitting_it(
    label: str, content: Any
) -> None:
    """⛔ THE REGRESSION READING BLOCKS INTRODUCED, and the reason this file exists at all.

    `generate_one` returns straight into `predictions`, and `main` counts the task as done, so an
    empty reading was written to the MTRAG submission as the system's answer with rc=0. Before
    block lists were read, `(content or "").strip()` raised `AttributeError` on them, so the task
    was quarantined LOUDLY into `.failed.jsonl` with rc=1. Reading them correctly routed every
    UNREADABLE block list into the silent hole `content=None` was already in.

    Measured before the fix: content `[{"type": "reasoning", "reasoning": "..."}]` produced
    `predictions: [{"text": ""}]`, `written=1`, `failed=0`, rc=0, in one billed call.

    Re-raised directly rather than wrapped, exactly as `CompletionTruncated` already is: "gave up
    after 4 attempts, re-run to resume" is advice that cannot work for a body this reader cannot
    read, and it costs ONE billed attempt instead of four.
    """
    from benchmarks.mtrag.generation import generate_one

    calls = {"n": 0}

    class _Client:
        @property
        def chat(self) -> "_Client":
            return self

        @property
        def completions(self) -> "_Client":
            return self

        def create(self, **_: Any) -> object:
            calls["n"] += 1
            return _reply(content)

    with pytest.raises(EmptyCompletion):
        generate_one(_Client(), "m", [{"role": "user", "content": "x"}], 128)

    assert calls["n"] == 1, "a shape that cannot be read repeats; it must not be retried"


def test_the_shape_note_survives_a_hostile_class() -> None:
    """`_shape_note` runs while an `EmptyCompletion` is being CONSTRUCTED, so a raise there
    replaces the typed error with an untyped one and loses the classification that keeps it out of
    the retry loop. `isinstance` consults `__class__` when the type check misses, so that call had
    to move inside the guard too."""
    from benchmarks.llm import _shape_note

    class _HostileClass:
        @property
        def __class__(self) -> type:  # type: ignore[override]
            raise RuntimeError("hostile __class__")

    assert "could not describe" in _shape_note(_HostileClass(), "")


def _mtrag_refusal(content: Any, reason: Any = "stop") -> EmptyCompletion:
    """Drive mtrag's refusal and hand back the exception, so its MESSAGE can be asserted."""
    from benchmarks.mtrag.generation import generate_one

    calls = {"n": 0}

    class _Client:
        @property
        def chat(self) -> "_Client":
            return self

        @property
        def completions(self) -> "_Client":
            return self

        def create(self, **_: Any) -> object:
            calls["n"] += 1
            return types.SimpleNamespace(choices=[types.SimpleNamespace(
                message=types.SimpleNamespace(content=content), finish_reason=reason)])

    with pytest.raises(EmptyCompletion) as caught:
        generate_one(_Client(), "m", [{"role": "user", "content": "x"}], 128)
    assert calls["n"] == 1, "a shape that cannot be read repeats; it must not be retried"
    return caught.value


def test_the_mtrag_refusal_names_the_cause_not_just_the_emptiness() -> None:
    """⛔ The message was byte-identical for three causes the repo treats as distinguishable, and
    `.failed.jsonl` is the ONLY record of why a task was quarantined. On 842 tasks a systematic
    gateway encoding fault and a model emitting whitespace want different responses."""
    blocks = str(_mtrag_refusal([{"type": "reasoning", "reasoning": "x"}]))
    whitespace = str(_mtrag_refusal("   \n  "))

    assert "text blocks" in blocks
    assert blocks != whitespace, "two causes, two messages"


def test_the_mtrag_refusal_keeps_provider_text_out_of_its_message() -> None:
    """⛔ It rendered the provider-chosen `finish_reason` with `!r`, which is precisely what
    `EmptyCompletion`'s docstring forbids: that message is substring-matched by `is_terminal` on
    the bare markers "401" and "402". The raw value belongs on the attribute."""
    exc = _mtrag_refusal(None, reason="upstream 402 refused")

    assert "402" not in str(exc), "a provider string must not reach a substring classifier"
    assert "unrecognised" in str(exc)
    assert exc.finish_reason == "upstream 402 refused", "withheld from the message, not discarded"


def test_a_hostile_finish_reason_repr_costs_one_call_not_four() -> None:
    """The f-string is evaluated INSIDE the `try`, so a `finish_reason` whose `__repr__` raises
    turned this single-billed-call typed refusal into four attempts and an untyped wrapper."""

    class _HostileRepr:
        def __repr__(self) -> str:
            raise RuntimeError("hostile repr")

    exc = _mtrag_refusal(None, reason=_HostileRepr())  # asserts calls == 1 internally
    assert "unrecognised" in str(exc)
