"""One wire shape, one reading rule, in both OpenAI-compatible clients this repo ships.

`message.content` can arrive as a LIST of text blocks rather than a string. A gateway that does
that is returning a well formed answer in a shape the OpenAI schema also permits, and the repo
handled it two ways:

    recall/truth_extraction/_openai_engine.py:_text_of   joined the block text
    benchmarks/llm.py:_complete_once                     refused it as unusable

`_text_of`'s reasoning is the stronger one and its comment says why: discarding the answer "would
refuse a WELL FORMED reply and blame the model for it". Refusing it in the benchmark client fails
questions that the extraction client would have read correctly, and it blames the system under
test for its gateway's encoding, which is the same class of measurement error `EmptyCompletion`
was added to prevent.

⚠️ **THE TWO CLIENTS STILL DIVERGE, AND THAT PART IS DELIBERATE.** They now agree on how to READ
the content and disagree only on what an EMPTY reading means, because the consequence differs:

    _text_of        returns ""      safe: `_batch_rungs("")` raises `ExtractionBatchRejected`,
                                    which a reviewer sees as a batch rejection
    _complete_once  raises          necessary: nothing downstream re-reads the answer, so "" was
                                    SCORED as the system's answer

Verified rather than assumed, in `test_an_empty_reading_is_refused_here_and_tolerated_there`.

🔑 The reading rule is MIRRORED, not imported, and the last test in this file is what makes that
safe. Importing `_text_of` into `benchmarks/llm.py` would make the benchmark client depend on the
truth-extraction subsystem, which is a dependency direction a reviewer would rightly question, and
it does not fit anyway: `_text_of` takes a whole reply and flattens an empty `choices` list into
`""`, while `_complete_once` has to tell that case apart as `NoCompletionChoices` (transient) from
empty content (permanent). So the coupling is pinned by a test that drives BOTH implementations
over the same table and asserts they agree, which fails loudly if either one drifts. This session
learned that a duplicated rule kept in step by a comment does drift: `is_terminal` ran with two of
three status spellings for a whole branch and nothing went red.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from benchmarks.llm import EmptyCompletion, OpenRouterLLM
from recall.truth_extraction._openai_engine import _text_of

#: The shapes both readers must agree on, and what the agreed reading is. Dicts and objects both
#: appear because a gateway may serialise blocks either way, and `_text_of` handles both.
_BLOCK_SHAPES: list[tuple[str, Any, str]] = [
    ("one text block", [{"type": "text", "text": "an answer"}], "an answer"),
    ("two blocks join in order", [{"type": "text", "text": "an "}, {"type": "text", "text": "answer"}], "an answer"),
    ("object blocks, not dicts", [types.SimpleNamespace(text="an answer")], "an answer"),
    ("a non-text block contributes nothing", [{"type": "image", "url": "x"}, {"type": "text", "text": "hi"}], "hi"),
    ("a block whose text is None", [types.SimpleNamespace(text=None), {"type": "text", "text": "hi"}], "hi"),
    ("an empty list reads as nothing", [], ""),
    ("a list of nothing useful reads as nothing", [{"type": "image", "url": "x"}], ""),
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
    """A whole chat completion carrying `content`, for `_text_of`, which takes the reply."""
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content))]
    )


@pytest.mark.parametrize(
    "label,content,expected",
    [pytest.param(*case, id=case[0]) for case in _BLOCK_SHAPES if case[2]],
)
def test_block_content_is_read_rather_than_refused(
    monkeypatch: pytest.MonkeyPatch, label: str, content: Any, expected: str
) -> None:
    """The defect. `_complete_once` refused every one of these with `EmptyCompletion`, naming the
    type, so a gateway that encodes text as blocks failed EVERY question of a run while the
    extraction client read the identical body correctly."""
    _install_fake_openai(monkeypatch, content)

    assert OpenRouterLLM(model="m", api_key="k").complete("s", "u") == expected


@pytest.mark.parametrize(
    "label,content", [pytest.param(c[0], c[1], id=c[0]) for c in _BLOCK_SHAPES if not c[2]]
)
def test_a_block_list_carrying_no_text_is_still_refused(
    monkeypatch: pytest.MonkeyPatch, label: str, content: Any
) -> None:
    """Guards the guard. Reading blocks must not become a way to return `""` after all: a list
    with nothing readable in it is exactly as unscorable as `content=None` was."""
    _install_fake_openai(monkeypatch, content)

    with pytest.raises(EmptyCompletion):
        OpenRouterLLM(model="m", api_key="k").complete("s", "u")


def test_a_plain_string_is_still_returned_verbatim(monkeypatch: pytest.MonkeyPatch) -> None:
    """The overwhelmingly common path, unchanged and still unstripped."""
    _install_fake_openai(monkeypatch, "  an answer  ")

    assert OpenRouterLLM(model="m", api_key="k").complete("s", "u") == "  an answer  "


def test_a_non_str_non_list_content_is_still_refused_by_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dict, an int, anything else: still unusable, and the message still names the shape that
    arrived so the operator sees a provider problem rather than an `AttributeError`."""
    _install_fake_openai(monkeypatch, {"text": "hi"})

    with pytest.raises(EmptyCompletion) as caught:
        OpenRouterLLM(model="m", api_key="k").complete("s", "u")

    assert "dict" in str(caught.value)


def test_an_empty_reading_is_refused_here_and_tolerated_there() -> None:
    """⚠️ Pins the DELIBERATE divergence, so nobody 'reconciles' it away.

    The two clients agree on the reading and disagree on the policy for an empty one, because the
    consequence differs: `_text_of` returning "" meets `_batch_rungs`, which refuses it as a batch
    rejection a reviewer can see, whereas `complete` returning "" was scored as the system's answer
    with nothing downstream to catch it.
    """
    from recall.truth_extraction._normalize import _batch_rungs

    assert _text_of(_reply([])) == "", "the extraction client tolerates an empty reading"
    with pytest.raises(Exception) as caught:
        _batch_rungs("")
    assert "json" in str(caught.value), "and something downstream refuses it"


def test_both_clients_read_every_block_shape_identically() -> None:
    """🔑 THE ARM THAT KEEPS THEM IN STEP, and the reason mirroring the rule is acceptable.

    Drives `_text_of` and `benchmarks.llm`'s reader over the same table and asserts they agree. A
    rule duplicated in two modules and kept in step by a comment DOES drift: in this same
    subsystem, `is_terminal` ran with two of three status spellings for an entire branch and
    nothing went red until a rebase surfaced it.
    """
    from benchmarks.llm import _assistant_text

    for label, content, expected in _BLOCK_SHAPES:
        mine = _assistant_text(content)
        theirs = _text_of(_reply(content))
        assert mine == theirs == expected, (
            f"{label}: benchmarks read {mine!r}, truth_extraction read {theirs!r}, "
            f"expected {expected!r}"
        )
