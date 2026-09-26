"""C9's anchored compile never resends a cut-off answer, and skips an oversized payload.

On the official Textual Full of 2026-09-25, USD 67 of the USD 85 spent on the compiler went to
2,802 Adds whose answer stopped at ``max_tokens``: the identical prompt was sent three times and
every answer was thrown away. Retries rescued 92 of 2,900 truncated first answers. Compiles whose
prompt was over about 40k tokens succeeded 15% of the time.

Red proof, 2026-09-26, each by mutating the named line of ``recall_aml/compiler.py`` with this
file unchanged, then restoring it:

* ``test_a_cut_off_answer_is_not_sent_again``: deleting the ``except CompilerOutputTruncated:
  raise`` clause in the v3 attempt loop of ``OpenAICompiler._compile_anchored`` made three calls
  and failed ``len(calls) == 1``.
* ``test_an_oversized_payload_is_skipped_without_a_call``: making the ``if encoded_chars >
  limit:`` condition false failed with ``DID NOT RAISE CompilerInputTooLarge``, because the
  payload was sent.
* ``test_served_c9_bounds_the_compile``: removing ``anchor_compile_max_payload_chars=150_000``
  from the C9 variant in ``recall_aml/variants.py`` failed the equality on the limit.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from recall_aml.compiler import CompilerInputTooLarge, CompilerOutputTruncated, OpenAICompiler
from recall_aml.models import Message

CUT_OFF = '{"records":[{"kind":"procedure","action":"moved the qu'


def _client(answers: list[tuple[str, str]]) -> tuple[Any, list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []

    def create(**request: Any) -> Any:
        calls.append(request)
        content, finish = answers[min(len(calls), len(answers)) - 1]
        return SimpleNamespace(
            choices=[SimpleNamespace(finish_reason=finish, message=SimpleNamespace(content=content))]
        )

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))), calls


def _messages(words: int) -> list[Message]:
    text = " ".join(f"step{i} moved the queue to postgres" for i in range(words))
    return [Message(role="user", content=text)]


def test_a_cut_off_answer_is_not_sent_again() -> None:
    client, calls = _client([(CUT_OFF, "length")])
    compiler = OpenAICompiler(client, sleep=lambda _: None)

    with pytest.raises(CompilerOutputTruncated):
        compiler.compile_anchored_v3(_messages(20), "s", [])

    assert len(calls) == 1


def test_other_failures_are_still_retried() -> None:
    client, calls = _client([("not json", "stop"), ('{"records": []}', "stop")])
    compiler = OpenAICompiler(client, sleep=lambda _: None)

    compiler.compile_anchored_v3(_messages(20), "s", [])

    assert len(calls) == 2


def test_an_oversized_payload_is_skipped_without_a_call() -> None:
    client, calls = _client([('{"records": []}', "stop")])
    compiler = OpenAICompiler(client, sleep=lambda _: None, max_anchor_payload_chars=2_000)

    with pytest.raises(CompilerInputTooLarge):
        compiler.compile_anchored_v3(_messages(400), "s", [])

    assert calls == []


def test_a_payload_under_the_limit_is_compiled() -> None:
    client, calls = _client([('{"records": []}', "stop")])
    compiler = OpenAICompiler(client, sleep=lambda _: None, max_anchor_payload_chars=1_000_000)

    compiler.compile_anchored_v3(_messages(400), "s", [])

    assert len(calls) == 1
    assert json.loads(calls[0]["messages"][1]["content"].split("<stored_data>")[1].split("</stored_data>")[0])


def test_served_c9_bounds_the_compile() -> None:
    from recall_aml.__main__ import build_compiler
    from recall_aml.config import HostedSettings
    from recall_aml.variants import variant

    settings = HostedSettings(
        database_url="postgresql://unused", api_key="k", git_commit="c", openrouter_api_key="or"
    )
    compiler = build_compiler(
        settings,
        variant("C9_routed_specialists_grounded_graph_atomic"),
        client_factory=lambda **_: object(),
    )

    assert compiler is not None
    assert compiler._max_anchor_payload_chars == 150_000
