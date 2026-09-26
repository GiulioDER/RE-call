"""What an anchored compile asks the model to write: ``full`` (served), ``lean`` and ``select``.

On the official Textual Full of 2026-09-25, 57,222 of 75,709 accepted compiled records (76%) had
every generated text field removed as not verbatim and were backfilled from their first cited
anchor. ``select`` asks only for what survived in those records (kind and cited anchors), and
``lean`` drops the keys the compiler overwrites. C9 keeps ``full`` until a replay decides
(docs/preregistrations/2026-09-26-c9-compile-output.md).

Red proof, 2026-09-26, each by mutating the named line with this file unchanged, then restoring:

* ``test_served_c9_keeps_the_full_output``: setting the ``HostedVariant.anchor_compile_output``
  default to ``"select"`` in ``recall_aml/variants.py`` failed ``== "full"`` on the variant.
  Passing ``anchor_output_mode="select"`` instead of the variant's value in ``build_compiler``
  (``recall_aml/__main__.py``) failed the builder assertion the same way.
* ``test_select_sends_the_select_prompt_and_backfills``: returning
  ``ANCHOR_COMPILER_SYSTEM_PROMPT`` for ``select`` in ``OpenAICompiler._anchor_system_prompt``
  failed the prompt assertion; leaving the answer to ``AnchoredCompilerPayload`` in
  ``_anchored_payload`` failed with a ``ValidationError`` for the missing ``source_session_id``.
* ``test_lean_keeps_a_verbatim_field_and_fills_the_session``: filling ``source_session_id`` from
  ``item`` instead of the Add's session failed ``len(records) == 1`` (the record was rejected as
  from another session).
* ``test_full_is_unchanged``: sending ``ANCHOR_COMPILER_LEAN_SYSTEM_PROMPT`` for ``full`` failed
  the prompt assertion.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from recall_aml.compiler import (
    ANCHOR_COMPILER_LEAN_SYSTEM_PROMPT,
    ANCHOR_COMPILER_SELECT_SYSTEM_PROMPT,
    ANCHOR_COMPILER_SYSTEM_PROMPT,
    OpenAICompiler,
    build_evidence_anchors,
)
from recall_aml.models import Message

SESSION = "session-7"
TEXT = "The worker queue moved to postgres. Run make migrate before deploying the worker."


def _client(answer: dict[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []

    def create(**request: Any) -> Any:
        calls.append(request)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop", message=SimpleNamespace(content=json.dumps(answer))
                )
            ]
        )

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))), calls


def _anchor_id() -> str:
    anchors = build_evidence_anchors([Message(role="user", content=TEXT)], SESSION, identifier_version=3)
    return anchors[0].id


def _system(call: dict[str, Any]) -> str:
    return str(call["messages"][0]["content"])


def test_an_unknown_output_mode_is_refused() -> None:
    with pytest.raises(ValueError, match="anchor_output_mode"):
        OpenAICompiler(object(), anchor_output_mode="short")


def test_served_c9_keeps_the_full_output() -> None:
    from recall_aml.__main__ import build_compiler
    from recall_aml.config import HostedSettings
    from recall_aml.variants import variant

    served = variant("C9_routed_specialists_grounded_graph_atomic")
    settings = HostedSettings(
        database_url="postgresql://unused", api_key="k", git_commit="c", openrouter_api_key="or"
    )
    compiler = build_compiler(settings, served, client_factory=lambda **_: object())

    assert served.anchor_compile_output == "full"
    assert compiler is not None
    assert compiler._anchor_output_mode == "full"


def test_full_is_unchanged() -> None:
    answer = {
        "records": [
            {
                "kind": "procedure",
                "action": "Run make migrate before deploying the worker.",
                "evidence_anchor_ids": [_anchor_id()],
                "source_session_id": SESSION,
                "supersedes": [],
            }
        ]
    }
    client, calls = _client(answer)
    records = OpenAICompiler(client, sleep=lambda _: None).compile_anchored_v3(
        [Message(role="user", content=TEXT)], SESSION, []
    )

    assert _system(calls[0]) == ANCHOR_COMPILER_SYSTEM_PROMPT
    assert [record.action for record in records] == ["Run make migrate before deploying the worker."]


def test_select_sends_the_select_prompt_and_backfills() -> None:
    client, calls = _client(
        {"records": [{"kind": "procedure", "evidence_anchor_ids": [_anchor_id()]}]}
    )
    records = OpenAICompiler(
        client, sleep=lambda _: None, anchor_output_mode="select"
    ).compile_anchored_v3([Message(role="user", content=TEXT)], SESSION, [])

    assert _system(calls[0]) == ANCHOR_COMPILER_SELECT_SYSTEM_PROMPT
    assert len(calls) == 1
    [record] = records
    assert record.kind == "procedure"
    assert record.source_session_id == SESSION
    assert record.supersedes == []
    assert record.event_time is None
    # A procedure is backfilled into ``action`` from its first cited anchor, exactly as the full
    # path backfills a record whose generated fields were all removed.
    assert record.action == record.evidence_spans[0].quote[:700]
    assert record.action and record.action in TEXT


def test_lean_keeps_a_verbatim_field_and_fills_the_session() -> None:
    answer = {
        "records": [
            {
                "kind": "constraint",
                "problem": "The worker queue moved to postgres.",
                "evidence_anchor_ids": [_anchor_id()],
                # Keys the lean shape no longer asks for are ignored, not refused.
                "source_session_id": "a session id the model misremembered",
                "supersedes": ["mem_0123"],
            }
        ]
    }
    client, calls = _client(answer)
    records = OpenAICompiler(
        client, sleep=lambda _: None, anchor_output_mode="lean"
    ).compile_anchored_v3([Message(role="user", content=TEXT)], SESSION, [])

    assert _system(calls[0]) == ANCHOR_COMPILER_LEAN_SYSTEM_PROMPT
    assert len(records) == 1
    [record] = records
    assert record.problem == "The worker queue moved to postgres."
    assert record.source_session_id == SESSION
    assert record.supersedes == []
