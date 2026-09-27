"""C9 never resends a cut-off compile answer and skips an oversized payload; other variants keep
resending as before.

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

CCA audit of #775, 2026-09-27. ``test_a_cut_off_answer_is_not_sent_again`` now builds the compiler
with ``resend_truncated=False``, because refusing a resend became C9's setting (FIX-003) instead of
every variant's behaviour; ``test_a_variant_that_keeps_resends_still_retries_a_cut_off_answer`` is
the control for everything else. Red proof, same method:

* ``test_other_failures_are_still_retried`` (CODE-012): widening the v3 loop's ``except
  CompilerOutputTruncated:`` to ``except ValueError:`` let the first schema error escape
  (``JSONDecodeError``) instead of retrying.
* ``test_a_payload_under_the_limit_is_compiled`` (CODE-012): inverting the gate to ``if
  encoded_chars <= limit:`` skipped the payload (``CompilerInputTooLarge``, 16,939 chars against
  1,000,000).
* ``test_a_variant_that_keeps_resends_still_retries_a_cut_off_answer`` (FIX-003): against the
  pre-fix compiler, which refused every variant's resend, it failed ``len(calls) == 3`` with 1.
* ``test_served_c9_refuses_to_resend_a_cut_off_answer`` (FIX-003): dropping
  ``resend_truncated=behavior.compile_resend_truncated`` from ``build_compiler`` made C9 retry the
  cut-off answer until it failed to parse (``JSONDecodeError``, not ``CompilerOutputTruncated``).
* ``test_prior_records_alone_never_skip_a_small_add`` (FIX-002, BUG-001): against the pre-fix
  compiler, which measured the limit on anchors plus prior records, the small Add was skipped
  (``CompilerInputTooLarge``, 146,247 chars against 20,000).
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from recall_aml.compiler import (
    CompilerInputTooLarge,
    CompilerOutputTruncated,
    OpenAICompiler,
    StoredCodingRecord,
)
from recall_aml.models import CodingMemoryRecord, EvidenceSpan, Message

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
    compiler = OpenAICompiler(client, sleep=lambda _: None, resend_truncated=False)

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


def test_a_variant_that_keeps_resends_still_retries_a_cut_off_answer() -> None:
    """Only a variant that refuses resends (C9) stops at the first cut-off answer (FIX-003)."""
    client, calls = _client([(CUT_OFF, "length")])
    compiler = OpenAICompiler(client, sleep=lambda _: None)

    with pytest.raises(ValueError):
        compiler.compile(_messages(20), "s", [])

    assert len(calls) == 3


def test_served_c9_refuses_to_resend_a_cut_off_answer() -> None:
    """C9's compiler, built as the service builds it, makes one call on a cut-off answer."""
    from recall_aml.__main__ import build_compiler
    from recall_aml.config import HostedSettings
    from recall_aml.variants import variant

    client, calls = _client([(CUT_OFF, "length")])
    settings = HostedSettings(
        database_url="postgresql://unused", api_key="k", git_commit="c", openrouter_api_key="or"
    )
    compiler = build_compiler(
        settings,
        variant("C9_routed_specialists_grounded_graph_atomic"),
        client_factory=lambda **_: client,
    )

    assert compiler is not None
    with pytest.raises(CompilerOutputTruncated):
        compiler.compile_anchored_v3(_messages(20), "s", [])
    assert len(calls) == 1


def _large_prior(n: int) -> list[StoredCodingRecord]:
    quote = "moved the queue to postgres and kept the retry " * 30
    span = EvidenceSpan(message_ordinal=0, start=0, end=len(quote), quote=quote)
    record = CodingMemoryRecord(
        kind="procedure",
        action="moved the queue",
        evidence_spans=[span, span],
        evidence_quotes=[quote, quote],
        source_session_id="s",
    )
    return [StoredCodingRecord(f"mem_{i:064d}", record) for i in range(n)]


def test_prior_records_alone_never_skip_a_small_add() -> None:
    """The limit is measured on the Add's own anchors, as the replay measured it (FIX-002).

    Otherwise a session whose 24 prior records encode past the limit would skip every later Add,
    and a skipped Add writes no record, so the prior set, and the skip, would never change.
    """
    client, calls = _client([('{"records": []}', "stop")])
    compiler = OpenAICompiler(client, sleep=lambda _: None, max_anchor_payload_chars=20_000)

    compiler.compile_anchored_v3(_messages(20), "s", _large_prior(24))

    assert len(calls) == 1
