"""An anchored compile over the model's context window is refitted once, not retried identically.

Measured 2026-09-24 against C9's own compiler: 802,804 encoded characters asked gpt-4o-mini for
about 201,000 tokens and got HTTP 400 in 0.5 s; three identical retries then failed the same
way and the Add kept no compiled record. Each behaviour test records the mutation it was proved
red against, with this file unchanged.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import recall_aml.compiler as compiler_module
from recall_aml.compiler import OpenAICompiler, fit_anchor_payload
from recall_aml.models import Message


class ContextOverflow(Exception):
    status_code = 400


def stored_data(call: dict[str, Any]) -> dict[str, Any]:
    content = call["messages"][1]["content"]
    return json.loads(content.removeprefix("<stored_data>").removesuffix("</stored_data>"))


def session(anchor_count: int) -> list[Message]:
    # One message of distinct 1,440 character steps yields one anchor per step (1,600 chars,
    # 160 overlap), so anchor order is session order.
    steps = [f"step{i:03d} " + f"ran pytest on module{i} and fixed parser.py line {i}. " * 24 for i in range(anchor_count)]
    return [Message(role="tool", content="".join(step[:1_440] for step in steps))]


def compiler_with_window(window_chars: int, calls: list[dict[str, Any]]) -> OpenAICompiler:
    """A provider that refuses any prompt whose stored data exceeds ``window_chars``."""

    def complete(**kwargs: Any) -> Any:
        calls.append(kwargs)
        data = stored_data(kwargs)
        if len(kwargs["messages"][1]["content"]) > window_chars:
            raise ContextOverflow("maximum context length exceeded")
        first = data["anchors"][0]["id"]
        answer = {
            "records": [
                {
                    "kind": "successful repair",
                    "action": "fixed parser.py",
                    "evidence_anchor_ids": [first],
                    "source_session_id": "session",
                }
            ]
        }
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(answer)))]
        )

    return OpenAICompiler(
        SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=complete))),
        sleep=lambda _: None,
    )


def test_an_over_window_payload_is_fitted_after_its_first_failure(monkeypatch):
    """Red proof: the `if sent is payload:` block in `_compile_anchored` deleted, so every attempt
    resends the full payload. This test then failed on `assert records is not None`."""
    monkeypatch.setattr(compiler_module, "ANCHOR_PAYLOAD_BUDGET_CHARS", 20_000)
    calls: list[dict[str, Any]] = []
    compiler = compiler_with_window(25_000, calls)
    messages = session(40)

    try:
        records = compiler.compile_anchored_v3(messages, "session", [])
    except ContextOverflow:
        records = None

    assert records is not None
    assert len(calls) == 2
    full, fitted = stored_data(calls[0])["anchors"], stored_data(calls[1])["anchors"]
    assert len(fitted) < len(full)
    assert fitted[0]["id"] == full[0]["id"] and fitted[-1]["id"] == full[-1]["id"]
    assert [a["id"] for a in fitted] == [a["id"] for a in full if a["id"] in {b["id"] for b in fitted}]


def test_a_payload_that_compiles_is_sent_in_full_even_over_the_budget(monkeypatch):
    """Red proof: `sent` initialised to `fit_anchor_payload(payload, ...) or payload`, fitting up
    front. This test then failed on the anchor count equality (the first call was trimmed)."""
    monkeypatch.setattr(compiler_module, "ANCHOR_PAYLOAD_BUDGET_CHARS", 20_000)
    calls: list[dict[str, Any]] = []
    compiler = compiler_with_window(10_000_000, calls)
    messages = session(40)

    records = compiler.compile_anchored_v3(messages, "session", [])

    assert records
    assert len(calls) == 1
    assert len(stored_data(calls[0])["anchors"]) == len(
        compiler_module.build_evidence_anchors(messages, "session", identifier_version=3)
    )


def test_fitting_keeps_both_ends_in_order_within_the_budget():
    """Red proof: the tail loop of `fit_anchor_payload` disabled (`tail_start` never moves). This
    test then failed on `assert kept[-1] == anchors[-1]`."""
    anchors = [{"id": f"a{i:03d}", "excerpt": "x" * 900} for i in range(50)]
    payload = {"session_id": "s", "anchors": anchors, "prior_records": []}

    fitted = fit_anchor_payload(payload, 12_000)

    assert fitted is not None
    kept = fitted["anchors"]
    assert kept[0] == anchors[0]
    assert kept[-1] == anchors[-1]
    assert kept == sorted(kept, key=lambda anchor: anchor["id"])
    assert len(compiler_module._encode_stored_data(fitted)) <= 12_000


def test_fitting_leaves_a_payload_that_fits_alone():
    """Nonbehavioural control: under the budget there is nothing to fit, so a failed attempt on a
    payload within the budget is retried exactly as before."""
    payload = {"session_id": "s", "anchors": [{"id": "a000", "excerpt": "x"}], "prior_records": []}
    assert fit_anchor_payload(payload, 10_000) is None
