"""A v3 anchored compile accepts a bare ``a<index>`` citation of an anchor it sent.

Found by the live C9 BEAM probe of 2026-09-24: 11 of 88 compiles kept no record. Replaying the
served compile on the same public BEAM batches showed gpt-4o-mini citing ``a162`` for
``a162_b7ab58bc7af5c114``, every citation of the answer, so every record was rejected. Each test
records the mutation of ``recall_aml/compiler.py`` it was proved red against, with this file
unchanged.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import recall_aml.compiler as compiler_module
from recall_aml.compiler import ANCHOR_PAYLOAD_BUDGET_CHARS, OpenAICompiler
from recall_aml.models import Message


def _messages(anchor_count: int) -> list[Message]:
    """One message whose text splits into about ``anchor_count`` anchors, each distinct."""
    words = " ".join(f"topic{i} detail{i} value{i}" for i in range(anchor_count * 90))
    return [Message(role="user", content=words)]


def _proposal(anchor_ids: list[str]) -> dict[str, Any]:
    # Fields quote nothing from the text, so the exact-text recovery path cannot rescue a record.
    return {
        "kind": "architectural decision",
        "action": "the user settled on a plan",
        "evidence_anchor_ids": anchor_ids,
        "source_session_id": "session",
    }


def _compiler(answers: list[Any]) -> tuple[OpenAICompiler, list[dict]]:
    calls: list[dict] = []

    def complete(**kwargs: Any) -> Any:
        calls.append(kwargs)
        answer = answers[len(calls) - 1]
        if isinstance(answer, Exception):
            raise answer
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(answer)))]
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=complete)))
    return OpenAICompiler(client, sleep=lambda _: None), calls


def _compile(
    compiler: OpenAICompiler, messages: list[Message], version: int = 3
) -> list[Any] | None:
    method = compiler.compile_anchored_v3 if version == 3 else compiler.compile_anchored
    try:
        return method(messages, "session", [])
    except Exception:  # BROAD-CATCH: a rejected answer surfaces as a raise; the test asserts on it
        return None


def test_a_bare_index_citation_keeps_the_record_and_its_evidence() -> None:
    """Red proof: the two lines that call
    ``resolve_bare_anchor_ids`` and count it deleted. This test then failed on ``assert records``: every
    citation was unknown, the only record was rejected, and the compile kept nothing."""
    messages = _messages(4)
    anchors = compiler_module.build_evidence_anchors(messages, "session", identifier_version=3)
    compiler, _ = _compiler([{"records": [_proposal(["a001", "a002"])]}])

    records = _compile(compiler, messages)

    assert records
    quotes = [span.quote for span in records[0].evidence_spans]
    assert quotes == [anchors[1].quote, anchors[2].quote]


def test_a_well_formed_id_with_another_anchors_hash_stays_unknown() -> None:
    """Red proof: the lookup made by prefix, ``anchor_id.partition("_")[0] in by_index`` with the
    appended id taken the same way, which resolves by index whatever hash follows. This test then
    failed on ``assert records == []``: ``a001_<hash of a002>`` was read as a001."""
    messages = _messages(4)
    anchors = compiler_module.build_evidence_anchors(messages, "session", identifier_version=3)
    mixed = "a001_" + anchors[2].id.split("_", 1)[1]
    compiler, _ = _compiler([{"records": [_proposal([mixed])]}])

    records = _compile(compiler, messages)

    assert records == []


def test_an_index_the_fitted_retry_did_not_send_stays_unknown() -> None:
    """Red proof: ``sent_anchor_ids`` built from ``anchors`` (every anchor) instead of the payload
    actually sent. This test then failed on ``assert unsent not in cited``: an anchor the model
    never saw on the fitted retry was accepted as evidence."""
    messages = _messages(260)
    anchors = compiler_module.build_evidence_anchors(messages, "session", identifier_version=3)
    fitted = compiler_module.fit_anchor_payload(
        {"session_id": "session",
         "anchors": [compiler_module._anchor_payload(a) for a in anchors],
         "prior_records": []},
        ANCHOR_PAYLOAD_BUDGET_CHARS,
    )
    assert fitted is not None
    sent = {a["id"].split("_", 1)[0] for a in fitted["anchors"]}
    unsent = next(a.id.split("_", 1)[0] for a in anchors if a.id.split("_", 1)[0] not in sent)
    compiler, calls = _compiler([
        RuntimeError("context length exceeded"),
        {"records": [_proposal([unsent, "a000"])]},
    ])

    records = _compile(compiler, messages)

    assert len(calls) == 2
    assert records
    cited = {span.quote for span in records[0].evidence_spans}
    assert anchors[0].quote in cited
    assert next(a.quote for a in anchors if a.id.startswith(unsent + "_")) not in cited


def test_a_v2_id_head_is_never_an_index() -> None:
    """v2 ids are ``anchor_<hash>``; their head ``anchor`` must not become a resolvable index.

    Red proof: the ``_BARE_ANCHOR_INDEX.fullmatch(index)`` condition removed from the map. This
    test then failed on ``assert resolved == ["anchor"]``: a citation of ``anchor`` resolved to the
    last v2 anchor."""
    sent = ["anchor_" + "1" * 64, "anchor_" + "2" * 64]

    resolved, count = compiler_module.resolve_bare_anchor_ids(["anchor"], sent)

    assert resolved == ["anchor"]
    assert count == 0
