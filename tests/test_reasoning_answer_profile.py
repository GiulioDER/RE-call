"""The opt-in ``dated`` answer profile of `recall_reasoning_query`.

Measured before it was added (the maintainer's research log, RC-1, 2026-09-29): on 120
LongMemEval-S questions, same evidence and model, accuracy 0.525 to 0.633, paired +0.108 (95% CI
+0.033 to +0.183), mostly because the plain prompt declined questions its evidence covered. These
tests pin what shipped to what was measured, keep the default unchanged, and hold both front doors
to the same setting. Each names its red proof.
"""

from __future__ import annotations

import ast
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from recall.answer_provider import resolve_answer_profile
from recall.evidence import (
    DATED_READER_CONTRACT,
    DATED_SYSTEM_PROMPT,
    EVIDENCE_CLOSE,
    EVIDENCE_OPEN,
    SYSTEM_PROMPT,
    EvidenceBundle,
    EvidenceItem,
    _user_message,
    render_dated_evidence_prompt,
    render_evidence_prompt,
)
from recall.reasoning import (
    GenerationSelection,
    ReasoningPolicy,
    ReasoningProviderPorts,
    ReasoningRequest,
    reason,
    reasoning_response_from_dict,
)
from recall.types import (
    Chunk,
    EvidenceCard,
    Provenance,
    RetrievalDiagnostics,
    StalenessReport,
    TrustedHit,
    TrustedResult,
    Validity,
)

ROOT = Path(__file__).resolve().parents[1]
#: sha256 of the contract text RC-1 measured (`GENERIC_CONTRACT` in the RC-1 harness at 64a68d2d).
MEASURED_CONTRACT_SHA256 = "b7fc95bd09f4665d6d93309c735409e413d698531b956e72d237653547f83ca9"

REBUILT = datetime(2026, 9, 1, tzinfo=timezone.utc)
FIRST = datetime(2023, 5, 20, 9, 0, tzinfo=timezone.utc)
VALID = datetime(2023, 4, 1, tzinfo=timezone.utc)


def _item(cid: str, *, valid_from: datetime | None = None) -> EvidenceItem:
    return EvidenceItem(
        chunk_id=cid, text=f"text {cid}", source=f"{cid}.md", ordinal=0, indexed_at=REBUILT,
        valid_from=valid_from, valid_until=None, cosine=0.9, confidence=0.9,
    )


def _card(cid: str, first: datetime) -> EvidenceCard:
    return EvidenceCard(
        card_id="", chunk_id=cid, source=f"{cid}.md", source_digest="d", valid_from=None,
        valid_until=None, first_indexed_at=first, indexed_at=REBUILT, tenant_id="t",
        generation_id="g", pipeline_fingerprint="p", corpus_fingerprint="c", calibration_id="cal",
        calibration_status="certified", trust_state="trusted", verdict="ok", confidence=0.9, rank=1,
    )


def _bundle(items: tuple[EvidenceItem, ...], cards: tuple[EvidenceCard, ...] = ()) -> EvidenceBundle:
    return EvidenceBundle(
        query="when did I buy the bike?", decision="answer", reason_code=None,
        decision_state="supported", calibrated=True, stale=False, embedding_profile="e",
        retrieval_profile="r", index_generation="g", items=items, cards=cards,
    )


def _payload(user: str) -> dict:
    return json.loads(user[len(EVIDENCE_OPEN):-len(EVIDENCE_CLOSE)])


def test_the_default_profile_renders_exactly_what_it_always_did() -> None:
    """Invariant: with no profile the boundary is unchanged: the fixed system prompt and the old
    user message, with no question date and no `date` key, so existing callers see no change.

    Red proof: routing `render_evidence_prompt` through the dated renderer (returning
    ``render_dated_evidence_prompt(bundle, "q")``) adds `question_date` and fails the user equality.
    """
    bundle = _bundle((_item("c1"),), (_card("c1", FIRST),))
    system, user = render_evidence_prompt(bundle)
    assert system == SYSTEM_PROMPT
    assert user == _user_message(bundle.query, bundle.items)
    assert "question_date" not in _payload(user) and "date" not in _payload(user)["evidence"][0]


def test_the_shipped_contract_is_the_measured_one() -> None:
    """Invariant: the instructions that ship are byte for byte those RC-1 measured; a wording
    change needs a new measurement, and this test is where that is noticed.

    Red proof: changing "LATEST" to "latest" in `DATED_READER_CONTRACT` changes the digest.
    """
    assert hashlib.sha256(DATED_READER_CONTRACT.encode("utf-8")).hexdigest() == MEASURED_CONTRACT_SHA256


def test_dated_puts_the_contract_after_the_safety_prompt_and_adds_the_question_date() -> None:
    """Invariant: the safety contract still comes first, the reading instructions follow, the
    system message is the module constant itself (no interpolation site, the rule
    `render_evidence_prompt` is held to), and the caller's question date is in the data payload.

    Red proof: defining ``DATED_SYSTEM_PROMPT = DATED_READER_CONTRACT + SYSTEM_PROMPT`` fails the
    first equality.
    """
    system, user = render_dated_evidence_prompt(
        _bundle((_item("c1"),)), "2023-05-30T20:36:00+00:00"
    )
    assert system == SYSTEM_PROMPT + DATED_READER_CONTRACT
    assert system is DATED_SYSTEM_PROMPT
    assert _payload(user)["question_date"] == "2023-05-30T20:36:00+00:00"


def test_an_items_date_is_authored_validity_then_first_write_never_the_rebuild_time() -> None:
    """Invariant: `date` is `valid_from` when authored, else the chunk's FIRST write from its card,
    else `indexed_at`. A generation rebuild stamps `indexed_at` with the build time, so preferring
    it would date every memory of a rebuilt corpus the same day.

    Red proof: `date = item.indexed_at` (ignoring the card) makes c2's date the rebuild time and
    fails the second equality; dropping `valid_from` from the chain fails the first.
    """
    items = (_item("c1", valid_from=VALID), _item("c2"), _item("c3"))
    cards = (_card("c1", FIRST), _card("c2", FIRST))
    _, user = render_dated_evidence_prompt(_bundle(items, cards), "q")
    dates = [entry["date"] for entry in _payload(user)["evidence"]]
    assert dates[0] == VALID.isoformat()
    assert dates[1] == FIRST.isoformat()
    assert dates[2] == REBUILT.isoformat()


def test_a_dated_call_without_a_question_date_is_refused() -> None:
    """Invariant: a missing date does not silently render a dated prompt with no date in it.

    Red proof: removing the ``if not question_date`` check lets the call through.
    """
    with pytest.raises(ValueError, match="requires question_date"):
        render_dated_evidence_prompt(_bundle((_item("c1"),)), "")


def test_the_profile_setting_defaults_to_dated_keeps_plain_and_refuses_a_typo() -> None:
    """Invariant: `RECALL_REASONING_ANSWER_PROFILE` unset (or empty) is ``dated``, the default since
    2026-09-29; ``plain`` stays reachable as the opt-out; case is forgiven; an unknown value stops
    the front door with the variable named.

    Red proof: returning the raw value without the membership check returns "datd"; reverting the
    default to ``"plain"`` fails the first equality.
    """
    assert resolve_answer_profile({}) == "dated"
    assert resolve_answer_profile({"RECALL_REASONING_ANSWER_PROFILE": ""}) == "dated"
    assert resolve_answer_profile({"RECALL_REASONING_ANSWER_PROFILE": " Plain "}) == "plain"
    with pytest.raises(ValueError, match="RECALL_REASONING_ANSWER_PROFILE"):
        resolve_answer_profile({"RECALL_REASONING_ANSWER_PROFILE": "datd"})


# ------------------------------------------------------------------------------ through reason()

NOW = datetime(2026, 8, 10, tzinfo=timezone.utc)


def _result() -> TrustedResult:
    chunk = Chunk("c1", "/corpus/bike.md", "I bought a blue bike.", {"file": "bike.md", "ord": 0})
    hit = TrustedHit(
        chunk=chunk, cosine=0.91, confidence=0.97, verdict="ok",
        provenance=Provenance(chunk.source, "bike.md", 0, NOW), validity=Validity(None, None, None),
    )
    return TrustedResult(
        query="when did I buy the bike?", hits=[hit], abstained=False, reason="", gap_warning=False,
        staleness=StalenessReport(False, NOW, timedelta(0), timedelta(days=1)),
        diagnostics=RetrievalDiagnostics(embedding_profile="e", retrieval_profile="quality",
                                         index_generation="gen_1", stage_ms={}),
        calibration_id="cal-1", calibration_status="certified", tenant_id="acme", generation_id="gen_1",
        pipeline_fingerprint="pipe-a", corpus_fingerprint="corpus-a", query_set_digest="query-a",
    )


def _request(profile: str, seen: list[tuple[str, str]], as_of: datetime | None) -> ReasoningRequest:
    def answer(system: str, user: str) -> dict:
        seen.append((system, user))
        return {"answer": "a blue bike", "citations": ["c1"], "insufficient_evidence": False}

    return ReasoningRequest(
        query="when did I buy the bike?", tenant_id="acme",
        generation=GenerationSelection(generation_id="gen_1", pipeline_fingerprint="pipe-a",
                                       corpus_fingerprint="corpus-a"),
        providers=ReasoningProviderPorts(retriever=lambda _r: _result(), answer_provider=answer),
        policy=ReasoningPolicy(), as_of=as_of, answer_profile=profile,
    )


def test_reason_renders_the_requested_profile_and_records_it() -> None:
    """Invariant: `reason` sends the provider the profile the request names, dates the question by
    `as_of` when pinned, and records the profile in the response, which survives a round trip.

    Red proof: making the plain branch unconditional in `_answer_from_evidence` (``if True:``)
    sends the plain prompt and fails the system equality.
    """
    seen: list[tuple[str, str]] = []
    pinned = datetime(2023, 5, 30, 20, 36, tzinfo=timezone.utc)
    response = reason(_request("dated", seen, pinned))
    system, user = seen[0]
    assert system == SYSTEM_PROMPT + DATED_READER_CONTRACT
    assert _payload(user)["question_date"] == pinned.isoformat()
    assert response.outcome == "answered" and response.diagnostics.answer_profile == "dated"
    decoded = reasoning_response_from_dict(json.loads(json.dumps(response.to_dict())))
    assert decoded.diagnostics.answer_profile == "dated"


def test_reason_defaults_to_dated_and_plain_is_the_opt_out() -> None:
    """Invariant: a request that names no profile gets the dated prompt, dated by the moment of the
    call when no `as_of` is pinned; naming ``plain`` gets the previous prompt unchanged.

    Red proof: reverting `ReasoningRequest.answer_profile`'s default to ``"plain"`` sends the plain
    prompt and fails the first system equality.
    """
    seen: list[tuple[str, str]] = []
    request = _request("plain", seen, None)
    request.answer_profile = ReasoningRequest.__dataclass_fields__["answer_profile"].default
    before = datetime.now(timezone.utc)
    reason(request)
    assert seen[0][0] == SYSTEM_PROMPT + DATED_READER_CONTRACT
    assert datetime.fromisoformat(_payload(seen[0][1])["question_date"]) >= before
    reason(_request("plain", seen, None))
    assert seen[1][0] == SYSTEM_PROMPT


def test_an_old_payload_reads_as_plain() -> None:
    """Invariant: a serialized response from before this field existed deserializes as ``plain``,
    because every such response was produced by the plain prompt, whatever the default is now.

    Red proof: deserializing with ``diagnostics_payload["answer_profile"]`` (no default) raises
    KeyError on the old payload.
    """
    seen: list[tuple[str, str]] = []
    response = reason(_request("plain", seen, None))
    payload = response.to_dict()
    del payload["diagnostics"]["answer_profile"]
    assert reasoning_response_from_dict(payload).diagnostics.answer_profile == "plain"


def test_a_request_with_an_unknown_profile_is_refused() -> None:
    """Invariant: the library refuses an unknown profile at the request, before any retrieval.

    Red proof: removing the ``answer_profile not in ANSWER_PROFILES`` check constructs the request.
    """
    with pytest.raises(ValueError, match="answer_profile"):
        _request("datd", [], None)


# ------------------------------------------------------------------------------ both front doors


def _calls_with_keyword(path: Path, function: str, keyword: str) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", getattr(node.func, "attr", None)) == function
        and any(k.arg == keyword for k in node.keywords)
    )


def test_both_front_doors_resolve_and_pass_the_profile() -> None:
    """Invariant: the MCP server and the CLI both resolve the setting and pass it to
    `reasoning_query`; a facility reachable from one of two front doors is the defect PR 553 names.

    Red proof: deleting ``answer_profile=answer_profile`` from the CLI's `reasoning_query` call
    fails the CLI count.
    """
    server = ROOT / "recall_mcp" / "server.py"
    cli = ROOT / "recall" / "cli_commands" / "reasoning_cmd.py"
    assert _calls_with_keyword(server, "reasoning_query", "answer_profile") >= 1
    assert _calls_with_keyword(cli, "reasoning_query", "answer_profile") >= 1
    assert "resolve_answer_profile(runtime_env)" in server.read_text(encoding="utf-8")
    assert "resolve_answer_profile()" in cli.read_text(encoding="utf-8")
