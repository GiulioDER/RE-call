"""The opt-in ``dated-premise`` answer profile of `recall_reasoning_query`.

Measured before it was added (the maintainer's research log, SP-2, 2026-10-23): on 400 STALE
instances (arXiv 2605.06527) with a declared ``supersedes`` and successor expansion, stale-premise
queries passed 149/400 against 0/400 for ``dated``; with the contract and no structural notice,
1/400, because the reader wrote corrections citing nothing; on 200 FactConsolidation questions the
profile scored 190 against 176. These tests pin what ships to what was measured, keep ``dated``
unchanged, and hold the outcome, the envelope and the notice to their rules. Each names its red
proof; the proof is recorded with the change that added this file.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone

import pytest

from recall import answer_provider
from recall.answer_provider import OllamaAnswerProvider, resolve_answer_profile
from recall.evidence import (
    DATED_PREMISE_SYSTEM_PROMPT,
    DATED_READER_CONTRACT,
    DATED_SYSTEM_PROMPT,
    EVIDENCE_CLOSE,
    EVIDENCE_OPEN,
    PREMISE_READER_CONTRACT,
    SYSTEM_PROMPT,
    AnswerEnvelope,
    EvidenceBundle,
    EvidenceItem,
    EvidenceValidationError,
    ground_premise_correction,
    normalize_citations,
    parse_premise_answer_envelope,
    render_dated_evidence_prompt,
    render_dated_premise_prompt,
    stale_premise_notice,
    validate_premise_answer,
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
    Provenance,
    RetrievalDiagnostics,
    StalenessReport,
    TrustedHit,
    TrustedResult,
    Validity,
)

#: sha256 of the contract SP-2 measured (`PREMISE_READER_CONTRACT` in its harness).
MEASURED_PREMISE_CONTRACT_SHA256 = "d18f6dc2280f5e894042b111b137098d4f9b5728751a8482cfb5b5ce61b8b6c3"

NOW = datetime(2026, 8, 10, tzinfo=timezone.utc)
CORRECTION = "The question assumes the user lives in Seattle; the newest evidence says Austin."


def _item(cid: str, source: str) -> EvidenceItem:
    return EvidenceItem(
        chunk_id=cid, text=f"text {cid}", source=source, ordinal=0, indexed_at=NOW,
        valid_from=None, valid_until=None, cosine=0.9, confidence=0.9,
    )


def _bundle(items: tuple[EvidenceItem, ...], decision: str = "answer") -> EvidenceBundle:
    return EvidenceBundle(
        query="since I live in Seattle, which neighbourhood groups should I join?",
        decision=decision, reason_code=None,  # type: ignore[arg-type]
        decision_state="supported", calibrated=True, stale=False, embedding_profile="e",
        retrieval_profile="r", index_generation="g", items=items,
    )


def _hit(cid: str, file: str, verdict: str, superseded_by: str | None = None, source: str | None = None) -> TrustedHit:
    chunk = Chunk(cid, source or f"/corpus/{file}", f"text {cid}", {"file": file, "ord": 0})
    return TrustedHit(
        chunk=chunk, cosine=0.9, confidence=0.9, verdict=verdict,  # type: ignore[arg-type]
        provenance=Provenance(chunk.source, file, 0, NOW),
        validity=Validity(None, None, superseded_by),
    )


def _result(hits: list[TrustedHit]) -> TrustedResult:
    return TrustedResult(
        query="since I live in Seattle, which neighbourhood groups should I join?", hits=hits,
        abstained=False, reason="", gap_warning=False,
        staleness=StalenessReport(False, NOW, timedelta(0), timedelta(days=1)),
        diagnostics=RetrievalDiagnostics(embedding_profile="e", retrieval_profile="quality",
                                         index_generation="gen_1", stage_ms={}),
        calibration_id="cal-1", calibration_status="certified", tenant_id="acme", generation_id="gen_1",
        pipeline_fingerprint="pipe-a", corpus_fingerprint="corpus-a", query_set_digest="query-a",
    )


def _payload(user: str) -> dict:
    return json.loads(user[len(EVIDENCE_OPEN):-len(EVIDENCE_CLOSE)])


# ------------------------------------------------------------------------------ the prompt


def test_the_shipped_premise_contract_is_the_measured_one() -> None:
    """Invariant: the premise instructions are byte for byte those SP-2 measured.

    Red proof: changing "do not only decline" to "do not decline" in `PREMISE_READER_CONTRACT`
    changes the digest.
    """
    assert hashlib.sha256(PREMISE_READER_CONTRACT.encode("utf-8")).hexdigest() == MEASURED_PREMISE_CONTRACT_SHA256


def test_the_premise_prompt_extends_the_dated_one_and_names_the_notice_only_when_there_is_one() -> None:
    """Invariant: the system message is the dated one followed by the premise contract, returned by
    identity; the payload is the dated payload plus ``premise_correction`` in the schema, and
    ``superseded_notice`` appears exactly when a notice is given; the ``dated`` payload is unchanged.

    Red proof: dropping the ``if notice:`` guard in `render_dated_premise_prompt` writes an empty
    notice and fails the ``not in`` assertion.
    """
    bundle = _bundle((_item("c1", "new.md"),))
    system, user = render_dated_premise_prompt(bundle, "2024-01-01", ())
    assert system == SYSTEM_PROMPT + DATED_READER_CONTRACT + PREMISE_READER_CONTRACT
    assert system is DATED_PREMISE_SYSTEM_PROMPT
    dated = _payload(render_dated_evidence_prompt(bundle, "2024-01-01")[1])
    premise = _payload(user)
    assert "superseded_notice" not in premise
    assert premise["answer_schema"] == {**dated["answer_schema"], "premise_correction": "string or null"}
    assert {k: v for k, v in premise.items() if k != "answer_schema"} == {k: v for k, v in dated.items() if k != "answer_schema"}
    assert "premise_correction" not in dated["answer_schema"] and "superseded_notice" not in dated
    assert render_dated_evidence_prompt(bundle, "2024-01-01")[0] is DATED_SYSTEM_PROMPT
    _, with_notice = render_dated_premise_prompt(bundle, "2024-01-01", ("c1",))
    assert _payload(with_notice)["superseded_notice"] == {"replaced_by": ["c1"]}


def test_the_profile_is_reachable_through_the_front_door_setting() -> None:
    """Invariant: ``RECALL_REASONING_ANSWER_PROFILE=dated-premise`` resolves, so both front doors
    (which call `resolve_answer_profile`) can opt in; the default stays ``dated``.

    Red proof: removing ``"dated-premise"`` from `ANSWER_PROFILES` raises ValueError.
    """
    assert resolve_answer_profile({"RECALL_REASONING_ANSWER_PROFILE": "Dated-Premise"}) == "dated-premise"
    assert resolve_answer_profile({}) == "dated"


# ------------------------------------------------------------------------------ the notice


def test_the_notice_names_the_successors_evidence_and_nothing_else() -> None:
    """Invariant: the notice is the bundle items from the memo that replaced a ``superseded`` hit,
    matched whether the successor is named root-relative or by its absolute source; items from other
    memos, and successors of hits that were not superseded, are not in it.

    Red proof: dropping ``hit.verdict == "superseded"`` from `stale_premise_notice` lets the
    ok hit's (stale) pointer name other.md and fails the first equality.
    """
    bundle = _bundle((_item("c1", "notes/new.md"), _item("c2", "other.md"), _item("c3", "/root/notes/new.md")))
    result = _result([
        _hit("c1", "notes/new.md", "ok"),
        _hit("c3", "notes/new.md", "ok", source="/root/notes/new.md"),
        _hit("c2", "other.md", "ok", superseded_by="other.md"),
        _hit("c0", "old.md", "superseded", superseded_by="notes/new.md"),
    ])
    assert stale_premise_notice(result, bundle) == ("c1", "c3")
    assert stale_premise_notice(_result([_hit("c2", "other.md", "ok")]), bundle) == ()


# ------------------------------------------------------------------------------ the envelope


def test_the_premise_envelope_reads_three_keys_as_no_correction_and_refuses_anything_else() -> None:
    """Invariant: four keys parse; the three v1 keys read as ``premise_correction: None`` (the
    omission SP-2's smoke saw); an extra key, a non-string correction and a blank one are handled.

    Red proof: removing the three-key set from the accepted ``fields`` in
    `parse_premise_answer_envelope` refuses the second case.
    """
    four = {"answer": None, "citations": ["c1"], "insufficient_evidence": True, "premise_correction": CORRECTION}
    assert parse_premise_answer_envelope(json.dumps(four)).premise_correction == CORRECTION
    three = {"answer": "x", "citations": ["c1"], "insufficient_evidence": False}
    assert parse_premise_answer_envelope(three) == AnswerEnvelope("x", ("c1",), False, None)
    assert parse_premise_answer_envelope({**four, "premise_correction": "  "}).premise_correction is None
    with pytest.raises(EvidenceValidationError, match="premise answer envelope"):
        parse_premise_answer_envelope({**four, "note": "y"})
    with pytest.raises(EvidenceValidationError, match="premise_correction must be"):
        parse_premise_answer_envelope({**four, "premise_correction": 3})


def test_a_correction_must_cite_known_evidence_and_may_accompany_an_abstention() -> None:
    """Invariant: a correction with citations is valid beside an abstention or an answer; without a
    citation, or citing an unknown id, it is refused; with no correction the ``dated`` rules apply,
    so an abstention that cites is still refused.

    Red proof: deleting the ``if not envelope.citations`` check in `validate_premise_answer`
    accepts the uncited correction and fails the second assertion.
    """
    bundle = _bundle((_item("c1", "new.md"),))
    assert validate_premise_answer(AnswerEnvelope(None, ("c1",), True, CORRECTION), bundle).valid
    assert not validate_premise_answer(AnswerEnvelope(None, (), True, CORRECTION), bundle).valid
    assert not validate_premise_answer(AnswerEnvelope(None, ("c9",), True, CORRECTION), bundle).valid
    assert validate_premise_answer(AnswerEnvelope("Austin groups", ("c1",), False, CORRECTION), bundle).valid
    assert not validate_premise_answer(AnswerEnvelope(None, ("c1",), True, None), bundle).valid


def test_the_notice_grounds_only_an_uncited_correction() -> None:
    """Invariant: a correction the reader left uncited takes the notice as its citations; a cited
    correction keeps its own; with no notice nothing is added.

    Red proof: dropping ``not envelope.citations`` from `ground_premise_correction` replaces the
    reader's own citation with the notice and fails the second equality.
    """
    uncited = AnswerEnvelope(None, (), True, CORRECTION)
    assert ground_premise_correction(uncited, ("c1",)) == (AnswerEnvelope(None, ("c1",), True, CORRECTION), True)
    cited = AnswerEnvelope(None, ("c2",), True, CORRECTION)
    assert ground_premise_correction(cited, ("c1",)) == (cited, False)
    assert ground_premise_correction(uncited, ()) == (uncited, False)


def test_normalising_citations_keeps_the_correction() -> None:
    """Invariant: collapsing repeated citations does not drop ``premise_correction``.

    Red proof: rebuilding the envelope from its three original fields in `normalize_citations`
    (``AnswerEnvelope(envelope.answer, tuple(unique), envelope.insufficient_evidence)``) drops it.
    """
    envelope = normalize_citations(AnswerEnvelope(None, ("c1", "c1"), True, CORRECTION))
    assert envelope == AnswerEnvelope(None, ("c1",), True, CORRECTION)


# ------------------------------------------------------------------------------ through reason()


def _request(profile: str, provider_output: dict, hits: list[TrustedHit]) -> ReasoningRequest:
    return ReasoningRequest(
        query="since I live in Seattle, which neighbourhood groups should I join?", tenant_id="acme",
        generation=GenerationSelection(generation_id="gen_1", pipeline_fingerprint="pipe-a",
                                       corpus_fingerprint="corpus-a"),
        providers=ReasoningProviderPorts(retriever=lambda _r: _result(hits),
                                         answer_provider=lambda _s, _u: dict(provider_output)),
        policy=ReasoningPolicy(), as_of=NOW, answer_profile=profile,
    )


SUPERSEDED_AND_SUCCESSOR = [
    _hit("c1", "new.md", "ok"),
    _hit("c0", "old.md", "superseded", superseded_by="new.md"),
]


def test_reason_returns_a_grounded_correction_as_its_own_outcome_and_round_trips() -> None:
    """Invariant: under ``dated-premise`` an uncited correction with an abstention becomes outcome
    ``corrected``: no answer, the correction, the notice's chunk as its citation, and diagnostics
    that say the notice grounded it; the response survives a JSON round trip.

    Red proof: deleting the ``if envelope.premise_correction is not None:`` branch in
    `_answer_from_evidence` returns ``abstained`` and fails the outcome equality.
    """
    output = {"answer": None, "citations": [], "insufficient_evidence": True, "premise_correction": CORRECTION}
    response = reason(_request("dated-premise", output, SUPERSEDED_AND_SUCCESSOR))
    assert response.outcome == "corrected"
    assert response.answer is None and response.premise_correction == CORRECTION
    assert [c.chunk_id for c in response.citations] == ["c1"]
    assert response.diagnostics.premise_notice == ("c1",)
    assert response.diagnostics.premise_citations_from_notice is True
    decoded = reasoning_response_from_dict(json.loads(json.dumps(response.to_dict())))
    assert decoded.outcome == "corrected" and decoded.premise_correction == CORRECTION
    assert decoded.diagnostics.premise_notice == ("c1",) and decoded.diagnostics.premise_citations_from_notice


def test_reason_refuses_an_uncited_correction_when_no_notice_can_ground_it() -> None:
    """Invariant: without a superseded hit there is no notice, and a correction citing nothing is
    refused, as an answer citing nothing is.

    Red proof: deleting the ``if not envelope.citations`` check in `validate_premise_answer` lets
    the response through and fails the raises.
    """
    output = {"answer": None, "citations": [], "insufficient_evidence": True, "premise_correction": CORRECTION}
    with pytest.raises(EvidenceValidationError, match="requires at least one citation"):
        reason(_request("dated-premise", output, [_hit("c1", "new.md", "ok")]))


def test_the_dated_profile_is_unchanged_and_an_old_payload_reads_without_premise_fields() -> None:
    """Invariant: under ``dated`` the reader still gets the dated prompt and the three-key
    envelope (a fourth key is refused), the response carries no correction, and a payload
    serialized before these fields existed deserializes with their defaults.

    Red proof: reading ``payload["premise_correction"]`` without a default in
    `reasoning_response_from_dict` raises KeyError on the old payload.
    """
    plain = {"answer": "a group", "citations": ["c1"], "insufficient_evidence": False}
    response = reason(_request("dated", plain, SUPERSEDED_AND_SUCCESSOR))
    assert response.outcome == "answered" and response.premise_correction is None
    assert response.diagnostics.premise_notice == ()
    payload = response.to_dict()
    del payload["premise_correction"]
    del payload["diagnostics"]["premise_notice"]
    del payload["diagnostics"]["premise_citations_from_notice"]
    decoded = reasoning_response_from_dict(payload)
    assert decoded.premise_correction is None and decoded.diagnostics.premise_notice == ()
    with pytest.raises(EvidenceValidationError):
        reason(_request("dated", {**plain, "premise_correction": None}, SUPERSEDED_AND_SUCCESSOR))


# ------------------------------------------------------------------------------ the Ollama schema


class _Response:
    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps({"message": {"content": "{}"}, "eval_count": 1}).encode()


def test_the_native_ollama_schema_follows_the_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invariant: the native Ollama call asks for the four-key envelope exactly when the system
    message is the premise prompt, and for the three-key envelope otherwise.

    Red proof: passing ``premise=False`` unconditionally from `OllamaAnswerProvider.__call__`
    leaves ``premise_correction`` out of the first schema.
    """
    formats: list[dict] = []

    def _urlopen(req: object, *, timeout: float) -> _Response:
        formats.append(json.loads(getattr(req, "data").decode())["format"])
        return _Response()

    monkeypatch.setattr(answer_provider.request, "urlopen", _urlopen)
    provider = OllamaAnswerProvider(
        answer_provider._NativeOllamaClient("http://127.0.0.1:11434/v1", timeout=12), model_id="qwen3:4b"
    )
    provider(DATED_PREMISE_SYSTEM_PROMPT, "user")
    provider(DATED_SYSTEM_PROMPT, "user")
    assert "premise_correction" in formats[0]["properties"] and "premise_correction" in formats[0]["required"]
    assert "premise_correction" not in formats[1]["properties"]
    assert formats[1]["required"] == ["answer", "citations", "insufficient_evidence"]
