"""W1: conversation facts at Add (``recall_aml.conversation_records``), with a stub model.

Each test names the mutation of the production code it was watched to fail on (the red proof).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from recall_aml.conversation_records import (
    CONVERSATION_FACTS_SYSTEM_PROMPT,
    extract_conversation_facts,
    extraction_payload,
    normalise_key_part,
)
from recall_aml.models import Message

STAMP = datetime(2023, 5, 8, 13, 56, tzinfo=UTC)
MESSAGES = [
    Message(role="user", content="Caroline: I moved to Lisbon in March and my new job starts Monday.", timestamp=STAMP),
    Message(role="user", content="Melanie: I have never been to Portugal, but I love grilled sardines.", timestamp=STAMP),
]


class StubCompiler:
    def __init__(self, facts: list[dict[str, Any]]) -> None:
        self.facts = facts
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def json_object(self, system: str, payload: dict[str, Any], **_: Any) -> dict[str, Any]:
        self.calls.append((system, payload))
        return {"facts": self.facts}


def _ids() -> list[str]:
    payload, anchors = extraction_payload(MESSAGES, "s1", [])
    return [a.id for a in anchors]


def _fact(**overrides: Any) -> dict[str, Any]:
    base = {
        "anchors": [_ids()[0]], "speaker": "Caroline", "subject": "Caroline", "attribute": "Home city",
        "value": "I moved to Lisbon in March", "relation": "state", "event_date": "2023-03", "sensitive": False,
    }
    return {**base, **overrides}


def test_a_grounded_fact_is_kept_with_its_key_speaker_dates_and_source() -> None:
    """Invariant: a fact whose value is in a cited anchor is kept, keyed ``subject|attribute`` in
    normalised form, with the anchor's timestamp as its mention time.

    Red proof: building the key from the raw ``subject`` and ``attribute`` (no
    `normalise_key_part`) gives ``Caroline|Home city`` and fails the key assertion.
    """
    result = extract_conversation_facts(StubCompiler([_fact()]), MESSAGES, "s1", [])
    assert len(result.facts) == 1
    fact = result.facts[0]
    assert fact.key == "caroline|home city"
    assert (fact.relation, fact.event_date, fact.mention_time, fact.message_ordinals) == ("state", "2023-03", STAMP, (0,))
    assert fact.rendered() == '[fact · caroline / home city · state on 2023-03] Caroline: "I moved to Lisbon in March"'


def test_a_value_not_in_the_cited_evidence_is_dropped() -> None:
    """Invariant: the model cannot put words in a speaker's mouth: the value must occur in a cited
    anchor (the second anchor holds Melanie's line, not this value).

    Red proof: replacing the verbatim check in `validate_facts` with ``True`` keeps the invented
    value and fails the count.
    """
    invented = _fact(value="I moved to Madrid in March")
    elsewhere = _fact(anchors=[_ids()[1]])
    result = extract_conversation_facts(StubCompiler([invented, elsewhere]), MESSAGES, "s1", [])
    assert result.facts == []
    assert result.dropped == {"value_not_in_evidence": 2}


def test_unknown_anchors_bad_relations_and_bad_dates_are_refused_and_bare_ids_resolve() -> None:
    """Invariant: an anchor that was not sent and a relation outside the set drop the fact; a
    malformed date is cleared, not kept; a bare ``a000`` resolves to the sent anchor.

    Red proof: accepting any relation (dropping the `RELATIONS` check) keeps the ``opinion`` fact
    and fails the dropped counts.
    """
    facts = [
        _fact(anchors=["a999_0000000000000000"]),
        _fact(relation="opinion"),
        _fact(anchors=["a000"], event_date="March"),
    ]
    result = extract_conversation_facts(StubCompiler(facts), MESSAGES, "s1", [])
    assert result.dropped == {"no_known_anchor": 1, "unknown_relation": 1}
    assert len(result.facts) == 1 and result.facts[0].event_date is None


def test_keys_normalise_so_the_same_slot_matches_across_adds() -> None:
    """Invariant: case, punctuation and one leading article or possessive do not make a new key.

    Red proof: dropping the `_ARTICLE` removal in `normalise_key_part` makes "my job" differ from
    "job" and fails the equality.
    """
    assert normalise_key_part("  My  Job!") == normalise_key_part("job") == "job"
    assert normalise_key_part("The trip to Japan") == "trip to japan"


def test_the_call_carries_the_known_keys_and_the_conversation_prompt() -> None:
    """Invariant: the model is sent the tenant's known keys (deduplicated, most recent kept) with
    the anchors, under the conversation prompt, so it can reuse a key.

    Red proof: omitting ``known_keys`` from `extraction_payload` fails the payload assertion.
    """
    stub = StubCompiler([])
    extract_conversation_facts(stub, MESSAGES, "s1", ["caroline|job", "caroline|job", "melanie|food"])
    system, payload = stub.calls[0]
    assert system == CONVERSATION_FACTS_SYSTEM_PROMPT
    assert payload.get("known_keys") == ["caroline|job", "melanie|food"]
    assert [a["role"] for a in payload["anchors"]] == ["user", "user"]
