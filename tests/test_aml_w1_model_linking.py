"""W1 model-side key linking: the closed-list extraction (arm A) and the Add-time key matcher
(arm B). Each test names its red proof."""

from __future__ import annotations

from datetime import UTC, datetime
import sys
from pathlib import Path

from recall_aml.compiler import EvidenceAnchor
from recall_aml.conversation_records import closed_known_keys, validate_facts
from recall_aml.key_matching import match_payload, validate_matches

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def _anchor(anchor_id: str, quote: str) -> EvidenceAnchor:
    return EvidenceAnchor(
        id=anchor_id, message_ordinal=0, start=0, end=len(quote), quote=quote, role="user",
        timestamp=datetime(2024, 4, 2, tzinfo=UTC), exact_code_tokens=(),
    )


def test_a_closed_list_fact_takes_the_cited_key_and_an_unsent_id_is_dropped() -> None:
    """Invariant: a fact with a sent ``key_id`` takes that key exactly (whatever subject and
    attribute it wrote), an unsent id drops the fact, and ``key_id`` null coins a key as before.

    Red proof: ignoring ``key_id`` in `validate_facts` coins "zoom call with the director|time"
    for the first fact and fails the first equality.
    """
    anchors = [_anchor("a1", "the call moved to April 22 at 11 AM"), _anchor("a2", "I love hiking")]
    raw = {"facts": [
        {"key_id": "k2", "anchors": ["a1"], "subject": "zoom call with the director", "attribute": "time",
         "value": "April 22 at 11 AM", "relation": "state"},
        {"key_id": "k9", "anchors": ["a1"], "value": "April 22", "relation": "state"},
        {"key_id": None, "anchors": ["a2"], "subject": "user", "attribute": "hobby", "value": "hiking",
         "relation": "preference"},
    ]}
    known = {"k1": "user|home city", "k2": "zoom call|date and time"}
    out = validate_facts(raw, anchors, known)
    assert [f.key for f in out.facts] == ["zoom call|date and time", "user|hobby"]
    assert out.dropped == {"unknown_key_id": 1}


def test_the_closed_list_sends_the_most_recent_keys_with_their_latest_value() -> None:
    """Invariant: past the cap, the MOST RECENTLY stated keys are sent, each numbered with its
    latest value, and the id map matches the numbering.

    Red proof: keeping the first `MAX_KNOWN_KEYS` instead of the last in `closed_known_keys`
    sends the oldest keys and fails the first equality.
    """
    latest = [(f"thing {i}|size", f"{i} cm") for i in range(205)]
    entries, ids = closed_known_keys(latest)
    assert entries[0] == {"id": "k1", "key": "thing 5 | size", "latest_value": "5 cm"}
    assert len(entries) == 200 and ids["k200"] == "thing 204|size"


def test_matches_naming_unsent_ids_are_ignored() -> None:
    """Invariant: only a match between a sent new id and a sent existing id is kept; a new key
    maps onto at most one existing key (the first answer).

    Red proof: accepting any existing id in `validate_matches` (dropping the membership check on
    ``existing_ids``, looking the id up with ``.get``) maps "n2" onto the never-sent "k7" and fails
    the equality.
    """
    payload, new_ids, existing_ids = match_payload(
        [("zoom call with the director|time", "April 22"), ("user|hobby", "hiking")],
        [("zoom call|date and time", "April 21")],
    )
    assert payload["existing"] == [{"id": "k1", "key": "zoom call | date and time", "latest_value": "April 21"}]
    raw = {"matches": [
        {"new": "n1", "existing": "k1"}, {"new": "n1", "existing": None},
        {"new": "n2", "existing": "k7"}, {"new": "n9", "existing": "k1"},
    ]}
    assert validate_matches(raw, new_ids, {**existing_ids}) == {"zoom call with the director|time": "zoom call|date and time"}


def test_a_relation_match_merges_only_when_the_relation_is_a_same_property_one() -> None:
    """Invariant (fix C): a match merges only when its stated relation is restates, updates or
    contradicts; "related", "different" or a missing relation never merges, even with an id.

    Red proof: dropping the relation check in `validate_relation_matches` merges the "related"
    pair and fails the equality.
    """
    from recall_aml.key_matching import validate_relation_matches

    new_ids = {"n1": "car|asking price", "n2": "car|mileage", "n3": "car|colour"}
    existing_ids = {"k1": "car|price"}
    raw = {"matches": [
        {"new": "n1", "relation": "updates", "existing": "k1"},
        {"new": "n2", "relation": "related", "existing": "k1"},
        {"new": "n3", "existing": "k1"},
    ]}
    assert validate_relation_matches(raw, new_ids, existing_ids) == {"car|asking price": "car|price"}


def test_the_verifier_confirms_only_an_explicit_true_for_a_sent_pair() -> None:
    """Invariant (fix D): a proposed merge survives only on an explicit ``"same": true`` for a
    pair that was sent; a string "false", a missing verdict or an unsent pair id rejects it.

    Red proof: accepting any truthy ``same`` (dropping ``is True`` in `validate_verdicts`) lets
    the string "false" confirm p2 and fails the equality.
    """
    from recall_aml.key_matching import validate_verdicts

    pair_ids = {"p1": "car|asking price", "p2": "car|mileage", "p3": "car|colour"}
    raw = {"verdicts": [
        {"pair": "p1", "same": True}, {"pair": "p2", "same": "false"}, {"pair": "p9", "same": True},
    ]}
    assert validate_verdicts(raw, pair_ids) == {"car|asking price"}


def test_the_verified_matcher_drops_every_merge_the_verifier_did_not_confirm() -> None:
    """Invariant (fix D): the plain matcher's proposals pass only if the second call confirms them,
    and the verifier sees both values for each proposed pair.

    Red proof: returning the proposals unfiltered from `match_new_keys_verified` keeps the mileage
    merge and fails the equality.
    """
    from recall_aml.key_matching import MERGE_VERIFIER_SYSTEM_PROMPT, match_new_keys_verified

    sent: list[dict] = []

    class Compiler:
        def json_object(self, system, payload, **_):
            sent.append(dict(payload))
            if system == MERGE_VERIFIER_SYSTEM_PROMPT:
                return {"verdicts": [{"pair": "p1", "same": True}, {"pair": "p2", "same": False}]}
            return {"matches": [{"new": "n1", "existing": "k1"}, {"new": "n2", "existing": "k1"}]}

    out = match_new_keys_verified(
        Compiler(), [("car|asking price", "$9,000"), ("car|mileage", "80,000 km")], [("car|price", "$9,500")]
    )
    assert out == {"car|asking price": "car|price"}
    assert sent[1]["pairs"][1]["new"] == {"key": "car | mileage", "value": "80,000 km"}
    assert sent[1]["pairs"][1]["existing"] == {"key": "car | price", "latest_value": "$9,500"}


def test_the_replay_sends_only_unseen_keys_and_maps_every_later_occurrence() -> None:
    """Invariant: the matcher sees each extracted key once, the first time it appears, together
    with the existing canonical keys and their latest values; a matched key takes the existing key
    in every later Add too, and keeps its extracted key as ``raw_key``.

    Red proof: sending every key of an Add to the matcher (dropping the ``not in canonical``
    condition in `replay`) re-sends the matched key in the third Add and fails the calls equality.
    """
    from aml_w1_key_match import cluster_width, replay

    calls: list[tuple[list, list]] = []

    def matcher(new, existing):
        calls.append((list(new), list(existing)))
        return {"zoom call with the director|time": "zoom call|date and time"} if any(
            k == "zoom call with the director|time" for k, _ in new) else {}

    records = [
        {"add": "c0:a0", "conversation": 0, "facts": [{"key": "zoom call|date and time", "value": "April 21"}]},
        {"add": "c0:a1", "conversation": 0, "facts": [{"key": "zoom call with the director|time", "value": "April 22"}]},
        {"add": "c0:a2", "conversation": 0, "facts": [{"key": "zoom call with the director|time", "value": "April 23"}]},
    ]
    out = replay(records, matcher)
    assert calls == [([("zoom call with the director|time", "April 22")], [("zoom call|date and time", "April 21")])]
    assert [f["key"] for r in out for f in r["facts"]] == ["zoom call|date and time"] * 3
    assert out[2]["facts"][0]["raw_key"] == "zoom call with the director|time"
    assert cluster_width(out) == (2, 1)


def test_remember_latest_moves_a_restated_key_to_the_recent_end() -> None:
    """Invariant: a key stated again takes its new value AND moves to the most recent end, so the
    closed list keeps a slot the conversation is still talking about when it has to cut.

    Red proof: updating the value in place (dropping the ``pop`` in `remember_latest`) leaves the
    key first and fails the equality.
    """
    from aml_w1_link_test import remember_latest

    latest = {"a|x": "1", "b|y": "2"}
    remember_latest(latest, [{"key": "a|x", "value": "3"}])
    assert list(latest.items()) == [("b|y", "2"), ("a|x", "3")]
