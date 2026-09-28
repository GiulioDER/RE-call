"""W5: sensitive masking in place and preference pinning. Each test names its red proof."""

from __future__ import annotations

from datetime import UTC, datetime

from recall_aml.conversation_records import ConversationFact
from recall_aml.models import SearchItem
from recall_aml.preference_pin import is_advice_query, rendered_preference, select_preferences
from recall_aml.sensitive_mask import mask_text, masked_items


def test_identifiers_are_replaced_in_place_by_typed_placeholders() -> None:
    """Invariant: each identifier becomes its placeholder and the rest of the sentence stays.

    Red proof: dropping the email pattern from `mask_text` leaves the address and fails the first
    equality.
    """
    assert mask_text("mail me at jane.doe@example.com today")[0] == "mail me at [EMAIL] today"
    assert mask_text("my card is 4111 1111 1111 1111 ok")[0] == "my card is [CARD NUMBER] ok"
    assert mask_text("ssn 123-45-6789, call +1 415-555-0132")[0] == "ssn [SSN], call [PHONE]"
    assert mask_text("key sk-abcdefghijklmnopqrstuvwxyz123")[0] == "key [API KEY]"
    assert mask_text("passport number: X1234567")[0] == "passport number: [PASSPORT NUMBER]"
    assert mask_text("I live at 221 Baker Street now")[0] == "I live at [STREET ADDRESS] now"


def test_look_alikes_and_health_details_are_left_alone() -> None:
    """Invariant: a number failing the Luhn check, digit groups with no phone shape, versions,
    dates and health or therapy details are not masked (E3 rewards using the latter).

    Red proof: accepting any 13 to 19 digit run as a card (dropping `luhn_valid`) masks the order
    number and fails the equality.
    """
    for text in (
        "order 1234 5678 9012 3456 shipped",
        "version 3.10.2 released on 2024-03-15",
        "I have anxiety and my therapist suggested sertraline",
    ):
        assert mask_text(text)[0] == text, text


def test_masked_items_keep_every_item_and_count_what_was_masked() -> None:
    """Invariant: masking never drops an item (the reader still needs the context); it counts per
    type.

    Red proof: filtering out items that contained an identifier in `masked_items` loses the first
    item and fails the length assertion.
    """
    items = [
        SearchItem(id="a", content="send it to jane@example.com", source="r", session_id="s", kind="raw", score=1.0),
        SearchItem(id="b", content="nothing here", source="r", session_id="s", kind="raw", score=0.5),
    ]
    out, counts = masked_items(items)
    assert [i.id for i in out] == ["a", "b"] and out[0].content == "send it to [EMAIL]" and out[1] is items[1]
    assert counts == {"EMAIL": 1}


def _pref(value: str, day: int, key: str, relation: str = "preference") -> ConversationFact:
    subject, _, attribute = key.partition("|")
    return ConversationFact(
        key=key, subject=subject, attribute=attribute, value=value, relation=relation, speaker="user",
        event_date=None, mention_time=datetime(2024, 3, day, tzinfo=UTC), anchor_ids=("a",),
        message_ordinals=(0,), sensitive=False,
    )


def test_advice_questions_pin_matching_preferences_newest_per_key() -> None:
    """Invariant: an advice question selects preference facts that share words with it, one per
    key at its newest statement; other relations and unrelated preferences are not pinned.

    Red proof: keeping the FIRST statement per key in `select_preferences` (``>`` flipped to
    ``<``) pins the old spicy-food preference and fails the equality.
    """
    facts = [
        _pref("I love spicy food", 1, "user|food"),
        _pref("I have gone off spicy food, mild dishes now", 9, "user|food"),
        _pref("I prefer window seats on flights", 5, "user|travel seat"),
        _pref("I ate spicy food yesterday", 4, "user|dinner", relation="event"),
    ]
    assert is_advice_query("Can you recommend a restaurant with food I would enjoy?")
    assert not is_advice_query("When did I move to Lisbon?")
    chosen = select_preferences(facts, "Can you recommend a restaurant with food I would enjoy?")
    assert [f.value for f in chosen] == ["I have gone off spicy food, mild dishes now"]
    assert rendered_preference(chosen[0]) == 'user stated (2024-03-09): "I have gone off spicy food, mild dishes now"'


def test_the_service_masks_only_when_asked_and_c9_is_unchanged(monkeypatch) -> None:
    """Invariant: with ``RECALL_AML_SENSITIVE_MASK=1`` a returned item's email is masked and the
    profile says so; the served C9 has masking off.

    Red proof: dropping the `masked_items` call from `HostedService.search` returns the raw email
    and fails the ``[EMAIL]`` assertion.
    """
    from recall_aml.variants import variant
    from tests.test_aml_relative_dates_and_adjacency import CONTEXT_QUERY, _add, _search
    from tests.test_aml_specialist_fusion import _service

    assert variant("C9_routed_specialists_grounded_graph_atomic").sensitive_masking is False
    monkeypatch.setenv("RECALL_AML_SENSITIVE_MASK", "1")
    service, _, _, _, _ = _service("C7_routed_specialists")
    _add(service, "w5-user", "w5-a", "I adopted a puppy, write to jane.doe@example.com about it.", 2)
    response = _search(service, "w5-user", CONTEXT_QUERY)
    assert any("[EMAIL]" in str(item.content) for item in response.data)
    assert not any("jane.doe@example.com" in str(item.content) for item in response.data)
    assert "+sensitive-masked-v1" in service.search_content_profile
