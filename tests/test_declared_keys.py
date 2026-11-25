"""Declared fact keys (`recall.declared_keys`): the subject lookup, rule R(a'), the receipt's key part.

Invariants and the failure each one catches:
- K1 `subject_view` keeps a subject's facts and contested groups whatever their case and spacing,
  and only the predicate asked for when one is: a lookup that misses "USER " for "user" sends the
  writer to a second spelling, which is a second key.
- K2 `subject_view` lists subjects that contain or are contained in the one asked for, and never
  the one asked for: the writer's only way to tell one entity spelled two ways from two entities.
- K3 `successor_pairs` links each statement of a key to the one of the latest earlier start,
  whatever the value: Valve, Valve, Epic links 1 to 2 and 2 to 3. The first wording ("another
  value") left 1 unlinked, and it stayed in force against 3 (recall-lab WK-1, 2026-10-23).
- K4 `successor_pairs` never orders an undated statement, two of one start, or two values of a
  set-valued predicate.
- K5 `key_receipt` says `current` for a key only this memo holds, `contested` with the proposal
  R(a') makes when an older dated memo holds another value, the proposal reversed for a backdated
  memo, a note instead of a proposal when the memos are undated, and `not_single_valued` or
  `unchecked` where no answer can be given.
- K6 `settle_target` names the older memo of rule R(a') for a memo not yet written, nothing when
  nothing is in force or the memo is backdated, and declines an undated memo, two older memos of
  one start, an unserved listing, a declared `set`, and an `email` fact declared `single`.

Red proof, 2026-10-23, on the remote test host, each mutation applied alone and its test run alone,
then restored and the whole file green (recorded in the pull request):
- M1 (K1) `subject_name` without `casefold()`: the "USER " lookup lists no fact.
- M2 (K2) `subject_view`'s look-alike test without `folded != wanted`: the subject itself is listed.
- M3 (K3) `successor_pairs` pairing only statements whose objects differ: (1, 2) is missing.
- M4 (K4) `successor_pairs` dropping the equal-start grouping (`groupby` replaced by one group per
  statement): the same-day pair is ordered.
- M5 (K5) `key_receipt`'s proposal with `superseded` and `superseder` swapped, the direction error
  `recall.reasoning_proposals._extracted` warns about: the proposal names pref 2 as superseded.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from recall.declared_keys import (
    STATED_KEY_SUCCESSOR_RULE,
    KeyedMemo,
    key_receipt,
    settle_target,
    subject_name,
    subject_view,
    successor_pairs,
)
from recall.types import AtomicFact


def _fact(subject: str, predicate: str, obj: str, day: int | None = None) -> AtomicFact:
    start = datetime(2026, 1, day, tzinfo=UTC) if day else None
    return AtomicFact("memory", subject, predicate, obj, valid_from=start)


def _standing(fact: AtomicFact, *sources: str) -> dict[str, Any]:
    return {"fact_id": f"fact_{fact.object}", "fact": fact.to_payload(), "sources": list(sources),
            "evidence_card_ids": [], "asserted_at": "2026-01-01T00:00:00+00:00"}


def _contested(*values: tuple[AtomicFact, str]) -> dict[str, Any]:
    first = values[0][0]
    return {"namespace": first.namespace, "subject": first.subject, "predicate": first.predicate,
            "context": first.context,
            "values": [{"fact": f.to_payload(), "fact_id": f"fact_{f.object}", "sources": [s],
                        "evidence_card_ids": []} for f, s in values]}


def _listing(facts: list[dict[str, Any]], contested: list[dict[str, Any]] | None = None,
             served: str = "ok") -> dict[str, Any]:
    return {"as_of": "2026-02-01T00:00:00+00:00", "facts": facts, "contested": contested or [],
            "served_state": served, "served_lookups_failed": []}


def test_the_lookup_matches_a_subject_whatever_its_case_and_spacing() -> None:
    """K1."""
    listing = _listing(
        [_standing(_fact("user", "drink", "tea"), "m/a.md"), _standing(_fact("user", "city", "Lyon"), "m/b.md"),
         _standing(_fact("Fatah", "chairperson", "Mahmoud Abbas"), "m/c.md")],
        [_contested((_fact("user", "employer", "ACME"), "m/d.md"), (_fact("user", "employer", "Initech"), "m/e.md"))],
    )
    view = subject_view(listing, "USER  ")
    assert sorted(f["fact"]["predicate"] for f in view["facts"]) == ["city", "drink"], view["facts"]
    assert [g["predicate"] for g in view["contested"]] == ["employer"], view["contested"]
    narrowed = subject_view(listing, "user", "drink")
    assert [f["fact"]["object"] for f in narrowed["facts"]] == ["tea"] and narrowed["contested"] == [], narrowed
    assert view["served_state"] == "ok" and view["subject"] == "USER  ", "the listing's other fields were dropped"
    assert subject_name("  TÉnaiste ") == subject_name("ténaiste"), "the fold is not NFC and case blind"


def test_the_lookup_lists_look_alike_subjects_and_never_the_one_asked_for() -> None:
    """K2."""
    listing = _listing([
        _standing(_fact("Fatah", "chairperson", "Mahmoud Abbas"), "m/a.md"),
        _standing(_fact("Fatah movement", "founded by", "Yasser Arafat"), "m/b.md"),
        _standing(_fact("Hamas", "chairperson", "X"), "m/c.md"),
    ])
    view = subject_view(listing, "fatah")
    assert [f["fact"]["subject"] for f in view["facts"]] == ["Fatah"], view["facts"]
    assert view["similar_subjects"] == [{"subject": "Fatah movement", "predicate": "founded by"}], view["similar_subjects"]
    assert subject_view(listing, "The Fatah movement")["similar_subjects"] == [
        {"subject": "Fatah movement", "predicate": "founded by"}, {"subject": "Fatah", "predicate": "chairperson"},
    ], "a subject contained in the one asked for is a look-alike too, longest first"


def test_rule_r_a_prime_links_the_latest_earlier_statement_whatever_the_value() -> None:
    """K3."""
    first = KeyedMemo("m/1.md", _fact("Portal", "developer", "Valve"), datetime(2026, 1, 1, tzinfo=UTC))
    second = KeyedMemo("m/2.md", _fact("Portal", "developer", "Valve"), datetime(2026, 1, 2, tzinfo=UTC))
    third = KeyedMemo("m/3.md", _fact("Portal", "developer", "Epic Games"), datetime(2026, 1, 3, tzinfo=UTC))
    other = KeyedMemo("m/4.md", _fact("Portal", "genre", "puzzle"), datetime(2026, 1, 4, tzinfo=UTC))
    pairs = [(o.name, n.name) for o, n in successor_pairs([third, other, first, second])]
    assert pairs == [("m/1.md", "m/2.md"), ("m/2.md", "m/3.md")], pairs


def test_rule_r_a_prime_never_orders_undated_same_day_or_set_valued_statements() -> None:
    """K4."""
    day = datetime(2026, 1, 1, tzinfo=UTC)
    undated = KeyedMemo("m/u.md", _fact("user", "city", "Paris"), None)
    a = KeyedMemo("m/a.md", _fact("user", "city", "Lyon"), day)
    b = KeyedMemo("m/b.md", _fact("user", "city", "Turin"), day)
    assert successor_pairs([undated, a, b]) == [], "an undated or a same-day statement was ordered"
    one = KeyedMemo("m/e1.md", _fact("user", "email", "a@example.org"), day)
    two = KeyedMemo("m/e2.md", _fact("user", "email", "b@example.org"), datetime(2026, 1, 2, tzinfo=UTC))
    assert successor_pairs([one, two]) == [], "two values of a set-valued predicate were ordered"


def test_the_receipt_names_the_key_its_state_and_the_edge_to_settle() -> None:
    """K5."""
    tea = _fact("user", "drink", "tea", 1)
    coffee = _fact("user", "drink", "coffee", 2)
    alone = key_receipt(_listing([_standing(tea, "m/1.md")]), "m/1.md", tea, name_of=lambda n: n[2:-3])
    assert alone["state"] == "current" and alone["proposals"] == [], alone
    assert alone["in_force"] == [{"value": "tea", "valid_from": tea.to_payload()["valid_from"], "memos": ["1"]}], alone
    both = _listing([], [_contested((tea, "m/1.md"), (coffee, "m/2.md"))])
    later = key_receipt(both, "m/2.md", coffee, name_of=lambda n: n[2:-3])
    assert later["state"] == "contested", later
    assert later["proposals"] == [{"rule_id": STATED_KEY_SUCCESSOR_RULE, "status": "requires_review",
                                   "relation": "supersedes", "superseded": "1", "superseder": "2"}], later
    backdated = key_receipt(both, "m/1.md", tea, name_of=lambda n: n[2:-3])
    assert [(p["superseded"], p["superseder"]) for p in backdated["proposals"]] == [("1", "2")], backdated
    undated = _listing([], [_contested((_fact("user", "drink", "tea"), "m/1.md"), (_fact("user", "drink", "milk"), "m/2.md"))])
    noted = key_receipt(undated, "m/2.md", _fact("user", "drink", "milk"))
    assert noted["state"] == "contested" and noted["proposals"] == [] and "supersedes" in noted["note"], noted
    email = _fact("user", "email", "a@example.org", 1)
    assert key_receipt(_listing([_standing(email, "m/1.md")]), "m/1.md", email)["state"] == "not_single_valued"
    assert key_receipt(_listing([], served="degraded"), "m/1.md", tea)["state"] == "unchecked"
    hidden = key_receipt(_listing([_standing(coffee, "m/2.md")]), "m/1.md", tea)
    assert hidden["state"] == "not_in_force", hidden


def test_the_settle_target_is_the_rules_older_memo_and_the_table_decides_cardinality() -> None:
    """K6. `settle_target` names the memo rule R(a') gives a memo not yet written, and otherwise
    why the pair stays a proposal. An `email` fact declared `single` is still declined: G1 reads
    RE-call's table, not the declaration, which a capable writer got wrong for email, phone number
    and nickname every time (the maintainer's private study CD-1, 2026-11-25).
    """
    tea, coffee = _fact("user", "drink", "tea", 1), _fact("user", "drink", "coffee", 2)
    one = _listing([_standing(tea, "m/1.md")])
    assert settle_target(one, coffee, None) == ("m/1.md", None)
    assert settle_target(one, _fact("user", "drink", "coffee"), datetime(2026, 1, 2, tzinfo=UTC)) == ("m/1.md", None)
    assert settle_target(_listing([]), coffee, None) == (None, None), "nothing in force, nothing to settle"
    assert settle_target(one, _fact("user", "drink", "coffee"), None) == (None, "undated")
    early = AtomicFact("memory", "user", "drink", "milk", valid_from=datetime(2025, 12, 1, tzinfo=UTC))
    assert settle_target(one, early, None) == (None, None), "a backdated memo supersedes nothing"
    declared = AtomicFact("memory", "user", "email", "b@example.org", valid_from=datetime(2026, 1, 2, tzinfo=UTC),
                          cardinality="single")
    email = AtomicFact("memory", "user", "email", "a@example.org", valid_from=datetime(2026, 1, 1, tzinfo=UTC),
                       cardinality="single")
    assert settle_target(_listing([_standing(email, "m/1.md")]), declared, None) == (None, "not_single_in_table")
    as_set = AtomicFact("memory", "user", "drink", "milk", valid_from=datetime(2026, 1, 2, tzinfo=UTC), cardinality="set")
    assert settle_target(one, as_set, None) == (None, "not_single_valued")
    same_day = _listing([], [_contested((tea, "m/1.md"), (_fact("user", "drink", "milk", 1), "m/2.md"))])
    assert settle_target(same_day, coffee, None) == (None, "several_older_memos")
    assert settle_target(_listing([_standing(tea, "m/1.md")], served="degraded"), coffee, None) == (None, "unchecked")


def test_settle_declines_while_two_earlier_memos_hold_the_key() -> None:
    """K7. With two earlier values of a key in force, `settle_target` declines `several_older_memos`
    whatever their starts: rule R(a') pairs a new memo with the latest earlier start only, so
    settling it would leave the older value in force beside the new one, the key still contested.

    Red proof (CCA run 20261125T120125Z, DOC-003), on the remote test host: against the pre-fix
    `recall.declared_keys.settle_target` (the PR head before the audit fixes), which counted only
    the latest earlier start group, this returned `('m/2.md', None)` and failed the assertion.
    """
    tea, milk, coffee = (_fact("user", "drink", "tea", 1), _fact("user", "drink", "milk", 2),
                         _fact("user", "drink", "coffee", 3))
    held = _listing([], [_contested((tea, "m/1.md"), (milk, "m/2.md"))])
    assert settle_target(held, coffee, None) == (None, "several_older_memos")


def test_settle_declines_while_any_other_memo_holds_the_key() -> None:
    """K8. A settle that would leave another holder of the key in force (undated, of the new memo's
    start, or later than it) is declined `other_memos_hold_key`: the key would read contested after
    a write that claims to settle it. One earlier memo alone still settles.

    Red proof (CCA run 20261125T120125Z, fix audit FA-1-5), on the remote test host: against the
    pass-1 snapshot of `recall.declared_keys.settle_target`, which counted only earlier-dated holders,
    the undated case returned `('m/1.md', None)` and failed the first assertion.
    """
    milk, coffee = _fact("user", "drink", "milk", 1), _fact("user", "drink", "coffee", 3)
    for other in (_fact("user", "drink", "tea"), _fact("user", "drink", "juice", 3), _fact("user", "drink", "soda", 5)):
        held = _listing([], [_contested((milk, "m/1.md"), (other, "m/2.md"))])
        assert settle_target(held, coffee, None) == (None, "other_memos_hold_key"), other
    assert settle_target(_listing([_standing(milk, "m/1.md")]), coffee, None) == ("m/1.md", None)


def test_settle_reads_the_table_whatever_the_predicate_is_spelled() -> None:
    """K9. `phone number`, `Email-Address` and `PHONE_NUMBER` are the table's set-valued
    `phone_number` and `email_address`, so a settle of them is declined `not_single_in_table`: the
    table spells predicates with underscores, and a writer that spells one with a space or a hyphen
    would otherwise read as single-valued and retire the other value (owner's decision on CCA
    STAKES-002, 2026-11-25). The ledger's own lookup is unchanged.

    Red proof (CCA run 20261125T120125Z, STAKES-002), on the remote test host: against
    `recall.declared_keys.settle_precondition` before this change (the predicate casefolded only),
    `phone number` returned `('m/1.md', None)` and failed the first assertion.
    """
    for predicate in ("phone number", "Email-Address", "PHONE_NUMBER"):
        older = AtomicFact("memory", "user", predicate, "1", valid_from=datetime(2026, 1, 1, tzinfo=UTC))
        newer = AtomicFact("memory", "user", predicate, "2", valid_from=datetime(2026, 1, 2, tzinfo=UTC))
        assert settle_target(_listing([_standing(older, "m/1.md")]), newer, None) == (None, "not_single_in_table"), predicate
