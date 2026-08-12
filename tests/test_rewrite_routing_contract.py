"""Properties of the relation to key resolver.

`recall/rewrite.py` routed on a bare KEY while `reasoning_proposals/_extracted.py` emits a
RELATION with the key prefixed into `object_id`. That worked under proposal schema v1 only by
coincidence: the three v1 relation names happened to equal their key names. Schema v2 added
`declares_validity` and `declares_status`, and neither is a key, so both raised `unknown_key`
before this resolver existed.

The properties, one test each:

1. `supersedes` is the ONLY relation whose edit lands on `object_id`.
2. Every other routable relation edits `subject_id`.
3. `declares_validity` takes its key from the `object_id` prefix.
4. A validity prefix that is not one of the two validity keys is refused, not coerced.
5. `supersedes` cannot be smuggled in through a validity prefix.
6. `declares_status` routes to the derived block, never to frontmatter.
7. `references` is refused: it names no key, so there is nowhere to write it.
8. Every relation in `PROPOSED_RELATIONS` is either routed or explicitly refused, so adding a
   relation to the vocabulary without routing it fails this file rather than failing a user.
9. `plan_rewrite` itself handles a v2 relation, which is the regression this task closes.
"""
from datetime import datetime, timezone
from pathlib import Path

import pytest

from recall.frontmatter import validity_bounds
from recall.promotion import (
    PromotedFact,
    accept_reviewed_proposal,
    promote_accepted_proposal,
    review_proposal,
)
from recall.reasoning_proposals import InferenceProposal
from recall.reasoning_proposals.types import PROPOSED_RELATIONS
from recall.rewrite import RewriteRefused, destination, plan_rewrite, route_relation
from recall.truth_extraction.types import STATUS_VOCABULARY

_WHEN = datetime(2026, 8, 12, tzinfo=timezone.utc)


def _proposal(
    *,
    relation: str,
    subject_id: str,
    object_id: str,
) -> InferenceProposal:
    return InferenceProposal(
        id=f"ip_{relation}_{subject_id}_{object_id}",
        source_evidence_ids=("node-a", "node-b"),
        proposed_relation=relation,  # type: ignore[arg-type]
        subject_id=subject_id,
        object_id=object_id,
        explanation="the body states the relation the frontmatter does not declare",
        model_id="rules",
        pipeline_id="pipe-a",
        provider_id="recall.deterministic",
        provider_revision="rev-a",
        confidence=None,
        uncertainty=(),
        generation_id="gen-a",
    )


def _fact(*, relation: str, subject_id: str, object_id: str) -> PromotedFact:
    """A PromotedFact that has actually been through the review machine.

    Hand constructing one would either bypass `_refuse_untrusted` or fail it for the wrong
    reason, and this file is about routing, not about the gate.
    """
    reviewed = review_proposal(
        _proposal(relation=relation, subject_id=subject_id, object_id=object_id),
        reviewer_id="reviewer@example.com",
        reviewed_at=_WHEN,
        audit_note="Read the memo; the claim is stated and unhedged.",
    )
    return promote_accepted_proposal(accept_reviewed_proposal(reviewed), promoted_at=_WHEN)


def test_supersedes_edits_the_object_and_names_the_subject():
    routed = route_relation("supersedes", "old.md", "new.md")
    assert routed.edit_file == "new.md"
    assert routed.key == "supersedes"
    assert routed.value == "old.md"


def test_every_other_routable_relation_edits_the_subject():
    for relation, object_id in (
        ("declares_validity", "valid_from:2026-07-14"),
        ("declares_status", "status:deprecated"),
        ("contradicts", "other.md"),
        ("same_entity", "Alias"),
    ):
        routed = route_relation(relation, "subject.md", object_id)
        assert routed.edit_file == "subject.md", relation


def test_declares_validity_takes_its_key_from_the_prefix():
    routed = route_relation("declares_validity", "memo.md", "valid_until:2026-12-31")
    assert routed.key == "valid_until"
    assert routed.value == "2026-12-31"
    assert destination(routed.key) == "frontmatter"


def test_an_unknown_validity_prefix_is_refused_not_coerced():
    with pytest.raises(RewriteRefused, match="unroutable_validity"):
        route_relation("declares_validity", "memo.md", "expires_on:2026-12-31")


def test_supersedes_cannot_be_smuggled_through_a_validity_prefix():
    """`supersedes` IS in VALIDITY_KEYS, so a loose membership check would let this through."""
    with pytest.raises(RewriteRefused, match="unroutable_validity"):
        route_relation("declares_validity", "memo.md", "supersedes:victim.md")


def test_declares_status_routes_to_the_derived_block():
    routed = route_relation("declares_status", "memo.md", "status:deprecated")
    assert routed.key == "status"
    assert routed.value == "deprecated"
    assert destination(routed.key) == "derived"


def test_references_is_refused_because_it_has_no_key():
    with pytest.raises(RewriteRefused, match="unroutable_relation"):
        route_relation("references", "a.md", "b.md")


def test_every_relation_in_the_vocabulary_is_routed_or_refused():
    for relation in PROPOSED_RELATIONS:
        object_id = {
            "declares_validity": "valid_from:2026-07-14",
            "declares_status": "status:active",
        }.get(relation, "other.md")
        try:
            routed = route_relation(relation, "subject.md", object_id)
        except RewriteRefused:
            continue
        # A routed key must have a destination. Before this resolver, `declares_validity`
        # reached `destination()` as a relation and raised here.
        destination(routed.key)


def test_an_unparseable_validity_date_is_refused():
    """A date `recall index` cannot parse aborts the WHOLE corpus, not just this memo.

    `index.py:570` calls `validity_bounds` to fail fast and re-raises, so `valid_until:
    yesterday` in one file stops the run for all 792. Routing the key correctly and then
    writing an unusable value is the failure this refusal exists to prevent.
    """
    for bad in ("yesterday", "2026-13-99", "2026-07-14:2026-08-01", "14/07/2026"):
        with pytest.raises(RewriteRefused, match="unroutable_validity"):
            route_relation("declares_validity", "memo.md", f"valid_from:{bad}")


def test_a_well_formed_validity_date_is_not_over_refused():
    routed = route_relation("declares_validity", "memo.md", "valid_from:2026-07-14")
    assert routed.value == "2026-07-14"
    # The value this writes must survive the parser that reads it back.
    assert validity_bounds({"valid_from": routed.value})[0] is not None


def test_the_writer_is_not_stricter_than_the_reader():
    """`2026-7-4` is unpadded, and `validity_bounds` accepts it.

    Refusing it here would reject a value a human is allowed to have authored by hand. The
    guard's job is "never write what the reader cannot parse", not "impose a second format".
    Both sides call `validity_bounds`, so they cannot drift apart.
    """
    assert validity_bounds({"valid_from": "2026-7-4"})[0] is not None
    assert route_relation("declares_validity", "m.md", "valid_from:2026-7-4").value == "2026-7-4"


def test_an_off_vocabulary_status_is_refused():
    """`STATUS_VOCABULARY` is closed so a model cannot invent a taxonomy per memo."""
    for bad in ("whatever-the-model-said", "active:extra", "ACTIVE", "in/review", ""):
        with pytest.raises(RewriteRefused, match="unroutable_status"):
            route_relation("declares_status", "memo.md", f"status:{bad}")


def test_every_vocabulary_status_is_accepted():
    for good in STATUS_VOCABULARY:
        assert route_relation("declares_status", "memo.md", f"status:{good}").value == good


def test_a_second_status_is_refused_rather_than_appended(tmp_path: Path):
    """`status` is single valued. The derived block dedups on `key: value`, which is right for
    `contradicts` and wrong here: three promotions would leave three contradictory statuses."""
    from recall.rewrite import apply_rewrite

    (tmp_path / "memo.md").write_text("# memo\n\nBody.\n", encoding="utf-8", newline="\n")
    first = apply_rewrite(
        tmp_path,
        _fact(relation="declares_status", subject_id="memo.md", object_id="status:active"),
        apply=True,
    )
    assert first.written is True

    second = apply_rewrite(
        tmp_path,
        _fact(relation="declares_status", subject_id="memo.md", object_id="status:deprecated"),
        apply=True,
    )
    assert second.written is False
    assert "already declares status" in (second.refusal or "")
    assert (tmp_path / "memo.md").read_text(encoding="utf-8").count("status:") == 1


def test_a_second_contradicts_is_still_appended(tmp_path: Path):
    """The single-valued rule must not leak onto genuinely multi-valued keys."""
    from recall.rewrite import apply_rewrite

    (tmp_path / "memo.md").write_text("# memo\n\nBody.\n", encoding="utf-8", newline="\n")
    for other in ("a.md", "b.md"):
        result = apply_rewrite(
            tmp_path,
            _fact(relation="contradicts", subject_id="memo.md", object_id=other),
            apply=True,
        )
        assert result.written is True, other
    assert (tmp_path / "memo.md").read_text(encoding="utf-8").count("contradicts:") == 2


def test_plan_rewrite_handles_a_validity_relation(tmp_path: Path):
    """The regression: `destination("declares_validity")` raised unknown_key."""
    (tmp_path / "memo.md").write_text("Body.\n", encoding="utf-8", newline="\n")
    fact = _fact(
        relation="declares_validity", subject_id="memo.md", object_id="valid_from:2026-07-14"
    )
    plan = plan_rewrite(tmp_path, fact)
    assert plan.key == "valid_from"
    assert plan.value == "2026-07-14"
    assert plan.block == "frontmatter"
    assert plan.edit_file.endswith("memo.md")


def test_plan_rewrite_still_puts_supersedes_on_the_superseding_memo(tmp_path: Path):
    """The direction rule survives the refactor: inverting it demotes the live memo."""
    (tmp_path / "old.md").write_text("Older.\n", encoding="utf-8", newline="\n")
    (tmp_path / "new.md").write_text("Newer.\n", encoding="utf-8", newline="\n")
    fact = _fact(relation="supersedes", subject_id="old.md", object_id="new.md")
    plan = plan_rewrite(tmp_path, fact)
    assert plan.edit_file.endswith("new.md")
    assert plan.value == "old.md"
