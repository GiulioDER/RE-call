"""The dashboard's review queue: what it shows, and the two decisions, through the real rewrite chain.

Red proof, 2026-10-05, each mutation alone against `recall/dashboard/review.py`, failing for the stated
reason (JUnit XML), then restored; all green after:
- V3 the memo digest not compared: `test_accept_refuses_when_the_memo_changed_after_it_was_shown`,
  DID NOT RAISE ReviewRefused.
- V4 an accepted report left open: `test_accept_writes_supersedes_into_the_newer_memo_and_closes_the_report`,
  assert 'pending' == 'accepted'.
- V5 reject not recorded in the ledger: `test_reject_records_the_claim_closes_the_report_and_hides_it`,
  is_rejected False.
- V6 an agent report and an arbiter proposal of one claim not merged:
  `test_an_arbiter_proposal_for_the_same_claim_is_one_row`, ('arbiter',) == ('agent', 'arbiter').
- V1 and V2 (the rejected and declared filters removed) SURVIVED the first version of these tests: a
  decided report is closed, so the queue emptied anyway. The filters matter for an arbiter proposal of a
  rejected claim and for a report the memo declares by hand; V1b and V2b, against the two tests added
  for exactly those cases, fail with one extra queue item each.
- V7 the dashboard's own name-and-note check removed SURVIVES by design: `review_proposal` and
  `RejectionLedger.reject` refuse an anonymous decision underneath, which is the check that matters;
  the dashboard's copy only words the refusal for a person.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from recall.dashboard import review
from recall.document import parse_document
from recall.frontmatter import supersedes_targets
from recall.rewrite import RejectionLedger, RewriteRefused, claim_key, default_ledger_path
from recall.stale_reports import StaleReportQueue, default_queue_path

NOW = datetime(2026, 10, 5, 12, 0, tzinfo=UTC)
OLD_QUOTE = "it is still being prepared for PyPI"
NEW_QUOTE = "was published to PyPI on 2 September"


def _no_arbiter(_root: Path):
    raise RewriteRefused("the supersession arbiter is off")


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    (tmp_path / "old.md").write_text(f"The release is 0.12.0 and {OLD_QUOTE}.\n", encoding="utf-8")
    (tmp_path / "new.md").write_text(f"Release 0.12.0 {NEW_QUOTE}.\n", encoding="utf-8")
    with StaleReportQueue(default_queue_path(tmp_path)) as queue:
        queue.submit(
            tmp_path,
            stale_source="old.md",
            replacing_source="new.md",
            stale_quote=OLD_QUOTE,
            current_quote=NEW_QUOTE,
            client="mcp:acme",
            task="which release is current?",
            reported_at=NOW,
        )
    return tmp_path


def _queue(root: Path) -> review.Queue:
    return review.build_queue(root, arbiter_proposals=_no_arbiter)


def _supersedes(path: Path) -> list[str]:
    return list(supersedes_targets(parse_document(path.read_text(encoding="utf-8")).meta.get("supersedes")))


def test_a_pending_report_is_listed_with_its_quotes_and_the_arbiter_note(corpus: Path) -> None:
    queue = _queue(corpus)
    (item,) = queue.items
    assert (item.stale, item.replacing, item.origins) == ("old.md", "new.md", ("agent",))
    assert (item.stale_quote, item.current_quote) == (OLD_QUOTE, NEW_QUOTE)
    assert item.claim == claim_key("supersedes", "old.md", "new.md")
    assert any("arbiter is off" in note for note in queue.notes)


def test_accept_writes_supersedes_into_the_newer_memo_and_closes_the_report(corpus: Path) -> None:
    (item,) = _queue(corpus).items
    shown = review.memo_digest(corpus, review.preview(corpus, item, NOW).edit_file)
    plan = review.accept(corpus, item, reviewer="giulio", note="confirmed", shown_digest=shown, now=NOW)
    assert plan.edit_file == "new.md"
    assert _supersedes(corpus / "new.md") == ["old.md"]
    assert _supersedes(corpus / "old.md") == []
    with StaleReportQueue(default_queue_path(corpus)) as queue:
        assert queue.get(item.claim).status == "accepted"
    # declared now, so it leaves the queue without any record of its own
    assert _queue(corpus).items == ()


def test_accept_refuses_when_the_memo_changed_after_it_was_shown(corpus: Path) -> None:
    (item,) = _queue(corpus).items
    shown = review.memo_digest(corpus, "new.md")
    (corpus / "new.md").write_text(f"Release 0.12.0 {NEW_QUOTE}. Edited since.\n", encoding="utf-8")
    with pytest.raises(review.ReviewRefused, match="changed since you reviewed it"):
        review.accept(corpus, item, reviewer="giulio", note="confirmed", shown_digest=shown, now=NOW)
    assert _supersedes(corpus / "new.md") == []


@pytest.mark.parametrize("reviewer,note", [("", "confirmed"), ("giulio", " ")])
def test_a_decision_needs_a_name_and_a_note(corpus: Path, reviewer: str, note: str) -> None:
    (item,) = _queue(corpus).items
    shown = review.memo_digest(corpus, "new.md")
    with pytest.raises(review.ReviewRefused):
        review.accept(corpus, item, reviewer=reviewer, note=note, shown_digest=shown, now=NOW)
    with pytest.raises(review.ReviewRefused):
        review.reject(corpus, item, reviewer=reviewer, note=note, now=NOW)
    assert _supersedes(corpus / "new.md") == []
    if default_ledger_path(corpus).exists():
        with RejectionLedger(default_ledger_path(corpus)) as ledger:
            assert not ledger.claims()


def test_reject_records_the_claim_closes_the_report_and_hides_it(corpus: Path) -> None:
    (item,) = _queue(corpus).items
    review.reject(corpus, item, reviewer="giulio", note="not a replacement", now=NOW)
    with RejectionLedger(default_ledger_path(corpus)) as ledger:
        assert ledger.is_rejected(item.claim)
    with StaleReportQueue(default_queue_path(corpus)) as queue:
        assert queue.get(item.claim).status == "rejected"
    assert _queue(corpus).items == ()
    assert _supersedes(corpus / "new.md") == []


def test_an_arbiter_proposal_for_the_same_claim_is_one_row(corpus: Path) -> None:
    from recall.reasoning_proposals.types import InferenceProposal

    def _arbiter(_root: Path):
        return (
            InferenceProposal(
                id="ip_1",
                source_evidence_ids=("n1", "n2"),
                proposed_relation="supersedes",
                subject_id="old.md",
                object_id="new.md",
                explanation="judged a supersession pair",
                model_id="m",
                pipeline_id="p",
                provider_id="arb",
                provider_revision="r",
                confidence=0.97,
                uncertainty=(),
                generation_id="filesystem",
                status="requires_review",
                rule_id="supersession_arbiter.stated_probability",
                metadata={"quote_older": OLD_QUOTE, "quote_newer": NEW_QUOTE},
            ),
        )

    (item,) = review.build_queue(corpus, arbiter_proposals=_arbiter).items
    assert item.origins == ("agent", "arbiter")


def _arbiter_for(stale: str, replacing: str):
    from recall.reasoning_proposals.types import InferenceProposal

    def _arbiter(_root: Path):
        return (
            InferenceProposal(
                id="ip_2",
                source_evidence_ids=("n1", "n2"),
                proposed_relation="supersedes",
                subject_id=stale,
                object_id=replacing,
                explanation="judged a supersession pair",
                model_id="m",
                pipeline_id="p",
                provider_id="arb",
                provider_revision="r",
                confidence=0.97,
                uncertainty=(),
                generation_id="filesystem",
                status="requires_review",
                rule_id="supersession_arbiter.stated_probability",
                metadata={"quote_older": OLD_QUOTE, "quote_newer": NEW_QUOTE},
            ),
        )

    return _arbiter


def test_a_rejected_claim_stays_hidden_when_the_arbiter_proposes_it_again(corpus: Path) -> None:
    (item,) = _queue(corpus).items
    review.reject(corpus, item, reviewer="giulio", note="not a replacement", now=NOW)
    assert review.build_queue(corpus, arbiter_proposals=_arbiter_for("old.md", "new.md")).items == ()


def test_a_report_the_memo_already_declares_by_hand_is_hidden(corpus: Path) -> None:
    (corpus / "new.md").write_text(f"---\nsupersedes: old.md\n---\nRelease 0.12.0 {NEW_QUOTE}.\n", encoding="utf-8")
    assert _queue(corpus).items == ()
