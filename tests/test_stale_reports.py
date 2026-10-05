"""The stale-report queue records an agent's claim and nothing else, and only a checkable claim.

Red proof, 2026-10-05: each mutation applied alone to `recall/stale_reports.py`, each failing the
named test for the stated reason (JUnit XML), then restored; all 13 green after.
- R1 `submit` skips the `current_quote` check: `test_a_quote_that_cannot_be_checked_is_refused`, two
  cases, DID NOT RAISE.
- R2 `grounded` without the 20-character minimum: the same test, the short-quote case, DID NOT RAISE.
- R3 `submit` without the self-pair check:
  `test_a_report_must_name_two_different_memos_inside_the_corpus`, DID NOT RAISE.
- R4 `resolve_memo` without the root confinement: the same test, DID NOT RAISE.
- R5 `resolve_memo` without the sidecar rule: the same test (the `.recall/old-copy.md` case carries
  verbatim quotes, so only that rule can refuse it), DID NOT RAISE.
- R6 `resolve_memo` accepting absolute paths: `test_an_absolute_path_is_refused_even_inside_the_root`,
  DID NOT RAISE.
- R7 `submit` ignoring the rejection ledger: `test_a_claim_a_person_rejected_is_refused`, DID NOT RAISE.
- R8 no pending bound: `test_the_pending_queue_is_bounded`, DID NOT RAISE.
- R9 `mark_reviewed` without the reviewer and note check:
  `test_a_review_needs_a_named_person_and_closes_once`, DID NOT RAISE.
- R10 `submit` reopening a closed claim: the same test, DID NOT RAISE.
- R11 a repeat resetting the count to 1:
  `test_a_repeated_report_raises_the_count_instead_of_adding_a_row`, AssertionError 1 == 2.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from recall.rewrite import RejectionLedger, claim_key, default_ledger_path
from recall.stale_reports import (
    StaleReportQueue,
    StaleReportRefused,
    default_queue_path,
)

OLD_TEXT = "The release is 0.12.0 and it is still being prepared for PyPI.\n"
NEW_TEXT = "---\nsupersedes: nothing\n---\nRelease 0.12.0 was published to PyPI on 2 September.\n"
OLD_QUOTE = "it is still being prepared for PyPI"
NEW_QUOTE = "was published to PyPI on 2 September"
NOW = datetime(2026, 10, 5, 12, 0, tzinfo=UTC)


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    (tmp_path / "old.md").write_text(OLD_TEXT, encoding="utf-8")
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "new.md").write_text(NEW_TEXT, encoding="utf-8")
    return tmp_path


def _submit(queue: StaleReportQueue, root: Path, **overrides: object):
    args = {
        "stale_source": "old.md",
        "replacing_source": "notes/new.md",
        "stale_quote": OLD_QUOTE,
        "current_quote": NEW_QUOTE,
        "client": "test",
        "task": "which release is current?",
        "reported_at": NOW,
    }
    args.update(overrides)
    return queue.submit(root, **args)  # type: ignore[arg-type]


def test_a_checkable_report_is_queued_and_the_memos_are_untouched(corpus: Path) -> None:
    before = {p: p.read_bytes() for p in corpus.rglob("*.md")}
    with StaleReportQueue(default_queue_path(corpus)) as queue:
        report = _submit(queue, corpus)
        assert (report.stale_source, report.replacing_source, report.status) == ("old.md", "notes/new.md", "pending")
        assert report.claim_key == claim_key("supersedes", "old.md", "notes/new.md")
        assert [r.claim_key for r in queue.list()] == [report.claim_key]
    assert {p: p.read_bytes() for p in corpus.rglob("*.md")} == before


@pytest.mark.parametrize(
    "overrides",
    [
        {"current_quote": "this sentence is not in the newer memo"},
        {"stale_quote": "this sentence is not in the older memo"},
        {"current_quote": "published"},  # verbatim but shorter than the minimum
    ],
)
def test_a_quote_that_cannot_be_checked_is_refused(corpus: Path, overrides: dict) -> None:
    with StaleReportQueue(default_queue_path(corpus)) as queue:
        with pytest.raises(StaleReportRefused):
            _submit(queue, corpus, **overrides)
        assert queue.count("pending") == 0


@pytest.mark.parametrize(
    "overrides",
    [
        {"replacing_source": "old.md", "current_quote": OLD_QUOTE},  # a memo superseding itself
        {"stale_source": "../outside.md"},
        {"stale_source": "missing.md"},
        {"stale_source": ".recall/old-copy.md"},  # quotes are verbatim there: only the sidecar rule refuses
    ],
)
def test_a_report_must_name_two_different_memos_inside_the_corpus(corpus: Path, overrides: dict) -> None:
    (corpus.parent / "outside.md").write_text(OLD_TEXT, encoding="utf-8")
    (corpus / ".recall").mkdir(exist_ok=True)
    (corpus / ".recall" / "old-copy.md").write_text(OLD_TEXT, encoding="utf-8")
    with StaleReportQueue(default_queue_path(corpus)) as queue:
        with pytest.raises(StaleReportRefused):
            _submit(queue, corpus, **overrides)
        assert queue.count("pending") == 0


def test_an_absolute_path_is_refused_even_inside_the_root(corpus: Path) -> None:
    with StaleReportQueue(default_queue_path(corpus)) as queue:
        with pytest.raises(StaleReportRefused):
            _submit(queue, corpus, stale_source=str(corpus / "old.md"))


def test_a_repeated_report_raises_the_count_instead_of_adding_a_row(corpus: Path) -> None:
    with StaleReportQueue(default_queue_path(corpus)) as queue:
        _submit(queue, corpus)
        again = _submit(queue, corpus, client="another agent")
        assert again.report_count == 2
        assert queue.count("pending") == 1


def test_a_claim_a_person_rejected_is_refused(corpus: Path) -> None:
    key = claim_key("supersedes", "old.md", "notes/new.md")
    with RejectionLedger(default_ledger_path(corpus)) as ledger:
        ledger.reject(key, reviewer_id="giulio", reason="not a replacement", rejected_at=NOW)
    with StaleReportQueue(default_queue_path(corpus)) as queue:
        with pytest.raises(StaleReportRefused, match="already rejected"):
            _submit(queue, corpus)
        assert queue.count("pending") == 0


def test_the_pending_queue_is_bounded(corpus: Path) -> None:
    (corpus / "third.md").write_text("Release 0.12.0 was published to PyPI on 2 September, again.\n", encoding="utf-8")
    with StaleReportQueue(default_queue_path(corpus)) as queue:
        _submit(queue, corpus, max_pending=1)
        with pytest.raises(StaleReportRefused, match="pending reports"):
            _submit(queue, corpus, replacing_source="third.md", max_pending=1)
        assert queue.count("pending") == 1


def test_a_review_needs_a_named_person_and_closes_once(corpus: Path) -> None:
    with StaleReportQueue(default_queue_path(corpus)) as queue:
        key = _submit(queue, corpus).claim_key
        with pytest.raises(StaleReportRefused):
            queue.mark_reviewed(key, status="accepted", reviewer_id=" ", note="ok", reviewed_at=NOW)
        with pytest.raises(StaleReportRefused):
            queue.mark_reviewed(key, status="accepted", reviewer_id="giulio", note="", reviewed_at=NOW)
        done = queue.mark_reviewed(key, status="accepted", reviewer_id="giulio", note="confirmed", reviewed_at=NOW)
        assert (done.status, done.reviewer_id) == ("accepted", "giulio")
        with pytest.raises(StaleReportRefused):
            queue.mark_reviewed(key, status="rejected", reviewer_id="giulio", note="changed mind", reviewed_at=NOW)
        # a closed claim is not reopened by a new report
        with pytest.raises(StaleReportRefused, match="already accepted"):
            _submit(queue, corpus)
