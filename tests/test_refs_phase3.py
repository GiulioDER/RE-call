"""Phase 3 of resolving `supersedes:` references at write time: a claim's identity names its memo.

`rewrite.claim_key` keyed both sides by file stem, so a claim about `a/notes.md` and one about
`b/notes.md` were one claim: the queue merged them into one item, and rejecting either hid both.
A side whose stem another memo shares now keeps its folder; every other side keys as before, and
a rejection recorded under the older key still applies (`claim_keys`).

Red proof, 2026-11-06 (recorded in the pull request):
- against the pre-change `recall/rewrite.py`, `recall/stale_reports.py`, `recall/dashboard/review.py`
  and `recall/cli_commands/extract_rewrite.py`, the namesake tests fail in their own assertion;
- the older-key test is green before and after (before, the two keys are one); it is red under the
  mutation of `RejectionLedger.is_rejected` that ignores `older` ("an older rejection stopped
  applying").
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from recall.dashboard import review
from recall.rewrite import RejectionLedger, RewriteRefused, claim_key, default_ledger_path, plan_rewrite
from recall.stale_reports import StaleReportQueue, StaleReportRefused, default_queue_path
from tests.test_rewrite_routing_contract import _fact

NOW = datetime(2026, 11, 6, 12, 0, tzinfo=UTC)
OLD = "the deploy host for staging is alpha"
NEW = "the deploy host for staging is now gamma"


@pytest.fixture
def namesakes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    def no_arbiter(*args: object, **kwargs: object) -> tuple[()]:
        raise RewriteRefused("the supersession arbiter is off")

    monkeypatch.setattr(review, "corpus_proposals", no_arbiter)
    for folder in ("a", "b"):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "notes.md").write_text(f"{folder.upper()}: {OLD}.\n", encoding="utf-8")
    (tmp_path / "new.md").write_text(f"New: {NEW}.\n", encoding="utf-8")
    return tmp_path


def _report(root: Path, stale: str) -> None:
    with StaleReportQueue(default_queue_path(root)) as queue:
        queue.submit(root, stale_source=stale, replacing_source="new.md", stale_quote=OLD, current_quote=NEW,
                     client="mcp:test", task=None, reported_at=NOW)


def test_claims_about_two_namesakes_are_two_queue_items(namesakes: Path) -> None:
    _report(namesakes, "a/notes.md")
    _report(namesakes, "b/notes.md")
    stale = sorted(item.stale for item in review.build_queue(namesakes).items)
    assert stale == ["a/notes.md", "b/notes.md"], f"two claims about namesakes were merged: {stale}"


def test_rejecting_one_namesake_claim_leaves_the_other(namesakes: Path) -> None:
    _report(namesakes, "a/notes.md")
    _report(namesakes, "b/notes.md")
    items = {item.stale: item for item in review.build_queue(namesakes).items}
    assert set(items) == {"a/notes.md", "b/notes.md"}, f"two claims about namesakes were merged: {sorted(items)}"
    review.reject(namesakes, items["a/notes.md"], reviewer="ana", note="a is still right", now=NOW)
    left = [item.stale for item in review.build_queue(namesakes).items]
    assert left == ["b/notes.md"], f"rejecting one namesake's claim hid the other: {left}"


def test_a_rewrite_plan_keys_each_namesake_apart(namesakes: Path) -> None:
    claim_a = plan_rewrite(namesakes, _fact(relation="supersedes", subject_id="a/notes.md", object_id="new.md")).claim
    claim_b = plan_rewrite(namesakes, _fact(relation="supersedes", subject_id="b/notes.md", object_id="new.md")).claim
    assert claim_a != claim_b, "one claim key for two memos"


def test_an_older_rejection_still_applies(namesakes: Path) -> None:
    """A rejection made under the file-name key, before folders counted, was made against every
    memo the name could mean, so a new report of the claim is still refused."""
    with RejectionLedger(default_ledger_path(namesakes)) as ledger:
        ledger.reject(claim_key("supersedes", "b/notes.md", "new.md"), reviewer_id="ana", reason="decided", rejected_at=NOW)
    try:
        _report(namesakes, "b/notes.md")
        outcome = "accepted"
    except StaleReportRefused as exc:
        outcome = str(exc)
    assert outcome != "accepted", "an older rejection stopped applying"


def test_a_unique_name_keeps_the_key_it_always_had(tmp_path: Path) -> None:
    """A control, green before and after: nothing stored under a unique name changes key."""
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "old.md").write_text(f"Old: {OLD}.\n", encoding="utf-8")
    (tmp_path / "new.md").write_text(f"New: {NEW}.\n", encoding="utf-8")
    with StaleReportQueue(default_queue_path(tmp_path)) as queue:
        report = queue.submit(tmp_path, stale_source="notes/old.md", replacing_source="new.md", stale_quote=OLD,
                              current_quote=NEW, client="mcp:test", task=None, reported_at=NOW)
    assert report.claim_key == claim_key("supersedes", "old.md", "new.md")
