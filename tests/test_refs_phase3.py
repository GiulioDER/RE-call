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


# ---- the review of the first version of this change found these four (red against that commit)


def test_a_file_in_a_dot_directory_is_not_a_namesake(tmp_path: Path) -> None:
    """`.trash/notes.md` is no memo, so `a/notes.md` keeps the key it always had."""
    (tmp_path / "a").mkdir()
    (tmp_path / ".trash").mkdir()
    (tmp_path / "a" / "notes.md").write_text(f"A: {OLD}.\n", encoding="utf-8")
    (tmp_path / ".trash" / "notes.md").write_text("deleted\n", encoding="utf-8")
    (tmp_path / "new.md").write_text(f"New: {NEW}.\n", encoding="utf-8")
    with StaleReportQueue(default_queue_path(tmp_path)) as queue:
        report = queue.submit(tmp_path, stale_source="a/notes.md", replacing_source="new.md", stale_quote=OLD,
                              current_quote=NEW, client="mcp:test", task=None, reported_at=NOW)
    assert report.claim_key == claim_key("supersedes", "a/notes.md", "new.md"), "a hidden file changed a memo's key"


def _gain_a_namesake(root: Path) -> None:
    """`a/notes.md` alone, reported, then `b/notes.md` appears: the row keeps the older key."""
    (root / "a").mkdir()
    (root / "a" / "notes.md").write_text(f"A: {OLD}.\n", encoding="utf-8")
    (root / "new.md").write_text(f"New: {NEW}.\n", encoding="utf-8")
    _report(root, "a/notes.md")
    (root / "b").mkdir()
    (root / "b" / "notes.md").write_text(f"B: {OLD}.\n", encoding="utf-8")


def test_a_claim_reported_again_after_gaining_a_namesake_counts_on_its_row(tmp_path: Path) -> None:
    _gain_a_namesake(tmp_path)
    _report(tmp_path, "a/notes.md")
    with StaleReportQueue(default_queue_path(tmp_path)) as queue:
        pending = queue.list("pending")
    assert [(r.stale_source, r.report_count) for r in pending] == [("a/notes.md", 2)], (
        f"a second pending row was opened for one claim: {[(r.stale_source, r.report_count) for r in pending]}"
    )


def test_an_older_row_and_its_database_report_are_one_item_closed_by_one_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.test_dashboard_review_database import _row

    def no_arbiter(*args: object, **kwargs: object) -> tuple[()]:
        raise RewriteRefused("the supersession arbiter is off")

    monkeypatch.setattr(review, "corpus_proposals", no_arbiter)
    _gain_a_namesake(tmp_path)
    rows = [_row("a/notes.md", "new.md", stale_quote=OLD, current_quote=NEW)]
    items = review.build_queue(tmp_path, database_reports=lambda: rows).items
    assert len(items) == 1, f"one claim was shown as {len(items)} items"
    review.reject(tmp_path, items[0], reviewer="ana", note="a is still right", now=NOW)
    left = review.build_queue(tmp_path, database_reports=lambda: rows).items
    assert not left, f"a rejected claim stayed pending under its other key: {[i.claim for i in left]}"


def test_the_preview_of_a_claim_rejected_under_its_older_key_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The dry run must not promise a write that `--apply` refuses."""
    import recall.rewrite as rewrite

    monkeypatch.setenv("RECALL_TRUTH_EXTRACTION", "1")
    from recall.cli import main
    from tests.test_rewrite_routing_contract import _proposal

    for folder in ("a", "b"):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "notes.md").write_text(f"{folder}: {OLD}.\n", encoding="utf-8")
    (tmp_path / "new.md").write_text(f"New: {NEW}.\n", encoding="utf-8")
    proposal = _proposal(relation="supersedes", subject_id="a/notes.md", object_id="new.md")
    monkeypatch.setattr(rewrite, "corpus_proposals", lambda *args, **kwargs: (proposal,))
    with RejectionLedger(default_ledger_path(tmp_path)) as ledger:
        ledger.reject(claim_key("supersedes", "a/notes.md", "new.md"), reviewer_id="ana", reason="no", rejected_at=NOW)
    with pytest.raises(SystemExit):
        main(["rewrite", "apply", str(tmp_path), "--proposal", proposal.id, "--reviewer", "ana", "--note", "n"])
    out = capsys.readouterr().out
    assert "already rejected" in out and "Re-run with --apply" not in out, f"the preview promised a refused write: {out}"
