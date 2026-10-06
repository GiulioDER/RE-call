"""Agent stale reports recorded in the corpus database reach the review queue, checked against this folder.

A served MCP on another host records each report as a `stale_report` row in `recall_audit_events`
(`recall_mcp.stale_reports_api`); the dashboard reads those rows over its read-only connection and
merges them with the local sidecar. The person still decides on the files in front of them.

Invariants and the failure each one catches:
- Q1 a row naming served sources (`recall/old.md`) becomes one queue item over this folder's names,
  so the preview and the accept edit the files a person reads. (Its claim key needs no mapping:
  `claim_key` compares stems, so it already agrees with the sidecar and the rejection ledger.)
- Q2 a row whose quote is not verbatim in this folder's memo is not shown, and the notes say so.
- Q3 a row whose memo is not here is not shown; of two memos a served name could mean, the one
  matching more of its path is chosen, never one in a folder the name does not include.
- Q4 several rows for one pair are one item with their count; a sidecar report of the same claim
  stays one item that keeps the sidecar's identity, so a decision closes the sidecar row.
- Q5 a rejection hides a database-only claim and an accept writes the declaration, then hides it,
  though the rows themselves are append-only and never change.
- Q6 an unreachable database leaves the folder's own reports on the page, with a note.
- Q7 the page wires it: with a database connected, /queue lists the claim and /reject closes it.
- Q8 `db.stale_reports` returns only `stale_report` rows, only from the tenants it is given, newest
  first: a report filed in another tenant never reaches this folder's queue. (Until the audit of
  51e83dde it read every tenant, which let a report filed anywhere on the database reach this
  queue with only a 20 character quote check between it and a person.)
- Q9 an item takes its memo names and its quotes from the same row, so a quote is never shown under
  a memo it is not in (two memos with one stem share a claim key).
- Q10 a read that reaches `MAX_STALE_REPORTS` says so in the notes.
- Q11 a query that fails while it runs is a `DatabaseUnavailable`, so the queue degrades to the
  folder's own reports instead of dropping the request.
- Q12 a row as `report_stale` writes it is a row the review queue reads: the payload contract is
  exercised end to end, writer to reader, not by two hand-written copies.
- Q13 a quote is searched for once per version of a memo, not on every page: the search hashes
  every window of the memo, so repeating it per request costs seconds on a grown ledger.
- Q14 the rows of a claim already decided (rejected, or declared in the newer memo) are skipped
  before any quote is searched for, and the notes count them.
- Q15 a memo that cannot be read for a moment (locked by an editor or a scanner) is not remembered
  as holding no quote: the next page reads it again.

Red proof, 2026-10-06, each mutation alone, run through pytest with JUnit XML (Q8 against this
checkout's own session database, not skipped), then restored and green:
- R1 (Q1) `pinned[key]` set to the served names instead of the local ones:
  `test_a_database_report_is_one_item_over_this_folders_names`, 'recall/old.md' != 'old.md'.
  A first attempt, keying the claim on the served names, SURVIVED, and correctly: `claim_key`
  normalises to stems, so that mutation is equivalent. It is why Q1 asserts the names.
- R2 (Q2) the `grounded` check in `database_items` made `if not True`:
  `test_a_quote_not_verbatim_here_is_not_shown`, the queue held one item.
- R3 (Q3) `local_memo` comparing file names only (`name.split("/")[-1:]`):
  `test_a_memo_not_here_is_not_shown_and_the_longest_path_wins`, AssertionError "a served name
  matched a memo in a subfolder it does not name". A first version of this test asserted an
  "ambiguous" refusal and its mutation SURVIVED: two different relative paths cannot match one
  source at the same length, so that branch was dead and was removed.
- R4 (Q4) a database item replacing the sidecar's item outright (`if current is None` made
  `if True`): `test_reports_of_one_pair_are_one_item_and_the_sidecar_keeps_its_identity`,
  AssertionError "the sidecar's report lost its identity to the database's".
- R5 (Q5) `key not in rejected` removed from `build_queue`'s filter:
  `test_reject_and_accept_close_a_database_only_claim`, AssertionError "a rejected claim came back
  from the database rows".
- R6 (Q6) `except DatabaseUnavailable` in `build_queue` narrowed to an unrelated exception:
  `test_an_unreachable_database_leaves_the_folder_queue`, DatabaseUnavailable escaped the queue.
- R7 (Q7) `DashboardApp._reports_source` returning None:
  `test_the_queue_page_reads_the_database_and_reject_closes_it`, the review link was not on /queue.
- R8 (Q8) the `event_type = %s` filter in `db.stale_reports` made `(true or event_type = %s)`:
  `test_stale_reports_reads_only_stale_report_rows_from_every_tenant` (the name it had then),
  AssertionError "not newest first, or a tenant was missing" (the use_report and search rows came
  first).
The failure each produced is recorded in the line that names it. The quote rows these tests build
now carry fingerprints instead of text (the audit of 51e83dde, Q12); R1 to R8 were run before that
change, against rows that carried the quotes.

Red proof for Q8's tenant filter and Q7's binding, and for Q9 to Q12, added by the same audit,
2026-10-06, on a Linux host, each mutation alone, JUnit XML (Q8 against a real database, not
skipped), then restored and green. The tautology checker reported INCONCLUSIVE for each, because
the pre-fix tree lacks `quote_fingerprint` and `stale_row` and the module does not collect there;
the proof is a mutation of the new code:
- N6 (Q8) `tenant_id = any(%s)` made `(true or tenant_id = any(%s))`:
  `test_stale_reports_reads_only_stale_report_rows_of_the_named_tenants`, the third tenant's row
  came first.
- N7 (Q7) the server asking for `reports_tenants + ("re-call-code-gen",)`:
  `test_the_queue_page_reads_the_database_and_reject_closes_it`, AssertionError "the queue read
  reports from tenants this folder is not bound to".
- N8 (Q9) the names taken from the oldest row and the quotes from the newest, as the code first
  did: `test_an_item_takes_its_names_and_quotes_from_one_row`, AssertionError "... is shown under
  old.md, which does not contain it".
- N9 (Q10) the limit note made `if False`: `test_a_read_that_reaches_the_limit_says_so`.
- N10 (Q11) `DashboardDB.connect` no longer turning a statement's psycopg error into
  DatabaseUnavailable: `test_a_query_that_fails_while_it_runs_degrades_the_queue`, QueryCanceled
  escaped.
- N11 (Q12) the writer renaming `stale_quote_sha256`: `test_a_row_as_the_tool_writes_it_is_a_row_the_queue_reads`,
  AssertionError "the row the tool wrote is not one the queue reads" (its quote counted as not
  verbatim).

A second repair round, after the regression review found the quote search repeated on every page:
- O1 (Q13) the `lru_cache` removed from `_quote_in`: `test_a_quote_is_searched_once_per_version_of_a_memo`,
  AssertionError "the quotes were searched for again on the next page", 4 == 2. (A first run of
  this proof failed on the test's own `cache_clear` call instead, which proves nothing; the helper
  now tolerates the cache being absent.)
- O2 (Q14) the decided-claim skip made `if False`: `test_a_decided_claim_is_skipped_before_its_quotes_are_searched`,
  AssertionError "a rejected claim's quotes were searched for".
- P1 (Q15), a third round, after the second regression review found read failures cached: the
  `except OSError: return None` put back inside the cached `_quote_in`:
  `test_a_read_that_fails_once_is_not_remembered`, AssertionError "a read that failed once kept the
  claim off the queue", 0 == 1.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import urlencode

import psycopg
import pytest

from recall.dashboard import db as dbq
from recall.dashboard import review
from recall.dashboard.server import COOKIE, DashboardApp
from recall.document import parse_document
from recall.frontmatter import supersedes_targets
from recall.rewrite import RejectionLedger, RewriteRefused, claim_key, default_ledger_path
from recall.stale_reports import StaleReportQueue, default_queue_path, grounded, quote_fingerprint
from tests.conftest import TEST_DSN, requires_db

NOW = datetime(2026, 10, 6, 12, 0, tzinfo=UTC)
OLD_QUOTE = "it is still being prepared for PyPI"
NEW_QUOTE = "was published to PyPI on 2 September"
KEY = claim_key("supersedes", "old.md", "new.md")


def _no_arbiter(_root: Path):
    raise RewriteRefused("the supersession arbiter is off")


def _row(
    stale: str = "recall/old.md",
    replacing: str = "recall/new.md",
    *,
    minutes: int = 0,
    stale_quote: str = OLD_QUOTE,
    current_quote: str = NEW_QUOTE,
    **extra: object,
) -> dict[str, object]:
    """A row as `db.stale_reports` returns it: fingerprints of the quotes, never their text."""
    stale_sha, stale_chars = quote_fingerprint(stale_quote)
    current_sha, current_chars = quote_fingerprint(current_quote)
    row: dict[str, object] = {
        "event_id": f"evt-{uuid.uuid4().hex[:8]}", "tenant": "memory", "created_at": NOW + timedelta(minutes=minutes),
        "stale_source": stale, "replacing_source": replacing,
        "stale_quote_sha256": stale_sha, "stale_quote_chars": stale_chars,
        "current_quote_sha256": current_sha, "current_quote_chars": current_chars,
        "claim_key": claim_key("supersedes", stale, replacing), "client": "mcp:memory", "task": "which release is current?",
    }
    row.update(extra)
    return row


@pytest.fixture
def corpus(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.delenv("RECALL_SUPERSESSION_ARBITER", raising=False)
    (tmp_path / "old.md").write_text(f"The release is 0.12.0 and {OLD_QUOTE}.\n", encoding="utf-8")
    (tmp_path / "new.md").write_text(f"Release 0.12.0 {NEW_QUOTE}.\n", encoding="utf-8")
    return tmp_path


def _queue(root: Path, rows: list[dict[str, object]]) -> review.Queue:
    return review.build_queue(root, arbiter_proposals=_no_arbiter, database_reports=lambda: rows)


def test_a_database_report_is_one_item_over_this_folders_names(corpus: Path) -> None:
    (item,) = _queue(corpus, [_row()]).items
    assert (item.claim, item.stale, item.replacing) == (KEY, "old.md", "new.md")
    assert item.origins == ("agent",)
    assert (item.stale_quote, item.current_quote) == (OLD_QUOTE, NEW_QUOTE)
    assert any("served as recall/old.md and recall/new.md" in d for d in item.details)
    assert any("tenant memory" in d for d in item.details)


def test_a_quote_not_verbatim_here_is_not_shown(corpus: Path) -> None:
    queue = _queue(corpus, [_row(current_quote="this sentence is in the served text but not here")])
    assert queue.items == ()
    assert any("1 report(s) quote text that is not verbatim" in note for note in queue.notes)


def test_a_memo_not_here_is_not_shown_and_the_longest_path_wins(corpus: Path) -> None:
    queue = _queue(corpus, [_row("recall/old.md", "recall/elsewhere.md"), _row("recall/gone.md", "recall/new.md")])
    assert queue.items == ()
    assert any("2 report(s) name a memo this folder does not hold" in note for note in queue.notes)

    (corpus / "a").mkdir()
    (corpus / "a" / "new.md").write_text(f"Release 0.12.0 {NEW_QUOTE}, in a subfolder.\n", encoding="utf-8")
    (top,) = _queue(corpus, [_row("recall/old.md", "recall/new.md")]).items
    assert top.replacing == "new.md", "a served name matched a memo in a subfolder it does not name"
    (nested,) = _queue(corpus, [_row("recall/old.md", "recall/a/new.md")]).items
    assert nested.replacing == "a/new.md"


def test_reports_of_one_pair_are_one_item_and_the_sidecar_keeps_its_identity(corpus: Path) -> None:
    rows = [_row(minutes=5, client="mcp:memory"), _row(minutes=1, client="mcp:other"), _row("old.md", "new.md", minutes=3)]
    (item,) = _queue(corpus, rows).items
    assert item.details[0].startswith("reported 3 time(s) by mcp:memory, mcp:other")

    with StaleReportQueue(default_queue_path(corpus)) as queue:
        queue.submit(corpus, stale_source="old.md", replacing_source="new.md", stale_quote=OLD_QUOTE,
                      current_quote=NEW_QUOTE, client="mcp:local", task=None, reported_at=NOW)
    (merged,) = _queue(corpus, rows).items
    assert merged.proposal.model_id == "mcp:local", "the sidecar's report lost its identity to the database's"
    assert merged.details[0] == "reported 1 time(s) by mcp:local"
    assert any("reported 3 time(s)" in d and "corpus database" in d for d in merged.details)


def test_reject_and_accept_close_a_database_only_claim(corpus: Path) -> None:
    rows = [_row()]
    (item,) = _queue(corpus, rows).items
    review.reject(corpus, item, reviewer="giulio", note="the newer memo is about a different release", now=NOW)
    assert _queue(corpus, rows).items == (), "a rejected claim came back from the database rows"
    with RejectionLedger(default_ledger_path(corpus)) as ledger:
        assert ledger.is_rejected(KEY)
    assert not default_queue_path(corpus).exists(), "a decision on a database report created a sidecar"

    (other := corpus / "older.md").write_text(f"Earlier: {OLD_QUOTE}, said a draft.\n", encoding="utf-8")
    rows = [_row("recall/older.md", "recall/new.md")]
    (item,) = _queue(corpus, rows).items
    assert item.stale == other.name
    plan = review.preview(corpus, item, NOW)
    review.accept(corpus, item, reviewer="giulio", note="confirmed", shown_digest=review.memo_digest(corpus, plan.edit_file), now=NOW)
    meta = parse_document((corpus / "new.md").read_text(encoding="utf-8")).meta
    assert list(supersedes_targets(meta.get("supersedes"))) == ["older.md"]
    assert _queue(corpus, rows).items == ()


def test_an_unreachable_database_leaves_the_folder_queue(corpus: Path) -> None:
    with StaleReportQueue(default_queue_path(corpus)) as queue:
        queue.submit(corpus, stale_source="old.md", replacing_source="new.md", stale_quote=OLD_QUOTE,
                     current_quote=NEW_QUOTE, client="mcp:local", task=None, reported_at=NOW)

    def down() -> list[dict[str, object]]:
        raise dbq.DatabaseUnavailable("connection refused")

    queue = review.build_queue(corpus, arbiter_proposals=_no_arbiter, database_reports=down)
    assert [item.claim for item in queue.items] == [KEY]
    assert any("unreachable" in note and "connection refused" in note for note in queue.notes)


PORT = 8766
SIGNED = {"Host": f"127.0.0.1:{PORT}", "Cookie": f"{COOKIE}=tok123"}


def test_the_queue_page_reads_the_database_and_reject_closes_it(corpus: Path, monkeypatch) -> None:
    asked: list[tuple[str, ...]] = []

    def reports(db: dbq.DashboardDB, tenants: tuple[str, ...], limit: int = dbq.MAX_STALE_REPORTS) -> list[dict[str, object]]:
        asked.append(tenants)
        return [_row()]

    monkeypatch.setattr(dbq, "stale_reports", reports)
    monkeypatch.setattr(dbq, "tenants", lambda db: [])
    app = DashboardApp(corpus, port=PORT, token="tok123", db=dbq.DashboardDB("postgresql://unused"))
    page = app.handle("GET", "/queue", SIGNED).body.decode()
    assert asked and set(asked) == {("memory",)}, "the queue read reports from tenants this folder is not bound to"
    assert "review?" + urlencode({"claim": KEY}) in page
    form = urlencode({"csrf": "tok123", "claim": KEY, "reviewer": "giulio", "note": "not the same release"}).encode()
    assert app.handle("POST", "/reject", SIGNED, form).status == 303
    assert "Nothing to review" in app.handle("GET", "/queue", SIGNED).body.decode()


@pytest.fixture
def ledger_rows(make_store) -> Iterator[tuple[str, str]]:
    make_store(8)  # applies the migrations that create recall_audit_events
    one, two, other = ("sr-" + uuid.uuid4().hex[:10] for _ in range(3))
    rows = [
        (one, "stale_report", _row(), "2026-10-06T10:00:00+00:00"),
        (two, "stale_report", _row("recall/a.md", "recall/b.md"), "2026-10-06T11:00:00+00:00"),
        (other, "stale_report", _row("recall/c.md", "recall/d.md"), "2026-10-06T11:30:00+00:00"),
        (one, "use_report", {"task": "t", "effect": "helped", "used": ["recall/new.md"]}, "2026-10-06T12:00:00+00:00"),
        (one, "search_decision", {"query": "q", "hits": []}, "2026-10-06T12:30:00+00:00"),
    ]
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        for i, (tenant, kind, payload, at) in enumerate(rows):
            body = {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in payload.items() if k not in ("event_id", "tenant", "created_at")}
            conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (tenant,))
            conn.execute(
                "INSERT INTO recall_audit_events (tenant_id, event_id, event_type, actor, payload, created_at) "
                "VALUES (%s, %s, %s, 'agent-report', %s, %s)",
                (tenant, f"evt_{i}_{uuid.uuid4().hex[:6]}", kind, json.dumps(body), at),
            )
    yield one, two
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        for tenant in (one, two, other):
            conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (tenant,))
            conn.execute("DELETE FROM recall_audit_events WHERE tenant_id = %s", (tenant,))


@requires_db
def test_stale_reports_reads_only_stale_report_rows_of_the_named_tenants(ledger_rows: tuple[str, str]) -> None:
    one, two = ledger_rows
    found = dbq.stale_reports(dbq.DashboardDB(TEST_DSN), (one, two))
    assert [r["tenant"] for r in found] == [two, one], "not newest first, a named tenant was missing, or another tenant was read"
    newest = found[0]
    assert (newest["stale_source"], newest["replacing_source"]) == ("recall/a.md", "recall/b.md")
    assert (newest["stale_quote_sha256"], newest["stale_quote_chars"]) == quote_fingerprint(OLD_QUOTE)
    assert (newest["current_quote_sha256"], newest["current_quote_chars"], newest["client"]) == (*quote_fingerprint(NEW_QUOTE), "mcp:memory")
    assert isinstance(newest["created_at"], datetime)


def test_an_item_takes_its_names_and_quotes_from_one_row(corpus: Path) -> None:
    (corpus / "archive").mkdir()
    archived = "the archived draft says the release is still pending review"
    (corpus / "archive" / "old.md").write_text(f"Archive: {archived}.\n", encoding="utf-8")
    rows = [
        _row("recall/archive/old.md", "recall/new.md", minutes=9, stale_quote=archived),
        _row("recall/old.md", "recall/new.md", minutes=1),
    ]
    (item,) = _queue(corpus, rows).items
    assert grounded(item.stale_quote, (corpus / item.stale).read_text(encoding="utf-8")), (
        f"{item.stale_quote!r} is shown under {item.stale}, which does not contain it"
    )
    assert (item.stale, item.stale_quote) == ("archive/old.md", archived)


def test_a_read_that_reaches_the_limit_says_so(corpus: Path, monkeypatch) -> None:
    monkeypatch.setattr(review, "MAX_STALE_REPORTS", 2)
    queue = _queue(corpus, [_row(minutes=2), _row(minutes=1)])
    assert any("only the newest 2 reports were read" in note for note in queue.notes)
    assert not any("only the newest" in note for note in _queue(corpus, [_row()]).notes)


class _Connection:
    def execute(self, *_: object) -> object:
        raise psycopg.errors.QueryCanceled("canceling statement due to statement timeout")

    def close(self) -> None:
        pass


def test_a_query_that_fails_while_it_runs_degrades_the_queue(corpus: Path, monkeypatch) -> None:
    monkeypatch.setattr(psycopg, "connect", lambda *a, **k: _Connection())
    db = dbq.DashboardDB("postgresql://unused")
    with pytest.raises(dbq.DatabaseUnavailable, match="QueryCanceled: canceling statement"):
        dbq.stale_reports(db, ("memory",))
    with StaleReportQueue(default_queue_path(corpus)) as queue:
        queue.submit(corpus, stale_source="old.md", replacing_source="new.md", stale_quote=OLD_QUOTE,
                     current_quote=NEW_QUOTE, client="mcp:local", task=None, reported_at=NOW)
    queue_ = review.build_queue(corpus, arbiter_proposals=_no_arbiter, database_reports=lambda: dbq.stale_reports(db, ("memory",)))
    assert [item.claim for item in queue_.items] == [KEY]
    assert any("unreachable" in note for note in queue_.notes)


def test_a_row_as_the_tool_writes_it_is_a_row_the_queue_reads(corpus: Path, tmp_path_factory) -> None:
    from recall_mcp.stale_reports_api import report_stale

    written: list[dict[str, object]] = []

    class Store:
        tenant = "memory"

        def source_content_hashes(self) -> dict[str, str]:
            return {"recall/old.md": "a", "recall/new.md": "b"}

        def source_chunk_texts(self, source: str) -> list[str]:
            return [(corpus / source.split("/")[-1]).read_text(encoding="utf-8")]

        def append_audit_event(self, event_type: str, payload: dict[str, object], **kwargs: object) -> str:
            written.append(payload)
            return str(kwargs.get("event_id"))

    elsewhere = tmp_path_factory.mktemp("serving-checkout")
    report_stale(Store(), stale_source="recall/old.md", replacing_source="recall/new.md", stale_quote=OLD_QUOTE,
                 current_quote=NEW_QUOTE, task="which release?", env={"RECALL_INDEX_ROOT": str(elsewhere)}, now=NOW)
    (payload,) = written
    queue = _queue(corpus, [dbq.stale_row("evt_x", "memory", payload, NOW)])
    assert len(queue.items) == 1, f"the row the tool wrote is not one the queue reads: {queue.notes}"
    (item,) = queue.items
    assert (item.stale, item.replacing, item.stale_quote, item.current_quote) == ("old.md", "new.md", OLD_QUOTE, NEW_QUOTE)


def _counting(monkeypatch) -> list[int]:
    calls: list[int] = []
    real = review.find_quote

    def counted(*args: object) -> str | None:
        calls.append(1)
        return real(*args)  # type: ignore[arg-type]

    getattr(review._quote_in, "cache_clear", lambda: None)()  # absent when the cache is mutated away
    monkeypatch.setattr(review, "find_quote", counted)
    return calls


def test_a_quote_is_searched_once_per_version_of_a_memo(corpus: Path, monkeypatch) -> None:
    calls = _counting(monkeypatch)
    rows = [_row()]
    assert len(_queue(corpus, rows).items) == 1
    first = len(calls)
    assert first == 2
    assert len(_queue(corpus, rows).items) == 1
    assert len(calls) == first, "the quotes were searched for again on the next page"
    (corpus / "new.md").write_text(f"Release 0.12.0 {NEW_QUOTE}, edited.\n", encoding="utf-8")
    assert len(_queue(corpus, rows).items) == 1
    assert len(calls) == first + 1, "an edited memo was not searched again"


def test_a_decided_claim_is_skipped_before_its_quotes_are_searched(corpus: Path, monkeypatch) -> None:
    rows = [_row()]
    (item,) = _queue(corpus, rows).items
    review.reject(corpus, item, reviewer="giulio", note="not the same release", now=NOW)
    calls = _counting(monkeypatch)
    queue = _queue(corpus, rows)
    assert queue.items == ()
    assert calls == [], "a rejected claim's quotes were searched for"
    assert any("1 report(s) about claims already decided" in note for note in queue.notes)


def test_a_read_that_fails_once_is_not_remembered(corpus: Path, monkeypatch) -> None:
    import pathlib

    review._quote_in.cache_clear()
    real = pathlib.Path.read_text
    failures = [PermissionError("The process cannot access the file because another process has locked it")]

    def flaky(self: pathlib.Path, *args: object, **kwargs: object) -> str:
        if self.name == "new.md" and kwargs.get("errors") == "replace" and failures:  # the quote search's read only
            raise failures.pop()
        return real(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(pathlib.Path, "read_text", flaky)
    assert _queue(corpus, [_row()]).items == ()
    assert len(_queue(corpus, [_row()]).items) == 1, "a read that failed once kept the claim off the queue"
