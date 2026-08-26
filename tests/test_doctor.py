"""`recall doctor` has to work when the install does not, which is the only time anyone runs it.

Every test here is written against that constraint. A diagnostic is used at exactly the moment the
database is down, the package is half-installed, or the corpus is somewhere other than where the
config says, so the failure modes that matter are the ones where this command itself falls over,
refuses to start, or takes minutes before printing anything.

Four properties, each of which is a way this would be worse than useless:

* **It writes nothing.** A doctor that repairs cannot be run to find out what is wrong, because
  running it changes the answer.
* **It never builds an embedder.** `resolve_embedder("fastembed")` downloads a model on a cold
  cache. A diagnostic that costs 130 MB and a wait before its first line is one people stop running.
* **Every `fail` carries the command that fixes it.** Reporting a problem without a repair moves
  the work rather than doing it.
* **The exit code means BLOCKED, not imperfect.** An uncalibrated corpus is a warning; if it exited
  non-zero, every script would have to ignore the exit code, which is the same as not having one.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from recall import doctor
from recall.doctor import Check, Report, run_checks
from recall.store import DEFAULT_TABLE


def test_a_blocking_failure_exits_non_zero_and_a_warning_does_not() -> None:
    """The distinction the exit code exists to carry, asserted in both directions.

    Both halves matter. Without the first the code is never 1 and nothing can gate on it; without
    the second it is always 1 on a healthy-but-uncalibrated install, and a code that is always 1 is
    read by nobody.
    """
    assert Report([Check("x", "warn", "d"), Check("y", "ok", "d")]).exit_code() == 0
    assert Report([Check("x", "skip", "d")]).exit_code() == 0
    assert Report([Check("x", "warn", "d"), Check("y", "fail", "d")]).exit_code() == 1


def test_every_failure_names_the_command_that_fixes_it() -> None:
    """⚠️ Asserted over the REAL run, not over hand-built `Check`s.

    A rule about how findings are written is only worth having if it binds the findings this
    command actually produces. Run against a DSN nothing listens on, so the database, corpus and
    calibration branches are all exercised without needing a container.
    """
    report = run_checks(
        dsn="postgresql://recall:recall@127.0.0.1:1/recall",
        embedder="hashing",
        trust_mode="strict",
        project_root=Path.cwd(),
    )
    unhelpful = [c.name for c in report.checks if c.status == "fail" and not c.fix]
    assert not unhelpful, f"failing checks with no repair line: {unhelpful}"
    assert report.failed, "the test is vacuous if nothing failed against an unreachable database"


def test_the_embedder_check_never_constructs_an_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    """⛔ The cold-cache trap, pinned by making the expensive call explode.

    `resolve_embedder` is what downloads. If any path in `run_checks` reaches it, this raises
    rather than quietly costing the user minutes on the one command they ran because they were
    already stuck.
    """
    import recall.embeddings

    def _explode(*args: Any, **kwargs: Any):
        raise AssertionError("doctor resolved an embedder; that downloads a model")

    monkeypatch.setattr(recall.embeddings, "resolve_embedder", _explode)
    report = run_checks(
        dsn="postgresql://recall:recall@127.0.0.1:1/recall",
        embedder="fastembed",
        trust_mode="development",
        project_root=Path.cwd(),
    )
    embedder = next(c for c in report.checks if c.name == "embedder")
    assert "fastembed" in embedder.detail


def test_a_missing_backend_is_a_failure_that_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """The most common install mistake: `pip install recall-rag` without the extra."""
    monkeypatch.setattr(doctor.importlib.util, "find_spec", lambda name: None)
    check = doctor._embedder_check("fastembed")
    assert check.status == "fail"
    assert check.fix is not None and "recall-rag[fastembed]" in check.fix


def test_hashing_needs_no_backend_and_is_never_reported_as_broken() -> None:
    """An offline embedder that needs nothing must not be reported as missing a dependency."""
    assert doctor._embedder_check("hashing").status == "ok"


def test_the_console_script_check_names_the_ones_the_plugin_invokes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⛔ This is the only check that explains a plugin install with no tools.

    A Claude Code plugin manifest is written once and shipped to every machine, so it cannot name
    an interpreter: it invokes `recall-mcp` and `recall-hooks` by bare name. A marketplace install
    with no `pip install recall-rag` behind it therefore fails to spawn its own server, and the
    client reports that as absent tools, which is also what a missing config, a dead database and
    pending migrations look like.
    """
    assert {"recall-mcp", "recall-hooks"} <= set(doctor.CONSOLE_SCRIPTS)

    monkeypatch.setattr(shutil, "which", lambda name: None if name == "recall-mcp" else "/x/" + name)
    check = doctor._console_scripts_check()
    assert check.status == "fail"
    # Asserted on the LIST, not on the whole string: the explanatory clause after it names both
    # scripts on purpose, and a substring check over the whole detail would forbid the sentence
    # that tells the reader why either one matters.
    assert check.detail.startswith("not on PATH: recall-mcp."), check.detail

    monkeypatch.setattr(shutil, "which", lambda name: "/x/" + name)
    assert doctor._console_scripts_check().status == "ok"


def test_a_table_that_is_not_an_identifier_is_refused_before_any_query() -> None:
    """Refused in the caller, and composed as an identifier in the query. Two gates, on purpose."""
    with pytest.raises(ValueError, match="identifier"):
        run_checks(dsn="postgresql://x", embedder="hashing", table="drop table; --")


class _FakeCursor:
    def __init__(self, rows: list[tuple]) -> None:
        self._rows = rows

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return self._rows


class _FakeConn:
    """The smallest thing that answers the four queries `_corpus_checks` asks.

    A fake rather than a container, because what is under test is the REPORTING decision — does an
    empty configured corpus name the populated one? — and that decision is the same whatever
    PostgreSQL is running. The queries themselves are covered by the DB-backed suite.
    """

    def __init__(
        self,
        *,
        tables: dict[str, list[tuple[str, int]]],
        shapes: dict[str, tuple[str, ...]] | None = None,
        can_bypass_rls: bool = True,
    ) -> None:
        self.tables = tables
        #: Columns each table has. Absent means "a full corpus". A table listed here with fewer
        #: columns is one the shape filter must EXCLUDE, which is the bookkeeping-table case.
        self.shapes = shapes or {}
        #: Whether this role can read past row-level security. False makes `_populated_corpora`
        #: append its "other tenants are hidden" caveat, which is the honest answer for a role that
        #: cannot see them rather than the confident "nothing holds rows" it used to print.
        self.can_bypass_rls = can_bypass_rls

    def __enter__(self):
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def transaction(self):
        """⚠️ Kept after this fake caused a VACUOUS PASS, which is worth recording.

        `_populated_corpora` once wrapped each sibling count in `conn.transaction()`. A fake without
        this method raised `AttributeError`, the broad `except` around the count treated it as
        "could not count", and the table's name appeared in the detail under `[not counted: ...]`
        rather than with its row count. The assertion was `"quickstart_chunks" in detail`, so it
        matched the FAILURE text and went green while the behaviour under test never ran.

        The bound now lives on the session (`_connect` passes `options=-c statement_timeout=...`),
        so nothing calls this any more. It stays because a fake that quietly lacks a method the
        code later starts calling reproduces the same trap, and because the lesson is cheaper to
        keep than to relearn.
        """
        return self

    def execute(self, query: Any, params: tuple | None = None) -> _FakeCursor:
        """⛔ **Answer only what this fake actually understands, and RAISE otherwise.**

        Every branch here is a place the fake can diverge from PostgreSQL, so the list is kept as
        short as the code allows and anything unrecognised is a loud AssertionError rather than an
        empty result. An empty result would be absorbed by `_populated_corpora`'s broad `except` and
        reported as "not counted", which is exactly how this file produced a green test over code
        that never ran. The DB-backed suite in `tests/test_doctor_db.py` is what covers the SQL
        itself; these tests cover the REPORTING decisions, which are the same whatever the database.
        """
        text = (
            str(query.as_string()) if hasattr(query, "as_string") else str(query)
        )
        if "a.atttypmod" in text:
            # `_embedding_width`. ⚠️ Checked BEFORE the discovery branch: both queries join
            # `pg_class` and both mention `a.attname`, so ordering is the only thing separating
            # them. Matching the discovery branch first returned (relname, attname) rows and the
            # width read got a string where it wanted an int — the fake diverging from the code by
            # one dispatch line, which is this file's recurring failure mode.
            return _FakeCursor([(384,)])
        if "pg_class" in text and "a.attname" in text:
            # `_chunk_tables`: one round trip returning (relname, attname) for the whole schema.
            # Every fixture table is given the full chunk shape unless it is declared otherwise, so
            # a test that wants a NON-corpus table says so explicitly via `shapes`.
            rows: list[tuple] = []
            for name in self.tables:
                for column in self.shapes.get(name, ("tenant_id", "embedding", "source")):
                    rows.append((name, column))
            return _FakeCursor(rows)
        if "rolsuper" in text:
            return _FakeCursor([(self.can_bypass_rls,)])
        if "GROUP BY tenant_id" in text:
            for name, counts in self.tables.items():
                if f'"{name}"' in text:
                    return _FakeCursor(list(counts))
            return _FakeCursor([])
        if "count(*)" in text:
            assert params is not None
            for name, counts in self.tables.items():
                if f'"{name}"' in text:
                    return _FakeCursor([(sum(c for t, c in counts if t == params[0]),)])
            return _FakeCursor([(0,)])
        raise AssertionError(f"unexpected query: {text}")


def test_an_empty_configured_corpus_names_the_one_that_is_populated() -> None:
    """⚠️ The headline behaviour, and the reason this command exists at all.

    This is the shipped defect it was written for: the Claude Code plugin was pointed at `chunks`
    while `recall quickstart` had indexed into `quickstart_chunks`, and `recall_search` answered
    `0 relevant memory hit(s)` with no error. "Empty" and "wrong table" are the same observation
    from inside a search; the difference is only visible to something that looks at the OTHER
    tables, which is what this does.

    So the assertion is not that it fails. It is that the failure names `quickstart_chunks` and
    `quickstart`, because a failure that says only "no rows" leaves the user exactly where they
    were.
    """
    conn = _FakeConn(tables={"chunks": [], "quickstart_chunks": [("quickstart", 22)]})

    checks = list(doctor._corpus_checks(conn, table="chunks", tenant="default"))
    corpus = next(c for c in checks if c.name == "corpus")
    assert corpus.status == "fail"
    # The ROW COUNT, not just the table name. Naming the table is satisfied by the "could not count
    # in time" branch too, and asserting only the name is how this test first passed vacuously.
    assert "quickstart_chunks/quickstart (22)" in corpus.detail, corpus.detail
    assert "not counted in time" not in corpus.detail, corpus.detail
    assert corpus.fix is not None and "--table" in corpus.fix


def test_a_populated_configured_corpus_passes_and_says_how_many() -> None:
    conn = _FakeConn(tables={"chunks": [("default", 91)]})

    checks = list(doctor._corpus_checks(conn, table=DEFAULT_TABLE, tenant="default"))
    corpus = next(c for c in checks if c.name == "corpus")
    assert corpus.status == "ok"
    assert "91" in corpus.detail


def test_an_unreachable_database_skips_the_corpus_rather_than_reporting_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """"0 chunks" against a database nobody could reach is a lie, and an actionable-looking one.

    Also pins that the connect is attempted ONCE. It used to be twice, once per check group, so the
    reader whose database was down waited two full connect timeouts before the first line appeared,
    and that reader is the entire audience for this command.
    """
    attempts: list[str] = []

    def _refuse(dsn: str, *, tenant: str | None = None):
        # The keyword matters: `_connect` now sets the RLS tenant GUC on the connection it opens
        # (audit F01), so a stub with the old positional-only signature raises TypeError at CALL
        # time and never records the attempt — which made this assertion read `0 == 1` and look
        # like a behaviour change rather than a stale double.
        attempts.append(dsn)
        raise OSError("nope")

    monkeypatch.setattr(doctor, "_connect", _refuse)
    report = run_checks(dsn="postgresql://x", embedder="hashing", trust_mode="development")

    assert len(attempts) == 1, f"connected {len(attempts)} times; each one is a timeout to wait out"
    database = next(c for c in report.checks if c.name == "database")
    corpus = next(c for c in report.checks if c.name == "corpus")
    assert database.status == "fail"
    assert database.fix is not None
    assert corpus.status == "skip"
    assert "unreachable" in corpus.detail


@pytest.mark.parametrize(
    "block",
    [
        {"command": "recall-mcp"},
        {"command": "python", "args": ["-m", "recall_mcp.server"]},
        {"command": "C:\\venv\\Scripts\\recall-mcp.exe", "args": []},
    ],
)
def test_every_spelling_of_our_own_server_block_is_recognised(block: dict) -> None:
    """A false "not registered" sends somebody re-running an installer that already worked.

    Three spellings are in the wild and all three are ours: the console script (what the plugin and
    `recall setup` write) and `python -m recall_mcp.server` (what the wizard and the manual block in
    `docs/USING_WITH_CLAUDE.md` write).
    """
    assert doctor._is_recall_server(block) is True


@pytest.mark.parametrize("block", [{"command": "some-other-server"}, {}, None, "recall-mcp"])
def test_somebody_elses_server_is_not_counted_as_ours(block: object) -> None:
    assert doctor._is_recall_server(block) is False


def test_the_report_renders_the_repair_line_only_where_there_is_something_to_repair() -> None:
    """An `ok` row with a `fix` would train the reader to skip the arrows that matter."""
    assert "->" not in Check("x", "ok", "fine", fix="do a thing").render()
    assert "-> do a thing" in Check("x", "fail", "broken", fix="do a thing").render()


def test_the_status_markers_are_ascii() -> None:
    """A box-drawing glyph that renders as a replacement character in a Windows console makes a
    diagnostic look broken at the exact moment its credibility matters."""
    for marker in doctor.MARKERS.values():
        assert marker.isascii(), marker


def test_a_password_never_reaches_the_report_even_from_inside_the_exception() -> None:
    """⛔ `redacted_dsn` alone is not enough, and this is the case that proves it.

    `redacted_dsn` cleans the string this module formats. The password can be inside the EXCEPTION,
    because psycopg builds its message out of the connection string it was handed: a malformed DSN
    comes back echoed whole, and `scrub_dsn_secrets` documents that measurement. This command's
    output is the single most likely thing in this package to be pasted into a bug report, so the
    leak would land in public.

    Asserted over the WHOLE rendered report rather than the one check, because a password that
    escaped into any line is equally published.
    """
    secret = "hunter2SuperSecret"
    report = run_checks(
        dsn=f"not-a-dsn://recall:{secret}@127.0.0.1:1/recall",
        embedder="hashing",
        trust_mode="development",
        project_root=Path.cwd(),
    )
    rendered = report.render()
    assert secret not in rendered, rendered
    database = next(c for c in report.checks if c.name == "database")
    assert database.status == "fail"
    assert secret not in database.detail


def test_a_role_that_cannot_see_other_tenants_still_gets_the_index_command() -> None:
    """⛔ **The regression the anti-regression gate caught, pinned.**

    The RLS caveat was first APPENDED to the string `_populated_corpora` returns. Every caller
    tests that string for truth, so on a `NOSUPERUSER NOBYPASSRLS` role with an empty corpus the
    report read:

        These do hold rows: [other tenants are hidden from this role by row-level security]

    and the only correct instruction, `recall index <folder>`, was suppressed — the exact
    wrong-population defect this command exists to end, reproduced on precisely the hardened role
    the RLS fix was written for.

    The caveat is now its own Check, so "cannot see" can never be read as "found something".
    """
    conn = _FakeConn(tables={"chunks": []}, can_bypass_rls=False)
    checks = list(doctor._corpus_checks(conn, table="chunks", tenant="default"))

    corpus = next(c for c in checks if c.name == "corpus")
    assert corpus.status == "fail"
    assert corpus.fix is not None
    assert "index <folder>" in corpus.fix, (
        "an unprivileged role with an empty corpus lost the index instruction: " + repr(corpus.fix)
    )
    assert "hidden from this role" not in corpus.detail, corpus.detail

    # The fact is not dropped, only moved somewhere it cannot be mistaken for a finding.
    visibility = next(c for c in checks if c.name == "rls visibility")
    assert visibility.status == "warn"


DEAD_DSN = "postgresql://recall:recall@127.0.0.1:1/recall"


def test_a_divergent_server_config_is_reported_with_BOTH_pairs_named(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⛔ **The P1 six auditors converged on, and it shipped with no test at all.**

    `recall doctor` inspects its own `--table`/`--tenant` while the MCP server reads
    `RECALL_TABLE`/`RECALL_TENANT`, so the diagnostic could bless one corpus while the server served
    another. The fix reports the divergence rather than inheriting it, deliberately: defaulting the
    CLI flags from the environment would silently redirect `recall index`, which PRUNES, and
    `recall forget`, which is irreversible.

    The architect gate deleted the whole emitting block and the audited set still passed 287/287.
    A conditional emission with no test is indistinguishable from a comment, which is the third time
    that exact shape has shipped in this branch.

    Three assertions, and the third is the one with teeth: the REPAIR must name the SERVED pair, not
    the reported one. It is a single `or` away from being backwards, and backwards would send the
    reader to re-diagnose the corpus they already looked at.
    """
    monkeypatch.setenv("RECALL_TABLE", "quickstart_chunks")
    monkeypatch.setenv("RECALL_TENANT", "work")
    report = run_checks(
        dsn=DEAD_DSN, embedder="hashing", table="chunks", tenant="default",
        trust_mode="development",
    )
    check = next((c for c in report.checks if c.name == "server config"), None)
    assert check is not None, "the divergence was not reported at all"
    assert check.status == "warn"

    assert "'chunks'/'default'" in check.detail, check.detail
    assert "'quickstart_chunks'/'work'" in check.detail, check.detail

    assert check.fix is not None
    assert "--table quickstart_chunks --tenant work" in check.fix, (
        "the repair must name the corpus the SERVER opens, not the one just inspected: " + check.fix
    )


def test_a_matching_server_config_produces_no_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    """The control. Without it, an unconditional warn satisfies the test above.

    Every correctly configured install would then grow a spurious warning, and a diagnostic that
    cries wolf on the common case is one people stop reading — the same argument the calibration
    verdict turns on.
    """
    monkeypatch.setenv("RECALL_TABLE", "chunks")
    monkeypatch.setenv("RECALL_TENANT", "default")
    report = run_checks(
        dsn=DEAD_DSN, embedder="hashing", table="chunks", tenant="default",
        trust_mode="development",
    )
    assert not [c for c in report.checks if c.name == "server config"], (
        "a server pointed at the corpus under audit is not a divergence"
    )


@pytest.mark.parametrize("mode", ["strict", "development"])
def test_a_missing_calibration_is_a_WARNING_in_every_mode(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """⛔ **Asserted over the REAL run, because a hand-built Check is what hid this.**

    `_calibration_check` returned `fail` under the default strict mode, which falsified five
    published sentences promising a missing calibration will not fail a script. The fix made it
    `warn` in every mode; the architect gate flipped it back to `fail` and the audited set still
    passed 287/287.

    `test_a_blocking_failure_exits_non_zero_and_a_warning_does_not` LOOKS like it covers this and
    does not: it asserts over `Check` objects built by hand, so it pins `Report.exit_code()`'s
    arithmetic and says nothing about the verdict this function actually returns.

    Both modes, because "warn only in development" would satisfy a single-mode test while leaving
    every fresh install exiting 1. The strictness must still be REPORTED, since it is what decides
    whether queries are refused — that is the difference between softening the verdict and losing
    the fact.
    """
    monkeypatch.delenv("RECALL_TRUST_MODE", raising=False)
    report = run_checks(dsn=DEAD_DSN, embedder="hashing", trust_mode=mode)
    calibration = next(c for c in report.checks if c.name == "calibration")

    assert calibration.status == "warn", (
        f"a missing calibration exits 1 in {mode} mode, which five published sentences say it does "
        "not"
    )
    assert calibration.fix is not None
    if mode == "strict":
        assert "STRICT" in calibration.detail, calibration.detail

    # The verdict must not be able to block on its own.
    only_calibration = Report([Check("x", "ok", "d"), calibration])
    assert only_calibration.exit_code() == 0
