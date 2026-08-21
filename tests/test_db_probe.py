"""The database probe decides once, and says what it saw. Both were false, and both cost coverage.

The defect these guard against produced a false GREEN, which is why it survived so long. `require_db()`
re-probed on every test that reached a DB fixture, with a 2 second timeout. On a loaded machine a
probe lost that race and the test **skipped**. Nothing failed, nothing warned, and the only visible
trace was a skip count nobody compares between runs.

Measured 2026-08-21, same commit, same machine, same container: an otherwise idle run reported
`6209 passed, 34 skipped`, and a run sharing the host with a type check and a documentation gate
reported `6176 passed, 88 skipped`. 21 of the difference was newly added tests. **54 was tests that
had passed and now skipped.**

Nothing here needs a database. Every probe outcome is injected, so these run identically on a host
with a container and on one without, which is the only way to test a mechanism whose whole subject
is what happens when the database is not there.
"""

from __future__ import annotations

import psycopg
import pytest

from tests import conftest
from tests.conftest import DB_UNREACHABLE, _db_available, _probe_database, db_unreachable_reason


@pytest.fixture(autouse=True)
def _isolated_probe():
    """Clear the memoised probe around every test in this file, and put it back afterwards.

    ⚠️ **Not tidiness. Without the teardown these tests would poison the rest of the session.**
    The probe is cached for the life of the process, deliberately, so a test that clears it and
    leaves an injected *failure* behind hands every later DB test a cached "no database" and turns
    the remainder of the suite into skips. That is the exact failure this file exists to prevent,
    reintroduced by the file that prevents it.

    Clearing on the way out rather than restoring a value: the next real caller then pays one
    genuine probe and gets the truth, which is cheaper than trying to remember what the truth was.
    """
    _probe_database.cache_clear()
    yield
    _probe_database.cache_clear()


def _counting_connect(outcomes: list[BaseException | None]) -> tuple[object, list[int]]:
    """A `psycopg.connect` stand-in that replays `outcomes` and counts its calls."""
    calls = [0]

    def connect(*_args, **_kwargs):
        index = calls[0]
        calls[0] += 1
        outcome = outcomes[index] if index < len(outcomes) else outcomes[-1]
        if outcome is not None:
            raise outcome

        class _Conn:
            def close(self) -> None:
                return None

        return _Conn()

    return connect, calls


# ------------------------------------------------------------------------------------------
# Decided once.
# ------------------------------------------------------------------------------------------


def test_the_probe_connects_once_however_many_times_it_is_asked(monkeypatch) -> None:
    """The fix itself.

    Before this, sixteen `require_db()` call sites each opened a connection per test, so the answer
    to "is there a database" was re-decided hundreds of times per run under whatever load happened
    to exist at that moment. Asking once means every test in a run agrees.
    """
    connect, calls = _counting_connect([None])
    monkeypatch.setattr(psycopg, "connect", connect)

    assert all(_db_available() for _ in range(25))
    assert calls[0] == 1, "the probe reconnected, so the answer is still a function of the load"


def test_a_failed_probe_is_also_remembered(monkeypatch) -> None:
    """Caching only the successes would leave the flake exactly where it was.

    A machine too busy to answer is the case that matters, and it is the case that would re-probe
    forever if the cache held only the happy answer.
    """
    connect, calls = _counting_connect([psycopg.OperationalError("timeout expired")])
    monkeypatch.setattr(psycopg, "connect", connect)

    assert _db_available() is False
    first_round = calls[0]
    assert _db_available() is False
    assert calls[0] == first_round, "a failing probe was retried after it had already answered"


# ------------------------------------------------------------------------------------------
# Decided robustly.
# ------------------------------------------------------------------------------------------


def test_a_configured_database_is_retried_before_it_is_written_off(monkeypatch) -> None:
    """One decision for a whole run must not itself be a coin flip.

    Two lost races followed by a success is exactly the pattern a loaded host produces, and the old
    probe would have taken the first timeout as proof there was no database at all.
    """
    monkeypatch.setenv("RECALL_TEST_DSN", "postgresql://recall:recall@127.0.0.1:5999/recall")
    connect, calls = _counting_connect(
        [psycopg.OperationalError("timeout expired"), psycopg.OperationalError("timeout"), None]
    )
    monkeypatch.setattr(psycopg, "connect", connect)

    assert _db_available() is True
    assert calls[0] == 3


def test_a_configured_probe_waits_longer_than_the_two_seconds_that_caused_this(monkeypatch) -> None:
    """The timeout is asserted, not just the retry count.

    2 seconds is the number that produced 54 silent skips. A test that only checked the retries
    would stay green if someone put the 2 back.
    """
    seen: list[float] = []

    def connect(*_args, **kwargs):
        seen.append(kwargs.get("connect_timeout", 0))
        raise psycopg.OperationalError("timeout expired")

    monkeypatch.setenv("RECALL_TEST_DSN", "postgresql://recall:recall@127.0.0.1:5999/recall")
    monkeypatch.setattr(psycopg, "connect", connect)
    _db_available()

    assert seen, "the probe never attempted a connection"
    assert all(timeout > 2 for timeout in seen), (
        f"a configured database is probed with {seen}, and 2 seconds is the bound that turned "
        "database tests into skips on a busy host"
    )


def test_a_refusal_is_not_retried_even_with_a_database_configured(monkeypatch) -> None:
    """Retry ambiguity, never certainty.

    A refused connection is a complete answer: nothing is listening on that port, and asking twice
    more cannot change it. This is not only tidiness. `test_requires_db_coverage.py` runs a real
    subprocess against a dead port and requires it to skip cleanly, so retrying refusals would
    triple the fixed cost of the guard protecting the whole fixture set, on every run.
    """
    monkeypatch.setenv("RECALL_TEST_DSN", "postgresql://recall:recall@127.0.0.1:5999/recall")
    connect, calls = _counting_connect([psycopg.OperationalError("connection refused")])
    monkeypatch.setattr(psycopg, "connect", connect)

    assert _db_available() is False
    assert calls[0] == 1, "a definitive refusal was retried"


def test_an_unconfigured_run_waits_only_briefly(monkeypatch) -> None:
    """`_UNCONFIGURED_DSN` normally refuses instantly, but a host that DROPs instead would wait.

    Every database-less run on every contributor's machine pays this, so it is bounded well below
    the patience a configured database earns.
    """
    seen: list[float] = []

    def connect(*_args, **kwargs):
        seen.append(kwargs.get("connect_timeout", 0))
        raise psycopg.OperationalError("connection refused")

    monkeypatch.delenv("RECALL_TEST_DSN", raising=False)
    monkeypatch.setattr(psycopg, "connect", connect)

    assert _db_available() is False
    assert seen == [2]


# ------------------------------------------------------------------------------------------
# And says what it saw.
# ------------------------------------------------------------------------------------------


def test_the_reason_keeps_the_actionable_constant_as_its_prefix(monkeypatch) -> None:
    """`test_requires_db_coverage.py` asserts `DB_UNREACHABLE in proc.stdout` on a real subprocess.

    Rebuilding the wording instead of extending it would break that guard, and it would break it
    for a reason that has nothing to do with what the guard checks.
    """
    connect, _calls = _counting_connect([psycopg.OperationalError("connection refused")])
    monkeypatch.setattr(psycopg, "connect", connect)

    reason = db_unreachable_reason()
    assert reason.startswith(DB_UNREACHABLE)


def test_a_timeout_and_a_refusal_do_not_read_the_same(monkeypatch) -> None:
    """The half of the defect a cache alone does not fix.

    "Nothing is listening" and "something is listening and did not answer in time" want opposite
    responses from whoever reads the report: start a container, or stop blaming the code. The old
    reason spelled both of them "not reachable".
    """
    connect, _calls = _counting_connect([psycopg.OperationalError("connection refused")])
    monkeypatch.setattr(psycopg, "connect", connect)
    refused = db_unreachable_reason()

    _probe_database.cache_clear()
    connect, _calls = _counting_connect([psycopg.OperationalError("timeout expired")])
    monkeypatch.setattr(psycopg, "connect", connect)
    timed_out = db_unreachable_reason()

    assert refused != timed_out
    assert "refused" in refused
    assert "timeout" in timed_out


def test_a_reachable_database_gets_the_bare_constant(monkeypatch) -> None:
    """No diagnosis to append when there was no failure, and none is invented."""
    connect, _calls = _counting_connect([None])
    monkeypatch.setattr(psycopg, "connect", connect)

    assert db_unreachable_reason() == DB_UNREACHABLE


def test_an_error_with_no_message_still_names_its_class(monkeypatch) -> None:
    """`str(exc)` is the empty string for several psycopg errors.

    A reason ending in "Probe saw: " would be worse than no diagnosis, because it looks like the
    mechanism ran and found nothing worth saying.
    """
    connect, _calls = _counting_connect([psycopg.OperationalError()])
    monkeypatch.setattr(psycopg, "connect", connect)

    assert "OperationalError" in db_unreachable_reason()


def test_a_multiline_error_is_collapsed_to_one_line(monkeypatch) -> None:
    """psycopg connection errors span lines, and this text becomes a pytest skip reason.

    Left alone it breaks the `-rs` report into fragments that no longer read as one cause.
    """
    connect, _calls = _counting_connect(
        [psycopg.OperationalError("connection failed:\n\thost unreachable\n\tno route")]
    )
    monkeypatch.setattr(psycopg, "connect", connect)

    reason = db_unreachable_reason()
    assert "\n" not in reason
    assert "host unreachable" in reason


# ------------------------------------------------------------------------------------------
# The property the rest of the suite depends on.
# ------------------------------------------------------------------------------------------


def test_require_db_skips_with_the_same_reason_the_report_will_show(monkeypatch) -> None:
    """`require_db()` is what sixteen fixtures call, so its skip is the one operators actually read.

    Asserted against the injected probe rather than against the real database. An earlier draft of
    this test compared the collection-time mark with a fresh call to `_db_available()`, which is a
    test whose outcome depends on whether a container happens to be running: flaky in exactly the
    way this file exists to remove.
    """
    connect, _calls = _counting_connect([psycopg.OperationalError("connection refused")])
    monkeypatch.setattr(psycopg, "connect", connect)

    with pytest.raises(pytest.skip.Exception) as skipped:
        conftest.require_db()
    assert str(skipped.value).startswith(DB_UNREACHABLE)
    assert "refused" in str(skipped.value)


def test_require_db_is_silent_when_the_database_answers(monkeypatch) -> None:
    connect, _calls = _counting_connect([None])
    monkeypatch.setattr(psycopg, "connect", connect)
    conftest.require_db()  # must not raise
