"""Reaching the database without a database must SKIP, and this proves the mechanism that makes
it skip.

`tests/test_store_cosines_for.py` and `tests/test_store_query_latency.py` both requested
`make_store` without `@requires_db`. With no local Postgres they did not skip, they produced 3
failures and 7 errors of `psycopg.errors.ConnectionTimeout`, and each attempt burned the connect
timeout: 213 s to arrive at a red that meant nothing, while every other DB test in the same run
skipped cleanly. A suite that reports red when it is actually clean teaches people to ignore red,
which is the expensive part of the defect, not the two files.

The first fix was the mark, and the first guard tried to prove by static analysis that every test
reaching the database carried one. That guard was wrong twice, in the same direction both times:
it stayed green while the property it named went unchecked over a growing share of the suite,
because it derived truth from a hand-written list (first of fixtures, then of DSN names). Three
audit rounds turned up twenty findings, and the third showed the analysis promoting 227 ordinary
local names to "holds a connection string" and costing 14.8 s per run in a checkout with sibling
worktrees. An approximation of a dependency graph is the wrong instrument for a property the
runtime can simply enforce.

So the fixtures refuse instead. `make_store`, `cli_table` and `unprivileged_dsn` all call
`conftest.require_db()`, which skips when `_db_available()` is false. That is exact, needs no
parsing, and makes the two files that started this skip correctly with every mark removed.

BE PRECISE ABOUT THE SCOPE, because the previous draft of this docstring was not. The refusal
covers the tests that reach the database THROUGH those fixtures, roughly 202 of 391. It does NOT
cover the other 189: 164 go through a module-local fixture that opens its own connection, like
`tests/test_tenancy.py::tenant_table` calling `apply_migrations(TEST_DSN, ...)`, and 25 connect in
their own body. For those, `@requires_db` is still the only protection, and removing one still
costs the psycopg connect timeout per test. They were never protected by a runtime mechanism, so
this is not a regression, but a file claiming to prove a property must not claim more than it
proves. `require_db()` is exported so those fixtures can opt in one line at a time.

What is left for a guard is the mechanism: the declared fixture set is what conftest actually
defines, every one of them refuses, they refuse with the reason operators are told to act on, and,
proved by running rather than by reading, unmarked tests requesting them skip. `@requires_db`
remains worth carrying, because skipping at collection never sets the fixture up at all, but it is
now an optimisation rather than the only thing between a forgotten mark and a 213 s red.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from tests.conftest import DB_BACKED_FIXTURES, DB_UNREACHABLE

TESTS_DIR = pathlib.Path(__file__).parent
CONFTEST = TESTS_DIR / "conftest.py"

#: Names in conftest that hold a connection string. A fixture mentioning one reaches the database.
#: Matched as `ast.Name` nodes, not as a substring of the source: a substring test also fires on a
#: docstring that merely discusses the DSN, and on `RECALL_TEST_DSN`, which contains `TEST_DSN`.
DSN_NAMES = frozenset({"TEST_DSN", "_UNCONFIGURED_DSN"})

AnyFunc = ast.FunctionDef | ast.AsyncFunctionDef


def _conftest_tree() -> ast.Module:
    return ast.parse(CONFTEST.read_text(encoding="utf-8"), filename=str(CONFTEST))


def _fixture_decorator(fn: AnyFunc) -> ast.expr | None:
    """The `@pytest.fixture` decorator on `fn`, or None. Matches the bare and called forms."""
    for dec in fn.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        if isinstance(target, ast.Attribute) and target.attr == "fixture":
            return dec
        if isinstance(target, ast.Name) and target.id == "fixture":
            return dec
    return None


def _is_autouse(dec: ast.expr) -> bool:
    if not isinstance(dec, ast.Call):
        return False
    return any(
        kw.arg == "autouse" and isinstance(kw.value, ast.Constant) and kw.value.value is True
        for kw in dec.keywords
    )


def _requested(fn: AnyFunc) -> set[str]:
    a = fn.args
    return {x.arg for x in (*a.posonlyargs, *a.args, *a.kwonlyargs)}


def _reaches_db(fn: AnyFunc) -> bool:
    return any(isinstance(n, ast.Name) and n.id in DSN_NAMES for n in ast.walk(fn))


def _conftest_fixtures() -> dict[str, tuple[AnyFunc, ast.expr]]:
    tree = _conftest_tree()
    out: dict[str, tuple[AnyFunc, ast.expr]] = {}
    for node in ast.walk(tree):
        if isinstance(node, AnyFunc):
            dec = _fixture_decorator(node)
            if dec is not None:
                out[node.name] = (node, dec)
    return out


def _close_over(requests: dict[str, set[str]], seed: set[str]) -> set[str]:
    """Grow `seed` to a fixpoint: a fixture requesting a DB fixture is itself one.

    Split out and given a synthetic test of its own because conftest happens not to contain a
    wrapper fixture today, so this loop cannot fail against the real file. An untestable branch is
    one nobody notices breaking, which is the shape of defect this whole file is about.

    Terminates: each continuing iteration adds a name, and the result is bounded by
    `seed | set(requests)`.
    """
    names = set(seed)
    changed = True
    while changed:
        changed = False
        for name, requested in requests.items():
            if name not in names and names.intersection(requested):
                names.add(name)
                changed = True
    return names


def _db_fixtures() -> dict[str, tuple[AnyFunc, ast.expr]]:
    """Conftest fixtures that reach the database, directly or by requesting one that does."""
    fixtures = _conftest_fixtures()
    names = _close_over(
        {name: _requested(fn) for name, (fn, _) in fixtures.items()},
        {name for name, (fn, _) in fixtures.items() if _reaches_db(fn)},
    )
    return {name: fixtures[name] for name in names}


def _refuses_without_a_database(fn: AnyFunc) -> bool:
    """Does this fixture refuse to proceed when the database is unreachable?

    Two accepted shapes, and only two, because a detector that accepts many shapes accepts wrong
    ones. A call to `require_db()`, which is how every requestable fixture does it; or, for the
    autouse fixture, which cannot skip anything and must no-op instead, an `if` on `_db_available`
    with a bail in ITS OWN body.

    Three properties, all of them load-bearing, and each pinned by a row of the table below.

    POSITION: the refusal must be the FIRST statement after any docstring, not merely present
    somewhere. A fixture that connects and then refuses has already paid the connect timeout, and
    scanning for a match anywhere accepted exactly that.

    POLARITY: the `if` must test `not <probe>`. `if _db_available(): return` bails precisely when
    the database IS reachable, which is the opposite of a guard, and reads identically to a scan
    that only asks whether the probe is mentioned.

    DEPTH: `fn.body`, so a `require_db()` sitting inside a nested helper that nobody calls is not
    mistaken for the fixture's own refusal, and `node.body`, so a bail in the `else` branch is not
    either.
    """
    for node in fn.body:
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            continue  # module or function docstring, not a statement that does anything
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            func = node.value.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            return name == "require_db"
        if isinstance(node, ast.If):
            test = node.test
            if not (isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not)):
                return False
            calls_probe = any(
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Name | ast.Attribute)
                and (inner.func.id if isinstance(inner.func, ast.Name) else inner.func.attr)
                == "_db_available"
                for inner in ast.walk(test.operand)
            )
            if not calls_probe:
                return False
            # The branch's own contents matter, not just that a `Return` is somewhere in it. The
            # same ordering rule applied one level in: `if not _db_available(): conn =
            # psycopg.connect(...); return` bails, but connects first, which is the defect this
            # whole file is about. A generator fixture must yield before returning, or pytest
            # reports "did not yield a value" for every test that uses it.
            shape = [type(stmt) for stmt in node.body]
            if any(isinstance(n, ast.Yield) for n in ast.walk(fn)):
                return shape == [ast.Expr, ast.Return] and isinstance(
                    node.body[0].value, ast.Yield
                )
            return shape == [ast.Return]
        return False  # the first thing this fixture does is something other than refusing
    return False


def test_db_backed_fixtures_matches_the_conftest_source() -> None:
    """The declaration must equal the fixtures that actually reach the database.

    A new DB fixture added to conftest and left out of the tuple would never be checked for the
    refusal below, which is the whole protection. That is the same drift that made the original
    defect possible, one level up, so it is checked rather than trusted.
    """
    fixtures = _conftest_fixtures()
    assert fixtures, "found no fixtures in conftest.py; the parse is wrong"

    db = _db_fixtures()
    assert db, "no conftest fixture references a DSN; the parse is wrong"

    requestable = {name for name, (_, dec) in db.items() if not _is_autouse(dec)}
    assert requestable == set(DB_BACKED_FIXTURES), (
        f"DB_BACKED_FIXTURES is {sorted(DB_BACKED_FIXTURES)} but the requestable fixtures that "
        f"reach the database are {sorted(requestable)}. Update the tuple, or the new one's "
        "refusal goes unchecked."
    )


def test_every_db_fixture_refuses_without_a_database() -> None:
    """The load-bearing assertion. Without this, a forgotten mark is a 213 s red again.

    Every route into conftest's database goes through one of these fixtures, so a fixture that
    connects before checking `_db_available()` reopens the exact hole that
    `test_store_cosines_for.py` and `test_store_query_latency.py` fell into.
    """
    unguarded = [name for name, (fn, _) in _db_fixtures().items() if not _refuses_without_a_database(fn)]
    assert not unguarded, (
        f"{unguarded} do not refuse a missing database in the one shape this guard accepts. The "
        "rule is deliberately narrow: `require_db()` must be the FIRST statement after the "
        "docstring, with nothing before it, or (for the autouse fixture, which cannot skip) "
        "`if not _db_available():` as the first statement with a bare bail as its whole body. "
        "If your refusal is present but lower down, move it to the top: a fixture that connects "
        "and then refuses has already paid the psycopg connect timeout, which is the entire "
        "defect this file exists to prevent."
    )


def test_the_skip_reason_is_the_one_operators_are_told_to_act_on() -> None:
    """A skip nobody can act on is only marginally better than a failure.

    The earlier form of the second assertion was `DB_UNREACHABLE in CONFTEST.read_text()`, which
    is true BY CONSTRUCTION: the constant is a literal in that very file, so its value is
    necessarily a substring of the file's own source whatever it is. It could not fail, which is
    the most expensive kind of check, because it reads as protection. What the constant exists for
    is that every refusal cites it rather than a hardcoded copy, so that is what is asserted.
    """
    assert "docker compose up -d" in DB_UNREACHABLE

    require_db = next(
        n
        for n in ast.walk(_conftest_tree())
        if isinstance(n, AnyFunc) and n.name == "require_db"
    )
    skips = [
        call
        for call in ast.walk(require_db)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "skip"
    ]
    assert len(skips) == 1, f"expected one skip in require_db, found {len(skips)}"
    assert any(
        isinstance(arg, ast.Name) and arg.id == "DB_UNREACHABLE" for arg in skips[0].args
    ), (
        "require_db() skips with something other than DB_UNREACHABLE, so a DB-less run reports a "
        "different reason depending on how the test was skipped."
    )


def _only(source: str) -> AnyFunc:
    return next(n for n in ast.walk(ast.parse(source)) if isinstance(n, AnyFunc))


@pytest.mark.parametrize(
    ("label", "source", "refuses"),
    [
        (
            "mentions the probe but never branches on it",
            'def f():\n    """Callers should consult _db_available() first."""\n'
            "    x = _db_available\n    return psycopg.connect(TEST_DSN)\n",
            False,
        ),
        (
            "branches, but on something that is not the probe",
            "def f():\n    if os.environ.get('CI'):\n        pytest.skip('x')\n"
            "    return psycopg.connect(TEST_DSN)\n",
            False,
        ),
        (
            "negates something that is not the probe, and does bail",
            # Isolates the "is it the probe?" check specifically: this row satisfies position,
            # polarity and bail, so it is the only one whose verdict flips if that check goes.
            "def f():\n    if not os.environ.get('CI'):\n        return None\n"
            "    return psycopg.connect(TEST_DSN)\n",
            False,
        ),
        (
            "connects first and refuses afterwards, having already paid the timeout",
            "def f():\n    apply_migrations(TEST_DSN, table='t', dim=4)\n    require_db()\n"
            "    yield 't'\n",
            False,
        ),
        (
            "bails when the database IS reachable, which is a guard with its polarity inverted",
            "def f():\n    if _db_available():\n        yield\n        return\n"
            "    yield psycopg.connect(TEST_DSN)\n",
            False,
        ),
        (
            "refuses only inside a nested helper that nothing calls",
            "def f():\n    def unused():\n        require_db()\n"
            "    return psycopg.connect(TEST_DSN)\n",
            False,
        ),
        (
            "connects inside the guarded branch before returning",
            "def f():\n    if not _db_available():\n        conn = psycopg.connect(TEST_DSN)\n"
            "        return\n    yield 't'\n",
            False,
        ),
        (
            "a generator whose guard returns without yielding, which errors every test using it",
            "def f():\n    if not _db_available():\n        return\n"
            "    yield psycopg.connect(TEST_DSN)\n",
            False,
        ),
        (
            "does anything at all before refusing",
            # Deliberately strict, and this row says so. The rule is "the refusal is the first
            # statement after the docstring", with no judgement about whether what precedes it
            # looks harmless, because deciding which statements can reach a socket is the sort of
            # approximation that made the previous three versions of this file wrong.
            "def f():\n    x = 1\n    require_db()\n    yield x\n",
            False,
        ),
        (
            "branches on the probe but does not bail",
            "def f():\n    if not _db_available():\n        print('no database')\n"
            "    return psycopg.connect(TEST_DSN)\n",
            False,
        ),
        (
            "bails only in the else branch, so the guarded path is the connecting one",
            "def f():\n    if not _db_available():\n        print('x')\n    else:\n"
            "        return psycopg.connect(TEST_DSN)\n",
            False,
        ),
        (
            "the only return belongs to a helper defined inside the branch",
            "def f():\n    if not _db_available():\n        def inner():\n            return 1\n"
            "    return psycopg.connect(TEST_DSN)\n",
            False,
        ),
        (
            "calls require_db, which is how every requestable fixture does it",
            "def f():\n    require_db()\n    return psycopg.connect(TEST_DSN)\n",
            True,
        ),
        (
            "branches on the probe and returns, which is how the autouse one bails",
            "def f():\n    if not _db_available():\n        yield\n        return\n"
            "    yield psycopg.connect(TEST_DSN)\n",
            True,
        ),
    ],
)
def test_the_refusal_detector_distinguishes_a_real_check_from_a_mention(
    label: str, source: str, refuses: bool
) -> None:
    """Non-vacuity, one negative shape per thing the detector actually tests.

    The earlier version of this test used a negative with no `if` in it at all, so neither the
    "branches on the probe" nor the "actually bails" half of the detector was exercised: mutating
    either one away left it green. Each row here fails exactly one of them.
    """
    assert _refuses_without_a_database(_only(source)) is refuses, label


def test_a_wrapper_fixture_inherits_the_database_from_what_it_requests() -> None:
    """Conftest has no wrapper fixture today, so `_close_over` cannot fail against the real file.

    If one is ever added, `store(make_store)`, it must be recognised as DB-backed so its refusal
    is checked too. Pinned on a synthetic rather than left to the day someone writes it.
    """
    requests = {"make_store": set(), "store": {"make_store"}, "pure": {"tmp_path"}}
    assert _close_over(requests, {"make_store"}) == {"make_store", "store"}
    assert "pure" not in _close_over(requests, {"make_store"})


def test_every_db_fixture_skips_when_asked_for_without_a_database(
    pytestconfig: pytest.Config,
) -> None:
    """The refusal proved by RUNNING it, not by reading it.

    `_refuses_without_a_database` reads the source, and a source reader can be fooled by a shape it
    does not understand. This runs a real pytest session in a subprocess with the DSN pointed at a
    closed port, on tests that carry NO mark, and requires every one of them to be `skipped`. That
    is the actual claim of this file: an unmarked test cannot reach the database without one.

    One subprocess for all of them rather than one each. The session cost is conftest's imports
    plus a `_db_available()` timeout, so per-fixture parametrisation tripled a fixed cost to prove
    the same thing; the individual outcomes are still checked, by name, in the `-rs` report.
    """
    import os
    import re
    import subprocess
    import sys
    import uuid

    names = sorted(DB_BACKED_FIXTURES)
    body = "\n\n".join(
        f"def test_probe_{name}({name}):\n"
        f'    raise AssertionError("reached {name} without a database")'
        for name in names
    )
    # Under `tests/` so `conftest.py` applies to it, but deliberately NOT matching `test_*.py`, so
    # a full suite running concurrently cannot collect it. Unique per run, because a fixed name
    # means two concurrent sessions delete each other's file mid-collection. A leading dot would
    # make the module name a relative import and fail collection.
    tmp = TESTS_DIR / f"probe_db_refusal_{uuid.uuid4().hex[:8]}.py"
    env = {
        **os.environ,
        # A port nothing listens on, so `_db_available()` is false however the machine is set up,
        # which is what lets this run identically with or without a local container.
        "RECALL_TEST_DSN": "postgresql://recall:recall@127.0.0.1:1/recall",
        # Leave nothing behind under tests/: the abandoned per-fixture version of this test left
        # three stale `.pyc` files in `tests/__pycache__`.
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    # `_reject_unsafe_test_dsn` refuses when the two DSNs are equal, which would surface here as a
    # confusing collection error rather than a clear message.
    env.pop("RECALL_DSN", None)
    try:
        # Written inside the try, so an interrupt between creating and deleting it still cleans up.
        tmp.write_text(body + "\n", encoding="utf-8", newline="\n")
        proc = subprocess.run(
            # `-v` rather than `-q`: the `-rs` report identifies a skip by file and line, so the
            # per-fixture assertions below need the test NAMES that verbose output prints.
            [sys.executable, "-m", "pytest", str(tmp), "-v", "-p", "no:randomly", "-rs"],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(pytestconfig.rootpath),
        )
    finally:
        tmp.unlink(missing_ok=True)

    assert proc.returncode == 0, f"the probe session did not succeed\n{proc.stdout[-3000:]}"
    # Anchored: a bare `f"{len(names)} skipped" in stdout` also matches "13 skipped".
    assert re.search(rf"(?<!\d){len(names)} skipped", proc.stdout), (
        "requesting the DB fixtures with no reachable database did not skip them all. This is "
        f"the regression the whole file exists to prevent.\n{proc.stdout[-3000:]}"
    )
    for name in names:
        # `SKIPPED`, not just the name: under `-v` the name is printed for a FAILED test too, so
        # matching the name alone checked that the probe was collected, not that it skipped.
        assert re.search(rf"test_probe_{name}\b.*SKIPPED", proc.stdout), (
            f"{name!r} was not reported as skipped, so its refusal is unproven"
        )
    assert DB_UNREACHABLE in proc.stdout, "the skip did not cite the actionable reason"


def test_every_dsn_name_still_exists_in_conftest() -> None:
    """`DSN_NAMES` must name constants that are really there.

    The detector above matches `ast.Name` nodes against this set, so a name that no longer exists
    in conftest contributes nothing and takes no test with it: the set silently shrinks and the
    fixtures it was meant to catch stop being classified as reaching the database. That is the
    failure mode where a guard keeps passing while covering less, which is worse than a red one.

    It has already happened once. `_LOCAL_DEV_DSN` was removed when the suite stopped defaulting to
    the shared container on port 5432, and this set still named it for one commit. Nothing failed,
    because no fixture referenced the new `_UNCONFIGURED_DSN` yet; the next fixture that did would
    have been classified as DB-free and escaped the refusal check entirely.
    """
    assigned = {
        target.id
        for node in ast.walk(_conftest_tree())
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    missing = DSN_NAMES - assigned
    assert not missing, (
        f"DSN_NAMES names {sorted(missing)}, which conftest.py no longer assigns. Update the set "
        "to the current constant, or the detector silently stops covering those fixtures."
    )
