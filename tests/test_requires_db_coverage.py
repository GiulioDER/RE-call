"""Every test that reaches the database must carry `@requires_db`, and this proves it.

`tests/test_store_cosines_for.py` and `tests/test_store_query_latency.py` both requested
`make_store` without the mark. With no local Postgres they did not skip, they produced 3 failures
and 7 errors of `psycopg.errors.ConnectionTimeout`, and each attempt burned the connect timeout:
213 s to arrive at a red that meant nothing. Every other DB test in the suite skipped cleanly in
the same run. A suite that reports red when it is actually clean teaches people to ignore red,
which is the expensive part of the defect, not the two files.

Nothing prevented that, and nothing prevented it recurring, because the obligation lived in
whether the author remembered. So it is derived from source here instead, in the same shape
`recall/store.py` already uses for `TIMED_PUBLIC_METHODS` and `STORE_QUERY_LEGS`: declare the set,
then require the declaration to equal what the code actually does.

The load-bearing part is the transitive walk. Seven of the ten original offenders never named a
conftest fixture: they requested a module-local `store` fixture that requested `make_store`. A
guard matching only direct requests would have passed on them, and today 68 of the suite's 202
DB-backed tests reach the database that way.
"""

from __future__ import annotations

import ast
import functools
import pathlib

import pytest

from tests.conftest import DB_BACKED_FIXTURES

TESTS_DIR = pathlib.Path(__file__).parent

#: Anything naming this in a conftest fixture body is talking to the database.
DSN_REFERENCE = "TEST_DSN"


@functools.lru_cache(maxsize=None)
def _parse(path: pathlib.Path) -> ast.Module:
    """Cached: several guards below scan the whole `tests/` tree, and parsing it once per guard
    both wastes the time and re-emits any `SyntaxWarning` a scanned file happens to carry."""
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _fixture_decorator(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> ast.expr | None:
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


def _requested(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """Fixture names this function asks for, which is simply its parameter list."""
    a = fn.args
    return [x.arg for x in (*a.posonlyargs, *a.args, *a.kwonlyargs)]


def _functions(tree: ast.Module) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    # `ast.walk`, not `tree.body`: several modules put tests in `class Test...` bodies, and a
    # top-level-only scan would skip them without saying so.
    return [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)]


def _close_over_fixtures(fixtures: dict[str, list[str]], seed: set[str]) -> set[str]:
    """Grow `seed` to a fixpoint: a fixture requesting a DB-backed fixture is itself DB-backed."""
    db = set(seed)
    changed = True
    while changed:
        changed = False
        for name, requests in fixtures.items():
            if name not in db and db.intersection(requests):
                db.add(name)
                changed = True
    return db


def _conftest_fixtures() -> dict[str, tuple[list[str], ast.expr, str]]:
    tree = _parse(TESTS_DIR / "conftest.py")
    out: dict[str, tuple[list[str], ast.expr, str]] = {}
    for fn in _functions(tree):
        dec = _fixture_decorator(fn)
        if dec is not None:
            out[fn.name] = (_requested(fn), dec, ast.unparse(fn))
    return out


def test_db_backed_fixtures_matches_the_conftest_source() -> None:
    """The declaration must equal the requestable fixtures that actually touch the database.

    A new DB fixture added to conftest and forgotten here would exempt every test using it from
    the coverage guard below, in silence. That is the same drift that made the original defect
    possible, one level up, so it gets checked rather than trusted.
    """
    fixtures = _conftest_fixtures()
    assert fixtures, "found no fixtures in conftest.py; the parse is wrong"

    touches_db = {name for name, (_, _, src) in fixtures.items() if DSN_REFERENCE in src}
    requests = {name: reqs for name, (reqs, _, _) in fixtures.items()}
    touches_db = _close_over_fixtures(requests, touches_db)
    assert touches_db, f"no conftest fixture references {DSN_REFERENCE}; the parse is wrong"

    requestable = {name for name in touches_db if not _is_autouse(fixtures[name][1])}
    assert requestable == set(DB_BACKED_FIXTURES), (
        f"DB_BACKED_FIXTURES is {sorted(DB_BACKED_FIXTURES)} but the requestable fixtures that "
        f"reach the database are {sorted(requestable)}. Update the tuple: the coverage guard is "
        "only as wide as it."
    )


def test_an_autouse_db_fixture_guards_itself() -> None:
    """Autouse fixtures are exempt from marking, so they must refuse the database on their own.

    They cannot be requested, so no test can be marked on their account, which is why
    `DB_BACKED_FIXTURES` leaves them out. That exemption is only safe while each one checks
    `_db_available()` before connecting. One that did not would fail the entire session with no
    DB, and this is what stops the exemption from becoming a hole.
    """
    fixtures = _conftest_fixtures()
    unguarded = [
        name
        for name, (_, dec, src) in fixtures.items()
        if DSN_REFERENCE in src and _is_autouse(dec) and "_db_available" not in src
    ]
    assert not unguarded, (
        f"{unguarded} are autouse, reach the database, and never check `_db_available()`. An "
        "autouse fixture cannot be skipped per test, so it must no-op without a DB."
    )


def _module_is_marked(tree: ast.Module) -> bool:
    """Does a module-level `pytestmark` apply `requires_db` to everything in the file?

    Both shapes in the suite count: `pytestmark = requires_db` and the list form
    `pytestmark = [requires_db, pytest.mark.timeout(300)]`.
    """
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets):
            continue
        value = node.value
        applied = value.elts if isinstance(value, ast.List | ast.Tuple) else [value]
        if any(isinstance(e, ast.Name) and e.id == "requires_db" for e in applied):
            return True
    return False


def _collect_db_tests() -> tuple[list[str], list[str]]:
    """Every test reaching the database, split into (all of them, the unmarked offenders)."""
    found: list[str] = []
    offenders: list[str] = []
    for path in sorted(TESTS_DIR.glob("test_*.py")):
        tree = _parse(path)
        functions = _functions(tree)
        fixtures = {fn.name: _requested(fn) for fn in functions if _fixture_decorator(fn)}
        db = _close_over_fixtures(fixtures, set(DB_BACKED_FIXTURES))
        marked_module = _module_is_marked(tree)
        for fn in functions:
            if not fn.name.startswith("test_") or _fixture_decorator(fn):
                continue
            used = db.intersection(_requested(fn))
            if not used:
                continue
            found.append(f"{path.name}::{fn.name}")
            marked = any(
                isinstance(d, ast.Name) and d.id == "requires_db" for d in fn.decorator_list
            )
            if not (marked_module or marked):
                offenders.append(f"{path.name}::{fn.name} requests {sorted(used)}")
    return found, offenders


def test_every_db_backed_test_carries_the_marker() -> None:
    """The guard itself. Unmarked, these do not skip without a DB, they time out and go red."""
    found, offenders = _collect_db_tests()
    assert found, "found no DB-backed tests at all; the scan is broken, not the suite"
    assert not offenders, (
        "these tests reach the database but do not carry `@requires_db`, so without a local "
        "Postgres they fail on a connection timeout instead of skipping:\n  "
        + "\n  ".join(offenders)
    )


def test_the_scan_sees_tests_that_reach_the_db_through_a_local_fixture() -> None:
    """Non-vacuity for the transitive walk, which is the half that is easy to get wrong.

    Seven of the ten tests that prompted this guard requested a module-local `store` fixture, not
    a conftest one. Were `_close_over_fixtures` reduced to direct requests, every assertion above
    would still pass while the majority of the suite's DB tests went unexamined. So require the
    indirect case to be genuinely present.
    """
    found, _ = _collect_db_tests()
    direct = set()
    for path in sorted(TESTS_DIR.glob("test_*.py")):
        tree = _parse(path)
        for fn in _functions(tree):
            if fn.name.startswith("test_") and set(DB_BACKED_FIXTURES).intersection(_requested(fn)):
                direct.add(f"{path.name}::{fn.name}")

    indirect = set(found) - direct
    assert indirect, (
        "every DB-backed test was found by a direct fixture request, so the transitive walk is "
        "doing nothing and a `store`-style wrapper fixture would hide a test from this guard."
    )


@pytest.mark.parametrize("shape", ["pytestmark = requires_db", "pytestmark = [requires_db]"])
def test_module_level_marker_is_recognised_in_both_shapes(shape: str) -> None:
    """Both spellings appear in the suite, and reading only one would produce false offenders."""
    assert _module_is_marked(ast.parse(shape))
    assert not _module_is_marked(ast.parse(shape.replace("requires_db", "requires_fastembed")))
