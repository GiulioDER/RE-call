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

A test reaches the database two ways, and the FIRST version of this guard only understood one of
them. It walked fixture-to-fixture request edges out from `DB_BACKED_FIXTURES`, which finds
`store(make_store)` wrappers but is blind to a module-local fixture that opens its own connection
and asks conftest for nothing, like `tests/test_tenancy.py::tenant_table` calling
`apply_migrations(TEST_DSN, ...)` directly. There are 22 such fixtures, and the guard examined 202
of the suite's 363 DB-backed tests while its own docstring claimed it proved the property for all
of them. Deleting a mark from any of the other 161 left it green. So the seed now includes any
fixture that names a DSN itself, which is the same classifier `test_db_backed_fixtures_matches_
the_conftest_source` already applied to conftest, and both halves have a non-vacuity test.

Where this guard cannot read a construct it FAILS rather than skipping it, because a shape it
silently ignores is a test it silently exempts. `test_no_module_uses_a_shape_this_guard_cannot_read`
is that backstop.
"""

from __future__ import annotations

import ast
import functools
import pathlib

import pytest

from tests.conftest import DB_BACKED_FIXTURES

TESTS_DIR = pathlib.Path(__file__).parent

#: Module-level names that hold a connection string. A fixture mentioning one of these reaches the
#: database on its own account. Matched as `ast.Name` nodes rather than as a substring of the
#: unparsed source: a substring test also fires on the docstring of a fixture that merely discusses
#: the DSN, and on the unrelated environment variable `RECALL_TEST_DSN`, which contains `TEST_DSN`.
DSN_NAMES = frozenset({"TEST_DSN", "_LOCAL_DEV_DSN"})

#: pytest's default `python_files`. `pyproject.toml` does not override it, so both patterns collect.
TEST_FILE_PATTERNS = ("test_*.py", "*_test.py")

AnyFunc = ast.FunctionDef | ast.AsyncFunctionDef


def _test_modules() -> list[pathlib.Path]:
    """Every file pytest would collect from `tests/`, including any future subdirectory."""
    found: set[pathlib.Path] = set()
    for pattern in TEST_FILE_PATTERNS:
        found.update(TESTS_DIR.rglob(pattern))
    return sorted(p for p in found if "__pycache__" not in p.parts)


def _parse(path: pathlib.Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


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


def _registered_name(fn: AnyFunc, dec: ast.expr) -> str:
    """The name pytest registers the fixture under, which `@pytest.fixture(name=...)` overrides.

    Keying on `fn.name` alone would put `_store_impl` in the DB set while tests request `store`,
    and every one of them would be exempt.
    """
    if isinstance(dec, ast.Call):
        for kw in dec.keywords:
            if kw.arg == "name" and isinstance(kw.value, ast.Constant):
                return str(kw.value.value)
    return fn.name


def _requested(fn: AnyFunc) -> set[str]:
    """Fixture names this function consumes: its parameters plus any `usefixtures` marker."""
    a = fn.args
    names = {x.arg for x in (*a.posonlyargs, *a.args, *a.kwonlyargs)}
    for dec in fn.decorator_list:
        if not isinstance(dec, ast.Call):
            continue
        func = dec.func
        if isinstance(func, ast.Attribute) and func.attr == "usefixtures":
            names.update(a.value for a in dec.args if isinstance(a, ast.Constant))
    return names


def _reaches_dsn(fn: AnyFunc) -> bool:
    """Does this function's body name a DSN, and so open its own connection?"""
    return any(isinstance(n, ast.Name) and n.id in DSN_NAMES for n in ast.walk(fn))


def _functions(tree: ast.AST) -> list[AnyFunc]:
    # `ast.walk`, not `tree.body`: several modules put tests and fixtures in `class Test...`
    # bodies, and a top-level-only scan would skip them without saying so.
    return [n for n in ast.walk(tree) if isinstance(n, AnyFunc)]


def _names_requires_db(node: ast.expr) -> bool:
    """`requires_db`, or `conftest.requires_db`. Both are the same mark."""
    if isinstance(node, ast.Name):
        return node.id == "requires_db"
    return isinstance(node, ast.Attribute) and node.attr == "requires_db"


def _body_applies_the_mark(body: list[ast.stmt]) -> bool:
    """Does a `pytestmark` statement in this body apply `requires_db` to everything under it?

    Covers the plain assignment, the annotated form, and augmentation, at module OR class level.
    """
    for node in body:
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign | ast.AugAssign):
            targets = [node.target]
        else:
            continue
        if not any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in targets):
            continue
        value = node.value
        if value is None:
            continue
        applied = value.elts if isinstance(value, ast.List | ast.Tuple) else [value]
        if any(_names_requires_db(e) for e in applied):
            return True
    return False


def _iter_tests(node: ast.AST, inherited: bool) -> list[tuple[AnyFunc, bool]]:
    """Every function in this tree paired with whether a CONTAINER already marked it.

    A `pytestmark` in a class body, or `@requires_db` on the class itself, marks every test method
    inside it. Reading only module level would report those methods as unmarked offenders.
    """
    out: list[tuple[AnyFunc, bool]] = []
    for child in getattr(node, "body", []):
        if isinstance(child, ast.ClassDef):
            marked = (
                inherited
                or _body_applies_the_mark(child.body)
                or any(_names_requires_db(d) for d in child.decorator_list)
            )
            out.extend(_iter_tests(child, marked))
        elif isinstance(child, AnyFunc):
            out.append((child, inherited))
    return out


def _close_over_fixtures(fixtures: dict[str, set[str]], seed: set[str]) -> set[str]:
    """Grow `seed` to a fixpoint: a fixture requesting a DB-backed fixture is itself DB-backed.

    Terminates: every iteration that continues adds a name to `db`, and `db` is bounded by
    `seed | set(fixtures)`. A cyclic or self-referential fixture is added once and then fails the
    membership test.
    """
    db = set(seed)
    changed = True
    while changed:
        changed = False
        for name, requests in fixtures.items():
            if name not in db and db.intersection(requests):
                db.add(name)
                changed = True
    return db


def _conftest_path() -> pathlib.Path:
    return TESTS_DIR / "conftest.py"


def _conftest_fixtures() -> dict[str, tuple[set[str], ast.expr, AnyFunc]]:
    tree = _parse(_conftest_path())
    out: dict[str, tuple[set[str], ast.expr, AnyFunc]] = {}
    for fn in _functions(tree):
        dec = _fixture_decorator(fn)
        if dec is not None:
            out[_registered_name(fn, dec)] = (_requested(fn), dec, fn)
    return out


@functools.lru_cache(maxsize=1)
def _analyse() -> dict[str, list[str]]:
    """One pass over every collectable module. Returns findings, never the parse trees.

    Caching the FINDINGS rather than the ASTs is deliberate: an `lru_cache` over `_parse` would
    retain ~68 MB of `ast.Module` objects for the rest of the pytest session, long after the six
    tests here have finished with them.
    """
    db_tests: list[str] = []
    offenders: list[str] = []
    direct: list[str] = []
    via_wrapper: list[str] = []
    via_self_reaching: list[str] = []
    unreadable: list[str] = []

    for path in _test_modules():
        tree = _parse(path)
        where = path.relative_to(TESTS_DIR).as_posix()

        # A construct this guard cannot resolve must be reported, not skipped. Skipping it exempts
        # every test behind it, which is the exact failure the whole file exists to prevent.
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "pytest":
                for alias in node.names:
                    if alias.name == "fixture" and alias.asname not in (None, "fixture"):
                        unreadable.append(
                            f"{where}:{node.lineno} imports pytest.fixture as {alias.asname!r}, "
                            "so its fixtures are invisible here"
                        )
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr == "usefixtures":
                    for arg in node.args:
                        if not isinstance(arg, ast.Constant):
                            unreadable.append(
                                f"{where}:{node.lineno} passes a non-literal to usefixtures"
                            )

        functions = _functions(tree)
        fixtures: dict[str, set[str]] = {}
        self_reaching: set[str] = set()
        for fn in functions:
            dec = _fixture_decorator(fn)
            if dec is None:
                continue
            if isinstance(dec, ast.Call):
                for kw in dec.keywords:
                    if kw.arg == "name" and not isinstance(kw.value, ast.Constant):
                        unreadable.append(
                            f"{where}:{fn.lineno} sets a non-literal fixture name= , so the name "
                            "tests request cannot be resolved"
                        )
            name = _registered_name(fn, dec)
            fixtures[name] = _requested(fn)
            if _reaches_dsn(fn):
                self_reaching.add(name)

        # Seeded from BOTH routes to the database: conftest's requestable fixtures, and any local
        # fixture that names a DSN itself. Over-approximating here is safe (it can only demand a
        # mark on a test that does not need one, which fails loudly); under-approximating is what
        # silently exempts a test, so the seed errs wide.
        #
        # The conftest-only closure is computed alongside the full one purely so the two halves can
        # be told apart below. Measuring them together lets either half die while the other keeps
        # the non-vacuity assertion alive, which is how the DSN half went missing the first time.
        conftest_seed = set(DB_BACKED_FIXTURES)
        db_conftest = _close_over_fixtures(fixtures, conftest_seed)
        db = _close_over_fixtures(fixtures, conftest_seed | self_reaching)

        module_marked = _body_applies_the_mark(tree.body)
        for fn, container_marked in _iter_tests(tree, module_marked):
            if not fn.name.startswith("test_") or _fixture_decorator(fn):
                continue
            requested = _requested(fn)
            used = db.intersection(requested)
            if not used:
                continue
            ident = f"{where}::{fn.name}"
            db_tests.append(ident)
            used_conftest = db_conftest.intersection(requested)
            if used & conftest_seed:
                direct.append(ident)
            if used_conftest - conftest_seed:
                # Reached through a wrapper fixture the CLOSURE added, e.g. `store(make_store)`.
                via_wrapper.append(ident)
            if used and not used_conftest:
                # Reachable only because a local fixture opens its own connection.
                via_self_reaching.append(ident)
            marked = container_marked or any(_names_requires_db(d) for d in fn.decorator_list)
            if not marked:
                offenders.append(f"{ident} requests {sorted(used)}")

    return {
        "db_tests": db_tests,
        "offenders": offenders,
        "direct": direct,
        "via_wrapper": via_wrapper,
        "via_self_reaching": via_self_reaching,
        "unreadable": unreadable,
    }


def test_db_backed_fixtures_matches_the_conftest_source() -> None:
    """The declaration must equal the requestable conftest fixtures that touch the database.

    A new DB fixture added to conftest and forgotten here would exempt every test using it from
    the coverage guard below, in silence. That is the same drift that made the original defect
    possible, one level up, so it gets checked rather than trusted.
    """
    fixtures = _conftest_fixtures()
    assert fixtures, "found no fixtures in conftest.py; the parse is wrong"

    touches_db = {name for name, (_, _, fn) in fixtures.items() if _reaches_dsn(fn)}
    requests = {name: reqs for name, (reqs, _, _) in fixtures.items()}
    touches_db = _close_over_fixtures(requests, touches_db)
    assert touches_db, "no conftest fixture references a DSN; the parse is wrong"

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
    unguarded = [
        name
        for name, (_, dec, fn) in _conftest_fixtures().items()
        if _reaches_dsn(fn) and _is_autouse(dec) and "_db_available" not in ast.unparse(fn)
    ]
    assert not unguarded, (
        f"{unguarded} are autouse, reach the database, and never check `_db_available()`. An "
        "autouse fixture cannot be skipped per test, so it must no-op without a DB."
    )


def test_only_one_conftest_defines_fixtures_for_this_scan() -> None:
    """`_conftest_fixtures` reads one file. A second conftest would go unread, and unread means
    its DB fixtures seed nothing and its users are exempt."""
    conftests = [p for p in TESTS_DIR.rglob("conftest.py") if "__pycache__" not in p.parts]
    assert conftests == [_conftest_path()], (
        f"expected exactly one conftest under tests/, found {[str(p) for p in conftests]}. "
        "Widen `_conftest_fixtures` to read all of them."
    )


def test_no_module_uses_a_shape_this_guard_cannot_read() -> None:
    """A construct the scan cannot resolve must fail here rather than be skipped in silence.

    This is the same treatment `test_store_query_legs_matches_the_actual_timer_labels` gives a
    timer whose label it cannot read: a guard that quietly drops what it does not understand
    reports success over a shrinking share of the code.
    """
    unreadable = _analyse()["unreadable"]
    assert not unreadable, "constructs this guard cannot read:\n  " + "\n  ".join(unreadable)


def test_every_db_backed_test_carries_the_marker() -> None:
    """The guard itself. Unmarked, these do not skip without a DB, they time out and go red."""
    result = _analyse()
    assert result["db_tests"], "found no DB-backed tests at all; the scan is broken, not the suite"
    assert not result["offenders"], (
        "these tests reach the database but do not carry `@requires_db`, so without a local "
        "Postgres they fail on a connection timeout instead of skipping:\n  "
        + "\n  ".join(result["offenders"])
    )


def test_the_scan_sees_both_routes_to_the_database() -> None:
    """Non-vacuity for the two halves of the seed, which are what the guard's reach depends on.

    Route one is a wrapper fixture requesting a conftest fixture (`store(make_store)`); route two
    is a fixture that opens its own connection and requests nothing (`tenant_table`). The first
    version of this guard implemented only route one and reported success over 202 of 363 tests.
    If either half stops contributing, the guard has quietly narrowed and this says so.
    """
    result = _analyse()
    assert result["via_wrapper"], (
        "no test was found through a wrapper fixture that the closure added, so the transitive "
        "walk is doing nothing and a `store(make_store)`-style wrapper would hide a test from "
        "this guard."
    )
    assert result["via_self_reaching"], (
        "no test was found via a fixture that opens its own connection, so the DSN half of the "
        "seed is doing nothing and a `tenant_table`-style fixture would hide a test from this "
        "guard."
    )


@pytest.mark.parametrize(
    "shape",
    [
        "pytestmark = requires_db",
        "pytestmark = [requires_db]",
        "pytestmark = (requires_db,)",
        "pytestmark: list = [requires_db]",
        "pytestmark = conftest.requires_db",
    ],
)
def test_every_pytestmark_spelling_is_recognised(shape: str) -> None:
    """Each spelling read as unmarked would fail a correctly marked module, so read them all."""
    assert _body_applies_the_mark(ast.parse(shape).body)
    assert not _body_applies_the_mark(
        ast.parse(shape.replace("requires_db", "requires_fastembed")).body
    )


def test_a_class_level_mark_covers_its_methods() -> None:
    """`pytestmark` in a class body marks the tests inside it, and so does a class decorator."""
    in_body = ast.parse(
        "class TestThing:\n    pytestmark = requires_db\n    def test_x(self, make_store): pass\n"
    )
    on_class = ast.parse(
        "@requires_db\nclass TestThing:\n    def test_x(self, make_store): pass\n"
    )
    plain = ast.parse("class TestThing:\n    def test_x(self, make_store): pass\n")

    assert [marked for _, marked in _iter_tests(in_body, False)] == [True]
    assert [marked for _, marked in _iter_tests(on_class, False)] == [True]
    assert [marked for _, marked in _iter_tests(plain, False)] == [False]


def test_a_renamed_or_usefixtures_consumer_is_still_seen() -> None:
    """Two shapes that made the earlier version blind, pinned so they cannot regress."""
    renamed = ast.parse(
        'import pytest\n'
        '@pytest.fixture(name="store")\n'
        'def _store_impl(make_store): return make_store(8)\n'
        'def test_y(store): pass\n'
    )
    functions = _functions(renamed)
    fixtures = {
        _registered_name(fn, dec): _requested(fn)
        for fn in functions
        if (dec := _fixture_decorator(fn)) is not None
    }
    db = _close_over_fixtures(fixtures, set(DB_BACKED_FIXTURES))
    test_y = next(fn for fn in functions if fn.name == "test_y")
    assert db.intersection(_requested(test_y)), "a fixture(name=...) wrapper hid its consumer"

    used = ast.parse(
        'import pytest\n@pytest.mark.usefixtures("make_store")\ndef test_uses_db(): pass\n'
    )
    consumer = next(fn for fn in _functions(used) if fn.name == "test_uses_db")
    assert "make_store" in _requested(consumer), "a usefixtures consumer was invisible"
