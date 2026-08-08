"""The cleanup predicate in `benchmarks/check_generation_parity.py`, over all 16 flag combinations.

That predicate is the only thing standing between a bug and DROPPED POSTGRESQL TABLES, each of
which costs about an hour of embedding to rebuild. It has already been wrong twice in one day:
once dropping generations another process had built, and once, while fixing that, silently turning
`--drop-generations` into a no-op in the only stage the driver runs.

⚠️ The predicate is written out below, which on its own would compare the source with a copy of
itself, the defect class this repository keeps recording. It is BOUND to the shipped code instead:
`_skip_conditions` extracts the `if <cond>: continue` tests straight out of the module with `ast`,
and `test_predicate_is_pinned_to_the_shipped_source` asserts they equal `EXPECTED_SKIPS` verbatim.
Edit the cleanup loop and this file fails loudly rather than drifting into agreement with nothing.

No database, no fixtures: this reads a source file and evaluates booleans.
"""
from __future__ import annotations

import ast
import itertools
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[1] / "benchmarks" / "check_generation_parity.py"

#: Verbatim, in order, as `ast.unparse` renders them. This is the binding to the shipped code.
EXPECTED_SKIPS = [
    "args.keep_tables",
    "not gen.get('scratch') and (not args.drop_generations)",
]


def _skip_conditions() -> list[str]:
    """The `if <cond>: continue` tests inside the cleanup loop, read from the module."""
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.For):
            continue
        if not (isinstance(node.target, ast.Name) and node.target.id == "gen"):
            continue
        if not (isinstance(node.iter, ast.Name) and node.iter.id == "built"):
            continue
        conds = [
            ast.unparse(stmt.test)
            for stmt in node.body
            if isinstance(stmt, ast.If)
            and len(stmt.body) == 1
            and isinstance(stmt.body[0], ast.Continue)
        ]
        if conds:
            return conds
    pytest.fail(
        f"no `for gen in built:` loop with `continue` guards found in {MODULE.name}. "
        f"The extractor is stale, which is a failure of THIS test, not evidence about the code."
    )


def drops(*, keep_tables: bool, owned: bool, scratch: bool, drop_generations: bool) -> bool:
    """Mirrors the two pinned skip conditions. `owned` is deliberately unused; see below."""
    if keep_tables:
        return False
    if not scratch and not drop_generations:
        return False
    return True


def test_predicate_is_pinned_to_the_shipped_source() -> None:
    """If this fails, the cleanup loop changed and every assertion below describes old code."""
    assert _skip_conditions() == EXPECTED_SKIPS, (
        "the cleanup loop's guards changed. Re-derive the truth table below against the NEW "
        "conditions, then update EXPECTED_SKIPS. Do NOT update EXPECTED_SKIPS alone: that would "
        "leave this test agreeing with a copy of itself, which is exactly what it exists to stop."
    )


@pytest.mark.parametrize(
    ("keep_tables", "owned", "scratch", "drop_generations"),
    list(itertools.product([False, True], repeat=4)),
)
def test_truth_table(keep_tables: bool, owned: bool, scratch: bool, drop_generations: bool) -> None:
    """All 16 cells, against the intent stated in the module's cleanup comment."""
    want = (not keep_tables) and (scratch or drop_generations)
    got = drops(
        keep_tables=keep_tables,
        owned=owned,
        scratch=scratch,
        drop_generations=drop_generations,
    )
    assert got == want


def test_keep_tables_drops_nothing() -> None:
    """`--keep-tables` is the operator's stop button and must win over every other flag."""
    for owned, scratch, drop_generations in itertools.product([False, True], repeat=3):
        assert not drops(
            keep_tables=True, owned=owned, scratch=scratch, drop_generations=drop_generations
        ), f"--keep-tables dropped a table at owned={owned} scratch={scratch}"


def test_a_generation_is_never_dropped_implicitly() -> None:
    """The finding that started this: a compare stage dropped four arms it had been HANDED.

    A generation costs about an hour of embedding, and keeping it is what makes a compare re-run
    free. It may only go when the operator asks for it by name.
    """
    for owned in (False, True):
        assert not drops(
            keep_tables=False, owned=owned, scratch=False, drop_generations=False
        ), f"a generation was dropped without --drop-generations at owned={owned}"


def test_drop_generations_is_not_a_no_op() -> None:
    """The over-correction: gating on `owned` first made this flag inert where it is USED.

    A compare stage marks every generation `owned=False` by construction, so a predicate that
    skipped not-owned rows before considering the flag killed it in the only stage the driver runs,
    while its help text still advertised the old behaviour.
    """
    assert drops(keep_tables=False, owned=False, scratch=False, drop_generations=True), (
        "--drop-generations is a no-op on a generation this process did not create, which is the "
        "only kind a compare stage ever holds"
    )


def test_a_control_table_is_always_disposable() -> None:
    """Scratch tables are created per run and must not accumulate on a shared host."""
    for owned, drop_generations in itertools.product([False, True], repeat=2):
        assert drops(
            keep_tables=False, owned=owned, scratch=True, drop_generations=drop_generations
        ), f"a control table survived at owned={owned} drop_generations={drop_generations}"
