"""Run mypy for mutmut's type filter, dropping the errors mutmut cannot attribute to a mutant.

    python scripts/mutmut_type_filter.py mypy --output json <files>

`scripts/mutation_changed.sh` sets this as mutmut's `type_check_command`. mutmut runs it inside
`mutants/`, reads mypy's JSON lines from stdout, and discards every mutant whose generated function
holds an error. The literal `mypy` in the arguments is what makes mutmut parse the output as
mypy's: it tests the command list for that element, not the executable.

Why it exists: mutmut 3.8.0 RAISES, and the whole run is lost, when an error falls outside every
generated mutant function ("Could not find mutant for type error"). Its rewriting produces such
errors on its own, without any mutant at fault. Seen on #854 (2026-10-02): mutmut mutated the
default of a stub in `recall.current_state.StateStore`, a `Protocol`, which added the generated
methods `xǁStateStoreǁiter_chunks__mutmut_orig` and `..._1` to the Protocol's members. From then on
`PgVectorStore` no longer satisfies `StateStore`, and every call `project_current_state(self, ...)`
in `recall/store.py` is a type error: 140 of them. 139 sit inside mutant functions, which mutmut
would discard as caught by the type checker although no mutant caused them; one sits in the
original `dependency_projection` body, which no mutant owns, and that one crashed the run.

Two kinds of error are dropped, and every other line passes through unchanged:

* **generated**: the error names a mutmut-generated identifier. A mutant's body is a copy of the
  original with one change, so it cannot name one; only the rewriting can. For a Protocol error
  the hint lists the missing members, and the error is dropped only when EVERY listed member is
  generated: a mutant that passes an object missing a real member keeps its error. mypy lists
  them only while at most two are missing, and each mutated Protocol method adds its `_orig`
  copy plus one method per mutant, so from two mutants on the error comes with no hint at all.
  Such an error is dropped when the type it expects is a Protocol that holds generated methods
  in one of the checked files (`broken_protocols`, matched by bare class name) and the type it got
  does not contain `None`: mutmut's mutant that passes `None` keeps its error. Without the member
  list a mutant's own error cannot always be told apart: one that removes an argument (`f(store,
  n)` to `f(n)`) or names another module's class of the same name loses its error too. That costs
  test time and loses no mutant, since mutmut then tests the mutant instead of discarding it.
* **unowned**: the error lies outside every generated mutant function of a file that has some.
  mutmut would raise on it. In a file with no mutant functions mutmut already ignores errors, so
  those pass through.

Each dropped error is appended as one JSON line to `$MUTMUT_TYPE_FILTER_REPORT` when that is set,
so the job summary can say what the filter could not use.

When mypy itself fails (exit 2 and up: a crash, a bad flag, a missing file; or killed by a
signal, a negative exit), or this filter cannot run (mypy not found, a source it cannot read or
parse), nothing is passed on and the report records the filter as `unavailable`. mutmut then
filters nothing and tests every mutant, which costs time and loses no mutant; the report is what
keeps that from passing as "no mutant failed the type check", which is how mutmut reads an empty
stdout.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

GENERATED_SEPARATOR = "ǁ"  # mutmut's CLASS_NAME_SEPARATOR, in every generated method name
GENERATED_MARKER = "__mutmut"
MISSING_MEMBERS = "protocol members:"
# The incompatibility messages mypy writes for an argument, an assignment and a return value.
_INCOMPATIBLE = re.compile(
    r'incompatible type "(?P<got>[^"]+)"; expected "(?P<want>[^"]+)"'
    r'|expression has type "(?P<got2>[^"]+)", variable has type "(?P<want2>[^"]+)"'
    r'|got "(?P<got3>[^"]+)", expected "(?P<want3>[^"]+)"'
)


def is_generated_name(name: str) -> bool:
    """mutmut's own test (`is_mutated_method_name`): `x_f__mutmut_3`, `xǁCǁf__mutmut_orig`."""
    return name.startswith(("x_", "x" + GENERATED_SEPARATOR)) and GENERATED_MARKER in name


def _names_generated(text: str) -> bool:
    return GENERATED_SEPARATOR in text or any(
        is_generated_name(token) for token in text.replace('"', " ").replace(",", " ").split()
    )


def _is_protocol_base(base: ast.expr) -> bool:
    if isinstance(base, ast.Subscript):
        base = base.value
    return (isinstance(base, ast.Name) and base.id == "Protocol") or (
        isinstance(base, ast.Attribute) and base.attr == "Protocol"
    )


def broken_protocols(sources: list[str]) -> frozenset[str]:
    """Names of the Protocol classes that hold mutmut-generated methods in these sources."""
    names = set()
    for source in sources:
        for node in ast.walk(ast.parse(source)):
            if (
                isinstance(node, ast.ClassDef)
                and any(_is_protocol_base(base) for base in node.bases)
                and any(
                    isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and is_generated_name(item.name)
                    for item in node.body
                )
            ):
                names.add(node.name)
    return frozenset(names)


def _expects_broken_protocol(message: str, protocols: frozenset[str]) -> bool:
    match = _INCOMPATIBLE.search(message)
    if not match:
        return False
    got = match["got"] or match["got2"] or match["got3"]
    want = match["want"] or match["want2"] or match["want3"]
    return want.rsplit(".", 1)[-1] in protocols and "None" not in got


def is_generated_error(diagnostic: dict, protocols: frozenset[str] = frozenset()) -> bool:
    """True when the error exists only because of mutmut's renaming, not because of a mutant."""
    message = diagnostic.get("message") or ""
    hint = diagnostic.get("hint") or ""
    if MISSING_MEMBERS in hint:
        members = [
            member.strip()
            for line in hint.split(MISSING_MEMBERS, 1)[1].splitlines()
            for member in line.split(",")
            if member.strip()
        ]
        return bool(members) and all(is_generated_name(member) for member in members)
    if not hint and _expects_broken_protocol(message, protocols):
        return True  # more than two members missing: mypy lists none
    return _names_generated(message)


def mutant_spans(source: str) -> list[tuple[int, int]]:
    """Line spans of mutmut's generated functions, which it writes undecorated."""
    return [
        (node.lineno, node.end_lineno or node.lineno)
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and is_generated_name(node.name)
    ]


def filter_diagnostics(
    lines: list[str], read_source, protocols: frozenset[str] = frozenset()
) -> tuple[list[str], list[dict]]:
    """Split mypy's JSON lines into those to pass on and the errors to drop, with a reason each."""
    kept: list[str] = []
    dropped: list[dict] = []
    spans_by_file: dict[str, list[tuple[int, int]]] = {}
    for line in lines:
        try:
            diagnostic = json.loads(line)
        except json.JSONDecodeError:
            kept.append(line)  # not mypy's JSON: let mutmut report what it got
            continue
        if not isinstance(diagnostic, dict) or diagnostic.get("severity") != "error":
            kept.append(line)
            continue
        reason = None
        if is_generated_error(diagnostic, protocols):
            reason = "generated"
        else:
            path = diagnostic["file"]
            if path not in spans_by_file:
                spans_by_file[path] = mutant_spans(read_source(path))
            spans = spans_by_file[path]
            if spans and not any(start <= diagnostic["line"] <= end for start, end in spans):
                reason = "unowned"
        if reason is None:
            kept.append(line)
        else:
            dropped.append({"reason": reason, **diagnostic})
    return kept, dropped


def _posix(path: str) -> str:
    """mypy writes `recall\\store.py` on Windows, and `Path` on Linux keeps the backslash."""
    return path.replace("\\", "/")


def summarise(report_lines: list[str]) -> str:
    """The job summary's account of the report: what the type filter could not use, per file."""
    entries = [json.loads(line) for line in report_lines if line.strip()]
    if not entries:
        return ""
    unavailable = [entry for entry in entries if entry["reason"] == "unavailable"]
    if unavailable:
        first = unavailable[0]
        code = first["returncode"]
        ran = "mypy did not run" if code is None else f"mypy exited {code}"
        return (
            f"**Type filter unavailable**: {ran} "
            f"(`{first['message']}`). No mutant was discarded for a type error; all were tested."
        )
    counts: dict[tuple[str, str], int] = {}
    for entry in entries:
        key = (_posix(entry["file"]), entry["reason"])
        counts[key] = counts.get(key, 0) + 1
    by_file = ", ".join(f"`{path}` {count} {reason}" for (path, reason), count in sorted(counts.items()))
    first = entries[0]
    return (
        f"Type filter: ignored {len(entries)} mypy error(s) that mutmut cannot pin on one mutant "
        f"(generated: caused by its own renaming; unowned: outside every mutant). {by_file}. "
        f"First: `{_posix(first['file'])}:{first['line']}` {first['message']}"
    )


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[0] == "--summary":
        print(summarise(Path(argv[1]).read_text(encoding="utf-8").splitlines()))
        return 0
    if not argv:
        print("usage: mutmut_type_filter.py mypy --output json <files>", file=sys.stderr)
        print("       mutmut_type_filter.py --summary <report>", file=sys.stderr)
        return 2
    returncode: int | None = None
    try:
        completed = subprocess.run(argv, capture_output=True, encoding="utf-8", check=False)  # noqa: S603
        returncode = completed.returncode
        sys.stderr.write(completed.stderr)
        if returncode >= 2 or returncode < 0:
            tail = (completed.stderr or completed.stdout).strip().splitlines()[-3:]
            dropped = [{"reason": "unavailable", "returncode": returncode, "message": " / ".join(tail)}]
        else:
            read = lambda path: Path(path).read_text(encoding="utf-8")  # noqa: E731
            protocols = broken_protocols([read(arg) for arg in argv[1:] if arg.endswith(".py")])
            kept, dropped = filter_diagnostics(completed.stdout.splitlines(), read, protocols)
            for line in kept:
                print(line)
    except Exception as error:  # noqa: BLE001 - a filter that cannot run must not read as a clean check
        dropped = [
            {"reason": "unavailable", "returncode": returncode,
             "message": f"filter error: {type(error).__name__}: {error}"}
        ]
        returncode = 2
    report = os.environ.get("MUTMUT_TYPE_FILTER_REPORT")
    if report and dropped:
        with open(report, "a", encoding="utf-8") as handle:
            for diagnostic in dropped:
                handle.write(json.dumps(diagnostic, ensure_ascii=False) + "\n")
    return returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
