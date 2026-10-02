"""scripts/mutmut_type_filter.py: keep mutmut's type filter for real mutants, and stop its crash.

Invariant: an error caused by a mutant (inside its generated function, naming no generated
identifier) reaches mutmut unchanged; an error mutmut's own rewriting caused, or one outside every
mutant, does not, and is reported instead. The failure it guards against is #854's report-only
mutation job dying with "Could not find mutant for type error ...store.py:85752", after a mutated
Protocol stub made every `project_current_state(self, ...)` call ill-typed.

The sources below are laid out the way mutmut 3.8.0 writes a mutated file: the original method
behind a trampoline decorator under its own name, then `..__mutmut_orig` and `..__mutmut_N` copies.

Red proof, 2026-10-02, on a Linux host: each mutation below was applied to the named production
line of scripts/mutmut_type_filter.py, this file was run, and the listed test failed in its own
assertion (not in import or setup); restoring the line turned it green.

| mutation | production line | test that went red |
|---|---|---|
| `is_generated_error` returns False | the `if is_generated_error(diagnostic)` branch | test_a_generated_protocol_error_inside_a_mutant_is_dropped |
| `all(` to `any(` over the missing members | `is_generated_error` | test_a_real_missing_member_keeps_the_error |
| the `unowned` branch removed | `if spans and not any(...)` | test_an_error_no_mutant_owns_is_dropped |
| `if spans and` removed | the same line | test_a_file_without_mutants_passes_through |
| `severity != "error"` check removed | `filter_diagnostics` | test_notes_and_non_json_lines_pass_through |
| `returncode >= 2` to `>= 3` | `main` | test_a_mypy_failure_passes_nothing_and_reports_unavailable |
| `is_generated_error` returns True | the same branch | test_a_real_mutant_error_is_kept |
| report write removed | `main` | test_main_drops_generated_errors_into_the_report |
| backslash normalisation removed | `_posix` | test_the_summary_counts_per_file_and_reason |

End to end, the same day, against mutmut 3.8.0's own `filter_mutants_with_type_checker` and mypy
2.4.0 over #854's `recall/store.py` and `recall/current_state.py` as mutmut rewrites them: with
bare mypy it raised "Could not find mutant for type error .../store.py:85752", the CI failure;
through this script it returned 1,192 mutants to discard and reported 141 dropped errors, 140 in
`store.py` and 1 in `trust.py`, all `generated`. The two `StateStore` errors left were mutants
that pass `None` as the store, which is what the filter is for.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "mutmut_type_filter", REPO_ROOT / "scripts" / "mutmut_type_filter.py"
)
assert _spec and _spec.loader
type_filter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(type_filter)

MUTATED = """\
class StateStore(Protocol):
    @_mutmut_mutated(mutants_xǁStateStoreǁiter_chunks__mutmut)
    def iter_chunks(self, batch_size: int = 1000) -> list[str]: ...

    def xǁStateStoreǁiter_chunks__mutmut_orig(self, batch_size: int = 1000) -> list[str]: ...

    def xǁStateStoreǁiter_chunks__mutmut_1(self, batch_size: int = 1001) -> list[str]: ...


class PgVectorStore:
    @_mutmut_mutated(mutants_xǁPgVectorStoreǁdependency_projection__mutmut)
    def dependency_projection(self) -> object:
        return project_current_state(self)  # ORIGINAL BODY

    def xǁPgVectorStoreǁdependency_projection__mutmut_orig(self) -> object:
        return project_current_state(self)  # ORIG COPY

    def xǁPgVectorStoreǁdependency_projection__mutmut_1(self) -> object:
        return project_current_state(None)  # MUTANT 1
"""

UNMUTATED = """\
def caller(store: PgVectorStore) -> object:
    return project_current_state(store)  # NO MUTANTS HERE
"""

GENERATED_HINT = (
    '"PgVectorStore" is missing following "StateStore" protocol members:\n'
    "    xǁStateStoreǁiter_chunks__mutmut_1, xǁStateStoreǁiter_chunks__mutmut_orig"
)
PROTOCOL_MESSAGE = (
    'Argument 1 to "project_current_state" has incompatible type "PgVectorStore"; '
    'expected "StateStore"'
)


def _line(source: str, marker: str) -> int:
    return next(n for n, text in enumerate(source.splitlines(), 1) if marker in text)


def _error(path: str, line: int, message: str, hint: str | None = None, severity: str = "error") -> str:
    return json.dumps(
        {"file": path, "line": line, "column": 0, "message": message, "hint": hint,
         "code": "arg-type", "severity": severity}
    )


def _run(lines: list[str]) -> tuple[list[str], list[dict]]:
    sources = {"recall/store.py": MUTATED, "recall/trust.py": UNMUTATED}
    return type_filter.filter_diagnostics(lines, sources.__getitem__)


def test_generated_names_follow_mutmut() -> None:
    assert type_filter.is_generated_name("x_project__mutmut_3")
    assert type_filter.is_generated_name("xǁStateStoreǁiter_chunks__mutmut_orig")
    assert not type_filter.is_generated_name("x_value")
    assert not type_filter.is_generated_name("iter_chunks")


def test_a_generated_protocol_error_inside_a_mutant_is_dropped() -> None:
    line = _error("recall/store.py", _line(MUTATED, "ORIG COPY"), PROTOCOL_MESSAGE, GENERATED_HINT)
    kept, dropped = _run([line])
    assert kept == []
    assert [entry["reason"] for entry in dropped] == ["generated"]


def test_the_854_crash_line_is_dropped() -> None:
    """The original body: no mutant owns it, and the error is mutmut's own."""
    line = _error("recall/store.py", _line(MUTATED, "ORIGINAL BODY"), PROTOCOL_MESSAGE, GENERATED_HINT)
    kept, dropped = _run([line])
    assert kept == []
    assert len(dropped) == 1


def test_a_real_mutant_error_is_kept() -> None:
    line = _error(
        "recall/store.py",
        _line(MUTATED, "MUTANT 1"),
        'Argument 1 to "project_current_state" has incompatible type "None"; expected "StateStore"',
    )
    kept, dropped = _run([line])
    assert kept == [line]
    assert dropped == []


def test_a_real_missing_member_keeps_the_error() -> None:
    hint = (
        '"object" is missing following "StateStore" protocol members:\n'
        "    tenant, xǁStateStoreǁiter_chunks__mutmut_1"
    )
    line = _error("recall/store.py", _line(MUTATED, "MUTANT 1"), PROTOCOL_MESSAGE, hint)
    kept, dropped = _run([line])
    assert kept == [line]
    assert dropped == []


def test_an_error_no_mutant_owns_is_dropped() -> None:
    line = _error(
        "recall/store.py", _line(MUTATED, "ORIGINAL BODY"), "Incompatible return value type"
    )
    kept, dropped = _run([line])
    assert kept == []
    assert [entry["reason"] for entry in dropped] == ["unowned"]


def test_a_file_without_mutants_passes_through() -> None:
    """mutmut already ignores errors in a file with no mutants, so they are not ours to drop."""
    line = _error("recall/trust.py", _line(UNMUTATED, "NO MUTANTS"), "Incompatible return value type")
    kept, dropped = _run([line])
    assert kept == [line]
    assert dropped == []


def test_notes_and_non_json_lines_pass_through() -> None:
    note = _error("recall/store.py", _line(MUTATED, "ORIGINAL BODY"), "See docs", severity="note")
    garbage = "mypy: crashed half way"
    kept, dropped = _run([note, garbage])
    assert kept == [note, garbage]
    assert dropped == []


def _fake_mypy(stdout_lines: list[str], returncode: int) -> list[str]:
    script = (
        "import sys\n"
        f"sys.stdout.write({json.dumps(chr(10).join(stdout_lines))} + chr(10))\n"
        "sys.stderr.write('mypy: error: something broke' + chr(10))\n"
        f"sys.exit({returncode})\n"
    )
    return [sys.executable, "-c", script]


def _entries(report: Path) -> list[dict]:
    """The report's entries; an absent report is none, so a missing write fails an assertion."""
    if not report.exists():
        return []
    return [json.loads(text) for text in report.read_text(encoding="utf-8").splitlines()]


def test_a_mypy_failure_passes_nothing_and_reports_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    report = tmp_path / "report.jsonl"
    monkeypatch.setenv("MUTMUT_TYPE_FILTER_REPORT", str(report))
    assert type_filter.main(_fake_mypy(["half a report"], 2)) == 2
    assert capsys.readouterr().out == ""
    entries = _entries(report)
    assert [entry["reason"] for entry in entries] == ["unavailable"]
    assert "something broke" in entries[0]["message"]
    summary = type_filter.summarise(report.read_text(encoding="utf-8").splitlines())
    assert summary.startswith("**Type filter unavailable**: mypy exited 2")


def test_main_drops_generated_errors_into_the_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    report = tmp_path / "report.jsonl"
    monkeypatch.setenv("MUTMUT_TYPE_FILTER_REPORT", str(report))
    generated = _error("recall/store.py", 12, PROTOCOL_MESSAGE, GENERATED_HINT)
    assert type_filter.main(_fake_mypy([generated], 1)) == 1
    assert capsys.readouterr().out == ""
    entries = _entries(report)
    assert [(entry["reason"], entry["line"]) for entry in entries] == [("generated", 12)]


def test_the_summary_counts_per_file_and_reason() -> None:
    entries = [
        json.dumps({"reason": "generated", "file": "recall\\store.py", "line": 12, "message": "m"}),
        json.dumps({"reason": "generated", "file": "recall\\store.py", "line": 40, "message": "m"}),
        json.dumps({"reason": "unowned", "file": "recall/x.py", "line": 7, "message": "m"}),
    ]
    summary = type_filter.summarise(entries)
    assert "ignored 3 mypy error(s)" in summary
    assert "`recall/store.py` 2 generated" in summary
    assert "`recall/x.py` 1 unowned" in summary
    assert type_filter.summarise([]) == ""
