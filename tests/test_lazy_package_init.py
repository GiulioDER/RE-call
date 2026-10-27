"""`recall/__init__.py` resolves its re-exports on first access and loads nothing heavy itself.

Invariants:

* importing a leaf module such as `recall.errors` does not load the re-exported layers (reasoning,
  graphs, fact ledgers) or the database driver;
* every name the package exports still resolves, to the very object its module defines, through
  `from recall import X`, `recall.X`, and `from recall import *`;
* `import recall; recall.<submodule>` keeps working, as it did when the eager imports set those
  attributes as a side effect;
* the runtime table `_LAZY_EXPORTS` and the `TYPE_CHECKING` imports name the same module for every
  name, so type checkers and the runtime agree.

Failure modes caught: the eager package init made `import recall.errors` cost about 1.2 s and
load psycopg (measured 2026-10-01); a lazy table that loses an entry, points at the wrong module,
or drops the submodule fallback breaks a public import without any error at package load.

Red proof, 2026-10-01, on a Linux test host, each case run alone against a copy of this tree:

* `test_importing_a_leaf_module_loads_no_reexported_layer` with the eager `recall/__init__.py` of
  the parent commit `162b9f84`: failed at its final assertion, all four of `psycopg`,
  `recall.fact_ledger`, `recall.reasoning` and `recall.semantic_graph` loaded.
* `test_every_exported_name_resolves_to_its_modules_object`, mutation `"reason"` deleted from
  `_LAZY_EXPORTS`: failed at the asserted `getattr`, `AttributeError: module 'recall' has no
  attribute 'reason'`.
* `test_the_lazy_table_matches_the_type_checking_imports`, mutation `"federate"` pointed at
  `recall.retriever`: failed at its assertion, `recall.retriever != recall.federation`.
* `test_submodules_are_reachable_as_package_attributes`, mutation: the submodule fallback in
  `__getattr__` removed: failed at the return-code assertion, the child raised
  `AttributeError: module 'recall' has no attribute 'evidence'`.
* `test_star_import_binds_every_exported_name`, mutation `"reason"` deleted: failed at the
  return-code assertion, the child's `from recall import *` raised `AttributeError`.

`test_an_unknown_attribute_is_still_an_attribute_error` is a nonbehavioural control: it pins that
the fallback does not turn a missing name into some other error, and it passes on both versions.
"""

from __future__ import annotations

import ast
import importlib
import subprocess
import sys
from pathlib import Path

import recall

_INIT = Path(recall.__file__)


def _run(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


def test_importing_a_leaf_module_loads_no_reexported_layer() -> None:
    probe = (
        "import sys, recall.errors; "
        "print(','.join(sorted(m for m in ('recall.reasoning', 'recall.semantic_graph', "
        "'recall.fact_ledger', 'psycopg') if m in sys.modules)))"
    )
    result = _run(probe)
    assert result.returncode == 0, result.stderr
    loaded = [name for name in result.stdout.strip().split(",") if name]
    assert loaded == []


def test_every_exported_name_resolves_to_its_modules_object() -> None:
    table = recall._LAZY_EXPORTS
    for name in recall.__all__:
        value = getattr(recall, name)
        assert value is getattr(importlib.import_module(table[name]), name), name


def test_the_lazy_table_matches_the_type_checking_imports() -> None:
    tree = ast.parse(_INIT.read_text(encoding="utf-8"))
    declared: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.If) and ast.unparse(node.test) == "TYPE_CHECKING":
            for statement in node.body:
                assert isinstance(statement, ast.ImportFrom)
                for alias in statement.names:
                    declared[alias.name] = str(statement.module)
    assert set(declared) == set(recall._LAZY_EXPORTS)
    for name, module in declared.items():
        assert recall._LAZY_EXPORTS[name] == module, f"{name}: {recall._LAZY_EXPORTS[name]} != {module}"
    assert set(recall.__all__) <= set(declared)


def test_star_import_binds_every_exported_name() -> None:
    result = _run(
        "from recall import *\n"
        "import recall\n"
        "print(sorted(name for name in recall.__all__ if name not in globals()))\n"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"


def test_submodules_are_reachable_as_package_attributes() -> None:
    result = _run("import recall; print(recall.evidence.EvidencePolicy.__name__)")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "EvidencePolicy"


def test_an_unknown_attribute_is_still_an_attribute_error() -> None:
    result = _run(
        "import recall\n"
        "try:\n    recall.no_such_name\nexcept AttributeError:\n    print('AttributeError')\n"
    )
    assert result.stdout.strip() == "AttributeError", result.stderr
