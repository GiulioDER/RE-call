"""The split of `recall.provenance_controller` kept every moved name the same object.

`recall.fact_events` (the event vocabulary) and `recall.fact_projection` (retraction, withdrawal,
supersession and the request-id forms) were moved verbatim out of the controller, which
re-exports every moved name. Code that imports a name from the controller and code that imports
it from its new module must get ONE object: `isinstance`, `except` and enum comparisons silently
stop matching across two copies, and a re-export replaced by a pasted copy would still import and
still mostly pass.

Red proof, observed on the test host on 2026-11-04 against this file and the split at 2ef233139
(each mutation applied alone, then restored by bytes; both tests then passed again):

- `test_every_moved_name_is_one_object_through_both_import_paths` failed with
  `AssertionError: assert ['RetractionKey'] == []` when the controller's `RetractionKey`
  re-export was replaced by a pasted copy of its definition, and with
  `assert ['FactSupersession'] == []` when a copy of the `FactSupersession` class was pasted
  after the re-export block.
- `test_the_moved_modules_never_import_the_controller` failed with
  `AssertionError: recall.fact_projection` when `recall/fact_projection.py` gained
  `import recall.provenance_controller`, `from recall import provenance_controller` or
  `from . import provenance_controller`, and with `AssertionError: recall.fact_events` when
  `recall/fact_events.py` gained `from recall import fact_projection`. The first version of this
  test (2ef233139) passed under `from recall import provenance_controller`, which is why
  `_imported_modules` reads every import spelling.
"""

from __future__ import annotations

import ast
import inspect
from types import ModuleType

import recall.fact_events as fact_events
import recall.fact_projection as fact_projection
import recall.provenance_controller as provenance_controller

MOVED_TO_EVENTS = ("DecisionCode", "_canonical", "canonical_json", "fact_identity", "_as_utc", "FactEvent")
MOVED_TO_PROJECTION = (
    "asserts_fact",
    "applications",
    "RetractionKey",
    "retraction_key",
    "retracted_keys",
    "withdrawn_event_ids",
    "FactWindow",
    "_EARLIEST",
    "_LATEST",
    "SUPERSESSION_REQUEST_SUFFIX",
    "RETRACTION_REQUEST_PREFIX",
    "WITHDRAWAL_REQUEST_PREFIX",
    "STATED_REQUEST_PREFIX",
    "reserved_request_id",
    "fact_window",
    "_holds",
    "_without",
    "Interval",
    "_span",
    "_merge",
    "FactSupersession",
    "_FactSuccessorIndex",
    "fact_supersession",
    "ordered_after",
    "fact_conflict_key",
)


def test_every_moved_name_is_one_object_through_both_import_paths() -> None:
    pairs = [(fact_events, name) for name in MOVED_TO_EVENTS] + [
        (fact_projection, name) for name in MOVED_TO_PROJECTION
    ]
    differ = [
        name
        for module, name in pairs
        if getattr(provenance_controller, name) is not getattr(module, name)
    ]
    assert differ == []


def _imported_modules(module: ModuleType) -> set[str]:
    """Every module `module`'s source can import, in each spelling an import statement allows:
    `import a.b`, `from a.b import c`, `from a import b` (which may import the submodule `a.b`)
    and the relative forms, resolved against the module's package."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(inspect.getsource(module))):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = (module.__package__ or "").split(".")
                base = ".".join(parts[: len(parts) - node.level + 1])
                source = f"{base}.{node.module}" if node.module else base
            else:
                source = node.module or ""
            found.add(source)
            found.update(f"{source}.{alias.name}" for alias in node.names)
    return found


def test_the_moved_modules_never_import_the_controller() -> None:
    """The controller depends on the projections, and the projections on the event vocabulary,
    never the reverse: an import back would be a cycle that only works while the modules happen
    to be imported in one order."""
    for module in (fact_events, fact_projection):
        assert "recall.provenance_controller" not in _imported_modules(module), module.__name__
    assert "recall.fact_projection" not in _imported_modules(fact_events), fact_events.__name__
