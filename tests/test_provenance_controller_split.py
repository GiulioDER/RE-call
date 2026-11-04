"""The split of `recall.provenance_controller` kept every moved name the same object.

`recall.fact_events` (the event vocabulary) and `recall.fact_projection` (the supersession
index) were moved verbatim out of the controller, which re-exports every moved name. Code that
imports a name from the controller and code that imports it from its new module must get ONE
object: `isinstance`, `except` and enum comparisons silently stop matching across two copies,
and a re-export replaced by a pasted copy would still import and still mostly pass.

Red proof (2026-11-04, on the test host): with the controller's `RetractionKey` re-export
replaced by a pasted copy of its definition,
`test_every_moved_name_is_one_object_through_both_import_paths` failed on its assertion naming
`RetractionKey`; with a pasted copy of the `FactSupersession` class after the re-export block, it
failed naming `FactSupersession`. Restored, it passes.
"""

from __future__ import annotations

import ast
import inspect

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


def test_the_moved_modules_never_import_the_controller() -> None:
    """The controller depends on the projections, never the reverse: an import back would be a
    cycle that only works while the controller happens to be imported first."""
    for module in (fact_events, fact_projection):
        tree = ast.parse(inspect.getsource(module))
        imported = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        } | {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        assert "recall.provenance_controller" not in imported, module.__name__
    assert "recall.fact_projection" not in {
        node.module
        for node in ast.walk(ast.parse(inspect.getsource(fact_events)))
        if isinstance(node, ast.ImportFrom)
    }
