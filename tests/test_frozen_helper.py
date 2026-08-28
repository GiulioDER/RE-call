"""CODE-002: one freeze implementation feeds every content hash.

The freeze helper normalizes values before canonical_sha256 turns them into graph,
node, edge, entity, mention, relation and proposal identities. Three modules used to
carry line for line copies of it; they now alias one shared implementation, and this
file is the tripwire that keeps them from diverging again.
"""

from __future__ import annotations

from types import MappingProxyType

from recall._frozen import freeze_value
from recall.reasoning_graph import _freeze_projection_value
from recall.reasoning_proposals.types import _freeze_value
from recall.semantic_graph import _freeze


def test_all_three_import_paths_are_the_same_function_object() -> None:
    assert _freeze is freeze_value
    assert _freeze_projection_value is freeze_value
    assert _freeze_value is freeze_value


def test_a_nested_structure_freezes_identically_through_every_path() -> None:
    value = {
        "b": [1, {"z": 2, "a": 3}],
        "a": {5, 4},
        7: ("x", frozenset({"y"})),
    }

    frozen = [
        helper(value)
        for helper in (freeze_value, _freeze, _freeze_projection_value, _freeze_value)
    ]

    assert all(repr(item) == repr(frozen[0]) for item in frozen)
    assert isinstance(frozen[0], MappingProxyType)
    assert list(frozen[0]) == [7, "a", "b"], "keys are sorted by their str form"
    assert frozen[0]["a"] == (4, 5), "sets become tuples sorted by repr"
    assert frozen[0]["b"] == (1, frozen[0]["b"][1]), "sequences become tuples"
    assert list(frozen[0]["b"][1]) == ["a", "z"]
