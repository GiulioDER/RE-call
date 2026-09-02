"""Shared freeze helper for immutable graph and proposal values.

The frozen output of :func:`freeze_value` feeds ``canonical_sha256`` content hashes:
graph, node, edge, entity, mention, relation and proposal identities are all computed
over values this function has normalized. The importing modules (``recall.semantic_graph``,
``recall.reasoning_graph``, ``recall.reasoning_proposals.types``) alias it under their
historical private names, and it must never diverge between those call sites: any change
here changes every identity at once, which is an explicit migration, never a refactor.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any


def freeze_value(value: Any) -> Any:
    """Recursively convert ``value`` into an immutable, deterministically ordered form."""
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                key: freeze_value(item)
                for key, item in sorted(value.items(), key=lambda entry: str(entry[0]))
            }
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(freeze_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((freeze_value(item) for item in value), key=repr))
    return value
