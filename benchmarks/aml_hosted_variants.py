"""Frozen attribution arms for the AML Hosted preregistration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HostedVariant:
    name: str
    raw: bool
    compiler: bool
    facets: bool
    reranker: bool
    context_chars: int | None


VARIANTS = (
    HostedVariant("A0", True, False, False, False, None),
    HostedVariant("A1", True, True, False, False, None),
    HostedVariant("A2", True, True, True, False, None),
    HostedVariant("A3", True, True, True, True, None),
    HostedVariant("A4-5000", True, True, True, True, 5_000),
    HostedVariant("A4-7000", True, True, True, True, 7_000),
    HostedVariant("A4-9000", True, True, True, True, 9_000),
)


def variant(name: str) -> HostedVariant:
    for item in VARIANTS:
        if item.name == name:
            return item
    raise KeyError(f"unknown AML Hosted variant {name!r}")
