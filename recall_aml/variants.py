"""Registered RE-call Hosted behaviors used by local attribution runs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HostedVariant:
    name: str
    raw: bool
    compiler: bool
    facets: bool
    reranker: bool
    pack: bool
    context_chars: int | None


VARIANTS = (
    HostedVariant("A0_raw", True, False, False, False, False, None),
    HostedVariant("A1_compiler", True, True, False, False, False, None),
    HostedVariant("A2_facets", True, True, True, False, False, None),
    HostedVariant("A3_rerank", True, True, True, True, False, None),
    HostedVariant("A4_pack_5000", True, True, True, True, True, 5_000),
    HostedVariant("A4_pack_7000", True, True, True, True, True, 7_000),
    HostedVariant("A4_pack_9000", True, True, True, True, True, 9_000),
)
DEFAULT_VARIANT = "A4_pack_7000"


def variant(name: str) -> HostedVariant:
    for item in VARIANTS:
        if item.name == name:
            return item
    raise ValueError(f"unknown AML Hosted variant {name!r}")
