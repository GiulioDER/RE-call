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
    learned_sparse: bool = False
    task_conditioned: bool = False
    reranker_model: str | None = None
    entailment: bool = False


ATTRIBUTION_VARIANTS = (
    HostedVariant("A0_raw", True, False, False, False, False, None),
    HostedVariant("A1_compiler", True, True, False, False, False, None),
    HostedVariant("A2_facets", True, True, True, False, False, None),
    HostedVariant("A3_rerank", True, True, True, True, False, None),
    HostedVariant("A4_pack_5000", True, True, True, True, True, 5_000),
    HostedVariant("A4_pack_7000", True, True, True, True, True, 7_000),
    HostedVariant("A4_pack_9000", True, True, True, True, True, 9_000),
)
EXPERIENCE_VARIANTS = (
    HostedVariant("E0_raw", True, False, False, False, False, None),
    HostedVariant("E1_compiled", False, True, False, False, False, None),
    HostedVariant("E2_compiled_raw", True, True, False, False, False, None),
)
CODING_MATRIX_VARIANTS = (
    HostedVariant("C0_raw_lexical", True, False, False, False, False, None),
    HostedVariant("C1_splade", True, False, False, False, False, None, learned_sparse=True),
    HostedVariant("C2_procedure", True, True, False, False, False, None, learned_sparse=True),
    HostedVariant("C3_rerank", True, True, False, True, False, None, learned_sparse=True),
    HostedVariant(
        "C4_task_pack",
        True,
        True,
        True,
        True,
        True,
        7_000,
        learned_sparse=True,
        task_conditioned=True,
    ),
)
CLEAN_RERANK_VARIANTS = (
    HostedVariant("B0_raw", True, False, False, False, False, None),
    HostedVariant(
        "B1_raw_rerank",
        True,
        False,
        False,
        True,
        False,
        None,
        reranker_model="rerank-2.5",
    ),
    HostedVariant(
        "B2_raw_rerank3",
        True,
        False,
        False,
        True,
        False,
        None,
        reranker_model="rerank-3",
    ),
    HostedVariant(
        "B3_raw_entailment",
        True,
        False,
        False,
        False,
        False,
        None,
        entailment=True,
    ),
)
VARIANTS = (
    ATTRIBUTION_VARIANTS + EXPERIENCE_VARIANTS + CODING_MATRIX_VARIANTS + CLEAN_RERANK_VARIANTS
)
DEFAULT_VARIANT = "A4_pack_7000"


def variant(name: str) -> HostedVariant:
    for item in VARIANTS:
        if item.name == name:
            return item
    raise ValueError(f"unknown AML Hosted variant {name!r}")
