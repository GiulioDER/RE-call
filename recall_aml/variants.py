"""Registered RE-call Hosted behaviors used by local attribution runs."""

from __future__ import annotations

from dataclasses import dataclass


REPOSITORY_KINDS = frozenset(
    {"architectural decision", "constraint", "repository fact"}
)
EXPERIENCE_KINDS = frozenset(
    {
        "symptom",
        "root cause",
        "failed attempt",
        "successful repair",
        "procedure",
        "validation",
    }
)


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
    code_aware: bool = False
    anchor_compiler: bool = False
    anchor_compiler_version: int = 2
    raw_rescue_tail: bool = False
    compiled_kinds: frozenset[str] | None = None
    drop_compiler_fallback: bool = False
    multimodal_preserve: bool = False
    multimodal_native: bool = False


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
    HostedVariant("B1_raw_rerank", True, False, False, True, False, None),
)
CODE_AWARE_VARIANTS = (
    HostedVariant("M0_raw", True, False, False, False, False, None),
    HostedVariant(
        "M1_code_neighbors",
        True,
        False,
        False,
        False,
        False,
        None,
        code_aware=True,
    ),
)
ANCHOR_COMPILER_VARIANTS = (
    HostedVariant("V2_raw", True, False, False, False, False, None),
    HostedVariant(
        "V2_anchor_raw",
        True,
        True,
        False,
        False,
        False,
        None,
        anchor_compiler=True,
    ),
    HostedVariant("V3_raw", True, False, False, False, False, None),
    HostedVariant(
        "V3_anchor_raw",
        True,
        True,
        False,
        False,
        False,
        None,
        anchor_compiler=True,
        anchor_compiler_version=3,
        raw_rescue_tail=True,
    ),
)
MULTIVIEW_RETRIEVAL_VARIANTS = (
    HostedVariant("M0_multiview_raw", True, False, False, False, False, None),
    HostedVariant(
        "M2_repository_raw",
        True,
        True,
        False,
        False,
        False,
        None,
        anchor_compiler=True,
        compiled_kinds=REPOSITORY_KINDS,
        drop_compiler_fallback=True,
    ),
    HostedVariant(
        "M3_experience_raw",
        True,
        True,
        False,
        False,
        False,
        None,
        anchor_compiler=True,
        compiled_kinds=EXPERIENCE_KINDS,
        drop_compiler_fallback=True,
    ),
)
MULTIMODAL_VARIANTS = (
    HostedVariant("MM0_caption", True, False, False, False, False, None),
    HostedVariant(
        "MM1_preserve",
        True,
        False,
        False,
        False,
        False,
        None,
        multimodal_preserve=True,
    ),
    HostedVariant(
        "MM2_dual",
        True,
        False,
        False,
        False,
        False,
        None,
        multimodal_preserve=True,
        multimodal_native=True,
    ),
)
VARIANTS = (
    ATTRIBUTION_VARIANTS
    + EXPERIENCE_VARIANTS
    + CODING_MATRIX_VARIANTS
    + CLEAN_RERANK_VARIANTS
    + CODE_AWARE_VARIANTS
    + ANCHOR_COMPILER_VARIANTS
    + MULTIVIEW_RETRIEVAL_VARIANTS
    + MULTIMODAL_VARIANTS
)
DEFAULT_VARIANT = "A4_pack_7000"


def variant(name: str) -> HostedVariant:
    for item in VARIANTS:
        if item.name == name:
            return item
    raise ValueError(f"unknown AML Hosted variant {name!r}")
