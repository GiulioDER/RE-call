from __future__ import annotations

import ast
from pathlib import Path

import recall_mcp.models as models
import recall_mcp.service as service


RESPONSE_MODELS = (
    "SearchHit",
    "SearchResult",
    "EvidenceItemModel",
    "EvidenceCardModel",
    "EvidenceResult",
    "ReasoningProjectionResult",
    "CurrentStateRecordModel",
    "CurrentStateResult",
    "RelatedResult",
    "ReasoningProposalItem",
    "ReasoningProposalResult",
    "ReasoningAuditResult",
    "IndexResult",
    "ForgetResult",
    "MemoryStatsResult",
    "InventoryEntry",
    "InventoryResult",
    "RewritePlanResult",
)


def test_service_reexports_the_canonical_response_models() -> None:
    assert all(getattr(service, name) is getattr(models, name) for name in RESPONSE_MODELS)


def test_service_does_not_declare_response_models() -> None:
    tree = ast.parse(Path(service.__file__).read_text(encoding="utf-8"))
    declared = {
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name in RESPONSE_MODELS
    }
    assert declared == set()
