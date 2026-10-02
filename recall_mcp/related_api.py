"""The `recall_related` tool body: trusted evidence structurally related to one chunk.

Moved out of `recall_mcp.service`, which re-exports every name defined here. Collaborators are
imported from the modules that own them, so a test patches a collaborator on THIS module.
"""

from __future__ import annotations

from pydantic import (
    BaseModel,
    Field,
)

from recall.calibration import Calibration
from recall.observability import get_logger as _get_logger
from recall.related import trusted_related
from recall.security_policy import (
    AccessContext,
    SourceSecurityPolicy,
)
from recall.store import PgVectorStore
from recall.trust_policy import TrustPolicy

from recall_mcp.models import EvidenceItemModel
from recall_mcp.retrieval import _trusted_evidence_item_model


_log = _get_logger("mcp.service")


class RelatedResult(BaseModel):
    """Related evidence whose candidates each passed an independent trust evaluation."""

    seed_chunk_id: str = Field(description="Chunk that seeded the structural relation.")
    relation: str = Field(description="source | ordinal | supersession.")
    generation_id: str = Field(description="Generation identity shared by seed and items.")
    items: list[EvidenceItemModel] = Field(description="Trusted related evidence items.")
    rejected_count: int = Field(description="Candidates rejected by independent trust checks.")
    explanation: dict[str, object] | None = Field(
        default=None, description="Optional structured explanation when explain=true."
    )


def related_memory(
    store: PgVectorStore,
    seed_chunk_id: str,
    *,
    relation: str = "source",
    max_items: int = 5,
    calibration: Calibration | None = None,
    policy: TrustPolicy | None = None,
    explain: bool = False,
    security_policy: SourceSecurityPolicy | None = None,
    access_context: AccessContext | None = None,
) -> RelatedResult:
    """Return structurally related evidence after independent trust evaluation.

    Args:
        store: tenant and generation bound read store.
        seed_chunk_id: chunk that defines the relation.
        relation: `source`, `ordinal`, or `supersession`.
        max_items: positive bounded candidate limit.
        calibration: optional trust calibration, resolved from the store when omitted.
        explain: include stable machine readable explanation metadata.

    Raises:
        ValueError: if the relation, seed, or item limit is invalid.
    """
    result = trusted_related(
        store,
        seed_chunk_id,
        relation=relation,  # type: ignore[arg-type]
        max_items=max_items,
        calibration=calibration,
        policy=policy,
        explain=explain,
        security_policy=security_policy,
        access_context=access_context,
    )
    items = [_trusted_evidence_item_model(item) for item in result.items]
    return RelatedResult(
        seed_chunk_id=result.seed_chunk_id,
        relation=result.relation,
        generation_id=result.generation_id,
        items=items,
        rejected_count=result.rejected_count,
        explanation=result.explanation,
    )
