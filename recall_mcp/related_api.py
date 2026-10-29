"""The `recall_related` tool body: trusted evidence structurally related to one chunk, and the
`recall_records` tool body: audit records by exact key, the other structural read.

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
from recall.records import RecordsQueryError, lookup_records
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
    relation: str = Field(description="source | ordinal | supersession | caused_by.")
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
        relation: `source`, `ordinal`, `supersession`, or `caused_by`.
        max_items: returned-item limit, from 1 to 50. Supersession evaluates the complete matching
            lineage before limiting returned items; source, ordinal, and `caused_by` relations also limit candidates.
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


def records_memory(
    store: PgVectorStore,
    *,
    repo: str | None = None,
    anchors: list[dict[str, str | None]] | None = None,
    ids: list[str] | None = None,
    kinds: list[str] | None = None,
    security_policy: SourceSecurityPolicy | None = None,
    access_context: AccessContext | None = None,
) -> dict[str, object]:
    """Audit records by exact key (`recall.records.lookup_records`), the C5 envelope.

    Raises:
        recall.records.RecordsQueryError: for a query that cannot be answered as asked.
    """
    # The tool has no `all_records`, so the library's refusal, which names it, would offer the
    # caller a parameter this tool does not take.
    if anchors is None and ids is None:
        raise RecordsQueryError("give anchors (with repo) or ids")
    return lookup_records(
        store,
        repo=repo,
        anchors=anchors,
        ids=ids,
        kinds=kinds,
        security_policy=security_policy,
        access_context=access_context,
    )
