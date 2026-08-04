"""Optional framework integrations for RE-call.

Each submodule targets one host framework and imports that framework at module load, so it raises
a clear error when the corresponding extra is not installed. Importing ``recall.integrations``
itself pulls in nothing heavy — pick the submodule you need, e.g. ``recall.integrations.langchain``.
"""
from __future__ import annotations

from typing import Any

from recall.types import TrustedHit, TrustedResult


def result_trust_metadata(result: TrustedResult) -> dict[str, Any]:
    """Result-level trust and lineage identity, shared by the framework adapters.

    Separate from `trust_metadata`, which is per hit. These keys answer "what produced this
    answer, and may I rely on it", and they have to travel with every document or node because a
    framework will happily hand a single document to a chain that never sees the result object.

    `recall_trust_state` and `recall_failure_code` are the load-bearing pair. Without them a
    degraded result is indistinguishable from a trusted one once the hits are unpacked, which is
    the exact boundary at which the trust signal used to be lost.
    """
    return {
        "recall_trust_state": result.trust_state,
        "recall_failure_code": result.failure_code,
        "recall_calibrated": result.calibrated,
        "recall_calibration_id": result.calibration_id,
        "recall_calibration_status": result.calibration_status,
        "recall_tenant_id": result.tenant_id,
        "recall_generation_id": result.generation_id,
        "recall_pipeline_fingerprint": result.pipeline_fingerprint,
        "recall_corpus_fingerprint": result.corpus_fingerprint,
        "recall_query_set_digest": result.query_set_digest,
    }


def trust_metadata(hit: TrustedHit) -> dict[str, Any]:
    """The ``recall_*`` + provenance metadata contract shared by the framework integrations.

    Defined once so the LangChain and LlamaIndex adapters cannot drift — they previously built
    near-identical dicts inline and had already diverged on ``chunk_id`` (present under LangChain,
    absent under LlamaIndex). ``chunk_id`` is now included for both; the LlamaIndex node also
    carries it as its native ``id_``, harmless redundancy in exchange for one contract.
    """
    prov = hit.provenance
    val = hit.validity
    return {
        "recall_verdict": hit.verdict,
        "recall_confidence": hit.confidence,
        "recall_cosine": hit.cosine,
        "chunk_id": hit.chunk.id,
        "source": prov.source,
        "file": prov.file,
        "ord": prov.ord,
        "indexed_at": prov.indexed_at.isoformat() if prov.indexed_at is not None else None,
        "superseded_by": val.superseded_by,
        "valid_from": val.valid_from.isoformat() if val.valid_from is not None else None,
        "valid_until": val.valid_until.isoformat() if val.valid_until is not None else None,
    }
