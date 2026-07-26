"""Optional framework integrations for RE-call.

Each submodule targets one host framework and imports that framework at module load, so it raises
a clear error when the corresponding extra is not installed. Importing ``recall.integrations``
itself pulls in nothing heavy — pick the submodule you need, e.g. ``recall.integrations.langchain``.
"""
from __future__ import annotations

from typing import Any

from recall.types import TrustedHit


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
