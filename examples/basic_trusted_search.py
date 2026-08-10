"""Five-minute trusted-search example against the bundled demo corpus.

Before running this file, index the demo corpus into a local Postgres table:

    docker compose up -d --wait
    python -m recall.cli --table recall_quickstart \
        --migration-dsn postgresql://recall:recall@localhost:5432/recall \
        schema --dim 384 apply
    RECALL_TRUST_MODE=development python -m recall.cli --table recall_quickstart index corpus

The example uses development trust mode explicitly because the bundled demo corpus has no certified
production calibration. Production code should omit that policy and publish a generation-bound
calibration instead.
"""
from __future__ import annotations

import os

from recall.embeddings import Embedder, FastEmbedEmbedder, HashingEmbedder
from recall.store import PgVectorStore
from recall.trust import trusted_search
from recall.trust_policy import TrustPolicy


def make_embedder(name: str) -> Embedder:
    if name == "hashing":
        return HashingEmbedder(dim=64)
    if name == "fastembed":
        return FastEmbedEmbedder()
    raise ValueError(f"unknown embedder: {name!r} (use 'fastembed' or 'hashing')")


def main() -> None:
    dsn = os.environ.get("RECALL_SERVING_DSN", "postgresql://recall:recall@localhost:5432/recall")
    table = os.environ.get("RECALL_TABLE", "recall_quickstart")

    embedder = make_embedder(os.environ.get("RECALL_EMBEDDER", "fastembed"))
    with PgVectorStore(dsn, dim=embedder.dim, table=table) as store:
        result = trusted_search(
            store,
            embedder,
            "how many requests per second can a client make?",
            k=3,
            policy=TrustPolicy.development(),
        )

    if result.abstained:
        print(f"ABSTAIN: {result.reason}")
    elif result.trust_state == "degraded":
        print(f"DEGRADED: {result.failure_code or result.calibration_status}")
    else:
        print("OK")
    for hit in result.hits:
        print(
            f"{hit.verdict:10} confidence={hit.confidence:.2f} "
            f"source={hit.provenance.source}"
        )


if __name__ == "__main__":
    main()
