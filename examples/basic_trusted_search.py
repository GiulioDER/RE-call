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

from recall.store import PgVectorStore
from recall.trust_policy import TrustPolicy
from recall_mcp.service import make_embedder, search_memory


def main() -> None:
    dsn = os.environ.get("RECALL_SERVING_DSN", "postgresql://recall:recall@localhost:5432/recall")
    table = os.environ.get("RECALL_TABLE", "recall_quickstart")

    embedder = make_embedder(os.environ.get("RECALL_EMBEDDER", "fastembed"))
    with PgVectorStore(dsn, dim=embedder.dim, table=table) as store:
        result = search_memory(
            store,
            embedder,
            "how many requests per second can a client make?",
            k=3,
            policy=TrustPolicy.development(),
        )

    print(result.advice)
    for hit in result.hits:
        print(f"{hit.verdict:10} confidence={hit.confidence:.2f} source={hit.source}")


if __name__ == "__main__":
    main()
