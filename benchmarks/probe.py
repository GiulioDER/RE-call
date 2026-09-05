"""Print the chunk count in this corpus from the recall package.

This is called by `scripts/run_probe.sh`, which CI runs as a single entry point.
It connects to the corpus database and reports the count of indexed chunks.
"""

import os
import sys
from pathlib import Path


def main():
    """Connect to the corpus and print the chunk count."""
    # Add the repo to the path so this standalone probe can import recall from any cwd.
    repo_root = Path(__file__).parent.parent
    sys.path.insert(0, str(repo_root))

    import psycopg
    from recall.store import PgVectorStore

    # Get DSN from environment, or use a default pointing at this machine's test container
    dsn = os.getenv(
        "RECALL_TEST_DSN",
        "postgresql://localhost:5432/recall"
    )

    try:
        # Ensure pgvector extension is installed
        try:
            with psycopg.connect(dsn) as conn:
                with conn.cursor() as cur:
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                conn.commit()
        except Exception:
            # Extension creation might fail on a database where we're not superuser,
            # but it may already be installed. Try to proceed anyway.
            pass

        # Create a store with the configured DSN
        store = PgVectorStore(dsn=dsn, dim=1024)
        # Ensure schema is created if it doesn't exist
        store.ensure_schema()
        count = store.count()
        print(f"Chunk count: {count}")
        store.close()
        return 0
    except Exception as e:
        print(f"Error querying corpus: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
