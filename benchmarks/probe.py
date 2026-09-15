"""Report the corpus chunk count.

Connects to the configured PostgreSQL database and queries the total chunk count
from the recall_chunks_v1 table. Prints the result in the format: chunks=<count>

Environment variables:
    RECALL_SERVING_DSN: PostgreSQL connection string (defaults to RECALL_DSN)
    RECALL_DSN: PostgreSQL connection string (default: postgresql://recall:recall@localhost:5432/recall)
"""

import os
import sys

try:
    import psycopg
except ImportError:
    print("Error: psycopg is required. Install it with: pip install psycopg[binary]", file=sys.stderr)
    sys.exit(1)


def get_chunk_count() -> int:
    """Query the database for the total chunk count."""
    # Get DSN from environment with fallback
    dsn = os.environ.get(
        "RECALL_SERVING_DSN",
        os.environ.get("RECALL_DSN", "postgresql://recall:recall@localhost:5432/recall"),
    )
    
    try:
        # Set a connection timeout of 5 seconds
        with psycopg.connect(dsn, connect_timeout=5) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM recall_chunks_v1")
                result = cur.fetchone()
                if result:
                    return int(result[0])
                return 0
    except Exception as e:
        print(f"Error querying database: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> int:
    count = get_chunk_count()
    print(f"chunks={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
