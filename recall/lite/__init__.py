"""RE-call on one SQLite file: a local store that needs no database server.

`LiteStore` stores chunks and ranks them; the rules that decide trust (verdicts, supersession
resolution, calibration maths, indexing) stay in the shared code the Postgres store also uses. A
few rules are copied rather than shared, because the originals are SQL or live in a module that
loads psycopg, and tests pin each copy to its original: `Scope.predicate` (evaluated in Python by
`recall.lite.store._matches`), `recall.store._numeric_query_terms` (`_numeric_terms`),
`recall.store._chunk_identity` (`_chunk_identity`), and PostgreSQL's `ts_rank` (the keyword leg).
"""

from recall.lite.store import (
    LITE_DSN_PREFIX,
    LiteStore,
    LiteStoreError,
    LiteUnsupported,
    is_lite_dsn,
    lite_dsn_path,
)
from recall.lite.writer_lock import LiteWriterBusy, LiteWriterCoalesced, lite_lane, lite_lane_status

__all__ = [
    "LITE_DSN_PREFIX",
    "LiteStore",
    "LiteStoreError",
    "LiteUnsupported",
    "LiteWriterBusy",
    "LiteWriterCoalesced",
    "is_lite_dsn",
    "lite_dsn_path",
    "lite_lane",
    "lite_lane_status",
]
