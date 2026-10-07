"""RE-call on one SQLite file: a local store that needs no database server.

`LiteStore` stores chunks and ranks them; every rule that decides trust (verdicts, supersession
resolution, calibration maths, indexing) stays in the shared code the Postgres store also uses.
"""

from recall.lite.store import LITE_DSN_PREFIX, LiteStore, LiteStoreError, is_lite_dsn

__all__ = ["LITE_DSN_PREFIX", "LiteStore", "LiteStoreError", "is_lite_dsn"]
