"""RE-call on one SQLite file: a local store that needs no database server.

`LiteStore` stores chunks and ranks them; the rules that decide trust (verdicts, supersession
resolution, calibration maths, indexing) stay in the shared code the Postgres store also uses. One
rule is copied rather than shared: `Scope.predicate` is SQL, so the lite store evaluates the same
scope arms in Python (`recall.lite.store._matches`), and its tests pin the copy to the original.
"""

from recall.lite.store import LITE_DSN_PREFIX, LiteStore, LiteStoreError, is_lite_dsn

__all__ = ["LITE_DSN_PREFIX", "LiteStore", "LiteStoreError", "is_lite_dsn"]
