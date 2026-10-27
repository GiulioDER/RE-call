"""The allowlist for physical table names that reach SQL through the control plane.

A leaf so the store can validate a routed table name without importing the control plane.
`recall.control_plane` re-exports both names.
"""

from __future__ import annotations

import re

#: Physical table identifiers are interpolated into SQL, so this is an allowlist, not a filter.
#:
#: `str.isidentifier()` was the previous gate and is too weak in three separate ways, each a live
#: defect rather than a hypothetical. It accepts non-ASCII, because `café` is a valid Python
#: identifier. It accepts uppercase, and PostgreSQL folds an unquoted `Chunks_G1` to `chunks_g1`,
#: so the registry row and the physical table silently disagree. It accepts any length, and
#: PostgreSQL truncates identifiers at NAMEDATALEN-1 = 63 bytes, so two registry rows differing
#: only after byte 63 map to ONE table. The allowlist below refuses all three.
#:
#: The bound is 46, not 63, because nothing is named by the table name alone. Every derived
#: object suffixes it, and the longest suffix shipped is `_tenant_isolation` at 17 bytes
#: (`recall/migrations/sql/0001_v08_baseline.sql`). A 63-byte table therefore yields an 80-byte
#: policy or index name that PostgreSQL silently truncates, and `readiness_facts` compares index
#: names by EXACT string, so readiness would report a missing index for one that exists and is
#: valid. A bound is named by what it bounds: this one bounds the derived identifier.
_TABLE_NAME = re.compile(r"^[a-z_][a-z0-9_]{0,45}$")


def validate_table_name(table: str) -> str:
    """Return `table` if it matches the physical-identifier allowlist, else raise.

    This is the chokepoint for every table name that reaches an f-string THROUGH THE CONTROL
    PLANE: it runs on the way IN (`register_generation`) and again on the way OUT (`_generation`),
    so a row written by another client, or by a direct `INSERT`, cannot smuggle an identifier into
    a query at read time.

    It is not yet the only gate in the codebase. `PgVectorStore.__init__` and
    `recall.schema._validate_target` still use `str.isidentifier()`, so a store constructed
    directly, or `recall schema apply --table`, can still carry a name this allowlist would
    refuse. Those two are a deliberate follow-up rather than an oversight: tightening them changes
    behaviour for tables that already exist, which needs a compatibility decision this change does
    not make. Do not describe this function as the single chokepoint until they are converted.
    """
    if not isinstance(table, str) or not _TABLE_NAME.fullmatch(table):
        raise ValueError(
            "physical table must match ^[a-z_][a-z0-9_]{0,45}$ (lowercase ASCII, at most 46 "
            f"bytes, leaving room for the 17-byte `_tenant_isolation` suffix); got {table!r}"
        )
    return table
