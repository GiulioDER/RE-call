"""The documented PostgreSQL 19 tsvector recompute must lift forced row level security.

Invariant: every table the recompute block in docs/DEPENDENCIES.md updates or joins, and that a
migration puts under FORCE ROW LEVEL SECURITY, is switched to NO FORCE inside the block's
transaction before the first UPDATE and forced again before COMMIT; VACUUM stays outside it.

Failure mode caught: FORCE RLS binds the tables' owner too, and the tenant policy matches nothing
with ``recall.tenant_id`` unset, so a block without NO FORCE prints ``UPDATE 0`` and changes no row
for any role that is not a superuser. Measured 2026-11-24 on pgvector/pgvector:pg19 with the real
migrations applied by a NOBYPASSRLS owner role: the block without NO FORCE printed ``UPDATE 0``
twice and left 5 and 5 stale rows; this block printed ``UPDATE 5`` twice and left none.

Red proof (CCA run 20261124T213512Z, finding FA-1-1): against the pass-1 snapshot of
docs/DEPENDENCIES.md (the block with two bare UPDATEs and a VACUUM),
``tests/test_tsv_recompute_doc.py::test_recompute_lifts_forced_rls_on_every_table_it_touches``
fails its assertion naming recall_chunks_v1, recall_generations and chunks.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "DEPENDENCIES.md"
MIGRATIONS = ROOT / "recall" / "migrations" / "sql"
# The legacy table's default name; migrations spell it __RECALL_TABLE__ (recall/db_constants.py).
LEGACY_TABLE = "chunks"


def _recompute_block() -> str:
    blocks = re.findall(r"```sql\n(.*?)```", DOC.read_text(encoding="utf-8"), re.S)
    matching = [b for b in blocks if "SET tsv" in b]
    assert len(matching) == 1, "expected exactly one tsvector recompute block in DEPENDENCIES.md"
    return matching[0]


def _forced_tables() -> set[str]:
    forced = set()
    for path in MIGRATIONS.glob("*.sql"):
        for table in re.findall(
            r"ALTER TABLE\s+(\S+)\s+FORCE ROW LEVEL SECURITY", path.read_text(encoding="utf-8")
        ):
            forced.add(LEGACY_TABLE if table == "__RECALL_TABLE__" else table)
    return forced


def test_recompute_lifts_forced_rls_on_every_table_it_touches() -> None:
    block = _recompute_block()
    statements = [s.strip() for s in block.split(";") if s.strip()]
    updates = [i for i, s in enumerate(statements) if s.upper().startswith("UPDATE")]
    assert updates, "the recompute block has no UPDATE"
    touched = set()
    for i in updates:
        touched.update(re.findall(r"(?:UPDATE|FROM)\s+(\w+)", statements[i]))
    forced = touched & _forced_tables()
    assert forced, "no touched table is under FORCE ROW LEVEL SECURITY; the check observes nothing"

    def index_of(text: str) -> int:
        return statements.index(text) if text in statements else -1

    missing = sorted(
        t
        for t in forced
        if not (
            0 <= index_of(f"ALTER TABLE {t} NO FORCE ROW LEVEL SECURITY") < updates[0]
            and index_of(f"ALTER TABLE {t} FORCE ROW LEVEL SECURITY") > updates[-1]
        )
    )
    assert not missing, f"FORCE RLS not lifted around the UPDATEs for: {', '.join(missing)}"
    assert statements[0].upper() == "BEGIN" and statements[-1].upper() == "COMMIT"


def test_vacuum_is_not_inside_the_recompute_transaction() -> None:
    assert "VACUUM" not in _recompute_block().upper()
