"""The write-time hook finds what plain `recall setup` indexed (issue 1425).

Invariant: with no active generation for the tenant, the hook's lexical search reads the legacy
table its config names, under that table's tenant isolation, so a memo `recall setup` wrote is
found. Failure mode it catches: the hook searched only the active generation in
`recall_chunks_v1`, while plain `recall setup` indexes into the legacy `chunks` table
(`recall.memory_index.index_memory_directory`), so on a setup-only install it found nothing.

Red proof receipt ``write-time-legacy-01``, 2026-11-19: the database test ran in CI against this
branch's parent commit (the hook unchanged) and failed on its `== ["crlf-churn.md"]` assertion
with `[]`; it passes with the fix. The database cannot be run on the maintainer's hosts during the
AML validation window, which is why the red run is CI's. Production symbol targeted:
`recall_hooks.write_time._search_connection`.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from tests.conftest import TEST_DSN, requires_db

_OPTIONS = {"k": 3, "connect_timeout": 5}


def test_the_hook_names_the_tenant_setting_the_store_isolates_by() -> None:
    """The hook spells the row-level-security setting out rather than import `recall` (its import
    cost is measured in `recall_hooks/write_time.py`), so it is pinned to the store's constant.

    A deliberately nonbehavioral control: it compares two constants, so red proof does not apply.
    """
    from recall.db_constants import TENANT_GUC
    from recall_hooks.write_time import LEGACY_TENANT_GUC

    assert LEGACY_TENANT_GUC == TENANT_GUC


@requires_db
def test_the_hook_finds_a_memo_plain_setup_indexed(tmp_path: Path) -> None:
    import psycopg

    from recall.memory_index import index_memory_directory
    from recall_hooks.write_time import search

    table = f"wt_legacy_{uuid.uuid4().hex[:10]}"
    tenant = f"wt-legacy-{uuid.uuid4().hex[:6]}"
    memory = tmp_path / "memory"
    memory.mkdir()
    (memory / "crlf-churn.md").write_text(
        "Pass newline to write_text, or Windows writes CRLF churn into every regenerated file.\n",
        encoding="utf-8",
    )
    (memory / "unrelated.md").write_text("The dashboard caches warm views per store.\n", encoding="utf-8")
    try:
        index_memory_directory(
            dsn=TEST_DSN, embedder_name="hashing", memory_dir=memory,
            tenant=tenant, table=table, env={}, print_fn=lambda *a, **k: None,
        )

        hits = search(
            "write_text newline CRLF churn",
            {"dsn": TEST_DSN, "tenant": tenant, "table": table},
            _OPTIONS,
        )

        assert [name for name, _, _ in hits] == ["crlf-churn.md"]
        # Another tenant's hook must not see this tenant's rows.
        assert search(
            "write_text newline CRLF churn",
            {"dsn": TEST_DSN, "tenant": "someone-else", "table": table},
            _OPTIONS,
        ) == []
    finally:
        with psycopg.connect(TEST_DSN, autocommit=True) as conn:
            conn.execute(f'DROP TABLE IF EXISTS "{table}"')


@requires_db
def test_a_missing_legacy_table_is_no_hits_not_an_error() -> None:
    """A wizard install whose generation is not active yet has no legacy table; that must read as
    nothing found, not as a failure that starts the hook's cooldown.

    Passes on the unfixed hook by construction (it never read the legacy table). Red proof is by
    mutation, in CI: with the fix but without its missing-table handling, this test failed with
    `psycopg.errors.UndefinedTable`."""
    from recall_hooks.write_time import search

    hits = search(
        "anything at all here",
        {"dsn": TEST_DSN, "tenant": f"none-{uuid.uuid4().hex[:6]}", "table": f"absent_{uuid.uuid4().hex[:10]}"},
        _OPTIONS,
    )

    assert hits == []
