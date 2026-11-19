"""The write-time hook finds what plain `recall setup` indexed (issue 1425).

Invariant: with no active generation for the tenant, the hook's lexical search reads the legacy
table its config names, under that table's tenant isolation, so a memo `recall setup` wrote is
found. Failure mode it catches: the hook searched only the active generation in
`recall_chunks_v1`, while plain `recall setup` indexes into the legacy `chunks` table
(`recall.memory_index.index_memory_directory`), so on a setup-only install it found nothing.

Red proof receipt ``write-time-legacy-01``, 2026-11-19, in CI (the database cannot run on the
maintainer's hosts during the AML validation window): with the hook as on master `2e546bec1`,
`test_the_hook_finds_a_memo_plain_setup_indexed` passed its indexing check and failed on
`assert [] == ['crlf-churn.md']` (run 44598039479, commit 9e5dfbdbf); it passes with the fix.
An earlier red run of it (44591719683) was invalid and is not counted: the test then never
created its table, so it would have failed with a correct hook too. Production symbol targeted:
`recall_hooks.write_time._search_connection`.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from tests.conftest import TEST_DSN, requires_db
from tests.test_absent_memo_facts import manager as manager  # noqa: F401  # the fixture

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

    from recall.embeddings import resolve_embedder
    from recall.memory_index import index_memory_directory
    from recall.schema import apply_migrations
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
    printed: list[str] = []
    try:
        # As `recall setup` does (`_prepare_schema_for_embedder`) before it indexes the memo folder.
        apply_migrations(TEST_DSN, table=table, dim=resolve_embedder("hashing").dim)
        index_memory_directory(
            dsn=TEST_DSN, embedder_name="hashing", memory_dir=memory,
            tenant=tenant, table=table, env={}, print_fn=lambda *a, **k: printed.append(str(a[0])),
        )
        # Indexing is best effort and only prints on failure, so an empty search could otherwise
        # be a failed index rather than a hook that looks in the wrong place.
        assert printed and printed[-1].startswith("Indexed ") and "from 2 files" in printed[-1], printed

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
def test_a_role_without_bypassrls_searches_the_active_generation_not_the_legacy_table(
    manager: Any, unprivileged_dsn: str, tmp_path: Path
) -> None:
    """Invariant: for a tenant with an active generation, the hook searches that generation, also
    under a role row-level security binds (a serving role that is neither superuser nor BYPASSRLS).

    Failure mode: the has-active-generation probe ran before the tenant setting existed on the
    connection, and `recall_generations` is RLS forced, so under such a role it always read false
    and the hook served the tenant's legacy rows instead, which the generation route never serves
    (CCA run 20261119T212834Z, BUG-001). The superuser `TEST_DSN` bypasses RLS, so the test above
    cannot see this.

    Red proof: this test, with `recall_hooks/write_time.py` as on master f25e5ab1e (the probe
    without the tenant on the connection), failed in its first assertion, the legacy memo returned
    in place of the generation's; recorded in the PR. Production symbol targeted:
    `recall_hooks.write_time.tenant_option` as used by `search`.
    """
    import psycopg

    from recall.embeddings import resolve_embedder
    from recall.memory_index import index_memory_directory
    from recall.schema import apply_migrations
    from recall_hooks.write_time import search
    from tests.conftest import role_is_unprivileged
    from tests.test_gc_retraction import _promote

    assert role_is_unprivileged(unprivileged_dsn)
    tenant = manager.tenant_id
    _promote(manager, {"generation-memo.md": b"quokka lantern generation memo"}, "v1")
    table = f"wt_rls_{uuid.uuid4().hex[:10]}"
    memory = tmp_path / "memory"
    memory.mkdir()
    (memory / "legacy-memo.md").write_text("quokka lantern legacy memo\n", encoding="utf-8")
    printed: list[str] = []
    try:
        apply_migrations(TEST_DSN, table=table, dim=resolve_embedder("hashing").dim)
        index_memory_directory(
            dsn=TEST_DSN, embedder_name="hashing", memory_dir=memory,
            tenant=tenant, table=table, env={}, print_fn=lambda *a, **k: printed.append(str(a[0])),
        )
        assert printed and printed[-1].startswith("Indexed "), printed

        hits = search(
            "quokka lantern", {"dsn": unprivileged_dsn, "tenant": tenant, "table": table}, _OPTIONS
        )

        assert [name for name, _, _ in hits] == ["generation-memo.md"]
    finally:
        with psycopg.connect(TEST_DSN, autocommit=True) as conn:
            conn.execute(f'DROP TABLE IF EXISTS "{table}"')


def test_the_tenant_on_the_connection_is_escaped_for_libpq() -> None:
    """Invariant: whitespace or a backslash in the tenant cannot split or end the libpq options
    string (libpq splits it on every whitespace character).

    Red proof by mutation: with `tenant_option` returning the tenant unescaped, and with it
    escaping only spaces and backslashes, this failed on the equality below; recorded in the PR.
    """
    from recall_hooks.write_time import tenant_option

    assert tenant_option("a b\\c\td") == "-c recall.tenant_id=a\\ b\\\\c\\\td"


@requires_db
def test_a_missing_legacy_table_is_no_hits_not_an_error() -> None:
    """A wizard install whose generation is not active yet has no legacy table; that must read as
    nothing found, not as a failure that starts the hook's cooldown.

    Passes on the unfixed hook by construction (it never read the legacy table). Red proof is by
    mutation, in CI: with the fix but without its missing-table handling (commit fa8472dcb, run
    44594904129), this test failed with `psycopg.errors.UndefinedTable`."""
    from recall_hooks.write_time import search

    hits = search(
        "anything at all here",
        {"dsn": TEST_DSN, "tenant": f"none-{uuid.uuid4().hex[:6]}", "table": f"absent_{uuid.uuid4().hex[:10]}"},
        _OPTIONS,
    )

    assert hits == []
