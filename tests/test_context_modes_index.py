"""The context-mode invariants asserted where the rows actually exist.

`tests/test_context_modes.py` tests `contextual_passages` in isolation. That is necessary and not
sufficient: the claim is about what RE-call STORES, and between the function and the row sit
`Indexer`, the chunker, the metadata dict and two generations. So the load-bearing invariant is
re-asserted here against real PostgreSQL rows, read back with SQL rather than inferred from the
call that wrote them.
"""
from __future__ import annotations

import contextlib
import hashlib
import uuid
from pathlib import Path
from typing import NamedTuple

import psycopg
import pytest

from recall.control_plane import ControlPlane
from recall.context import ContextPolicy, context_policy_for_profile
from recall.embeddings import EmbeddingProfile, HashingEmbedder, embedding_profile_id
from recall.embedding_registry import REGISTERED_PROFILES
from recall.index import Indexer, ShadowIndexTarget
from recall.schema import apply_migrations
from recall.store import PgVectorStore

from tests.conftest import TEST_DSN, requires_db

DIM = 384

#: The symmetric baseline first: it is the arm every other one is compared against.
PROFILE_FOR_MODE = {
    "none": "bge-small-symmetric-v1",
    "document": "bge-small-context-document-v1",
    "section": "bge-small-context-section-v1",
    "neighbor": "bge-small-context-neighbor-v1",
}

def _para(word: str) -> str:
    """A paragraph past `chunk_text`'s 800-character budget, so each file really is split.

    A one-chunk file would make neighbour mode indistinguishable from section mode here, and the
    invariant would be asserted over a corpus that cannot exercise the thing under test.
    """
    return (f"{word} " * 140).strip() + "."


CORPUS = {
    "with-frontmatter.md": (
        f"---\ntitle: A Document\n---\n\n# Heading One\n\n{_para('alpha')}\n\n"
        f"## Nested Heading\n\n{_para('beta')}\n\n{_para('gamma')}\n"
    ),
    "no-headings.md": f"{_para('delta')}\n\n{_para('epsilon')}\n\n{_para('zeta')}\n",
    "nested-headings.md": (
        f"# One\n\n{_para('eta')}\n\n## Two\n\n{_para('theta')}\n\n"
        f"### Three\n\n{_para('iota')}\n"
    ),
}


class _ProfiledEmbedder:
    """A `HashingEmbedder` carrying a registered profile's real identity.

    The identity matters: `Indexer.__init__` refuses an embedder whose profile's context version
    disagrees with the index's context policy, and that check is SKIPPED for a legacy
    (`legacy-unverified`) profile. Indexing these tests with a bare `HashingEmbedder` would take
    the exempt path and prove nothing about the profiles the modes actually ship under.

    The vector is a function of the text it is handed, so two modes producing the same vector
    would mean they produced the same embedding text.
    """

    def __init__(self, profile_id: str) -> None:
        self.profile: EmbeddingProfile = REGISTERED_PROFILES[profile_id].identity(
            artifact_digest=REGISTERED_PROFILES[profile_id].artifact_digest or "a" * 64
        )
        self.name = profile_id
        self._inner = HashingEmbedder(dim=DIM)

    @property
    def dim(self) -> int:
        return DIM

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._inner.embed(texts)


def _write_corpus(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for name, text in CORPUS.items():
        (corpus / name).write_text(text, encoding="utf-8")
    return corpus


class Row(NamedTuple):
    """One stored chunk, named.

    Positional indices (`r[2]`, `r[3]`) carried the load-bearing assertions of this file while
    their meaning lived only in a docstring, so reordering the SELECT would have re-aimed
    "mode altered stored text" at another column without failing anything.
    """

    source: str
    ord: int
    text: str
    content_hash: str | None
    index_fingerprint: str | None
    context_mode: str | None
    context_version: str | None
    text_start: int | None
    text_end: int | None
    heading_hierarchy: list[str] | None


def _rows(store: PgVectorStore) -> list[Row]:
    """Every mode-sensitive stored field, read with SQL against the table.

    Read directly rather than through a store helper: the helpers carry their own predicates, and
    the question here is what is ON DISK.

    ⚠️ The `set_config` is not decorative, but it is currently unfalsifiable: the shipped test
    role is a superuser, so FORCE ROW LEVEL SECURITY is inert and this read would return the same
    rows without it. `test_the_tenant_guc_is_what_makes_this_read_correct` establishes the premise
    on an unprivileged role instead of leaving it asserted by a docstring.
    """
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (store.tenant,))
        return [
            Row(*row)
            for row in conn.execute(
                f"SELECT source, (metadata->>'ord')::int, text, metadata->>'content_hash', "
                f"metadata->>'index_fingerprint', metadata->>'context_mode', "
                f"metadata->>'context_version', (metadata->>'text_start')::int, "
                f"(metadata->>'text_end')::int, metadata->'heading_hierarchy' "
                f"FROM {store.table} WHERE tenant_id = %s "
                f"ORDER BY source, (metadata->>'ord')::int",
                (store.tenant,),
            ).fetchall()
        ]


def raw_hash(rows: list[Row]) -> str:
    """The identity of the STORED text and its per-file hash, in a stable order.

    Refuses a NULL rather than folding it. `metadata->>'x'` is SQL NULL when the key is absent,
    and this used to interpolate it as the string "None" on both sides of a comparison — so
    stripping `content_hash` from every row left the test that is NAMED for content hashes green.
    An equality assertion over a missing key is satisfied by absence.
    """
    digest = hashlib.sha256()
    for row in rows:
        if row.content_hash is None:
            raise AssertionError(f"{row.source} carries no content_hash; the comparison is void")
        digest.update(
            f"{row.source}\x00{row.ord}\x00{row.text}\x00{row.content_hash}\x00".encode("utf-8")
        )
    return digest.hexdigest()


def expected_content_hashes() -> dict[str, str]:
    """`{basename: sha256}` computed from the corpus BYTES, not from what was stored."""
    return {
        name: hashlib.sha256(text.encode("utf-8")).hexdigest() for name, text in CORPUS.items()
    }


@requires_db
def test_raw_text_and_content_hashes_are_identical_across_generations_and_modes(
    tmp_path, make_store
) -> None:
    """The load-bearing invariant, over four independently indexed generations.

    Four separate `Indexer` runs into four separate tables, not one run fanned out: a shared run
    computes `chunk_text` ONCE and hands the same list to every target, so it could not detect a
    mode that changed the chunking. Independent runs can.
    """
    corpus = _write_corpus(tmp_path)
    stored: dict[str, list[Row]] = {}
    for mode, profile_id in PROFILE_FOR_MODE.items():
        store = make_store(DIM)
        embedder = _ProfiledEmbedder(profile_id)
        stats = Indexer(
            store, embedder, context_policy=context_policy_for_profile(profile_id)
        ).index_path(corpus)
        assert stats.files == len(CORPUS)
        stored[mode] = _rows(store)

    baseline = stored["none"]
    assert len(baseline) > len(CORPUS), "the corpus must produce more than one chunk per file"

    # Pin the baseline to the corpus BYTES before comparing arms. Comparing two arms' hashes to
    # each other is satisfied by both being absent: stripping `content_hash` from every stored row
    # left this test green, because `metadata->>'content_hash'` is NULL on both sides.
    expected = expected_content_hashes()
    for row in baseline:
        assert row.content_hash == expected[Path(row.source).name], row.source
        assert row.index_fingerprint and len(row.index_fingerprint) == 64
        assert row.text_start is not None and row.text_end is not None

    for mode in ("document", "section", "neighbor"):
        rows = stored[mode]
        assert [(r.source, r.ord) for r in rows] == [(r.source, r.ord) for r in baseline]
        assert [r.text for r in rows] == [r.text for r in baseline], f"{mode} altered stored text"
        assert [r.content_hash for r in rows] == [r.content_hash for r in baseline], (
            f"{mode} altered content_hash"
        )
        # The offsets and the heading path address the same bytes of the same document too: a
        # mode may change what is EMBEDDED and nothing else that is stored.
        assert [(r.text_start, r.text_end) for r in rows] == [
            (r.text_start, r.text_end) for r in baseline
        ], f"{mode} moved the stored offsets"
        assert [r.heading_hierarchy for r in rows] == [
            r.heading_hierarchy for r in baseline
        ], f"{mode} altered the stored heading hierarchy"
        assert raw_hash(rows) == raw_hash(baseline)
        # The deliberate exception: `index_fingerprint` MUST differ, or re-indexing the same
        # corpus under a new mode would be skipped as unchanged and the generation would keep
        # vectors built under the old one. This assertion alone does not prove the MODE is what
        # made it differ — the profile ID differs here too — so the mode's own contribution is
        # isolated in the test below.
        assert all(r.index_fingerprint != b.index_fingerprint for r, b in zip(rows, baseline)), (
            f"{mode} shares the baseline's index fingerprint; a re-index would be skipped"
        )


@requires_db
def test_the_index_fingerprint_changes_when_only_the_context_mode_changes(
    tmp_path, make_store
) -> None:
    """One embedder, one profile ID, two context policies.

    The test above indexes each mode under its own profile, so its fingerprints differ whether or
    not the mode is part of them: removing the mode from the fingerprint input leaves that test
    green. Holding the embedder fixed is the only way to attribute the difference to the mode,
    and a legacy (`legacy-unverified`) profile is what allows one embedder to be indexed under
    two policies at all.
    """
    corpus = _write_corpus(tmp_path)
    fingerprints = {}
    for mode in ("none", "document", "section", "neighbor"):
        store = make_store(DIM)
        # The SAME embedder identity in every arm: `hashing-384`, legacy profile, no context
        # version to disagree with the policy.
        Indexer(
            store, HashingEmbedder(dim=DIM),
            context_policy=ContextPolicy(mode=mode),  # type: ignore[arg-type]
        ).index_path(corpus)
        rows = _rows(store)
        assert rows
        assert all(r.index_fingerprint and len(r.index_fingerprint) == 64 for r in rows), (
            "every row carries an index fingerprint"
        )
        fingerprints[mode] = [r.index_fingerprint for r in rows]
        # The premise: the profile ID really is identical across arms.
        assert embedding_profile_id(HashingEmbedder(dim=DIM)) == "hashing-384"

    for mode in ("document", "section", "neighbor"):
        assert fingerprints[mode] != fingerprints["none"], (
            f"the {mode} fingerprint equals the baseline's under an identical embedder: the "
            "context mode is not part of the fingerprint, so switching mode would skip re-indexing"
        )
    # And the three modes are distinct from each other, not merely from the baseline.
    assert len({tuple(v) for v in fingerprints.values()}) == 4


@requires_db
def test_the_tenant_guc_is_what_makes_this_read_correct(tmp_path, make_store,
                                                        unprivileged_dsn: str) -> None:
    """`_rows` sets `recall.tenant_id`, and on the shipped role that setting does nothing.

    `docker-compose.yml` ships `POSTGRES_USER=recall`, the cluster superuser, for whom FORCE ROW
    LEVEL SECURITY is inert — every read in this file returns the same rows with the `set_config`
    deleted. That is the SAFE direction of the trap (no false zero), but it means a future GUC
    mistake here is invisible in CI, and this repository has already misread a policy-hidden row
    as a fact about the data four times.

    So the premise is established on a role that RLS actually constrains: naming another tenant
    must hide the rows, and naming the owning tenant must reveal them. The second half is what
    stops "returned zero" from being explained by an empty table.
    """
    corpus = _write_corpus(tmp_path)
    store = make_store(DIM)
    Indexer(store, HashingEmbedder(dim=DIM)).index_path(corpus)
    assert _rows(store), "the table must hold rows before absence can mean anything"

    with psycopg.connect(unprivileged_dsn, autocommit=True) as conn:
        role, superuser, bypass = conn.execute(
            "SELECT current_user, rolsuper, rolbypassrls FROM pg_roles WHERE rolname = current_user"
        ).fetchone()
        assert not superuser and not bypass, f"{role!r} makes this assertion vacuous"

        # An UNPREDICATED count: only the policy can answer it, so this measures the POLICY and
        # not a `WHERE tenant_id = %s` the query carried in with it.
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", ("some-other-tenant",))
        hidden = conn.execute(f"SELECT count(*) FROM {store.table}").fetchone()[0]
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (store.tenant,))
        visible = conn.execute(f"SELECT count(*) FROM {store.table}").fetchone()[0]

    assert hidden == 0, "another tenant's GUC must not see these rows"
    assert visible > 0, "the owning tenant's GUC must see them; otherwise zero proves nothing"


@requires_db
@pytest.mark.parametrize("mode", ["none", "document", "section", "neighbor"])
def test_context_mode_and_version_are_recorded_in_chunk_metadata(
    tmp_path, make_store, mode: str
) -> None:
    corpus = _write_corpus(tmp_path)
    profile_id = PROFILE_FOR_MODE[mode]
    store = make_store(DIM)
    policy = context_policy_for_profile(profile_id)
    assert policy.mode == mode, "the registry must be what selects the mode, not this test"

    Indexer(store, _ProfiledEmbedder(profile_id), context_policy=policy).index_path(corpus)

    rows = _rows(store)
    assert rows
    assert {r.context_mode for r in rows} == {mode}
    assert {r.context_version for r in rows} == {policy.version}
    # And the profile identity carries it too, so the row and the vector agree on the mode.
    assert REGISTERED_PROFILES[profile_id].context_version == (
        "raw-v1" if mode == "none" else f"context-{mode}-v1"
    )


@requires_db
@pytest.mark.parametrize(
    ("active_mode", "shadow_mode"),
    [
        # The realistic migration: the symmetric baseline serves while a context mode is built.
        ("none", "section"),
        # Two CONTEXT modes. Without this pair the active generation is always `none`, where the
        # embedding text and the chunk are the same string — so an indexer that stored embedding
        # text instead of chunk text would leave the active table looking correct. A mutation
        # sweep found exactly that hole.
        ("document", "neighbor"),
    ],
)
def test_a_dual_write_leaves_byte_identical_raw_text_in_both_generations(
    tmp_path, active_mode: str, shadow_mode: str
) -> None:
    """The real dual-write path, with the two generations on DIFFERENT context modes.

    Both must hold the same stored text, or a cutover changes what the corpus says rather than
    only how it is embedded.
    """
    suffix = uuid.uuid4().hex[:10]
    tenant = f"t_{suffix}"
    active_id, shadow_id = f"g_active_{suffix}", f"g_shadow_{suffix}"
    active_table, shadow_table = f"c_active_{suffix}", f"c_shadow_{suffix}"

    # Everything provisioned INSIDE the try, including the two stores. Provisioning outside it
    # meant a failure partway through (the second `apply_migrations`, either `register_generation`)
    # left a physical table and a registry row behind with no teardown path at all — and a
    # generation row whose physical table is gone is exactly what `invalid_generations` exists to
    # complain about, in every later test on the same database.
    control = ControlPlane(TEST_DSN)
    active = shadow = None
    try:
        control.apply_migrations()
        for table in (active_table, shadow_table):
            apply_migrations(TEST_DSN, table=table, dim=DIM)
        control.register_generation(active_id, active_table, "profile-a", DIM)
        control.register_generation(shadow_id, shadow_table, "profile-b", DIM)
        control.set_generation_state(active_id, "ready", chunk_count=0, source_count=0)
        active = PgVectorStore(
            TEST_DSN, dim=DIM, table=active_table, tenant=tenant, generation_id=active_id
        )
        shadow = PgVectorStore(
            TEST_DSN, dim=DIM, table=shadow_table, tenant=tenant, generation_id=shadow_id
        )
        corpus = _write_corpus(tmp_path)
        Indexer(
            active,
            _ProfiledEmbedder(PROFILE_FOR_MODE[active_mode]),
            context_policy=ContextPolicy(mode=active_mode),  # type: ignore[arg-type]
            shadow=ShadowIndexTarget(
                store=shadow,
                embedder=_ProfiledEmbedder(PROFILE_FOR_MODE[shadow_mode]),
                control_plane=control,
                context_policy=ContextPolicy(mode=shadow_mode),  # type: ignore[arg-type]
            ),
        ).index_path(corpus)

        active_rows, shadow_rows = _rows(active), _rows(shadow)
        assert active_rows and len(active_rows) == len(shadow_rows)
        # Pinned to the corpus bytes, so "both generations agree" cannot be satisfied by both
        # being empty or by both carrying a NULL hash.
        expected = expected_content_hashes()
        for row in active_rows:
            assert row.content_hash == expected[Path(row.source).name], row.source
        assert [(r.source, r.ord, r.text, r.content_hash) for r in active_rows] == [
            (r.source, r.ord, r.text, r.content_hash) for r in shadow_rows
        ]
        assert [(r.text_start, r.text_end) for r in active_rows] == [
            (r.text_start, r.text_end) for r in shadow_rows
        ]
        assert raw_hash(active_rows) == raw_hash(shadow_rows)
        # Not vacuous: the two generations really were indexed under different modes, and their
        # fingerprints differ, so neither is a copy of the other.
        assert {r.context_mode for r in active_rows} == {active_mode}
        assert {r.context_mode for r in shadow_rows} == {shadow_mode}
        assert all(
            a.index_fingerprint != s.index_fingerprint
            for a, s in zip(active_rows, shadow_rows)
        )
        assert control.pending_events(tenant) == []
    finally:
        for store in (active, shadow):
            # A raising close() must not skip the DDL cleanup below it.
            if store is not None:
                with contextlib.suppress(Exception):
                    store.close()
        with psycopg.connect(TEST_DSN, autocommit=True) as conn:
            conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (tenant,))
            conn.execute("DELETE FROM recall_migration_events WHERE tenant_id = %s", (tenant,))
            conn.execute("DELETE FROM recall_tenant_routes WHERE tenant_id = %s", (tenant,))
            conn.execute(
                "DELETE FROM recall_index_generations WHERE generation_id = ANY(%s)",
                ([active_id, shadow_id],),
            )
            for table in (active_table, shadow_table):
                conn.execute(f"DROP TABLE IF EXISTS {table}")
                conn.execute(
                    "DELETE FROM recall_schema_migrations WHERE target_table = %s", (table,)
                )
