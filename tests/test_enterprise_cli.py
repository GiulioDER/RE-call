"""`recall-enterprise` end to end, including the five subcommands that did not exist.

`recall/enterprise_cli.py` had no test of any kind, and three pieces of machinery had no operator
entry point at all: `replay_pending` (whose only reference in the whole repository was its own
definition), `validate_generation_parity` and `check_enterprise_readiness`. A recovery mechanism
that cannot be invoked is not a recovery mechanism, and `cutover` refuses forever while an event
is pending, so the gap was a deadlock with a documented workaround nobody could run.

Every command is driven through `main()` and `sys.argv`, not by calling the private helpers, so
the argument parsing and the exit codes are part of what is under test. An operator reads the exit
code.
"""
from __future__ import annotations

import uuid

import psycopg
import pytest

from recall.control_plane import ControlPlane
from recall.embeddings import HashingEmbedder
from recall.enterprise_cli import main
from recall.schema import apply_migrations
from recall.store import PgVectorStore
from recall.types import Chunk
from tests.conftest import requires_db

pytestmark = requires_db

DIM = 32


@pytest.fixture
def cli(unprivileged_dsn: str, monkeypatch):
    """A tenant with two registered generations, driven through the real CLI entry point."""
    monkeypatch.setenv("RECALL_DSN", unprivileged_dsn)
    suffix = uuid.uuid4().hex[:10]

    class _Env:
        dsn = unprivileged_dsn
        tenant = f"t_{suffix}"
        active_id = f"g_active_{suffix}"
        shadow_id = f"g_shadow_{suffix}"
        active_table = f"c_active_{suffix}"
        shadow_table = f"c_shadow_{suffix}"
        control = ControlPlane(unprivileged_dsn)

        def run(self, *argv: str) -> int:
            """Run one subcommand; return its exit status the way a shell would see it.

            `SystemExit` carrying a STRING is CPython's "print this to stderr and exit 1", which
            is how `raise SystemExit("unknown generation: ...")` behaves for a real operator.
            Reading `.code` as an integer would have scored those as exit 0, so three refusal
            tests would have passed against a CLI that reported success.
            """
            monkeypatch.setattr("sys.argv", ["recall-enterprise", *argv])
            try:
                main()
            except SystemExit as exit_status:
                code = exit_status.code
                if code is None:
                    return 0
                return code if isinstance(code, int) else 1
            return 0

        def store(self, generation_id: str, table: str) -> PgVectorStore:
            return PgVectorStore(
                self.dsn, dim=DIM, table=table, tenant=self.tenant, generation_id=generation_id
            )

    env = _Env()
    env.control.apply_migrations()
    for table in (env.active_table, env.shadow_table):
        apply_migrations(unprivileged_dsn, table=table, dim=DIM)
    env.control.register_generation(env.active_id, env.active_table, "profile-a", DIM)
    env.control.register_generation(env.shadow_id, env.shadow_table, "profile-a", DIM)
    env.control.set_generation_state(env.active_id, "ready", chunk_count=0, source_count=0)
    try:
        yield env
    finally:
        with psycopg.connect(unprivileged_dsn, autocommit=True) as conn:
            conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (env.tenant,))
            conn.execute("DELETE FROM recall_migration_events WHERE tenant_id = %s", (env.tenant,))
            conn.execute("DELETE FROM recall_tenant_routes WHERE tenant_id = %s", (env.tenant,))
            conn.execute(
                "DELETE FROM recall_index_generations WHERE generation_id = ANY(%s)",
                ([env.active_id, env.shadow_id],),
            )
            for table in (env.active_table, env.shadow_table):
                conn.execute(f"DROP TABLE IF EXISTS {table}")
                conn.execute(
                    "DELETE FROM recall_schema_migrations WHERE target_table = %s", (table,)
                )


def _pending_index_event(cli, sources: list[str]) -> str:
    embedder = HashingEmbedder(dim=DIM)
    chunks = [
        Chunk(id=f"{s}#0", source=s, text=f"contents of {s}", metadata={}) for s in sources
    ]
    records = [
        {"id": c.id, "source": c.source, "text": c.text, "metadata": c.metadata,
         "embedding": embedder.embed([c.text])[0]}
        for c in chunks
    ]
    operation_id = f"op-{uuid.uuid4().hex[:8]}"
    cli.control.append_event(
        cli.tenant, operation_id, "index",
        {
            "active_generation": cli.active_id,
            "shadow_generation": cli.shadow_id,
            "sources": sources,
            "active_chunks": records,
            "chunks": records,
        },
        active_count=len(chunks),
    )
    return operation_id


# --------------------------------------------------------------------------------------------
# The parser surface
# --------------------------------------------------------------------------------------------


def test_every_documented_subcommand_is_reachable() -> None:
    from recall.enterprise_cli import _parser

    choices = set(_parser()._subparsers._group_actions[0].choices)
    assert {
        "migrate", "create-generation", "mark-ready", "set-route", "cutover",
        "replay", "parity", "readiness", "status", "retire",
    } <= choices


def test_the_dsn_is_chosen_by_command_class(monkeypatch) -> None:
    """A read-only subcommand must not run on the DDL credential, and vice versa.

    `readiness` reports whether row level security constrains "the runtime database role", but
    `check_rls_effective` reads `current_user` of the connection it was handed. Run on the
    migration role, a green verdict certifies a credential that may never serve a request. Every
    subcommand previously read one `RECALL_DSN`, which this repository defines two ways.
    """
    from recall.enterprise_cli import _SERVING_COMMANDS, _dsn

    monkeypatch.setenv("RECALL_SERVING_DSN", "postgresql://serving@localhost/recall")
    monkeypatch.setenv("RECALL_MIGRATION_DSN", "postgresql://migrating@localhost/recall")
    monkeypatch.delenv("RECALL_DSN", raising=False)

    for command in ("readiness", "status", "parity", "replay"):
        assert "serving@" in _dsn(command), command
    for command in ("migrate", "create-generation", "mark-ready", "set-route", "cutover", "retire"):
        assert "migrating@" in _dsn(command), command

    # The classification is not hand-maintained twice: every parser subcommand is in exactly one
    # class, so a new one cannot silently default to the wrong credential unnoticed.
    from recall.enterprise_cli import _parser

    choices = set(_parser()._subparsers._group_actions[0].choices)
    assert _SERVING_COMMANDS <= choices, _SERVING_COMMANDS - choices

    # RECALL_DSN still works for single-variable deployments, both ways.
    monkeypatch.delenv("RECALL_SERVING_DSN")
    monkeypatch.delenv("RECALL_MIGRATION_DSN")
    monkeypatch.setenv("RECALL_DSN", "postgresql://legacy@localhost/recall")
    assert "legacy@" in _dsn("readiness")
    assert "legacy@" in _dsn("migrate")

    monkeypatch.delenv("RECALL_DSN")
    with pytest.raises(SystemExit, match="RECALL_SERVING_DSN"):
        _dsn("readiness")
    with pytest.raises(SystemExit, match="RECALL_MIGRATION_DSN"):
        _dsn("migrate")


def test_status_names_an_unusable_registry_row_instead_of_dying_on_it(cli, capsys) -> None:
    """The diagnostic must survive the thing it diagnoses.

    The identifier allowlist validates on the way OUT, so a row written under the older
    `str.isidentifier()` rule raises when read. Letting that escape made `status`, the one command
    an operator would use to FIND the bad row, the command that crashes on it.
    """
    import json

    bad = f"BadTable_{uuid.uuid4().hex[:6]}"  # uppercase: legal before, rejected now
    with psycopg.connect(cli.dsn, autocommit=True) as conn:
        conn.execute(
            "INSERT INTO recall_index_generations "
            "(generation_id, physical_table, embedding_profile, dimension, state) "
            "VALUES (%s, %s, 'profile-a', %s, 'ready')",
            (f"g_legacy_{uuid.uuid4().hex[:8]}", bad, DIM),
        )
    try:
        assert cli.run("status", "--json") == 0, "status died on a row it is meant to report"
        report = json.loads(capsys.readouterr().out)
        offenders = [r["physical_table"] for r in report["invalid_generations"]]
        assert bad in offenders, report["invalid_generations"]
        # And the valid rows are still listed, so the bad row does not hide the good ones.
        assert cli.active_id in {g["generation_id"] for g in report["generations"]}
    finally:
        with psycopg.connect(cli.dsn, autocommit=True) as conn:
            conn.execute(
                "DELETE FROM recall_index_generations WHERE physical_table = %s", (bad,)
            )


def test_no_subcommand_accepts_a_physical_table_except_create_generation() -> None:
    """An operator names a GENERATION; the registry names the table.

    `create-generation` is the one place a table name legitimately enters, because that is where
    it is registered. Every later command resolves it from `recall_index_generations`. A new
    subcommand that took a `--table` would reintroduce exactly the client-supplied identifier the
    security boundary forbids, and would do it in a place nobody would think to look.
    """
    from recall.enterprise_cli import _parser

    subparsers = _parser()._subparsers._group_actions[0].choices
    for name, parser in subparsers.items():
        if name == "create-generation":
            continue
        options = {
            option for action in parser._actions for option in (action.option_strings or [action.dest])
        }
        assert not {"table", "--table", "physical_table", "--physical-table"} & options, (
            f"subcommand {name!r} accepts a physical table name from the operator"
        )


# --------------------------------------------------------------------------------------------
# replay
# --------------------------------------------------------------------------------------------


def test_replay_drains_the_outbox_and_unblocks_cutover(cli) -> None:
    """The deadlock, and its drain, in one test."""
    assert cli.run("set-route", cli.tenant, cli.active_id, "--shadow-generation", cli.shadow_id) == 0
    cli.run("mark-ready", cli.shadow_id, "--chunks", "1", "--sources", "1")
    _pending_index_event(cli, ["/corpus/a.md", "/corpus/b.md"])

    # Before the drain, cutover is refused. This is the state a crash leaves behind.
    with pytest.raises(RuntimeError, match="pending"):
        cli.run("cutover", cli.tenant)

    assert cli.run("replay", cli.tenant) == 0
    assert cli.control.pending_events(cli.tenant) == []

    active = cli.store(cli.active_id, cli.active_table)
    shadow = cli.store(cli.shadow_id, cli.shadow_table)
    try:
        assert active.count() == 2
        assert shadow.count() == 2
    finally:
        active.close()
        shadow.close()

    assert cli.run("cutover", cli.tenant) == 0
    route = cli.control.route(cli.tenant)
    assert route is not None and route.active.generation_id == cli.shadow_id


def test_replay_on_an_empty_outbox_is_a_success_not_an_error(cli) -> None:
    """An operator runs this during an incident. "Nothing to do" must not read as failure."""
    assert cli.run("replay", cli.tenant) == 0


def test_replay_refuses_a_generation_that_is_not_in_the_registry(cli) -> None:
    """The event names a generation; the CLI will not invent a table for it."""
    cli.control.append_event(
        cli.tenant, "op-orphan", "index",
        {
            "active_generation": cli.active_id,
            "shadow_generation": "g_never_registered",
            "sources": ["/corpus/a.md"],
            "active_chunks": [],
            "chunks": [],
        },
    )
    assert cli.run("replay", cli.tenant) != 0
    assert len(cli.control.pending_events(cli.tenant)) == 1, "a refused replay consumed the event"


# --------------------------------------------------------------------------------------------
# parity
# --------------------------------------------------------------------------------------------


def test_parity_passes_on_matching_generations_and_fails_on_a_gap(cli) -> None:
    embedder = HashingEmbedder(dim=DIM)
    chunks = [
        Chunk(id="a#0", source="/corpus/a.md", text="alpha", metadata={}),
        Chunk(id="b#0", source="/corpus/b.md", text="beta", metadata={}),
    ]
    vectors = [embedder.embed([c.text])[0] for c in chunks]
    cli.run("set-route", cli.tenant, cli.active_id, "--shadow-generation", cli.shadow_id)

    active = cli.store(cli.active_id, cli.active_table)
    shadow = cli.store(cli.shadow_id, cli.shadow_table)
    try:
        active.upsert(chunks, vectors)
        shadow.upsert(chunks, vectors)
        assert cli.run("parity", cli.tenant) == 0

        # Remove one source from the shadow only: the generations now disagree.
        shadow.delete_sources(["/corpus/b.md"])
        assert cli.run("parity", cli.tenant) == 1
    finally:
        active.close()
        shadow.close()


def test_parity_without_a_shadow_generation_says_so(cli) -> None:
    cli.run("set-route", cli.tenant, cli.active_id)
    assert cli.run("parity", cli.tenant) != 0


def test_parity_refuses_two_empty_generations_instead_of_passing_vacuously(cli, capsys) -> None:
    """The misleading GREEN at the parity step of the cutover sequence.

    Two empty generations cannot disagree, so every comparison `validate_generation_parity` makes
    is vacuously satisfied and `GenerationParity.valid` is True. Before this guard the command
    printed "parity: OK" and exited 0 over 0 active and 0 shadow chunks, which is exactly the
    state the reference deployment was in -- and "parity: OK" is what an operator reads as
    permission to run cutover.

    Two limits on what this guard is, both of which an earlier version of this docstring asserted
    away. It does NOT stand between an operator and a promoted empty index: `ControlPlane.cutover`
    calls `_require_non_empty_shadow` before its own parity check, deliberately outside the
    `--allow-divergent-corpus` branch, so an empty shadow was never promotable. And emptiness is
    not the only vacuous green here: `source_raw_hashes()` coalesces an absent content hash to the
    empty string, so two generations whose rows all lack one compare equal while certifying
    nothing. That mode is gated in `benchmarks/check_generation_parity.py`, not here.

    This asserts the REASON, not just the exit code: a non-zero that arrived because the stores
    failed to open, or because RLS was not forced, would satisfy a bare `!= 0` while leaving the
    vacuous-green hole open.
    """
    cli.run("set-route", cli.tenant, cli.active_id, "--shadow-generation", cli.shadow_id)

    active = cli.store(cli.active_id, cli.active_table)
    shadow = cli.store(cli.shadow_id, cli.shadow_table)
    try:
        assert active.count() == 0 and shadow.count() == 0, "fixture is not the empty state"
        assert cli.run("parity", cli.tenant) == 1
    finally:
        active.close()
        shadow.close()

    captured = capsys.readouterr()
    assert "vacuous" in captured.err, captured.err
    assert "parity: OK" not in captured.out, "the vacuous green survived"


def test_parity_reports_a_missing_shadow_as_missing_sources_not_as_vacuity(cli, capsys) -> None:
    """The discriminating population, and the one the emptiness guard must NOT claim.

    `_cmd_parity`'s emptiness refusal is scoped to BOTH generations being empty. A populated
    active against an empty shadow is a real divergence and must keep failing on missing sources,
    with that reason, so an operator is told what is actually wrong.

    Without this test the scoping is asserted only in a comment: every other parity test has
    either both sides populated or both sides empty, so mutating the guard's `and` to `or`
    leaves the whole suite green while turning a specific, actionable failure into
    "the comparison is vacuous". Verified by running that mutation — it survives without this
    test and is caught by it.
    """
    embedder = HashingEmbedder(dim=DIM)
    chunks = [Chunk(id="a#0", source="/corpus/a.md", text="alpha", metadata={})]
    vectors = [embedder.embed([c.text])[0] for c in chunks]
    cli.run("set-route", cli.tenant, cli.active_id, "--shadow-generation", cli.shadow_id)

    active = cli.store(cli.active_id, cli.active_table)
    shadow = cli.store(cli.shadow_id, cli.shadow_table)
    try:
        active.upsert(chunks, vectors)
        assert active.count() > 0 and shadow.count() == 0, "fixture is not the asymmetric state"
        assert cli.run("parity", cli.tenant) == 1
    finally:
        active.close()
        shadow.close()

    captured = capsys.readouterr()
    assert "missing sources" in captured.err, captured.err
    assert "vacuous" not in captured.err, "a real divergence was restated as a vacuity error"


def test_parity_still_reports_catalog_failures_when_both_generations_are_empty(
    cli, capsys, monkeypatch
) -> None:
    """`indexes_valid` and `rls_enabled` are catalog facts, so they can be false on an EMPTY pair.

    An earlier version of the emptiness refusal returned before the failure-printing loop, which
    made this command silent about a shadow whose row-level security was not forced — and
    `readiness` evaluates `route.active` only, so this is the sole step that inspects the
    shadow's RLS at all. The refusal must not suppress the reason.

    The parity result is substituted rather than provoked: forcing a real catalog into that state
    would mean dropping RLS on a live table mid-test, and what is under test here is the CLI's
    reporting order, not the validator's derivation.
    """
    import recall.enterprise_cli as ecli
    from recall.migration import GenerationParity

    monkeypatch.setattr(
        ecli,
        "validate_generation_parity",
        lambda active, shadow: GenerationParity(
            valid=False,
            active_chunks=0,
            shadow_chunks=0,
            missing_sources=(),
            extra_sources=(),
            hash_mismatches=(),
            failures=("row level security is not forced on both generations",),
        ),
    )
    cli.run("set-route", cli.tenant, cli.active_id, "--shadow-generation", cli.shadow_id)
    assert cli.run("parity", cli.tenant) == 1

    captured = capsys.readouterr()
    assert "row level security is not forced" in captured.err, captured.err
    assert "vacuous" in captured.err, "the emptiness refusal itself must still fire"


def test_parity_still_passes_when_both_generations_hold_the_same_chunks(cli) -> None:
    """The negative control for the guard above.

    A guard that refuses an empty pair could be satisfied by refusing everything. This is the
    same populated-and-matching case `test_parity_passes_on_matching_generations_and_fails_on_a_gap`
    covers, asserted again here so that the emptiness check cannot be broadened into a blanket
    refusal without a test going red.
    """
    embedder = HashingEmbedder(dim=DIM)
    chunks = [Chunk(id="a#0", source="/corpus/a.md", text="alpha", metadata={})]
    vectors = [embedder.embed([c.text])[0] for c in chunks]
    cli.run("set-route", cli.tenant, cli.active_id, "--shadow-generation", cli.shadow_id)

    active = cli.store(cli.active_id, cli.active_table)
    shadow = cli.store(cli.shadow_id, cli.shadow_table)
    try:
        active.upsert(chunks, vectors)
        shadow.upsert(chunks, vectors)
        assert cli.run("parity", cli.tenant) == 0
    finally:
        active.close()
        shadow.close()


# --------------------------------------------------------------------------------------------
# readiness
# --------------------------------------------------------------------------------------------


def test_readiness_reports_ready_and_returns_zero(cli, monkeypatch) -> None:
    """Wiring, not the checks themselves: `tests/test_enterprise_readiness.py` owns those.

    The embedder is substituted because constructing the real profile needs the `fastembed` extra
    and downloads-at-runtime are on this program's standing out-of-scope list. What is being
    tested here is that the subcommand resolves the tenant's active generation, opens ITS table,
    and turns the result into an exit code.
    """
    import recall_mcp.service as service

    from recall.embeddings import EmbeddingProfile

    profile = EmbeddingProfile(
        profile_id="profile-a", model_name="test-model", artifact_digest="a" * 64,
        dimension=DIM, query_mode="query_embed", passage_mode="passage_embed",
    )

    class _Embedder:
        def __init__(self) -> None:
            self.profile = profile
            self.dim = DIM

    monkeypatch.setattr(service, "make_profile_embedder", lambda *_a, **_k: _Embedder())
    cli.run("set-route", cli.tenant, cli.active_id)
    assert cli.run("readiness", cli.tenant) == 0


def test_readiness_returns_non_zero_when_the_generation_profile_disagrees(cli, monkeypatch):
    import recall_mcp.service as service

    from recall.embeddings import EmbeddingProfile

    profile = EmbeddingProfile(
        profile_id="a-completely-different-profile", model_name="test-model",
        artifact_digest="a" * 64, dimension=DIM,
        query_mode="query_embed", passage_mode="passage_embed",
    )

    class _Embedder:
        def __init__(self) -> None:
            self.profile = profile
            self.dim = DIM

    monkeypatch.setattr(service, "make_profile_embedder", lambda *_a, **_k: _Embedder())
    cli.run("set-route", cli.tenant, cli.active_id)
    assert cli.run("readiness", cli.tenant) == 1


def test_readiness_without_a_route_says_so(cli) -> None:
    assert cli.run("readiness", cli.tenant) != 0


# --------------------------------------------------------------------------------------------
# status
# --------------------------------------------------------------------------------------------


def test_status_reports_generations_routes_and_outbox_depth(cli, capsys) -> None:
    cli.run("set-route", cli.tenant, cli.active_id, "--shadow-generation", cli.shadow_id)
    _pending_index_event(cli, ["/corpus/a.md"])

    assert cli.run("status", "--tenant", cli.tenant, "--json") == 0
    import json

    report = json.loads(capsys.readouterr().out)
    ids = {g["generation_id"] for g in report["generations"]}
    assert {cli.active_id, cli.shadow_id} <= ids
    assert report["route"]["active"] == cli.active_id
    assert report["route"]["shadow"] == cli.shadow_id
    assert len(report["pending_events"]) == 1
    assert report["control_plane_ledger"] == "control plane ledger is current"


def test_status_never_prints_a_pending_events_payload(cli, capsys) -> None:
    """A status command is not a retrieval path.

    The payload of a pending event holds the full text and vectors of every chunk in the batch.
    Printing it would put corpus content on an operator's terminal and into whatever collects
    that terminal's output, for a tenant whose data the operator may have no right to read.
    """
    secret = "/corpus/private-memo.md"
    cli.run("set-route", cli.tenant, cli.active_id, "--shadow-generation", cli.shadow_id)
    _pending_index_event(cli, [secret])

    cli.run("status", "--tenant", cli.tenant, "--json")
    as_json = capsys.readouterr().out
    cli.run("status", "--tenant", cli.tenant)
    as_text = capsys.readouterr().out

    for rendering, label in ((as_json, "json"), (as_text, "text")):
        assert "contents of" not in rendering, f"{label} status printed corpus text"
        assert secret not in rendering, f"{label} status printed a source path"
        # The payload key, not the substring: `embedding_profile` is a legitimate generation
        # field and matching on `embedding` alone made this assertion fire on the good case.
        assert '"embedding"' not in rendering, f"{label} status printed vectors"
        assert '"active_chunks"' not in rendering, f"{label} status printed the replay payload"
        assert '"payload"' not in rendering, f"{label} status printed the replay payload"


# --------------------------------------------------------------------------------------------
# retire
# --------------------------------------------------------------------------------------------


def test_retire_refuses_a_routed_generation_and_accepts_an_unrouted_one(cli) -> None:
    cli.run("set-route", cli.tenant, cli.active_id, "--shadow-generation", cli.shadow_id)
    with pytest.raises(RuntimeError, match="active generation"):
        cli.run("retire", cli.active_id, "--tenant", cli.tenant)

    # Drop the shadow from the route, then retirement is allowed.
    cli.run("set-route", cli.tenant, cli.active_id)
    assert cli.run("retire", cli.shadow_id, "--tenant", cli.tenant) == 0
    retired = cli.control.generation(cli.shadow_id)
    assert retired is not None and retired.state == "retired"


def test_retire_reports_an_unknown_generation(cli) -> None:
    with pytest.raises(KeyError):
        cli.run("retire", "g_does_not_exist", "--tenant", cli.tenant)
