"""An erasure is authorized on every name the rows it removes are authorized by.

A row is authorized by its root-relative `file`, or by its `source` when it has none (the prefix
arm of `recall.scope.Scope.sql`, and the name the indexer decided when it wrote the row). Forget
and withdraw resolve a requested name to every source whose `file` OR `source` it matches, so
deciding only the requested name let a caller allowed to erase one spelling of a memo (its
absolute path, which is what the CLI and the lite memo tools pass) erase rows the policy denies
that caller under their `file`. A withdrawal also reaches memos forgotten earlier through their
`SupersessionRecord`, which keeps the same two names.

Invariants and the failure each one catches:
- A1 `forget_memory` refuses a source whose `file` the policy denies, though the requested
  absolute path is allowed, and deletes nothing.
- A2 `withdraw_memory` of a memo forgotten earlier is refused when its record's `file` is denied,
  so what the forget kept retired stays retired.
- A3 the check refuses nothing it should allow: a source whose every name is allowed is erased.
- A4 `recall forget` on a lite DSN refuses the same source, before its dry run.
- A5 `recall withdraw` on a lite DSN refuses a forgotten memo by its record.
- A6 PostgreSQL's `PgVectorStore` refuses as the lite store does.
- A7 `GenerationStore` decides the adopted v0.8 rows by their `file` too.
- A8 `GenerationStore.erasure_subjects` reads the generation table's `file`.
- A9 `recall withdraw` on PostgreSQL refuses a forgotten memo by its record.

Red proof: on master `91bad3448` (before this change) each test fails in its intended assertion,
named in its docstring: the erasure goes through, or, for A8, `erasure_subjects` does not exist
(A8 is therefore also proved by mutation). A3 is the deliberate control and passes on master.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import psycopg
import pytest

from recall.generation_store import GenerationStore
from recall.lite import LiteStore
from recall.security_policy import AccessContext, SourceRule, SourceSecurityPolicy
from recall.types import Chunk
from recall_mcp.lifecycle import forget_memory, withdraw_memory
from tests.conftest import TEST_DSN, requires_db
from tests.lite_cli_support import _offline, run_cli  # noqa: F401  # _offline: autouse fixture
from tests.test_generations import (  # noqa: F401  # manager: fixture
    _Embedder,
    _manifest,
    _pipeline,
    _reader,
    _ready,
    manager,
)

HASHING_DIM = 64
ERASURE = frozenset({"erasure"})
MEMOS = {
    "notes/old.md": "# Pay\n\nThe old salary band.\n",
    "notes/a.md": "# Note\n\nA public note.\n",
    "hr/salaries.md": "---\nsupersedes: notes/old.md\n---\n# Pay\n\nThe new salary bands.\n",
}


def _folder(root: Path) -> Path:
    for name, text in MEMOS.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def _policy(allowed_prefix: str) -> SourceSecurityPolicy:
    """Anyone may erase under `allowed_prefix` and under `notes`; only `hr-admin` under `hr`."""
    return SourceSecurityPolicy(
        (
            SourceRule(allowed_prefix, purposes=ERASURE),
            SourceRule("notes", purposes=ERASURE),
            SourceRule("hr", purposes=ERASURE, principals=frozenset({"hr-admin"})),
        )
    )


def _context(principal: str, tenant: str = "default") -> AccessContext:
    return AccessContext(principal, tenant, purpose="erasure")


def _lite(tmp_path: Path) -> tuple[Path, str]:
    folder = _folder(tmp_path / "mem")
    dsn = "sqlite:///" + (tmp_path / "store" / "memory.db").as_posix()
    run_cli(["--embedder", "hashing", "--dsn", dsn, "index", str(folder)])
    return folder, dsn


def _open(tmp_path: Path) -> LiteStore:
    return LiteStore(tmp_path / "store" / "memory.db", dim=HASHING_DIM)


def _write_policy(tmp_path: Path, allowed_prefix: str) -> Path:
    # Every list spelled out: the loader rejects an absent one, whose default is a tuple.
    rules = [
        {"prefix": prefix, "purposes": ["erasure"], "principals": principals, "redactions": []}
        for prefix, principals in ((allowed_prefix, []), ("notes", []), ("hr", ["hr-admin"]))
    ]
    path = tmp_path / "policy.json"
    path.write_text(json.dumps({"default_deny": True, "rules": rules}), encoding="utf-8")
    return path


def test_a1_forget_refuses_a_source_whose_file_is_denied(tmp_path: Path) -> None:
    """Node tests/test_erasure_authorization.py::test_a1_forget_refuses_a_source_whose_file_is_denied.

    Red on master `91bad3448`: `DID NOT RAISE PermissionError`; the forget deleted hr/salaries.md.
    """
    folder, _ = _lite(tmp_path)
    salaries = str((folder / "hr" / "salaries.md").resolve())
    with _open(tmp_path) as store:
        with pytest.raises(PermissionError, match="hr/salaries.md"):
            forget_memory(
                store, [salaries], security_policy=_policy(str(folder)), security_context=_context("alice")
            )
        assert store.sources_for_identifiers([salaries]), "the refused forget deleted the rows"


def test_a2_withdraw_of_a_forgotten_memo_refuses_a_denied_record(tmp_path: Path) -> None:
    """Node tests/test_erasure_authorization.py::test_a2_withdraw_of_a_forgotten_memo_refuses_a_denied_record.

    Red on master `91bad3448`: `DID NOT RAISE PermissionError`; alice's withdrawal gave notes/old.md
    back although she may not erase hr/salaries.md.
    """
    folder, _ = _lite(tmp_path)
    salaries = str((folder / "hr" / "salaries.md").resolve())
    policy = _policy(str(folder))
    with _open(tmp_path) as store:
        assert store.supersession_graph().successor("notes/old.md") == "hr/salaries.md"
        forget_memory(store, [salaries], security_policy=policy, security_context=_context("hr-admin"))
        assert store.supersession_graph().successor("notes/old.md") == "hr/salaries.md"
        with pytest.raises(PermissionError, match="hr/salaries.md"):
            withdraw_memory(store, [salaries], security_policy=policy, security_context=_context("alice"))
        assert store.supersession_graph().successor("notes/old.md") == "hr/salaries.md", (
            "the refused withdrawal gave notes/old.md back"
        )


def test_a3_a_source_whose_every_name_is_allowed_is_erased(tmp_path: Path) -> None:
    """Node tests/test_erasure_authorization.py::test_a3_a_source_whose_every_name_is_allowed_is_erased.

    Control, not a red proof: it passes on master too, and guards the other direction, a check
    that refuses what the policy allows. Red by mutation: `_authorize_erasure` raising whenever
    `subjects` is non-empty fails it with `PermissionError: source 'notes/a.md'`.
    """
    folder, _ = _lite(tmp_path)
    note = str((folder / "notes" / "a.md").resolve())
    with _open(tmp_path) as store:
        result = forget_memory(
            store, [note], security_policy=_policy(str(folder)), security_context=_context("alice")
        )
        assert result.sources_removed == [note]
        assert not store.sources_for_identifiers([note])


def test_a4_cli_forget_refuses_a_source_whose_file_is_denied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Node tests/test_erasure_authorization.py::test_a4_cli_forget_refuses_a_source_whose_file_is_denied.

    Red on master `91bad3448`: the "source authorization denied for hr/salaries.md" assertion; the
    dry run listed the source as one it would forget.
    """
    folder, dsn = _lite(tmp_path)
    salaries = str((folder / "hr" / "salaries.md").resolve())
    monkeypatch.setenv("RECALL_SOURCE_POLICY_FILE", str(_write_policy(tmp_path, str(folder))))
    monkeypatch.setenv("RECALL_PRINCIPAL", "alice")
    base = ["--embedder", "hashing", "--dsn", dsn]
    for argv in ([*base, "forget", salaries], [*base, "forget", salaries, "--yes"]):
        out = run_cli(argv)
        assert "source authorization denied for hr/salaries.md" in out, out
    with _open(tmp_path) as store:
        assert store.sources_for_identifiers([salaries]), "the refused forget deleted the rows"


def test_a5_cli_withdraw_refuses_a_forgotten_memo_by_its_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Node tests/test_erasure_authorization.py::test_a5_cli_withdraw_refuses_a_forgotten_memo_by_its_record.

    Red on master `91bad3448`: the "source authorization denied for hr/salaries.md" assertion;
    alice's withdrawal ran and gave notes/old.md back.
    """
    folder, dsn = _lite(tmp_path)
    salaries = str((folder / "hr" / "salaries.md").resolve())
    monkeypatch.setenv("RECALL_SOURCE_POLICY_FILE", str(_write_policy(tmp_path, str(folder))))
    base = ["--embedder", "hashing", "--dsn", dsn]
    monkeypatch.setenv("RECALL_PRINCIPAL", "hr-admin")
    forgot = run_cli([*base, "forget", salaries, "--yes"])
    assert "EXIT" not in forgot, forgot
    monkeypatch.setenv("RECALL_PRINCIPAL", "alice")
    out = run_cli([*base, "withdraw", salaries, "--yes"])
    assert "source authorization denied for hr/salaries.md" in out, out
    with _open(tmp_path) as store:
        assert store.supersession_graph().successor("notes/old.md") == "hr/salaries.md"


@requires_db
def test_a6_postgres_forget_refuses_a_source_whose_file_is_denied(make_store) -> None:
    """Node tests/test_erasure_authorization.py::test_a6_postgres_forget_refuses_a_source_whose_file_is_denied.

    Red on master `91bad3448`: `DID NOT RAISE PermissionError`; the forget deleted the row.
    """
    store = make_store(3)
    source = "/data/import/hr/salaries.md"
    store.upsert([Chunk("c1", source, "the new salary bands", {"file": "hr/salaries.md"})], [[1.0, 0.0, 0.0]])
    with pytest.raises(PermissionError, match="hr/salaries.md"):
        forget_memory(
            store, [source], security_policy=_policy("data"), security_context=_context("alice", store.tenant)
        )
    assert store.count() == 1, "the refused forget deleted the row"


@requires_db
def test_a7_generation_store_decides_adopted_rows_by_their_file(manager) -> None:  # noqa: F811
    """Node tests/test_erasure_authorization.py::test_a7_generation_store_decides_adopted_rows_by_their_file.

    Red on master `91bad3448`: `DID NOT RAISE PermissionError`; the forget erased the adopted rows.
    """
    tenant = manager.tenant_id
    uri = "/data/import/hr/salaries.md"
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (tenant,))
        conn.execute(
            "INSERT INTO chunks (tenant_id, id, source, text, embedding, metadata) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (tenant, f"legacy-{uuid.uuid4().hex[:8]}", uri, "adopted body",
             "[" + ",".join(["0.0"] * 63) + ",1.0]", json.dumps({"file": "hr/salaries.md"})),
        )
    store = GenerationStore(TEST_DSN, 64, tenant=tenant)
    try:
        with pytest.raises(PermissionError, match="hr/salaries.md"):
            forget_memory(store, [uri], security_policy=_policy("data"), security_context=_context("alice", tenant))
    finally:
        store.close()
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (tenant,))
        surviving = conn.execute(
            "SELECT count(*) FROM chunks WHERE tenant_id = %s AND source = %s", (tenant, uri)
        ).fetchone()[0]
    assert surviving == 1, "the refused forget erased the adopted rows"


@requires_db
def test_a8_generation_store_reads_the_generation_tables_file(manager) -> None:  # noqa: F811
    """Node tests/test_erasure_authorization.py::test_a8_generation_store_reads_the_generation_tables_file.

    Red on master `91bad3448`: `AttributeError: 'GenerationStore' object has no attribute
    'erasure_subjects'`, which is not a behavioural red, so also proved by mutation: with the
    `recall_chunks_v1` query of `GenerationStore._erasure_subject_rows` returning no rows, it fails
    on the equality (`{}` against the built row's name).
    """
    data = b"---\nstatus: current\n---\na built memo"
    manifest = _manifest(manager.tenant_id, data)
    uri = manifest.objects[0].uri
    _ready(manager, manifest, _pipeline("model-a"), _reader(manifest, data), _Embedder(1))
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (manager.tenant_id,))
        names = {
            str(row[0])
            for row in conn.execute(
                "SELECT DISTINCT COALESCE(NULLIF(metadata->>'file', ''), source_uri) "
                "FROM recall_chunks_v1 WHERE tenant_id = %s AND source_uri = %s",
                (manager.tenant_id, uri),
            ).fetchall()
        }
    assert names, "the build wrote no row to decide"
    store = GenerationStore(TEST_DSN, 64, tenant=manager.tenant_id)
    try:
        assert store.erasure_subjects([uri]) == {uri: names}
    finally:
        store.close()


@requires_db
def test_a9_postgres_cli_withdraw_refuses_a_forgotten_memo_by_its_record(
    tmp_path: Path, cli_table: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Node tests/test_erasure_authorization.py::test_a9_postgres_cli_withdraw_refuses_a_forgotten_memo_by_its_record.

    Red on master `91bad3448`: the "source authorization denied for hr/salaries.md" assertion; the
    withdrawal ran.
    """
    folder = _folder(tmp_path / "mem")
    salaries = str((folder / "hr" / "salaries.md").resolve())
    base = ["--embedder", "hashing", "--dsn", TEST_DSN, "--table", cli_table]
    run_cli([*base, "index", str(folder)])
    monkeypatch.setenv("RECALL_SOURCE_POLICY_FILE", str(_write_policy(tmp_path, str(folder))))
    monkeypatch.setenv("RECALL_PRINCIPAL", "hr-admin")
    forgot = run_cli([*base, "forget", salaries, "--yes"])
    assert "EXIT" not in forgot, forgot
    monkeypatch.setenv("RECALL_PRINCIPAL", "alice")
    out = run_cli([*base, "withdraw", salaries, "--yes"])
    assert "source authorization denied for hr/salaries.md" in out, out
