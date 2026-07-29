"""The runner: ingest once per distinct corpus state, and let invariants stop a bad run early.

Prior work: [[project-recall-abstention-bounded-domain-2026-07-24]] — six abstention signals
measured, all failed; this measures the axis they were measured on, not a seventh signal.

These tests use a fake MemorySystem on purpose. The Postgres-backed adapter is exercised by the
real run; what needs pinning here is the runner's own logic, which is where a silent defect would
cost a whole overnight job.
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import pytest

from benchmarks.ladder.adapter import Document, Response
from benchmarks.ladder.invariants import InvariantViolation
from benchmarks.ladder.manifest import (
    LABEL_ANSWERABLE,
    LABEL_UNANSWERABLE,
    MANIFEST_VERSION_V1,
    RING_ORIGINAL,
    Instance,
    read_manifest,
    write_manifest,
)
from benchmarks.ladder.run import AdapterSmokeCheckFailed, main, run

DOCS = {f"c/D1:{i}": f"turn {i} about the support group" for i in range(1, 5)}


class _Fake:
    name = "fake"

    def __init__(self, *, leak: bool = False) -> None:
        self._docs: dict[str, str] = {}
        self.ingest_calls = 0
        self._leak = leak

    def ingest(self, docs: Iterable[Document]) -> None:
        self.ingest_calls += 1
        incoming = {d.doc_id: d.text for d in docs}
        # A leaking system MERGES instead of replacing — exactly what invariant 1 exists to catch.
        self._docs = {**self._docs, **incoming} if self._leak else incoming

    def indexed_doc_ids(self) -> frozenset[str]:
        return frozenset(self._docs)

    def query(self, question: str) -> Response:
        if not self._docs:
            return Response(answer=None)
        first = sorted(self._docs)[0]
        return Response(answer=self._docs[first], cited_ids=(first,), tokens=10)


def _manifest(tmp_path: Path) -> Path:
    instances = [
        Instance(
            instance_id="p1#original", corpus="locomo", source_question_id="q1",
            question="when?", label=LABEL_ANSWERABLE, ring=RING_ORIGINAL,
            excised_doc_ids=(), gold_doc_ids=("c/D1:1",), pair_id="p1",
        ),
        Instance(
            instance_id="p1#d0", corpus="locomo", source_question_id="q1",
            question="when?", label=LABEL_UNANSWERABLE, ring=0,
            excised_doc_ids=("c/D1:1",), gold_doc_ids=("c/D1:1",), pair_id="p1",
        ),
        Instance(
            instance_id="p2#d0", corpus="locomo", source_question_id="q2",
            question="who?", label=LABEL_UNANSWERABLE, ring=0,
            excised_doc_ids=("c/D1:1",), gold_doc_ids=("c/D1:1",), pair_id="p2",
        ),
    ]
    path = tmp_path / "manifest.jsonl"
    write_manifest(
        path, instances, ring_widths=[0], corpus_hashes={"locomo": "x"},
        manifest_version=MANIFEST_VERSION_V1,
    )
    return path


def test_writes_one_row_per_instance(tmp_path: Path):
    out = tmp_path / "responses.jsonl"
    run(_manifest(tmp_path), _Fake(), out, documents=DOCS, cluster_members={"c": tuple(DOCS)})
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert {r["instance_id"] for r in rows} == {"p1#original", "p1#d0", "p2#d0"}


def test_ingests_once_per_distinct_excision_set_not_once_per_instance(tmp_path: Path):
    system = _Fake()
    run(_manifest(tmp_path), system, tmp_path / "r.jsonl", documents=DOCS,
        cluster_members={"c": tuple(DOCS)})
    # Two distinct states: nothing excised, and {c/D1:1} excised.
    assert system.ingest_calls == 2


def test_abstention_is_recorded_as_a_boolean(tmp_path: Path):
    out = tmp_path / "r.jsonl"
    run(_manifest(tmp_path), _Fake(), out, documents=DOCS, cluster_members={"c": tuple(DOCS)})
    rows = {
        json.loads(line)["instance_id"]: json.loads(line)
        for line in out.read_text(encoding="utf-8").splitlines()
    }
    assert isinstance(rows["p1#d0"]["abstained"], bool)


def test_a_system_that_merges_instead_of_replacing_is_stopped(tmp_path: Path):
    with pytest.raises(InvariantViolation, match="still indexed"):
        run(_manifest(tmp_path), _Fake(leak=True), tmp_path / "r.jsonl", documents=DOCS,
            cluster_members={"c": tuple(DOCS)})


def test_resume_skips_instances_already_written(tmp_path: Path):
    out = tmp_path / "r.jsonl"
    manifest = _manifest(tmp_path)
    run(manifest, _Fake(), out, documents=DOCS, cluster_members={"c": tuple(DOCS)})
    system = _Fake()
    run(manifest, system, out, documents=DOCS, cluster_members={"c": tuple(DOCS)}, resume=True)
    assert system.ingest_calls == 0


def test_ingest_is_scoped_to_the_questions_own_conversation(tmp_path: Path):
    """A question is scored against its own conversation, not the whole corpus.

    This is the difference between indexing 646 turns per state and 5 882 — at ~1 500 states, the
    difference between a run an adopter can finish and one nobody will.
    """
    two_clusters = dict(DOCS)
    two_clusters.update({f"other/D1:{i}": f"unrelated turn {i}" for i in range(1, 4)})
    system = _Fake()
    run(
        _manifest(tmp_path),
        system,
        tmp_path / "r.jsonl",
        documents=two_clusters,
        cluster_members={"c": tuple(DOCS), "other": ("other/D1:1", "other/D1:2", "other/D1:3")},
    )
    assert all(not d.startswith("other/") for d in system.indexed_doc_ids())


def test_invariant_three_still_fires_on_a_fully_resumed_run(tmp_path: Path):
    """A resume where every original was already scored must not skip the check silently."""
    out = tmp_path / "r.jsonl"
    manifest = _manifest(tmp_path)
    # Hand-write a completed run in which the answerable original was ABSTAINED on.
    rows = [
        {"instance_id": "p1#original", "system": "fake", "abstained": True, "cited_ids": [], "tokens": 0},
        {"instance_id": "p1#d0", "system": "fake", "abstained": True, "cited_ids": [], "tokens": 0},
        {"instance_id": "p2#d0", "system": "fake", "abstained": True, "cited_ids": [], "tokens": 0},
    ]
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    with pytest.raises(InvariantViolation, match="broken"):
        run(manifest, _Fake(), out, documents=DOCS, cluster_members={"c": tuple(DOCS)}, resume=True)


class _BadSignature:
    """An adapter whose method NAMES match `MemorySystem` but whose signatures do not.

    `MemorySystem` is `runtime_checkable`, which checks names only (Task 7 review) — an adapter
    shaped like this passes `isinstance` and must instead be caught by a real call, not by typing.
    """

    name = "bad"

    def ingest(self, docs, extra_required_arg) -> None:  # missing default -> TypeError on call
        raise AssertionError("should never be reached")

    def indexed_doc_ids(self) -> frozenset[str]:
        return frozenset()

    def query(self, question: str) -> Response:
        return Response(answer=None)


def test_smoke_check_calls_ingest_and_query_and_fails_fast_on_a_bad_signature():
    from benchmarks.ladder.run import AdapterSmokeCheckFailed, smoke_check

    with pytest.raises(AdapterSmokeCheckFailed):
        smoke_check(_BadSignature())


def test_smoke_check_passes_for_a_well_shaped_system():
    from benchmarks.ladder.run import smoke_check

    smoke_check(_Fake())  # must not raise


def test_run_rejects_an_adapter_whose_query_signature_is_wrong(tmp_path: Path):
    """runtime_checkable passes this class — it has all three method NAMES. Only a signature
    check catches it, and catching it here beats catching it forty minutes in."""

    class _WrongArity(_Fake):
        def query(self):  # missing `question`
            return Response(answer=None)

    with pytest.raises(AdapterSmokeCheckFailed, match="query"):
        run(_manifest(tmp_path), _WrongArity(), tmp_path / "r.jsonl",
            documents=DOCS, cluster_members={"c": tuple(DOCS)})


def test_run_rejects_an_adapter_missing_a_method_entirely(tmp_path: Path):
    class _NoIndexedIds:
        name = "broken"

        def ingest(self, docs):
            return None

        def query(self, question):
            return Response(answer=None)

    with pytest.raises(AdapterSmokeCheckFailed, match="indexed_doc_ids"):
        run(_manifest(tmp_path), _NoIndexedIds(), tmp_path / "r.jsonl",
            documents=DOCS, cluster_members={"c": tuple(DOCS)})


def test_the_signature_check_does_not_disturb_the_ingest_counters(tmp_path: Path):
    """The reason this is a signature check and not a functional smoke call."""
    system = _Fake()
    run(_manifest(tmp_path), system, tmp_path / "r.jsonl",
        documents=DOCS, cluster_members={"c": tuple(DOCS)})
    assert system.ingest_calls == 2


# --- FIX-E: --table/--tenant are real flags, threaded into RecallSystem -----------------------


def test_main_threads_table_and_tenant_flags_into_recall_system(tmp_path: Path, monkeypatch):
    """`--table`/`--tenant` must actually reach `RecallSystem.__init__` — the whole point is that
    two concurrent runs against the same --dsn can use distinct table/tenant pairs for isolation."""
    import benchmarks.ladder.sources.locomo as locomo_mod
    import benchmarks.ladder.systems.recall_system as recall_system_mod
    from benchmarks.ladder.sources.locomo import SourceCorpus

    captured: dict[str, str] = {}

    class _StubRecallSystem:
        name = "recall"

        def __init__(self, dsn, *, table, tenant, embedder=None):
            captured["dsn"] = dsn
            captured["table"] = table
            captured["tenant"] = tenant
            captured["embedder"] = embedder
            self._docs: dict[str, str] = {}

        def ingest(self, docs: Iterable[Document]) -> None:
            self._docs = {d.doc_id: d.text for d in docs}

        def indexed_doc_ids(self) -> frozenset[str]:
            return frozenset(self._docs)

        def query(self, question: str) -> Response:
            if not self._docs:
                return Response(answer=None)
            first = sorted(self._docs)[0]
            return Response(answer=self._docs[first], cited_ids=(first,), tokens=1)

    stub_corpus = SourceCorpus(
        documents=tuple(DOCS.items()), questions=(), cluster_members={"c": tuple(DOCS)},
        content_hash="x",
    )
    monkeypatch.setattr(locomo_mod, "load_locomo", lambda path: stub_corpus)
    monkeypatch.setattr(recall_system_mod, "RecallSystem", _StubRecallSystem)

    manifest = _manifest(tmp_path)
    out = tmp_path / "out.jsonl"
    rc = main([
        "--manifest", str(manifest),
        "--locomo", str(tmp_path / "unused_locomo.json"),
        "--out", str(out),
        "--dsn", "postgresql://fake",
        "--table", "custom_table",
        "--tenant", "custom_tenant",
    ])
    assert rc == 0
    assert captured["table"] == "custom_table"
    assert captured["tenant"] == "custom_tenant"


def test_main_defaults_table_and_tenant_when_flags_are_omitted(tmp_path: Path, monkeypatch):
    """Existing invocations (no --table/--tenant) must behave identically to before this fix."""
    import benchmarks.ladder.sources.locomo as locomo_mod
    import benchmarks.ladder.systems.recall_system as recall_system_mod
    from benchmarks.ladder.sources.locomo import SourceCorpus
    from benchmarks.ladder.systems.recall_system import DEFAULT_TABLE, DEFAULT_TENANT

    captured: dict[str, str] = {}

    class _StubRecallSystem:
        name = "recall"

        def __init__(self, dsn, *, table, tenant, embedder=None):
            captured["table"] = table
            captured["tenant"] = tenant
            captured["embedder"] = embedder
            self._docs: dict[str, str] = {}

        def ingest(self, docs: Iterable[Document]) -> None:
            self._docs = {d.doc_id: d.text for d in docs}

        def indexed_doc_ids(self) -> frozenset[str]:
            return frozenset(self._docs)

        def query(self, question: str) -> Response:
            if not self._docs:
                return Response(answer=None)
            first = sorted(self._docs)[0]
            return Response(answer=self._docs[first], cited_ids=(first,), tokens=1)

    stub_corpus = SourceCorpus(
        documents=tuple(DOCS.items()), questions=(), cluster_members={"c": tuple(DOCS)},
        content_hash="x",
    )
    monkeypatch.setattr(locomo_mod, "load_locomo", lambda path: stub_corpus)
    monkeypatch.setattr(recall_system_mod, "RecallSystem", _StubRecallSystem)

    manifest = _manifest(tmp_path)
    out = tmp_path / "out.jsonl"
    main([
        "--manifest", str(manifest),
        "--locomo", str(tmp_path / "unused_locomo.json"),
        "--out", str(out),
        "--dsn", "postgresql://fake",
    ])
    assert captured["table"] == DEFAULT_TABLE
    assert captured["tenant"] == DEFAULT_TENANT


def test_recall_system_guard_names_flags_that_actually_exist(monkeypatch):
    """The non-empty-table guard used to tell the user to use `--table`, a flag that did not
    exist. Now that it does (FIX-E), the guard message must actually be accurate — and must not
    claim it can atomically prevent a race between two genuinely concurrent processes."""
    import benchmarks.ladder.systems.recall_system as rs_mod

    class _FakeEmbedder:
        dim = 8

        def embed(self, texts):
            return [[0.0] * 8 for _ in texts]

    class _NonEmptyStore:
        def __init__(self, dsn, dim, table, *, tenant):
            self.table = table
            self.tenant = tenant

        def ensure_schema(self):
            pass

        def count(self):
            return 3

        def close(self):
            pass

    monkeypatch.setattr(rs_mod, "FastEmbedEmbedder", _FakeEmbedder)
    monkeypatch.setattr(rs_mod, "PgVectorStore", _NonEmptyStore)

    with pytest.raises(RuntimeError) as exc_info:
        rs_mod.RecallSystem("postgresql://fake", table="custom_table", tenant="custom_tenant")

    msg = str(exc_info.value)
    assert "--table" in msg
    assert "--tenant" in msg
    assert "custom_table" in msg


# --- FIX-SEC1: no predictable shared-temp cache path -------------------------------------------


def test_recall_system_default_cache_path_is_not_the_old_shared_fixed_name(monkeypatch):
    """The old default (`<tempdir>/ladder_recall_embed_cache.sqlite`) is predictable and shared
    across every user of the host — CWE-377/CWE-59. Two distinct instances must not collide."""
    import tempfile as tempfile_mod

    import benchmarks.ladder.systems.recall_system as rs_mod

    class _FakeEmbedder:
        dim = 8

        def embed(self, texts):
            return [[0.0] * 8 for _ in texts]

    class _EmptyStore:
        def __init__(self, dsn, dim, table, *, tenant):
            self.table = table
            self.tenant = tenant

        def ensure_schema(self):
            pass

        def count(self):
            return 0

        def close(self):
            pass

    captured_paths: list[Path] = []

    class _FakeCache:
        def __init__(self, path):
            captured_paths.append(Path(path))

        def close(self):
            pass

    monkeypatch.setattr(rs_mod, "FastEmbedEmbedder", _FakeEmbedder)
    monkeypatch.setattr(rs_mod, "PgVectorStore", _EmptyStore)
    monkeypatch.setattr(rs_mod, "EmbeddingCache", _FakeCache)

    system_a = rs_mod.RecallSystem("postgresql://fake", table="t1", tenant="tenant-a")
    system_b = rs_mod.RecallSystem("postgresql://fake", table="t2", tenant="tenant-b")

    old_shared_default = Path(tempfile_mod.gettempdir()) / "ladder_recall_embed_cache.sqlite"
    assert captured_paths[0] != old_shared_default
    assert captured_paths[1] != old_shared_default
    # Two different tenants (i.e. two different runs) must not be handed the same cache path.
    assert captured_paths[0] != captured_paths[1]

    system_a.close()
    system_b.close()


@pytest.mark.skipif(__import__("os").name != "posix", reason="permission bits are POSIX-only")
def test_recall_system_cache_dir_is_owner_only(monkeypatch):
    """Created under a directory THIS process makes with restrictive permissions — the pattern
    `ingest()` already uses via `tempfile.mkdtemp`, per FIX-SEC1."""
    import os
    import stat

    import benchmarks.ladder.systems.recall_system as rs_mod

    class _FakeEmbedder:
        dim = 8

        def embed(self, texts):
            return [[0.0] * 8 for _ in texts]

    class _EmptyStore:
        def __init__(self, dsn, dim, table, *, tenant):
            self.table = table
            self.tenant = tenant

        def ensure_schema(self):
            pass

        def count(self):
            return 0

        def close(self):
            pass

    monkeypatch.setattr(rs_mod, "FastEmbedEmbedder", _FakeEmbedder)
    monkeypatch.setattr(rs_mod, "PgVectorStore", _EmptyStore)

    system = rs_mod.RecallSystem("postgresql://fake", table="t1", tenant="tenant-a")
    try:
        cache_dir = system._cache_path.parent
        mode = stat.S_IMODE(os.stat(cache_dir).st_mode)
        assert mode == 0o700
    finally:
        system.close()


# --- FIX-F: --expected-digest actually arms the fifth invariant --------------------------------


def test_expected_digest_matching_the_manifest_is_accepted(tmp_path: Path):
    manifest = _manifest(tmp_path)
    _instances, header = read_manifest(manifest)
    out = tmp_path / "r.jsonl"
    run(manifest, _Fake(), out, documents=DOCS, cluster_members={"c": tuple(DOCS)},
        expected_digest=header["digest"])
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert {r["instance_id"] for r in rows} == {"p1#original", "p1#d0", "p2#d0"}


def test_expected_digest_mismatch_is_rejected_before_any_scoring(tmp_path: Path):
    manifest = _manifest(tmp_path)
    out = tmp_path / "r.jsonl"
    with pytest.raises(InvariantViolation, match="digest"):
        run(manifest, _Fake(), out, documents=DOCS, cluster_members={"c": tuple(DOCS)},
            expected_digest="deadbeef")
    # Nothing scored: the digest check ran BEFORE the output file was ever opened.
    assert not out.exists()


def test_expected_digest_omitted_prints_a_not_armed_notice(tmp_path: Path, capsys):
    run(_manifest(tmp_path), _Fake(), tmp_path / "r.jsonl", documents=DOCS,
        cluster_members={"c": tuple(DOCS)})
    captured = capsys.readouterr()
    assert "NOT ARMED" in captured.out


# --- FIX-BUG1: --resume must survive a truncated trailing line ---------------------------------


def test_resume_treats_a_truncated_trailing_line_as_not_yet_recorded(tmp_path: Path):
    """A process killed mid-write (SIGKILL, OOM, power loss) leaves a truncated final line.
    Resume must treat it as 'not yet recorded', not raise `JSONDecodeError` before scoring
    starts — that was the bug: the run could not resume at all until a human hand-edited the
    file."""
    out = tmp_path / "r.jsonl"
    out.write_text(
        '{"instance_id": "p1#original", "system": "fake", "abstained": false, "cited_ids": [], '
        '"tokens": 5}\n'
        '{"instance_id": "p1#d0", "system": "fake", "abst',
        encoding="utf-8",
    )
    manifest = _manifest(tmp_path)
    system = _Fake()
    n = run(manifest, system, out, documents=DOCS, cluster_members={"c": tuple(DOCS)}, resume=True)
    # p1#original was validly recorded and must be skipped; the truncated p1#d0 and the
    # never-written p2#d0 must both be (re)scored.
    assert n == 2
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert {r["instance_id"] for r in rows} == {"p1#original", "p1#d0", "p2#d0"}
    # The truncated tail must not have been glued onto the next append: every line must parse.
    for line in out.read_text(encoding="utf-8").splitlines():
        json.loads(line)


def test_a_malformed_line_in_the_middle_of_the_file_stays_loud(tmp_path: Path):
    """Corruption in the MIDDLE of the file is a different signal than a truncated tail — it must
    not be silently swallowed the same way."""
    out = tmp_path / "r.jsonl"
    out.write_text(
        '{"instance_id": "p1#original", "system": "fake", "abst\n'
        '{"instance_id": "p1#d0", "system": "fake", "abstained": true, "cited_ids": [], '
        '"tokens": 0}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not valid JSON"):
        run(_manifest(tmp_path), _Fake(), out, documents=DOCS, cluster_members={"c": tuple(DOCS)},
            resume=True)


def test_omitting_the_embedder_flag_leaves_the_arm_on_the_shipped_default(tmp_path: Path, monkeypatch):
    """A different embedder is a separately labelled arm (SUITE-DESIGN rule 4).

    So the absence of `--embedder` must reach RecallSystem as None — letting it default inside the
    adapter — rather than this layer picking a model. If main() ever started naming a default here,
    an arm could change identity without its label changing, which is the same defect class as a
    headline naming a contrast that never ran.
    """
    import benchmarks.ladder.sources.locomo as locomo_mod
    import benchmarks.ladder.systems.recall_system as recall_system_mod
    from benchmarks.ladder.sources.locomo import SourceCorpus

    captured: dict[str, object] = {}

    class _Stub:
        name = "recall"

        def __init__(self, dsn, *, table, tenant, embedder=None):
            captured["embedder"] = embedder
            self._docs: dict[str, str] = {}

        def ingest(self, docs: Iterable[Document]) -> None:
            self._docs = {d.doc_id: d.text for d in docs}

        def indexed_doc_ids(self) -> frozenset[str]:
            return frozenset(self._docs)

        def query(self, question: str) -> Response:
            # Must ANSWER, not abstain: invariant 3 rejects a run where every answerable original
            # was declined, on the grounds that such questions are broken rather than hard. A stub
            # that abstained unconditionally would trip that guard and this test would be measuring
            # the guard instead of the flag. (It did, on the first draft.)
            first = sorted(self._docs)[0] if self._docs else None
            return Response(answer=None) if first is None else Response(answer=self._docs[first])

    monkeypatch.setattr(recall_system_mod, "RecallSystem", _Stub)
    monkeypatch.setattr(
        locomo_mod,
        "load_locomo",
        lambda _p: SourceCorpus(
            documents=tuple(DOCS.items()),
            questions=(),
            cluster_members={"c": tuple(DOCS)},
            content_hash="x",
        ),
    )
    manifest = _manifest(tmp_path)
    out = tmp_path / "r.jsonl"
    main(["--manifest", str(manifest), "--locomo", "x.json", "--out", str(out), "--dsn", "d"])
    assert captured["embedder"] is None
