"""Regression proofs for the surviving PR70 through PR79 evaluation paths.

Each node is run against a214dab75 before the repair and fails its stated
assertion, then runs unchanged against the fixed source. Receipts retain exact
node IDs and source hashes. External services and paid models are not used.
"""
from __future__ import annotations

import json
import ntpath
import sys
import uuid
from pathlib import PureWindowsPath

import pytest

from recall.calibration import load_for
from recall.embeddings import HashingEmbedder
from recall.eval import locomo, locomo_abstention
from recall.eval.longmemeval import ConversionError, convert
from recall.eval.longmemeval_perq import populate_haystack
from recall.index import Indexer
from recall.store import PgVectorStore

from .conftest import TEST_DSN, requires_db
from .test_eval_longmemeval import _instance


@pytest.mark.parametrize("sample_id", ["C:..", "C:payload", "C:"])
def test_locomo_refuses_drive_sample_before_storage(tmp_path, monkeypatch, sample_id):
    """SEC-001: unsafe sample IDs must fail before the storage assertion sentinel."""
    dataset = tmp_path / "data.json"
    dataset.write_text(json.dumps([{"sample_id": sample_id, "conversation": {}, "qa": []}]))
    monkeypatch.setattr(locomo, "_make_embedder", lambda _: HashingEmbedder(dim=16))

    def storage(*args, **kwargs):
        raise AssertionError("unsafe sample_id reached storage before refusal")

    monkeypatch.setattr(locomo, "PgVectorStore", storage)
    assert ntpath.normpath(str(PureWindowsPath("C:/audit/workspace") / "C:..")) == "C:\\audit"
    assert locomo._dia_id_to_filename("D1:3") == "D1_3.md"
    with pytest.raises(ValueError, match="sample_id"):
        locomo.run(dataset, dsn="unused", embedder_name="hashing", k=1, limit=None,
                   keep_corpus=tmp_path / "corpus", table="unused")


@requires_db
@pytest.mark.parametrize("uppercase", [False, True])
def test_haystack_alias_refusal_preserves_master(tmp_path, uppercase):
    """STAKES-001: baseline TRUNCATE loses two rows through either table alias."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "s1.md").write_text("the deployment target is staging")
    (corpus / "s2.md").write_text("the oncall rotation is weekly")
    emb = HashingEmbedder(dim=16)
    table = "alias_" + uuid.uuid4().hex[:8]
    store = PgVectorStore(TEST_DSN, dim=emb.dim, table=table)
    sub = None
    try:
        store.ensure_schema()
        Indexer(store, emb).index_path(corpus)
        before = store.count()
        assert before == 2
        rejected = False
        try:
            sub = populate_haystack(TEST_DSN, emb.dim, table,
                                    table.upper() if uppercase else table, ["s1.md"])
        except ValueError:
            rejected = True
        assert store.count() == before, "source corpus was truncated through scratch alias"
        assert rejected, "source/destination alias was not rejected"
    finally:
        if sub is not None:
            sub.close()
        store.drop_table()
        store.close()


def test_reconversion_refuses_stale_files_without_changing_previous_output(tmp_path):
    """BUG-001: baseline silently retains s2.md while emitting the smaller questions."""
    instances = [_instance(), _instance(question_id="q2", haystack_session_ids=["s2"],
                                       answer_session_ids=["s2"])]
    convert(instances, tmp_path)
    convert(instances, tmp_path)
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises(ConversionError, match="existing corpus"):
        convert([_instance()], tmp_path)
    assert {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("field", ["n_answerable", "n_unanswerable"])
@pytest.mark.parametrize("count", [sys.maxsize + 1, 10**200, 10**400])
def test_impossible_calibration_counts_are_unknown(tmp_path, field, count):
    """DAT-002: baseline either certifies impossible counts or overflows load_for."""
    path = tmp_path / "calibration.json"
    payload = dict(embedder="e", threshold=0.7, scale=0.05, separability=0.95,
                   n_answerable=200, n_unanswerable=200)
    path.write_text(json.dumps(payload))
    assert load_for("e", path).certified is True
    payload[field] = count
    path.write_text(json.dumps(payload))
    for _ in range(2):
        error = None
        back = None
        try:
            back = load_for("e", path)
        except OverflowError as exc:
            error = exc
        assert error is None, "malformed sample count escaped the calibration load boundary"
        assert back is not None
        assert getattr(back, field) is None, "impossible sample count survived diagnosis validation"
        assert back.certified is None


@requires_db
@pytest.mark.parametrize("present", [0, 1])
def test_ablation_refuses_incomplete_tenant_before_scoring(tmp_path, monkeypatch, present):
    """BUG-002: baseline scores a missing/partial tenant instead of refusing it."""
    conv = {"session_1": [{"dia_id": "D1:1", "speaker": "A", "text": "deploy to staging"},
                          {"dia_id": "D1:2", "speaker": "A", "text": "back up nightly"}]}
    qa = [{"category": 1, "question": "where deploy?", "evidence": ["D1:1"]},
          {"category": 5, "question": "where does B deploy?"}]
    dataset = tmp_path / "data.json"
    dataset.write_text(json.dumps([{"sample_id": "main", "conversation": conv, "qa": qa}]))
    corpus = tmp_path / "corpus"
    locomo.write_conversation_corpus(conv, corpus)
    emb = HashingEmbedder(dim=16)
    table = "coverage_" + uuid.uuid4().hex[:8]
    own = PgVectorStore(TEST_DSN, dim=emb.dim, table=table, tenant="locomo-main")
    other = PgVectorStore(TEST_DSN, dim=emb.dim, table=table, tenant="locomo-other")
    monkeypatch.setattr(locomo_abstention, "_make_embedder", lambda _: emb)
    monkeypatch.setattr(locomo_abstention, "PgVectorStore",
                        lambda dsn, dim, tenant, table: PgVectorStore(
                            dsn, dim=dim, tenant=tenant, table=own.table))

    def run():
        return locomo_abstention.run(dataset, dsn=TEST_DSN, embedder_name="hashing",
                                    k=2, answerable_sample=0, limit=None, seed=0)

    try:
        own.ensure_schema()
        Indexer(other, emb).index_path(corpus)
        if present:
            Indexer(own, emb).index_path(corpus, glob="D1_1.md")
        with pytest.raises(ValueError, match="indexed corpus"):
            run()
        Indexer(own, emb).index_path(corpus)
        result = run()
        assert result["conversations"] == 1
        for mode in locomo_abstention.MODES:
            assert result["modes"][mode]["adversarial_abstention"]["n"] == 1
            assert result["modes"][mode]["answerable_false_abstain"]["n"] == 1
    finally:
        own.drop_table()
        own.close()
        other.close()
