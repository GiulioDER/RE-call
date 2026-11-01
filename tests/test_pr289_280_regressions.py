"""PR289 through PR280 regressions, proved against d8d0bc04c on the remote host.

Each defect case must fail its stated assertion before its production repair.
The test node, baseline and failure are also retained in the CCA proof receipt.
"""
from __future__ import annotations

import json
import zipfile
from dataclasses import replace

import pytest

from benchmarks.enterprise_rag import load_documents
from recall import index as index_module
from recall.derived_block import DerivedBlockError, render_derived_block, verify_derived_block
from recall.document import ParsedDocument, parse_document
from recall.embeddings import HashingEmbedder
from recall.index import Indexer
from recall.rewrite import apply_rewrite
from recall.truth_extraction._normalize import human_body_of
from scripts.enterprise_rag_build_eval_cache import build_cache
from tests.conftest import requires_db
from tests.test_corpus_rewrite_contract import _corpus, _fact
from tests.test_derived_block_contract import _entry


@pytest.mark.parametrize("relation,object_id", [
    ("contradicts", "new_decision_2026-06-01.md"),
    ("same_entity", "new_decision_2026-06-01.md"),
    ("declares_status", "status:active"),
])
def test_reviewed_annotations_never_become_evidence(tmp_path, relation, object_id):
    """FIX-001: baseline fails human_body equality after the real reviewed writer."""
    _corpus(tmp_path)
    memo = tmp_path / "old_decision_2026-01-01.md"
    before = parse_document(memo.read_text()).human_body
    result = apply_rewrite(
        tmp_path, _fact(relation=relation, object_id=object_id), apply=True
    )
    assert result.written
    after = memo.read_text()
    assert parse_document(after).human_body == before, "machine annotations became evidence"
    assert human_body_of(after) == before
    assert parse_document(after).derived_text


@requires_db
def test_index_rebuilds_existing_contaminated_memo(tmp_path, make_store, monkeypatch):
    """FIX-001: baseline skips the unchanged contaminated file instead of repairing it.

    Seed through the old parser and derivation identity, then restore the live implementation.
    The public indexing boundary must rebuild once and reuse the repaired result afterwards.
    """
    raw = "# Memo\n\nAuthored evidence.\n\n<!-- recall:derived -->\nstatus: adopted\n<!-- /recall:derived -->\n"
    (tmp_path / "memo.md").write_text(raw)
    store = make_store(64)
    embedder = HashingEmbedder(dim=64)
    with monkeypatch.context() as old:
        old.setattr(index_module, "_body_derivation_hash", lambda text, digest: digest)
        old.setattr(index_module, "parse_document", lambda text: ParsedDocument({}, text.rstrip(), ""))
        Indexer(store, embedder).index_path(tmp_path)
    assert store.query_sparse("adopted", k=1), "seed must contain the stale machine annotation"
    repaired = Indexer(store, embedder).index_path(tmp_path)
    assert repaired.files == 1, "unchanged contaminated memo was reused"
    assert not store.query_sparse("adopted", k=1)
    assert Indexer(store, embedder).index_path(tmp_path).skipped == 1


@pytest.mark.parametrize("escape", ["parent", "absolute", "symlink"])
def test_eval_cache_refuses_archive_destination_escape(tmp_path, escape):
    """FIX-002: baseline writes outside sources_out and fails the explicit refusal assertion."""
    sources = tmp_path / "sources"
    sources.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    if escape == "parent":
        member = "../outside/dsid_one__note.txt"
    elif escape == "absolute":
        member = (outside / "dsid_one__note.txt").as_posix()
    else:
        (sources / "link").symlink_to(outside, target_is_directory=True)
        member = "link/dsid_one__note.txt"
    questions = tmp_path / "questions.jsonl"
    questions.write_text(json.dumps({"expected_doc_ids": ["dsid_one"]}) + "\n")
    answers = tmp_path / "answers.jsonl"
    answers.write_text("")
    archive = tmp_path / "documents.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(member, "Title\nContent")
    refused = False
    try:
        build_cache(questions=questions, answers=answers, documents_zip=archive,
                    sources_out=sources, index_out=tmp_path / "index.json")
    except ValueError:
        refused = True
    assert refused, "archive path escaped the requested output directory"
    assert not list(outside.iterdir()), "refusal must precede writing"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("nested/dsid_one__note.txt", "Title\nContent")
    build_cache(questions=questions, answers=answers, documents_zip=archive,
                sources_out=sources, index_out=tmp_path / "index.json")
    assert json.loads((tmp_path / "index.json").read_text()) == {
        "dsid_one": "nested/dsid_one__note.json"
    }


@pytest.mark.parametrize("zipped", [False, True], ids=["file", "zip"])
def test_indented_json_documents_preserve_jsonl_support(tmp_path, zipped):
    """FIX-003: baseline rejects the indented object and fails the document-ID assertion."""
    row = {"doc_id": "dsid_one", "title": "Title", "content": "Evidence"}
    text = json.dumps(row, indent=2)
    path = tmp_path / ("documents.zip" if zipped else "document.json")
    if zipped:
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("document.json", text)
    else:
        path.write_text(text)
    jsonl = tmp_path / "documents.jsonl"
    jsonl.write_text(json.dumps({**row, "doc_id": "dsid_two"}) + "\n"
                     + json.dumps({**row, "doc_id": "dsid_three"}) + "\n")
    try:
        documents = list(load_documents([path, jsonl]))
    except json.JSONDecodeError:
        documents = []
    assert [doc.doc_id for doc in documents] == ["dsid_one", "dsid_two", "dsid_three"], (
        "formatting a supported JSON object must not change its acceptance"
    )


@pytest.mark.parametrize("field", ["provider", "reviewer", "note"])
def test_derived_renderer_refuses_carriage_return_in_audit_field(field):
    """FIX-004: baseline accepts terminal CR and fails the explicit refusal assertion."""
    entry = replace(_entry(), **{field: "reviewed\r"})
    refused = False
    try:
        render_derived_block([entry])
    except DerivedBlockError:
        refused = True
    assert refused, "renderer accepted a field its parser cannot preserve"
    valid = replace(entry, **{field: "reviewed"})
    assert verify_derived_block(render_derived_block([valid])).entries == (valid,)


def test_derived_renderer_refuses_newline_in_proposal_id():
    """FIX-004: baseline accepts a terminal LF after64 hex characters, violating round trips."""
    refused = False
    try:
        render_derived_block([_entry(proposal="a" * 64 + "\n")])
    except DerivedBlockError:
        refused = True
    assert refused, "proposal ID newline was accepted"
    valid = _entry(proposal="a" * 64)
    assert verify_derived_block(render_derived_block([valid])).entries == (valid,)
