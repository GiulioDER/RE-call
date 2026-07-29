"""The manifest is the released artifact — these tests are its contract with a stranger.

Prior work: [[project-recall-abstention-bounded-domain-2026-07-24]] — six abstention signals
measured, all failed; this measures the axis they were measured on, not a seventh signal.

A reader who distrusts our builder must be able to verify the digest and read the instances
without running our code. So the digest must be stable across instance ordering and across a
write/read round trip, and it must CHANGE when any scored field changes.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.ladder.manifest import (
    LABEL_ANSWERABLE,
    LABEL_UNANSWERABLE,
    MANIFEST_VERSION,
    Instance,
    instance_from_dict,
    instance_to_dict,
    manifest_digest,
    read_manifest,
    write_manifest,
)


def _inst(instance_id: str = "i1", ring: int = 0, **kw) -> Instance:
    base = dict(
        instance_id=instance_id,
        corpus="locomo",
        source_question_id="locomo_0_qa3",
        question="When did Caroline go to the support group?",
        label=LABEL_UNANSWERABLE,
        ring=ring,
        excised_doc_ids=("D1:3",),
        gold_doc_ids=("D1:3",),
        pair_id="p1",
    )
    base.update(kw)
    return Instance(**base)


def test_round_trips_through_dict_unchanged():
    inst = _inst()
    assert instance_from_dict(instance_to_dict(inst)) == inst


def test_digest_is_stable_across_instance_order():
    a, b = _inst("i1"), _inst("i2", ring=4)
    kw = dict(ring_widths=[0, 4], corpus_hashes={"locomo": "abc123"})
    assert manifest_digest([a, b], **kw) == manifest_digest([b, a], **kw)


def test_digest_changes_when_an_excised_id_changes():
    kw = dict(ring_widths=[0], corpus_hashes={"locomo": "abc123"})
    before = manifest_digest([_inst()], **kw)
    after = manifest_digest([_inst(excised_doc_ids=("D1:4",))], **kw)
    assert before != after


def test_digest_changes_when_the_label_changes():
    kw = dict(ring_widths=[0], corpus_hashes={"locomo": "abc123"})
    before = manifest_digest([_inst()], **kw)
    after = manifest_digest([_inst(label=LABEL_ANSWERABLE)], **kw)
    assert before != after


def test_write_then_read_preserves_instances_and_digest(tmp_path: Path):
    instances = [_inst("i1"), _inst("i2", ring=4)]
    path = tmp_path / "manifest.jsonl"
    digest = write_manifest(
        path, instances, ring_widths=[0, 4], corpus_hashes={"locomo": "abc123"}
    )
    read_back, header = read_manifest(path)
    assert read_back == instances
    assert header["digest"] == digest
    assert header["manifest_version"] == MANIFEST_VERSION
    assert header["ring_widths"] == [0, 4]
    assert header["corpus_hashes"] == {"locomo": "abc123"}


def test_read_rejects_a_manifest_whose_digest_does_not_match_its_body(tmp_path: Path):
    path = tmp_path / "manifest.jsonl"
    write_manifest(path, [_inst("i1")], ring_widths=[0], corpus_hashes={})
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[1] = lines[1].replace('"D1:3"', '"D9:9"')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="digest"):
        read_manifest(path)


def test_an_unknown_label_is_refused_at_construction():
    with pytest.raises(ValueError, match="label"):
        _inst(label="maybe")


def test_a_list_of_doc_ids_is_refused_because_a_released_artifact_cannot_be_mutable():
    with pytest.raises(TypeError, match="tuples"):
        _inst(excised_doc_ids=["D1:3"])


def test_editing_the_corpus_hash_in_the_header_is_refused(tmp_path: Path):
    """corpus_hashes says WHICH corpus this manifest came from — forging it is the attack."""
    path = tmp_path / "manifest.jsonl"
    write_manifest(path, [_inst("i1")], ring_widths=[0], corpus_hashes={"locomo": "abc123"})
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[0] = lines[0].replace("abc123", "deadbeef")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="digest"):
        read_manifest(path)


def test_editing_the_ring_widths_in_the_header_is_refused(tmp_path: Path):
    path = tmp_path / "manifest.jsonl"
    write_manifest(path, [_inst("i1")], ring_widths=[0, 4], corpus_hashes={})
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[0] = lines[0].replace("[0, 4]", "[0, 999]")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="digest"):
        read_manifest(path)


def test_a_truncated_file_reports_this_file_rather_than_a_raw_json_error(tmp_path: Path):
    path = tmp_path / "manifest.jsonl"
    write_manifest(path, [_inst("i1")], ring_widths=[0], corpus_hashes={})
    path.write_text(path.read_text(encoding="utf-8")[:40], encoding="utf-8")
    with pytest.raises(ValueError, match="truncated|not a manifest"):
        read_manifest(path)
