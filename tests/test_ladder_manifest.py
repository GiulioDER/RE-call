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
    MANIFEST_VERSION_V1,
    MANIFEST_VERSION_V2,
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
        path,
        instances,
        ring_widths=[0, 4],
        corpus_hashes={"locomo": "abc123"},
        manifest_version=MANIFEST_VERSION_V1,
    )
    read_back, header = read_manifest(path)
    assert read_back == instances
    assert header["digest"] == digest
    # Pins the VALUE PASSED IN, not a module-level default — a default is exactly how a v1
    # rebuild came to be stamped "2.0" (FIX-D). Written with V1 above; must read back as V1.
    assert header["manifest_version"] == MANIFEST_VERSION_V1
    assert header["ring_widths"] == [0, 4]
    assert header["corpus_hashes"] == {"locomo": "abc123"}


def test_write_manifest_requires_manifest_version_with_no_default(tmp_path: Path):
    """A default is how FIX-D regressed: `write_manifest` must refuse to guess the version."""
    path = tmp_path / "manifest.jsonl"
    with pytest.raises(TypeError):
        write_manifest(path, [_inst("i1")], ring_widths=[0], corpus_hashes={})


def test_write_manifest_writes_unix_newlines_even_on_windows(tmp_path: Path):
    """FIX-CRLF: the artifact is frozen and cited, so its bytes must not depend on the OS that
    built it. Default text-mode newline translation turns every \\n into \\r\\n on Windows while
    the digest is computed over \\n-joined lines — same content, different bytes. Assert directly
    on the bytes written to disk, not on the in-memory digest, since the bug is byte-level."""
    path = tmp_path / "manifest.jsonl"
    write_manifest(
        path,
        [_inst("i1")],
        ring_widths=[0],
        corpus_hashes={},
        manifest_version=MANIFEST_VERSION_V1,
    )
    raw = path.read_bytes()
    assert b"\r" not in raw


def test_read_rejects_a_manifest_whose_digest_does_not_match_its_body(tmp_path: Path):
    path = tmp_path / "manifest.jsonl"
    write_manifest(
        path, [_inst("i1")], ring_widths=[0], corpus_hashes={}, manifest_version=MANIFEST_VERSION_V1
    )
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
    write_manifest(
        path,
        [_inst("i1")],
        ring_widths=[0],
        corpus_hashes={"locomo": "abc123"},
        manifest_version=MANIFEST_VERSION_V1,
    )
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[0] = lines[0].replace("abc123", "deadbeef")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="digest"):
        read_manifest(path)


def test_editing_the_ring_widths_in_the_header_is_refused(tmp_path: Path):
    path = tmp_path / "manifest.jsonl"
    write_manifest(
        path,
        [_inst("i1")],
        ring_widths=[0, 4],
        corpus_hashes={},
        manifest_version=MANIFEST_VERSION_V1,
    )
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[0] = lines[0].replace("[0, 4]", "[0, 999]")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="digest"):
        read_manifest(path)


def test_a_truncated_file_reports_this_file_rather_than_a_raw_json_error(tmp_path: Path):
    path = tmp_path / "manifest.jsonl"
    write_manifest(
        path, [_inst("i1")], ring_widths=[0], corpus_hashes={}, manifest_version=MANIFEST_VERSION_V1
    )
    path.write_text(path.read_text(encoding="utf-8")[:40], encoding="utf-8")
    with pytest.raises(ValueError, match="truncated|not a manifest"):
        read_manifest(path)


# --- v2: scope_cluster_ids ------------------------------------------------------------------


def test_scope_cluster_ids_defaults_to_empty_tuple():
    inst = _inst()
    assert inst.scope_cluster_ids == ()


def test_scope_cluster_ids_round_trips_through_dict():
    inst = _inst(scope_cluster_ids=("conv-1", "conv-2"))
    assert instance_from_dict(instance_to_dict(inst)) == inst


def test_a_v1_shaped_dict_with_no_scope_cluster_ids_key_still_deserialises():
    d = instance_to_dict(_inst())
    del d["scope_cluster_ids"]
    inst = instance_from_dict(d)
    assert inst.scope_cluster_ids == ()


def test_the_frozen_v1_manifest_still_reads_with_its_digest_intact():
    path = Path("results/ladder/manifest.jsonl")
    if not path.exists():
        pytest.skip("full ladder manifest is archived outside the source tree")
    instances, header = read_manifest(path)
    assert header["digest"] == "6bfe2d2b094eefaf64409a3eddbb26d62b9e7709346540b2d068a4be300632b1"
    assert header["manifest_version"] == MANIFEST_VERSION_V1
    assert len(instances) == 1800
    assert all(inst.scope_cluster_ids == () for inst in instances)


def test_read_manifest_accepts_a_2_0_header(tmp_path: Path):
    path = tmp_path / "manifest.jsonl"
    write_manifest(
        path, [_inst("i1")], ring_widths=[0], corpus_hashes={}, manifest_version=MANIFEST_VERSION_V1
    )
    lines = path.read_text(encoding="utf-8").splitlines()
    header = __import__("json").loads(lines[0])
    header["manifest_version"] = MANIFEST_VERSION_V2
    lines[0] = __import__("json").dumps(header, sort_keys=True, ensure_ascii=False)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    instances, header_back = read_manifest(path)
    assert header_back["manifest_version"] == MANIFEST_VERSION_V2
    assert len(instances) == 1
