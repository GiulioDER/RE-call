"""The v2 builder: fractional rungs + distractor conversations, and the same manifest every time.

Prior work: [[project-recall-abstention-bounded-domain-2026-07-24]] — six abstention signals
measured, all failed; this measures the axis they were measured on, not a seventh signal.

Mirrors `test_ladder_build.py`'s determinism discipline, plus the property that is new here:
distractor selection must be reproducible **across processes**, not merely across calls in one —
`hash()` is salted per `PYTHONHASHSEED` and would silently break a manifest nobody could rebuild.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.ladder.build_v2 import build_v2_instances, main, select_distractors
from benchmarks.ladder.manifest import (
    LABEL_ANSWERABLE,
    LABEL_UNANSWERABLE,
    MANIFEST_VERSION_V2,
    RING_ORIGINAL,
    manifest_digest,
    read_manifest,
)
from benchmarks.ladder.rings import FractionSpec, fraction_to_ring
from benchmarks.ladder.sources.locomo import load_locomo

# Four conversations so distractor selection (n=2) has real candidates to choose among, and so
# "own cluster never among distractors" is a meaningful assertion rather than vacuous.
FOUR_CONVERSATIONS = [
    {
        "sample_id": "conv-0",
        "conversation": {
            "session_1_date_time": "7 May 2023",
            "session_1": [
                {"dia_id": "D1:1", "speaker": "Caroline", "text": "I went to the support group."},
                {"dia_id": "D1:2", "speaker": "Melanie", "text": "How was the support group?"},
                {"dia_id": "D1:3", "speaker": "Caroline", "text": "We talked about the group."},
                {"dia_id": "D1:4", "speaker": "Melanie", "text": "I ran a charity race."},
            ],
        },
        "qa": [
            {
                "question": "When did Caroline go to the support group?",
                "answer": "7 May 2023",
                "evidence": ["D1:1"],
                "category": 2,
            },
            {
                "question": "What did Melanie run?",
                "answer": "a charity race",
                "evidence": ["D1:4"],
                "category": 1,
            },
        ],
    },
    {
        "sample_id": "conv-1",
        "conversation": {
            "session_1_date_time": "3 June 2023",
            "session_1": [
                {"dia_id": "D1:1", "speaker": "Ravi", "text": "I adopted a dog last week."},
                {"dia_id": "D1:2", "speaker": "Priya", "text": "What breed is the dog?"},
                {"dia_id": "D1:3", "speaker": "Ravi", "text": "A golden retriever puppy."},
            ],
        },
        "qa": [
            {
                "question": "What breed did Ravi adopt?",
                "answer": "golden retriever",
                "evidence": ["D1:3"],
                "category": 1,
            },
        ],
    },
    {
        "sample_id": "conv-2",
        "conversation": {
            "session_1_date_time": "11 July 2023",
            "session_1": [
                {"dia_id": "D1:1", "speaker": "Zoe", "text": "I painted the fence blue."},
                {"dia_id": "D1:2", "speaker": "Adam", "text": "Nice, which fence?"},
            ],
        },
        "qa": [
            {
                "question": "What colour did Zoe paint the fence?",
                "answer": "blue",
                "evidence": ["D1:1"],
                "category": 1,
            },
        ],
    },
    {
        "sample_id": "conv-3",
        "conversation": {
            "session_1_date_time": "2 August 2023",
            "session_1": [
                {"dia_id": "D1:1", "speaker": "Lee", "text": "I started a pottery class."},
                {"dia_id": "D1:2", "speaker": "Nina", "text": "How is the pottery class going?"},
            ],
        },
        "qa": [
            {
                "question": "What class did Lee start?",
                "answer": "pottery",
                "evidence": ["D1:1"],
                "category": 1,
            },
        ],
    },
]

SPEC = FractionSpec(fractions=(0.0, 0.25, 0.5, 0.75, 1.0))


def _corpus(tmp_path: Path):
    path = tmp_path / "locomo.json"
    path.write_text(json.dumps(FOUR_CONVERSATIONS), encoding="utf-8")
    return load_locomo(path)


def _digest(instances) -> str:
    return manifest_digest(
        instances,
        ring_widths=[fraction_to_ring(f) for f in SPEC.fractions],
        corpus_hashes={"locomo": "x"},
    )


# --- select_distractors -----------------------------------------------------------------------


def test_select_distractors_excludes_the_own_cluster():
    all_ids = ["conv-0", "conv-1", "conv-2", "conv-3"]
    chosen = select_distractors("conv-0", all_ids, n=2, seed=0)
    assert "conv-0" not in chosen


def test_select_distractors_picks_n():
    all_ids = ["conv-0", "conv-1", "conv-2", "conv-3"]
    assert len(select_distractors("conv-0", all_ids, n=2, seed=0)) == 2


def test_select_distractors_result_is_sorted():
    all_ids = ["conv-0", "conv-1", "conv-2", "conv-3"]
    chosen = select_distractors("conv-3", all_ids, n=2, seed=0)
    assert list(chosen) == sorted(chosen)


def test_select_distractors_is_deterministic_for_the_same_seed():
    all_ids = ["conv-0", "conv-1", "conv-2", "conv-3"]
    a = select_distractors("conv-0", all_ids, n=2, seed=0)
    b = select_distractors("conv-0", all_ids, n=2, seed=0)
    assert a == b


def test_select_distractors_depends_on_the_seed():
    all_ids = ["conv-0", "conv-1", "conv-2", "conv-3", "conv-4", "conv-5"]
    a = select_distractors("conv-0", all_ids, n=2, seed=0)
    b = select_distractors("conv-0", all_ids, n=2, seed=1)
    assert a != b


def test_select_distractors_refuses_to_draw_more_than_available():
    with pytest.raises(ValueError, match="candidate"):
        select_distractors("conv-0", ["conv-0", "conv-1"], n=2, seed=0)


def test_select_distractors_is_reproducible_across_processes_with_different_pythonhashseed():
    """The trap the v1 builder was tested against: `hash()` is salted per `PYTHONHASHSEED`.

    If `select_distractors` used `hash(cluster_id)` anywhere, this would be flaky by construction
    — two subprocesses with different hash seeds would draw different distractors for the same
    conversation, and the manifest would not be rebuildable from the corpus alone. Runs the
    selection in two real subprocesses (not just reasoning about the source) so the test would
    actually fail if that regressed.
    """
    script = (
        "import json\n"
        "from benchmarks.ladder.build_v2 import select_distractors\n"
        "all_ids = ['conv-0', 'conv-1', 'conv-2', 'conv-3', 'conv-4', 'conv-5']\n"
        "print(json.dumps(select_distractors('conv-0', all_ids, n=2, seed=0)))\n"
    )
    repo_root = Path(__file__).resolve().parents[1]
    results = []
    for hashseed in ("111", "999999"):
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(repo_root),
            env={**__import__("os").environ, "PYTHONHASHSEED": hashseed},
            capture_output=True,
            text=True,
            check=True,
        )
        results.append(json.loads(proc.stdout.strip()))
    assert results[0] == results[1]


# --- build_v2_instances -----------------------------------------------------------------------


def test_every_question_yields_one_answerable_original_plus_one_per_fraction(tmp_path: Path):
    corpus = _corpus(tmp_path)
    instances = build_v2_instances(corpus, SPEC, corpus_name="locomo", sample=None)
    # 5 fractions + 1 answerable original = 6 per question
    assert len(instances) == len(corpus.questions) * 6


def test_the_answerable_original_excises_nothing(tmp_path: Path):
    instances = build_v2_instances(_corpus(tmp_path), SPEC, corpus_name="locomo", sample=None)
    originals = [i for i in instances if i.label == LABEL_ANSWERABLE]
    assert originals
    assert all(i.excised_doc_ids == () for i in originals)


def test_every_question_in_a_conversation_gets_the_same_distractors(tmp_path: Path):
    corpus = _corpus(tmp_path)
    instances = build_v2_instances(corpus, SPEC, corpus_name="locomo", sample=None)
    # conv-0 has two questions in the fixture.
    conv0 = [i for i in instances if i.source_question_id.startswith("conv-0/")]
    assert conv0
    scopes = {i.scope_cluster_ids for i in conv0}
    assert len(scopes) == 1


def test_a_questions_own_cluster_is_never_among_its_distractors(tmp_path: Path):
    """`select_distractors` itself is unit-tested for this; here we check the builder actually
    calls it with the question's OWN cluster excluded, by checking the distractor identity
    directly against what `select_distractors` would produce for the same corpus."""
    corpus = _corpus(tmp_path)
    all_ids = sorted(corpus.cluster_members)
    instances = build_v2_instances(corpus, SPEC, corpus_name="locomo", sample=None)
    for inst in instances:
        own = inst.source_question_id.split("/", 1)[0]
        assert own in inst.scope_cluster_ids  # own cluster IS part of the scope
        distractors = tuple(sorted(set(inst.scope_cluster_ids) - {own}))
        assert distractors == select_distractors(own, all_ids, n=2, seed=0)
        assert own not in select_distractors(own, all_ids, n=2, seed=0)


def test_scope_cluster_ids_is_set_on_every_instance_including_the_original(tmp_path: Path):
    corpus = _corpus(tmp_path)
    instances = build_v2_instances(corpus, SPEC, corpus_name="locomo", sample=None)
    assert all(i.scope_cluster_ids for i in instances)
    originals = [i for i in instances if i.ring == RING_ORIGINAL]
    assert originals
    assert all(i.scope_cluster_ids for i in originals)


def test_scope_cluster_ids_is_sorted_own_cluster_plus_distractors(tmp_path: Path):
    corpus = _corpus(tmp_path)
    instances = build_v2_instances(corpus, SPEC, corpus_name="locomo", sample=None)
    for inst in instances:
        assert list(inst.scope_cluster_ids) == sorted(inst.scope_cluster_ids)
        assert len(inst.scope_cluster_ids) == 3  # own + 2 distractors


def test_r1_excises_exactly_the_own_cluster_and_nothing_from_distractors(tmp_path: Path):
    corpus = _corpus(tmp_path)
    instances = build_v2_instances(corpus, SPEC, corpus_name="locomo", sample=None)
    for inst in instances:
        if inst.ring != fraction_to_ring(1.0):
            continue
        own = inst.source_question_id.split("/", 1)[0]
        own_cluster = set(corpus.cluster_members[own])
        distractors = set(inst.scope_cluster_ids) - {own}
        distractor_docs = {
            d for c in distractors for d in corpus.cluster_members.get(c, ())
        }
        assert set(inst.excised_doc_ids) == own_cluster
        assert not (set(inst.excised_doc_ids) & distractor_docs)


def test_r0_excises_only_gold(tmp_path: Path):
    corpus = _corpus(tmp_path)
    instances = build_v2_instances(corpus, SPEC, corpus_name="locomo", sample=None)
    for inst in instances:
        if inst.ring == fraction_to_ring(0.0):
            assert set(inst.excised_doc_ids) == set(inst.gold_doc_ids)


def test_a_family_shares_one_pair_id(tmp_path: Path):
    instances = build_v2_instances(_corpus(tmp_path), SPEC, corpus_name="locomo", sample=None)
    by_question: dict[str, set[str]] = {}
    for inst in instances:
        by_question.setdefault(inst.source_question_id, set()).add(inst.pair_id)
    assert all(len(pairs) == 1 for pairs in by_question.values())


def test_instance_ids_are_unique(tmp_path: Path):
    instances = build_v2_instances(_corpus(tmp_path), SPEC, corpus_name="locomo", sample=None)
    ids = [i.instance_id for i in instances]
    assert len(ids) == len(set(ids))


def test_two_builds_of_the_same_corpus_produce_the_same_digest(tmp_path: Path):
    corpus = _corpus(tmp_path)
    a = build_v2_instances(corpus, SPEC, corpus_name="locomo", sample=None)
    b = build_v2_instances(corpus, SPEC, corpus_name="locomo", sample=None)
    assert _digest(a) == _digest(b)


def test_the_bm25_index_covers_the_whole_corpus_so_sampling_does_not_reshape_rings(tmp_path: Path):
    """Ring order must not depend on which questions were drawn — otherwise the sample silently
    reshapes the x-axis, exactly the property `build.py`'s builder pins."""
    corpus = _corpus(tmp_path)
    full = {
        i.instance_id: i.excised_doc_ids
        for i in build_v2_instances(corpus, SPEC, corpus_name="locomo", sample=None)
    }
    drawn = build_v2_instances(corpus, SPEC, corpus_name="locomo", sample=1, sample_seed=0)
    assert drawn
    for inst in drawn:
        assert inst.excised_doc_ids == full[inst.instance_id]


def test_distractor_seed_changes_the_manifest(tmp_path: Path):
    corpus = _corpus(tmp_path)
    a = build_v2_instances(corpus, SPEC, corpus_name="locomo", sample=None, distractor_seed=0)
    b = build_v2_instances(corpus, SPEC, corpus_name="locomo", sample=None, distractor_seed=1)
    assert _digest(a) != _digest(b)


# --- CLI -----------------------------------------------------------------------------------


def test_cli_writes_a_readable_manifest(tmp_path: Path):
    src = tmp_path / "locomo.json"
    src.write_text(json.dumps(FOUR_CONVERSATIONS), encoding="utf-8")
    out = tmp_path / "manifest.jsonl"
    rc = main(
        [
            "--locomo",
            str(src),
            "--out",
            str(out),
            "--fractions",
            "0.00,0.50,1.00",
            "--sample",
            "0",
        ]
    )
    assert rc == 0
    instances, header = read_manifest(out)
    assert instances
    assert "locomo" in header["corpus_hashes"]
    assert all(i.scope_cluster_ids for i in instances)


def test_cli_defaults_match_the_preregistration(tmp_path: Path):
    src = tmp_path / "locomo.json"
    src.write_text(json.dumps(FOUR_CONVERSATIONS), encoding="utf-8")
    out = tmp_path / "manifest.jsonl"
    # sample defaults to 200, larger than our 4-conversation fixture, so every question survives.
    assert main(["--locomo", str(src), "--out", str(out)]) == 0
    instances, _header = read_manifest(out)
    corpus = load_locomo(src)
    assert len(instances) == len(corpus.questions) * 6


def test_cli_stamps_the_manifest_v2_not_v1(tmp_path: Path):
    """FIX-D companion: `build_v2.py` IS the v2 builder, so its header must say so explicitly —
    this pins the value it actually passes, not the module default it happened to share before
    the fix (see `test_ladder_build.py::test_cli_stamps_the_manifest_v1_not_v2` for the arm that
    was actually wrong)."""
    src = tmp_path / "locomo.json"
    src.write_text(json.dumps(FOUR_CONVERSATIONS), encoding="utf-8")
    out = tmp_path / "manifest.jsonl"
    assert main(["--locomo", str(src), "--out", str(out), "--sample", "0"]) == 0
    _instances, header = read_manifest(out)
    assert header["manifest_version"] == MANIFEST_VERSION_V2 == "2.0"


# --- FIX-ENV2: --sample-questions is the canonical flag name (matches build.py's) -----------


def test_sample_questions_is_the_canonical_flag_name(tmp_path: Path):
    """`build.py` uses `--sample-questions` for this exact pre-registered concept; `build_v2.py`
    used to call it `--sample` with identical help text — two names for one thing. The canonical
    name must now work here too, and select the pre-registered subset size."""
    src = tmp_path / "locomo.json"
    src.write_text(json.dumps(FOUR_CONVERSATIONS), encoding="utf-8")
    out = tmp_path / "manifest.jsonl"
    rc = main(
        [
            "--locomo",
            str(src),
            "--out",
            str(out),
            "--fractions",
            "0.00,1.00",
            "--sample-questions",
            "1",
            "--sample-seed",
            "0",
        ]
    )
    assert rc == 0
    instances, _header = read_manifest(out)
    assert len({i.source_question_id for i in instances}) == 1


def test_sample_alias_still_works_and_matches_sample_questions(tmp_path: Path):
    """`--sample` must keep working as an explicit alias, not be silently dropped — it is not a
    concept anyone renamed on purpose, it was the flag every prior invocation used."""
    src = tmp_path / "locomo.json"
    src.write_text(json.dumps(FOUR_CONVERSATIONS), encoding="utf-8")
    out_canonical = tmp_path / "canonical.jsonl"
    out_alias = tmp_path / "alias.jsonl"
    common = [
        "--locomo",
        str(src),
        "--fractions",
        "0.00,1.00",
        "--sample-seed",
        "0",
    ]
    assert main([*common, "--out", str(out_canonical), "--sample-questions", "1"]) == 0
    assert main([*common, "--out", str(out_alias), "--sample", "1"]) == 0
    _canonical_instances, canonical_header = read_manifest(out_canonical)
    _alias_instances, alias_header = read_manifest(out_alias)
    assert canonical_header["digest"] == alias_header["digest"]


def test_sample_questions_default_is_still_200(tmp_path: Path):
    """The v2 pre-registration fixes 200 for this arm — renaming the flag must not change the
    value. Spies on `build_v2_instances` (via a monkeypatched module attribute, restored in a
    `finally`) to capture the `sample=` the CLI actually threads through when neither
    `--sample-questions` nor `--sample` is given."""
    import benchmarks.ladder.build_v2 as build_v2_module

    src = tmp_path / "locomo.json"
    src.write_text(json.dumps(FOUR_CONVERSATIONS), encoding="utf-8")

    captured = {}
    orig_build = build_v2_module.build_v2_instances

    def _spy(*args, **kwargs):
        captured["sample"] = kwargs.get("sample")
        return orig_build(*args, **kwargs)

    build_v2_module.build_v2_instances = _spy
    try:
        out = tmp_path / "manifest.jsonl"
        assert main(["--locomo", str(src), "--out", str(out)]) == 0
    finally:
        build_v2_module.build_v2_instances = orig_build
    assert captured["sample"] == 200
