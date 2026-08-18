"""The synthetic corpus is evaluation INFRASTRUCTURE — if it is wrong, every number it
produces is wrong in a way that looks like a result. These tests pin the properties the
generated set must have for the measurements taken on it to mean anything:

- ground truth ids match what the harness actually computes from a chunk;
- every ground-truth document is exactly one chunk (so `:0` is a valid id);
- unanswerable queries are genuinely unanswerable (their subject is absent);
- the supersession pairs are ADVERSARIAL — the query is worded closer to the stale doc, which
  is the whole failure mode the trust layer exists to catch;
- generation is deterministic for a seed.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from recall.eval.synthetic import generate
from recall.frontmatter import parse_frontmatter
from recall.index import chunk_text


@pytest.fixture
def corpus(tmp_path):
    return generate(tmp_path / "c", n_answerable=12, n_unanswerable=6, n_successor=8,
                    n_abstain=4, n_filler_chunks=50, seed=7)


def _bodies(root):
    return {
        p.relative_to(root).as_posix(): parse_frontmatter(p.read_text(encoding="utf-8"))
        for p in sorted(root.rglob("*.md"))
    }


def test_every_ground_truth_doc_is_exactly_one_chunk(corpus):
    """Ground-truth ids end in `:0`, which is only true if the doc does not split."""
    referenced = set()
    for q in corpus.queries:
        for key in ("relevant_ids", "stale_ids", "successor_ids"):
            referenced.update(i.rsplit(":", 1)[0] for i in q.get(key, []))
    docs = _bodies(corpus.root)
    for rel in referenced:
        assert rel in docs, f"query references {rel}, which was not generated"
        assert len(chunk_text(docs[rel][1])) == 1, f"{rel} splits into more than one chunk"


def test_ground_truth_ids_match_the_harness_key_format(corpus):
    """`_key` builds `{metadata['file']}:{ord}` — ids must be spelled the same way."""
    for q in corpus.queries:
        for key in ("relevant_ids", "stale_ids", "successor_ids"):
            for ident in q.get(key, []):
                rel, _, ordinal = ident.rpartition(":")
                assert ordinal == "0"
                assert rel.endswith(".md")


def test_unanswerable_queries_have_no_document_about_their_subject(corpus):
    """A "gap" query whose subject is quietly present measures nothing."""
    text = " ".join(b for _, (_, b) in _bodies(corpus.root).items()).lower()
    unans = [q for q in corpus.queries if not q.get("trust") and not q["answerable"]]
    assert unans
    for q in unans:
        assert q["subject"].lower() not in text


def test_supersession_pairs_are_adversarial(corpus):
    """The stale doc must be the LEXICALLY closer match, or the test is not the hard case.

    The failure mode being measured is a stale memory outranking its successor. If the query
    were worded closer to the successor, plain search would get it right and the trust layer
    would have nothing to prove.
    """
    docs = _bodies(corpus.root)
    pairs = [q for q in corpus.queries if q.get("expect") == "successor"]
    assert pairs
    for q in pairs:
        stale = docs[q["stale_ids"][0].rsplit(":", 1)[0]][1].lower()
        succ = docs[q["successor_ids"][0].rsplit(":", 1)[0]][1].lower()
        terms = [t for t in q["query"].lower().split() if len(t) > 3]
        assert sum(t in stale for t in terms) > sum(t in succ for t in terms)


def test_successor_declares_supersedes_pointing_at_the_stale_file(corpus):
    docs = _bodies(corpus.root)
    for q in [q for q in corpus.queries if q.get("expect") == "successor"]:
        stale_rel = q["stale_ids"][0].rsplit(":", 1)[0]
        succ_rel = q["successor_ids"][0].rsplit(":", 1)[0]
        assert docs[succ_rel][0]["supersedes"] == stale_rel.rsplit("/", 1)[-1]


def test_abstain_docs_are_expired(corpus):
    """An expect=abstain query is only correct if its document is genuinely out of window."""
    from datetime import date

    docs = _bodies(corpus.root)
    abstain = [q for q in corpus.queries if q.get("expect") == "abstain"]
    assert abstain
    for q in abstain:
        meta = docs[q["stale_ids"][0].rsplit(":", 1)[0]][0]
        assert date.fromisoformat(str(meta["valid_until"])) < date.today()


def test_query_counts_match_the_request(corpus):
    plain = [q for q in corpus.queries if not q.get("trust")]
    assert sum(1 for q in plain if q["answerable"]) == 12
    assert sum(1 for q in plain if not q["answerable"]) == 6
    assert sum(1 for q in corpus.queries if q.get("expect") == "successor") == 8
    assert sum(1 for q in corpus.queries if q.get("expect") == "abstain") == 4


def test_query_ids_are_unique(corpus):
    ids = [q["id"] for q in corpus.queries]
    assert len(ids) == len(set(ids))


def test_generation_is_deterministic_for_a_seed(tmp_path):
    a = generate(tmp_path / "a", n_answerable=5, n_unanswerable=3, n_successor=3,
                 n_abstain=2, n_filler_chunks=20, seed=11)
    b = generate(tmp_path / "b", n_answerable=5, n_unanswerable=3, n_successor=3,
                 n_abstain=2, n_filler_chunks=20, seed=11)
    assert json.dumps(a.queries) == json.dumps(b.queries)
    assert _bodies(a.root).keys() == _bodies(b.root).keys()
    assert [v for _, v in sorted(_bodies(a.root).items())] == \
           [v for _, v in sorted(_bodies(b.root).items())]


def test_a_different_seed_gives_a_different_corpus(tmp_path):
    a = generate(tmp_path / "a", n_answerable=5, n_unanswerable=3, n_successor=3,
                 n_abstain=2, n_filler_chunks=20, seed=1)
    b = generate(tmp_path / "b", n_answerable=5, n_unanswerable=3, n_successor=3,
                 n_abstain=2, n_filler_chunks=20, seed=2)
    assert json.dumps(a.queries) != json.dumps(b.queries)


def test_reported_chunk_count_is_the_real_one(corpus):
    """`n_chunks` is the corpus's advertised scale — it must equal what indexing will produce.

    Regression test for a real defect: filler paragraphs shorter than `max_chars` are PACKED
    several-to-a-chunk by `chunk_text`, so a request for 50 filler chunks produced 10 while
    still reporting 50. Every per-scale number computed from that corpus would have been
    attributed to a corpus four times its actual size.
    """
    docs = _bodies(corpus.root)
    actual = sum(len(chunk_text(body)) for _, (_, body) in docs.items())
    assert actual == corpus.n_chunks


def test_filler_reaches_the_requested_chunk_count(corpus):
    """Index pressure is the point of the filler — under-delivering it silently shrinks scale."""
    docs = _bodies(corpus.root)
    filler_chunks = sum(
        len(chunk_text(body)) for rel, (_, body) in docs.items() if rel.startswith("filler/")
    )
    assert filler_chunks == 50


def test_filler_never_answers_a_query(corpus):
    """Filler exists to create pressure, not to be retrieved — it must not contain a subject."""
    docs = _bodies(corpus.root)
    filler = " ".join(b for rel, (_, b) in docs.items() if rel.startswith("filler/")).lower()
    for q in corpus.queries:
        assert q["subject"].lower() not in filler


def test_regeneration_does_not_inherit_the_previous_runs_files(tmp_path):
    """Regression: a corpus is a glob, so leftovers are indexed as if they belonged to this run.

    A run configured for 600 chunks silently indexed 20,600 because 100 filler files from an
    earlier, larger run were still in the directory — while `n_chunks` still reported 600. Every
    scale and latency figure was attributed to the wrong corpus size, and nothing failed.
    """
    root = tmp_path / "c"
    big = generate(root, n_answerable=2, n_unanswerable=1, n_successor=1, n_abstain=1,
                   n_filler_chunks=40, seed=3)
    assert len(list((root / "filler").glob("*.md"))) > 0

    small = generate(root, n_answerable=2, n_unanswerable=1, n_successor=1, n_abstain=1,
                     n_filler_chunks=0, seed=3)
    assert not (root / "filler").exists()
    actual = sum(len(chunk_text(body)) for _, (_, body) in _bodies(root).items())
    assert actual == small.n_chunks
    assert small.n_chunks < big.n_chunks


def test_generate_refuses_to_wipe_a_directory_it_did_not_create(tmp_path):
    """The fix for the above must not be a delete-anything footgun: `--out ~/notes` must fail."""
    victim = tmp_path / "notes"
    victim.mkdir()
    (victim / "important.md").write_text("do not delete me", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        generate(victim, n_answerable=1, n_unanswerable=1, n_successor=1, n_abstain=1,
                 n_filler_chunks=0, seed=1)
    assert (victim / "important.md").read_text(encoding="utf-8") == "do not delete me"


def test_generate_accepts_an_empty_directory(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    c = generate(empty, n_answerable=1, n_unanswerable=1, n_successor=1, n_abstain=1,
                 n_filler_chunks=0, seed=1)
    assert c.n_chunks == 4


def test_no_offtopic_subject_word_appears_in_recall_s_own_python_sources():
    """The pool must stay out of the corpora it exists to be disjoint FROM.

    These subjects lived as Python literals in `recall/eval/synthetic.py`, which put all 25 of them
    into any code corpus rooted at this repository. `offtopic_subjects_absent_from` then
    disqualified every one, and the wizard's `code` tenant could not build a gap class at all.
    Measured 2026-08-18 over `recall/**/*.py`: 0 of 25 survived; with that one file excluded, 12.

    The refusal it produced blamed the operator's corpus for overlapping the pool and told them to
    supply a domain-specific subject list, which was a confident and wrong diagnosis. That is the
    real cost: not the failure, but a failure that misdirected.

    This is a guard on the whole tree rather than on one module, because the pool is data and data
    can be pasted back into source by anyone, at any time, in any file.
    """
    from recall.eval.synthetic import _OFFTOPIC_SUBJECTS, _OFFTOPIC_TEMPLATES
    from recall.wizard.queryset import DEFAULT_PER_CLASS, offtopic_subjects_absent_from

    root = Path(__file__).resolve().parents[1] / "recall"
    assert root.is_dir(), "the package must be found for this guard to mean anything"

    sources = sorted(root.rglob("*.py"))
    assert len(sources) > 50, f"only {len(sources)} sources found; the walk is not covering the tree"

    texts = [path.read_text(encoding="utf-8", errors="ignore") for path in sources]
    survivors = offtopic_subjects_absent_from(texts)
    capacity = len(survivors) * len(_OFFTOPIC_TEMPLATES)

    # CAPACITY, not zero overlap. Demanding that no subject word appear anywhere in the tree is
    # both unachievable and the wrong question: `tuning`, `timing`, `methods` and `systems` are
    # ordinary software words and always will be. What the wizard needs is enough SURVIVING
    # subjects to build a full gap class, so that is what is asserted.
    assert capacity >= DEFAULT_PER_CLASS, (
        f"only {len(survivors)} of {len(_OFFTOPIC_SUBJECTS)} off-topic subjects survive against "
        f"recall's own Python sources, giving {capacity} gap questions against the "
        f"{DEFAULT_PER_CLASS} a default run needs. The commonest cause is the pool being pasted "
        "back into source as literals: keep it in `recall/eval/offtopic_subjects.json`, which no "
        "code or docs glob matches."
    )


def test_no_distinctive_pool_word_appears_in_recall_source():
    """The capacity guard has a threshold, so it tolerates erosion. This one detects a QUOTE.

    Ordinary software vocabulary kills subjects one at a time: `tuning`, `timing`, `patterns` and
    `methods` are words any codebase contains, and demanding their absence is neither achievable
    nor useful. What must never appear is the word that carries a subject's off-topic-ness, and the
    pool declares those itself in its `distinctive` field, because the only thing that knows which
    word is the domain anchor is the pool.

    ⚠️ Caught for real, and by the measurement rather than by review. The comment in
    `recall/wizard/queryset.py` explaining that the pool had been poisoned named the three words it
    had collided on, which put them straight back into every code corpus rooted here and
    disqualified those three subjects. Survivors fell from 12 to 11 and the capacity guard stayed
    GREEN, because 55 still clears 40. An example drawn from the pool, written anywhere under
    `recall/`, is not an example. It is corpus.

    An earlier version of this test counted subjects disqualified per file and failed on `store.py`
    and `index.py`, which contain `tuning` and `timing` innocently. Rarity is the signal, not count.
    """
    from recall.eval.synthetic import _OFFTOPIC_DATA, _OFFTOPIC_SUBJECTS
    from recall.eval.vocab import word_tokens

    distinctive = _OFFTOPIC_DATA["distinctive"]
    assert len(distinctive) == len(_OFFTOPIC_SUBJECTS), (
        "every subject needs an anchor word, or this guard silently covers only some of the pool"
    )

    root = Path(__file__).resolve().parents[1] / "recall"
    sources = sorted(root.rglob("*.py"))
    assert len(sources) > 50, f"only {len(sources)} sources found; the walk is not covering the tree"

    offenders: dict[str, list[str]] = {}
    for path in sources:
        words = set(word_tokens([path.read_text(encoding="utf-8", errors="ignore")]))
        if hits := sorted(w for w in distinctive if w in words):
            offenders[path.relative_to(root).as_posix()] = hits

    assert not offenders, (
        f"distinctive off-topic pool words found in recall source: {offenders}. Each one silently "
        "disqualifies its subject for every code corpus rooted at this repository. Refer to "
        "`recall/eval/offtopic_subjects.json` rather than quoting a subject in code or a comment."
    )


def test_no_offtopic_subject_overlaps_the_generated_corpus_vocabulary():
    """The module states this invariant and never checked it, and it was false.

    The comment above the pool says the subjects are "deliberately from a domain the corpus never
    mentions — no overlap with the subject adjectives/nouns, the aspect names, or the filler
    vocabulary". `saffron` was in `_ADJECTIVES` AND in the subject "saffron harvesting by hand", so
    that gap query was about a word the generated corpus actually contains.

    That is the failure the pool exists to prevent, stated in this module's own words: a gap class
    sharing the corpus vocabulary is not separable, and any abstention rate measured on it means
    nothing. An invariant asserted in a comment is not asserted.
    """
    from recall.eval.synthetic import (
        _ADJECTIVES,
        _ASPECTS,
        _FILLER_NOUNS,
        _NOUNS,
        _OFFTOPIC_SUBJECTS,
    )
    from recall.eval.vocab import word_tokens

    generated: set[str] = set(_ADJECTIVES) | set(_NOUNS) | set(_FILLER_NOUNS)
    for aspect, unit in _ASPECTS:
        generated |= set(word_tokens([aspect, unit]))

    offenders = {
        subject: hits
        for subject in _OFFTOPIC_SUBJECTS
        if (hits := [w for w in word_tokens([subject]) if len(w) > 3 and w in generated])
    }
    assert not offenders, (
        f"off-topic subjects sharing vocabulary with the generated corpus: {offenders}. A gap "
        "query about a word the corpus contains is not a gap query, and every abstention rate "
        "measured on this set would be measuring the wrong thing."
    )
