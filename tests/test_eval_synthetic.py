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
