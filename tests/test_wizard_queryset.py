"""Generating the labelled query set that calibration needs, without asking a human to write it.

Calibration binds to a generation and needs >=20 answerable and >=20 unanswerable labelled
queries. Nothing in the repository produced them; `recall setup` asks for a path to a file the user
must already have. That is the step this exists to remove.

The design constraint is a measurement, not an intuition. `recall/eval/synthetic.py:69-75` records
that building an unanswerable query by suffixing a nonsense token onto an answerable one produced a
set that was NOT separable at all — median top cosine 0.830 against answerable 0.923, with 0% below
the weakest answerable query. Unanswerable queries must therefore come from a genuinely disjoint
domain, and these tests pin that rather than trusting it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from recall.calibration import MIN_CALIBRATION_SAMPLES
from recall.wizard.queryset import (
    DEFAULT_PER_CLASS,
    QuerySetError,
    canonicalize,
    chunks_from_directory,
    generate_offline,
    offtopic_subjects_absent_from,
    prepare_for_calibration,
    require_balance,
)

CORPUS = [
    "# Retry policy\n\nRequests are retried on 5xx responses with exponential backoff. "
    "Client errors in the 4xx range are never retried, because a malformed request stays "
    "malformed.\n",
    "# Read-through cache\n\nThe read-through cache reduced p99 latency from 840ms to 120ms. "
    "Entries expire after 300 seconds and are evicted under memory pressure.\n",
    "# Storage engine choice\n\nPostgres was chosen over a document store because the access "
    "pattern is relational and the team already operated Postgres in production.\n",
    "# Deployment topology\n\nEach region runs three replicas behind an anycast load balancer. "
    "Failover is automatic and takes under ninety seconds.\n",
    "# Rate limiting\n\nThe limiter uses a token bucket per tenant, refilled at a steady rate. "
    "Burst capacity is twice the sustained rate.\n",
]


# --------------------------------------------------------------------------------------
# canonicalize: the shape v2's loader actually accepts
# --------------------------------------------------------------------------------------


def test_canonicalize_output_is_accepted_by_the_v2_loader(tmp_path) -> None:
    """The only shape test that counts: the real loader parses it.

    `canonical_query_set` requires a non-empty `query` and a BOOLEAN `answerable` on every entry,
    refuses duplicates, and refuses an empty array.
    """
    import json

    from recall.calibration_v2 import load_query_set

    entries = [{"query": f"question {i}", "answerable": i % 2 == 0} for i in range(40)]
    out = tmp_path / "q.json"
    out.write_text(json.dumps(prepare_for_calibration(entries, min_per_class=20)), encoding="utf-8")

    # `load_query_set` returns (canonical_entries, digest). The digest is what binds a calibration
    # artifact to the exact question set it was measured on, so a set that parses but produces no
    # digest would be useless downstream.
    loaded, digest = load_query_set(out)
    assert len(loaded) == 40
    assert len(digest) == 64
    assert sum(1 for e in loaded if e["answerable"]) == 20


def test_trust_entries_are_stripped() -> None:
    """Both `recall/eval/queries.json` and `synthetic.py` emit them, and v2's loader refuses them.

    A `trust` entry carries no `answerable` key at all, so `canonical_query_set` raises on the
    whole file rather than skipping the entry. The legacy `recall calibrate` path filters them
    (`recall/setup.py:1146`); v2's loader fails first, so a set handed to v2 must be filtered here.
    """
    entries = [{"query": "a", "answerable": True}, {"trust": True, "expect": "successor", "query": "b"}]
    got = canonicalize(entries)
    assert got == [{"query": "a", "answerable": True}]


def test_duplicate_queries_are_removed_not_passed_through() -> None:
    """`canonical_query_set` refuses a duplicate outright, so a generator must not emit one."""
    entries = [
        {"query": "same question", "answerable": True},
        {"query": "  Same Question  ", "answerable": True},
        {"query": "other", "answerable": False},
    ]
    got = canonicalize(entries)
    assert len(got) == 2


def test_a_non_boolean_answerable_is_refused() -> None:
    """`answerable: "yes"` is truthy in Python and rejected by the loader. Catch it here."""
    with pytest.raises(QuerySetError, match="boolean"):
        canonicalize([{"query": "a", "answerable": "yes"}])


def test_an_empty_query_is_refused() -> None:
    with pytest.raises(QuerySetError, match="non-empty"):
        canonicalize([{"query": "   ", "answerable": True}])


def test_a_class_below_the_floor_is_refused_naming_both_counts() -> None:
    """Certification needs 20 per class. Failing here beats failing after a full retrieval run."""
    entries = [{"query": f"q{i}", "answerable": True} for i in range(25)]
    entries += [{"query": f"u{i}", "answerable": False} for i in range(3)]
    with pytest.raises(QuerySetError) as exc:
        require_balance(canonicalize(entries))
    assert "25" in str(exc.value) and "3" in str(exc.value)


def test_the_floor_is_the_librarys_own_constant() -> None:
    """Not a second copy of 20 that can drift away from certification's real requirement."""
    from recall.wizard import queryset

    assert queryset.MIN_PER_CLASS is MIN_CALIBRATION_SAMPLES


def test_only_the_three_recognised_keys_survive() -> None:
    """`canonical_query_set` drops everything else anyway; carrying extras invites a digest that
    looks stable while its inputs are not."""
    got = canonicalize(
        [{"query": "a", "answerable": True, "relevant_ids": ["x:0"], "note": "drop me"}]
    )
    assert got == [{"query": "a", "answerable": True, "relevant_ids": ["x:0"]}]


# --------------------------------------------------------------------------------------
# The offline generator
# --------------------------------------------------------------------------------------


def test_offline_generates_a_balanced_certifiable_shaped_set() -> None:
    entries = generate_offline(CORPUS, per_class=3)
    answerable = [e for e in entries if e["answerable"]]
    unanswerable = [e for e in entries if not e["answerable"]]
    assert len(answerable) == len(unanswerable) == 3
    assert len({e["query"] for e in entries}) == 6


def test_offline_is_deterministic_for_a_seed() -> None:
    """The measurement re-runs this; two runs must produce the same artifact."""
    assert generate_offline(CORPUS, per_class=3, seed=7) == generate_offline(
        CORPUS, per_class=3, seed=7
    )


def test_unanswerable_query_SUBJECTS_share_no_content_word_with_the_corpus() -> None:
    """The measured constraint, asserted over the part the generator actually controls.

    An earlier version of this test asserted it over the whole rendered query and passed — but
    only because the five-chunk fixture happened to avoid the template vocabulary. On this
    repository's real `docs/`, 40 of 40 gap queries share a content word with the corpus, because
    the five templates contribute "explains", "actually", "work", "difficult", "predict",
    "leading", "theory" and "described", and `offtopic_subjects_absent_from` filters the SUBJECT
    only. The test claimed a guarantee the code does not make.

    Template overlap is not the hazard, and that is measured rather than assumed:
    `recall/eval/synthetic.py:74` records that questions built from these exact templates sit at
    median top cosine 0.570 with 78% below the answerable floor. What destroyed separability was
    sharing the SUBJECT vocabulary — 0.830 against 0.923, with nothing below the floor. So the
    subject is what must be disjoint, and the subject is what this asserts.
    """
    from recall.eval.vocab import word_tokens

    from recall.wizard.queryset import offtopic_subjects_absent_from

    corpus_words = set(word_tokens(CORPUS))
    subjects = offtopic_subjects_absent_from(CORPUS)
    assert subjects, "precondition: the fixture must not exhaust the off-topic pool"

    for subject in subjects:
        content = [w for w in word_tokens([subject]) if len(w) > 3]
        overlap = corpus_words.intersection(content)
        assert not overlap, f"subject {subject!r} reuses corpus words {sorted(overlap)}"

    # And every gap query really is built from one of those cleared subjects.
    for entry in generate_offline(CORPUS, per_class=3):
        if entry["answerable"]:
            continue
        assert any(s in entry["query"] for s in subjects), entry["query"]


def test_answerable_queries_carry_terms_that_are_distinctive_not_generic() -> None:
    """A query built from words common to every chunk retrieves nothing in particular."""
    entries = [e for e in generate_offline(CORPUS, per_class=3) if e["answerable"]]
    from recall.eval.vocab import word_tokens

    for entry in entries:
        words = set(word_tokens([entry["query"]]))
        # At least one word that appears in exactly one chunk of the corpus.
        rare = [
            w
            for w in words
            if sum(1 for chunk in CORPUS if w in chunk.lower()) == 1
        ]
        assert rare, f"{entry['query']!r} carries no term unique to a single chunk"


def test_answerable_queries_contain_no_identifiers_or_symbols() -> None:
    """Ranking by rarity alone surfaced symbols, which measure string matching, not meaning.

    Run against this repository's real `docs/`, the first version produced
    "why was doubly sources_not_found fell decided this way" and
    "what is the behaviour of id_rsa promo_ uuid8". Every one of those tokens is genuinely rare,
    and none is a topic; a question built from them retrieves the one chunk holding that symbol,
    which looks like excellent separability while measuring nothing about retrieval quality.
    """
    corpus = [
        "# Tenant isolation\n\nThe helper `_tenant_isolation` and the test "
        "`test_parity_reports_a_missing_shadow` guard uuid8 and promo_ and id_rsa handling "
        "across every tenant boundary in the system.\n"
    ] + CORPUS
    answerable = [e["query"] for e in generate_offline(corpus, per_class=4) if e["answerable"]]

    # Precondition, asserted rather than assumed. Which chunks get sampled depends on
    # `random.sample`, so without this the loop below could run over three clean queries and pass
    # having never seen the identifier chunk at all.
    assert any("tenant isolation" in q for q in answerable), answerable

    for query in answerable:
        for word in query.split():
            assert "_" not in word, f"identifier leaked into {query!r}"
            assert not any(c.isdigit() for c in word), f"symbol leaked into {query!r}"


def test_a_heading_is_preferred_over_inferred_terms() -> None:
    """A heading is the one place a document states its own topic."""
    corpus = [
        "# Token bucket rate limiting\n\nThe limiter refills steadily and permits a burst of "
        "twice the sustained rate before shedding load.\n",
        *CORPUS,
    ]
    answerable = [e["query"] for e in generate_offline(corpus, per_class=6) if e["answerable"]]
    assert any("token bucket rate limiting" in q for q in answerable)


def test_a_corpus_the_offtopic_list_collides_with_is_refused() -> None:
    """If the corpus IS about glaciers, the gap class must not be glacier questions.

    Refused rather than silently narrowed: a set built from the few surviving subjects would be
    repetitive, and duplicate gap queries understate the variance of any rate measured on them —
    the reason `synthetic.generate` refuses to reuse a subject.
    """
    glacier_corpus = [
        f"# Note {i}\n\n" + " ".join(subject for subject in _ALL_OFFTOPIC_SUBJECTS)
        for i in range(5)
    ]
    with pytest.raises(QuerySetError, match="off-topic|disjoint"):
        generate_offline(glacier_corpus, per_class=3)


def test_too_few_chunks_is_refused_rather_than_recycled() -> None:
    """Reusing a chunk would emit duplicate queries, which the loader refuses anyway."""
    with pytest.raises(QuerySetError, match="chunk"):
        generate_offline(CORPUS[:2], per_class=10)


def test_the_default_is_double_the_certification_floor() -> None:
    """40 per class, so the Hanley-Menil lower bound is not the binding constraint."""
    assert DEFAULT_PER_CLASS == 2 * MIN_CALIBRATION_SAMPLES


def test_generated_output_survives_canonicalize_unchanged() -> None:
    """The generator must already emit the canonical shape, not rely on being cleaned up."""
    entries = generate_offline(CORPUS, per_class=3)
    assert prepare_for_calibration(entries, min_per_class=3) == entries


# --------------------------------------------------------------------------------------
# Reading chunks the same way indexing will
# --------------------------------------------------------------------------------------


def test_chunks_come_from_the_same_chunker_indexing_uses(tmp_path) -> None:
    """A query generated from text that was never a chunk cannot retrieve that chunk."""
    from recall.index import chunk_text

    root = tmp_path / "docs"
    root.mkdir()
    body = "# Title\n\n" + ("paragraph one. " * 60) + "\n\n" + ("paragraph two. " * 60) + "\n"
    (root / "a.md").write_text(body, encoding="utf-8")

    assert chunks_from_directory(root) == chunk_text(body)


def test_chunks_from_directory_refuses_an_empty_corpus(tmp_path) -> None:
    """`QuerySetError` specifically, not any ValueError: this module documents that type, and a
    caller writing `except QuerySetError` must actually catch the common failures."""
    root = tmp_path / "empty"
    root.mkdir()
    with pytest.raises(QuerySetError, match="nothing to build"):
        chunks_from_directory(root)


def test_a_missing_directory_names_itself_rather_than_blaming_the_glob(tmp_path) -> None:
    """`candidate_files` reaches its non-directory branch for a path that does not exist, and its
    message is about the glob — which sends the reader to fix the wrong thing."""
    with pytest.raises(QuerySetError, match="does not exist"):
        chunks_from_directory(tmp_path / "nope")


def test_an_unreadable_file_is_not_silently_dropped(tmp_path, monkeypatch) -> None:
    """A partly unreadable corpus must not shrink in silence.

    Swallowing the OSError made "every file is unreadable" indistinguishable from "there are no
    files", and made a partial read indistinguishable from a complete one — so questions would be
    generated about only the readable part with nothing saying so.
    """
    root = tmp_path / "docs"
    root.mkdir()
    (root / "a.md").write_text("# A\n\nreadable content here.\n", encoding="utf-8")
    (root / "b.md").write_text("# B\n\nalso readable.\n", encoding="utf-8")

    real_read = Path.read_text

    def deny(self, *args, **kwargs):
        if self.name == "b.md":
            raise PermissionError(13, "Permission denied", str(self))
        return real_read(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", deny)
    with pytest.raises(QuerySetError) as exc:
        chunks_from_directory(root)
    assert "b.md" in str(exc.value)
    assert "1 of 2" in str(exc.value)


def test_a_file_that_vanished_mid_walk_is_skipped_not_fatal(tmp_path, monkeypatch) -> None:
    """A wizard runs against a live user directory, where files legitimately disappear.

    Distinguished from the unreadable case above: a file that is GONE is not part of the corpus,
    which is how `index_path` treats it, while a file that is still there and cannot be read means
    the corpus is silently smaller than it looks.
    """
    root = tmp_path / "docs"
    root.mkdir()
    (root / "a.md").write_text("# A\n\nreadable content here.\n", encoding="utf-8")
    (root / "b.md").write_text("# B\n\nswap file, about to vanish.\n", encoding="utf-8")

    real_read = Path.read_text

    def vanish(self, *args, **kwargs):
        if self.name == "b.md":
            raise FileNotFoundError(2, "No such file or directory", str(self))
        return real_read(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", vanish)
    chunks = chunks_from_directory(root)
    assert chunks and all("swap file" not in c for c in chunks)


def test_an_all_vanished_corpus_does_not_blame_the_glob(tmp_path, monkeypatch) -> None:
    """An unmounted share finds every file in the walk and none in the read.

    Without the count in the message that reports as "no chunks matching '**/*.md'", which sends
    the reader to fix a pattern that was correct.
    """
    root = tmp_path / "docs"
    root.mkdir()
    for name in ("a.md", "b.md"):
        (root / name).write_text(f"# {name}\n\nbody.\n", encoding="utf-8")

    def gone(self, *args, **kwargs):
        raise FileNotFoundError(2, "No such file or directory", str(self))

    monkeypatch.setattr(Path, "read_text", gone)
    with pytest.raises(QuerySetError) as exc:
        chunks_from_directory(root)
    assert "2 of 2" in str(exc.value)
    assert "disappeared" in str(exc.value)


def test_the_sample_spreads_across_the_corpus_rather_than_its_head() -> None:
    """Oversampling plus `sorted(pool)` took only the pool's lowest-indexed quarter.

    Measured on this repository's `docs/` at the default: sorted order drew all 40 questions from
    3 of 51 files, sampled order from 21 of 51 — the exact head-of-corpus bias the sampling
    comment says it exists to prevent, reintroduced by the oversampling fix.

    The first version of this test was a guard that could not fail. Its token helper had period
    100, so `_word(i) == _word(i + 100)`: every match counted twice, and `max(positions) > 100`
    was satisfied by any match at all. It passed verbatim against the buggy code. The token is now
    unique across the whole range, and the margin is real — sorted order reaches position 33..64,
    sampled order 166..199.
    """
    corpus = [f"# Section {_word(i)}\n\nThe {_word(i)} subsystem behaves distinctly.\n" for i in range(200)]
    assert len({_word(i) for i in range(200)}) == 200, "tokens must identify one chunk each"

    answerable = [e["query"] for e in generate_offline(corpus, per_class=20, seed=0) if e["answerable"]]
    positions = [i for i in range(200) if any(_word(i) in q for q in answerable)]

    assert len(positions) == 20, f"{len(positions)} chunks traced, expected exactly 20"
    assert max(positions) > 100, f"sample never reached past chunk {max(positions)} of 200"


def _word(i: int) -> str:
    """A pronounceable token unique for every `i` below 1000, alphabetic so `_is_prose` keeps it.

    Three syllables, not two. Two gave a period of 100 and made the test above vacuous.
    """
    syllables = ("ka", "lo", "mi", "ru", "ta", "ne", "vo", "shi", "pe", "du")
    return syllables[i // 100 % 10] + syllables[i // 10 % 10] + syllables[i % 10] + "x"


# --------------------------------------------------------------------------------------
# relevant_ids, which the LLM generator will supply and the loader type-checks
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["doc:1", ["a", 2], {"a": 1}, 7])
def test_a_badly_typed_relevant_ids_is_refused_here_not_downstream(bad) -> None:
    """`canonical_query_set` requires list[str] and aborts the WHOLE file on a bad one."""
    with pytest.raises(QuerySetError, match="relevant_ids"):
        canonicalize([{"query": "a", "answerable": True, "relevant_ids": bad}])


def test_a_null_relevant_ids_is_dropped_rather_than_refused() -> None:
    """`"relevant_ids": null` is the likely shape from a model asked for an optional field."""
    assert canonicalize([{"query": "a", "answerable": True, "relevant_ids": None}]) == [
        {"query": "a", "answerable": True}
    ]


def test_a_trust_entry_that_is_also_labelled_is_kept() -> None:
    """Dropping on the flag alone silently discarded a well-formed labelled entry, and the loss
    then surfaced as a class-count shortfall blamed on the corpus."""
    got = canonicalize([{"query": "a", "answerable": True, "trust": True}])
    assert got == [{"query": "a", "answerable": True}]


def test_a_non_object_entry_is_refused_rather_than_skipped() -> None:
    """Skipping turned a JSON array of strings into '0 answerable and 0 unanswerable'."""
    with pytest.raises(QuerySetError, match="not an object"):
        canonicalize(["just a string"])


# --------------------------------------------------------------------------------------
# The vocabulary guard, exercised directly
# --------------------------------------------------------------------------------------


def test_offtopic_subjects_absent_from_filters_on_content_words() -> None:
    """Named through the pool rather than spelled out, and that is not only tidiness.

    A literal subject in a test file is a literal subject in the CORPUS whenever a code corpus is
    rooted at this repository, and the filter then disqualifies it for everyone. That is exactly
    how the pool came to disqualify all 25 of its own subjects: they were Python literals in
    `recall/eval/synthetic.py`. Referencing the pool keeps this test honest and keeps `tests/` out
    of the vocabulary the filter has to avoid.
    """
    from recall.eval.synthetic import _OFFTOPIC_SUBJECTS

    overlapping = _OFFTOPIC_SUBJECTS[1]
    subjects = offtopic_subjects_absent_from([f"a note about {overlapping} at home"])
    assert overlapping not in subjects
    assert len(subjects) > 10, "one collision must not empty the pool"


_ALL_OFFTOPIC_SUBJECTS: list[str] = []


def setup_module() -> None:
    from recall.eval.synthetic import _OFFTOPIC_SUBJECTS

    _ALL_OFFTOPIC_SUBJECTS.extend(_OFFTOPIC_SUBJECTS)
