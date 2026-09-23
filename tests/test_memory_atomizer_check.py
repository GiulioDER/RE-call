"""Offline invariants of the memory-tenant atomizer check.

Red proofs (2026-09-23, each mutation applied to ``scripts/memory_atomizer_check.py`` and
reverted):

* ``test_sampling_never_uses_a_consumed_source_and_golds_every_containing_chunk`` failed when
  ``sample_memory_spans`` ignored ``excluded`` (``source not in excluded`` removed), and, with the
  exclusion restored, when gold kept only the sampled chunk.
* ``test_excluded_sources_are_found_at_any_depth`` failed when ``walk`` stopped recursing into
  lists.
* ``test_micro_views_are_word_ranges_of_their_own_chunk`` failed when the window size was taken
  from the C8 word window (160) instead of ``MICRO_SIZE``.
* ``test_paired_counts_gains_and_losses_against_the_first_arm`` failed when losses were counted
  with the arms swapped.
"""

from __future__ import annotations

from scripts import memory_atomizer_check as check


def _chunk(cid: str, source: str, ordinal: int, text: str) -> check.StoredChunk:
    return check.StoredChunk(cid, source, ordinal, text)


def _corpus() -> list[check.StoredChunk]:
    chunks = []
    for s in range(30):
        source = f"recall/memo-{s}.md"
        body = " ".join(f"s{s}w{i}" for i in range(80))
        chunks.append(_chunk(f"c{s}-0", source, 0, body))
        # The second chunk repeats the first chunk's words 40..79, as overlapping chunkers do.
        chunks.append(_chunk(f"c{s}-1", source, 1, " ".join(body.split()[40:]) + " tail words here"))
    return chunks


def test_sampling_never_uses_a_consumed_source_and_golds_every_containing_chunk() -> None:
    corpus = _corpus()
    excluded = {f"recall/memo-{s}.md" for s in range(0, 30, 2)}
    spans = check.sample_memory_spans(corpus, excluded, sources=30)
    assert spans
    assert not {span.source for span in spans} & excluded
    text_of = {chunk.chunk_id: chunk.text for chunk in corpus}
    for span in spans:
        assert span.chunk_id in span.gold_chunk_ids
        containing = {
            chunk.chunk_id
            for chunk in corpus
            if chunk.source == span.source and span.text in text_of[chunk.chunk_id]
        }
        assert set(span.gold_chunk_ids) == containing
    assert any(len(span.gold_chunk_ids) == 2 for span in spans)


def test_excluded_sources_are_found_at_any_depth() -> None:
    records = [
        {"rows": [{"gold_sources": ["recall/a.md"], "nested": {"source": "steel/b.md"}}]},
        [{"gold_source": "cca-demos/c.md"}, {"source": "not-a-path"}],
    ]
    assert check.excluded_sources(records) == {"recall/a.md", "steel/b.md", "cca-demos/c.md"}


def test_micro_views_are_word_ranges_of_their_own_chunk() -> None:
    corpus = _corpus()[:4]
    views = check.micro_chunk_views(corpus)
    assert views
    for chunk, _, text in views:
        assert len(text.split()) <= check.MICRO_SIZE
        assert text in chunk.text
    assert len(views) > len(corpus) * 3
    per_source: dict[str, list[str]] = {}
    for chunk, _, text in views:
        per_source.setdefault(chunk.source, []).append(text)
    assert all(len(texts) == len(set(texts)) for texts in per_source.values())


def test_paired_counts_gains_and_losses_against_the_first_arm() -> None:
    base = [1, 7, None, 3, 8]
    arm = [2, 5, 6, 9, None]
    assert check.paired(base, arm, 6) == {"gains": 2, "losses": 1, "net": 1}
