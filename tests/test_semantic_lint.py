"""Write-time SEMANTIC lint — the missing-edge catch the static lint can't do.

The static `recall lint` checks edges you *wrote* (dangling / cycle / version-sibling). It is
blind to the edge you *forgot*: a new memo that is really about a prior settled decision but
never declares `supersedes:`. That relation is not in the frontmatter — it's in the meaning —
so only retrieval can find it. This is the write-time mirror of anti-re-litigation: query the
index with the new memo before committing; any high-similarity CLOSED decision it does not
reference is a candidate unlinked chain, surfaced for a one-keystroke confirm.

Pure core (no DB, no embedder) tested exhaustively here; the DB driver is proven on a planted
orphan corpus (requires_db).
"""
from __future__ import annotations

from recall.semantic_lint import (
    ChainCandidate,
    UnlinkedChain,
    find_unlinked_chains,
    is_closed_decision,
)

from tests.conftest import TEST_DSN, requires_db


def _c(name, cosine, closed=True):
    return ChainCandidate(name=name, cosine=cosine, is_closed_decision=closed)


# ── pure core: find_unlinked_chains ─────────────────────────────────────────

def test_flags_high_sim_closed_decision_not_referenced():
    cands = [_c("cache_ttl_v1.md", 0.82)]
    out = find_unlinked_chains("cache_ttl_v3.md", supersedes=set(), candidates=cands,
                               threshold=0.70)
    assert out == [UnlinkedChain("cache_ttl_v3.md", "cache_ttl_v1.md", 0.82)]


def test_skips_referenced_candidate():
    # the new memo already declares supersedes: cache_ttl_v2.md — no missing edge
    cands = [_c("cache_ttl_v2.md", 0.90)]
    out = find_unlinked_chains("cache_ttl_v3.md", supersedes={"cache_ttl_v2.md"},
                               candidates=cands, threshold=0.70)
    assert out == []


def test_skips_self():
    cands = [_c("a.md", 0.99)]
    assert find_unlinked_chains("a.md", set(), cands, 0.70) == []


def test_skips_below_threshold():
    cands = [_c("a.md", 0.55)]
    assert find_unlinked_chains("new.md", set(), cands, 0.70) == []


def test_skips_open_hypothesis_only_closed_decisions_flag():
    # an OPEN investigation legitimately coexists with a new related memo — not a missing edge
    cands = [_c("h021_reranking.md", 0.88, closed=False)]
    assert find_unlinked_chains("new.md", set(), cands, 0.70) == []


def test_sorts_by_cosine_descending():
    cands = [_c("a.md", 0.72), _c("b.md", 0.91), _c("c.md", 0.80)]
    out = find_unlinked_chains("new.md", set(), cands, 0.70)
    assert [u.prior for u in out] == ["b.md", "c.md", "a.md"]


# ── prior-decision detection: is_closed_decision ────────────────────────────

def test_is_closed_decision_true_for_settled_status():
    assert is_closed_decision("Rate limit is 100 rps. Status: adopted.")
    assert is_closed_decision("The experiment failed. Status: falsified.")
    assert is_closed_decision("This lane is closed.\nStatus: closed")


def test_is_closed_decision_true_for_closure_prose():
    assert is_closed_decision("This replaces the original lazy expiry choice.")
    assert is_closed_decision("Superseded by the new gateway approach.")


def test_is_closed_decision_false_for_open_or_plain():
    # an open hypothesis is not a settled decision
    assert not is_closed_decision("Early signal positive. Status: open, collecting data.")
    assert not is_closed_decision("Reciprocal Rank Fusion combines several ranked lists.")


# ── DB driver: semantic_lint over a planted-orphan corpus ───────────────────

def _write(d, name, text):
    (d / name).write_text(text, encoding="utf-8")


def _planted_corpus(tmp_path):
    d = tmp_path / "corpus"
    d.mkdir()
    # correctly-linked pair (control: must NOT flag)
    _write(d, "retry_v1.md",
           "# Retry policy\nOutbound calls retry three times with backoff. Status: adopted.")
    _write(d, "retry_v2.md",
           "---\nsupersedes: retry_v1.md\n---\n# Retry policy revision\n"
           "Retries reduced to a single attempt after the thundering-herd incident. "
           "This replaces the three-attempt policy.")
    # PLANTED ORPHAN: a revision of the pricing-cache decision that forgets the supersedes link
    _write(d, "pricing_cache_v1.md",
           "# Pricing snapshot cache TTL\nThe pricing snapshot cache TTL is 15 minutes, "
           "refreshed lazily on first read after expiry. Status: adopted.")
    _write(d, "pricing_cache_rev.md",
           "# Snapshot caching update\nSnapshot pricing cache entries now expire after 60 "
           "seconds, refreshed proactively by a background worker instead of lazily.")
    # unrelated settled decision (control: must NOT flag — nothing similar)
    _write(d, "backups.md",
           "# Backups\nDatabase backups run every six hours with thirty-day retention. "
           "Status: adopted.")
    return d


@requires_db
def test_semantic_lint_flags_only_the_planted_orphan(tmp_path):
    import pytest

    from recall.semantic_lint import semantic_lint

    try:
        from recall.embeddings import FastEmbedEmbedder

        emb = FastEmbedEmbedder()
    except Exception:  # pragma: no cover - fastembed extra not installed
        pytest.skip("fastembed not installed (recall[fastembed])")

    corpus = _planted_corpus(tmp_path)
    # threshold 0.70 = bge-small's calibrated value (FINDINGS §2). Below it (e.g. 0.60) the
    # unrelated pairs land in bge-small's noise floor — the exact non-transfer this whole
    # project is about — so the threshold MUST be calibrated per embedder.
    findings = semantic_lint(TEST_DSN, emb, corpus_dir=corpus, threshold=0.70)

    flagged = {(f.new_memo, f.prior) for f in findings}
    # the orphan revision surfaces its unreferenced predecessor, and nothing else survives:
    assert flagged == {("pricing_cache_rev.md", "pricing_cache_v1.md")}
    # the correctly-linked pair is never flagged (retry_v2 references retry_v1, either direction)
    assert not any("retry" in f.prior for f in findings)
    # the unrelated settled decision is never flagged
    assert not any(f.prior == "backups.md" for f in findings)


@requires_db
def test_long_multichunk_memo_still_finds_its_prior(tmp_path):
    # BUG-001 (audit): self-hits are excluded only AFTER the retriever truncates to top-k, so a
    # memo long enough to produce >= k chunks fills every slot with its own chunks and the real
    # prior is dropped — a false negative that scales with memo length (the long-memo case the
    # tool targets). Must survive that.
    import pytest

    try:
        from recall.embeddings import FastEmbedEmbedder

        emb = FastEmbedEmbedder()
    except Exception:  # pragma: no cover
        pytest.skip("fastembed not installed (recall[fastembed])")

    from recall.semantic_lint import semantic_lint

    d = tmp_path / "corpus"
    d.mkdir()
    (d / "pricing_cache_v1.md").write_text(
        "# Pricing snapshot cache TTL\nThe pricing snapshot cache TTL is 15 minutes, refreshed "
        "lazily on first read after expiry. Status: adopted.", encoding="utf-8")
    # a long revision of the SAME decision (~15 chunks at 800 chars) that omits the supersedes
    sentence = ("Snapshot pricing cache entries now expire after 60 seconds and are refreshed "
                "proactively by a background worker instead of the old lazy read path. ")
    (d / "pricing_cache_rev.md").write_text(
        "# Snapshot caching update\n" + sentence * 90, encoding="utf-8")

    findings = semantic_lint(TEST_DSN, emb, corpus_dir=d, threshold=0.70)
    assert any(f.new_memo == "pricing_cache_rev.md" and f.prior == "pricing_cache_v1.md"
               for f in findings)


@requires_db
def test_duplicate_basenames_in_subdirs_are_not_misqueried(tmp_path):
    # DAT-001 (audit): keying per-file state by basename shadows same-named files in different
    # subdirs (the BUG-004 class the static lint fixed by keying on rel-path). A shadowed file
    # must not be queried with another file's body and fabricate a finding.
    import pytest

    try:
        from recall.embeddings import FastEmbedEmbedder

        emb = FastEmbedEmbedder()
    except Exception:  # pragma: no cover
        pytest.skip("fastembed not installed (recall[fastembed])")

    from recall.semantic_lint import semantic_lint

    d = tmp_path / "corpus"
    (d / "a").mkdir(parents=True)
    (d / "b").mkdir(parents=True)
    # a prior closed decision about pricing cache
    (d / "pricing_cache_v1.md").write_text(
        "# Pricing snapshot cache TTL\nThe pricing snapshot cache TTL is 15 minutes. "
        "Status: adopted.", encoding="utf-8")
    # two DIFFERENT files sharing basename topic.md: a/ is about backups, b/ about pricing cache
    (d / "a" / "topic.md").write_text(
        "# Backup retention\nDatabase backups run every six hours, thirty-day retention. "
        "Status: adopted.", encoding="utf-8")
    (d / "b" / "topic.md").write_text(
        "# Snapshot caching update\nSnapshot pricing cache entries now expire after 60 seconds, "
        "refreshed proactively instead of lazily.", encoding="utf-8")
    # a clean, uniquely-named orphan that MUST still be caught
    (d / "clean_rev.md").write_text(
        "# Cache refresh change\nThe pricing snapshot cache now refreshes proactively every 60 "
        "seconds rather than lazily on read.", encoding="utf-8")

    findings = semantic_lint(TEST_DSN, emb, corpus_dir=d, threshold=0.70)
    # the ambiguous basename must never appear as a subject (it can't be disambiguated)
    assert not any(f.new_memo == "topic.md" for f in findings)
    # the unambiguous orphan is still surfaced
    assert any(f.new_memo == "clean_rev.md" and f.prior == "pricing_cache_v1.md"
               for f in findings)


@requires_db
def test_cli_lint_semantic_reports_the_orphan(tmp_path, capsys):
    import pytest

    try:
        # instantiate, not just import: FastEmbedEmbedder imports fine without the extra and
        # only raises ImportError in __init__ (which is what main() triggers below)
        from recall.embeddings import FastEmbedEmbedder

        FastEmbedEmbedder()
    except Exception:  # pragma: no cover
        pytest.skip("fastembed not installed (recall[fastembed])")

    from recall.cli import main

    corpus = _planted_corpus(tmp_path)
    main(["--dsn", TEST_DSN, "--embedder", "fastembed",
          "lint", str(corpus), "--semantic", "--threshold", "0.70"])
    out = capsys.readouterr().out
    assert "unlinked-chain" in out
    assert "pricing_cache_v1.md" in out
    assert "retry_v1.md" not in out  # the linked pair stays quiet
