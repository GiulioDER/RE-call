"""The full chain, which no single branch ever ran end to end.

    ExtractedClaim -> InferenceProposal -> ReviewedProposal -> PromotedFact -> file edit
      truth_extraction   _extracted.py     promotion.py       promotion.py     rewrite.py

Every link existed on some branch before this consolidation. None ran all four, so the seams
between them were never exercised by anything.

Properties, one test each:

1. An extracted claim arrives as a proposal that `requires_review`, with no confidence.
2. Promotion refuses without a named reviewer.
3. Promotion refuses without an audit note.
4. A promoted supersession rewrites the SUPERSEDING memo and names the superseded one.
5. The superseded memo is not touched. Inverting the direction demotes the live memo beneath
   the one it replaced, the exact failure the trust layer exists to prevent.
6. A rewrite refuses anything that did not pass promotion.
7. The written edge is one the trust layer can actually read back.
8. A claim the ladder refused never reaches the corpus at all.
"""
from datetime import datetime, timezone
from pathlib import Path

import pytest

from recall.frontmatter import parse_frontmatter, supersedes_key
from recall.promotion import (
    accept_reviewed_proposal,
    promote_accepted_proposal,
    review_proposal,
)
from recall.reasoning_graph import build_reasoning_graph
from recall.reasoning_proposals._extracted import ExtractedClaimProposalProvider
from recall.reasoning_proposals._providers import proposal_report
from recall.rewrite import RewriteRefused, apply_rewrite, plan_rewrite
from recall.truth_extraction._engine import DeterministicExtractionEngine
from recall.truth_extraction.extract import extract_corpus_claims
from recall.types import Chunk

_WHEN = datetime(2026, 8, 12, tzinfo=timezone.utc)

OLD = "old_decision_2026-01-01.md"
NEW = "new_decision_2026-02-01.md"
DOCUMENTS = {
    OLD: "# old decision\n\nThe original call, made in January.\n",
    NEW: f"# new decision\n\nThis memo supersedes {OLD} after the February review.\n",
}


def _graph(documents=DOCUMENTS):
    return build_reasoning_graph(
        [
            Chunk(name.split("_")[0], f"/corpus/{name}", text, {"file": name, "ord": 0})
            for name, text in documents.items()
        ],
        tenant_id="acme",
        generation_id="gen_1",
        pipeline_fingerprint="pipe-a",
        include_text=True,
    )


def _proposals(documents=DOCUMENTS):
    """Every proposal the REAL pipeline emits for `documents`.

    Built through `extract_corpus_claims`, `ExtractedClaimProposalProvider` and
    `proposal_report`, which is the route the library takes, so this goes red if any link
    changes shape. Hand constructing an InferenceProposal here would test nothing about the
    chain, which is the only thing this file is for.
    """
    extractions = extract_corpus_claims(documents, engine=DeterministicExtractionEngine())
    provider = ExtractedClaimProposalProvider(extractions)
    report = proposal_report(_graph(documents), model_provider=provider)
    assert report.provider_failures == (), report.provider_failures
    return tuple(
        proposal
        for proposal in (*report.proposals, *report.rejected_proposals)
        if proposal.provider_id == provider.provider_id
    )


@pytest.fixture
def supersession():
    found = [p for p in _proposals() if p.proposed_relation == "supersedes"]
    assert found, "the deterministic engine did not find the supersession stated in the prose"
    return found[0]


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    for name, text in DOCUMENTS.items():
        (tmp_path / name).write_text(text, encoding="utf-8", newline="\n")
    return tmp_path


def _promote(proposal, *, reviewer="reviewer@example.com", note="Read both memos."):
    reviewed = review_proposal(
        proposal, reviewer_id=reviewer, reviewed_at=_WHEN, audit_note=note
    )
    return promote_accepted_proposal(accept_reviewed_proposal(reviewed), promoted_at=_WHEN)


def test_an_extracted_claim_arrives_as_requires_review(supersession):
    """The measured prior is that four candidates survived mechanical rules and all four were
    wrong. Swapping rules for a model is not evidence that changed."""
    assert supersession.status == "requires_review"
    assert supersession.confidence is None


def test_promotion_refuses_without_a_named_reviewer(supersession):
    with pytest.raises(ValueError, match="reviewer identity is required"):
        _promote(supersession, reviewer="   ")


def test_promotion_refuses_without_an_audit_note(supersession):
    with pytest.raises(ValueError, match="audit note is required"):
        _promote(supersession, note="   ")


def test_a_promoted_supersession_declares_the_edge_on_the_superseding_memo(
    corpus, supersession
):
    fact = _promote(supersession)
    plan = plan_rewrite(corpus, fact)
    assert plan.edit_file == NEW
    assert supersedes_key(plan.value) == supersedes_key(OLD)

    result = apply_rewrite(corpus, fact, apply=True)
    assert result.written is True

    meta, _ = parse_frontmatter((corpus / NEW).read_text(encoding="utf-8"))
    assert supersedes_key(meta["supersedes"]) == supersedes_key(OLD)


def test_the_superseded_memo_is_left_untouched(corpus, supersession):
    """Inverting the direction demotes the live memo beneath the one it replaced."""
    before = (corpus / OLD).read_text(encoding="utf-8")
    apply_rewrite(corpus, _promote(supersession), apply=True)
    assert (corpus / OLD).read_text(encoding="utf-8") == before
    assert "supersedes" not in parse_frontmatter(
        (corpus / OLD).read_text(encoding="utf-8")
    )[0]


def test_a_rewrite_refuses_a_proposal_that_did_not_pass_promotion(corpus, supersession):
    """`promotion.py` held a complete gate with no caller outside its own tests. This is the
    reason it exists, so the gate has to actually stop an unreviewed proposal."""
    with pytest.raises((RewriteRefused, TypeError, ValueError, AttributeError)):
        apply_rewrite(corpus, supersession, apply=True)  # type: ignore[arg-type]


def test_the_written_edge_is_readable_by_the_trust_layer(corpus, supersession):
    """A declared edge the trust layer cannot resolve is the defect `supersedes_key` exists for."""
    apply_rewrite(corpus, _promote(supersession), apply=True)
    meta, _ = parse_frontmatter((corpus / NEW).read_text(encoding="utf-8"))
    target = supersedes_key(meta["supersedes"])
    assert (corpus / f"{target}.md").exists(), "the declared target does not resolve to a file"


def test_a_dry_run_writes_nothing(corpus, supersession):
    before = {name: (corpus / name).read_text(encoding="utf-8") for name in DOCUMENTS}
    result = apply_rewrite(corpus, _promote(supersession))
    assert result.written is False
    assert {name: (corpus / name).read_text(encoding="utf-8") for name in DOCUMENTS} == before


def test_the_adapter_imports_on_its_own():
    """`recall.truth_extraction` re-exported the adapter, which imports back into it.

    That cycle only showed itself when the adapter was imported FIRST, on a fresh interpreter,
    which is why every existing test missed it: they all imported the package first and the
    module was already in `sys.modules`. A subprocess is the only honest check.
    """
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-c", "import recall.reasoning_proposals._extracted"],
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, f"importing the adapter first still fails:\n{out.stderr}"


def test_the_dependency_runs_one_way():
    """truth_extraction must not import reasoning_proposals: the query planner imports the
    latter, and pulling extraction in behind it would put a model on the query path."""
    from pathlib import Path

    source = Path("recall/truth_extraction/__init__.py").read_text(encoding="utf-8")
    offending = [
        line
        for line in source.splitlines()
        if line.startswith(("import ", "from ")) and "reasoning_proposals" in line
    ]
    assert not offending, f"truth_extraction imports reasoning_proposals: {offending}"


def test_a_claim_the_ladder_refused_never_reaches_the_corpus():
    """The ladder is the whole defence. A target outside the corpus must produce no proposal."""
    documents = {
        NEW: f"# new decision\n\nThis memo supersedes {OLD} after the February review.\n"
    }
    relations = [p.proposed_relation for p in _proposals(documents)]
    assert "supersedes" not in relations, (
        "a supersession naming a target that is not in the corpus produced a proposal"
    )
